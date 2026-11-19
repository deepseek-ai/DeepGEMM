import torch
from typing import Callable, Tuple

import deep_gemm
from deep_gemm.testing import assert_bitwise_equal
from deep_gemm.utils import align, ceil_div, get_mk_alignment_for_contiguous_layout


def assert_psum_zero_padding(a: torch.Tensor | tuple, d: torch.Tensor, grouped_layout: torch.Tensor, dtype_label: str) -> None:
    a_data = a[0] if isinstance(a, tuple) else a
    for group_idx, current_m in enumerate(grouped_layout.cpu().tolist()):
        aligned_m = align(current_m, get_mk_alignment_for_contiguous_layout())
        if current_m < aligned_m:
            a_padding = a_data[current_m: aligned_m]
            d_padding = d[current_m: aligned_m]
            assert torch.equal(a_padding, torch.zeros_like(a_padding)), f'{group_idx=}, nonzero {dtype_label} input padding'
            assert torch.equal(d_padding, torch.zeros_like(d_padding)), f'{group_idx=}, nonzero {dtype_label} output padding'


def assert_direct_output_matches_fp32_accumulation(
        direct_output: torch.Tensor,
        launch: Callable[[torch.Tensor, torch.Tensor | None], None],
        label: str) -> None:
    """Check direct output against the same workload accumulated from zero in FP32.

    ``launch(output, accumulator)`` reruns the direct-output workload with the
    same inputs, shapes, group layout, K lengths, and kernel options, changing
    only the output mode. This helper passes one zero-initialized FP32 tensor as
    both D and C, so accumulation mode computes ``D = GEMM + 0``. FP32 direct
    output must match this baseline bitwise; BF16 direct output must match the
    baseline after one FP32-to-BF16 cast.
    """
    assert direct_output.dtype in (torch.float, torch.bfloat16)

    fp32_baseline = torch.zeros_like(direct_output, dtype=torch.float)
    launch(fp32_baseline, fp32_baseline)
    expected = fp32_baseline if direct_output.dtype == torch.float else fp32_baseline.to(torch.bfloat16)
    assert_bitwise_equal(direct_output, expected, label)


def convert_to_fp8(x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # E4M3 exactly represents every finite E2M1 value. Convert the encoding without requantizing.
    data, sf = x
    if data.dtype == torch.float8_e4m3fn:
        return x
    if data.stride(-1) != 1:
        converted, sf = convert_to_fp8((data.mT, sf))
        return converted.mT, sf
    lut = torch.tensor((0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c), dtype=torch.uint8, device=data.device)
    packed = data.view(torch.uint8)
    lo, hi = packed & 0xf, packed >> 4
    codes = torch.stack((lut[(lo & 7).long()] | ((lo & 8) << 4),
                         lut[(hi & 7).long()] | ((hi & 8) << 4)), dim=-1).flatten(-2)
    return codes.view(torch.float8_e4m3fn), sf


def to_cublaslt_vec16_sf_layout(sf: torch.Tensor) -> torch.Tensor:
    # Convert per-32 MXFP4 power-of-two scales into the equivalent per-16 NVFP4 layout.
    assert sf.dim() == 2 and sf.dtype == torch.float
    assert (sf.view(torch.int32) & 0x7fffff == 0).all(), 'SF must be positive powers of two'
    exp = (sf.view(torch.int32) >> 23) - 127
    assert (exp >= -9).all() and (exp <= 8).all(), 'SF out of the UE4M3 range'
    ue4m3 = torch.where(exp >= -6, (exp + 7) << 3, 1 << (exp + 9)).to(torch.uint8).repeat_interleave(2, dim=1)

    mn, sf_k = ue4m3.shape
    num_mn_tiles, num_k_tiles = ceil_div(mn, 128), ceil_div(sf_k, 4)
    padded = torch.zeros((num_mn_tiles * 128, num_k_tiles * 4), device=sf.device, dtype=torch.uint8)
    padded[:mn, :sf_k] = ue4m3
    return padded.view(num_mn_tiles, 4, 32, num_k_tiles, 4).permute(0, 3, 2, 1, 4).contiguous()


def make_cublas_gemm(a: Tuple[torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor],
                     d: torch.Tensor, c: torch.Tensor | None) -> Callable[[], None]:
    if a[0].dtype == torch.int8 and b[0].dtype == torch.int8:
        a_nvfp4 = a[0], to_cublaslt_vec16_sf_layout(a[1])
        b_nvfp4 = b[0], to_cublaslt_vec16_sf_layout(b[1])
        return lambda: deep_gemm.cublaslt_nvfp4_gemm_nt(a_nvfp4, b_nvfp4, d, c=c)
    a_fp8, b_fp8 = convert_to_fp8(a), convert_to_fp8(b)
    return lambda: deep_gemm.cublaslt_gemm_nt(a_fp8[0], b_fp8[0], d, c=c)
