import sys
from pathlib import Path
from typing import Callable

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import bench, calc_diff
from deep_gemm.utils.math import per_token_cast_to_fp8


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SM90 MXFP8FP8 tests")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"SM90 MXFP8FP8 tests require sm_90, got sm_{major}x")


def _cast_back_from_fp8_1d(x: torch.Tensor, sf: torch.Tensor, gran_k: int) -> torch.Tensor:
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def _e8m0_from_fp32_pow2(sf: torch.Tensor) -> torch.Tensor:
    assert sf.dtype == torch.float32 or sf.dtype == torch.float
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _pack_ue8m0_u8_to_i32(sf: torch.Tensor) -> torch.Tensor:
    assert sf.dtype == torch.uint8
    if sf.shape[-1] % 4 != 0:
        padded = torch.zeros(
            (*sf.shape[:-1], ((sf.shape[-1] + 3) // 4) * 4),
            device=sf.device,
            dtype=sf.dtype,
        )
        padded[..., : sf.shape[-1]] = sf
        sf = padded
    sf_i32 = sf.contiguous().view(*sf.shape[:-1], sf.shape[-1] // 4, 4).to(torch.int32)
    return (
        sf_i32[..., 0]
        | torch.bitwise_left_shift(sf_i32[..., 1], 8)
        | torch.bitwise_left_shift(sf_i32[..., 2], 16)
        | torch.bitwise_left_shift(sf_i32[..., 3], 24)
    ).contiguous()


def _pack_ue8m0_u8_to_i32_mn_major(sf: torch.Tensor) -> torch.Tensor:
    packed = _pack_ue8m0_u8_to_i32(sf)
    return packed.transpose(-1, -2).contiguous().transpose(-1, -2)


def _fp32_from_e8m0_u8(sf: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_left_shift(sf.to(torch.int32), 23).contiguous().view(torch.float32)


def test_packed_ue8m0_i32_byte_order_matches_sm100_layout():
    _require_sm90()
    import deep_gemm.utils.layout

    sf = torch.tensor(
        [[2.0, 4.0, 8.0, 16.0], [32.0, 64.0, 128.0, 256.0]],
        device="cuda",
        dtype=torch.float32,
    )
    expected = _pack_ue8m0_u8_to_i32(_e8m0_from_fp32_pow2(sf))
    packed = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)

    assert torch.equal(packed.cpu(), expected.cpu())


def _tflops(m: int, n: int, k: int, elapsed: float) -> float:
    return 2.0 * m * n * k / elapsed / 1e12


def _time_kernel(fn: Callable[[], None]) -> float:
    fn()
    return bench(fn, num_warmups=5, num_tests=10)


def _make_contiguous_case(groups: int, m_per_group: int, n: int, k: int):
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data, a_sf_fp32 = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=32)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)
    a = (a_data, _e8m0_from_fp32_pow2(a_sf_fp32))
    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=32)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        start = group_id * m_per_group
        end = start + m_per_group
        b_dequant = _cast_back_from_fp8_1d(b_data[group_id], b_sf_fp32[group_id], gran_k=32)
        ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)
    return a, (b_data, _e8m0_from_fp32_pow2(b_sf_fp32)), grouped_layout, ref


def test_m_grouped_mxfp8_fp8_contiguous_e8m0_scale_accuracy():
    _require_sm90()
    # SM90 grouped-contiguous WGMMA/TMA maps one B group per M tile.
    groups, m_per_group, n, k = 2, 128, 48, 128
    a, b, grouped_layout, ref = _make_contiguous_case(groups, m_per_group, n, k)
    d = torch.empty_like(ref)

    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(a, b, d, grouped_layout)
    diff = calc_diff(d, ref)
    assert diff < 0.03


def test_m_grouped_mxfp8_fp8_masked_e8m0_scale_accuracy():
    _require_sm90()
    groups, max_m, n, k = 2, 32, 48, 128
    masked_m = torch.tensor([7, 19], device="cuda", dtype=torch.int32)
    a_ref = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf_fp32 = torch.empty((groups, max_m, k // 32), device="cuda", dtype=torch.float32)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        a_data[group_id], a_sf_fp32[group_id] = per_token_cast_to_fp8(
            a_ref[group_id], use_ue8m0=True, gran_k=32
        )
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    a = (a_data, _e8m0_from_fp32_pow2(a_sf_fp32))
    b = (b_data, _e8m0_from_fp32_pow2(b_sf_fp32))
    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=32)
    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    for group_id, valid_m in enumerate(masked_m.tolist()):
        b_dequant = _cast_back_from_fp8_1d(b_data[group_id], b_sf_fp32[group_id], gran_k=32)
        ref[group_id, :valid_m] = (a_dequant[group_id, :valid_m] @ b_dequant.t()).to(torch.bfloat16)

    d = torch.empty_like(ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m=max_m)
    diff = max(
        calc_diff(d[group_id, :valid_m], ref[group_id, :valid_m])
        for group_id, valid_m in enumerate(masked_m.tolist())
    )
    assert diff < 0.03


def test_m_grouped_mxfp8_fp8_contiguous_packed_int32_scale_accuracy():
    _require_sm90()
    groups, m_per_group, n, k = 2, 128, 48, 640
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data, a_sf_fp32 = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=128)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    a = (a_data, _pack_ue8m0_u8_to_i32(_e8m0_from_fp32_pow2(a_sf_fp32)))
    b = (b_data, _pack_ue8m0_u8_to_i32(_e8m0_from_fp32_pow2(b_sf_fp32)))
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(
        m_per_group
    )

    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=128)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        start = group_id * m_per_group
        end = start + m_per_group
        b_dequant = _cast_back_from_fp8_1d(b_data[group_id], b_sf_fp32[group_id], gran_k=32)
        ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)

    d = torch.empty_like(ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
        a, b, d, grouped_layout, recipe_a=(1, 128), recipe_b=(1, 32)
    )
    diff = calc_diff(d, ref)
    assert diff < 0.03


def test_m_grouped_mxfp8_fp8_contiguous_deepep_normal_scale_layout_accuracy():
    _require_sm90()
    torch.manual_seed(0)
    # Matches SGLang DeepEP normal layout:
    #   A scale: packed int32 MN-major non-contiguous view, gran_k=128
    #   B scale: raw uint8 [expert, n, k/32], gran_k=32
    groups, m_per_group, n, k = 3, 128, 80, 640
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
    a_data, _ = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=128)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    for group_id in range(groups):
        b_data[group_id], _ = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    # Keep exponents near 127 (scale 1.0). Wider synthetic ranges produce very
    # large BF16 outputs, where a normal one-ULP BF16 difference has a misleading
    # absolute error while relative/cosine error is still essentially zero.
    a_exp = (
        126
        + (torch.arange(m, device="cuda", dtype=torch.uint8).view(m, 1) % 2)
        + (torch.arange(k // 128, device="cuda", dtype=torch.uint8).view(1, -1) % 2)
    )
    b_exp = (
        126
        + (torch.arange(groups, device="cuda", dtype=torch.uint8).view(groups, 1, 1) % 2)
        + (torch.arange(n, device="cuda", dtype=torch.uint8).view(1, n, 1) % 2)
        + (torch.arange(k // 32, device="cuda", dtype=torch.uint8).view(1, 1, -1) % 2)
    )

    a_scale_i32 = _pack_ue8m0_u8_to_i32_mn_major(a_exp)
    b_scale_u8 = b_exp.contiguous()
    a = (a_data, a_scale_i32)
    b = (b_data, b_scale_u8)
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(
        m_per_group
    )

    a_dequant = _cast_back_from_fp8_1d(a_data, _fp32_from_e8m0_u8(a_exp), gran_k=128)
    b_scale_fp32 = _fp32_from_e8m0_u8(b_exp)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        start = group_id * m_per_group
        end = start + m_per_group
        b_dequant = _cast_back_from_fp8_1d(b_data[group_id], b_scale_fp32[group_id], gran_k=32)
        ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)

    d = torch.empty_like(ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
        a, b, d, grouped_layout, recipe_a=(1, 128), recipe_b=(1, 32)
    )
    diff = calc_diff(d, ref)
    max_abs_diff = (d.float() - ref.float()).abs().max().item()
    ref_absmax = ref.float().abs().max().item()
    max_rel_diff = max_abs_diff / max(ref_absmax, 1.0)
    print(
        "DeepEP-normal scale layout diff: "
        f"calc_diff={diff:.6f}, max_abs_diff={max_abs_diff:.6f}, "
        f"ref_absmax={ref_absmax:.6f}, max_rel_diff={max_rel_diff:.6f}, "
        f"a_scale_shape={tuple(a_scale_i32.shape)}, a_scale_stride={tuple(a_scale_i32.stride())}, "
        f"b_scale_shape={tuple(b_scale_u8.shape)}, b_scale_stride={tuple(b_scale_u8.stride())}"
    )
    assert diff < 0.03


def test_m_grouped_mxfp8_fp8_masked_packed_int32_mn_major_scale_accuracy():
    _require_sm90()
    groups, max_m, n, k = 3, 128, 64, 640
    masked_m = torch.tensor([7, 65, 113], device="cuda", dtype=torch.int32)
    a_ref = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf_fp32 = torch.empty((groups, max_m, k // 128), device="cuda", dtype=torch.float32)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        a_data[group_id], a_sf_fp32[group_id] = per_token_cast_to_fp8(
            a_ref[group_id], use_ue8m0=True, gran_k=128
        )
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    a = (a_data, _pack_ue8m0_u8_to_i32_mn_major(_e8m0_from_fp32_pow2(a_sf_fp32)))
    b = (b_data, _e8m0_from_fp32_pow2(b_sf_fp32))
    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=128)
    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    for group_id, valid_m in enumerate(masked_m.tolist()):
        b_dequant = _cast_back_from_fp8_1d(b_data[group_id], b_sf_fp32[group_id], gran_k=32)
        ref[group_id, :valid_m] = (a_dequant[group_id, :valid_m] @ b_dequant.t()).to(torch.bfloat16)

    d = torch.empty_like(ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_masked(
        a, b, d, masked_m, expected_m=max_m, recipe_a=(1, 128), recipe_b=(1, 32)
    )
    diff = max(
        calc_diff(d[group_id, :valid_m], ref[group_id, :valid_m])
        for group_id, valid_m in enumerate(masked_m.tolist())
    )
    assert diff < 0.03


def test_m_grouped_mxfp8_vs_fp8_perf_contiguous_and_masked():
    _require_sm90()
    groups, n, k = 4, 1024, 1024

    # Contiguous: one B group per M tile.
    m_per_group = 128
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
    a_data, a_mx_sf_fp32 = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=32)
    a = (a_data, _e8m0_from_fp32_pow2(a_mx_sf_fp32))
    a_fp8 = per_token_cast_to_fp8(a_ref, use_ue8m0=False, gran_k=128)
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)

    b_mx_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_mx_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    b_fp8_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // 128), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        b_mx_data[group_id], b_mx_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=False, gran_k=128
        )
    b_mx = (b_mx_data, _e8m0_from_fp32_pow2(b_mx_sf_fp32))
    b_fp8 = (b_fp8_data, b_fp8_sf)
    d_mx = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    d_fp8 = torch.empty_like(d_mx)

    def run_mx_contiguous():
        deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(a, b_mx, d_mx, grouped_layout)

    def run_fp8_contiguous():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a_fp8, b_fp8, d_fp8, grouped_layout, recipe_a=(1, 128), recipe_b=(1, 128)
        )

    mx_contiguous_elapsed = _time_kernel(run_mx_contiguous)
    fp8_contiguous_elapsed = _time_kernel(run_fp8_contiguous)
    contiguous_diff = float(calc_diff(d_mx, d_fp8))

    # Masked: same shape class, but allow uneven active rows per group.
    max_m = 128
    masked_m = torch.tensor([128, 96, 64, 32], device="cuda", dtype=torch.int32)
    a_ref_masked = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_masked = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
    a_masked_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_masked_sf_fp32 = torch.empty((groups, max_m, k // 32), device="cuda", dtype=torch.float32)
    a_fp8_masked_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_fp8_masked_sf = torch.empty((groups, max_m, k // 128), device="cuda", dtype=torch.float32)
    b_mx_masked_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_mx_masked_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    b_fp8_masked_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_masked_sf = torch.empty((groups, n, k // 128), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        a_masked_data[group_id], a_masked_sf_fp32[group_id] = per_token_cast_to_fp8(
            a_ref_masked[group_id], use_ue8m0=True, gran_k=32
        )
        a_fp8_masked_data[group_id], a_fp8_masked_sf[group_id] = per_token_cast_to_fp8(
            a_ref_masked[group_id], use_ue8m0=False, gran_k=128
        )
        b_mx_masked_data[group_id], b_mx_masked_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref_masked[group_id], use_ue8m0=True, gran_k=32
        )
        b_fp8_masked_data[group_id], b_fp8_masked_sf[group_id] = per_token_cast_to_fp8(
            b_ref_masked[group_id], use_ue8m0=False, gran_k=128
        )
    a_masked = (a_masked_data, _e8m0_from_fp32_pow2(a_masked_sf_fp32))
    a_fp8_masked = (a_fp8_masked_data, a_fp8_masked_sf)
    b_mx_masked = (b_mx_masked_data, _e8m0_from_fp32_pow2(b_mx_masked_sf_fp32))
    b_fp8_masked = (b_fp8_masked_data, b_fp8_masked_sf)
    d_mx_masked = torch.empty((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    d_fp8_masked = torch.empty_like(d_mx_masked)

    def run_mx_masked():
        deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_masked(
            a_masked, b_mx_masked, d_mx_masked, masked_m, expected_m=max_m
        )

    def run_fp8_masked():
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a_fp8_masked,
            b_fp8_masked,
            d_fp8_masked,
            masked_m,
            expected_m=max_m,
            recipe_a=(1, 128),
            recipe_b=(1, 128),
        )

    mx_masked_elapsed = _time_kernel(run_mx_masked)
    fp8_masked_elapsed = _time_kernel(run_fp8_masked)
    masked_diff = max(
        float(calc_diff(d_mx_masked[group_id, :valid_m], d_fp8_masked[group_id, :valid_m]))
        for group_id, valid_m in enumerate(masked_m.tolist())
    )

    masked_active_m = int(masked_m.sum().item())
    rows = [
        (
            "contiguous",
            m,
            mx_contiguous_elapsed,
            fp8_contiguous_elapsed,
            _tflops(m, n, k, mx_contiguous_elapsed),
            _tflops(m, n, k, fp8_contiguous_elapsed),
            contiguous_diff,
        ),
        (
            "masked",
            masked_active_m,
            mx_masked_elapsed,
            fp8_masked_elapsed,
            _tflops(masked_active_m, n, k, mx_masked_elapsed),
            _tflops(masked_active_m, n, k, fp8_masked_elapsed),
            masked_diff,
        ),
    ]
    print("kernel | active M | MXFP8 us | FP8 us | MXFP8 TFLOPS | FP8 TFLOPS | speedup | diff")
    print("-- | -- | -- | -- | -- | -- | -- | --")
    for name, active_m, mx_elapsed, fp8_elapsed, mx_tflops, fp8_tflops, diff in rows:
        print(
            f"{name} | {active_m} | {mx_elapsed * 1e6:.0f} | {fp8_elapsed * 1e6:.0f} | "
            f"{mx_tflops:.1f} | {fp8_tflops:.1f} | {fp8_elapsed / mx_elapsed:.2f}x | {diff:.4f}"
        )

    assert mx_contiguous_elapsed > 0
    assert fp8_contiguous_elapsed > 0
    assert mx_masked_elapsed > 0
    assert fp8_masked_elapsed > 0
    assert contiguous_diff == contiguous_diff
    assert masked_diff == masked_diff
