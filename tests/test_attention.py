import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8

from generators import get_arch_major, generate_normal, get_ue8m0_usage, get_kernel_types, MajorTypeAB


def apply_skip_head_mid(d: torch.Tensor, head_splits: Tuple[int, int, int]):
    left, mid, right = head_splits
    m, n = d.shape
    assert n % (left + right) == 0
    num_heads = n // (left + right)

    # Split and insert padding tensor
    d = d.view(m, num_heads, -1)
    d_left = d[:, :, :left]
    d_right = d[:, :, -right:]

    d_mid = torch.zeros((m, num_heads, mid), dtype=d.dtype, device=d.device)
    return torch.cat([d_left, d_mid, d_right], dim=2).view(m, -1)


def test_gemm_skip_head_mid() -> None:
    print('Testing GEMM skip head mid:')
    head_splits = (128, 64, 128)

    major_a, major_b = MajorTypeAB.KMajor,  MajorTypeAB.KMajor
    out_dtype, accumulate = torch.bfloat16, False

    for kernel_type in get_kernel_types(dtype=torch.float8_e4m3fn):
        for m in (128, 4096):
            for n, k in [(32768, 512), (8192, 512)]:
                kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
                use_ue8m0 = get_ue8m0_usage(kernel_type)
                disable_ue8m0_cast = not use_ue8m0

                a, b, _, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
                d = apply_skip_head_mid(d, head_splits)
                ref_d = apply_skip_head_mid(ref_d, head_splits)

                deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f'{m=}, {n=}, {k=}, {kernel_opt}, {diff:.5f}'

                t = bench_kineto(lambda: deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast),
                                'fp8_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}): '
                    f'{t * 1e6:4.0f} us | '
                    f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                    f'{(count_bytes(a, b, d)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    test_gemm_skip_head_mid()
