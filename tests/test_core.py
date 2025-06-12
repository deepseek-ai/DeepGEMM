# PyTorch has its own NVRTC, which may have a lower version than the system
# So try to disable PyTorch's NVRTC, or import NVRTC before PyTorch
import cuda.bindings.nvrtc as nvrtc
print(f'NVRTC version: {nvrtc.nvrtcVersion()[1:]}')

import random
import torch

import deep_gemm
from deep_gemm.utils.layout import MajorTypeAB
from deep_gemm.testing.bench import bench_kineto
from deep_gemm.testing.numeric import calc_diff, count_bytes

from generators import (
    enumerate_normal, enumerate_grouped_contiguous, enumerate_grouped_masked,
    generate_normal, generate_grouped_contiguous, generate_grouped_masked,
)


def test_gemm() -> None:
    print('Testing GEMM:')
    for m, k, n, major_a, major_b, accumulate, out_dtype in enumerate_normal():
        major_opt  = 'N' if major_a == MajorTypeAB.KMajor else 'T'
        major_opt += 'T' if major_b == MajorTypeAB.KMajor else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'accumulate={int(accumulate)}'

        a, b, c, d, ref_d = generate_normal(m, k, n, major_a, major_b, accumulate, out_dtype)
        deep_gemm.fp8_gemm_nt(a, b, d, c=c)
        diff = calc_diff(d, ref_d)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {major_opt=}, {diff:.5f}'

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.fp8_gemm_nt(a, b, d)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, MemLayout={major_opt}, {out_opt}, {acc_opt}):'
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes((a, b, c, d)) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, expected_m_per_group, k, n, major_a, major_b in enumerate_grouped_contiguous():
        # TODO: make a stronger test
        major_opt  = 'N' if major_a == MajorTypeAB.KMajor else 'T'
        major_opt += 'T' if major_b == MajorTypeAB.KMajor else 'N'

        m, a, b, m_indices, d, ref_d = generate_grouped_contiguous(num_groups, expected_m_per_group, k, n, major_a, major_b)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices)
        diff = calc_diff(d, ref_d)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {major_opt}, {diff:.5f}'

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, MemLayout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes((a, b, d)) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing grouped masked GEMM:')

    # TODO: merge Hopper's tests
    for num_groups, m, k, n in enumerate_grouped_masked():
        # Test correctness
        masked_m_candidates = list(filter(lambda candidate: candidate <= m, (128, 256, 384)))
        for i in range(10):
            a, b, d, ref_d = generate_grouped_masked(num_groups, m, k, n)
            masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
            for j in range(num_groups):
                masked_m[j] = random.choice(masked_m_candidates)
            expected_m = min(int(masked_m.float().mean()) + 1, m)
            deep_gemm.fp8_m_grouped_gemm_nt_masked(a, b, d, masked_m, expected_m)
            for j in range(num_groups):
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, d, ref_d = generate_grouped_masked(num_groups, m, k, n)
        masked_m = torch.ones((num_groups, ), device='cuda', dtype=torch.int) * m

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.fp8_m_grouped_gemm_nt_masked(a, b, d, masked_m, m)

        # Test performance with fixed shapes
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes((a, b, d)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
