import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    ignore_env, get_arch_major
)

from generators import (
    enumerate_normal_w4, enumerate_m_grouped_contiguous_w4, enumerate_m_grouped_masked_w4,
    generate_normal_w4, generate_m_grouped_contiguous_w4, generate_m_grouped_masked_w4,
)


@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 9)
def test_gemm() -> None:
    print('Testing SM90 W4AFP8 GEMM:')
    for m, n, k in enumerate_normal_w4():
        accumulate = False
        out_dtype = torch.bfloat16

        a, b, c, d, ref_d = generate_normal_w4(m, n, k, accumulate, out_dtype)
        deep_gemm.sm90_w4afp8_gemm_nt(a, b, d, c=c)
        diff = calc_diff(d, ref_d)
        assert diff < 0.001, (f'{m=}, {n=}, {k=}, {out_dtype=}, {diff:.5f}')

        def test_func():
            deep_gemm.sm90_w4afp8_gemm_nt(a, b, d, c=c)

        # Test performance
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}): '
              f'{t * 1e6:6.1f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing SM90 W4AFP8 m-grouped contiguous GEMM:')

    for num_groups, expected_m_per_group, n, k in enumerate_m_grouped_contiguous_w4():
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous_w4(num_groups, expected_m_per_group, n, k)
        deep_gemm.sm90_m_grouped_w4afp8_gemm_nt_contiguous(a, b, d, m_indices)
        d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
        diff = calc_diff(d, ref_d)
        assert diff < 0.001, f'{m=}, {n=}, {k=}, {diff:.5f}'

        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous_w4(num_groups, expected_m_per_group, n, k)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.sm90_m_grouped_w4afp8_gemm_nt_contiguous(a, b, d, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing SM90 W4AFP8 m-grouped masked GEMM:')

    for num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked_w4():
        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked_w4(num_groups, max_m, expected_m_per_group, n, k)
            deep_gemm.sm90_m_grouped_w4afp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)
            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

        # Test performance
        a, b, masked_m, d, ref_d = generate_m_grouped_masked_w4(num_groups, max_m, expected_m_per_group, n, k)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.sm90_m_grouped_w4afp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)

        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
