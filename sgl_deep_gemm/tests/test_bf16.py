import numpy as np
import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes
)
from generators import (
    get_arch_major, layout_masked_to_psum, align,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,
    generate_k_grouped_contiguous_psum,
    get_mk_alignment_for_contiguous_layout
)


def check_bf16_psum_zero_padding(a: torch.Tensor, d: torch.Tensor, grouped_layout: torch.Tensor) -> None:
    for group_idx, current_m in enumerate(grouped_layout.cpu().tolist()):
        aligned_m = align(current_m, get_mk_alignment_for_contiguous_layout())
        if current_m < aligned_m:
            a_padding = a[current_m: aligned_m]
            d_padding = d[current_m: aligned_m]
            assert torch.equal(a_padding, torch.zeros_like(a_padding)), f'{group_idx=}, nonzero BF16 input padding'
            assert torch.equal(d_padding, torch.zeros_like(d_padding)), f'{group_idx=}, nonzero BF16 output padding'


def test_gemm() -> None:
    print('Testing GEMM:')
    scores = []
    for kernel_type, _, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)
            func_name = f'bf16_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else a.T
                b = b if major_b.is_k_major() else b.T
                assert a.is_contiguous() and b.is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, c=c)
            diff = calc_diff(d, ref_d)
            assert diff < 1e-5, (f'{m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                   f'{diff:.5f}, alias={test_alias}')
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)

        t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), 'bf16_gemm', suppress_kineto_output=True)
        cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:7.1f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{(cublas_t + split_k_t) / t:.2f}x cuBLAS')
        if cublas_t > 0:
            scores.append((cublas_t + split_k_t) / t)
    print(f"Average speedup over cuBLASLt: {float(np.prod(scores)) ** (1.0 / len(scores)):.3f}x\n")


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for _, _, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout, ensure_zero_padding in enumerate_m_grouped_contiguous(torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for test_alias in (False, True):
            m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                              use_bf16=True, use_psum_layout=use_psum_layout)
            func_name = f"m_grouped_bf16_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else b.mT
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, grouped_layout, use_psum_layout=use_psum_layout,
                                          ensure_zero_padding=ensure_zero_padding)
            if use_psum_layout:
                for j in range(num_groups):
                    start = 0 if j == 0 else align(grouped_layout[j - 1], get_mk_alignment_for_contiguous_layout())
                    end = grouped_layout[j]
                    diff = calc_diff(d[start : end], ref_d[start : end])
                    assert diff < 1e-5, (f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}, '
                                         f'alias={test_alias}, {ensure_zero_padding=}')
                if ensure_zero_padding:
                    check_bf16_psum_zero_padding(a, d, grouped_layout)
            else:
                diff = calc_diff(d, ref_d)
                assert diff < 1e-5, f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                          use_bf16=True, use_psum_layout=use_psum_layout)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, use_psum_layout=use_psum_layout,
                                                        ensure_zero_padding=ensure_zero_padding)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, layout={major_opt}, '
              f'psum={use_psum_layout}, zero_pad={ensure_zero_padding}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for _, _, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.bfloat16):
        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m_per_group * 1.2))
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for i in range(num_tests):
            a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k,
                                                                         use_bf16=True, use_psum_layout=use_psum_layout)
            if use_psum_layout:
                a_psum = layout_masked_to_psum(a, psum_m)
                d_psum = layout_masked_to_psum(d, psum_m)

            # noinspection PyShadowingNames
            def test_func():
                if use_psum_layout:
                    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_psum, b, d_psum, psum_m,
                                                                use_psum_layout=True, expected_m_for_psum_layout=expected_m_per_group)
                else:
                    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group)

            test_func()
            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                if use_psum_layout:
                    d_slice = d_psum[: psum_m[j]] if j == 0 else d_psum[align(psum_m[j - 1], get_mk_alignment_for_contiguous_layout()): psum_m[j]]
                else:
                    d_slice = d[j, :masked_m[j].item()]
                diff = calc_diff(d_slice, ref_d[j, :masked_m[j].item()])
                assert diff < 1e-5, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'


            # Test performance with fixed shapes
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'psum={1 if use_psum_layout else 0}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    for num_groups, m, n, major_a, major_b, real_ks_cpu, aligned_ks_cpu, _, _, alignment, use_psum_layout in enumerate_k_grouped_contiguous(torch.bfloat16):
        include_k_tail = get_arch_major() == 10 and alignment == 32
        for test_real_ks_cpu, test_aligned_ks_cpu, _, test_k_tail in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu, alignment, use_psum_layout, include_k_tail):
            if use_psum_layout:
                total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous_psum(num_groups, m, n, major_a, major_b, test_real_ks_cpu, k_alignment=alignment, use_bf16=True, gran_k=alignment)
            else:
                total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, test_aligned_ks_cpu, use_bf16=True)
            c_orig = c.clone() if use_psum_layout else None
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, test_aligned_ks_cpu, grouped_layout, c, use_psum_layout=use_psum_layout)

            diff = calc_diff(d, ref_d)
            assert diff < 1e-5, f'{m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, {test_aligned_ks_cpu=}, {use_psum_layout=}, {test_k_tail=}, {diff:.7f}'

            # Unsynced psum paths
            if use_psum_layout:
                c.copy_(c_orig)
                deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, None, grouped_layout, c,
                                                            use_psum_layout=True)
                diff = calc_diff(d, ref_d)
                assert diff < 1e-5, f'None ks_cpu path: {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, {diff:.7f}'

                c.copy_(c_orig)
                deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, [], grouped_layout, c,
                                                            use_psum_layout=True)
                diff = calc_diff(d, ref_d)
                assert diff < 1e-5, f'empty ks_cpu path: {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, {diff:.7f}'

        # Test performance
        if use_psum_layout:
            total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous_psum(num_groups, m, n, major_a, major_b, real_ks_cpu, k_alignment=alignment, use_bf16=True, gran_k=alignment)
        else:
            total_k, a, b, c, d, ref_d, grouped_layout, _ = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, aligned_ks_cpu, use_bf16=True)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, aligned_ks_cpu, grouped_layout, c, use_psum_layout=use_psum_layout)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={total_k:5}, align={alignment:3}, psum={int(use_psum_layout)}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * total_k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')

    print()


def test_cublaslt_gemm() -> None:
    print('Testing cuBLASLt GEMM:')
    for kernel_type, _, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(dtype=torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)
        deep_gemm.cublaslt_gemm_nt(a, b, d, c=c)
        diff = calc_diff(d, ref_d)
        # BF16 accumulation has lower precision than cuBLASLt's FP32 accumulation
        threshold = 1e-5 if (accumulate and out_dtype == torch.bfloat16) else 6e-7
        assert diff < threshold, f'{diff=}, ({m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=})'

        t_nvjet, t_gemv, t_gemm = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
        t = t_nvjet + t_gemv + t_gemm
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    if get_arch_major() >= 9:
        test_gemm()
        test_m_grouped_gemm_contiguous()
        test_m_grouped_gemm_masked()
        test_k_grouped_gemm_contiguous()

    test_cublaslt_gemm()
