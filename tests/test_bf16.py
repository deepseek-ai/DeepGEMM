import contextlib
import numpy as np
import os
import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    assert_bitwise_equal, assert_check, calc_diff, count_bytes
)
from utils import (
    assert_direct_output_matches_fp32_accumulation,
    assert_stochastic_bf16_matches_fp32_accumulation,
    assert_psum_zero_padding,
)
from generators import (
    get_arch_major,
    enumerate_normal, enumerate_batched_syrk_symm, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,
)


def test_gemm() -> None:
    print('Testing GEMM:')
    scores = []
    use_alpha_options = (False, True) if get_arch_major() == 10 else (False,)
    use_bias_options = (False, True) if get_arch_major() == 10 else (False,)
    for kernel_type, _, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.bfloat16):
        deep_gemm.use_deterministic_algorithms(True)
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        for test_alias in (False, True):
            func_name = f'bf16_gemm_{major_opt.lower() if test_alias else "nt"}'
            for use_alpha in use_alpha_options:
                alpha = random.uniform(-1.0, 1.0) if use_alpha else None
                for with_bias in use_bias_options:
                    if with_bias and accumulate:
                        continue
                    a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                         kernel_type, use_bf16=True, alpha=alpha, with_bias=with_bias)
                    if test_alias:
                        a = a if major_a.is_k_major() else a.T
                        b = b if major_b.is_k_major() else b.T
                        assert a.is_contiguous() and b.is_contiguous()
                    epilogue_options = [dict(alpha=alpha)]
                    if with_bias:
                        epilogue_options.append(dict(epilogue=deep_gemm.epilogue.Alpha(alpha) if use_alpha else deep_gemm.epilogue.Identity()))
                        if use_alpha:
                            epilogue_options.extend((dict(alpha=1.0), dict(epilogue=deep_gemm.epilogue.Alpha(1.0))))
                    saved_c = c.clone() if with_bias else None
                    for epilogue_args in epilogue_options:
                        expect_error = with_bias and (epilogue_args.get('alpha') is not None or 'epilogue' in epilogue_args)
                        try:
                            getattr(deep_gemm, func_name)(a, b, d, c=c, **epilogue_args)
                        except RuntimeError as error:
                            if not expect_error:
                                raise
                            assert 'The epilogue class, `alpha`, the FP8 output pair and bias are exclusive' in str(error), str(error)
                            continue
                        assert not expect_error, 'Bias with alpha or an explicit epilogue must be rejected'
                        diff = calc_diff(d, ref_d)
                        assert_check(diff < 1e-5, (f'{m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                               f'{use_alpha=}, {alpha=}, {with_bias=}, {diff:.5f}, alias={test_alias}'))
                        if with_bias:
                            assert_check(torch.equal(c, saved_c))
                    if with_bias and not use_alpha:
                        fp32_output = torch.empty_like(d, dtype=torch.float32)
                        getattr(deep_gemm, func_name)(a, b, fp32_output, alpha=alpha)
                        assert_check(calc_diff(d, (fp32_output + c.float()).to(out_dtype)) < 1e-5)

        for with_bias in use_bias_options:
            if with_bias and accumulate:
                continue
            deep_gemm.use_deterministic_algorithms(True)
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                 kernel_type, use_bf16=True, with_bias=with_bias)
            if get_arch_major() == 10 and not accumulate and not with_bias and out_dtype == torch.bfloat16:
                deep_gemm.bf16_gemm_nt(a, b, d, epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                assert_stochastic_bf16_matches_fp32_accumulation(
                    d,
                    lambda output, accumulator: deep_gemm.bf16_gemm_nt(a, b, output, c=accumulator),
                    f'BF16 GEMM stochastic rounding, {m=}, {n=}, {k=}, {major_opt=}')

            t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), 'bf16_gemm', suppress_kineto_output=True)
            cublas_t = split_k_t = 0
            if not with_bias:
                deep_gemm.use_deterministic_algorithms(False)
                cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), ('nvjet', 'splitKreduce'), suppress_kineto_output=True)
            print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}, bias={int(with_bias)}): '
                  f'{t * 1e6:7.1f} us | '
                  f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                  f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate or with_bias)) / 1e9 / t:4.0f} GB/s'
                  + (f' | {(cublas_t + split_k_t) / t:.2f}x cuBLAS' if not with_bias else ''))
            if cublas_t > 0:
                scores.append((cublas_t + split_k_t) / t)
    if scores:
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
            m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                                          use_bf16=True, use_psum_layout=use_psum_layout)
            func_name = f"m_grouped_bf16_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else b.mT
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, grouped_layout, use_psum_layout=use_psum_layout,
                                          ensure_zero_padding=ensure_zero_padding)
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert_check(diff < 1e-5, f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}, alias={test_alias}, {ensure_zero_padding=}')
            if use_psum_layout and ensure_zero_padding:
                assert_psum_zero_padding(a, d, grouped_layout, 'BF16')
        m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
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
            a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k,
                use_bf16=True, use_psum_layout=use_psum_layout)

            def test_func():
                if use_psum_layout:
                    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout,
                                                                use_psum_layout=True, expected_m_for_psum_layout=expected_m_per_group)
                else:
                    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, grouped_layout, expected_m_per_group)

            test_func()
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert_check(diff < 1e-5, f'{max_m=}, {n=}, {k=}, {num_groups=}, {diff:.5f}')


            # Test performance with fixed shapes
            valid_m = int(valid_mask.sum().item())
            t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * (valid_m / (max_m * num_groups)) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'psum={1 if use_psum_layout else 0}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    for num_groups, m, n, major_a, major_b, real_ks_cpu, _, _, _, alignment, use_psum_layout, accumulate, out_dtype in \
            enumerate_k_grouped_contiguous(torch.bfloat16):
        for test_real_ks_cpu in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu):
            total_k, a, b, c, d, ref_d, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, test_real_ks_cpu, use_bf16=True,
                use_psum_layout=use_psum_layout, k_alignment=alignment,
                accumulate=accumulate, out_dtype=out_dtype)
            initial_d = d.clone()
            if not accumulate:
                initial_d.fill_(float('nan'))
            host_ks_options = (host_ks_cpu, None, []) if use_psum_layout else (host_ks_cpu, )
            for test_host_ks_cpu in host_ks_options:
                d.copy_(initial_d)
                deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                    a, b, d, test_host_ks_cpu, grouped_layout, c, use_psum_layout=use_psum_layout)
                if accumulate:
                    diff = calc_diff(d, ref_d)
                    assert_check(diff < 1e-5, (f'{m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                                        f'{test_host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, '
                                        f'{out_dtype=}, {diff:.7f}'))
                else:
                    case_label = (f'BF16 K-grouped direct output, {m=}, {n=}, {total_k=}, '
                                  f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, '
                                  f'{out_dtype=}')
                    launch = lambda output, accumulator: deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                        a, b, output, test_host_ks_cpu, grouped_layout, accumulator,
                        use_psum_layout=use_psum_layout)
                    assert_direct_output_matches_fp32_accumulation(d, launch, case_label)
                    if out_dtype == torch.bfloat16:
                        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                            a, b, d, test_host_ks_cpu, grouped_layout, None, use_psum_layout=use_psum_layout,
                            epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                        assert_stochastic_bf16_matches_fp32_accumulation(d, launch, f'stochastic rounding, {case_label}')

        # Test performance
        _, a, b, c, d, _, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
            num_groups, m, n, major_a, major_b, real_ks_cpu, use_bf16=True,
            use_psum_layout=use_psum_layout, k_alignment=alignment,
            accumulate=accumulate, out_dtype=out_dtype)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, host_ks_cpu, grouped_layout, c, use_psum_layout=use_psum_layout)

        t = bench_kineto(test_func, 'bf16_gemm', suppress_kineto_output=True)
        logical_k = sum(real_ks_cpu)
        out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={logical_k:5}, align={alignment:3}, '
              f'psum={int(use_psum_layout)}, acc={int(accumulate)}, {out_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * logical_k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')

    print()


def test_cublaslt_gemm() -> None:
    print('Testing cuBLASLt GEMM:')
    use_alpha_options = (False, True)
    for kernel_type, _, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(dtype=torch.bfloat16):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        # BF16 accumulation has lower precision than cuBLASLt's FP32 accumulation
        threshold = 1e-5 if (accumulate and out_dtype == torch.bfloat16) else 6e-7
        for use_alpha in use_alpha_options:
            alpha = random.uniform(-1.0, 1.0) if use_alpha else None
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                 kernel_type, use_bf16=True, alpha=alpha)
            deep_gemm.use_deterministic_algorithms(False)
            deep_gemm.bf16_gemm_nt(a, b, d, c=c, alpha=alpha)
            diff = calc_diff(d, ref_d)
            assert_check(diff < threshold, (f'{diff=}, {use_alpha=}, {alpha=}, '
                                      f'({m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=})'))

        t_nvjet, t_gemv, t_gemm = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
        t = t_nvjet + t_gemv + t_gemm
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()


def test_cublaslt_batched_syrk() -> None:
    print('Testing batched SYRK (default dispatch):')
    for num_batches, m, k, out_dtype in enumerate_batched_syrk_symm():
        out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
        rows, cols = min(m, k), max(m, k)
        threshold = 6e-7
        for k_major in (True, False):
            for batch_shape, padding in (((), (3, 5)), ((num_batches,), (3, 5)), ((num_batches,), (0, 0))):
                shape = (rows, cols) if k_major else (cols, rows)
                a = torch.randn(*batch_shape, shape[0] + padding[0], shape[1] + padding[1],
                                device='cuda', dtype=out_dtype)[..., :shape[0], :shape[1]]
                if not k_major:
                    a = a.mT
                major_opt = 'N' if a.stride(-1) == 1 else 'T'

                d = torch.empty(*batch_shape, rows + padding[0], rows + padding[1],
                                device='cuda', dtype=out_dtype)[..., :rows, :rows]
                deep_gemm.batched_syrk(a, d)
                ref_d = (a.float() @ a.float().mT).to(out_dtype)
                diff = calc_diff(d, ref_d)
                assert_check(diff < threshold, (
                    f'{diff=}, ({batch_shape=}, {padding=}, {rows=}, {cols=}, {major_opt=}, {out_dtype=})'
                ))

            t_nvjet, t_gemv, t_gemm = bench_kineto(
                lambda: deep_gemm.batched_syrk(a, d),
                ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
            t = t_nvjet + t_gemv + t_gemm
            print(f' > Perf (batch={num_batches}, m={rows:6}, n={rows:6}, k={cols:6}, '
                  f'layout={major_opt}, {out_opt}): '
                  f'{t * 1e6:5.0f} us | '
                  f'{2 * num_batches * rows * rows * cols / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_batched_syrk_epilogue() -> None:
    print('Testing batched SYRK epilogue (default dispatch: values; forced native: bitwise symmetry and repeatability):')
    routes = ('default', 'native') if get_arch_major() == 9 else ('default', )

    # Default dispatch may pick cuBLASLt, which gives no symmetry guarantee, so only values are checked there
    def run(route, a, d, **kwargs):
        with batched_syrk_route(route):
            deep_gemm.batched_syrk(a, d, **kwargs)
            if route == 'native':
                d_again = torch.empty_like(d)
                deep_gemm.batched_syrk(a, d_again, **kwargs)
                assert_bitwise_equal(d, d_again, 'native SYRK repeatability')

    for b, n, k, k_major, out_dtype in (
        (1, 64, 128, True, torch.bfloat16),
        (1, 136, 72, False, torch.bfloat16),
        (2, 320, 256, True, torch.bfloat16),
        (3, 520, 1040, True, torch.bfloat16),
        (3, 520, 1040, False, torch.bfloat16),
        (4, 1536, 2048, True, torch.float32),
        (4, 1536, 2048, False, torch.bfloat16),
        (8, 2048, 2048, True, torch.bfloat16),
    ):
        shape = (n, k) if k_major else (k, n)
        a = torch.randn(b, shape[0], shape[1], device='cuda', dtype=torch.bfloat16)
        if not k_major:
            a = a.mT
        ref = (a.float() @ a.float().mT).to(out_dtype)
        for route in routes:
            d = torch.empty(b, n, n, device='cuda', dtype=out_dtype)
            run(route, a, d)
            diff = calc_diff(d, ref)
            assert_check(diff < 1e-5, f'SYRK diff={diff}, case={(route, b, n, k, k_major, out_dtype)}')
            if route == 'native':
                assert_bitwise_equal(d, d.mT, f'native SYRK symmetry {(b, n, k, k_major, out_dtype)}')

    # Edge cases: k=0, n=0, b=0
    for out_dtype in (torch.bfloat16, torch.float32):
        a0 = torch.empty(2, 128, 0, device='cuda', dtype=torch.bfloat16)
        d0 = torch.full((2, 128, 128), 7.0, device='cuda', dtype=out_dtype)
        deep_gemm.batched_syrk(a0, d0)
        assert_check(torch.equal(d0, torch.zeros_like(d0)), 'k=0 did not zero D')

        c0 = torch.randn(2, 128, 128, device='cuda', dtype=out_dtype)
        deep_gemm.batched_syrk(a0, d0, c=c0, alpha=2.0, beta=-1.5)
        assert_check(calc_diff(d0, (-1.5 * c0.float()).to(out_dtype)) < 1e-5, 'k=0 with C mismatch')

        deep_gemm.batched_syrk(torch.empty(2, 0, 128, device='cuda', dtype=torch.bfloat16),
                               torch.empty(2, 0, 0, device='cuda', dtype=out_dtype))
        deep_gemm.batched_syrk(torch.empty(0, 128, 128, device='cuda', dtype=torch.bfloat16),
                               torch.empty(0, 128, 128, device='cuda', dtype=out_dtype))

    # Fused C / alpha / beta epilogue (symmetric C, non-symmetric C, in-place C==D, mixed dtypes, beta=0)
    alpha, beta = 2.0315, -4.7750
    for b, n, k, k_major, d_dtype, c_dtype in (
        (4, 1536, 2048, True, torch.bfloat16, torch.bfloat16),
        (4, 1536, 2048, False, torch.bfloat16, torch.bfloat16),
        (2, 520, 640, True, torch.bfloat16, torch.bfloat16),
        (2, 520, 640, True, torch.bfloat16, torch.float32),
        (2, 520, 640, True, torch.float32, torch.bfloat16),
        (2, 520, 640, True, torch.float32, torch.float32),
        (4, 1536, 2048, True, torch.float32, torch.float32),
    ):
        shape = (n, k) if k_major else (k, n)
        a = torch.randn(b, shape[0], shape[1], device='cuda', dtype=torch.bfloat16) * 0.1
        if not k_major:
            a = a.mT
        # `(C + C.mT) * 0.5` is bitwise symmetric, which the native bitwise symmetry guarantee requires
        c_sym = torch.randn(b, n, n, device='cuda', dtype=c_dtype) * 0.5
        c_sym = ((c_sym.float() + c_sym.float().mT) * 0.5).to(c_dtype)
        c_nonsym = torch.randn(b, n, n, device='cuda', dtype=c_dtype) * 0.5
        c_nan = torch.full((b, n, n), float('nan'), device='cuda', dtype=c_dtype)
        gram = a.float() @ a.float().mT
        case = (b, n, k, k_major, d_dtype, c_dtype)
        for route in routes:
            d = torch.empty(b, n, n, device='cuda', dtype=d_dtype)
            run(route, a, d, c=c_sym, alpha=alpha, beta=beta)
            ref = (alpha * gram + beta * c_sym.float()).to(d_dtype)
            assert_check(calc_diff(d, ref) < 1e-4, f'Fused symmetric C mismatch: {(route, ) + case}')
            if route == 'native':
                assert_bitwise_equal(d, d.mT, f'native fused symmetric C symmetry {case}')
            d_sym = d.clone()

            run(route, a, d, c=c_nonsym, alpha=alpha, beta=beta)
            ref_ns = (alpha * gram + beta * c_nonsym.float()).to(d_dtype)
            assert_check(calc_diff(d, ref_ns) < 1e-4, f'Fused non-symmetric C mismatch: {(route, ) + case}')

            if d_dtype == c_dtype:
                d_inplace = c_sym.clone()
                with batched_syrk_route(route):
                    deep_gemm.batched_syrk(a, d_inplace, c=d_inplace, alpha=alpha, beta=beta)
                assert_check(calc_diff(d_inplace, ref) < 1e-4, f'In-place C==D mismatch: {(route, ) + case}')
                if route == 'native':
                    assert_bitwise_equal(d_inplace, d_sym, f'native in-place C==D {case}')

            # beta=0 must ignore C even if C contains NaNs
            run(route, a, d, c=c_nan, alpha=alpha, beta=0.0)
            assert_check(not torch.isnan(d).any().item(), f'beta=0 produced NaN from C: {(route, ) + case}')
            assert_check(calc_diff(d, (alpha * gram).to(d_dtype)) < 1e-4, f'beta=0 result mismatch: {(route, ) + case}')

    # Muon Newton-Schulz Step 1 + fused Step 2 (B = b * A + c * (A @ A))
    b_ns, c_ns = -4.7750, 2.0315
    for batch, n, k in ((4, 576, 5120), (4, 1536, 5120)):
        x = torch.randn(batch, n, k, device='cuda', dtype=torch.bfloat16) / (k ** 0.5)
        for route in routes:
            gram = torch.empty(batch, n, n, device='cuda', dtype=torch.bfloat16)
            run(route, x, gram)
            ref_gram = (x.float() @ x.float().mT).to(torch.bfloat16)
            assert_check(calc_diff(gram, ref_gram) < 1e-5, f'Muon step 1 mismatch: {(route, batch, n, k)}')
            if route == 'native':
                assert_bitwise_equal(gram, gram.mT, f'native Muon step 1 symmetry {(batch, n, k)}')
            poly = torch.empty_like(gram)
            run(route, gram, poly, c=gram, alpha=c_ns, beta=b_ns)
            ref_poly = (b_ns * gram.float() + c_ns * (gram.float() @ gram.float().mT)).to(torch.bfloat16)
            assert_check(calc_diff(poly, ref_poly) < 1e-4, f'Muon step 2 mismatch: {(route, batch, n, k)}')
            if route == 'native':
                assert_bitwise_equal(poly, poly.mT, f'native Muon step 2 symmetry {(batch, n, k)}')
    print(' > All batched SYRK epilogue tests passed!\n')


@contextlib.contextmanager
def batched_syrk_env(**kwargs):
    old = {key: os.environ.get(key) for key in kwargs}
    os.environ.update({key: str(value) for key, value in kwargs.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key)
            else:
                os.environ[key] = value


def batched_syrk_route(route: str):
    # `DG_SYRK_FORCE_*` are internal test controls; 'default' keeps the normal dispatch
    if route == 'default':
        return contextlib.nullcontext()
    return batched_syrk_env(**{'DG_SYRK_FORCE_NATIVE' if route == 'native' else 'DG_SYRK_FORCE_FALLBACK': 1})


def batched_syrk_rejects(message: str, *args, **kwargs) -> bool:
    try:
        deep_gemm.batched_syrk(*args, **kwargs)
    except RuntimeError as e:
        return message in str(e)
    return False


def test_batched_syrk_routes() -> None:
    print('Testing batched SYRK routes (forced native and forced cuBLASLt fallback):')

    # Small integer `A` and `C` values with a few fractional bits keep every FP32 intermediate exact, so
    # results must match bitwise: a missing tile or rounding `C` before `beta` shows up as a mismatch
    def make_a(b, n, k, k_major):
        a = torch.randint(-2, 3, (b, n, k) if k_major else (b, k, n), device='cuda').to(torch.bfloat16)
        return a if k_major else a.mT

    def make_c(*shape, dtype):
        c = torch.randint(-4, 4, shape, device='cuda') + torch.randint(0, 512, shape, device='cuda') * 2 ** -9 + 2 ** -10
        return c.to(dtype)

    def ref_syrk(a, c, alpha, beta, dtype):
        ref = alpha * (a.float() @ a.float().mT)
        return (ref if c is None or beta == 0 else ref + beta * c.float()).to(dtype)

    alpha, beta = 2.0, 1.5
    routes = ('native', 'fallback') if get_arch_major() == 9 else ('fallback', )
    for route in routes:
        # Odd tile-row counts, partial `n`/`k` tails, and an odd number of tiles (unpaired final CTA)
        blocks = ((64, 64), (128, 128), (128, 256)) if route == 'native' else ((0, 0), )
        for b, n, k, k_major, d_dtype, c_dtype in (
            (3, 200, 136, True, torch.bfloat16, None),
            (2, 264, 72, False, torch.bfloat16, torch.bfloat16),
            (1, 392, 200, True, torch.bfloat16, torch.float),
            (5, 136, 64, True, torch.float, torch.bfloat16),
            (2, 520, 128, False, torch.float, torch.float),
        ):
            for block_m, block_n in blocks:
                if block_n == 256 and d_dtype != torch.bfloat16:
                    continue
                a = make_a(b, n, k, k_major)
                c = None if c_dtype is None else make_c(b, n, n, dtype=c_dtype)
                d = torch.empty(b, n, n, device='cuda', dtype=d_dtype)
                with batched_syrk_route(route), batched_syrk_env(DG_SYRK_BLOCK_M=block_m, DG_SYRK_BLOCK_N=block_n):
                    deep_gemm.batched_syrk(a, d, c=c, alpha=alpha, beta=beta)
                assert_bitwise_equal(d, ref_syrk(a, c, alpha, beta, d_dtype), f'{route=}, {(b, n, k, k_major, d_dtype, c_dtype, block_m, block_n)}')

        with batched_syrk_route(route):
            # Misaligned `C` views: a one-element offset with padded rows, and an odd batch stride
            b, n, k = 3, 128, 64
            a = make_a(b, n, k, True)
            for c_dtype in (torch.bfloat16, torch.float):
                c_offset = make_c(b, n, n + 3, dtype=c_dtype)[..., 1:n + 1]
                c_odd_batch = make_c(b * (n * n + 1), dtype=c_dtype).as_strided((b, n, n), (n * n + 1, n, 1))
                for c in (c_offset, c_odd_batch):
                    for d_dtype in (torch.bfloat16, torch.float):
                        d = torch.empty(b, n, n, device='cuda', dtype=d_dtype)
                        deep_gemm.batched_syrk(a, d, c=c, alpha=alpha, beta=beta)
                        assert_bitwise_equal(d, ref_syrk(a, c, alpha, beta, d_dtype), f'{route=}, misaligned C, {c_dtype=}, {d_dtype=}')

            # Pointer-only offsets of `A`, `C`, or `D` with aligned pitches and batch strides
            # (only `C` can be offset on the native route, as `A` and `D` must be TMA-encodable)
            for offset_idx in (range(3) if route == 'fallback' else (1, )):
                def make_view(buf, rows, cols, offset):
                    return buf[offset:offset + b * rows * (cols + 64)].view(b, rows, cols + 64)[..., :cols]
                a_buf = make_a(1, 1, b * n * (k + 64) + 8, True).flatten()
                c_buf = make_c(b * n * (n + 64) + 8, dtype=torch.bfloat16)
                d_buf = torch.empty(b * n * (n + 64) + 8, device='cuda', dtype=torch.bfloat16)
                a_view = make_view(a_buf, n, k, int(offset_idx == 0))
                c_view = make_view(c_buf, n, n, int(offset_idx == 1))
                d_view = make_view(d_buf, n, n, int(offset_idx == 2))
                ref = ref_syrk(a_view, c_view, alpha, beta, torch.bfloat16)
                deep_gemm.batched_syrk(a_view, d_view, c=c_view, alpha=alpha, beta=beta)
                assert_bitwise_equal(d_view, ref, f'{route=}, pointer offset on operand {"ACD"[offset_idx]}')

            # FP32 `C` with a BF16 `D` (the fallback casts an FP32 temporary) on a non-default stream
            stream = torch.cuda.Stream()
            c = make_c(b, n, n, dtype=torch.float)
            d = torch.empty(b, n, n, device='cuda', dtype=torch.bfloat16)
            ref = ref_syrk(a, c, alpha, beta, torch.bfloat16)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                deep_gemm.batched_syrk(a, d, c=c, alpha=alpha, beta=beta)
            torch.cuda.current_stream().wait_stream(stream)
            assert_bitwise_equal(d, ref, f'{route=}, FP32 C -> BF16 D on a side stream')

            # Batches interleaved across rows: distinct elements despite failing both fast overlap checks
            d_interleaved = torch.empty(256, device='cuda', dtype=torch.bfloat16).as_strided((2, 8, 8), (32, 24, 1))
            a_small = make_a(2, 8, 16, True)
            deep_gemm.batched_syrk(a_small, d_interleaved, alpha=alpha)
            assert_bitwise_equal(d_interleaved, ref_syrk(a_small, None, alpha, 0.0, torch.bfloat16), f'{route=}, interleaved D')

            # Asymmetric in-place `C == D`
            for dtype in (torch.bfloat16, torch.float):
                c = make_c(b, n, n, dtype=dtype)
                d = c.clone()
                deep_gemm.batched_syrk(a, d, c=d, alpha=alpha, beta=beta)
                assert_bitwise_equal(d, ref_syrk(a, c, alpha, beta, dtype), f'{route=}, in-place C, {dtype=}')

            # `beta == 0` ignores NaN `C`, `alpha == 0` ignores NaN `A`
            c_nan = torch.full((b, n, n), float('nan'), device='cuda', dtype=torch.float)
            a_nan = torch.full((b, n, k), float('nan'), device='cuda', dtype=torch.bfloat16)
            c = make_c(b, n, n, dtype=torch.float)
            d = torch.empty(b, n, n, device='cuda', dtype=torch.bfloat16)
            deep_gemm.batched_syrk(a, d, c=c_nan, alpha=alpha, beta=0.0)
            assert_bitwise_equal(d, ref_syrk(a, None, alpha, 0.0, torch.bfloat16), f'{route=}, beta=0 with NaN C')
            deep_gemm.batched_syrk(a_nan, d, c=c, alpha=0.0, beta=beta)
            assert_bitwise_equal(d, (beta * c).to(torch.bfloat16), f'{route=}, alpha=0 with NaN A')

    # Forcing the native route on an unsupported view must fail instead of silently falling back
    if get_arch_major() == 9:
        a = torch.randn(1, 128, 72, device='cuda', dtype=torch.bfloat16)[..., 1:65]
        d = torch.empty(1, 128, 128, device='cuda', dtype=torch.bfloat16)
        with batched_syrk_route('native'):
            assert_check(batched_syrk_rejects('native SM90 batched SYRK is not applicable', a, d),
                         'Forced native SYRK accepted a TMA-misaligned A')

        # Cluster-2 multicast with an unpaired final tile: 3 * 45 64x64 tiles = 135 tiles on 132 SMs
        a = make_a(3, 520, 136, True)
        c = make_c(3, 520, 520, dtype=torch.bfloat16)
        d = torch.empty(3, 520, 520, device='cuda', dtype=torch.bfloat16)
        with batched_syrk_route('native'), batched_syrk_env(DG_SYRK_BLOCK_M=64, DG_SYRK_BLOCK_N=64, DG_SYRK_CLUSTER_SIZE=2):
            deep_gemm.batched_syrk(a, d, c=c, alpha=alpha, beta=beta)
        assert_bitwise_equal(d, ref_syrk(a, c, alpha, beta, torch.bfloat16), 'cluster-2 native with an unpaired final tile')

    # Aliasing: partial `C`/`D` overlap, `A`/`D` overlap, and overlapping `D` batches are rejected,
    # while disjoint column windows of one padded buffer are accepted
    b, n, k = 2, 128, 64
    a = make_a(b, n, k, True)
    buf = make_c(b, n, n + 8, dtype=torch.bfloat16)
    assert_check(batched_syrk_rejects('C must be exactly D or not overlap D', a, buf[..., 8:], c=buf[..., :n]),
                 'Partial C/D overlap was accepted')
    d = torch.empty(b, n, n, device='cuda', dtype=torch.bfloat16)
    assert_check(batched_syrk_rejects('A must not overlap D', d[..., :k], d), 'A/D overlap was accepted')
    for d_overlap in (torch.empty(n * n + n, device='cuda', dtype=torch.bfloat16).as_strided((b, n, n), (n, n, 1)),
                      # Neighbour of the accepted interleaved layout `(32, 24, 1)` whose rows do overlap
                      torch.empty(256, device='cuda', dtype=torch.bfloat16).as_strided((2, 8, 8), (28, 24, 1))):
        a_overlap = make_a(d_overlap.size(0), d_overlap.size(1), 16, True)
        assert_check(batched_syrk_rejects('D must not have overlapping elements', a_overlap, d_overlap),
                     f'Overlapping D elements were accepted: {d_overlap.stride()}')
        assert_check(batched_syrk_rejects('D must not have overlapping elements', a_overlap[..., :0], d_overlap),
                     f'Overlapping D elements were accepted with k=0: {d_overlap.stride()}')
    buf = make_c(b, n, 2 * n, dtype=torch.bfloat16)
    c, d = buf[..., :n], buf[..., n:]
    ref = ref_syrk(a, c, alpha, beta, torch.bfloat16)
    deep_gemm.batched_syrk(a, d, c=c, alpha=alpha, beta=beta)
    assert_bitwise_equal(d, ref, 'Disjoint padded C/D views')
    print(' > All batched SYRK route tests passed!\n')


def test_cublaslt_batched_symm() -> None:
    print('Testing cuBLASLt batched SYMM:')
    for num_batches, m, k, out_dtype in enumerate_batched_syrk_symm():
        out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
        rows, cols = min(m, k), max(m, k)
        threshold = 6e-7
        for k_major_a, k_major_b in ((True, True), (True, False), (False, True), (False, False)):
            for batch_shape, padding in (((), (3, 5)), ((num_batches,), (3, 5)), ((num_batches,), (0, 0))):
                a = torch.randn(*batch_shape, rows + padding[0], rows + padding[1],
                                device='cuda', dtype=out_dtype)[..., :rows, :rows]
                if not k_major_a:
                    a = a.mT
                a.copy_(a + a.mT)

                shape_b = (rows, cols) if k_major_b else (cols, rows)
                b = torch.randn(*batch_shape, shape_b[0] + padding[0], shape_b[1] + padding[1],
                                device='cuda', dtype=out_dtype)[..., :shape_b[0], :shape_b[1]]
                if not k_major_b:
                    b = b.mT
                major_opt  = 'N' if a.stride(-1) == 1 else 'T'
                major_opt += 'N' if b.stride(-1) == 1 else 'T'

                d = torch.empty(*batch_shape, rows + padding[0], cols + padding[1],
                                device='cuda', dtype=out_dtype)[..., :rows, :cols]
                deep_gemm.batched_symm(a, b, d)
                ref_d = (a.float() @ b.float()).to(out_dtype)
                diff = calc_diff(d, ref_d)
                assert_check(diff < threshold, (
                    f'{diff=}, ({batch_shape=}, {padding=}, {rows=}, {cols=}, {major_opt=}, {out_dtype=})'
                ))

            t_nvjet, t_gemv, t_gemm = bench_kineto(
                lambda: deep_gemm.batched_symm(a, b, d),
                ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
            t = t_nvjet + t_gemv + t_gemm
            print(f' > Perf (batch={num_batches}, m={rows:6}, n={cols:6}, k={rows:6}, '
                  f'layout={major_opt}, {out_opt}): '
                  f'{t * 1e6:5.0f} us | '
                  f'{2 * num_batches * rows * rows * cols / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
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
    test_cublaslt_batched_syrk()
    test_batched_syrk_epilogue()
    test_batched_syrk_routes()
    test_cublaslt_batched_symm()
