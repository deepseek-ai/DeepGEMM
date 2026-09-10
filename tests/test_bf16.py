import numpy as np
import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes
)
from utils import (
    assert_direct_output_matches_fp32_accumulation,
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
    for kernel_type, _, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal(torch.bfloat16):
        deep_gemm.use_deterministic_algorithms(True)
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'

        for test_alias in (False, True):
            for use_alpha in use_alpha_options:
                alpha = random.uniform(-1.0, 1.0) if use_alpha else None
                a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                     kernel_type, use_bf16=True, alpha=alpha)
                func_name = f'bf16_gemm_{major_opt.lower() if test_alias else "nt"}'
                if test_alias:
                    a = a if major_a.is_k_major() else a.T
                    b = b if major_b.is_k_major() else b.T
                    assert a.is_contiguous() and b.is_contiguous()
                getattr(deep_gemm, func_name)(a, b, d, c=c, alpha=alpha)
                diff = calc_diff(d, ref_d)
                assert diff < 1e-5, (f'{m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                       f'{use_alpha=}, {alpha=}, {diff:.5f}, alias={test_alias}')
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_bf16=True)

        t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), 'bf16_gemm', suppress_kineto_output=True)
        deep_gemm.use_deterministic_algorithms(False)
        cublas_t, split_k_t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), ('nvjet', 'reduce'), suppress_kineto_output=True)
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
            assert diff < 1e-5, f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}, alias={test_alias}, {ensure_zero_padding=}'
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
            assert diff < 1e-5, f'{max_m=}, {n=}, {k=}, {num_groups=}, {diff:.5f}'


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
                    assert diff < 1e-5, (f'{m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                                        f'{test_host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, '
                                        f'{out_dtype=}, {diff:.7f}')
                else:
                    case_label = (f'BF16 K-grouped direct output, {m=}, {n=}, {total_k=}, '
                                  f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, '
                                  f'{out_dtype=}')
                    assert_direct_output_matches_fp32_accumulation(
                        d,
                        lambda output, accumulator: deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
                            a, b, output, test_host_ks_cpu, grouped_layout, accumulator,
                            use_psum_layout=use_psum_layout),
                        case_label)

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
            assert diff < threshold, (f'{diff=}, {use_alpha=}, {alpha=}, '
                                      f'({m=}, {n=}, {k=}, {major_opt=}, {accumulate=}, {out_dtype=})')

        t_nvjet, t_gemv, t_gemm = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d, c=c), ('nvjet', 'gemv', 'gemm'), suppress_kineto_output=True)
        t = t_nvjet + t_gemv + t_gemm
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:5.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()


def test_cublaslt_batched_syrk() -> None:
    print('Testing cuBLASLt batched SYRK:')
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
                assert diff < threshold, (
                    f'{diff=}, ({batch_shape=}, {padding=}, {rows=}, {cols=}, {major_opt=}, {out_dtype=})'
                )

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
                assert diff < threshold, (
                    f'{diff=}, ({batch_shape=}, {padding=}, {rows=}, {cols=}, {major_opt=}, {out_dtype=})'
                )

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
    test_cublaslt_batched_symm()
