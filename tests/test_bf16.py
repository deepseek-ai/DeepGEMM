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


def sm120_bf16_matrix(values, padded=False, offset=0, extra=5, tma=True):
    from sm120_test_storage import native_matrix
    return native_matrix(values, padded, offset, extra, tma)


def exercise_sm120_bf16_native(layout, shape, out_dtype, alpha, c_mode,
                                padded=False, offset=0, graph=False, pdl=False):
    assert get_arch_major() == 12
    assert layout in ('nt', 'nn', 'tn', 'tt') and c_mode in ('none', 'same', 'different')
    m, n, k = shape
    generator = torch.Generator(device='cpu').manual_seed(1907 + m + n + k)
    a_values = (torch.randn((m, k), generator=generator) * 0.25).to(torch.bfloat16)
    b_values = (torch.randn((k, n), generator=generator) * 0.25).to(torch.bfloat16)
    c_values = torch.randn((m, n), generator=generator).to(out_dtype)
    a_physical = a_values if layout[0] == 'n' else a_values.T.contiguous()
    b_physical = b_values if layout[1] == 'n' else b_values.T.contiguous()
    a, a_storage = sm120_bf16_matrix(a_physical, padded, offset)
    b, b_storage = sm120_bf16_matrix(b_physical, padded, offset)
    d, d_storage = sm120_bf16_matrix(torch.zeros_like(c_values), padded, offset, extra=7)
    if c_mode == 'different':
        c, c_storage = sm120_bf16_matrix(c_values, padded, offset, extra=11, tma=False)
        assert c.data_ptr() != d.data_ptr()
        if padded:
            assert c.stride(0) != d.stride(0)
    else:
        c, c_storage = (d, d_storage) if c_mode == 'same' else (None, None)
    initial_c = c_values.cuda()
    d_valid = torch.zeros_like(d_storage, dtype=torch.bool)
    d_valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    if padded:
        assert a.stride(0) > a.size(1) and b.stride(0) > b.size(1)
        assert d.stride(0) > n
    if offset:
        assert all(t.storage_offset() * t.element_size() == offset * 16 for t in (a, b, d))
    function = getattr(deep_gemm, f'bf16_gemm_{layout}')

    def run():
        if c_mode == 'same':
            d.copy_(initial_c)
        function(a, b, d, c=c, alpha=alpha)

    def reference():
        result = a_values.float() @ b_values.float()
        result *= 1.0 if alpha is None else alpha
        if c_mode != 'none':
            result += c_values.float()
        return result

    def check():
        actual = d.cpu().float()
        expected = reference()
        assert torch.isfinite(actual).all()
        if alpha == 0:
            exact = torch.zeros_like(c_values) if c_mode == 'none' else c_values
            torch.testing.assert_close(d.cpu(), exact, rtol=0, atol=0)
        else:
            rtol, atol = (0.008, 0.015) if out_dtype == torch.bfloat16 else (2e-5, 2e-5)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            diff = calc_diff(actual, expected)
            assert diff < (1e-5 if out_dtype == torch.bfloat16 else 1e-10), f'{diff=}'
        assert (d_storage[~d_valid] == 19).all(), 'Output store touched padding or offset guards'
        torch.testing.assert_close(a.cpu(), a_values if layout[0] == 'n' else a_values.T, rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), b_values if layout[1] == 'n' else b_values.T, rtol=0, atol=0)
        if c_mode == 'different':
            torch.testing.assert_close(c.cpu(), c_values, rtol=0, atol=0)

    original_pdl = deep_gemm.get_pdl()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_pdl(pdl)
        for _ in range(3):
            d.fill_(float('nan'))
            run()
            check()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            check()
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in range(3):
                if phase:
                    a_values = (-0.5 * a_values.float()).to(torch.bfloat16)
                    b_values = (b_values.float() + 0.25).to(torch.bfloat16)
                    c_values = (-c_values.float()).to(out_dtype)
                    a.copy_(a_values if layout[0] == 'n' else a_values.T)
                    b.copy_(b_values if layout[1] == 'n' else b_values.T)
                    initial_c.copy_(c_values)
                    if c_mode == 'different':
                        c.copy_(c_values)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(original_pdl)
        # This fixture starts and ends in the known nondeterministic default state.
        deep_gemm.use_deterministic_algorithms(False)
    print(f' > Native SM120 BF16: {layout=}, {shape=}, {out_dtype=}, {alpha=}, '
          f'{c_mode=}, {padded=}, {offset=}, {graph=}, {pdl=}')


def test_sm120_bf16_native_layouts_alpha():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            for alpha in (None, 0.5, -1.0, 2.0):
                for c_mode in ('none', 'same', 'different'):
                    exercise_sm120_bf16_native(layout, (128, 128, 256), out_dtype, alpha, c_mode)


def test_sm120_bf16_native_strided_tails():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            shapes = ((1, 8, 13), (65, 18, 67), (129, 34, 129), (2049, 66, 65))
            for m, n, k in shapes:
                native_k = k if layout == 'nt' else (k + 7) // 8 * 8
                exercise_sm120_bf16_native(layout, (m, n, native_k), out_dtype, -1.0, 'different', padded=True, offset=1)
            exercise_sm120_bf16_native(layout, (67, 30, 63 if layout == 'nt' else 64), out_dtype, 0.5, 'same', padded=True)


def test_sm120_bf16_native_alpha_zero():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            for c_mode in ('none', 'same', 'different'):
                exercise_sm120_bf16_native(layout, (65, 18, 67 if layout == 'nt' else 72), out_dtype, 0.0, c_mode, padded=True, offset=1)


def test_sm120_bf16_native_graph_pdl():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            for pdl in (False, True):
                for c_mode, alpha in (('none', 2.0), ('same', 0.5), ('different', -1.0), ('same', 0.0)):
                    exercise_sm120_bf16_native(layout, (65, 34, 67 if layout == 'nt' else 72), out_dtype, alpha, c_mode,
                                                padded=True, offset=1, graph=True, pdl=pdl)


def sm120_grouped_intervals(lengths, alignment):
    intervals, end = [], 0
    for length in lengths:
        start = (end + alignment - 1) // alignment * alignment
        end = start + length
        intervals.append((start, end))
    return intervals


def exercise_sm120_grouped_bf16(mode, alignment, groups, n=64, k=72,
                                 zero_padding=False, all_empty=False, graph=False, pdl=False, nn=False):
    assert get_arch_major() == 12 and mode in ('labels', 'psum', 'masked')
    assert alignment in (32, 64, 128) and not (mode == 'masked' and (nn or zero_padding))
    expected_m = {32: 32, 64: 33, 128: 65}[alignment]
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m) == alignment
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_pdl = deep_gemm.get_pdl()
    max_m = alignment + 3
    capacity = groups * 2 * alignment + alignment
    m = max_m if mode == 'masked' else capacity
    generator = torch.Generator().manual_seed(731 + groups + alignment)
    a_shape = (groups, m, k) if mode == 'masked' else (m, k)
    original_a = (torch.randn(a_shape, generator=generator) * 0.25).to(torch.bfloat16)
    original_b = (torch.randn((groups, n, k), generator=generator) * 0.25).to(torch.bfloat16)
    a = original_a.cuda()
    b = (original_b.transpose(1, 2).contiguous() if nn else original_b).cuda()
    if mode == 'masked':
        storage = torch.full((8 + groups * m * n + 16,), 19, dtype=torch.bfloat16, device='cuda')
        d = storage.as_strided((groups, m, n), (m * n, n, 1), 8)
    else:
        d, storage = sm120_bf16_matrix(torch.zeros((m, n), dtype=torch.bfloat16), True, 1, extra=7)
    guard = torch.zeros_like(storage, dtype=torch.bool)
    guard.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    metadata = torch.empty((groups if mode != 'labels' else m,), dtype=torch.int32, device='cuda')

    def inputs(phase):
        lengths = [0 if all_empty or phase == 1 or g % 3 == 0 else
                   (alignment + 1 if g % 3 == 1 else alignment - 3) for g in range(groups)]
        if phase == 2 and not all_empty:
            lengths = [alignment - 1 if g % 2 == 0 else 0 for g in range(groups)]
        valid = torch.zeros(d.shape[:-1], dtype=torch.bool)
        av = original_a.clone() * (-1 if phase == 2 else 1)
        bv = original_b.clone() * (0.5 if phase == 2 else 1)
        expected = torch.full(d.shape, 0.0 if zero_padding else 7.0, dtype=torch.float32)
        if mode == 'masked':
            meta = lengths
            for g, length in enumerate(lengths):
                valid[g, :length] = True
                expected[g, :length] = av[g, :length].float() @ bv[g].float().T
        else:
            intervals = sm120_grouped_intervals(lengths, alignment)
            labels = torch.full((m,), -1, dtype=torch.int32)
            for g, (start, end) in enumerate(intervals):
                assert start % alignment == 0 and end <= m
                labels[start:end] = g
                valid[start:end] = True
                expected[start:end] = av[start:end].float() @ bv[g].float().T
            meta = labels if mode == 'labels' else [end for _, end in intervals]
        av[~valid] = float('nan')
        a.copy_(av)
        b.copy_(bv.transpose(1, 2) if nn else bv)
        metadata.copy_(torch.as_tensor(meta, dtype=torch.int32))
        return expected, valid

    def run():
        if mode == 'masked':
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, metadata, expected_m)
        else:
            kwargs = dict(use_psum_layout=mode == 'psum', ensure_zero_padding=zero_padding)
            if mode == 'psum' and not nn:
                kwargs['expected_m_for_psum_layout'] = 1
            getattr(deep_gemm, f'm_grouped_bf16_gemm_{"nn" if nn else "nt"}_contiguous')(
                a, b, d, metadata, **kwargs)

    def check(expected, valid):
        actual = d.cpu().float()
        assert torch.isfinite(actual[valid]).all()
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0.008, atol=0.015)
        if zero_padding:
            zero_rows = ~valid
            if mode == 'psum':
                zero_rows = torch.zeros_like(valid)
                for end in metadata.cpu().tolist():
                    zero_rows[end:(end + alignment - 1) // alignment * alignment] = True
            torch.testing.assert_close(actual[zero_rows], torch.zeros_like(actual[zero_rows]), rtol=0, atol=0)
        assert (storage[~guard] == 19).all(), 'Grouped output modified storage outside D'

    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(pdl)
        expected, valid = inputs(0)
        for _ in range(3):
            d.fill_(7)
            run()
            check(expected, valid)
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in (0, 1, 2):
                expected, valid = inputs(phase)
                d.fill_(7)
                captured.replay()
                torch.cuda.synchronize()
                check(expected, valid)
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
    print(f' > Native SM120 grouped BF16: {mode=}, {alignment=}, {groups=}, {n=}, {k=}, '
          f'{zero_padding=}, {all_empty=}, {graph=}, {pdl=}, {nn=}')


def test_sm120_grouped_bf16_alignment():
    assert get_arch_major() == 12
    old = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        for expected, alignment in ((None, 128), (0, 32), (1, 32), (32, 32), (33, 64), (64, 64), (65, 128)):
            actual = (deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout() if expected is None else
                      deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected))
            assert actual == alignment
            deep_gemm.set_mk_alignment_for_contiguous_layout(actual)
            assert deep_gemm.get_mk_alignment_for_contiguous_layout() == alignment
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old)


def test_sm120_grouped_bf16_contiguous():
    for alignment in (32, 64, 128):
        for mode in ('labels', 'psum'):
            for nn in (False, True):
                for zero_padding in (False, True):
                    for groups in (3, 5):
                        exercise_sm120_grouped_bf16(mode, alignment, groups, n=64 if groups == 3 else 72,
                                                    zero_padding=zero_padding, nn=nn)


def test_sm120_grouped_bf16_masked():
    for alignment in (32, 64, 128):
        for groups in (3, 5):
            for all_empty in (False, True):
                exercise_sm120_grouped_bf16('masked', alignment, groups, all_empty=all_empty)


def test_sm120_grouped_bf16_graph_padding():
    for alignment in (32, 64, 128):
        for mode in ('labels', 'psum', 'masked'):
            for pdl in (False, True):
                exercise_sm120_grouped_bf16(mode, alignment, 5, zero_padding=pdl and mode != 'masked',
                                            graph=True, pdl=pdl, nn=mode == 'psum')


def test_sm120_grouped_bf16_empty():
    for alignment in (32, 64, 128):
        for mode in ('labels', 'psum'):
            exercise_sm120_grouped_bf16(mode, alignment, 3, zero_padding=True, all_empty=True)
    a = torch.empty((0, 67), device='cuda', dtype=torch.bfloat16)
    b = torch.empty((3, 17, 67), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((0, 17), device='cuda', dtype=torch.bfloat16)
    for psum in (False, True):
        metadata = torch.zeros(3 if psum else 0, device='cuda', dtype=torch.int32)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, metadata, use_psum_layout=psum,
                                                  ensure_zero_padding=True)
        assert d.numel() == 0


def sm120_k_grouped_lengths(groups, alignment, psum, phase=0, empty=False):
    lengths = [((1, 127, 129, 0)[g % 4] if psum else (alignment, 0, 2 * alignment, 0)[g % 4])
               for g in range(groups)]
    if empty or (psum and phase == 1):
        return [0] * groups
    if psum and phase == 2:
        return list(reversed(lengths))
    return lengths


def exercise_sm120_k_grouped(api, groups, alignment, psum, out_dtype, c_mode,
                             shape=(48, 80), gran=128, packed=False, graph=False,
                             pdl=False, empty=False, ks_mode='list', default_recipe=False):
    assert get_arch_major() == 12 and api in ('bf16_tn', 'fp8_tn', 'fp8_nt')
    assert groups in (1, 3, 8) and alignment in (128, 256)
    assert not (api == 'fp8_nt' and (psum or alignment != 128))
    assert not packed or (api != 'bf16_tn' and gran == 32)
    assert c_mode in ('none', 'same', 'different') and ks_mode in ('list', 'none', 'empty')
    assert psum or ks_mode == 'list'
    assert not default_recipe or gran == 128
    m, n = shape
    quant = api != 'bf16_tn'
    assert not quant or (m % 4 == 0 and n % 4 == 0)
    lengths0 = sm120_k_grouped_lengths(groups, alignment, psum, empty=empty)
    padded0 = [(length + alignment - 1) // alignment * alignment for length in lengths0]
    total = sum(padded0)
    ks_cpu = padded0 if not psum or ks_mode == 'list' else (None if ks_mode == 'none' else [])
    assert not (graph and psum and ks_mode == 'list')
    dtype = torch.float8_e4m3fn if quant else torch.bfloat16
    a_shape, b_shape = ((m, total), (n, total)) if api == 'fp8_nt' else ((total, m), (total, n))
    a, b = torch.empty(a_shape, device='cuda', dtype=dtype), torch.empty(b_shape, device='cuda', dtype=dtype)
    sa = torch.empty((total // gran, m), device='cuda') if quant else None
    sb = torch.empty((total // gran, n), device='cuda') if quant else None
    metadata = torch.empty(groups, dtype=torch.int32, device='cuda')
    storage = torch.full((groups * m * n + 32,), 19, dtype=out_dtype, device='cuda')
    d = storage[16:-16].view(groups, m, n)
    c = torch.empty_like(d) if c_mode == 'different' else (d if c_mode == 'same' else None)
    initial_c = torch.empty_like(d)

    def inputs(phase):
        lengths = sm120_k_grouped_lengths(groups, alignment, psum, phase, empty)
        padded = [(length + alignment - 1) // alignment * alignment for length in lengths]
        assert sum(padded) <= total
        av, bv = torch.zeros((total, m), dtype=dtype), torch.zeros((total, n), dtype=dtype)
        sa_cpu, sb_cpu = torch.ones((total // gran, m)), torch.ones((total // gran, n))
        cv = (((torch.arange(groups * m * n).view(groups, m, n) + phase * 3) % 17 - 8).float() / 16).to(out_dtype)
        expected = torch.zeros((groups, m, n)) if c_mode == 'none' else cv.float().clone()
        meta, cursor, sf_cursor, word_prefix = [], 0, 0, [0]
        nt_a, nt_b = [], []
        for g, (length, capacity) in enumerate(zip(lengths, padded)):
            dim = torch.arange(capacity)[:, None]
            ag = (((dim * 3 + torch.arange(m)[None, :] * 5 + g * 7 + phase) % 15 - 7).float() / 4).to(dtype)
            bg = (((dim * 5 + torch.arange(n)[None, :] * 3 + g * 2 + phase * 2) % 13 - 6).float() / 4).to(dtype)
            ag[length:], bg[length:] = 0, 0
            ar, br = ag.float(), bg.float()
            if quant:
                rows = capacity // gran
                sga = torch.pow(2.0, ((torch.arange(rows)[:, None] + torch.arange(m)[None, :] + g + phase) % 3 - 3).float())
                sgb = torch.pow(2.0, ((torch.arange(rows)[:, None] * 2 + torch.arange(n)[None, :] + g * 2 + phase) % 3 - 3).float())
                sa_cpu[sf_cursor:sf_cursor + rows] = sga
                sb_cpu[sf_cursor:sf_cursor + rows] = sgb
                ar *= sga[torch.arange(capacity) // gran]
                br *= sgb[torch.arange(capacity) // gran]
                sf_cursor += rows
                word_prefix.append(word_prefix[-1] + (rows + 3) // 4)
            expected[g] += ar[:length].T @ br[:length]
            av[cursor:cursor + capacity], bv[cursor:cursor + capacity] = ag, bg
            nt_a.append(ag.T.contiguous().flatten())
            nt_b.append(bg.T.contiguous().flatten())
            meta.append(cursor + length if psum else capacity)
            cursor += capacity
        if api == 'fp8_nt':
            av, bv = torch.cat(nt_a).view(m, total), torch.cat(nt_b).view(n, total)
        a.copy_(av)
        b.copy_(bv)
        metadata.copy_(torch.tensor(meta, dtype=torch.int32))
        initial_c.copy_(cv)
        if c_mode == 'different':
            c.copy_(cv)
        if quant:
            sa.copy_(sa_cpu)
            sb.copy_(sb_cpu)
        return expected, cv, lengths, word_prefix

    function = getattr(deep_gemm, f'k_grouped_{"bf16" if not quant else "fp8"}_gemm_{"nt" if api == "fp8_nt" else "tn"}_contiguous')

    def run():
        aa, bb = a, b
        if quant:
            sf_a, sf_b = sa, sb
            if packed and total:
                sf_a = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    sa, metadata, ks_cpu, gran, alignment, psum)
                sf_b = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    sb, metadata, ks_cpu, gran, alignment, psum)
                for sf, mn in ((sf_a, m), (sf_b, n)):
                    assert sf.dtype == torch.int32 and sf.is_contiguous()
                    assert sf.stride() == (mn, 1) and sf.shape[1] == mn
            aa, bb = (a, sf_a), (b, sf_b)
        if c_mode == 'same':
            d.copy_(initial_c)
        kwargs = {} if not quant or default_recipe else dict(recipe=(1, 1, gran))
        function(aa, bb, d, ks_cpu, metadata, c=c, use_psum_layout=psum, **kwargs)

    def check():
        actual = d.cpu().float()
        tolerance = (0.008, 0.02) if out_dtype == torch.bfloat16 else (2e-4, 1e-4)
        torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
        for g, length in enumerate(lengths):
            if length == 0:
                torch.testing.assert_close(actual[g], expected[g], rtol=0, atol=0)
        assert (storage[:16] == 19).all() and (storage[-16:] == 19).all()
        if c_mode == 'different':
            torch.testing.assert_close(c.cpu(), cv, rtol=0, atol=0)
        if quant and total:
            for sf, mn in ((sa, m), (sb, n)):
                packed_sf = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    sf, metadata, ks_cpu, gran, alignment, psum)[:word_prefix[-1]].cpu()
                exponent = ((sf.cpu().view(torch.int32).to(torch.int64) >> 23) & 255)
                cursor, expected_words = 0, []
                for length in lengths:
                    rows = ((length + alignment - 1) // alignment * alignment) // gran
                    for row in range(0, rows, 4):
                        word = torch.zeros(mn, dtype=torch.int64)
                        for byte in range(min(4, rows - row)):
                            word |= exponent[cursor + row + byte] << (8 * byte)
                        expected_words.append(word.to(torch.int32))
                    cursor += rows
                expected_packed = torch.stack(expected_words) if expected_words else torch.empty((0, mn), dtype=torch.int32)
                assert len(expected_packed) == word_prefix[-1]
                torch.testing.assert_close(packed_sf[:len(expected_packed)], expected_packed, rtol=0, atol=0)

    old_alignment, old_pdl = deep_gemm.get_mk_alignment_for_contiguous_layout(), deep_gemm.get_pdl()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(pdl)
        expected, cv, lengths, word_prefix = inputs(0)
        for _ in range(3):
            d.fill_(float('nan'))
            run()
            check()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in (0, 1, 2):
                expected, cv, lengths, word_prefix = inputs(phase)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
    print(f' > Native SM120 K-grouped: {api=}, {groups=}, {alignment=}, {psum=}, {out_dtype=}, '
          f'{c_mode=}, {shape=}, {gran=}, {packed=}, {graph=}, {empty=}, {ks_mode=}, {word_prefix=}')


def test_sm120_k_grouped_bf16_contracts():
    for groups in (1, 3, 8):
        for psum in (False, True):
            for di, dtype in enumerate((torch.bfloat16, torch.float32)):
                for ci, c_mode in enumerate(('none', 'same', 'different')):
                    exercise_sm120_k_grouped('bf16_tn', groups, 128 if (di + ci) % 2 else 256, psum,
                                            dtype, c_mode, shape=((24, 40), (32, 32), (40, 72))[ci],
                                            ks_mode=('list', 'none', 'empty')[ci] if psum else 'list')


def test_sm120_k_grouped_bf16_graph():
    for dtype in (torch.bfloat16, torch.float32):
        for ci, c_mode in enumerate(('none', 'same', 'different')):
            exercise_sm120_k_grouped('bf16_tn', 8, 128 if ci % 2 else 256, True, dtype, c_mode,
                                    shape=(40, 72), graph=True, pdl=ci % 2 == 1, ks_mode='none')


def test_sm120_k_grouped_bf16_empty():
    for c_mode in ('none', 'same', 'different'):
        exercise_sm120_k_grouped('bf16_tn', 3, 128, True, torch.float32, c_mode, empty=True, ks_mode='empty')


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
