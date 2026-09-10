import random
import torch

import deep_gemm
from deep_gemm.utils import align, pack_ue8m0_to_int
from deep_gemm.testing import (
    assert_bitwise_equal,
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major
)
from utils import (
    assert_direct_output_matches_fp32_accumulation,
    assert_psum_zero_padding, convert_to_fp8, make_cublas_gemm,
)

from generators import (
    KernelType, QuantConfig, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,
)


def test_gemm() -> None:
    print('Testing GEMM:')
    use_alpha_options = (False, True) if get_arch_major() == 10 else (False,)
    for kernel_type, quant_config, m, n, k, major_a, major_b, accumulate, out_dtype, scores in \
            enumerate_normal(torch.float8_e4m3fn, collect_cublas_scores=True):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=(kernel_type.is_1d1d() and accumulate))

        for test_alias in (False, True):
            for use_alpha in use_alpha_options:
                alpha = random.uniform(-1.0, 1.0) if use_alpha else None
                a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                     kernel_type, use_ue8m0=use_ue8m0,
                                                     quant_config=quant_config, alpha=alpha)
                func_name = f'fp8_fp4_gemm_{major_opt.lower() if test_alias else "nt"}'
                if test_alias:
                    a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                    b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                    assert a[0].is_contiguous() and b[0].is_contiguous()
                getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast,
                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b, alpha=alpha)
                diff = calc_diff(d, ref_d)
                assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, '
                                                        f'{accumulate=}, {out_dtype=}, {use_alpha=}, {alpha=}, '
                                                        f'{diff:.5f}, alias={test_alias}')

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0, quant_config=quant_config)
        initial_d = d.clone()
        def test_func(a_=a, b_=b, d_=d):
            deep_gemm.fp8_fp4_gemm_nt(a_, b_, d_, c=d_ if accumulate else None,
                                       disable_ue8m0_cast=disable_ue8m0_cast,
                                       recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
        equivalent_d = d.clone()
        test_func(d_=equivalent_d)
        # Bitwise deterministic test
        for _ in range(20):
            d.copy_(initial_d)
            test_func()
            assert torch.equal(d, equivalent_d), f'{m=}, {n=}, {k=}, {accumulate=}'
        if quant_config.is_fp4_a or quant_config.is_fp4_b:
            equivalent_fp8_d = initial_d.clone()
            test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
            # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
            assert calc_diff(equivalent_d, equivalent_fp8_d) < 1e-14, (f'FP4/FP8 mismatch: {m=}, {n=}, {k=}, '
                                                                      f'{accumulate=}')
        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        cublas_func = make_cublas_gemm(a, b, d, c)
        cublas_times = bench_kineto(cublas_func, ('nvjet', 'bstensorop', 'reduce'),
                                    suppress_kineto_output=True, with_multiple_kernels=True)
        cublas_t = sum(cublas_times)
        if cublas_t > 0:
            scores.append(cublas_t / t)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{cublas_t / t:.2f}x cuBLAS speedup')


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for kernel_type, quant_config, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout, ensure_zero_padding in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for test_alias in (False, True):
            m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                                          use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                                          quant_config=quant_config)
            func_name = f"m_grouped_fp8_fp4_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            def test_func(a_=a, b_=b, d_=d):
                getattr(deep_gemm, func_name)(a_, b_, d_, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast,
                                              use_psum_layout=use_psum_layout, ensure_zero_padding=ensure_zero_padding,
                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            equivalent_d = d.clone()
            test_func(d_=equivalent_d)
            # Bitwise deterministic test
            for _ in range(20):
                test_func()
                assert torch.equal(d[valid_mask], equivalent_d[valid_mask]), f'{m=}, {n=}, {k=}, alias={test_alias}'
            if quant_config.is_fp4_a or quant_config.is_fp4_b:
                equivalent_fp8_d = d.clone()
                test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
                # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
                assert calc_diff(equivalent_d[valid_mask], equivalent_fp8_d[valid_mask]) < 1e-14, (
                    f'FP4/FP8 mismatch: {m=}, {n=}, {k=}, alias={test_alias}')
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, '
                                                    f'{diff:.5f}, alias={test_alias}, {ensure_zero_padding=}')
            if use_psum_layout and ensure_zero_padding:
                assert_psum_zero_padding(a, d, grouped_layout, 'FP8/FP4')
        m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                          use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                          quant_config=quant_config)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast, use_psum_layout=use_psum_layout,
                                                           ensure_zero_padding=ensure_zero_padding,
                                                           recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}, '
              f'psum={use_psum_layout}, zero_pad={ensure_zero_padding}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0
        expected_m = int(expected_m_per_group * 1.2)

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for i in range(num_tests):
            a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k,
                                                                                   use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                                   quant_config=quant_config)
            def test_func(a_=a, b_=b, d_=d):
                common = dict(disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                if use_psum_layout:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a_, b_, d_, grouped_layout,
                                                                   use_psum_layout=True, expected_m_for_psum_layout=expected_m, **common)
                else:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a_, b_, d_, grouped_layout, expected_m, **common)

            equivalent_d = d.clone()
            test_func(d_=equivalent_d)
            # Bitwise deterministic test
            for _ in range(20):
                test_func()
                assert torch.equal(d[valid_mask], equivalent_d[valid_mask]), f'{max_m=}, {n=}, {k=}'
            if quant_config.is_fp4_a or quant_config.is_fp4_b:
                equivalent_fp8_d = d.clone()
                test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
                # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
                assert calc_diff(equivalent_d[valid_mask], equivalent_fp8_d[valid_mask]) < 1e-14, (
                    f'FP4/FP8 mismatch: {max_m=}, {n=}, {k=}, {num_groups=}')
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < quant_config.max_diff(), f'{max_m=}, {n=}, {k=}, {kernel_opt}, {num_groups=}, {diff:.5f}'

            # Test performance with fixed shapes
            valid_m = int(valid_mask.sum().item())
            t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'{kernel_opt}, psum={1 if use_psum_layout else 0}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped GEMM:')

    arch_major = get_arch_major()
    test_options = [(torch.float8_e4m3fn, QuantConfig(),
                     deep_gemm.k_grouped_fp8_gemm_nt_contiguous if arch_major == 9 else
                     deep_gemm.k_grouped_fp8_gemm_tn_contiguous)]
    if arch_major == 10:
        test_options.append((torch.float4_e2m1fn_x2, QuantConfig((32, 32, True, True)),
                             deep_gemm.k_grouped_fp4_gemm_nt_contiguous))
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    for dtype, quant_config, gemm in test_options:
        is_fp4 = dtype == torch.float4_e2m1fn_x2
        dtype_opt = 'FP4' if is_fp4 else 'FP8'

        for num_groups, m, n, major_a, major_b, real_ks_cpu, _, _, gran_k, k_alignment, use_psum_layout, accumulate, out_dtype in \
                enumerate_k_grouped_contiguous(dtype):
            recipe = (1, 1, gran_k)

            for test_real_ks_cpu in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu):
                total_k, a, b, c, d, ref_d, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                    num_groups, m, n, major_a, major_b, test_real_ks_cpu,
                    use_ue8m0=use_ue8m0, gran_k=gran_k,
                    quant_config=quant_config if is_fp4 else None,
                    use_psum_layout=use_psum_layout, k_alignment=k_alignment,
                    accumulate=accumulate, out_dtype=out_dtype)

                initial_d = d.clone()
                if not accumulate:
                    initial_d.fill_(float('nan'))
                equivalent_d = initial_d.clone()
                gemm(a, b, equivalent_d, host_ks_cpu, grouped_layout, equivalent_d if accumulate else None,
                     recipe=recipe, use_psum_layout=use_psum_layout)
                if is_fp4 and accumulate:
                    fp8_a, fp8_b = convert_to_fp8(a), convert_to_fp8(b)
                    fp8_a = (fp8_a[0].T.contiguous(), fp8_a[1])
                    fp8_b = (fp8_b[0].T.contiguous(), fp8_b[1])
                    equivalent_fp8_d = initial_d.clone()
                    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                        fp8_a, fp8_b, equivalent_fp8_d, host_ks_cpu, grouped_layout, equivalent_fp8_d if accumulate else None,
                        recipe=recipe, use_psum_layout=use_psum_layout)
                    mismatch_message = (f'FP4/FP8 mismatch: {m=}, {n=}, {total_k=}, '
                                        f'{test_real_ks_cpu=}, {use_psum_layout=}')
                    assert calc_diff(equivalent_d, equivalent_fp8_d) < 1e-14, mismatch_message

                # Bitwise deterministic test
                host_ks_options = (host_ks_cpu, None, []) if use_psum_layout else (host_ks_cpu, )
                for test_host_ks_cpu in host_ks_options:
                    for stress_idx in range(20):
                        d.copy_(initial_d)
                        gemm(a, b, d, test_host_ks_cpu, grouped_layout, c,
                             recipe=recipe, use_psum_layout=use_psum_layout)
                        assert_bitwise_equal(
                            d, equivalent_d,
                            f'k-grouped self-consistency at {stress_idx=}, {dtype_opt}, {m=}, {n=}, {total_k=}, '
                            f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}'
                        )
                    if not accumulate:
                        case_label = (f'{dtype_opt} K-grouped direct output, {m=}, {n=}, {total_k=}, '
                                      f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, '
                                      f'{out_dtype=}')
                        assert_direct_output_matches_fp32_accumulation(
                            d,
                            lambda output, accumulator: gemm(
                                a, b, output, test_host_ks_cpu, grouped_layout, accumulator,
                                recipe=recipe, use_psum_layout=use_psum_layout),
                            case_label)

                if accumulate:
                    diff = calc_diff(d, ref_d)
                    assert diff < quant_config.max_diff(), (
                        f'{dtype_opt}, {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                        f'{host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}, {diff:.5f}')

                # gran_k=128 requires FP32 SF input; only gran_k=32 accepts
                # the per-group packed INT32 UE8M0 layout.
                if gran_k == 32:
                    sf_ks = [k // gran_k for k in host_ks_cpu]
                    ref_packed_a = torch.cat([
                        pack_ue8m0_to_int(torch.nn.functional.pad(
                            group_sf.T, (0, align(sf_k, 4) - sf_k)).contiguous()).T
                        for group_sf, sf_k in zip(a[1].split(sf_ks), sf_ks) if sf_k > 0
                    ])
                    ref_packed_b = torch.cat([
                        pack_ue8m0_to_int(torch.nn.functional.pad(
                            group_sf.T, (0, align(sf_k, 4) - sf_k)).contiguous()).T
                        for group_sf, sf_k in zip(b[1].split(sf_ks), sf_ks) if sf_k > 0
                    ])
                    packed_a = (
                        a[0], deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                            a[1], grouped_layout, host_ks_cpu, gran_k, k_alignment, use_psum_layout))
                    packed_b = (
                        b[0], deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                            b[1], grouped_layout, host_ks_cpu, gran_k, k_alignment, use_psum_layout))
                    assert torch.equal(packed_a[1], ref_packed_a)
                    assert torch.equal(packed_b[1], ref_packed_b)

                    packed_d = initial_d.clone()
                    gemm(packed_a, packed_b, packed_d, host_ks_cpu, grouped_layout, packed_d if accumulate else None,
                         recipe=recipe, use_psum_layout=use_psum_layout)
                    if accumulate:
                        packed_diff = calc_diff(packed_d, ref_d)
                        assert packed_diff < quant_config.max_diff(), (
                            f'pre-packed INT32 SF: {dtype_opt}, {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                            f'{host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}, '
                            f'{packed_diff:.5f}')
                    else:
                        case_label = (f'pre-packed INT32 SF direct output: {dtype_opt}, {m=}, {n=}, '
                                      f'{total_k=}, {test_real_ks_cpu=}, {host_ks_cpu=}, '
                                      f'{use_psum_layout=}, {out_dtype=}')
                        assert_direct_output_matches_fp32_accumulation(
                            packed_d,
                            lambda output, accumulator: gemm(
                                packed_a, packed_b, output, host_ks_cpu, grouped_layout, accumulator,
                                recipe=recipe, use_psum_layout=use_psum_layout),
                            case_label)

            _, a, b, c, d, _, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, real_ks_cpu,
                use_ue8m0=use_ue8m0, gran_k=gran_k,
                quant_config=quant_config if is_fp4 else None,
                use_psum_layout=use_psum_layout, k_alignment=k_alignment,
                accumulate=accumulate, out_dtype=out_dtype)

            # noinspection PyShadowingNames
            def test_func():
                gemm(a, b, d, host_ks_cpu, grouped_layout, c,
                     recipe=recipe, use_psum_layout=use_psum_layout)

            t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
            logical_k = sum(real_ks_cpu)
            out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
            print(f' > Perf ({dtype_opt}, {num_groups=:2}, m={m:5}, n={n:5}, k={logical_k:5}, gran_k={gran_k:3}, '
                  f'k_alignment={k_alignment:3}, psum={int(use_psum_layout)}, acc={int(accumulate)}, {out_opt}): '
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * m * n * logical_k / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
    test_k_grouped_gemm_contiguous()
