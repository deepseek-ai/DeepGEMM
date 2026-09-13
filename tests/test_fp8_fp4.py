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


def sm120_dense_quantized(rows, k, fp4, gran, phase, mn_major=False):
    row = torch.arange(rows)[:, None]
    dim = torch.arange(k)[None, :]
    indices = (row * 5 + dim * 3 + dim // 7 + phase).remainder(15)
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    if fp4:
        codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)[indices]
        if mn_major:
            assert rows % 2 == 0
            raw = (codes[0::2] | (codes[1::2] << 4)).T.contiguous().T.view(torch.int8)
            unpacked = torch.empty((rows, k), dtype=torch.uint8)
            unpacked[0::2] = raw.view(torch.uint8) & 15
            unpacked[1::2] = raw.view(torch.uint8) >> 4
        else:
            assert k % 2 == 0
            raw = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
            unpacked = torch.empty((rows, k), dtype=torch.uint8)
            unpacked[:, 0::2] = raw.view(torch.uint8) & 15
            unpacked[:, 1::2] = raw.view(torch.uint8) >> 4
        magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])[unpacked.long() & 7]
        decoded = torch.where((unpacked & 8) != 0, -magnitude, magnitude)
    else:
        raw = values[indices].to(torch.float8_e4m3fn)
        raw = raw.T.contiguous().T if mn_major else raw
        decoded = raw.float()
    sf = torch.pow(2.0, ((row + torch.arange((k + gran - 1) // gran)[None, :] + phase) % 3 - 4).float())
    decoded *= sf[:, torch.arange(k) // gran]
    return raw, sf, decoded


def sm120_dense_device_matrix(values, padded=False, offset=0, extra=5, tma=True):
    from sm120_test_storage import native_matrix
    return native_matrix(values, padded, offset, extra, tma)


def exercise_sm120_dense_fp8_fp4(fmt, layout, shape, out_dtype, alpha, c_mode, sf_kind,
                                 grans=(32, 128), padded=False, graph=False, pdl=False, common_recipe=False,
                                 mn_grans=(1, 1), compare_default=False, legacy_alias=False):
    assert get_arch_major() == 12
    assert fmt in ((False, False), (False, True), (True, False), (True, True))
    assert layout in ('nt', 'nn', 'tn', 'tt') and c_mode in ('none', 'same', 'different')
    assert sf_kind in ('float', 'packed')
    m, n, k = shape
    assert fmt[0] == fmt[1] or k % 128 == 0
    mn_a = layout[0] == 't' and fmt != (True, True)
    mn_b = layout[1] == 'n' and fmt != (True, True)

    assert sf_kind != 'packed' or mn_grans == (1, 1)
    if compare_default:
        assert grans == (128, 128)
        assert mn_grans == ((1, 128) if sf_kind == 'float' else (1, 1))
    assert not legacy_alias or (fmt == (False, False) and layout == 'nt')

    def operand(rows, fp4, gran, mn_gran, phase, mn_major):
        raw, old_sf, decoded = sm120_dense_quantized(rows, k, fp4, gran, phase, mn_major)
        if mn_gran == 128:
            decoded /= old_sf[:, torch.arange(k) // gran]
            sf = torch.pow(2.0, ((torch.arange((rows + 127) // 128)[:, None]
                                 + torch.arange((k + gran - 1) // gran)[None, :] + phase) % 3 - 4).float())
            decoded *= sf[torch.arange(rows) // 128][:, torch.arange(k) // gran]
            return raw, sf, decoded
        assert mn_gran == 1
        return raw, old_sf, decoded

    def make_inputs(phase):
        av, sa, ar = operand(m, fmt[0], grans[0], mn_grans[0], phase, mn_a)
        bv, sb, br = operand(n, fmt[1], grans[1], mn_grans[1], phase + 2, mn_b)
        cv = (((torch.arange(m)[:, None] * 3 + torch.arange(n)[None, :] + phase) % 17 - 8).float() / 16).to(out_dtype)
        return av, bv, sa, sb, ar, br, cv

    av, bv, sa_cpu, sb_cpu, ar, br, cv = make_inputs(0)
    a = av.cuda() if mn_a else sm120_dense_device_matrix(av)[0]
    b = bv.cuda() if mn_b else sm120_dense_device_matrix(bv)[0]
    assert (a.stride(0) == 1) if mn_a else (a.stride(1) == 1)
    assert (b.stride(0) == 1) if mn_b else (b.stride(1) == 1)
    sa, sb = sa_cpu.cuda(), sb_cpu.cuda()
    d, storage = sm120_dense_device_matrix(torch.zeros_like(cv), padded, int(padded), extra=7)
    c = sm120_dense_device_matrix(cv, padded, int(padded), extra=11, tma=False)[0] if c_mode == 'different' else (d if c_mode == 'same' else None)
    initial_c = cv.cuda()
    valid = torch.zeros_like(storage, dtype=torch.bool)
    valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    if padded and c_mode == 'different':
        assert c.stride(0) != d.stride(0)
    kwargs = (dict(recipe=(*mn_grans, grans[0])) if common_recipe or compare_default else
              dict(recipe_a=(mn_grans[0], grans[0]), recipe_b=(mn_grans[1], grans[1])))
    assert not common_recipe or grans[0] == grans[1]
    function = getattr(deep_gemm, 'fp8_gemm_nt' if legacy_alias else f'fp8_fp4_gemm_{layout}')
    explicit_d = sm120_dense_device_matrix(torch.zeros_like(cv), padded, int(padded), extra=7)[0] if compare_default else None

    def run():
        sf_a, sf_b = sa, sb
        if sf_kind == 'packed':
            sf_a = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sa)
            sf_b = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sb)
            assert sf_a.dtype == sf_b.dtype == torch.int32
            assert sf_a.stride(0) == sf_b.stride(0) == 1
            assert sf_a.stride(1) == align(m, 4) and sf_b.stride(1) == align(n, 4)
        aa = (a, sf_a) if layout[0] == 'n' else (a.T, sf_a.T)
        bb = (b, sf_b) if layout[1] == 't' else (b.T, sf_b.T)
        if compare_default:
            explicit_c = c
            if c_mode == 'same':
                explicit_d.copy_(initial_c)
                explicit_c = explicit_d
            function(aa, bb, explicit_d, c=explicit_c, alpha=alpha,
                     disable_ue8m0_cast=sf_kind == 'packed' and alpha is not None, **kwargs)
        if c_mode == 'same':
            d.copy_(initial_c)
        function(aa, bb, d, c=c, alpha=alpha,
                 disable_ue8m0_cast=sf_kind == 'packed' and alpha is not None,
                 **({} if compare_default else kwargs))

    def check():
        if compare_default:
            assert_bitwise_equal(d, explicit_d, 'default versus explicit dense recipe')
        expected = (ar @ br.T) * (1.0 if alpha is None else alpha)
        if c_mode != 'none':
            expected += cv.float()
        actual = d.cpu().float()
        assert torch.isfinite(actual).all()
        if alpha == 0:
            torch.testing.assert_close(actual, torch.zeros_like(expected) if c_mode == 'none' else cv.float(), rtol=0, atol=0)
        else:
            rtol, atol = (0.008, 0.02) if out_dtype == torch.bfloat16 else (2e-4, 1e-4)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            assert calc_diff(actual, expected) < (1e-5 if out_dtype == torch.bfloat16 else 1e-8)
        assert (storage[~valid] == 19).all(), 'Output store touched guard cells'
        if c_mode == 'different':
            torch.testing.assert_close(c.cpu(), cv, rtol=0, atol=0)

    original_pdl = deep_gemm.get_pdl()
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
                    av, bv, sa_cpu, sb_cpu, ar, br, cv = make_inputs(phase)
                    a.copy_(av)
                    b.copy_(bv)
                    sa.copy_(sa_cpu)
                    sb.copy_(sb_cpu)
                    initial_c.copy_(cv)
                    if c_mode == 'different':
                        c.copy_(cv)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(original_pdl)
    print(f' > Native SM120 dense quantized: {fmt=}, {layout=}, {shape=}, {out_dtype=}, '
          f'{alpha=}, {c_mode=}, {sf_kind=}, {grans=}, {padded=}, {graph=}, {pdl=}')


def test_sm120_dense_fp8_fp4_formats_alpha():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for li, layout in enumerate(('nt', 'nn', 'tn', 'tt')):
            for di, dtype in enumerate((torch.bfloat16, torch.float32)):
                for ai, alpha in enumerate((None, 0.0, 0.5, -1.0, 2.0)):
                    index = fi + li + di + ai
                    grans = ((32, 32), (32, 128), (128, 32), (128, 128))[index % 4]
                    exercise_sm120_dense_fp8_fp4(fmt, layout, (32, 64, 256), dtype, alpha,
                                                ('none', 'same', 'different')[index % 3],
                                                ('float', 'packed')[index % 2], grans,
                                                common_recipe=grans[0] == grans[1])


def test_sm120_dense_fp8_fp4_swap_boundary():
    for fmt in ((False, False), (False, True), (True, False), (True, True)):
        for m in (15, 16, 17):
            for c_mode in ('none', 'same'):
                exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, 33, 256), torch.bfloat16, 0.5,
                                            c_mode, 'packed', (32, 128), padded=True)


def test_sm120_dense_fp8_fp4_tails_large():
    for fmt in ((False, False), (False, True), (True, False), (True, True)):
        for dtype in (torch.bfloat16, torch.float32):
            k = 130 if fmt == (True, True) else (131 if fmt == (False, False) else 256)
            exercise_sm120_dense_fp8_fp4(fmt, 'nt', (65, 67, k), dtype, -1.0,
                                        'different', 'float', (128, 32), padded=True)
            exercise_sm120_dense_fp8_fp4(fmt, 'nt', (2049, 256, 256), dtype, 2.0,
                                        'none', 'packed', (32, 128))


def test_sm120_dense_fp8_fp4_graph_pdl():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for li, layout in enumerate(('nt', 'nn', 'tn', 'tt')):
            for pdl in (False, True):
                exercise_sm120_dense_fp8_fp4(fmt, layout, (32, 64, 256),
                                            torch.bfloat16 if pdl else torch.float32,
                                            -1.0 if pdl else 0.5, ('none', 'same', 'different')[(fi + li) % 3],
                                            'packed' if pdl else 'float', (128, 32), padded=True, graph=True, pdl=pdl)


def test_sm120_dense_fp8_fp4_explicit_rejections():
    assert get_arch_major() == 12
    raw, sf, _ = sm120_dense_quantized(32, 128, False, 32, 0)
    a = raw.cuda(), sf.cuda()
    d = torch.empty((32, 32), device='cuda', dtype=torch.bfloat16)
    cases = [
        (a, a, dict(recipe_a=(1, 32))),
        (a, a, dict(recipe=(1, 1, 32), recipe_a=(1, 32), recipe_b=(1, 32))),
        (a, a, dict(recipe=(2, 1, 32))),
        (a, a, dict(recipe=(1, 1, 64))),
        (a, a, dict(recipe=(1, 1, 32), disable_ue8m0_cast=True)),
    ]
    short8, short_sf, _ = sm120_dense_quantized(32, 130, False, 32, 0)
    short4, short_sf4, _ = sm120_dense_quantized(32, 130, True, 32, 0)
    cases.append(((short8.cuda(), short_sf.cuda()), (short4.cuda(), short_sf4.cuda()), dict(recipe=(1, 1, 32))))
    block_sf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf[:1].cuda())
    cases.append(((a[0], block_sf), a, dict(recipe_a=(128, 32), recipe_b=(1, 32))))
    for aa, bb, kwargs in cases:
        try:
            deep_gemm.fp8_fp4_gemm_nt(aa, bb, d, **kwargs)
        except RuntimeError as error:
            assert 'Assertion' in str(error) or 'assert' in str(error), str(error)
        else:
            raise AssertionError(f'Invalid dense input contract was not rejected: {kwargs}')


def test_sm120_dense_scaling_defaults():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for si, sf_kind in enumerate(('float', 'packed')):
            for i, (m, n) in enumerate(((15, 127), (16, 128), (17, 129), (129, 255), (257, 257))):
                exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, n, 256),
                                            torch.bfloat16 if i % 2 else torch.float32,
                                            (None, 0.0, 0.5, -1.0, 2.0)[i],
                                            ('none', 'same', 'different')[(fi + si + i) % 3], sf_kind,
                                            (128, 128), padded=True,
                                            mn_grans=(1, 128) if sf_kind == 'float' else (1, 1),
                                            compare_default=True, legacy_alias=fmt == (False, False))


def test_sm120_dense_scaling_blockwise_tails():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for gi, mn_grans in enumerate(((128, 1), (1, 128), (128, 128))):
            for i, (m, n) in enumerate(((127, 129), (128, 255), (129, 257), (255, 127), (257, 128))):
                grans = ((32, 128), (128, 32), (32, 32), (128, 128))[(fi + gi + i) % 4]
                exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, n, 256),
                                            torch.bfloat16 if (gi + i) % 2 else torch.float32,
                                            (None, 0.0, 0.5, -1.0, 2.0)[i],
                                            ('none', 'same', 'different')[(fi + i) % 3], 'float', grans,
                                            padded=True, mn_grans=mn_grans,
                                            common_recipe=grans[0] == grans[1])


def test_sm120_dense_scaling_graph():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for pdl in (False, True):
            exercise_sm120_dense_fp8_fp4(fmt, 'nt', (129, 257, 256),
                                        torch.bfloat16 if pdl else torch.float32, -1.0,
                                        ('none', 'same', 'different')[fi % 3], 'float', (128, 128),
                                        padded=True, graph=True, pdl=pdl, mn_grans=(1, 128), compare_default=True)
            exercise_sm120_dense_fp8_fp4(fmt, 'tt', (128, 256, 256),
                                        torch.bfloat16 if pdl else torch.float32, 0.5, 'same', 'float', (32, 128),
                                        padded=True, graph=True, pdl=pdl, mn_grans=(128, 128))


def sm120_quant_grouped_intervals(lengths, alignment):
    intervals, end = [], 0
    for length in lengths:
        start = (end + alignment - 1) // alignment * alignment
        end = start + length
        intervals.append((start, end))
    return intervals


def exercise_sm120_quant_grouped(mode, fmt, alignment, sf_kind, grans=(32, 128),
                                  mn_grans=(1, 1), nn=False, zero_padding=False,
                                  all_empty=False, graph=False, pdl=False, defaults=False, k=256):
    assert get_arch_major() == 12 and mode in ('labels', 'psum', 'masked')
    assert alignment in (32, 64, 128) and sf_kind in ('float', 'packed')
    assert not (mode == 'masked' and (nn or zero_padding))
    assert sf_kind != 'packed' or mn_grans == (1, 1)
    assert fmt[0] == fmt[1] or k % 128 == 0
    assert not defaults or (grans == (128, 128) and
                            mn_grans == ((1, 128) if sf_kind == 'float' else (1, 1)))
    groups = 5
    m = alignment + 3 if mode == 'masked' else groups * 2 * alignment + alignment
    n = 80 if nn else 72
    mn_b = nn and fmt != (True, True)
    expected_m = {32: 32, 64: 33, 128: 65}[alignment]
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m) == alignment

    def operand(rows, fp4, gran, mn_gran, phase, mn_major=False):
        raw, old_sf, decoded = sm120_dense_quantized(rows, k, fp4, gran, phase, mn_major)
        decoded /= old_sf[:, torch.arange(k) // gran]
        sf = torch.pow(2.0, ((torch.arange((rows + mn_gran - 1) // mn_gran)[:, None]
                             + torch.arange((k + gran - 1) // gran)[None, :] + phase) % 3 - 4).float())
        decoded *= sf[torch.arange(rows) // mn_gran][:, torch.arange(k) // gran]
        return raw, sf, decoded

    def cpu_inputs(phase):
        aa = [operand(m, fmt[0], grans[0], mn_grans[0], phase + g * 2)
              for g in range(groups if mode == 'masked' else 1)]
        bb = [operand(n, fmt[1], grans[1], mn_grans[1], phase + g * 2 + 1, mn_b)
              for g in range(groups)]
        av, sa, ar = [torch.stack([item[i] for item in aa]) for i in range(3)]
        if mode != 'masked':
            av, sa, ar = av[0], sa[0], ar[0]
        bv, sb, br = [torch.stack([item[i] for item in bb]) for i in range(3)]
        lengths = [0 if all_empty or phase == 1 or g % 3 == 0 or g == groups - 1 else
                   (alignment + 1 if g % 3 == 1 else alignment - 3) for g in range(groups)]
        if phase == 2 and not all_empty:
            lengths = [alignment - 1 if g % 2 == 0 else 0 for g in range(groups)]
        shape = (groups, m, n) if mode == 'masked' else (m, n)
        valid = torch.zeros(shape[:-1], dtype=torch.bool)
        expected = torch.full(shape, 0.0 if zero_padding else 7.0)
        if mode == 'masked':
            meta = torch.tensor(lengths, dtype=torch.int32)
            for g, length in enumerate(lengths):
                valid[g, :length] = True
                expected[g, :length] = ar[g, :length] @ br[g].T
        else:
            intervals = sm120_quant_grouped_intervals(lengths, alignment)
            labels = torch.full((m,), -1, dtype=torch.int32)
            for g, (start, end) in enumerate(intervals):
                assert start % alignment == 0 and end <= m
                labels[start:end] = g
                valid[start:end] = True
                expected[start:end] = ar[start:end] @ br[g].T
            meta = labels if mode == 'labels' else torch.tensor([end for _, end in intervals], dtype=torch.int32)
        if mode != 'labels':
            for block in range(sa.shape[-2]):
                active = valid[..., block * mn_grans[0]:(block + 1) * mn_grans[0]].any(dim=-1)
                sa[..., block, :] = torch.where(active[..., None], sa[..., block, :],
                                                   float('nan') if mode == 'psum' else 1.0)
        return av, bv, sa, sb, meta, expected, valid

    av, bv, sa_cpu, sb_cpu, meta, expected, valid = cpu_inputs(0)
    a = av.cuda()
    b = (bv.transpose(1, 2).contiguous().transpose(1, 2) if mn_b else bv).cuda()
    assert a.stride(-1) == 1 and (b.stride(-2) == 1 if mn_b else b.stride(-1) == 1)
    sa, sb, metadata = sa_cpu.cuda(), sb_cpu.cuda(), meta.cuda()
    if mode == 'masked':
        storage = torch.full((8 + groups * m * n + 16,), 19, dtype=torch.bfloat16, device='cuda')
        d = storage.as_strided((groups, m, n), (m * n, n, 1), 8)
    else:
        d, storage = sm120_dense_device_matrix(torch.zeros((m, n), dtype=torch.bfloat16), True, 1, 7)
    guard = torch.zeros_like(storage, dtype=torch.bool)
    guard.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    explicit_d = torch.empty_like(d) if defaults else None
    recipes = dict(recipe_a=(mn_grans[0], grans[0]), recipe_b=(mn_grans[1], grans[1]))

    def packed(sf, rows, gran, psum=None):
        result = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, psum)
        words = (k + 4 * gran - 1) // (4 * gran)
        assert result.dtype == torch.int32 and result.shape[-2:] == (rows, words)
        assert result.stride(-2) == 1 and result.stride(-1) == align(rows, 4)
        assert result.data_ptr() % 16 == 0
        if result.ndim == 3:
            assert result.stride(0) == words * align(rows, 4)
        return result

    def run():
        sf_a, sf_b = sa, sb
        if sf_kind == 'packed':
            if mode == 'masked':
                active = torch.arange(m, device='cuda')[None, :] < metadata[:, None]
                sf_a = sa.masked_fill(~active[..., None], 1.0)
            sf_a = packed(sf_a, m, grans[0], metadata if mode == 'psum' else None)
            sf_b = packed(sb, n, grans[1])
        aa = a, sf_a
        bb = (b.transpose(1, 2), sf_b.transpose(1, 2)) if nn else (b, sf_b)
        if mode == 'masked':
            function = deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked
            positional, kwargs = (metadata, expected_m), {}
        else:
            function = getattr(deep_gemm, f'm_grouped_fp8_fp4_gemm_{"nn" if nn else "nt"}_contiguous')
            positional = (metadata,)
            kwargs = dict(use_psum_layout=mode == 'psum', ensure_zero_padding=zero_padding)
            if mode == 'psum' and not nn:
                kwargs['expected_m_for_psum_layout'] = 1
        if defaults:
            explicit_d.fill_(7)
            function(aa, bb, explicit_d, *positional, **kwargs, **recipes)
        function(aa, bb, d, *positional, **kwargs, **({} if defaults else recipes))

    def check():
        actual = d.cpu().float()
        assert torch.isfinite(actual[valid]).all()
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0.008, atol=0.02)
        comparison_rows = valid.clone()
        if zero_padding:
            zero_rows = ~valid
            if mode == 'psum':
                zero_rows = torch.zeros_like(valid)
                for end in metadata.cpu().tolist():
                    zero_rows[end:align(end, alignment)] = True
            torch.testing.assert_close(actual[zero_rows], torch.zeros_like(actual[zero_rows]), rtol=0, atol=0)
            comparison_rows |= zero_rows
        if defaults:
            assert_bitwise_equal(d[comparison_rows], explicit_d[comparison_rows],
                                 'grouped default versus explicit scaling')
        assert (storage[~guard] == 19).all(), 'Grouped quantized output modified guard cells'

    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(pdl)
        for _ in range(3):
            d.fill_(7)
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
                av, bv, sa_cpu, sb_cpu, meta, expected, valid = cpu_inputs(phase)
                a.copy_(av)
                b.copy_(bv)
                sa.copy_(sa_cpu)
                sb.copy_(sb_cpu)
                metadata.copy_(meta)
                d.fill_(7)
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
    print(f' > Native SM120 grouped quantized: {mode=}, {fmt=}, {alignment=}, {sf_kind=}, '
          f'{grans=}, {mn_grans=}, {nn=}, {zero_padding=}, {all_empty=}, {graph=}, {pdl=}, {defaults=}, {k=}')


def test_sm120_quant_grouped_formats():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for ai, alignment in enumerate((32, 64, 128)):
                for si, sf_kind in enumerate(('float', 'packed')):
                    index = mi + fi + ai + si
                    exercise_sm120_quant_grouped(mode, fmt, alignment, sf_kind,
                                                grans=((32, 32), (32, 128), (128, 32), (128, 128))[index % 4],
                                                nn=mode != 'masked' and index % 2 == 0,
                                                zero_padding=mode != 'masked' and (fi + si) % 2 == 0,
                                                k=128 if index % 2 else 256)


def test_sm120_quant_grouped_block_scales():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for gi, mn_grans in enumerate(((128, 1), (1, 128), (128, 128))):
                exercise_sm120_quant_grouped(mode, fmt, (32, 64, 128)[gi], 'float',
                                            grans=((32, 128), (128, 32), (32, 32))[(fi + gi) % 3],
                                            mn_grans=mn_grans, nn=mode != 'masked' and fi % 2 == 1,
                                            zero_padding=mode != 'masked' and (mi + fi + gi) % 2 == 0)


def test_sm120_quant_grouped_defaults():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for si, sf_kind in enumerate(('float', 'packed')):
                exercise_sm120_quant_grouped(mode, fmt, (32, 64, 128)[(mi + fi + si) % 3], sf_kind,
                                            grans=(128, 128), mn_grans=(1, 128) if sf_kind == 'float' else (1, 1),
                                            defaults=True, nn=mode != 'masked' and si == 1,
                                            zero_padding=mode != 'masked' and fi % 2 == 1)


def test_sm120_quant_grouped_graph_mutation():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for ai, alignment in enumerate((32, 64, 128)):
                packed = (fi + ai) % 2 == 1
                exercise_sm120_quant_grouped(mode, fmt, alignment, 'packed' if packed else 'float',
                                            grans=(32, 128) if ai % 2 else (128, 32),
                                            mn_grans=(1, 1) if packed else ((128, 128) if ai == 2 else (1, 128)),
                                            graph=True, pdl=(mi + fi + ai) % 2 == 1,
                                            nn=mode != 'masked' and fi % 2 == 1,
                                            zero_padding=mode != 'masked' and (fi + ai) % 2 == 0)


def test_sm120_quant_grouped_all_empty():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            exercise_sm120_quant_grouped(mode, fmt, (32, 64, 128)[(mi + fi) % 3],
                                        'packed' if fi % 2 else 'float', all_empty=True,
                                        nn=mode != 'masked' and fi % 2 == 1,
                                        zero_padding=mode != 'masked' and fi % 2 == 0)


def test_sm120_quant_grouped_zero_m():
    for fmt in ((False, False), (False, True), (True, False), (True, True)):
        a = torch.empty((0, 64 if fmt[0] else 128), dtype=torch.int8 if fmt[0] else torch.float8_e4m3fn, device='cuda')
        b = torch.empty((3, 65, 64 if fmt[1] else 128), dtype=torch.int8 if fmt[1] else torch.float8_e4m3fn, device='cuda')
        sa = torch.empty((0, 1), device='cuda')
        sb = torch.ones((3, 65, 1), device='cuda')
        d = torch.empty((0, 65), dtype=torch.bfloat16, device='cuda')
        for psum in (False, True):
            metadata = torch.zeros(3 if psum else 0, dtype=torch.int32, device='cuda')
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((a, sa), (b, sb), d, metadata,
                                                         recipe=(1, 1, 128), use_psum_layout=psum,
                                                         ensure_zero_padding=True)
            assert d.numel() == 0


def test_sm120_k_grouped_fp8_contracts():
    from test_bf16 import exercise_sm120_k_grouped
    for api in ('fp8_tn', 'fp8_nt'):
        for gi, groups in enumerate((1, 3, 8)):
            for di, dtype in enumerate((torch.bfloat16, torch.float32)):
                for si, (gran, packed) in enumerate(((128, False), (32, False), (32, True))):
                    psum = api == 'fp8_tn' and (gi + di + si) % 2 == 0
                    exercise_sm120_k_grouped(api, groups, 256 if psum else 128, psum, dtype,
                                            ('none', 'same', 'different')[(gi + di + si) % 3],
                                            shape=(16, 48) if gi % 2 else (48, 80), gran=gran, packed=packed,
                                            ks_mode=('list', 'none', 'empty')[si] if psum else 'list',
                                            default_recipe=gran == 128)


def test_sm120_k_grouped_fp8_graph():
    from test_bf16 import exercise_sm120_k_grouped
    for api in ('fp8_tn', 'fp8_nt'):
        for di, dtype in enumerate((torch.bfloat16, torch.float32)):
            for ci, c_mode in enumerate(('none', 'same', 'different')):
                psum = api == 'fp8_tn'
                exercise_sm120_k_grouped(api, 8, 256 if psum and ci == 2 else 128, psum, dtype, c_mode,
                                        gran=128 if ci == 0 else 32, packed=ci == 2,
                                        graph=True, pdl=(di + ci) % 2 == 1, ks_mode='none' if psum else 'list',
                                        default_recipe=ci == 0)


def test_sm120_k_grouped_fp8_empty():
    from test_bf16 import exercise_sm120_k_grouped
    for api in ('fp8_tn', 'fp8_nt'):
        for c_mode in ('none', 'same', 'different'):
            exercise_sm120_k_grouped(api, 3, 128, False, torch.float32, c_mode, empty=True)


def test_sm120_k_grouped_fp4_pending_rejection():
    assert get_arch_major() == 12
    a = torch.zeros((20, 128), dtype=torch.int8, device='cuda')
    b = torch.zeros((36, 128), dtype=torch.int8, device='cuda')
    sa, sb = torch.ones((8, 20), device='cuda'), torch.ones((8, 36), device='cuda')
    d = torch.full((1, 20, 36), 7, dtype=torch.bfloat16, device='cuda')
    metadata = torch.tensor([256], dtype=torch.int32, device='cuda')
    try:
        deep_gemm.k_grouped_fp4_gemm_nt_contiguous((a, sa), (b, sb), d, [256], metadata)
    except RuntimeError as error:
        assert 'SM120 K-grouped FP4 NT is not implemented' in str(error), str(error)
    else:
        raise AssertionError('Deferred SM120 K-grouped FP4 unexpectedly accepted input')
    assert (d == 7).all()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
    test_k_grouped_gemm_contiguous()
