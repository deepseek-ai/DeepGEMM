"""SM120a M-Grouped FP8/FP4 GEMM correctness tests (contiguous + masked)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major
from generators import (
    generate_m_grouped_contiguous, generate_m_grouped_masked,
    layout_masked_to_psum, MajorTypeAB,
    get_ue8m0_usage, KernelType, QuantConfig,
    reset_seed, get_mk_alignment_for_contiguous_layout, align
)


def test_m_grouped_contiguous():
    assert get_arch_major() == 12, f"Expected SM120a, got arch_major={get_arch_major()}"
    print("=" * 70)
    print("SM120a M-Grouped Contiguous FP8 GEMM — Correctness")
    print("=" * 70)

    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    disable_ue8m0_cast = not use_ue8m0

    configs = [
        # (num_groups, expected_m, n, k, use_psum)
        ( 4, 8192, 7168, 2048, False),
        ( 8, 4096, 7168, 2048, False),
        ( 4, 8192, 4096, 4096, False),
        ( 8, 4096, 4096, 2048, False),
        ( 4, 8192, 7168, 2048, True),
        ( 8, 4096, 7168, 2048, True),
    ]

    num_total = 0
    num_passed = 0

    for quant_config in QuantConfig.get_list_from_dtype(torch.float8_e4m3fn):
        qc_name = "FP4" if quant_config.is_fp4_a else "FP8"
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        for num_groups, expected_m, n, k, use_psum in configs:
            alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
            deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
            reset_seed()

            num_total += 1
            label = f"{qc_name} {num_groups}g {expected_m}x{n}x{k} psum={use_psum}"
            try:
                m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
                    num_groups, expected_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                    use_ue8m0=use_ue8m0, use_psum_layout=use_psum, quant_config=quant_config)

                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast,
                    use_psum_layout=use_psum,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

                max_diff = 0.0
                if use_psum:
                    for j in range(num_groups):
                        start = 0 if j == 0 else align(grouped_layout[j - 1].item(), get_mk_alignment_for_contiguous_layout())
                        end = grouped_layout[j].item()
                        diff = calc_diff(d[start:end], ref_d[start:end])
                        max_diff = max(max_diff, diff)
                else:
                    max_diff = calc_diff(d, ref_d)

                passed = max_diff < quant_config.max_diff()
                if passed:
                    num_passed += 1
                status = "PASS" if passed else "FAIL"
                print(f"  {label:45s} — {status} (diff={max_diff:.6f})")
            except Exception as e:
                print(f"  {label:45s} — ERROR: {e}")

    print(f"\nContiguous: {num_passed}/{num_total} passed\n")
    return num_passed == num_total


def test_m_grouped_masked():
    assert get_arch_major() == 12, f"Expected SM120a, got arch_major={get_arch_major()}"
    print("=" * 70)
    print("SM120a M-Grouped Masked FP8 GEMM — Correctness")
    print("=" * 70)

    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    disable_ue8m0_cast = not use_ue8m0

    configs = [
        # (num_groups, max_m, expected_m, n, k)
        ( 6, 4096, 1024, 7168, 2048),
        (32, 4096,  192, 7168, 2048),
        ( 6, 4096, 1024, 4096, 4096),
        ( 6, 4096,   20, 4096, 2048),
        (32, 4096,   20, 4096, 2048),
    ]

    num_total = 0
    num_passed = 0

    for quant_config in QuantConfig.get_list_from_dtype(torch.float8_e4m3fn):
        qc_name = "FP4" if quant_config.is_fp4_a else "FP8"
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        for num_groups, max_m, expected_m, n, k in configs:
            alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m * 1.2))
            deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
            reset_seed()

            num_total += 1
            label = f"{qc_name} {num_groups}g max_m={max_m} exp={expected_m} {n}x{k}"
            try:
                a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
                    num_groups, max_m, expected_m, n, k,
                    use_ue8m0=use_ue8m0, quant_config=quant_config)

                deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
                    a, b, d, masked_m, int(expected_m * 1.2),
                    disable_ue8m0_cast=disable_ue8m0_cast,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

                max_diff = 0.0
                for j in range(num_groups):
                    mm = masked_m[j].item()
                    if mm == 0:
                        continue
                    diff = calc_diff(d[j, :mm], ref_d[j, :mm])
                    max_diff = max(max_diff, diff)

                passed = max_diff < quant_config.max_diff()
                if passed:
                    num_passed += 1
                status = "PASS" if passed else "FAIL"
                print(f"  {label:50s} — {status} (diff={max_diff:.6f})")
            except Exception as e:
                print(f"  {label:50s} — ERROR: {e}")

    print(f"\nMasked: {num_passed}/{num_total} passed\n")
    return num_passed == num_total


def test_performance():
    assert get_arch_major() == 12
    print("=" * 70)
    print("SM120a M-Grouped FP8 GEMM — Performance")
    print("=" * 70)

    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

    print(f"\n{'Type':>12s} {'Config':>30s} | {'Time (us)':>10s} | {'TFLOPS':>8s}")
    print("-" * 70)

    # Contiguous performance
    for num_groups, expected_m, n, k in [
        ( 4, 8192, 7168, 2048),
        ( 8, 4096, 7168, 2048),
        ( 4, 8192, 4096, 4096),
    ]:
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        reset_seed()
        m, a, b, gl, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_ue8m0=use_ue8m0, use_psum_layout=False)

        t = bench_kineto(
            lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a, b, d, gl, disable_ue8m0_cast=False, use_psum_layout=False),
            'gemm_', suppress_kineto_output=True)
        tflops = 2.0 * m * n * k / t / 1e12
        label = f"{num_groups}g m~{expected_m} {n}x{k}"
        print(f"{'contiguous':>12s} {label:>30s} | {t*1e6:10.1f} | {tflops:8.1f}")

    # Masked performance
    for num_groups, max_m, expected_m, n, k in [
        ( 6, 4096, 1024, 7168, 2048),
        (32, 4096,  192, 7168, 2048),
        ( 6, 4096, 1024, 4096, 4096),
    ]:
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(int(expected_m * 1.2))
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        reset_seed()
        a, b, masked_m, psum_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m, n, k, use_ue8m0=use_ue8m0)

        valid_m = masked_m.sum().item()
        t = bench_kineto(
            lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
                a, b, d, masked_m, int(expected_m * 1.2), disable_ue8m0_cast=False),
            'gemm_', suppress_kineto_output=True)
        tflops = 2.0 * valid_m * n * k / t / 1e12
        label = f"{num_groups}g exp={expected_m} {n}x{k}"
        print(f"{'masked':>12s} {label:>30s} | {t*1e6:10.1f} | {tflops:8.1f}")

    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    c_pass = test_m_grouped_contiguous()
    m_pass = test_m_grouped_masked()
    if not (c_pass and m_pass):
        print("Some tests FAILED!")
        sys.exit(1)
    else:
        print("All M-grouped tests PASSED!\n")
        test_performance()
