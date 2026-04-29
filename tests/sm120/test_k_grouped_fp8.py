"""SM120a K-Grouped Contiguous FP8 GEMM correctness and performance tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major
from generators import (
    generate_k_grouped_contiguous, MajorTypeAB,
    get_ue8m0_usage, KernelType, reset_seed,
    set_mk_alignment_for_contiguous_layout, align
)


def test_correctness():
    assert get_arch_major() == 12, f"Expected SM120a, got arch_major={get_arch_major()}"
    print("=" * 60)
    print("SM120a K-Grouped Contiguous FP8 GEMM — Correctness")
    print("=" * 60)

    major_a, major_b = MajorTypeAB.KMajor, MajorTypeAB.KMajor
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

    configs = [
        # (num_groups, m, n, expected_k, gran_k, label)
        ( 4, 4096, 7168, 2048, 128, " 4g  4096x7168  gk=128"),
        ( 8, 4096, 7168, 2048, 128, " 8g  4096x7168  gk=128"),
        (16, 4096, 7168, 2048, 128, "16g  4096x7168  gk=128"),
        ( 4, 7168, 2048, 2048, 128, " 4g  7168x2048  gk=128"),
        ( 4, 4096, 7168, 2048,  32, " 4g  4096x7168  gk=32"),
        ( 8, 4096, 7168, 2048,  32, " 8g  4096x7168  gk=32"),
    ]

    num_total = 0
    num_passed = 0

    for num_groups, m, n, expected_k, gran_k, label in configs:
        set_mk_alignment_for_contiguous_layout(gran_k)
        reset_seed()
        random.seed(num_groups * 1000 + gran_k)
        ks = [align(int(expected_k * random.uniform(0.7, 1.3)), 128) for _ in range(num_groups)]
        recipe = (1, 1, gran_k)

        num_total += 1
        try:
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, ks,
                use_ue8m0=use_ue8m0, gran_k=gran_k
            )
            ks_tensor = torch.tensor(ks, dtype=torch.int32, device='cuda')

            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, c, recipe=recipe)

            diff = calc_diff(d, ref_d)
            passed = diff < 0.01
            if passed:
                num_passed += 1
            status = "PASS" if passed else "FAIL"
            print(f"  {label} — {status} (diff={diff:.6f})")
        except Exception as e:
            print(f"  {label} — ERROR: {e}")

    # Test with zero-K groups (edge case)
    num_total += 1
    try:
        gran_k = 128
        set_mk_alignment_for_contiguous_layout(gran_k)
        reset_seed()
        random.seed(42)
        m, n = 4096, 7168
        ks_with_zero = [0, 2048, 0, 1920, 2176, 0]
        num_groups = len(ks_with_zero)
        recipe = (1, 1, gran_k)

        k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
            num_groups, m, n, major_a, major_b, ks_with_zero,
            use_ue8m0=use_ue8m0, gran_k=gran_k
        )
        ks_tensor = torch.tensor(ks_with_zero, dtype=torch.int32, device='cuda')
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks_with_zero, ks_tensor, c, recipe=recipe)

        diff = calc_diff(d, ref_d)
        passed = diff < 0.01
        if passed:
            num_passed += 1
        status = "PASS" if passed else "FAIL"
        print(f"  zero-K groups edge case  — {status} (diff={diff:.6f})")
    except Exception as e:
        print(f"  zero-K groups edge case  — ERROR: {e}")

    print(f"\nResult: {num_passed}/{num_total} passed\n")
    return num_passed == num_total


def test_performance():
    assert get_arch_major() == 12
    print("=" * 60)
    print("SM120a K-Grouped Contiguous FP8 GEMM — Performance")
    print("=" * 60)

    major_a, major_b = MajorTypeAB.KMajor, MajorTypeAB.KMajor
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    gran_k = 128
    set_mk_alignment_for_contiguous_layout(gran_k)
    recipe = (1, 1, gran_k)

    bench_configs = [
        # (num_groups, m, n, expected_k, label)
        ( 4, 4096, 7168, 8192, "EP64  4g"),
        ( 8, 4096, 7168, 4096, "EP32  8g"),
        (16, 4096, 7168, 2048, "EP16 16g"),
        ( 4, 7168, 2048, 8192, "EP64  4g (alt)"),
    ]

    print(f"{'Config':>20s} | {'sum_K':>8s} | {'Time (us)':>10s} | {'TFLOPS':>8s}")
    print("-" * 56)

    for num_groups, m, n, expected_k, label in bench_configs:
        reset_seed()
        random.seed(num_groups * 100 + expected_k)
        ks = [align(int(expected_k * random.uniform(0.7, 1.3)), 128) for _ in range(num_groups)]
        sum_k = sum(ks)

        try:
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, ks,
                use_ue8m0=use_ue8m0, gran_k=gran_k
            )
            ks_tensor = torch.tensor(ks, dtype=torch.int32, device='cuda')

            t = bench_kineto(
                lambda: deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
                    a, b, d, ks, ks_tensor, c, recipe=recipe),
                'gemm_', suppress_kineto_output=True
            )
            tflops = 2.0 * m * n * sum_k / t / 1e12
            print(f"{label:>20s} | {sum_k:8d} | {t * 1e6:10.1f} | {tflops:8.1f}")
        except Exception as e:
            print(f"{label:>20s} | ERROR: {e}")


if __name__ == '__main__':
    torch.manual_seed(0)
    all_pass = test_correctness()
    if all_pass:
        test_performance()
    else:
        print("Correctness tests failed, skipping performance tests.")
