"""SM120a Dense FP4 GEMM correctness and performance tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes, get_arch_major
from generators import (
    generate_normal, KernelType, MajorTypeAB, QuantConfig,
    get_ue8m0_usage, reset_seed
)


def test_correctness():
    assert get_arch_major() == 12, f"Expected SM120a, got arch_major={get_arch_major()}"
    print("=" * 60)
    print("SM120a Dense FP4 GEMM — Correctness")
    print("=" * 60)

    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)

    configs = [
        # (gran_k_a, gran_k_b, is_fp4_a, is_fp4_b, label)
        (32, 32, True, True, "FP4xFP4 gk=32"),
        (128, 128, True, True, "FP4xFP4 gk=128"),
    ]

    shapes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (128, 4096, 7168),
        (4096, 7168, 2048),
    ]

    num_total = 0
    num_passed = 0

    for gran_k_a, gran_k_b, is_fp4_a, is_fp4_b, label in configs:
        qc = QuantConfig((gran_k_a, gran_k_b, is_fp4_a, is_fp4_b))
        recipe, recipe_a, recipe_b = qc.get_recipes()
        print(f"\n  Config: {label}")

        for m, n, k in shapes:
            num_total += 1
            reset_seed()
            try:
                a, b, c, d, ref_d = generate_normal(
                    m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                    accumulate=False, out_dtype=torch.bfloat16,
                    kernel_type=kernel_type, use_ue8m0=use_ue8m0, quant_config=qc
                )
                deep_gemm.fp8_fp4_gemm_nt(
                    a, b, d, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b
                )
                diff = calc_diff(d, ref_d)
                passed = diff < qc.max_diff()
                status = "PASS" if passed else "FAIL"
                if passed:
                    num_passed += 1
                print(f"    M={m:5d}, N={n:5d}, K={k:5d} — {status} (diff={diff:.6f})")
            except Exception as e:
                print(f"    M={m:5d}, N={n:5d}, K={k:5d} — ERROR: {e}")

    print(f"\nResult: {num_passed}/{num_total} passed\n")
    return num_passed == num_total


def test_performance():
    assert get_arch_major() == 12
    print("=" * 60)
    print("SM120a Dense FP4 GEMM — Performance (vs FP8)")
    print("=" * 60)

    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)

    qc_fp4 = QuantConfig((32, 32, True, True))
    recipe_fp4, recipe_a_fp4, recipe_b_fp4 = qc_fp4.get_recipes()

    qc_fp8 = QuantConfig()
    recipe_fp8, recipe_a_fp8, recipe_b_fp8 = qc_fp8.get_recipes()

    shapes = [
        (4096, 7168, 2048),
        (4096, 2112, 7168),
        (4096, 24576, 1536),
        (1, 7168, 2048),
        (128, 7168, 2048),
    ]

    print(f"{'M':>6s} {'N':>6s} {'K':>6s} | {'FP4 (us)':>10s} {'FP4 TFLOPS':>11s} | {'FP8 (us)':>10s} {'FP8 TFLOPS':>11s} | {'Speedup':>8s}")
    print("-" * 82)

    for m, n, k in shapes:
        reset_seed()
        fp4_str = ""
        fp8_str = ""
        fp4_t = fp8_t = None

        # FP4
        try:
            a4, b4, _, d4, _ = generate_normal(
                m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                accumulate=False, out_dtype=torch.bfloat16,
                kernel_type=kernel_type, use_ue8m0=use_ue8m0, quant_config=qc_fp4
            )
            fp4_t = bench_kineto(
                lambda: deep_gemm.fp8_fp4_gemm_nt(a4, b4, d4, recipe=recipe_fp4, recipe_a=recipe_a_fp4, recipe_b=recipe_b_fp4),
                'gemm_', suppress_kineto_output=True
            )
            fp4_tflops = 2 * m * n * k / fp4_t / 1e12
            fp4_str = f"{fp4_t * 1e6:10.1f} {fp4_tflops:10.1f}T"
        except Exception as e:
            fp4_str = f"{'ERROR':>10s} {str(e)[:10]:>11s}"

        # FP8
        reset_seed()
        try:
            a8, b8, _, d8, _ = generate_normal(
                m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                accumulate=False, out_dtype=torch.bfloat16,
                kernel_type=kernel_type, use_ue8m0=use_ue8m0, quant_config=qc_fp8
            )
            fp8_t = bench_kineto(
                lambda: deep_gemm.fp8_fp4_gemm_nt(a8, b8, d8, recipe=recipe_fp8, recipe_a=recipe_a_fp8, recipe_b=recipe_b_fp8),
                'gemm_', suppress_kineto_output=True
            )
            fp8_tflops = 2 * m * n * k / fp8_t / 1e12
            fp8_str = f"{fp8_t * 1e6:10.1f} {fp8_tflops:10.1f}T"
        except Exception as e:
            fp8_str = f"{'ERROR':>10s} {str(e)[:10]:>11s}"

        speedup = f"{fp8_t / fp4_t:.2f}x" if fp4_t and fp8_t else "N/A"
        print(f"{m:6d} {n:6d} {k:6d} | {fp4_str} | {fp8_str} | {speedup:>8s}")


if __name__ == '__main__':
    torch.manual_seed(0)
    all_pass = test_correctness()
    if all_pass:
        test_performance()
    else:
        print("Correctness tests failed, skipping performance tests.")
