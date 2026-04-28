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
                status = "PASS" if passed else f"FAIL"
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
    print("SM120a Dense FP4 GEMM — Performance")
    print("=" * 60)

    qc = QuantConfig((32, 32, True, True))
    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = qc.get_recipes()

    shapes = [
        (4096, 7168, 2048),
        (4096, 2112, 7168),
        (4096, 24576, 1536),
        (1, 7168, 2048),
        (128, 7168, 2048),
    ]

    print(f"{'M':>6s} {'N':>6s} {'K':>6s} | {'Time (us)':>10s} | {'TFLOPS':>8s} | {'BW (GB/s)':>10s}")
    print("-" * 56)

    for m, n, k in shapes:
        reset_seed()
        a, b, c, d, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            accumulate=False, out_dtype=torch.bfloat16,
            kernel_type=kernel_type, use_ue8m0=use_ue8m0, quant_config=qc
        )

        try:
            t = bench_kineto(
                lambda: deep_gemm.fp8_fp4_gemm_nt(a, b, d, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b),
                'gemm_', suppress_kineto_output=True
            )
            tflops = 2 * m * n * k / t / 1e12
            bw = count_bytes(a, b, d) / 1e9 / t
            print(f"{m:6d} {n:6d} {k:6d} | {t * 1e6:10.1f} | {tflops:8.1f} | {bw:10.0f}")
        except Exception as e:
            print(f"{m:6d} {n:6d} {k:6d} | ERROR: {e}")


if __name__ == '__main__':
    torch.manual_seed(0)
    all_pass = test_correctness()
    if all_pass:
        test_performance()
    else:
        print("Correctness tests failed, skipping performance tests.")
