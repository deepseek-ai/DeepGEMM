"""SM120a Dense FP8×FP4 Mixed Precision GEMM correctness and performance tests.

FP8 A (e4m3) × FP4 B (e2m1) using mxf8f6f4 block_scale MMA with
TMA .b4x16_p64 padded SMEM + ldmatrix.m8n16.x2.b8x16.b4x16_p64 hardware unpack.
"""

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
    print("SM120a Dense FP8×FP4 Mixed Precision GEMM — Correctness")
    print("=" * 60)

    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    qc = QuantConfig((128, 32, False, True))
    recipe, recipe_a, recipe_b = qc.get_recipes()

    shapes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (128, 4096, 7168),
        (4096, 576, 7168),
        (4096, 7168, 2048),
        (4096, 2112, 7168),
        (4096, 24576, 1536),
    ]

    num_total = 0
    num_passed = 0

    for m, n, k in shapes:
        num_total += 1
        reset_seed()
        try:
            a, b, c, d, ref_d = generate_normal(
                m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                False, torch.bfloat16, kernel_type,
                use_ue8m0=use_ue8m0, quant_config=qc
            )
            deep_gemm.fp8_fp4_gemm_nt(
                a, b, d, c=c, disable_ue8m0_cast=not use_ue8m0,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b
            )
            diff = calc_diff(d, ref_d)
            passed = diff < qc.max_diff()
            if passed:
                num_passed += 1
            print(f"  M={m:5}, N={n:5}, K={k:5} — {'PASS' if passed else 'FAIL'} (diff={diff:.6f})")
        except Exception as e:
            print(f"  M={m:5}, N={n:5}, K={k:5} — ERROR: {e}")

    print(f"\nResult: {num_passed}/{num_total} passed")
    return num_passed == num_total


def test_performance():
    print("\n" + "=" * 60)
    print("SM120a Dense FP8×FP4 Mixed — Performance")
    print("=" * 60)

    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    qc = QuantConfig((128, 32, False, True))
    recipe, recipe_a, recipe_b = qc.get_recipes()

    shapes = [
        (4096, 7168, 2048),
        (4096, 2112, 7168),
        (4096, 24576, 1536),
        (1, 7168, 2048),
        (128, 7168, 2048),
    ]

    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Time (us)':>10} | {'TFLOPS':>8} | {'BW (GB/s)':>10}")
    print("-" * 70)

    for m, n, k in shapes:
        reset_seed()
        a, b, c, d, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, kernel_type,
            use_ue8m0=use_ue8m0, quant_config=qc
        )
        t = bench_kineto(
            lambda: deep_gemm.fp8_fp4_gemm_nt(
                a, b, d, c=c, disable_ue8m0_cast=not use_ue8m0,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b
            ),
            'gemm_', suppress_kineto_output=True
        )
        tflops = 2 * m * n * k / t / 1e12
        bw = (count_bytes(a, b, d)) / 1e9 / t
        print(f"{m:6} {n:6} {k:6} | {t * 1e6:10.1f} | {tflops:8.0f}T | {bw:10.0f}")


if __name__ == '__main__':
    ok = test_correctness()
    test_performance()
    if not ok:
        sys.exit(1)
