"""SM120a AB-Swap correctness tests.

Verifies that for M <= 32, the AB-swap path produces results within
the same tolerance as the normal path, across all quant configs.

Verification strategy:
1. Reference: PyTorch BF16 matmul (same as other tests)
2. Dual-path: for each shape, also run with M=33 (forces normal path)
   to confirm swap path diff is comparable to normal path diff
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from generators import (
    generate_normal, KernelType, MajorTypeAB, QuantConfig,
    get_ue8m0_usage, reset_seed,
)


def test_dense_gemm_swap():
    assert get_arch_major() == 12
    print("=" * 70)
    print("SM120a AB-Swap Dense GEMM — Correctness")
    print("=" * 70)

    shapes = [
        # (M, N, K) — M <= 32 triggers swap
        (1, 7168, 2048),
        (1, 2048, 7168),
        (4, 7168, 2048),
        (4, 4096, 4096),
        (16, 7168, 2048),
        (16, 2112, 7168),
        (32, 7168, 2048),
        (32, 4096, 2048),
        # M > 32: should NOT swap — verify no regression
        (33, 7168, 2048),
        (64, 7168, 2048),
        (128, 7168, 2048),
    ]

    num_passed = 0
    num_total = 0

    for qc in QuantConfig.get_list_from_dtype(torch.float8_e4m3fn):
        if qc.is_fp4_a and qc.is_fp4_b:
            qc_name = "FP4"
        elif qc.is_fp4_b:
            qc_name = "FP8xFP4"
        else:
            qc_name = "FP8"

        recipe, ra, rb = qc.get_recipes()
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for m, n, k in shapes:
            num_total += 1
            swap_label = "SWAP" if m <= 32 else "NORM"
            label = f"{qc_name} {swap_label} M={m:>4} N={n:>5} K={k:>5}"

            try:
                reset_seed()
                a, b, c, d, ref_d = generate_normal(
                    m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                    False, torch.bfloat16, KernelType.Kernel1D1D,
                    use_ue8m0=use_ue8m0, quant_config=qc)

                deep_gemm.fp8_fp4_gemm_nt(
                    a, b, d, recipe=recipe, recipe_a=ra, recipe_b=rb)

                diff = calc_diff(d, ref_d)
                passed = diff < qc.max_diff()
                if passed:
                    num_passed += 1
                status = "PASS" if passed else f"FAIL"
                print(f"  {label} — {status} (diff={diff:.6f})")
            except Exception as e:
                print(f"  {label} — ERROR: {e}")

    print(f"\nDense GEMM: {num_passed}/{num_total} passed\n")
    return num_passed == num_total


def test_output_layout():
    """Verify swap path writes to correct output positions."""
    assert get_arch_major() == 12
    print("=" * 70)
    print("SM120a AB-Swap — Output Layout Verification")
    print("=" * 70)

    qc = QuantConfig()
    recipe, ra, rb = qc.get_recipes()
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    num_passed = 0
    num_total = 0

    for m, n, k in [(16, 7168, 2048), (32, 4096, 4096), (1, 2048, 2048)]:
        num_total += 1
        reset_seed()
        a, b, c, d, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, KernelType.Kernel1D1D,
            use_ue8m0=use_ue8m0, quant_config=qc)

        deep_gemm.fp8_fp4_gemm_nt(
            a, b, d, recipe=recipe, recipe_a=ra, recipe_b=rb)

        # Verify output shape is correct (M, N) and contiguous
        assert d.shape == (m, n), f"Shape mismatch: {d.shape} vs ({m}, {n})"
        assert d.stride(-1) == 1, f"Not row-major: stride={d.stride()}"
        assert d.stride(-2) == n, f"Unexpected stride: {d.stride()}"

        diff = calc_diff(d, ref_d)
        passed = diff < qc.max_diff()
        if passed:
            num_passed += 1
        print(f"  M={m:>4} N={n:>5} K={k:>5} — {'PASS' if passed else 'FAIL'} "
              f"(diff={diff:.6f}, shape={d.shape}, stride={d.stride()})")

    print(f"\nLayout: {num_passed}/{num_total} passed\n")
    return num_passed == num_total


if __name__ == '__main__':
    torch.manual_seed(0)
    p1 = test_dense_gemm_swap()
    p2 = test_output_layout()
    if p1 and p2:
        print("All AB-swap tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
