"""Test split-K with AB-swap for small-M FP8 dense GEMM.

Verifies that enabling split-K on the swap path produces correct results
and improves SM utilization for memory-bound M=1..16 shapes.
"""
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, bench_kineto, count_bytes, get_arch_major

from generators import (
    KernelType, MajorTypeAB, QuantConfig,
    generate_normal, get_ue8m0_usage
)


def test_one(m, n, k, quant_config=None):
    if quant_config is None:
        quant_config = QuantConfig()
    kernel_type = KernelType.Kernel1D1D
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=False)

    a, b, c, d, ref_d = generate_normal(
        m, n, k,
        MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        False, torch.bfloat16, kernel_type,
        use_ue8m0=use_ue8m0, quant_config=quant_config
    )

    deep_gemm.fp8_fp4_gemm_nt(
        a, b, d, c=c,
        disable_ue8m0_cast=not use_ue8m0,
        recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b
    )

    diff = calc_diff(d, ref_d)
    max_diff = quant_config.max_diff()

    t = bench_kineto(
        lambda: deep_gemm.fp8_fp4_gemm_nt(
            a, b, d, c=c,
            disable_ue8m0_cast=not use_ue8m0,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b
        ),
        'gemm_', suppress_kineto_output=True
    )

    tflops = 2 * m * n * k / t / 1e12
    bw = count_bytes(a, b, d) / 1e9 / t
    status = 'PASS' if diff < max_diff else 'FAIL'

    print(f'  {status}  m={m:5}, n={n:5}, k={k:5} | diff={diff:.6f} | '
          f'{t * 1e6:6.1f} us | {tflops:4.0f} TFLOPS | {bw:4.0f} GB/s')
    assert diff < max_diff, f'diff={diff:.6f} >= {max_diff}'


def main():
    print(f'GPU arch: SM{get_arch_major() * 10}')
    print()

    qc_fp8 = QuantConfig()

    print('FP8x FP8 swap shapes (M=1):')
    for n, k in [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512),
                 (7168, 16384), (4096, 7168), (7168, 2048)]:
        test_one(1, n, k, qc_fp8)

    print('\nFP8x FP8 swap shapes (M=8, M=16):')
    for m in [8, 16]:
        for n, k in [(2112, 7168), (7168, 2048)]:
            test_one(m, n, k, qc_fp8)

    print('\nFP8x FP8 non-swap regression (M=128, 4096):')
    for m in [128, 4096]:
        for n, k in [(2112, 7168), (7168, 2048)]:
            test_one(m, n, k, qc_fp8)

    print('\nAll tests passed.')


if __name__ == '__main__':
    main()
