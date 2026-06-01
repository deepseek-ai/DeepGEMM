import random
import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes

from generators import (
    KernelType, MajorTypeAB, QuantConfig,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked,
)


# FP4xFP4 (MXFP4): both operands packed FP4 (E2M1) with VS=32 UE8M0 scales.
# Dispatches to the SM100_MMA_MXF4_SS path; the API rejects bf16 D so we
# always allocate fp32 D below.
FP4_FP4 = QuantConfig((32, 32, True, True))


def test_gemm() -> None:
    print('Testing GEMM:')
    nk_list = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512),
               (7168, 16384), (4096, 7168), (7168, 2048)]
    m_list = [128, 4096]
    kernel_type = KernelType.Kernel1D1D
    out_dtype = torch.float
    recipe, recipe_a, recipe_b = FP4_FP4.get_recipes()

    for m in m_list:
        for n, k in nk_list:
            a, b, c, d, ref_d = generate_normal(
                m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                accumulate=False, out_dtype=out_dtype, kernel_type=kernel_type,
                use_ue8m0=True, quant_config=FP4_FP4)
            deep_gemm.fp8_fp4_gemm_nt(
                a, b, d, c=c, disable_ue8m0_cast=False,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            diff = calc_diff(d, ref_d)
            assert diff < FP4_FP4.max_diff(), \
                f'{m=}, {n=}, {k=}, {diff:.5f}'

            t = bench_kineto(
                lambda: deep_gemm.fp8_fp4_gemm_nt(
                    a, b, d, c=c, disable_ue8m0_cast=False,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b),
                'sm100_fp4_gemm', suppress_kineto_output=True)
            print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, 1D1D, layout=NT, FP32): '
                  f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')
    m_group_list = [(4, 8192), (8, 4096)]
    n_k_list = [(4096, 7168), (7168, 2048), (24576, 1536), (32768, 512)]
    out_dtype = torch.float
    recipe, recipe_a, recipe_b = FP4_FP4.get_recipes()

    for num_groups, expected_m_per_group in m_group_list:
        for n, k in n_k_list:
            m, a, b, m_indices, d_bf16, _ref_bf16 = generate_m_grouped_contiguous(
                num_groups, expected_m_per_group, n, k,
                MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                use_ue8m0=True, quant_config=FP4_FP4)
            # generate_m_grouped_contiguous returns bf16 d/ref; allocate fp32 d for
            # the FP4xFP4 API and cast the reference for calc_diff.
            d = torch.empty_like(d_bf16, dtype=out_dtype)
            ref_d = _ref_bf16.to(out_dtype)
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a, b, d, m_indices,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                disable_ue8m0_cast=False, use_psum_layout=False)
            diff = calc_diff(d, ref_d)
            assert diff < FP4_FP4.max_diff(), \
                f'{num_groups=}, {m=}, {n=}, {k=}, {diff:.5f}'

            t = bench_kineto(
                lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a, b, d, m_indices,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                    disable_ue8m0_cast=False, use_psum_layout=False),
                'sm100_fp4_gemm', suppress_kineto_output=True)
            print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:5}, '
                  f'n={n:5}, k={k:5}, 1D1D, FP32): '
                  f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')
    max_m = 4096
    m_group_list = [(1, 1024), (2, 512), (4, 256)]
    n_k_list = [(4096, 7168), (7168, 2048)]
    out_dtype = torch.float
    recipe, recipe_a, recipe_b = FP4_FP4.get_recipes()

    for num_groups, expected_m_per_group in m_group_list:
        for n, k in n_k_list:
            a, b, masked_m, _psum_m, d_bf16, _ref_bf16 = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k,
                use_ue8m0=True, quant_config=FP4_FP4)
            d = torch.empty_like(d_bf16, dtype=out_dtype)
            ref_d = _ref_bf16.to(out_dtype)
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
                a, b, d, masked_m, expected_m_per_group,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                disable_ue8m0_cast=False)
            # Diff over the valid (non-padding) rows per group only.
            for g in range(num_groups):
                mm = int(masked_m[g].item())
                diff = calc_diff(d[g, :mm], ref_d[g, :mm])
                assert diff < FP4_FP4.max_diff(), \
                    f'{num_groups=}, group={g}, masked_m={mm}, {n=}, {k=}, {diff:.5f}'

            t = bench_kineto(
                lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
                    a, b, d, masked_m, expected_m_per_group,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                    disable_ue8m0_cast=False),
                'sm100_fp4_gemm', suppress_kineto_output=True)
            print(f' > Perf (num_groups={num_groups}, max_m={max_m}, expected_m_per_group={expected_m_per_group:5}, '
                  f'n={n:5}, k={k:5}, 1D1D, FP32): '
                  f'{t * 1e6:6.1f} us | {2 * num_groups * expected_m_per_group * n * k / t / 1e12:4.0f} TFLOPS')


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
