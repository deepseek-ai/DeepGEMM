import time

import numpy as np
import torch

import deep_gemm
from deep_gemm.testing import assert_check, bench_kineto, calc_diff

COEFFICIENTS = (3.4445, -4.775, 2.0315)


def test_batched_newton_schulz() -> None:
    print('Testing BF16 batched Newton-Schulz:')
    b_m_k_list = [
        (16, 4608, 5120),
        (16, 1536, 5120),
        (16, 576, 5120),
        (8, 1536, 24576),
        (8, 512, 32768),
        (4, 5120, 16384),
        (8, 4096, 7168),
        (8, 2048, 7168),
    ]
    total_time = 0.0
    identity, linear, quadratic = map(lambda value: float(np.float32(value)), COEFFICIENTS)
    for shape in b_m_k_list:
        source = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        source.div_(source.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        original = source.clone()
        rows = shape[-2]
        gram = torch.empty((*shape[:-2], rows, rows), device='cuda', dtype=torch.bfloat16)
        polynomial = torch.empty_like(gram)
        expected = torch.empty_like(source)
        output = torch.empty_like(source)
        deep_gemm.batched_syrk(source, gram)
        deep_gemm.batched_syrk(gram, polynomial)
        linear_term = (gram.float() * linear).double()
        polynomial_fp32 = (polynomial.double() * quadratic + linear_term).float()
        polynomial_fp32.diagonal(dim1=-2, dim2=-1).add_(identity)
        polynomial.copy_(polynomial_fp32)
        deep_gemm.batched_symm(polynomial, source, expected)

        def run() -> None:
            deep_gemm.experimental.batched_bf16_newton_schulz(source, output, *COEFFICIENTS)

        run()
        assert_check(torch.isfinite(output).all() and calc_diff(output, expected) < 1e-5,
                     f'Newton-Schulz numerical mismatch: {shape=}')
        first_output = output.clone()
        run()
        assert_check(torch.equal(output.view(torch.int16), first_output.view(torch.int16)),
                     f'Newton-Schulz is not deterministic: {shape=}')
        fused_time, poly_time, gemm_time, ready_time = bench_kineto(
            run, ('sm100_bf16_newton_schulz_impl', 'bf16_newton_schulz_polynomial_impl', 'nvjet', 'FillFunctor<long>'),
            with_multiple_kernels=True, suppress_kineto_output=True)
        assert_check(fused_time > 0 or (poly_time > 0 and gemm_time > 0),
                     f'No complete Newton-Schulz kernel timing: {shape=}')
        kernel_time = fused_time + poly_time + 3 * gemm_time + ready_time
        total_time += kernel_time
        assert_check(torch.equal(output.view(torch.int16), first_output.view(torch.int16)),
                     f'Newton-Schulz is not deterministic after benchmarking: {shape=}')
        assert_check(torch.equal(source, original), 'Newton-Schulz changed its input')
        print(f' > Perf {shape=}: {kernel_time * 1e6:.3f} us GPU kernel sum (direct API)')
        time.sleep(0.3)
    print(f' > SUM: {total_time * 1e6:.3f} us')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_batched_newton_schulz()
