import pytest
import torch
import random

import deep_gemm
from deep_gemm.testing import (
    test_filter as filter_test,
    bench_kineto,
    calc_diff, count_bytes
)
from deep_gemm.utils import align
from generators import get_arch_major


@filter_test(lambda: get_arch_major() >= 9)
def test_hc_prenorm_gemm() -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Testing hyperconnection prenorm GEMM:')
    for m in (13, 137, 4096, 8192):
        for n, k in [(24, 28672), (24, 7680), (24, 7168)]:
            for num_splits in [None, 16]:
                a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((n, k), dtype=torch.float, device='cuda')
                d = torch.empty((m, n), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
                s = torch.empty((m, ), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m), dtype=torch.float, device='cuda')
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
                final_d = d if num_splits is None else d.sum(0)
                final_s = s if num_splits is None else s.sum(0)

                ref_d = a.float() @ b.T
                ref_s = a.float().square().sum(-1)

                diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
                assert diff < 1e-8, f'{m=}, {n=}, {k=}, {diff:.10f}'

                t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits), 'tf32_hc_prenorm_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits={(num_splits or 0):2}): '
                      f'{t * 1e6:4.0f} us | '
                      f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                      f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
    print()


@pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10, 12)',
    reason='requires a supported HC architecture',
)
@pytest.mark.parametrize('splits', (None, 1, 3))
@pytest.mark.parametrize('zero_batch_stride', (False, True))
def test_hc_prenorm_empty_layout_contract(splits, zero_batch_stride):
    n, k = 24, 64
    a = torch.empty((0, k), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((n, k), dtype=torch.float32, device='cuda')
    storage = torch.full((16,), 19, dtype=torch.float32, device='cuda')
    shape = (0, n) if splits is None else (splits, 0, n)
    strides = (n, 1) if splits is None else (0 if zero_batch_stride else n, n, 1)
    d = storage.as_strided(shape, strides, 8)
    s = torch.empty((0,) if splits is None else (splits, 0), device='cuda')
    if get_arch_major() != 12 and splits is not None and not zero_batch_stride:
        with pytest.raises(RuntimeError, match=r't.stride\(0\)'):
            deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=splits)
    else:
        deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=splits)
    assert d.numel() == s.numel() == 0
    assert (storage == 19).all()


@pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)
@pytest.mark.parametrize('invalid', ('b_cpu', 'd_cpu', 's_cpu', 'a_dtype', 'b_dtype', 'd_dtype',
                                   's_dtype', 'd_shape', 's_shape', 'zero_n', 'zero_k', 'zero_splits'))
def test_sm120_hc_prenorm_empty_validation(invalid):
    n, k, splits = 24, 64, 3
    if invalid == 'zero_n':
        n = 0
    if invalid == 'zero_k':
        k = 0
    if invalid == 'zero_splits':
        splits = 0
    tensors = dict(
        a=torch.empty((0, k), dtype=torch.bfloat16, device='cuda'),
        b=torch.ones((n, k), device='cuda'),
        d=torch.empty((splits, 0, n), device='cuda'),
        s=torch.empty((splits, 0), device='cuda'),
    )
    name, kind = invalid.split('_', 1)
    if kind == 'cpu':
        tensors[name] = tensors[name].cpu()
    elif kind == 'dtype':
        tensors[name] = tensors[name].to(torch.float16)
    elif kind == 'shape':
        tensors[name] = torch.empty((splits, 0, n + 1) if name == 'd' else (splits + 1, 0), device='cuda')
    with pytest.raises(RuntimeError, match='Assertion'):
        deep_gemm.tf32_hc_prenorm_gemm(tensors['a'], tensors['b'], tensors['d'], tensors['s'], num_splits=splits)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_hc_prenorm_gemm()
