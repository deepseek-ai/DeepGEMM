import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, test_filter


@test_filter(lambda: get_arch_major() >= 9)
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


@test_filter(lambda: get_arch_major() == 12)
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

    for splits in (None, 1, 3):
        for zero_batch_stride in (False, True):
            test_hc_prenorm_empty_layout_contract(splits=splits, zero_batch_stride=zero_batch_stride)
    for invalid in ('b_cpu', 'd_cpu', 's_cpu', 'a_dtype', 'b_dtype', 'd_dtype', 's_dtype', 'd_shape', 's_shape', 'zero_n', 'zero_k', 'zero_splits'):
        test_sm120_hc_prenorm_empty_validation(invalid=invalid)
