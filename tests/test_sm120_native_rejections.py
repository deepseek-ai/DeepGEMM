import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, test_filter


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
@pytest.mark.parametrize('invalid', ('a_offset', 'a_stride', 'd_offset', 'd_stride'))
def test_sm120_native_tma_rejections(dtype, invalid):
    a = torch.zeros((64, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.zeros_like(a)
    d = torch.full((64, 64), 7, dtype=dtype, device='cuda')
    if invalid == 'a_offset':
        a = torch.zeros(64 * 128 + 1, dtype=a.dtype, device='cuda')[1:].view(64, 128)
    if invalid == 'a_stride':
        a = torch.zeros((64, 129), dtype=a.dtype, device='cuda')[:, :128]
    if invalid == 'd_offset':
        d = torch.full((64 * 64 + 1,), 7, dtype=dtype, device='cuda')[1:].view(64, 64)
    if invalid == 'd_stride':
        d = torch.full((64, 65), 7, dtype=dtype, device='cuda')[:, :64]
    with pytest.raises(RuntimeError, match='data_ptr|num_gmem_outer_stride_bytes'):
        deep_gemm.bf16_gemm_nt(a, b, d)
    assert (d == 7).all()


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_native_alignment32_small_n_rejection():
    a = torch.zeros((32, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.zeros((1, 17, 128), dtype=torch.bfloat16, device='cuda')
    d = torch.full((32, 24), 7, dtype=torch.bfloat16, device='cuda')[:, :17]
    labels = torch.zeros(32, dtype=torch.int32, device='cuda')
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(32)
        with pytest.raises(RuntimeError, match='m-grouped with N <= 32 requires alignment >= 64'):
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, labels, ensure_zero_padding=False)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(previous)
    assert (d == 7).all()


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('layout', ('nn', 'tn', 'tt'))
def test_sm120_native_repacked_odd_k_rejection(layout):
    a = torch.zeros((64, 67), dtype=torch.bfloat16, device='cuda')
    b = torch.zeros((64, 67), dtype=torch.bfloat16, device='cuda')
    aa = a if layout[0] == 'n' else a.T.contiguous()
    bb = b if layout[1] == 't' else b.T.contiguous()
    d = torch.full((64, 64), 7, dtype=torch.bfloat16, device='cuda')
    with pytest.raises(RuntimeError, match='num_gmem_outer_stride_bytes'):
        getattr(deep_gemm, f'bf16_gemm_{layout}')(aa, bb, d)
    assert (d == 7).all()


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('quant', (False, True))
def test_sm120_native_k_grouped_small_stride_rejection(quant):
    m, n = (20, 36) if quant else (3, 5)
    dtype = torch.float8_e4m3fn if quant else torch.bfloat16
    a = torch.zeros((128, m), dtype=dtype, device='cuda')
    b = torch.zeros((128, n), dtype=dtype, device='cuda')
    if quant:
        a, b = (a, torch.ones((1, m), device='cuda')), (b, torch.ones((1, n), device='cuda'))
    d = torch.full((1, m, n), 7, dtype=torch.bfloat16, device='cuda')
    metadata = torch.tensor([128], dtype=torch.int32, device='cuda')
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        with pytest.raises(RuntimeError, match='num_gmem_outer_stride_bytes'):
            getattr(deep_gemm, f'k_grouped_{"fp8" if quant else "bf16"}_gemm_tn_contiguous')(
                a, b, d, [128], metadata)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(previous)
    assert (d == 7).all()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    for dtype in (torch.bfloat16, torch.float32):
        for invalid in ('a_offset', 'a_stride', 'd_offset', 'd_stride'):
            test_sm120_native_tma_rejections(dtype=dtype, invalid=invalid)
    test_sm120_native_alignment32_small_n_rejection()
    for layout in ('nn', 'tn', 'tt'):
        test_sm120_native_repacked_odd_k_rejection(layout=layout)
    for quant in (False, True):
        test_sm120_native_k_grouped_small_stride_rejection(quant=quant)
