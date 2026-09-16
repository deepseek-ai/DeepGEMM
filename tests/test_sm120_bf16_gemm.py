import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, test_filter
from sm120_exercise import (
    exercise_sm120_bf16_native,
    exercise_sm120_grouped_bf16,
    exercise_sm120_k_grouped,
)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_bf16_native_layouts_alpha():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            for alpha in (None, 0.5, -1.0, 2.0):
                for c_mode in ('none', 'same', 'different'):
                    exercise_sm120_bf16_native(layout, (128, 128, 256), out_dtype, alpha, c_mode)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_bf16_native_strided_tails():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            shapes = ((1, 8, 13), (65, 18, 67), (129, 34, 129), (2049, 66, 65))
            for m, n, k in shapes:
                native_k = k if layout == 'nt' else (k + 7) // 8 * 8
                exercise_sm120_bf16_native(layout, (m, n, native_k), out_dtype, -1.0, 'different', padded=True, offset=1)
            exercise_sm120_bf16_native(layout, (67, 30, 63 if layout == 'nt' else 64), out_dtype, 0.5, 'same', padded=True)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_bf16_native_alpha_zero():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            for c_mode in ('none', 'same', 'different'):
                exercise_sm120_bf16_native(layout, (65, 18, 67 if layout == 'nt' else 72), out_dtype, 0.0, c_mode, padded=True, offset=1)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_bf16_native_graph_pdl():
    assert get_arch_major() == 12
    for layout in ('nt', 'nn', 'tn', 'tt'):
        for out_dtype in (torch.bfloat16, torch.float32):
            for pdl in (False, True):
                for c_mode, alpha in (('none', 2.0), ('same', 0.5), ('different', -1.0), ('same', 0.0)):
                    exercise_sm120_bf16_native(layout, (65, 34, 67 if layout == 'nt' else 72), out_dtype, alpha, c_mode,
                                                padded=True, offset=1, graph=True, pdl=pdl)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_grouped_bf16_alignment():
    assert get_arch_major() == 12
    old = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        for expected, alignment in ((None, 128), (0, 32), (1, 32), (32, 32), (33, 64), (64, 64), (65, 128)):
            actual = (deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout() if expected is None else
                      deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected))
            assert actual == alignment
            deep_gemm.set_mk_alignment_for_contiguous_layout(actual)
            assert deep_gemm.get_mk_alignment_for_contiguous_layout() == alignment
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_grouped_bf16_contiguous():
    for alignment in (32, 64, 128):
        for mode in ('labels', 'psum'):
            for nn in (False, True):
                for zero_padding in (False, True):
                    for groups in (3, 5):
                        exercise_sm120_grouped_bf16(mode, alignment, groups, n=64 if groups == 3 else 72,
                                                    zero_padding=zero_padding, nn=nn)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_grouped_bf16_masked():
    for alignment in (32, 64, 128):
        for groups in (3, 5):
            for all_empty in (False, True):
                exercise_sm120_grouped_bf16('masked', alignment, groups, all_empty=all_empty)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_grouped_bf16_graph_padding():
    for alignment in (32, 64, 128):
        for mode in ('labels', 'psum', 'masked'):
            for pdl in (False, True):
                exercise_sm120_grouped_bf16(mode, alignment, 5, zero_padding=pdl and mode != 'masked',
                                            graph=True, pdl=pdl, nn=mode == 'psum')


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_grouped_bf16_empty():
    for alignment in (32, 64, 128):
        for mode in ('labels', 'psum'):
            exercise_sm120_grouped_bf16(mode, alignment, 3, zero_padding=True, all_empty=True)
    a = torch.empty((0, 67), device='cuda', dtype=torch.bfloat16)
    b = torch.empty((3, 17, 67), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((0, 17), device='cuda', dtype=torch.bfloat16)
    for psum in (False, True):
        metadata = torch.zeros(3 if psum else 0, device='cuda', dtype=torch.int32)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, metadata, use_psum_layout=psum,
                                                  ensure_zero_padding=True)
        assert d.numel() == 0


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_k_grouped_bf16_contracts():
    for groups in (1, 3, 8):
        for psum in (False, True):
            for di, dtype in enumerate((torch.bfloat16, torch.float32)):
                for ci, c_mode in enumerate(('none', 'same', 'different')):
                    exercise_sm120_k_grouped('bf16_tn', groups, 128 if (di + ci) % 2 else 256, psum,
                                            dtype, c_mode, shape=((24, 40), (32, 32), (40, 72))[ci],
                                            ks_mode=('list', 'none', 'empty')[ci] if psum else 'list')


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_k_grouped_bf16_graph():
    for dtype in (torch.bfloat16, torch.float32):
        for ci, c_mode in enumerate(('none', 'same', 'different')):
            exercise_sm120_k_grouped('bf16_tn', 8, 128 if ci % 2 else 256, True, dtype, c_mode,
                                    shape=(40, 72), graph=True, pdl=ci % 2 == 1, ks_mode='none')


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_k_grouped_bf16_empty():
    for c_mode in ('none', 'same', 'different'):
        exercise_sm120_k_grouped('bf16_tn', 3, 128, True, torch.float32, c_mode, empty=True, ks_mode='empty')


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('logical_ks', ((154, 147, 121, 109, 128, 120, 149, 112), (121, 121, 121)))
@pytest.mark.parametrize('alias', (False, True))
def test_sm120_k_grouped_bf16_descriptor_reuse(logical_ks, alias):
    m, n = 768, 2048
    host_ks = [(k + 127) // 128 * 128 for k in logical_ks]
    generator = torch.Generator().manual_seed(0)
    av = torch.zeros((sum(host_ks), m), dtype=torch.bfloat16)
    bv = torch.zeros((sum(host_ks), n), dtype=torch.bfloat16)
    cv = torch.randn((len(host_ks), m, n), generator=generator) * 32
    expected, start = [], 0
    for g, (logical_k, host_k) in enumerate(zip(logical_ks, host_ks)):
        av[start:start + logical_k] = torch.randn((logical_k, m), generator=generator, dtype=torch.bfloat16)
        bv[start:start + logical_k] = torch.randn((logical_k, n), generator=generator, dtype=torch.bfloat16)
        expected.append((av[start:start + host_k].double().T @ bv[start:start + host_k].double()
                         + cv[g].double()).float())
        start += host_k
    expected = torch.stack(expected)
    a, b, initial = av.cuda(), bv.cuda(), cv.cuda()
    metadata = torch.tensor(host_ks, dtype=torch.int32, device='cuda')
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_num_sms(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        first = None
        for _ in range(8):
            d = initial.clone()
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, host_ks, metadata, d if alias else initial)
            actual = d.cpu()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-4)
            if first is not None:
                torch.testing.assert_close(actual, first, rtol=0, atol=0)
            first = actual
        torch.testing.assert_close(initial.cpu(), cv, rtol=0, atol=0)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_sm120_bf16_native_layouts_alpha()
    test_sm120_bf16_native_strided_tails()
    test_sm120_bf16_native_alpha_zero()
    test_sm120_bf16_native_graph_pdl()
    test_sm120_grouped_bf16_alignment()
    test_sm120_grouped_bf16_contiguous()
    test_sm120_grouped_bf16_masked()
    test_sm120_grouped_bf16_graph_padding()
    test_sm120_grouped_bf16_empty()
    test_sm120_k_grouped_bf16_contracts()
    test_sm120_k_grouped_bf16_graph()
    test_sm120_k_grouped_bf16_empty()
    for logical_ks in ((154, 147, 121, 109, 128, 120, 149, 112), (121, 121, 121)):
        for alias in (False, True):
            test_sm120_k_grouped_bf16_descriptor_reuse(logical_ks=logical_ks, alias=alias)
