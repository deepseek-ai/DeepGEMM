import pytest
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major
from sm120_test_storage import native_matrix
from test_bf16 import exercise_sm120_bf16_native, sm120_grouped_intervals


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)


@pytest.mark.parametrize('n', (1, 7, 17, 33))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_bf16_odd_n_dense(n, dtype):
    for mi, m in enumerate((1, 2, 65, 129)):
        for li, layout in enumerate(('nt', 'nn', 'tn', 'tt')):
            exercise_sm120_bf16_native(
                layout, (m, n, 128), dtype, (None, 0.5, -1.0, 2.0)[(mi + li) % 4],
                ('none', 'same', 'different')[(mi + li) % 3], padded=True, offset=1)


@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
@pytest.mark.parametrize('c_mode', ('none', 'same', 'different'))
def test_sm120_bf16_odd_n_graph_alpha_zero(dtype, c_mode):
    for i, n in enumerate((1, 7, 17, 33)):
        for alpha in (0.0, -0.5):
            exercise_sm120_bf16_native(
                ('nt', 'nn', 'tn', 'tt')[i], (65, n, 128), dtype, alpha,
                c_mode, padded=True, offset=1, graph=True, pdl=i % 2 == 1)


def exercise_sm120_bf16_odd_n_grouped(mode, n, nn, zero_padding, graph):
    assert get_arch_major() == 12
    assert mode in ('labels', 'psum') and not nn
    groups, alignment, k = 3, 128, 128
    m = 7 * alignment
    shape = (m, n)
    av = ((torch.arange(m * k).reshape(m, k) % 13 - 6).float() / 4).to(torch.bfloat16)
    bv = ((torch.arange(groups * n * k).reshape(groups, n, k) % 11 - 5).float() / 4).to(torch.bfloat16)
    a = av.cuda()
    b_flat, b_storage = native_matrix(bv.reshape(-1, k), False, 1)
    b = b_flat.view(bv.shape)
    assert b.stride() == (n * k, k, 1)
    d, storage = native_matrix(torch.zeros(shape, dtype=torch.bfloat16), True, 1, extra=7)
    guard = torch.zeros_like(storage, dtype=torch.bool)
    guard.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    metadata = torch.empty(groups if mode == 'psum' else m, dtype=torch.int32, device='cuda')

    def inputs(phase):
        lengths = ((1, 2, 129), (0, 0, 0), (127, 1, 0))[phase]
        valid = torch.zeros(m, dtype=torch.bool)
        expected = torch.zeros(shape)
        values = av.clone() * (-1 if phase == 2 else 1)
        labels = torch.full((m,), -1, dtype=torch.int32)
        intervals = sm120_grouped_intervals(lengths, alignment)
        for g, (start, end) in enumerate(intervals):
            valid[start:end] = True
            labels[start:end] = g
            expected[start:end] = values[start:end].float() @ bv[g].float().T
        meta = labels if mode == 'labels' else [end for _, end in intervals]
        values[~valid] = float('nan')
        a.copy_(values)
        metadata.copy_(torch.as_tensor(meta, dtype=torch.int32))
        return expected, valid

    def run():
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            a, b, d, metadata, use_psum_layout=mode == 'psum', ensure_zero_padding=zero_padding)

    def check(expected, valid):
        actual = d.cpu().float()
        assert torch.isfinite(actual[valid]).all()
        torch.testing.assert_close(actual[..., -1][valid], expected[..., -1][valid], rtol=0.008, atol=0.015)
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0.008, atol=0.015)
        if zero_padding:
            zeros = ~valid
            if mode == 'psum':
                zeros = torch.zeros_like(valid)
                for end in metadata.cpu().tolist():
                    zeros[end:(end + alignment - 1) // alignment * alignment] = True
            torch.testing.assert_close(actual[zeros], torch.zeros_like(actual[zeros]), rtol=0, atol=0)
        assert (storage[~guard] == 19).all()
        torch.testing.assert_close(b.cpu(), bv, rtol=0, atol=0)
        assert (b_storage[:8] == 19).all() and (b_storage[8 + bv.numel():] == 19).all()

    old_alignment, old_pdl = deep_gemm.get_mk_alignment_for_contiguous_layout(), deep_gemm.get_pdl()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(graph)
        expected, valid = inputs(0)
        for _ in range(3):
            d.fill_(float('nan'))
            run()
            check(expected, valid)
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in (0, 1, 2):
                expected, valid = inputs(phase)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check(expected, valid)
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
    print(f' > Native SM120 BF16 odd-N grouped: {mode=}, {n=}, {nn=}, {zero_padding=}, {graph=}')


@pytest.mark.parametrize('n', (1, 7, 17, 33))
@pytest.mark.parametrize('mode', ('labels', 'psum'))
def test_sm120_bf16_odd_n_grouped(n, mode):
    for zero_padding in (False, True):
        exercise_sm120_bf16_odd_n_grouped(mode, n, False, zero_padding, graph=True)
