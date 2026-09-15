import pytest
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major
from sm120_reference_heuristic import predict_dense_fp8
from test_bf16 import exercise_sm120_bf16_native, exercise_sm120_k_grouped
from test_fp8_fp4 import exercise_sm120_dense_fp8_fp4, sm120_dense_quantized


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)


@pytest.mark.parametrize('layout', ('nt', 'nn', 'tn', 'tt'))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_bf16(layout, dtype):
    exercise_sm120_bf16_native(layout, (128, 128, 256), dtype, -0.5, 'different', graph=True)


@pytest.mark.parametrize('fmt', ((False, False), (False, True), (True, False), (True, True)))
@pytest.mark.parametrize('grans', ((32, 32), (32, 128), (128, 32), (128, 128)))
def test_sm120_sf_branches(fmt, grans):
    exercise_sm120_dense_fp8_fp4(fmt, 'nt', (128, 128, 512), torch.float32,
                                -0.5, 'different', 'packed', grans)


@pytest.mark.parametrize('m', (8, 16, 17))
@pytest.mark.parametrize('fmt', ((False, False), (True, True)))
def test_sm120_swap_boundary(m, fmt):
    exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, 128, 512), torch.bfloat16,
                                0.5, 'none', 'packed', (32, 128))


def test_sm120_subtile():
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        prediction = predict_dense_fp8(2048, 512, 512, deep_gemm.get_num_sms())
        assert prediction['store_m'] < prediction['block_m'] and prediction['swizzle_cd'] == 128
        exercise_sm120_dense_fp8_fp4((False, False), 'nt', (2048, 512, 512),
                                    torch.bfloat16, None, 'none', 'packed', (128, 128))
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
@pytest.mark.parametrize('alpha', (None, 0.0, -0.5, 2.0))
@pytest.mark.parametrize('c_mode', ('none', 'same', 'different'))
def test_sm120_split_k(dtype, alpha, c_mode):
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        prediction = predict_dense_fp8(32, 256, 16384, deep_gemm.get_num_sms(),
                                       output_bytes=torch.empty((), dtype=dtype).element_size())
        assert prediction['split_k'] > 1, prediction
        exercise_sm120_dense_fp8_fp4((False, False), 'nt', (32, 256, 16384), dtype,
                                    alpha, c_mode, 'packed', (128, 128), graph=True)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@pytest.mark.parametrize('api', ('bf16_tn', 'fp8_tn', 'fp8_nt'))
@pytest.mark.parametrize('c_mode', ('none', 'same', 'different'))
def test_sm120_k_grouped(api, c_mode):
    exercise_sm120_k_grouped(api, 8, 128, False, torch.float32, c_mode,
                            shape=(64, 128), gran=128)


@pytest.mark.parametrize('api', ('bf16_tn', 'fp8_tn'))
@pytest.mark.parametrize('alignment', (128, 256))
def test_sm120_k_psum_adapter(api, alignment):
    exercise_sm120_k_grouped(api, 8, alignment, True, torch.float32, 'same',
                            shape=(64, 128), gran=128, graph=True, ks_mode='none')


@pytest.mark.parametrize('api,alignment,ks', (
    ('bf16_tn', 32, (0, 128, 256, 0)),
    ('fp8_tn', 32, (0, 64, 192, 0)),
    ('fp8_tn', 64, (0, 64, 192, 0)),
    ('fp8_nt', 128, (0, 128, 256, 0)),
))
@pytest.mark.parametrize('packed', (False, True))
def test_sm120_legacy_k_alignment(api, alignment, ks, packed):
    assert get_arch_major() == 12
    m, n, gran = 64, 128, 128
    quant = api != 'bf16_tn'
    dtype = torch.float8_e4m3fn if quant else torch.bfloat16
    av = (((torch.arange(sum(ks) * m).reshape(-1, m) % 13) - 6).float() / 4).to(dtype)
    bv = (((torch.arange(sum(ks) * n).reshape(-1, n) % 11) - 5).float() / 4).to(dtype)
    a, b = av.cuda(), bv.cuda()
    expected = torch.empty((len(ks), m, n))
    cursor = 0
    for g, length in enumerate(ks):
        expected[g] = av[cursor:cursor + length].float().T @ bv[cursor:cursor + length].float()
        cursor += length
    metadata = torch.tensor(ks, dtype=torch.int32, device='cuda')
    if api == 'fp8_nt':
        a = torch.cat([x.T.contiguous().flatten() for x in a.split(ks)]).view(m, -1)
        b = torch.cat([x.T.contiguous().flatten() for x in b.split(ks)]).view(n, -1)
    if quant:
        rows = sum((length + gran - 1) // gran for length in ks)
        sa, sb = torch.ones((rows, m), device='cuda'), torch.ones((rows, n), device='cuda')
        if packed:
            sa = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sa, metadata, list(ks), gran, alignment, False)
            sb = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sb, metadata, list(ks), gran, alignment, False)
        a, b = (a, sa), (b, sb)
    d = torch.empty((len(ks), m, n), device='cuda')
    initial = torch.full_like(d, 0.25)
    function = getattr(deep_gemm, f'k_grouped_{"fp8" if quant else "bf16"}_gemm_{api[-2:]}_contiguous')
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        for _ in range(3):
            d.copy_(initial)
            function(a, b, d, list(ks), metadata, c=d, **(dict(recipe=(1, 1, gran)) if quant else {}))
            torch.testing.assert_close(d.cpu(), expected + 0.25, rtol=2e-4, atol=1e-4)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@pytest.mark.parametrize('quant', (False, True))
@pytest.mark.parametrize('mode', ('labels', 'psum', 'masked'))
@pytest.mark.parametrize('alignment', (32, 64, 128))
def test_sm120_m_grouped(quant, mode, alignment):
    assert get_arch_major() == 12
    groups, capacity, n, k = 3, 2 * alignment, 128, 256
    lengths = (0, alignment + 1, alignment - 3)
    rows = groups * capacity
    if quant:
        av, sa, ar = sm120_dense_quantized(rows, k, False, 128, 0)
        bv, sb, br = sm120_dense_quantized(groups * n, k, False, 128, 1)
        bv, sb, br = bv.reshape(groups, n, k), sb.reshape(groups, n, -1), br.reshape(groups, n, k)
    else:
        av = ((torch.arange(rows * k).view(rows, k) % 13 - 6).float() / 4).to(torch.bfloat16)
        bv = ((torch.arange(groups * n * k).view(groups, n, k) % 11 - 5).float() / 4).to(torch.bfloat16)
        ar, br = av.float(), bv.float()
    expected = torch.zeros((rows, n))
    valid = torch.zeros(rows, dtype=torch.bool)
    labels = torch.full((rows,), -1, dtype=torch.int32)
    ends, cursor = [], 0
    for g, length in enumerate(lengths):
        start = g * capacity if mode == 'masked' else cursor
        end = start + length
        expected[start:end] = ar[start:end] @ br[g].T
        valid[start:end] = True
        labels[start:end] = g
        ends.append(end)
        cursor = (end + alignment - 1) // alignment * alignment
    if mode == 'masked':
        av = av.view(groups, capacity, k)
        if quant:
            sa = sa.view(groups, capacity, -1)
        expected, valid = expected.view(groups, capacity, n), valid.view(groups, capacity)
        metadata = torch.tensor(lengths, dtype=torch.int32, device='cuda')
    else:
        metadata = (labels if mode == 'labels' else torch.tensor(ends, dtype=torch.int32)).cuda()
    a, b = av.cuda(), bv.cuda()
    if quant:
        a, b = (a, sa.cuda()), (b, sb.cuda())
    d = torch.full(expected.shape, 7, dtype=torch.bfloat16, device='cuda')
    family = 'fp8_fp4' if quant else 'bf16'
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        for _ in range(3):
            d.fill_(7)
            kwargs = dict(recipe=(1, 1, 128)) if quant else {}
            if mode == 'masked':
                getattr(deep_gemm, f'm_grouped_{family}_gemm_nt_masked')(a, b, d, metadata, alignment, **kwargs)
            else:
                getattr(deep_gemm, f'm_grouped_{family}_gemm_nt_contiguous')(
                    a, b, d, metadata, use_psum_layout=mode == 'psum', ensure_zero_padding=False, **kwargs)
            torch.testing.assert_close(d.cpu().float()[valid], expected[valid], rtol=0.008, atol=0.02)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
