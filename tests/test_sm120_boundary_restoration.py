import pytest
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major
from test_fp8_fp4 import sm120_dense_quantized


@pytest.mark.parametrize('mn_a,mn_b', ((False, True), (True, False), (True, True)))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_restored_symmetric_fp4_mn(mn_a, mn_b, dtype):
    assert get_arch_major() == 12
    av, sa, ar = sm120_dense_quantized(32, 128, True, 32, 0, mn_major=mn_a)
    bv, sb, br = sm120_dense_quantized(32, 128, True, 32, 1, mn_major=mn_b)
    a, b = (av.cuda(), sa.cuda()), (bv.cuda(), sb.cuda())
    assert a[0].stride(0 if mn_a else 1) == 1
    assert b[0].stride(0 if mn_b else 1) == 1
    d = torch.empty((32, 32), dtype=dtype, device='cuda')
    expected = ar @ br.T
    for _ in range(3):
        d.fill_(float('nan'))
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, recipe=(1, 1, 32))
        torch.testing.assert_close(d.cpu().float(), expected,
                                   rtol=0.008 if dtype == torch.bfloat16 else 2e-4,
                                   atol=0.02 if dtype == torch.bfloat16 else 1e-4)


@pytest.mark.parametrize('api', ('tn', 'nt'))
@pytest.mark.parametrize('orientation', ('sf_k_mn', 'mn_sf_k'))
@pytest.mark.parametrize('contiguous', (False, True))
def test_sm120_restored_legacy_sf_orientation(api, orientation, contiguous):
    assert get_arch_major() == 12
    m, n, gran, ks = 64, 128, 128, (0, 128, 256, 0)
    a_parts, b_parts, sa_parts, sb_parts, expected = [], [], [], [], []
    for g, k in enumerate(ks):
        av, sa, ar = sm120_dense_quantized(m, k, False, gran, g)
        bv, sb, br = sm120_dense_quantized(n, k, False, gran, g + 1)
        a_parts.append(av.T.contiguous())
        b_parts.append(bv.T.contiguous())
        sa_parts.append(sa.T.contiguous())
        sb_parts.append(sb.T.contiguous())
        expected.append(ar @ br.T + 0.25)
    if api == 'tn':
        a, b = torch.cat(a_parts).cuda(), torch.cat(b_parts).cuda()
    else:
        a = torch.cat([p.T.contiguous().flatten() for p in a_parts]).view(m, -1).cuda()
        b = torch.cat([p.T.contiguous().flatten() for p in b_parts]).view(n, -1).cuda()

    def sf_input(parts):
        value = torch.cat(parts)
        if orientation == 'mn_sf_k':
            value = value.T
        if contiguous:
            return value.contiguous().cuda()
        storage = torch.empty((value.shape[0], value.shape[1] * 2), device='cuda')
        result = storage[:, ::2]
        result.copy_(value)
        assert not result.is_contiguous()
        return result

    sa, sb = sf_input(sa_parts), sf_input(sb_parts)
    d = torch.empty((len(ks), m, n), dtype=torch.float32, device='cuda')
    metadata = torch.tensor(ks, dtype=torch.int32, device='cuda')
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        for _ in range(3):
            d.fill_(0.25)
            getattr(deep_gemm, f'k_grouped_fp8_gemm_{api}_contiguous')(
                (a, sa), (b, sb), d, list(ks), metadata, c=d, recipe=(1, 1, gran))
            torch.testing.assert_close(d.cpu(), torch.stack(expected), rtol=2e-4, atol=1e-4)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
