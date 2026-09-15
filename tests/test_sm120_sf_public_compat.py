import math

import pytest
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from sm120_test_storage import native_matrix
from test_sm120_gemm import fp8_operand


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)


MN_GRANS = (2, 3, 16, 32, 64, 256)


def packed_reference(sf):
    values = sf.contiguous().view(torch.int32).bitwise_right_shift(23).to(torch.uint8)
    padded = torch.zeros((*values.shape[:-1], (values.shape[-1] + 3) // 4 * 4), dtype=torch.uint8)
    padded[..., :values.shape[-1]] = values
    return padded.contiguous().view(torch.int32)


@pytest.mark.parametrize('gran_mn', MN_GRANS)
def test_sm120_sf_arbitrary_mn_transform(gran_mn):
    assert get_arch_major() == 12
    for mn in (gran_mn + 1, 2 * gran_mn + 1):
        for gran_k in (32, 128):
            k = 257
            for groups in (None, 2):
                shape = ((groups,) if groups else ()) + (math.ceil(mn / gran_mn), math.ceil(k / gran_k))
                sf = torch.pow(2.0, (torch.arange(math.prod(shape)).reshape(shape) % 5 - 6).float())
                device_sf = sf.cuda()
                out = deep_gemm.transform_sf_into_required_layout(device_sf, mn, k, (gran_mn, gran_k), num_groups=groups)
                expanded = sf.index_select(-2, torch.arange(mn) // gran_mn)
                assert out.dtype == torch.int32 and out.stride(-2) == 1
                assert out.stride(-1) == (mn + 3) // 4 * 4 and out.data_ptr() % 16 == 0
                if groups:
                    assert out.stride(0) == out.stride(-1) * out.size(-1)
                torch.testing.assert_close(out.cpu(), packed_reference(expanded), rtol=0, atol=0)
                torch.testing.assert_close(device_sf.cpu(), sf, rtol=0, atol=0)


@pytest.mark.parametrize('gran_mn', MN_GRANS)
def test_sm120_sf_arbitrary_mn_dense(gran_mn):
    for m in (16, 32, 33):
        for independent in (False, True):
            n, k = gran_mn + 1, 256
            gran_b = MN_GRANS[(MN_GRANS.index(gran_mn) + 1) % len(MN_GRANS)]
            ka, kb = (32, 128) if independent else (128, 128)
            aa, sa, ar = fp8_operand(1, m, k, gran_mn, ka, 1)
            bb, sb, br = fp8_operand(1, n, k, gran_b, kb, 2)
            a = native_matrix(aa[0], True, 1)[0], sa[0].cuda()
            b = native_matrix(bb[0], True, 1)[0], sb[0].cuda()
            d, storage = native_matrix(torch.zeros((m, n), dtype=torch.float32), True, 1)
            valid = torch.zeros_like(storage, dtype=torch.bool)
            valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
            kwargs = dict(recipe_a=(gran_mn, ka), recipe_b=(gran_b, kb)) if independent else dict(recipe=(gran_mn, gran_b, 128))
            d.fill_(float('nan'))
            deep_gemm.fp8_fp4_gemm_nt(a, b, d, **kwargs)
            expected = ar[0] @ br[0].T
            assert torch.isfinite(d).all()
            torch.testing.assert_close(d.cpu(), expected, rtol=2e-4, atol=1e-4)
            assert calc_diff(d.cpu(), expected) < 1e-8
            assert (storage[~valid] == 19).all()


@pytest.mark.parametrize('gran_mn', (1, 128))
def test_sm120_sf_standalone_float_disable_cast(gran_mn):
    mn, k = 259, 257
    for groups in (None, 2):
        for transposed in (False, True):
            shape = ((groups,) if groups else ()) + (math.ceil(mn / gran_mn), math.ceil(k / 128))
            sf = torch.arange(math.prod(shape)).reshape(shape).float() / 10 + 0.123
            values = sf.transpose(-1, -2).contiguous().transpose(-1, -2) if transposed else sf
            x = values.cuda()
            out = deep_gemm.transform_sf_into_required_layout(x, mn, k, (gran_mn, 128), num_groups=groups, disable_ue8m0_cast=True)
            assert out.dtype == torch.float32 and out.shape == x.shape
            torch.testing.assert_close(out.cpu(), sf, rtol=0, atol=0)
            if gran_mn == 1:
                assert out.stride(-2) == 1 and out.stride(-1) == (mn + 3) // 4 * 4
            else:
                assert out.data_ptr() == x.data_ptr() and out.stride() == x.stride()


def test_sm120_sf_invalid_contracts():
    sf = torch.ones((3, 2), device='cuda')
    for gran_mn in (0, -1):
        with pytest.raises(RuntimeError, match='gran_mn > 0'):
            deep_gemm.transform_sf_into_required_layout(sf, 5, 256, (gran_mn, 128))
    with pytest.raises(RuntimeError, match='sf.size'):
        deep_gemm.transform_sf_into_required_layout(sf, 5, 256, (3, 128))
    integer_sf = torch.zeros((3, 1), dtype=torch.int32, device='cuda')
    with pytest.raises(RuntimeError, match='gran_mn == 1'):
        deep_gemm.transform_sf_into_required_layout(integer_sf, 5, 256, (2, 128))
    for mn_gran, k_gran in ((2, 128), (1, 32)):
        shape = (math.ceil(5 / mn_gran), math.ceil(256 / k_gran))
        with pytest.raises(RuntimeError, match='disable_ue8m0_cast'):
            deep_gemm.transform_sf_into_required_layout(torch.ones(shape, device='cuda'), 5, 256, (mn_gran, k_gran), disable_ue8m0_cast=True)
    a = torch.ones((16, 128), dtype=torch.float8_e4m3fn, device='cuda')
    d = torch.zeros((16, 16), dtype=torch.bfloat16, device='cuda')
    for gran_mn in (1, 128):
        x = torch.ones((math.ceil(16 / gran_mn), 1), device='cuda')
        with pytest.raises(RuntimeError, match='disable_ue8m0_cast'):
            deep_gemm.fp8_fp4_gemm_nt((a, x), (a, x), d, recipe=(gran_mn, gran_mn, 128), disable_ue8m0_cast=True)


@pytest.mark.parametrize('gran_mn', MN_GRANS)
def test_sm120_sf_psum_metadata(gran_mn):
    mn, k = 384, 256
    sf = torch.pow(2.0, (torch.arange(math.ceil(mn / gran_mn) * 2).reshape(-1, 2) % 3 - 5).float())
    layout = torch.tensor([1, 130, 259], dtype=torch.int32, device='cuda')
    old = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        out = deep_gemm.transform_sf_into_required_layout(sf.cuda(), mn, k, (gran_mn, 128), psum_layout=layout)
        expanded = sf[torch.arange(mn) // gran_mn]
        valid = torch.zeros(mn, dtype=torch.bool)
        valid[0:1] = True
        valid[128:130] = True
        valid[256:259] = True
        torch.testing.assert_close(out.cpu()[valid], packed_reference(expanded)[valid], rtol=0, atol=0)
        assert (out.cpu()[~valid] == 0).all()
        with pytest.raises(RuntimeError, match='psum_layout'):
            deep_gemm.transform_sf_into_required_layout(sf.cuda(), mn, k, (gran_mn, 128), psum_layout=layout.to(torch.int64))
        with pytest.raises(RuntimeError, match='psum_layout'):
            deep_gemm.transform_sf_into_required_layout(sf.cuda().unsqueeze(0), mn, k, (gran_mn, 128), num_groups=1, psum_layout=layout)
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old)
