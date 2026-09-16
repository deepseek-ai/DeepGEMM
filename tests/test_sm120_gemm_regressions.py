import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, calc_diff, test_filter
from deep_gemm.utils import align
from sm120_reference_heuristic import predict_dense_fp8
from sm120_exercise import (
    exercise_sm120_bf16_native,
    exercise_sm120_dense_fp8_fp4,
    exercise_sm120_k_grouped,
    sm120_dense_device_matrix,
    sm120_dense_quantized,
    sm120_quant_grouped_intervals,
)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('layout', ('nt', 'nn', 'tn', 'tt'))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_bf16(layout, dtype):
    exercise_sm120_bf16_native(layout, (128, 128, 256), dtype, -0.5, 'different', graph=True)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('fmt', ((False, False), (False, True), (True, False), (True, True)))
@pytest.mark.parametrize('grans', ((32, 32), (32, 128), (128, 32), (128, 128)))
def test_sm120_sf_branches(fmt, grans):
    exercise_sm120_dense_fp8_fp4(fmt, 'nt', (128, 128, 512), torch.float32,
                                -0.5, 'different', 'packed', grans)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('m', (8, 16, 17))
@pytest.mark.parametrize('fmt', ((False, False), (True, True)))
def test_sm120_swap_boundary(m, fmt):
    exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, 128, 512), torch.bfloat16,
                                0.5, 'none', 'packed', (32, 128))


@test_filter(lambda: get_arch_major() == 12)
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


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
@pytest.mark.parametrize('alpha', (None, 0.0, -0.5, 2.0))
@pytest.mark.parametrize('c_mode', ('none', 'same', 'different'))
@pytest.mark.parametrize('padded', (False, True))
def test_sm120_split_k(dtype, alpha, c_mode, padded):
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(32)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        prediction = predict_dense_fp8(32, 256, 16384, deep_gemm.get_num_sms(),
                                       output_bytes=torch.empty((), dtype=dtype).element_size())
        assert prediction['split_k'] > 1, prediction
        exercise_sm120_dense_fp8_fp4((False, False), 'nt', (32, 256, 16384), dtype,
                                    alpha, c_mode, 'packed', (128, 128), padded=padded,
                                    graph=True, pdl=True)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
@pytest.mark.parametrize('alpha', (None, 0.0, -0.5, 2.0))
def test_sm120_split_k_swapped_strides(dtype, alpha):
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_num_sms(32)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        prediction = predict_dense_fp8(128, 8, 16384, 32, swapped=True,
                                       output_bytes=torch.empty((), dtype=dtype).element_size())
        assert prediction['split_k'] > 1 and prediction['swizzle_cd'] == 0, prediction
        exercise_sm120_dense_fp8_fp4((False, False), 'nt', (8, 128, 16384), dtype,
                                    alpha, 'none', 'packed', (128, 128), padded=True,
                                    graph=True, pdl=True)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('api', ('bf16_tn', 'fp8_tn', 'fp8_nt'))
@pytest.mark.parametrize('c_mode', ('none', 'same', 'different'))
def test_sm120_k_grouped(api, c_mode):
    exercise_sm120_k_grouped(api, 8, 128, False, torch.float32, c_mode,
                            shape=(64, 128), gran=128)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('api', ('bf16_tn', 'fp8_tn'))
@pytest.mark.parametrize('alignment', (128, 256))
def test_sm120_k_psum_adapter(api, alignment):
    exercise_sm120_k_grouped(api, 8, alignment, True, torch.float32, 'same',
                            shape=(64, 128), gran=128, graph=True, ks_mode='none')


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('api,alignment,ks', (
    ('bf16_tn', 32, (0, 128, 256, 0)),
    ('fp8_tn', 32, (0, 64, 192, 0)),
    ('fp8_tn', 64, (0, 64, 192, 0)),
    ('fp8_nt', 128, (0, 128, 256, 0)),
))
@pytest.mark.parametrize('packed', (False, True))
def test_sm120_legacy_k_alignment(api, alignment, ks, packed):
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


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('quant', (False, True))
@pytest.mark.parametrize('mode', ('labels', 'psum', 'masked'))
@pytest.mark.parametrize('alignment', (32, 64, 128))
def test_sm120_m_grouped(quant, mode, alignment):
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


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('mode', ('labels', 'psum', 'masked', 'masked_aligned'))
@pytest.mark.parametrize('pdl', (False, True), ids=('fp8b', 'fp4b-pdl'))
def test_sm120_grouped_tma_independent_streams(mode, pdl):
    aligned_masked = mode == 'masked_aligned'
    mode = 'masked' if aligned_masked else mode
    groups, alignment, k = 2 if aligned_masked else 3, 64, 384
    m, n = (128, 72) if aligned_masked else ((67, 24) if mode == 'masked' else (6 * alignment, 33))
    shape = (groups, m, n) if mode == 'masked' else (m, n)
    jobs = []
    for phase in (0, 1):
        aa = [sm120_dense_quantized(m, k, False, 32, phase + g * 2)
              for g in range(groups if mode == 'masked' else 1)]
        bb = [sm120_dense_quantized(n, k, pdl, 128, phase + g * 2 + 3) for g in range(groups)]
        av, sa, ar = [torch.stack([item[i] for item in aa]) for i in range(3)]
        if mode != 'masked':
            av, sa, ar = av[0], sa[0], ar[0]
        bv, sb, br = [torch.stack([item[i] for item in bb]) for i in range(3)]
        lengths = [65, 0, 61] if phase == 0 else [0, 63, 67]
        if aligned_masked:
            lengths = [65, 0] if phase == 0 else [0, 127]
        valid = torch.zeros(shape[:-1], dtype=torch.bool)
        expected = torch.zeros(shape)
        if mode == 'masked':
            meta = torch.tensor(lengths, dtype=torch.int32)
            for g, length in enumerate(lengths):
                valid[g, :length] = True
                expected[g, :length] = ar[g, :length] @ br[g].T
            storage = torch.full((8 + groups * m * n + 16,), 19, dtype=torch.bfloat16, device='cuda')
            d = storage.as_strided(shape, (m * n, n, 1), 8)
        else:
            intervals = sm120_quant_grouped_intervals(lengths, alignment)
            labels = torch.full((m,), -1, dtype=torch.int32)
            for g, (start, end) in enumerate(intervals):
                labels[start:end] = g
                valid[start:end] = True
                expected[start:end] = ar[start:end] @ br[g].T
            meta = labels if mode == 'labels' else torch.tensor([end for _, end in intervals], dtype=torch.int32)
            d, storage = sm120_dense_device_matrix(torch.zeros(shape, dtype=torch.bfloat16), True, 1, 7)
        guard = torch.zeros_like(storage, dtype=torch.bool)
        guard.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
        jobs.append(((av.cuda(), sa.cuda()), (bv.cuda(), sb.cuda()), d, meta.cuda(),
                     storage, guard, expected, valid, torch.cuda.Stream()))

    def run(job):
        a, b, d, metadata = job[:4]
        d.fill_(7)
        kwargs = dict(recipe_a=(1, 32), recipe_b=(1, 128))
        if mode == 'masked':
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, metadata, 33, **kwargs)
        else:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a, b, d, metadata, use_psum_layout=mode == 'psum', ensure_zero_padding=True, **kwargs)

    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(pdl)
        for job in jobs:
            run(job)
        current = torch.cuda.current_stream()
        for job in jobs:
            job[-1].wait_stream(current)
        snapshots = []
        for _ in range(3):
            for job in jobs:
                with torch.cuda.stream(job[-1]):
                    run(job)
                    snapshots.append((job, job[2].clone()))
        for job in jobs:
            current.wait_stream(job[-1])
        for job, snapshot in snapshots:
            _, _, d, metadata, storage, guard, expected, valid, _ = job
            actual = snapshot.cpu().float()
            assert torch.isfinite(actual[valid]).all()
            torch.testing.assert_close(actual[valid], expected[valid], rtol=0.008, atol=0.02)
            if mode == 'labels':
                assert (actual[~valid] == 0).all()
            elif mode == 'psum':
                ends = metadata.cpu().tolist()
                for end in ends:
                    assert (actual[end:align(end, alignment)] == 0).all()
                assert (actual[align(ends[-1], alignment):] == 7).all()
            assert (storage[~guard] == 19).all(), 'Independent streams modified output guard cells'
    finally:
        for job in jobs:
            job[-1].synchronize()
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('mode', ('dense', 'labels', 'psum', 'masked'))
@pytest.mark.parametrize('fmt', ((False, False), (False, True), (True, False), (True, True)))
@pytest.mark.parametrize('gran_k', (32, 128))
@pytest.mark.parametrize('float_scales', ((False, False), (True, False), (False, True), (True, True)))
def test_sm120_gemm_disable_ue8m0_cast(mode, fmt, gran_k, float_scales):
    groups, rows, n, k = 2, 128, 64, 512
    m = rows if mode in ('dense', 'masked') else groups * rows
    av, sa, ar = sm120_dense_quantized(m, k, fmt[0], gran_k, 0)
    bv, sb, br = sm120_dense_quantized(n, k, fmt[1], gran_k, 2)
    if mode == 'masked':
        av, sa, ar = (v.unsqueeze(0).repeat(groups, 1, 1) for v in (av, sa, ar))
    if mode != 'dense':
        bv, sb, br = (v.unsqueeze(0).repeat(groups, 1, 1) for v in (bv, sb, br))
    a, b, sa, sb = av.cuda(), bv.cuda(), sa.cuda(), sb.cuda()
    packed_a = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sa)
    packed_b = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sb)
    aa = a, sa if float_scales[0] else packed_a
    bb = b, sb if float_scales[1] else packed_b
    if mode == 'dense':
        function, positional, kwargs = deep_gemm.fp8_fp4_gemm_nt, (), {}
        expected = ar @ br.T
    elif mode == 'masked':
        function = deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked
        positional = (torch.full((groups,), rows, dtype=torch.int32, device='cuda'), rows)
        kwargs, expected = {}, ar @ br.mT
    else:
        function = deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous
        metadata = (torch.arange(groups, device='cuda', dtype=torch.int32).repeat_interleave(rows)
                    if mode == 'labels' else torch.arange(1, groups + 1, device='cuda', dtype=torch.int32) * rows)
        positional, kwargs = (metadata,), dict(use_psum_layout=mode == 'psum')
        expected = torch.cat([ar[g * rows:(g + 1) * rows] @ br[g].T for g in range(groups)])
    d = torch.full(expected.shape, 7, dtype=torch.bfloat16, device='cuda')
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(rows)
        for recipes in (dict(recipe=(1, 1, gran_k)), dict(recipe_a=(1, gran_k), recipe_b=(1, gran_k))):
            for disable in (True, False):
                d.fill_(7)
                if disable and any(float_scales):
                    torch.cuda.synchronize()
                    with pytest.raises(RuntimeError, match='disable_ue8m0_cast'):
                        function(aa, bb, d, *positional, disable_ue8m0_cast=disable, **kwargs, **recipes)
                    torch.cuda.synchronize()
                    assert (d == 7).all()
                else:
                    function(aa, bb, d, *positional, disable_ue8m0_cast=disable, **kwargs, **recipes)
                    actual = d.cpu().float()
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.02)
                    assert calc_diff(actual, expected) < 1e-5
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(previous)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('mode, padded_a', (('labels', False), ('labels', True), ('psum', False), ('masked', False)))
@pytest.mark.parametrize('fp4_b', (False, True))
@pytest.mark.parametrize('gran_k', (32, 128))
@pytest.mark.parametrize('k', (384, 512, 640, 1024, 1536))
def test_sm120_grouped_pipeline_stage_reuse(mode, padded_a, fp4_b, gran_k, k):
    groups, rows, n = 2, 128, 64
    m = rows if mode == 'masked' else groups * rows
    av, sa, ar = sm120_dense_quantized(m, k, False, gran_k, 0)
    bv, sb, br = sm120_dense_quantized(n, k, fp4_b, gran_k, 2)
    if mode == 'masked':
        av, sa, ar = (v.unsqueeze(0).repeat(groups, 1, 1) for v in (av, sa, ar))
    bv, sb, br = (v.unsqueeze(0).repeat(groups, 1, 1) for v in (bv, sb, br))
    a = av.cuda()
    if padded_a:
        storage_a = torch.zeros((m, k + 16), dtype=a.dtype, device='cuda')
        storage_a[:, :k].copy_(a)
        a = storage_a[:, :k]
    assert a.is_contiguous() != padded_a
    b = bv.cuda()
    sa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sa.cuda())
    sb = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sb.cuda())
    if mode == 'masked':
        expected = ar @ br.mT
        metadata = torch.full((groups,), rows, dtype=torch.int32, device='cuda')
    else:
        expected = torch.cat([ar[g * rows:(g + 1) * rows] @ br[g].T for g in range(groups)])
        metadata = (torch.arange(groups, dtype=torch.int32, device='cuda').repeat_interleave(rows)
                    if mode == 'labels' else torch.arange(1, groups + 1, dtype=torch.int32, device='cuda') * rows)
    storage_d = torch.full((expected.numel() + 16,), 19, dtype=torch.bfloat16, device='cuda')
    d = storage_d[8:-8].view(expected.shape)
    previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(rows)
        for _ in range(3):
            d.fill_(float('nan'))
            kwargs = dict(recipe=(1, 1, gran_k), disable_ue8m0_cast=True)
            if mode == 'masked':
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((a, sa), (b, sb), d, metadata, rows, **kwargs)
            else:
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    (a, sa), (b, sb), d, metadata, use_psum_layout=mode == 'psum', **kwargs)
            actual = d.cpu().float()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.02)
            assert calc_diff(actual, expected) < 1e-5
            assert (storage_d[:8] == 19).all() and (storage_d[-8:] == 19).all()
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(previous)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    for layout in ('nt', 'nn', 'tn', 'tt'):
        for dtype in (torch.bfloat16, torch.float32):
            test_sm120_bf16(layout=layout, dtype=dtype)
    for fmt in ((False, False), (False, True), (True, False), (True, True)):
        for grans in ((32, 32), (32, 128), (128, 32), (128, 128)):
            test_sm120_sf_branches(fmt=fmt, grans=grans)
    for m in (8, 16, 17):
        for fmt in ((False, False), (True, True)):
            test_sm120_swap_boundary(m=m, fmt=fmt)
    test_sm120_subtile()
    for dtype in (torch.bfloat16, torch.float32):
        for alpha in (None, 0.0, -0.5, 2.0):
            for c_mode in ('none', 'same', 'different'):
                for padded in (False, True):
                    test_sm120_split_k(dtype=dtype, alpha=alpha, c_mode=c_mode, padded=padded)
    for dtype in (torch.bfloat16, torch.float32):
        for alpha in (None, 0.0, -0.5, 2.0):
            test_sm120_split_k_swapped_strides(dtype=dtype, alpha=alpha)
    for api in ('bf16_tn', 'fp8_tn', 'fp8_nt'):
        for c_mode in ('none', 'same', 'different'):
            test_sm120_k_grouped(api=api, c_mode=c_mode)
    for api in ('bf16_tn', 'fp8_tn'):
        for alignment in (128, 256):
            test_sm120_k_psum_adapter(api=api, alignment=alignment)
    for api, alignment, ks in (('bf16_tn', 32, (0, 128, 256, 0)), ('fp8_tn', 32, (0, 64, 192, 0)), ('fp8_tn', 64, (0, 64, 192, 0)), ('fp8_nt', 128, (0, 128, 256, 0))):
        for packed in (False, True):
            test_sm120_legacy_k_alignment(api=api, alignment=alignment, ks=ks, packed=packed)
    for quant in (False, True):
        for mode in ('labels', 'psum', 'masked'):
            for alignment in (32, 64, 128):
                test_sm120_m_grouped(quant=quant, mode=mode, alignment=alignment)
    for mode in ('labels', 'psum', 'masked', 'masked_aligned'):
        for pdl in (False, True):
            test_sm120_grouped_tma_independent_streams(mode=mode, pdl=pdl)
    for mode in ('dense', 'labels', 'psum', 'masked'):
        for fmt in ((False, False), (False, True), (True, False), (True, True)):
            for gran_k in (32, 128):
                for float_scales in ((False, False), (True, False), (False, True), (True, True)):
                    test_sm120_gemm_disable_ue8m0_cast(mode=mode, fmt=fmt, gran_k=gran_k, float_scales=float_scales)
    for mode, padded_a in (('labels', False), ('labels', True), ('psum', False), ('masked', False)):
        for fp4_b in (False, True):
            for gran_k in (32, 128):
                for k in (384, 512, 640, 1024, 1536):
                    test_sm120_grouped_pipeline_stage_reuse(mode=mode, padded_a=padded_a, fp4_b=fp4_b, gran_k=gran_k, k=k)
