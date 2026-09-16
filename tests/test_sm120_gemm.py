import math

import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major, test_filter
from sm120_test_storage import native_matrix
from sm120_exercise import (
    exercise_sm120_dense_fp8_fp4,
    exercise_sm120_k_grouped,
    exercise_sm120_quant_grouped,
    sm120_dense_quantized,
)


def guarded_compact(values):
    offset = 16 // values.element_size()
    storage = torch.full((offset + values.numel() + offset,), 19, dtype=values.dtype, device='cuda')
    tensor = storage[offset:offset + values.numel()].view(values.shape)
    tensor.copy_(values)
    return tensor, storage, offset


def check_compact_guards(storage, offset):
    assert (storage[:offset] == 19).all() and (storage[-offset:] == 19).all()


def split_k_ranges(k, splits):
    blocks, remainder = divmod(k // 64, splits)
    return [((i * blocks + min(i, remainder)) * 64,
             (i * blocks + min(i, remainder) + blocks + (i < remainder)) * 64)
            for i in range(splits)]


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('m', (0, 1, 127, 128, 129))
@pytest.mark.parametrize('n', (8, 24, 128))
def test_sm120_hc_prenorm_contract(m, n):
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for k in (64, 192, 1024):
            av = ((torch.arange(m * k).reshape(m, k) % 15 - 7).float() / 8).to(torch.bfloat16)
            bv = (torch.arange(n * k).reshape(n, k) % 13 - 6).float() / 16
            for splits in (None, 1, 3, 16, 19):
                count = splits or 1
                a, a_storage = native_matrix(av, True, 1)
                b, b_storage = native_matrix(bv, True, 1)
                d_shape = (m, n) if splits is None else (count, m, n)
                s_shape = (m,) if splits is None else (count, m)
                d, ds, do = guarded_compact(torch.zeros(d_shape))
                s, ss, so = guarded_compact(torch.zeros(s_shape))
                d.fill_(float('nan'))
                s.fill_(float('nan'))
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=splits)
                actual_d, actual_s = d.cpu().reshape(count, m, n), s.cpu().reshape(count, m)
                for i, (start, end) in enumerate(split_k_ranges(k, count)):
                    ref_d = av[:, start:end].float() @ bv[:, start:end].T
                    ref_s = av[:, start:end].float().square().sum(-1)
                    assert torch.isfinite(actual_d[i]).all() and torch.isfinite(actual_s[i]).all()
                    torch.testing.assert_close(actual_d[i], ref_d, rtol=0, atol=0)
                    torch.testing.assert_close(actual_s[i], ref_s, rtol=0, atol=0)
                if m:
                    assert calc_diff(actual_d.sum(0), av.float() @ bv.T) < 1e-8
                    assert calc_diff(actual_s.sum(0), av.float().square().sum(-1)) < 1e-8
                check_compact_guards(ds, do)
                check_compact_guards(ss, so)
                torch.testing.assert_close(a.cpu(), av, rtol=0, atol=0)
                torch.testing.assert_close(b.cpu(), bv, rtol=0, atol=0)
                for operand, storage in ((a, a_storage), (b, b_storage)):
                    valid = torch.zeros_like(storage, dtype=torch.bool)
                    valid.as_strided(operand.shape, operand.stride(), operand.storage_offset()).fill_(True)
                    assert (storage[~valid] == 19).all()
                print(f' > SM120 HC fixture: {m=}, {n=}, {k=}, {splits=}')
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('offset', (0, 1))
def test_sm120_hc_strided_output(offset):
    m, n, k = 129, 24, 192
    av = torch.ones((m, k), dtype=torch.bfloat16)
    bv = torch.ones((n, k), dtype=torch.float32) / 4
    a = native_matrix(av, True, offset, tma=False)[0]
    b = native_matrix(bv, True, offset, tma=False)[0]
    d, storage = native_matrix(torch.zeros((m, n)), True, offset, extra=5, tma=False)
    sq, ss, so = guarded_compact(torch.zeros(m))
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sq)
    torch.testing.assert_close(d.cpu(), torch.full((m, n), 48.0), rtol=0, atol=0)
    torch.testing.assert_close(sq.cpu(), torch.full((m,), 192.0), rtol=0, atol=0)
    mask = torch.zeros_like(storage, dtype=torch.bool)
    mask.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    assert (storage[~mask] == 19).all()
    check_compact_guards(ss, so)
    torch.testing.assert_close(a.cpu(), av, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), bv, rtol=0, atol=0)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('splits', (None, 3, 16))
@pytest.mark.parametrize('pdl', (False, True))
def test_sm120_hc_prenorm_graph(splits, pdl):
    m, n, k = 129, 24, 192
    av = ((torch.arange(m * k).reshape(m, k) % 15 - 7).float() / 8).to(torch.bfloat16)
    bv = (torch.arange(n * k).reshape(n, k) % 13 - 6).float() / 16
    a, b = av.cuda(), bv.cuda()
    count = splits or 1
    d, ds, do = guarded_compact(torch.zeros((m, n) if splits is None else (count, m, n)))
    s, ss, so = guarded_compact(torch.zeros((m,) if splits is None else (count, m)))
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_pdl(pdl)
        def run():
            deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=splits)
        for _ in range(3):
            run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for scale in (1.0, -0.5, 0.0):
            a.copy_(av * scale)
            d.fill_(float('nan'))
            s.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            dd, ss_actual = d.cpu().reshape(count, m, n), s.cpu().reshape(count, m)
            for i, (start, end) in enumerate(split_k_ranges(k, count)):
                v = av[:, start:end].float() * scale
                torch.testing.assert_close(dd[i], v @ bv[:, start:end].T, rtol=0, atol=0)
                torch.testing.assert_close(ss_actual[i], v.square().sum(-1), rtol=0, atol=0)
            check_compact_guards(ds, do)
            check_compact_guards(ss, so)
        torch.testing.assert_close(b.cpu(), bv, rtol=0, atol=0)
    finally:
        deep_gemm.set_pdl(old_pdl)


def fp8_operand(groups, rows, k, gran_mn, gran_k, phase):
    values = ((torch.arange(groups * rows * k).reshape(groups, rows, k) * 3 + phase) % 15 - 7).float() / 4
    raw = values.to(torch.float8_e4m3fn)
    sf = torch.pow(2.0, ((torch.arange(groups * math.ceil(rows / gran_mn) * math.ceil(k / gran_k))
                         .reshape(groups, math.ceil(rows / gran_mn), math.ceil(k / gran_k)) + phase) % 3 - 4).float())
    decoded = raw.float() * sf[:, torch.arange(rows) // gran_mn][:, :, torch.arange(k) // gran_k]
    return raw, sf, decoded


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('expr', ('bhr,hdr->bhd', 'bhd,hdr->bhr', 'bhd,bhr->hdr'))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_fp8_einsum_contract(expr, dtype):
    old_pdl = deep_gemm.get_pdl()
    try:
        for m, n, k in ((16, 64, 256), (32, 64, 256), (33, 64, 256), (1024, 512, 128)):
            for recipe in ((1, 128, 128), (128, 1, 32)):
                h = 2
                aa, sa, ar = fp8_operand(h, m, k, recipe[0], recipe[2], 1)
                bb, sb, br = fp8_operand(h, n, k, recipe[1], recipe[2], 2)
                if expr == 'bhr,hdr->bhd':
                    aa, sa = aa.permute(1, 0, 2).contiguous(), sa.permute(1, 0, 2).contiguous()
                elif expr == 'bhd,hdr->bhr':
                    aa, sa = aa.permute(1, 0, 2).contiguous(), sa.permute(1, 0, 2).contiguous()
                    bb, sb = bb.permute(0, 2, 1).contiguous(), sb.permute(0, 2, 1).contiguous()
                else:
                    aa, sa = aa.permute(2, 0, 1).contiguous(), sa.permute(2, 0, 1).contiguous()
                    bb, sb = bb.permute(2, 0, 1).contiguous(), sb.permute(2, 0, 1).contiguous()
                product = ar @ br.transpose(1, 2)
                if expr != 'bhd,bhr->hdr':
                    product = product.permute(1, 0, 2).contiguous()
                a, b = (aa.cuda(), sa.cuda()), (bb.cuda(), sb.cuda())
                cv = ((torch.arange(product.numel()).reshape(product.shape) % 7 - 3).float() / 16).to(dtype)
                for c_mode in ('none', 'same', 'different'):
                    if m == 33:
                        flat, storage = native_matrix(torch.zeros((cv.shape[0] * cv.shape[1], cv.shape[2]), dtype=dtype), True, 1)
                        d = flat.view(cv.shape)
                    else:
                        d, storage, offset = guarded_compact(torch.zeros_like(cv))
                    storage_valid = torch.zeros_like(storage, dtype=torch.bool)
                    storage_valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
                    c = d if c_mode == 'same' else (guarded_compact(cv)[0] if c_mode == 'different' else None)
                    def run():
                        deep_gemm.fp8_einsum(expr, a, b, d, c=c, recipe=recipe)
                    def check(scale=1.0):
                        expected = product * scale + (cv.float() if c_mode != 'none' else 0)
                        actual = d.cpu().float()
                        assert torch.isfinite(actual).all()
                        torch.testing.assert_close(actual, expected, rtol=0.008 if dtype == torch.bfloat16 else 2e-4,
                                                   atol=0.02 if dtype == torch.bfloat16 else 1e-4)
                        assert calc_diff(actual, expected) < (1e-5 if dtype == torch.bfloat16 else 1e-8)
                        assert (storage[~storage_valid] == 19).all()
                        if c_mode == 'different':
                            torch.testing.assert_close(c.cpu(), cv, rtol=0, atol=0)
                    deep_gemm.set_pdl(m == 33)
                    for _ in range(3):
                        d.copy_(cv) if c_mode == 'same' else d.fill_(float('nan'))
                        run()
                        check()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        run()
                    for phase in (0, 1, 2):
                        scale = 2.0 ** phase
                        a[1].copy_(sa * scale)
                        d.copy_(cv) if c_mode == 'same' else d.fill_(float('nan'))
                        graph.replay()
                        torch.cuda.synchronize()
                        check(scale)
                    a[1].copy_(sa)
                    for actual, expected in ((a[0], aa), (a[1], sa), (b[0], bb), (b[1], sb)):
                        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    print(f' > SM120 FP8 einsum fixture: {expr=}, {dtype=}, {m=}, {recipe=}, {c_mode=}')
    finally:
        deep_gemm.set_pdl(old_pdl)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_bf16_batch_reduction(dtype):
    for groups, m, n, k in ((1, 64, 64, 64), (3, 128, 64, 128), (5, 64, 128, 192)):
        av = ((torch.arange(groups * m * k).reshape(groups, m, k) % 15 - 7).float() / 8).to(torch.bfloat16)
        bv = ((torch.arange(groups * n * k).reshape(groups, n, k) % 13 - 6).float() / 8).to(torch.bfloat16)
        a, b = av.cuda(), bv.cuda()
        initial = torch.full((m, n), 0.25, dtype=dtype)
        d, storage, offset = guarded_compact(initial)
        c = d if dtype == torch.float32 else None
        for _ in range(3):
            d.copy_(initial) if c is not None else d.fill_(float('nan'))
            deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)
            expected = (av.float() @ bv.float().transpose(1, 2)).sum(0) + (initial.float() if c is not None else 0)
            actual = d.cpu().float()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected.to(dtype).float(), rtol=0, atol=0)
            assert calc_diff(actual, expected) < 1e-5
            check_compact_guards(storage, offset)
        torch.testing.assert_close(a.cpu(), av, rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), bv, rtol=0, atol=0)
        print(f' > SM120 BF16 batch reduction fixture: {groups=}, {m=}, {n=}, {k=}, {dtype=}')


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_bf16_batch_reduction_empty_graph(dtype):
    for groups, m, n, k in ((0, 64, 64, 64), (2, 0, 64, 64), (2, 64, 0, 64), (2, 64, 64, 0), (3, 64, 64, 128)):
        a = torch.ones((groups, m, k), dtype=torch.bfloat16, device='cuda')
        b = torch.ones((groups, n, k), dtype=torch.bfloat16, device='cuda')
        d, storage, offset = guarded_compact(torch.full((m, n), 0.25, dtype=dtype))
        c = d if dtype == torch.float32 else None
        def run():
            deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)
        for _ in range(3):
            run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for scale in (1.0, 0.5, 0.0):
            a.fill_(scale)
            d.fill_(0.25 if c is not None else float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full((m, n), groups * k * scale + (0.25 if c is not None else 0), dtype=dtype)
            torch.testing.assert_close(d.cpu(), expected, rtol=0, atol=0)
            check_compact_guards(storage, offset)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_fp8_fp4_formats_alpha():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for li, layout in enumerate(('nt', 'nn', 'tn', 'tt')):
            for di, dtype in enumerate((torch.bfloat16, torch.float32)):
                for ai, alpha in enumerate((None, 0.0, 0.5, -1.0, 2.0)):
                    index = fi + li + di + ai
                    grans = ((32, 32), (32, 128), (128, 32), (128, 128))[index % 4]
                    exercise_sm120_dense_fp8_fp4(fmt, layout, (32, 64, 256), dtype, alpha,
                                                ('none', 'same', 'different')[index % 3],
                                                ('float', 'packed')[index % 2], grans,
                                                common_recipe=grans[0] == grans[1])


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_fp8_fp4_swap_boundary():
    for fmt in ((False, False), (False, True), (True, False), (True, True)):
        for m in (15, 16, 17):
            for c_mode in ('none', 'same'):
                exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, 33, 256), torch.bfloat16, 0.5,
                                            c_mode, 'packed', (32, 128), padded=True)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_fp8_fp4_tails_large():
    for fmt in ((False, False), (False, True), (True, False), (True, True)):
        for dtype in (torch.bfloat16, torch.float32):
            k = 130 if fmt == (True, True) else (131 if fmt == (False, False) else 256)
            exercise_sm120_dense_fp8_fp4(fmt, 'nt', (65, 67, k), dtype, -1.0,
                                        'different', 'float', (128, 32), padded=True)
            exercise_sm120_dense_fp8_fp4(fmt, 'nt', (2049, 256, 256), dtype, 2.0,
                                        'none', 'packed', (32, 128))


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_fp8_fp4_graph_pdl():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for li, layout in enumerate(('nt', 'nn', 'tn', 'tt')):
            for pdl in (False, True):
                exercise_sm120_dense_fp8_fp4(fmt, layout, (32, 64, 256),
                                            torch.bfloat16 if pdl else torch.float32,
                                            -1.0 if pdl else 0.5, ('none', 'same', 'different')[(fi + li) % 3],
                                            'packed' if pdl else 'float', (128, 32), padded=True, graph=True, pdl=pdl)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_fp8_fp4_explicit_rejections():
    assert get_arch_major() == 12
    raw, sf, _ = sm120_dense_quantized(32, 128, False, 32, 0)
    a = raw.cuda(), sf.cuda()
    d = torch.empty((32, 32), device='cuda', dtype=torch.bfloat16)
    cases = [
        (a, a, dict(recipe_a=(1, 32))),
        (a, a, dict(recipe=(1, 1, 32), recipe_a=(1, 32), recipe_b=(1, 32))),
        (a, a, dict(recipe=(0, 1, 32))),
        (a, a, dict(recipe=(1, 1, 64))),
        (a, a, dict(recipe=(1, 1, 32), disable_ue8m0_cast=True)),
    ]
    short8, short_sf, _ = sm120_dense_quantized(32, 130, False, 32, 0)
    short4, short_sf4, _ = sm120_dense_quantized(32, 130, True, 32, 0)
    cases.append(((short8.cuda(), short_sf.cuda()), (short4.cuda(), short_sf4.cuda()), dict(recipe=(1, 1, 32))))
    block_sf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf[:1].cuda())
    cases.append(((a[0], block_sf), a, dict(recipe_a=(128, 32), recipe_b=(1, 32))))
    for aa, bb, kwargs in cases:
        try:
            deep_gemm.fp8_fp4_gemm_nt(aa, bb, d, **kwargs)
        except RuntimeError as error:
            assert 'Assertion' in str(error) or 'assert' in str(error), str(error)
        else:
            raise AssertionError(f'Invalid dense input contract was not rejected: {kwargs}')


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_scaling_defaults():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for si, sf_kind in enumerate(('float', 'packed')):
            for i, (m, n) in enumerate(((15, 127), (16, 128), (17, 129), (129, 255), (257, 257))):
                exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, n, 256),
                                            torch.bfloat16 if i % 2 else torch.float32,
                                            (None, 0.0, 0.5, -1.0, 2.0)[i],
                                            ('none', 'same', 'different')[(fi + si + i) % 3], sf_kind,
                                            (128, 128), padded=True,
                                            mn_grans=(1, 128) if sf_kind == 'float' else (1, 1),
                                            compare_default=True, legacy_alias=fmt == (False, False))


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_scaling_blockwise_tails():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for gi, mn_grans in enumerate(((128, 1), (1, 128), (128, 128))):
            for i, (m, n) in enumerate(((127, 129), (128, 255), (129, 257), (255, 127), (257, 128))):
                grans = ((32, 128), (128, 32), (32, 32), (128, 128))[(fi + gi + i) % 4]
                exercise_sm120_dense_fp8_fp4(fmt, 'nt', (m, n, 256),
                                            torch.bfloat16 if (gi + i) % 2 else torch.float32,
                                            (None, 0.0, 0.5, -1.0, 2.0)[i],
                                            ('none', 'same', 'different')[(fi + i) % 3], 'float', grans,
                                            padded=True, mn_grans=mn_grans,
                                            common_recipe=grans[0] == grans[1])


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_dense_scaling_graph():
    for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
        for pdl in (False, True):
            exercise_sm120_dense_fp8_fp4(fmt, 'nt', (129, 257, 256),
                                        torch.bfloat16 if pdl else torch.float32, -1.0,
                                        ('none', 'same', 'different')[fi % 3], 'float', (128, 128),
                                        padded=True, graph=True, pdl=pdl, mn_grans=(1, 128), compare_default=True)
            exercise_sm120_dense_fp8_fp4(fmt, 'tt', (128, 256, 256),
                                        torch.bfloat16 if pdl else torch.float32, 0.5, 'same', 'float', (32, 128),
                                        padded=True, graph=True, pdl=pdl, mn_grans=(128, 128))


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('fp4_b', (False, True), ids=('fp8b', 'fp4b'))
@pytest.mark.parametrize('mode, alignment, n, zero_padding, graph, padded_a', (
    pytest.param('labels', 32, 1, False, False, False, id='labels-a32-n1'),
    pytest.param('labels', 64, 33, True, True, True, id='labels-a64-n33-graph-padded'),
    pytest.param('labels', 128, 65, False, False, False, id='labels-a128-n65'),
    pytest.param('labels', 32, 72, True, False, False, id='labels-a32-n72'),
    pytest.param('psum', 32, 33, True, True, False, id='psum-a32-n33-graph'),
    pytest.param('psum', 64, 1, False, False, True, id='psum-a64-n1-padded'),
    pytest.param('psum', 128, 65, True, False, False, id='psum-a128-n65'),
    pytest.param('psum', 64, 72, False, False, False, id='psum-a64-n72'),
    pytest.param('masked', 32, 8, False, False, False, id='masked-a32-n8'),
    pytest.param('masked', 64, 24, False, True, False, id='masked-a64-n24-graph'),
    pytest.param('masked', 128, 72, False, False, False, id='masked-a128-n72'),
))
def test_sm120_grouped_tma_layout_contract(mode, alignment, n, zero_padding, graph, padded_a, fp4_b):
    exercise_sm120_quant_grouped(mode, (False, fp4_b), alignment, 'packed' if fp4_b else 'float',
                                grans=(32, 128), zero_padding=zero_padding, graph=graph, pdl=fp4_b,
                                k=384, n=n, padded_a=padded_a)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_quant_grouped_formats():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for ai, alignment in enumerate((32, 64, 128)):
                for si, sf_kind in enumerate(('float', 'packed')):
                    index = mi + fi + ai + si
                    exercise_sm120_quant_grouped(mode, fmt, alignment, sf_kind,
                                                grans=((32, 32), (32, 128), (128, 32), (128, 128))[index % 4],
                                                nn=mode != 'masked' and index % 2 == 0,
                                                zero_padding=mode != 'masked' and (fi + si) % 2 == 0,
                                                k=128 if index % 2 else 256)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_quant_grouped_block_scales():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for gi, mn_grans in enumerate(((128, 1), (1, 128), (128, 128))):
                exercise_sm120_quant_grouped(mode, fmt, (32, 64, 128)[gi], 'float',
                                            grans=((32, 128), (128, 32), (32, 32))[(fi + gi) % 3],
                                            mn_grans=mn_grans, nn=mode != 'masked' and fi % 2 == 1,
                                            zero_padding=mode != 'masked' and (mi + fi + gi) % 2 == 0)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_quant_grouped_defaults():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for si, sf_kind in enumerate(('float', 'packed')):
                exercise_sm120_quant_grouped(mode, fmt, (32, 64, 128)[(mi + fi + si) % 3], sf_kind,
                                            grans=(128, 128), mn_grans=(1, 128) if sf_kind == 'float' else (1, 1),
                                            defaults=True, nn=mode != 'masked' and si == 1,
                                            zero_padding=mode != 'masked' and fi % 2 == 1)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_quant_grouped_graph_mutation():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            for ai, alignment in enumerate((32, 64, 128)):
                packed = (fi + ai) % 2 == 1
                exercise_sm120_quant_grouped(mode, fmt, alignment, 'packed' if packed else 'float',
                                            grans=(32, 128) if ai % 2 else (128, 32),
                                            mn_grans=(1, 1) if packed else ((128, 128) if ai == 2 else (1, 128)),
                                            graph=True, pdl=(mi + fi + ai) % 2 == 1,
                                            nn=mode != 'masked' and fi % 2 == 1,
                                            zero_padding=mode != 'masked' and (fi + ai) % 2 == 0)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_quant_grouped_all_empty():
    for mi, mode in enumerate(('labels', 'psum', 'masked')):
        for fi, fmt in enumerate(((False, False), (False, True), (True, False), (True, True))):
            exercise_sm120_quant_grouped(mode, fmt, (32, 64, 128)[(mi + fi) % 3],
                                        'packed' if fi % 2 else 'float', all_empty=True,
                                        nn=mode != 'masked' and fi % 2 == 1,
                                        zero_padding=mode != 'masked' and fi % 2 == 0)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_quant_grouped_zero_m():
    for fmt in ((False, False), (False, True), (True, False), (True, True)):
        a = torch.empty((0, 64 if fmt[0] else 128), dtype=torch.int8 if fmt[0] else torch.float8_e4m3fn, device='cuda')
        b = torch.empty((3, 65, 64 if fmt[1] else 128), dtype=torch.int8 if fmt[1] else torch.float8_e4m3fn, device='cuda')
        sa = torch.empty((0, 1), device='cuda')
        sb = torch.ones((3, 65, 1), device='cuda')
        d = torch.empty((0, 65), dtype=torch.bfloat16, device='cuda')
        for psum in (False, True):
            metadata = torch.zeros(3 if psum else 0, dtype=torch.int32, device='cuda')
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((a, sa), (b, sb), d, metadata,
                                                         recipe=(1, 1, 128), use_psum_layout=psum,
                                                         ensure_zero_padding=True)
            assert d.numel() == 0


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_k_grouped_fp8_contracts():
    from sm120_exercise import exercise_sm120_k_grouped
    for api in ('fp8_tn', 'fp8_nt'):
        for gi, groups in enumerate((1, 3, 8)):
            for di, dtype in enumerate((torch.bfloat16, torch.float32)):
                for si, (gran, packed) in enumerate(((128, False), (32, False), (32, True))):
                    psum = api == 'fp8_tn' and (gi + di + si) % 2 == 0
                    exercise_sm120_k_grouped(api, groups, 256 if psum else 128, psum, dtype,
                                            ('none', 'same', 'different')[(gi + di + si) % 3],
                                            shape=(16, 48) if gi % 2 else (48, 80), gran=gran, packed=packed,
                                            ks_mode=('list', 'none', 'empty')[si] if psum else 'list',
                                            default_recipe=gran == 128)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_k_grouped_fp8_graph():
    from sm120_exercise import exercise_sm120_k_grouped
    for api in ('fp8_tn', 'fp8_nt'):
        for di, dtype in enumerate((torch.bfloat16, torch.float32)):
            for ci, c_mode in enumerate(('none', 'same', 'different')):
                psum = api == 'fp8_tn'
                exercise_sm120_k_grouped(api, 8, 256 if psum and ci == 2 else 128, psum, dtype, c_mode,
                                        gran=128 if ci == 0 else 32, packed=ci == 2,
                                        graph=True, pdl=(di + ci) % 2 == 1, ks_mode='none' if psum else 'list',
                                        default_recipe=ci == 0)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_k_grouped_fp8_empty():
    from sm120_exercise import exercise_sm120_k_grouped
    for api in ('fp8_tn', 'fp8_nt'):
        for c_mode in ('none', 'same', 'different'):
            exercise_sm120_k_grouped(api, 3, 128, False, torch.float32, c_mode, empty=True)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_k_grouped_fp4_unsupported_rejection():
    assert get_arch_major() == 12
    a = torch.zeros((20, 128), dtype=torch.int8, device='cuda')
    b = torch.zeros((36, 128), dtype=torch.int8, device='cuda')
    sa, sb = torch.ones((8, 20), device='cuda'), torch.ones((8, 36), device='cuda')
    d = torch.full((1, 20, 36), 7, dtype=torch.bfloat16, device='cuda')
    metadata = torch.tensor([256], dtype=torch.int32, device='cuda')
    try:
        deep_gemm.k_grouped_fp4_gemm_nt_contiguous((a, sa), (b, sb), d, [256], metadata)
    except RuntimeError as error:
        assert 'SM120 K-grouped FP4 NT is not implemented' in str(error), str(error)
    else:
        raise AssertionError('Unsupported SM120 K-grouped FP4 unexpectedly accepted input')
    assert (d == 7).all()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    for m in (0, 1, 127, 128, 129):
        for n in (8, 24, 128):
            test_sm120_hc_prenorm_contract(m=m, n=n)
    for offset in (0, 1):
        test_sm120_hc_strided_output(offset=offset)
    for splits in (None, 3, 16):
        for pdl in (False, True):
            test_sm120_hc_prenorm_graph(splits=splits, pdl=pdl)
    for expr in ('bhr,hdr->bhd', 'bhd,hdr->bhr', 'bhd,bhr->hdr'):
        for dtype in (torch.bfloat16, torch.float32):
            test_sm120_fp8_einsum_contract(expr=expr, dtype=dtype)
    for dtype in (torch.bfloat16, torch.float32):
        test_sm120_bf16_batch_reduction(dtype=dtype)
    for dtype in (torch.bfloat16, torch.float32):
        test_sm120_bf16_batch_reduction_empty_graph(dtype=dtype)
    test_sm120_dense_fp8_fp4_formats_alpha()
    test_sm120_dense_fp8_fp4_swap_boundary()
    test_sm120_dense_fp8_fp4_tails_large()
    test_sm120_dense_fp8_fp4_graph_pdl()
    test_sm120_dense_fp8_fp4_explicit_rejections()
    test_sm120_dense_scaling_defaults()
    test_sm120_dense_scaling_blockwise_tails()
    test_sm120_dense_scaling_graph()
    for fp4_b in (False, True):
        for mode, alignment, n, zero_padding, graph, padded_a in (('labels', 32, 1, False, False, False), ('labels', 64, 33, True, True, True), ('labels', 128, 65, False, False, False), ('labels', 32, 72, True, False, False), ('psum', 32, 33, True, True, False), ('psum', 64, 1, False, False, True), ('psum', 128, 65, True, False, False), ('psum', 64, 72, False, False, False), ('masked', 32, 8, False, False, False), ('masked', 64, 24, False, True, False), ('masked', 128, 72, False, False, False)):
            test_sm120_grouped_tma_layout_contract(fp4_b=fp4_b, mode=mode, alignment=alignment, n=n, zero_padding=zero_padding, graph=graph, padded_a=padded_a)
    test_sm120_quant_grouped_formats()
    test_sm120_quant_grouped_block_scales()
    test_sm120_quant_grouped_defaults()
    test_sm120_quant_grouped_graph_mutation()
    test_sm120_quant_grouped_all_empty()
    test_sm120_quant_grouped_zero_m()
    test_sm120_k_grouped_fp8_contracts()
    test_sm120_k_grouped_fp8_graph()
    test_sm120_k_grouped_fp8_empty()
    test_sm120_k_grouped_fp4_unsupported_rejection()
