import math

import pytest
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from sm120_test_storage import native_matrix


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
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


@pytest.mark.parametrize('m', (0, 1, 127, 128, 129))
@pytest.mark.parametrize('n', (8, 24, 128))
def test_sm120_hc_prenorm_contract(m, n):
    assert get_arch_major() == 12
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


@pytest.mark.parametrize('expr', ('bhr,hdr->bhd', 'bhd,hdr->bhr', 'bhd,bhr->hdr'))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_fp8_einsum_contract(expr, dtype):
    assert get_arch_major() == 12
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


@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_bf16_batch_reduction(dtype):
    assert get_arch_major() == 12
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
