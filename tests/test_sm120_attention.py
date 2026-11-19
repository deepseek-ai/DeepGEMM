import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major, assert_bitwise_equal, test_filter
from deep_gemm.utils import ceil_div, per_token_cast_to_fp4, cast_back_from_fp4
from test_sm120_gemm import fp8_operand
from sm120_test_storage import native_matrix


def mqa_operand(shape, fp4, phase):
    values = ((torch.arange(torch.tensor(shape).prod().item()).reshape(shape) * 3 + phase) % 13 - 6).float() / 4
    if fp4:
        rows = values.numel() // shape[-1]
        exponents = (torch.arange(rows)[:, None] + torch.arange(shape[-1] // 32)[None, :] + phase) % 5 - 2
        amplitudes = torch.pow(2.0, exponents.float()).repeat_interleave(32, dim=-1).reshape(shape)
        values *= amplitudes
        packed, scales = per_token_cast_to_fp4(values.reshape(-1, shape[-1]).cuda(), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        decoded = cast_back_from_fp4(packed, scales, gran_k=32, use_packed_ue8m0=True).cpu().reshape(shape).float()
        return packed.reshape(*shape[:-1], shape[-1] // 2), scales.reshape(shape[:-1]), decoded
    raw = values.to(torch.float8_e4m3fn)
    return raw.cuda(), None, raw.float()


def mqa_reference(q, kv, weights):
    return (torch.einsum('mhd,nd->mhn', q.float(), kv.float()).relu() * weights.float()[..., None]).sum(1)


def graph_mqa_call(call, weights):
    original = weights.clone()
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_pdl(True)
        for _ in range(3):
            eager = call()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = call()
        weights.copy_(original * 0.5)
        output.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        mutated = output.clone()
        weights.copy_(original)
        output.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        return eager, output, mutated
    finally:
        weights.copy_(original)
        deep_gemm.set_pdl(old_pdl)


def check_logits(actual, expected, dtype):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=0.008 if dtype == torch.bfloat16 else 2e-4,
                               atol=0.02 if dtype == torch.bfloat16 else 1e-4)
    if actual.numel():
        assert calc_diff(actual.float(), expected) < (3e-5 if dtype == torch.bfloat16 else 5e-6)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('fp4,dim', ((False, 32), (False, 64), (False, 128), (True, 128)))
@pytest.mark.parametrize('heads', (16, 32, 64))
def test_sm120_dense_mqa_contract(fp4, dim, heads):
    for dtype, compressed in ((torch.float32, False), (torch.bfloat16, False), (torch.float32, True)):
        seq, tokens = (3, 8192) if not fp4 and heads == 16 else (9, 257)
        q, qsf, qr = mqa_operand((seq, heads, dim), fp4, 1)
        kv, kvsf, kr = mqa_operand((tokens, dim), fp4, 2)
        if not fp4:
            kvsf = torch.pow(2.0, (torch.arange(tokens) % 3 - 4).float()).cuda()
            kr *= kvsf.cpu()[:, None]
        weights_cpu = ((torch.arange(seq * heads).reshape(seq, heads) % 7 - 3).float() / 8)
        weights = weights_cpu.cuda()
        starts = torch.tensor([0, 1, 127] * ((seq + 2) // 3), dtype=torch.int32)[:seq]
        ends = torch.tensor([tokens, 1, tokens - 1] * ((seq + 2) // 3), dtype=torch.int32)[:seq]
        ks, ke = starts.cuda(), ends.cuda()
        expected = mqa_reference(qr, kr, weights_cpu)
        width = int((ends - starts).max()) if compressed else 0
        call = lambda: deep_gemm.fp8_fp4_mqa_logits((q, qsf), (kv, kvsf), weights, ks, ke,
                                                   clean_logits=not compressed, max_seqlen_k=width, logits_dtype=dtype)
        if heads == 16:
            eager, result, mutated = graph_mqa_call(call, weights)
            for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                columns = slice(0, end - start) if compressed else slice(start, end)
                check_logits(mutated[row, columns].cpu(), expected[row, start:end] * 0.5, dtype)
                torch.testing.assert_close(eager[row, columns], result[row, columns], rtol=0, atol=0)
        else:
            result = call()
        actual = result.cpu()
        for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            values = actual[row, :end - start] if compressed else actual[row, start:end]
            check_logits(values, expected[row, start:end], dtype)
            if not compressed:
                assert torch.isneginf(actual[row, :start]).all() and torch.isneginf(actual[row, end:]).all()
        if not fp4 and dtype == torch.float32 and not compressed:
            legacy = deep_gemm.fp8_mqa_logits(q, (kv, kvsf), weights, ks, ke)
            torch.testing.assert_close(legacy, result, rtol=0, atol=0)
        print(f' > SM120 dense MQA: {fp4=}, {dim=}, {heads=}, {dtype=}, {compressed=}')


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('fp4,page', ((False, 32), (False, 64), (False, 128), (False, 256), (True, 32), (True, 64), (True, 128), (True, 256)))
@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16))
def test_sm120_paged_mqa_contract(fp4, page, dtype):
    dim, heads, pages, batch = 128, 16, 8, 3
    kv, sf, decoded = mqa_operand((pages * page, dim), fp4, 2)
    if not fp4:
        sf = torch.pow(2.0, (torch.arange(pages * page) % 3 - 4).float()).cuda()
        decoded *= sf.cpu()[:, None]
    data_bytes = dim // 2 if fp4 else dim
    page_bytes = page * (data_bytes + 4)
    stride = (page_bytes + 511) // 512 * 512
    backing = torch.full((pages * stride + 32,), 19, dtype=torch.uint8, device='cuda')
    cache = backing.as_strided((pages, page, 1, data_bytes + 4), (stride, data_bytes + 4, data_bytes + 4, 1))
    for p in range(pages):
        backing[p * stride:p * stride + page * data_bytes].copy_(kv[p * page:(p + 1) * page].contiguous().view(torch.uint8).reshape(-1))
        backing[p * stride + page * data_bytes:p * stride + page_bytes].copy_(sf[p * page:(p + 1) * page].contiguous().view(torch.uint8).reshape(-1))
    before = backing.clone()
    table_cpu = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    table = table_cpu.cuda()
    for next_n, varlen, empty in ((1, False, False), (2, False, False), (3, False, False),
                                 (4, False, False), (5, False, False), (1, True, False), (1, True, True)):
        table_cpu = torch.tensor([[0, 1], [0, 1], [4, 5]] if varlen else [[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
        table.copy_(table_cpu)
        indices = torch.tensor([0, 0, 1], dtype=torch.int32, device='cuda') if varlen else None
        q, qsf, qr = mqa_operand((batch, next_n, heads, dim), fp4, 1)
        weights_cpu = torch.full((batch * next_n, heads), 0.125)
        weights = weights_cpu.cuda()
        contexts_cpu = torch.stack([torch.linspace(0, 2 * page - i, next_n).int() if next_n > 1 else torch.tensor([2 * page - i], dtype=torch.int32) for i in range(batch)])
        if varlen:
            contexts_cpu = torch.tensor([[page - 1], [2 * page], [0]], dtype=torch.int32)
        if empty:
            contexts_cpu.zero_()
        contexts = contexts_cpu.cuda()
        meta = deep_gemm.get_paged_mqa_logits_metadata(contexts, page, deep_gemm.get_num_sms(), indices=indices)
        call = lambda: deep_gemm.fp8_fp4_paged_mqa_logits((q, qsf), cache, weights, contexts, table, meta, 2 * page, indices=indices, logits_dtype=dtype)
        mutated = None
        if next_n == 3 or varlen:
            eager, out, mutated = graph_mqa_call(call, weights)
        else:
            out = call()
        for row in range(batch):
            keys = torch.cat([decoded[p * page:(p + 1) * page] for p in table_cpu[row].tolist()])
            ref = mqa_reference(qr[row], keys, weights_cpu[row * next_n:(row + 1) * next_n])
            for token in range(next_n):
                end = contexts_cpu[row, token].item()
                check_logits(out[row * next_n + token, :end].cpu(), ref[token, :end], dtype)
                if mutated is not None:
                    check_logits(mutated[row * next_n + token, :end].cpu(), ref[token, :end] * 0.5, dtype)
                    torch.testing.assert_close(eager[row * next_n + token, :end], out[row * next_n + token, :end], rtol=0, atol=0)
        assert torch.equal(backing, before)
        if not fp4 and next_n == 1 and dtype == torch.float32:
            legacy = deep_gemm.fp8_paged_mqa_logits(q, cache, weights, contexts, table, meta, 2 * page, indices=indices)
            for row in range(batch):
                end = contexts_cpu[row, 0].item()
                torch.testing.assert_close(legacy[row, :end], out[row, :end], rtol=0, atol=0)
        with pytest.raises(RuntimeError, match='clean_logits'):
            deep_gemm.fp8_fp4_paged_mqa_logits((q, qsf), cache, weights, contexts, table, meta, 2 * page, clean_logits=True)
        print(f' > SM120 paged MQA: {fp4=}, {page=}, {next_n=}')


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('varlen', (False, True))
def test_sm120_paged_mqa_page32_smem_edge(varlen):
    # page32 + 64 heads + paired atoms exceeds the 99 KiB SMEM budget with three
    # KV stages (by 4 bytes); the launcher must fall back to two stages.
    page, dim, heads, pages, batch = 32, 128, 64, 16, 3
    next_n = 1 if varlen else 2
    kv, sf, decoded = mqa_operand((pages * page, dim), False, 2)
    sf = torch.pow(2.0, (torch.arange(pages * page) % 3 - 4).float()).cuda()
    decoded *= sf.cpu()[:, None]
    page_bytes = page * (dim + 4)
    stride = (page_bytes + 511) // 512 * 512
    backing = torch.full((pages * stride + 32,), 19, dtype=torch.uint8, device='cuda')
    cache = backing.as_strided((pages, page, 1, dim + 4), (stride, dim + 4, dim + 4, 1))
    for p in range(pages):
        backing[p * stride:p * stride + page * dim].copy_(kv[p * page:(p + 1) * page].contiguous().view(torch.uint8).reshape(-1))
        backing[p * stride + page * dim:p * stride + page_bytes].copy_(sf[p * page:(p + 1) * page].contiguous().view(torch.uint8).reshape(-1))
    pages_per_req = 4
    table_cpu = torch.arange(batch * pages_per_req, dtype=torch.int32).reshape(batch, pages_per_req)
    if varlen:
        # Atoms group adjacent tokens with equal indices: atom a reads block-table
        # row a, and a paired atom's context length comes from its second token.
        table_cpu[1] = table_cpu[0]
    table = table_cpu.cuda()
    indices = torch.tensor([0, 0, 1], dtype=torch.int32, device='cuda') if varlen else None
    q, qsf, qr = mqa_operand((batch, next_n, heads, dim), False, 1)
    weights_cpu = torch.full((batch * next_n, heads), 0.125)
    weights = weights_cpu.cuda()
    if varlen:
        contexts_cpu = torch.tensor([[page - 1], [2 * page], [0]], dtype=torch.int32)
    else:
        contexts_cpu = torch.stack([torch.linspace(0, pages_per_req * page - i, next_n).int() for i in range(batch)])
    contexts = contexts_cpu.cuda()
    meta = deep_gemm.get_paged_mqa_logits_metadata(contexts, page, deep_gemm.get_num_sms(), indices=indices)
    out = deep_gemm.fp8_fp4_paged_mqa_logits((q, qsf), cache, weights, contexts, table, meta, pages_per_req * page,
                                             indices=indices, logits_dtype=torch.bfloat16)
    for row in range(batch):
        keys = torch.cat([decoded[p * page:(p + 1) * page] for p in table_cpu[row].tolist()])
        ref = mqa_reference(qr[row], keys, weights_cpu[row * next_n:(row + 1) * next_n])
        for token in range(next_n):
            end = contexts_cpu[row, token].item()
            check_logits(out[row * next_n + token, :end].cpu(), ref[token, :end], torch.bfloat16)
    print(f' > SM120 paged MQA page32 smem edge: {varlen=}')


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_sm120_skip_head_mid(dtype):
    m, n, k = 1024, 512, 128
    aa, sa, ar = fp8_operand(1, m, k, 1, 128, 1)
    bb, sb, br = fp8_operand(1, n, k, 1, 128, 2)
    a, b = (aa[0].cuda(), sa[0].cuda()), (bb[0].cuda(), sb[0].cuda())
    d, storage = native_matrix(torch.full((m, 640), 19, dtype=dtype), True, 1)
    deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, (128, 64, 128), recipe=(1, 1, 128))
    actual = d.cpu().float().reshape(m, 2, 320)
    expected = (ar[0] @ br[0].T).reshape(m, 2, 256)
    check_logits(actual[..., :128], expected[..., :128], dtype)
    check_logits(actual[..., 192:], expected[..., 128:], dtype)
    assert (actual[..., 128:192] == 19).all()
    valid = torch.zeros_like(storage, dtype=torch.bool)
    valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    assert (storage[~valid] == 19).all()


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
@pytest.mark.parametrize('packed_a,packed_b', ((False, False), (False, True), (True, False)))
def test_sm120_skip_head_mid_reject_fp32_scale_without_cast(dtype, packed_a, packed_b):
    a = torch.ones((1, 512), dtype=torch.float8_e4m3fn, device='cuda')
    b = torch.ones((256, 512), dtype=torch.float8_e4m3fn, device='cuda')
    sa, sb = torch.ones((1, 4), device='cuda'), torch.ones((256, 4), device='cuda')
    packed_sa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sa)
    packed_sb = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sb)
    sfa, sfb = packed_sa if packed_a else sa, packed_sb if packed_b else sb
    d, storage = native_matrix(torch.full((1, 320), 19, dtype=dtype), True, 1)
    before = storage.clone()
    match = 'sfb.scalar_type' if packed_a else 'Unsupported architecture or scaling factor types'
    with pytest.raises(RuntimeError, match=match):
        deep_gemm.fp8_gemm_nt_skip_head_mid((a, sfa), (b, sfb), d, (128, 64, 128),
                                          recipe=(1, 1, 128), disable_ue8m0_cast=True)
    assert torch.equal(storage, before)
    for scale_a, scale_b, disable_cast in ((sfa, sfb, False), (packed_sa, packed_sb, True)):
        d.fill_(19)
        deep_gemm.fp8_gemm_nt_skip_head_mid((a, scale_a), (b, scale_b), d, (128, 64, 128),
                                          recipe=(1, 1, 128), disable_ue8m0_cast=disable_cast)
        actual = d.cpu()
        torch.testing.assert_close(actual[:, :128], torch.full((1, 128), 512, dtype=dtype), rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 192:], torch.full((1, 128), 512, dtype=dtype), rtol=0, atol=0)
        assert (actual[:, 128:192] == 19).all()
        valid = torch.zeros_like(storage, dtype=torch.bool)
        valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
        assert (storage[~valid] == 19).all()


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dim,heads,page', ((32, 32, 64), (64, 64, 128)))
def test_sm120_paged_mqa_metadata_graph(dim, heads, page):
    batch, pages = 7, 6
    q, _, qr = mqa_operand((batch, 1, heads, dim), False, 1)
    kv, _, kr = mqa_operand((pages * page, dim), False, 2)
    data_bytes = page * dim
    stride = (page * (dim + 4) + 511) // 512 * 512
    backing = torch.full((pages * stride + 32,), 19, dtype=torch.uint8, device='cuda')
    cache = backing.as_strided((pages, page, 1, dim + 4), (stride, dim + 4, dim + 4, 1))
    scales = torch.ones((pages, page), dtype=torch.float32, device='cuda')
    for p in range(pages):
        backing[p * stride:p * stride + data_bytes].copy_(kv[p * page:(p + 1) * page].view(torch.uint8).reshape(-1))
        backing[p * stride + data_bytes:p * stride + data_bytes + page * 4].copy_(scales[p].view(torch.uint8))
    original_cache = backing.clone()
    ids_cpu = torch.tensor([0, 0, 0, 0, 0, 1, 2], dtype=torch.int32)
    ids = ids_cpu.cuda()
    table_cpu = torch.tensor([[0, 1]] * 5 + [[2, 3], [4, 5]], dtype=torch.int32)
    table = table_cpu.cuda()
    ctx_cpu = torch.tensor([[1], [page - 1], [page], [page + 1], [2 * page], [page // 2], [0]], dtype=torch.int32)
    ctx = ctx_cpu.cuda()
    weights = torch.full((batch, heads), 0.125, device='cuda')
    def run():
        schedule = deep_gemm.get_paged_mqa_logits_metadata(ctx, page, deep_gemm.get_num_sms(), indices=ids)
        return deep_gemm.fp8_fp4_paged_mqa_logits((q, None), cache, weights, ctx, table, schedule, 2 * page, indices=ids)
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_pdl(True)
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
            output = run()
        for phase in (0, 1, 2):
            contexts = ctx_cpu if phase == 0 else (torch.zeros_like(ctx_cpu) if phase == 1 else ctx_cpu // 2)
            ctx.copy_(contexts)
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            for row in range(batch):
                keys = torch.cat([kr[p * page:(p + 1) * page] for p in table_cpu[row].tolist()])
                expected = mqa_reference(qr[row], keys, weights[row:row + 1].cpu())[0]
                end = contexts[row, 0].item()
                check_logits(output[row, :end].cpu(), expected[:end], torch.float32)
            assert torch.equal(backing, original_cache)
    finally:
        deep_gemm.set_pdl(old_pdl)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_mqa_host_rejections():
    q, _, _ = mqa_operand((3, 16, 128), False, 1)
    kv, _, _ = mqa_operand((128, 128), False, 2)
    sf = torch.ones(128, device='cuda')
    weights = torch.ones((3, 16), device='cuda')
    starts = torch.zeros(3, dtype=torch.int32, device='cuda')
    ends = torch.full((3,), 128, dtype=torch.int32, device='cuda')
    def call(qvalue=(None, None), kvvalue=None, w=None, **kwargs):
        return deep_gemm.fp8_fp4_mqa_logits((q, None) if qvalue[0] is None else qvalue,
                                           (kv, sf) if kvvalue is None else kvvalue,
                                           weights if w is None else w, starts, ends, **kwargs)
    for dtype in (torch.float16, torch.bfloat16):
        with pytest.raises(RuntimeError, match='weights.scalar_type'):
            call(w=weights.to(dtype))
    with pytest.raises(RuntimeError, match='arch_major'):
        call(qvalue=(q, torch.zeros((3, 16), dtype=torch.int32, device='cuda')))
    with pytest.raises(RuntimeError, match='kv_sf.scalar_type'):
        call(kvvalue=(kv, sf.to(torch.int32)))
    with pytest.raises(RuntimeError, match='logits_dtype'):
        call(logits_dtype=torch.float16)
    with pytest.raises(RuntimeError, match='schedule_meta'):
        call(schedule_meta=torch.zeros(1, dtype=torch.int32, device='cuda'))
    with pytest.raises(RuntimeError, match='max_seqlen_k'):
        call(max_seqlen_k=-1)
    with pytest.raises(RuntimeError, match='clean_logits'):
        call(max_seqlen_k=128, clean_logits=True)
    with pytest.raises(RuntimeError, match='arch_major'):
        deep_gemm.get_mqa_logits_metadata(starts, ends, 128, 16)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
@pytest.mark.parametrize('padded', (False, True))
@pytest.mark.parametrize('pdl', (False, True))
@pytest.mark.parametrize('shape,splits,sms,branch', (
    ((128, 16, 128), (8, 8, 8), 2, 'scalar'),
    ((65, 16, 128), (8, 8, 8), 2, 'scalar'),
    ((129, 16, 128), (0, 8, 8), 2, 'scalar'),
    ((1, 16, 128), (8, 8, 0), 2, 'scalar'),
    ((65, 16, 128), (8, 0, 8), 2, 'scalar'),
    ((32, 256, 16384), (128, 64, 128), 32, 'split'),
    ((32, 512, 16384), (128, 64, 128), 32, 'split'),
    ((32, 128, 16384), (0, 64, 128), 32, 'split'),
    ((32, 128, 16384), (128, 64, 0), 32, 'split'),
    ((32, 256, 16384), (128, 0, 128), 32, 'split'),
    ((128, 256, 2048), (128, 64, 128), 2, 'tma'),
    ((1024, 512, 128), (128, 64, 128), 32, 'tma'),
))
def test_sm120_skip_head_stores(dtype, padded, pdl, shape, splits, sms, branch):
    from sm120_reference_heuristic import predict_dense_fp8

    m, n, k = shape
    left, mid, right = splits
    width = n + n // (left + right) * mid
    aa, sa, ar = fp8_operand(1, m, k, 1, 128, 1)
    bb, sb, br = fp8_operand(1, n, k, 1, 128, 2)
    a, b = (aa[0].cuda(), sa[0].cuda()), (bb[0].cuda(), sb[0].cuda())
    d, storage = native_matrix(torch.full((m, width), 19, dtype=dtype), padded, int(padded))
    logical = torch.arange(n)
    physical = logical + (logical + right) // (left + right) * mid
    expected = ar[0] @ br[0].T
    untouched = torch.ones(width, dtype=torch.bool)
    untouched[physical] = False
    valid = torch.zeros_like(storage, dtype=torch.bool)
    valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    old_sms = deep_gemm.get_num_sms()
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_pdl(pdl)
        deep_gemm.set_num_sms(sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        prediction = predict_dense_fp8(m, n, k, sms, output_bytes=d.element_size())
        assert (prediction['split_k'] > 1) == (branch == 'split'), prediction
        if branch != 'split':
            assert (prediction['swizzle_cd'] > 0) == (branch == 'tma'), prediction

        def call():
            deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, splits, recipe=(1, 1, 128))

        def check():
            actual = d.cpu()
            check_logits(actual[:, physical], expected, dtype)
            assert (actual[:, untouched] == 19).all()
            assert (storage[~valid] == 19).all()

        call()
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        for _ in range(2):
            storage.fill_(19)
            graph.replay()
            torch.cuda.synchronize()
            check()
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)


def check_mqa_logits_chunked(actual, q, kv, weights, ks, ke, compressed):
    stats = torch.zeros(2, device=actual.device, dtype=torch.float64)
    q_chunk = 256
    kv_chunk = max(1, (128 * 1024 * 1024) // (q_chunk * q.size(1) * 4))
    for m0 in range(0, q.size(0), q_chunk):
        m1 = min(m0 + q_chunk, q.size(0))
        q_slice = q[m0:m1].float()
        w = weights[m0:m1].float().T.contiguous()
        for n0 in range(0, kv.size(0), kv_chunk):
            n1 = min(n0 + kv_chunk, kv.size(0))
            columns = torch.arange(n0, n1, device=actual.device)[None, :]
            mask = (columns >= ks[m0:m1, None]) & (columns < ke[m0:m1, None])
            scores = torch.einsum('mhd,nd->hmn', q_slice, kv[n0:n1].float()).relu_()
            reference = torch.einsum('hmn,hm->mn', scores, w)
            del scores
            if compressed:
                offsets = (columns - ks[m0:m1, None]).clamp(0, actual.size(1) - 1).long()
                values = actual[m0:m1].gather(1, offsets)
            else:
                values = actual[m0:m1, n0:n1]
                assert torch.equal(values == float('-inf'), ~mask)
            x, y = values.double().masked_fill(~mask, 0), reference.double().masked_fill(~mask, 0)
            stats[0] += (x * x + y * y).sum()
            stats[1] += 2 * (x * y).sum()
    denominator, numerator = stats.tolist()
    return 1 - numerator / denominator if denominator else 0.0


def sm120_sparse_candidates(starts, ends, block, count):
    candidates, sizes = [], []
    for row, (start, end) in enumerate(zip(starts, ends)):
        available = ceil_div(end - start, block)
        size = min(count, available)
        local = sorted(random.Random(1701 + row).sample(range(available), size))
        prefix = [start // block + index for index in local]
        candidates.append(prefix + [prefix[-1] if prefix else 0] * (count - size))
        sizes.append(size)
    return torch.tensor(candidates, dtype=torch.int32), sizes


def check_sm120_sparse_metadata(metadata, candidates, sizes, starts, block, fmt,
                                 page=0, table=None, requests=None, unaligned=False):
    import struct

    assert metadata.dtype == torch.uint8 and metadata.ndim == 1 and metadata.is_contiguous()
    raw = bytes(metadata.cpu().tolist())
    splits, waves, flag = struct.unpack_from('<III', raw)
    assert flag == int(unaligned)
    capacity = (640 if fmt == 'mxfp4' else 512) // block
    split_bytes = 16 + capacity * 8
    num_sms = deep_gemm.get_num_sms()
    assert waves >= 1 and 16 + splits * split_bytes + waves * num_sms * 16 <= len(raw)
    expected = {}
    for row, size in enumerate(sizes):
        for slot in range(size):
            logical = candidates[row, slot].item()
            physical = (table[row, logical * block // page].item() * (page // block)
                        + logical % (page // block)) if page else logical * block + starts[row] % block
            expected[row, slot] = physical
    actual, split_queries = {}, []
    for split in range(splits):
        offset = 16 + split * split_bytes
        query, packed_count, base0, base1 = struct.unpack_from('<IIII', raw, offset)
        size = packed_count & 0x7fffffff
        assert 0 < size <= capacity and query < len(sizes)
        split_queries.append(query)
        physical_blocks = []
        for index in range(capacity):
            physical, slots = struct.unpack_from('<II', raw, offset + 16 + index * 8)
            if index >= size:
                assert slots == 0xffffffff
                continue
            physical_blocks.append(physical)
            assert slots != 0xffffffff
            for half, base in enumerate((base0, base1)):
                slot = (slots >> (16 * half)) & 0xffff
                if slot == 0xffff:
                    continue
                row = query + half
                assert row < len(sizes)
                if page and half:
                    assert requests[row] == requests[query]
                key = row, base + slot
                assert key not in actual, f'Duplicate metadata output slot: {key}'
                actual[key] = physical
        if packed_count & 0x80000000:
            assert not page and not unaligned and size == capacity
            assert physical_blocks == list(range(physical_blocks[0], physical_blocks[0] + size * block, block))
    assert actual == expected, 'Metadata must map every valid prefix slot exactly once'
    visited = []
    for index in range(waves * num_sms):
        begin, end, query, num_queries = struct.unpack_from('<IIII', raw, 16 + splits * split_bytes + index * 16)
        if begin == end:
            assert (begin, end, query, num_queries) == (0, 0, 0, 0)
            continue
        assert 0 <= begin < end <= splits and 1 <= num_queries <= 2
        assert query + num_queries <= len(sizes)
        assert all(split_queries[split] == query for split in range(begin, end))
        if page and num_queries == 2:
            assert requests[query] == requests[query + 1]
        visited.extend(range(begin, end))
    assert sorted(visited) == list(range(splits))


def sm120_sparse_quantized(rows, fmt, phase):
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    dim = torch.arange(128, dtype=torch.int64)[None, :]
    values = (row * 3 + dim * 7 + dim // 32 + phase).remainder(5) - 2
    exponents = (torch.arange(4)[None, :] + row + phase).remainder(4)
    decoded = values * (1 << exponents).repeat_interleave(32, dim=1)
    scales = (exponents + 127).to(torch.uint8).contiguous().view(torch.int32).flatten()
    if fmt == 'mxfp4':
        codes = torch.tensor([12, 10, 0, 2, 4], dtype=torch.uint8)[values + 2]
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    else:
        packed = values.to(torch.float8_e4m3fn)
    return (packed.cuda(), scales.cuda()), decoded


def sm120_sparse_fractional_quantized(rows, fmt, phase):
    generator = torch.Generator().manual_seed(8171 + phase)
    if fmt == 'mxfp4':
        value_units = torch.tensor([-24, -16, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 16, 24])
        codebook = torch.tensor([15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
    else:
        value_units = torch.tensor([-14, -10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10, 14])
    indices = torch.randint(len(value_units), (rows, 32), generator=generator).repeat(1, 4)
    values = value_units[indices]
    exponents = (torch.arange(4)[None, :] + torch.arange(rows)[:, None] + phase).remainder(4) - 2
    # Integer units are 1/16; equal raw groups have deliberately unequal scales.
    decoded_units = values * (1 << (exponents + 2)).repeat_interleave(32, dim=1)
    scales = (exponents + 127).to(torch.uint8).contiguous().view(torch.int32).flatten()
    if fmt == 'mxfp4':
        codes = codebook[indices]
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    else:
        packed = (values.float() / 4).to(torch.float8_e4m3fn)
    return (packed.cuda(), scales.cuda()), decoded_units


def sm120_sparse_round_bf16_integer(value):
    # Round once per BF16 FMA without an FP32 intermediate rounding.
    sign = -1 if value < 0 else 1
    value = abs(value)
    shift = max(0, value.bit_length() - 8)
    if shift:
        quotient, remainder = divmod(value, 1 << shift)
        half = 1 << (shift - 1)
        value = (quotient + (remainder > half or (remainder == half and quotient % 2))) << shift
    return sign * value


def sm120_sparse_reference(q, kv, weights, candidates, sizes, starts, ends, block,
                            page=0, table=None, output_unit_exponent=0):
    rows, count = candidates.shape
    expected = torch.zeros((rows, count * block), dtype=torch.bfloat16)
    math_expected = torch.zeros((rows, count * block), dtype=torch.float32)
    mask = torch.zeros((rows, count * block), dtype=torch.bool)
    round_bf16 = sm120_sparse_round_bf16_integer
    for row in range(rows):
        tokens = (candidates[row, :sizes[row]].long()[:, None] * block
                  + starts[row] % block + torch.arange(block)).flatten()
        valid = (tokens >= starts[row]) & (tokens < ends[row])
        columns = torch.arange(tokens.numel())[valid]
        tokens = tokens[valid]
        if page:
            tokens = table[row, tokens // page].long() * page + tokens % page
        assert (kv[tokens].abs() @ q[row].abs().T < 2 ** 24).all(), 'FP32 dot partials must be exact in fixture units'
        scores = (kv[tokens] @ q[row].T).clamp(min=0)
        math_expected[row, columns] = (scores * weights[row]).sum(-1).float()
        unique_scores, inverse = torch.unique(scores, dim=0, return_inverse=True)
        row_weights = weights[row].tolist()
        logits = []
        for score in unique_scores.tolist():
            sums = [0, 0, 0, 0]
            for head in range(32):
                lane = head % 4
                sums[lane] = round_bf16(round_bf16(score[head]) * int(row_weights[head]) + sums[lane])
            logits.append(round_bf16(round_bf16(sums[0] + sums[2]) + round_bf16(sums[1] + sums[3])))
        expected[row, columns] = torch.tensor(logits, dtype=torch.bfloat16)[inverse]
        mask[row, columns] = True
    expected *= 2.0 ** output_unit_exponent
    math_expected *= 2.0 ** output_unit_exponent
    return expected.cuda(), math_expected.cuda(), mask.cuda()


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_metadata():
    for fmt in ('mxfp4', 'mxfp8'):
        dtype = torch.int8 if fmt == 'mxfp4' else torch.float8_e4m3fn
        for block in (8, 16):
            for count in (4, 8, 68):
                for page, unaligned in ((0, False), (0, True), (block, False), (64, False), (128, False)):
                    starts = [0, 0, 0] if page else [2 * block + int(unaligned), 2 * block + int(unaligned), 7 * block + 3 * int(unaligned)]
                    lengths = [0, 1, 0] if count == 4 else [2 * count * block + 1, 2 * count * block + 3, block - 1]
                    ends = [start + length for start, length in zip(starts, lengths)]
                    requests = [5, 5, 9]
                    width = max(1, ceil_div(max(ends), page)) if page else 0
                    table = torch.arange(2 * width - 1, -1, -1, dtype=torch.int32).view(2, width)[[0, 0, 1]] if page else None
                    candidates, sizes = sm120_sparse_candidates(starts, ends, block, count)
                    gpu_candidates = candidates.cuda()
                    gpu_starts = torch.tensor(starts, dtype=torch.int32, device='cuda')
                    gpu_ends = torch.tensor(ends, dtype=torch.int32, device='cuda')
                    if page:
                        kwargs = dict(context_lens=gpu_ends, block_table=table.cuda(),
                                      indices=torch.tensor(requests, dtype=torch.int32, device='cuda'), page_kv=page,
                                      sparse_kv_block_indices=gpu_candidates, qk_dtype=dtype, sparse_block_kv=block)
                        build = lambda: deep_gemm.get_paged_sparse_mqa_logits_metadata(**kwargs)
                    else:
                        kwargs = dict(cu_seq_len_k_start=gpu_starts, cu_seq_len_k_end=gpu_ends,
                                      num_kv_tokens=max(ends), sparse_kv_block_indices=gpu_candidates,
                                      qk_dtype=dtype, sparse_block_kv=block, use_unaligned_ks=unaligned)
                        build = lambda: deep_gemm.get_sparse_mqa_logits_metadata(**kwargs)
                    def check(metadata, current_sizes):
                        check_sm120_sparse_metadata(metadata, candidates, current_sizes, starts, block, fmt,
                                                    page, table, requests, unaligned)
                    capture_stream = torch.cuda.Stream()
                    capture_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(capture_stream):
                        warm_metadata = build()
                    torch.cuda.current_stream().wait_stream(capture_stream)
                    check(warm_metadata, sizes)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        captured = build()
                    for empty in (False, True, False):
                        gpu_ends.copy_(gpu_starts if empty else torch.tensor(ends, dtype=torch.int32, device='cuda'))
                        graph.replay()
                        torch.cuda.synchronize()
                        check(captured, [0] * 3 if empty else sizes)
    for dtype in (torch.int8, torch.float8_e4m3fn):
        zeros = torch.zeros(3, dtype=torch.int32, device='cuda')
        candidates = torch.zeros((3, 4), dtype=torch.int32, device='cuda')
        metadata = deep_gemm.get_sparse_mqa_logits_metadata(zeros, zeros, 0, candidates, dtype, 8)
        check_sm120_sparse_metadata(metadata, candidates.cpu(), [0] * 3, [0] * 3, 8,
                                    'mxfp4' if dtype == torch.int8 else 'mxfp8')


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_metadata_wave_histogram_reuse():
    original_num_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(2)
        for fmt in ('mxfp4', 'mxfp8'):
            dtype = torch.int8 if fmt == 'mxfp4' else torch.float8_e4m3fn
            for block in (8, 16):
                for rows in (1, 16, 17):
                    count, page = 2048, 64
                    starts = [0] * rows
                    ends = [count * block - 1 if row < 3 else (count - 1) * block + 1 for row in range(rows)]
                    requests = [5] + [9] * (rows - 1)
                    width = ceil_div(max(ends), page)
                    table = torch.arange(2 * width - 1, -1, -1, dtype=torch.int32).view(2, width)[[0] + [1] * (rows - 1)]
                    candidates = torch.arange(count, dtype=torch.int32).repeat(rows, 1)
                    gpu_ends = torch.tensor(ends, dtype=torch.int32, device='cuda')
                    kwargs = dict(context_lens=gpu_ends, block_table=table.cuda(),
                                  indices=torch.tensor(requests, dtype=torch.int32, device='cuda'), page_kv=page,
                                  sparse_kv_block_indices=candidates.cuda(), qk_dtype=dtype, sparse_block_kv=block)
                    def build():
                        return deep_gemm.get_paged_sparse_mqa_logits_metadata(**kwargs)
                    def check(metadata, empty=False):
                        check_sm120_sparse_metadata(metadata, candidates, [0 if empty else count] * rows,
                                                    starts, block, fmt, page, table, requests)
                        header = metadata.cpu().view(torch.int32)[:3].tolist()
                        if not empty:
                            assert (header[1] > 8) == (rows > 1)
                            if (fmt, block, rows) == ('mxfp4', 8, 16):
                                assert header == [234, 18, 0]
                    check(build())
                    capture_stream = torch.cuda.Stream()
                    capture_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(capture_stream):
                        warm_metadata = build()
                    torch.cuda.current_stream().wait_stream(capture_stream)
                    check(warm_metadata)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        captured = build()
                    for empty in (False, True, False):
                        gpu_ends.copy_(torch.tensor(starts if empty else ends, dtype=torch.int32, device='cuda'))
                        graph.replay()
                        torch.cuda.synchronize()
                        check(captured, empty)
    finally:
        deep_gemm.set_num_sms(original_num_sms)


def check_sm120_sparse_entry_partition_mapping(split_headers, q_token_base):
    chunks = [(index, chunk) for index, (base, packed, *_) in enumerate(split_headers)
              if base == q_token_base and (packed & 0x7fffffff) <= 80
              for chunk in range(ceil_div(packed & 0x7fffffff, 8))]
    assignments = []
    for partition in range(4):
        begin, end = len(chunks) * partition // 4, len(chunks) * (partition + 1) // 4
        prefix, assigned = 0, []
        for index, (base, packed, *_) in enumerate(split_headers):
            blocks = packed & 0x7fffffff
            count = ceil_div(blocks, 8) if base == q_token_base and blocks <= 80 else 0
            local_begin = min(count, max(0, begin - prefix))
            local_end = min(count, max(0, end - prefix))
            assigned.extend((index, chunk) for chunk in range(local_begin, local_end))
            prefix += count
        assert assigned == chunks[begin:end]
        assignments.append(assigned)
    flattened = [chunk for assigned in assignments for chunk in assigned]
    assert flattened == chunks and len(set(flattened)) == len(flattened)
    counts = [len(assigned) for assigned in assignments]
    assert max(counts) - min(counts) <= 1
    return counts


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_entry_partition_mapping():
    for tail in range(81):
        for wave in range(17):
            base = wave * 2
            headers = [(base, count | (0x80000000 if index % 2 else 0), index * 80, index * 79)
                       for index, count in enumerate((0, 1, 7, 8, 9, 79, 80, tail))]
            headers[2:2] = [(base + 1, 80, 0, 0), (base, 81, 0, 0), (base, 0x7fffffff, 0, 0)]
            check_sm120_sparse_entry_partition_mapping(headers, base)
    for counts in ((), (0,), (1,), (1, 1), (1, 1, 1), (80, 1), (80, 80, 1)):
        check_sm120_sparse_entry_partition_mapping([(0, count, 0, 0) for count in counts], 0)


def exercise_sm120_sparse_contract(fmt, block, count, page=0, unaligned=False, broadcast_cache=False, fractional=False,
                                    paged_rows=None, physical_tail=None, contiguous_rows=None,
                                    min_waves=1, require_split_slot_progression=False, physical_start=5):
    rows = paged_rows if paged_rows is not None else (1 if count == 4096 else 3)
    if paged_rows is not None:
        assert page and not broadcast_cache
    if contiguous_rows is not None:
        assert not page and unaligned and paged_rows is None and physical_tail is None and not broadcast_cache
        rows = contiguous_rows
    starts = ([0] * rows if page else [2 * block + int(unaligned), 2 * block + int(unaligned),
                                     7 * block + 3 * int(unaligned)][:rows])
    lengths = [count * block] if count == 4096 else [2 * count * block + 3, (count - 1) * block - 1, 0]
    if broadcast_cache:
        assert page and rows == 3
        lengths = [block - 1, 2 * count * block + 3, (count - 1) * block - 1]
    if paged_rows is not None:
        lengths = [count * block - 1 if row < 3 else (count - 1) * block + 1 for row in range(rows)]
    if physical_tail is not None:
        assert not page and unaligned and rows == 3 and paged_rows is None
        assert 0 <= physical_start < physical_tail
        starts = [physical_start, physical_start,
                  physical_tail - min(physical_tail - physical_start, 2 * block + 3)]
        if count == 4:
            starts = [physical_tail - 2 * block - 3] * 2 + [physical_tail - block - 1]
        lengths = [physical_tail - starts[0], physical_tail - starts[1] - 1, physical_tail - starts[2]]
    if contiguous_rows is not None:
        starts = [2 * block * ((row // 2) % 3) + 1 + ((row // 2) % (block - 1)) for row in range(rows)]
        lengths = [min(641 - start, count * block - (1 if row % 2 else block + 1))
                   if row % 4 < 2 else 641 - start - row % 2 for row, start in enumerate(starts)]
    ends = [start + length for start, length in zip(starts, lengths)]
    candidates, sizes = sm120_sparse_candidates(starts, ends, block, count)
    if physical_tail is not None:
        for row, size in enumerate(sizes):
            assert size == ceil_div(ends[row] - starts[row], block)
            assert (candidates[row, size - 1].item() + 1) * block + starts[row] % block > physical_tail
    quantized = sm120_sparse_fractional_quantized if fractional else sm120_sparse_quantized
    q, q_ref = quantized(rows * 32, fmt, 0)
    q = q[0].view(rows, 32, -1), q[1].view(rows, 32)
    if fractional:
        generator = torch.Generator().manual_seed(9817)
        weights_ref = torch.randint(1, 16, (rows, 16), generator=generator)
        weights_ref = torch.stack((weights_ref, -weights_ref), dim=-1).flatten(1)
    else:
        weights_ref = (torch.arange(rows * 32).view(rows, 32) % 5 - 1)
    weights = torch.empty((rows, 40), device='cuda', dtype=torch.bfloat16)[:, :32]
    weights.copy_(weights_ref.float() / (16 if fractional else 1))
    request_rows = [0, 1, 1] if broadcast_cache else [0, 0, 1]
    table, requests = None, ([5, 9, 9] if broadcast_cache else [5, 5, 9])[:rows]
    if paged_rows is not None:
        request_rows = [0] + [1] * (rows - 1)
        requests = [5] + [9] * (rows - 1)
    if page:
        width = ceil_div(max(ends), page)
        num_pages = 2 * width
        table = torch.arange(num_pages - 1, -1, -1, dtype=torch.int32).view(2, width)[request_rows[:rows]]
        kv, kv_ref = quantized(num_pages * page, fmt, 2)
        elem_dim = kv[0].size(1)
        page_bytes = page * (elem_dim + 4)
        stride = ceil_div(page_bytes, 512) * 512 + 512
        storage = torch.full((num_pages, stride), 255, dtype=torch.uint8, device='cuda')
        storage[:, :page * elem_dim] = kv[0].view(num_pages, -1).view(torch.uint8)
        storage[:, page * elem_dim:page_bytes] = kv[1].view(num_pages, page).view(torch.uint8)
        cache = storage[:, :page_bytes].view(num_pages, page, 1, elem_dim + 4)
        if broadcast_cache:
            cache = cache[:1].expand(num_pages, -1, -1, -1)
            kv_ref = kv_ref[:page].repeat(num_pages, 1)
            assert cache.stride(0) == 0
        q = q[0][:, None], q[1][:, None]
        metadata_kwargs = dict(context_lens=torch.tensor(ends, device='cuda', dtype=torch.int32),
                               block_table=table.cuda(), indices=torch.tensor(requests, device='cuda', dtype=torch.int32),
                               page_kv=page, sparse_kv_block_indices=candidates.cuda(), qk_dtype=q[0].dtype,
                               sparse_block_kv=block)
        build = lambda: deep_gemm.get_paged_sparse_mqa_logits_metadata(**metadata_kwargs)
        compute = lambda metadata: deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
            q=q, kv_cache=cache, weights=weights, metadata=metadata,
            num_max_sparse_blocks=count, sparse_block_kv=block)
    else:
        num_kv = physical_tail if physical_tail is not None else ceil_div(max(ends), 128) * 128
        kv, kv_ref = quantized(num_kv, fmt, 2)
        if physical_tail is not None:
            assert kv[0].data_ptr() % 16 == 0 and kv[1].data_ptr() % 16 == 0
            assert kv[0].untyped_storage().nbytes() == num_kv * kv[0].size(1)
            assert kv[1].untyped_storage().nbytes() == num_kv * 4
        metadata_kwargs = dict(cu_seq_len_k_start=torch.tensor(starts, device='cuda', dtype=torch.int32),
                               cu_seq_len_k_end=torch.tensor(ends, device='cuda', dtype=torch.int32),
                               num_kv_tokens=kv[0].size(0), sparse_kv_block_indices=candidates.cuda(),
                               qk_dtype=q[0].dtype, sparse_block_kv=block)
        if unaligned:
            metadata_kwargs['use_unaligned_ks'] = True
        build = lambda: deep_gemm.get_sparse_mqa_logits_metadata(**metadata_kwargs)
        compute_kwargs = dict(q=q, kv=kv, weights=weights, num_max_sparse_blocks=count, sparse_block_kv=block)
        if unaligned:
            compute_kwargs['use_unaligned_ks'] = True
        compute = lambda metadata: deep_gemm.fp8_fp4_sparse_mqa_logits(metadata=metadata, **compute_kwargs)
    expected, math_expected, mask = sm120_sparse_reference(
        q_ref.view(rows, 32, 128), kv_ref, weights_ref, candidates, sizes, starts, ends, block, page, table,
        output_unit_exponent=-12 if fractional else 0)
    metadata = build()
    check_sm120_sparse_metadata(metadata, candidates, sizes, starts, block, fmt, page, table, requests, unaligned)
    assert metadata.cpu().view(torch.int32)[1].item() >= min_waves
    if not page and fmt == 'mxfp4' and block == 8 and count == 2048:
        import struct

        raw = bytes(metadata.cpu().tolist())
        num_splits, num_waves, _ = struct.unpack_from('<III', raw)
        split_bytes = 16 + 80 * 8
        headers = [struct.unpack_from('<IIII', raw, 16 + split * split_bytes) for split in range(num_splits)]
        for index in range(num_waves * deep_gemm.get_num_sms()):
            begin, end, base, num_q = struct.unpack_from('<IIII', raw, 16 + num_splits * split_bytes + index * 16)
            if begin <= end <= num_splits and 1 <= num_q <= 2 and base + num_q <= rows:
                check_sm120_sparse_entry_partition_mapping(headers[begin:end], base)
    if require_split_slot_progression:
        import struct

        raw = bytes(metadata.cpu().tolist())
        num_splits, num_waves, _ = struct.unpack_from('<III', raw)
        split_bytes = 16 + ((640 if fmt == 'mxfp4' else 512) // block) * 8
        headers = [struct.unpack_from('<IIII', raw, 16 + split * split_bytes) for split in range(num_splits)]
        progressed = False
        for index in range(num_waves * deep_gemm.get_num_sms()):
            begin, end, _, _ = struct.unpack_from('<IIII', raw, 16 + num_splits * split_bytes + index * 16)
            for split in range(begin, end - 1):
                first, second = headers[split:split + 2]
                progressed |= first[0] == second[0] and first[2] < second[2] and first[3] < second[3]
        assert progressed, 'Fixture must exercise distinct Q-slot bases across splits within one entry'
    if broadcast_cache or (paged_rows is not None and (rows > 1 or count == 2048)):
        assert metadata.cpu().view(torch.int32)[1].item() > 1, 'Low-SM paged case must schedule multiple waves'
    if paged_rows is not None or contiguous_rows is not None:
        assert mask.any(dim=1).all() and (expected[mask] != 0).any()
    def check(logits):
        assert logits.dtype == torch.bfloat16 and logits.shape == (rows, count * block)
        assert logits.stride() == (ceil_div(count * block, 512) * 512, 1)
        assert_bitwise_equal(logits[mask], expected[mask], 'exact dyadic-fixture BF16 FMA reference (not an SM100 oracle)')
        diff = calc_diff(logits[mask], math_expected[mask])
        assert diff < 3e-4, f'Separate tolerance-only math check: {diff=}'
    baseline = compute(metadata)
    check(baseline)
    for _ in range(30):
        assert_bitwise_equal(compute(metadata)[mask], baseline[mask], 'sparse 30-run self-consistency')
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        warm_metadata = build()
        warm_logits = compute(warm_metadata)
    torch.cuda.current_stream().wait_stream(capture_stream)
    check(warm_logits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_metadata = build()
        captured = compute(captured_metadata)
    for empty in (False, True, False):
        end_key = 'context_lens' if page else 'cu_seq_len_k_end'
        metadata_kwargs[end_key].copy_(torch.tensor(starts if empty else ends, dtype=torch.int32, device='cuda'))
        graph.replay()
        torch.cuda.synchronize()
        check_sm120_sparse_metadata(captured_metadata, candidates, [0] * rows if empty else sizes,
                                    starts, block, fmt, page, table, requests, unaligned)
        if not empty:
            check(captured)
    print(f' > Sparse contract: {fmt=}, {rows=}, {block=}, {count=}, {page=}, {unaligned=}, {fractional=}, '
          f'{physical_tail=}, num_sms={deep_gemm.get_num_sms()}; exact fixture + math tolerance + 30 repeats')


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_contiguous():
    for fmt in ('mxfp4', 'mxfp8'):
        for block in (8, 16):
            for count in (4, 8, 68):
                for unaligned in (False, True):
                    exercise_sm120_sparse_contract(fmt, block, count, unaligned=unaligned)
            exercise_sm120_sparse_contract(fmt, block, 8, unaligned=True, fractional=True)
    original_num_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(1)
        for fmt in ('mxfp4', 'mxfp8'):
            for block in (8, 16):
                exercise_sm120_sparse_contract(fmt, block, 4096, unaligned=True)
    finally:
        deep_gemm.set_num_sms(original_num_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_paged():
    for fmt in ('mxfp4', 'mxfp8'):
        for block in (8, 16):
            for count, page in ((4, block), (8, 64), (68, 128)):
                exercise_sm120_sparse_contract(fmt, block, count, page=page)
            exercise_sm120_sparse_contract(fmt, block, 8, page=64, fractional=True)
    original_num_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(1)
        for fmt in ('mxfp4', 'mxfp8'):
            exercise_sm120_sparse_contract(fmt, 8, 68, page=64, broadcast_cache=True)
            for block in (8, 16):
                for rows in (1, 17):
                    exercise_sm120_sparse_contract(fmt, block, 68, page=64, paged_rows=rows)
    finally:
        deep_gemm.set_num_sms(original_num_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_contiguous_pipeline_tails():
    original_num_sms = deep_gemm.get_num_sms()
    try:
        for num_sms in (1, 2):
            deep_gemm.set_num_sms(num_sms)
            for fmt in ('mxfp4', 'mxfp8'):
                for block in (8, 16):
                    for physical_tail, count in ((137, 4), (137, 20), (641, 84), (1283, 164)):
                        exercise_sm120_sparse_contract(fmt, block, count, unaligned=True,
                                                        physical_tail=physical_tail)
    finally:
        deep_gemm.set_num_sms(original_num_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_eligible_pipeline_tails():
    original_num_sms = deep_gemm.get_num_sms()
    try:
        for num_sms, physical_tail in ((1, 137), (2, 641), (1, 1283)):
            deep_gemm.set_num_sms(num_sms)
            exercise_sm120_sparse_contract('mxfp4', 8, 2048, unaligned=True, fractional=True,
                                            physical_tail=physical_tail,
                                            min_waves=2 if num_sms == 1 else 1,
                                            require_split_slot_progression=physical_tail == 1283)
        for start, tail in ((4, 137), (4, 138), (4, 139), (4, 135), (5, 143), (6, 141), (7, 141)):
            deep_gemm.set_num_sms(1 if tail % 2 else 2)
            exercise_sm120_sparse_contract('mxfp4', 8, 2048, unaligned=True, fractional=True,
                                            physical_tail=tail, physical_start=start)
    finally:
        deep_gemm.set_num_sms(original_num_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_entry_balance_row_boundary():
    original_num_sms = deep_gemm.get_num_sms()
    try:
        for num_sms, rows, min_waves in ((1, 17, 8), (2, 33, 8), (2, 63, 16), (2, 64, 16), (2, 65, 16)):
            deep_gemm.set_num_sms(num_sms)
            exercise_sm120_sparse_contract('mxfp4', 8, 2048, unaligned=True, fractional=True,
                                            contiguous_rows=rows, min_waves=min_waves)
    finally:
        deep_gemm.set_num_sms(original_num_sms)


@test_filter(lambda: get_arch_major() == 12)
def test_sm120_sparse_concurrent_graphs():
    original_num_sms = deep_gemm.get_num_sms()
    streams, fixtures = [torch.cuda.Stream(), torch.cuda.Stream()], []
    try:
        deep_gemm.set_num_sms(2)
        for index, stream in enumerate(streams):
            rows, count, block, num_kv = 3, 2048, 8, 1283
            starts = [5, 5, 1264] if index == 0 else [13, 13, 1256]
            ends = [1283, 1282, 1283] if index == 0 else [1275, 1274, 1281]
            candidates, sizes = sm120_sparse_candidates(starts, ends, block, count)
            with torch.cuda.stream(stream):
                q, q_ref = sm120_sparse_fractional_quantized(rows * 32, 'mxfp4', index)
                q = q[0].view(rows, 32, -1), q[1].view(rows, 32)
                kv, kv_ref = sm120_sparse_fractional_quantized(num_kv, 'mxfp4', index + 2)
                units = torch.arange(rows * 16).view(rows, 16) % 15 + 1
                weights_ref = torch.stack((units, -units), dim=-1).flatten(1)
                weights = (weights_ref.float() / 16).to(device='cuda', dtype=torch.bfloat16)
                expected, _, mask = sm120_sparse_reference(q_ref.view(rows, 32, 128), kv_ref, weights_ref,
                                                          candidates, sizes, starts, ends, block,
                                                          output_unit_exponent=-12)
                kwargs = dict(cu_seq_len_k_start=torch.tensor(starts, device='cuda', dtype=torch.int32),
                              cu_seq_len_k_end=torch.tensor(ends, device='cuda', dtype=torch.int32),
                              num_kv_tokens=num_kv, sparse_kv_block_indices=candidates.cuda(),
                              qk_dtype=q[0].dtype, sparse_block_kv=block, use_unaligned_ks=True)
                metadata = deep_gemm.get_sparse_mqa_logits_metadata(**kwargs)
            stream.synchronize()
            check_sm120_sparse_metadata(metadata, candidates, sizes, starts, block, 'mxfp4', unaligned=True)
            with torch.cuda.stream(stream):
                warm = deep_gemm.fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, count, block,
                                                         use_unaligned_ks=True)
            stream.synchronize()
            assert_bitwise_equal(warm[mask], expected[mask], 'independent stream exact fractional warmup')
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured_metadata = deep_gemm.get_sparse_mqa_logits_metadata(**kwargs)
                output = deep_gemm.fp8_fp4_sparse_mqa_logits(q, kv, weights, captured_metadata, count, block,
                                                           use_unaligned_ks=True)
            fixtures.append(dict(graph=graph, metadata=captured_metadata, output=output, expected=expected, mask=mask,
                                 kwargs=kwargs, q=q, kv=kv, weights=weights, candidates=candidates, sizes=sizes,
                                 starts=starts, ends=ends))
        assert fixtures[0]['metadata'].data_ptr() != fixtures[1]['metadata'].data_ptr()
        assert fixtures[0]['output'].data_ptr() != fixtures[1]['output'].data_ptr()
        for repeat in range(30):
            empty = (repeat % 3 == 1, repeat % 3 == 2)
            for index, (stream, fixture) in enumerate(zip(streams, fixtures)):
                with torch.cuda.stream(stream):
                    fixture['kwargs']['cu_seq_len_k_end'].copy_(torch.tensor(
                        fixture['starts'] if empty[index] else fixture['ends'], device='cuda', dtype=torch.int32))
                    fixture['graph'].replay()
            for stream in streams:
                stream.synchronize()
            for index, fixture in enumerate(fixtures):
                check_sm120_sparse_metadata(fixture['metadata'], fixture['candidates'],
                                            [0] * 3 if empty[index] else fixture['sizes'], fixture['starts'],
                                            8, 'mxfp4', unaligned=True)
                if not empty[index]:
                    assert_bitwise_equal(fixture['output'][fixture['mask']], fixture['expected'][fixture['mask']],
                                         'concurrent independent graph exact fractional reference')
    finally:
        for stream in streams:
            stream.synchronize()
        deep_gemm.set_num_sms(original_num_sms)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    for fp4, dim in ((False, 32), (False, 64), (False, 128), (True, 128)):
        for heads in (16, 32, 64):
            test_sm120_dense_mqa_contract(fp4=fp4, dim=dim, heads=heads)
    for fp4, page in ((False, 64), (False, 128), (False, 256), (True, 32), (True, 64), (True, 128), (True, 256)):
        for dtype in (torch.float32, torch.bfloat16):
            test_sm120_paged_mqa_contract(fp4=fp4, page=page, dtype=dtype)
    for dtype in (torch.bfloat16, torch.float32):
        test_sm120_skip_head_mid(dtype=dtype)
    for dtype in (torch.bfloat16, torch.float32):
        for packed_a, packed_b in ((False, False), (False, True), (True, False)):
            test_sm120_skip_head_mid_reject_fp32_scale_without_cast(dtype=dtype, packed_a=packed_a, packed_b=packed_b)
    for dim, heads, page in ((32, 32, 64), (64, 64, 128)):
        test_sm120_paged_mqa_metadata_graph(dim=dim, heads=heads, page=page)
    test_sm120_mqa_host_rejections()
    for dtype in (torch.bfloat16, torch.float32):
        for padded in (False, True):
            for pdl in (False, True):
                for shape, splits, sms, branch in (((128, 16, 128), (8, 8, 8), 2, 'scalar'), ((65, 16, 128), (8, 8, 8), 2, 'scalar'), ((129, 16, 128), (0, 8, 8), 2, 'scalar'), ((1, 16, 128), (8, 8, 0), 2, 'scalar'), ((65, 16, 128), (8, 0, 8), 2, 'scalar'), ((32, 256, 16384), (128, 64, 128), 32, 'split'), ((32, 512, 16384), (128, 64, 128), 32, 'split'), ((32, 128, 16384), (0, 64, 128), 32, 'split'), ((32, 128, 16384), (128, 64, 0), 32, 'split'), ((32, 256, 16384), (128, 0, 128), 32, 'split'), ((128, 256, 2048), (128, 64, 128), 2, 'tma'), ((1024, 512, 128), (128, 64, 128), 32, 'tma')):
                    test_sm120_skip_head_stores(dtype=dtype, padded=padded, pdl=pdl, shape=shape, splits=splits, sms=sms, branch=branch)
    test_sm120_sparse_metadata()
    test_sm120_sparse_metadata_wave_histogram_reuse()
    test_sm120_sparse_entry_partition_mapping()
    test_sm120_sparse_contiguous()
    test_sm120_sparse_paged()
    test_sm120_sparse_contiguous_pipeline_tails()
    test_sm120_sparse_eligible_pipeline_tails()
    test_sm120_sparse_entry_balance_row_boundary()
    test_sm120_sparse_concurrent_graphs()
