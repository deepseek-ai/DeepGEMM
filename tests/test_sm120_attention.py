import pytest
import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp4, cast_back_from_fp4
from test_sm120_gemm import fp8_operand
from sm120_test_storage import native_matrix


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)


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


@pytest.mark.parametrize('fp4,dim', ((False, 32), (False, 64), (False, 128), (True, 128)))
@pytest.mark.parametrize('heads', (16, 32, 64))
def test_sm120_dense_mqa_contract(fp4, dim, heads):
    assert get_arch_major() == 12
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


@pytest.mark.parametrize('fp4,page', ((False, 64), (False, 128), (False, 256), (True, 32), (True, 64), (True, 128), (True, 256)))
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


@pytest.mark.parametrize('identity', (False, True))
def test_sm120_skip_head_split_k_guard(identity):
    m, n, k = 32, 256, 16384
    aa, sa, ar = fp8_operand(1, m, k, 1, 128, 1)
    bb, sb, br = fp8_operand(1, n, k, 1, 128, 2)
    a, b = (aa[0].cuda(), sa[0].cuda()), (bb[0].cuda(), sb[0].cuda())
    splits = (128, 0, 128) if identity else (128, 64, 128)
    d, storage = native_matrix(torch.full((m, n if identity else 320), 19, dtype=torch.bfloat16), True, 1)
    before = storage.clone()
    if identity:
        deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, splits, recipe=(1, 1, 128))
        check_logits(d.cpu(), ar[0] @ br[0].T, torch.bfloat16)
        valid = torch.zeros_like(storage, dtype=torch.bool)
        valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
        assert (storage[~valid] == 19).all()
    else:
        with pytest.raises(RuntimeError, match='split-K reduction'):
            deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, splits, recipe=(1, 1, 128))
        assert torch.equal(storage, before)


def test_sm120_skip_head_direct_guard():
    aa, sa, _ = fp8_operand(1, 128, 128, 1, 128, 1)
    bb, sb, _ = fp8_operand(1, 16, 128, 1, 128, 2)
    d, storage = native_matrix(torch.full((128, 24), 19, dtype=torch.bfloat16), True, 1)
    before = storage.clone()
    with pytest.raises(RuntimeError, match='direct stores bound transformed'):
        deep_gemm.fp8_gemm_nt_skip_head_mid((aa[0].cuda(), sa[0].cuda()), (bb[0].cuda(), sb[0].cuda()),
                                          d, (8, 8, 8), recipe=(1, 1, 128))
    assert torch.equal(storage, before)
