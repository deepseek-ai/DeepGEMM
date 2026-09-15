import subprocess
import sys
from pathlib import Path

import pytest
import torch

import deep_gemm
from deep_gemm.testing import assert_bitwise_equal, calc_diff, get_arch_major
from deep_gemm.utils import per_custom_dims_cast_to_fp8
from test_attention import ref_diff_tol, ref_fp8_mqa_logits, to_mqa_weights


def require_arch(major):
    if not torch.cuda.is_available() or get_arch_major() != major:
        pytest.skip(f'requires SM{major}0')


def graph_replay(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    return graph, output


@pytest.mark.parametrize('seq_len,num_heads', [(4, 8), (12, 16), (516, 32), (512, 64)])
@pytest.mark.parametrize('head_dim', [32, 64, 128])
@pytest.mark.parametrize('logits_dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('compressed', [False, True])
def test_sm100_fp16_dense_tail_graph(seq_len, num_heads, head_dim, logits_dtype, compressed):
    require_arch(10)
    torch.manual_seed(123)
    seq_len_kv, window = 769, 127
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = (torch.randn(seq_len, num_heads, device='cuda') * 0.1).half()
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
    rows = torch.arange(seq_len, device='cuda')
    ks = torch.where(rows % 2 == 0, 1, seq_len_kv - window).int()
    ke = ks + window
    ks[::4], ke[::4] = 0, 0
    kwargs = dict(q=(q_fp8, None), kv=(kv_fp8, kv_sf), weights=weights,
                  cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                  clean_logits=not compressed, max_seqlen_k=window if compressed else 0,
                  logits_dtype=logits_dtype)
    call = lambda: deep_gemm.fp8_fp4_mqa_logits(**kwargs)
    eager = call()
    graph, captured = graph_replay(call)

    def validate(actual):
        reference, _ = ref_fp8_mqa_logits(q, kv, weights.float(), ks, ke)
        simulated_kv = (kv_fp8.float() * kv_sf[:, None]).bfloat16()
        simulated, _ = ref_fp8_mqa_logits(q_fp8.bfloat16(), simulated_kv, weights.float(), ks, ke)
        if compressed:
            positions = ks[:, None] + torch.arange(window, device='cuda')[None, :]
            mask = positions < ke[:, None]
            reference = reference.gather(1, positions.long())
            simulated = simulated.gather(1, positions.long())
            assert actual.shape == (seq_len, window)
        else:
            mask = reference != float('-inf')
            assert torch.equal(actual == float('-inf'), ~mask)
            assert actual.shape == (seq_len, seq_len_kv)
        assert actual.dtype == logits_dtype
        actual = actual.float().masked_fill(~mask, 0)
        assert calc_diff(actual, reference.masked_fill(~mask, 0)) < 1e-3
        assert calc_diff(actual, simulated.masked_fill(~mask, 0)) < ref_diff_tol(True)
        return actual

    eager_masked = validate(eager)
    for _ in range(5):
        graph.replay()
        assert_bitwise_equal(validate(captured), eager_masked, 'FP16 dense graph replay')
    ks.zero_()
    ke.fill_(window)
    graph.replay()
    validate(captured)
    ke.zero_()
    graph.replay()
    validate(captured)


@pytest.mark.parametrize('seq_len,seq_len_kv', [(2048, 8192), (2048, 65536), (8192, 8192), (8192, 65536)])
@pytest.mark.parametrize('num_heads', [32, 64])
@pytest.mark.parametrize('head_dim', [32, 64, 128])
@pytest.mark.parametrize('logits_dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('compressed', [False, True])
def test_sm100_fp16_dense_large(seq_len, seq_len_kv, num_heads, head_dim, logits_dtype, compressed):
    from test_attention import check_mqa_logits_chunked

    require_arch(10)
    torch.manual_seed(123)
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = (torch.randn(seq_len, num_heads, device='cuda') * 0.1).half()
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)
    rows = torch.arange(seq_len, device='cuda', dtype=torch.int32)
    ks = (rows % 3) * 257
    ke = seq_len_kv - seq_len + rows + 1
    ke = torch.maximum(ks, ke)
    ks[::32], ke[::32] = 0, 0
    window = int((ke - ks).max())
    kwargs = dict(q=(q_fp8, None), kv=(kv_fp8, kv_sf), weights=weights,
                  cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
                  clean_logits=not compressed, max_seqlen_k=window if compressed else 0,
                  logits_dtype=logits_dtype)
    actual = deep_gemm.fp8_fp4_mqa_logits(**kwargs)
    assert actual.shape == (seq_len, window if compressed else seq_len_kv)
    assert actual.dtype == logits_dtype
    assert check_mqa_logits_chunked(actual, q, kv, weights, ks, ke, compressed) < 1e-3
    simulated_kv = (kv_fp8.float() * kv_sf[:, None]).bfloat16()
    assert check_mqa_logits_chunked(
        actual, q_fp8.bfloat16(), simulated_kv, weights, ks, ke, compressed) < ref_diff_tol(True)
    repeated = deep_gemm.fp8_fp4_mqa_logits(**kwargs)
    for m0 in range(0, seq_len, 256):
        m1 = min(m0 + 256, seq_len)
        if compressed:
            mask = torch.arange(window, device='cuda')[None, :] < (ke[m0:m1] - ks[m0:m1])[:, None]
            assert_bitwise_equal(actual[m0:m1][mask], repeated[m0:m1][mask], 'large FP16 dense repeat')
        else:
            assert_bitwise_equal(actual[m0:m1], repeated[m0:m1], 'large FP16 dense repeat')


@pytest.mark.parametrize('seq_len,num_heads,with_metadata', [(510, 32, False), (512, 12, False), (512, 32, True)])
def test_sm100_fp16_dense_rejections(seq_len, num_heads, with_metadata):
    require_arch(10)
    q = torch.zeros(seq_len, num_heads, 64, device='cuda').to(torch.float8_e4m3fn)
    kv = torch.zeros(256, 64, device='cuda').to(torch.float8_e4m3fn)
    sf = torch.ones(256, device='cuda')
    weights = torch.ones(seq_len, num_heads, device='cuda', dtype=torch.float16)
    ks = torch.zeros(seq_len, device='cuda', dtype=torch.int32)
    ke = torch.full_like(ks, 256)
    meta = torch.empty(1, device='cuda', dtype=torch.int32) if with_metadata else None
    with pytest.raises((RuntimeError, AssertionError)):
        deep_gemm.fp8_fp4_mqa_logits((q, None), (kv, sf), weights, ks, ke, schedule_meta=meta)


@pytest.mark.parametrize('weights_dtype,logits_dtype', [
    (torch.float32, torch.float32), (torch.float32, torch.bfloat16), (torch.bfloat16, torch.bfloat16)])
def test_sm100_unified_dense_scheduled_preserved(weights_dtype, logits_dtype):
    require_arch(10)
    torch.manual_seed(789)
    seq_len, num_heads, head_dim, seq_len_kv = 13, 12, 64, 512
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda').to(torch.float8_e4m3fn)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda').to(torch.float8_e4m3fn)
    sf = torch.ones(seq_len_kv, device='cuda')
    weights = to_mqa_weights(torch.randn(seq_len, num_heads, device='cuda'), weights_dtype)
    ks = torch.zeros(seq_len, device='cuda', dtype=torch.int32)
    ke = torch.full_like(ks, seq_len_kv)
    meta = deep_gemm.get_mqa_logits_metadata(ks, ke, seq_len_kv, num_heads)
    eager = deep_gemm.fp8_fp4_mqa_logits((q, None), (kv, sf), weights, ks, ke, logits_dtype=logits_dtype)
    scheduled = deep_gemm.fp8_fp4_mqa_logits(
        (q, None), (kv, sf), weights, ks, ke, logits_dtype=logits_dtype, schedule_meta=meta)
    reference, _ = ref_fp8_mqa_logits(q.float(), kv.float(), weights.float(), ks, ke)
    assert calc_diff(scheduled, reference) < ref_diff_tol(logits_dtype == torch.bfloat16)
    assert_bitwise_equal(eager, scheduled, 'unified dense scheduled path')


@pytest.mark.parametrize('page_kv', [32, 64])
@pytest.mark.parametrize('next_n', [1, 2, 4])
@pytest.mark.parametrize('head_dim,num_heads', [(32, 32), (64, 64), (128, 32)])
@pytest.mark.parametrize('logits_dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('all_zero', [False, True])
def test_sm90_paged_pages_zero_tail_graph(page_kv, next_n, head_dim, num_heads, logits_dtype, all_zero):
    require_arch(9)
    run_sm90_paged_graph_case(page_kv, next_n, head_dim, num_heads, logits_dtype, all_zero, 2)


def test_sm90_paged_page32_next4_hardware_sms_graph():
    require_arch(9)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    run_sm90_paged_graph_case(32, 4, 64, 32, torch.float32, False, num_sms)


def run_sm90_paged_graph_case(page_kv, next_n, head_dim, num_heads, logits_dtype, all_zero, num_sms):
    torch.manual_seed(456)
    old_num_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(num_sms)
    try:
        batch_size, max_context_len = 7, 513
        lengths = [0, 33, 0, 257, 1, 0, 513]
        if all_zero:
            lengths = [0] * batch_size
        context_lens = torch.tensor(lengths, device='cuda', dtype=torch.int32)[:, None].repeat(1, next_n)
        if next_n > 1:
            context_lens[:, 0] = 0
        max_pages = (max_context_len + page_kv - 1) // page_kv
        num_pages = batch_size * max_pages
        block_table = torch.full((batch_size, max_pages), -1, device='cuda', dtype=torch.int32)
        page_order = torch.randperm(num_pages, device='cuda').int().view(batch_size, max_pages)
        for row, length in enumerate(lengths):
            count = (length + page_kv - 1) // page_kv
            block_table[row, :count] = page_order[row, :count]
        q = torch.randn(batch_size, next_n, num_heads, head_dim, device='cuda').to(torch.float8_e4m3fn)
        weights = to_mqa_weights(torch.randn(batch_size * next_n, num_heads, device='cuda'), torch.float32)
        kv = torch.randn(num_pages, page_kv, head_dim, device='cuda').to(torch.float8_e4m3fn)
        scales = torch.rand(num_pages, page_kv, device='cuda') + 0.1
        fused = torch.empty(num_pages, page_kv * (head_dim + 4), device='cuda', dtype=torch.uint8)
        fused[:, :page_kv * head_dim] = kv.view(torch.uint8).reshape(num_pages, -1)
        fused[:, page_kv * head_dim:] = scales.view(torch.uint8).reshape(num_pages, -1)
        fused = fused.view(num_pages, page_kv, 1, head_dim + 4)
        slots = deep_gemm.get_num_sms() // (2 if next_n == 4 else 1)

        def call():
            meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_kv, slots)
            logits = deep_gemm.fp8_fp4_paged_mqa_logits(
                (q, None), fused, weights, context_lens, block_table, meta, max_context_len,
                logits_dtype=logits_dtype)
            return logits, meta

        meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_kv, slots)
        assert meta.shape == (slots + 1, 2)
        assert torch.equal(meta[-1], torch.tensor([batch_size, 0], device='cuda', dtype=torch.int32))
        if all_zero:
            assert torch.equal(meta, meta[-1:].expand_as(meta))
        if next_n == 4:
            wrong_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_kv, deep_gemm.get_num_sms())
            with pytest.raises((RuntimeError, AssertionError)):
                deep_gemm.fp8_fp4_paged_mqa_logits(
                    (q, None), fused, weights, context_lens, block_table, wrong_meta, max_context_len)
        eager, eager_meta = call()
        graph, (captured, captured_meta) = graph_replay(call)
        reference = torch.zeros(batch_size * next_n, max_context_len, device='cuda')
        for row, length in enumerate(lengths):
            if not length:
                continue
            pages = block_table[row, :(length + page_kv - 1) // page_kv].long()
            keys = (kv.float()[pages] * scales[pages, :, None]).reshape(-1, head_dim)[:length]
            scores = torch.einsum('nhd,kd->nhk', q[row].float(), keys).relu()
            reference[row * next_n:(row + 1) * next_n, :length] = (
                scores * weights[row * next_n:(row + 1) * next_n, :, None]).sum(1)
        mask = torch.arange(max_context_len, device='cuda')[None, :] < context_lens.reshape(-1, 1)
        expected = reference.masked_fill(~mask, 0)
        eager_masked = eager.float().masked_fill(~mask, 0)
        assert eager.shape == (batch_size * next_n, max_context_len)
        assert eager.dtype == logits_dtype
        if not all_zero:
            assert calc_diff(eager_masked, expected) < ref_diff_tol(logits_dtype == torch.bfloat16)
        for _ in range(5):
            graph.replay()
            assert_bitwise_equal(captured.float().masked_fill(~mask, 0), eager_masked, 'SM90 paged graph replay')
        original_context_lens = context_lens.clone()
        original_block_table = block_table.clone()
        context_lens.zero_()
        block_table.fill_(-1)
        graph.replay()
        sentinel = torch.tensor([batch_size, 0], device='cuda', dtype=torch.int32)
        assert torch.equal(captured_meta, sentinel.expand(slots + 1, 2))
        context_lens.copy_(original_context_lens)
        block_table.copy_(original_block_table)
        captured.zero_()
        graph.replay()
        assert_bitwise_equal(captured_meta, eager_meta, 'SM90 paged restored metadata')
        restored = captured.float().masked_fill(~mask, 0)
        assert_bitwise_equal(restored, eager_masked, 'SM90 paged restored-live graph replay')
        assert calc_diff(restored, expected) < ref_diff_tol(logits_dtype == torch.bfloat16)
    finally:
        deep_gemm.set_num_sms(old_num_sms)


@pytest.mark.parametrize('case', [
    'start_past_end', 'end_out_of_batch', 'start_out_of_batch', 'sentinel_nonzero',
    'kv_past_end', 'stale_zero', 'stale_shrink', 'exact_end',
])
@pytest.mark.parametrize('next_n', [1, 4])
@pytest.mark.parametrize('page_kv', [32, 64])
def test_sm90_paged_scheduler_bounds(case, next_n, page_kv):
    require_arch(9)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case, str(next_n), str(page_kv)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_sm90_paged_scheduler_bounds(case, next_n, page_kv):
    deep_gemm.set_num_sms(4 if next_n == 4 else 2)
    batch_size, head_dim, num_heads, max_context_len = 3, 32, 32, 513
    max_pages = (max_context_len + page_kv - 1) // page_kv
    context_lens = torch.zeros(batch_size, next_n, device='cuda', dtype=torch.int32)
    block_table = torch.full((batch_size, max_pages), -1, device='cuda', dtype=torch.int32)
    q = torch.ones(batch_size, next_n, num_heads, head_dim, device='cuda').to(torch.float8_e4m3fn)
    weights = torch.ones(batch_size * next_n, num_heads, device='cuda')
    kv = torch.ones(max_pages, page_kv, head_dim, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.ones(max_pages, page_kv, device='cuda')
    fused = torch.empty(max_pages, page_kv * (head_dim + 4), device='cuda', dtype=torch.uint8)
    fused[:, :page_kv * head_dim] = kv.view(torch.uint8).reshape(max_pages, -1)
    fused[:, page_kv * head_dim:] = scales.view(torch.uint8).reshape(max_pages, -1)
    fused = fused.view(max_pages, page_kv, 1, head_dim + 4)
    if case in ('stale_zero', 'stale_shrink', 'exact_end'):
        context_lens[0].fill_(max_context_len)
        block_table[0] = torch.arange(max_pages, device='cuda', dtype=torch.int32)
        metadata = deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_kv, 2)
        torch.cuda.synchronize()
        if case == 'stale_zero':
            context_lens.zero_()
            block_table.fill_(-1)
        elif case == 'stale_shrink':
            context_lens[0].fill_(1)
            block_table[0, 1:].fill_(-1)
    else:
        bounds = {
            'start_past_end': ((2, 0), (1, 0)),
            'end_out_of_batch': ((0, 0), (0x7fffffff, 0)),
            'start_out_of_batch': ((batch_size + 1, 0), (0x7fffffff, 0)),
            'sentinel_nonzero': ((batch_size, 1), (batch_size, 2)),
            'kv_past_end': ((0, 2), (0, 1)),
        }
        begin, end = bounds[case]
        metadata = torch.tensor([begin, end, end], device='cuda', dtype=torch.int32)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q, None), fused, weights, context_lens, block_table, metadata, max_context_len)
    torch.cuda.synchronize()
    if case in ('stale_shrink', 'exact_end'):
        valid = 1 if case == 'stale_shrink' else max_context_len
        assert torch.equal(logits[:next_n, :valid], torch.full_like(logits[:next_n, :valid], num_heads * head_dim))


if __name__ == '__main__':
    run_sm90_paged_scheduler_bounds(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
