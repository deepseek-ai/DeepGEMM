import torch
import pytest

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major, test_filter as _test_filter
from deep_gemm.utils.math import ceil_div, per_block_cast_to_fp8, per_token_cast_to_fp8


def _cast_kv_cache_to_fp8(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_kv, num_heads, head_dim = kv_cache.shape
    assert num_heads == 1

    scale = kv_cache.abs().float().amax(dim=3, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv_cache * scale.reciprocal()).to(torch.float8_e4m3fn)
    kv_simulated = kv_fp8.float() * scale

    fused = torch.empty(
        (num_blocks, block_kv * (head_dim + 4)),
        device=kv_cache.device,
        dtype=torch.uint8,
    )
    fused[:, : block_kv * head_dim] = kv_fp8.view(num_blocks, block_kv * head_dim).view(torch.uint8)
    fused[:, block_kv * head_dim :] = scale.view(num_blocks, block_kv).view(torch.uint8)
    return fused.view(num_blocks, block_kv, num_heads, head_dim + 4), kv_simulated


def _paged_mqa_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.shape
    _, block_kv, _, _ = kv_cache.shape
    logits = torch.full(
        (batch_size * next_n, max_context_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            for token_idx in range(context_len):
                block_idx = int(block_table[batch_idx, token_idx // block_kv].item())
                kv = kv_cache[block_idx, token_idx % block_kv, 0].float()
                score = (q[batch_idx, next_idx].float() * kv).sum(dim=1)
                logits[row, token_idx] = (score.relu() * weights[row]).sum()

    return logits


@_test_filter(lambda: get_arch_major() >= 12)
def test_sm120_hc_prenorm_gemm_kernel_path() -> None:
    torch.manual_seed(0)

    m, n, k = 5, 8, 256
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.float32, device="cuda")
    for num_splits in (None, 3):
        d = torch.empty((m, n), dtype=torch.float32, device="cuda") if num_splits is None else \
            torch.empty((num_splits, m, n), dtype=torch.float32, device="cuda")
        sqr_sum = torch.empty((m,), dtype=torch.float32, device="cuda") if num_splits is None else \
            torch.empty((num_splits, m), dtype=torch.float32, device="cuda")

        deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)

        final_d = d if num_splits is None else d.sum(dim=0)
        final_sqr_sum = sqr_sum if num_splits is None else sqr_sum.sum(dim=0)
        ref_d = a.float() @ b.T
        ref_sqr_sum = a.float().square().sum(dim=-1)
        diff = max(calc_diff(final_d, ref_d), calc_diff(final_sqr_sum, ref_sqr_sum))
        assert diff < 1e-6, f"{num_splits=}, {diff=}"


@_test_filter(lambda: get_arch_major() >= 12)
def test_sm120_hc_prenorm_gemm_larger_tiles() -> None:
    torch.manual_seed(6)

    for m, n, k, num_splits in (
        (70, 16, 512, 2),
        (70, 32, 512, 4),
        (129, 32, 1024, 4),
    ):
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((n, k), dtype=torch.float32, device="cuda")
        d = torch.empty((num_splits, m, n), dtype=torch.float32, device="cuda")
        sqr_sum = torch.empty((num_splits, m), dtype=torch.float32, device="cuda")

        deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)

        ref_d = a.float() @ b.T
        ref_sqr_sum = a.float().square().sum(dim=-1)
        diff = max(calc_diff(d.sum(dim=0), ref_d), calc_diff(sqr_sum.sum(dim=0), ref_sqr_sum))
        assert diff < 1e-6, f"{m=}, {n=}, {k=}, {num_splits=}, {diff=}"


@_test_filter(lambda: get_arch_major() >= 12)
def test_sm120_fp8_paged_mqa_logits_scalar_path() -> None:
    torch.manual_seed(1)

    batch_size, next_n, num_heads, head_dim = 2, 2, 32, 32
    block_kv = 64
    max_context_len = 96
    num_blocks = 4

    q = torch.randn((batch_size, next_n, num_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    q_fp8 = q.to(torch.float8_e4m3fn)
    q_simulated = q_fp8.float()

    kv_cache = torch.randn((num_blocks, block_kv, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    fused_kv_cache, kv_simulated = _cast_kv_cache_to_fp8(kv_cache)

    weights = torch.randn((batch_size * next_n, num_heads), device="cuda", dtype=torch.float32)
    context_lens = torch.tensor([[17, 23], [31, 47]], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)

    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        deep_gemm.get_num_sms(),
    )
    expected = _paged_mqa_reference(
        q_simulated,
        kv_simulated,
        weights,
        context_lens,
        block_table,
        max_context_len,
    )
    valid_mask = torch.arange(max_context_len, device="cuda").unsqueeze(0) < context_lens.view(-1, 1)
    for logits_dtype in (torch.float32, torch.bfloat16):
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            q=(q_fp8, None),
            kv_cache=fused_kv_cache,
            weights=weights,
            context_lens=context_lens,
            block_table=block_table,
            schedule_meta=schedule_meta,
            max_context_len=max_context_len,
            clean_logits=False,
            logits_dtype=logits_dtype,
        )

        assert logits.dtype == logits_dtype
        diff = calc_diff(logits.float()[valid_mask], expected[valid_mask])
        threshold = 1e-6 if logits_dtype == torch.float32 else 2e-6
        assert diff < threshold, f"{logits_dtype=}, {diff=}"


@_test_filter(lambda: get_arch_major() >= 12)
def test_sm120_fp8_paged_mqa_logits_v4_shape_scalar_path() -> None:
    torch.manual_seed(4)

    batch_size, next_n, num_heads, head_dim = 1, 1, 64, 128
    block_kv = 64
    max_context_len = 65
    num_blocks = 2

    q = torch.randn((batch_size, next_n, num_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    q_fp8 = q.to(torch.float8_e4m3fn)
    q_simulated = q_fp8.float()
    kv_cache = torch.randn((num_blocks, block_kv, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    fused_kv_cache, kv_simulated = _cast_kv_cache_to_fp8(kv_cache)
    weights = torch.randn((batch_size * next_n, num_heads), device="cuda", dtype=torch.float32)
    context_lens = torch.tensor([[max_context_len]], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)

    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        deep_gemm.get_num_sms(),
    )
    expected = _paged_mqa_reference(
        q_simulated,
        kv_simulated,
        weights,
        context_lens,
        block_table,
        max_context_len,
    )
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        q=(q_fp8, None),
        kv_cache=fused_kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_meta=schedule_meta,
        max_context_len=max_context_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )

    valid_mask = torch.arange(max_context_len, device="cuda").unsqueeze(0) < context_lens
    diff = calc_diff(logits.float()[valid_mask], expected[valid_mask])
    assert diff < 1e-6, f"{diff=}"


@_test_filter(lambda: get_arch_major() >= 12)
@pytest.mark.parametrize(
    "token_groups, cache_q",
    ((1, False), (2, False), (4, False), (8, False), (2, True), (4, True), (8, True)),
)
def test_sm120_fp8_paged_mqa_logits_tiled_path(monkeypatch, token_groups: int, cache_q: bool) -> None:
    torch.manual_seed(5)
    monkeypatch.setenv("DG_SM120_PAGED_MQA_TILED", "1")
    monkeypatch.setenv("DG_SM120_PAGED_MQA_TILED_GROUPS", str(token_groups))
    monkeypatch.setenv("DG_SM120_PAGED_MQA_TILED_CACHE_Q", "1" if cache_q else "0")

    batch_size, next_n, num_heads, head_dim = 2, 1, 64, 128
    block_kv = 64
    max_context_len = 73
    num_blocks = 4

    q = torch.randn((batch_size, next_n, num_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    q_fp8 = q.to(torch.float8_e4m3fn)
    q_simulated = q_fp8.float()
    kv_cache = torch.randn((num_blocks, block_kv, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    fused_kv_cache, kv_simulated = _cast_kv_cache_to_fp8(kv_cache)
    weights = torch.randn((batch_size * next_n, num_heads), device="cuda", dtype=torch.float32)
    context_lens = torch.tensor([[max_context_len], [37]], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)

    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        deep_gemm.get_num_sms(),
    )
    expected = _paged_mqa_reference(
        q_simulated,
        kv_simulated,
        weights,
        context_lens,
        block_table,
        max_context_len,
    )
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        q=(q_fp8, None),
        kv_cache=fused_kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_meta=schedule_meta,
        max_context_len=max_context_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )

    valid_mask = torch.arange(max_context_len, device="cuda").unsqueeze(0) < context_lens
    diff = calc_diff(logits.float()[valid_mask], expected[valid_mask])
    assert diff < 1e-6, f"{diff=}"


@_test_filter(lambda: get_arch_major() >= 12)
def test_sm120_fp8_paged_mqa_logits_varlen_scalar_path() -> None:
    torch.manual_seed(2)

    raw_batch_size, num_heads, head_dim = 3, 32, 32
    block_kv = 32
    max_context_len = 80
    num_blocks = 8
    tokens_per_seq = torch.tensor([1, 2, 1], device="cuda", dtype=torch.int32)
    indices = torch.arange(raw_batch_size, device="cuda", dtype=torch.int32)
    indices = indices.repeat_interleave(tokens_per_seq)
    batch_size = int(tokens_per_seq.sum().item())
    next_n = 1

    base_context_lens = torch.tensor([11, 19, 7], device="cuda", dtype=torch.int32)
    offsets = torch.cat([
        torch.arange(int(token_count.item()), device="cuda", dtype=torch.int32)
        for token_count in tokens_per_seq
    ])
    context_lens = (base_context_lens.repeat_interleave(tokens_per_seq) + offsets).view(-1, 1)

    q = torch.randn(
        (batch_size, next_n, num_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)
    q_simulated = q_fp8.float()
    kv_cache = torch.randn((num_blocks, block_kv, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    fused_kv_cache, kv_simulated = _cast_kv_cache_to_fp8(kv_cache)
    weights = torch.randn((batch_size * next_n, num_heads), device="cuda", dtype=torch.float32)

    base_block_table = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 0],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    block_table = base_block_table.repeat_interleave(tokens_per_seq, dim=0)

    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        block_kv,
        deep_gemm.get_num_sms(),
        indices=indices,
    )
    expected = _paged_mqa_reference(
        q_simulated,
        kv_simulated,
        weights,
        context_lens,
        block_table,
        max_context_len,
    )
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        q=(q_fp8, None),
        kv_cache=fused_kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_meta=schedule_meta,
        max_context_len=max_context_len,
        clean_logits=False,
        logits_dtype=torch.float32,
        indices=indices,
    )

    valid_mask = torch.arange(max_context_len, device="cuda").unsqueeze(0) < context_lens
    diff = calc_diff(logits.float()[valid_mask], expected[valid_mask])
    assert diff < 1e-6, f"{diff=}"


@_test_filter(lambda: get_arch_major() >= 12)
def test_sm120_fp8_einsum_scalar_path() -> None:
    torch.manual_seed(3)

    batch_size, num_heads, rank, output_dim = 3, 4, 256, 128
    x = torch.randn((batch_size, num_heads, rank), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((num_heads, output_dim, rank), device="cuda", dtype=torch.bfloat16)
    expected = torch.einsum("bhr,hdr->bhd", x, y)

    x_fp8, x_sf = per_token_cast_to_fp8(x.view(-1, rank), use_ue8m0=False)
    x_in = x_fp8.view(batch_size, num_heads, rank), x_sf.view(batch_size, num_heads, ceil_div(rank, 128))
    y_fp8 = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_sf = torch.empty(
        (num_heads, ceil_div(output_dim, 128), ceil_div(rank, 128)),
        device="cuda",
        dtype=torch.float32,
    )
    for head_idx in range(num_heads):
        y_fp8[head_idx], y_sf[head_idx] = per_block_cast_to_fp8(y[head_idx], use_ue8m0=False)

    actual = torch.empty((batch_size, num_heads, output_dim), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", x_in, (y_fp8, y_sf), actual)

    diff = calc_diff(actual, expected)
    assert diff < 1e-3, f"{diff=}"


@_test_filter(lambda: get_arch_major() >= 12)
def test_sm120_fp8_einsum_v4_shape_scalar_path() -> None:
    torch.manual_seed(5)

    batch_size, num_heads, rank, output_dim = 5, 64, 128, 128
    x = torch.randn((batch_size, num_heads, rank), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((num_heads, output_dim, rank), device="cuda", dtype=torch.bfloat16)
    expected = torch.einsum("bhr,hdr->bhd", x, y)

    x_fp8, x_sf = per_token_cast_to_fp8(x.view(-1, rank), use_ue8m0=False)
    x_in = x_fp8.view(batch_size, num_heads, rank), x_sf.view(batch_size, num_heads, 1)
    y_fp8 = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_sf = torch.empty((num_heads, 1, 1), device="cuda", dtype=torch.float32)
    for head_idx in range(num_heads):
        y_fp8[head_idx], y_sf[head_idx] = per_block_cast_to_fp8(y[head_idx], use_ue8m0=False)

    actual = torch.empty((batch_size, num_heads, output_dim), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", x_in, (y_fp8, y_sf), actual)

    diff = calc_diff(actual, expected)
    assert diff < 1e-3, f"{diff=}"
