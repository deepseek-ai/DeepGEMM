import dataclasses
import os
import random
import torch
from typing import Tuple, List

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    assert_bitwise_equal, calc_diff, count_bytes,
    get_arch_major,
    test_filter
)
from deep_gemm.utils import (ceil_div, per_token_cast_to_fp4, cast_back_from_fp4, per_token_cast_to_fp8,
                             cast_back_from_fp8, per_custom_dims_cast_to_fp8)


def sample_mqa_cases(name: str, cases: List[tuple]) -> List[tuple]:
    num_cases = os.getenv('DG_MQA_NUM_CASES')
    if num_cases is None:
        selected = cases
    else:
        rng = random.Random({'prefill': 0, 'paged': 100000, 'sparse': 200000}[name])
        selected = rng.sample(cases, min(int(num_cases), len(cases)))
    print(f' > {name}: running {len(selected)}/{len(cases)} cases')
    return selected


def to_mqa_weights(weights: torch.Tensor) -> torch.Tensor:
    # Rows are padded to 16 bytes for the TMA loads (BF16 on SM100, float on SM90)
    element_size = weights.element_size()
    stride = ceil_div(weights.size(1) * element_size, 16) * 16 // element_size
    storage = torch.empty((weights.size(0), stride), device=weights.device, dtype=weights.dtype)
    result = storage[:, :weights.size(1)]
    result.copy_(weights)
    return result


def dtype_tag(dtype: torch.dtype) -> str:
    return 'BF16' if dtype == torch.bfloat16 else 'FP32'


# SM100 takes MXFP4 / MXFP8 with BF16 weights and logits; SM90 takes E4M3 with one float scale per KV token, float
# weights and float logits
def mqa_logits_formats():
    if get_arch_major() == 10:
        return [(fmt, torch.bfloat16) for fmt in ('mxfp4', 'mxfp8')]
    return [('fp8', torch.float)]


def mqa_logits_heads(is_mxfp4: bool):
    if get_arch_major() == 10:
        heads = (8, 12, 16, 20, 32, 64)
        head_dims = (64, 128) if is_mxfp4 else (32, 64, 128)
        return heads, head_dims
    return (32, 64), (32, 64, 128)


def quantize_mqa_q(q: torch.Tensor, fmt: str):
    # Returns the kernel input `(q_fp, q_sf)` and the dequantized BF16 copy for the simulated reference
    if fmt == 'fp8':
        q_fp = q.to(torch.float8_e4m3fn)
        return (q_fp, None), q_fp.to(torch.bfloat16)
    is_mxfp4 = fmt == 'mxfp4'
    head_dim = q.size(-1)
    q_fp, q_sf = (per_token_cast_to_fp4 if is_mxfp4 else per_token_cast_to_fp8)(
        q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q_simulated = (cast_back_from_fp4 if is_mxfp4 else cast_back_from_fp8)(
        q_fp, q_sf, gran_k=32, use_packed_ue8m0=True).to(torch.bfloat16)
    return (q_fp.view(*q.shape[:-1], head_dim // 2 if is_mxfp4 else head_dim), q_sf.view(*q.shape[:-1])), \
        q_simulated.view(q.shape)


def mqa_logits_tolerances(fmt: str):
    return (1e-3, 5e-6) if fmt == 'fp8' else (0.02, 3e-5)


def ref_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                   cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor):
    seq_len_kv = kv.shape[0]

    seq_len = q.shape[0]
    q = q.float()
    k = kv.float()
    w = weights.transpose(0, 1).contiguous()       # [num_heads, seq_len]

    # Chunk along KV so the temporary score tensor stays bounded
    kv_chunk = max(1, (256 * 1024 * 1024) // max(1, seq_len * q.shape[1] * 4))   # ~cap score chunk bytes
    positions = torch.arange(0, seq_len_kv, device='cuda')
    logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float, device='cuda')
    cost = torch.zeros((), dtype=torch.long, device='cuda')
    for n0 in range(0, seq_len_kv, kv_chunk):
        n1 = min(n0 + kv_chunk, seq_len_kv)
        score = torch.einsum('mhd,nd->hmn', q, k[n0:n1])           # [H, M, chunk]
        chunk_logits = torch.einsum('hmn,hm->mn', score.relu(), w)  # sum over heads -> [M, chunk]
        cols = positions[n0:n1]
        mask = (cols[None, :] >= cu_seqlen_ks[:, None]) & (cols[None, :] < cu_seqlen_ke[:, None])
        logits[:, n0:n1] = chunk_logits.masked_fill(~mask, float('-inf'))
        cost += mask.sum()

    return logits, cost


def test_mqa_logits():
    # Helper functions
    def generate_ks_ke_tests(seq_len: int, seq_len_kv: int, disable_cp: bool):
        if disable_cp:
            ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
            ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)
            return ks, ke
        assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
        chunk_size = seq_len // 2
        cp_size = seq_len_kv // seq_len
        # Select an arbitrary CP rank
        cp_id = cp_size // 3
        ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        ke = torch.zeros(seq_len, dtype=torch.int,  device='cuda')
        for i in range(chunk_size):
            ke[i] = cp_id * chunk_size + i
            ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
        return ks, ke

    def enumerate_mqa_logits():
        for fmt, dtype in mqa_logits_formats():
            heads, head_dims = mqa_logits_heads(fmt == 'mxfp4')
            for seq_len in (2048, 8192):
                for seq_len_kv in (8192, 65536):
                    for num_heads in heads:
                        for head_dim in head_dims:
                            for disable_cp in (False, True):
                                if not disable_cp and (seq_len_kv % seq_len != 0 or seq_len % 2 != 0):
                                    continue
                                yield fmt, dtype, seq_len, seq_len_kv, num_heads, head_dim, disable_cp

    print('Testing MQA Logits:')

    for fmt, dtype, seq_len, seq_len_kv, num_heads, head_dim, disable_cp in sample_mqa_cases('prefill', list(enumerate_mqa_logits())):
        # Generate random inputs
        q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(seq_len, num_heads, device='cuda', dtype=dtype)
        kernel_weights = to_mqa_weights(weights)
        ks, ke = generate_ks_ke_tests(seq_len, seq_len_kv, disable_cp)

        # Calculate reference logits
        ref_logits, ref_cost = ref_mqa_logits(q, kv, kernel_weights.float(), ks, ke)

        # Quantize Q and KV
        q_in, q_simulated = quantize_mqa_q(q, fmt)
        if fmt == 'fp8':
            kv_in = per_custom_dims_cast_to_fp8(kv, (0, ), False)
            kv_simulated = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)
        else:
            kv_in, kv_simulated = quantize_mqa_q(kv, fmt)

        # Calculate reference logits
        simulated_logits, _ = ref_mqa_logits(q_simulated, kv_simulated, kernel_weights.float(), ks, ke)

        # Prepare kwargs
        max_seqlen_k = (ke - ks).max().item()
        kernel_kwargs = dict(
            q=q_in, kv=kv_in, weights=kernel_weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
            max_seqlen_k=max_seqlen_k,
        )

        # Run kernel
        logits = deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs)
        assert logits.dtype == dtype

        self_mask = torch.arange(logits.size(1), device='cuda')[None, :] < (ke - ks)[:, None]
        masked_logits = logits.masked_fill(~self_mask, 0)
        for _ in range(20):
            logits_again = deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs).masked_fill(~self_mask, 0)
            assert_bitwise_equal(logits_again, masked_logits, 'mqa logits self-consistency')

        workspace = None
        if get_arch_major() == 10:
            workspace = deep_gemm.get_mqa_logits_metadata(ks, ke, seq_len_kv, num_heads)
            scheduled_logits = deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs, schedule_meta=workspace)
            scheduled_logits = scheduled_logits.masked_fill(~self_mask, 0)
            assert_bitwise_equal(scheduled_logits, masked_logits, 'mqa logits scheduled path')

        # Expand the compressed result for comparison with the dense reference.
        assert logits.size() == (seq_len, max_seqlen_k)
        tmp = torch.full((seq_len, seq_len_kv), float('-inf'), device='cuda')
        for i in range(seq_len):
            tmp[i, ks[i] : ke[i]] = logits[i, : ke[i] - ks[i]]
        logits = tmp

        # Validation
        ref_neginf_mask = (ref_logits == float('-inf'))
        neginf_mask = (logits == float('-inf'))
        assert torch.equal(neginf_mask, ref_neginf_mask)

        ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
        simulated_logits = simulated_logits.masked_fill(ref_neginf_mask, 0)
        logits = logits.masked_fill(ref_neginf_mask, 0)
        diff = calc_diff(logits, ref_logits)
        simulated_diff = calc_diff(logits, simulated_logits)
        diff_tol, simulated_diff_tol = mqa_logits_tolerances(fmt)
        assert diff < diff_tol, f"Diff: {diff}"
        assert simulated_diff < simulated_diff_tol, f"Simulated Diff: {simulated_diff}"

        # Profiling
        tflops = 2 * ref_cost * num_heads * head_dim / 1e12
        t = bench_kineto(lambda: deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs), 'mqa_logits')
        t_scheduled = t_build = 0
        if workspace is not None:
            t_scheduled = bench_kineto(lambda: deep_gemm.fp8_fp4_mqa_logits(
                **kernel_kwargs, schedule_meta=workspace), 'mqa_logits')
            t_build = bench_kineto(lambda: deep_gemm.get_mqa_logits_metadata(
                ks, ke, seq_len_kv, num_heads), 'mqa_logits_metadata')
        reduce_relus = ref_cost * num_heads
        relu_per_sm_cycle = reduce_relus / (t * deep_gemm.get_num_sms() * 1.9 * 1e9)
        print(f' > Fmt={fmt:5}, DType={dtype_tag(dtype):4}, '
              f'SQ={seq_len:4}, SK={seq_len_kv:5}, H={num_heads:2}, D={head_dim:3}, CP={0 if disable_cp else 1}: '
              f'{tflops / t:4.0f} TFLOPS, {t * 1e6:4.0f} us '
              f'(scheduled {t_scheduled * 1e6:4.0f} us, build {t_build * 1e6:4.1f} us), '
              f'{(count_bytes(q_in, kv_in, kernel_weights, ks, ke) + ref_cost * dtype.itemsize) / t / 1e9:4.0f} GB/s, '
              f'{relu_per_sm_cycle:4.1f} relu/cyc/SM')
    print()


def ref_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor,
                         weights: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor,
                         max_model_len: int):
    # `q` is `[batch_size, next_n, num_heads, dim]`, `context_lens` `[batch_size, next_n]`: token `j` of query `i`
    # attends to `[0, context_lens[i, j])` of the query's block table
    batch_size, next_n, num_heads, dim = q.size()
    _, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    for i in range(batch_size):
        q_context_lens = context_lens[i]
        context_len = q_context_lens.max().item()
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()

        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]                 # [num_blocks, block_size, kv_heads, dim]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)    # [kv_heads, dim, total_tokens]
        qx = q[i].transpose(0, 1)                       # q[i]: [next_n, num_heads, dim] -> [num_heads, next_n, dim]
        s = torch.matmul(qx, kx).to(logits.dtype)       # [num_heads, next_n, dim] @ [1, dim, total_tokens] -> [num_heads, next_n, total_tokens]

        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = k_offsets[None, :] < q_context_lens[:, None]     # [next_n, total_tokens]
        s = torch.where(mask[None, :, :], s, float('-inf'))
        s = torch.relu(s) * weight_slice[..., None]             # weight_slice: [num_heads, next_n] -> [num_heads, next_n, 1]
        s = s.sum(dim=0)                                        # [next_n, total_tokens]
        logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(mask, s, float('-inf'))

    return logits


def kv_cache_cast_to_mxfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1 and head_dim in (64, 128)
    x_scaled, sf = per_token_cast_to_fp4(
        x.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    x_cast_back = cast_back_from_fp4(
        x_scaled, sf, gran_k=32, use_packed_ue8m0=True).view(num_blocks, block_size, 1, head_dim)

    x_fp4 = torch.empty((num_blocks, block_size * (head_dim // 2 + 4)), device=x.device, dtype=torch.uint8)
    x_fp4[:, :block_size * head_dim // 2] = x_scaled.view(num_blocks, block_size * head_dim // 2).view(torch.uint8)
    x_fp4[:, block_size * head_dim // 2:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4), x_cast_back.to(x.dtype)


def kv_cache_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # SM90 fused layout: per-token E4M3 values followed by one float scale per token
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_cast_back = x_scaled.float() * sf

    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[:, :block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_cast_back.to(x.dtype)


def kv_cache_cast_to_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1 and head_dim in (32, 64, 128)
    x_scaled, sf = per_token_cast_to_fp8(
        x.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    x_cast_back = cast_back_from_fp8(
        x_scaled, sf, gran_k=32, use_packed_ue8m0=True).view(num_blocks, block_size, 1, head_dim)

    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[:, :block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_cast_back.to(x.dtype)


def test_paged_mqa_logits():
    # SM100 flattens varlen requests into `[num_q_tokens, 1]` rows tied together by `indices`; SM90 takes
    # `[batch_size, next_n]` queries. Both give every row its own context length
    is_varlen = get_arch_major() == 10

    def enumerate_paged_mqa_logits():
        max_kv_pool_tokens = 32 * 1024 * 1024
        max_q_tokens = 16 * 1024
        for fmt, dtype in mqa_logits_formats():
            heads, head_dims = mqa_logits_heads(fmt == 'mxfp4')
            for block_kv in ((64, 128, 32) if is_varlen else (64, )):
                for num_sequences in (64, 256, ):
                    for max_tokens_per_batch in ((6, 8) if is_varlen else (1, 2)):
                        for num_heads in heads:
                            for head_dim in head_dims:
                                for avg_kv in (8192, 65536, 256 * 1024):
                                    if num_sequences * avg_kv > max_kv_pool_tokens:
                                        continue
                                    if num_sequences * max_tokens_per_batch > max_q_tokens:
                                        continue
                                    yield fmt, dtype, block_kv, num_sequences, max_tokens_per_batch, num_heads, head_dim, avg_kv

    print('Testing Paged MQA Logits:')

    for fmt, dtype, block_kv, num_sequences, max_tokens_per_batch, num_heads, head_dim, avg_kv in sample_mqa_cases('paged', list(enumerate_paged_mqa_logits())):
        # Query layout: varlen rows `(num_q_tokens, 1)` or fixed `(num_sequences, next_n)`
        if is_varlen:
            tokens_per_seq = torch.randint(1, max_tokens_per_batch + 1, (num_sequences,), device='cuda', dtype=torch.int)
            indices = torch.arange(num_sequences, device='cuda', dtype=torch.int).repeat_interleave(tokens_per_seq)
            batch_size, next_n = tokens_per_seq.sum().item(), 1
        else:
            indices = None
            batch_size, next_n = num_sequences, max_tokens_per_batch
        num_q_tokens = batch_size * next_n

        # Generate random inputs
        q = torch.randn((batch_size, next_n, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
        weights = torch.randn((num_q_tokens, num_heads), device='cuda', dtype=dtype)
        kernel_weights = to_mqa_weights(weights)
        context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (num_sequences,), device='cuda', dtype=torch.int)

        # Per-token context lengths: the tokens of a sequence see growing prefixes, the last one the full length
        if is_varlen:
            offsets_within_seq = torch.cat([torch.arange(n.item(), device='cuda', dtype=torch.int) for n in tokens_per_seq])
            context_lens_nextn = (context_lens.repeat_interleave(tokens_per_seq) + offsets_within_seq).view(-1, 1)
            max_ctx_len_per_seq = context_lens + (tokens_per_seq - 1)
        else:
            context_lens_nextn = ((context_lens.unsqueeze(1) + 1) * torch.rand(batch_size, next_n, device='cuda')).int()
            context_lens_nextn[:, -1] = context_lens
            max_ctx_len_per_seq = context_lens

        # Assign block tables (per-sequence, sized by the largest ctx_len within the sequence)
        seq_sum_lens = context_lens.sum().item()
        num_blocks_per_query = ceil_div(max_ctx_len_per_seq, block_kv)
        max_model_len = num_blocks_per_query.max().item() * block_kv
        num_total_blocks = num_blocks_per_query.sum().item()
        kv_cache = torch.randn((num_total_blocks, block_kv, 1, head_dim), device='cuda', dtype=torch.bfloat16)
        block_table = torch.zeros((num_sequences, num_blocks_per_query.max().item()), device='cuda', dtype=torch.int)
        block_idx_pool = torch.randperm(num_total_blocks, device='cuda', dtype=torch.int)
        offset = 0
        for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
            block_table[i, :num_blocks] = block_idx_pool[offset : offset + num_blocks]
            offset += num_blocks
        if is_varlen:
            block_table = block_table.repeat_interleave(tokens_per_seq, dim=0)

        # Calculate reference logits
        ref_logits = ref_paged_mqa_logits(q, kv_cache, kernel_weights.float(), context_lens_nextn, block_table, max_model_len)
        q_weight_bytes = count_bytes(q, kernel_weights)

        # Quantize Q and KV cache
        kv_cache_cast = {'fp8': kv_cache_cast_to_fp8, 'mxfp4': kv_cache_cast_to_mxfp4, 'mxfp8': kv_cache_cast_to_mxfp8}[fmt]
        q_in, q_simulated = quantize_mqa_q(q, fmt)
        kv_in, kv_simulated = kv_cache_cast(kv_cache)
        del q, kv_cache

        # Calculate simulated reference logits
        simulated_logits = ref_paged_mqa_logits(q_simulated, kv_simulated, kernel_weights.float(), context_lens_nextn, block_table, max_model_len)
        positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(num_q_tokens, -1)
        ref_neginf_mask = ~(positions < context_lens_nextn.view(-1, 1))

        # Run Kernel
        assert block_table.min().item() >= 0
        assert block_table.max().item() < num_total_blocks
        assert context_lens_nextn.max().item() <= max_model_len
        metadata_kwargs = dict(
            context_lens=context_lens_nextn, block_kv=block_kv,
            num_sms=deep_gemm.get_num_sms(), indices=indices,
        )
        kernel_kwargs = dict(
            q=q_in, kv_cache=kv_in, weights=kernel_weights,
            context_lens=context_lens_nextn, block_table=block_table,
            schedule_meta=deep_gemm.get_paged_mqa_logits_metadata(**metadata_kwargs),
            max_context_len=max_model_len,
            indices=indices,
        )
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kernel_kwargs)

        self_mask = ~ref_neginf_mask
        masked_logits = logits.masked_fill(~self_mask, 0)
        for _ in range(20):
            logits_again = deep_gemm.fp8_fp4_paged_mqa_logits(**kernel_kwargs).masked_fill(~self_mask, 0)
            assert_bitwise_equal(logits_again, masked_logits, 'paged mqa logits self-consistency')

        # Validation
        assert logits.dtype == dtype
        logits = logits.to(torch.float)

        logits_masked = logits.masked_fill(ref_neginf_mask, 0)
        ref_masked = ref_logits.masked_fill(ref_neginf_mask, 0)
        simulated_masked = simulated_logits.masked_fill(ref_neginf_mask, 0)
        diff = calc_diff(logits_masked, ref_masked)
        simulated_diff = calc_diff(logits_masked, simulated_masked)
        diff_tol, simulated_diff_tol = mqa_logits_tolerances(fmt)
        assert diff < diff_tol, f"Diff: {diff}"
        assert simulated_diff < simulated_diff_tol, f"Simulated Diff: {simulated_diff}"

        # Profiling
        sum_lens = context_lens_nextn.sum().item()
        tflops_calc = 2 * sum_lens * num_heads * head_dim / 1e12
        kv_bytes_per_token = head_dim / (2 if fmt == 'mxfp4' else 1) + 4
        # KV is read once per sequence; per-token sum_lens overcounts it.
        total_bytes = q_weight_bytes + seq_sum_lens * kv_bytes_per_token + (sum_lens * dtype.itemsize)

        metadata_t = bench_kineto(
            lambda: deep_gemm.get_paged_mqa_logits_metadata(**metadata_kwargs),
            'paged_mqa_logits_metadata',
        )
        t = bench_kineto(lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**kernel_kwargs), 'paged_mqa_logits')
        reduce_relus = sum_lens * num_heads
        relu_per_sm_cycle = reduce_relus / (t * deep_gemm.get_num_sms() * 1.9 * 1e9)
        tokens_desc = f'MaxTPR={max_tokens_per_batch:2}' if is_varlen else f'NextN={next_n:2}'
        print(f' > Fmt={fmt:5}, DType={dtype_tag(dtype):4}, '
              f'PAGE_KV={block_kv:3}, BSZ={num_sequences:4}, {tokens_desc}, H={num_heads:2}, D={head_dim:3}, L={avg_kv:6}: '
              f'{tflops_calc / t:4.0f} TFLOPS, {t * 1e6:4.0f} us, Metadata={metadata_t * 1e6:4.0f} us, '
              f'{total_bytes / t / 1e9:4.0f} GB/s, {relu_per_sm_cycle:4.1f} relu/cyc/SM')

        del metadata_kwargs, kernel_kwargs, logits, ref_neginf_mask, positions
        del q_in, q_simulated, kv_in, kv_simulated, weights, kernel_weights, context_lens, context_lens_nextn, block_table
        torch.cuda.empty_cache()
    print()


def make_sparse_kv_block_indices(context_lens: List[int], request_indices: List[int] | None,
                                 sparse_block_kv: int, num_max_sparse_blocks: int,
                                 seed: int, context_starts: List[int] | None = None) -> Tuple[torch.Tensor, List[int]]:
    rng = random.Random(seed)
    request_indices = [0] * len(context_lens) if request_indices is None else request_indices
    context_starts = [0] * len(context_lens) if context_starts is None else context_starts
    indices, num_blocks_per_q = [], []
    previous_blocks, previous_request_idx = None, None
    for context_start, context_len, request_idx in zip(context_starts, context_lens, request_indices):
        first_block = context_start // sparse_block_kv
        num_available_blocks = ceil_div(max(0, context_len - context_start), sparse_block_kv)
        block_end = first_block + num_available_blocks
        num_sparse_blocks = min(num_available_blocks, num_max_sparse_blocks)
        if num_sparse_blocks == num_available_blocks:
            blocks = list(range(first_block, block_end))
        elif request_idx != previous_request_idx:
            blocks = rng.sample(range(first_block, block_end), num_sparse_blocks)
        else:
            previous = [block_idx for block_idx in previous_blocks if first_block <= block_idx < block_end]
            retained = rng.sample(previous, min(round(num_sparse_blocks * 0.8), len(previous)))
            retained_set = set(retained)
            replacements = set()
            while len(retained) + len(replacements) < num_sparse_blocks:
                block_idx = rng.randrange(first_block, block_end)
                if block_idx not in retained_set:
                    replacements.add(block_idx)
            blocks = retained + list(replacements)
        blocks.sort()
        indices.append(blocks + [blocks[-1] if blocks else 0] * (num_max_sparse_blocks - num_sparse_blocks))
        num_blocks_per_q.append(num_sparse_blocks)
        previous_blocks, previous_request_idx = blocks, request_idx
    return torch.tensor(indices, device='cuda', dtype=torch.int32), num_blocks_per_q


@test_filter(lambda: get_arch_major() == 10)
def test_sparse_mqa_logits() -> None:
    head_dim, page_kv = 128, 64

    def enumerate_sparse_mqa_logits():
        avg_kv_lens = (4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024)
        num_sms = deep_gemm.get_num_sms()
        for fmt in ('mxfp4', 'mxfp8'):
            # Contiguous KV
            for num_max_sparse_blocks in (2048, 1024, 512):
                for num_q_tokens in (8192, ):
                    for avg_kv_len in avg_kv_lens:
                        for use_unaligned_ks in (False, True):
                            yield fmt, False, num_q_tokens, avg_kv_len, 8, num_max_sparse_blocks, use_unaligned_ks

            # Paged varlen KV
            for num_max_sparse_blocks in (2048, 1024, 512):
                for num_q_tokens in (512, ):
                    for avg_kv_len in avg_kv_lens:
                        yield fmt, True, num_q_tokens, avg_kv_len, 8, num_max_sparse_blocks, False

            # Small Q counts and both sparse block sizes.
            split_kv = 640 if fmt == 'mxfp4' else 512
            for sparse_block_kv in (8, 16):
                for use_unaligned_ks in (False, True):
                    yield fmt, False, 9, split_kv, sparse_block_kv, 128, use_unaligned_ks

        # Contiguous metadata edge cases use MXFP4
        for case in (
            (1, 1, 16, 4), (2, 639, 16, 1024), (2, 0, 16, 4),
            (3, 640, 16, 1024), (5, 641, 16, 1024),
            (num_sms - 1, 16 * 1024 + 3, 16, 1024),
            (num_sms, 16 * 1024 + 3, 16, 1024),
            (num_sms + 1, 16 * 1024 + 3, 16, 1024),
            (2 * num_sms + 1, 64 * 1024 + 7, 16, 4096),
        ):
            for use_unaligned_ks in (False, True):
                yield 'mxfp4', False, *case, use_unaligned_ks

        # Paged metadata edge cases use MXFP4
        for case in (
            (True, 1, 64, 16, 8), (True, 3, 64, 16, 8),
            (True, num_sms - 1, 16 * 1024 + 3, 16, 1024),
            (True, num_sms + 1, 16 * 1024 + 3, 16, 1024),
            (True, 2 * num_sms + 1, 16 * 1024 + 3, 16, 1024),
            (True, 3, 1024 * 1024, 8, 2048),
        ):
            yield 'mxfp4', *case, False
        for use_unaligned_ks in (False, True):
            yield 'mxfp8', False, 3, 640, 16, 1024, use_unaligned_ks
        yield 'mxfp8', True, 3, 64, 16, 8, False

    print('Testing MXFP4/MXFP8 Sparse MQA Logits:')
    torch.manual_seed(0)
    cases = sample_mqa_cases('sparse', list(enumerate_sparse_mqa_logits()))
    cases = [(num_heads, *case) for case in cases for num_heads in (8, 12, 20, 32)]
    aligned_sparse_times = {}
    for num_heads, fmt, is_paged, num_q_tokens, avg_kv_len, sparse_block_kv, num_max_sparse_blocks, use_unaligned_ks in cases:
        is_mxfp4 = fmt == 'mxfp4'
        cast_fwd = per_token_cast_to_fp4 if is_mxfp4 else per_token_cast_to_fp8
        kv_cache_cast = kv_cache_cast_to_mxfp4 if is_mxfp4 else kv_cache_cast_to_mxfp8
        elem_dim = head_dim // 2 if is_mxfp4 else head_dim
        rng = random.Random(num_q_tokens * 1000000 + avg_kv_len + sparse_block_kv)
        request_sizes = []
        if is_paged:
            remaining_q_tokens = num_q_tokens
            while remaining_q_tokens > 0:
                request_size = min(rng.randint(2, 6), remaining_q_tokens)
                request_sizes.append(request_size)
                remaining_q_tokens -= request_size
        else:
            num_requests = min(rng.randint(2, 4), num_q_tokens)
            request_ends = sorted(rng.sample(range(1, num_q_tokens), num_requests - 1)) + [num_q_tokens]
            request_sizes = [request_end - request_begin
                             for request_begin, request_end in zip([0] + request_ends, request_ends)]
        request_indices = [request_idx for request_idx, request_size in enumerate(request_sizes)
                           for _ in range(request_size)]
        if is_paged:
            context_starts = [0] * num_q_tokens
            batch_size = len(request_sizes)
            request_context_lens = [rng.randint(int(0.7 * avg_kv_len) // sparse_block_kv,
                                                int(1.3 * avg_kv_len) // sparse_block_kv) * sparse_block_kv
                                    for _ in range(batch_size)]
            context_lens = [context_len + q_offset
                            for context_len, request_size in zip(request_context_lens, request_sizes)
                            for q_offset in range(request_size)]
        else:
            aligned_starts, unaligned_starts, context_lengths = [], [], []
            aligned_kv_end = unaligned_kv_end = 0
            for request_size in request_sizes:
                aligned_start = ceil_div(aligned_kv_end, sparse_block_kv) * sparse_block_kv
                unaligned_start = ceil_div(unaligned_kv_end, sparse_block_kv) * sparse_block_kv + rng.randrange(1, sparse_block_kv)
                aligned_starts.extend([aligned_start] * request_size)
                unaligned_starts.extend([unaligned_start] * request_size)
                context_lengths.extend(avg_kv_len + q_offset for q_offset in range(request_size))
                aligned_kv_end = aligned_start + context_lengths[-1]
                unaligned_kv_end = unaligned_start + context_lengths[-1]
            assert all(start % sparse_block_kv == 0 for start in aligned_starts)
            assert all(start % sparse_block_kv != 0 for start in unaligned_starts)
            context_starts = unaligned_starts if use_unaligned_ks else aligned_starts
            context_lens = [start + length for start, length in zip(context_starts, context_lengths)]
            num_kv_tokens = max(1, aligned_kv_end, unaligned_kv_end)

        q_shape = (num_q_tokens, 1, num_heads) if is_paged else (num_q_tokens, num_heads)
        q_fp, q_sf = cast_fwd(
            torch.randn((*q_shape, head_dim), device='cuda', dtype=torch.bfloat16).view(-1, head_dim),
            use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q = q_fp.view(*q_shape, elem_dim), q_sf.view(*q_shape)
        weights = to_mqa_weights(torch.randn((num_q_tokens, num_heads), device='cuda', dtype=torch.bfloat16))
        sparse_indices, num_sparse_blocks = make_sparse_kv_block_indices(
            context_lens, request_indices, sparse_block_kv, num_max_sparse_blocks, avg_kv_len, context_starts)

        if is_paged:
            max_context_lens = [context_len + request_size - 1
                                for context_len, request_size in zip(request_context_lens, request_sizes)]
            num_pages_per_request = [ceil_div(context_len, page_kv) for context_len in max_context_lens]
            max_num_pages, num_pages = max(num_pages_per_request), sum(num_pages_per_request)
            page_bytes = page_kv * (elem_dim + 4)
            page_stride_bytes = ceil_div(page_bytes, 512) * 512
            kv_storage = torch.empty((num_pages, page_stride_bytes), device='cuda', dtype=torch.uint8)
            kv_cache = kv_storage.as_strided((num_pages, page_kv, 1, elem_dim + 4),
                                             (page_stride_bytes, elem_dim + 4, elem_dim + 4, 1))
            for page_begin in range(0, num_pages, 16 * 1024):
                num_pages_to_copy = min(16 * 1024, num_pages - page_begin)
                kv_pages, kv_pages_reference = kv_cache_cast(torch.randn(
                    (num_pages_to_copy, page_kv, 1, head_dim), device='cuda', dtype=torch.bfloat16))
                kv_cache[page_begin:page_begin + num_pages_to_copy].copy_(kv_pages)
                del kv_pages, kv_pages_reference

            indices = torch.tensor(request_indices, device='cuda', dtype=torch.int32)
            context_lens_tensor = torch.tensor(context_lens, device='cuda', dtype=torch.int32)
            page_pool = torch.randperm(num_pages, device='cuda', dtype=torch.int32)
            request_block_table = torch.zeros((batch_size, max_num_pages), device='cuda', dtype=torch.int32)
            page_begin = 0
            for request_idx, num_request_pages in enumerate(num_pages_per_request):
                page_end = page_begin + num_request_pages
                request_block_table[request_idx, :num_request_pages] = page_pool[page_begin:page_end]
                page_begin = page_end
            block_table = request_block_table[indices.long()].contiguous()
            metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
                context_lens_tensor, block_table, indices, page_kv, sparse_indices, q[0].dtype, sparse_block_kv)
            sparse_kwargs = dict(q=q, kv_cache=kv_cache, weights=weights, metadata=metadata,
                                 num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)
            context_lens_2d = context_lens_tensor.view(-1, 1)
            full_kwargs = dict(
                q=q, kv_cache=kv_cache, weights=weights, context_lens=context_lens_2d, block_table=block_table,
                schedule_meta=deep_gemm.get_paged_mqa_logits_metadata(
                    context_lens_2d, page_kv, deep_gemm.get_num_sms(), indices),
                max_context_len=max(context_lens), indices=indices)
            run_sparse = lambda: deep_gemm.fp8_fp4_paged_sparse_mqa_logits(**sparse_kwargs)
            run_full = lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**full_kwargs)
            sparse_kernel_name, full_kernel_name = 'sm100_paged_sparse_mqa_logits', 'paged_mqa_logits'
            case = f'Paged BSZ={batch_size:3}, SQ={num_q_tokens:4}'
        else:
            kv_fp, kv_sf = cast_fwd(
                torch.randn((num_kv_tokens, head_dim), device='cuda', dtype=torch.bfloat16),
                use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            kv = kv_fp, kv_sf.view(num_kv_tokens)
            starts = torch.tensor(context_starts, device='cuda', dtype=torch.int32)
            ends = torch.tensor(context_lens, device='cuda', dtype=torch.int32)
            metadata_kwargs = dict(
                cu_seq_len_k_start=starts, cu_seq_len_k_end=ends, num_kv_tokens=num_kv_tokens,
                sparse_kv_block_indices=sparse_indices, qk_dtype=q[0].dtype, sparse_block_kv=sparse_block_kv)
            if use_unaligned_ks:
                metadata_kwargs['use_unaligned_ks'] = True
            metadata = deep_gemm.get_sparse_mqa_logits_metadata(**metadata_kwargs)
            sparse_kwargs = dict(q=q, kv=kv, weights=weights, metadata=metadata,
                                 num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)
            if use_unaligned_ks:
                sparse_kwargs['use_unaligned_ks'] = True
            full_kwargs = dict(q=q, kv=kv, weights=weights, cu_seq_len_k_start=starts, cu_seq_len_k_end=ends,
                               max_seqlen_k=num_kv_tokens)
            run_sparse = lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(**sparse_kwargs)
            run_full = lambda: deep_gemm.fp8_fp4_mqa_logits(**full_kwargs)
            sparse_kernel_name, full_kernel_name = 'sm100_sparse_mqa_logits', 'mqa_logits'
            case = f'Contiguous SQ={num_q_tokens:4}, UnalignedKS={int(use_unaligned_ks)}'

        sparse_logits, full_logits = run_sparse(), run_full()
        sparse_output_bytes = count_bytes(sparse_logits)
        kv_offsets = torch.arange(sparse_block_kv, device='cuda', dtype=torch.int32)
        context_starts_tensor = torch.tensor(context_starts, device='cuda', dtype=torch.int32)
        block_offsets = context_starts_tensor % sparse_block_kv
        token_indices = (sparse_indices.unsqueeze(-1) * sparse_block_kv
                         + block_offsets[:, None, None] + kv_offsets).flatten(1).long()
        num_sparse_blocks = torch.tensor(num_sparse_blocks, device='cuda')
        num_sparse_tokens = num_sparse_blocks * sparse_block_kv
        valid_mask = torch.arange(token_indices.size(1), device='cuda')[None, :] < num_sparse_tokens[:, None]
        valid_mask &= token_indices >= context_starts_tensor[:, None]
        valid_mask &= token_indices < torch.tensor(context_lens, device='cuda')[:, None]
        sparse_logits = sparse_logits[valid_mask]
        full_token_indices = token_indices - context_starts_tensor[:, None]
        full_logits = full_logits.gather(
            1, full_token_indices.clamp(min=0, max=full_logits.size(1) - 1))[valid_mask]
        assert_bitwise_equal(sparse_logits, full_logits, 'sparse MQA logits')

        for _ in range(30):
            assert_bitwise_equal(run_sparse()[valid_mask], sparse_logits, 'sparse MQA logits self-consistency')

        sparse_t = bench_kineto(run_sparse, sparse_kernel_name)
        full_t = bench_kineto(run_full, full_kernel_name)
        comparison = ''
        if not is_paged:
            key = num_heads, fmt, num_q_tokens, avg_kv_len, sparse_block_kv, num_max_sparse_blocks
            if use_unaligned_ks and key in aligned_sparse_times:
                comparison = f', unaligned/aligned {sparse_t / aligned_sparse_times[key]:.2f}x'
            elif not use_unaligned_ks:
                aligned_sparse_times[key] = sparse_t
        valid_block_mask = torch.arange(num_max_sparse_blocks, device='cuda')[None, :] < num_sparse_blocks[:, None]
        if is_paged:
            request_indices_2d = indices[:, None].expand_as(sparse_indices)
            block_keys = torch.stack((request_indices_2d[valid_block_mask], sparse_indices[valid_block_mask]), dim=-1)
            num_union_blocks = torch.unique(block_keys, dim=0).size(0)
        else:
            block_starts = sparse_indices * sparse_block_kv + block_offsets[:, None]
            num_union_blocks = block_starts[valid_block_mask].unique().numel()
        num_sum_blocks = num_sparse_blocks.sum().item()
        kv_bytes_per_block = sparse_block_kv * (elem_dim + 4)
        total_bytes = count_bytes(q, weights, metadata) + sparse_output_bytes + num_union_blocks * kv_bytes_per_block
        tflops = 2 * num_sum_blocks * sparse_block_kv * num_heads * head_dim / 1e12
        reduce_relus = num_sum_blocks * sparse_block_kv * num_heads
        relu_per_sm_cycle = reduce_relus / (sparse_t * deep_gemm.get_num_sms() * 1.9 * 1e9)
        print(f' > Fmt={fmt:5}, H={num_heads:2}, {case}, KV={avg_kv_len:7}, SPARSE_BLOCK_KV={sparse_block_kv:2}, '
              f'MAX_BLOCKS={num_max_sparse_blocks:4}: sparse {sparse_t * 1e6:5.1f} us, '
              f'{tflops / sparse_t:4.0f} TFLOPS, '
              f'{total_bytes / sparse_t / 1e9:4.0f} GB/s, '
              f'{relu_per_sm_cycle:4.1f} relu/cyc/SM ',
              f'(full {full_t * 1e6:6.1f} us, {full_t / sparse_t:5.2f}x{comparison})')
        torch.cuda.empty_cache()
    print()




if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_mqa_logits()
    test_paged_mqa_logits()
    test_sparse_mqa_logits()
