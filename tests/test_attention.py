import dataclasses
import random
import torch
from typing import Tuple, List

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    ignore_env, get_arch_major,
    test_filter
)
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8, per_token_cast_to_fp4, cast_back_from_fp4

from generators import get_arch_major, generate_normal, get_ue8m0_usage, get_kernel_types, reset_seed, MajorTypeAB


def apply_skip_head_mid(d: torch.Tensor, head_splits: Tuple[int, int, int]):
    left, mid, right = head_splits
    m, n = d.shape
    assert n % (left + right) == 0
    num_heads = n // (left + right)

    # Split and insert padding tensor
    d = d.view(m, num_heads, -1)
    d_left = d[:, :, :left]
    d_right = d[:, :, -right:]

    d_mid = torch.zeros((m, num_heads, mid), dtype=d.dtype, device=d.device)
    return torch.cat([d_left, d_mid, d_right], dim=2).view(m, -1)


def test_gemm_skip_head_mid() -> None:
    print('Testing GEMM skip head mid:')
    head_splits = (128, 64, 128)

    major_a, major_b = MajorTypeAB.KMajor,  MajorTypeAB.KMajor
    out_dtype, accumulate = torch.bfloat16, False

    for kernel_type in get_kernel_types(dtype=torch.float8_e4m3fn):
        for m in (128, 4096):
            for n, k in [(32768, 512), (8192, 512)]:
                kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
                use_ue8m0 = get_ue8m0_usage(kernel_type)
                disable_ue8m0_cast = not use_ue8m0

                a, b, _, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
                d = apply_skip_head_mid(d, head_splits)
                ref_d = apply_skip_head_mid(ref_d, head_splits)

                deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast)
                diff = calc_diff(d, ref_d)
                assert diff < 0.001, f'{m=}, {n=}, {k=}, {kernel_opt}, {diff:.5f}'

                t = bench_kineto(lambda: deep_gemm.fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, disable_ue8m0_cast=disable_ue8m0_cast),
                                 'gemm_', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}): '
                      f'{t * 1e6:4.0f} us | '
                      f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                      f'{(count_bytes(a, b, d)) / 1e9 / t:4.0f} GB/s')
    print()


def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                       cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor, cost_only: bool = False):
    seq_len_kv = kv.shape[0]

    if cost_only:
        start = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
        end   = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)
        count_ones_per_row = (end - start).clamp(min=0)
        return count_ones_per_row.sum()

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = torch.arange(0, seq_len_kv, device='cuda')[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device='cuda')[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum('mhd,nd->hmn', q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float('-inf'))

    cost = mask.sum()
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
        for is_fp4 in ((True, False) if get_arch_major() == 10 else (False, )):
            for logits_dtype in (torch.float, torch.bfloat16):
                for compressed_logits, clean_logits in [(False, True), (True, False)]:
                    for seq_len in (2048, 4096):
                        for seq_len_kv in (4096, 8192):
                            for num_heads, head_dim in [(64, 128)]:
                                for disable_cp in (False, True):
                                    yield is_fp4, logits_dtype, compressed_logits, clean_logits, seq_len, seq_len_kv, num_heads, head_dim, disable_cp

    print('Testing FP8 MQA Logits:')
    for is_fp4, logits_dtype, compressed_logits, clean_logits, seq_len, seq_len_kv, num_heads, head_dim, disable_cp in enumerate_mqa_logits():
        # Generate random inputs
        q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
        ks, ke = generate_ks_ke_tests(seq_len, seq_len_kv, disable_cp)

        # Calculate reference logits
        ref_logits, ref_cost = ref_fp8_mqa_logits(q, kv, weights, ks, ke)

        # Quantize Q and KV to FP4 / FP8
        if is_fp4:
            q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            q_in = (q_fp4[0].view(seq_len, num_heads, head_dim // 2), q_fp4[1].view(seq_len, num_heads))
            q_simulated = cast_back_from_fp4(q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True).view(seq_len, num_heads, head_dim).to(torch.bfloat16)

            kv_fp4 = per_token_cast_to_fp4(kv.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            kv_in = (kv_fp4[0].view(seq_len_kv, head_dim // 2), kv_fp4[1].view(seq_len_kv))
            kv_simulated = cast_back_from_fp4(kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True).view(seq_len_kv, head_dim).to(torch.bfloat16)
        else:
            q_in = q.to(torch.float8_e4m3fn), None
            q_simulated = q_in[0].to(torch.bfloat16)
            kv_in = per_custom_dims_cast_to_fp8(kv, (0, ), False)
            kv_simulated = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)

        # Calculate reference logits
        simulated_logits, _ = ref_fp8_mqa_logits(q_simulated, kv_simulated, weights, ks, ke)

        # Prepare kwargs
        kernel_kwargs = dict(
            q=q_in, kv=kv_in, weights=weights,
            cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
            clean_logits=clean_logits, max_seqlen_k=0,
            logits_dtype=logits_dtype
        )
        if compressed_logits:
            max_seqlen_k = (ke - ks).max().item()
            kernel_kwargs['max_seqlen_k'] = max_seqlen_k

        # Run kernel
        logits = deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs)

        # Post process for compressed logits
        if compressed_logits:
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
        assert diff < 0.02 if is_fp4 else 1e-3, f"Diff: {diff}"
        assert simulated_diff < 5e-6, f"Simulated Diff: {simulated_diff}"

        # Profiling
        tflops = 2 * ref_cost * num_heads * head_dim / 1e12
        t, clean_t = bench_kineto(lambda: deep_gemm.fp8_fp4_mqa_logits(**kernel_kwargs), ('mqa_logits', 'clean_logits'))
        clean_bytes = (seq_len * seq_len_kv - ref_cost) * 4 + count_bytes(ks, ke)

        print(f' > FP4={is_fp4}, BF16={logits_dtype == torch.bfloat16}, S={seq_len:4}, SKV={seq_len_kv:6}, H={num_heads:3}, D={head_dim:3}, CP={0 if disable_cp else 1}: '
              f'{tflops / t:4.0f} TFLOPS, {t * 1e6:4.0f} us, '
              f'{(count_bytes(q_in, kv_in, weights, ks, ke) + ref_cost * 4) / t / 1e9:4.0f} GB/s', end='')
        print(f' | clean: {clean_t * 1e6:3.0f} us, {clean_bytes / clean_t / 1e9:4.0f} GB/s' if clean_logits else '')
    print()


def ref_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor,
                         weights: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor,
                         max_model_len: int, use_2d_context_lens: bool):
    batch_size, next_n, num_heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.full((next_n, ), context_len, device='cuda', dtype=torch.int32) if use_2d_context_lens \
            else torch.arange(context_len - next_n, context_len, device='cuda')
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()

        num_blocks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blocks]
        kv_slice = kv_cache[block_idxs]                 # [num_blocks, block_size, kv_heads, dim]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)    # [kv_heads, dim, total_tokens]
        qx = q[i].transpose(0, 1)                       # q[i]: [next_n, num_heads, dim] -> [num_heads, next_n, dim]
        s = torch.matmul(qx, kx).to(logits.dtype)       # [num_heads, next_n, dim] @ [1, dim, total_tokens] -> [num_heads, next_n, total_tokens]

        total_len = num_blocks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float('-inf'))     # mask shape: [1, next_n, total_tokens]
        s = torch.relu(s) * weight_slice[..., None]             # weight_slice: [num_heads, next_n] -> [num_heads, next_n, 1]
        s = s.sum(dim=0)                                        # [next_n, total_tokens]
        logits[i * next_n:(i + 1) * next_n, :total_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))

    return logits


def test_paged_mqa_logits():

    # Helper functions
    def kv_cache_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks, block_size, num_heads, head_dim = x.shape
        assert num_heads == 1
        x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
        sf = x_amax / 448.0
        x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
        x_cast_back = x_scaled.float() * sf

        x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
        x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(torch.uint8)
        x_fp8[ :, block_size * head_dim :] = sf.view(num_blocks, block_size).view(torch.uint8)
        return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_cast_back.to(x.dtype)

    def kv_cache_cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
        num_blocks, block_size, num_heads, head_dim = x.shape
        assert num_heads == 1 and head_dim == 128
        x_scaled, sf = per_token_cast_to_fp4(x.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        x_cast_back = cast_back_from_fp4(x_scaled, sf, gran_k=32, use_packed_ue8m0=True).view(num_blocks, block_size, 1, head_dim)

        x_fp4 = torch.empty((num_blocks, block_size * (head_dim // 2 + 4)), device=x.device, dtype=torch.uint8)
        x_fp4[ :, : block_size * head_dim // 2] = x_scaled.view(num_blocks, block_size * head_dim // 2).view(torch.uint8)
        x_fp4[ :, block_size * head_dim // 2 :] = sf.view(num_blocks, block_size).view(torch.uint8)
        return x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4), x_cast_back.to(x.dtype)

    def enumerate_paged_mqa_logits():
        arch_major = get_arch_major()
        for is_varlen in ((True, False) if arch_major == 10 else (False, )):
            for is_fp4 in ((True, False) if arch_major == 10 else (False, )):
                for logits_dtype in (torch.float, torch.bfloat16):
                    for block_kv in ((32, 64) if arch_major == 10 else (64, )):
                        for use_2d_context_lens, clean_logits in [(True, False)]:
                            for batch_size in (256, ):
                                for next_n in ((1, ) if is_varlen else ((1, 2, 4, 5, 6) if arch_major == 10 else (1, 2))):
                                    for max_tokens_per_batch in ((1, 4, 10) if is_varlen else (1, )):
                                        for num_heads, head_dim in [(64, 128)]:
                                            for avg_kv in (8192, 32768):
                                                yield is_varlen, is_fp4, logits_dtype, block_kv, use_2d_context_lens, clean_logits, batch_size, next_n, max_tokens_per_batch, num_heads, head_dim, avg_kv


    print('Testing FP8/FP4 Paged MQA Logits:')
    max_model_len = 111 * 1024
    num_total_blocks = max_model_len * 5

    for is_varlen, is_fp4, logits_dtype, block_kv, use_2d_context_lens, clean_logits, batch_size, next_n, max_tokens_per_batch, num_heads, head_dim, avg_kv in enumerate_paged_mqa_logits():
        # Varlen: flatten raw_batch_size sequences with variable tokens into (batch_size, 1, ...)
        raw_batch_size, raw_next_n = batch_size, next_n
        if is_varlen:
            tokens_per_seq = torch.randint(1, max_tokens_per_batch + 1, (raw_batch_size,), device='cuda', dtype=torch.int)
            indices = torch.arange(raw_batch_size, device='cuda', dtype=torch.int).repeat_interleave(tokens_per_seq)
            batch_size, next_n = tokens_per_seq.sum().item(), 1
        else:
            tokens_per_seq, indices = None, None

        # Generate random inputs
        q = torch.randn((batch_size, next_n, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
        kv_cache = torch.randn((num_total_blocks, block_kv, 1, head_dim), device='cuda', dtype=torch.bfloat16)
        weights = torch.randn((batch_size * next_n, num_heads), device='cuda', dtype=torch.float)
        context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (raw_batch_size,), device='cuda', dtype=torch.int)

        if is_varlen:
            max_ctx_len_per_seq = context_lens + (tokens_per_seq - 1)
        else:
            max_ctx_len_per_seq = context_lens

        # Assign block tables (per-sequence, sized by the largest ctx_len within the sequence)
        seq_sum_lens = context_lens.sum().item()
        num_blocks_per_query = ceil_div(max_ctx_len_per_seq, block_kv)
        block_table = torch.empty((raw_batch_size, num_blocks_per_query.max().item()), device='cuda', dtype=torch.int)
        block_idx_pool = torch.randperm(num_total_blocks, device='cuda', dtype=torch.int)
        offset = 0
        for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
            block_table[i, :num_blocks] = block_idx_pool[offset : offset + num_blocks]
            offset += num_blocks
        if is_varlen:
            context_lens = context_lens.repeat_interleave(tokens_per_seq)
            offsets_within_seq = torch.cat([
                torch.arange(n.item(), device='cuda', dtype=torch.int)
                for n in tokens_per_seq
            ])
            context_lens = context_lens + offsets_within_seq
            block_table = block_table.repeat_interleave(tokens_per_seq, dim=0)

        # Calculate reference logits
        ref_logits = ref_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, max_model_len, use_2d_context_lens)

        # Quantize Q and KV cache to FP4 / FP8
        if is_fp4:
            q_fp4 = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            q_in = (q_fp4[0].view(batch_size, next_n, num_heads, head_dim // 2), q_fp4[1].view(batch_size, next_n, num_heads))
            q_simulated = cast_back_from_fp4(q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True).view(batch_size, next_n, num_heads, head_dim).to(torch.bfloat16)
            kv_in, kv_simulated = kv_cache_cast_to_fp4(kv_cache)
        else:
            q_in = q.to(torch.float8_e4m3fn), None
            q_simulated = q_in[0].to(torch.bfloat16)
            kv_in, kv_simulated = kv_cache_cast_to_fp8(kv_cache)

        # Calculate simulated reference logits
        simulated_logits = ref_paged_mqa_logits(q_simulated, kv_simulated, weights, context_lens, block_table, max_model_len, use_2d_context_lens)

        # Prepare masks and context lengths with NextN
        positions = torch.arange(max_model_len, device='cuda').unsqueeze(0).expand(batch_size * next_n, -1)
        if use_2d_context_lens:
            if is_varlen:
                # Varlen: context_lens is already per-token (shape [total_tokens]);
                # just reshape to (total_tokens, 1) so each token keeps its own ctx_len.
                context_lens_nextn = context_lens.view(-1, 1)
            else:
                context_lens_nextn = ((context_lens.unsqueeze(1) + 1) * torch.rand(batch_size, next_n, device='cuda')).int()
                # Ensure last token matches actual length
                context_lens_nextn[:, -1] = context_lens
            ref_neginf_mask = ~(positions < context_lens_nextn.view(-1, 1))
        else:
            context_lens_nextn = context_lens
            offsets = torch.arange(batch_size * next_n, device='cuda')
            limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
            ref_neginf_mask = ~(positions <= limits)

        # Run Kernel
        kernel_kwargs = dict(
            q=q_in, kv_cache=kv_in, weights=weights,
            context_lens=context_lens_nextn, block_table=block_table,
            schedule_meta=deep_gemm.get_paged_mqa_logits_metadata(context_lens_nextn, block_kv, deep_gemm.get_num_sms(), indices=indices),
            max_context_len=max_model_len, clean_logits=clean_logits, logits_dtype=logits_dtype,
            indices=indices,
        )
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kernel_kwargs)

        # Validation
        assert logits.dtype == logits_dtype
        logits = logits.to(torch.float)

        if clean_logits:
            assert torch.equal(logits == float('-inf'), ref_neginf_mask), "Mask mismatch"

        logits_masked = logits.masked_fill(ref_neginf_mask, 0)
        ref_masked = ref_logits.masked_fill(ref_neginf_mask, 0)
        simulated_masked = simulated_logits.masked_fill(ref_neginf_mask, 0)
        diff = calc_diff(logits_masked, ref_masked)
        simulated_diff = calc_diff(logits_masked, simulated_masked)
        assert diff < 0.02 if is_fp4 else 1e-3, f"Diff: {diff}"
        assert simulated_diff < 5e-6, f"Simulated Diff: {simulated_diff}"

        # Profiling
        sum_lens = context_lens.sum().item()
        tflops_calc = 2 * sum_lens * next_n * num_heads * head_dim / 1e12
        kv_bytes_per_token = head_dim / (2 if is_fp4 else 1) + 4
        # KV is read once per sequence; for varlen sum_lens overcounts (per-token), so use seq_sum_lens
        kv_sum_lens = seq_sum_lens if is_varlen else sum_lens
        total_bytes = count_bytes(q, weights) + kv_sum_lens * kv_bytes_per_token + (sum_lens * next_n * logits_dtype.itemsize)

        t, clean_t = bench_kineto(lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**kernel_kwargs), ('paged_mqa_logits', 'clean_logits'))
        print(f' > FP4={is_fp4}, BF16={logits_dtype == torch.bfloat16}, BLOCK_KV={block_kv}, BSZ={raw_batch_size:3}, NextN={raw_next_n:1}, H={num_heads:2}, D={head_dim:2}, L={avg_kv:6}: '
              f'{tflops_calc / t:4.0f} TFLOPS, {t * 1e6:3.0f} us, {total_bytes / t / 1e9:4.0f} GB/s', end='')
        if is_varlen:
            print(f' | Varlen, MaxTPB={max_tokens_per_batch}, NumTokens={batch_size}', end='')
        print(f' | clean: {clean_t*1e6:3.0f} us' if clean_logits else '')
    print()




if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    test_gemm_skip_head_mid()
    test_mqa_logits()
    test_paged_mqa_logits()
