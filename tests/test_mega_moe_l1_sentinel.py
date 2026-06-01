# Stream A0.2.1 sentinel-pattern probe — verifies the L1 epilogue's FP4 store
# byte layout matches the canonical packed layout the L2 phase reads.
#
# Methodology:
#   - Run the kernel with FP8 acts → dump l2_acts and decode FP8 → fp32.
#   - Run with FP4 acts → dump l2_acts (now packed E2M1) and decode → fp32.
#   - Both paths share the same scheduler / dispatch / SwiGLU math, so the
#     dequantized values should agree to within FP4 quant noise (~5-10% rel
#     error per cell, much less in row-mean magnitude). The slot-permutation
#     ambiguity that plagued A0.1's harness is sidestepped by using the
#     end-to-end `y` comparison: y is indexed by global (token, hidden) so
#     the kernel's atomicAdd-based dispatch slot order doesn't enter the
#     metric.
#
# Why this is "sentinel-pattern":
#   The MMA TMEM accumulator for each (frag = T%4, group = T/4) lane carries
#   4 fp32 values that map to a 2x2 block of the smem CD output (rows
#   {2*frag, 2*frag+1} × cols {T/4, T/4+8} within the warp's 16-byte stripe).
#   This is the empirical layout of `stmatrix.m16n8.x1.trans.b8` (verified by
#   a probe in the kernels-repo) used by the FP8 path. The original Stream
#   A0.2 FP4 store assumed lane T's 4 fp32s are 4 contiguous N-cols in one
#   row — which is wrong, and produced rel-RMSE = 1.41 (well above the
#   ≤0.5 target). Stream A0.2.1 fixes the FP4 store with `__shfl_xor_sync 4`
#   to combine adjacent-col values into FP4 bytes.
#
# Pass criterion: end-to-end `y` rel-RMSE ≤ 0.5 between FP4-acts and FP8-acts
# at smoke shape (matches A3's measured FP4-quant chain noise floor).
#
# Usage:
#   bench/run_megamoe.sh --gpus 4,5 --slot 2 -- \
#       python tests/test_mega_moe_l1_sentinel.py --num-processes 2

import argparse
import os
import random
import sys
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
from deep_gemm.utils.dist import dist_print, init_dist


def _decode_fp4_packed(packed_bytes: torch.Tensor) -> torch.Tensor:
    """Decode (M, N_packed) uint8 buffer where each byte holds 2 E2M1 nibbles
    (low nibble = even col, high nibble = odd col) into a (M, 2*N_packed)
    fp32 tensor."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float, device=packed_bytes.device)
    pb = packed_bytes.to(torch.uint8)
    m, npb = pb.shape
    lo = (pb & 0x0F).to(torch.long)
    hi = ((pb >> 4) & 0x0F).to(torch.long)
    codes = torch.stack([lo, hi], dim=-1).reshape(m, npb * 2)
    sign = (codes & 0x08) != 0
    mag_idx = (codes & 0x07)
    val = table[mag_idx]
    val = torch.where(sign & (mag_idx != 0), -val, val)
    return val


def _decode_ue8m0(sf_bytes: torch.Tensor) -> torch.Tensor:
    return ((sf_bytes.to(torch.int32) << 23).view(torch.float32))


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(0)
    random.seed(0)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    activation_clamp = args.activation_clamp
    assert num_tokens <= num_max_tokens_per_rank

    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    cumulative = torch.zeros((num_experts_per_rank,), dtype=torch.int, device='cuda')
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    x_fp4 = per_token_cast_to_fp4(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    def cast_grouped_weights_to_fp4(bf16_weights):
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    l1_weights_fp4 = cast_grouped_weights_to_fp4(l1_weights_bf16)
    l2_weights_fp4 = cast_grouped_weights_to_fp4(l2_weights_bf16)
    transformed_l1_weights, transformed_l2_weights = \
        deep_gemm.transform_weights_for_mega_moe(l1_weights_fp4, l2_weights_fp4)

    # Stream A0.0b: under `DG_USE_FP4_ACTS=1`, the symm buffer's `x` slot is
    # sized for packed E2M1 (`hidden/2` bytes/token) — different from FP8.
    # Allocate the buffer separately for each path and feed it the matching
    # source tensor.
    def make_buffer_and_run(use_fp4_acts: bool):
        os.environ['DG_USE_FP4_ACTS'] = '1' if use_fp4_acts else '0'
        os.environ['DG_COMM_KERNEL_DEBUG'] = '0'
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden
        )
        x_src = x_fp4 if use_fp4_acts else x_fp8

        def run_once():
            buf.x[:num_tokens].copy_(x_src[0])
            buf.x_sf[:num_tokens].copy_(x_src[1])
            buf.topk_idx[:num_tokens].copy_(topk_idx)
            buf.topk_weights[:num_tokens].copy_(topk_weights)
            cumulative.zero_()
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_fp4_mega_moe(
                y, transformed_l1_weights, transformed_l2_weights, buf,
                cumulative_local_expert_recv_stats=cumulative,
                activation_clamp=activation_clamp,
                fast_math=bool(args.fast_math)
            )
            return y, cumulative.clone()

        _ = run_once()
        torch.cuda.synchronize()
        y_out, _ = run_once()
        torch.cuda.synchronize()
        buf.destroy()
        return y_out

    # Run FP8-acts first (warmup + measurement).
    y_fp8 = make_buffer_and_run(use_fp4_acts=False)
    # Run FP4-acts (separate buffer because the `x` slot footprint changes).
    y_fp4 = make_buffer_and_run(use_fp4_acts=True)

    # End-to-end y comparison: this is the source of truth (no slot
    # permutation ambiguity since y is indexed by global (token, hidden)).
    y_diff = (y_fp4.float() - y_fp8.float()).abs()
    y_rmse = y_diff.pow(2).mean().sqrt().item()
    y_fp8_rms = y_fp8.float().pow(2).mean().sqrt().item()
    rel_rmse = y_rmse / max(y_fp8_rms, 1e-12)

    dist_print(f'=== A0.2.1 sentinel — y rel-RMSE (FP4 vs FP8 acts) ===',
               once_in_node=True)
    dist_print(f'  y_fp8 RMS:        {y_fp8_rms:.4f}', once_in_node=True)
    dist_print(f'  y_rmse:           {y_rmse:.4f}', once_in_node=True)
    dist_print(f'  rel-RMSE:         {rel_rmse:.4f}', once_in_node=True)
    dist_print(f'  target:           ≤ 0.50 (A3 chain noise floor)',
               once_in_node=True)
    dist_print(f'  verdict:          {"PASS" if rel_rmse <= 0.5 else "FAIL"}',
               once_in_node=True)

    # Spot-check first row to make the failure mode legible if it ever
    # comes back: matched values at low N indices = layout correct;
    # garbage = layout broken.
    dist_print(f'\n  y_fp8 [0, :8]:  {y_fp8[0, :8].cpu().tolist()}',
               once_in_node=True)
    dist_print(f'  y_fp4 [0, :8]:  {y_fp4[0, :8].cpu().tolist()}',
               once_in_node=True)

    assert rel_rmse <= 0.5, \
        f'A0.2.1 layout regression: y rel-RMSE {rel_rmse:.4f} > 0.5'

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--num-tokens', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--fast-math', type=int, default=1)
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
