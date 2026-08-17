"""SM90 (Hopper) MegaMoE correctness test.

Ported from upstream PR https://github.com/sgl-project/DeepGEMM/pull/36
(`tests/test_mega_moe_hopper.py` on the old `dev` branch) and adapted to follow the structure and
test harness (`init_dist`/`dist_print`, multiprocessing-spawn driver) of `tests/test_mega_moe.py`
(the SM100 FP4 MegaMoE test in this repo), and to call the SM90 API surface added in
`deep_gemm.mega` (`Sm90SymmBuffer`, `get_symm_buffer_for_sm90_mega_moe`,
`transform_weights_for_mega_moe_sm90`, `fp8_mega_moe_sm90`).

Unlike the SM100 path, `fp8_mega_moe_sm90`:
  * is FP8xFP8-only (no FP4 weights, no `bf16xbf16` variant here);
  * uses plain float scale factors at block-(128, 128)/(token, 128) granularity (no UE8M0 packing);
  * has no shared-expert support;
  * runs on a simpler pool-based scheduler (no ring buffer).

This test is skipped cleanly (prints a message and returns) on any non-Hopper (SM90) GPU.
"""

import argparse
import random
from typing import Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist
from deep_gemm.testing import calc_diff, get_arch_major


def _quantize_weights_block_128_128(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a `(num_groups, n, k)` BF16 weight tensor into FP8 e4m3 with block-(128, 128)
    float scale factors, matching the layout `fp8_mega_moe_sm90` expects for L1/L2 weights."""
    num_groups, n, k = bf16_weights.shape
    w = torch.empty((num_groups, n, k), device='cuda', dtype=torch.float8_e4m3fn)
    w_sf = torch.empty((num_groups, n // 128, k // 128), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = per_block_cast_to_fp8(bf16_weights[i], use_ue8m0=False, gran_k=128)
    return w, w_sf


def _reference_mega_moe(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
    l1_weights: torch.Tensor, l2_weights: torch.Tensor,
    num_experts_per_rank: int, activation_clamp: float,
) -> torch.Tensor:
    """Naive reference: dense routed SwiGLU MLP in FP32, evaluated per (token, topk) pair.
    NOTES: this assumes a single rank (EP=1), matching how this test invokes the kernel."""
    num_tokens, hidden = x.shape
    num_topk = topk_idx.shape[1]
    y = torch.zeros((num_tokens, hidden), dtype=torch.float32, device='cuda')
    x_f32 = x.float()
    l1_f32 = l1_weights.float()
    l2_f32 = l2_weights.float()
    for t in range(num_tokens):
        for k in range(num_topk):
            e = topk_idx[t, k].item()
            if e < 0 or e >= num_experts_per_rank:
                continue
            w = topk_weights[t, k].item()
            gate_up = x_f32[t] @ l1_f32[e].T
            intermediate_hidden = gate_up.shape[0] // 2
            gate, up = gate_up[:intermediate_hidden], gate_up[intermediate_hidden:]
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(min=-activation_clamp, max=activation_clamp)
            act = torch.nn.functional.silu(gate) * up
            out = act @ l2_f32[e].T
            y[t] += w * out
    return y.to(torch.bfloat16)


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if not torch.cuda.is_available() or get_arch_major() != 9:
        dist_print(f'Skipping SM90 MegaMoE test: requires a Hopper (SM90) GPU, '
                   f'got arch major {get_arch_major() if torch.cuda.is_available() else "N/A"}',
                   once_in_node=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens if args.num_tokens > 0 else num_max_tokens_per_rank
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    activation_clamp = args.activation_clamp
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate the SM90 symmetric buffer
    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
    )

    # Random inputs
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    # Keep only experts routed to this rank, and remap to local expert indices for the reference
    local_mask = (topk_idx // num_experts_per_rank) == rank_idx
    local_topk_idx = torch.where(local_mask, topk_idx % num_experts_per_rank, -1)

    # Quantize activations (per-token, per-128 K, float SF) and weights (block-128x128, float SF)
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)
    l1_weights_fp8, l1_weights_sf = _quantize_weights_block_128_128(l1_weights_bf16)
    l2_weights_fp8, l2_weights_sf = _quantize_weights_block_128_128(l2_weights_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_weights_fp8, l1_weights_sf), (l2_weights_fp8, l2_weights_sf))

    # Copy inputs into the symmetric buffer
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
    if num_tokens < num_max_tokens_per_rank:
        buffer.x[num_tokens:].zero_()
        buffer.topk_idx[num_tokens:].fill_(-1)
        buffer.topk_weights[num_tokens:].zero_()

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_mega_moe_sm90(
        y, transformed_l1, transformed_l2, buffer,
        activation_clamp=activation_clamp,
    )

    if num_ranks == 1:
        y_ref = _reference_mega_moe(
            x, local_topk_idx, topk_weights,
            l1_weights_bf16, l2_weights_bf16,
            num_experts_per_rank, activation_clamp)
        diff = calc_diff(y.float(), y_ref.float())
        dist_print(f' > EP {rank_idx:2}/{num_ranks} | correctness diff: {diff:.6f}', once_in_node=True)
        assert diff < 0.05, f'MegaMoE SM90 correctness check failed: diff={diff}'
    else:
        # Cross-rank combine correctness needs a distributed reference; only run the smoke check
        # (kernel executes without error / NaNs) for multi-rank configurations.
        assert torch.isfinite(y.float()).all(), 'MegaMoE SM90 produced non-finite outputs'
        dist_print(f' > EP {rank_idx:2}/{num_ranks} | smoke test passed (finite outputs)', once_in_node=True)

    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 (Hopper) MegaMoE correctness test')
    parser.add_argument('--num-processes', type=int, default=1, help='Number of processes to spawn (default: 1)')
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=512, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max if 0)')
    parser.add_argument('--hidden', type=int, default=2048, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=1024, help='Intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=2, help='Number of expert selections')
    args = parser.parse_args()

    if args.num_processes == 1:
        test(0, 1, args)
    else:
        torch.multiprocessing.spawn(
            test, args=(args.num_processes, args), nprocs=args.num_processes
        )
