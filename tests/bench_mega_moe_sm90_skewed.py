"""SM90 MegaMoE bench with power-law expert load skew.

Adds --skew-alpha (Zipf exponent). The score generator biases the topk to
favor a power-law subset of experts:

    bias[i] = log( 1 / (rank[i]+1)^alpha * num_experts ) * gain
    scores  = N(0, 1) + bias

alpha=0 => uniform (matches upstream bench).
alpha=1 => Zipfian harmonic — hot experts get ~ln(N) more tokens than cold.
alpha=2 => strong skew — top expert can get 5-10x mean.

Per-rank expert permutation is randomized so hot experts spread across ranks
(matches production behavior where placement is shuffled).

Reports observed skew via num_recv/num_touched and per-expert recv distribution.
"""
import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, get_arch_major


def _quantize_grouped_fp8_block_128_128(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g, n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0
    chunk_g = 4
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    sf = torch.empty((g, n // 128, k // 128), dtype=torch.float, device=w.device)
    for start in range(0, g, chunk_g):
        end = min(start + chunk_g, g)
        w_view = w[start:end].view(end - start, n // 128, 128, k // 128, 128).float()
        sf_chunk = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4) / 448.0
        w_fp8[start:end].copy_(
            (w_view / sf_chunk.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn).view(end - start, n, k))
        sf[start:end].copy_(sf_chunk)
    return w_fp8, sf.contiguous()


def _generate_skewed_scores(num_tokens: int, num_experts: int, num_topk: int,
                             alpha: float, gain: float, generator):
    """Generate scores that produce a Zipfian topk distribution.

    Adds a per-expert bias derived from Zipfian rank to N(0,1) scores. The
    final topk[i] is therefore drawn from a softer-than-Zipf but still
    power-law-tailed distribution.
    """
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float,
                         device='cuda', generator=generator)
    if alpha > 0:
        # Zipf weights: probs[k] = 1 / (k+1)^alpha, normalized
        ranks = torch.arange(1, num_experts + 1, device='cuda', dtype=torch.float)
        probs = 1.0 / ranks.pow(alpha)
        probs = probs / probs.sum()
        # Bias scale relative to uniform expert prob
        bias = torch.log(probs * num_experts) * gain
        # Shuffle so hot experts are not concentrated at low ids (mimics
        # production where init_expert_location randomizes placement).
        perm = torch.randperm(num_experts, device='cuda', generator=generator)
        bias = bias[perm]
        scores = scores + bias.unsqueeze(0)
    return scores


def _run_one_config(args, num_tokens, num_max_tokens_per_rank,
                    hidden, intermediate_hidden,
                    num_experts, num_topk, num_ranks, rank_idx, group):
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
    )

    gen = torch.Generator(device='cuda')
    gen.manual_seed(rank_idx * 1009 + int(num_tokens))

    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda', generator=gen) * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda', generator=gen) * 0.05

    scores = _generate_skewed_scores(num_tokens, num_experts, num_topk,
                                      args.skew_alpha, args.skew_gain, gen)
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf),
    )

    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')
    use_skew_hint = args.skew_alpha > 0.0

    def run_fused():
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_w)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        old_skew_hint = os.environ.get('DG_SM90_MOE_SKEW_HINT')
        if use_skew_hint:
            os.environ['DG_SM90_MOE_SKEW_HINT'] = '1'
        try:
            deep_gemm.fp8_mega_moe(
                y, transformed_l1, transformed_l2, buffer,
                cumulative_local_expert_recv_stats=cum_stats,
                recipe=(128, 128, 128),
                activation='swiglu',
                activation_clamp=10.0,
                fast_math=True,
            )
        finally:
            if use_skew_hint:
                if old_skew_hint is None:
                    os.environ.pop('DG_SM90_MOE_SKEW_HINT', None)
                else:
                    os.environ['DG_SM90_MOE_SKEW_HINT'] = old_skew_hint
        return y

    run_fused()
    dist.barrier()
    t_fused = bench_kineto(run_fused, 'sm90_fp8_mega_moe',
                           barrier=lambda: dist.barrier(),
                           num_tests=args.num_tests,
                           suppress_kineto_output=True,
                           with_multiple_kernels=os.environ.get(
                               'DG_SM90_MOE_SPLIT_L1_L2', '1') != '0')

    # Local expert count distribution
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    local_mask = ((gathered_topk_idx >= rank_idx * num_experts_per_rank) &
                  (gathered_topk_idx < (rank_idx + 1) * num_experts_per_rank))
    gathered_local = gathered_topk_idx.clone()
    gathered_local[~local_mask] = -1
    num_recv_tokens = int(local_mask.sum().item())
    num_touched_experts = max(0, int(torch.unique(gathered_local.flatten()).numel()) - 1)

    # Per-local-expert recv count (for skew diagnostics)
    if num_recv_tokens > 0:
        per_expert_count = torch.zeros(num_experts_per_rank, dtype=torch.int64, device='cuda')
        local_ids = gathered_local[local_mask] - rank_idx * num_experts_per_rank
        per_expert_count.scatter_add_(0, local_ids.to(torch.int64),
                                       torch.ones_like(local_ids, dtype=torch.int64))
        counts = per_expert_count.cpu().tolist()
        mean_c = sum(counts) / num_experts_per_rank
        max_c = max(counts)
        min_c = min(counts)
        nonzero = sum(1 for c in counts if c > 0)
        skew_max_mean = max_c / mean_c if mean_c > 0 else 0.0
    else:
        max_c = min_c = mean_c = 0
        skew_max_mean = 0.0
        nonzero = 0

    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden +
        num_touched_experts * hidden * intermediate_hidden +
        num_recv_tokens * hidden +
        num_recv_tokens * intermediate_hidden +
        num_recv_tokens * intermediate_hidden +
        num_recv_tokens * hidden * 2
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    dist_print(
        f' tokens={num_tokens:5d}  recv={num_recv_tokens:6d}  nz_exp={nonzero:3d}/{num_experts_per_rank}  '
        f'max/mean={skew_max_mean:.2f}  '
        f'{t_fused * 1e6:7.1f} us  {tflops:6.1f} TFLOPS  {hbm_gbs:6.0f} GB/s  (rank{rank_idx})',
        once_in_node=True,
    )

    dist.barrier()
    buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90', once_in_node=True)
        dist.destroy_process_group()
        return

    batches = args.batches if args.batches else [1, 2, 4, 8, 16, 32]

    dist_print(
        f'SM90 MegaMoE bench (skew_alpha={args.skew_alpha} gain={args.skew_gain}): '
        f'ranks={num_ranks} hidden={args.hidden} ih={args.intermediate_hidden} '
        f'experts={args.num_experts} topk={args.num_topk} fast_math=True',
        once_in_node=True,
    )

    num_max_tokens_per_rank = max(batches)
    for num_tokens in batches:
        _run_one_config(
            args, num_tokens, num_max_tokens_per_rank,
            args.hidden, args.intermediate_hidden,
            args.num_experts, args.num_topk,
            num_ranks, rank_idx, group,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 MegaMoE bench (skewed routing)')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)
    parser.add_argument('--batches', type=int, nargs='+', default=None)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--skew-alpha', type=float, default=0.0,
                        help='Zipf exponent for expert bias (0=uniform)')
    parser.add_argument('--skew-gain', type=float, default=1.0,
                        help='Scale of log-bias added to scores (default 1.0)')
    args = parser.parse_args()
    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        np = args.num_processes
        torch.multiprocessing.spawn(test, args=(np, args), nprocs=np)
