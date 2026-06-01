"""SM90 (Hopper) MegaMoE benchmark — patched to support power-law routing
and aggregate per-rank timing.

Adds vs upstream `tests/bench_mega_moe_sm90.py`:
  --skew-alpha     Zipf exponent for expert popularity (0 = uniform, upstream default)
  --skew-gain      Scale of log-bias added to scores (default 1.0)
  --skew-seed      Seed for the per-expert bias permutation. Constant across
                   ranks so all ranks see the SAME hot/cold expert assignment
                   (matches production: gate is shared globally).

Output is aggregated across all ranks:
  - per-rank t_us, recv_tokens (max/mean/min)
  - per-rank max/mean ratio (intra-rank imbalance)
  - inter-rank max/mean t_us ratio (cross-rank imbalance)
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


def _make_global_bias(num_experts: int, alpha: float, gain: float, seed: int):
    """Power-law (Zipf) bias per expert, identical across all ranks.

    Generated with a deterministic seed so all ranks see the same hot/cold
    expert assignment. Returns None if alpha == 0 (uniform).
    """
    if alpha <= 0:
        return None
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    ranks = torch.arange(1, num_experts + 1, device='cuda', dtype=torch.float)
    probs = 1.0 / ranks.pow(alpha)
    probs = probs / probs.sum()
    bias = torch.log(probs * num_experts) * gain
    perm = torch.randperm(num_experts, device='cuda', generator=gen)
    return bias[perm]


def _run_one_config(args, num_tokens, num_max_tokens_per_rank,
                    hidden, intermediate_hidden,
                    num_experts, num_topk, num_ranks, rank_idx, group,
                    activation_clamp, fast_math,
                    print_perf=True):
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
    )

    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05

    # Per-rank random scores + global skew bias (added to all rows)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    global_bias = _make_global_bias(num_experts, args.skew_alpha, args.skew_gain,
                                     args.skew_seed)
    if global_bias is not None:
        scores = scores + global_bias.unsqueeze(0)

    replica_for = {}
    if args.num_redundant_experts > 0:
        assert args.num_redundant_experts % num_ranks == 0, 'redundant experts must divide ranks'
        num_replicas_per_rank = args.num_redundant_experts // num_ranks
        assert 0 < num_replicas_per_rank < num_experts_per_rank, 'invalid redundant expert count'
        replica_slots = []
        for r in range(num_ranks):
            base = r * num_experts_per_rank
            replica_slots += list(range(base + num_experts_per_rank - num_replicas_per_rank,
                                        base + num_experts_per_rank))
        logical_mask = torch.ones(num_experts, dtype=torch.bool, device='cuda')
        logical_mask[torch.tensor(replica_slots, dtype=torch.long, device='cuda')] = False
        scores[:, ~logical_mask] = -float('inf')
        if global_bias is not None:
            hot_order = torch.argsort(global_bias.masked_fill(~logical_mask, -float('inf')), descending=True).tolist()
        else:
            hot_order = torch.arange(num_experts, device='cuda')[logical_mask].tolist()
        hot_experts = hot_order[:args.num_redundant_experts]
        replica_for = {int(h): int(s) for h, s in zip(hot_experts, replica_slots)}
        if rank_idx == 0 and print_perf:
            print(
                f'eplb_sim redundant={args.num_redundant_experts} '
                f'replicas_per_rank={num_replicas_per_rank} '
                f'dispatch={args.replica_dispatch}',
                flush=True,
            )

    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if replica_for and args.replica_dispatch == 'hash':
        token_ids = torch.arange(num_tokens, device='cuda')[:, None]
        slot_ids = torch.arange(num_topk, device='cuda')[None, :]
        choose_replica = ((token_ids * num_topk + slot_ids + rank_idx) & 1).bool()
        mapped = topk_idx.clone()
        for logical_expert, replica_slot in replica_for.items():
            mapped = torch.where((topk_idx == logical_expert) & choose_replica,
                                 torch.full_like(mapped, replica_slot), mapped)
        topk_idx = mapped
    elif replica_for and args.replica_dispatch == 'static':
        mapped = topk_idx.clone()
        for logical_expert, replica_slot in replica_for.items():
            logical_rank = logical_expert // num_experts_per_rank
            replica_rank = replica_slot // num_experts_per_rank
            if rank_idx == logical_rank:
                chosen = logical_expert
            elif rank_idx == replica_rank:
                chosen = replica_slot
            else:
                chosen = replica_slot if ((rank_idx + logical_expert) & 1) else logical_expert
            if chosen != logical_expert:
                mapped = torch.where(topk_idx == logical_expert,
                                     torch.full_like(mapped, chosen), mapped)
        topk_idx = mapped
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf),
    )

    phase_profile_enabled = os.environ.get('DG_SM90_MOE_PHASE_PROFILE', '0') != '0'
    phase_profile_ints = 64 if phase_profile_enabled else 0
    cum_stats = torch.zeros(num_experts_per_rank + phase_profile_ints, dtype=torch.int, device='cuda')
    use_eplb_hint = bool(replica_for)
    use_skew_hint = global_bias is not None
    use_masked_hint = args.masked_ratio > 0

    def run_fused():
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_w)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        old_eplb_hint = os.environ.get('DG_SM90_MOE_EPLB_HINT')
        old_skew_hint = os.environ.get('DG_SM90_MOE_SKEW_HINT')
        old_masked_hint = os.environ.get('DG_SM90_MOE_MASKED_HINT')
        if use_eplb_hint:
            os.environ['DG_SM90_MOE_EPLB_HINT'] = '1'
        if use_skew_hint:
            os.environ['DG_SM90_MOE_SKEW_HINT'] = '1'
        if use_masked_hint:
            os.environ['DG_SM90_MOE_MASKED_HINT'] = '1'
        try:
            deep_gemm.fp8_mega_moe(
                y, transformed_l1, transformed_l2, buffer,
                cumulative_local_expert_recv_stats=cum_stats,
                recipe=(128, 128, 128),
                activation='swiglu',
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            )
        finally:
            if use_eplb_hint:
                if old_eplb_hint is None:
                    os.environ.pop('DG_SM90_MOE_EPLB_HINT', None)
                else:
                    os.environ['DG_SM90_MOE_EPLB_HINT'] = old_eplb_hint
            if use_skew_hint:
                if old_skew_hint is None:
                    os.environ.pop('DG_SM90_MOE_SKEW_HINT', None)
                else:
                    os.environ['DG_SM90_MOE_SKEW_HINT'] = old_skew_hint
            if use_masked_hint:
                if old_masked_hint is None:
                    os.environ.pop('DG_SM90_MOE_MASKED_HINT', None)
                else:
                    os.environ['DG_SM90_MOE_MASKED_HINT'] = old_masked_hint
        return y

    run_fused()
    dist.barrier()
    if phase_profile_enabled:
        cum_stats.zero_()
        torch.cuda.synchronize()
        dist.barrier()
    t_fused = bench_kineto(run_fused, 'sm90_fp8_mega_moe',
                           barrier=lambda: dist.barrier(),
                           num_tests=args.num_tests,
                           suppress_kineto_output=True,
                           with_multiple_kernels=os.environ.get(
                               'DG_SM90_MOE_SPLIT_L1_L2', '1') != '0')

    # Per-rank token receive counts + per-local-expert distribution
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    local_mask = ((gathered_topk_idx >= rank_idx * num_experts_per_rank) &
                  (gathered_topk_idx < (rank_idx + 1) * num_experts_per_rank))
    gathered_local = gathered_topk_idx.clone()
    gathered_local[~local_mask] = -1
    num_recv_tokens = int(local_mask.sum().item())
    num_touched_experts = max(0, int(torch.unique(gathered_local.flatten()).numel()) - 1)

    if num_recv_tokens > 0:
        per_expert_count = torch.zeros(num_experts_per_rank, dtype=torch.int64, device='cuda')
        local_ids = gathered_local[local_mask] - rank_idx * num_experts_per_rank
        per_expert_count.scatter_add_(0, local_ids.to(torch.int64),
                                       torch.ones_like(local_ids, dtype=torch.int64))
        counts = per_expert_count.cpu().tolist()
        local_max = max(counts)
        local_mean = sum(counts) / num_experts_per_rank
        local_max_mean = local_max / local_mean if local_mean > 0 else 0.0
    else:
        local_max_mean = 0.0

    # Cross-rank aggregation
    info = torch.tensor([t_fused, float(num_recv_tokens), float(num_touched_experts),
                         local_max_mean], device='cuda', dtype=torch.float64)
    gather_buf = [torch.zeros_like(info) for _ in range(num_ranks)]
    dist.all_gather(gather_buf, info, group=group)

    phase_gather_buf = None
    phase_names = [
        'dispatch_total', 'dispatch_pull', 'math_loop', 'combine_barrier',
        'combine_reduce', 'gemm_core', 'l1_epilogue', 'l2_epilogue',
    ]
    if phase_profile_enabled:
        torch.cuda.synchronize()
        num_profile_metrics = len(phase_names)
        profile = cum_stats[
            num_experts_per_rank:num_experts_per_rank + phase_profile_ints
        ].view(torch.int64)
        phase_values = []
        for i in range(num_profile_metrics):
            total = float(profile[i].item())
            max_v = float(profile[num_profile_metrics + i].item())
            count = float(profile[2 * num_profile_metrics + i].item())
            avg = total / count if count else 0.0
            phase_values.extend([avg, max_v, count])
        phase_info = torch.tensor(phase_values, device='cuda', dtype=torch.float64)
        phase_gather_buf = [torch.zeros_like(phase_info) for _ in range(num_ranks)]
        dist.all_gather(phase_gather_buf, phase_info, group=group)

    if rank_idx == 0 and print_perf:
        all_t = [g[0].item() for g in gather_buf]
        all_recv = [int(g[1].item()) for g in gather_buf]
        all_touch = [int(g[2].item()) for g in gather_buf]
        all_lmm = [g[3].item() for g in gather_buf]
        t_mean = sum(all_t) / num_ranks
        t_max = max(all_t)
        t_min = min(all_t)
        recv_mean = sum(all_recv) / num_ranks
        recv_max = max(all_recv)
        recv_min = min(all_recv)
        inter_max_mean = t_max / t_mean if t_mean > 0 else 0.0
        # Aggregate TFLOPS based on max wall time (real serving sees max)
        total_flops = 2 * sum(all_recv) * (hidden * intermediate_hidden * 3)
        tflops_agg = total_flops / 1e12 / t_max if t_max > 0 else 0.0
        # Per-rank avg view: mean recv across ranks per mean time
        tflops_mean = (2 * recv_mean * hidden * intermediate_hidden * 3) / 1e12 / t_mean

        print(f'tokens={num_tokens:5d}  '
              f'recv[mean/max/min]={recv_mean:6.0f}/{recv_max}/{recv_min}  '
              f'local_max/mean[mean]={sum(all_lmm)/num_ranks:.2f}  '
              f'inter_max/mean={inter_max_mean:.3f}  '
              f't[mean/max/min]us={t_mean*1e6:7.1f}/{t_max*1e6:7.1f}/{t_min*1e6:7.1f}  '
              f'TFLOPS_agg={tflops_agg:6.1f}  TFLOPS_mean={tflops_mean:6.1f}',
              flush=True)
        if phase_gather_buf is not None:
            phase_by_rank = [g.cpu().tolist() for g in phase_gather_buf]
            for i, name in enumerate(phase_names):
                avg_values = [rank_values[3 * i] for rank_values in phase_by_rank]
                max_values = [rank_values[3 * i + 1] for rank_values in phase_by_rank]
                count_values = [rank_values[3 * i + 2] for rank_values in phase_by_rank]
                avg_mean = sum(avg_values) / len(avg_values)
                avg_max = max(avg_values)
                max_max = max(max_values)
                count_max = max(count_values)
                print(
                    f'   phase {name:16s} avg_mean={avg_mean:10.0f} '
                    f'avg_max={avg_max:10.0f} max={max_max:10.0f} '
                    f'count_max={count_max:8.0f}',
                    flush=True,
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

    if rank_idx == 0:
        print(f'SM90 MegaMoE bench: ranks={num_ranks} hidden={args.hidden} '
              f'ih={args.intermediate_hidden} experts={args.num_experts} '
              f'topk={args.num_topk} skew_alpha={args.skew_alpha} '
              f'skew_gain={args.skew_gain} masked_ratio={args.masked_ratio} '
              f'fast_math={bool(args.fast_math)}', flush=True)

    num_max_tokens_per_rank = max(batches)
    for num_tokens in batches:
        _run_one_config(
            args, num_tokens, num_max_tokens_per_rank,
            args.hidden, args.intermediate_hidden,
            args.num_experts, args.num_topk,
            num_ranks, rank_idx, group,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 MegaMoE bench (skew-aware)')

    parser.add_argument('--ncu-profile-only', action='store_true')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)

    parser.add_argument('--batches', type=int, nargs='+', default=None)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--skew-alpha', type=float, default=0.0,
                        help='Zipf exponent for expert popularity; 0=uniform')
    parser.add_argument('--skew-gain', type=float, default=1.0,
                        help='Multiplier on log-bias added to scores')
    parser.add_argument('--skew-seed', type=int, default=0,
                        help='Seed for the global hot/cold expert permutation')
    parser.add_argument('--num-redundant-experts', type=int, default=0,
                        help='EPLB replica simulation: reserve physical expert slots as hot-expert replicas')
    parser.add_argument('--replica-dispatch', choices=('hash', 'static'), default='hash',
                        help='Replica remap model: token-level hash or SGLang static source-rank approximation')

    args = parser.parse_args()

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        np = args.num_processes
        torch.multiprocessing.spawn(test, args=(np, args), nprocs=np)
