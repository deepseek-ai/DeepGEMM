import argparse
import os
import random
import sys
import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_nvfp4 import quantize_to_nvfp4
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, get_arch_major

def _run_one_config(args, num_tokens, num_max_tokens_per_rank,
                    hidden, intermediate_hidden,
                    num_experts, num_topk, num_ranks, rank_idx, group,
                    activation_clamp, fast_math):
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Symmetric buffer (one per config: cheaper to recreate than to keep max-size)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        mma_type="fp4xfp4",
    )

    # Inputs (bf16, then quantised)
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    # NVFP4: quantize real BF16 weights directly to packed NVFP4 + per-16 UE4M3 SF.
    # Do not route through block-FP8 weight quantization here; that produces
    # byte-domain FP8 values and requires a separate block SF that this kernel
    # ABI does not consume.
    l1_packed, l1_scale = quantize_to_nvfp4(l1_bf, group_size=16)
    l2_packed, l2_scale = quantize_to_nvfp4(l2_bf, group_size=16)
    transformed_l1, transformed_l2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
        (l1_packed, l1_scale), (l2_packed, l2_scale),
    )
    kernel_name = 'sm90_nvfp4_mega_moe'

    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')
    reuse_output = os.environ.get('DG_BENCH_REUSE_OUTPUT', '0') != '0'
    output = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') if reuse_output else None

    # Stage inputs once; bench-loop re-copies them each call (bench helper expects
    # an idempotent ``fn``).
    def run():
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_w)
        y = output if output is not None else torch.empty(
            (num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.nvfp4_mega_moe(
            y, transformed_l1, transformed_l2, buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return y

    if args.ncu_profile_only:
        dist_print(f'[NCU] tokens={num_tokens} hidden={hidden} ih={intermediate_hidden}',
                   once_in_node=True)
        run()
        torch.cuda.synchronize()
        dist.barrier()
        buffer.destroy()
        return

    # Warm up + benchmark
    run()
    dist.barrier()
    show_kineto = os.environ.get('DG_SHOW_KINETO', '0') != '0'
    t_nvfp4 = bench_kineto(run, kernel_name,
                           barrier=dist.barrier,
                           num_tests=args.num_tests,
                           suppress_kineto_output=not show_kineto)

    t_rank = torch.tensor([t_nvfp4], dtype=torch.float64, device="cuda")
    t_rank_max = t_rank.clone()
    t_rank_sum = t_rank.clone()
    dist.all_reduce(t_rank_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(t_rank_sum, op=dist.ReduceOp.SUM)
    t_nvfp4_rank_max = float(t_rank_max.item())
    t_nvfp4_rank_mean = float(t_rank_sum.item()) / num_ranks

    # Count tokens that landed on this rank for stats
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) |
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops_nvfp4 = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_nvfp4)
    num_touched_experts = max(0, torch.unique(gathered_topk_idx.flatten()).numel() - 1)
    # NVFP4 weights = 0.5 byte/value plus 1 UE4M3 scale byte per 16 K values.
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * (hidden // 2 + hidden // 16) +
        num_touched_experts * hidden * (intermediate_hidden // 2 + intermediate_hidden // 16) +
        num_recv_tokens * hidden +                                  # L1 acts read
        num_recv_tokens * intermediate_hidden +                     # L1 out write
        num_recv_tokens * intermediate_hidden +                     # L2 acts read
        num_recv_tokens * hidden * 2                                # L2 out write
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_nvfp4)

    dist_print(
        f' tokens={num_tokens:4d}  recv={num_recv_tokens:5d}  experts={num_touched_experts:4d}  '
        f'nvfp4={t_nvfp4 * 1e6:7.1f}us '
        f'mean_rank={t_nvfp4_rank_mean * 1e6:7.1f}us max_rank={t_nvfp4_rank_max * 1e6:7.1f}us '
        f'({tflops_nvfp4:5.1f}TF, {hbm_gbs:4.0f}GB/s)  (rank{rank_idx})',
        once_in_node=True,
    )
    dist.barrier()
    buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    if args.batches is None:
        batches = [8, 16, 32, 64, 128]
    else:
        batches = args.batches

    dist_print(
        f'SM90 NVFP4 MegaMoE bench: ranks={num_ranks} hidden={args.hidden} '
        f'ih={args.intermediate_hidden} experts={args.num_experts} topk={args.num_topk} '
        f'masked_ratio={args.masked_ratio} fast_math={bool(args.fast_math)}',
        once_in_node=True,
    )

    # In NCU mode we run only one batch (the first one in `batches`) so that
    # ncu's `--launch-count 1` is unambiguous.
    if args.ncu_profile_only:
        batches = batches[:1]

    num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(batches)
    if num_max_tokens_per_rank < max(batches):
        raise ValueError(
            f"num_max_tokens_per_rank={num_max_tokens_per_rank} is smaller than "
            f"the largest batch M={max(batches)}"
        )
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
    parser = argparse.ArgumentParser(description='SM90 NVFP4 MegaMoE benchmark')

    parser.add_argument('--ncu-profile-only', action='store_true')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)

    parser.add_argument('--batches', type=int, nargs='+', default=None,
                        help='List of num_tokens to sweep (default: 8 16 32 64 128)')
    parser.add_argument('--hidden', type=int, default=6144)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=None,
                        help='Fix buffer capacity independently of the measured batch list')
    args = parser.parse_args()

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        np = args.num_processes
        torch.multiprocessing.spawn(test, args=(np, args), nprocs=np)
