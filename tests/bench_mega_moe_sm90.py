"""Benchmark the SM90 FP8 MegaMoE kernel on Flash and Pro model shapes."""

import argparse
import json
import statistics
from typing import Dict, Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist


MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    'flash': {
        'hidden': 4096,
        'intermediate_hidden': 2048,
        'num_experts': 256,
        'num_topk': 6,
    },
    'pro': {
        'hidden': 7168,
        'intermediate_hidden': 3072,
        'num_experts': 384,
        'num_topk': 6,
    },
}

DEFAULT_BATCHES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
PHASE_KERNEL_NAMES = (
    'sm90_fp8_mega_moe_l1_impl',
    'sm90_fp8_mega_moe_l2_impl',
)


def _stable_seed(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name)) & 0x7fffffff


def _quantize_grouped_fp8_block_128_128(
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = weights.shape
    assert n % 128 == 0 and k % 128 == 0

    weights_fp8 = torch.empty_like(weights, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (num_groups, n // 128, k // 128),
        dtype=torch.float,
        device=weights.device,
    )
    for start in range(0, num_groups, 4):
        end = min(start + 4, num_groups)
        block = weights[start:end].view(
            end - start, n // 128, 128, k // 128, 128,
        ).float()
        block_scales = block.abs().amax(dim=(-1, -3)).clamp(1e-4) / 448.0
        weights_fp8[start:end].copy_(
            (block / block_scales.unsqueeze(-1).unsqueeze(-3))
            .to(torch.float8_e4m3fn)
            .view(end - start, n, k)
        )
        scales[start:end].copy_(block_scales)
    return weights_fp8, scales.contiguous()


def _benchmark_case(
    args: argparse.Namespace,
    model_name: str,
    num_tokens: int,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
) -> None:
    model = MODEL_CONFIGS[model_name]
    hidden = model['hidden']
    intermediate_hidden = model['intermediate_hidden']
    num_experts = model['num_experts']
    num_topk = model['num_topk']
    num_experts_per_rank = num_experts // num_ranks
    assert num_experts % num_ranks == 0
    assert num_tokens <= args.num_max_tokens_per_rank

    case_seed = (
        args.seed
        + rank_idx * 1000003
        + _stable_seed(f'{model_name}:{num_tokens}')
    )
    torch.manual_seed(case_seed)
    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group,
        num_experts,
        args.num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )

    x_bf16 = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device='cuda',
    )
    l1_bf16 = torch.randn(
        (num_experts_per_rank, 2 * intermediate_hidden, hidden),
        dtype=torch.bfloat16,
        device='cuda',
    ) * 0.05
    l2_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device='cuda',
    ) * 0.05
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float, device='cuda',
    )
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False,
    )
    if args.masked_ratio > 0:
        mask = torch.rand_like(topk_idx, dtype=torch.float) < args.masked_ratio
        topk_idx.masked_fill_(mask, -1)
        topk_weights.masked_fill_(mask, 0)

    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False,
    )
    l1_fp8, l1_sf = _quantize_grouped_fp8_block_128_128(l1_bf16)
    l2_fp8, l2_sf = _quantize_grouped_fp8_block_128_128(l2_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_fp8, l1_sf), (l2_fp8, l2_sf),
    )
    del x_bf16, l1_bf16, l2_bf16, scores
    cumulative_recv_stats = torch.zeros(
        num_experts_per_rank, dtype=torch.int, device='cuda',
    )
    y = torch.empty(
        (num_tokens, hidden), dtype=torch.bfloat16, device='cuda',
    )

    def run_sm90() -> torch.Tensor:
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        deep_gemm.fp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_recv_stats,
            recipe=(128, 128, 128),
            activation='swiglu',
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
        return y

    if args.ncu_profile_only:
        dist_print(
            f'[NCU] model={model_name} M={num_tokens}', once_in_node=True,
        )
        run_sm90()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        buffer.destroy()
        return

    repeats = args.repeats
    if repeats is None:
        repeats = args.small_repeats if num_tokens <= 128 else args.large_repeats

    run_sm90()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    rank0_observations = []
    max_rank_observations = []
    for repeat in range(repeats):
        phase_times = bench_kineto(
            run_sm90,
            PHASE_KERNEL_NAMES,
            barrier=lambda: dist.barrier(group=group),
            num_tests=args.num_tests,
            suppress_kineto_output=True,
        )
        local_time = sum(phase_times)
        max_rank_time = torch.tensor(local_time, dtype=torch.float64, device='cuda')
        dist.all_reduce(max_rank_time, op=dist.ReduceOp.MAX, group=group)

        rank0_observations.append(local_time)
        max_rank_observations.append(max_rank_time.item())
        if rank_idx == 0:
            print('BENCH_OBS_JSON ' + json.dumps({
                'model': model_name,
                'm': num_tokens,
                'repeat': repeat,
                'rank0_us': local_time * 1e6,
                'max_rank_us': max_rank_time.item() * 1e6,
                'l1_rank0_us': phase_times[0] * 1e6,
                'l2_rank0_us': phase_times[1] * 1e6,
                'num_tests': args.num_tests,
                'num_max_tokens_per_rank': args.num_max_tokens_per_rank,
                'seed': args.seed,
            }, sort_keys=True), flush=True)

    if rank_idx == 0:
        median_time = statistics.median(max_rank_observations)
        print(
            f'[{model_name:5s}] M={num_tokens:4d} obs={repeats:2d} '
            f'max-rank median={median_time * 1e6:8.1f} us '
            f'range={min(max_rank_observations) * 1e6:.1f}-'
            f'{max(max_rank_observations) * 1e6:.1f} us',
            flush=True,
        )
        print('BENCH_SUMMARY_JSON ' + json.dumps({
            'model': model_name,
            'm': num_tokens,
            'observations': repeats,
            'rank0_median_us': statistics.median(rank0_observations) * 1e6,
            'max_rank_median_us': median_time * 1e6,
            'max_rank_min_us': min(max_rank_observations) * 1e6,
            'max_rank_max_us': max(max_rank_observations) * 1e6,
            'num_tests': args.num_tests,
            'num_max_tokens_per_rank': args.num_max_tokens_per_rank,
        }, sort_keys=True), flush=True)

    dist.barrier(group=group)
    buffer.destroy()


def _benchmark_worker(
    local_rank: int,
    num_local_ranks: int,
    args: argparse.Namespace,
) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print(
            f'[SKIP] SM90 MegaMoE benchmark requires SM90; got SM{get_arch_major()}0',
            once_in_node=True,
        )
        dist.destroy_process_group()
        return

    models = args.model_config[:1] if args.ncu_profile_only else args.model_config
    batches = args.batches[:1] if args.ncu_profile_only else args.batches
    for model_name in models:
        model = MODEL_CONFIGS[model_name]
        dist_print(
            f'SM90 MegaMoE benchmark: model={model_name} ranks={num_ranks} '
            f'H={model["hidden"]} IH={model["intermediate_hidden"]} '
            f'E={model["num_experts"]} topk={model["num_topk"]}',
            once_in_node=True,
        )
        for num_tokens in batches:
            _benchmark_case(
                args, model_name, num_tokens, rank_idx, num_ranks, group,
            )
        torch.cuda.empty_cache()
        dist.barrier(group=group)

    dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)
    parser.add_argument(
        '--model-config', nargs='+', choices=sorted(MODEL_CONFIGS),
        default=['flash', 'pro'],
    )
    parser.add_argument('--batches', type=int, nargs='+', default=DEFAULT_BATCHES)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--small-repeats', type=int, default=50)
    parser.add_argument('--large-repeats', type=int, default=3)
    parser.add_argument(
        '--repeats', type=int, default=None,
        help='Override the small/large repeat counts for every M.',
    )
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--fast-math', type=int, choices=[0, 1], default=1)
    parser.add_argument('--ncu-profile-only', action='store_true')
    args = parser.parse_args()

    assert args.num_processes > 0
    assert args.batches and min(args.batches) >= 0
    assert args.num_max_tokens_per_rank >= max(args.batches)
    assert args.small_repeats > 0 and args.large_repeats > 0
    assert args.repeats is None or args.repeats > 0
    assert args.num_tests > 0
    assert 0 <= args.masked_ratio <= 1
    return args


if __name__ == '__main__':
    benchmark_args = _parse_args()
    if benchmark_args.local_rank_idx is not None:
        _benchmark_worker(
            benchmark_args.local_rank_idx,
            benchmark_args.num_processes,
            benchmark_args,
        )
    else:
        torch.multiprocessing.spawn(
            _benchmark_worker,
            args=(benchmark_args.num_processes, benchmark_args),
            nprocs=benchmark_args.num_processes,
        )
