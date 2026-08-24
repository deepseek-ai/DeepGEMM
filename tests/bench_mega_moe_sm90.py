"""Benchmark the SM90 Humming FP8 x MXFP4 persistent MegaMoE kernel.

The benchmark measures the CUDA kernel selected by ``bench_kineto``. Weight
quantization and preprocessing happen before timing, and input-buffer copies
are not included in the reported kernel duration. Distributed results report
both rank-0 time and the maximum rank-local average.

DeepGEMM imports are intentionally delayed until the CUDA worker starts so
``--help`` and ``py_compile`` remain usable on non-CUDA hosts.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "flash": {
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 6,
    },
    "pro": {
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
    },
}

DEFAULT_BATCHES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
KERNEL_NAME = "sm90_fp8_mxfp4_mega_moe_persistent_impl"


class _SingleProcessGroup:
    """Minimal rank-0 group for profiler runs that should avoid NCCL."""

    @staticmethod
    def size() -> int:
        return 1

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def barrier() -> None:
        torch.cuda.synchronize()


def _stable_seed(name: str) -> int:
    return (
        sum((index + 1) * ord(character) for index, character in enumerate(name))
        & 0x7FFFFFFF
    )


def _barrier(group: Any) -> None:
    if isinstance(group, _SingleProcessGroup):
        group.barrier()
    else:
        dist.barrier(group=group)


def _quantize_grouped_mxfp4(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create packed E2M1 ``[E,N,K/2]`` and K32 UE8M0 FP32 scales."""
    from deep_gemm.utils import per_token_cast_to_fp4

    num_experts, n, k = weight.shape
    assert k % 32 == 0
    packed = torch.empty(
        (num_experts, n, k // 2), dtype=torch.int8, device=weight.device
    )
    scale = torch.empty(
        (num_experts, n, k // 32),
        dtype=torch.float32,
        device=weight.device,
    )
    for expert_idx in range(num_experts):
        packed[expert_idx], scale[expert_idx] = per_token_cast_to_fp4(
            weight[expert_idx], use_ue8m0=True, gran_k=32
        )
    return packed, scale


def _quantize_block_fp8(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a shared-expert weight with FP32 block-(128,128) scales."""
    from deep_gemm.utils import per_block_cast_to_fp8

    fp8, scale = per_block_cast_to_fp8(weight, use_ue8m0=False, gran_k=128)
    assert scale.dtype == torch.float32
    return fp8, scale.contiguous()


def _copy_shared_l1_scale(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy K128 input scales into the shared L1 column-major view."""
    num_tokens, num_scale_columns = src.shape
    assert dst.size(0) >= num_tokens and dst.size(1) == num_scale_columns
    dst.zero_()
    dst[:num_tokens].copy_(src)


def _emit_json(prefix: str, payload: Dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)


def _cache_mode(flush_l2: int) -> str:
    return "cold_l2" if flush_l2 else "no_explicit_l2_flush"


def _benchmark_case(
    args: argparse.Namespace,
    model_name: str,
    num_tokens: int,
    rank_idx: int,
    num_ranks: int,
    group: Any,
) -> None:
    import deep_gemm
    from deep_gemm.testing import bench_kineto
    from deep_gemm.utils import per_token_cast_to_fp8

    model = MODEL_CONFIGS[model_name]
    hidden = model["hidden"]
    intermediate_hidden = model["intermediate_hidden"]
    num_experts = args.num_experts_override or model["num_experts"]
    num_topk = model["num_topk"]
    num_local_experts = num_experts // num_ranks
    num_shared_experts = args.num_shared_experts

    if num_experts % num_ranks != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by num_ranks={num_ranks}"
        )
    if num_ranks > 64:
        raise ValueError("SM90 Humming MegaMoE supports at most 64 ranks")
    if num_topk > num_experts:
        raise ValueError(
            f"num_topk={num_topk} must not exceed num_experts={num_experts}"
        )
    if num_topk + int(num_shared_experts > 0) > 32:
        raise ValueError("num_topk plus the shared-expert contribution must be <= 32")
    if num_tokens > args.num_max_tokens_per_rank:
        raise ValueError(
            f"num_tokens={num_tokens} exceeds capacity {args.num_max_tokens_per_rank}"
        )

    case_seed = (
        args.seed
        + rank_idx * 1000003
        + _stable_seed(f"mxfp4:{model_name}:{num_tokens}:shared={num_shared_experts}")
    )
    torch.manual_seed(case_seed)
    random.seed(case_seed)

    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group,
        num_experts,
        args.num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts=num_shared_experts,
    )

    try:
        x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        x_fp8, x_scale = per_token_cast_to_fp8(
            x_bf16,
            use_ue8m0=False,
            gran_k=128,
            use_packed_ue8m0=False,
        )
        del x_bf16

        l1_bf16 = (
            torch.randn(
                (num_local_experts, 2 * intermediate_hidden, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        l1_quantized = _quantize_grouped_mxfp4(l1_bf16)
        del l1_bf16

        l2_bf16 = (
            torch.randn(
                (num_local_experts, hidden, intermediate_hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        l2_quantized = _quantize_grouped_mxfp4(l2_bf16)
        del l2_bf16

        transformed_l1, transformed_l2 = (
            deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
                l1_quantized, l2_quantized
            )
        )
        del l1_quantized, l2_quantized

        scores = torch.randn(
            (num_tokens, num_experts), dtype=torch.float32, device="cuda"
        )
        topk_weights, topk_idx = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False
        )
        del scores
        if args.masked_ratio > 0:
            mask = torch.rand_like(topk_idx, dtype=torch.float32) < args.masked_ratio
            topk_idx.masked_fill_(mask, -1)
            topk_weights.masked_fill_(mask, 0.0)

        route_stats = None
        if args.report_route_stats:
            if num_ranks == 1:
                global_topk_idx = topk_idx
            else:
                from deep_gemm.utils.dist import uneven_all_gather

                global_topk_idx = uneven_all_gather(topk_idx, group=group)
            valid_topk_idx = global_topk_idx[global_topk_idx >= 0].long()
            global_recv_counts = torch.bincount(
                valid_topk_idx, minlength=num_experts
            ).reshape(num_ranks, num_local_experts)
            routed_m_blocks = torch.div(
                global_recv_counts + 63, 64, rounding_mode="floor"
            )
            if rank_idx == 0:
                route_stats = [
                    {
                        "rank": route_rank,
                        "recv_tokens": int(
                            global_recv_counts[route_rank].sum().item()
                        ),
                        "m_blocks": int(
                            routed_m_blocks[route_rank].sum().item()
                        ),
                        "experts_over_block_m": int(
                            (global_recv_counts[route_rank] > 64).sum().item()
                        ),
                        "max_expert_tokens": int(
                            global_recv_counts[route_rank].max().item()
                        ),
                        "expert_counts": global_recv_counts[
                            route_rank
                        ].cpu().tolist(),
                    }
                    for route_rank in range(num_ranks)
                ]
            del global_topk_idx, valid_topk_idx

        transformed_shared_l1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        transformed_shared_l2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if num_shared_experts > 0:
            shared_generator = torch.Generator(device="cuda")
            shared_generator.manual_seed(
                args.seed + _stable_seed(f"shared:{model_name}:{num_shared_experts}")
            )
            shared_intermediate_hidden = num_shared_experts * intermediate_hidden
            shared_l1_bf16 = (
                torch.randn(
                    (2 * shared_intermediate_hidden, hidden),
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=shared_generator,
                )
                * 0.05
            )
            shared_l1_quantized = _quantize_block_fp8(shared_l1_bf16)
            del shared_l1_bf16
            shared_l2_bf16 = (
                torch.randn(
                    (hidden, shared_intermediate_hidden),
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=shared_generator,
                )
                * 0.05
            )
            shared_l2_quantized = _quantize_block_fp8(shared_l2_bf16)
            del shared_l2_bf16
            transformed_shared_l1, transformed_shared_l2 = (
                deep_gemm.transform_shared_weights_for_fp8_mega_moe_sm90(
                    shared_l1_quantized, shared_l2_quantized
                )
            )
            del shared_l1_quantized, shared_l2_quantized

        output = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        cumulative_recv_stats = torch.zeros(
            num_local_experts, dtype=torch.int32, device="cuda"
        )

        if num_shared_experts > 0:
            assert buffer.shared_l1_acts_sf is not None
            _copy_shared_l1_scale(buffer.shared_l1_acts_sf, x_scale)

        def run_kernel() -> torch.Tensor:
            buffer.x[:num_tokens].copy_(x_fp8)
            buffer.x_sf[:num_tokens].copy_(x_scale)
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)
            deep_gemm.fp8_mxfp4_mega_moe(
                output,
                transformed_l1,
                transformed_l2,
                buffer,
                shared_l1_weights=transformed_shared_l1,
                shared_l2_weights=transformed_shared_l2,
                cumulative_local_expert_recv_stats=cumulative_recv_stats,
                recipe=(1, 1, 32),
                activation="swiglu",
                activation_clamp=(
                    args.activation_clamp
                    if math.isfinite(args.activation_clamp)
                    else None
                ),
                fast_math=bool(args.fast_math),
            )
            return output

        case_metadata = {
            "implementation": args.implementation,
            "kernel": KERNEL_NAME,
            "model": model_name,
            "m": num_tokens,
            "num_ranks": num_ranks,
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "num_shared_experts": num_shared_experts,
            "num_max_tokens_per_rank": buffer.num_max_tokens_per_rank,
            "fast_math": args.fast_math,
            "activation_clamp": args.activation_clamp,
            "masked_ratio": args.masked_ratio,
            "seed": args.seed,
        }
        if route_stats is not None:
            _emit_json(
                "BENCH_ROUTE_JSON",
                {**case_metadata, "block_m": 64, "per_rank": route_stats},
            )

        if args.profile_only:
            if rank_idx == 0:
                _emit_json("PROFILE_CASE_JSON", case_metadata)
            use_ncu_range = bool(int(os.environ.get("DG_NCU_RANGE", "0")))
            if use_ncu_range:
                torch.cuda.cudart().cudaProfilerStart()
            run_kernel()
            torch.cuda.synchronize()
            _barrier(group)
            if use_ncu_range:
                torch.cuda.cudart().cudaProfilerStop()
            return

        for _ in range(args.num_warmups):
            run_kernel()
        torch.cuda.synchronize()
        _barrier(group)

        local_nonfinite = torch.tensor(
            [int(not bool(torch.isfinite(output).all().item()))],
            dtype=torch.int32,
            device="cuda",
        )
        if num_ranks > 1:
            dist.all_reduce(local_nonfinite, op=dist.ReduceOp.MAX, group=group)
        if local_nonfinite.item() != 0:
            raise RuntimeError("warmup produced non-finite output")

        repeats = args.repeats
        if repeats is None:
            repeats = args.small_repeats if num_tokens <= 128 else args.large_repeats

        rank0_observations = []
        max_rank_observations = []
        for repeat in range(repeats):
            kernel_time = bench_kineto(
                run_kernel,
                KERNEL_NAME,
                barrier=lambda: _barrier(group),
                num_tests=args.num_tests,
                suppress_kineto_output=True,
                flush_l2=bool(args.flush_l2),
            )
            if not math.isfinite(kernel_time) or kernel_time <= 0:
                raise RuntimeError(
                    f"failed to find a positive duration for {KERNEL_NAME}; "
                    f"got {kernel_time}"
                )

            max_rank_time = torch.tensor(
                kernel_time, dtype=torch.float64, device="cuda"
            )
            if num_ranks > 1:
                dist.all_reduce(max_rank_time, op=dist.ReduceOp.MAX, group=group)

            rank_times_us = None
            if args.report_rank_times:
                gathered_rank_times = [
                    torch.zeros_like(max_rank_time) for _ in range(num_ranks)
                ]
                if num_ranks > 1:
                    dist.all_gather(
                        gathered_rank_times, max_rank_time.new_tensor(kernel_time),
                        group=group,
                    )
                else:
                    gathered_rank_times[0].copy_(max_rank_time)
                rank_times_us = [
                    rank_time.item() * 1e6
                    for rank_time in gathered_rank_times
                ]

            rank0_observations.append(kernel_time)
            max_rank_observations.append(max_rank_time.item())
            if rank_idx == 0:
                _emit_json(
                    "BENCH_OBS_JSON",
                    {
                        **case_metadata,
                        "repeat": repeat,
                        "rank0_us": kernel_time * 1e6,
                        "persistent_rank0_us": kernel_time * 1e6,
                        "max_rank_us": max_rank_time.item() * 1e6,
                        "num_tests": args.num_tests,
                        "flush_l2": bool(args.flush_l2),
                        "cache_mode": _cache_mode(args.flush_l2),
                        **(
                            {"rank_times_us": rank_times_us}
                            if rank_times_us is not None else {}
                        ),
                    },
                )

        if rank_idx == 0:
            rank0_median = statistics.median(rank0_observations)
            max_rank_median = statistics.median(max_rank_observations)
            print(
                f"[mxfp4/{model_name}] M={num_tokens:4d} "
                f"obs={repeats:2d} "
                f"max-rank median={max_rank_median * 1e6:8.1f} us "
                f"range={min(max_rank_observations) * 1e6:.1f}-"
                f"{max(max_rank_observations) * 1e6:.1f} us "
                f"cache={_cache_mode(args.flush_l2)}",
                flush=True,
            )
            _emit_json(
                "BENCH_SUMMARY_JSON",
                {
                    **case_metadata,
                    "observations": repeats,
                    "rank0_median_us": rank0_median * 1e6,
                    "persistent_rank0_median_us": rank0_median * 1e6,
                    "max_rank_median_us": max_rank_median * 1e6,
                    "max_rank_min_us": min(max_rank_observations) * 1e6,
                    "max_rank_max_us": max(max_rank_observations) * 1e6,
                    "num_tests": args.num_tests,
                    "num_warmups": args.num_warmups,
                    "flush_l2": bool(args.flush_l2),
                    "cache_mode": _cache_mode(args.flush_l2),
                    "timing_scope": "persistent_kernel_only",
                },
            )

        _barrier(group)
    finally:
        buffer.destroy()


def _benchmark_worker(
    local_rank: int,
    num_local_ranks: int,
    args: argparse.Namespace,
) -> None:
    from deep_gemm.testing import get_arch_major
    from deep_gemm.utils.dist import init_dist

    if args.no_dist:
        if local_rank != 0 or num_local_ranks != 1:
            raise ValueError("--no-dist requires exactly one worker")
        torch.cuda.set_device(0)
        torch.set_default_device("cuda")
        rank_idx, num_ranks, group = 0, 1, _SingleProcessGroup()
    else:
        rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    try:
        if get_arch_major() != 9:
            raise RuntimeError(
                f"SM90 Humming MegaMoE benchmark requires SM90, "
                f"got SM{get_arch_major()}0"
            )

        plan = {
            "implementation": args.implementation,
            "kernel": KERNEL_NAME,
            "models": args.model_config,
            "batches": args.batches,
            "num_ranks": num_ranks,
            "profile_only": args.profile_only,
            "flush_l2": bool(args.flush_l2),
            "cache_mode": _cache_mode(args.flush_l2),
        }
        if rank_idx == 0:
            _emit_json("BENCH_PLAN_JSON", plan)

        for model_name in args.model_config:
            model = MODEL_CONFIGS[model_name]
            num_experts = args.num_experts_override or model["num_experts"]
            if rank_idx == 0:
                print(
                    f"SM90 Humming MegaMoE: model={model_name} "
                    f"ranks={num_ranks} H={model['hidden']} "
                    f"I={model['intermediate_hidden']} E={num_experts} "
                    f"topk={model['num_topk']} "
                    f"shared={args.num_shared_experts}",
                    flush=True,
                )
            for num_tokens in args.batches:
                _benchmark_case(
                    args,
                    model_name,
                    num_tokens,
                    rank_idx,
                    num_ranks,
                    group,
                )
            torch.cuda.empty_cache()
            _barrier(group)
    finally:
        if not args.no_dist and dist.is_initialized():
            dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument(
        "--no-dist",
        action="store_true",
        help="run one GPU without NCCL; requires --num-processes 1",
    )
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--implementation", choices=("mxfp4",), default="mxfp4")
    parser.add_argument(
        "--model-config",
        nargs="+",
        choices=sorted(MODEL_CONFIGS),
        default=["flash", "pro"],
    )
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument(
        "--num-experts-override",
        type=int,
        default=None,
        help="override model expert count, primarily for single-rank profiling",
    )
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--small-repeats", type=int, default=50)
    parser.add_argument("--large-repeats", type=int, default=3)
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="override repeat count for every token batch",
    )
    parser.add_argument("--num-tests", type=int, default=20)
    parser.add_argument(
        "--flush-l2",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 flushes L2 before each sample; 0 disables the explicit flush",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-rank-times",
        action="store_true",
        help="include every rank's local kernel time in BENCH_OBS_JSON",
    )
    parser.add_argument(
        "--report-route-stats",
        action="store_true",
        help="emit per-rank receive counts and routed M64 block counts",
    )
    parser.add_argument("--masked-ratio", type=float, default=0.0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--profile-only",
        "--ncu-profile-only",
        dest="profile_only",
        action="store_true",
        help="launch exactly one selected case without Kineto timing",
    )
    args = parser.parse_args()

    if args.num_processes <= 0:
        parser.error("--num-processes must be positive")
    if args.no_dist and args.num_processes != 1:
        parser.error("--no-dist requires --num-processes 1")
    if not args.batches or min(args.batches) <= 0:
        parser.error("--batches must contain positive token counts")
    if args.num_max_tokens_per_rank < max(args.batches):
        parser.error("--num-max-tokens-per-rank must cover every batch")
    if args.num_experts_override is not None and args.num_experts_override <= 0:
        parser.error("--num-experts-override must be positive")
    if args.num_shared_experts < 0:
        parser.error("--num-shared-experts must be non-negative")
    if args.num_warmups <= 0:
        parser.error("--num-warmups must be positive")
    if args.small_repeats <= 0 or args.large_repeats <= 0:
        parser.error("repeat counts must be positive")
    if args.repeats is not None and args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.num_tests <= 0:
        parser.error("--num-tests must be positive")
    if not 0 <= args.masked_ratio <= 1:
        parser.error("--masked-ratio must be between 0 and 1")
    if args.profile_only and (len(args.model_config) != 1 or len(args.batches) != 1):
        parser.error("--profile-only requires exactly one --model-config and one batch")
    if not args.profile_only and int(os.environ.get("DG_USE_NVIDIA_TOOLS", 0)):
        parser.error(
            "DG_USE_NVIDIA_TOOLS disables Kineto timing; use --profile-only "
            "with NVIDIA tools or unset the variable"
        )
    return args


if __name__ == "__main__":
    benchmark_args = _parse_args()
    if benchmark_args.local_rank_idx is not None:
        _benchmark_worker(
            benchmark_args.local_rank_idx,
            benchmark_args.num_processes,
            benchmark_args,
        )
    elif benchmark_args.no_dist:
        _benchmark_worker(0, 1, benchmark_args)
    else:
        torch.multiprocessing.spawn(
            _benchmark_worker,
            args=(benchmark_args.num_processes, benchmark_args),
            nprocs=benchmark_args.num_processes,
        )
