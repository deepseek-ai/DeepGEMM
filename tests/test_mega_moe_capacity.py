"""Manual SM100 Mega MoE capacity regression test.

Run on an 8-GPU SM100+ machine:

    python tests/test_mega_moe_capacity.py \
        --num-processes 8 \
        --small-capacity 32 \
        --large-capacity 200000

This verifies that ``num_max_tokens_per_rank`` only sizes the symmetric-memory
buffer.  For the same live tokens, inputs, and weights, changing the capacity
must not change the Mega MoE output.  The default large capacity intentionally
crosses the historical 32-bit SF-pool offset overflow threshold.
"""

import argparse
import hashlib
import os
from typing import Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils.dist import init_dist


FP4_BLOCK = 32


def _pack_fp4_scales(scale: torch.Tensor, mn: int, k: int, groups: int) -> torch.Tensor:
    return deep_gemm.transform_sf_into_required_layout(
        scale.float(), mn, k, (1, FP4_BLOCK), groups
    )


def _make_weights(
    local_rank: int,
    local_experts: int,
    hidden: int,
    intermediate: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000 + local_rank)

    w13 = torch.randint(
        -8,
        8,
        (local_experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        device="cuda",
        generator=gen,
    )
    s13_raw = torch.randint(
        120,
        132,
        (local_experts, 2 * intermediate, hidden // FP4_BLOCK),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    ).view(torch.float8_e8m0fnu)
    s13 = _pack_fp4_scales(s13_raw, 2 * intermediate, hidden, local_experts)

    w2 = torch.randint(
        -8,
        8,
        (local_experts, hidden, intermediate // 2),
        dtype=torch.int8,
        device="cuda",
        generator=gen,
    )
    s2_raw = torch.randint(
        120,
        132,
        (local_experts, hidden, intermediate // FP4_BLOCK),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    ).view(torch.float8_e8m0fnu)
    s2 = _pack_fp4_scales(s2_raw, hidden, intermediate, local_experts)

    return deep_gemm.transform_weights_for_mega_moe((w13, s13), (w2, s2))


def _make_inputs(
    local_rank: int,
    tokens: int,
    hidden: int,
    topk: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2000 + local_rank)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=gen)
    weights = torch.full((tokens, topk), 1.0 / topk, dtype=torch.float32, device="cuda")

    # Route each rank to its local expert range. This keeps the regression
    # deterministic while still exercising symmetric-memory dispatch.
    base = local_rank * local_experts
    indices = torch.arange(base, base + topk, dtype=torch.int64, device="cuda")
    return x, weights, indices.unsqueeze(0).repeat(tokens, 1)


def _run_once(
    capacity: int,
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    l1: Tuple[torch.Tensor, torch.Tensor],
    l2: Tuple[torch.Tensor, torch.Tensor],
    num_experts: int,
    topk: int,
    hidden: int,
    intermediate: int,
    activation_clamp: float,
) -> torch.Tensor:
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group=dist.group.WORLD,
        num_experts=num_experts,
        num_max_tokens_per_rank=capacity,
        num_topk=topk,
        hidden=hidden,
        intermediate_hidden=intermediate,
        use_fp8_dispatch=True,
        activation="swiglu",
    )

    x_fp8, x_sf = deep_gemm.utils.per_token_cast_to_fp8(
        x.contiguous(),
        use_ue8m0=True,
        gran_k=FP4_BLOCK,
        use_packed_ue8m0=True,
    )
    tokens = x.size(0)
    buffer.x[:tokens].copy_(x_fp8)
    buffer.x_sf[:tokens].copy_(x_sf)
    buffer.topk_idx[:tokens].copy_(indices.contiguous())
    buffer.topk_weights[:tokens].copy_(weights.contiguous())

    y = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_fp4_mega_moe(
        y,
        l1,
        l2,
        buffer,
        recipe=(1, 1, FP4_BLOCK),
        activation="swiglu",
        activation_clamp=activation_clamp,
        fast_math=True,
    )
    dist.barrier()
    torch.cuda.synchronize()
    buffer.destroy()
    del buffer
    torch.cuda.empty_cache()
    dist.barrier()
    return y


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.md5(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _worker(local_rank: int, num_processes: int, args: argparse.Namespace) -> None:
    rank, world_size, _ = init_dist(local_rank, num_processes)
    if torch.cuda.get_device_capability()[0] < 10:
        if rank == 0:
            print("Mega MoE capacity regression requires SM100+; skipping")
        dist.destroy_process_group()
        return

    assert args.num_experts % world_size == 0
    local_experts = args.num_experts // world_size
    l1, l2 = _make_weights(local_rank, local_experts, args.hidden, args.intermediate)
    x, weights, indices = _make_inputs(
        local_rank, args.tokens, args.hidden, args.topk, local_experts
    )

    y_small = _run_once(
        args.small_capacity,
        x,
        weights,
        indices,
        l1,
        l2,
        args.num_experts,
        args.topk,
        args.hidden,
        args.intermediate,
        args.activation_clamp,
    ).detach()
    y_small_cpu = y_small.cpu()
    del y_small
    torch.cuda.empty_cache()

    y_large = _run_once(
        args.large_capacity,
        x,
        weights,
        indices,
        l1,
        l2,
        args.num_experts,
        args.topk,
        args.hidden,
        args.intermediate,
        args.activation_clamp,
    ).detach().cpu()

    diff = (y_large.float() - y_small_cpu.float()).abs()
    row = {
        "rank": rank,
        "small_hash": _digest(y_small_cpu),
        "large_hash": _digest(y_large),
        "mean_abs_diff": diff.mean().item(),
        "max_abs_diff": diff.max().item(),
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, row)
    mismatched = [item for item in gathered if item["small_hash"] != item["large_hash"]]
    max_abs = max(float(item["max_abs_diff"]) for item in gathered)
    if rank == 0:
        print(f"token_alignment={deep_gemm._C.get_token_alignment_for_mega_moe()}")
        for item in gathered:
            print(item)

    dist.barrier()
    assert not mismatched and max_abs == 0.0, (
        f"Mega MoE output changed with capacity {args.small_capacity} -> "
        f"{args.large_capacity}: mismatched={mismatched}, max_abs={max_abs}"
    )
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Mega MoE output invariance across symm-buffer capacities."
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--small-capacity", type=int, default=32)
    parser.add_argument("--large-capacity", type=int, default=200000)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--activation-clamp", type=float, default=7.0)
    args = parser.parse_args()

    if torch.cuda.device_count() < args.num_processes:
        raise SystemExit(
            "Mega MoE capacity regression needs at least "
            f"{args.num_processes} CUDA devices, got {torch.cuda.device_count()}"
        )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "8361")
    torch.multiprocessing.spawn(
        _worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
