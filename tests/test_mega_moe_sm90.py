"""Multi-rank runtime validation for SM90 FP8 x MXFP4 MegaMoE.

The default smoke suite runs routed-only and shared-expert cases through the
single Humming processed-MXFP4 triplet contract. ``--suite standard`` also
covers heuristic token bands, fast-math modes, activation clamps, and 0/1/max
token boundaries. ``--suite full`` adds production-shaped and randomized cases.

This file intentionally delays importing DeepGEMM until the spawned CUDA
worker starts, so ``--help`` and ``py_compile`` work on non-CUDA hosts.
"""

import argparse
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


Scenario = Tuple[str, Dict[str, Any]]


class _CorrectnessMismatch(AssertionError):
    """A correctness failure already reduced to every participating rank."""


class _SingleProcessGroup:
    """Minimal rank-0 group for sanitizer runs that must avoid NCCL."""

    @staticmethod
    def size() -> int:
        return 1

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def barrier() -> None:
        torch.cuda.synchronize()


def _quantize_grouped_mxfp4(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create packed E2M1 ``[E,N,K/2]`` plus K32 UE8M0 FP32 powers."""
    from deep_gemm.utils import per_token_cast_to_fp4

    num_experts, n, k = weight.shape
    packed = torch.empty(
        (num_experts, n, k // 2), dtype=torch.int8, device=weight.device)
    sf = torch.empty(
        (num_experts, n, k // 32), dtype=torch.float32, device=weight.device)
    for expert_idx in range(num_experts):
        packed[expert_idx], sf[expert_idx] = per_token_cast_to_fp4(
            weight[expert_idx], use_ue8m0=True, gran_k=32)
    return packed, sf


def _quantize_block_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight with natural FP32 block-(128, 128) scales."""
    from deep_gemm.utils import per_block_cast_to_fp8

    fp8, sf = per_block_cast_to_fp8(
        weight, use_ue8m0=False, gran_k=128)
    assert sf.dtype == torch.float32
    assert sf.shape == (weight.size(0) // 128, weight.size(1) // 128)
    return fp8, sf.contiguous()


def _dequant_block_fp8(weight: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Restore a 2D FP8 tensor from natural block-(128, 128) scales."""
    n, k = weight.shape
    assert n % 128 == 0 and k % 128 == 0
    assert sf.shape == (n // 128, k // 128)
    blocks = weight.float().view(n // 128, 128, k // 128, 128)
    return (
        blocks * sf[:, None, :, None]
    ).view(n, k)


def _copy_shared_l1_sf(
    dst: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """Copy K128 FP32 scales into the shared L1 column-major view."""
    num_tokens, num_sf_columns = src.shape
    assert dst.size(0) >= num_tokens and dst.size(1) == num_sf_columns
    dst.zero_()
    if num_tokens > 0:
        dst[:num_tokens].copy_(src)


def _dequant_mxfp4(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Dequantize arbitrary-prefix packed E2M1 tensors with K32 scales."""
    *prefix, n, packed_k = packed.shape
    k = packed_k * 2
    codes = torch.empty((*prefix, n, k), dtype=torch.int8, device=packed.device)
    codes[..., 0::2] = packed & 0x0f
    codes[..., 1::2] = (packed >> 4) & 0x0f
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    values = magnitudes[(codes & 0x07).to(torch.long)]
    values = torch.where((codes & 0x08) != 0, -values, values)
    return (
        values.view(*prefix, n, k // 32, 32) * sf.unsqueeze(-1)
    ).view(*prefix, n, k)


def _deinterleave_l1(tensor: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    """Restore natural ``[gate | up]`` rows from the SM90 SwiGLU layout."""
    num_experts, num_rows, *tail = tensor.shape
    half = num_rows // 2
    interleaved = tensor.view(
        num_experts, half // granularity, 2, granularity, *tail)
    result = torch.empty_like(tensor)
    result[:, :half].copy_(
        interleaved[:, :, 0].reshape(num_experts, half, *tail))
    result[:, half:].copy_(
        interleaved[:, :, 1].reshape(num_experts, half, *tail))
    return result


def _restore_sm90_mxfp4_scale_layout(
    tensor: torch.Tensor,
    hidden: int,
) -> torch.Tensor:
    """Restore the processed ``[K128,N,4]`` payload to logical scale rows."""
    if hidden > 8192:
        return tensor
    num_experts, num_rows, num_k32_groups = tensor.shape
    assert num_k32_groups % 4 == 0
    return tensor.view(
        num_experts, num_k32_groups // 4, num_rows, 4
    ).permute(0, 2, 1, 3).contiguous().view(tensor.shape)


def _swiglu_fp32(gate_up: torch.Tensor, clamp: float) -> torch.Tensor:
    half = gate_up.size(-1) // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _gather_token_sizes(
    num_local_tokens: int,
    num_ranks: int,
    group: dist.ProcessGroup,
) -> List[int]:
    if num_ranks == 1:
        return [num_local_tokens]
    local_size = torch.tensor([num_local_tokens], dtype=torch.long, device='cuda')
    all_sizes = torch.empty(num_ranks, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(all_sizes, local_size, group=group)
    return all_sizes.cpu().tolist()


def _reference_mega_moe(
    x_fp8_local: torch.Tensor,
    x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor,
    topk_weights_local: torch.Tensor,
    l1_weight_local: torch.Tensor,
    l1_sf_local: torch.Tensor,
    l2_weight_local: torch.Tensor,
    l2_sf_local: torch.Tensor,
    shared_l1_weight: Optional[torch.Tensor],
    shared_l1_sf: Optional[torch.Tensor],
    shared_l2_weight: Optional[torch.Tensor],
    shared_l2_sf: Optional[torch.Tensor],
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float,
) -> torch.Tensor:
    """Cross-rank PyTorch oracle independent of the CUDA kernel."""
    from deep_gemm.utils.dist import uneven_all_gather

    sizes = _gather_token_sizes(x_fp8_local.size(0), num_ranks, group)
    if sum(sizes) == 0:
        return torch.empty((0, hidden), dtype=torch.bfloat16, device='cuda')

    if num_ranks == 1:
        x_fp8 = x_fp8_local
        x_sf = x_sf_local
        topk_idx = topk_idx_local
        topk_weights = topk_weights_local
    else:
        x_fp8 = uneven_all_gather(x_fp8_local, group=group)
        x_sf = uneven_all_gather(x_sf_local, group=group)
        topk_idx = uneven_all_gather(topk_idx_local, group=group)
        topk_weights = uneven_all_gather(topk_weights_local, group=group)
    num_global_tokens = x_fp8.size(0)

    x = (
        x_fp8.float().view(num_global_tokens, hidden // 128, 128) *
        x_sf.unsqueeze(-1)
    ).view(num_global_tokens, hidden)
    combine = torch.zeros(
        num_global_tokens,
        num_topk,
        hidden,
        dtype=torch.float32,
        device='cuda',
    )
    num_local_experts = num_experts // num_ranks
    local_expert_start = rank_idx * num_local_experts
    local_expert_end = local_expert_start + num_local_experts
    local_routes = ((topk_idx >= local_expert_start) &
                    (topk_idx < local_expert_end))
    valid_expert_ids = torch.unique(topk_idx[local_routes]).cpu().tolist()
    for expert_id in valid_expert_ids:
        routes = (topk_idx == expert_id).nonzero(as_tuple=False)
        selected = routes[:, 0]
        selected_slots = routes[:, 1]
        local_expert = expert_id - local_expert_start

        # Dequantize each routed expert once. This keeps the oracle independent
        # of the CUDA kernel while bounding production-shape memory to one
        # expert instead of materializing a per-token weight batch.
        l1_weight = _dequant_mxfp4(
            l1_weight_local[local_expert],
            l1_sf_local[local_expert],
        )
        l1_output = torch.matmul(x[selected], l1_weight.transpose(0, 1))
        del l1_weight
        l1_output = _swiglu_fp32(l1_output, activation_clamp)
        l1_output *= topk_weights[selected, selected_slots].unsqueeze(-1)

        num_selected = selected.numel()
        l1_view = l1_output.view(num_selected, intermediate_hidden // 64, 64)
        l1_scale = l1_view.abs().amax(dim=-1).clamp(1e-4) / 448.0
        l1_quantized = (
            l1_view / l1_scale.unsqueeze(-1)
        ).to(torch.float8_e4m3fn).float()
        l2_input = (
            l1_quantized * l1_scale.unsqueeze(-1)
        ).view(num_selected, intermediate_hidden)

        l2_weight = _dequant_mxfp4(
            l2_weight_local[local_expert],
            l2_sf_local[local_expert],
        )
        l2_output = torch.matmul(l2_input, l2_weight.transpose(0, 1))
        del l2_weight
        combine[selected, selected_slots] = l2_output.to(torch.bfloat16).float()

    # Each route is computed only by the rank that owns its expert. Summing
    # outputs, instead of replicating all expert weights, keeps the oracle
    # independent and memory-bounded at production shapes.
    if num_ranks > 1:
        dist.all_reduce(combine, op=dist.ReduceOp.SUM, group=group)

    combine = combine.to(torch.bfloat16).float()
    if shared_l1_weight is not None:
        assert shared_l1_sf is not None
        assert shared_l2_weight is not None
        assert shared_l2_sf is not None
        shared_l1 = _dequant_block_fp8(
            shared_l1_weight, shared_l1_sf)
        shared_l1_output = torch.matmul(x, shared_l1.transpose(0, 1))
        del shared_l1
        shared_l1_output = _swiglu_fp32(
            shared_l1_output, activation_clamp)

        shared_width = shared_l1_output.size(1)
        shared_l1_view = shared_l1_output.view(
            num_global_tokens, shared_width // 64, 64)
        shared_l1_scale = (
            shared_l1_view.abs().amax(dim=-1).clamp(1e-4) / 448.0)
        shared_l1_quantized = (
            shared_l1_view / shared_l1_scale.unsqueeze(-1)
        ).to(torch.float8_e4m3fn).float()
        shared_l2_input = (
            shared_l1_quantized * shared_l1_scale.unsqueeze(-1)
        ).view(num_global_tokens, shared_width)

        shared_l2 = _dequant_block_fp8(
            shared_l2_weight, shared_l2_sf)
        shared_output = torch.matmul(
            shared_l2_input, shared_l2.transpose(0, 1))
        del shared_l2
        combine = torch.cat(
            [combine, shared_output.to(torch.bfloat16).float().unsqueeze(1)],
            dim=1,
        )

    output = combine.sum(dim=1).to(torch.bfloat16)
    rank_start = sum(sizes[:rank_idx])
    return output[rank_start:rank_start + sizes[rank_idx]].contiguous()


def _run_scenario(
    name: str,
    config: Dict[str, Any],
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    diff_tolerance: float,
) -> float:
    import deep_gemm
    from deep_gemm.mega import _restore_mxfp4_sign_bits_from_sm90
    from deep_gemm.testing import calc_diff
    from deep_gemm.utils import per_token_cast_to_fp8

    num_max_tokens = config['num_max_tokens_per_rank']
    num_tokens_by_rank = config.get('num_tokens_by_rank')
    if num_tokens_by_rank is None:
        num_tokens = config.get('num_tokens', num_max_tokens)
    else:
        assert len(num_tokens_by_rank) == num_ranks
        num_tokens = num_tokens_by_rank[rank_idx]
    hidden = config['hidden']
    intermediate_hidden = config['intermediate_hidden']
    num_experts = config['num_experts']
    num_topk = config['num_topk']
    num_shared_experts = config.get('num_shared_experts', 0)
    fast_math = config.get('fast_math', True)
    activation_clamp = config.get('activation_clamp', 10.0)
    masked_ratio = config.get('masked_ratio', 0.0)
    repeat_count = config.get('repeat_count', 1)

    assert num_tokens <= num_max_tokens
    assert num_experts % num_ranks == 0
    assert num_max_tokens % 128 == 0
    assert hidden % 512 == 0 and intermediate_hidden % 256 == 0
    assert num_shared_experts >= 0
    assert num_topk + int(num_shared_experts > 0) <= 32
    assert repeat_count > 0
    num_local_experts = num_experts // num_ranks

    seed = rank_idx * 100003 + sum(
        (index + 1) * ord(character) for index, character in enumerate(name))
    torch.manual_seed(seed)
    random.seed(seed)

    if num_tokens == 0:
        x_fp8 = torch.empty(
            (0, hidden), dtype=torch.float8_e4m3fn, device='cuda')
        x_sf = torch.empty((0, hidden // 128), dtype=torch.float32, device='cuda')
    else:
        x_bf16 = torch.randn(
            num_tokens, hidden, dtype=torch.bfloat16, device='cuda')
        x_fp8, x_sf = per_token_cast_to_fp8(
            x_bf16,
            use_ue8m0=False,
            gran_k=128,
            use_packed_ue8m0=False,
        )
        del x_bf16

    l1_bf16 = torch.randn(
        num_local_experts,
        2 * intermediate_hidden,
        hidden,
        dtype=torch.bfloat16,
        device='cuda',
    ) * 0.05
    l1_quantized = _quantize_grouped_mxfp4(l1_bf16)
    del l1_bf16
    l2_bf16 = torch.randn(
        num_local_experts,
        hidden,
        intermediate_hidden,
        dtype=torch.bfloat16,
        device='cuda',
    ) * 0.05
    l2_quantized = _quantize_grouped_mxfp4(l2_bf16)
    del l2_bf16

    hot_route_rank = config.get('hot_route_rank')
    if hot_route_rank is None:
        scores = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device='cuda')
        topk_weights, topk_idx = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False)
    else:
        assert 0 <= hot_route_rank < num_ranks
        assert num_topk <= num_local_experts
        hot_expert_start = hot_route_rank * num_local_experts
        topk_idx = (
            torch.arange(num_topk, dtype=torch.int64, device='cuda') +
            hot_expert_start
        ).expand(num_tokens, -1).clone()
        topk_weights = torch.randn(
            num_tokens, num_topk, dtype=torch.float32, device='cuda')
    if masked_ratio:
        mask = torch.rand_like(topk_idx, dtype=torch.float32) < masked_ratio
        topk_idx.masked_fill_(mask, -1)
        topk_weights.masked_fill_(mask, 0.0)

    # Independent cumulative receive-count contract for local experts.
    if num_ranks == 1:
        global_topk_idx = topk_idx
    else:
        from deep_gemm.utils.dist import uneven_all_gather
        global_topk_idx = uneven_all_gather(topk_idx, group=group)
    valid_topk_idx = global_topk_idx[global_topk_idx >= 0].long()
    expected_global_recv_stats = torch.bincount(
        valid_topk_idx, minlength=num_experts)
    local_expert_start = rank_idx * num_local_experts
    local_recv_counts = expected_global_recv_stats[
        local_expert_start:local_expert_start + num_local_experts
    ].to(torch.int32)
    expected_recv_stats = local_recv_counts * repeat_count

    transformed_l1, transformed_l2 = (
        deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            l1_quantized, l2_quantized))
    assert len(transformed_l1) == 3 and len(transformed_l2) == 3
    reference_l1_weight = _restore_mxfp4_sign_bits_from_sm90(
        _deinterleave_l1(transformed_l1[0]))
    reference_l1_sf = (
        transformed_l1[2][:, None, None] *
        torch.exp2(_deinterleave_l1(
            _restore_sm90_mxfp4_scale_layout(
                transformed_l1[1], hidden)).float()))
    reference_l2_weight = _restore_mxfp4_sign_bits_from_sm90(
        transformed_l2[0])
    reference_l2_sf = (
        transformed_l2[2][:, None, None] *
        torch.exp2(
            _restore_sm90_mxfp4_scale_layout(
                transformed_l2[1], hidden).float()))

    transformed_shared_l1 = transformed_shared_l2 = None
    reference_shared_l1_weight = reference_shared_l1_sf = None
    reference_shared_l2_weight = reference_shared_l2_sf = None
    if num_shared_experts > 0:
        # Shared weights are replicated, so use a rank-independent generator.
        # Keeping this separate from the routed RNG also makes the routed-only
        # scenarios byte-for-byte stable when shared coverage is added.
        shared_generator = torch.Generator(device='cuda')
        shared_generator.manual_seed(
            700001 + sum(
                (index + 1) * ord(character)
                for index, character in enumerate(name)))
        shared_intermediate_hidden = (
            num_shared_experts * intermediate_hidden)
        shared_l1_bf16 = torch.randn(
            2 * shared_intermediate_hidden,
            hidden,
            dtype=torch.bfloat16,
            device='cuda',
            generator=shared_generator,
        ) * 0.05
        shared_l2_bf16 = torch.randn(
            hidden,
            shared_intermediate_hidden,
            dtype=torch.bfloat16,
            device='cuda',
            generator=shared_generator,
        ) * 0.05
        shared_l1_quantized = _quantize_block_fp8(shared_l1_bf16)
        shared_l2_quantized = _quantize_block_fp8(shared_l2_bf16)
        del shared_l1_bf16, shared_l2_bf16
        transformed_shared_l1, transformed_shared_l2 = (
            deep_gemm.transform_shared_weights_for_fp8_mega_moe_sm90(
                shared_l1_quantized, shared_l2_quantized))
        reference_shared_l1_weight, reference_shared_l1_sf = (
            shared_l1_quantized)
        reference_shared_l2_weight, reference_shared_l2_sf = (
            shared_l2_quantized)

    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group,
        num_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts=num_shared_experts,
    )
    try:
        # SM90 shares main's twelve-view host ABI even when shared experts are
        # disabled.  The routed L2 activation SF remains Hopper-specific FP32
        # K64: one logical scale column per 64 intermediate elements.
        assert buffer.shared_l1_acts.data_ptr() == buffer.x.data_ptr()
        if num_shared_experts == 0:
            assert buffer.shared_l1_acts_sf is None
            assert buffer.shared_l2_acts is None
            assert buffer.shared_l2_acts_sf is None
        else:
            shared_intermediate_hidden = (
                num_shared_experts * intermediate_hidden)
            assert buffer.shared_l1_acts_sf is not None
            assert buffer.shared_l2_acts is not None
            assert buffer.shared_l2_acts_sf is not None
            assert buffer.shared_l1_acts_sf.shape[1] == hidden // 128
            assert buffer.shared_l2_acts.shape == (
                num_max_tokens, shared_intermediate_hidden)
            assert buffer.shared_l2_acts_sf.shape[1] == (
                shared_intermediate_hidden // 64)
            assert buffer.shared_l1_acts_sf.stride() == (
                1, buffer.shared_l1_acts_sf.shape[0])
            assert buffer.shared_l2_acts_sf.stride() == (
                1, buffer.shared_l2_acts_sf.shape[0])
        assert buffer.l1_acts.shape[1] == hidden
        assert buffer.l1_acts_sf.shape[1] == hidden // 128
        assert buffer.l2_acts.shape[1] == intermediate_hidden
        assert buffer.l2_acts_sf.shape[1] == intermediate_hidden // 64
        assert buffer.l1_acts_sf.stride() == (1, buffer.l1_acts_sf.shape[0])
        assert buffer.l2_acts_sf.stride() == (1, buffer.l2_acts_sf.shape[0])
        if config.get('require_ring_wrap', False):
            num_logical_pool_blocks = int(
                ((local_recv_counts + 63) // 64).sum().item())
            num_physical_ring_blocks = buffer.l1_acts.shape[0] // 64
            did_wrap = torch.tensor(
                [int(num_logical_pool_blocks > num_physical_ring_blocks)],
                dtype=torch.int32,
                device='cuda',
            )
            if num_ranks > 1:
                dist.all_reduce(did_wrap, op=dist.ReduceOp.MIN, group=group)
            if rank_idx == 0:
                print(
                    f'  [RING] {name:<40} '
                    f'logical_blocks(rank0)={num_logical_pool_blocks} '
                    f'physical_blocks={num_physical_ring_blocks} '
                    f'all_ranks_wrap={did_wrap.item()}',
                    flush=True,
                )
            if did_wrap.item() == 0:
                raise AssertionError(
                    f'{name}: scenario did not force ring wrap; logical_blocks='
                    f'{num_logical_pool_blocks}, physical_blocks='
                    f'{num_physical_ring_blocks}')

        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        if num_shared_experts > 0:
            # The SM90 descriptor indexes m_idx directly. The exposed view is
            # already column-major, so tensor copy performs the physical layout.
            _copy_shared_l1_sf(buffer.shared_l1_acts_sf, x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        output = torch.empty(
            num_tokens, hidden, dtype=torch.bfloat16, device='cuda')
        recv_stats = torch.zeros(
            num_local_experts, dtype=torch.int32, device='cuda')
        for _ in range(repeat_count):
            deep_gemm.fp8_mxfp4_mega_moe(
                output,
                transformed_l1,
                transformed_l2,
                buffer,
                shared_l1_weights=transformed_shared_l1,
                shared_l2_weights=transformed_shared_l2,
                cumulative_local_expert_recv_stats=recv_stats,
                recipe=(1, 1, 32),
                activation='swiglu',
                activation_clamp=(
                    activation_clamp if math.isfinite(activation_clamp) else None),
                fast_math=fast_math,
            )
            torch.cuda.synchronize()

        stats_failed = torch.tensor(
            [int(not torch.equal(recv_stats, expected_recv_stats))],
            dtype=torch.int32,
            device='cuda',
        )
        if num_ranks > 1:
            dist.all_reduce(stats_failed, op=dist.ReduceOp.MAX, group=group)
        if stats_failed.item() != 0:
            raise _CorrectnessMismatch(
                f'{name}: cumulative expert receive stats mismatch; '
                f'actual={recv_stats.cpu().tolist()}, '
                f'expected={expected_recv_stats.cpu().tolist()}')

        reference = _reference_mega_moe(
            x_fp8,
            x_sf,
            topk_idx,
            topk_weights,
            reference_l1_weight,
            reference_l1_sf,
            reference_l2_weight,
            reference_l2_sf,
            reference_shared_l1_weight,
            reference_shared_l1_sf,
            reference_shared_l2_weight,
            reference_shared_l2_sf,
            rank_idx,
            num_ranks,
            group,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
            activation_clamp,
        )
        diff = 0.0 if output.numel() == 0 else float(calc_diff(output, reference))
        local_failed = torch.tensor(
            [int(not math.isfinite(diff) or diff >= diff_tolerance)],
            dtype=torch.int32,
            device='cuda',
        )
        if num_ranks > 1:
            dist.all_reduce(local_failed, op=dist.ReduceOp.MAX, group=group)
        if local_failed.item() != 0:
            raise _CorrectnessMismatch(
                f'{name}: diff {diff:.6f} exceeded tolerance {diff_tolerance}')
        return diff
    finally:
        buffer.destroy()


def _smoke_scenarios(num_ranks: int) -> List[Scenario]:
    base = dict(
        num_max_tokens_per_rank=128,
        num_tokens=64,
        hidden=512,
        intermediate_hidden=512,
        num_experts=8 * num_ranks,
        num_topk=2,
        fast_math=True,
        activation_clamp=10.0,
    )
    routed = [('smoke.routed', dict(base, repeat_count=2))]
    shared = [
        (f'smoke.shared_s{num_shared_experts}', dict(
            base,
            num_shared_experts=num_shared_experts,
            repeat_count=2,
        ))
        for num_shared_experts in (1, 2)
    ]
    return routed + shared


def _standard_scenarios(num_ranks: int) -> List[Scenario]:
    scenarios: List[Scenario] = []
    base = dict(
        hidden=512,
        intermediate_hidden=512,
        num_experts=8 * num_ranks,
        num_topk=2,
        activation_clamp=10.0,
    )
    # Token-per-expert bands used by the SM90 heuristic.
    for num_tokens in (64, 256, 512, 2048):
        scenarios.append((
            f'heuristic.t{num_tokens}',
            dict(
                base,
                num_max_tokens_per_rank=(
                    (num_tokens + 127) // 128 * 128),
                num_tokens=num_tokens,
                fast_math=True,
            ),
        ))
    # Explicit fast-math and clamp behavior.
    for fast_math in (False, True):
        scenarios.append((
            f'fast_math.{int(fast_math)}',
            dict(
                base,
                num_max_tokens_per_rank=128,
                num_tokens=128,
                fast_math=fast_math,
            ),
        ))
    for clamp in (1.0, math.inf):
        scenarios.append((
            f'clamp.{clamp}',
            dict(
                base,
                num_max_tokens_per_rank=128,
                num_tokens=128,
                fast_math=True,
                activation_clamp=clamp,
            ),
        ))
    # 0, 1, and maximum-token boundaries plus masked routing.
    for num_tokens in (0, 1, 128):
        scenarios.append((
            f'token_boundary.{num_tokens}',
            dict(
                base,
                num_max_tokens_per_rank=128,
                num_tokens=num_tokens,
                fast_math=True,
            ),
        ))
    scenarios.append((
        'masked_routes',
        dict(
            base,
            num_max_tokens_per_rank=128,
            num_tokens=128,
            fast_math=True,
            masked_ratio=0.5,
        ),
    ))
    return scenarios


def _full_scenarios(
    num_ranks: int,
    stress_count: int,
) -> List[Scenario]:
    scenarios: List[Scenario] = [
        ('ring_wrap.h2048', dict(
            num_max_tokens_per_rank=128,
            num_tokens=128,
            hidden=2048,
            intermediate_hidden=1024,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.flash_m128', dict(
            num_max_tokens_per_rank=128,
            num_tokens=128,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.flash_m8', dict(
            num_max_tokens_per_rank=128,
            num_tokens=8,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.flash_m16', dict(
            num_max_tokens_per_rank=128,
            num_tokens=16,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.flash_m32', dict(
            num_max_tokens_per_rank=128,
            num_tokens=32,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.flash_m64', dict(
            num_max_tokens_per_rank=128,
            num_tokens=64,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.flash_m256', dict(
            num_max_tokens_per_rank=256,
            num_tokens=256,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
        )),
        ('production.flash_m512', dict(
            num_max_tokens_per_rank=512,
            num_tokens=512,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
        )),
        ('production.flash_m1024', dict(
            num_max_tokens_per_rank=1024,
            num_tokens=1024,
            hidden=4096,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
        )),
        ('production.pro_m256', dict(
            num_max_tokens_per_rank=256,
            num_tokens=256,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=48 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
        )),
        ('production.pro_m512', dict(
            num_max_tokens_per_rank=512,
            num_tokens=512,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=48 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
        )),
        # Compatibility probes intentionally exercise dimensions rather than
        # model-name dispatch.  They cover shared experts and wider top-k so
        # optimization selectors cannot silently assume the DSV4 topology.
        ('compat.h6144_i2048_e256_topk8_shared1_m512', dict(
            num_max_tokens_per_rank=512,
            num_tokens=512,
            hidden=6144,
            intermediate_hidden=2048,
            num_experts=32 * num_ranks,
            num_topk=8,
            num_shared_experts=1,
            fast_math=True,
            activation_clamp=10.0,
        )),
        ('compat.h7168_i3072_e896_topk16_shared2_m512', dict(
            num_max_tokens_per_rank=512,
            num_tokens=512,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=112 * num_ranks,
            num_topk=16,
            num_shared_experts=2,
            fast_math=True,
            activation_clamp=10.0,
        )),
        ('production.pro_m8', dict(
            num_max_tokens_per_rank=128,
            num_tokens=8,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=48 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
        )),
        ('production.pro_m16', dict(
            num_max_tokens_per_rank=128,
            num_tokens=16,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=48 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.pro_m32', dict(
            num_max_tokens_per_rank=128,
            num_tokens=32,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=48 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.pro_m64', dict(
            num_max_tokens_per_rank=128,
            num_tokens=64,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=48 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
        ('production.pro_m128', dict(
            num_max_tokens_per_rank=128,
            num_tokens=128,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=48 * num_ranks,
            num_topk=6,
            fast_math=True,
            activation_clamp=10.0,
            require_ring_wrap=True,
        )),
    ]
    if num_ranks > 1:
        scenarios.extend([
            ('swap_ab_cross_rank_bound.flash_m8', dict(
                num_max_tokens_per_rank=128,
                num_tokens=8,
                hidden=4096,
                intermediate_hidden=2048,
                num_experts=32 * num_ranks,
                num_topk=6,
                fast_math=True,
                activation_clamp=10.0,
                hot_route_rank=0,
            )),
            ('swap_ab_cross_rank_bound.flash_m16', dict(
                num_max_tokens_per_rank=128,
                num_tokens=16,
                hidden=4096,
                intermediate_hidden=2048,
                num_experts=32 * num_ranks,
                num_topk=6,
                fast_math=True,
                activation_clamp=10.0,
                hot_route_rank=0,
            )),
            ('dispatch_mixed_protocol.flash_m32_m64', dict(
                num_max_tokens_per_rank=128,
                num_tokens_by_rank=(32,) + (64,) * (num_ranks - 1),
                hidden=4096,
                intermediate_hidden=2048,
                num_experts=32 * num_ranks,
                num_topk=6,
                fast_math=True,
                activation_clamp=10.0,
                hot_route_rank=1,
            )),
            ('dispatch_mixed_protocol.flash_m1024_m64', dict(
                num_max_tokens_per_rank=1024,
                num_tokens_by_rank=(1024,) + (64,) * (num_ranks - 1),
                hidden=4096,
                intermediate_hidden=2048,
                num_experts=32 * num_ranks,
                num_topk=6,
                fast_math=True,
                activation_clamp=10.0,
                hot_route_rank=1,
            )),
        ])
    rng = random.Random(0xC0FFEE)
    for index in range(stress_count):
        num_tokens = rng.choice((32, 64, 128, 256, 512))
        scenarios.append((f'stress.{index:03d}', dict(
            num_max_tokens_per_rank=(num_tokens + 127) // 128 * 128,
            num_tokens=num_tokens,
            hidden=rng.choice((512, 1024, 2048)),
            intermediate_hidden=rng.choice((512, 1024, 2048)),
            num_experts=8 * num_ranks,
            num_topk=rng.choice((1, 2, 4)),
            fast_math=rng.choice((False, True)),
            activation_clamp=rng.choice((1.0, 10.0, math.inf)),
            masked_ratio=rng.choice((0.0, 0.0, 0.3, 0.7)),
        )))
    return scenarios


def _test_worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    from deep_gemm.testing import get_arch_major
    from deep_gemm.utils.dist import init_dist

    if args.no_dist:
        assert local_rank == 0 and num_local_ranks == 1
        torch.cuda.set_device(0)
        torch.set_default_device('cuda')
        rank_idx, num_ranks, group = 0, 1, _SingleProcessGroup()
    else:
        rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        if rank_idx == 0:
            print(
                f'[SKIP] test_mega_moe_sm90.py requires SM90, got SM{get_arch_major()}0',
                flush=True,
            )
        if not args.no_dist:
            dist.destroy_process_group()
        return

    scenarios = _smoke_scenarios(num_ranks)
    if args.suite in ('standard', 'full'):
        scenarios.extend(_standard_scenarios(num_ranks))
    if args.suite == 'full':
        scenarios.extend(_full_scenarios(
            num_ranks, args.stress_count))
    if args.filter:
        scenarios = [
            scenario for scenario in scenarios if args.filter in scenario[0]
        ]
    if args.fast_math is not None:
        scenarios = [
            (name, dict(config, fast_math=bool(args.fast_math)))
            for name, config in scenarios
        ]

    if not scenarios:
        if rank_idx == 0:
            print('[FAIL] scenario filter selected zero tests', flush=True)
        if not args.no_dist:
            dist.destroy_process_group()
        raise SystemExit(2)

    if rank_idx == 0:
        print(
            f'SM90 MXFP4 MegaMoE plan: {len(scenarios)} scenarios, '
            f'{num_ranks} ranks, suite={args.suite}',
            flush=True,
        )
    failures: List[str] = []
    successes = 0
    for name, config in scenarios:
        try:
            diff = _run_scenario(
                name,
                config,
                rank_idx,
                num_ranks,
                group,
                args.diff_tolerance,
            )
            successes += 1
            if rank_idx == 0:
                print(
                    f'  [PASS] {name:<40} diff={diff:.6f} '
                    f'fast_math={int(config.get("fast_math", True))}',
                    flush=True,
                )
        except Exception as exception:
            # A rank-local Python/JIT/launch exception can leave peers inside
            # the NVLink kernel. Let `mp.spawn` terminate sibling processes
            # immediately. Only globally reduced correctness mismatches are safe to collect,
            # because `_run_scenario` first reduces them to every rank.
            if num_ranks > 1 and not isinstance(
                    exception, _CorrectnessMismatch):
                raise
            failures.append(name)
            if rank_idx == 0:
                print(f'  [FAIL] {name}: {exception}', flush=True)
            if args.fail_fast or not isinstance(
                    exception, _CorrectnessMismatch):
                break

    if rank_idx == 0:
        num_executed = successes + len(failures)
        print(
            f'SUMMARY total={num_executed} success={successes} '
            f'failed={len(failures)} planned={len(scenarios)}',
            flush=True,
        )
    group.barrier()
    if not args.no_dist:
        dist.destroy_process_group()
    if failures:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Cross-rank SM90 FP8 x MXFP4 MegaMoE runtime validation')
    parser.add_argument(
        '--num-processes', type=int, default=2,
        help='number of local GPU ranks to spawn (default: 2)')
    parser.add_argument(
        '--no-dist', action='store_true',
        help='run one GPU without NCCL (intended for compute-sanitizer)')
    parser.add_argument(
        '--suite', choices=('smoke', 'standard', 'full'), default='smoke',
        help='scenario breadth; all cases use Humming processed MXFP4 triplets')
    parser.add_argument(
        '--fast-math', type=int, choices=(0, 1), default=None,
        help='override fast_math for every selected scenario')
    parser.add_argument(
        '--diff-tolerance', type=float, default=0.01,
        help='maximum calc_diff value (default: 0.01)')
    parser.add_argument(
        '--stress-count', type=int, default=8,
        help='number of randomized scenarios in the full suite')
    parser.add_argument(
        '--filter', default='', help='only run scenario names containing this text')
    parser.add_argument(
        '--fail-fast', action='store_true', help='stop after the first failed scenario')
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = _parse_args()
    if cli_args.no_dist:
        if cli_args.num_processes != 1:
            raise ValueError('--no-dist requires --num-processes 1')
        _test_worker(0, 1, cli_args)
    else:
        torch.multiprocessing.spawn(
            _test_worker,
            args=(cli_args.num_processes, cli_args),
            nprocs=cli_args.num_processes,
        )
