import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_nvfp4 import (
    FP4_VALUES,
    dequantize_nvfp4_to_fp32,
    nvfp4_scale_to_tile_major,
    quantize_to_nvfp4,
    ue4m3_to_fp32,
)
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


def _pack_nvfp4_modelopt(nibbles: torch.Tensor) -> torch.Tensor:
    *outer_shape, k = nibbles.shape
    assert k % 2 == 0
    pairs = nibbles.view(*outer_shape, k // 2, 2).to(torch.int16)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8).contiguous()


_CUDA_DEQUANT_EXT = None


def _load_cuda_dequant_ext():
    global _CUDA_DEQUANT_EXT
    if _CUDA_DEQUANT_EXT is not None:
        return _CUDA_DEQUANT_EXT

    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

__global__ void nvfp4_lut_bytes_kernel(uint8_t* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 128 * 16)
        return;

    const uint32_t scale = static_cast<uint32_t>(idx / 16);
    const uint32_t nibble = static_cast<uint32_t>(idx & 15);
    const uint32_t packed_byte = (nibble << 4) | nibble;
    const uint32_t q = packed_byte * 0x01010101u;
    const uint2 lut = deep_gemm::nvfp4::load_e2m1_ue4m3_lut(scale);
    const uint2 deq = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(q, lut);

    const uint8_t* hi = reinterpret_cast<const uint8_t*>(&deq.x);
    const uint8_t* lo = reinterpret_cast<const uint8_t*>(&deq.y);
    out[idx * 2] = hi[0];
    out[idx * 2 + 1] = lo[0];
}

torch::Tensor nvfp4_lut_bytes_cuda() {
    auto out = torch::empty({128, 16, 2}, torch::device(torch::kCUDA).dtype(torch::kUInt8));
    nvfp4_lut_bytes_kernel<<<8, 256>>>(out.data_ptr<uint8_t>());
    return out;
}
"""
    cpp_src = "torch::Tensor nvfp4_lut_bytes_cuda();"
    _CUDA_DEQUANT_EXT = load_inline(
        name="deepgemm_nvfp4_dequant_lut_test",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["nvfp4_lut_bytes_cuda"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _CUDA_DEQUANT_EXT


def _run_cuda_dequant_lut_unit_test() -> None:
    ext = _load_cuda_dequant_ext()
    got = ext.nvfp4_lut_bytes_cuda().cpu()

    scales = torch.arange(128, dtype=torch.uint8, device="cpu")
    nibbles = torch.arange(16, dtype=torch.uint8, device="cpu")
    mag = FP4_VALUES[(nibbles & 0x7).long()].view(1, 16)
    signed = torch.where(((nibbles >> 3) & 0x1).bool().view(1, 16), -mag, mag)
    expected = ((signed * ue4m3_to_fp32(scales).view(128, 1)) / 8.0).to(
        torch.float8_e4m3fn
    ).view(torch.uint8)
    expected = torch.where(
        ((expected & 0x7F) == 0) & (((nibbles >> 3) & 0x1).bool().view(1, 16)),
        expected | 0x80,
        expected,
    )
    expected = expected.unsqueeze(-1).expand(128, 16, 2).contiguous()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    compensated = got[..., 0].contiguous().view(torch.float8_e4m3fn).float() * 8.0
    exact = signed * ue4m3_to_fp32(scales).view(128, 1)
    high_scale_nonzero = (scales.view(-1, 1) >= 80) & (exact != 0)
    relative_error = (compensated - exact).abs() / exact.abs().clamp_min(1e-30)
    if relative_error[high_scale_nonzero].max().item() > 0.05:
        raise AssertionError("compensated NVFP4 LUT saturates in the high-scale range")
    print("NVFP4 CUDA dequant LUT unit test: PASS", flush=True)


def _run_dequant_unit_test() -> None:
    scales = torch.tensor([0x00, 0x01, 0x07, 0x08, 0x38, 0x3F, 0x7E, 0x7F], dtype=torch.uint8)
    nibbles = torch.arange(16, dtype=torch.uint8).view(1, 1, 16).expand(scales.numel(), 1, 16).clone()
    packed = _pack_nvfp4_modelopt(nibbles)
    got = dequantize_nvfp4_to_fp32(packed, scales.view(-1, 1, 1), group_size=16)

    mag = FP4_VALUES.to(nibbles.device)[(nibbles & 0x7).long()]
    signed = torch.where(((nibbles >> 3) & 0x1).bool(), -mag, mag)
    expected = signed * ue4m3_to_fp32(scales).view(-1, 1, 1)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    tile_weight = torch.randn(2, 256, 256, dtype=torch.bfloat16) * 0.1
    tile_packed, tile_scale = quantize_to_nvfp4(tile_weight, group_size=16)
    tile_scale_tm = nvfp4_scale_to_tile_major(tile_scale)
    torch.testing.assert_close(
        dequantize_nvfp4_to_fp32(tile_packed, tile_scale, group_size=16),
        dequantize_nvfp4_to_fp32(tile_packed, tile_scale_tm, group_size=16),
        rtol=0,
        atol=0,
    )

    print('NVFP4 dequant unit test: PASS', flush=True)




def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _quantize_expert_chunks(
    weight: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts, n, k = weight.shape
    packed = torch.empty(
        (num_experts, n, k // 2), dtype=torch.uint8, device=weight.device
    )
    scales = torch.empty(
        (num_experts, n, k // 16), dtype=torch.uint8, device=weight.device
    )
    for start in range(0, num_experts, chunk_size):
        end = min(start + chunk_size, num_experts)
        packed[start:end], scales[start:end] = quantize_to_nvfp4(
            weight[start:end], group_size=16
        )
    return packed, scales


def _dequantize_expert_chunks(
    packed: torch.Tensor, scales: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    num_experts, n, packed_k = packed.shape
    output = torch.empty(
        (num_experts, n, packed_k * 2),
        dtype=torch.float32,
        device=packed.device,
    )
    for start in range(0, num_experts, chunk_size):
        end = min(start + chunk_size, num_experts)
        output[start:end] = dequantize_nvfp4_to_fp32(
            packed[start:end], scales[start:end], group_size=16
        )
    return output


def _run_case(args: argparse.Namespace, m_tokens: int, weight_scale: float,
              global_scale_mode: str,
              rank_idx: int, group: dist.ProcessGroup,
              heterogeneous: bool = False,
              token_sequence: list[int] | None = None) -> None:
    call_m_tokens = token_sequence or [m_tokens] * args.kernel_repeats
    allocation_tokens = max(call_m_tokens)
    torch.manual_seed(
        args.seed + m_tokens + int(weight_scale * 1000000) +
        (100000 if global_scale_mode != "none" else 0)
    )

    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_ranks = dist.get_world_size(group)
    if args.expert_chunk_size <= 0:
        raise ValueError("expert_chunk_size must be positive")
    if num_experts % num_ranks != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by num_ranks={num_ranks}"
        )
    num_local_experts = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(32, m_tokens)
    if num_max_tokens_per_rank < allocation_tokens:
        raise ValueError(
            f"num_max_tokens_per_rank={num_max_tokens_per_rank} is smaller "
            f"than M={allocation_tokens}"
        )

    rank_batches = [None] * num_ranks
    dist.all_gather_object(rank_batches, call_m_tokens, group=group)
    if rank_idx == 0:
        print(
            f"=== NVFP4 correctness M={rank_batches}, "
            f"NE={num_experts}, NL={num_local_experts}, NK={num_topk}, "
            f"NMT={num_max_tokens_per_rank}, weight_scale={weight_scale:g}, "
            f"global_scale_mode={global_scale_mode}, "
            "reference=exact-nvfp4 ===",
            flush=True,
        )

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        mma_type="fp4xfp4",
        activation="swiglu",
    )

    x_bf = torch.randn(
        (allocation_tokens, hidden), dtype=torch.bfloat16, device="cuda"
    )
    l1_bf = torch.randn(
        (num_local_experts, 2 * intermediate_hidden, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * weight_scale
    l2_bf = torch.randn(
        (num_local_experts, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * weight_scale
    scores = torch.randn(
        (allocation_tokens, num_experts), dtype=torch.float, device="cuda"
    )
    topk_weights, topk_idx = torch.topk(scores.sigmoid(), num_topk, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    if args.route_destination_rank is not None:
        destination_rank = args.route_destination_rank
        if not 0 <= destination_rank < num_ranks:
            raise ValueError("route-destination-rank is outside the EP group")
        if num_local_experts < num_topk:
            raise ValueError(
                "route-destination-rank requires at least top-k local experts"
            )
        first_expert = destination_rank * num_local_experts
        topk_idx.copy_(
            torch.arange(
                first_expert,
                first_expert + num_topk,
                dtype=topk_idx.dtype,
                device=topk_idx.device,
            ).expand_as(topk_idx)
        )

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)

    if global_scale_mode == "none":
        l1_global_scales = None
        l2_global_scales = None
        l1_quant_input = l1_bf
        l2_quant_input = l2_bf
    elif global_scale_mode == "expert":
        l1_global_scales = torch.linspace(
            0.73, 1.37, num_local_experts, dtype=torch.float32, device="cuda"
        )
        l2_global_scales = torch.linspace(
            1.31, 0.67, num_local_experts, dtype=torch.float32, device="cuda"
        )
        l1_quant_input = l1_bf
        l2_quant_input = l2_bf
    elif global_scale_mode == "modelopt":
        l1_global_scales = (
            l1_bf.float().abs().amax(dim=(1, 2)) / (FP4_VALUES[-1] * 448.0)
        ).clamp_min(1e-30)
        l2_global_scales = (
            l2_bf.float().abs().amax(dim=(1, 2)) / (FP4_VALUES[-1] * 448.0)
        ).clamp_min(1e-30)
        l1_quant_input = l1_bf / l1_global_scales[:, None, None]
        l2_quant_input = l2_bf / l2_global_scales[:, None, None]
    else:
        raise ValueError(f"unsupported global_scale_mode={global_scale_mode}")

    l1_packed, l1_scale = _quantize_expert_chunks(
        l1_quant_input, args.expert_chunk_size
    )
    l2_packed, l2_scale = _quantize_expert_chunks(
        l2_quant_input, args.expert_chunk_size
    )
    if not heterogeneous:
        l1_dequant = _dequantize_expert_chunks(
            l1_packed, l1_scale, args.expert_chunk_size
        )
        l2_dequant = _dequantize_expert_chunks(
            l2_packed, l2_scale, args.expert_chunk_size
        )

    transformed_l1, transformed_l2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
        (l1_packed, l1_scale), (l2_packed, l2_scale),
    )
    cumulative_stats = torch.zeros(num_local_experts, dtype=torch.int, device="cuda")
    buffer.x[:allocation_tokens].copy_(x_fp8)
    buffer.x_sf[:allocation_tokens].copy_(x_sf)
    buffer.topk_idx[:allocation_tokens].copy_(topk_idx.to(torch.int32))
    buffer.topk_weights[:allocation_tokens].copy_(topk_weights.to(torch.float32))

    y_storage = torch.zeros(
        (allocation_tokens, hidden), dtype=torch.bfloat16, device="cuda"
    )
    first_outputs = {}
    y_kernel = y_storage[:m_tokens]
    for repeat_idx, current_m in enumerate(call_m_tokens):
        collective_m = [None] * num_ranks
        dist.all_gather_object(collective_m, current_m, group=group)
        repeat_key = tuple(collective_m)
        y_kernel = y_storage[:current_m]
        y_kernel.zero_()
        deep_gemm.nvfp4_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_stats,
            l1_global_scales=l1_global_scales,
            l2_global_scales=l2_global_scales,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
        torch.cuda.synchronize()
        if not torch.isfinite(y_kernel).all().item():
            raise AssertionError(
                f"rank={rank_idx}, M={current_m}, repeat={repeat_idx}: "
                "kernel produced non-finite values"
            )
        first_output = first_outputs.get(repeat_key)
        if first_output is None:
            first_outputs[repeat_key] = y_kernel.clone()
        elif repeat_idx + 1 == len(call_m_tokens):
            repeat_diff = (
                y_kernel.float() - first_output.float()
            ).abs().max().item()
            per_token_diff = (
                y_kernel.float() - first_output.float()
            ).abs().amax(dim=1)
            changed_tokens = (per_token_diff != 0).nonzero().flatten()
            print(
                f"rank={rank_idx} M={current_m} calls={len(call_m_tokens)} "
                f"last_vs_first_max_abs_diff={repeat_diff:.4e} "
                f"changed_token_count={changed_tokens.numel()} "
                f"first_changed_tokens={changed_tokens[:32].cpu().tolist()}",
                flush=True,
            )
            if repeat_diff > args.repeat_atol:
                raise AssertionError(
                    f"rank={rank_idx}, M={current_m}: repeated kernel output "
                    f"diff {repeat_diff} exceeds {args.repeat_atol}"
                )
        dist.barrier(group=group)

    if heterogeneous:
        print(
            f"rank={rank_idx} heterogeneous-M finite PASS: M={m_tokens}, "
            f"abs_max={y_kernel.abs().max().item():.4e}",
            flush=True,
        )
        buffer.destroy()
        return

    x_ref = (
        x_fp8.float().view(m_tokens, hidden // 128, 128)
        * x_sf.float().unsqueeze(2)
    ).view(m_tokens, hidden)
    y_ref = torch.zeros((m_tokens, hidden), device="cuda", dtype=torch.float32)
    for token_idx in range(m_tokens):
        for topk_i in range(num_topk):
            expert_idx = topk_idx[token_idx, topk_i].item()
            expert_rank = expert_idx // num_local_experts
            if expert_rank != rank_idx:
                continue
            local_expert_idx = expert_idx - rank_idx * num_local_experts
            assert 0 <= local_expert_idx < num_local_experts
            route_weight = topk_weights[token_idx, topk_i].item()
            l1_global_scale = 1.0 if l1_global_scales is None else l1_global_scales[local_expert_idx].item()
            l2_global_scale = 1.0 if l2_global_scales is None else l2_global_scales[local_expert_idx].item()
            l1_out = (l1_dequant[local_expert_idx].float() @ x_ref[token_idx]) * l1_global_scale
            gate, up = l1_out[:intermediate_hidden], l1_out[intermediate_hidden:]
            gate = gate.clamp(max=args.activation_clamp)
            up = up.clamp(min=-args.activation_clamp, max=args.activation_clamp)
            intermediate = _silu(gate) * up * route_weight
            y_ref[token_idx] += (l2_dequant[local_expert_idx].float() @ intermediate) * l2_global_scale

    if num_ranks > 1:
        dist.all_reduce(y_ref, op=dist.ReduceOp.SUM, group=group)

    finite = torch.isfinite(y_kernel).all().item()
    diff = y_kernel.float() - y_ref
    abs_max_diff = diff.abs().max().item()
    abs_mean_diff = diff.abs().mean().item()
    ref_abs_max = y_ref.abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(y_kernel.float(), y_ref, dim=-1)
    cosine_min = cosine.min().item()
    cosine_mean = cosine.mean().item()
    worst_token_idx = cosine.argmin().item()
    norm_ratio = (
        torch.linalg.vector_norm(y_kernel.float()) /
        torch.linalg.vector_norm(y_ref).clamp_min(1e-30)
    ).item()

    if rank_idx == 0:
        print(f"cum_stats: {cumulative_stats.cpu().tolist()}", flush=True)
        print(
            f"Kernel: abs_max={y_kernel.abs().max().item():.4e} "
            f"abs_mean={y_kernel.abs().mean().item():.4e} finite={finite}",
            flush=True,
        )
        print(
            f"Ref:    abs_max={y_ref.abs().max().item():.4e} "
            f"abs_mean={y_ref.abs().mean().item():.4e}",
            flush=True,
        )
        print(
            f"Diff: max_abs={abs_max_diff:.4e} "
            f"mean_abs={abs_mean_diff:.4e}",
            flush=True,
        )
        print(
            f"Per-token cosine sim: min={cosine_min:.4f} mean={cosine_mean:.4f}",
            flush=True,
        )
        print(f"Norm ratio: kernel/ref={norm_ratio:.4f}", flush=True)

    if cosine_min < args.cosine_min_threshold:
        kernel_worst = y_kernel[worst_token_idx].float()
        ref_worst = y_ref[worst_token_idx]
        print(
            f"rank={rank_idx} worst_token={worst_token_idx} "
            f"cosine={cosine_min:.6f} "
            f"kernel_norm={torch.linalg.vector_norm(kernel_worst).item():.6e} "
            f"ref_norm={torch.linalg.vector_norm(ref_worst).item():.6e} "
            f"diff_norm={torch.linalg.vector_norm(kernel_worst - ref_worst).item():.6e} "
            f"topk_weights={topk_weights[worst_token_idx].cpu().tolist()} "
            f"topk_idx={topk_idx[worst_token_idx].cpu().tolist()}",
            flush=True,
        )

    if not finite:
        raise AssertionError(f"M={m_tokens}, weight_scale={weight_scale:g}: kernel produced non-finite values")
    small_signal_abs_pass = (
        ref_abs_max <= args.small_signal_ref_abs_max
        and abs_max_diff <= args.small_signal_abs_max_threshold
        and abs_mean_diff <= args.small_signal_abs_mean_threshold
    )
    if small_signal_abs_pass:
        if rank_idx == 0:
            print(
                f"Small-signal absolute check PASS: ref_abs_max={ref_abs_max:.4e} "
                f"max_abs_diff={abs_max_diff:.4e} mean_abs_diff={abs_mean_diff:.4e}",
                flush=True,
            )
    elif cosine_mean < args.cosine_mean_threshold:
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: cosine_mean={cosine_mean:.4f} < {args.cosine_mean_threshold:.4f}"
        )
    if not small_signal_abs_pass and cosine_min < args.cosine_min_threshold:
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: cosine_min={cosine_min:.4f} < {args.cosine_min_threshold:.4f}"
        )
    if not small_signal_abs_pass and not (args.norm_ratio_min <= norm_ratio <= args.norm_ratio_max):
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: norm_ratio={norm_ratio:.4f} "
            f"outside [{args.norm_ratio_min:.4f}, {args.norm_ratio_max:.4f}]"
        )

    if rank_idx == 0:
        print(
            f"PASS M={m_tokens} weight_scale={weight_scale:g}: "
            f"global_scale_mode={global_scale_mode} "
            f"cosine_min={cosine_min:.4f} cosine_mean={cosine_mean:.4f}",
            flush=True,
        )

    buffer.destroy()


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    try:
        if args.num_sms is not None:
            deep_gemm.set_num_sms(args.num_sms)
        if rank_idx == 0:
            _run_dequant_unit_test()
            _run_cuda_dequant_lut_unit_test()
        dist.barrier(group=group)
        if get_arch_major() != 9:
            if rank_idx == 0:
                print(f"[SKIP] requires SM90, got SM{get_arch_major()}0", flush=True)
            return
        for weight_scale in args.weight_scales:
            for global_scale_mode in args.global_scale_modes:
                if args.rank_batches:
                    if len(args.rank_batches) != num_local_ranks:
                        raise ValueError(
                            "rank-batches must have one value per process"
                        )
                    _run_case(
                        args,
                        args.rank_batches[rank_idx],
                        weight_scale,
                        global_scale_mode,
                        rank_idx,
                        group,
                        heterogeneous=True,
                    )
                elif args.rank_batch_sequences:
                    if len(args.rank_batch_sequences) != num_local_ranks:
                        raise ValueError(
                            "rank-batch-sequences must have one value per process"
                        )
                    sequence = [
                        int(value)
                        for value in args.rank_batch_sequences[rank_idx].split(",")
                    ]
                    sequence_lengths = [None] * num_local_ranks
                    dist.all_gather_object(
                        sequence_lengths, len(sequence), group=group
                    )
                    if len(set(sequence_lengths)) != 1:
                        raise ValueError(
                            "all rank batch sequences must have equal length"
                        )
                    _run_case(
                        args,
                        sequence[0],
                        weight_scale,
                        global_scale_mode,
                        rank_idx,
                        group,
                        heterogeneous=True,
                        token_sequence=sequence,
                    )
                else:
                    for m_tokens in args.batches:
                        _run_case(
                            args,
                            m_tokens,
                            weight_scale,
                            global_scale_mode,
                            rank_idx,
                            group,
                        )
    finally:
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SM90 NVFP4 MegaMoE correctness gate")
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument(
        "--rank-batches",
        nargs="+",
        type=int,
        help="Use a different local token count on each rank.",
    )
    parser.add_argument(
        "--rank-batch-sequences",
        nargs="+",
        help="Comma-separated per-call token counts for each rank.",
    )
    parser.add_argument("--kernel-repeats", type=int, default=1)
    parser.add_argument("--repeat-atol", type=float, default=0.0)
    parser.add_argument("--route-destination-rank", type=int)
    parser.add_argument("--num-sms", type=int)
    parser.add_argument("--expert-chunk-size", type=int, default=8)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-scales", nargs="+", type=float, default=[100.0])
    parser.add_argument(
        "--global-scale-modes",
        nargs="+",
        choices=["none", "expert", "modelopt"],
        default=["modelopt"],
        help="Run with no global scales and/or per-expert non-unit L1/L2 global scales.",
    )
    parser.add_argument("--cosine-mean-threshold", type=float, default=0.995)
    parser.add_argument("--cosine-min-threshold", type=float, default=0.995)
    parser.add_argument("--norm-ratio-min", type=float, default=0.99)
    parser.add_argument("--norm-ratio-max", type=float, default=1.01)
    parser.add_argument("--small-signal-ref-abs-max", type=float, default=1e-4)
    parser.add_argument("--small-signal-abs-max-threshold", type=float, default=1e-4)
    parser.add_argument("--small-signal-abs-mean-threshold", type=float, default=2e-5)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    torch.multiprocessing.spawn(
        _worker,
        args=(parsed.num_processes, parsed),
        nprocs=parsed.num_processes,
        join=True,
    )
