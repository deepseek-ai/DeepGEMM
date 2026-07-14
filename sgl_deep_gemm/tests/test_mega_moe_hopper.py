"""SM90 (Hopper) MegaMoE fused-kernel and same-pipeline baseline benchmark.

This follows the structure of ``tests/test_mega_moe.py`` for the SM100 FP4
path, with the compute path changed to SM90 FP8:

* fused: calls ``deep_gemm.fp8_mega_moe`` (kernel symbol
  ``sm90_fp8_mega_moe_impl``) with weights transformed by
  ``transform_weights_for_mega_moe_sm90`` and a ``SymmBuffer``.
* baseline: DeepEP dispatch, two grouped FP8 GEMMs, Triton SwiGLU, and DeepEP
  combine with untransformed weights. The current SM90 grouped GEMM path accepts
  per-128-K L2 activation SF, while the fused SM90 MegaMoE L1 epilogue writes
  per-64-K L2 activation SF to avoid cross-CTA synchronization. This is a
  same-pipeline performance reference, not a bitwise correctness oracle.
* low-latency baseline (optional, ``--run-low-latency-baseline``): mirrors the
  sglang low-latency MoE pipeline (see
  ``sglang/srt/layers/moe/token_dispatcher/deepep.py::_DeepEPDispatcherImplLowLatency``):
  ``Buffer.low_latency_dispatch`` (use_fp8=True) -> per-expert masked-layout
  FP8 grouped GEMM -> masked SwiGLU + FP8 quant -> masked FP8 grouped GEMM ->
  ``Buffer.low_latency_combine`` (which applies topk weights internally). This
  is the canonical decode path used in production EP serving.
* fused-only sweep (optional, ``--fused-only-sweep``): replaces the old
  standalone SM90 benchmark harness. It sweeps token counts, measures
  only the fused SM90 kernel, and keeps the ``--ncu-profile-only`` /
  ``--local-rank-idx`` interface for single-rank NCU profiling.
* accuracy mode (optional, ``--accuracy``): runs the former layered SM90
  correctness suite with a PyTorch BF16/FP32 reference. It covers smoke,
  heuristic branches, shape sweeps, edge cases, and optional random stress.
* output: TFLOPS, overlap-adjusted TFLOPS, HBM GB/s, NVLink GB/s, fused time,
  reduction estimate, and ``t_baseline / t_fused``.
"""

import argparse
import math
import os
import random
import sys
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Tuple, List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getenv("DG_TEST_USE_SOURCE_TREE", "0") == "1" and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major

_missing_sgl_symbols = [
    name for name in ("fp8_mega_moe", "transform_weights_for_mega_moe_sm90")
    if not hasattr(deep_gemm, name)
]
if _missing_sgl_symbols:
    raise RuntimeError(
        "SM90 MegaMoE tests require the sgl-deep-gemm wheel built with "
        "`bash build_sgl_deep_gemm.sh`; missing symbols: "
        f"{', '.join(_missing_sgl_symbols)}"
    )

try:
    import deep_ep as _deep_ep
    _deep_ep_import_error = None
except Exception as ex:
    _deep_ep = None
    _deep_ep_import_error = ex


# Must match the template entry point in
# deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh so bench_kineto can
# select the fused MegaMoE GPU region from the trace.
SM90_KERNEL_NAME = "sm90_fp8_mega_moe_impl"


# Max finite value of FP8 e4m3fn; quantization uses amax / 448 as the scale.
FP8_E4M3_MAX = 448.0
# Triton >= 3 requires Python globals read by a JIT kernel to be tl.constexpr,
# otherwise compilation can fail with NameError. Host-side torch code still uses
# the plain float above.
_FP8_E4M3_MAX_TL = tl.constexpr(448.0)
L1_ACT_SF_GRAN = 128
FUSED_L2_ACT_SF_GRAN = 64
BASELINE_L2_ACT_SF_GRAN = 128
WEIGHT_SF_GRAN_MN = 128
WEIGHT_SF_GRAN_K = 128


# ============================================================================
# Section 1: Triton SwiGLU + FP8 quantization kernel.
# ----------------------------------------------------------------------------
# The baseline L2 path uses DeepGEMM SM90 grouped FP8 GEMM, which accepts
# per-128-K activation SF. The scale values still use the same power-of-two
# rounding as the fused epilogue to avoid adding an exact-FP32-scale difference.
# Input  x        : (M, 2*H) bf16, laid out as [gate_part | up_part].
# Input  topk_w   : (M,) fp32, optional.
# Output y        : (M, H) fp8_e4m3fn.
# Output y_sf     : (M, H / BLOCK_K) fp32, row-major.
# ============================================================================


@triton.jit
def _swiglu_apply_weight_to_fp8_kernel(
    x_ptr,
    topk_w_ptr,
    y_ptr,
    y_sf_ptr,
    M,
    H,  # Runtime shape
    stride_xm,
    stride_xn,  # x: (M, 2H) stride
    stride_ym,
    stride_yn,  # y: (M, H) stride
    stride_sfm,
    stride_sfk,  # y_sf: (M, H / BLOCK_K) stride
    clamp_value,  # Ignored when HAS_CLAMP=False
    HAS_TOPK: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,  # = num_per_channels
):
    # One program handles BLOCK_M tokens and one BLOCK_K column tile.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Row indices handled by this program.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Column indices inside the current K block, in the H dimension.
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # 1) Load gate from [0, H) and up from [H, 2H).
    # stride_xn is an element stride, so H + offs_k is also element-based.
    gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
    up_ptrs = x_ptr + offs_m[:, None] * stride_xm + (H + offs_k[None, :]) * stride_xn
    gate = tl.load(gate_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # 2) Optional clamp: one-sided for gate, two-sided for up.
    if HAS_CLAMP:
        gate = tl.minimum(gate, clamp_value)
        up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)

    # 3) SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up, accumulated in FP32.
    y = gate * tl.sigmoid(gate) * up

    # 4) Optional MoE weight scaling with a per-token scalar.
    if HAS_TOPK:
        w = tl.load(topk_w_ptr + offs_m, mask=mask_m, other=1.0)
        y = y * w[:, None]

    # 5) Per-row absmax in the current K block -> scale.
    amax = tl.max(tl.abs(y), axis=1)  # (BLOCK_M,)
    sf = tl.maximum(amax / _FP8_E4M3_MAX_TL, 1.0e-30)
    if USE_UE8M0_SCALE:
        # Match deep_gemm/common/math.cuh::get_e4m3_sf_and_sf_inv:
        # scale = 2 ** ceil(log2(amax / 448)).
        sf = tl.exp2(tl.ceil(tl.log2(sf)))

    # 6) Quantize to FP8 e4m3fn.
    y_fp8 = (y / sf[:, None]).to(tl.float8e4nv)

    # 7) Store y and sf.
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yn
    tl.store(y_ptrs, y_fp8, mask=mask_m[:, None])

    sf_ptrs = y_sf_ptr + offs_m * stride_sfm + pid_k * stride_sfk
    tl.store(sf_ptrs, sf, mask=mask_m)


def swiglu_apply_weight_to_fp8_triton(
    x: torch.Tensor,
    topk_weights: torch.Tensor | None,
    clamp_value: float | None = None,
    num_per_channels: int = BASELINE_L2_ACT_SF_GRAN,
    use_ue8m0_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU + FP8 quantization. Semantically equivalent to:
    gate, up = x[:, :H], x[:, H:]
    y = silu(gate.clamp(max=c)) * up.clamp(-c, c) * topk_w
    y_sf = y.view(M, H/np, np).abs().amax(-1) / 448
    if use_ue8m0_scale: y_sf = ceil_to_power_of_2(y_sf)
    y_fp8 = (y / y_sf.unsqueeze(-1)).to(fp8)
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.is_contiguous(), "This implementation expects contiguous x"
    M, two_H = x.shape
    H = two_H // 2
    assert H % num_per_channels == 0, f"H={H} must be divisible by {num_per_channels}"

    y = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=x.device)
    y_sf = torch.empty((M, H // num_per_channels), dtype=torch.float32, device=x.device)

    # BLOCK_M=16 keeps register pressure low for the Triton reference kernel.
    BLOCK_M = 16
    grid = (triton.cdiv(M, BLOCK_M), H // num_per_channels)

    # Triton still needs a valid pointer when HAS_TOPK=False; x is a placeholder.
    topk_ptr = topk_weights if topk_weights is not None else x

    _swiglu_apply_weight_to_fp8_kernel[grid](
        x,
        topk_ptr,
        y,
        y_sf,
        M,
        H,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_sf.stride(0),
        y_sf.stride(1),
        float(clamp_value) if clamp_value is not None else 0.0,
        HAS_TOPK=topk_weights is not None,
        HAS_CLAMP=clamp_value is not None,
        USE_UE8M0_SCALE=use_ue8m0_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_K=num_per_channels,
    )
    return y, y_sf


# ============================================================================
# Section 2: grouped weight block-(128, 128) FP8 quantization.
# ----------------------------------------------------------------------------
# SM90 m_grouped_fp8_gemm_nt_contiguous expects each (128, 128) weight block to
# share one FP32 SF, with K as the inner contiguous SF dimension (K-major).
# Unlike the SM100 FP4 path:
#   * deep_gemm.transform_sf_into_required_layout is not needed.
#   * SF is FP32, not packed UE8M0.
# ============================================================================


def _quantize_grouped_fp8_block_128_128(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(G, N, K) bf16 -> (G, N, K) fp8_e4m3fn plus FP32 block SF."""
    g, n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0, f"weight N={n}, K={k} must be multiples of 128"

    # Split (N, K) into (N/128, 128, K/128, 128) block interiors.
    w_view = w.view(g, n // 128, 128, k // 128, 128).float()

    # In-block absmax -> scale = amax / 448; clamp avoids all-zero scales.
    amax = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4)  # (G, N/128, K/128)
    sf = amax / FP8_E4M3_MAX

    # Divide by the owning block's SF before casting to FP8.
    w_fp8 = (w_view / sf.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return w_fp8.view(g, n, k).contiguous(), sf.contiguous()


# ============================================================================
# Section 3: layered accuracy reference and scenarios.
# ============================================================================


def _dequant_block_128_128(w_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantize_grouped_fp8_block_128_128. Returns fp32."""
    *prefix, n, k = w_fp8.shape
    assert n % 128 == 0 and k % 128 == 0
    w_view = w_fp8.float().view(*prefix, n // 128, 128, k // 128, 128)
    return (w_view * sf.unsqueeze(-1).unsqueeze(-3)).view(*prefix, n, k)


def _dequant_per_token_per_128_k(x_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Dequantize (M, K) fp8 with per-token, per-128-K float scales."""
    m, k = x_fp8.shape
    assert k % 128 == 0
    x_view = x_fp8.float().view(m, k // 128, 128)
    return (x_view * sf.unsqueeze(-1)).view(m, k)


def _swiglu_fp32(gate_up: torch.Tensor, clamp: float) -> torch.Tensor:
    """SwiGLU matching the fused SM90 path's clamp semantics."""
    n2 = gate_up.size(-1)
    half = n2 // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _reference_fused(
    x_fp8_local: torch.Tensor,
    x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor,
    topk_weights_local: torch.Tensor,
    l1_w_fp8: torch.Tensor,
    l1_w_sf: torch.Tensor,
    l2_w_fp8: torch.Tensor,
    l2_w_sf: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float,
) -> torch.Tensor:
    """PyTorch BF16/FP32 reference for this rank's fused output."""
    num_experts_per_rank = num_experts // num_ranks

    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)
    mg = x_fp8_g.size(0)

    local_size = torch.tensor([x_fp8_local.size(0)], device="cuda", dtype=torch.long)
    sizes_t = torch.empty(num_ranks, dtype=torch.long, device="cuda")
    dist.all_gather_into_tensor(sizes_t, local_size, group=group)
    sizes_list = sizes_t.tolist()
    assert sum(sizes_list) == mg

    l1_w_g = [torch.empty_like(l1_w_fp8) for _ in range(num_ranks)]
    l1_sf_g = [torch.empty_like(l1_w_sf) for _ in range(num_ranks)]
    l2_w_g = [torch.empty_like(l2_w_fp8) for _ in range(num_ranks)]
    l2_sf_g = [torch.empty_like(l2_w_sf) for _ in range(num_ranks)]
    dist.all_gather(l1_w_g, l1_w_fp8, group=group)
    dist.all_gather(l1_sf_g, l1_w_sf, group=group)
    dist.all_gather(l2_w_g, l2_w_fp8, group=group)
    dist.all_gather(l2_sf_g, l2_w_sf, group=group)
    l1_w_all = torch.stack(l1_w_g, dim=0)
    l1_sf_all = torch.stack(l1_sf_g, dim=0)
    l2_w_all = torch.stack(l2_w_g, dim=0)
    l2_sf_all = torch.stack(l2_sf_g, dim=0)

    combine_buf = torch.zeros(mg, num_topk, hidden, dtype=torch.float32, device="cuda")
    x_fp32 = _dequant_per_token_per_128_k(x_fp8_g, x_sf_g)

    chunk = 256
    for k in range(num_topk):
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx_full = mask.nonzero(as_tuple=False).squeeze(-1)
        for c0 in range(0, sel_idx_full.numel(), chunk):
            sel_idx = sel_idx_full[c0:c0 + chunk]
            eids = topk_idx_g[sel_idx, k]
            weights = topk_w_g[sel_idx, k]
            x_sel = x_fp32[sel_idx]

            dst_rank = (eids // num_experts_per_rank).long()
            dst_local = (eids % num_experts_per_rank).long()

            l1_w_sel = _dequant_block_128_128(
                l1_w_all[dst_rank, dst_local],
                l1_sf_all[dst_rank, dst_local],
            )
            l1_y = torch.einsum("sk,snk->sn", x_sel, l1_w_sel)
            del l1_w_sel

            l1_y = _swiglu_fp32(l1_y, activation_clamp) * weights.unsqueeze(-1)
            s, ih = l1_y.shape
            assert ih == intermediate_hidden and ih % 64 == 0
            l1_view = l1_y.view(s, ih // 64, 64)
            amax = l1_view.abs().amax(dim=-1).clamp(1e-4)
            sf2 = amax / FP8_E4M3_MAX
            l1_q = (l1_view / sf2.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
            l2_in = (l1_q * sf2.unsqueeze(-1)).view(s, ih)

            l2_w_sel = _dequant_block_128_128(
                l2_w_all[dst_rank, dst_local],
                l2_sf_all[dst_rank, dst_local],
            )
            l2_y = torch.einsum("sn,smn->sm", l2_in, l2_w_sel)
            del l2_w_sel

            combine_buf[sel_idx, k] = l2_y.to(torch.bfloat16).float()

    y_full_bf16 = combine_buf.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    return y_full_bf16[start:end].contiguous()


def _run_accuracy_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    diff_tol: float,
):
    num_max = cfg["num_max_tokens_per_rank"]
    num_tokens = cfg.get("num_tokens", num_max)
    hidden = cfg["hidden"]
    intermediate_hidden = cfg["intermediate_hidden"]
    num_experts = cfg["num_experts"]
    num_topk = cfg["num_topk"]
    masked_ratio = cfg.get("masked_ratio", 0.0)
    activation_clamp = cfg.get("activation_clamp", 10.0)
    fast_math = cfg.get("fast_math", True)

    assert num_experts % num_ranks == 0, (
        f"{name}: experts {num_experts} not divisible by ranks {num_ranks}"
    )
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

    verbose = bool(int(os.environ.get("DG_TEST_VERBOSE", "0")))

    def trace(stage: str):
        if verbose:
            print(f"[rank{rank_idx}] {name} :: {stage}", flush=True)

    trace("begin")
    torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)
    random.seed(rank_idx * 1000 + abs(hash(name)) % 1000)

    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    x_fp8 = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_weights_bf16)
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_weights_bf16)

    trace("weight_transform")
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        l1_weights, l2_weights
    )

    trace("alloc_symm_buffer")
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    cum_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")

    trace("copy_inputs")
    buffer.x[:num_tokens].copy_(x_fp8[0])
    buffer.x_sf[:num_tokens].copy_(x_fp8[1])
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)

    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    trace("launch_fused")
    deep_gemm.fp8_mega_moe(
        y_fused,
        transformed_l1,
        transformed_l2,
        buffer,
        cumulative_local_expert_recv_stats=cum_stats,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
        fast_math=fast_math,
    )
    torch.cuda.synchronize()

    trace("reference")
    y_ref = _reference_fused(
        x_fp8[0],
        x_fp8[1],
        topk_idx,
        topk_weights,
        l1_weights[0],
        l1_weights[1],
        l2_weights[0],
        l2_weights[1],
        rank_idx,
        num_ranks,
        group,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        activation_clamp,
    )

    diff = calc_diff(y_fused, y_ref)
    ok = diff < diff_tol
    dist_print(
        f"  [{name:<32}] diff={diff:.4f} (tol={diff_tol:.2f}) "
        f"{'OK' if ok else 'FAIL'}",
        once_in_node=True,
    )
    assert ok, f"{name}: diff={diff} >= tol={diff_tol}"
    if num_tokens > 0 and masked_ratio < 1.0:
        assert cum_stats.sum().item() >= 0

    buffer.destroy()
    dist.barrier()


_ACCURACY_SMOKE = dict(
    num_max_tokens_per_rank=64,
    num_tokens=64,
    hidden=512,
    intermediate_hidden=512,
    num_experts=8,
    num_topk=2,
)


def _accuracy_layer1_smoke() -> List[Tuple[str, Dict[str, Any]]]:
    return [("L1.smoke", dict(_ACCURACY_SMOKE))]


def _accuracy_layer2_heuristic_branches(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(
        hidden=1024,
        intermediate_hidden=1024,
        num_experts=8 * num_ranks,
        num_topk=2,
    )
    out: List[Tuple[str, Dict[str, Any]]] = []
    for tokens, label in [(64, "small"), (256, "midA"), (512, "midB"), (2048, "large")]:
        cfg = dict(base)
        cfg.update(num_max_tokens_per_rank=tokens, num_tokens=tokens)
        out.append((f"L2.heur.{label}.t{tokens}", cfg))
    return out


def _accuracy_layer3_shape_sweep(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    base_experts = 8 * num_ranks
    for hidden in (512, 2048):
        for ih in (512, 2048):
            for topk in (1, 2, 4):
                if topk > base_experts:
                    continue
                cfg = dict(
                    num_max_tokens_per_rank=128,
                    num_tokens=128,
                    hidden=hidden,
                    intermediate_hidden=ih,
                    num_experts=base_experts,
                    num_topk=topk,
                )
                out.append((f"L3.h{hidden}_ih{ih}_k{topk}", cfg))
    return out


def _accuracy_layer4_edges(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(
        num_max_tokens_per_rank=128,
        hidden=512,
        intermediate_hidden=512,
        num_experts=8 * num_ranks,
        num_topk=2,
    )
    out = []
    for masked_ratio in (0.0, 0.3, 0.7):
        cfg = dict(base)
        cfg.update(num_tokens=128, masked_ratio=masked_ratio)
        out.append((f"L4.mask{masked_ratio:.1f}", cfg))
    cfg = dict(base)
    cfg.update(num_tokens=128, masked_ratio=1.0)
    out.append(("L4.mask_all", cfg))
    for clamp in (1.0, 10.0, math.inf):
        cfg = dict(base)
        cfg.update(num_tokens=128, activation_clamp=clamp)
        out.append((f"L4.clamp{clamp}", cfg))
    for fast_math in (True, False):
        cfg = dict(base)
        cfg.update(num_tokens=128, fast_math=fast_math)
        out.append((f"L4.fm{int(fast_math)}", cfg))
    cfg = dict(base)
    cfg.update(num_tokens=0)
    out.append(("L4.tokens0", cfg))
    cfg = dict(base)
    cfg.update(num_tokens=base["num_max_tokens_per_rank"])
    out.append(("L4.tokens_max", cfg))
    return out


def _accuracy_layer5_stress(num_ranks: int, num_tests: int) -> List[Tuple[str, Dict[str, Any]]]:
    rng = random.Random(0xC0FFEE)
    out = []
    for i in range(num_tests):
        cfg = dict(
            num_max_tokens_per_rank=rng.choice([32, 64, 128, 256, 512]),
            hidden=rng.choice([512, 1024, 2048]),
            intermediate_hidden=rng.choice([512, 1024, 2048]),
            num_experts=8 * num_ranks,
            num_topk=rng.choice([1, 2, 4]),
            masked_ratio=rng.choice([0.0, 0.0, 0.3, 0.5]),
            activation_clamp=rng.choice([1.0, 10.0, math.inf]),
            fast_math=rng.choice([True, False]),
        )
        cfg["num_tokens"] = cfg["num_max_tokens_per_rank"]
        out.append((f"L5.rand{i:03d}", cfg))
    return out


def _run_accuracy_tests(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if get_arch_major() != 9:
        dist_print(
            f"[SKIP] test_mega_moe_hopper accuracy requires SM90; got SM{get_arch_major()}0",
            once_in_node=True,
        )
        dist.destroy_process_group()
        return

    layers: List[Tuple[str, Dict[str, Any]]] = []
    if 1 in args.layers:
        layers += _accuracy_layer1_smoke()
    if 2 in args.layers:
        layers += _accuracy_layer2_heuristic_branches(num_ranks)
    if 3 in args.layers:
        layers += _accuracy_layer3_shape_sweep(num_ranks)
    if 4 in args.layers:
        layers += _accuracy_layer4_edges(num_ranks)
    if 5 in args.layers:
        layers += _accuracy_layer5_stress(num_ranks, args.num_correctness_tests or 8)
    if args.filter:
        layers = [(name, cfg) for name, cfg in layers if args.filter in name]

    dist_print(
        f"SM90 MegaMoE accuracy plan: {len(layers)} scenarios across "
        f"layers {sorted(args.layers)} on {num_ranks} ranks",
        once_in_node=True,
    )

    failures: List[str] = []
    for name, cfg in layers:
        try:
            _run_accuracy_scenario(name, cfg, rank_idx, num_ranks, group, args.diff_tol)
        except AssertionError as ex:
            dist_print(f"  [{name}] FAIL: {ex}", once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    dist_print("", once_in_node=True)
    if failures:
        dist_print(
            f"FAILED {len(failures)}/{len(layers)} scenarios: {failures}",
            once_in_node=True,
        )
    else:
        dist_print(f"PASSED all {len(layers)} scenarios", once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


# ============================================================================
# Section 4: optional deep_ep import for dispatch/combine.
# ============================================================================


def _import_deep_ep():
    if _deep_ep is None:
        dist_print(f"Failed to import deep_ep: {_deep_ep_import_error}", once_in_node=True)
        return None
    return _deep_ep


class _DeepEPHandle:
    def __init__(self, raw_handle, psum_num_recv_tokens_per_expert: torch.Tensor):
        self.raw_handle = raw_handle
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert


class _DeepEPBufferCompat:
    """Compatibility shim for newer DeepEP versions that expose Buffer, not ElasticBuffer."""

    def __init__(self, deep_ep, group, num_nvl_bytes: int):
        self.buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=0,
            explicitly_destroy=True,
        )

    def dispatch(
        self,
        x,
        *,
        topk_idx,
        topk_weights,
        num_experts: int,
        expert_alignment: int,
        **_,
    ):
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = (
            self.buffer.get_dispatch_layout(topk_idx, num_experts)
        )
        recv_x, _, recv_topk_weights, num_recv_tokens_per_expert, raw_handle, event = self.buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
        )
        psum = torch.tensor(
            num_recv_tokens_per_expert, dtype=torch.int, device=topk_idx.device
        ).cumsum(dim=0, dtype=torch.int)
        return recv_x, None, recv_topk_weights, _DeepEPHandle(raw_handle, psum), event

    def combine(self, x, *, handle):
        raw_handle = handle.raw_handle if isinstance(handle, _DeepEPHandle) else handle
        return self.buffer.combine(x, handle=raw_handle)

    def barrier(self, use_comm_stream: bool = False):
        torch.cuda.synchronize()
        dist.barrier()

    def destroy(self):
        self.buffer.destroy()


def _make_deep_ep_buffer(deep_ep, group, num_max_tokens_per_rank, hidden, num_topk, sym_buffer_bytes):
    if hasattr(deep_ep, "ElasticBuffer"):
        return deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=True,
            explicitly_destroy=True,
            allow_multiple_reduction=False,
        )
    nvl_alignment = 2 * 1024 * 1024
    num_nvl_bytes = ((int(sym_buffer_bytes) + nvl_alignment - 1) // nvl_alignment) * nvl_alignment
    return _DeepEPBufferCompat(deep_ep, group, num_nvl_bytes=num_nvl_bytes)


def _make_deep_ep_low_latency_buffer(
    deep_ep, group, num_max_dispatch_tokens_per_rank, hidden, num_experts
):
    """Build a DeepEP ``Buffer`` configured for low-latency dispatch/combine.

    Mirrors the buffer construction used by sglang's
    ``_DeepEPDispatcherImplLowLatency`` (see
    ``sglang/srt/layers/moe/token_dispatcher/deepep.py``): RDMA bytes from
    ``get_low_latency_rdma_size_hint`` and one QP per local expert.
    """
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
    )
    return deep_ep.Buffer(
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // group.size(),
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )


# ----------------------------------------------------------------------------
# Masked SwiGLU + FP8 quantization (low-latency layout).
# ----------------------------------------------------------------------------
# DeepEP low-latency dispatch returns tokens packed as
#   x: [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, 2*IH]
# with a per-expert valid-token count ``masked_m[g]``. The masked GEMM does
# not care about the trailing junk rows, but to feed the L2 masked GEMM with
# correct scales we still need a per-token-per-128-K FP32 scale tensor with
# the same masked-layout convention. This kernel produces:
#   y    : [E, M, IH]                    fp8_e4m3fn
#   y_sf : [E, M, IH // BLOCK_K]         fp32 (row-major; DeepGEMM SM90 path
#                                              accepts this layout)
# Modeled on sglang's ``_silu_and_mul_post_quant_kernel`` in
# ``sglang/srt/layers/moe/ep_moe/kernels.py``.


@triton.jit
def _swiglu_masked_post_quant_kernel(
    x_ptr,
    stride_x_e,
    stride_x_m,
    stride_x_n,
    y_ptr,
    stride_y_e,
    stride_y_m,
    stride_y_n,
    y_sf_ptr,
    stride_sf_e,
    stride_sf_m,
    stride_sf_k,
    masked_m_ptr,
    H,
    clamp_value,
    HAS_CLAMP: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_k = tl.program_id(0)  # column tile within IH
    pid_m = tl.program_id(1)  # token-stripe within this expert
    pid_e = tl.program_id(2)  # expert

    num_token_stripes = tl.num_programs(1)
    num_valid_tokens = tl.load(masked_m_ptr + pid_e)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Element ptrs for one (expert, token_index, k_block).
    x_base = x_ptr + pid_e * stride_x_e + offs_k * stride_x_n
    y_base = y_ptr + pid_e * stride_y_e + offs_k * stride_y_n
    sf_base = y_sf_ptr + pid_e * stride_sf_e + pid_k * stride_sf_k

    for token in tl.range(pid_m, num_valid_tokens, num_token_stripes, num_stages=NUM_STAGES):
        gate = tl.load(x_base + token * stride_x_m).to(tl.float32)
        up = tl.load(x_base + token * stride_x_m + H * stride_x_n).to(tl.float32)

        if HAS_CLAMP:
            gate = tl.minimum(gate, clamp_value)
            up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)

        y = gate * tl.sigmoid(gate) * up

        amax = tl.max(tl.abs(y))
        sf = tl.maximum(amax / _FP8_E4M3_MAX_TL, 1.0e-30)
        if USE_UE8M0_SCALE:
            sf = tl.exp2(tl.ceil(tl.log2(sf)))

        y_fp8 = (y / sf).to(tl.float8e4nv)

        tl.store(y_base + token * stride_y_m, y_fp8)
        tl.store(sf_base + token * stride_sf_m, sf)


def swiglu_masked_post_quant_to_fp8(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    quant_group_size: int = BASELINE_L2_ACT_SF_GRAN,
    clamp_value: float | None = None,
    use_ue8m0_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU + per-(token, BLOCK_K) FP8 quant on masked-layout input.

    Input:
        x          : (E, M, 2*H) bf16, contiguous
        masked_m   : (E,) int, number of valid rows per expert
    Returns:
        y          : (E, M, H) fp8_e4m3fn
        y_sf       : (E, M, H // quant_group_size) fp32 (row-major)

    The MoE low-latency path applies topk weights inside
    ``low_latency_combine``, so this kernel does NOT multiply by topk weights.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.is_contiguous(), "Expects contiguous masked-layout input"
    assert x.dim() == 3 and x.shape[-1] % 2 == 0
    E, M, two_H = x.shape
    H = two_H // 2
    assert H % quant_group_size == 0
    assert masked_m.shape == (E,)

    y = torch.empty((E, M, H), dtype=torch.float8_e4m3fn, device=x.device)
    y_sf = torch.empty(
        (E, M, H // quant_group_size), dtype=torch.float32, device=x.device
    )

    BLOCK_K = quant_group_size
    # Heuristic similar to sglang's silu_and_mul_masked_post_quant_fwd.
    block_num_per_expert = 64 if E < 4 else 32

    grid = (H // BLOCK_K, block_num_per_expert, E)

    _swiglu_masked_post_quant_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y_sf,
        y_sf.stride(0),
        y_sf.stride(1),
        y_sf.stride(2),
        masked_m,
        H,
        float(clamp_value) if clamp_value is not None else 0.0,
        HAS_CLAMP=clamp_value is not None,
        USE_UE8M0_SCALE=use_ue8m0_scale,
        BLOCK_K=BLOCK_K,
        NUM_STAGES=4,
        num_warps=1,
    )
    return y, y_sf


# ============================================================================
# Section 5: CUDA event median timing, independent of tilelang.do_bench.
# ============================================================================


def _bench_cuda_events(
    fn, num_warmup: int = 5, num_repeat: int = 20, l2_flush_gb: float = 8.0
) -> float:
    """Return median runtime of fn in seconds."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(num_repeat):
        # Flush L2 to avoid optimistic timings from repeated cache hits.
        if l2_flush_gb > 0:
            free_bytes, _ = torch.cuda.mem_get_info()
            flush_bytes = min(int(l2_flush_gb * 1e9), int(free_bytes * 0.5))
            if flush_bytes >= 4:
                torch.empty(flush_bytes // 4, dtype=torch.int, device="cuda").zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times_ms.append(s.elapsed_time(e))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] / 1e3


# ============================================================================
# Section 6: fused-only sweep / NCU profile mode.
# ============================================================================


def _run_fused_only_config(
    args: argparse.Namespace,
    num_tokens: int,
    num_max_tokens_per_rank: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    rank_idx: int,
    group: dist.ProcessGroup,
):
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0, (
        f"num_experts={num_experts} must be divisible by num_ranks={num_ranks}"
    )
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64, (
        f"SM90 fused kernel requires intermediate_hidden <= 4096, got {intermediate_hidden}"
    )

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )

    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    x_fp8 = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_weights_bf16)
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_weights_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        l1_weights, l2_weights
    )

    cum_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")
    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    clamp_arg = args.activation_clamp if math.isfinite(args.activation_clamp) else None

    def run_fused():
        buffer.x[:num_tokens].copy_(x_fp8[0])
        buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        deep_gemm.fp8_mega_moe(
            y_fused,
            transformed_l1,
            transformed_l2,
            buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=clamp_arg,
            fast_math=bool(args.fast_math),
        )
        return y_fused

    if args.ncu_profile_only:
        dist_print(
            f"[NCU] tokens={num_tokens} hidden={hidden} ih={intermediate_hidden}",
            once_in_node=True,
        )
        run_fused()
        torch.cuda.synchronize()
        dist.barrier()
        buffer.destroy()
        return

    run_fused()
    dist.barrier()
    t_fused = bench_kineto(
        run_fused,
        SM90_KERNEL_NAME,
        barrier=lambda: dist.barrier(),
        num_tests=args.num_bench_tests,
        suppress_kineto_output=True,
    )

    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[
        (gathered_topk_idx < rank_idx * num_experts_per_rank)
        | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
    ] = -1
    local_expert_ids = gathered_topk_idx[gathered_topk_idx != -1]
    num_recv_tokens = int(local_expert_ids.numel())
    num_touched_experts = int(torch.unique(local_expert_ids).numel()) if num_recv_tokens else 0

    def safe_div(a, b):
        return float("nan") if b == 0 else a / b

    tflops = safe_div(
        2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12,
        t_fused,
    )
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden
        + num_touched_experts * hidden * intermediate_hidden
        + num_recv_tokens * hidden
        + num_recv_tokens * intermediate_hidden
        + num_recv_tokens * intermediate_hidden
        + num_recv_tokens * hidden * 2
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    dist_print(
        f" tokens={num_tokens:4d}  recv={num_recv_tokens:5d}  "
        f"experts={num_touched_experts:4d}  {t_fused * 1e6:7.1f} us  "
        f"{tflops:6.1f} TFLOPS  {hbm_gbs:6.0f} GB/s  (rank{rank_idx})",
        once_in_node=True,
    )

    dist.barrier()
    buffer.destroy()


def _run_fused_only_sweep(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(
            f"[SKIP] test_mega_moe_hopper fused-only sweep requires SM90; got SM{get_arch_major()}0",
            once_in_node=True,
        )
        dist.destroy_process_group()
        return

    batches = args.batches if args.batches is not None else [1, 2, 4, 8, 16, 32]
    if args.ncu_profile_only:
        batches = batches[:1]

    dist_print(
        f"SM90 MegaMoE fused-only sweep: ranks={num_ranks} hidden={args.hidden} "
        f"ih={args.intermediate_hidden} experts={args.num_experts} topk={args.num_topk} "
        f"masked_ratio={args.masked_ratio} fast_math={bool(args.fast_math)}",
        once_in_node=True,
    )

    num_max_tokens_per_rank = max(batches)
    for num_tokens in batches:
        _run_fused_only_config(
            args,
            num_tokens,
            num_max_tokens_per_rank,
            args.hidden,
            args.intermediate_hidden,
            args.num_experts,
            args.num_topk,
            num_ranks,
            rank_idx,
            group,
        )

    dist.barrier()
    dist.destroy_process_group()


# ============================================================================
# Section 7: per-rank comparison entry point.
# ============================================================================


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    if args.accuracy:
        _run_accuracy_tests(local_rank, num_local_ranks, args)
        return

    if args.fused_only_sweep:
        _run_fused_only_sweep(local_rank, num_local_ranks, args)
        return

    # Initialize distributed state; rank_idx is global rank and group is NCCL.
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(
            f"[SKIP] test_mega_moe_hopper requires SM90; got SM{get_arch_major()}0",
            once_in_node=True,
        )
        dist.destroy_process_group()
        return

    # Shape parameters, with names matching tests/test_mega_moe.py.
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = (
        max(
            0,
            args.num_max_tokens_per_rank
            - random.randint(0, args.num_max_removed_tokens),
        )
        if args.num_tokens == 0
        else args.num_tokens
    )
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0, (
        f"num_experts={num_experts} must be divisible by num_ranks={num_ranks}"
    )

    # SM90 fused-kernel shape constraints from csrc/apis/sm90_mega.hpp::fp8_mega_moe:
    #   * H and IH must be multiples of 128 (L1 input per-128-K SF and
    #     block-(128,128) weight SF).
    #   * IH / 64 <= 64, i.e. IH <= 4096, because l2_arrival_mask is uint64
    #     with one bit per 64-column block.
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64, (
        f"SM90 fused kernel requires intermediate_hidden <= 4096, got {intermediate_hidden}"
    )

    # ---- Create BF16 token and weight inputs ----
    # x: local tokens for this rank.
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    # L1 weight maps hidden -> 2*intermediate_hidden (gate and up packed).
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    # L2 weight maps intermediate_hidden -> hidden.
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )

    # Routing: scores -> topk_idx (M, K) and topk_weights (M, K).
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    # Keep separate recv counters so fused and baseline do not overwrite each other.
    cum_stats_fused = torch.zeros(
        (num_experts_per_rank,), dtype=torch.int, device="cuda"
    )
    cum_stats_baseline = cum_stats_fused.clone()

    # ---- BF16 -> FP8 quantization ----
    # x_fp8 is (token_fp8 (M, hidden), token_sf (M, hidden//128) row-major FP32).
    # SM90 expects FP32 SF, not packed UE8M0.
    x_fp8 = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )

    # Weight quantization: (G, N, K) bf16 -> FP8 e4m3fn plus block FP32 SF.
    # The DeepEP grouped-GEMM baseline uses these untransformed tuples directly.
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_weights_bf16)
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_weights_bf16)

    # Fused path: interleave gate/up along N for FP8 L1 weights; SF is unchanged.
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        l1_weights, l2_weights
    )

    # SwiGLU clamp: finite values enable clamp; inf maps to None and disables it.
    clamp_arg = args.activation_clamp if math.isfinite(args.activation_clamp) else None
    run_baseline_enabled = args.run_baseline or bool(args.check_output_diff)
    run_ll_baseline_enabled = bool(args.run_low_latency_baseline)

    # ---- M-dimension alignment for the grouped-GEMM baseline ----
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    # ---- Allocate fused SymmBuffer and output buffer ----
    sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def run_fused():
        # Match the SM100 test: DG_COMM_KERNEL_DEBUG=1 zeros the whole
        # sym_buffer at kernel exit, so inputs must be re-copied every call.
        sym_buffer.x[:num_tokens].copy_(x_fp8[0])
        sym_buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

        deep_gemm.fp8_mega_moe(
            y_fused,
            transformed_l1,
            transformed_l2,
            sym_buffer,
            cumulative_local_expert_recv_stats=cum_stats_fused,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=clamp_arg,
            fast_math=bool(args.fast_math),
        )
        return y_fused

    # ---- Print config ----
    dist_print("Config (SM90 fused MegaMoE):", once_in_node=True)
    dist_print(f" > Tokens: {num_tokens}/{num_max_tokens_per_rank}", once_in_node=True)
    dist_print(
        f" > Hidden: {hidden}, Intermediate: {intermediate_hidden}", once_in_node=True
    )
    dist_print(
        f" > Experts: {num_topk}/{num_experts} (per-rank: {num_experts_per_rank})",
        once_in_node=True,
    )
    dist_print(f" > Masked ratio: {args.masked_ratio}", once_in_node=True)
    dist_print(
        f" > Activation SF: fused L2 per-{FUSED_L2_ACT_SF_GRAN} FP32 pow2, "
        f"baseline L2 per-{BASELINE_L2_ACT_SF_GRAN} FP32 pow2 "
        f"(SM90 grouped-GEMM constraint)",
        once_in_node=True,
    )
    dist_print(
        f" > Baseline: {'enabled' if run_baseline_enabled else 'disabled'}",
        once_in_node=True,
    )
    dist_print(
        f" > Low-latency baseline: {'enabled' if run_ll_baseline_enabled else 'disabled'}",
        once_in_node=True,
    )
    dist_print(
        f" > Buffer: {sym_buffer.buffer.nbytes / 2**30:.3f} GiB", once_in_node=True
    )
    dist_print(once_in_node=True)

    # Match tests/test_mega_moe.py: NCU mode runs only the fused kernel to avoid
    # baseline noise in the profile.
    if args.ncu_profile_only:
        dist_print("Run fused SM90 mega-MoE kernel:", once_in_node=True)
        y = run_fused()
        torch.cuda.synchronize()
        assert y.shape == (num_tokens, hidden) and y.dtype == torch.bfloat16
        dist_print(" > Done, exiting", once_in_node=True)
        dist.barrier()
        sym_buffer.destroy()
        dist.destroy_process_group()
        return

    # ---- Allocate DeepEP buffer for the baseline ----
    deep_ep = (
        _import_deep_ep()
        if (run_baseline_enabled or run_ll_baseline_enabled)
        else None
    )
    ep_buffer = None
    if deep_ep is not None and run_baseline_enabled:
        ep_buffer = _make_deep_ep_buffer(
            deep_ep,
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            sym_buffer.buffer.nbytes,
        )

    # ---- Allocate DeepEP buffer for the low-latency baseline ----
    # Low-latency mode requires its own ``Buffer(low_latency_mode=True, ...)``
    # with ``num_qps_per_rank == num_local_experts`` and RDMA bytes sized via
    # ``get_low_latency_rdma_size_hint``. See sglang's
    # ``DeepEPBuffer.get_deepep_buffer`` for the canonical setup.
    ll_buffer = None
    if deep_ep is not None and run_ll_baseline_enabled:
        ll_buffer = _make_deep_ep_low_latency_buffer(
            deep_ep,
            group,
            num_max_tokens_per_rank,
            hidden,
            num_experts,
        )

    # ----------------------------------------------------------------
    # Baseline body: dispatch -> L1 GEMM -> SwiGLU+quantize -> L2 GEMM -> combine.
    # It uses the same FP8 weights and FP32 block-(128,128) SF as the fused path,
    # but without the fused-only gate/up interleave.
    # ----------------------------------------------------------------
    def run_baseline():
        recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
            x_fp8,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cum_stats_baseline,
            num_experts=num_experts,
            expert_alignment=alignment,
            do_cpu_sync=False,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=False,  # SM90: row-major float SF
        )
        n = recv_x[0].size(0)

        # L1 GEMM: FP8 token @ FP8 W1 -> BF16 intermediate activation (gate||up).
        l1_y = torch.empty(
            (n, intermediate_hidden * 2), dtype=torch.bfloat16, device="cuda"
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            recv_x,
            l1_weights,
            l1_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )

        # Triton SwiGLU + FP8 quantization, including topk weight scaling.
        # The fused SM90 MegaMoE L2 activation SF is per-64-K. The current
        # DeepGEMM SM90 grouped GEMM supports only per-128-K activation SF, so
        # the baseline uses per-128-K FP32 scales with the same power-of-two
        # rounding rule as the fused epilogue.
        l1_y = swiglu_apply_weight_to_fp8_triton(
            x=l1_y,
            topk_weights=recv_topk_weights,
            clamp_value=clamp_arg,
            num_per_channels=BASELINE_L2_ACT_SF_GRAN,
            use_ue8m0_scale=True,
        )

        # L2 GEMM: FP8 intermediate activation @ FP8 W2 -> BF16.
        l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device="cuda")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            l1_y,
            l2_weights,
            l2_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )

        # DeepEP combine: gather each token's topk expert outputs back to source rank.
        return ep_buffer.combine(l2_y, handle=handle)[0]

    # ----------------------------------------------------------------
    # Low-latency baseline body. Mirrors the sglang
    # ``_DeepEPDispatcherImplLowLatency`` pipeline:
    #   1. ``low_latency_dispatch(use_fp8=True)`` -> per-expert packed FP8 tokens
    #      with shape ``[E_local, M_max, hidden]`` plus FP32 scales
    #      ``[E_local, M_max, hidden // 128]`` and per-expert ``masked_m``.
    #   2. Masked grouped FP8 GEMM with L1 weights.
    #   3. Masked SwiGLU + per-128-K FP8 quantize. (topk weights are NOT
    #      applied here — ``low_latency_combine`` applies them internally.)
    #   4. Masked grouped FP8 GEMM with L2 weights.
    #   5. ``low_latency_combine`` (reduces with topk weights).
    # ----------------------------------------------------------------
    if run_ll_baseline_enabled:
        M_max_ll = num_max_tokens_per_rank * num_ranks
        # Expected per-expert mean of ``masked_m`` after dispatch. Used as the
        # ``expected_m`` hint for the DeepGEMM masked kernel selector.
        expected_m_ll = max(
            1,
            (num_max_tokens_per_rank * num_ranks * num_topk + num_experts - 1)
            // num_experts,
        )
        # Pre-allocate per-call output buffers; the masked GEMM writes into them
        # in place and ignores rows past ``masked_m[g]``.
        ll_l1_y = torch.empty(
            (num_experts_per_rank, M_max_ll, intermediate_hidden * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ll_l2_y = torch.empty(
            (num_experts_per_rank, M_max_ll, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ll_combined = torch.empty(
            (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
        )
        # DeepEP low-latency dispatch requires int64 topk indices.
        topk_idx_ll = topk_idx.to(torch.int64)

    def run_baseline_low_latency():
        # 1) Low-latency dispatch with FP8 cast.
        (recv_x_data, recv_x_sf), masked_m, ll_handle, event, hook = (
            ll_buffer.low_latency_dispatch(
                x_bf16,
                topk_idx_ll,
                num_max_tokens_per_rank,
                num_experts,
                use_fp8=True,
                round_scale=False,
                use_ue8m0=False,
                async_finish=False,
                return_recv_hook=False,
            )
        )

        # 2) L1 masked grouped FP8 GEMM:
        #    (E_local, M_max, hidden) @ (E_local, 2*IH, hidden)^T -> (E_local, M_max, 2*IH).
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (recv_x_data, recv_x_sf),
            l1_weights,
            ll_l1_y,
            masked_m,
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

        # 3) Masked SwiGLU + per-128-K FP8 quant. Topk weights are NOT applied
        #    here — they are reduced inside ``low_latency_combine``.
        l1_fp8, l1_sf = swiglu_masked_post_quant_to_fp8(
            ll_l1_y,
            masked_m,
            quant_group_size=BASELINE_L2_ACT_SF_GRAN,
            clamp_value=clamp_arg,
            use_ue8m0_scale=False,
        )

        # 4) L2 masked grouped FP8 GEMM:
        #    (E_local, M_max, IH) @ (E_local, H, IH)^T -> (E_local, M_max, H).
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (l1_fp8, l1_sf),
            l2_weights,
            ll_l2_y,
            masked_m,
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

        # 5) Low-latency combine: per-token weighted reduction across topk
        #    expert replicas; outputs (num_tokens, hidden) bf16.
        combined_x, event, hook = ll_buffer.low_latency_combine(
            ll_l2_y,
            topk_idx_ll,
            topk_weights,
            ll_handle,
            use_logfmt=False,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
            out=ll_combined,
        )
        return combined_x

    # ---- Run once to check fused and optional baseline paths ----
    y = run_fused()
    assert y.shape == (num_tokens, hidden) and y.dtype == torch.bfloat16, (
        f"unexpected fused output shape/dtype: shape={y.shape}, dtype={y.dtype}"
    )
    if ep_buffer is not None:
        out_b = run_baseline()
        assert out_b.shape == (num_tokens, hidden) and out_b.dtype == torch.bfloat16, (
            f"unexpected baseline output shape/dtype: shape={out_b.shape}, dtype={out_b.dtype}"
        )
        if args.check_output_diff:
            diff = (y.float() - out_b.float()).abs()
            denom = out_b.float().abs().mean().clamp_min(1e-12)
            dist_print(
                "Output diff (fused vs per-128 baseline):", once_in_node=True
            )
            dist_print(
                f" > max_abs={diff.max().item():.6e}, "
                f"mean_abs={diff.mean().item():.6e}, "
                f"mean_abs/mean_ref={diff.mean().div(denom).item():.6e}",
                once_in_node=True,
            )
            dist_print(once_in_node=True)
    if ll_buffer is not None:
        out_ll = run_baseline_low_latency()
        assert out_ll.shape == (num_tokens, hidden) and out_ll.dtype == torch.bfloat16, (
            f"unexpected LL baseline output shape/dtype: shape={out_ll.shape}, dtype={out_ll.dtype}"
        )
        if args.check_output_diff:
            diff = (y.float() - out_ll.float()).abs()
            denom = out_ll.float().abs().mean().clamp_min(1e-12)
            dist_print(
                "Output diff (fused vs low-latency baseline):", once_in_node=True
            )
            dist_print(
                f" > max_abs={diff.max().item():.6e}, "
                f"mean_abs={diff.mean().item():.6e}, "
                f"mean_abs/mean_ref={diff.mean().div(denom).item():.6e}",
                once_in_node=True,
            )
            dist_print(once_in_node=True)

    # ---- Count tokens routed to this rank and touched local experts ----
    # Gather all topk_idx tensors and mark entries outside this rank's local
    # expert range as -1. Remaining entries are routed (token, slot) pairs.
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[
        (gathered_topk_idx < rank_idx * num_experts_per_rank)
        | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
    ] = -1
    local_expert_ids = gathered_topk_idx[gathered_topk_idx != -1]
    num_recv_tokens = int(local_expert_ids.numel())
    num_touched_experts = int(torch.unique(local_expert_ids).numel())

    # ---- benchmark ----
    # Fused: bench_kineto selects the sm90_fp8_mega_moe_impl GPU region only.
    t_fused = bench_kineto(
        run_fused,
        SM90_KERNEL_NAME,
        num_tests=args.num_bench_tests,
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False)
        if ep_buffer is not None
        else dist.barrier(),
        trace_path=(
            f"{args.dump_profile_traces}/mega_moe_hopper_rank{rank_idx}.json"
            if args.dump_profile_traces
            else None
        ),
    )
    # Baseline: use CUDA event median timing for consistency across SM90 setups.
    t_baseline = (
        _bench_cuda_events(
            run_baseline,
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
        )
        if ep_buffer is not None
        else 0.0
    )
    # Low-latency baseline timing (same CUDA-event median methodology).
    t_baseline_ll = (
        _bench_cuda_events(
            run_baseline_low_latency,
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
        )
        if ll_buffer is not None
        else 0.0
    )

    def safe_div(a, b):
        return float("nan") if b == 0 else a / b

    # End-to-end TFLOPS: three matmuls (L1 gate, L1 up, L2), each 2*M*N*K.
    tflops = safe_div(
        2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused
    )

    # HBM byte estimate (SM90 weights are FP8 = 1B/elem, unlike SM100 FP4).
    l1_weight_bytes = num_touched_experts * intermediate_hidden * 2 * hidden
    l2_weight_bytes = num_touched_experts * hidden * intermediate_hidden
    l1_weight_sf_bytes = (
        num_touched_experts
        * (intermediate_hidden * 2 // WEIGHT_SF_GRAN_MN)
        * (hidden // WEIGHT_SF_GRAN_K)
        * 4
    )
    l2_weight_sf_bytes = (
        num_touched_experts
        * (hidden // WEIGHT_SF_GRAN_MN)
        * (intermediate_hidden // WEIGHT_SF_GRAN_K)
        * 4
    )
    l1_input_sf_bytes = num_recv_tokens * (hidden // L1_ACT_SF_GRAN) * 4
    l2_act_sf_bytes = (
        num_recv_tokens * (intermediate_hidden // FUSED_L2_ACT_SF_GRAN) * 4
    )
    num_hbm_bytes = (
        l1_weight_bytes
        + l2_weight_bytes  # weights (FP8)
        + l1_weight_sf_bytes
        + l2_weight_sf_bytes  # weight SF (FP32)
        + num_recv_tokens * hidden
        + l1_input_sf_bytes  # L1 input read (FP8 + SF)
        + num_recv_tokens * intermediate_hidden
        + l2_act_sf_bytes  # L1 output write (FP8 + SF)
        + num_recv_tokens * intermediate_hidden
        + l2_act_sf_bytes  # L2 input read (FP8 + SF)
        + num_recv_tokens * hidden * 2  # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pulls token + input SF + topk weight; combine writes BF16.
    num_nvlink_bytes = num_recv_tokens * (hidden + hidden // 32 + 4 + hidden * 2)
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Serial lower bound for combine reduction, using 6.5e12 B/s as an estimate.
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Overlap adjustment: remove the non-overlapped serial reduction estimate.
    approx_factor = t_fused / max(t_fused - t_reduction, 1e-12)

    # Baseline uses the same FLOPs and HBM byte estimate, with t_baseline.
    tflops_baseline = safe_div(
        2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_baseline
    )
    hbm_gbs_baseline = safe_div(num_hbm_bytes / 1e9, t_baseline)
    nvlink_gbs_baseline = safe_div(num_nvlink_bytes / 1e9, t_baseline)
    # Low-latency baseline pads each expert's activation to ``M_max_ll``, so
    # the weights are streamed once per expert regardless of routing. NVLink
    # bytes match the normal-mode baseline (same per-routed-token volume).
    tflops_baseline_ll = safe_div(
        2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_baseline_ll
    )
    hbm_gbs_baseline_ll = safe_div(num_hbm_bytes / 1e9, t_baseline_ll)
    nvlink_gbs_baseline_ll = safe_div(num_nvlink_bytes / 1e9, t_baseline_ll)

    def fmt_perf_line(
        name: str,
        t: float,
        compute_tflops: float,
        hbm_gbs_: float,
        nvlink_gbs_: float,
        reduction_us: float | None = None,
        speedup: float | None = None,
    ) -> str:
        reduction = f"{reduction_us:13.1f}" if reduction_us is not None else f"{'-':>13}"
        speedup_text = (
            f"{speedup:6.2f}x {'fused faster' if speedup > 1 else 'baseline faster'}"
            if speedup is not None else
            f"{'-':>21}"
        )
        return (
            f" > {name:<10} {rank_idx:2d}/{num_ranks:<2d} "
            f"{num_recv_tokens:12d} "
            f"{num_touched_experts:14d} | "
            f"{compute_tflops:15.0f} "
            f"{hbm_gbs_:9.0f} "
            f"{nvlink_gbs_:9.0f} "
            f"{t * 1e6:9.0f} "
            f"{reduction} "
            f"{speedup_text}"
        )

    dist_print("Performance:", once_in_node=True)
    dist_print(
        " > kind       EP    recv_tokens active_experts | "
        "compute(TFLOPS) HBM(GB/s) NVL(GB/s)  time(us) reduction(us) speedup",
        once_in_node=True,
    )
    dist_print(
        fmt_perf_line(
            "[fused]",
            t_fused,
            tflops * approx_factor,
            hbm_gbs * approx_factor,
            nvlink_gbs * approx_factor,
            reduction_us=t_reduction * 1e6,
        )
    )
    if ep_buffer is not None:
        speedup = safe_div(t_baseline, t_fused)
        dist_print(
            fmt_perf_line(
                "[baseline]",
                t_baseline,
                tflops_baseline,
                hbm_gbs_baseline,
                nvlink_gbs_baseline,
                speedup=speedup,
            )
        )
    else:
        reason = (
            "disabled; pass --run-baseline or --check-output-diff to compare"
            if not run_baseline_enabled
            else "deep_ep unavailable"
        )
        dist_print(f" > [baseline] ({reason})", once_in_node=True)
    if ll_buffer is not None:
        speedup_ll = safe_div(t_baseline_ll, t_fused)
        dist_print(
            fmt_perf_line(
                "[ll_base]",
                t_baseline_ll,
                tflops_baseline_ll,
                hbm_gbs_baseline_ll,
                nvlink_gbs_baseline_ll,
                speedup=speedup_ll,
            )
        )
    elif run_ll_baseline_enabled:
        dist_print(" > [ll_base] (deep_ep unavailable)", once_in_node=True)

    # ---- Cleanup ----
    dist.barrier()
    sym_buffer.destroy()
    if ep_buffer is not None:
        ep_buffer.destroy()
    if ll_buffer is not None:
        ll_buffer.destroy()
    dist.destroy_process_group()


# ============================================================================
# Section 8: argparse + spawn.
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SM90 MegaMoE: fused (deep_gemm.fp8_mega_moe) vs DeepEP+grouped-FP8 baseline"
    )

    # Resources.
    parser.add_argument(
        "--ncu-profile-only",
        action="store_true",
        help="Run the fused SM90 kernel once for NCU/Nsight profiling",
    )
    parser.add_argument(
        "--fused-only-sweep",
        action="store_true",
        help="Run the fused-only token sweep benchmark mode",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run the layered SM90 accuracy suite instead of benchmark modes",
    )
    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of spawned processes, one per GPU"
    )
    parser.add_argument(
        "--local-rank-idx",
        type=int,
        default=None,
        help="Local rank for single-process mode, useful for external launchers/NCU",
    )

    # Model shape.
    # SM90 fused kernel requires intermediate_hidden <= 4096.
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=0,
        help="Actual per-rank token count; 0 means num-max-tokens-per-rank",
    )
    parser.add_argument(
        "--num-max-removed-tokens",
        type=int,
        default=0,
        help="Max random token removals per rank when num-tokens is 0",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
        help="Token counts for --fused-only-sweep (default: 1 2 4 8 16 32)",
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument(
        "--intermediate-hidden",
        type=int,
        default=3072,
        help="Intermediate dimension, constrained to <= 4096 by SM90 l2_arrival_mask",
    )
    parser.add_argument(
        "--activation-clamp",
        type=float,
        default=10.0,
        help="Clamp threshold for gate/up before SwiGLU; pass inf to disable",
    )
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument(
        "--masked-ratio",
        type=float,
        default=0.0,
        help="Randomly mask some topk expert selections to test sparse routing edges",
    )
    parser.add_argument(
        "--fast-math",
        type=int,
        default=1,
        help="Whether fused SwiGLU uses fast math (0/1)",
    )

    # Timing.
    parser.add_argument(
        "--num-bench-tests",
        "--num-tests",
        dest="num_bench_tests",
        type=int,
        default=30,
        help="Number of bench_kineto iterations for the fused kernel",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=5, help="baseline cuda events warmup"
    )
    parser.add_argument(
        "--num-repeat", type=int, default=20, help="Baseline CUDA event timing iterations"
    )
    parser.add_argument(
        "--l2-flush-gb",
        type=float,
        default=8.0,
        help="Temporary write size used to flush L2 before baseline timing; 0 disables it",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Enable the DeepEP+grouped-FP8 baseline; disabled by default",
    )
    parser.add_argument(
        "--run-low-latency-baseline",
        action="store_true",
        help=(
            "Enable the sglang low-latency baseline "
            "(DeepEP low_latency_dispatch -> masked grouped FP8 GEMM -> masked "
            "SwiGLU+FP8 quant -> masked FP8 GEMM -> low_latency_combine); "
            "disabled by default"
        ),
    )
    parser.add_argument(
        "--check-output-diff",
        type=int,
        default=0,
        help="If nonzero, print fused vs per-128 baseline output differences",
    )
    parser.add_argument(
        "--dump-profile-traces",
        type=str,
        default="",
        help="If nonempty, write one fused Chrome trace per rank to this directory",
    )

    # Accuracy mode.
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Accuracy layers to run with --accuracy (1..5); default: 1 2 3 4",
    )
    parser.add_argument(
        "--num-correctness-tests",
        type=int,
        default=None,
        help="Layer-5 random stress scenario count for --accuracy",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Substring filter for --accuracy scenario names",
    )
    parser.add_argument(
        "--diff-tol",
        type=float,
        default=0.07,
        help="calc_diff tolerance for --accuracy; default: 0.07",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop --accuracy on the first failing scenario",
    )

    args = parser.parse_args()

    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: external launcher sets MASTER_ADDR/PORT/WORLD_SIZE/RANK.
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Multi-process mode: one process per GPU; test() creates the NCCL group.
        torch.multiprocessing.spawn(
            test, args=(args.num_processes, args), nprocs=args.num_processes
        )
