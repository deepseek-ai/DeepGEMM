"""SM90 (Hopper) MegaMoE benchmark / NCU-profile harness.

Mirrors ``tests/test_mega_moe.py``'s ``--ncu-profile-only`` /
``--local-rank-idx`` interface so the same ``scripts/run_ncu_mega_moe.sh``
pattern can drive it for SM90.

In normal (non-NCU) mode it runs a list of ``num_tokens`` values (default:
1, 2, 4, 8, 16, 32) and reports per-call kernel time via the same
``bench_kineto`` helper used by the SM100 perf test, plus a rough TFLOPS /
HBM GB/s figure useful for tracking optimisation deltas.
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

import triton
import triton.language as tl

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major



# ============================================================================
# Inlined DeepEP V1 (classic ``deep_ep.Buffer``) contiguous baseline.
# Classic intranode contiguous dispatch + 2 grouped FP8 GEMMs + Triton
# SwiGLU/FP8 quant + classic combine. V1 combine reduces WITHOUT weights,
# so the topk weight is applied in the SwiGLU quant step.
# ============================================================================

# FP8 e4m3fn max representable value.
# Newer Triton (>= 3.x) requires Python globals read by a jit kernel to be
# ``tl.constexpr`` instances, else a compile-time NameError.
_FP8_E4M3_MAX_TL = tl.constexpr(448.0)
BASELINE_L2_ACT_SF_GRAN = 128


# ============================================================================
# Module 1: Triton SwiGLU + FP8 quant (contiguous layout, applies 1D topk weight)
# ----------------------------------------------------------------------------
# Input x: (M, 2*H) bf16 with inner [gate | up]; optional per-row topk weight;
# output y: (M, H) fp8_e4m3fn + y_sf: (M, H/BLOCK_K) fp32 row-major.
# ============================================================================


@triton.jit
def _swiglu_apply_weight_to_fp8_kernel(
    x_ptr,
    topk_w_ptr,
    y_ptr,
    y_sf_ptr,
    M,
    H,  # runtime shape
    stride_xm,
    stride_xn,  # stride of x: (M, 2H)
    stride_ym,
    stride_yn,  # stride of y: (M, H)
    stride_sfm,
    stride_sfk,  # stride of y_sf: (M, H/BLOCK_K)
    clamp_value,  # meaningless when HAS_CLAMP=False
    HAS_TOPK: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,  # = num_per_channels
):
    # One program handles (BLOCK_M tokens) x (the BLOCK_K columns of the pid_k-th K-block)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Row indices: this program owns [pid_m*BLOCK_M, pid_m*BLOCK_M+BLOCK_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Column indices within the current K-block (in the H dimension, not 2H)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # ---- 1) load gate (first half [0, H)) and up (second half [H, 2H)) of x ----
    # Note stride_xn is the element stride (usually == 1), and the H + offs_k offset is in elements
    gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
    up_ptrs = x_ptr + offs_m[:, None] * stride_xm + (H + offs_k[None, :]) * stride_xn
    gate = tl.load(gate_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # ---- 2) optional clamp (matching the tilelang impl: gate one-sided max, up two-sided) ----
    if HAS_CLAMP:
        gate = tl.minimum(gate, clamp_value)
        up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)

    # ---- 3) SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up (accumulated in FP32) ----
    y = gate * tl.sigmoid(gate) * up

    # ---- 4) optional MoE weight scaling (per-token scalar) ----
    if HAS_TOPK:
        w = tl.load(topk_w_ptr + offs_m, mask=mask_m, other=1.0)
        y = y * w[:, None]

    # ---- 5) per-row absmax -> scale within the current K-block ----
    amax = tl.max(tl.abs(y), axis=1)  # (BLOCK_M,)
    sf = tl.maximum(amax / _FP8_E4M3_MAX_TL, 1.0e-30)
    if USE_UE8M0_SCALE:
        # Matching deep_gemm/common/math.cuh::get_e4m3_sf_and_sf_inv:
        # scale = 2 ** ceil(log2(amax / 448)).
        sf = tl.exp2(tl.ceil(tl.log2(sf)))

    # ---- 6) quantize to FP8 e4m3fn ----
    y_fp8 = (y / sf[:, None]).to(tl.float8e4nv)

    # ---- 7) write back y and sf ----
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
    """SwiGLU + FP8 quantization. Semantically equivalent to the PyTorch reference:
    gate, up = x[:, :H], x[:, H:]
    y = silu(gate.clamp(max=c)) * up.clamp(-c, c) * topk_w
    y_sf = y.view(M, H/np, np).abs().amax(-1) / 448
    if use_ue8m0_scale: y_sf = ceil_to_power_of_2(y_sf)
    y_fp8 = (y / y_sf.unsqueeze(-1)).to(fp8)
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.is_contiguous(), "this implementation assumes x is contiguous to avoid stride miscalculation"
    M, two_H = x.shape
    H = two_H // 2
    assert H % num_per_channels == 0, f"H={H} must be a multiple of {num_per_channels}"

    y = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=x.device)
    y_sf = torch.empty((M, H // num_per_channels), dtype=torch.float32, device=x.device)

    # BLOCK_M = 16: each program handles 16 tokens x 128 columns, low register pressure and easy to tune
    BLOCK_M = 16
    grid = (triton.cdiv(M, BLOCK_M), H // num_per_channels)

    # When HAS_TOPK=False we still must pass a valid pointer (Triton disallows nullptr); use x as a placeholder
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
# Module 2: V1 contiguous pipeline.
# ============================================================================


def _build_expand_indices(recv_topk_idx, num_recv_tokens_per_expert_list, device):
    """Build the index plan that expands V1's COMPACT per-received-token recv_x
    into the EXPERT-CONTIGUOUS, alignment-padded grouped layout the grouped GEMM
    needs (i.e. replicate by hand what V2 do_expand=True does internally).

    V1 facts (measured + legacy.py/test_intranode.py):
      - recv_x is COMPACT: [n_recv, hidden] (one row per received token; NOT
        expert-grouped, NOT padded). n_recv == rows routed to this rank.
      - recv_topk_idx is [n_recv, num_topk], expert ids REBASED to LOCAL experts
        (0..E_local-1), -1 for slots routing to other ranks.
      - num_recv_tokens_per_expert_list is PADDED per-local-expert counts (each
        rounded up to expert_alignment). N_grouped = sum(padded counts).

    Returns (row_to_token [N_grouped] int64 with -1 for pad rows,
             row_w        [N_grouped] float topk weight per grouped row (0 for pad),
             psum         [E_local] int32 cumsum of padded counts).
    The grouped layout: expert e occupies rows [psum[e-1], psum[e]); within it
    the first (real-count) rows are the tokens selecting e (in token order),
    remaining rows are padding.
    """
    padded = torch.tensor(list(num_recv_tokens_per_expert_list),
                          dtype=torch.long, device=device)            # [E]
    E = padded.numel()
    psum = padded.cumsum(0).to(torch.int32)                            # [E]
    N = int(padded.sum().item())
    n_recv, num_topk = recv_topk_idx.shape

    # For each (token, slot) with a valid local expert, record (expert, token, weight).
    flat_expert = recv_topk_idx.reshape(-1)                            # [n_recv*topk]
    valid = flat_expert >= 0
    tok_of = torch.arange(n_recv, device=device).repeat_interleave(num_topk)[valid]
    exp_of = flat_expert[valid].to(torch.long)                        # local expert id

    # Real (unpadded) count per expert from the valid pairs.
    real_cnt = torch.bincount(exp_of, minlength=E)                    # [E]
    # Stable sort the valid pairs by expert so each expert's tokens are grouped.
    order = torch.argsort(exp_of, stable=True)
    exp_sorted = exp_of[order]
    tok_sorted = tok_of[order]

    # Within-expert position of each pair (0..real_cnt[e]-1), via segment ranks.
    real_psum_excl = torch.zeros(E, dtype=torch.long, device=device)
    real_psum_excl[1:] = real_cnt.cumsum(0)[:-1]
    within = torch.arange(exp_sorted.numel(), device=device) - real_psum_excl[exp_sorted]

    # Padded base offset per expert (where expert e's block starts in N_grouped).
    pad_base = torch.zeros(E, dtype=torch.long, device=device)
    pad_base[1:] = padded.cumsum(0)[:-1]
    dst_row = pad_base[exp_sorted] + within                          # [num_valid]

    row_to_token = torch.full((N,), -1, dtype=torch.long, device=device)
    row_to_token[dst_row] = tok_sorted

    return row_to_token, psum, real_cnt


def run_v1_contiguous(
    deep_ep,
    group,
    x_fp8,
    x_sf,
    topk_idx,
    topk_w,
    l1_w_fp8,
    l1_w_sf,
    l2_w_fp8,
    l2_w_sf,
    num_experts,
    hidden,
    intermediate_hidden,
    clamp,
    fast_math,
    alignment,
):
    """V1 (classic ``deep_ep.Buffer``) contiguous-layout MoE baseline.

    V1 dispatch yields a COMPACT per-received-token recv_x; we manually expand it
    to the expert-contiguous padded grouped layout (replicating V2 do_expand),
    run the 2 grouped GEMMs + SwiGLU, then index_add back to the compact
    per-token order (summing the topk-expert contributions of each token) before
    ``buffer.combine`` (which scatters compact [n_recv, hidden] back to source).

    Returns ``(make_run, buffer, weighted=True)``.

    Timing note: EVERYTHING that a real forward repeats every step lives inside
    ``run()`` and is therefore timed -- ``get_dispatch_layout``, ``dispatch``
    (the all-to-all), the compact->expert-contiguous expand, the two grouped
    GEMMs + SwiGLU, the un-expand, and ``combine``. Only the persistent
    ``Buffer`` allocation is outside ``run()`` (mirroring how V2/fused reuse a
    pre-allocated buffer). This makes V1 end-to-end and comparable to the V2 and
    fused timings (which also include dispatch + combine).
    """
    buffer = deep_ep.Buffer(group, num_nvl_bytes=int(1e9), num_rdma_bytes=0,
                            explicitly_destroy=True)
    device = topk_idx.device
    l1_weights = (l1_w_fp8, l1_w_sf)
    l2_weights = (l2_w_fp8, l2_w_sf)

    def run():
        # ---- dispatch (all-to-all) -- timed, like a real forward ------------
        (num_tokens_per_rank, num_tokens_per_rdma_rank,
         num_tokens_per_expert, is_token_in_rank, _) = buffer.get_dispatch_layout(
            topk_idx, num_experts)
        (recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list,
         handle, _) = buffer.dispatch(
            (x_fp8, x_sf),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_w,
            expert_alignment=alignment,
        )
        recv_fp8, recv_sf = recv_x  # compact [n_recv, hidden], [n_recv, hidden//128]
        n_recv = recv_fp8.size(0)

        # ---- compact -> expert-contiguous expand plan -----------------------
        row_to_token, psum, _ = _build_expand_indices(
            recv_topk_idx, num_recv_tokens_per_expert_list, device)
        N = row_to_token.numel()
        valid_row = row_to_token >= 0                    # [N] non-pad rows
        safe_src = torch.clamp(row_to_token, min=0)      # gather index (pad->0)

        # Per grouped row: the topk weight of (token, this-expert). expert_of_row
        # from padded boundaries; gather the matching slot weight from recv_topk_weights.
        padded = torch.tensor(list(num_recv_tokens_per_expert_list), dtype=torch.long, device=device)
        expert_of_row = torch.repeat_interleave(torch.arange(padded.numel(), device=device), padded)  # [N]
        match = (recv_topk_idx[safe_src] == expert_of_row[:, None])    # [N, topk]
        row_w = (recv_topk_weights[safe_src] * match.to(recv_topk_weights.dtype)).sum(dim=1)
        row_w = torch.where(valid_row, row_w, torch.zeros_like(row_w)).contiguous()

        # ---- expand -> 2x grouped FP8 GEMM (DeepGEMM) + SwiGLU --------------
        gx = recv_fp8.index_select(0, safe_src)          # [N, hidden] fp8
        gsf = recv_sf.index_select(0, safe_src)          # [N, hidden//128]
        l1_y = torch.empty((N, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (gx, gsf), l1_weights, l1_y, psum, use_psum_layout=True, recipe=(1, 128, 128))
        l1_q, l1_q_sf = swiglu_apply_weight_to_fp8_triton(
            x=l1_y, topk_weights=row_w, clamp_value=clamp,
            num_per_channels=BASELINE_L2_ACT_SF_GRAN, use_ue8m0_scale=False)
        l2_y = torch.empty((N, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (l1_q, l1_q_sf), l2_weights, l2_y, psum, use_psum_layout=True, recipe=(1, 128, 128))

        # ---- un-expand + combine (all-to-all) -- timed ----------------------
        # Sum each token's topk-expert grouped rows back to compact [n_recv].
        compact = torch.zeros((n_recv, hidden), dtype=torch.bfloat16, device='cuda')
        compact.index_add_(0, safe_src, torch.where(valid_row[:, None], l2_y, torch.zeros_like(l2_y)))
        return buffer.combine(compact, handle=handle)[0]

    return run, buffer, True


# ============================================================================
# Low-latency buffer factory + masked SwiGLU/FP8 quant (low-latency layout).
# ============================================================================


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


# ============================================================================
# Masked SwiGLU + FP8 quant (low-latency layout; topk weights applied
# ----------------------------------------------------------------------------
# Copied verbatim from tests/test_mega_moe_hopper.py (kernel ~line 371, wrapper
# ~line 428). Does NOT apply topk weights — low_latency_combine applies them.
# ============================================================================


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
# V1 low-latency pipeline: low_latency_dispatch -> masked GEMM -> masked SwiGLU
# ============================================================================


def run_v1_low_latency(
    deep_ep,
    group,
    x_bf16,
    topk_idx,
    topk_w,
    l1_w_fp8,
    l1_w_sf,
    l2_w_fp8,
    l2_w_sf,
    num_experts,
    num_experts_per_rank,
    num_max_tokens_per_rank,
    hidden,
    intermediate_hidden,
    clamp,
    fast_math,
):
    """V1 (classic ``deep_ep.Buffer``) low-latency-layout MoE baseline.

    Mirrors the proven low-latency body of ``test_mega_moe_hopper.py``:
    ``low_latency_dispatch(use_fp8=True)`` -> masked L1 GEMM -> masked SwiGLU/
    FP8 quant (no weights) -> masked L2 GEMM -> ``low_latency_combine`` (which
    reduces with topk weights internally).

    Returns ``(make_run, ll_buffer)`` where ``make_run`` is a zero-arg callable
    for the bench timer and ``ll_buffer`` must be ``destroy()``-ed by the caller.
    """
    num_ranks = group.size()
    # DeepEP low_latency_dispatch asserts NVSHMEM_QP_DEPTH >= (tokens + 1) * 2
    # (legacy.py); the default is 1024, which only allows tokens <= 511. Raise it
    # to fit this scenario's token count BEFORE constructing the buffer (the QP
    # depth is latched in Buffer.__init__ from this env var).
    _need_qp_depth = (num_max_tokens_per_rank + 1) * 2
    if int(os.environ.get('NVSHMEM_QP_DEPTH', '1024')) < _need_qp_depth:
        os.environ['NVSHMEM_QP_DEPTH'] = str(_need_qp_depth)
    ll_buffer = _make_deep_ep_low_latency_buffer(
        deep_ep, group, num_max_tokens_per_rank, hidden, num_experts)

    M_max_ll = num_max_tokens_per_rank * num_ranks
    # Expected per-expert mean of ``masked_m`` after dispatch; the ``expected_m``
    # hint for the DeepGEMM masked kernel selector. Matches ``expected_m_ll`` in
    # test_mega_moe_hopper.py: ceil(num_max_tokens_per_rank * num_ranks * num_topk
    # / num_experts), where num_topk is recovered from topk_idx's last dim.
    num_topk = topk_idx.shape[-1]
    expected_m_ll = max(
        1,
        (num_max_tokens_per_rank * num_ranks * num_topk + num_experts - 1)
        // num_experts,
    )

    num_tokens = x_bf16.size(0)
    ll_l1_y = torch.empty(
        (num_experts_per_rank, M_max_ll, intermediate_hidden * 2),
        dtype=torch.bfloat16, device='cuda')
    ll_l2_y = torch.empty(
        (num_experts_per_rank, M_max_ll, hidden),
        dtype=torch.bfloat16, device='cuda')
    ll_combined = torch.empty(
        (num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    topk_idx_ll = topk_idx.to(torch.int64)

    l1_weights = (l1_w_fp8, l1_w_sf)
    l2_weights = (l2_w_fp8, l2_w_sf)

    def run():
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

        # 2) L1 masked grouped FP8 GEMM.
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (recv_x_data, recv_x_sf),
            l1_weights,
            ll_l1_y,
            masked_m,
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

        # 3) Masked SwiGLU + per-128-K FP8 quant (topk weights NOT applied here).
        l1_fp8, l1_sf = swiglu_masked_post_quant_to_fp8(
            ll_l1_y,
            masked_m,
            quant_group_size=BASELINE_L2_ACT_SF_GRAN,
            clamp_value=clamp,
            use_ue8m0_scale=False,
        )

        # 4) L2 masked grouped FP8 GEMM.
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (l1_fp8, l1_sf),
            l2_weights,
            ll_l2_y,
            masked_m,
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

        # 5) Low-latency combine (per-token weighted reduction across topk).
        combined_x, event, hook = ll_buffer.low_latency_combine(
            ll_l2_y,
            topk_idx_ll,
            topk_w,
            ll_handle,
            use_logfmt=False,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
            out=ll_combined,
        )
        return combined_x

    return run, ll_buffer


def import_baseline():
    """Load the DeepEP + TileLang unfused implementation for the perf baseline.

    Returns ``(deep_ep, tilelang_ops, do_bench, is_legacy_loaded)``. On any
    import failure the benchmark falls back to fused-only mode.
    NOTE: single-node runs need ``EP_DISABLE_GIN=1`` so ElasticBuffer init does
    not assert on the absent RDMA GIN.
    """
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


# ---------------------------------------------------------------------------
# DeepEP baselines: V2 (ElasticBuffer) is timed inline below; the V1 contiguous
# baseline (`run_v1_contiguous`, inlined above) is driven from `_run_one_config`.
# ---------------------------------------------------------------------------
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


def _run_one_config(args, num_tokens, num_max_tokens_per_rank,
                    hidden, intermediate_hidden,
                    num_experts, num_topk, num_ranks, rank_idx, group,
                    activation_clamp, fast_math,
                    print_perf=True, baseline_ctx=None):
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Symmetric buffer (one per config: cheaper to recreate than to keep max-size)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
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
    l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf),
    )

    phase_profile_enabled = os.environ.get('DG_SM90_MOE_PHASE_PROFILE', '0') != '0'
    phase_profile_ints = 64 if phase_profile_enabled else 0
    cum_stats = torch.zeros(num_experts_per_rank + phase_profile_ints, dtype=torch.int, device='cuda')
    use_masked_hint = args.masked_ratio > 0

    # Kernel selection: DG_SM90_MOE_KERNEL ∈ {auto(default), cooperative}
    # (the pingpong kernel was removed; only the N-split cooperative kernel remains)
    _kernel = os.environ.get('DG_SM90_MOE_KERNEL', 'auto')
    _moe_fn = {'auto': deep_gemm.fp8_mega_moe,
               'cooperative': deep_gemm.fp8_mega_moe_cooperative}[_kernel]

    # Stage inputs once; bench-loop re-copies them each call (bench helper expects
    # an idempotent ``fn``).
    def run_sm90():
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_w)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        old_masked_hint = os.environ.get('DG_SM90_MOE_MASKED_HINT')
        if use_masked_hint:
            os.environ['DG_SM90_MOE_MASKED_HINT'] = '1'
        try:
            _moe_fn(
                y, transformed_l1, transformed_l2, buffer,
                cumulative_local_expert_recv_stats=cum_stats,
                recipe=(128, 128, 128),
                activation='swiglu',
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            )
        finally:
            if use_masked_hint:
                if old_masked_hint is None:
                    os.environ.pop('DG_SM90_MOE_MASKED_HINT', None)
                else:
                    os.environ['DG_SM90_MOE_MASKED_HINT'] = old_masked_hint
        return y

    if args.ncu_profile_only:
        dist_print(f'[NCU] tokens={num_tokens} hidden={hidden} ih={intermediate_hidden}',
                   once_in_node=True)
        # Profiled rank: a single launch (NCU application-replay re-runs it N times).
        # Peer ranks: loop the kernel so all 8 ranks are live at every cross-rank
        # nvlink_barrier during each of the profiled rank's replays. The shell
        # kills the peers once NCU finishes, so the loop count is just a safe
        # upper bound on the replay count. NO host `dist.barrier()` here: NCU only
        # replays the CUDA kernel (the in-kernel nvlink_barrier is what couples the
        # ranks), and a host collective would mismatch counts and deadlock.
        if rank_idx == args.ncu_rank:
            run_sm90()
            torch.cuda.synchronize()
        else:
            for _ in range(args.ncu_peer_iters):
                run_sm90()
            torch.cuda.synchronize()
        buffer.destroy()
        return

    # Warm up + benchmark. EVERYTHING is timed with CUDA events, end-to-end
    # (one timed call), so the fused kernel / V1 / V2 are all measured the same way.
    run_sm90()
    dist.barrier()
    if phase_profile_enabled:
        cum_stats.zero_()
        torch.cuda.synchronize()
        dist.barrier()
    from deep_gemm.testing import bench as _bench_e2e
    t_sm90 = _bench_e2e(run_sm90, num_warmups=5, num_tests=args.num_tests)
    t_sm90_e2e = t_sm90  # single end-to-end CUDA-event time; used everywhere
    dist.barrier()

    # Count tokens that landed on this rank for stats
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) |
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_sm90)
    num_touched_experts = max(0, torch.unique(gathered_topk_idx.flatten()).numel() - 1)
    # FP8 weights = 1 byte, FP8 acts = 1 byte, BF16 output = 2 bytes
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden +    # L1 weights
        num_touched_experts * hidden * intermediate_hidden +        # L2 weights
        num_recv_tokens * hidden +                                  # L1 acts read
        num_recv_tokens * intermediate_hidden +                     # L1 out write
        num_recv_tokens * intermediate_hidden +                     # L2 acts read
        num_recv_tokens * hidden * 2                                # L2 out write
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_sm90)

    if print_perf:
        dist_print(
            f' tokens={num_tokens:4d}  recv={num_recv_tokens:5d}  experts={num_touched_experts:4d}  '
            f'{t_sm90 * 1e6:7.1f} us  {tflops:6.1f} TFLOPS  {hbm_gbs:6.0f} GB/s  (rank{rank_idx})',
            once_in_node=True,
        )
        if phase_profile_enabled:
            torch.cuda.synchronize()
            names = [
                'dispatch_total', 'dispatch_pull', 'math_loop', 'combine_barrier',
                'combine_reduce', 'gemm_core', 'l1_epilogue', 'l2_epilogue',
            ]
            num_profile_metrics = len(names)
            profile = cum_stats[
                num_experts_per_rank:num_experts_per_rank + phase_profile_ints
            ].view(torch.int64).cpu().tolist()
            for i, name in enumerate(names):
                total = profile[i]
                max_v = profile[num_profile_metrics + i]
                count = profile[2 * num_profile_metrics + i]
                avg = float(total) / count if count else 0.0
                dist_print(
                    f'   phase {name:16s} avg={avg:10.0f} max={max_v:10d} count={count}',
                    once_in_node=True,
                )

    # ---- DeepEP + TileLang unfused baseline (optional) ----------------------
    if baseline_ctx is not None and baseline_ctx.get('enabled'):
        deep_ep = baseline_ctx['deep_ep']
        tilelang_ops = baseline_ctx['tilelang_ops']
        tilelang_bench = baseline_ctx['do_bench']
        baseline_version = baseline_ctx.get('version', 'v2')
        bl_warmup = baseline_ctx.get('warmup', 5)
        bl_repeat = baseline_ctx.get('repeat', 1)

        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        l1_weights = (l1_w_fp8, l1_w_sf)
        l2_weights = (l2_w_fp8, l2_w_sf)
        x_in = (x_fp8, x_sf)
        clamp = activation_clamp if (activation_clamp is not None and activation_clamp == activation_clamp) else None

        def _make_ep_buffer(ver):
            # Only V2 (ElasticBuffer) goes through this factory; V1 uses the
            # classic Buffer inside run_v1_contiguous (inlined above).
            return deep_ep.ElasticBuffer(
                group,
                num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
                num_topk=num_topk, use_fp8_dispatch=True,
                explicitly_destroy=True, allow_multiple_reduction=False,
                num_gpu_timeout_secs=10, num_cpu_timeout_secs=30)

        def _make_run_baseline(ep_buffer):
            def run_baseline():
                recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                    x_in, topk_idx=topk_idx, topk_weights=topk_w,
                    num_experts=num_experts, expert_alignment=alignment,
                    do_cpu_sync=False, do_handle_copy=False,
                    do_expand=True, use_tma_aligned_col_major_sf=True)
                n = recv_x[0].size(0)
                psum = handle.psum_num_recv_tokens_per_expert
                l1_y = torch.empty((n, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    recv_x, l1_weights, l1_y, psum, use_psum_layout=True, recipe=(1, 128, 128))
                # noinspection PyCallingNonCallable
                l1_q, l1_q_sf = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_y, topk_weights=recv_topk_weights, avail_tokens=psum[-1],
                    num_per_channels=128, use_col_major_scales=False,
                    round_scale=False, ue8m0_scale=False, output_bf16=False,
                    clamp_value=clamp, fast_math=fast_math)
                # TileLang returns SF as `y_sf.T` (SM100 MN-major); SM90 1d2d GEMM
                # wants per-token-major, so transpose back.
                l1_q_sf = l1_q_sf.T
                l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (l1_q, l1_q_sf), l2_weights, l2_y, psum, use_psum_layout=True, recipe=(1, 128, 128))
                return ep_buffer.combine(l2_y, handle=handle)[0]
            return run_baseline

        def _run_v2():
            """Time the V2 (ElasticBuffer) baseline. Returns seconds or None."""
            ep_buffer = None
            try:
                ep_buffer = _make_ep_buffer('v2')
                # End-to-end (CUDA-event) timing — measured the same way as t_sm90_e2e
                # (CUDA-graph backend would erase launch overhead and not be
                # comparable to the end-to-end fused-kernel time).
                t = tilelang_bench(
                    _make_run_baseline(ep_buffer), _n_warmup=bl_warmup, _n_repeat=bl_repeat,
                    backend='event', return_mode='median') / 1e3
                if print_perf:
                    dist_print(
                        f'   -> baseline[v2]={t * 1e6:8.1f} us  '
                        f'{safe_div(t, t_sm90_e2e):.2f}x speedup  (rank{rank_idx})',
                        once_in_node=True)
                return t
            except Exception as ex:
                dist_print(f'   -> baseline[v2] FAILED: {ex}', once_in_node=True)
                return None
            finally:
                if ep_buffer is not None:
                    try:
                        ep_buffer.destroy()
                    except Exception:
                        pass

        def _run_v1_contig():
            """V1 classic-Buffer CONTIGUOUS baseline (end-to-end, incl. dispatch)."""
            v1c_buffer = None
            try:
                run_v1c, v1c_buffer, weighted = run_v1_contiguous(
                    deep_ep, group, x_fp8, x_sf, topk_idx, topk_w,
                    l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
                    num_experts, hidden, intermediate_hidden,
                    clamp, fast_math, alignment)
                t = tilelang_bench(
                    run_v1c, _n_warmup=bl_warmup, _n_repeat=bl_repeat,
                    backend='event', return_mode='median') / 1e3
                if print_perf:
                    note = '' if weighted else '  (speed-only, unweighted)'
                    dist_print(
                        f'   -> baseline[v1-contig]={t * 1e6:8.1f} us  '
                        f'{safe_div(t, t_sm90_e2e):.2f}x speedup{note}  (rank{rank_idx})',
                        once_in_node=True)
                return t
            except Exception as ex:
                dist_print(f'   -> baseline[v1-contig] FAILED: {ex}', once_in_node=True)
                return None
            finally:
                if v1c_buffer is not None:
                    try:
                        v1c_buffer.destroy()
                    except Exception:
                        pass

        def _run_v1_ll():
            """V1 classic-Buffer LOW-LATENCY baseline (low_latency_dispatch ->
            masked GEMM -> masked SwiGLU -> masked GEMM -> low_latency_combine).
            Uses NVLink low-latency on a single node."""
            # Low-latency pre-allocates a per-(expert, max-token) RDMA buffer of
            # ~num_dispatch_tokens * hidden * num_experts bytes. DeepEP asserts
            # num_rdma_bytes / 16 < INT_MAX, which overflows once the per-rank
            # dispatch capacity reaches ~4096 (about 35 GB). Low-latency targets
            # small decode batches, so size the buffer to THIS scenario's token
            # count and skip only when a single batch itself exceeds the limit.
            _LL_MAX_TOKENS = 2048
            if num_tokens > _LL_MAX_TOKENS:
                if print_perf:
                    dist_print(
                        f'   -> baseline[v1-ll]     skipped '
                        f'(num_tokens={num_tokens} > {_LL_MAX_TOKENS}; '
                        f'low-latency buffer would overflow)  (rank{rank_idx})',
                        once_in_node=True)
                return None
            ll_buffer = None
            try:
                run_ll, ll_buffer = run_v1_low_latency(
                    deep_ep, group, x_bf, topk_idx, topk_w,
                    l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
                    num_experts, num_experts_per_rank, num_tokens,
                    hidden, intermediate_hidden, clamp, fast_math)
                t = tilelang_bench(
                    run_ll, _n_warmup=bl_warmup, _n_repeat=bl_repeat,
                    backend='event', return_mode='median') / 1e3
                if print_perf:
                    dist_print(
                        f'   -> baseline[v1-ll]    ={t * 1e6:8.1f} us  '
                        f'{safe_div(t, t_sm90_e2e):.2f}x speedup  (rank{rank_idx})',
                        once_in_node=True)
                return t
            except Exception as ex:
                dist_print(f'   -> baseline[v1-ll] FAILED: {ex}', once_in_node=True)
                return None
            finally:
                if ll_buffer is not None:
                    try:
                        ll_buffer.destroy()
                    except Exception:
                        pass

        def _run_v1():
            # V1 has two layouts; run both so contiguous vs low-latency are visible.
            _run_v1_contig()
            _run_v1_ll()

        if baseline_version == 'v1':
            _run_v1()
        elif baseline_version == 'v2':
            _run_v2()
        else:  # 'both'
            _run_v1()
            _run_v2()

    dist.barrier()
    buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    forced_num_sms = int(os.environ.get('DG_SM90_MOE_SET_NUM_SMS', '0'))
    if forced_num_sms > 0:
        deep_gemm.set_num_sms(forced_num_sms)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    if args.batches is None:
        batches = [1, 2, 4, 8, 16, 32]
    else:
        batches = args.batches

    # Optionally load the DeepEP + TileLang baseline for speedup comparison.
    baseline_ctx = None
    if args.baseline:
        deep_ep, tilelang_ops, do_bench, is_legacy_loaded = import_baseline()
        baseline_ctx = dict(
            enabled=is_legacy_loaded,
            deep_ep=deep_ep, tilelang_ops=tilelang_ops, do_bench=do_bench,
            version=args.baseline_version,
            warmup=args.baseline_warmup, repeat=args.baseline_repeat)

    dist_print(
        f'SM90 MegaMoE bench: ranks={num_ranks} hidden={args.hidden} '
        f'ih={args.intermediate_hidden} experts={args.num_experts} topk={args.num_topk} '
        f'masked_ratio={args.masked_ratio} fast_math={bool(args.fast_math)} '
        f'baseline={bool(args.baseline)}',
        once_in_node=True,
    )

    # In NCU mode we run only one batch (the first one in `batches`) so that
    # ncu's `--launch-count 1` is unambiguous.
    if args.ncu_profile_only:
        batches = batches[:1]

    num_max_tokens_per_rank = max(batches)
    for num_tokens in batches:
        _run_one_config(
            args, num_tokens, num_max_tokens_per_rank,
            args.hidden, args.intermediate_hidden,
            args.num_experts, args.num_topk,
            num_ranks, rank_idx, group,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            baseline_ctx=baseline_ctx,
        )

    # In NCU mode the profiled rank and the looping peers run the kernel a
    # different number of times, so a host collective here would mismatch and
    # hang. The shell orchestrator tears the peers down instead.
    if not args.ncu_profile_only:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 MegaMoE benchmark')

    parser.add_argument('--ncu-profile-only', action='store_true')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)
    # NCU "profile only rank R" support: the profiled rank runs the kernel once
    # (NCU replays it); the other ranks act as live peers and must keep launching
    # the kernel enough times to outlast all of the profiled rank's replays.
    parser.add_argument('--ncu-rank', type=int, default=0,
                        help='Which rank NCU profiles; peers loop to stay alive')
    parser.add_argument('--ncu-peer-iters', type=int, default=2000,
                        help='Kernel launches a peer rank does in NCU mode (>= NCU replay count)')

    parser.add_argument('--batches', type=int, nargs='+', default=None,
                        help='List of num_tokens to benchmark (default: 1 2 4 8 16 32)')
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--num-tests', type=int, default=1,
                        help='Timed CUDA-event iterations for the fused kernel (default 1 = single shot)')
    parser.add_argument('--baseline', action='store_true',
                        help='Also benchmark the DeepEP + TileLang unfused baseline '
                             'and report speedup (needs EP_DISABLE_GIN=1 single-node)')
    parser.add_argument('--baseline-version', type=str, default='v2',
                        choices=['v1', 'v2', 'both'],
                        help="DeepEP baseline API: 'v2' = ElasticBuffer (default), "
                             "'v1' = classic Buffer dispatch/combine, "
                             "'both' = run V1 and V2 and report each")
    parser.add_argument('--baseline-warmup', type=int, default=5,
                        help='Warmup iters for the baseline timing (do_bench _n_warmup)')
    parser.add_argument('--baseline-repeat', type=int, default=1,
                        help='Timed repeats for the baseline (do_bench _n_repeat); '
                             'median is reported')

    args = parser.parse_args()

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        np = args.num_processes
        torch.multiprocessing.spawn(test, args=(np, args), nprocs=np)
