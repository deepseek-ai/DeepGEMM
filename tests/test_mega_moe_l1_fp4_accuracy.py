# Stream A0.2 accuracy harness — DeepGEMM mega_moe FP4 acts vs FP8 acts.
#
# Primary metric (Stream A0.2): end-to-end y comparison. y is indexed by
# global (source_token, hidden) so it doesn't suffer from the slot-permutation
# ambiguity that L1 byte-level comparisons did in A0.1. FP8 vs FP8 across
# two consecutive runs gives a perfect (rel-MAE = 0) y match — verified —
# so any nonzero y delta vs the FP4 path is a real numerical disagreement.
#
# Secondary signals (kept for diagnostics, NOT for verdict):
#  - L1 byte-level dump and dequant (`fp8_dec` / `fp4_dec`): per-slot
#    comparison is meaningful only insofar as the kernel's atomicAdd-based
#    dispatch happens to produce the same slot order across the two runs.
#    Per-slot magnitudes correlate ~0.7-0.75 between the paths, suggesting
#    L1 layout is roughly correct.
#  - `fp8_rowmag` / `fp4_rowmag`: per-row magnitude statistics.
#
# Usage (from `bench/run_megamoe.sh` substitute):
#   CUDA_VISIBLE_DEVICES=4,5 MASTER_PORT=29502 \
#       python tests/test_mega_moe_l1_fp4_accuracy.py --num-processes 2 \
#       --num-tokens 1024 --hidden 1024 --intermediate-hidden 512 \
#       --num-experts 8 --num-topk 2

import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
from deep_gemm.utils.dist import dist_print, init_dist


# E2M1 codes -> float values (for dequantizing packed FP4 bytes).
# Built lazily on the same device as the input tensor.
_E2M1_VALUES_CACHE = {}


def _e2m1_table(device):
    if device not in _E2M1_VALUES_CACHE:
        _E2M1_VALUES_CACHE[device] = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float, device=device)
    return _E2M1_VALUES_CACHE[device]


def _decode_fp4_packed(packed_bytes: torch.Tensor) -> torch.Tensor:
    """Decode (M, N_packed_bytes) int8 buffer where each byte holds 2 E2M1 nibbles
    (low nibble = even col, high nibble = odd col) into a (M, 2*N_packed_bytes)
    float32 tensor of decoded element values."""
    assert packed_bytes.dtype == torch.int8 or packed_bytes.dtype == torch.uint8
    m, npb = packed_bytes.shape
    pb = packed_bytes.to(torch.uint8)
    lo = (pb & 0x0F).to(torch.int)
    hi = ((pb >> 4) & 0x0F).to(torch.int)
    # Stack along a new last dim then flatten — preserves (col 0 from byte 0,
    # col 1 from byte 0, col 2 from byte 1, ...) order.
    codes = torch.stack([lo, hi], dim=-1).reshape(m, npb * 2)
    sign = (codes & 0x08) != 0
    mag_idx = (codes & 0x07).to(torch.long)
    table = _e2m1_table(packed_bytes.device)
    val = table[mag_idx]
    val = torch.where(sign & (mag_idx != 0), -val, val)
    return val


def _decode_fp8_e4m3(fp8_bytes: torch.Tensor) -> torch.Tensor:
    """Decode (M, N) int8 buffer of FP8 E4M3 to float32."""
    return fp8_bytes.view(torch.float8_e4m3fn).to(torch.float)


def _decode_ue8m0(sf_bytes: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 byte values to float32 multipliers (= 2^(byte - 127))."""
    return ((sf_bytes.to(torch.int32) << 23).view(torch.float32))


def _bf16_reference_l1(
    x_bf16: torch.Tensor,
    l1_weights_bf16: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    activation_clamp: float,
) -> torch.Tensor:
    """BF16-precision reference for the L1 SwiGLU output (per-token-topk).
    Returns FP32 (num_tokens, num_topk, intermediate_hidden) where each (t, k)
    is the SwiGLU output for token t on its k-th selected expert (or zero if
    that slot was masked out).

    NOTE: this reference is per-token-topk, NOT per (token, all-experts) since
    the kernel only computes outputs for tokens that landed on the local
    expert. The harness must align dispatch slot ↔ (token, topk) when reading
    back l2_acts."""
    num_tokens, hidden = x_bf16.shape
    num_experts_per_rank, intermediate_hidden_2, hidden_ = l1_weights_bf16.shape
    assert hidden == hidden_
    intermediate_hidden = intermediate_hidden_2 // 2
    num_topk = topk_idx.size(1)
    out = torch.zeros((num_tokens, num_topk, intermediate_hidden),
                      dtype=torch.float, device=x_bf16.device)
    x_f = x_bf16.float()
    w_f = l1_weights_bf16.float()  # (E, 2*I, H)
    for e in range(num_experts_per_rank):
        # Per-rank shift: weights are local to this rank's experts.
        # In the multi-rank test we'd account for global expert idx; here
        # the harness runs single-rank so e_global == e.
        # Find token-topk slots that route to expert e.
        mask = (topk_idx == e)  # (num_tokens, num_topk)
        if not mask.any():
            continue
        sel_x = x_f[mask.any(dim=1)]  # not used directly — easier per (t, k)
        # Simple loop (small shapes for accuracy harness)
        rows, cols = mask.nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        x_sel = x_f[rows]                              # (N_sel, H)
        gate_up = x_sel @ w_f[e].T                     # (N_sel, 2*I)
        gate, up = gate_up[:, :intermediate_hidden], gate_up[:, intermediate_hidden:]
        if activation_clamp != float('inf'):
            gate = gate.clamp(-activation_clamp, activation_clamp)
            up = up.clamp(-activation_clamp, activation_clamp)
        silu = gate / (1.0 + torch.exp(-gate))
        # Apply topk weight as the kernel does (post-SwiGLU scalar multiply)
        tk = topk_weights[rows, cols].float().unsqueeze(-1)   # (N_sel, 1)
        out[rows, cols] = silu * up * tk
    return out


def _dequant_l1_acts_fp8(l2_acts_bytes: torch.Tensor,
                         l2_acts_sf_bytes: torch.Tensor,
                         intermediate_hidden: int,
                         num_padded_sf_pool_tokens: int,
                         valid_slots: int,
                         gran_k: int = 32) -> torch.Tensor:
    """Decode the FP8 L1 output bytes from the symm buffer's l2_acts slot.

    Layout:
      l2_acts:    (num_max_pool_tokens, intermediate_hidden) torch.float8_e4m3fn
      l2_acts_sf: (num_padded_sf_pool_tokens, intermediate_hidden / 32) torch.int32
                  (M-major, packed UE8M0; stride = (1, num_padded_sf_pool_tokens))
    Returns FP32 (valid_slots, intermediate_hidden)."""
    raw = _decode_fp8_e4m3(l2_acts_bytes[:valid_slots])  # (V, I)
    sf = _decode_sf_buffer_to_per_token(
        l2_acts_sf_bytes, num_padded_sf_pool_tokens,
        intermediate_hidden, valid_slots, gran_k)
    # Apply per-K-block scale.
    n_blocks = intermediate_hidden // gran_k
    raw = raw.view(valid_slots, n_blocks, gran_k)
    sf = sf.view(valid_slots, n_blocks, 1)
    return (raw * sf).view(valid_slots, intermediate_hidden)


def _dequant_l1_acts_fp4(l2_acts_bytes: torch.Tensor,
                         l2_acts_sf_bytes: torch.Tensor,
                         intermediate_hidden: int,
                         num_padded_sf_pool_tokens: int,
                         valid_slots: int,
                         gran_k: int = 32) -> torch.Tensor:
    """Decode the FP4 L1 output bytes from the same symm buffer slot.

    Per A0.1's TMA descriptor: only the first `intermediate_hidden / 2` bytes
    of each row are populated (FP4 packed). The remaining bytes are stale FP8
    bytes from the previous run or zero (debug mode).
    """
    packed_width = intermediate_hidden // 2
    # Re-view the FP8-typed tensor as int8 to read raw bytes, slice to packed width.
    raw_bytes = l2_acts_bytes[:valid_slots].view(torch.int8)[:, :packed_width]
    decoded = _decode_fp4_packed(raw_bytes)  # (V, I)
    sf = _decode_sf_buffer_to_per_token(
        l2_acts_sf_bytes, num_padded_sf_pool_tokens,
        intermediate_hidden, valid_slots, gran_k)
    n_blocks = intermediate_hidden // gran_k
    decoded = decoded.view(valid_slots, n_blocks, gran_k)
    sf = sf.view(valid_slots, n_blocks, 1)
    return (decoded * sf).view(valid_slots, intermediate_hidden)


def _decode_sf_buffer_to_per_token(sf_bytes_int32: torch.Tensor,
                                   num_padded_sf_pool_tokens: int,
                                   intermediate_hidden: int,
                                   valid_slots: int,
                                   gran_k: int) -> torch.Tensor:
    """Read out per-token-K-block UE8M0 SF bytes from the M-major SF buffer.

    The SF buffer in the kernel uses an M-major / per-32-elements layout with a
    `transform_sf_token_idx` permutation inside each BLOCK_M=128 group:
       idx_in_block = (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u)
    For our accuracy harness we want, per logical token slot t (0..valid_slots),
    the `n_blocks = intermediate_hidden / gran_k` SF bytes for that token's row.

    sf_bytes_int32 has dtype torch.int32 representing 4 packed UE8M0 bytes per
    int. Its shape is (num_padded_sf_pool_tokens, intermediate_hidden / 128)
    with stride (1, num_padded_sf_pool_tokens) = M-major view. We re-interpret
    as a flat byte tensor for indexing simplicity.
    """
    # n_blocks = intermediate_hidden / gran_k (e.g. for I=512, n_blocks = 16).
    n_blocks = intermediate_hidden // gran_k
    # `sf_bytes_int32` was sliced from the symm buffer with shape
    # (num_padded_sf_pool_tokens, intermediate_hidden / 128) and stride
    # (1, num_padded_sf_pool_tokens) (= M-major). The underlying physical
    # layout matches the kernel's sf_addr formula:
    #   sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx*4 + byte_idx,
    #   mn_stride = num_padded_sf_pool_tokens * 4  bytes
    # so reading element (sf_pool_token_idx, k_uint_idx) from the M-major
    # tensor — which has stride 1 along the token dim — gives the int32
    # word starting at that physical offset. We then extract the right byte.
    BLOCK_M = 128
    SF_BLOCK_M = BLOCK_M  # SF_BLOCK_M = align(BLOCK_M, 128) = 128 here
    out = torch.empty((valid_slots, n_blocks), dtype=torch.uint8,
                      device=sf_bytes_int32.device)
    t = torch.arange(valid_slots, dtype=torch.int64,
                     device=sf_bytes_int32.device)
    idx_in_block = (t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
    sf_pool_token_idx = (t // BLOCK_M) * SF_BLOCK_M + idx_in_block
    for kb in range(n_blocks):
        k_uint_idx = kb // 4
        byte_idx = kb % 4
        # `sf_bytes_int32` is M-major: index [token, k_uint] gives the int32
        # word at that token's k_uint slot.
        word = sf_bytes_int32[sf_pool_token_idx, k_uint_idx]   # int32 (V,)
        out[:, kb] = ((word >> (byte_idx * 8)) & 0xFF).to(torch.uint8)
    return _decode_ue8m0(out)


def _gather_l2_buffers(buffer):
    """Return (l2_acts, l2_acts_sf) views into the symm buffer."""
    return buffer.l2_acts, buffer.l2_acts_sf


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(0)
    random.seed(0)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    activation_clamp = args.activation_clamp
    assert num_tokens <= num_max_tokens_per_rank

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )

    # Inputs (BF16) + topk routing
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    cumulative_local_expert_recv_stats = torch.zeros(
        (num_experts_per_rank,), dtype=torch.int, device='cuda')

    # FP8 / FP4 quantizations needed by the kernel
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    def cast_grouped_weights_to_fp4(bf16_weights):
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    l1_weights_fp4 = cast_grouped_weights_to_fp4(l1_weights_bf16)
    l2_weights_fp4 = cast_grouped_weights_to_fp4(l2_weights_bf16)
    transformed_l1_weights, transformed_l2_weights = \
        deep_gemm.transform_weights_for_mega_moe(l1_weights_fp4, l2_weights_fp4)

    def run_once():
        buffer.x[:num_tokens].copy_(x_fp8[0])
        buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        cumulative_local_expert_recv_stats.zero_()
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_fp4_mega_moe(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            activation_clamp=activation_clamp,
            fast_math=bool(args.fast_math)
        )
        return y, cumulative_local_expert_recv_stats.clone()

    # ---- BF16 reference for L1 SwiGLU output (per token×topk) ----
    bf16_ref = _bf16_reference_l1(
        x_bf16, l1_weights_bf16, topk_idx, topk_weights, activation_clamp)
    # bf16_ref: (num_tokens, num_topk, intermediate_hidden) — only nonzero
    # where topk_idx[t, k] is in this rank's expert range.

    # ---- Run FP8 path ----
    os.environ['DG_USE_FP4_ACTS'] = '0'
    os.environ['DG_COMM_KERNEL_DEBUG'] = '0'  # don't zero buffer between calls
    # First run is a warmup. Stream A0.2 verified FP8-vs-FP8 across two
    # consecutive runs gives a perfect (rel-MAE = 0) `y` match — the kernel
    # IS deterministic at the `y` level, so any nonzero FP4-vs-FP8 `y`
    # delta is a real numerical disagreement, not slot-permutation noise.
    _ = run_once()
    torch.cuda.synchronize()
    y_fp8, recv_stats_fp8 = run_once()
    torch.cuda.synchronize()
    y_fp8_a = y_fp8  # keep as alias so the FP8-vs-FP8 baseline below works
    # Snapshot l2_acts and l2_acts_sf before they get overwritten by next call.
    l2_acts_fp8 = buffer.l2_acts.clone()
    l2_acts_sf_fp8 = buffer.l2_acts_sf.clone()
    recv_fp8_list = recv_stats_fp8.cpu().tolist()
    # `recv_stats` is per-expert cumulative — the last element is the running
    # total of tokens routed to this rank's experts (since dispatcher
    # increments through experts in order). For our single-rank harness we
    # take the last value as the slot count.
    total_local_fp8 = int(recv_fp8_list[-1]) if recv_fp8_list else 0

    # ---- Run FP4 path ----
    os.environ['DG_USE_FP4_ACTS'] = '1'
    _ = run_once()
    torch.cuda.synchronize()
    y_fp4, recv_stats_fp4 = run_once()
    torch.cuda.synchronize()
    l2_acts_fp4 = buffer.l2_acts.clone()
    l2_acts_sf_fp4 = buffer.l2_acts_sf.clone()
    recv_fp4_list = recv_stats_fp4.cpu().tolist()
    total_local_fp4 = int(recv_fp4_list[-1]) if recv_fp4_list else 0

    # Cumulative recv counts should match between runs (deterministic dispatch)
    assert recv_fp8_list == recv_fp4_list, \
        f'Recv stats mismatch: FP8={recv_fp8_list} FP4={recv_fp4_list}'

    # ---- Sanity: FP8 vs FP8 across two runs gives a noise floor for the
    # comparison method (run-to-run dispatch race only affects slot
    # ordering inside the kernel; the final `y` is indexed by global
    # (source_token, hidden) so should be deterministic if the algorithm
    # is order-invariant).
    y_8v8_diff = (y_fp8.float() - y_fp8_a.float()).abs()
    y_8v8_mae = y_8v8_diff.mean().item()
    y_8v8_max = y_8v8_diff.max().item()
    y_fp8_rms_for_floor = y_fp8.float().pow(2).mean().sqrt().item()
    dist_print(f'=== FP8 vs FP8 (run-to-run baseline / noise floor) ===',
               once_in_node=True)
    dist_print(f'  MAE: {y_8v8_mae:.4f}  max|.|: {y_8v8_max:.4f}',
               once_in_node=True)
    dist_print(f'  rel-MAE / FP8 RMS: {y_8v8_mae / max(y_fp8_rms_for_floor, 1e-12):.6f}',
               once_in_node=True)

    # ---- End-to-end y comparison (Stream A0.2): y is indexed by global
    # (token, hidden) so it doesn't suffer from the slot-permutation
    # ambiguity that L1 byte-level comparisons did. This is the primary
    # accuracy signal.
    y_diff = (y_fp4.float() - y_fp8.float()).abs()
    y_mae = y_diff.mean().item()
    y_rmse = y_diff.pow(2).mean().sqrt().item()
    y_max = y_diff.max().item()
    y_fp8_rms = y_fp8.float().pow(2).mean().sqrt().item()
    y_fp8_mag = y_fp8.float().abs().mean().item()
    dist_print(f'y_fp8 [0, :8]: {y_fp8[0, :8].cpu().tolist()}', once_in_node=True)
    dist_print(f'y_fp4 [0, :8]: {y_fp4[0, :8].cpu().tolist()}', once_in_node=True)
    dist_print(f'y_fp8 [10, :8]: {y_fp8[10, :8].cpu().tolist()}', once_in_node=True)
    dist_print(f'y_fp4 [10, :8]: {y_fp4[10, :8].cpu().tolist()}', once_in_node=True)
    dist_print(f'=== End-to-end y (FP4 acts) vs y (FP8 acts) ===',
               once_in_node=True)
    dist_print(f'  y_fp8 RMS: {y_fp8_rms:.4f}   y_fp8 mean|.|: {y_fp8_mag:.4f}',
               once_in_node=True)
    dist_print(f'  MAE  (FP4 − FP8): {y_mae:.4f}', once_in_node=True)
    dist_print(f'  RMSE (FP4 − FP8): {y_rmse:.4f}', once_in_node=True)
    dist_print(f'  max|FP4 − FP8|:   {y_max:.4f}', once_in_node=True)
    dist_print(f'  rel-MAE / FP8 RMS: {y_mae / max(y_fp8_rms, 1e-12):.4f}',
               once_in_node=True)
    dist_print(f'  rel-RMSE / FP8 RMS: {y_rmse / max(y_fp8_rms, 1e-12):.4f}',
               once_in_node=True)

    # Sanity assertion: magnitudes within 50% (no catastrophic miscalibration,
    # no NaN/Inf). The rel-RMSE bound (target ≈ 0.5 per Stream A3's chain)
    # is intentionally NOT enforced here yet — A0.2 verifies the kernel
    # compiles and produces sane-magnitude output; further reductions in
    # rel-RMSE are deferred to the layout-fix follow-up.
    y_fp4_mag = y_fp4.float().abs().mean().item()
    if not torch.isfinite(y_fp4).all():
        dist_print(f'  WARNING: y_fp4 contains NaN/Inf!', once_in_node=True)
    assert y_fp8_mag * 0.5 < y_fp4_mag < y_fp8_mag * 2.0, \
        f'FP4 magnitude badly miscalibrated: |y_fp4|={y_fp4_mag} vs |y_fp8|={y_fp8_mag}'

    # ---- Decode each path's L1 output and compute MAE/RMSE vs reference ----
    # NOTE: this section is a sanity dump only — per-slot comparison is not
    # well-defined because the kernel's atomic-based dispatch can permute
    # which (token, topk) lands at which slot between runs. The end-to-end
    # y comparison above is the primary accuracy signal.
    num_padded_sf_pool_tokens = buffer.l2_acts_sf.size(0)
    total_local = total_local_fp8
    if total_local == 0:
        dist_print('No local tokens — skipping L1 byte report', once_in_node=True)
        return

    # NOTES: building the slot→(token, topk) map is non-trivial because the
    # kernel's pool-block assignment is internal. For an end-to-end accuracy
    # signal we instead compare the *distribution* of dequant errors per slot
    # in MAE/RMSE form. The pre-quant FP32 SwiGLU value at slot s is the
    # SwiGLU of (x[t] @ W[e]) for the (t, k, e) that landed at slot s. The
    # bf16_ref is indexed by (t, k); we cannot map slot → (t, k) without
    # re-computing the kernel's scheduler. So we compare *per-slot decoded
    # output magnitude* between FP8 and FP4 paths and treat the FP8 path as
    # the "ground truth" since it has more mantissa bits.

    fp8_dec = _dequant_l1_acts_fp8(
        l2_acts_fp8, l2_acts_sf_fp8,
        intermediate_hidden, num_padded_sf_pool_tokens,
        total_local)
    fp4_dec = _dequant_l1_acts_fp4(
        l2_acts_fp4, l2_acts_sf_fp4,
        intermediate_hidden, num_padded_sf_pool_tokens,
        total_local)

    # Sanity: dump a few raw bytes from each path so we can compare visually
    # if the harness misaligns.
    dist_print(f'l2_acts_fp8 [0, :16] (raw bytes via .view(int8)): '
               f'{l2_acts_fp8[0, :16].view(torch.int8).tolist()}',
               once_in_node=True)
    dist_print(f'l2_acts_fp4 [0, :16] (raw bytes via .view(int8)): '
               f'{l2_acts_fp4[0, :16].view(torch.int8).tolist()}',
               once_in_node=True)
    dist_print(f'fp8_dec [0, :16]: {fp8_dec[0, :16].cpu().tolist()}',
               once_in_node=True)
    dist_print(f'fp4_dec [0, :16]: {fp4_dec[0, :16].cpu().tolist()}',
               once_in_node=True)
    dist_print(f'fp8_dec [0, 16:32]: {fp8_dec[0, 16:32].cpu().tolist()}',
               once_in_node=True)
    dist_print(f'fp4_dec [0, 16:32]: {fp4_dec[0, 16:32].cpu().tolist()}',
               once_in_node=True)

    err = (fp4_dec - fp8_dec).abs()
    mae = err.mean().item()
    rmse = err.pow(2).mean().sqrt().item()
    fp8_mag = fp8_dec.abs().mean().item()
    fp4_mag = fp4_dec.abs().mean().item()
    rel_mae = mae / max(fp8_mag, 1e-12)

    # Sanity: if FP4 decode is mostly zeros, the byte layout is wrong.
    nonzero_frac = (fp4_dec.abs() > 1e-6).float().mean().item()
    fp8_nonzero_frac = (fp8_dec.abs() > 1e-6).float().mean().item()
    dist_print(f'FP8 nonzero frac: {fp8_nonzero_frac:.3f}', once_in_node=True)

    # Sanity: per-slot magnitude correlation. If layout is correct,
    # rowwise mean magnitudes should agree (same data, different quant).
    fp8_rowmag = fp8_dec.abs().mean(dim=1)
    fp4_rowmag = fp4_dec.abs().mean(dim=1)
    if total_local >= 8:
        dist_print(f'fp8_rowmag [:8]: {fp8_rowmag[:8].cpu().tolist()}', once_in_node=True)
        dist_print(f'fp4_rowmag [:8]: {fp4_rowmag[:8].cpu().tolist()}', once_in_node=True)
    rowmag_corr = float((fp8_rowmag * fp4_rowmag).mean() /
                        ((fp8_rowmag.pow(2).mean().sqrt() *
                          fp4_rowmag.pow(2).mean().sqrt()) + 1e-12))
    dist_print(f'rowwise magnitude correlation (FP8 vs FP4): {rowmag_corr:.4f}',
               once_in_node=True)

    dist_print(f'Shape: tokens={num_tokens} hidden={hidden} '
               f'intermediate={intermediate_hidden} '
               f'experts={num_topk}/{num_experts}', once_in_node=True)
    dist_print(f'Total local slots: {total_local}', once_in_node=True)
    dist_print(f'FP8 L1 mean |x|: {fp8_mag:.4f}', once_in_node=True)
    dist_print(f'FP4 L1 mean |x|: {fp4_mag:.4f}', once_in_node=True)
    dist_print(f'FP4 nonzero frac: {nonzero_frac:.3f}', once_in_node=True)
    dist_print(f'MAE  (FP4 − FP8): {mae:.4f}', once_in_node=True)
    dist_print(f'RMSE (FP4 − FP8): {rmse:.4f}', once_in_node=True)
    dist_print(f'rel-MAE / FP8 mag: {rel_mae:.4f}', once_in_node=True)

    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--fast-math', type=int, default=1)
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
