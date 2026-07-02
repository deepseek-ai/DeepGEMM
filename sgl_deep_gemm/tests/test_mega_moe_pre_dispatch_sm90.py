# Correctness probe for `deep_gemm.mega_moe_pre_dispatch_sm90` (SM90 / Hopper).
#
# SM90 variant of the fused mega-MoE pre-dispatch. Unlike the Blackwell
# `mega_moe_pre_dispatch` (covered by test_mega_moe_pre_dispatch.py), the SM90
# kernel emits plain FP8 (no packed-FP4 branch) with per-(token, 128-channel)
# FP32 scales, and folds a `routed_scaling_factor` into the topk_weights write.
# This test verifies, against a pure-torch reference:
#   1) per-token group-128 FP8 quantization of x -> buf_x[:M], buf_x_sf[:M]
#      (dequantised output within one e4m3 mantissa step),
#   2) buf.topk_idx[:M] == topk_idx ; buf.topk_weights[:M] == weights * alpha,
#   3) pad rows: buf.topk_idx[M:] == -1 ; buf.topk_weights[M:] == 0.
#
# Single-GPU; SM90-only (skips cleanly on non-Hopper or when the wheel lacks
# the kernel). Run directly: `python test_mega_moe_pre_dispatch_sm90.py`.

import argparse
import sys

import torch

import deep_gemm

# fp8-e4m3 representable max; per-group scale = amax / FP8_E4M3_MAX.
FP8_E4M3_MAX = 448.0


def _has_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 9


def _ref_quant_fp8_group128(x: torch.Tensor):
    """Per-token, per-128-channel-group absmax FP8-e4m3 quant with fp32 scales.

    Returns (q_fp8 [M, H], scale_fp32 [M, H/128]) such that
    ``q_fp8.float().reshape(M, G, 128) * scale[..., None] ≈ x``.
    """
    M, H = x.shape
    G = H // 128
    xf = x.float().reshape(M, G, 128)
    amax = xf.abs().amax(dim=-1).clamp(min=1e-12)  # (M, G)
    scale = amax / FP8_E4M3_MAX  # (M, G) fp32
    q = (xf / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q.reshape(M, H), scale


def _alloc(padded_max: int, hidden: int, top_k: int, device):
    num_groups = hidden // 128
    buf_x = torch.empty((padded_max, hidden), dtype=torch.float8_e4m3fn, device=device)
    buf_x_sf = torch.empty((padded_max, num_groups), dtype=torch.float32, device=device)
    buf_idx = torch.empty((padded_max, top_k), dtype=torch.int64, device=device)
    buf_w = torch.empty((padded_max, top_k), dtype=torch.float32, device=device)
    return buf_x, buf_x_sf, buf_idx, buf_w


def _reference(x, topk_idx, topk_weights, padded_max, routed_scaling_factor):
    M, _ = x.shape
    buf_x, buf_x_sf, buf_idx, buf_w = _alloc(padded_max, x.shape[1], topk_idx.shape[1], x.device)
    if M > 0:
        qx, sf = _ref_quant_fp8_group128(x)
        buf_x[:M].copy_(qx)
        buf_x_sf[:M].copy_(sf)
        buf_idx[:M].copy_(topk_idx)
        buf_w[:M].copy_(topk_weights * routed_scaling_factor)
    if M < padded_max:
        buf_idx[M:].fill_(-1)
        buf_w[M:].zero_()
    return buf_x, buf_x_sf, buf_idx, buf_w


def _run_kernel(x, topk_idx, topk_weights, padded_max, routed_scaling_factor):
    M, H = x.shape
    K = topk_idx.shape[1]
    buf_x, buf_x_sf, buf_idx, buf_w = _alloc(padded_max, H, K, x.device)
    # Poison the padded region so a missing pad-write shows up as a hard failure.
    if M < padded_max:
        buf_idx[M:].fill_(0x4242)
        buf_w[M:].fill_(float("nan"))

    if M > 0:
        x_in, idx_in, w_in = x, topk_idx, topk_weights
    else:
        x_in = x.new_empty((0, H), dtype=x.dtype)
        idx_in = x.new_empty((0, K), dtype=torch.int32)
        w_in = x.new_empty((0, K), dtype=torch.float32)

    deep_gemm.mega_moe_pre_dispatch_sm90(
        x_in,
        idx_in,
        w_in,
        buf_x,
        buf_x_sf,
        buf_idx,
        buf_w,
        num_tokens=M,
        group_size=128,
        routed_scaling_factor=routed_scaling_factor,
    )
    torch.cuda.synchronize()
    return buf_x, buf_x_sf, buf_idx, buf_w


def _check_case(M, H, K, P, scale, seed):
    assert H % 128 == 0 and H % 8 == 0, f"hidden {H} must be a multiple of 128"
    assert M <= P, f"num_tokens {M} must be <= padded_max {P}"
    device = torch.device("cuda")
    torch.manual_seed(seed)

    # Mix of magnitudes so absmax / clamp logic is exercised.
    x = torch.randn(M, H, dtype=torch.bfloat16, device=device) * 4.0
    topk_idx = torch.randint(0, 256, (M, K), dtype=torch.int32, device=device)
    topk_weights = torch.randn(M, K, dtype=torch.float32, device=device)

    ref_x, ref_sf, ref_idx, ref_w = _reference(x, topk_idx, topk_weights, P, scale)
    out_x, out_sf, out_idx, out_w = _run_kernel(x, topk_idx, topk_weights, P, scale)

    if M > 0:
        G = H // 128
        # Scales: same amax/448 formula; allow small fp32 drift.
        torch.testing.assert_close(out_sf[:M], ref_sf[:M], rtol=1e-3, atol=0)

        def _dequant(bx, bsf):
            return bx[:M].float().reshape(M, G, 128) * bsf[:M].unsqueeze(-1)

        out_deq = _dequant(out_x, out_sf)
        ref_deq = _dequant(ref_x, ref_sf)
        # Kernel vs reference: at most one e4m3 mantissa step (1/8) apart.
        torch.testing.assert_close(out_deq, ref_deq, rtol=1.0 / 8, atol=0)
        # Sanity: dequantised values approximate the original input.
        torch.testing.assert_close(
            out_deq, x.float().reshape(M, G, 128), rtol=0.25, atol=1e-4
        )

    # topk_idx pass-through + pad-fill (exact); weights scaled by alpha.
    torch.testing.assert_close(out_idx, ref_idx, rtol=0, atol=0)
    torch.testing.assert_close(out_w, ref_w, rtol=1e-6, atol=0)

    pad = "exact-fill" if M == P else f"{P - M} pad rows"
    print(f"  PASS  M={M} H={H} K={K} P={P} alpha={scale} ({pad})")


# (M, H, K, padded_max, routed_scaling_factor)
CASES = [
    (0, 2048, 8, 32, 1.0),       # zero tokens -> all pad
    (1, 1024, 4, 8, 2.5),
    (7, 2048, 8, 16, 1.0),
    (7, 2048, 8, 16, 2.5),
    (32, 4096, 8, 32, 2.5),
    (128, 7168, 8, 256, 2.5),
    (128, 7168, 8, 128, 1.0),    # exact-fill, no pad rows
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # Accept (and ignore) --num-processes so the shared test runner can pass it.
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    if not _has_hopper():
        major = torch.cuda.get_device_capability()[0]
        print(f"SKIP: SM90/Hopper required (got arch major {major})")
        return 0
    if not hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90"):
        print("SKIP: installed deep_gemm has no mega_moe_pre_dispatch_sm90")
        return 0

    for i, (M, H, K, P, s) in enumerate(CASES):
        _check_case(M, H, K, P, s, seed=(args.seed ^ (0xC0FFEE + i)))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
