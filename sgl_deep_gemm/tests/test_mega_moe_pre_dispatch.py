# Bytewise + correctness probe for `deep_gemm.mega_moe_pre_dispatch`.
#
# The fused pre-dispatch kernel produces the exact byte layout DeepGEMM's
# mega-MoE symmetric `x`, `x_sf`, `topk_idx`, and `topk_weights` slots expect.
# This test verifies bit-for-bit equivalence against the in-tree host helpers
# (`per_token_cast_to_fp8`, `per_token_cast_to_fp4`) for both the FP8 and the
# packed FP4 dtype branches, plus the pad-fill correctness contract.
#
# Single-GPU; no distributed init needed.

import argparse
import sys
import torch

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8


def _alloc_outputs(padded_max: int, hidden: int, top_k: int,
                   group_size: int, use_fp4_acts: bool):
    num_groups = hidden // group_size
    assert num_groups % 4 == 0
    if use_fp4_acts:
        buf_x = torch.empty((padded_max, hidden // 2), dtype=torch.int8, device='cuda')
    else:
        buf_x = torch.empty((padded_max, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    buf_x_sf = torch.empty((padded_max, num_groups // 4), dtype=torch.int32, device='cuda')
    buf_topk_idx = torch.empty((padded_max, top_k), dtype=torch.int64, device='cuda')
    buf_topk_weights = torch.empty((padded_max, top_k), dtype=torch.float32, device='cuda')
    # Sentinel-fill so any write-correctness bug shows up as a non-zero diff.
    buf_x.fill_(0)
    buf_x_sf.fill_(0)
    buf_topk_idx.fill_(0)
    buf_topk_weights.fill_(0)
    return buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights


def _run_one(use_fp4_acts: bool, args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    M = args.num_tokens
    P = args.padded_max
    H = args.hidden
    K = args.top_k
    G = args.group_size
    assert P >= M, 'padded_max must be >= num_tokens'

    # --- Inputs (BF16 acts, int32 topk_idx, float topk_weights) ---
    x = torch.randn((M, H), dtype=torch.bfloat16, device='cuda')
    # Use plausible expert ids in [0, num_experts) and float weights.
    num_experts = args.num_experts
    topk_idx = torch.randint(0, num_experts, (M, K), dtype=torch.int32, device='cuda')
    topk_weights = torch.randn((M, K), dtype=torch.float32, device='cuda')

    buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights = _alloc_outputs(P, H, K, G, use_fp4_acts)

    # --- Kernel under test ---
    deep_gemm.mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens=M, group_size=G, use_fp4_acts=use_fp4_acts,
    )
    torch.cuda.synchronize()

    # --- Reference (host helper) ---
    if use_fp4_acts:
        ref_x, ref_sf = per_token_cast_to_fp4(
            x, use_ue8m0=True, gran_k=G, use_packed_ue8m0=True)
    else:
        ref_x, ref_sf = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=G, use_packed_ue8m0=True)

    # --- Bytewise compare on valid-token rows ---
    if use_fp4_acts:
        # ref_x is int8 (M, H/2); buf_x[:M] is int8 (M, H/2). Compare raw bytes.
        kernel_bytes = buf_x[:M].view(torch.uint8)
        ref_bytes = ref_x.view(torch.uint8)
    else:
        # ref_x is float8_e4m3fn (M, H); compare via uint8 view.
        kernel_bytes = buf_x[:M].view(torch.uint8)
        ref_bytes = ref_x.view(torch.uint8)
    diff_x = (kernel_bytes != ref_bytes)
    if diff_x.any().item():
        bad = diff_x.nonzero()
        first = bad[0].tolist()
        i, j = first[0], first[1]
        raise AssertionError(
            f'[{"FP4" if use_fp4_acts else "FP8"}] buf_x mismatch '
            f'at row {i}, col {j}: kernel={int(kernel_bytes[i, j])} '
            f'ref={int(ref_bytes[i, j])} (total mismatches={int(diff_x.sum())})')

    # SF byte layout: (M, num_groups/4) int32 → (M, num_groups) UE8M0 bytes.
    kernel_sf_bytes = buf_x_sf[:M].view(torch.uint8)
    ref_sf_bytes = ref_sf.view(torch.uint8)
    diff_sf = (kernel_sf_bytes != ref_sf_bytes)
    if diff_sf.any().item():
        bad = diff_sf.nonzero()
        first = bad[0].tolist()
        i, j = first[0], first[1]
        raise AssertionError(
            f'[{"FP4" if use_fp4_acts else "FP8"}] buf_x_sf mismatch '
            f'at row {i}, byte {j}: kernel={int(kernel_sf_bytes[i, j])} '
            f'ref={int(ref_sf_bytes[i, j])} (total mismatches={int(diff_sf.sum())})')

    # --- topk pass-through and pad-fill ---
    # Valid rows: int32 → int64 widening match.
    if not torch.equal(buf_topk_idx[:M], topk_idx.to(torch.int64)):
        raise AssertionError('topk_idx pass-through mismatch on valid rows')
    if not torch.equal(buf_topk_weights[:M], topk_weights):
        raise AssertionError('topk_weights pass-through mismatch on valid rows')
    # Pad rows.
    if P > M:
        if not torch.all(buf_topk_idx[M:] == -1).item():
            raise AssertionError('pad rows of buf_topk_idx must equal -1')
        if not torch.all(buf_topk_weights[M:] == 0.0).item():
            raise AssertionError('pad rows of buf_topk_weights must equal 0.0')

    print(f'  PASS  '
          f'[{"FP4" if use_fp4_acts else "FP8"}] '
          f'M={M} P={P} H={H} K={K} G={G} — bytewise equal vs host helper '
          f'+ pad-fill correct')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tokens', type=int, default=512)
    parser.add_argument('--padded-max', type=int, default=576)  # > num_tokens to exercise pad
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--group-size', type=int, default=32)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', choices=['fp8', 'fp4', 'both'], default='both')
    args = parser.parse_args()

    if args.dtype in ('fp8', 'both'):
        _run_one(use_fp4_acts=False, args=args)
    if args.dtype in ('fp4', 'both'):
        _run_one(use_fp4_acts=True, args=args)
    print('OK')


if __name__ == '__main__':
    sys.exit(main())
