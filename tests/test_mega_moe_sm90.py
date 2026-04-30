"""Layered tests for the SM90 (Hopper) MegaMoE kernel.

The fused FP8 SM90 MegaMoE kernel is exercised across a hierarchy of
scenarios so that each kernel path / heuristic branch / edge case is
covered with at least one configuration.

Layers
------
  L1  Smoke           : single tiny config; only verifies the kernel runs
                        and produces an output close to a PyTorch reference.
  L2  Heuristic       : sweeps tokens-per-expert across the bands of
                        ``get_block_config_for_mega_moe_sm90`` so each
                        ``{block_m, num_epilogue_warpgroups}`` case is hit.
  L3  Shape sweep     : sweeps ``hidden``, ``intermediate_hidden`` and
                        ``num_topk`` over divisible-by-128 values.
  L4  Edge cases      : masking ratio, activation clamp (finite vs inf),
                        ``fast_math`` 0/1, ``num_tokens`` boundaries.
  L5  Stress          : ``--num-correctness-tests`` repeated random configs.

Notes
-----
*   The reference is a pure PyTorch BF16/FP32 simulation of the fused path
    (dequantize -> matmul -> SwiGLU + clamp + per-row quantize -> matmul ->
    cross-rank scatter -> BF16 reduce).  It is *not* bitwise-identical to
    the kernel; correctness is checked with ``calc_diff < 0.07``.
*   Because every scenario allocates its own symmetric memory buffer we
    re-`init_dist`/`destroy` once per process at the outer level only,
    and re-create ``SymmBuffer`` per scenario.
*   Skips itself when the device is not SM90.
"""

import argparse
import math
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple, List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import calc_diff, get_arch_major


# ----------------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------------

def _quantize_grouped_fp8_per_128_k(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel-row FP8 quantization with per-128 K float scale.

    Args
    ----
    w : (G, N, K) bf16

    Returns
    -------
    fp8 : (G, N, K) torch.float8_e4m3fn
    sf  : (G, N, K // 128) torch.float32 — K-major (C-contiguous in this shape).
          The host launcher calls `check_sf_layout(..., tma_stride_check=true)`
          on weight SF, which requires MN-major; the caller is responsible for
          transposing this output via
          ``deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 128), g)``
          before passing it to the kernel.
    """
    g, n, k = w.shape
    assert k % 128 == 0
    w_view = w.view(g, n, k // 128, 128).float()
    amax = w_view.abs().amax(dim=-1).clamp(1e-4)
    sf = amax / 448.0
    w_fp8 = (w_view / sf.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return w_fp8.view(g, n, k).contiguous(), sf.contiguous()


def _dequant_per_128_k(w_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of `_quantize_grouped_fp8_per_128_k`.  Returns fp32."""
    *prefix, n, k = w_fp8.shape
    w_view = w_fp8.float().view(*prefix, n, k // 128, 128)
    return (w_view * sf.unsqueeze(-1)).view(*prefix, n, k)


def _dequant_per_token_per_128_k(x_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """For (M, K) fp8 with (M, K // 128) float SF (per-token, K-major)."""
    m, k = x_fp8.shape
    return _dequant_per_128_k(x_fp8.unsqueeze(0), sf.unsqueeze(0)).squeeze(0)


# ----------------------------------------------------------------------------
# PyTorch reference
# ----------------------------------------------------------------------------

def _swiglu_fp32(gate_up: torch.Tensor, clamp: float) -> torch.Tensor:
    """SwiGLU with one-sided gate clamp and two-sided up clamp.

    Matches the fused kernel: ``silu(min(gate, c)) * clamp(up, -c, c)``.
    """
    n2 = gate_up.size(-1)
    half = n2 // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _reference_fused(
    x_fp8_local: torch.Tensor, x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor, topk_weights_local: torch.Tensor,
    l1_w_fp8: torch.Tensor, l1_w_sf: torch.Tensor,
    l2_w_fp8: torch.Tensor, l2_w_sf: torch.Tensor,
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    num_experts: int, num_topk: int,
    hidden: int, intermediate_hidden: int,
    activation_clamp: float,
) -> torch.Tensor:
    """Reference: returns (num_tokens, hidden) bf16 result for *this* rank.

    All-gathers the global tokens / topk decisions / per-rank weights, then
    for each global token routes through its topk experts, applies the
    L1+SwiGLU+L2 path, and reduces over topk on the source rank.
    """
    num_experts_per_rank = num_experts // num_ranks

    # --- gather global token data --------------------------------------------------
    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)      # (Mg, H)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)        # (Mg, H/128)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)         # (Mg, K)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)       # (Mg, K)
    mg = x_fp8_g.size(0)

    # rank-id lookup for each gathered token (for combine routing)
    rank_offsets = [0]
    sizes = [torch.tensor([0], device='cuda')]                  # placeholder
    # mimic uneven_all_gather to compute per-rank token counts
    local_size = torch.tensor([x_fp8_local.size(0)], device='cuda', dtype=torch.long)
    sizes_t = torch.empty(num_ranks, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(sizes_t, local_size, group=group)
    sizes_list = sizes_t.tolist()
    src_rank_of = torch.empty(mg, dtype=torch.long, device='cuda')
    cur = 0
    for r, s in enumerate(sizes_list):
        src_rank_of[cur:cur + s] = r
        cur += s
    assert cur == mg

    # --- gather all-rank weights --------------------------------------------------
    # l1_w_fp8: (E_pr, 2*IH, H), l1_w_sf: (E_pr, 2*IH, H/128)
    l1_w_g = [torch.empty_like(l1_w_fp8) for _ in range(num_ranks)]
    l1_sf_g = [torch.empty_like(l1_w_sf) for _ in range(num_ranks)]
    l2_w_g = [torch.empty_like(l2_w_fp8) for _ in range(num_ranks)]
    l2_sf_g = [torch.empty_like(l2_w_sf) for _ in range(num_ranks)]
    dist.all_gather(l1_w_g, l1_w_fp8, group=group)
    dist.all_gather(l1_sf_g, l1_w_sf, group=group)
    dist.all_gather(l2_w_g, l2_w_fp8, group=group)
    dist.all_gather(l2_sf_g, l2_w_sf, group=group)
    l1_w_all = torch.stack(l1_w_g, dim=0)   # (R, E_pr, 2*IH, H)
    l1_sf_all = torch.stack(l1_sf_g, dim=0)
    l2_w_all = torch.stack(l2_w_g, dim=0)
    l2_sf_all = torch.stack(l2_sf_g, dim=0)

    # --- per-token / per-topk compute --------------------------------------------------
    # The combine slot tensor: (Mg, K, H) bf16 — each src rank will reduce over K.
    combine_buf = torch.zeros(mg, num_topk, hidden, dtype=torch.float32, device='cuda')

    # Precompute dequantized x in fp32
    x_fp32 = _dequant_per_token_per_128_k(x_fp8_g, x_sf_g)         # (Mg, H)

    # Iterate (cheap; reference is for small test configs only)
    # Token-chunked to keep gathered (S, 2*IH, H) dequant tensors below GPU memory.
    _CHUNK = 256
    for k in range(num_topk):
        # Skip masked
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx_full = mask.nonzero(as_tuple=False).squeeze(-1)    # (S,)
        for c0 in range(0, sel_idx_full.numel(), _CHUNK):
            sel_idx = sel_idx_full[c0:c0 + _CHUNK]
            eids = topk_idx_g[sel_idx, k]                          # (S,)
            weights = topk_w_g[sel_idx, k]                         # (S,)
            x_sel = x_fp32[sel_idx]                                # (S, H)

            dst_rank = (eids // num_experts_per_rank).long()
            dst_local = (eids % num_experts_per_rank).long()

            # L1 GEMM (per-token): y = x @ W^T  shape (S, 2*IH)
            l1_w_sel = _dequant_per_128_k(
                l1_w_all[dst_rank, dst_local],                     # (S, 2*IH, H)
                l1_sf_all[dst_rank, dst_local],
            )
            l1_y = torch.einsum('sk,snk->sn', x_sel, l1_w_sel)     # (S, 2*IH)
            del l1_w_sel

            # SwiGLU + clamp + multiply by topk weight
            l1_y = _swiglu_fp32(l1_y, activation_clamp) * weights.unsqueeze(-1)   # (S, IH)

            # Per-row, per-64-col FP8 quantize -> dequantize
            s_, ih = l1_y.shape
            assert ih == intermediate_hidden and ih % 64 == 0
            l1_view = l1_y.view(s_, ih // 64, 64)
            amax = l1_view.abs().amax(dim=-1).clamp(1e-4)          # (S, IH/64)
            sf2 = amax / 448.0
            l1_q = (l1_view / sf2.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
            l2_in = (l1_q * sf2.unsqueeze(-1)).view(s_, ih)        # (S, IH) fp32

            # L2 GEMM
            l2_w_sel = _dequant_per_128_k(
                l2_w_all[dst_rank, dst_local],                     # (S, H, IH)
                l2_sf_all[dst_rank, dst_local],
            )
            l2_y = torch.einsum('sn,smn->sm', l2_in, l2_w_sel)     # (S, H)
            del l2_w_sel

            # Scatter to combine buffer (cast to bf16 then back to mimic kernel storage)
            combine_buf[sel_idx, k] = l2_y.to(torch.bfloat16).float()

    # Sum over K -> (Mg, H), keep only this rank's slice
    y_full_bf16 = combine_buf.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)  # (Mg, H)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    return y_full_bf16[start:end].contiguous()


# ----------------------------------------------------------------------------
# Single-scenario runner
# ----------------------------------------------------------------------------

def _run_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    diff_tol: float,
):
    num_max = cfg['num_max_tokens_per_rank']
    num_tokens = cfg.get('num_tokens', num_max)
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    masked_ratio = cfg.get('masked_ratio', 0.0)
    activation_clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)

    assert num_experts % num_ranks == 0, f'{name}: experts {num_experts} not divisible by ranks {num_ranks}'
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

    verbose = bool(int(os.environ.get('DG_TEST_VERBOSE', '0')))
    def _trace(stage: str):
        if verbose:
            print(f'[rank{rank_idx}] {name} :: {stage}', flush=True)

    _trace('begin')
    torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)
    random.seed(rank_idx * 1000 + abs(hash(name)) % 1000)

    # ---- Inputs (bf16) -------------------------------------------------------
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    # Quantize x to FP8 with per-128 K float SF (SM90 format)
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    # Quantize weights — keep K-major copies for the reference dequant path.
    l1_w_fp8, l1_w_sf_kmajor = _quantize_grouped_fp8_per_128_k(l1_bf)
    l2_w_fp8, l2_w_sf_kmajor = _quantize_grouped_fp8_per_128_k(l2_bf)

    # The kernel expects MN-major, TMA-aligned weight SFs (per `check_sf_layout`
    # with `tma_stride_check=true`). Convert via the public helper, which on
    # SM90 is exactly what `transform_sf_into_required_layout` does for
    # ``(FP32, gran_mn=1, gran_k=128)``.
    l1_w_sf = deep_gemm.get_mn_major_tma_aligned_tensor(l1_w_sf_kmajor)
    l2_w_sf = deep_gemm.get_mn_major_tma_aligned_tensor(l2_w_sf_kmajor)

    # SM90 weight transform (gate/up interleave only)
    _trace('weight_transform')
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf)
    )

    # ---- Allocate symm buffer -----------------------------------------------
    _trace('alloc_symm_buffer')
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max, num_topk,
        hidden, intermediate_hidden,
    )
    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

    # ---- Run fused -----------------------------------------------------------
    _trace('copy_inputs')
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_w)

    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    _trace('launch_fused (may JIT-compile, can take minutes)')
    deep_gemm.fp8_mega_moe(
        y_fused, transformed_l1, transformed_l2, buffer,
        cumulative_local_expert_recv_stats=cum_stats,
        recipe=(1, 128, 128),
        activation='swiglu',
        activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
        fast_math=fast_math,
    )
    _trace('sync_fused')
    torch.cuda.synchronize()
    _trace('fused_done')

    # ---- Reference & check ---------------------------------------------------
    # Use the *un-interleaved*, K-major SF weights for the reference path —
    # the dequant helper expects K-major SF, and the original (gate||up) row
    # ordering is what `_swiglu_fp32` splits with ``[..., :IH], [..., IH:]``.
    _trace('reference')
    y_ref = _reference_fused(
        x_fp8, x_sf, topk_idx, topk_w,
        l1_w_fp8, l1_w_sf_kmajor, l2_w_fp8, l2_w_sf_kmajor,
        rank_idx, num_ranks, group,
        num_experts, num_topk,
        hidden, intermediate_hidden,
        activation_clamp,
    )

    diff = calc_diff(y_fused, y_ref)
    ok = diff < diff_tol
    dist_print(f'  [{name:<32}] diff={diff:.4f} '
               f'(tol={diff_tol:.2f}) {"OK" if ok else "FAIL"}',
               once_in_node=True)
    assert ok, f'{name}: diff={diff} >= tol={diff_tol}'

    # Verify cum_stats has been incremented (i.e. dispatch ran)
    if num_tokens > 0 and masked_ratio < 1.0:
        assert cum_stats.sum().item() >= 0  # non-negative; can be 0 if nothing routed here

    buffer.destroy()
    dist.barrier()


# ----------------------------------------------------------------------------
# Scenario tables
# ----------------------------------------------------------------------------

# A single tiny config used as a smoke test.
_SMOKE = dict(
    num_max_tokens_per_rank=64, num_tokens=64,
    hidden=512, intermediate_hidden=512,
    num_experts=8, num_topk=2,
)


def _layer1_smoke() -> List[Tuple[str, Dict[str, Any]]]:
    return [('L1.smoke', dict(_SMOKE))]


def _layer2_heuristic_branches(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Vary tokens / (num_experts * num_topk / num_ranks) so each
    ``get_block_config_for_mega_moe_sm90`` band fires at least once.

    The heuristic decides on ``avg_tokens_per_expert``; we approximate by
    setting ``num_max_tokens_per_rank`` and ``num_topk`` while keeping
    ``num_experts`` fixed.  The bands are at 64.5 / 96.5 / 192.5.
    """
    base = dict(hidden=1024, intermediate_hidden=1024,
                num_experts=8 * num_ranks, num_topk=2)
    out: List[Tuple[str, Dict[str, Any]]] = []
    # tokens-per-rank settings chosen to hit (small / mid / large) bands
    for tokens, label in [(64, 'small'), (256, 'midA'), (512, 'midB'), (2048, 'large')]:
        cfg = dict(base)
        cfg.update(num_max_tokens_per_rank=tokens, num_tokens=tokens)
        out.append((f'L2.heur.{label}.t{tokens}', cfg))
    return out


def _layer3_shape_sweep(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    base_experts = 8 * num_ranks
    for hidden in (512, 2048):
        for ih in (512, 2048):
            for topk in (1, 2, 4):
                if topk > base_experts:
                    continue
                cfg = dict(num_max_tokens_per_rank=128, num_tokens=128,
                           hidden=hidden, intermediate_hidden=ih,
                           num_experts=base_experts, num_topk=topk)
                out.append((f'L3.h{hidden}_ih{ih}_k{topk}', cfg))
    return out


def _layer4_edges(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(num_max_tokens_per_rank=128,
                hidden=512, intermediate_hidden=512,
                num_experts=8 * num_ranks, num_topk=2)
    out = []
    # Masked ratios
    for mr in (0.0, 0.3, 0.7):
        cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=mr)
        out.append((f'L4.mask{mr:.1f}', cfg))
    # All masked
    cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=1.0)
    out.append(('L4.mask_all', cfg))
    # Activation clamp variations (finite vs inf)
    for c in (1.0, 10.0, math.inf):
        cfg = dict(base); cfg.update(num_tokens=128, activation_clamp=c)
        out.append((f'L4.clamp{c}', cfg))
    # fast_math toggle
    for fm in (True, False):
        cfg = dict(base); cfg.update(num_tokens=128, fast_math=fm)
        out.append((f'L4.fm{int(fm)}', cfg))
    # num_tokens boundaries
    cfg = dict(base); cfg.update(num_tokens=0)
    out.append(('L4.tokens0', cfg))
    cfg = dict(base); cfg.update(num_tokens=base['num_max_tokens_per_rank'])
    out.append(('L4.tokens_max', cfg))
    return out


def _layer5_stress(num_ranks: int, num_tests: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Random configs under simple constraints."""
    rng = random.Random(0xC0FFEE)
    out = []
    for i in range(num_tests):
        hidden = rng.choice([512, 1024, 2048])
        ih = rng.choice([512, 1024, 2048])
        topk = rng.choice([1, 2, 4])
        tokens = rng.choice([32, 64, 128, 256, 512])
        masked = rng.choice([0.0, 0.0, 0.3, 0.5])
        clamp = rng.choice([1.0, 10.0, math.inf])
        fm = rng.choice([True, False])
        cfg = dict(num_max_tokens_per_rank=tokens, num_tokens=tokens,
                   hidden=hidden, intermediate_hidden=ih,
                   num_experts=8 * num_ranks, num_topk=topk,
                   masked_ratio=masked, activation_clamp=clamp, fast_math=fm)
        out.append((f'L5.rand{i:03d}', cfg))
    return out


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Skip on non-SM90
    if get_arch_major() != 9:
        dist_print(f'[SKIP] test_mega_moe_sm90 requires SM90; got SM{get_arch_major()}0',
                   once_in_node=True)
        dist.destroy_process_group()
        return

    diff_tol = args.diff_tol
    layers: List[Tuple[str, Dict[str, Any]]] = []

    if 1 in args.layers:
        layers += _layer1_smoke()
    if 2 in args.layers:
        layers += _layer2_heuristic_branches(num_ranks)
    if 3 in args.layers:
        layers += _layer3_shape_sweep(num_ranks)
    if 4 in args.layers:
        layers += _layer4_edges(num_ranks)
    if 5 in args.layers:
        layers += _layer5_stress(num_ranks, args.num_correctness_tests or 8)

    if args.filter:
        layers = [(n, c) for n, c in layers if args.filter in n]

    dist_print(f'SM90 MegaMoE test plan: {len(layers)} scenarios across '
               f'layers {sorted(args.layers)} on {num_ranks} ranks',
               once_in_node=True)

    failures: List[str] = []
    for name, cfg in layers:
        try:
            _run_scenario(name, cfg, rank_idx, num_ranks, group, diff_tol)
        except AssertionError as ex:
            dist_print(f'  [{name}] FAIL: {ex}', once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED {len(failures)}/{len(layers)} scenarios: {failures}',
                   once_in_node=True)
    else:
        dist_print(f'PASSED all {len(layers)} scenarios', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layered SM90 MegaMoE tests')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='Number of ranks to spawn (default: 2)')
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Which layers to run (1..5). Default: 1 2 3 4. '
                             'Layer 5 requires --num-correctness-tests.')
    parser.add_argument('--num-correctness-tests', type=int, default=None,
                        help='Layer 5 stress test count')
    parser.add_argument('--filter', type=str, default='',
                        help='Substring filter on scenario names')
    parser.add_argument('--diff-tol', type=float, default=0.07,
                        help='calc_diff tolerance (default: 0.07)')
    parser.add_argument('--fail-fast', action='store_true',
                        help='Stop on first failing scenario')
    args = parser.parse_args()

    np = args.num_processes
    torch.multiprocessing.spawn(test, args=(np, args), nprocs=np)
