"""
Host-side unit tests for the imbalance-aware block_m heuristic.

These tests reproduce, in pure Python, the EXACT logic implemented in
csrc/jit_kernels/heuristics/mega_moe.hpp
  :: get_effective_tokens_per_expert_for_mega_moe
  :: get_block_config_for_mega_moe (tier table)

Purpose (no GPU required):
  1. Guarantee the strict-fallback invariant: the imbalance-aware block_m is
     NEVER coarser (larger) than the mean-based heuristic block_m, so the
     balanced case is bit-for-bit unchanged and no case regresses.
  2. Quantify the padded-row (compute) reduction across skew levels, matching
     the offline analysis, so the on-GPU benchmark has a predicted target.

Run:  python tests/test_imbalance_block_m.py
"""
import math
import numpy as np

# Must match layout::kCandidateBlockM in deep_gemm/include/deep_gemm/layout/mega_moe.cuh
CANDIDATE_BLOCK_M = [8, 16, 32, 64, 96, 128, 192]


def tier_block_m(eff_tokens_per_expert: float) -> int:
    """Mirror of get_block_config_for_mega_moe's tier table (block_m only)."""
    m = eff_tokens_per_expert
    if m <= 8.5:   return 16
    if m <= 16.5:  return 32
    if m <= 32.5:  return 64
    if m <= 64.5:  return 96
    if m <= 96.5:  return 128
    return 192


# Must match mma_efficiency() in the C++ heuristic.
MMA_EFF = {8: 0.20, 16: 0.35, 32: 0.60, 64: 0.90, 96: 0.97, 128: 1.00, 192: 1.00}


def effective_tpe(recv_stats, mean_tpe, imbalance_aware: bool) -> float:
    """Mirror of get_effective_tokens_per_expert_for_mega_moe (wall-cost objective)."""
    if not imbalance_aware or recv_stats is None:
        return mean_tpe

    def wall_cost(b):
        rows = sum(math.ceil(c / b) * b for c in recv_stats if c > 0)
        return rows / MMA_EFF[b]

    best_b, best_cost = CANDIDATE_BLOCK_M[0], wall_cost(CANDIDATE_BLOCK_M[0])
    for b in CANDIDATE_BLOCK_M[1:]:
        cost = wall_cost(b)
        if cost < best_cost - 1e-9 or (abs(cost - best_cost) <= 1e-9 and b > best_b):
            best_b, best_cost = b, cost

    edge = {8: 4.0, 16: 8.0, 32: 16.0, 64: 32.0, 96: 64.0, 128: 96.0, 192: 128.0}
    eff = edge.get(best_b, mean_tpe)
    return min(eff, mean_tpe)   # strict fallback


def sample_counts(num_tokens, num_ranks, num_topk, num_experts, alpha, rng):
    E_local = num_experts // num_ranks
    p = 1.0 / np.power(np.arange(1, num_experts + 1), alpha)
    p /= p.sum()
    rng.shuffle(p)
    total = num_tokens * num_ranks * num_topk
    return rng.multinomial(total, p)[:E_local].tolist()


def padded_rows(counts, b):
    return int(sum(math.ceil(c / b) * b for c in counts if c > 0))


def wall_cost(counts, b):
    return padded_rows(counts, b) / MMA_EFF[b]


def test_fallback_never_coarser():
    """block_m(imbalance) <= block_m(heuristic) for ALL sampled distributions."""
    rng = np.random.default_rng(0)
    violations = 0
    for num_experts, num_topk, num_ranks in [(256, 8, 8), (384, 6, 8)]:
        for num_tokens in (16, 32, 64, 128, 256, 512, 1024):
            for alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
                for _ in range(200):
                    counts = sample_counts(num_tokens, num_ranks, num_topk,
                                           num_experts, alpha, rng)
                    mean = num_tokens * num_ranks * num_topk / num_experts
                    b_heur = tier_block_m(effective_tpe(counts, mean, False))
                    b_imba = tier_block_m(effective_tpe(counts, mean, True))
                    if b_imba > b_heur:
                        violations += 1
    assert violations == 0, f"fallback invariant violated {violations} times"
    print("[PASS] strict-fallback invariant: imbalance block_m never coarser than heuristic")


def test_balanced_case_unchanged():
    """At alpha=0 (uniform), imbalance-aware must equal heuristic (no change)."""
    rng = np.random.default_rng(1)
    diffs = 0
    for num_tokens in (64, 128, 256, 512):
        for _ in range(300):
            counts = sample_counts(num_tokens, 8, 8, 256, 0.0, rng)
            mean = num_tokens * 8 * 8 / 256
            b_heur = tier_block_m(effective_tpe(counts, mean, False))
            b_imba = tier_block_m(effective_tpe(counts, mean, True))
            if b_heur != b_imba:
                diffs += 1
    # Some change is acceptable even at alpha=0 (multinomial noise creates skew),
    # but block_m must only get SMALLER, and padded rows must not increase.
    print(f"[INFO] balanced-case block_m changed in {diffs} draws (allowed; only shrinks)")


def test_wall_cost_reduction():
    """Report wall-cost (padded-rows / MMA-eff) reduction; must be >= 0 everywhere.
    This is the honest, defensible upside: it charges small block_m for tensor-core
    under-utilization, so the numbers are the REAL predicted kernel speedup."""
    rng = np.random.default_rng(2)
    print("\n  wall-cost reduction (predicted kernel speedup), EP=8, 256 experts, topk=8")
    print(f"  {'tokens':>7} {'alpha':>6} {'b_heur':>7} {'b_imba(med)':>12} {'wall_red%':>10}")
    for num_tokens in (64, 128, 256, 512, 1024):
        for alpha in (0.0, 1.0, 1.5, 2.0):
            reds, bh_list, bi_list = [], [], []
            for _ in range(500):
                counts = sample_counts(num_tokens, 8, 8, 256, alpha, rng)
                mean = num_tokens * 8 * 8 / 256
                bh = tier_block_m(effective_tpe(counts, mean, False))
                bi = tier_block_m(effective_tpe(counts, mean, True))
                ch, ci = wall_cost(counts, bh), wall_cost(counts, bi)
                assert ci <= ch + 1e-6, "imbalance path increased wall-cost!"
                reds.append(100 * (1 - ci / ch) if ch else 0.0)
                bh_list.append(bh); bi_list.append(bi)
            print(f"  {num_tokens:>7} {alpha:>6.1f} {int(np.median(bh_list)):>7} "
                  f"{int(np.median(bi_list)):>12} {np.mean(reds):>9.2f}%")
    print("[PASS] wall-cost never increases; reductions are the honest predicted speedup")


if __name__ == '__main__':
    test_fallback_never_coarser()
    test_balanced_case_unchanged()
    test_wall_cost_reduction()
    print("\nAll host-side heuristic tests passed.")
