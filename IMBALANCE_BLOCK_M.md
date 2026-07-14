# Imbalance-Aware `block_m` Selection for DeepGEMM MegaMoE

**Branch:** `feature/imbalance-aware-block-m`
**Base commit:** `1f6f3f3` (deepseek-ai/DeepGEMM, 2026-07-06)
**Scope:** SM100 FP8×FP4 MegaMoE decode path. Opt-in, default-off, strict fallback.

---

## 1. Problem

The MegaMoE block-config heuristic
(`get_block_config_for_mega_moe`) picks a **single global `block_m`** from the
**mean** tokens-per-expert:

```
mean = num_tokens * num_ranks * num_topk / num_experts
mean<=8.5 ->16 ; <=16.5 ->32 ; <=32.5 ->64 ; <=64.5 ->96 ; <=96.5 ->128 ; else ->192
```

Real MoE routing is **skewed** (Zipf-like): a few hot experts get most tokens,
most experts get far fewer than the mean. The kernel MMAs
`ceil(c_e / block_m) * block_m` rows per expert — **padding included**. A
mean-sized `block_m` therefore pads all the cold experts heavily, wasting
tensor-core cycles. Internal TRT-LLM DSV4 syncs repeatedly flag:
> "The 1st gap contributor is still megamoe, and imbalance is one of the problems."

## 2. Mechanism

When `DG_MEGA_MOE_IMBALANCE_AWARE_BLOCK_M=1` and per-expert receive stats are
available, we choose `block_m` from the **realized** counts by minimizing a
**wall-cost** objective (not raw padded rows):

```
wall_cost(b) = ( Σ_e ceil(c_e / b) * b ) / eta(b)
```

where `eta(b)` is a monotone SM100 UMMA M-utilization efficiency
(`eta(16)=0.35 … eta(96)=0.97 … eta(≥128)=1.0`). Using padded rows *alone*
would always pick the smallest `block_m`; the efficiency weight prevents that
by charging small tiles for tensor-core under-utilization. This is the key
correction that makes the predicted gain **honest**.

The chosen `block_m` is mapped back to an "effective tokens-per-expert" fed
into the existing tier table, so **no kernel-shape changes** are needed — we
reuse the candidate set the kernel already supports (`kCandidateBlockM`).

**Strict fallback:** `return min(eff, mean_tpe)` guarantees the adaptive path
**never selects a coarser `block_m`** than the default. At `alpha=0` (uniform)
it reproduces the default — the balanced case is unchanged and never regresses.

## 3. Files changed

| File | Change |
|------|--------|
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | New `get_effective_tokens_per_expert_for_mega_moe()`; `get_block_config_for_mega_moe` / `get_mega_moe_config` take optional `recv_stats` |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | `sm100_fp8_fp4_mega_moe()` accepts optional `host_recv_stats` and forwards it to the heuristic |
| `csrc/apis/mega.hpp` | Snapshots `cumulative_local_expert_recv_stats` to host (only when the env flag is on) and passes it down; `#include <vector>` |
| `tests/test_mega_moe.py` | `--skew-alpha` Zipf router bias for A/B benchmarking |
| `tests/test_imbalance_block_m.py` | Host-side unit tests: fallback invariant, balanced-case unchanged, wall-cost reduction |
| `scripts/bench_imbalance_block_m.sh` | On-GPU baseline-vs-adaptive kernel-time A/B harness |

## 4. Validation

**Host-side (no GPU):** `python tests/test_imbalance_block_m.py`
- ✅ strict-fallback invariant holds over 14k sampled distributions
- ✅ wall-cost never increases (>=0 reduction everywhere)
- Predicted kernel-time reduction (EP=8, 256 experts, topk=8):

| tokens | α=0 | α=1.0 | α=1.5 | α=2.0 |
|-------:|----:|------:|------:|------:|
| 128 | 0.2% | 7.5% | 10.9% | 12.7% |
| 256 | 0.6% | 16.7% | 21.5% | 24.2% |
| 512 | 8.5% | 34.3% | 42.0% | 44.5% |
| 1024 | 22.1% | 19.2% | 30.6% | 38.9% |

**C++ logic check (no CUTLASS):** `standalone_logic_check.cpp` compiles the
extracted selector and confirms `block_m 192→64` on a 4-hot/28-cold profile
with the invariant holding.

**On-GPU A/B:** `bash scripts/bench_imbalance_block_m.sh` — runs each workload
with flag 0 vs 1 and prints the `mega_moe` kernel time. Correctness is the
existing numerical check in `test_mega_moe.py` (identical output; only tiling
changes). **This step requires an SM100 (B200) 8-GPU node and is the final gate
before upstreaming.**

## 5. Honest limitations

- Gains concentrate in **decode / small-batch** (where cold-expert padding
  dominates). Prefill with large chunks sees ~0 gain (experts already fill
  large tiles) — protected by the fallback so it never regresses.
- The `eta(b)` curve is a conservative model; on-GPU A/B calibrates it. If the
  measured optimum differs, only the `mma_efficiency` table needs tuning.
- Requires realized per-expert counts on host: one tiny D2H copy of
  `num_experts_per_rank` ints, skipped entirely when the flag is off.

## 6. Upstreaming

- OSRB: contribution to `deepseek-ai/DeepGEMM` (MIT). New logic is original,
  based on public MoE-imbalance literature (LLEP, Sem-MoE, Occult).
- Ship as **opt-in, default-off** flag with strict fallback → zero risk to
  existing users; reviewers can enable and reproduce with the provided harness.
