# MegaMoE Adaptive Wave Sizing

MegaMoE processes a fixed number of local experts in each L1→L2 scheduler wave.
The upstream heuristic derives that number from the mean tokens per expert. That
is a good general fallback, but it cannot see the realized routing distribution.

Set `DG_MEGA_MOE_ADAPTIVE_WAVE=1` to enable the opt-in B200 FP8×FP4 policy. The policy:

- reads a delta of `cumulative_local_expert_recv_stats` from the preceding window;
- maintains an independent bounded cache entry for each logical receive-counter
  tensor, so multiple MegaMoE layers can alternate on one host thread;
- caches the sampled distribution and refreshes it every 256 launches. This
  interval was calibrated for stationary B200 routing and amortizes the
  synchronous device-to-host copy on the steady-state path;
- only changes the calibrated shape (EP 8, 256 experts, top-k 8, hidden 7168,
  intermediate 2048) and `127.5 < expected tokens/expert <= 128.5` band;
- skips receive-stat sampling entirely outside that calibrated shape and band;
- uses 8 experts/wave when the active-expert ratio is at or below 0.92;
  balanced and moderate-skew routing retain the upstream size;
- falls back to the upstream wave size on the first call, after counter resets or
  zero-delta samples, outside the calibrated tier, or when the requested tier
  exceeds ring capacity. Reset and zero-delta samples observe the same refresh
  interval instead of synchronizing on every launch.

The deliberately narrow gate is based on same-process, order-balanced 8×B200
measurements. A broader lower bound was rejected after the 96 tokens/expert
point regressed 1.48% under high skew. A balanced-routing wave-12 candidate was
also removed after an independent 6x30 repeat regressed 1.28% with 0/6 wins.
Other broad candidate policies looked promising in process-per-config sweeps,
but did not survive interleaved A/B validation.

## Validation

Correctness and configuration invariance:

```bash
python3 tests/test_mega_moe.py \
  --num-processes 8 --num-tokens 512 --num-max-tokens-per-rank 512 \
  --num-experts 256 --num-topk 8 --hidden 7168 --intermediate-hidden 2048 \
  --skew-alpha 1.5 --validate-config-invariance
```

Robust baseline/adaptive A/B (four measurements per side by default,
alternating order, taking the slowest of all eight ranks in each repetition,
then reporting the median):

```bash
bash scripts/bench_adaptive_wave_ab.sh
```

## B200 performance

The final-source validation used 8×B200, six alternating baseline/adaptive
measurements per case, and 30 profiled kernel calls per measurement. Distributed
latency is reduced as `median_repeat(max_rank(latency))`: each repetition first
takes the slowest of all eight ranks, then the six slowest-rank samples are
reduced by their median.

The common-token sweep used 256 experts, top-k 8, EP 8, hidden 7168, and
intermediate hidden 2048. Values below are observed baseline/adaptive deltas:

| tokens/rank | expected TPE | alpha 0.0 | alpha 1.0 | alpha 1.5 |
|---:|---:|---:|---:|---:|
| 64 | 16 | +0.933% | -1.223% | +0.752% |
| 128 | 32 | -1.015% | -0.077% | -0.361% |
| 256 | 64 | -0.246% | -0.280% | -0.168% |
| 384 | 96 | -0.763% | +0.676% | -1.009% |
| 512 | 128 | -0.789% | +0.041% | **+1.268%** |
| 1024 | 256 | +0.059% | -0.702% | +0.043% |
| 2048 | 512 | +0.087% | +0.041% | +0.114% |

Only the 512-token, alpha-1.5 cell changes the production kernel configuration
(upstream wave 16 to adaptive wave 8). Its exact-delivery-source result was
352.642 → 348.225 µs, a **1.268% speedup with 5/6 wins**. The same wave-8 branch
was positive across six independent runs and multiple B200 nodes: 0.634%,
1.365%, 2.231%, 1.580%, 2.072%, and 1.268% (median **1.473%**).

Every other cell retains the upstream block and wave configuration. Their
approximately -1.23% to +0.93% variation is the measured same-configuration
noise band and is not attributed to the policy. Two broader candidates were
explicitly rejected: 384-token high-skew wave 8 regressed 1.481% with 0/6 wins,
and 512-token balanced wave 12 regressed 1.276% with 0/6 wins in an independent
repeat.

Forced-wave calibration is available separately:

```bash
bash scripts/bench_mega_moe_wave_size.sh
```

`DG_MEGA_MOE_FORCE_EXPERTS_PER_WAVE` and `DG_MEGA_MOE_FORCE_BLOCK_M` are
benchmark-only overrides. They fail loudly on invalid values and are not used by
the production adaptive policy.
