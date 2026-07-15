# MegaMoE Adaptive Wave Sizing

For a detailed Chinese design note, see [`ADAPTIVE_WAVE_ZH.md`](ADAPTIVE_WAVE_ZH.md).

MegaMoE processes a fixed number of local experts in each L1→L2 scheduler wave.
The upstream heuristic derives that number from the mean tokens per expert. That
is a good general fallback, but it cannot see the realized routing distribution.

Set `DG_MEGA_MOE_ADAPTIVE_WAVE=1` to enable the opt-in B200 FP8×FP4 policy. The policy:

- reads a delta of `cumulative_local_expert_recv_stats` from the preceding window;
- caches the sampled distribution and refreshes it every 256 launches, avoiding a
  synchronous device-to-host copy on the steady-state path;
- only changes the calibrated shape (EP 8, 256 experts, top-k 8, hidden 7168,
  intermediate 2048) and `127.5 < expected tokens/expert <= 128.5` band;
- skips receive-stat sampling entirely outside that calibrated shape and band;
- uses 8 experts/wave when the active-expert ratio is at or below 0.92;
  balanced and moderate-skew routing retain the upstream size;
- falls back to the upstream wave size on the first call, after counter resets,
  outside the calibrated tier, or when the requested tier exceeds ring capacity.

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

Forced-wave calibration is available separately:

```bash
bash scripts/bench_mega_moe_wave_size.sh
```

`DG_MEGA_MOE_FORCE_EXPERTS_PER_WAVE` and `DG_MEGA_MOE_FORCE_BLOCK_M` are
benchmark-only overrides. They fail loudly on invalid values and are not used by
the production adaptive policy.
