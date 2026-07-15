#!/bin/bash
# Robust same-process, order-balanced A/B for Adaptive Wave Sizing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPERTS=${EXPERTS:-256}
TOPK=${TOPK:-8}
HIDDEN=${HIDDEN:-7168}
INTER=${INTER:-2048}
NPROC=${NPROC:-8}
TOKENS_LIST=${TOKENS_LIST:-"128 256 512 1024"}
ALPHAS=${ALPHAS:-"0.0 1.0 1.5"}
AB_REPEATS=${AB_REPEATS:-4}
AB_NUM_TESTS=${AB_NUM_TESTS:-20}
RAW_LOG_DIR=${RAW_LOG_DIR:-"adaptive_wave_ab_raw_${SLURM_JOB_ID:-local}"}
mkdir -p "$RAW_LOG_DIR"

echo "experts=$EXPERTS topk=$TOPK hidden=$HIDDEN inter=$INTER EP=$NPROC"
echo "raw_logs=$RAW_LOG_DIR repeats=$AB_REPEATS num_tests=$AB_NUM_TESTS"
echo "tokens,alpha,baseline_slowest_us,adaptive_slowest_us,speedup_pct"

for tokens in $TOKENS_LIST; do
  for alpha in $ALPHAS; do
    raw_log="$RAW_LOG_DIR/tokens_${tokens}_alpha_${alpha}.log"
    if ! output=$(python3 tests/test_mega_moe.py \
      --num-processes "$NPROC" \
      --num-tokens "$tokens" --num-max-tokens-per-rank "$tokens" \
      --num-experts "$EXPERTS" --num-topk "$TOPK" \
      --hidden "$HIDDEN" --intermediate-hidden "$INTER" \
      --skew-alpha "$alpha" --num-correctness-tests 0 \
      --validate-adaptive-wave-ab \
      --adaptive-wave-ab-repeats "$AB_REPEATS" \
      --adaptive-wave-ab-num-tests "$AB_NUM_TESTS" \
      2>&1); then
      printf '%s\n' "$output" > "$raw_log"
      printf '%s\n' "$output" >&2
      exit 1
    fi
    printf '%s\n' "$output" > "$raw_log"
    samples=$(printf '%s\n' "$output" | grep '^WAVE_AB_SAMPLES rank=' || true)
    if [[ $(printf '%s\n' "$samples" | grep -c '^WAVE_AB_SAMPLES rank=') -ne "$NPROC" ]]; then
      printf 'ERROR: expected %s A/B sample rows for tokens=%s alpha=%s\n' \
        "$NPROC" "$tokens" "$alpha" >&2
      printf '%s\n' "$output" >&2
      exit 1
    fi

    # System latency is the slowest rank in each repetition. Take that maximum
    # first, then the median across repetitions; max(median(rank)) can hide a
    # different straggler in each repetition and is not the distributed
    # critical path.
    if ! metrics=$(printf '%s\n' "$samples" | awk \
      -v expected_ranks="$NPROC" -v expected_repeats="$AB_REPEATS" '
      {
        split($0, baseline_parts, "baseline_us=")
        split(baseline_parts[2], sample_parts, " adaptive_us=")
        num_baseline = split(sample_parts[1], baseline, ",")
        num_adaptive = split(sample_parts[2], adaptive, ",")
        if (num_baseline != expected_repeats || num_adaptive != expected_repeats)
          exit 2
        for (i = 1; i <= expected_repeats; ++i) {
          if (ranks == 0 || baseline[i] > baseline_max[i])
            baseline_max[i] = baseline[i]
          if (ranks == 0 || adaptive[i] > adaptive_max[i])
            adaptive_max[i] = adaptive[i]
        }
        ++ranks
      }
      END {
        if (ranks != expected_ranks)
          exit 3
        for (i = 1; i <= expected_repeats; ++i) {
          baseline_sorted[i] = baseline_max[i]
          adaptive_sorted[i] = adaptive_max[i]
        }
        for (i = 1; i <= expected_repeats; ++i) {
          for (j = i + 1; j <= expected_repeats; ++j) {
            if (baseline_sorted[j] < baseline_sorted[i]) {
              tmp = baseline_sorted[i]
              baseline_sorted[i] = baseline_sorted[j]
              baseline_sorted[j] = tmp
            }
            if (adaptive_sorted[j] < adaptive_sorted[i]) {
              tmp = adaptive_sorted[i]
              adaptive_sorted[i] = adaptive_sorted[j]
              adaptive_sorted[j] = tmp
            }
          }
        }
        middle = int(expected_repeats / 2)
        if (expected_repeats % 2) {
          baseline_median = baseline_sorted[middle + 1]
          adaptive_median = adaptive_sorted[middle + 1]
        } else {
          baseline_median = (baseline_sorted[middle] + baseline_sorted[middle + 1]) / 2
          adaptive_median = (adaptive_sorted[middle] + adaptive_sorted[middle + 1]) / 2
        }
        printf "%.3f %.3f %.3f", baseline_median, adaptive_median,
          (baseline_median / adaptive_median - 1.0) * 100.0
      }'); then
      printf 'ERROR: malformed A/B samples for tokens=%s alpha=%s\n' \
        "$tokens" "$alpha" >&2
      printf '%s\n' "$output" >&2
      exit 1
    fi
    read -r baseline_max adaptive_max speedup_pct <<< "$metrics"
    printf '%s,%s,%s,%s,%s\n' \
      "$tokens" "$alpha" "$baseline_max" "$adaptive_max" "$speedup_pct"
  done
done
