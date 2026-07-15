#!/bin/bash
# Calibrate MegaMoE expert-wave sizes while keeping the upstream block_m tier.
# Each run writes all eight rank summaries; the CSV stream reports both rank 0
# and the slowest rank, which is the end-to-end EP latency that matters.
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
WAVE_SIZES=${WAVE_SIZES:-"4 8 12 16 24 32"}
RAW_LOG_DIR=${RAW_LOG_DIR:-"wave_size_calibration_raw_${SLURM_JOB_ID:-local}"}
mkdir -p "$RAW_LOG_DIR"

echo "experts=$EXPERTS topk=$TOPK hidden=$HIDDEN inter=$INTER EP=$NPROC"
echo "raw_logs=$RAW_LOG_DIR"
echo "tokens,alpha,experts_per_wave,rank0_us,slowest_rank_us"

for tokens in $TOKENS_LIST; do
  for alpha in $ALPHAS; do
    for wave_size in $WAVE_SIZES; do
      raw_log="$RAW_LOG_DIR/tokens_${tokens}_alpha_${alpha}_wave_${wave_size}.log"
      if ! output=$( \
        DG_MEGA_MOE_ADAPTIVE_WAVE=0 \
        DG_MEGA_MOE_FORCE_EXPERTS_PER_WAVE="$wave_size" \
        python3 tests/test_mega_moe.py \
          --num-processes "$NPROC" \
          --num-tokens "$tokens" \
          --num-max-tokens-per-rank "$tokens" \
          --num-experts "$EXPERTS" --num-topk "$TOPK" \
          --hidden "$HIDDEN" --intermediate-hidden "$INTER" \
          --skew-alpha "$alpha" --num-correctness-tests 0 \
          2>&1
      ); then
        printf '%s\n' "$output" > "$raw_log"
        printf '%s\n' "$output" >&2
        exit 1
      fi
      printf '%s\n' "$output" > "$raw_log"
      summary=$(printf '%s\n' "$output" | grep -E '^ > EP:[[:space:]]+0/' | head -1 || true)
      if [[ -z "$summary" ]]; then
        printf 'ERROR: no rank-0 summary for tokens=%s alpha=%s wave=%s\n' \
          "$tokens" "$alpha" "$wave_size" >&2
        printf '%s\n' "$output" >&2
        exit 1
      fi
      latency_us=$(printf '%s\n' "$summary" | sed -E 's/.*\|[[:space:]]*([0-9]+)[[:space:]]+us,.*/\1/')
      slowest_rank_us=$(printf '%s\n' "$output" \
        | grep -E '^ > EP:[[:space:]]+[0-9]+/' \
        | sed -E 's/.*\|[[:space:]]*([0-9]+)[[:space:]]+us,.*/\1/' \
        | sort -nr | head -1)
      if [[ -z "$slowest_rank_us" ]]; then
        printf 'ERROR: no per-rank latency for tokens=%s alpha=%s wave=%s\n' \
          "$tokens" "$alpha" "$wave_size" >&2
        exit 1
      fi
      printf '%s,%s,%s,%s,%s\n' \
        "$tokens" "$alpha" "$wave_size" "$latency_us" "$slowest_rank_us"
    done
  done
done
