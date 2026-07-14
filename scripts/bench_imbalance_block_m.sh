#!/bin/bash
# On-GPU A/B benchmark for imbalance-aware block_m selection (SM100 FP8xFP4 MegaMoE).
#
# For each (tokens, skew) point we run the SAME workload twice:
#   BASELINE : DG_MEGA_MOE_IMBALANCE_AWARE_BLOCK_M unset  (mean-based block_m)
#   ADAPTIVE : DG_MEGA_MOE_IMBALANCE_AWARE_BLOCK_M=1       (wall-cost block_m)
# and print the mega_moe kernel time from each so the speedup can be read off.
#
# Correctness is covered by test_mega_moe.py's built-in numerical check; this
# script is purely for the perf A/B. Requires 8 GPUs (EP=8) by default.
#
# Usage:  bash scripts/bench_imbalance_block_m.sh 2>&1 | tee imbalance_ab.log
set -u
cd "$(dirname "$0")/.."

EXPERTS=${EXPERTS:-256}
TOPK=${TOPK:-8}
HIDDEN=${HIDDEN:-7168}
INTER=${INTER:-2048}
NPROC=${NPROC:-8}

run_one () {
  local tokens=$1 alpha=$2 flag=$3
  DG_MEGA_MOE_IMBALANCE_AWARE_BLOCK_M=$flag \
  python3 tests/test_mega_moe.py \
    --num-processes "$NPROC" \
    --num-tokens "$tokens" \
    --num-max-tokens-per-rank "$tokens" \
    --num-experts "$EXPERTS" --num-topk "$TOPK" \
    --hidden "$HIDDEN" --intermediate-hidden "$INTER" \
    --skew-alpha "$alpha" \
    2>&1 | grep -E "EP: .*0/|us," | head -1
}

echo "experts=$EXPERTS topk=$TOPK hidden=$HIDDEN inter=$INTER EP=$NPROC"
echo "tokens  alpha  |  BASELINE (flag=0)          |  ADAPTIVE (flag=1)"
echo "-------------------------------------------------------------------------"
for tokens in 128 256 512 1024; do
  for alpha in 0.0 1.0 1.5; do
    base=$(run_one "$tokens" "$alpha" 0)
    adap=$(run_one "$tokens" "$alpha" 1)
    printf "%6s  %5s  |  %s  |  %s\n" "$tokens" "$alpha" "$base" "$adap"
  done
done
echo "-------------------------------------------------------------------------"
echo "Read the 'us,' field: adaptive should be <= baseline at alpha>0 (decode),"
echo "and equal at alpha=0 (strict fallback -> identical block_m)."
