#!/bin/bash
# SM120 full regression test suite
# Usage: bash tests/sm120/run_regression.sh [--perf]
# Without --perf: correctness only (fast)
# With --perf: correctness + performance

set -e
cd "$(dirname "$0")/../.."
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/home/scratch.yuasun/cache/deep_gemm}

PERF=0
if [ "$1" = "--perf" ]; then PERF=1; fi

PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local script="$2"
    TOTAL=$((TOTAL + 1))
    echo -n "[$TOTAL] $name ... "
    result=$(python "$script" 2>&1)
    if echo "$result" | grep -q "FAIL\|ERROR\|Traceback"; then
        echo "FAIL"
        echo "$result" | grep -E "FAIL|ERROR|Traceback" | head -3
        FAIL=$((FAIL + 1))
    else
        passed=$(echo "$result" | grep -oP '\d+/\d+ passed' | head -1)
        echo "OK ($passed)"
        PASS=$((PASS + 1))
        if [ $PERF -eq 1 ]; then
            echo "$result" | grep -E 'TFLOPS|Speedup|vs cuBLAS' | head -5
        fi
    fi
}

echo "============================================"
echo "SM120 Regression Test Suite"
echo "============================================"
echo

run_test "Dense FP8"           tests/sm120/test_dense_fp8.py
run_test "Dense FP4"           tests/sm120/test_dense_fp4.py
run_test "Dense FP8xFP4 Mixed" tests/sm120/test_dense_fp8_fp4_mixed.py
run_test "K-grouped FP8"       tests/sm120/test_k_grouped_fp8.py
run_test "M-grouped FP8/FP4"   tests/sm120/test_m_grouped_fp8.py
run_test "MQA Logits"          tests/sm120/test_mqa_logits.py

echo
echo "============================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL"
echo "============================================"

[ $FAIL -eq 0 ] || exit 1
