#!/usr/bin/env bash
set -euo pipefail
task_root=/root/sm120-work/single-token-final1
cd "$task_root/source"
export PATH=/usr/local/cuda/bin:/usr/local/miniconda3/envs/py312/bin:$PATH
export LD_PRELOAD=/root/sm120-toolchain/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/root/sm120-toolchain/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CPATH=/root/sm120-toolchain/include:${CPATH:-}
export LIBRARY_PATH=/root/sm120-toolchain/lib:${LIBRARY_PATH:-}
export CC=/root/sm120-toolchain/bin/x86_64-conda-linux-gnu-gcc
export CXX=/root/sm120-toolchain/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$task_root/replay-deps:$PWD:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export DG_USE_LOCAL_VERSION=0
export DG_JIT_CACHE_DIR="$task_root/jit-cache"
export MAX_JOBS=2
export OMP_NUM_THREADS=4
test ! -e "$task_root/validation"
test ! -e "$task_root/exit-code.txt"
trap 'code=$?; echo "$code" > "$task_root/exit-code.txt"' EXIT
python -c 'import pytest, safetensors; from safetensors import safe_open' > "$task_root/dependency-preflight.log" 2>&1
python -c 'import json,hashlib; from pathlib import Path; m=json.loads(Path("source-manifest.json").read_text()); assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in m.items()); print("SOURCE VERIFIED",len(m),flush=True)'
python -u -m tests.run_sm120_single_token_validation --output "$task_root/validation" > "$task_root/validation.log" 2>&1
python -u -m tests.bench_sm120_single_token_entry --seeds 25301 25309 --rounds 31 --output "$task_root/final-entry.json" > "$task_root/final-entry.log" 2>&1
python -u -m tests.replay_sm120_single_token --data /root/sm120-work/expert-pair11 --plan "$task_root/frozen-plan.json" --rounds 31 --output "$task_root/real-replay.json" > "$task_root/real-replay.log" 2>&1
