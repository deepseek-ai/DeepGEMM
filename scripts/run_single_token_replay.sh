#!/usr/bin/env bash
set -euo pipefail
task_root=${SM120_SINGLE_TOKEN_WORK:-/root/sm120-work/single-token-final1}
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
test ! -e "$task_root/real-replay.json"
test ! -e "$task_root/replay-exit-code.txt"
trap 'code=$?; echo "$code" > "$task_root/replay-exit-code.txt"' EXIT
python -m pip install --only-binary=:all: --target "$task_root/replay-deps" safetensors==0.6.2 > "$task_root/replay-dependency.log" 2>&1
python -c 'import safetensors; assert safetensors.__version__ == "0.6.2"; print(safetensors.__version__, safetensors.__file__)' > "$task_root/replay-dependency-version.txt"
python -u -m tests.replay_sm120_single_token --data /root/sm120-work/expert-pair11 --plan "$task_root/frozen-plan.json" --rounds 31 --output "$task_root/real-replay.json" > "$task_root/real-replay-retry.log" 2>&1
