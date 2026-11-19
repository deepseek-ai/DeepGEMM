#!/usr/bin/env bash
set -euo pipefail
task_root=/root/sm120-work/single-token-integration1
cd "$task_root/source"
export PATH=/usr/local/cuda/bin:/usr/local/miniconda3/envs/py312/bin:$PATH
export LD_PRELOAD=/root/sm120-toolchain/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/root/sm120-toolchain/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CPATH=/root/sm120-toolchain/include:${CPATH:-}
export LIBRARY_PATH=/root/sm120-toolchain/lib:${LIBRARY_PATH:-}
export CC=/root/sm120-toolchain/bin/x86_64-conda-linux-gnu-gcc
export CXX=/root/sm120-toolchain/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/tests"
export PYTHONDONTWRITEBYTECODE=1
export DG_USE_LOCAL_VERSION=0
export DG_JIT_CACHE_DIR="$task_root/jit-cache"
export MAX_JOBS=2
export OMP_NUM_THREADS=4
trap 'code=$?; echo "$code" > "$task_root/exit-code-retry.txt"' EXIT
python setup.py build_ext --inplace > "$task_root/build-retry.log" 2>&1
python -u -m tests.bench_sm120_single_token --models qwen30 --seeds 25011 --rounds 5 --output "$task_root/pilot.json" > "$task_root/pilot.log" 2>&1
