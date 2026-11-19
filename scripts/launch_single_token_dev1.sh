#!/usr/bin/env bash
set -euo pipefail
task_root=/root/sm120-work/single-token-integration1
test ! -e "$task_root"
mkdir -p "$task_root"
tar -xzf /root/sm120-work/uploads/single-token-integration-dev1.tar.gz -C "$task_root"
src="$task_root/source"
donor=/root/sm120-work/small-fusion-pr1/source
for dep in cutlass deep_jit; do
    cp -a "$donor/third-party/$dep" "$src/third-party/$dep"
done
for dep in cute cutlass; do
    ln -s "../../third-party/cutlass/include/$dep" "$src/deep_gemm/include/$dep"
done
export PATH=/usr/local/cuda/bin:/usr/local/miniconda3/envs/py312/bin:$PATH
export LD_PRELOAD=/root/sm120-toolchain/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/root/sm120-toolchain/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CC=/root/sm120-toolchain/bin/x86_64-conda-linux-gnu-gcc
export CXX=/root/sm120-toolchain/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$src:$src/tests"
export PYTHONDONTWRITEBYTECODE=1
export DG_USE_LOCAL_VERSION=0
export DG_JIT_CACHE_DIR="$task_root/jit-cache"
export MAX_JOBS=2
export OMP_NUM_THREADS=4
trap 'code=$?; echo "$code" > "$task_root/exit-code.txt"' EXIT
cd "$src"
python -c 'import json,hashlib; from pathlib import Path; m=json.loads(Path("source-manifest.json").read_text()); assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in m.items()); print("SOURCE VERIFIED",len(m),flush=True)'
python setup.py build_ext --inplace > "$task_root/build.log" 2>&1
python -u -m tests.bench_sm120_single_token --models qwen30 --seeds 25011 --rounds 5 --output "$task_root/pilot.json" > "$task_root/pilot.log" 2>&1
