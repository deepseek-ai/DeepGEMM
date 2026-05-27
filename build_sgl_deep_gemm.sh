#!/usr/bin/env bash
#
# Build a wheel for the `sgl-deep-gemm` distribution.
#
# Distribution name: sgl-deep-gemm. Top-level import name: `deep_gemm`
# (so existing call sites like `import deep_gemm` in sglang keep working).
#
# Build flow:
#   1. Initialises submodules (cutlass, fmt) — same prerequisite as `bash build.sh`.
#   2. Stages the package layout under build/deep_gemm/ with the Python
#      sub-modules pulled from the source deep_gemm/ tree (utils, testing,
#      legacy, mega).
#   3. Reads the version string from sgl_deep_gemm/VERSION.
#   4. Pre-compiles the tvm-ffi `_C.so` extension and bundles it into the wheel.
#   5. Invokes `python -m build` to produce dist/*.whl.

set -euo pipefail

PYTHON_EXE=$(which python3 || which python)
ROOT_DIR=$(realpath "$(dirname "$0")")
BUILD_DIR="${ROOT_DIR}/build"
PKG_DIR="${BUILD_DIR}/deep_gemm"
DIST_DIR="${ROOT_DIR}/dist"

cd "$ROOT_DIR"

if [[ ! -f "setup.py" || ! -d "sgl_deep_gemm" || ! -d "deep_gemm" || ! -d "csrc" ]]; then
    echo "Error: Run from the DeepGEMM project root." >&2
    exit 1
fi

echo "--- Initialising submodules ---"
git submodule update --init --recursive

echo "--- Linking CUTLASS headers into deep_gemm/include ---"
ln -sfn "${ROOT_DIR}/third-party/cutlass/include/cutlass" "${ROOT_DIR}/deep_gemm/include/cutlass"
ln -sfn "${ROOT_DIR}/third-party/cutlass/include/cute" "${ROOT_DIR}/deep_gemm/include/cute"

echo "--- Preparing build directory ---"
rm -rf "$BUILD_DIR"
mkdir -p "$PKG_DIR"

cp sgl_deep_gemm/LICENSE sgl_deep_gemm/README.md sgl_deep_gemm/pyproject.toml "$BUILD_DIR/"
cp sgl_deep_gemm/__init__.py "$PKG_DIR/"

# `__init__.py` imports `.utils`, `.testing`, `.legacy`, `.mega` — pulled from
# the existing deep_gemm/ tree.
for sub in utils testing legacy mega; do
    cp -r "deep_gemm/${sub}" "$PKG_DIR/"
done

# Headers required by the runtime JIT (same set the deep_gemm wheel ships).
mkdir -p "$PKG_DIR/include"
cp -r "${ROOT_DIR}/deep_gemm/include/deep_gemm" "$PKG_DIR/include/deep_gemm"
cp -r "${ROOT_DIR}/third-party/cutlass/include/cute" "$PKG_DIR/include/cute"
cp -r "${ROOT_DIR}/third-party/cutlass/include/cutlass" "$PKG_DIR/include/cutlass"

echo "--- Reading version from sgl_deep_gemm/VERSION ---"
if [[ ! -f "sgl_deep_gemm/VERSION" ]]; then
    echo "Error: sgl_deep_gemm/VERSION is missing — create it with the desired version (e.g. 0.0.1)." >&2
    exit 1
fi
# Strip surrounding whitespace; the file is the single source of truth.
tr -d '[:space:]' < sgl_deep_gemm/VERSION > "$PKG_DIR/VERSION"
echo "Version: $(cat "$PKG_DIR/VERSION")"

echo "--- Compiling _C.so ---"
ROOT_DIR="$ROOT_DIR" PKG_DIR="$PKG_DIR" "$PYTHON_EXE" -u - <<'PY'
import os, shutil, subprocess, sys, sysconfig
root_dir = os.environ['ROOT_DIR']
pkg_dir = os.environ['PKG_DIR']
sys.path.insert(0, root_dir)

# tvm_ffi.cpp.build runs ninja with capture_output=True, hiding compile logs
# until a failure. Patch subprocess.run so the ninja invocation streams to the
# terminal — leaves other internal calls (nvidia-smi, nvcc --version) alone.
_orig_run = subprocess.run
def _streamed_run(*args, **kwargs):
    cmd = kwargs.get('args') if 'args' in kwargs else (args[0] if args else None)
    is_ninja = isinstance(cmd, (list, tuple)) and cmd and 'ninja' in str(cmd[0])
    if is_ninja:
        kwargs.pop('capture_output', None)
        kwargs['stdout'] = None
        kwargs['stderr'] = None
    return _orig_run(*args, **kwargs)
subprocess.run = _streamed_run

import torch
import tvm_ffi.cpp
from setup import _find_cuda_home, _get_cuda_arch

cuda_home = _find_cuda_home()
os.environ.setdefault('TVM_FFI_CUDA_ARCH_LIST', _get_cuda_arch())

cxx_abi = int(torch.compiled_with_cxx11_abi())
extra_cflags = [
    '-std=c++17', '-O3', '-fPIC',
    '-Wno-psabi', '-Wno-deprecated-declarations',
    f'-D_GLIBCXX_USE_CXX11_ABI={cxx_abi}',
]
if int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')):
    extra_cflags.append('-DDG_JIT_USE_RUNTIME_API')

torch_dir = os.path.dirname(torch.__file__)
extra_include_paths = [
    f'{cuda_home}/include',
    sysconfig.get_path('include'),
    os.path.join(torch_dir, 'include'),
    os.path.join(torch_dir, 'include', 'torch', 'csrc', 'api', 'include'),
    os.path.join(root_dir, 'deep_gemm', 'include'),
    os.path.join(root_dir, 'third-party', 'cutlass', 'include'),
    os.path.join(root_dir, 'third-party', 'fmt', 'include'),
]
cccl = f'{cuda_home}/include/cccl'
if os.path.exists(cccl):
    extra_include_paths.append(cccl)

extra_ldflags = [
    f'-L{cuda_home}/lib64',
    f'-L{os.path.join(torch_dir, "lib")}',
    '-lcudart', '-lnvrtc', '-lcublasLt', '-lcublas',
    '-ltorch', '-ltorch_cpu', '-lc10', '-lc10_cuda', '-ltorch_cuda',
]

build_subdir = os.path.join(pkg_dir, '_C_build')
os.makedirs(build_subdir, exist_ok=True)
lib_path = tvm_ffi.cpp.build(
    name='_C',
    cpp_files=[os.path.join(root_dir, 'csrc', 'tvm_ffi_api.cpp')],
    extra_cflags=extra_cflags,
    extra_ldflags=extra_ldflags,
    extra_include_paths=extra_include_paths,
    build_directory=build_subdir,
)
target = os.path.join(pkg_dir, '_C.so')
if os.path.exists(target):
    os.remove(target)
shutil.copy2(lib_path, target)
shutil.rmtree(build_subdir, ignore_errors=True)
print(f"Built {target}")
PY

echo "--- Installing build frontend ---"
"$PYTHON_EXE" -m pip install --quiet --upgrade build

echo "--- Building wheel ---"
mkdir -p "$DIST_DIR"
"$PYTHON_EXE" -m build --wheel "$BUILD_DIR" --outdir "$DIST_DIR"

echo "--- Done ---"
ls -lh "$DIST_DIR"/sgl_deep_gemm-*.whl 2>/dev/null || ls -lh "$DIST_DIR"/sgl-deep-gemm-*.whl 2>/dev/null || ls -lh "$DIST_DIR"/sgl_deep_gemm*.whl
