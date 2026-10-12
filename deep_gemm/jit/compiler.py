import hashlib
import functools
import os
import re
import subprocess
import uuid
import torch
from typing import Tuple

from . import interleave_ffma
from .runtime import Runtime, RuntimeCache
from .template import typename_map

runtime_cache = RuntimeCache()

_jit_include_dir_default = f'{os.path.dirname(os.path.abspath(__file__))}/../include'
_jit_include_dir = f'{_jit_include_dir_default}/actlize_v1.0.0'

def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]

def set_jit_include_dir(new_jit_include_dir : str = None) -> None:
    global _jit_include_dir
    if new_jit_include_dir :
        _jit_include_dir = f'{os.path.dirname(os.path.abspath(__file__))}/../include' + "/" + new_jit_include_dir
    else:
        _jit_include_dir = f'{_jit_include_dir_default}/actlize_v1.0.0'
    # print("--------------------------- set_jit_include_dir = ", _jit_include_dir)

def get_jit_include_dir() -> str:
    # print("--------------------------- get_jit_include_dir = ", _jit_include_dir)
    return _jit_include_dir

@functools.lru_cache(maxsize=None)
def get_deep_gemm_version() -> str:
    # Update include directories
    include_dir = f'{os.path.dirname(os.path.abspath(__file__))}/../include/deep_gemm'
    assert os.path.exists(include_dir), f'Cannot find GEMM include directory {include_dir}'
    md5 = hashlib.md5()
    for root, _, files in os.walk(include_dir):
        for filename in sorted(filter(lambda x: x.endswith('.cuh'), files)):
            with open(os.path.join(root, filename), 'rb') as f:
                md5.update(f.read())

    # Update `interleave_ffma.py`
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/interleave_ffma.py', 'rb') as f:
        md5.update(f.read())
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_hgcc_compiler() -> Tuple[str, str]:
    """Find hgcc compiler path and version.
    Checks DG_JIT_HGCC_COMPILER env var first, then falls back to PPU_SDK default path.
    """
    paths = []
    if os.getenv('DG_JIT_HGCC_COMPILER'):
        paths.append(os.getenv('DG_JIT_HGCC_COMPILER'))
    # Default PPU SDK hgcc path
    paths.append('/usr/local/PPU_SDK/bin/hgcc')

    version_pattern = re.compile(r'version (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            try:
                output = os.popen(f'{path} --version 2>&1').read()
                match = version_pattern.search(output)
                version = match.group(1) if match else 'unknown'
            except Exception:
                version = 'unknown'
            return path, version
    raise RuntimeError('Cannot find any available hgcc compiler. '
                       'Set DG_JIT_HGCC_COMPILER or ensure /usr/local/PPU_SDK/bin/hgcc exists.')


@functools.lru_cache(maxsize=None)
def get_default_user_dir():
    if 'DG_CACHE_DIR' in os.environ:
        path = os.getenv('DG_CACHE_DIR')
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.expanduser('~') + '/.deep_gemm'


@functools.lru_cache(maxsize=None)
def get_tmp_dir():
    return f'{get_default_user_dir()}/tmp'


@functools.lru_cache(maxsize=None)
def get_cache_dir():
    return f'{get_default_user_dir()}/cache'


def make_tmp_dir():
    tmp_dir = get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def put(path, data, is_binary=False):
    # Write and do POSIX atomic replace
    tmp_file_path = f'{make_tmp_dir()}/file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}'
    with open(tmp_file_path, 'wb' if is_binary else 'w') as f:
        f.write(data)
    os.replace(tmp_file_path, path)

@functools.lru_cache(maxsize=None)
def is_ppu1v5_device():
    device_prop = torch.cuda.get_device_properties('cuda')
    if device_prop.major == 8 and device_prop.minor == 9:
        return True
    else:
        return False

def build(name: str, arg_defs: tuple, code: str) -> Runtime:
    """
    JIT compile a kernel into a shared library (.so).
    Flags are kept aligned with the C++ JIT compiler for consistency.
    """
    # --- Language standard & defines ---
    cpp_standard = int(os.getenv('DG_HGCC_OVERRIDE_CPP_STANDARD', 17))
    hgcc_flags = [
        f'-std=c++{cpp_standard}',
        '-shared',                  # produce a .so rather than a raw binary
        '-DUSE_HGGC', '-DUSE_CLANG', '-DUSE_ACWRAPPER',
    ]

    # --- Architecture ---
    if is_ppu1v5_device():
        hgcc_flags.append('-arch=ppu_15')
    else:
        hgcc_flags.append('-arch=ppu_10')

    # --- Optimization ---
    hgcc_flags.extend(['-ftemplate-depth=8192', '-O3', '-DNDEBUG'])


    # --- Host compiler flags (via -Xcompiler) ---
    hgcc_flags.extend(['-Xcompiler', '-fPIC', '-Xcompiler', '-Wno-deprecated-declarations', '-Xcompiler', '-Wno-abi'])

    # --- PPU LLVM backend tuning ---
    if is_ppu1v5_device():
        hgcc_flags.extend(['-Xllvm', '-ppu-patch-fence-ppu=false',
                           '-Xllvm', '-wno-loop-miss-transform'])

        # Per-kernel PPU tuning
        lower_name = name.lower()
        use_warp_interleaving = ('gemm_fp8' in lower_name) or \
                                ('mqa_logits' in lower_name and 'paged' not in lower_name)
        if not use_warp_interleaving:
            hgcc_flags.extend(['-Xllvm', '-ppu-simt-branch=false',
                               '-Xllvm', '-ppu-cg-to-kp1=true',
                               '-Xllvm', '-ppu-fix-uninit=true'])
        else:
            hgcc_flags.extend(['-Xllvm', '-ppu-cg-to-kp1=true',
                               '-Xllvm', '-ppu-fix-uninit=true',
                               '-Xllvm', '-ppu-blksync-nb-schedule-boundary=true',
                               '-Xllvm', '-ppu-simt-branch=false',
                               '-Xllvm', '-ppu-adjust-tsm-valu-war=13',
                               '-Xllvm', '-ppu-reassign-subregs=true',
                               '-Xllvm', '-ppu-pref-fma-reuse=true',
                               '-Xllvm', '-ppu-pref-mma-reuse=true'])
        if 'w4a16' in lower_name:
            hgcc_flags.extend(['-Xllvm', '-sort-copy-before-coalesce'])

    # --- Source language ---
    hgcc_flags.extend(['-x', 'hg'])

    # --- Include paths ---
    include_dirs = [get_jit_include_dir()]
    if _jit_include_dir_default not in include_dirs:
        include_dirs.append(_jit_include_dir_default)
    # Always include the deep_gemm headers (aligned with the C++ JIT compiler)
    deep_gemm_inc = f'{_jit_include_dir_default}/deep_gemm'
    if deep_gemm_inc not in include_dirs:
        include_dirs.append(deep_gemm_inc)
    # NOTE: do not add the SDK include dir explicitly -- the compiler finds its own
    # headers via built-in paths, and adding it causes GCC 13 <cmath> conflicts.

    # --- Build signature ---
    enable_sass_opt = 0
    signature = f'{name}$${get_deep_gemm_version()}$${code}$${get_hgcc_compiler()}$${hgcc_flags}$${enable_sass_opt}'
    name = f'kernel.{name}.{hash_to_hex(signature)}'
    path = f'{get_cache_dir()}/{name}'

    # Check runtime cache or file system hit
    global runtime_cache

    disable_cache = os.environ.get("DG_JIT_DISABLE_CACHE")
    if (runtime_cache[path] is not None) and (disable_cache is None or disable_cache == '0'):
        print(f'Using cached JIT runtime {path} {name} during build')

        if os.getenv('DG_JIT_DEBUG', None):
            print(f'Using cached JIT runtime {name} during build')
        return runtime_cache[path]

    # Write the code
    os.makedirs(path, exist_ok=True)
    args_path = f'{path}/kernel.args'
    src_path = f'{path}/kernel.cu'
    put(args_path, ', '.join([f"('{arg_def[0]}', {typename_map[arg_def[1]]})" for arg_def in arg_defs]))
    put(src_path, code)

    # Compile into a temporary SO file
    so_path = f'{path}/kernel.so'
    tmp_so_path = f'{make_tmp_dir()}/hgcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(so_path)}.so'

    # Compile
    command = [get_hgcc_compiler()[0],
               src_path, '-o', tmp_so_path,
               *hgcc_flags,
               *[f'-I{d}' for d in include_dirs]]

    if os.getenv('DG_JIT_DEBUG', None) or os.getenv('DG_JIT_PRINT_HGCC_COMMAND', False):
        print(f'Compiling JIT kernel {name} with command: {" ".join(command)}')
    return_code = subprocess.check_call(command)
    assert return_code == 0, f'Failed to compile {src_path}'

    # Interleave FFMA reuse (currently disabled)
    if enable_sass_opt:
        interleave_ffma.process(tmp_so_path)

    # Atomic replace SO file
    os.replace(tmp_so_path, so_path)

    # Put cache and return
    runtime_cache[path] = Runtime(path)
    return runtime_cache[path]
