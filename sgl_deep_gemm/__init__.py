from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import torch
import tvm_ffi

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Set some default environment provided at setup
try:
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Build & load the tvm-ffi _C module
# ---------------------------------------------------------------------------
def _find_cuda_home() -> str:
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


def _build_module(pkg_dir: str, cuda_home: str) -> str:
    """Build the _C shared library using tvm_ffi.cpp.build()."""
    import tvm_ffi.cpp

    root_dir = os.path.dirname(pkg_dir)
    cxx_abi = int(torch.compiled_with_cxx11_abi())

    os.environ.setdefault('TVM_FFI_CUDA_ARCH_LIST', _get_cuda_arch())

    extra_cflags = [
        '-std=c++17', '-O3', '-fPIC',
        '-Wno-psabi', '-Wno-deprecated-declarations',
        f'-D_GLIBCXX_USE_CXX11_ABI={cxx_abi}',
    ]
    if int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')):
        extra_cflags.append('-DDG_JIT_USE_RUNTIME_API')

    # Torch include/lib paths
    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, 'include')
    torch_include_csrc = os.path.join(torch_include, 'torch', 'csrc', 'api', 'include')
    torch_lib = os.path.join(torch_dir, 'lib')

    import sysconfig
    python_include = sysconfig.get_path('include')

    extra_include_paths = [
        f'{cuda_home}/include',
        python_include,
        torch_include,
        torch_include_csrc,
        os.path.join(root_dir, 'deep_gemm', 'include'),
        os.path.join(root_dir, 'third-party', 'cutlass', 'include'),
        os.path.join(root_dir, 'third-party', 'fmt', 'include'),
    ]
    cccl_path = f'{cuda_home}/include/cccl'
    if os.path.exists(cccl_path):
        extra_include_paths.append(cccl_path)

    extra_ldflags = [
        f'-L{cuda_home}/lib64',
        f'-L{torch_lib}',
        '-lcudart',
        '-lnvrtc',
        '-lcublasLt',
        '-lcublas',
        '-ltorch',
        '-ltorch_cpu',
        '-lc10',
        '-lc10_cuda',
        '-ltorch_cuda',
    ]

    build_dir = os.path.join(pkg_dir, '_C_build')
    os.makedirs(build_dir, exist_ok=True)

    lib_path = tvm_ffi.cpp.build(
        name='_C',
        cpp_files=[os.path.join(root_dir, 'csrc', 'tvm_ffi_api.cpp')],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_dir,
    )
    # Copy the .so into the package directory for easy loading
    target = os.path.join(pkg_dir, '_C.so')
    shutil.copy2(lib_path, target)
    return target


def _get_cuda_arch() -> str:
    try:
        status = subprocess.run(
            args=['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, check=True,
        )
        return status.stdout.decode('utf-8').strip().split('\n')[0]
    except Exception:
        return '9.0'


def _load_module() -> Module:
    """Load (or build then load) the compiled tvm-ffi module."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(pkg_dir, '_C.so')

    if not os.path.exists(lib_path):
        cuda_home = _find_cuda_home()
        print(f'[DeepGEMM] Building _C module with tvm-ffi (CUDA_HOME={cuda_home})...')
        lib_path = _build_module(pkg_dir, cuda_home)
        print(f'[DeepGEMM] Built _C module: {lib_path}')

    return tvm_ffi.load_module(lib_path)

_C: Module = _load_module()

# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------
set_num_sms = _C.set_num_sms
get_num_sms = _C.get_num_sms
# set_compile_mode / get_compile_mode are not exported on this branch.
set_tc_util = _C.set_tc_util
get_tc_util = _C.get_tc_util

# cuBLASLt Kernels
cublaslt_gemm_nt = _C.cublaslt_gemm_nt
cublaslt_gemm_nn = _C.cublaslt_gemm_nn
cublaslt_gemm_tn = _C.cublaslt_gemm_tn
cublaslt_gemm_tt = _C.cublaslt_gemm_tt

def _parse_tensor_or_tuple(input):
    if type(input) is tuple or type(input) is list:
        return input[0], input[1]
    elif isinstance(input, torch.Tensor):
        scale = torch.Tensor([1.0], dtype=torch.float32, device=input.device)
        return input, scale

    assert False, "Expected Tensor, (Tensor, Tensor) tuple, or [Tensor, Tensor] list"

# ---------------------------------------------------------------------------
# GEMM / Attention / Einsum wrappers (handle optional params in Python)
# ---------------------------------------------------------------------------
try:
    def fp8_fp4_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_nt(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast)

    def fp8_fp4_gemm_nn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_nn(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast)

    def fp8_fp4_gemm_tn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_tn(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast)

    def fp8_fp4_gemm_tt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_fp4_gemm_tt(a_data, a_sf, b_data, b_sf, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast)

    fp8_gemm_nt = fp8_fp4_gemm_nt
    fp8_gemm_nn = fp8_fp4_gemm_nn
    fp8_gemm_tn = fp8_fp4_gemm_tn
    fp8_gemm_tt = fp8_fp4_gemm_tt

    def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False, expected_m_for_psum_layout=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, expected_m_for_psum_layout)

    def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nn_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout)

    m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
    m_grouped_fp8_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_nn_contiguous

    def bf16_gemm_nt(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_nt(a, b, d, c, compiled_dims)

    def bf16_gemm_nn(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_nn(a, b, d, c, compiled_dims)

    def bf16_gemm_tn(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_tn(a, b, d, c, compiled_dims)

    def bf16_gemm_tt(a, b, d, c=None, compiled_dims=''):
        _C.bf16_gemm_tt(a, b, d, c, compiled_dims)

    def einsum(expr, a, b, d, c=None, use_cublaslt=False):
        _C.einsum(expr, a, b, d, c, use_cublaslt)

    def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128)):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_einsum(expr, a_data, a_sf, b_data, b_sf, d, c, recipe)

    def fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, recipe=None, compiled_dims='nk', disable_ue8m0_cast=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.fp8_gemm_nt_skip_head_mid(a_data, a_sf, b_data, b_sf, d, head_splits, recipe, compiled_dims, disable_ue8m0_cast)

    def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, indices=None):
        return _C.fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, indices)

    def fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=0):
        (kv_data, kv_sf) = _parse_tensor_or_tuple(kv)
        return _C.fp8_mqa_logits(q, kv_data, kv_sf, weights, ks, ke, clean_logits, max_seqlen_k)

    def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits=False, logits_dtype=torch.float, indices=None):
        logits_dtype_str = str(logits_dtype).split('.')[-1]
        (q, q_sf) = _parse_tensor_or_tuple(q)
        return _C.fp8_fp4_paged_mqa_logits(q, q_sf, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, logits_dtype_str, indices)

    def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=False, max_seqlen_k=0, logits_dtype=torch.float):
        (q, q_sf), (kv_data, kv_sf) = _parse_tensor_or_tuple(q), _parse_tensor_or_tuple(kv)
        logits_dtype_str = str(logits_dtype).split('.')[-1]
        return _C.fp8_fp4_mqa_logits(q, q_sf, kv_data, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k, logits_dtype_str)

    def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
        return _C.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices)

    def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
        _C.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits)

    def transform_sf_into_required_layout(sf, mn, k, recipe, num_groups=None, is_sfa=None, disable_ue8m0_cast=False):
        (recipe_a, recipe_b, recipe_c) = recipe if len(recipe) == 3 else (recipe[0], recipe[1], None)
        return _C.transform_sf_into_required_layout(sf, mn, k, recipe_a, recipe_b, recipe_c, num_groups, is_sfa, disable_ue8m0_cast)

    get_mk_alignment_for_contiguous_layout = _C.get_mk_alignment_for_contiguous_layout

    def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False):
        (a, a_sf), (b, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        return _C.m_grouped_fp8_fp4_gemm_nt_masked(a, a_sf, b, b_sf, d, masked_m, expected_m, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast)

    fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
    bf16_m_grouped_gemm_nt_masked = None

except AttributeError:
    pass

# Mega kernels
from .mega import (
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
)

# Some utils
from . import testing
from . import utils
from .utils import *

# Initialize CPP modules
_C.init(
    os.path.dirname(os.path.abspath(__file__)),
    _find_cuda_home()
)

def _read_version() -> str:
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')
    try:
        with open(version_file, 'r') as f:
            return f.read().strip()
    except OSError:
        return '0.0.0.dev0'

__version__ = _read_version()

# Allow `import deep_gemm.<name>` to resolve top-level public symbols, mirroring
# `from deep_gemm import <name>`. Without this, Python's import machinery only
# resolves submodules — top-level callables defined here are otherwise
# inaccessible via the dotted-import form.
import sys as _sys
import types as _types
for _name, _val in list(globals().items()):
    if _name.startswith('_') or _val is None or isinstance(_val, _types.ModuleType):
        continue
    _sys.modules.setdefault(f'{__name__}.{_name}', _val)
del _sys, _types, _name, _val
