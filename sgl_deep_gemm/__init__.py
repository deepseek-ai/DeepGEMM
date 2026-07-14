from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import tvm_ffi

from .cuda_helpers import find_cuda_home, get_cuda_arch

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
def _build_module(pkg_dir: str, cuda_home: str) -> str:
    """Build the _C shared library using tvm_ffi.cpp.build()."""
    import tvm_ffi.cpp

    root_dir = os.path.dirname(pkg_dir)
    cxx_abi = int(torch.compiled_with_cxx11_abi())

    os.environ.setdefault('TVM_FFI_CUDA_ARCH_LIST', get_cuda_arch())

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

def _load_module() -> Module:
    """Load (or build then load) the compiled tvm-ffi module."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(pkg_dir, '_C.so')

    if not os.path.exists(lib_path):
        cuda_home = find_cuda_home()
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
set_pdl = _C.set_pdl
get_pdl = _C.get_pdl

# cuBLASLt Kernels
def cublaslt_gemm_nt(a, b, d, c=None):
    _C.cublaslt_gemm_nt(a, b, d, c)


def cublaslt_gemm_nn(a, b, d, c=None):
    _C.cublaslt_gemm_nn(a, b, d, c)


def cublaslt_gemm_tn(a, b, d, c=None):
    _C.cublaslt_gemm_tn(a, b, d, c)


def cublaslt_gemm_tt(a, b, d, c=None):
    _C.cublaslt_gemm_tt(a, b, d, c)

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

    def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout)

    def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.m_grouped_fp8_fp4_gemm_nn_contiguous(a_data, a_sf, b_data, b_sf, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding)

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

    def k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims='mn', use_psum_layout=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.k_grouped_fp8_gemm_tn_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout, c, recipe, compiled_dims, use_psum_layout)

    def k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, grouped_layout, c=None, recipe=(1, 1, 128), compiled_dims='mn', use_psum_layout=False):
        (a_data, a_sf), (b_data, b_sf) = _parse_tensor_or_tuple(a), _parse_tensor_or_tuple(b)
        _C.k_grouped_fp8_gemm_nt_contiguous(a_data, a_sf, b_data, b_sf, d, ks, grouped_layout, c, recipe, compiled_dims, use_psum_layout)

    def m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None):
        _C.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout)

    def m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True):
        _C.m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding)

    def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims='nk'):
        _C.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims)

    def k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c=None, compiled_dims='mn', use_psum_layout=False):
        _C.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c, compiled_dims, use_psum_layout)

    bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked

except AttributeError:
    pass

# Mega kernels
from . import mega
from .mega import (
    SymmBuffer,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
    bf16_mega_moe,
    mega_moe_pre_dispatch,
    mega_moe_pre_dispatch_sm90,
)


def _from_dlpack_if_needed(tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.utils.dlpack.from_dlpack(tensor)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.view(dtype)
    return tensor


class SM90SymmBuffer:
    def __init__(self, group,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_sm90_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        (x, x_sf, topk_idx, topk_weights,
         l1_acts, l1_acts_sf, l2_acts, l2_acts_sf) = slice_input_buffers(self.buffer)
        self.x = _from_dlpack_if_needed(x, torch.float8_e4m3fn)
        self.x_sf = _from_dlpack_if_needed(x_sf)
        self.topk_idx = _from_dlpack_if_needed(topk_idx)
        self.topk_weights = _from_dlpack_if_needed(topk_weights)
        self.l1_acts = _from_dlpack_if_needed(l1_acts, torch.float8_e4m3fn)
        self.l1_acts_sf = _from_dlpack_if_needed(l1_acts_sf)
        self.l2_acts = _from_dlpack_if_needed(l2_acts, torch.float8_e4m3fn)
        self.l2_acts_sf = _from_dlpack_if_needed(l2_acts_sf)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_sm90_mega_moe(group,
                                      num_experts: int,
                                      num_max_tokens_per_rank: int, num_topk: int,
                                      hidden: int, intermediate_hidden: int,
                                      use_fp8_dispatch: bool = True,
                                      activation: str = 'swiglu') -> SM90SymmBuffer:
    from .utils.math import align

    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())
    return SM90SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def get_symm_buffer_for_mega_moe(group,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu'):
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
        assert mma_type.split('x')[0] == 'fp8'
        return get_symm_buffer_for_sm90_mega_moe(
            group, num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            True, activation
        )
    return mega.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch=use_fp8_dispatch,
        mma_type=mma_type,
        activation=activation
    )


def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    l1_fp8, l1_sf = l1_weights

    def _interleave_one(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    return (_interleave_one(l1_fp8), l1_sf), l2_weights


def transform_weights_for_mega_moe_sm90_fp4(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Pack SM90 FP4 MegaMoE weights without changing the generic mega module."""
    def _interleave_one(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    def _pack_fp32_sf_to_ue8m0_kmajor(sf_fp32: torch.Tensor) -> torch.Tensor:
        assert sf_fp32.dtype == torch.float32, f"unexpected SF dtype {sf_fp32.dtype}"
        e, n, k_groups = sf_fp32.shape
        assert k_groups % 4 == 0, f"K/32={k_groups} must be a multiple of 4"
        bits = sf_fp32.view(torch.int32)
        ue8m0 = (bits.bitwise_right_shift(23).bitwise_and(0xff)).to(torch.uint8)
        ue8m0 = ue8m0.contiguous().view(e, n, k_groups // 4, 4)
        return ue8m0.view(torch.int32).reshape(e, n, k_groups // 4).contiguous()

    def _as_packed_fp4_storage(fp4: torch.Tensor) -> torch.Tensor:
        assert fp4.dtype in (torch.int8, torch.uint8), f"unexpected FP4 dtype {fp4.dtype}"
        return fp4.contiguous().view(torch.int8)

    l1_fp4, l1_sf_fp32 = l1_weights
    l2_fp4, l2_sf_fp32 = l2_weights
    l1_fp4 = _as_packed_fp4_storage(l1_fp4)
    l2_fp4 = _as_packed_fp4_storage(l2_fp4)
    l1_fp4 = _interleave_one(l1_fp4)
    l1_sf_fp32 = _interleave_one(l1_sf_fp32)
    return (
        (l1_fp4, _pack_fp32_sf_to_ue8m0_kmajor(l1_sf_fp32)),
        (l2_fp4, _pack_fp32_sf_to_ue8m0_kmajor(l2_sf_fp32)),
    )


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9):
        return mega.fp8_fp4_mega_moe(
            y,
            l1_weights,
            l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats,
            recipe,
            activation,
            activation_clamp,
            fast_math,
        )

    (l1_weights_data, l1_weights_sf) = l1_weights
    (l2_weights_data, l2_weights_sf) = l2_weights
    _C.fp8_fp4_mega_moe_sm90(
        y,
        l1_weights_data, l1_weights_sf,
        l2_weights_data, l2_weights_sf,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


def fp8_mega_moe(y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SM90SymmBuffer,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True):
    (l1_weights_data, l1_weights_sf) = l1_weights
    (l2_weights_data, l2_weights_sf) = l2_weights
    _C.fp8_mega_moe(
        y,
        l1_weights_data, l1_weights_sf,
        l2_weights_data, l2_weights_sf,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


def mega_moe_pre_dispatch_sm90(x: torch.Tensor,
                               topk_idx: torch.Tensor,
                               topk_weights: torch.Tensor,
                               buf_x: torch.Tensor,
                               buf_x_sf: torch.Tensor,
                               buf_topk_idx: torch.Tensor,
                               buf_topk_weights: torch.Tensor,
                               num_tokens: int,
                               group_size: int = 128,
                               routed_scaling_factor: float = 1.0) -> None:
    _C.mega_moe_pre_dispatch_sm90(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens, group_size, float(routed_scaling_factor),
    )

# Some utils
from . import testing
from . import utils
from .utils import *

# Initialize CPP modules
_C.init(
    os.path.dirname(os.path.abspath(__file__)),
    find_cuda_home()
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
