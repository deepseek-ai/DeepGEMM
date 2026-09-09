import torch
from pathlib import Path


def _load_extension():
    so_files = list(Path(__file__).parent.glob('_C_extension*.so'))
    assert len(so_files) == 1, (
        f'Expected one _C_extension*.so file, found {len(so_files)}: {so_files}'
    )
    torch.ops.load_library(str(so_files[0]))


_load_extension()
_torch_ops = torch.ops.deep_gemm


def _bind_guarded_ops(*names):
    """Bind ops when all are registered (matches one C++ #if guard group)."""
    present = [name for name in names if hasattr(_torch_ops, name)]
    if not present:
        return
    assert len(present) == len(names), (
        f'Guard group mismatch: {sorted(set(names) - set(present))} missing while '
        f'{present} are registered — the C++ #if guards for these ops have diverged.'
    )
    globals().update({name: getattr(_torch_ops, name) for name in names})


init = _torch_ops.init
set_num_sms = _torch_ops.set_num_sms
get_num_sms = _torch_ops.get_num_sms
set_tc_util = _torch_ops.set_tc_util
get_tc_util = _torch_ops.get_tc_util
set_pdl = _torch_ops.set_pdl
get_pdl = _torch_ops.get_pdl
set_ignore_compile_dims = _torch_ops.set_ignore_compile_dims
get_mk_alignment_for_contiguous_layout = _torch_ops.get_mk_alignment_for_contiguous_layout
get_theoretical_mk_alignment_for_contiguous_layout = _torch_ops.get_theoretical_mk_alignment_for_contiguous_layout
cublaslt_gemm_nt = _torch_ops.cublaslt_gemm_nt
cublaslt_gemm_nn = _torch_ops.cublaslt_gemm_nn
cublaslt_gemm_tn = _torch_ops.cublaslt_gemm_tn
cublaslt_gemm_tt = _torch_ops.cublaslt_gemm_tt


def set_block_size_multiple_of(value):
    if isinstance(value, int):
        return _torch_ops.set_block_size_multiple_of([value])
    return _torch_ops.set_block_size_multiple_of(list(value))


def set_mk_alignment_for_contiguous_layout(value):
    return _torch_ops.set_mk_alignment_for_contiguous_layout(value)


def _unpack_ab_pair(a, b):
    return a[0], a[1], b[0], b[1]


def _unpack_q(q):
    if isinstance(q, tuple):
        q_fp = q[0]
        q_sf = q[1] if len(q) > 1 else None
    else:
        q_fp, q_sf = q, None
    return q_fp, q_sf


def _unpack_kv(kv):
    return kv[0], kv[1]


def _as_int_list(value):
    """Reject a bare scalar instead of letting int[N] silently broadcast it into a list."""
    return None if value is None else list(value)


def _register_deep_gemm_kernels():
    """Export DeepGEMM kernels only when C++ ops are registered."""
    def fp8_fp4_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='nk', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_nt(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast,
        )

    def fp8_fp4_gemm_nn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='nk', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_nn(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast,
        )

    def fp8_fp4_gemm_tn(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='mn', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_tn(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast,
        )

    def fp8_fp4_gemm_tt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                        compiled_dims='mn', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_fp4_gemm_tt(
            a_tensor, sfa, b_tensor, sfb, d, c, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast,
        )

    def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None,
                                             compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False,
                                             ensure_zero_padding=True, expected_m_for_psum_layout=None):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, grouped_layout, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding,
            expected_m_for_psum_layout,
        )

    def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None,
                                             compiled_dims='nk', disable_ue8m0_cast=False, use_psum_layout=False,
                                             ensure_zero_padding=True):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.m_grouped_fp8_fp4_gemm_nn_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, grouped_layout, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding,
        )

    def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None,
                                         compiled_dims='nk', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.m_grouped_fp8_fp4_gemm_nt_masked(
            a_tensor, sfa, b_tensor, sfb, d, masked_m, expected_m, _as_int_list(recipe), _as_int_list(recipe_a), _as_int_list(recipe_b),
            compiled_dims, disable_ue8m0_cast,
        )

    def k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 128),
                                         compiled_dims='mn', use_psum_layout=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.k_grouped_fp8_gemm_tn_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, ks_cpu, grouped_layout, c, list(recipe),
            compiled_dims, use_psum_layout,
        )

    def k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1, 1, 128),
                                         compiled_dims='mn', use_psum_layout=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.k_grouped_fp8_gemm_nt_contiguous(
            a_tensor, sfa, b_tensor, sfb, d, ks_cpu, grouped_layout, c, list(recipe),
            compiled_dims, use_psum_layout,
        )

    def fp8_gemm_nt_skip_head_mid(a, b, d, head_splits, recipe=None, compiled_dims='nk', disable_ue8m0_cast=False):
        a_tensor, sfa, b_tensor, sfb = _unpack_ab_pair(a, b)
        return _torch_ops.fp8_gemm_nt_skip_head_mid(
            a_tensor, sfa, b_tensor, sfb, d, list(head_splits), _as_int_list(recipe), compiled_dims, disable_ue8m0_cast,
        )

    def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128)):
        return _torch_ops.fp8_einsum(expr, a[0], a[1], b[0], b[1], d, c, list(recipe))

    def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=True,
                           max_seqlen_k=0, logits_dtype=torch.float32):
        q_fp, q_sf = _unpack_q(q)
        kv_fp, kv_sf = _unpack_kv(kv)
        return _torch_ops.fp8_fp4_mqa_logits(
            q_fp, q_sf, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end,
            clean_logits, max_seqlen_k, logits_dtype,
        )

    def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
                                 clean_logits=False, logits_dtype=torch.float32, indices=None):
        q_fp, q_sf = _unpack_q(q)
        return _torch_ops.fp8_fp4_paged_mqa_logits(
            q_fp, q_sf, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
            clean_logits, logits_dtype, indices,
        )

    def fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=True, max_seqlen_k=0):
        kv_fp, kv_sf = _unpack_kv(kv)
        return _torch_ops.fp8_mqa_logits(q, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits, max_seqlen_k)

    def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
                             clean_logits=False, indices=None):
        return _torch_ops.fp8_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, clean_logits, indices,
        )

    globals().update({
        'fp8_fp4_gemm_nt': fp8_fp4_gemm_nt,
        'fp8_fp4_gemm_nn': fp8_fp4_gemm_nn,
        'fp8_fp4_gemm_tn': fp8_fp4_gemm_tn,
        'fp8_fp4_gemm_tt': fp8_fp4_gemm_tt,
        'fp8_gemm_nt': fp8_fp4_gemm_nt,
        'fp8_gemm_nn': fp8_fp4_gemm_nn,
        'fp8_gemm_tn': fp8_fp4_gemm_tn,
        'fp8_gemm_tt': fp8_fp4_gemm_tt,
        'm_grouped_fp8_fp4_gemm_nt_contiguous': m_grouped_fp8_fp4_gemm_nt_contiguous,
        'm_grouped_fp8_fp4_gemm_nn_contiguous': m_grouped_fp8_fp4_gemm_nn_contiguous,
        'm_grouped_fp8_fp4_gemm_nt_masked': m_grouped_fp8_fp4_gemm_nt_masked,
        'm_grouped_fp8_gemm_nt_contiguous': m_grouped_fp8_fp4_gemm_nt_contiguous,
        'm_grouped_fp8_gemm_nn_contiguous': m_grouped_fp8_fp4_gemm_nn_contiguous,
        'm_grouped_fp8_gemm_nt_masked': m_grouped_fp8_fp4_gemm_nt_masked,
        'k_grouped_fp8_gemm_tn_contiguous': k_grouped_fp8_gemm_tn_contiguous,
        'k_grouped_fp8_gemm_nt_contiguous': k_grouped_fp8_gemm_nt_contiguous,
        'fp8_gemm_nt_skip_head_mid': fp8_gemm_nt_skip_head_mid,
        'fp8_einsum': fp8_einsum,
        'fp8_fp4_mqa_logits': fp8_fp4_mqa_logits,
        'fp8_fp4_paged_mqa_logits': fp8_fp4_paged_mqa_logits,
        'fp8_mqa_logits': fp8_mqa_logits,
        'fp8_paged_mqa_logits': fp8_paged_mqa_logits,
    })

    # DG_TENSORMAP_COMPATIBLE — gemm.hpp (BF16 impl conditional)
    _bind_guarded_ops(
        'bf16_gemm_nt',
        'bf16_gemm_nn',
        'bf16_gemm_tn',
        'bf16_gemm_tt',
        'm_grouped_bf16_gemm_nt_contiguous',
        'm_grouped_bf16_gemm_nn_contiguous',
        'm_grouped_bf16_gemm_nt_masked',
        'k_grouped_bf16_gemm_tn_contiguous',
    )

    # DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE — grouped only because these three
    # happen to share the same guard today; if any one's guard changes, split it out.
    _bind_guarded_ops(
        'einsum',                         # einsum.hpp
        'tf32_hc_prenorm_gemm',           # hyperconnection.hpp
        'get_paged_mqa_logits_metadata',  # attention.hpp
    )

    # DG_TENSORMAP_COMPATIBLE — layout.hpp (schema and impl conditional)
    _bind_guarded_ops(
        'transform_sf_into_required_layout',
        'get_tma_aligned_size',
        'get_mn_major_tma_aligned_tensor',
        'get_mn_major_tma_aligned_packed_ue8m0_tensor',
        'get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor',
    )


_register_deep_gemm_kernels()


def get_symm_buffer_size_for_mega_moe(*args, **kwargs):
    return _torch_ops.get_symm_buffer_size_for_mega_moe(*args, **kwargs)


def _slice_symm_buffer_for_mega_moe(buffer, *args, **kwargs):
    return _torch_ops._slice_symm_buffer_for_mega_moe(buffer, *args, **kwargs)


def get_symm_buffer_size_for_sm90_mega_moe(*args, **kwargs):
    return _torch_ops.get_symm_buffer_size_for_sm90_mega_moe(*args, **kwargs)


def _slice_symm_buffer_for_sm90_mega_moe(buffer, *args, **kwargs):
    return _torch_ops._slice_symm_buffer_for_sm90_mega_moe(buffer, *args, **kwargs)


# DG_TENSORMAP_COMPATIBLE — mega.hpp (C++ impl conditional; matches legacy pybind export guard)
_bind_guarded_ops(
    'get_token_alignment_for_mega_moe',
    'get_block_m_for_mega_moe',
)

# DG_TENSORMAP_COMPATIBLE — sm90_mega.hpp
_bind_guarded_ops('get_token_alignment_for_sm90_mega_moe')


def fp8_fp4_mega_moe(y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights,
                     cumulative_local_expert_recv_stats, sym_buffer,
                     sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk, recipe,
                     activation, activation_clamp, fast_math,
                     situ_beta=None, situ_linear_beta=None):
    shared_l1_w = shared_l1_sf = shared_l2_w = shared_l2_sf = None
    if shared_l1_weights is not None:
        shared_l1_w, shared_l1_sf = shared_l1_weights
        shared_l2_w, shared_l2_sf = shared_l2_weights
    return _torch_ops.fp8_fp4_mega_moe(
        y, l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1],
        shared_l1_w, shared_l1_sf, shared_l2_w, shared_l2_sf,
        cumulative_local_expert_recv_stats, sym_buffer, list(sym_buffer_ptrs), rank_idx,
        num_max_tokens_per_rank, num_experts, num_topk, list(recipe), activation,
        activation_clamp, fast_math, situ_beta, situ_linear_beta,
    )


def fp4_fp4_mega_moe(y, l1_weights, l2_weights,
                     shared_l1_weights, shared_l2_weights, x_bf16,
                     cumulative_local_expert_recv_stats, sym_buffer,
                     sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank,
                     num_experts, num_topk, recipe, activation,
                     activation_clamp, fast_math, l1_alphas, l2_alphas,
                     a2_scales, routed_scaling_factor):
    return _torch_ops.fp4_fp4_mega_moe(
        y, l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1],
        shared_l1_weights, shared_l2_weights, x_bf16,
        cumulative_local_expert_recv_stats, sym_buffer,
        list(sym_buffer_ptrs), rank_idx, num_max_tokens_per_rank,
        num_experts, num_topk, list(recipe), activation,
        activation_clamp, fast_math, l1_alphas, l2_alphas,
        a2_scales, routed_scaling_factor,
    )


def bf16_mega_moe(y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights,
                  cumulative_local_expert_recv_stats, sym_buffer,
                  sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk,
                  activation, activation_clamp, fast_math):
    return _torch_ops.bf16_mega_moe(
        y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats, sym_buffer,
        list(sym_buffer_ptrs), rank_idx, num_max_tokens_per_rank, num_experts, num_topk,
        activation, activation_clamp, fast_math,
    )


def fp8_mega_moe(y, l1_weights, l2_weights,
                 cumulative_local_expert_recv_stats, sym_buffer,
                 sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank,
                 num_experts, num_topk, recipe, activation,
                 activation_clamp, fast_math):
    return _torch_ops.fp8_mega_moe(
        y, l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1],
        cumulative_local_expert_recv_stats, sym_buffer,
        list(sym_buffer_ptrs), rank_idx, num_max_tokens_per_rank,
        num_experts, num_topk, list(recipe), activation,
        activation_clamp, fast_math,
    )


_UNCONDITIONAL_API = (
    # Runtime
    'init',
    'set_num_sms', 'get_num_sms',
    'set_tc_util', 'get_tc_util',
    'set_pdl', 'get_pdl',
    'set_ignore_compile_dims',
    'set_block_size_multiple_of',
    'set_mk_alignment_for_contiguous_layout',
    'get_mk_alignment_for_contiguous_layout',
    'get_theoretical_mk_alignment_for_contiguous_layout',
    # cuBLASLt GEMMs
    'cublaslt_gemm_nt', 'cublaslt_gemm_nn',
    'cublaslt_gemm_tn', 'cublaslt_gemm_tt',
    # Mega MoE (imported via deep_gemm.mega; always defined, fails at call if unregistered)
    'get_symm_buffer_size_for_mega_moe',
    '_slice_symm_buffer_for_mega_moe',
    'get_symm_buffer_size_for_sm90_mega_moe',
    '_slice_symm_buffer_for_sm90_mega_moe',
    'fp8_fp4_mega_moe',
    'fp4_fp4_mega_moe',
    'bf16_mega_moe',
    'fp8_mega_moe',
)

_DEEP_GEMM_API = (
    # FP8/FP4 GEMMs
    'fp8_fp4_gemm_nt', 'fp8_fp4_gemm_nn',
    'fp8_fp4_gemm_tn', 'fp8_fp4_gemm_tt',
    'fp8_gemm_nt', 'fp8_gemm_nn',
    'fp8_gemm_tn', 'fp8_gemm_tt',
    'fp8_gemm_nt_skip_head_mid',
    'm_grouped_fp8_fp4_gemm_nt_contiguous',
    'm_grouped_fp8_fp4_gemm_nn_contiguous',
    'm_grouped_fp8_fp4_gemm_nt_masked',
    'm_grouped_fp8_gemm_nt_contiguous',
    'm_grouped_fp8_gemm_nn_contiguous',
    'm_grouped_fp8_gemm_nt_masked',
    'k_grouped_fp8_gemm_tn_contiguous',
    'k_grouped_fp8_gemm_nt_contiguous',
    # BF16 GEMMs (guarded)
    'bf16_gemm_nt', 'bf16_gemm_nn',
    'bf16_gemm_tn', 'bf16_gemm_tt',
    'm_grouped_bf16_gemm_nt_contiguous',
    'm_grouped_bf16_gemm_nn_contiguous',
    'm_grouped_bf16_gemm_nt_masked',
    'k_grouped_bf16_gemm_tn_contiguous',
    # Einsum
    'einsum',
    'fp8_einsum',
    # Attention
    'fp8_fp4_mqa_logits',
    'get_paged_mqa_logits_metadata',
    'fp8_fp4_paged_mqa_logits',
    'fp8_mqa_logits',
    'fp8_paged_mqa_logits',
    # Hyperconnection (guarded)
    'tf32_hc_prenorm_gemm',
    # Layout (guarded)
    'transform_sf_into_required_layout',
    'get_tma_aligned_size',
    'get_mn_major_tma_aligned_tensor',
    'get_mn_major_tma_aligned_packed_ue8m0_tensor',
    'get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor',
    # Mega helpers (guarded)
    'get_token_alignment_for_mega_moe',
    'get_block_m_for_mega_moe',
    'get_token_alignment_for_sm90_mega_moe',
)

__all__ = list(_UNCONDITIONAL_API) + [name for name in _DEEP_GEMM_API if name in globals()]
