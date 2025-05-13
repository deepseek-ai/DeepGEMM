import torch
from typing import Tuple

from .gemm import get_best_configs
from .runtime import (
    FP8GemmRuntime, GemmType,
    make_2d_tma_a_desc, make_2d_tma_b_desc,
    make_2d_tma_d_desc, make_2d_tma_scales_a_desc)
from .tuner import jit_tuner
from .utils import get_col_major_tma_aligned_tensor, get_num_sms


def m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs: Tuple[torch.Tensor, torch.Tensor],
                                              rhs: Tuple[torch.Tensor, torch.Tensor],
                                              out: torch.Tensor, m_indices: torch.Tensor) -> None:
    """
    Perform a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.

    Requirements:
        LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling tensor requires a TMA-aligned transposed format, if your input does not match the requirement,
            this function will do a transposing with a set of slow PyTorch operations.
        On the M axis, inputs are grouped into several batches, of which batch sizes aligned to
            `get_m_alignment_for_contiguous_layout()` (128).

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`,
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m_sum, n]`, representing the result.
        m_indices: a tensor of shape `[m_sum]` with type `torch.int`.
            `m_indices[i]` records the group which the i-th row of the LHS belongs to,
            which means that the i-th row of the LHS matrix will be multiplied with `rhs[m_indices[i]]`.
            Values of `m_indices` in every-m-alignment-block must also be the same.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(
        m, n, k, 1, num_sms, is_grouped_contiguous=True)
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_a_desc(
        GemmType.GroupedContiguous, lhs, m, k, block_m, block_k, num_groups)
    tensor_map_b = make_2d_tma_b_desc(
        GemmType.GroupedContiguous, rhs, k, n, block_k, block_n, num_groups)
    tensor_map_d = make_2d_tma_d_desc(
        GemmType.GroupedContiguous, out, m, n, block_m, block_n, num_groups, smem_config[1])
    tensor_map_scales_a = make_2d_tma_scales_a_desc(
        GemmType.GroupedContiguous, lhs_scales, m, k, block_m, block_k, num_groups)

    kwargs = {
        'NUM_TMA_THREADS': num_tma_threads,
        'NUM_MATH_THREADS_PER_GROUP': num_math_threads_per_group,
        'M': m,
        'BLOCK_K': block_k,
        'GMEM_D': out,
        'SCALES_B': rhs_scales,
        'GROUPED_LAYOUT': m_indices,
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config[0],
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_SCALES_A': tensor_map_scales_a,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
    }

    runtime, best_keys = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages,
              'NUM_TMA_MULTICAST': tma_multicast_config[0],
              'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1],
              'GEMM_TYPE': GemmType.GroupedContiguous},
        space=(),
        kwargs=kwargs,
        runtime_cls=FP8GemmRuntime,
    )

    # Run the kernel
    runtime(**best_keys, **kwargs)


def m_grouped_gemm_fp8_fp8_bf16_nt_masked(lhs: Tuple[torch.Tensor, torch.Tensor],
                                          rhs: Tuple[torch.Tensor, torch.Tensor],
                                          out: torch.Tensor, masked_m: torch.Tensor, expected_m: int) -> None:
    """
    Perform a grouped GEMM (masked format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.

    Requirements:
        LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling tensor requires a TMA-aligned transposed format, if your input does not match the requirement,
            this function will do a transposing with a set of slow PyTorch operations.
        Moreover, this alignment requirement is different with the contiguous-format kernel, as we require that each batch
            should be separately transposed.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, m_max, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[num_groups, m_max, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             The second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[num_groups, m_max, n]`, representing the result.
        masked_m: a tensor of shape `[num_groups]`, `masked_m[i]` records actual rows of the `lhs[i]` matrix to compute
            in the i-th group.
        expected_m: a value hint (which is a value on CPU) for the M expectation of each batch,
            correctly setting this value may lead to better performance.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    num_groups, m, k = lhs.shape
    num_groups_, n, k_ = rhs.shape
    num_groups__, m_, n_ = out.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert lhs_scales.shape == (num_groups, m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and masked_m.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(
        expected_m, n, k, num_groups, num_sms, is_grouped_masked=True)

    # Extra checks for TMA store
    if num_groups > 1 and m > block_m:
        assert m % block_m == 0, f'For masked grouped GEMM, shape M should be multiple of the block M (current block M: {block_m})'

    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_a_desc(
        GemmType.GroupedMasked, lhs, m, k, block_m, block_k, num_groups)
    tensor_map_b = make_2d_tma_b_desc(
        GemmType.GroupedMasked, rhs, k, n, block_k, block_n, num_groups)
    tensor_map_d = make_2d_tma_d_desc(
        GemmType.GroupedMasked, out, m, n, block_m, block_n, num_groups, smem_config[1])
    tensor_map_scales_a = make_2d_tma_scales_a_desc(
        GemmType.GroupedMasked, lhs_scales, m, k, block_m, block_k, num_groups)

    kwargs = {
        'NUM_TMA_THREADS': num_tma_threads,
        'NUM_MATH_THREADS_PER_GROUP': num_math_threads_per_group,
        'M': m,
        'BLOCK_K': block_k,
        'GMEM_D': out,
        'SCALES_B': rhs_scales,
        'GROUPED_LAYOUT': masked_m,
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config[0],
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_SCALES_A': tensor_map_scales_a,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
    }

    runtime, best_keys = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages,
              'NUM_TMA_MULTICAST': tma_multicast_config[0],
              'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1],
              'GEMM_TYPE': GemmType.GroupedMasked},
        space=(),
        kwargs=kwargs,
        runtime_cls=FP8GemmRuntime,
    )

    # Run the kernel
    runtime(**best_keys, **kwargs)
