import torch
from typing import List, Tuple

from .runtime import (
    FP8WGradGemmRuntime, GemmType,
    make_2d_tma_a_desc, make_2d_tma_b_desc,
    make_2d_tma_d_desc, make_2d_tma_scales_a_desc, make_2d_tma_scales_b_desc)
from .gemm import get_best_configs
from .tuner import jit_tuner
from .utils import get_num_sms, get_col_major_tma_aligned_tensor, get_tma_aligned_size


def wgrad_gemm_fp8_fp8_fp32_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                               rhs: Tuple[torch.Tensor, torch.Tensor],
                               out: Tuple[torch.Tensor, torch.Tensor]):
    """
    Perform a weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
        Results will be accumulated into the output tensor.

    Requirements:
        LHS, RHS, and output tensors must be contiguous in dimension 1, i.e., stride(1) = 1.
        The stride(0) of LHS and RHS must be a multiple of 16, and the stride(0) of output must be a multiple of 4.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling and RHS scaling tensor require TMA-aligned transposed format, if your input does not match the requirement,
            this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`,
             the second element is an FP32 1x128 scaling tensor for RHS of shape `[n, ⌈k / 128⌉]`.
        out: the FP32 output tensor of shape `[m, n]`, which will be accumulated.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and m > 0
    assert lhs_scales.shape == (m, (k + 127) // 128) or lhs_scales.shape == ((k + 127) // 128, m)
    assert rhs_scales.shape == (n, (k + 127) // 128) or rhs_scales.shape == ((k + 127) // 128, n)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.float
    assert lhs.stride(1) == 1 and out.stride(1) == 1 and rhs.stride(1) == 1

    lhs_stride = lhs.stride(0)
    rhs_stride = rhs.stride(0)
    out_stride = out.stride(0)

    # The stride(0) of LHS, RHS, and output must be aligned to 16 bytes
    assert lhs_stride % 16 == 0 and rhs_stride % 16 == 0 and out_stride % 4 == 0

    # LHS and RHS scales must be transposed for TMA load
    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    if lhs_scales.shape == ((k + 127) // 128, m):
        lhs_scales = lhs_scales.permute(1, 0)
        assert get_tma_aligned_size(m, 4) == m and lhs_scales.stride(1) == m
    else:
        lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert lhs_scales.stride(0) == 1
    
    if rhs_scales.shape == ((k + 127) // 128, n):
        rhs_scales = rhs_scales.permute(1, 0)
        assert get_tma_aligned_size(n, 4) == n and rhs_scales.stride(1) == n
    else:
        rhs_scales = get_col_major_tma_aligned_tensor(rhs_scales)
    assert rhs_scales.stride(0) == 1

    # Do nothing if `k` is zero
    if k == 0:
        return

    # K must be aligned to 128
    aligned_k = (k + 127) // 128 * 128

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(
        m, n, aligned_k, 1, num_sms, is_fp32_out=True, is_wgrad=True)
    last_stages = (k + 127) // 128 % num_stages
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_a_desc(
        GemmType.Normal, lhs, m, k, block_m, block_k, 1, a_stride=lhs_stride)
    tensor_map_b = make_2d_tma_b_desc(
        GemmType.Normal, rhs, k, n, block_k, block_n, 1, b_stride=rhs_stride)
    tensor_map_d = make_2d_tma_d_desc(
        GemmType.Normal, out, m, n, block_m, block_n, 1, smem_config[1], d_stride=out_stride)
    tensor_map_scales_a = make_2d_tma_scales_a_desc(
        GemmType.Normal, lhs_scales, m, k, block_m, block_k)
    tensor_map_scales_b = make_2d_tma_scales_b_desc(
        GemmType.Normal, rhs_scales, n, k, block_n, block_k)

    kwargs = {
        'GEMM_TYPE': GemmType.Normal,
        'NUM_TMA_THREADS': num_tma_threads,
        'NUM_MATH_THREADS_PER_GROUP': num_math_threads_per_group,
        'K': aligned_k,
        'NUM_GROUPS': 1,
        'BLOCK_K': block_k,
        'GMEM_D': out,
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config[0],
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_SCALES_A': tensor_map_scales_a,
        'TENSOR_MAP_SCALES_B': tensor_map_scales_b,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
    }

    runtime, best_keys = jit_tuner.compile_and_tune(
        name='wgrad_gemm_fp8_fp8_fp32_nt',
        keys={'M': m, 'N': n,
              'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'NUM_STAGES': num_stages,
              'LAST_STAGES': last_stages,
              'NUM_TMA_MULTICAST': tma_multicast_config[0],
              'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1]},
        space=(),
        kwargs=kwargs,
        runtime_cls=FP8WGradGemmRuntime,
    )

    # Run the kernel
    runtime(**best_keys, **kwargs)


def k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                                         rhs: Tuple[torch.Tensor, torch.Tensor],
                                         out: torch.Tensor,
                                         batch_sizes: List[int]):
    """
    Perform a k-grouped weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
        Results will be accumulated into the output tensor.

    Requirements:
        This function handles multiple batches with varying k-dimensions, processing each batch sequentially.
        Each batch's LHS, RHS, and output tensors must be contiguous.
        The RHS and RHS scaling factors are required to be transposed.
        The LHS scaling and RHS scaling tensors require TMA-aligned transposed format.

    Arguments:
        lhs: the first element is a flattened FP8 tensor (typed `torch.float8_e4m3fn`) containing all batches of LHS data,
                 and the flattened shape is `[sum(m * k for k in batch_sizes)]`, where m is the number of rows.
             the second element is an FP32 scaling tensor for LHS with shape `[⌈k / 128⌉ for k in batch_sizes), m]`,
                 representing the per-128-channel scaling factors.
        rhs: the first element is a flattened FP8 tensor (typed `torch.float8_e4m3fn`) containing all batches of RHS data,
                 and the flattened shape is `[sum(n * k for k in batch_sizes)]`, where n is the number of rows.
             the second element is an FP32 scaling tensor for RHS with shape `[⌈k / 128⌉ for k in batch_sizes), n]`,
                 representing the per-128-channel scaling factors.
        out: The FP32 output tensor of shape [num_batches, m, n], which will be accumulated.
        batch_sizes: A list of integers specifying the k-dimension for each batch.
    """
    lhs, lhs_scales = lhs[0].view(-1), lhs[1]
    rhs, rhs_scales = rhs[0].view(-1), rhs[1]
    num_batches, m, n = out.shape

    lhs_offset, rhs_offset, scales_offset = 0, 0, 0

    for idx in range(num_batches):
        k = batch_sizes[idx]
        A = lhs[lhs_offset:lhs_offset + m * k].view(m, k)
        B = rhs[rhs_offset:rhs_offset + n * k].view(n, k)
        A_scales = lhs_scales[scales_offset:scales_offset + (k + 127) // 128]
        B_scales = rhs_scales[scales_offset:scales_offset + (k + 127) // 128]
        D = out[idx]

        wgrad_gemm_fp8_fp8_fp32_nt((A, A_scales), (B, B_scales), D)

        lhs_offset += m * k
        rhs_offset += n * k
        scales_offset += (k + 127) // 128