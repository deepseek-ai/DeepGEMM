import math
import torch
from typing import List, Tuple

from .gemm import get_best_configs
from .tuner import jit_tuner
from .utils import get_num_sms, get_col_major_tma_aligned_tensor, get_tma_aligned_size

# C++ code templates
includes = ('"deep_gemm/fp8_wgrad_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto M = {M}, N = {N};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto BLOCK_K = 128;
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kLastStages = {LAST_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};
constexpr auto kIsTMAMulticastOnA = {IS_TMA_MULTICAST_ON_A};

// Make a templated GEMM
using gemm_t = WgradGemm<M, N, BLOCK_M, BLOCK_N, BLOCK_K, kNumStages, kLastStages, kNumTMAMulticast, kIsTMAMulticastOnA>;

// Launch kernel
auto tma_a_desc = gemm_t::make_2d_tma_a_desc(lhs, m, k, a_stride);
auto tma_b_desc = gemm_t::make_2d_tma_b_desc(rhs, n, k, b_stride);
auto tma_scales_a_desc = gemm_t::make_2d_tma_scales_a_desc(lhs_scales, m, k);
auto tma_scales_b_desc = gemm_t::make_2d_tma_scales_b_desc(rhs_scales, n, k);
auto tma_d_desc = gemm_t::make_2d_tma_d_desc(out, m, n, d_stride);
gemm_t::run(k,
            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_scales_b_desc, tma_d_desc,
            stream, num_sms, smem_size);
"""


def wgrad_gemm_fp8_fp8_fp32_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                               rhs: Tuple[torch.Tensor, torch.Tensor],
                               out: Tuple[torch.Tensor, torch.Tensor]):
    """
    Do a weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
    LHS, RHS, and output tensors must be contiguous in dimension 1, i.e., stride(1) = 1.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling and RHS scaling tensor require TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 1x128 scaling tensor for RHS of shape `[n, ⌈k / 128⌉]`.
        out: the FP32 output tensor of shape `[m, n]`, representing the result.
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

    # LHS and RHS scales must be transposed for TMA load
    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    if lhs_scales.shape == ((k + 127) // 128, m):
        lhs_scales = lhs_scales.permute(1, 0)
        assert get_tma_aligned_size(m, 4) == m
    else:
        lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert lhs_scales.stride(0) == 1
    
    if rhs_scales.shape == ((k + 127) // 128, n):
        rhs_scales = rhs_scales.permute(1, 0)
        assert get_tma_aligned_size(n, 4) == n
    else:
        rhs_scales = get_col_major_tma_aligned_tensor(rhs_scales)
    assert rhs_scales.stride(0) == 1

    # Do nothing if `k` is zero
    if k == 0:
        return

    aligned_n = (n + 63) // 64 * 64
    aligned_k = (k + 127) // 128 * 128

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(m, aligned_n, aligned_k, 1, num_sms, is_fp32_out=True, is_wgrad=True)
    last_stages = (k + 127) // 128 % num_stages

    args = (lhs, lhs_scales, rhs, rhs_scales, out, m, n, k,
            lhs_stride, rhs_stride, out_stride,
            torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='gemm_fp8_fp8_fp32_nt_dptp128c_dyn',
        keys={'M': m, 'N': aligned_n, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'NUM_STAGES': num_stages,
              'LAST_STAGES': last_stages,
              'NUM_TMA_MULTICAST': tma_multicast_config[0],
              'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1]},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.float), ('m', int), ('n', int), ('k', int),
                  ('a_stride', int), ('b_stride', int), ('d_stride', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)


def k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                                         rhs: Tuple[torch.Tensor, torch.Tensor],
                                         out: torch.Tensor,
                                         batch_sizes: List[int]):
    """
    Perform a k-grouped weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
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
        out: The FP32 output tensor of shape [num_batches, m, n], representing the result.
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