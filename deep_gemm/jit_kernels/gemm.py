import torch
from typing import Tuple

from .tuner import jit_tuner
from .config import config_cache
from .utils import get_num_sms, ceil_div, get_col_major_tma_aligned_tensor, get_m_alignment_for_contiguous_layout

# C++ code templates
includes = ('"deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumTMAMulticast, GemmType::Normal>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m);
GemmType::run(out, rhs_scales, nullptr,
              m,
              tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
              stream, num_sms, smem_size);
"""

def gemm_fp8_fp8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                         rhs: Tuple[torch.Tensor, torch.Tensor],
                         out: torch.Tensor) -> None:
    """
    Do a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    assert n % 64 == 0 and k % 128 == 0

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == ((n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    block_m, block_n, num_stages, num_tma_multicast, smem_size = config_cache.compute_and_cache(m, n, k, 1, num_sms)
    args = (lhs, lhs_scales, rhs, rhs_scales, out, m, torch.cuda.current_stream(), num_sms, smem_size)
    runtime = jit_tuner.compile_and_tune(
        name='gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
