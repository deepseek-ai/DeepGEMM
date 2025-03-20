import torch
from typing import Tuple

from .gemm import get_best_configs
from .tuner import jit_tuner
from .utils import get_col_major_tma_aligned_tensor, get_num_sms

# C++ code templates
includes = ('"deep_gemm/fp8_gemm_varlen_groupM.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated grouped GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, {NUM_GROUPS}, kNumStages, kNumTMAMulticast, GemmType::{GEMM_TYPE}>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
auto tma_d_desc_128rows = GemmType::make_2d_tma_d_desc(out, m, 128);
auto tma_d_desc_64rows = GemmType::make_2d_tma_d_desc(out, m, 64);
auto tma_d_desc_32rows = GemmType::make_2d_tma_d_desc(out, m, 32);
auto tma_d_desc_16rows = GemmType::make_2d_tma_d_desc(out, m, 16);
auto tma_d_desc_8rows = GemmType::make_2d_tma_d_desc(out, m, 8);
auto tma_d_desc_4rows = GemmType::make_2d_tma_d_desc(out, m, 4);
auto tma_d_desc_2rows = GemmType::make_2d_tma_d_desc(out, m, 2);
auto tma_d_desc_1row = GemmType::make_2d_tma_d_desc(out, m, 1);

GemmType::run(out, rhs_scales, m_indices_pad,
              m, m_pad,
              tma_a_desc, tma_b_desc, tma_scales_a_desc,
              tma_d_desc_128rows, tma_d_desc_64rows, tma_d_desc_32rows,
             tma_d_desc_16rows, tma_d_desc_8rows, tma_d_desc_4rows,
             tma_d_desc_2rows, tma_d_desc_1row,
              group_pad_off, token_cumdiff, token_pad_end,
              stream, num_sms, smem_size);
"""

def m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous(lhs: Tuple[torch.Tensor, torch.Tensor],
                                              rhs: Tuple[torch.Tensor, torch.Tensor],
                                              size_per_group: torch.Tensor,
                                              ) -> torch.Tensor:
    """
    Do a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.
    On the M axis, inputs are grouped into several batches, of which batch sizes aligned to
        `get_m_alignment_for_contiguous_layout()` (128).

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m_sum, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m_sum, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[num_groups, n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[num_groups, ⌈n / 128⌉, ⌈k / 128⌉]`.
        size_per_group: a tensor of shape `[num_groups]` with type `torch.long`.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    out = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)

    # Type and shape checks
    assert (n % 512) == 0 and (k % 512) == 0
    assert  k == k_ 
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert size_per_group.dtype == torch.long
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and size_per_group.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()

    num_sms, block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms)
    
    size_per_group_padding = ((size_per_group + block_m - 1) // block_m) * block_m
    group_pad_off = torch.zeros(size_per_group.shape[0] + 1, device = "cuda", dtype = torch.long)
    # import pdb; pdb.set_trace()
    group_pad_off[1:] = size_per_group_padding.cumsum(0)
    M_pad = size_per_group_padding.sum().item()
    token_diff = size_per_group_padding - size_per_group
    token_cumdiff = token_diff.cumsum(0)
    token_pad_end = size_per_group_padding.cumsum(0) - token_cumdiff
    token_cumdiff = token_diff.cumsum(0) - token_diff


    group_indices = torch.arange(num_groups, device='cuda')
    repeats = (size_per_group_padding // block_m)
    m_indices_pad = torch.repeat_interleave(group_indices, repeats, dim=0).to(torch.int32)

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m_indices_pad, group_pad_off, token_cumdiff, token_pad_end,
            m, M_pad, num_groups,
            torch.cuda.current_stream(), num_sms, smem_size)
    runtime = jit_tuner.compile_and_tune(
        name='varlen_m_grouped_gemm_fp8_fp8_bf16_nt',
        keys={ 'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast, 'GEMM_TYPE': 'GroupedContiguous'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('m_indices_pad', torch.int32),
                  ('group_pad_off', torch.long),
                  ('token_cumdiff', torch.long),
                  ('token_pad_end', torch.long),
                  ('m', int), ('m_pad', int),
                  ('num_groups', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
    return out

