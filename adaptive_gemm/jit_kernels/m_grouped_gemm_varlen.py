import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_col_major_tma_aligned_tensor, get_num_sms, ceil_div

import triton
import triton.language as tl

@triton.jit
def repeat_interleave_kernel(
    group_ptr,
    repeats_ptr,
    repeat_cum_ptr,
    output_ptr,
    # BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    repeat = tl.load(repeats_ptr + pid)
    start = tl.load(repeat_cum_ptr + pid) - repeat
    group = tl.load(group_ptr + pid)

    for r in range(repeat):
        tl.store(output_ptr + start + r, group)


@torch.library.custom_op("moe::repeat_interleave", mutates_args=('m_indices_pad', ))
def repeat_interleave(
    group_indices: Tensor, 
    repeats: Tensor, 
    repeat_cum: Tensor, 
    m_indices_pad: Tensor, 
) -> None:
    grid = lambda args: (len(repeats), )
    repeat_interleave_kernel[grid](group_indices, repeats, repeat_cum, m_indices_pad)
    return


@repeat_interleave.register_fake
def _(
    group_indices: Tensor, 
    repeats: Tensor, 
    repeat_cum: Tensor, 
    m_indices_pad: Tensor, 
) -> None:
    return


# C++ code templates
includes = ('"adaptive_gemm/fp8_gemm_varlen_groupM.cuh"', )
template = """
using namespace adaptive_gemm;

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
def is_tma_multicast_legal(n: int, block_n: int, num_tma_multicast: int, num_sms: int) -> bool:
    if num_tma_multicast == 1:
        return True
    return (n % (block_n * num_tma_multicast) == 0) and num_sms % num_tma_multicast == 0


def get_smem_size(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128) -> int:
    smem_d = block_m * block_n * 2
    smem_a_per_stage = block_m * block_k
    smem_scales_a_per_stage = block_m * 4 + 128 # 16 bytes if a buffer for scale
    smem_b_per_stage = block_n * block_k
    smem_scales_b = ceil_div(k, block_k) * 4
    smem_barrier = num_stages * 8 * 2

    smem_size = 0
    smem_size += smem_d
    smem_size += num_stages * smem_a_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += num_stages * smem_b_per_stage
    smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    smem_size += smem_barrier
    return smem_size


def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     is_grouped_contiguous: bool = False) -> Tuple[int, int, int, int, int, int]:
    if not is_grouped_contiguous:
        # TODO: for some cases, smaller M block is better, add them into tuning space
        block_ms = (64 if m <= 64 else 128, )
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )
    # block_ns = tuple(range(16, 129, 8))
    block_ns = (128, )

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        for block_n in block_ns:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            if best_block_m is None or best_block_n is None:
                success = True
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util or (util == best_util and (block_m > best_block_m or (block_m == best_block_m and block_n < best_block_n)))
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)
    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_size, sm90_capacity = None, None, 232448
    for num_stages in (6, 5, 4) if 128 % best_block_n != 0 else (8, 7, 6, 5, 4):
        best_smem_size = get_smem_size(num_stages, k, best_block_m, best_block_n)
        if best_smem_size <= sm90_capacity:
            best_num_stages = num_stages
            break
    assert best_num_stages is not None

    # Decide the number of TMA multicast
    best_num_tma_multicast = 1
    if m >= 1024 and is_tma_multicast_legal(n, best_block_n, 2, num_sms) and num_groups == 1:
        best_num_tma_multicast = 2

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)
    num_min_sms = ceil_div(max(num_min_sms, num_sms - 8), best_num_tma_multicast) * best_num_tma_multicast
    assert num_min_sms <= num_sms and is_tma_multicast_legal(n, best_block_n, best_num_tma_multicast, num_min_sms)

    return num_min_sms, best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size


@torch.library.custom_op("moe::varlen_gmm_fp8", mutates_args=('out', ))
def varlen_gmm_fp8(
        lhs: Tensor, 
        lhs_scales: Tensor, 
        rhs: Tensor, 
        rhs_scales: Tensor, 
        out: Tensor, 
        m_indices_pad: Tensor, 
        group_pad_off: Tensor, 
        token_cumdiff: Tensor, 
        token_pad_end: Tensor,
        m: int, 
        M_pad: Tensor, 
        num_groups: int,
        num_sms: int) -> None:
    global includes, template
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    num_sms, block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms)
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
                  ('m', int), ('m_pad', torch.long),
                  ('num_groups', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
    return


@varlen_gmm_fp8.register_fake
def _(
        lhs: Tensor, 
        lhs_scales: Tensor, 
        rhs: Tensor, 
        rhs_scales: Tensor, 
        out: Tensor, 
        m_indices_pad: Tensor, 
        group_pad_off: Tensor, 
        token_cumdiff: Tensor, 
        token_pad_end: Tensor,
        m: int, 
        M_pad: Tensor, 
        num_groups: int,
        num_sms: int) -> None:
    return


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
    # assert (n % 512) == 0 and (k % 512) == 0
    assert  k == k_ 
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == (num_groups, (n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert size_per_group.dtype == torch.long
    lhs = lhs.contiguous()
    rhs = rhs.contiguous()
    size_per_group = size_per_group.contiguous()
    rhs_scales = rhs_scales.contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    # global includes, template

    # When communication overlaps with computing, both operations compete for SM resources.
    # Disable persistent kernel can lead to better performance.
    num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count * 10

    num_sms, block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms)
    
    size_per_group_padding = ((size_per_group + block_m - 1) // block_m) * block_m
    group_pad_off = torch.zeros(size_per_group.shape[0] + 1, device = "cuda", dtype = torch.long)
    # import pdb; pdb.set_trace()
    group_pad_off[1:] = size_per_group_padding.cumsum(0)
    M_pad = size_per_group_padding.sum()
    token_diff = size_per_group_padding - size_per_group
    token_cumdiff = token_diff.cumsum(0)
    token_pad_end = size_per_group_padding.cumsum(0) - token_cumdiff
    token_cumdiff = token_diff.cumsum(0) - token_diff


    group_indices = torch.arange(num_groups, device='cuda').to(torch.int32)
    repeats = (size_per_group_padding // block_m).to(torch.int32)
    m_indices_pad = torch.empty(m, device = "cuda", dtype = torch.int32)
    repeat_cum = repeats.cumsum(0)

    repeat_interleave(group_indices, repeats, repeat_cum, m_indices_pad)

    varlen_gmm_fp8(
        lhs, lhs_scales, rhs, rhs_scales, out,
        m_indices_pad, group_pad_off, token_cumdiff, token_pad_end,
        m, M_pad, num_groups, 
        num_sms)
    
    return out

