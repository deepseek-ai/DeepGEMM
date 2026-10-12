import torch
from typing import Tuple

from .gemm_int8 import get_best_configs
from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_extra_info, is_ppu1v5_device, GemmType
from .gemm import get_gemv_best_configs
import os

# C++ code templates
includes_cutlass3 = ('<deep_gemm/impls/int8_gemm_cutlass3.cuh>', )
template_cutlass3 = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementAB = {ElementAB};
using ElementAcc = {ElementAcc};
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kEnableSboOverlap = {ENABLE_SBO_OVERLAP};

// Make a templated grouped GEMM
using gemm_t = Gemm<ElementAB, ElementAcc, N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::{GEMM_TYPE}, kEnableSboOverlap, KernelType::{KERNEL_TYPE}>;

// Launch kernel
gemm_t::run(out, grouped_layout, block_m_info,
            m, expected_m, (ElementAB*)lhs, lhs_scales, (ElementAB*)rhs, rhs_scales,
            stream, num_sms, smem_size, signal);
"""

includes_gemv = ('<deep_gemm/impls/gemvt.cuh>', )
template_gemv = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto ThreadPerN = {ThreadPerN};
constexpr auto NPerThread = {NPerThread};
constexpr auto NUM_UNROLL = {NUM_UNROLL};
constexpr auto SWZL_SIZE_M = {SWZL_SIZE_M};
constexpr auto USE_SMALL_K = {USE_SMALL_K};
constexpr auto BlockSize = {BlockSize};

// Make a templated grouped GEMM
using gemm_v = Gemvt<int8_t, __ppu_bfloat16, {acc_type}, N, K, kNumGroups, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, BlockSize, USE_SMALL_K>;

// Launch kernel
gemm_v::run(out, grouped_layout,
            m, lhs, rhs,
            stream,
            lhs_scales, rhs_scales);
"""

def m_grouped_gemm_a8w8_per_channel_nt_contiguous(lhs: Tuple[torch.Tensor, torch.Tensor],
                                                  rhs: Tuple[torch.Tensor, torch.Tensor],
                                                  out: torch.Tensor, m_indices: torch.Tensor, configs = None) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, 1)
    assert rhs_scales.shape == (num_groups, n, 1)
    assert lhs.dtype == torch.int8 or lhs.dtype == torch.float8_e4m3fn
    assert rhs.dtype == torch.int8 or rhs.dtype == torch.float8_e4m3fn
    assert lhs.dtype == rhs.dtype
    assert lhs_scales.dtype == torch.float32
    assert rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedContiguous)
    expected_m = ceil_div(m, num_groups)

    ElementAB = "cutlass::float_e4m3_t" if lhs.dtype == torch.float8_e4m3fn else "int8_t"
    ElementAcc = "float" if lhs.dtype == torch.float8_e4m3fn else "int32_t"
    enable_moe_dynamic_tile = False

    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            m_indices, m_indices, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int())

    ## Default, MultistageOnN, OverlapPrologue, OverlapMainloop
    kernel_type = 'Default'
    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_int8_int8_bf16_nt',
        keys={'ElementAB' : ElementAB, "ElementAcc" : ElementAcc,
              'N': n, 'K': k,
              'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'ENABLE_SBO_OVERLAP': False,
              'GEMM_TYPE': 'GroupedContiguous',
              'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_cutlass3,
        arg_defs=(('lhs', lhs.dtype), ('lhs_scales', torch.float),
                  ('rhs', rhs.dtype), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('block_m_info', torch.int32),
                  ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32)),
        template=template_cutlass3,
        jit_include_dir='actlize_v1.0.0',
        args=args
    )
    # Run the kernel
    runtime(*args)

def m_grouped_gemm_int8_int8_bf16_nt_contiguous(lhs: Tuple[torch.Tensor, torch.Tensor],
                                                rhs: Tuple[torch.Tensor, torch.Tensor],
                                                out: torch.Tensor, m_indices: torch.Tensor, configs = None) -> None:
    m_grouped_gemm_a8w8_per_channel_nt_contiguous(lhs, rhs, out, m_indices, configs)

def m_grouped_gemm_a8w8_per_channel_nt_masked(lhs: Tuple[torch.Tensor, torch.Tensor],
                                              rhs: Tuple[torch.Tensor, torch.Tensor],
                                              out: torch.Tensor, masked_m: torch.Tensor, expected_m: int, configs = None,
                                              max_block_n: int = 256, enable_sbo_overlap: bool = False,
                                              signal: torch.Tensor = torch.empty(0).int()) -> None:
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
    assert lhs_scales.shape == (num_groups, m, 1)
    assert rhs_scales.shape == (num_groups, n, 1)
    assert lhs.dtype == torch.int8 or lhs.dtype == torch.float8_e4m3fn
    assert rhs.dtype == torch.int8 or rhs.dtype == torch.float8_e4m3fn
    assert lhs.dtype == rhs.dtype
    assert lhs_scales.dtype == torch.float32
    assert rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and masked_m.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    if enable_sbo_overlap:
        assert signal is not None
        assert signal.is_contiguous()
        assert signal.dtype == torch.int32


    # Auto-tuning with compilation
    num_sms = get_num_sms()
    if configs:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedMasked, max_block_n=max_block_n)

    ## Default, MultistageOnN, MoeDynamicTile, OverlapPrologue, OverlapMainloop
    kernel_type = 'Default'

    ElementAB = "cutlass::float_e4m3_t" if lhs.dtype == torch.float8_e4m3fn else "int8_t"
    ElementAcc = "float" if lhs.dtype == torch.float8_e4m3fn else "int32_t"
    enable_moe_dynamic_tile = get_extra_info()['use_moe_dynamic_tile']

    supported_N_K_list = [
        (4096, 7168), (7168, 2048), # dpsk
        (4096, 8192), (8192, 2048), # qwen3.5
    ]
    if is_ppu1v5_device() and (n, k) in supported_N_K_list:
        enable_moe_dynamic_tile = True
    if enable_sbo_overlap:
        # disable dynamic tile if enable_sbo_overlap, as the kNumNBlocks is not static
        enable_moe_dynamic_tile = False
    if enable_moe_dynamic_tile:
        # fix the block config to avoid unecessary jit compile
        block_m, block_n, block_k, warp_m, warp_n, num_stages = (128, 128, 128, 64, 64, 3)
        kernel_type = 'MoeDynamicTile'


    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            masked_m, masked_m, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], signal)

    runtime = jit_tuner.compile_and_tune(
        name='m_grouped_gemm_' + ElementAB + '_nt',
        keys={'ElementAB' : ElementAB, "ElementAcc" : ElementAcc,
              'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
              'WARP_M': warp_m, 'WARP_N': warp_n,
              'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
              'ENABLE_SBO_OVERLAP': enable_sbo_overlap,
              'GEMM_TYPE': 'GroupedMasked',
              'KERNEL_TYPE': kernel_type},
        space=(),
        includes=includes_cutlass3,
        arg_defs=(('lhs', lhs.dtype), ('lhs_scales', torch.float),
                  ('rhs', rhs.dtype), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('m', int), ('expected_m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('signal', torch.int32)),
        template=template_cutlass3,
        jit_include_dir='actlize_v1.0.0',
        args=args
    )
    # Run the kernel
    runtime(*args)

    return (block_m, ceil_div(n, block_n))

def m_grouped_gemm_int8_int8_bf16_nt_masked(lhs: Tuple[torch.Tensor, torch.Tensor],
                                            rhs: Tuple[torch.Tensor, torch.Tensor],
                                            out: torch.Tensor, masked_m: torch.Tensor, expected_m: int, configs = None,
                                            max_block_n: int = 256, enable_sbo_overlap: bool = False,
                                            signal: torch.Tensor = torch.empty(0).int()) -> None:
    return m_grouped_gemm_a8w8_per_channel_nt_masked(lhs, rhs, out, masked_m, expected_m, configs, max_block_n, enable_sbo_overlap, signal)

def m_grouped_gemm_a8w8_per_channel_nt_nopad(lhs: Tuple[torch.Tensor],
                                             rhs: Tuple[torch.Tensor],
                                             out: torch.Tensor, m_indices: torch.Tensor,
                                             m_rows: torch.Tensor = None, configs = None) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    num_groups, n, k_ = rhs.shape
    m_, n_ = out.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and k == k_ and n == n_
    assert lhs_scales.shape == (m, 1)
    assert rhs_scales.shape == (num_groups, n, 1)
    assert lhs.dtype == torch.int8 or lhs.dtype == torch.float8_e4m3fn
    assert rhs.dtype == torch.int8 or rhs.dtype == torch.float8_e4m3fn
    assert lhs.dtype == rhs.dtype
    assert lhs_scales.dtype == torch.float32
    assert rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and m_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    expected_m = ceil_div(m, num_groups)

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    use_gemv = False
    if k % 16 == 0 and ((m <= 2 * num_groups * 0.75 and k <= 32 * 8) or (m < 0.65 * num_groups and k > 256)) and not is_ppu1v5_device():
        # use gemmv if avg m small
        # ThreadPerN = 8
        # NUM_UNROLL = 1
        # SWZL_SIZE_M = 1
        # NPerThread = 1
        BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, USE_SMALL_K = get_gemv_best_configs(m, n, k, num_groups, num_sms, torch.int8)

        # GemV kernel requires N % NPerBlock == 0: both the host-side grid_y
        # and the per-thread boundary check assume full N-tiles.  Fall back to
        # the general GEMM path when N is not a multiple of NPerBlock.
        n_per_block = (NPerThread * BlockSize // ThreadPerN) if ThreadPerN != -1 else 1
        if ThreadPerN != -1 and lhs.dtype != torch.float8_e4m3fn and n % n_per_block == 0:
            args = (lhs, rhs, out,
                m_indices, m,
                torch.cuda.current_stream(),
                lhs_scales, rhs_scales)

            runtime = jit_tuner.compile_and_tune(
                name='m_grouped_gemv_int8_int8_bf16_nt',
                keys={'N': n, 'K': k, 'NUM_GROUPS': num_groups,
                    'ThreadPerN':ThreadPerN, 'NUM_UNROLL':NUM_UNROLL,
                    'SWZL_SIZE_M':SWZL_SIZE_M, 'NPerThread':NPerThread,
                    'BlockSize':BlockSize, 'USE_SMALL_K':USE_SMALL_K,
                    'acc_type':'int'},
                space=(),
                includes=includes_gemv,
                arg_defs=(('lhs', torch.int8),
                        ('rhs', torch.int8),
                        ('out', torch.bfloat16),
                        ('grouped_layout', torch.int32), ('m', int),
                        ('stream', torch.cuda.Stream),
                        ('lhs_scales', torch.float),
                        ('rhs_scales', torch.float)),
                template=template_gemv,
                args=args
            )
            use_gemv = True

    if use_gemv == False:
        if configs:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
        else:
            num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(expected_m, n, k, num_groups, num_sms, gemm_type=GemmType.GroupedNoPad)

        ## Default, MultistageOnN, MoeDynamicTile, OverlapPrologue, OverlapMainloop
        kernel_type = 'Default'
        if k < 4096:
            kernel_type = 'OverlapPrologue'

        ElementAB = "cutlass::float_e4m3_t" if lhs.dtype == torch.float8_e4m3fn else "int8_t"
        ElementAcc = "float" if lhs.dtype == torch.float8_e4m3fn else "int32_t"
        # if enable_moe_dynamic_tile:
        #     # fix the block config to avoid unecessary jit compile
        #     block_m, block_n, block_k, warp_m, warp_n, num_stages = (128, 128, 128, 64, 64, 3)
        #     kernel_type = 'MoeDynamicTile'

        if m_rows is None:
            counts = torch.bincount(m_indices)
            min_n = min(counts.size(0), num_groups)
            experts_for_rows = torch.zeros(num_groups, dtype=torch.int32, device='cuda')
            if min_n > 0:
                experts_for_rows[:min_n] = counts[:min_n]
            m_rows = experts_for_rows

        ## the largest blockM_num is, num_groups - 1 only has 1 token, the last group has (m-1) tokens, blockM_num = num_group -1  + ceil_div(m + 1 - num_group, block_m)
        ## total line num: blockM_num + 1, line0 is used to store the real blockM_num
        ## total_size = (blockM_num + 1) * 4 * sizeof(int) Byte
        block_m_info = torch.empty((num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4, dtype=torch.int32, device=m_rows.device)

        args = (lhs, lhs_scales, rhs, rhs_scales, out, m_rows, block_m_info, m, expected_m,
            torch.cuda.current_stream(), num_sms, smem_config[0], torch.empty(0).int())
        runtime = jit_tuner.compile_and_tune(
            name='m_grouped_gemm_' + ElementAB + '_nt',
            keys={'ElementAB' : ElementAB, "ElementAcc" : ElementAcc,
                  'N': n, 'K': k,
                  'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                  'WARP_M': warp_m, 'WARP_N': warp_n,
                  'NUM_GROUPS': num_groups, 'NUM_STAGES': num_stages,
                  'ENABLE_SBO_OVERLAP': False,
                  'GEMM_TYPE': 'GroupedNoPad',
                  'KERNEL_TYPE': kernel_type
                  },
            space=(),
            includes=includes_cutlass3,
            arg_defs=(  ('lhs', lhs.dtype), ('lhs_scales', torch.float),
                        ('rhs', rhs.dtype), ('rhs_scales', torch.float),
                        ('out', torch.bfloat16),
                        ('grouped_layout', torch.int32), ('block_m_info', torch.int32), ('m', int), ('expected_m', int),
                        ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                        ('signal', torch.int32)),
            template=template_cutlass3,
            jit_include_dir='actlize_v1.0.0',
            args=args
        )
    # Run the kernel
    runtime(*args)

def m_grouped_gemm_int8_int8_bf16_nt_nopad(lhs: Tuple[torch.Tensor],
                                     rhs: Tuple[torch.Tensor],
                                     out: torch.Tensor, m_indices: torch.Tensor,
                                     m_rows: torch.Tensor = None, configs = None) -> None:
    m_grouped_gemm_a8w8_per_channel_nt_nopad(lhs, rhs, out, m_indices, m_rows, configs)
