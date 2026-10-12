import math
import os
import torch
from functools import lru_cache
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout, is_ppu1v5_device, GemmType
from .gemm_search_space import MatmulHeuristicsTile
from .densegemm_adaptive_select_strategy import get_adaptive_configs, is_bf16_adaptive_enabled, get_warp_k

# C++ code templates
includes_bf16_cutlass3 = ('<deep_gemm/impls/bf16_gemm_cutlass3.cuh>', )
template_bf16_cutlass3 = """
using namespace deep_gemm;

constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = 1;
constexpr auto kNumStages = {NUM_STAGES};

using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N,
                    kNumGroups, kNumStages, GemmType::DenseGemm>;

gemm_t::run(out, nullptr, nullptr,
            m, 0, lhs, rhs,
            stream, num_sms, smem_size);
"""

# PPU1.5 DenseGemm-only cutlass3 path: standalone kernel that reads N/K at runtime.
# Do NOT reuse includes_bf16_densegemm/template_bf16_densegemm for grouped or MoE paths.
includes_bf16_densegemm = ('<deep_gemm/impls/bf16_densegemm_cutlass3.cuh>', )
template_bf16_densegemm = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementAB = cutlass::bfloat16_t;
using ElementAcc = float;
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto WARP_K = {WARP_K};
constexpr auto kNumStages = {NUM_STAGES};
constexpr bool kIsAlignedN = {IS_ALIGNED_N};

// Make a templated BF16 DenseGemm (no SHAPE_N/K in template - uses runtime args)
using gemm_t = BF16DenseGemm<ElementAB, ElementAcc, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumStages, {DENSE_S2_OPT}, KernelType::{KERNEL_TYPE}, kIsAlignedN>;

// Launch kernel
gemm_t::run(out, m, n, k, lhs, rhs,
            stream, num_sms, smem_size);
"""
def get_smem_config(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128, bpp: int = 2) -> Tuple[int, int, int]:
    swizzle_mode = 128
    block_n_padding = 0

    smem_d = block_m * (block_n + block_n_padding)
    smem_a_per_stage = block_m * block_k
    smem_b_per_stage = block_n * block_k

    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)

    assert int(swizzle_mode > 0) + int(block_n_padding > 0) <= 1

    return smem_size, swizzle_mode, block_n_padding

def get_smem_occ(block_m: int, block_n: int) -> Tuple[int]:
    if block_m is None:
        return 0

    # use static suppose.
    ppu_capacity = 262144
    block_k = 64
    bpp = 2
    num_stages = 2
    smem_d = block_m * block_n
    smem_a_per_stage = block_m * block_k
    smem_b_per_stage = block_n * block_k

    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)

    return 262144 // smem_size

# Adaptive tile selection for 890P DenseGemm (shared with int8 DenseGemm)
# lives in densegemm_adaptive_select_strategy.py. Adaptive detection is done
# per-shape via is_bf16_adaptive_shape() in the heuristic path below.


@lru_cache(maxsize=None)
def get_gemv_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int, dtype: torch.dtype):
    if dtype == torch.int8:
        Alignment = 16
    else:
        Alignment = 8
    small_k_algo_limit = 32 * Alignment
    SmallK = False

    if k <= small_k_algo_limit:
        if (k <= 8 * Alignment):
            BlockSize = 64
            ThreadPerN = 8
            NPerThread = 16
            NUM_UNROLL = 1
            SWZL_SIZE_M = 1
        elif (k <= 16 * Alignment):
            BlockSize = 64
            ThreadPerN = 16
            NPerThread = 16
            NUM_UNROLL = 1
            SWZL_SIZE_M = 1
        # elif (k <= 24 * Alignment):
        # block size 96 has accuracy issue.
        #     BlockSize = 96
        #     ThreadPerN = 24
        #     NPerThread = 16
        #     NUM_UNROLL = 1
        #     SWZL_SIZE_M = 1
        else:
            BlockSize = 64
            ThreadPerN = 32
            NPerThread = 16
            NUM_UNROLL = 1
            SWZL_SIZE_M = 1
        SmallK = True
        Stages = 5
    else:
        BlockSize = 256
        if k % (32 * 2 * Alignment) == 0:
            ThreadPerN = 32
            NUM_UNROLL = 2

            if m >= 16 * 8:
                NPerThread = 4
                SWZL_SIZE_M = 4
            elif m >= 4 * 8:
                NPerThread = 2
                SWZL_SIZE_M = 2
            else:
                NPerThread = 1
                SWZL_SIZE_M = 1
        elif dtype == torch.int8 and k % (32 * Alignment) == 0 and n % 32 == 0:
            # int8 K aligns to 512 but not 1024 (K=1536/2560/3584...).
            ThreadPerN = 32
            NUM_UNROLL = 1
            SWZL_SIZE_M = 1
            if m >= 16 * 8:
                NPerThread = 4
            elif m >= 4 * 8:
                NPerThread = 2
            else:
                NPerThread = 1
        elif k % (8 * Alignment) == 0:
            ThreadPerN = 8
            NUM_UNROLL = 1
            SWZL_SIZE_M = 1

            if m >= 8 * 8:
                NPerThread = 4
            else:
                NPerThread = 1
        else:
            print(f"DeepGemm: gemv not support m:{m}, n:{n}, k:{k}, groups:{num_groups}, num_sms:{num_sms}\n")
            ThreadPerN = -1
            NUM_UNROLL = -1
            SWZL_SIZE_M = -1
            NPerThread = -1

    return BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, SmallK

@lru_cache(maxsize=None)
def get_adaptive_configs_bf16(m: int, n: int, k: int, num_sms: int):
    """bf16 adaptive selector using the shared DenseGemm tile strategy.
    Wraps get_adaptive_configs(), drops warp_k, and computes bf16 smem_config
    (bpp=2). Returns 8-tuple matching the get_best_configs contract:
      (num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config)
    """
    ns, bm, bn, bk, wm, wn, wk, s = get_adaptive_configs(m, n, k, num_sms)
    smem_config = get_smem_config(s, k, bm, bn, bk, bpp=2)
    return ns, bm, bn, bk, wm, wn, s, smem_config

@lru_cache(maxsize=None)
def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     gemm_type: GemmType=GemmType.DenseGemm, max_block_n: int = 256) -> \
        Tuple[int, int, int, int, Tuple[int, bool], Tuple[int, int, int]]:

    # Adaptive tile selector for bf16 DenseGemm on PPU1.5
    if is_bf16_adaptive_enabled(m, n, k) and is_ppu1v5_device() and gemm_type == GemmType.DenseGemm:
        return get_adaptive_configs_bf16(m, n, k, num_sms)

    #FIXME: block m can add 16, and blockM/N could be 512
    if gemm_type != GemmType.GroupedContiguous:
        # block_ms = (32, 64, 128, 256)
        block_ms = (256, 128, 64, 32, 16) if k >= 384 else (64, 32, 16)
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )
        # block_ms = (16, 32)

    # block_ns = (256, 128, 64, 32)
    assert max_block_n > 0 and (max_block_n & (max_block_n - 1)) == 0
    block_ns = tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 1, 4, -1))) if k >= 384 else tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 2, 4, -1)))

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    #block size wasted
    # get_block_utils = lambda x, y: (x / y) % 1 if x % y != 0 else 1.0
    get_block_utils = lambda m, bm: (((m / bm ) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0
    # get_block_utils = lambda m, bm: ((m / bm) % 1 if m % bm != 0 else 1.0) if bm else 0

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    min_n_threshold = 1 if gemm_type == GemmType.DenseGemm else 32
    for block_m in block_ms:
        if is_ppu1v5_device() and ((m >= 128 and k > 2048) or m >= 256):
            block_ns_after_filter = filter(lambda bn: (bn != n and n >= min_n_threshold) and not (block_m == 16 and bn <= 32), block_ns)
        else:
            block_ns_after_filter = \
                filter(lambda bn: (block_m <= 128 or bn <= 128) and (bn != n and n >= min_n_threshold) and not (block_m == 16 and bn <= 32), block_ns)

        # NOTES: the block sizes can not be too large, so at least one dim less than 128
        for block_n in block_ns_after_filter:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)
            num_occ, best_num_occ = get_smem_occ(block_m, block_n), get_smem_occ(best_block_m, best_block_n)

            # print(f"block_m:{block_m}, block_n:{block_n}, best_block_m:{best_block_m}, best_block_n:{best_block_n}")
            # print(f'num_occ:{num_occ}, best_num_occ:{best_num_occ}')
            # # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
            # print(f'm_util:{get_block_utils(m, block_m)}, n_util:{get_block_utils(n, block_n)}, num_utils:{num_utils}')
            # print(f'best_m_util:{get_block_utils(m, best_block_m)}, best_n_util:{get_block_utils(n, best_block_n)}, best_num_utils:{best_num_utils}')

            if best_block_m is None or best_block_n is None:
                success = True
            elif (m < 512 or n < 512):
                # if single group block is small, balance wave, utils and occ
                occ_wave = num_waves / num_occ
                best_occ_wave = best_num_waves / best_num_occ
                ai_util = (block_m * block_n) / (block_m + block_n)
                best_ai_util = (best_block_m * best_block_n) / (best_block_m + best_block_n)

                valid_occ  = (num_occ / best_num_occ) >= 1
                valid_wave = (occ_wave / best_occ_wave) <= 1
                valid_util = (num_utils / best_num_utils) >= 1
                valid_ai   = (ai_util / best_ai_util) >= 1

                # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
                # print(f'ai:{ai_util}, best_ai:{best_ai_util}')
                # print(f'util_ratio:{(num_utils / best_num_utils)}, wave_ratio:{occ_wave / best_occ_wave}, occ_ratio:{(num_occ / best_num_occ)}, ai_ratio:{ai_util / best_ai_util}')

                # print(f'valid_occ:{valid_occ}, valid_wave:{valid_wave}, valid_util:{valid_util}, valid_ai:{valid_ai}')
                success = (valid_wave + valid_util + valid_occ + valid_ai) >= 3

            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, `block_n` larger is better
                    success |= block_m != best_block_m and block_n > best_block_n

            # print(f'm:{m}, n:{n}, k:{k}, block_m:{block_m}, block_n:{block_n}, num_waves:{num_waves}, best_num_waves:{best_num_waves}, success:{success}\n')
            # print(f'\n\n\n')
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)

    # best_block_m = 32
    # best_block_n = 128
    if not is_ppu1v5_device() and (m > 64 and n == 128 and k == 4096):
        best_block_m = 128
        best_block_n = 128

    #small m hbm bound or latency bound, wave is not usful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m < 20 and n < 512) :
        best_block_m = 16
        best_block_n = 64

    # best_block_n = 128
    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144

    block_k = 64
    if k <= 64:
        block_k = 32
    if k >= 4096 and (best_block_m <= 32 and best_block_n <= 64):
        block_k = 128

    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (8, 7, 6, 5, 4, 3, 2)))

    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16 or best_block_m == 32:
        stage_candidates = (3, 2)

    if best_block_m == 256 and best_block_n == 256:
        stage_candidates = (4,)

    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k)
        # print(f"num_stages:{num_stages}, best_smem_config:{best_smem_config}")
        if best_smem_config[0] <= ppu_capacity:
            occ = ppu_capacity // best_smem_config[0]
            if k < 512 or (best_block_m > 32 and best_block_n >= 64) and occ >= best_occ:
                # compute block and too small-k use higer occ rather than large stage
                best_num_stages = num_stages
                best_occ = occ
            else:
                best_num_stages = num_stages
                break

    # best_num_stages = 3
    assert best_smem_config is not None
    assert best_num_stages is not None

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)

    # num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)
    num_min_sms = ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups

    # assert num_min_sms <= num_sms

    warp_m = best_block_m // 2
    warp_n = best_block_n // 2
    if best_block_m == 256 and best_block_n == 256:
        warp_m = best_block_m // 4
        warp_n = best_block_n // 4
    elif best_block_m == 64 and best_block_n >= 128:
        warp_m = 32
        warp_n = best_block_n // 2 if best_block_n < 128 else best_block_n // 4
    elif best_block_m == 32 and best_block_n >= 64:
        warp_m = 32
        warp_n = best_block_n // 4
    elif best_block_n == 32 and n <= 128 and best_block_m >=64:
        warp_m = best_block_m // 4
        warp_n = 32
    elif best_block_m == 128 or best_block_m == 256 and best_block_n >= 32:
        warp_m = 64
        warp_n = best_block_n // 2 if best_block_n <= 128 else best_block_n // 4
    elif best_block_m == 16:
        warp_m = 16
        warp_n = best_block_n // 4 if best_block_n <= 128 else best_block_n // 8
    elif best_block_n == 128 or best_block_n == 256:
        warp_m = best_block_m // 2 if best_block_m != 32 else best_block_m
        warp_n = best_block_n // 4

    return min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config

#pre-configured optimal tiling greater than or equal to 4096
CONFIG_TILE_GREATER_4096 = [
    [128, 128, 64, 64, 32, 64, 2],\
    [256, 128, 64, 64, 64, 64, 2],\
    [512, 128, 64, 64, 64, 64, 3],\
    [128, 256, 64, 32, 128, 64, 2]\
    ]
def generate_search_space_v2(lhs: Tuple[torch.Tensor],
                             rhs: Tuple[torch.Tensor],
                             out: torch.Tensor, num_candidate:int):
    m, k = lhs.shape
    n, _ = rhs.shape
    shape = [m, n, k]
    device_props = torch.cuda.get_device_properties(device='cuda')
    if not (all(a >= 4096 and a % 64 == 0 for a in shape)\
            and (lhs.dtype == torch.bfloat16 or lhs.dtype == torch.float16)\
            and ("ZW810E" in device_props.name or "ZW810" in device_props.name)):
       return []
    candidate_tile = MatmulHeuristicsTile(shape, 2, CONFIG_TILE_GREATER_4096)
    tile_list = candidate_tile.get_candidate_tile(num_candidate)
    return [tile[3:11] for tile in tile_list]

def get_gemm_best_configs_v2(shape, dtype, num_sms):
    candidate_tile = MatmulHeuristicsTile(shape, dtype, CONFIG_TILE_GREATER_4096)
    tile_list = candidate_tile.get_candidate_tile(1)
    _,_,k = shape
    tile_item = tile_list[0]
    [bm, bn, bk, wm, wn, _, stages, num_min_sms] = tile_item[3:11]
    best_smem_config = get_smem_config(stages, k, bm, bn, bk)
    return num_min_sms, bm, bn, bk, wm, wn, stages, best_smem_config

def gemm_bf16_bf16_bf16_nt(lhs: Tuple[torch.Tensor],
                         rhs: Tuple[torch.Tensor],
                         out: torch.Tensor,
                         configs = None) -> None:
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    # assert n % 64 == 0 and k % 128 == 0

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs.dtype == torch.bfloat16
    assert rhs.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    shape = [m, n, k]
    device_props = torch.cuda.get_device_properties(device='cuda')
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
        warp_k = block_k  # default: no K-split for explicit configs
        dense_s2_opt = False  # External configs: no adaptive re-evaluation
    elif all(a >= 4096 and a % 64 == 0 for a in shape)\
       and ("ZW810E" in device_props.name or "ZW810" in device_props.name):
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_gemm_best_configs_v2(shape, 2, num_sms)
        warp_k = block_k
        dense_s2_opt = False
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, 1, num_sms)
        warp_k = block_k  # default WarpOnK=1
        # Heuristic path: inject adaptive warp_k/dense_s2_opt (mirrors C++ bf16_gemm.hpp:664-668)
        if is_ppu1v5_device() and is_bf16_adaptive_enabled(m, n, k):
            dense_s2_opt = True
            # The tile is already an adaptive tile selected by the adaptive branch in get_best_configs.
            # Compute warp_k on this tile with get_warp_k, equivalent to get_adaptive_configs(...)[6].
            warp_k = get_warp_k(block_m, block_n, block_k, warp_m, warp_n, num_stages)
        else:
            dense_s2_opt = False

    kernel_type = 'Default'

    if is_ppu1v5_device():
        # PPU1.5 DenseGemm path: N and K are runtime args, not compile keys.
        args = (lhs, rhs, out, m, n, k, torch.cuda.current_stream(), num_sms, smem_config[0])
        runtime = jit_tuner.compile_and_tune(
            name='gemm_bf16_bf16_bf16_nt',
            keys={'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                  'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k,
                  'NUM_STAGES': num_stages,
                  'KERNEL_TYPE': kernel_type,
                  'DENSE_S2_OPT': 'true' if dense_s2_opt else 'false',
                  'IS_ALIGNED_N': 'true' if n % block_n == 0 else 'false'},
            space=(),
            includes=includes_bf16_densegemm,
            arg_defs=(('lhs', torch.bfloat16),
                      ('rhs', torch.bfloat16),
                      ('out', torch.bfloat16), ('m', int), ('n', int), ('k', int),
                      ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
            template=template_bf16_densegemm,
            jit_include_dir='actlize_v1.0.0',
            args=args
        )
    else:
        args = (lhs, rhs, out, m, torch.cuda.current_stream(), num_sms, smem_config[0])
        runtime = jit_tuner.compile_and_tune(
            name='gemm_bf16_bf16_bf16_nt',
            keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                  'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages},
            space=(),
            includes=includes_bf16_cutlass3,
            arg_defs=(('lhs', torch.bfloat16), ('rhs', torch.bfloat16),
                      ('out', torch.bfloat16), ('m', int),
                      ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
            template=template_bf16_cutlass3,
            jit_include_dir='actlize_v1.0.0',
            args=args
        )
    runtime(*args)
