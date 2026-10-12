import math
import os
import torch
from functools import lru_cache
from typing import Tuple
import re

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout, is_ppu1v5_device, get_sm_count, GemmType
from .gemm_int8_lut import get_best_configs_from_lut
from .densegemm_adaptive_select_strategy import get_adaptive_configs, is_int8_adaptive_enabled, get_warp_k

# C++ code templates
includes_int8_cutlass3 = ('<deep_gemm/impls/int8_gemm_cutlass3.cuh>', )
template_int8_cutlass3 = """
using namespace deep_gemm;

using ElementAB = {ElementAB};
using ElementAcc = {ElementAcc};
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto kNumGroups = 1;
constexpr auto kNumStages = {NUM_STAGES};

using gemm_t = Gemm<ElementAB, ElementAcc, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                    WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::DenseGemm,
                    false, KernelType::{KERNEL_TYPE}>;

gemm_t::run(out, nullptr, nullptr,
            m, 0, (ElementAB*)lhs, lhs_scales, (ElementAB*)rhs, rhs_scales,
            stream, num_sms, smem_size);
"""

# PPU1.5 DenseGemm-only cutlass3 path: standalone kernel that reads N/K at runtime.
# Do NOT reuse includes_densegemm/template_densegemm for grouped or MoE paths.
includes_densegemm = ('<deep_gemm/impls/int8_densegemm_cutlass3.cuh>', )
template_densegemm = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementAB = {ElementAB};
using ElementAcc = {ElementAcc};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto WARP_K = {WARP_K};
constexpr auto kNumStages = {NUM_STAGES};
constexpr bool kIsAlignedN = {IS_ALIGNED_N};

// Make a templated DenseGemm (no SHAPE_N/K in template - uses runtime args)
using gemm_t = DenseGemm<ElementAB, ElementAcc, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, kNumStages, {DENSE_S2_OPT}, KernelType::{KERNEL_TYPE}, kIsAlignedN>;

// Launch kernel
gemm_t::run(out, m, n, k, (ElementAB*)lhs, lhs_scales, (ElementAB*)rhs, rhs_scales,
            stream, num_sms, smem_size);
"""

def get_smem_config(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128, bpp: int = 1) -> Tuple[int, int, int]:
    # Try swizzle first, as it does not waste shared memory
    swizzle_mode = 128
    # block_n_padding = get_block_n_padding_for_smem_d(block_n) if swizzle_mode == 0 else 0
    block_n_padding = 0

    smem_d = block_m * (block_n + block_n_padding)
    smem_a_per_stage = block_m * block_k
    # smem_scales_a_per_stage = block_m * 4
    smem_b_per_stage = block_n * block_k
    # smem_scales_b = ceil_div(k, block_k) * 4
    # smem_barrier = num_stages * 8 * 2

    # smem_size = 0
    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)
    # smem_size += num_stages * smem_a_per_stage
    # smem_size += num_stages * smem_scales_a_per_stage
    # smem_size += num_stages * smem_b_per_stage
    # smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    # smem_size += smem_barrier

    # Swizzle and padding are not compatible
    assert int(swizzle_mode > 0) + int(block_n_padding > 0) <= 1

    return smem_size, swizzle_mode, block_n_padding

def get_smem_occ(block_m: int, block_n: int, block_k: int, num_stages: int) -> Tuple[int]:
    if block_m is None:
        return 0

    # use static suppose.
    ppu_capacity = 262144
    bpp = 1
    smem_d = block_m * block_n
    smem_a_per_stage = block_m * block_k
    smem_b_per_stage = block_n * block_k

    smem_size_d = smem_d * 2
    smem_size_a = num_stages * smem_a_per_stage * bpp
    smem_size_b = num_stages * smem_b_per_stage * bpp

    smem_size = max(smem_size_d, smem_size_a + smem_size_b)

    return 262144 // smem_size

@lru_cache(maxsize=None)
def get_best_configs_dense_ppu1v5(m: int, n: int, k: int, num_groups: int, num_sms: int) -> \
        Tuple[int, int, int, int, int, int, int, int, int, dict]:

    #FIXME: block m can add 16, and blockM/N could be 512, and 48, 96 blockM.
    block_ms = (256, 192, 128, 64, 32, 16)
    block_ns = (256, 128, 64, 32)

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)
    get_block_utils = lambda m, bm: (((m / bm) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0
    get_block_ai = lambda block_m, block_n: (block_m * block_n) / (block_m + block_n)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES:
        # for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        # for PPU1.5: the tile 256x256 is good for many compute bound case
        if (m >= 128 and k >= 2048) or m >= 256:
            block_ns_after_filter = filter(lambda bn: (bn != n and n >= 1), block_ns)
        else:
            block_ns_after_filter = \
                filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 1) and not (block_m == 16 and bn == 32)), block_ns)
        for block_n in block_ns_after_filter:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)
            num_occ, best_num_occ = get_smem_occ(block_m, block_n, 128, 2), get_smem_occ(best_block_m, best_block_n, 128, 2)

            # print(f"block_m:{block_m}, block_n:{block_n}, best_block_m:{best_block_m}, best_block_n:{best_block_n}")
            # print(f'num_occ:{num_occ}, best_num_occ:{best_num_occ}')
            # # print(f'num_waves:{num_waves / num_occ}, best_num_waves:{best_num_waves / best_num_occ}')
            # print(f'm_util:{get_block_utils(m, block_m)}, n_util:{get_block_utils(n, block_n)}, num_utils:{num_utils}')
            # print(f'best_m_util:{get_block_utils(m, best_block_m)}, best_n_util:{get_block_utils(n, best_block_n)}, best_num_utils:{best_num_utils}')

            if best_block_m is None or best_block_n is None:
                success = True
            elif (m <= 256 or n < 512):
                # if single group block is small, balance wave, utils and occ
                occ_wave = num_waves / num_occ
                best_occ_wave = best_num_waves / best_num_occ
                ai_util = get_block_ai(block_m, block_n)
                best_ai_util = get_block_ai(best_block_m, best_block_n)

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

    #small m hbm bound or latency bound, wave is not usful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m < 20 and n < 512) :
        best_block_m = 16
        best_block_n = 64

    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144

    block_k = 128
    if k <= 256:
        block_k = 64
    if k >= 4096 and ((best_block_m <= 32 and best_block_n <= 64) or (best_block_m == 64 and best_block_n == 128)):
        block_k = 256

    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (5, 4, 3, 2)))

    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4):
        stage_candidates = (3, 2)
    if (best_block_m >= 128 and best_block_n >= 128):
        stage_candidates = (4,)

    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1)
        # print(f"num_stages:{num_stages}, best_smem_config:{best_smem_config}")
        if best_smem_config[0] <= ppu_capacity:
            occ = ppu_capacity // best_smem_config[0]
            if k < 512 or (best_block_m > 64 and best_block_n >= 64) and occ >= best_occ:
                # compute block use higer occ rather than large stage
                best_num_stages = num_stages
                best_occ = occ
            else:
                best_num_stages = num_stages
                break

    # best_num_stages = 2
    assert best_smem_config is not None
    assert best_num_stages is not None

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)

    if is_ppu1v5_device():
        num_min_sms = num_sms
    else:
        num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)

    assert num_min_sms <= num_sms

    warp_m = best_block_m // 2
    warp_n = best_block_n // 2

    if best_block_m >= 128 and best_block_n == 256:
        warp_m = best_block_m // 4
        warp_n = best_block_n // 4
    elif best_block_m == 32 and best_block_n >= 64:
        warp_m = 32
        warp_n = best_block_n // 4
    elif best_block_n == 32 and n <= 128 and best_block_m >= 64:
        warp_m = best_block_m // 4
        warp_n = 32
    elif best_block_m == 128 or best_block_m == 256 or best_block_m == 192 and best_block_n >= 32:
        warp_m = best_block_m // 4
        warp_n = best_block_n // 2 if best_block_n != 32 else best_block_n
    elif best_block_m == 16:
        warp_m = 16
        best_block_n = 64 if n < 512 else best_block_n
        warp_n = best_block_n // 4 if best_block_n <= 128 else best_block_n // 8
    elif best_block_n == 128 or best_block_n == 256:
        warp_m = best_block_m // 2 if best_block_m != 32 else best_block_m
        warp_n = best_block_n // 4

    return min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config

@lru_cache(maxsize=None)
def get_best_configs_ppu1v5(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     gemm_type: GemmType = GemmType.DenseGemm, max_block_n: int = 256) -> \
        Tuple[int, int, int, int, int, int, int, int, int, dict]:
    if gemm_type == GemmType.DenseGemm or gemm_type == GemmType.BatchGemm:
       return get_best_configs_dense_ppu1v5(m, n, k, num_groups, num_sms)

    #FIXME: block m can add 16, and blockM/N could be 512, and 48, 96 blockM.
    if gemm_type != GemmType.GroupedContiguous:
        block_ms = (256, 128, 64, 32, 16) if k >= 384 else (128, 64, 32, 16)
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )

    # block_ns = (256, 128, 64, 32)
    assert max_block_n > 0 and (max_block_n & (max_block_n - 1)) == 0
    block_ns = tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 1, 4, -1))) if k >= 384 else tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 2, 4, -1)))

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    get_block_utils = lambda m, bm: (((m / bm) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0
    get_block_ai = lambda block_m, block_n: (block_m * block_n) / (block_m + block_n)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        # NOTES:
        # for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        # for PPU1.5: the tile 256x256 is good for many compute bound case
        if is_ppu1v5_device() and ((m >= 128 and k > 2048) or (m >= 256 and k >= 512)):
            block_ns_after_filter = filter(lambda bn: (bn != n and n >= 32) and not (block_m == 16 and bn <= 32), block_ns)
        else:
            block_ns_after_filter = \
                filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= 32) and not (block_m == 16 and bn <= 32)), block_ns)
        for block_n in block_ns_after_filter:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)
            num_occ, best_num_occ = get_smem_occ(block_m, block_n, 128, 2), get_smem_occ(best_block_m, best_block_n, 128, 2)

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
                ai_util = get_block_ai(block_m, block_n)
                best_ai_util = get_block_ai(best_block_m, best_block_n)

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

    if is_ppu1v5_device() and (m >= 96 and m < 128 and n > 2048 and k > 2048 and num_groups >= 8):
        best_block_m = 192
        best_block_n = 256

    if (best_block_m - 10 <= m <= best_block_m) and best_block_m == 16:
        best_block_m = best_block_m * 2

    # qwen3-next & deepseek gemm2 need to fix some issues
    if (best_block_m - 10 <= m <= best_block_m) and (best_block_m == 32 or best_block_m == 64) and k > 256:
        best_block_m = best_block_m * 2

    #small m hbm bound or latency bound, wave is not usful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m < 10 and n < 512) :
        best_block_m = 16
        best_block_n = 64

    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144

    block_k = 128
    if k == 128:
        block_k = 64
    if k >= 4096 and (best_block_m <= 32 and best_block_n <= 64):
        block_k = 256

    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (8, 7, 6, 5, 4, 3, 2)))

    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16 or best_block_m == 32:
        stage_candidates = (3, 2)
    if best_block_m > 128 and best_block_n == 256:
        stage_candidates = (4,)

    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1)
        # print(f"num_stages:{num_stages}, best_smem_config:{best_smem_config}")
        if best_smem_config[0] <= ppu_capacity:
            occ = ppu_capacity // best_smem_config[0]
            if k < 512 or (best_block_m > 32 and best_block_n >= 64) and occ >= best_occ:
                # compute block use higer occ rather than large stage
                best_num_stages = num_stages
                best_occ = occ
            else:
                best_num_stages = num_stages
                break

    assert best_smem_config is not None
    assert best_num_stages is not None

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)

    if is_ppu1v5_device():
        num_min_sms = num_sms
    else:
        num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)

    assert num_min_sms <= num_sms

    warp_m = best_block_m // 2
    warp_n = best_block_n // 2

    if best_block_m > 128 and best_block_n == 256:
        warp_m = best_block_m // 4
        warp_n = best_block_n // 4
    elif best_block_m == 64 and best_block_n >= 128:
        warp_m = 64 if k < 256 else 32
        warp_n = 64
    elif best_block_m == 32 and best_block_n >= 64:
        warp_m = 32
        warp_n = best_block_n // 2 if best_block_n <= 128 else best_block_n // 4
    elif best_block_n == 32 and n <= 128 and best_block_m >=64:
        warp_m = best_block_m // 4
        warp_n = 32
    elif best_block_m == 128 or best_block_m == 256 and best_block_n >= 32:
        warp_m = 64
        warp_n = best_block_n // 2 if best_block_n <= 128 else best_block_n // 4
    elif best_block_m == 16:
        warp_m = 16
        best_block_n = 64 if n < 512 else best_block_n
        warp_n = best_block_n // 2 if best_block_n < 128 else best_block_n // 4
    elif best_block_n == 128 or best_block_n == 256:
        warp_m = best_block_m // 2 if best_block_m != 32 else best_block_m
        warp_n = 64

    # (best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages) = (16, 64, 256, 16, 16, 4)
    # print(best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages)

    return min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config

# ---- adaptive tile selector for int8 DenseGemm ----
@lru_cache(maxsize=None)
def get_adaptive_configs_int8(m: int, n: int, k: int, num_sms: int):
    """int8 adaptive selector using the shared DenseGemm tile strategy.

    Calls the bf16 strategy (densegemm_adaptive_select_strategy) and doubles
    block_k and warp_k — int8 has bpp=1 while bf16 has bpp=2, so doubled BK
    fits the same SMEM budget. SMEM config is recomputed with int8's formula.
    """
    ns, bm, bn, bk, wm, wn, wk, s = get_adaptive_configs(m, n, k, num_sms)
    # int8: double BK and WK (bpp=1 halves per-stage SMEM vs bf16 bpp=2)
    bk, wk = bk * 2, wk * 2
    smem_config = get_smem_config(s, k, bm, bn, bk, bpp=1)
    return ns, bm, bn, bk, wm, wn, s, smem_config


@lru_cache(maxsize=None)
def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     gemm_type: GemmType = GemmType.DenseGemm,
                     max_block_n: int = 256) -> \
        Tuple[int, int, int, int, int, int, int, int, int, dict]:
    lut_result = get_best_configs_from_lut(m, n, k)
    if num_groups == 1 and lut_result:
        best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages = lut_result
        best_smem_config = get_smem_config(best_stages, k, best_block_m, best_block_n, best_block_k, 1)
        return num_sms, best_block_m, best_block_n, best_block_k, best_warp_m, best_warp_n, best_stages, best_smem_config
    # Adaptive tile selector for int8 DenseGemm on PPU1.5
    if is_int8_adaptive_enabled(m, n, k) and is_ppu1v5_device() and gemm_type == GemmType.DenseGemm:
        return get_adaptive_configs_int8(m, n, k, num_sms)
    if is_ppu1v5_device():
        return get_best_configs_ppu1v5(m, n, k, num_groups, num_sms, gemm_type, max_block_n)
    #FIXME: block m can add 16, and blockM/N could be 512, and 48, 96 blockM.
    if gemm_type != GemmType.GroupedContiguous:
        block_ms = (256, 128, 64, 32, 16) if k > 384 else (64, 32, 16)
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )

    # block_ns = (256, 128, 64, 32)
    assert max_block_n > 0 and (max_block_n & (max_block_n - 1)) == 0
    block_ns = tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 1, 4, -1))) if k > 384 else tuple(map(lambda x: 2**x, range(max_block_n.bit_length() - 2, 4, -1)))

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    get_block_utils = lambda m, bm: (((m / bm) / ((m + bm -1) // bm)) if m % bm != 0 else 1.0) if bm else 0
    get_block_ai = lambda block_m, block_n: (block_m * block_n) / (block_m + block_n)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    min_n_threshold = 1 if gemm_type == GemmType.DenseGemm else 32
    for block_m in block_ms:
        # NOTES:
        # for PPU1.0: the block sizes can not be too large, so at least one dim less than 128
        # for PPU1.5: the tile 256x256 is good for many compute bound case
        if is_ppu1v5_device() and ((m >= 128 and k > 2048) or m >= 256):
            block_ns_after_filter = filter(lambda bn: (bn != n and n >= min_n_threshold) and not (block_m == 16 and bn <= 32), block_ns)
        else:
            block_ns_after_filter = \
                filter(lambda bn: ((block_m <= 128 or bn <= 128) and (bn != n and n >= min_n_threshold) and not (block_m == 16 and bn <= 32)), block_ns)
        for block_n in block_ns_after_filter:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            num_utils = get_block_utils(m, block_m) * get_block_utils(n, block_n)
            best_num_utils = get_block_utils(m, best_block_m) * get_block_utils(n, best_block_n)
            num_occ, best_num_occ = get_smem_occ(block_m, block_n, 128, 2), get_smem_occ(best_block_m, best_block_n, 128, 2)

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
                ai_util = get_block_ai(block_m, block_n)
                best_ai_util = get_block_ai(best_block_m, best_block_n)

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

    if is_ppu1v5_device() and (m >= 96 and m < 128 and n > 2048 and k > 2048 and num_groups >= 8):
        best_block_m = 192
        best_block_n = 256

    #small m hbm bound or latency bound, wave is not usful, for better occ for 810e hbm bound, use smallest blockN for m16
    if (m < 6 and n < 512) :
        best_block_m = 16
        best_block_n = 64

    if (best_block_m - 10 <= m <= best_block_m) and best_block_m == 16 and k > 384:
        best_block_m = best_block_m * 2

    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_config, ppu_capacity = None, None, 262144
    if not is_ppu1v5_device() and 64 < m < 128:
        best_block_m = 64

    block_k = 128
    if k <= 256:
        block_k = 64
    if k >= 4096 and (best_block_m <= 32 and best_block_n <= 128):
        block_k = 256

    stage_candidates = tuple(filter(lambda s: s <= k // block_k, (8, 7, 6, 5, 4, 3, 2)))

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)

    num_min_sms = num_sms
    # num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)

    assert num_min_sms <= num_sms

    warp_m = best_block_m // 2
    warp_n = best_block_n // 2

    if best_block_m > 128 and best_block_n == 256:
        warp_m = best_block_m // 4
        warp_n = best_block_n // 4
    elif best_block_m == 32 and best_block_n >= 64:
        warp_m = best_block_m // 2 if k > 256 else 32
        warp_n = best_block_n // 4 if best_block_n <= 128 else best_block_n // 8
    elif best_block_n == 32 and n <= 128 and best_block_m >=64:
        warp_m = best_block_m // 4
        warp_n = 32
    elif best_block_m == 128 or best_block_m == 256 and best_block_n >= 32:
        warp_m = 64
        warp_n = best_block_n // 2 if best_block_n <= 128 else best_block_n // 4
    elif best_block_m == 16:
        warp_m = 16
        best_block_n = 64 if n <= 512 else best_block_n
        warp_n = best_block_n // 4 if best_block_n <= 128 else best_block_n // 8
    elif best_block_n == 128 or best_block_n == 256:
        warp_m = best_block_m // 2 if best_block_m != 32 else best_block_m
        warp_n = best_block_n // 4

    # (best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages) = (16, 64, 256, 16, 16, 4)
    # print(best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages)

    if not stage_candidates or (128 % best_block_n != 0 and 128 // math.gcd(128, best_block_n) <= 4) or best_block_m == 16 or best_block_m == 32:
        stage_candidates = (3, 2)
    if best_block_m == 64 and best_block_n == 128:
        stage_candidates = (3, 2)
    if best_block_m > 128 and best_block_n == 256:
        stage_candidates = (4,)

    best_occ = 0
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(num_stages, k, best_block_m, best_block_n, block_k, 1)
        # print(f"num_stages:{num_stages}, best_smem_config:{best_smem_config}")
        if best_smem_config[0] <= ppu_capacity:
            occ = ppu_capacity // best_smem_config[0]
            if k < 512 or (best_block_m > 64 and best_block_n >= 64) and occ >= best_occ:
                # compute block use higer occ rather than large stage
                best_num_stages = num_stages
                best_occ = occ
            else:
                best_num_stages = num_stages
                break

    # best_num_stages = 2
    assert best_smem_config is not None
    assert best_num_stages is not None

    return min(num_min_sms, num_sms), best_block_m, best_block_n, block_k, warp_m, warp_n, best_num_stages, best_smem_config

# support both int8 and fp8 with per-tensor per-channel scales
def gemm_a8w8_per_channel_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                             rhs: Tuple[torch.Tensor, torch.Tensor],
                             out: torch.Tensor, configs = None) -> None:
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs.dtype == torch.int8 or lhs.dtype == torch.float8_e4m3fn
    assert rhs.dtype == torch.int8 or rhs.dtype == torch.float8_e4m3fn
    assert lhs.dtype == rhs.dtype
    assert lhs_scales.shape[0] == m and lhs_scales.dtype == torch.float32
    assert rhs_scales.shape[0] == n and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return
    kernel_type = 'Default'
    # if k < 4096:
    #     kernel_type = 'OverlapPrologue'

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
        warp_k = block_k  # default: no K-split for explicit configs
        dense_s2_opt = False  # External configs: no adaptive re-evaluation
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, 1, num_sms)
        warp_k = block_k
        # Heuristic path: inject adaptive warp_k/dense_s2_opt (mirrors C++ int8_gemm)
        if is_ppu1v5_device() and is_int8_adaptive_enabled(m, n, k):
            dense_s2_opt = True
            # The tile is already an adaptive tile (get_best_configs hit the adaptive branch).
            # get_warp_k is calibrated for BF16-scale block_k; INT8 doubles block_k (bpp=1),
            # so pass the original scale (block_k // 2) and double the result.
            warp_k = get_warp_k(block_m, block_n, block_k // 2, warp_m, warp_n, num_stages) * 2
        else:
            dense_s2_opt = False
    ElementAB = "cutlass::float_e4m3_t" if lhs.dtype == torch.float8_e4m3fn else "int8_t"
    ElementAcc = "float" if lhs.dtype == torch.float8_e4m3fn else "int32_t"

    if is_ppu1v5_device():
        # PPU1.5 DenseGemm path: N and K are runtime args, not compile keys.
        args = (lhs, lhs_scales, rhs, rhs_scales, out,
                m, n, k, torch.cuda.current_stream(), num_sms, smem_config[0])
        runtime = jit_tuner.compile_and_tune(
            name='gemm_' + ElementAB + '_bf16_nt',
            keys={'ElementAB': ElementAB, 'ElementAcc': ElementAcc,
                  'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                  'WARP_M': warp_m, 'WARP_N': warp_n, 'WARP_K': warp_k,
                  'NUM_STAGES': num_stages,
                  'KERNEL_TYPE': kernel_type,
                  'DENSE_S2_OPT': 'true' if dense_s2_opt else 'false',
                  'IS_ALIGNED_N': 'true' if n % block_n == 0 else 'false'},
            space=(),
            includes=includes_densegemm,
            arg_defs=(('lhs', lhs.dtype), ('lhs_scales', torch.float),
                      ('rhs', rhs.dtype), ('rhs_scales', torch.float),
                      ('out', torch.bfloat16), ('m', int), ('n', int), ('k', int),
                      ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
            template=template_densegemm,
            jit_include_dir='actlize_v1.0.0',
            args=args
        )
    else:
        args = (lhs, lhs_scales, rhs, rhs_scales, out,
                m, torch.cuda.current_stream(), num_sms, smem_config[0])
        runtime = jit_tuner.compile_and_tune(
            name='gemm_' + ElementAB + '_bf16_nt',
            keys={'ElementAB': ElementAB, 'ElementAcc': ElementAcc,
                  'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
                  'WARP_M': warp_m, 'WARP_N': warp_n, 'NUM_STAGES': num_stages,
                  'KERNEL_TYPE': kernel_type},
            space=(),
            includes=includes_int8_cutlass3,
            arg_defs=(('lhs', lhs.dtype), ('lhs_scales', torch.float),
                      ('rhs', rhs.dtype), ('rhs_scales', torch.float),
                      ('out', torch.bfloat16), ('m', int),
                      ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
            template=template_int8_cutlass3,
            jit_include_dir='actlize_v1.0.0',
            args=args
        )
    runtime(*args)

def gemm_int8_int8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                         rhs: Tuple[torch.Tensor, torch.Tensor],
                         out: torch.Tensor, configs = None) -> None:
    gemm_a8w8_per_channel_nt(lhs, rhs, out, configs)
