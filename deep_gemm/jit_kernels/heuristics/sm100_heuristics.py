import functools
import torch
from typing import Tuple

from .common import (
    MulticastConfig, SharedMemoryConfig,
    is_multicast_legal, get_swizzle_mode
)
from ...utils.math import align, ceil_div
from ...utils.layout import (
    GemmType, MajorTypeAB, MajorTypeCD,
    get_element_size, get_m_alignment_for_contiguous_layout
)


def get_sf_aligned_block_sizes(block_m: int, block_n: int, ab_dtype: torch.dtype):
    num_utccp_aligned_elems = 128
    assert block_m % num_utccp_aligned_elems == 0
    return {
        torch.bfloat16: (0, 0),
        torch.float8_e4m3fn: (align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)),
    }[ab_dtype]


def is_tmem_size_legal(block_m: int, block_n: int, ab_dtype: torch.float):
    # M waves or epilogue stages (* 2), SFA and SFB
    sf_block_m, sf_block_n = get_sf_aligned_block_sizes(block_m, block_n, ab_dtype)
    return ((2 * block_n) + (sf_block_m // 32) + (sf_block_n // 32)) <= 512


def get_smem_config(block_m: int, block_n: int, block_k: int,
                    major_a: MajorTypeAB, major_b: MajorTypeAB, major_d: MajorTypeCD,
                    ab_dtype: torch.dtype, cd_dtype: torch.dtype,
                    num_stages: int, multicast_config: MulticastConfig) -> SharedMemoryConfig:
    assert major_d == MajorTypeCD.NMajor

    ab_elem_size = get_element_size(ab_dtype)
    cd_elem_size = get_element_size(cd_dtype)

    load_block_m = multicast_config.get_ab_load_block_m(block_m)
    load_block_n = multicast_config.get_ab_load_block_n(block_n)
    swizzle_a_mode = get_swizzle_mode(block_k if major_a == MajorTypeAB.KMajor else load_block_m, ab_elem_size)
    swizzle_b_mode = get_swizzle_mode(block_k if major_b == MajorTypeAB.KMajor else load_block_n, ab_elem_size)
    swizzle_cd_mode = get_swizzle_mode(block_n if major_d == MajorTypeCD.NMajor else block_m, cd_elem_size)

    # 2 stages of STSM and TMA store
    # TODO: consider other layouts
    layout_ad_m = 128
    smem_d = min(block_m, layout_ad_m) * swizzle_cd_mode * 2

    # A/B shared memory
    smem_a_per_stage = load_block_m * block_k * ab_elem_size
    smem_b_per_stage = load_block_n * block_k * ab_elem_size

    # SF shared memory must be aligned to UTCCP
    # Each stage must prefetch next 4 stages' SF (including the current)
    sf_block_m, sf_block_n = get_sf_aligned_block_sizes(block_m, block_n, ab_dtype)
    smem_scales_a_per_stage = sf_block_m * 4
    smem_scales_b_per_stage = sf_block_n * 4

    # TODO: remove SF barriers for BF16 GEMMs
    # TMA full/empty barriers, with-SF full barriers, tensor memory full/empty barriers, accumulation full barrier
    # NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
    # NOTES: cases without accumulation will not use the accumulation full barrier
    smem_barrier = num_stages * 8 * 3 + 2 * 8 * 2 + 8
    smem_tmem_ptr = 4

    # Sum them up
    smem_size = 0
    smem_size += smem_d
    smem_size += num_stages * smem_a_per_stage
    smem_size += num_stages * smem_b_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += num_stages * smem_scales_b_per_stage
    smem_size += smem_barrier
    smem_size += smem_tmem_ptr

    return SharedMemoryConfig(smem_size, swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode)


@functools.lru_cache(maxsize=None)
def get_best_configs(gemm_type: GemmType,
                     m: int, n: int, k: int, num_groups: int,
                     major_a: MajorTypeAB, major_b: MajorTypeAB, major_d: MajorTypeCD,
                     ab_dtype: torch.dtype, cd_dtype: torch.dtype,
                     num_sms: int) -> \
        Tuple[int, int, int, int, int, MulticastConfig, SharedMemoryConfig]:
    assert ab_dtype == torch.float8_e4m3fn
    assert cd_dtype in (torch.bfloat16, torch.float)

    # `BLOCK_M` and `BLOCK_N` are selected according to MMA instructions
    if gemm_type == GemmType.GroupedContiguous:
        block_ms = (get_m_alignment_for_contiguous_layout(), )
    else:
        block_ms = (128, ) if major_b == MajorTypeAB.KMajor else (128, 256)
    # NOTES: some `% 32 == 16` cases are not compatible with 2-CTA TMA swizzling
    block_ns = tuple(range(16, 257, 16)) if major_b == MajorTypeAB.KMajor else tuple(range(32, 257, 32))

    # `BLOCK_K` is selected in a fixed manner
    block_k = 128 // get_element_size(ab_dtype)

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    # Decide block sizes by waves
    # TODO: move block size search into `common.py`
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
                success = util > best_util
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, larger `block_n` is better
                    success |= block_m != best_block_m and block_n > best_block_n
            success &= is_tmem_size_legal(block_m, block_n, ab_dtype)
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)
    assert best_block_m is not None and best_block_n is not None

    # Decide the number of TMA multicasts and whether broadcast on A
    best_multicast_config = MulticastConfig(1, True)

    # Try to multicast on the larger block side first
    is_legal = {
        # TODO: support other `tcgen05` layouts
        'A': False,
        'B': is_multicast_legal(m, best_block_m, 2, num_sms, True) and gemm_type == GemmType.Normal,
    }
    for i in ('A', 'B') if best_block_m > best_block_n else ('B', 'A'):
        if m >= 512 and is_legal[i]:
            best_multicast_config = MulticastConfig(2, i == 'A')
            break

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    # TODO: move stage search into `common.py`
    best_num_stages, best_smem_config, sm100_capacity = None, None, 232448
    stage_candidates = tuple(filter(lambda s: s <= max(k // 128, 1), (8, 7, 6, 5, 4, 3, 2, 1)))
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(best_block_m, best_block_n, block_k,
                                           major_a, major_b, major_d,
                                           ab_dtype, cd_dtype,
                                           num_stages, best_multicast_config)
        if best_smem_config.smem_size <= sm100_capacity:
            best_num_stages = num_stages
            break
    assert best_smem_config is not None
    assert best_num_stages is not None

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    # TODO: move min SM fix into `common.py`
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)
    num_min_sms = ceil_div(num_min_sms, best_multicast_config.num_multicast) * best_multicast_config.num_multicast
    assert num_min_sms <= num_sms

    return num_min_sms, best_block_m, best_block_n, block_k, best_num_stages, best_multicast_config, best_smem_config
