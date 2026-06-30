"""Hopper FP8xFP4 MegaMoE correctness and benchmark harness.

Mirrors :file:`test_mega_moe_hopper.py` but exercises the FP4-weight path:

* Activations: FP8 (E4M3) with per-128 K float SFA — same as the SM90 FP8 path.
* Weights:     packed FP4 (E2M1) with per-32 K UE8M0 SFB. Each storage byte
               holds 2 nibbles (low nibble = even K, high nibble = odd K).
* Kernel:      ``sm90_fp8_fp4_mega_moe_impl`` (decode-to-SMEM SS-mode WGMMA).
               The per-32 SFB is folded into the FP4 -> E4M3 dequant through
               a constant-memory UE8M0 lookup table.

The reference runs in FP32 by dequantizing the same packed FP4 weights and
their UE8M0 SFs, so any numerical disagreement should be at WGMMA accumulator
precision plus the exact-FP32-vs-pow2-promote difference (small).

Default mode runs the layered correctness suite. ``--bench`` switches to the
FP4 fused-vs-FP8 low-latency benchmark that used to live in a separate script.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ----------------------------------------------------------------------------
# CPU-only SM90 FP4 heuristic equivalence guard
# ----------------------------------------------------------------------------

GENERIC_FALLBACK = "generic"


def legacy_wave(
    num_experts_per_rank: int,
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
):
    small_block_n = block_m == 64 and block_n == 128
    e = expected_tokens_per_expert
    n = num_experts_per_rank
    if small_block_n and intermediate_hidden <= 2048 and 0.75 <= e < 1.0 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden <= 2048 and 1.5 <= e < 2.0 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden <= 2048 and 3.0 <= e < 6.0 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden <= 2048 and 6.0 <= e < 12.0 and n % 32 == 0:
        return 32
    if small_block_n and intermediate_hidden <= 2048 and 6.0 <= e < 12.0 and n % 8 == 0:
        return 8
    if small_block_n and intermediate_hidden <= 2048 and 24.0 <= e < 32.0 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden <= 2048 and 12.0 <= e < 24.0 and n % 32 == 0:
        return 32
    if small_block_n and intermediate_hidden <= 2048 and 12.0 <= e < 24.0 and n % 8 == 0:
        return 8
    if small_block_n and intermediate_hidden <= 2048 and 32.0 <= e < 64.0 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden >= 3072 and 0.0 < e < 0.25 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden >= 3072 and 0.25 <= e < 0.375 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden >= 3072 and 0.375 <= e < 0.75 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden >= 3072 and 0.25 <= e < 1.0 and n % 24 == 0:
        return 24
    if small_block_n and intermediate_hidden >= 3072 and 1.0 <= e < 1.5 and n > 0:
        return n
    if small_block_n and intermediate_hidden >= 3072 and 1.5 <= e < 3.0 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden >= 3072 and 3.0 <= e < 6.0 and n % 8 == 0:
        return 8
    if small_block_n and intermediate_hidden >= 3072 and 6.0 <= e < 12.0 and n % 16 == 0:
        return 16
    if small_block_n and intermediate_hidden >= 3072 and 6.0 <= e < 12.0 and n % 8 == 0:
        return 8
    if small_block_n and intermediate_hidden >= 3072 and 12.0 <= e < 24.0 and n % 24 == 0:
        return 24
    if small_block_n and intermediate_hidden >= 3072 and 12.0 <= e < 24.0 and n % 8 == 0:
        return 8
    if small_block_n and intermediate_hidden >= 3072 and 24.0 <= e < 64.0 and n % 8 == 0:
        return 8
    if e < 1.0 or e > 4.0:
        return n
    return GENERIC_FALLBACK


FLASH_RULES = (
    (0.75, 1.0, True, 16, 16),
    (1.5, 2.0, True, 16, 16),
    (3.0, 6.0, True, 16, 16),
    (6.0, 12.0, True, 32, 32),
    (6.0, 12.0, True, 8, 8),
    (24.0, 32.0, True, 16, 16),
    (12.0, 24.0, True, 32, 32),
    (12.0, 24.0, True, 8, 8),
    (32.0, 64.0, True, 16, 16),
)

PRO_RULES = (
    (0.0, 0.25, False, 16, 16),
    (0.25, 0.375, True, 16, 16),
    (0.375, 0.75, True, 16, 16),
    (0.25, 1.0, True, 24, 24),
    (1.0, 1.5, True, 1, None),
    (1.5, 3.0, True, 16, 16),
    (3.0, 6.0, True, 8, 8),
    (6.0, 12.0, True, 16, 16),
    (6.0, 12.0, True, 8, 8),
    (12.0, 24.0, True, 24, 24),
    (12.0, 24.0, True, 8, 8),
    (24.0, 64.0, True, 8, 8),
)


def apply_rules(rules, num_experts_per_rank: int, expected_tokens_per_expert: float):
    for min_e, max_e, include_min, divisor, wave in rules:
        lower_ok = expected_tokens_per_expert >= min_e if include_min else expected_tokens_per_expert > min_e
        if not lower_ok or expected_tokens_per_expert >= max_e:
            continue
        if wave is None:
            if num_experts_per_rank <= 0:
                continue
            return num_experts_per_rank
        if num_experts_per_rank % divisor == 0:
            return wave
    return None


def table_wave(
    num_experts_per_rank: int,
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
):
    small_block_n = block_m == 64 and block_n == 128
    if small_block_n and intermediate_hidden <= 2048:
        selected = apply_rules(FLASH_RULES, num_experts_per_rank, expected_tokens_per_expert)
        if selected is not None:
            return selected
    if small_block_n and intermediate_hidden >= 3072:
        selected = apply_rules(PRO_RULES, num_experts_per_rank, expected_tokens_per_expert)
        if selected is not None:
            return selected
    if expected_tokens_per_expert < 1.0 or expected_tokens_per_expert > 4.0:
        return num_experts_per_rank
    return GENERIC_FALLBACK


def legacy_threads(
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_epilogue_threads: int,
):
    small_block_n = block_m == 64 and block_n == 128
    e = expected_tokens_per_expert
    decode_heavy_small_batch = small_block_n and 0.0 < e <= 24.0
    pro_large_decode_assist_batch = small_block_n and intermediate_hidden >= 3072 and 24.0 <= e < 64.0
    pro_split_n_decode_threads = small_block_n and intermediate_hidden >= 3072 and 0.0 < e < 64.0
    flash_split_n_decode_threads = small_block_n and intermediate_hidden <= 2048 and 0.0 < e < 64.0
    two_wg_decode_offload = (
        block_m == 128 and block_n == 128 and num_epilogue_threads == 256 and e >= 64.0
    )
    dispatch = (
        64
        if decode_heavy_small_batch
        or flash_split_n_decode_threads
        or pro_large_decode_assist_batch
        or two_wg_decode_offload
        else 128
    )
    non_epilogue = (
        320
        if pro_split_n_decode_threads or flash_split_n_decode_threads
        else (
            192
            if decode_heavy_small_batch or pro_large_decode_assist_batch or two_wg_decode_offload
            else 128
        )
    )
    return dispatch, non_epilogue


def legacy_epilogue_threads(
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_epilogue_threads: int,
):
    e = expected_tokens_per_expert
    epilogue_warpgroups = num_epilogue_threads // 128
    split_n_eligible = block_m == 64 and block_n % 128 == 0
    split_n_band = 32.0 <= e < 64.0
    pro_split_n_band = intermediate_hidden >= 3072 and 0.0 < e < 64.0
    flash_split_n_band = intermediate_hidden <= 2048 and 0.0 < e < 64.0
    small_split_n_band = flash_split_n_band or pro_split_n_band
    default_split_n = (
        split_n_eligible
        and (split_n_band or small_split_n_band)
        and (intermediate_hidden <= 2048 or intermediate_hidden >= 3072)
    )
    if split_n_eligible and default_split_n:
        epilogue_warpgroups = 2
    return epilogue_warpgroups * 128


def table_epilogue_threads(
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_epilogue_threads: int,
):
    e = expected_tokens_per_expert
    epilogue_warpgroups = num_epilogue_threads // 128
    split_n_eligible = block_m == 64 and block_n % 128 == 0
    split_n_shape_band = (
        (intermediate_hidden <= 2048 or intermediate_hidden >= 3072) and 0.0 < e < 64.0
    )
    if split_n_eligible and split_n_shape_band:
        epilogue_warpgroups = 2
    return epilogue_warpgroups * 128


def table_threads(
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_epilogue_threads: int,
):
    small_block_n_kernel = block_m == 64 and block_n == 128
    e = expected_tokens_per_expert
    split_n_shape_band = (
        (intermediate_hidden <= 2048 or intermediate_hidden >= 3072) and 0.0 < e < 64.0
    )
    split_n_decode_thread_kernel_band = small_block_n_kernel and split_n_shape_band
    two_wg_decode_offload_kernel_band = (
        block_m == 128 and block_n == 128 and num_epilogue_threads == 256 and e >= 64.0
    )
    decode_assist_thread_kernel_band = two_wg_decode_offload_kernel_band or (
        small_block_n_kernel and 0.0 < e <= 24.0
    )
    dispatch = 64 if split_n_decode_thread_kernel_band or decode_assist_thread_kernel_band else 128
    non_epilogue = (
        320
        if split_n_decode_thread_kernel_band
        else (192 if decode_assist_thread_kernel_band else 128)
    )
    return dispatch, non_epilogue


def legacy_stage_cap(
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
):
    small_block_n = block_m == 64 and block_n == 128
    e = expected_tokens_per_expert
    stage4_decode_band = small_block_n and intermediate_hidden <= 2048 and 6.0 <= e < 12.0
    flash_lookahead_stage4 = small_block_n and intermediate_hidden <= 2048 and 3.0 < e < 6.0
    pro_ultra_small_stage5 = small_block_n and intermediate_hidden >= 3072 and 0.0 < e < 0.25
    pro_half_token_stage5 = small_block_n and intermediate_hidden >= 3072 and 0.375 <= e < 0.75
    pro_single_token_stage5 = small_block_n and intermediate_hidden >= 3072 and 1.0 <= e < 1.5
    pro_two_token_stage5 = small_block_n and intermediate_hidden >= 3072 and 1.5 <= e < 3.0
    pro_large_batch_stage5 = small_block_n and intermediate_hidden >= 3072 and 24.0 <= e < 64.0
    stage6_decode_band = small_block_n and (
        (0.375 <= e < 0.75)
        or (intermediate_hidden <= 2048 and 3.0 < e < 6.0)
        or (1.5 <= e < 3.0 and not pro_two_token_stage5)
    )
    stage5_decode_heavy_batch = small_block_n and 1.5 <= e <= 24.0
    if (
        stage4_decode_band
        or flash_lookahead_stage4
    ):
        return 4
    if (
        pro_ultra_small_stage5
        or pro_half_token_stage5
        or pro_single_token_stage5
        or pro_two_token_stage5
        or pro_large_batch_stage5
    ):
        return 5
    if stage6_decode_band:
        return 6
    if stage5_decode_heavy_batch:
        return 5
    return 0


STAGE_SHAPE_ANY = "any"
STAGE_SHAPE_FLASH = "flash"
STAGE_SHAPE_PRO = "pro"
STAGE_SHAPE_NOT_PRO = "not_pro"

STAGE_RULES = (
    (6.0, 12.0, True, False, STAGE_SHAPE_FLASH, 4),
    (3.0, 6.0, False, False, STAGE_SHAPE_FLASH, 4),
    (0.0, 0.25, False, False, STAGE_SHAPE_PRO, 5),
    (0.375, 0.75, True, False, STAGE_SHAPE_PRO, 5),
    (1.5, 3.0, True, False, STAGE_SHAPE_PRO, 5),
    (1.0, 1.5, True, False, STAGE_SHAPE_PRO, 5),
    (24.0, 64.0, True, False, STAGE_SHAPE_PRO, 5),
    (0.375, 0.75, True, False, STAGE_SHAPE_ANY, 6),
    (3.0, 6.0, False, False, STAGE_SHAPE_FLASH, 6),
    (1.5, 3.0, True, False, STAGE_SHAPE_NOT_PRO, 6),
    (1.5, 24.0, True, True, STAGE_SHAPE_ANY, 5),
)


def stage_shape_matches(shape, intermediate_hidden: int):
    flash_shape = intermediate_hidden <= 2048
    pro_shape = intermediate_hidden >= 3072
    if shape == STAGE_SHAPE_ANY:
        return True
    if shape == STAGE_SHAPE_FLASH:
        return flash_shape
    if shape == STAGE_SHAPE_PRO:
        return pro_shape
    if shape == STAGE_SHAPE_NOT_PRO:
        return not pro_shape
    raise AssertionError(f"unknown stage shape {shape}")


def table_stage_cap(
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
):
    if not (block_m == 64 and block_n == 128):
        return 0
    e = expected_tokens_per_expert
    for min_e, max_e, include_min, include_max, shape, cap in STAGE_RULES:
        lower_ok = e >= min_e if include_min else e > min_e
        upper_ok = e <= max_e if include_max else e < max_e
        if lower_ok and upper_ok and stage_shape_matches(shape, intermediate_hidden):
            return cap
    return 0


def legacy_config(
    num_experts_per_rank: int,
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_epilogue_threads: int,
):
    epilogue_threads = legacy_epilogue_threads(
        expected_tokens_per_expert, intermediate_hidden, block_m, block_n, num_epilogue_threads
    )
    dispatch_threads, non_epilogue_threads = legacy_threads(
        expected_tokens_per_expert, intermediate_hidden, block_m, block_n, epilogue_threads
    )
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": 128,
        "num_experts_per_wave": legacy_wave(
            num_experts_per_rank, expected_tokens_per_expert, intermediate_hidden, block_m, block_n
        ),
        "num_dispatch_threads": dispatch_threads,
        "num_non_epilogue_threads": non_epilogue_threads,
        "num_epilogue_threads": epilogue_threads,
        "default_num_stages_cap": legacy_stage_cap(
            expected_tokens_per_expert, intermediate_hidden, block_m, block_n
        ),
    }


def table_config(
    num_experts_per_rank: int,
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_epilogue_threads: int,
):
    epilogue_threads = table_epilogue_threads(
        expected_tokens_per_expert, intermediate_hidden, block_m, block_n, num_epilogue_threads
    )
    dispatch_threads, non_epilogue_threads = table_threads(
        expected_tokens_per_expert, intermediate_hidden, block_m, block_n, epilogue_threads
    )
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": 128,
        "num_experts_per_wave": table_wave(
            num_experts_per_rank, expected_tokens_per_expert, intermediate_hidden, block_m, block_n
        ),
        "num_dispatch_threads": dispatch_threads,
        "num_non_epilogue_threads": non_epilogue_threads,
        "num_epilogue_threads": epilogue_threads,
        "default_num_stages_cap": table_stage_cap(
            expected_tokens_per_expert, intermediate_hidden, block_m, block_n
        ),
    }


def legacy_api_features(
    num_experts_per_rank: int,
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
):
    e = expected_tokens_per_expert
    fp4_decode_lookahead_band = 3.0 <= e <= 6.0
    fp4_bigband_lookahead_band = 12.0 <= e <= 24.0
    fp4_pro_bigband_lookahead_band = intermediate_hidden >= 3072 and fp4_bigband_lookahead_band
    fp4_b4_skip_decode_band = 0.5 <= e < 1.0
    fp4_pro_single_token_per_expert_band = (
        intermediate_hidden >= 3072 and 1.0 <= e < 1.5 and num_experts_per_rank % 8 == 0
    )
    fp4_pro_split_n_mbarrier_band = intermediate_hidden >= 3072 and 0.0 < e < 64.0
    fp4_pro_two_tokens_per_expert_band = intermediate_hidden >= 3072 and 1.5 <= e < 3.0
    fp4_pro_mid_decode_assist_band = intermediate_hidden >= 3072 and 6.0 <= e < 12.0
    fp4_pro_large_decode_assist_batch = intermediate_hidden >= 3072 and 24.0 <= e < 64.0
    fp4_flash_two_tokens_per_expert_band = intermediate_hidden <= 2048 and 1.5 <= e < 2.0
    fp4_flash_half_token_per_expert_band = intermediate_hidden <= 2048 and 0.375 <= e < 0.5
    fp4_flash_decode_lookahead_band = intermediate_hidden <= 2048 and 3.0 <= e < 6.0
    fp4_flash_stage4_no_early_band = intermediate_hidden <= 2048 and 6.0 <= e < 12.0
    fp4_flash_wide_load_decode_band = intermediate_hidden <= 2048 and 6.0 <= e < 64.0
    fp4_pro_wide_load_decode_band = intermediate_hidden >= 3072 and 0.0 < e < 6.0
    fp4_flash_split_n_mbarrier_band = intermediate_hidden <= 2048 and 0.75 <= e < 64.0
    fp4_flash_small_mbarrier_band = intermediate_hidden <= 2048 and 0.0 < e < 0.5
    fp4_2wg_decode_offload_band = e >= 64.0
    default_math_wg_decode = (
        (0.0 < e < 0.375)
        or fp4_flash_half_token_per_expert_band
        or (1.0 <= e < 2.0)
        or fp4_pro_two_tokens_per_expert_band
        or fp4_decode_lookahead_band
        or fp4_b4_skip_decode_band
        or fp4_flash_split_n_mbarrier_band
        or fp4_pro_mid_decode_assist_band
        or fp4_pro_large_decode_assist_batch
        or fp4_bigband_lookahead_band
        or fp4_2wg_decode_offload_band
    )
    default_skip_loader_decode_assist = (
        (0.0 < e < 0.375)
        or fp4_flash_half_token_per_expert_band
        or fp4_pro_single_token_per_expert_band
        or (1.5 <= e < 3.0)
        or fp4_decode_lookahead_band
        or fp4_b4_skip_decode_band
        or fp4_flash_split_n_mbarrier_band
        or fp4_pro_mid_decode_assist_band
        or fp4_pro_large_decode_assist_batch
        or fp4_bigband_lookahead_band
        or fp4_2wg_decode_offload_band
    )
    default_wide_load_decode = (
        fp4_pro_wide_load_decode_band
        or fp4_pro_mid_decode_assist_band
        or fp4_pro_bigband_lookahead_band
        or fp4_flash_half_token_per_expert_band
        or fp4_flash_two_tokens_per_expert_band
        or fp4_flash_wide_load_decode_band
        or fp4_pro_large_decode_assist_batch
    )
    default_ss_early_b_decode = (
        (
            1.5 <= e <= 3.0
            and not fp4_pro_two_tokens_per_expert_band
            and not fp4_flash_two_tokens_per_expert_band
            and not fp4_flash_decode_lookahead_band
        )
        or (
            intermediate_hidden >= 3072
            and 6.0 <= e <= 24.0
            and not fp4_pro_mid_decode_assist_band
            and not fp4_pro_bigband_lookahead_band
            and not fp4_flash_stage4_no_early_band
        )
        or fp4_2wg_decode_offload_band
    )
    default_decode_done_mbarrier = (
        fp4_pro_split_n_mbarrier_band
        or fp4_flash_split_n_mbarrier_band
        or fp4_flash_small_mbarrier_band
        or (
            fp4_decode_lookahead_band
            and not fp4_flash_decode_lookahead_band
            and not fp4_flash_stage4_no_early_band
        )
        or fp4_bigband_lookahead_band
        or fp4_2wg_decode_offload_band
    )
    default_l2_arrival_counter = (
        (intermediate_hidden <= 2048 and 0.375 <= e < 0.75)
        or (intermediate_hidden >= 3072 and 0.25 <= e < 0.375)
    )
    return {
        "math_wg_participates": not default_math_wg_decode,
        "first_decode_assist_warp": 2 if default_skip_loader_decode_assist else 0,
        "wide_load_decode": default_wide_load_decode,
        "early_b_decode": default_ss_early_b_decode,
        "decode_done_mbarrier": default_decode_done_mbarrier,
        "l2_arrival_counter": default_l2_arrival_counter,
        "ss_nsplit": e >= 64.0,
        "swap_ab": (intermediate_hidden <= 2048 or intermediate_hidden >= 3072) and 0.0 < e <= 24.0,
        "swap_ab_fast_amax": intermediate_hidden >= 3072 and 12.0 <= e <= 24.0,
    }


def table_api_features(
    num_experts_per_rank: int,
    expected_tokens_per_expert: float,
    intermediate_hidden: int,
):
    e = expected_tokens_per_expert
    fp4_flash_shape = intermediate_hidden <= 2048
    fp4_pro_shape = intermediate_hidden >= 3072
    fp4_middle_shape = not fp4_flash_shape and not fp4_pro_shape
    fp4_decode_lookahead_shape_band = 3.0 <= e <= 6.0
    fp4_bigband_lookahead_shape_band = 12.0 <= e <= 24.0
    fp4_b4_skip_decode_shape_band = 0.5 <= e < 1.0
    fp4_pro_single_token_per_expert_shape_band = (
        fp4_pro_shape and 1.0 <= e < 1.5 and num_experts_per_rank % 8 == 0
    )
    fp4_pro_split_n_mbarrier_shape_band = fp4_pro_shape and 0.0 < e < 64.0
    fp4_pro_two_tokens_per_expert_shape_band = fp4_pro_shape and 1.5 <= e < 3.0
    fp4_pro_mid_decode_assist_shape_band = fp4_pro_shape and 6.0 <= e < 12.0
    fp4_pro_large_decode_assist_shape_band = fp4_pro_shape and 24.0 <= e < 64.0
    fp4_flash_two_tokens_per_expert_shape_band = fp4_flash_shape and 1.5 <= e < 2.0
    fp4_flash_half_token_per_expert_shape_band = fp4_flash_shape and 0.375 <= e < 0.5
    fp4_flash_decode_lookahead_shape_band = fp4_flash_shape and 3.0 <= e < 6.0
    fp4_flash_wide_load_decode_shape_band = fp4_flash_shape and 6.0 <= e < 64.0
    fp4_pro_wide_load_decode_shape_band = fp4_pro_shape and 0.0 < e < 64.0
    fp4_flash_split_n_mbarrier_shape_band = fp4_flash_shape and 0.75 <= e < 64.0
    fp4_flash_small_mbarrier_shape_band = fp4_flash_shape and 0.0 < e < 0.5
    fp4_2wg_decode_offload_shape_band = e >= 64.0
    fp4_shared_decode_assist_shape_band = (
        (0.0 < e < 0.375)
        or fp4_flash_half_token_per_expert_shape_band
        or fp4_b4_skip_decode_shape_band
        or fp4_decode_lookahead_shape_band
        or fp4_flash_split_n_mbarrier_shape_band
        or fp4_pro_mid_decode_assist_shape_band
        or fp4_pro_large_decode_assist_shape_band
        or fp4_bigband_lookahead_shape_band
        or fp4_2wg_decode_offload_shape_band
    )
    default_math_wg_decode = (
        fp4_shared_decode_assist_shape_band
        or (1.0 <= e < 2.0)
        or fp4_pro_two_tokens_per_expert_shape_band
    )
    default_skip_loader_decode_assist = (
        fp4_shared_decode_assist_shape_band
        or fp4_pro_single_token_per_expert_shape_band
        or (1.5 <= e < 3.0)
    )
    default_wide_load_decode = (
        fp4_pro_wide_load_decode_shape_band
        or fp4_flash_half_token_per_expert_shape_band
        or fp4_flash_two_tokens_per_expert_shape_band
        or fp4_flash_wide_load_decode_shape_band
    )
    default_ss_early_b_decode = (
        (
            1.5 <= e <= 3.0
            and not fp4_pro_two_tokens_per_expert_shape_band
            and not fp4_flash_two_tokens_per_expert_shape_band
            and not fp4_flash_decode_lookahead_shape_band
        )
        or fp4_2wg_decode_offload_shape_band
    )
    default_decode_done_mbarrier = (
        fp4_pro_split_n_mbarrier_shape_band
        or fp4_flash_split_n_mbarrier_shape_band
        or fp4_flash_small_mbarrier_shape_band
        or (fp4_middle_shape and fp4_decode_lookahead_shape_band)
        or (fp4_middle_shape and fp4_bigband_lookahead_shape_band)
        or fp4_2wg_decode_offload_shape_band
    )
    default_l2_arrival_counter = (
        (fp4_flash_shape and 0.375 <= e < 0.75)
        or (fp4_pro_shape and 0.25 <= e < 0.375)
    )
    return {
        "math_wg_participates": not default_math_wg_decode,
        "first_decode_assist_warp": 2 if default_skip_loader_decode_assist else 0,
        "wide_load_decode": default_wide_load_decode,
        "early_b_decode": default_ss_early_b_decode,
        "decode_done_mbarrier": default_decode_done_mbarrier,
        "l2_arrival_counter": default_l2_arrival_counter,
        "ss_nsplit": e >= 64.0,
        "swap_ab": (fp4_flash_shape or fp4_pro_shape) and 0.0 < e <= 24.0,
        "swap_ab_fast_amax": fp4_pro_shape and 12.0 <= e <= 24.0,
    }


def check_case(num_experts_per_rank, e, intermediate_hidden, block_m, block_n, num_epilogue_threads):
    old = legacy_wave(num_experts_per_rank, e, intermediate_hidden, block_m, block_n)
    new = table_wave(num_experts_per_rank, e, intermediate_hidden, block_m, block_n)
    assert old == new, (
        f"wave mismatch n={num_experts_per_rank} e={e} ih={intermediate_hidden} "
        f"block=({block_m},{block_n}) old={old} new={new}"
    )
    old_threads = legacy_threads(e, intermediate_hidden, block_m, block_n, num_epilogue_threads)
    new_threads = table_threads(e, intermediate_hidden, block_m, block_n, num_epilogue_threads)
    assert old_threads == new_threads, (
        f"thread mismatch n={num_experts_per_rank} e={e} ih={intermediate_hidden} "
        f"block=({block_m},{block_n}) epilogue_threads={num_epilogue_threads} "
        f"old={old_threads} new={new_threads}"
    )
    old_epilogue = legacy_epilogue_threads(
        e, intermediate_hidden, block_m, block_n, num_epilogue_threads
    )
    new_epilogue = table_epilogue_threads(
        e, intermediate_hidden, block_m, block_n, num_epilogue_threads
    )
    assert old_epilogue == new_epilogue, (
        f"epilogue mismatch n={num_experts_per_rank} e={e} ih={intermediate_hidden} "
        f"block=({block_m},{block_n}) epilogue_threads={num_epilogue_threads} "
        f"old={old_epilogue} new={new_epilogue}"
    )
    old_stage_cap = legacy_stage_cap(e, intermediate_hidden, block_m, block_n)
    new_stage_cap = table_stage_cap(e, intermediate_hidden, block_m, block_n)
    assert old_stage_cap == new_stage_cap, (
        f"stage cap mismatch n={num_experts_per_rank} e={e} ih={intermediate_hidden} "
        f"block=({block_m},{block_n}) old={old_stage_cap} new={new_stage_cap}"
    )
    old_config = legacy_config(
        num_experts_per_rank, e, intermediate_hidden, block_m, block_n, num_epilogue_threads
    )
    new_config = table_config(
        num_experts_per_rank, e, intermediate_hidden, block_m, block_n, num_epilogue_threads
    )
    assert old_config == new_config, (
        f"config mismatch n={num_experts_per_rank} e={e} ih={intermediate_hidden} "
        f"block=({block_m},{block_n}) epilogue_threads={num_epilogue_threads} "
        f"old={old_config} new={new_config}"
    )
    old_features = legacy_api_features(num_experts_per_rank, e, intermediate_hidden)
    new_features = table_api_features(num_experts_per_rank, e, intermediate_hidden)
    assert old_features == new_features, (
        f"API feature mismatch n={num_experts_per_rank} e={e} "
        f"ih={intermediate_hidden} old={old_features} new={new_features}"
    )


def _run_sm90_fp4_heuristic_checks():
    e_values = {
        0.0, 0.125, 0.249, 0.25, 0.374, 0.375, 0.499, 0.5,
        0.749, 0.75, 0.999, 1.0, 1.499, 1.5, 1.999, 2.0,
        2.999, 3.0, 3.001, 4.0, 5.999, 6.0, 6.001, 11.999,
        12.0, 12.001, 23.999, 24.0, 24.001, 31.999, 32.0,
        47.999, 48.0, 63.999, 64.0, 64.001,
    }
    num_experts_per_rank_values = (1, 7, 8, 15, 16, 23, 24, 31, 32, 47, 48, 64, 96)
    intermediate_hidden_values = (1024, 2048, 2500, 3072, 4096)
    block_values = ((64, 128), (64, 256), (128, 128), (64, 64))
    epilogue_thread_values = (128, 256, 512)

    checked = 0
    for n in num_experts_per_rank_values:
        for e in sorted(e_values):
            for ih in intermediate_hidden_values:
                for block_m, block_n in block_values:
                    for num_epilogue_threads in epilogue_thread_values:
                        check_case(n, e, ih, block_m, block_n, num_epilogue_threads)
                        checked += 1

    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        check_case(32, batch * 6 / 32, 2048, 64, 128, 256)
        check_case(48, batch * 6 / 48, 3072, 64, 128, 256)
        checked += 2

    print(f"SM90 FP4 heuristic equivalence passed: {checked} cases")



if __name__ == '__main__' and '--check-heuristics-only' in sys.argv:
    _run_sm90_fp4_heuristic_checks()
    sys.exit(0)

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getenv("DG_TEST_USE_SOURCE_TREE", "0") == "1" and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major
from test_mega_moe_hopper import (
    BASELINE_L2_ACT_SF_GRAN,
    _bench_cuda_events,
    _import_deep_ep,
    _make_deep_ep_buffer,
    _make_deep_ep_low_latency_buffer,
    _quantize_grouped_fp8_block_128_128,
    swiglu_apply_weight_to_fp8_triton,
    swiglu_masked_post_quant_to_fp8,
)


SM90_FP4_KERNEL_NAME = "sm90_fp8_fp4_mega_moe_impl"


# ----------------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------------

def _quantize_grouped_fp4_per32(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row, per-32 K FP4 quantization with UE8M0 SFB.

    Args
    ----
    w : (G, N, K) bf16, with K % 32 == 0 and K % 2 == 0.

    Returns
    -------
    fp4 : (G, N, K // 2) torch.int8 — packed E2M1; low nibble = even K.
    sf  : (G, N, K // 32) torch.float32 — UE8M0-rounded scale (= amax / 6
          rounded up to nearest power of two), the same dtype the SM100 FP4
          path expects pre-``transform_sf_into_required_layout``.
    """
    g, n, k = w.shape
    assert k % 32 == 0
    fp4 = torch.empty((g, n, k // 2), device=w.device, dtype=torch.int8)
    sf = torch.empty((g, n, k // 32), device=w.device, dtype=torch.float)
    for i in range(g):
        fp4[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
    return fp4, sf


def _dequant_fp4_per32(fp4: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_quantize_grouped_fp4_per32`. Returns (G, N, K) fp32."""
    g, n, half_k = fp4.shape
    k = half_k * 2
    pb = fp4.to(torch.uint8)
    lo = (pb & 0x0F).to(torch.int)
    hi = ((pb >> 4) & 0x0F).to(torch.int)
    codes = torch.stack([lo, hi], dim=-1).reshape(g, n, k)  # (G, N, K) int
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float, device=fp4.device,
    )
    sign = (codes & 0x08) != 0
    mag = (codes & 0x07).to(torch.long)
    val = table[mag]
    val = torch.where(sign & (mag != 0), -val, val)
    # Broadcast SF: each 32 K-cols share one SF.
    sf_broad = sf.unsqueeze(-1).expand(g, n, sf.size(-1), 32).reshape(g, n, k)
    return val * sf_broad


def _predecode_grouped_fp4_to_fp8_block_128_128(
    fp4: torch.Tensor,
    sf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert FP4 weights to the FP8 grouped-GEMM layout outside timed regions."""
    g, n, half_k = fp4.shape
    k = half_k * 2
    assert n % 128 == 0 and k % 128 == 0

    w_fp8 = torch.empty((g, n, k), device=fp4.device, dtype=torch.float8_e4m3fn)
    w_sf = torch.empty((g, n // 128, k // 128), device=fp4.device, dtype=torch.float32)
    for expert_idx in range(g):
        w_deq = _dequant_fp4_per32(
            fp4[expert_idx:expert_idx + 1],
            sf[expert_idx:expert_idx + 1],
        ).to(torch.bfloat16)
        q_fp8, q_sf = _quantize_grouped_fp8_block_128_128(w_deq)
        w_fp8[expert_idx:expert_idx + 1].copy_(q_fp8)
        w_sf[expert_idx:expert_idx + 1].copy_(q_sf)
        del w_deq, q_fp8, q_sf
        if expert_idx % 4 == 3:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return w_fp8, w_sf


def _randn_quantize_grouped_fp8_block_128_128(
    shape: Tuple[int, int, int],
    weight_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate and quantize grouped FP8 weights one expert at a time."""
    g, n, k = shape
    assert n % 128 == 0 and k % 128 == 0
    w_fp8 = torch.empty((g, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    w_sf = torch.empty((g, n // 128, k // 128), device="cuda", dtype=torch.float32)
    for expert_idx in range(g):
        w_bf16 = torch.empty((1, n, k), device="cuda", dtype=torch.bfloat16)
        w_bf16.normal_()
        w_bf16.mul_(weight_scale)
        q_fp8, q_sf = _quantize_grouped_fp8_block_128_128(w_bf16)
        w_fp8[expert_idx:expert_idx + 1].copy_(q_fp8)
        w_sf[expert_idx:expert_idx + 1].copy_(q_sf)
        del w_bf16, q_fp8, q_sf
        if expert_idx % 4 == 3:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return w_fp8, w_sf


def _dequant_per_token_per_128_k(
    x_fp8: torch.Tensor, sf: torch.Tensor
) -> torch.Tensor:
    """For (M, K) fp8 with (M, K // 128) float SF (per-token, K-major)."""
    m, k = x_fp8.shape
    assert k % 128 == 0
    x_view = x_fp8.float().view(m, k // 128, 128)
    return (x_view * sf.unsqueeze(-1)).view(m, k)


def _load_dsv4_checkpoint_layer_weights(
    model_path: str,
    layer_idx: int,
    rank_idx: int,
    num_ranks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    model_dir = Path(model_path)
    weight_map = json.loads((model_dir / 'model.safetensors.index.json').read_text())[
        'weight_map'
    ]
    num_experts = 256
    experts_per_rank = num_experts // num_ranks
    start = rank_idx * experts_per_rank
    end = start + experts_per_rank
    cache: Dict[str, Any] = {}

    def get_tensor(name: str) -> torch.Tensor:
        file_name = weight_map[name]
        if file_name not in cache:
            cache[file_name] = safe_open(str(model_dir / file_name),
                                         framework='pt', device='cpu')
        return cache[file_name].get_tensor(name)

    l1_w, l1_s, l2_w, l2_s = [], [], [], []
    for expert_id in range(start, end):
        prefix = f'layers.{layer_idx}.ffn.experts.{expert_id}'
        w1 = get_tensor(f'{prefix}.w1.weight')
        w3 = get_tensor(f'{prefix}.w3.weight')
        s1 = get_tensor(f'{prefix}.w1.scale')
        s3 = get_tensor(f'{prefix}.w3.scale')
        l1_w.append(torch.cat([w1, w3], dim=0))
        l1_s.append(torch.cat([s1.float(), s3.float()], dim=0))
        l2_w.append(get_tensor(f'{prefix}.w2.weight'))
        l2_s.append(get_tensor(f'{prefix}.w2.scale').float())

    return (
        torch.stack(l1_w, dim=0).cuda(),
        torch.stack(l1_s, dim=0).cuda(),
        torch.stack(l2_w, dim=0).cuda(),
        torch.stack(l2_s, dim=0).cuda(),
    )


def _m_grouped_fp8_gemm_nt_masked(*args, **kwargs):
    fn = (
        getattr(deep_gemm, "m_grouped_fp8_gemm_nt_masked", None)
        or getattr(deep_gemm, "fp8_m_grouped_gemm_nt_masked", None)
        or getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked", None)
    )
    if fn is None:
        raise AttributeError("no masked grouped FP8 GEMM API is exported by deep_gemm")
    return fn(*args, **kwargs)


def _safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def _all_rank_metrics(values: Tuple[float, ...]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered).cpu()


def _flush_l2_if_requested(l2_flush_gb: float):
    if l2_flush_gb <= 0:
        return
    free_bytes, _ = torch.cuda.mem_get_info()
    flush_bytes = min(int(l2_flush_gb * 1e9), int(free_bytes * 0.5))
    if flush_bytes >= 4:
        torch.empty(flush_bytes // 4, dtype=torch.int, device="cuda").zero_()


def _bench_cuda_event_sections(
    sections,
    num_warmup: int = 3,
    num_repeat: int = 10,
    l2_flush_gb: float = 0.0,
    barrier=None,
):
    for _ in range(num_warmup):
        for _, fn in sections:
            fn()
    torch.cuda.synchronize()

    section_times_ms = {name: [] for name, _ in sections}
    total_times_ms = []
    for _ in range(num_repeat):
        if barrier is not None:
            barrier()
        _flush_l2_if_requested(l2_flush_gb)
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        events = []
        total_start.record()
        for name, fn in sections:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((name, start, end))
        total_end.record()
        total_end.synchronize()
        total_times_ms.append(total_start.elapsed_time(total_end))
        for name, start, end in events:
            section_times_ms[name].append(start.elapsed_time(end))

    def median_sec(values_ms):
        values_ms = sorted(values_ms)
        return values_ms[len(values_ms) // 2] / 1e3

    section_times = {
        name: median_sec(values) for name, values in section_times_ms.items()
    }
    return section_times, median_sec(total_times_ms)


# ----------------------------------------------------------------------------
# PyTorch reference
# ----------------------------------------------------------------------------

def _swiglu_fp32(gate_up: torch.Tensor, clamp: float) -> torch.Tensor:
    """SwiGLU: silu(min(gate, c)) * clamp(up, -c, c)."""
    n2 = gate_up.size(-1)
    half = n2 // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _reference_fused(
    x_fp8_local: torch.Tensor, x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor, topk_weights_local: torch.Tensor,
    l1_w_fp4: torch.Tensor, l1_w_sf: torch.Tensor,
    l2_w_fp4: torch.Tensor, l2_w_sf: torch.Tensor,
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    num_experts: int, num_topk: int,
    hidden: int, intermediate_hidden: int,
    activation_clamp: float,
    reference_chunk: int = 0,
) -> torch.Tensor:
    """FP32 reference for the SM90 FP4 mega-MoE kernel.

    All-gathers tokens / topk decisions / per-rank weights, then for each
    global token routes through its topk experts, applies the L1+SwiGLU+L2
    path, and reduces over topk on the source rank.
    """
    num_experts_per_rank = num_experts // num_ranks

    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)
    mg = x_fp8_g.size(0)

    local_size = torch.tensor([x_fp8_local.size(0)], device='cuda', dtype=torch.long)
    sizes_t = torch.empty(num_ranks, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(sizes_t, local_size, group=group)
    sizes_list = sizes_t.tolist()

    l1_w_g = [torch.empty_like(l1_w_fp4) for _ in range(num_ranks)]
    l1_sf_g = [torch.empty_like(l1_w_sf) for _ in range(num_ranks)]
    l2_w_g = [torch.empty_like(l2_w_fp4) for _ in range(num_ranks)]
    l2_sf_g = [torch.empty_like(l2_w_sf) for _ in range(num_ranks)]
    dist.all_gather(l1_w_g, l1_w_fp4, group=group)
    dist.all_gather(l1_sf_g, l1_w_sf, group=group)
    dist.all_gather(l2_w_g, l2_w_fp4, group=group)
    dist.all_gather(l2_sf_g, l2_w_sf, group=group)
    l1_w_all = torch.stack(l1_w_g, dim=0)   # (R, E_pr, 2*IH, H/2)
    l1_sf_all = torch.stack(l1_sf_g, dim=0)
    l2_w_all = torch.stack(l2_w_g, dim=0)
    l2_sf_all = torch.stack(l2_sf_g, dim=0)

    combine_buf = torch.zeros(
        mg, num_topk, hidden, dtype=torch.float32, device='cuda')

    x_fp32 = _dequant_per_token_per_128_k(x_fp8_g, x_sf_g)  # (Mg, H)

    # Token-chunked dequant to bound peak memory of the per-token gather.
    _CHUNK = int(reference_chunk) if reference_chunk else int(os.getenv('DSV4_FP4_REFERENCE_CHUNK', '64'))
    for k in range(num_topk):
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx_full = mask.nonzero(as_tuple=False).squeeze(-1)
        for c0 in range(0, sel_idx_full.numel(), _CHUNK):
            sel_idx = sel_idx_full[c0:c0 + _CHUNK]
            eids = topk_idx_g[sel_idx, k]
            weights = topk_w_g[sel_idx, k]
            x_sel = x_fp32[sel_idx]                              # (S, H)

            dst_rank = (eids // num_experts_per_rank).long()
            dst_local = (eids % num_experts_per_rank).long()

            # L1 GEMM
            l1_w_sel = _dequant_fp4_per32(
                l1_w_all[dst_rank, dst_local],                   # (S, 2*IH, H/2) int8
                l1_sf_all[dst_rank, dst_local],                  # (S, 2*IH, H/32)
            )                                                    # (S, 2*IH, H) fp32
            l1_y = torch.einsum('sk,snk->sn', x_sel, l1_w_sel)
            del l1_w_sel

            l1_y = _swiglu_fp32(l1_y, activation_clamp) * weights.unsqueeze(-1)

            s_, ih = l1_y.shape
            assert ih == intermediate_hidden and ih % 64 == 0
            l1_view = l1_y.view(s_, ih // 64, 64)
            amax = l1_view.abs().amax(dim=-1).clamp(1e-4)
            sf2 = amax / 448.0
            l1_q = (l1_view / sf2.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
            l2_in = (l1_q * sf2.unsqueeze(-1)).view(s_, ih)

            l2_w_sel = _dequant_fp4_per32(
                l2_w_all[dst_rank, dst_local],                   # (S, H, IH/2) int8
                l2_sf_all[dst_rank, dst_local],                  # (S, H, IH/32)
            )                                                    # (S, H, IH) fp32
            l2_y = torch.einsum('sn,smn->sm', l2_in, l2_w_sel)
            del l2_w_sel

            combine_buf[sel_idx, k] = l2_y.to(torch.bfloat16).float()

    y_full_bf16 = combine_buf.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    return y_full_bf16[start:end].contiguous()


# ----------------------------------------------------------------------------
# Single-scenario runner
# ----------------------------------------------------------------------------

def _run_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    diff_tol: float,
):
    num_max = cfg['num_max_tokens_per_rank']
    num_tokens = cfg.get('num_tokens', num_max)
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    masked_ratio = cfg.get('masked_ratio', 0.0)
    activation_clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)
    input_pattern = cfg.get('input_pattern', 'random')
    routing_pattern = cfg.get('routing_pattern', 'random')
    num_launch_repeats = int(cfg.get('num_launch_repeats', 1))
    reference_chunk = cfg.get('reference_chunk')
    scenario_diff_tol = cfg.get('diff_tol', diff_tol)
    assert num_launch_repeats >= 1

    assert num_experts % num_ranks == 0
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    # Hard kernel constraints.
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64, (
        f'SM90 fused kernel requires intermediate_hidden <= 4096, '
        f'got {intermediate_hidden}'
    )

    torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)
    random.seed(rank_idx * 1000 + abs(hash(name)) % 1000)

    # ---- Inputs (bf16) ------------------------------------------------------
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    if input_pattern == 'sfa_poison':
        num_k_groups = hidden // 128
        group_scales = torch.tensor(
            [0.25, 1.0, 4.0, 16.0], dtype=torch.float, device='cuda')
        group_scales = group_scales[
            torch.arange(num_k_groups, device='cuda') % group_scales.numel()
        ].to(torch.bfloat16)
        x_bf = (
            x_bf.view(num_tokens, num_k_groups, 128)
            * group_scales.view(1, num_k_groups, 1)
        ).reshape(num_tokens, hidden)
    elif input_pattern != 'random':
        raise AssertionError(f'unknown input_pattern={input_pattern}')

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if routing_pattern == 'round_robin':
        token_idx = torch.arange(num_tokens, device='cuda').unsqueeze(1)
        topk_offset = torch.arange(num_topk, device='cuda').unsqueeze(0)
        global_token_idx = rank_idx * num_tokens + token_idx
        topk_idx = ((global_token_idx * num_topk + topk_offset) % num_experts).to(torch.long)
    elif routing_pattern != 'random':
        raise AssertionError(f'unknown routing_pattern={routing_pattern}')
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    # FP8 activations with per-128 K float SF (SM90 format) — same as SM90 FP8 path.
    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)

    if cfg.get('checkpoint_model_path'):
        l1_w_fp4, l1_w_sf, l2_w_fp4, l2_w_sf = _load_dsv4_checkpoint_layer_weights(
            cfg['checkpoint_model_path'],
            cfg.get('checkpoint_layer_idx', 0),
            rank_idx,
            num_ranks,
        )
    else:
        l1_bf = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden),
            dtype=torch.bfloat16, device='cuda') * 0.05
        l2_bf = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden),
            dtype=torch.bfloat16, device='cuda') * 0.05
        # FP4 weights with per-32 K UE8M0 SF — DSV4 standard.
        l1_w_fp4, l1_w_sf = _quantize_grouped_fp4_per32(l1_bf)
        l2_w_fp4, l2_w_sf = _quantize_grouped_fp4_per32(l2_bf)

    # SM90 FP4 weight transform: gate/up gran-8 N interleave + UE8M0 SFB k-major
    # packing into uint32. Both pieces are needed — see
    # ``transform_weights_for_mega_moe_sm90_fp4`` for layout details.
    transformed_l1, transformed_l2 = (
        deep_gemm.transform_weights_for_mega_moe_sm90_fp4(
            (l1_w_fp4, l1_w_sf), (l2_w_fp4, l2_w_sf)
        )
    )

    # ---- Allocate symm buffer ----------------------------------------------
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max, num_topk,
        hidden, intermediate_hidden,
    )
    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def run_fused_once():
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_w)
        cum_stats.zero_()
        y_fused.zero_()
        deep_gemm.fp8_fp4_mega_moe(
            y_fused, transformed_l1, transformed_l2, buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(1, 1, 32),
            activation='swiglu',
            activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
            fast_math=fast_math,
        )
        torch.cuda.synchronize()
        return y_fused

    # ---- Reference & check --------------------------------------------------
    y_ref = _reference_fused(
        x_fp8, x_sf, topk_idx, topk_w,
        l1_w_fp4, l1_w_sf, l2_w_fp4, l2_w_sf,
        rank_idx, num_ranks, group,
        num_experts, num_topk,
        hidden, intermediate_hidden,
        activation_clamp,
        reference_chunk=reference_chunk,
    )

    max_diff = 0.0
    ok = True
    for repeat_idx in range(num_launch_repeats):
        y_fused = run_fused_once()
        diff = calc_diff(y_fused, y_ref)
        max_diff = max(max_diff, float(diff))
        ok = ok and diff < scenario_diff_tol
    repeat_suffix = '' if num_launch_repeats == 1 else f' x{num_launch_repeats}'
    dist_print(f'  [{name:<32}] diff={max_diff:.4f}{repeat_suffix} '
               f'(tol={scenario_diff_tol:.2f}) {"OK" if ok else "FAIL"}',
               once_in_node=True)
    if not ok:
        for label, tensor in (('fused', y_fused), ('ref', y_ref)):
            tensor_f = tensor.float()
            dist_print(
                f'    {label}: abs_max={tensor_f.abs().max().item():.6g} '
                f'mean={tensor_f.mean().item():.6g} '
                f'nonzero={(tensor_f != 0).sum().item()}/{tensor_f.numel()} '
                f'finite={torch.isfinite(tensor_f).all().item()}',
                once_in_node=True,
            )
    assert ok, f'{name}: diff={max_diff} >= tol={scenario_diff_tol}'

    buffer.destroy()
    dist.barrier()


# ----------------------------------------------------------------------------
# Scenario tables (smaller than the SM90 FP8 test — FP4 reference is heavier)
# ----------------------------------------------------------------------------

_SMOKE = dict(
    num_max_tokens_per_rank=64, num_tokens=64,
    hidden=512, intermediate_hidden=512,
    num_experts=8, num_topk=2,
)


def _layer1_smoke() -> List[Tuple[str, Dict[str, Any]]]:
    return [('L1.smoke', dict(_SMOKE))]


def _layer3_shape_sweep(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base_experts = 8 * num_ranks
    out = []
    for hidden in (512, 1024):
        for ih in (512, 1024):
            for topk in (1, 2):
                cfg = dict(num_max_tokens_per_rank=128, num_tokens=128,
                           hidden=hidden, intermediate_hidden=ih,
                           num_experts=base_experts, num_topk=topk)
                out.append((f'L3.h{hidden}_ih{ih}_k{topk}', cfg))
    return out


def _layer4_edges(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(num_max_tokens_per_rank=128,
                hidden=512, intermediate_hidden=512,
                num_experts=8 * num_ranks, num_topk=2)
    out = []
    for mr in (0.0, 0.5):
        cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=mr)
        out.append((f'L4.mask{mr:.1f}', cfg))
    for c in (1.0, math.inf):
        cfg = dict(base); cfg.update(num_tokens=128, activation_clamp=c)
        out.append((f'L4.clamp{c}', cfg))
    cfg = dict(base); cfg.update(num_tokens=0)
    out.append(('L4.tokens0', cfg))
    return out


def _layer5_dsv4_shape(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    assert num_ranks == 8, 'DSV4 shape test expects 8 ranks'
    return [('L5.dsv4_h4096_ih2048_e256_k6', dict(
        num_max_tokens_per_rank=128, num_tokens=64,
        hidden=4096, intermediate_hidden=2048,
        num_experts=256, num_topk=6,
        activation_clamp=10.0,
    ))]


def _layer6_dsv4_checkpoint(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    assert num_ranks == 8, 'DSV4 checkpoint test expects 8 ranks'
    model_path = os.getenv('DSV4_FP4_MODEL_PATH')
    if not model_path:
        dist_print(
            '[SKIP] layer 6 DSV4 checkpoint test requires DSV4_FP4_MODEL_PATH',
            once_in_node=True)
        return []
    return [('L6.dsv4_ckpt_layer0_h4096_ih2048_e256_k6', dict(
        num_max_tokens_per_rank=128, num_tokens=64,
        hidden=4096, intermediate_hidden=2048,
        num_experts=256, num_topk=6,
        activation_clamp=10.0,
        checkpoint_model_path=model_path,
        checkpoint_layer_idx=0,
    ))]


def _layer7_dsv4_2wg(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    # 2-WG split-MN accuracy guard. num_tokens=512 drives
    # expected_tokens_per_expert = 512 * num_topk / experts_per_rank
    #                            = 512 * 6 / (256 / 8) = 96 >= 64,
    # so get_block_config_for_mega_moe_sm90 takes the auto_split_mn branch:
    # block_m=128 with TWO epilogue warpgroups. This is the only accuracy
    # scenario that exercises the 2-WG path; L1/L3/L4/L5 are all 1-WG
    # (num_tokens<=128 -> expected<64). It guards the default that turns the
    # math warpgroup's FP4 decode OFF on the 2-WG path (decode is offloaded
    # to the assist warps and written to the shared decoded-B smem tile, so the
    # numerics must be identical to the math-on path).
    assert num_ranks == 8, 'DSV4 2-WG shape test expects 8 ranks'
    return [('L7.dsv4_2wg_nt512_h4096_ih2048_e256_k6', dict(
        num_max_tokens_per_rank=512, num_tokens=512,
        hidden=4096, intermediate_hidden=2048,
        num_experts=256, num_topk=6,
        activation_clamp=10.0,
    ))]


def _layer8_pro_smoke(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ('L8.pro_b64_mid_h7168_ih3072_e384_k6', dict(
            num_max_tokens_per_rank=128, num_tokens=64,
            hidden=7168, intermediate_hidden=3072,
            num_experts=384, num_topk=6,
            activation_clamp=10.0,
        )),
        ('L8.pro_b128_1wg_h7168_ih3072_e384_k6', dict(
            num_max_tokens_per_rank=128, num_tokens=128,
            hidden=7168, intermediate_hidden=3072,
            num_experts=384, num_topk=6,
            activation_clamp=10.0,
        )),
        ('L8.pro_b256_1wg_h7168_ih3072_e384_k6', dict(
            num_max_tokens_per_rank=256, num_tokens=256,
            hidden=7168, intermediate_hidden=3072,
            num_experts=384, num_topk=6,
            activation_clamp=10.0,
        )),
        ('L8.pro_b512_2wg_h7168_ih3072_e384_k6', dict(
            num_max_tokens_per_rank=512, num_tokens=512,
            hidden=7168, intermediate_hidden=3072,
            num_experts=384, num_topk=6,
            activation_clamp=10.0,
        )),
    ]


def _layer9_swapab_small_batch(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    assert num_ranks == 8, 'swapAB small-batch test expects 8 ranks'
    common = dict(
        num_max_tokens_per_rank=128,
        num_topk=6,
        activation_clamp=10.0,
        routing_pattern='round_robin',
        diff_tol=0.05,
    )
    flash = dict(common, hidden=4096, intermediate_hidden=2048, num_experts=256)
    pro = dict(common, hidden=7168, intermediate_hidden=3072, num_experts=384,
               reference_chunk=16)
    pro_fast_amax = dict(common, hidden=1024, intermediate_hidden=3072,
                         num_experts=384, reference_chunk=16)

    return [
        ('L9.flash_swapab_b1_sfa_poison', dict(
            flash, num_tokens=1, input_pattern='sfa_poison',
            num_launch_repeats=5,
        )),
        ('L9.flash_swapab_b8', dict(flash, num_tokens=8)),
        ('L9.flash_swapab_b32', dict(flash, num_tokens=32)),
        ('L9.pro_swapab_b1_sfa_poison', dict(
            pro, num_tokens=1, input_pattern='sfa_poison',
            num_launch_repeats=5,
        )),
        ('L9.pro_swapab_b4', dict(pro, num_tokens=4)),
        ('L9.pro_swapab_b16', dict(pro, num_tokens=16)),
        ('L9.pro_swapab_b128_fast_amax', dict(
            pro_fast_amax, num_tokens=128, num_launch_repeats=2,
        )),
    ]


# ----------------------------------------------------------------------------
# Benchmark mode
# ----------------------------------------------------------------------------

def _run_benchmark(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(
            f"[SKIP] test_mega_moe_fp4_hopper --bench requires SM90; "
            f"got SM{get_arch_major()}0",
            once_in_node=True,
        )
        dist.destroy_process_group()
        return

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens if args.num_tokens else num_max_tokens_per_rank
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    run_fp4_runtime_enabled = args.fp4_mode == "runtime"
    run_fp4_predecode_enabled = args.fp4_mode == "predecode"
    run_fp4_timing_enabled = run_fp4_runtime_enabled or run_fp4_predecode_enabled
    run_fp8_normal_baseline_enabled = (
        args.run_normal_baseline and not args.ncu_profile_only
    )
    run_fp8_ll_baseline_enabled = (
        not args.skip_fp8_ll_baseline and not args.ncu_profile_only
    )
    run_low_latency_path_enabled = (
        run_fp8_ll_baseline_enabled or run_fp4_predecode_enabled
    )
    need_original_fp8_weights = (
        run_fp8_normal_baseline_enabled or run_fp8_ll_baseline_enabled
    )
    baseline_only_weights = (
        not run_fp4_timing_enabled and
        need_original_fp8_weights and
        not args.bench_check_reference
    )

    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_bf16 = None
    l2_bf16 = None
    if not baseline_only_weights:
        l1_bf16 = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        ) * args.weight_scale
        l2_bf16 = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden),
            dtype=torch.bfloat16,
            device="cuda",
        ) * args.weight_scale

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, num_topk, dim=-1, largest=True, sorted=False
    )
    topk_idx_ll = topk_idx.to(torch.int64)

    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )

    l1_fp4 = None
    l2_fp4 = None
    transformed_l1 = None
    transformed_l2 = None
    need_fp4_weights = (
        run_fp4_timing_enabled or args.bench_check_reference
    )
    if need_fp4_weights:
        assert l1_bf16 is not None and l2_bf16 is not None
        l1_fp4 = _quantize_grouped_fp4_per32(l1_bf16)
        l2_fp4 = _quantize_grouped_fp4_per32(l2_bf16)
    if run_fp4_runtime_enabled:
        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90_fp4(
            l1_fp4, l2_fp4
        )

    l1_fp8 = None
    l2_fp8 = None
    if baseline_only_weights:
        l1_fp8 = _randn_quantize_grouped_fp8_block_128_128(
            (num_experts_per_rank, intermediate_hidden * 2, hidden),
            args.weight_scale,
        )
        torch.cuda.empty_cache()
        l2_fp8 = _randn_quantize_grouped_fp8_block_128_128(
            (num_experts_per_rank, hidden, intermediate_hidden),
            args.weight_scale,
        )
    elif need_original_fp8_weights:
        assert l1_bf16 is not None and l2_bf16 is not None
        l1_fp8 = _quantize_grouped_fp8_block_128_128(l1_bf16)
        l2_fp8 = _quantize_grouped_fp8_block_128_128(l2_bf16)
    if l1_bf16 is not None:
        del l1_bf16
    if l2_bf16 is not None:
        del l2_bf16
    torch.cuda.empty_cache()

    predecode_l1_fp8 = None
    predecode_l2_fp8 = None
    if run_fp4_predecode_enabled:
        predecode_l1_fp8 = _predecode_grouped_fp4_to_fp8_block_128_128(*l1_fp4)
        predecode_l2_fp8 = _predecode_grouped_fp4_to_fp8_block_128_128(*l2_fp4)

    clamp_arg = args.activation_clamp if math.isfinite(args.activation_clamp) else None
    cum_stats = None
    sym_buffer = None
    y_fused = None
    if run_fp4_runtime_enabled:
        if not args.bench_skip_cum_stats:
            cum_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")
        sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def fp4_prepare_inputs():
        assert sym_buffer is not None
        sym_buffer.x[:num_tokens].copy_(x_fp8)
        sym_buffer.x_sf[:num_tokens].copy_(x_sf)
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def fp4_fused_kernel():
        assert transformed_l1 is not None and transformed_l2 is not None
        assert sym_buffer is not None and y_fused is not None
        deep_gemm.fp8_fp4_mega_moe(
            y_fused,
            transformed_l1,
            transformed_l2,
            sym_buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=clamp_arg,
            fast_math=bool(args.fast_math),
        )
        return y_fused

    def run_fp4_fused():
        fp4_prepare_inputs()
        return fp4_fused_kernel()

    if args.ncu_profile_only:
        assert run_fp4_runtime_enabled, "--ncu-profile-only requires --fp4-mode runtime"
        dist_print(
            f"[NCU] FP4 Hopper tokens={num_tokens} hidden={hidden} "
            f"ih={intermediate_hidden}",
            once_in_node=True,
        )
        run_fp4_fused()
        torch.cuda.synchronize()
        dist.barrier()
        if sym_buffer is not None:
            sym_buffer.destroy()
        dist.destroy_process_group()
        return

    normal_buffer = None
    ll_buffer = None
    if run_fp8_normal_baseline_enabled or run_low_latency_path_enabled:
        deep_ep = _import_deep_ep()
        if deep_ep is None:
            raise RuntimeError("deep_ep is required for baseline comparisons")

    if run_fp8_normal_baseline_enabled:
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        normal_buffer = _make_deep_ep_buffer(
            deep_ep,
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            0 if sym_buffer is None else sym_buffer.buffer.nbytes,
        )
        normal_cum_stats = torch.zeros(
            (num_experts_per_rank,), dtype=torch.int, device="cuda"
        )
        normal_state = {}

    def fp8_normal_dispatch():
        recv_x, _, recv_topk_weights, handle, _ = normal_buffer.dispatch(
            (x_fp8, x_sf),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=normal_cum_stats,
            num_experts=num_experts,
            expert_alignment=alignment,
            do_cpu_sync=False,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=False,
        )
        normal_state["recv_x"] = recv_x
        normal_state["recv_topk_weights"] = recv_topk_weights
        normal_state["handle"] = handle

    def fp8_normal_l1_gemm():
        recv_x = normal_state["recv_x"]
        handle = normal_state["handle"]
        l1_y = torch.empty(
            (recv_x[0].size(0), intermediate_hidden * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            recv_x,
            l1_fp8,
            l1_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )
        normal_state["l1_y"] = l1_y

    def fp8_normal_swiglu_quant():
        normal_state["l1_act"] = swiglu_apply_weight_to_fp8_triton(
            x=normal_state["l1_y"],
            topk_weights=normal_state["recv_topk_weights"],
            clamp_value=clamp_arg,
            num_per_channels=BASELINE_L2_ACT_SF_GRAN,
            use_ue8m0_scale=True,
        )

    def fp8_normal_l2_gemm():
        handle = normal_state["handle"]
        l2_y = torch.empty(
            (normal_state["l1_act"][0].size(0), hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            normal_state["l1_act"],
            l2_fp8,
            l2_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )
        normal_state["l2_y"] = l2_y

    def fp8_normal_combine():
        combined = normal_buffer.combine(
            normal_state["l2_y"],
            handle=normal_state["handle"],
        )[0]
        normal_state["combined"] = combined
        return combined

    def run_fp8_normal_baseline():
        fp8_normal_dispatch()
        fp8_normal_l1_gemm()
        fp8_normal_swiglu_quant()
        fp8_normal_l2_gemm()
        return fp8_normal_combine()

    if run_low_latency_path_enabled:

        ll_buffer = _make_deep_ep_low_latency_buffer(
            deep_ep, group, num_max_tokens_per_rank, hidden, num_experts
        )
        m_max_ll = num_max_tokens_per_rank * num_ranks
        expected_m_ll = max(
            1,
            (num_max_tokens_per_rank * num_ranks * num_topk + num_experts - 1)
            // num_experts,
        )
        ll_l1_y = torch.empty(
            (num_experts_per_rank, m_max_ll, intermediate_hidden * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ll_l2_y = torch.empty(
            (num_experts_per_rank, m_max_ll, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        ll_combined = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        ll_state = {}

    def fp8_ll_dispatch_into(state):
        (recv_x_data, recv_x_sf), masked_m, ll_handle, event, hook = (
            ll_buffer.low_latency_dispatch(
                x_bf16,
                topk_idx_ll,
                num_max_tokens_per_rank,
                num_experts,
                use_fp8=True,
                round_scale=False,
                use_ue8m0=False,
                async_finish=False,
                return_recv_hook=False,
            )
        )
        state["recv"] = (recv_x_data, recv_x_sf)
        state["masked_m"] = masked_m
        state["ll_handle"] = ll_handle

    def fp8_ll_l1_gemm_with(state, weights):
        _m_grouped_fp8_gemm_nt_masked(
            state["recv"],
            weights,
            ll_l1_y,
            state["masked_m"],
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

    def fp8_ll_swiglu_quant_into(state):
        l1_act_fp8, l1_act_sf = swiglu_masked_post_quant_to_fp8(
            ll_l1_y,
            state["masked_m"],
            quant_group_size=BASELINE_L2_ACT_SF_GRAN,
            clamp_value=clamp_arg,
            use_ue8m0_scale=False,
        )
        state["l1_act"] = (l1_act_fp8, l1_act_sf)

    def fp8_ll_l2_gemm_with(state, weights):
        _m_grouped_fp8_gemm_nt_masked(
            state["l1_act"],
            weights,
            ll_l2_y,
            state["masked_m"],
            expected_m_ll,
            disable_ue8m0_cast=True,
        )

    def fp8_ll_combine_from(state):
        combined, event, hook = ll_buffer.low_latency_combine(
            ll_l2_y,
            topk_idx_ll,
            topk_weights,
            state["ll_handle"],
            use_logfmt=False,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
            out=ll_combined,
        )
        state["combined"] = combined
        return combined

    def fp8_ll_dispatch():
        fp8_ll_dispatch_into(ll_state)

    def fp8_ll_l1_gemm():
        fp8_ll_l1_gemm_with(ll_state, l1_fp8)

    def fp8_ll_swiglu_quant():
        fp8_ll_swiglu_quant_into(ll_state)

    def fp8_ll_l2_gemm():
        fp8_ll_l2_gemm_with(ll_state, l2_fp8)

    def fp8_ll_combine():
        return fp8_ll_combine_from(ll_state)

    def run_low_latency_with_weights(l1_weights, l2_weights, state):
        fp8_ll_dispatch_into(state)
        fp8_ll_l1_gemm_with(state, l1_weights)
        fp8_ll_swiglu_quant_into(state)
        fp8_ll_l2_gemm_with(state, l2_weights)
        return fp8_ll_combine_from(state)

    def run_fp8_low_latency_baseline():
        return run_low_latency_with_weights(l1_fp8, l2_fp8, ll_state)

    predecode_state = {}

    def run_fp4_predecode_low_latency():
        return run_low_latency_with_weights(
            predecode_l1_fp8, predecode_l2_fp8, predecode_state)

    fused_out = None
    if fused_out is None:
        if run_fp4_predecode_enabled:
            fused_out = run_fp4_predecode_low_latency()
        elif run_fp4_runtime_enabled:
            fused_out = run_fp4_fused()
    if fused_out is not None:
        assert fused_out.shape == (num_tokens, hidden)
    if args.bench_check_reference:
        assert fused_out is not None, "--bench-check-reference requires an FP4 mode"
        assert l1_fp4 is not None and l2_fp4 is not None
        y_ref = _reference_fused(
            x_fp8, x_sf, topk_idx, topk_weights,
            l1_fp4[0], l1_fp4[1], l2_fp4[0], l2_fp4[1],
            rank_idx, num_ranks, group,
            num_experts, num_topk,
            hidden, intermediate_hidden,
            args.activation_clamp,
        )
        diff = calc_diff(fused_out, y_ref)
        ok = diff < args.diff_tol
        if rank_idx == 0:
            print(
                "BENCH_REFERENCE_JSON " + json.dumps(
                    {
                        "batch_per_rank": num_tokens,
                        "hidden": hidden,
                        "intermediate_hidden": intermediate_hidden,
                        "num_experts": num_experts,
                        "num_topk": num_topk,
                        "diff": round(float(diff), 6),
                        "diff_tol": args.diff_tol,
                        "ok": bool(ok),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        assert ok, f"bench reference diff={diff} >= tol={args.diff_tol}"
    if run_fp8_normal_baseline_enabled:
        normal_out = run_fp8_normal_baseline()
        assert normal_out.shape == (num_tokens, hidden)
    if run_fp8_ll_baseline_enabled:
        ll_out = run_fp8_low_latency_baseline()
        assert ll_out.shape == (num_tokens, hidden)
    torch.cuda.synchronize()

    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[
        (gathered_topk_idx < rank_idx * num_experts_per_rank)
        | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
    ] = -1
    local_expert_ids = gathered_topk_idx[gathered_topk_idx != -1]
    num_recv_tokens = int(local_expert_ids.numel())
    num_touched_experts = int(torch.unique(local_expert_ids).numel())

    if args.profile_breakdown:
        if run_fp4_predecode_enabled:
            fp4_sections = [
                ("fp4_predecode_ll_dispatch", lambda: fp8_ll_dispatch_into(predecode_state)),
                ("fp4_predecode_ll_l1_gemm",
                 lambda: fp8_ll_l1_gemm_with(predecode_state, predecode_l1_fp8)),
                ("fp4_predecode_ll_swiglu_quant",
                 lambda: fp8_ll_swiglu_quant_into(predecode_state)),
                ("fp4_predecode_ll_l2_gemm",
                 lambda: fp8_ll_l2_gemm_with(predecode_state, predecode_l2_fp8)),
                ("fp4_predecode_ll_combine",
                 lambda: fp8_ll_combine_from(predecode_state)),
            ]
        else:
            fp4_sections = [
                ("fp4_prepare_inputs", fp4_prepare_inputs),
                ("fp4_fused_kernel", fp4_fused_kernel),
            ]
        dist.barrier()
        fp4_profile, fp4_profile_total = _bench_cuda_event_sections(
            fp4_sections,
            num_warmup=args.profile_warmup,
            num_repeat=args.profile_repeat,
            l2_flush_gb=args.profile_l2_flush_gb,
            barrier=dist.barrier,
        )
        profile_names = ["fp4_total", *[name for name, _ in fp4_sections]]
        profile_values = [
            fp4_profile_total,
            *[fp4_profile[name] for name, _ in fp4_sections],
        ]
        if run_fp8_ll_baseline_enabled:
            fp8_ll_sections = [
                ("fp8_ll_dispatch", fp8_ll_dispatch),
                ("fp8_ll_l1_gemm", fp8_ll_l1_gemm),
                ("fp8_ll_swiglu_quant", fp8_ll_swiglu_quant),
                ("fp8_ll_l2_gemm", fp8_ll_l2_gemm),
                ("fp8_ll_combine", fp8_ll_combine),
            ]
            dist.barrier()
            fp8_profile, fp8_profile_total = _bench_cuda_event_sections(
                fp8_ll_sections,
                num_warmup=args.profile_warmup,
                num_repeat=args.profile_repeat,
                l2_flush_gb=args.profile_l2_flush_gb,
                barrier=dist.barrier,
            )
            profile_names += ["fp8_ll_total", *[name for name, _ in fp8_ll_sections]]
            profile_values += [
                fp8_profile_total,
                *[fp8_profile[name] for name, _ in fp8_ll_sections],
            ]
        profile_metrics = _all_rank_metrics(tuple(profile_values))
        profile_count_metrics = _all_rank_metrics(
            (float(num_recv_tokens), float(num_touched_experts))
        )
        if rank_idx == 0:
            profile_result = {
                "batch_per_rank": num_tokens,
                "num_ranks": num_ranks,
                "num_experts": num_experts,
                "num_topk": num_topk,
                "recv_tokens_total": int(profile_count_metrics[:, 0].sum().item()),
                "active_experts_max": int(profile_count_metrics[:, 1].max().item()),
                "profile_repeat": args.profile_repeat,
                "profile_warmup": args.profile_warmup,
                "profile_l2_flush_gb": args.profile_l2_flush_gb,
                "fp8_ll_baseline_enabled": run_fp8_ll_baseline_enabled,
            }
            for i, name in enumerate(profile_names):
                profile_result[f"{name}_us_max"] = round(
                    float(profile_metrics[:, i].max().item() * 1e6), 3
                )
                profile_result[f"{name}_us_mean"] = round(
                    float(profile_metrics[:, i].mean().item() * 1e6), 3
                )
            print("PROFILE_JSON " + json.dumps(profile_result, sort_keys=True), flush=True)
        dist.barrier()

    def bench_low_latency_pipeline(fn):
        _, total = _bench_cuda_event_sections(
            [("pipeline", fn)],
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
            barrier=dist.barrier,
        )
        return total

    if run_fp4_predecode_enabled:
        fused_timing_method = "cuda_events_predecode_ll_barrier"
        t_fused = bench_low_latency_pipeline(run_fp4_predecode_low_latency)
    elif run_fp4_runtime_enabled and args.fp4_runtime_timing == "cuda-events":
        fused_timing_method = "cuda_events_runtime_barrier"
        t_fused = bench_low_latency_pipeline(run_fp4_fused)
    elif run_fp4_runtime_enabled:
        t_fused = bench_kineto(
            run_fp4_fused,
            SM90_FP4_KERNEL_NAME,
            num_tests=args.num_bench_tests,
            barrier=dist.barrier,
            flush_l2=bool(args.kineto_flush_l2),
        )
        kineto_ok = torch.tensor([1 if t_fused > 0 else 0], dtype=torch.int, device="cuda")
        dist.all_reduce(kineto_ok, op=dist.ReduceOp.MIN)
        fused_timing_method = "kineto_kernel"
        if kineto_ok.item() == 0:
            fused_timing_method = "cuda_events_fallback"
            t_fused = _bench_cuda_events(
                run_fp4_fused,
                num_warmup=args.num_warmup,
                num_repeat=args.num_repeat,
                l2_flush_gb=args.l2_flush_gb,
            )
    else:
        fused_timing_method = "disabled"
        t_fused = 0.0

    t_ll = float("nan")
    t_normal = float("nan")
    if run_fp8_normal_baseline_enabled:
        dist.barrier()
        t_normal = _bench_cuda_events(
            run_fp8_normal_baseline,
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
        )
        dist.barrier()

    if run_fp8_ll_baseline_enabled:
        dist.barrier()
        t_ll = bench_low_latency_pipeline(run_fp8_low_latency_baseline)
        dist.barrier()

    metrics = _all_rank_metrics(
        (
            t_fused,
            t_normal,
            t_ll,
            float(num_recv_tokens),
            float(num_touched_experts),
        )
    )
    if rank_idx == 0:
        fused_us_max = None
        fused_us_mean = None
        if run_fp4_timing_enabled:
            fused_us_max = float(metrics[:, 0].max().item() * 1e6)
            fused_us_mean = float(metrics[:, 0].mean().item() * 1e6)
        normal_us_max = None
        normal_us_mean = None
        speedup_vs_fp8_normal_max = None
        if run_fp8_normal_baseline_enabled:
            normal_us_max = float(metrics[:, 1].max().item() * 1e6)
            normal_us_mean = float(metrics[:, 1].mean().item() * 1e6)
            if fused_us_max is not None:
                speedup_vs_fp8_normal_max = round(
                    _safe_div(normal_us_max, fused_us_max), 4
                )
        ll_us_max = None
        ll_us_mean = None
        speedup_vs_fp8_ll_max = None
        if run_fp8_ll_baseline_enabled:
            ll_us_max = float(metrics[:, 2].max().item() * 1e6)
            ll_us_mean = float(metrics[:, 2].mean().item() * 1e6)
            if fused_us_max is not None:
                speedup_vs_fp8_ll_max = round(_safe_div(ll_us_max, fused_us_max), 4)
        result = {
            "batch_per_rank": num_tokens,
            "num_ranks": num_ranks,
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "recv_tokens_total": int(metrics[:, 3].sum().item()),
            "active_experts_max": int(metrics[:, 4].max().item()),
            "fp4_megamoe_us_max": None if fused_us_max is None else round(fused_us_max, 3),
            "fp4_megamoe_us_mean": None if fused_us_mean is None else round(fused_us_mean, 3),
            "fp4_mode": args.fp4_mode,
            "fp4_timing_method": fused_timing_method,
            "fp8_normal_baseline_enabled": run_fp8_normal_baseline_enabled,
            "fp8_normal_baseline_us_max": (
                None if normal_us_max is None else round(normal_us_max, 3)
            ),
            "fp8_normal_baseline_us_mean": (
                None if normal_us_mean is None else round(normal_us_mean, 3)
            ),
            "speedup_vs_fp8_normal_max": speedup_vs_fp8_normal_max,
            "fp8_ll_baseline_enabled": run_fp8_ll_baseline_enabled,
            "fp8_ll_baseline_us_max": None if ll_us_max is None else round(ll_us_max, 3),
            "fp8_ll_baseline_us_mean": None if ll_us_mean is None else round(ll_us_mean, 3),
            "speedup_vs_fp8_ll_max": speedup_vs_fp8_ll_max,
            "num_bench_tests": args.num_bench_tests,
            "num_warmup": args.num_warmup,
            "num_repeat": args.num_repeat,
            "l2_flush_gb": args.l2_flush_gb,
            "kineto_flush_l2": bool(args.kineto_flush_l2),
        }
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)

    dist.barrier()
    if sym_buffer is not None:
        sym_buffer.destroy()
    if normal_buffer is not None:
        normal_buffer.destroy()
    if ll_buffer is not None:
        ll_buffer.destroy()
    dist.destroy_process_group()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    if args.bench:
        _run_benchmark(local_rank, num_local_ranks, args)
        return

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if get_arch_major() != 9:
        dist_print(
            f'[SKIP] test_mega_moe_fp4_hopper requires SM90; '
            f'got SM{get_arch_major()}0',
            once_in_node=True)
        dist.destroy_process_group()
        return

    diff_tol = args.diff_tol
    layers: List[Tuple[str, Dict[str, Any]]] = []
    if 1 in args.layers:
        layers += _layer1_smoke()
    if 3 in args.layers:
        layers += _layer3_shape_sweep(num_ranks)
    if 4 in args.layers:
        layers += _layer4_edges(num_ranks)
    if 5 in args.layers:
        layers += _layer5_dsv4_shape(num_ranks)
    if 6 in args.layers:
        layers += _layer6_dsv4_checkpoint(num_ranks)
    if 7 in args.layers:
        layers += _layer7_dsv4_2wg(num_ranks)
    if 8 in args.layers or args.pro_smoke:
        layers += _layer8_pro_smoke(num_ranks)
    if 9 in args.layers or args.swapab_smoke:
        layers += _layer9_swapab_small_batch(num_ranks)

    if args.filter:
        layers = [(n, c) for n, c in layers if args.filter in n]

    dist_print(f'SM90 FP4 MegaMoE test plan: {len(layers)} scenarios across '
               f'layers {sorted(args.layers)} on {num_ranks} ranks',
               once_in_node=True)

    failures: List[str] = []
    for name, cfg in layers:
        try:
            _run_scenario(name, cfg, rank_idx, num_ranks, group, diff_tol)
        except AssertionError as ex:
            dist_print(f'  [{name}] FAIL: {ex}', once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED {len(failures)}/{len(layers)} scenarios: {failures}',
                   once_in_node=True)
    else:
        dist_print(f'PASSED all {len(layers)} scenarios', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hopper FP4 MegaMoE tests and benchmark')
    parser.add_argument('--check-heuristics-only', action='store_true',
                        help='Run CPU-only SM90 FP4 heuristic equivalence checks and exit')
    parser.add_argument('--bench', action='store_true',
                        help='Run FP4 fused vs FP8 low-latency benchmark mode')
    parser.add_argument('--ncu-profile-only', action='store_true',
                        help='With --bench, run one FP4 fused kernel launch for NCU')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 3, 4],
                        help='Correctness layers to run (1, 3, 4, 5, 6, 7, 8, 9). '
                             'Default: 1 3 4. Layer 8 is the Pro smoke shape; '
                             'layer 9 is the Flash/Pro swapAB small-batch guard.')
    parser.add_argument('--pro-smoke', action='store_true',
                        help='Also run DeepSeek-V4-Pro smoke scenarios')
    parser.add_argument('--swapab-smoke', action='store_true',
                        help='Also run Flash/Pro small-batch swapAB correctness guards')
    parser.add_argument('--filter', type=str, default='')
    parser.add_argument('--diff-tol', type=float, default=0.10,
                        help='calc_diff tolerance (default 0.10; FP4 weights '
                             'introduce more quantization noise than FP8).')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--bench-check-reference', action='store_true',
                        help='With --bench, run the FP32 reference on the same shape before timing')
    parser.add_argument('--bench-skip-cum-stats', action='store_true',
                        help='With --bench, pass None for FP4 cumulative expert recv stats')

    parser.add_argument('--num-max-tokens-per-rank', type=int, default=1)
    parser.add_argument('--num-tokens', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--weight-scale', type=float, default=0.05)
    parser.add_argument(
        '--fp4-mode',
        choices=('runtime', 'predecode', 'none'),
        default='runtime',
        help='Benchmark mode: runtime FP4 kernel, FP4-predecoded low-latency lower bound, or FP8-baseline only',
    )
    parser.add_argument('--num-bench-tests', type=int, default=20)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument('--num-repeat', type=int, default=20)
    parser.add_argument('--l2-flush-gb', type=float, default=0.0)
    parser.add_argument('--profile-breakdown', action='store_true',
                        help='Emit PROFILE_JSON with CUDA-event stage timings')
    parser.add_argument('--skip-fp8-ll-baseline', action='store_true',
                        help='Only measure the FP4 fused path')
    parser.add_argument('--run-normal-baseline', action='store_true',
                        help='Also measure the normal DeepEP dispatch/combine FP8 baseline')
    parser.add_argument('--profile-warmup', type=int, default=3)
    parser.add_argument('--profile-repeat', type=int, default=10)
    parser.add_argument('--profile-l2-flush-gb', type=float, default=0.0)
    parser.add_argument('--kineto-flush-l2', type=int, default=0)
    parser.add_argument(
        '--fp4-runtime-timing',
        choices=('kineto', 'cuda-events'),
        default='kineto',
        help='Timing backend for --fp4-mode runtime',
    )
    args = parser.parse_args()

    np_ = args.num_processes
    if args.local_rank_idx is not None:
        test(args.local_rank_idx, np_, args)
    else:
        torch.multiprocessing.spawn(test, args=(np_, args), nprocs=np_)
