import sys
import time
import os
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import (
    cast_back_from_fp4,
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
)


def _cast_back_from_fp8_1d(x: torch.Tensor, sf: torch.Tensor, gran_k: int = 128) -> torch.Tensor:
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def _require_sm90() -> None:
    assert torch.cuda.is_available()
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(f"This benchmark is intended for SM90, got sm_{major}x")


def _time_cuda(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def _effective_bytes(
    groups: int,
    m_per_group: int,
    n: int,
    k: int,
    a_gran_k: int,
    *,
    fp8_b: bool,
    b_gran_k: int | None = None,
) -> int:
    b_gran_k = a_gran_k if b_gran_k is None else b_gran_k
    logical_m = groups * m_per_group
    a_scale_k = (k + a_gran_k - 1) // a_gran_k
    b_scale_k = (k + b_gran_k - 1) // b_gran_k
    a_bytes = logical_m * k + logical_m * a_scale_k * 4
    b_data_bytes = groups * n * k if fp8_b else groups * n * (k // 2)
    b_scale_bytes = groups * n * b_scale_k * 4
    d_bytes = logical_m * n * 2
    return a_bytes + b_data_bytes + b_scale_bytes + d_bytes


def _effective_masked_bytes(
    groups: int,
    masked_m_values: list[int],
    n: int,
    k: int,
    a_gran_k: int,
    *,
    fp8_b: bool,
    b_gran_k: int | None = None,
) -> int:
    b_gran_k = a_gran_k if b_gran_k is None else b_gran_k
    logical_m = sum(masked_m_values)
    a_scale_k = (k + a_gran_k - 1) // a_gran_k
    b_scale_k = (k + b_gran_k - 1) // b_gran_k
    a_bytes = logical_m * k + logical_m * a_scale_k * 4
    b_data_bytes = groups * n * k if fp8_b else groups * n * (k // 2)
    b_scale_bytes = groups * n * b_scale_k * 4
    d_bytes = logical_m * n * 2
    return a_bytes + b_data_bytes + b_scale_bytes + d_bytes


def _build_grouped_layout(groups: int, m_per_group: int):
    m = groups * m_per_group
    group_starts = [group_id * m_per_group for group_id in range(groups)]
    group_ends = [(group_id + 1) * m_per_group for group_id in range(groups)]
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)
    return m, group_starts, group_ends, grouped_layout


def _benchmark_case(groups: int, m_per_group: int, n: int, k: int, gran_k: int = 128) -> dict[str, float | int]:
    m, group_starts, group_ends, grouped_layout = _build_grouped_layout(groups, m_per_group)
    a_ref_src = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref_src, use_ue8m0=False, gran_k=gran_k)
    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    for group_id in range(groups):
        b_fp4[group_id], b_sf[group_id] = per_token_cast_to_fp4(
            b_ref_src[group_id], use_ue8m0=True, gran_k=gran_k
        )
    b_w4 = (b_fp4, b_sf)

    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=gran_k)
    b_fp8_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        b_dequant = cast_back_from_fp4(b_w4[0][group_id], b_w4[1][group_id], gran_k=gran_k)
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_dequant, use_ue8m0=False, gran_k=gran_k
        )
        start = group_starts[group_id]
        end = group_ends[group_id]
        if start != end:
            ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)
    b_fp8 = (b_fp8_data, b_fp8_sf)

    d_fp8 = torch.empty_like(ref)

    def run_fp8():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a,
            b_fp8,
            d_fp8,
            grouped_layout,
            recipe_a=(1, gran_k),
            recipe_b=(1, gran_k),
            use_psum_layout=False,
        )

    run_fp8()
    fp8_diff = calc_diff(d_fp8, ref)
    fp8_elapsed = _time_cuda(run_fp8)

    d_w4 = torch.empty_like(ref)

    def run_w4():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
            a,
            b_w4,
            d_w4,
            grouped_layout,
            gran_k=gran_k,
            compiled_dims="nk",
            use_psum_layout=False,
        )

    run_w4()
    w4_diff = calc_diff(d_w4, ref)
    w4_elapsed = _time_cuda(run_w4)

    fp8_threshold = 0.05
    assert w4_diff < 0.015
    assert fp8_diff < fp8_threshold

    w4_bytes = _effective_bytes(groups, m_per_group, n, k, gran_k, fp8_b=False)
    fp8_bytes = _effective_bytes(groups, m_per_group, n, k, gran_k, fp8_b=True)
    return {
        "groups": groups,
        "m_per_group": m_per_group,
        "n": n,
        "k": k,
        "w4_us": w4_elapsed * 1e6,
        "w4_gbps": w4_bytes / w4_elapsed / 1e9,
        "w4_diff": w4_diff,
        "fp8_us": fp8_elapsed * 1e6,
        "fp8_gbps": fp8_bytes / fp8_elapsed / 1e9,
        "fp8_diff": fp8_diff,
        "speedup": fp8_elapsed / w4_elapsed,
    }


def _print_markdown_table(rows: list[dict[str, float | int]]) -> None:
    print("groups | m/group | n | k | W4 us | W4 GB/s | W4 diff | FP8 us | FP8 GB/s | FP8 diff | Speedup")
    print("-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --")
    for row in rows:
        prefix = f"{row['groups']} | {row['m_per_group']} | {row['n']} | {row['k']} | "
        print(
            prefix +
            f"{row['w4_us']:.0f} | {row['w4_gbps']:.0f} | {row['w4_diff']:.4f} | "
            f"{row['fp8_us']:.0f} | {row['fp8_gbps']:.0f} | {row['fp8_diff']:.4f} | "
            f"{row['speedup']:.2f}x"
        )


def _masked_benchmark_case(
    groups: int,
    m_per_group: int,
    n: int,
    k: int,
    a_gran_k: int = 128,
    b_gran_k: int = 32,
) -> dict[str, float | int]:
    sm90_masked_w4 = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma", None)
    if sm90_masked_w4 is None:
        raise RuntimeError(
            "SM90 FP8xFP4 masked fused kernel is not exposed yet. "
            "Do not call generic m_grouped_fp8_fp4_gemm_nt_masked on SM90; "
            "it is currently routed to the SM100 FP8xFP4 masked path."
        )

    max_m = 128
    masked_m = torch.full((groups,), m_per_group, device="cuda", dtype=torch.int32)

    a_ref_src = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, k // a_gran_k), device="cuda", dtype=torch.float)
    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    use_packed_b_sf = bool(int(os.getenv("DG_W4_FUSE_SCALE_B_DECODE", "0")))
    b_sf_k = k // (b_gran_k * (4 if use_packed_b_sf else 1))
    b_sf = torch.empty((groups, n, b_sf_k), device="cuda", dtype=torch.int if use_packed_b_sf else torch.float)
    for group_id in range(groups):
        a_data[group_id], a_sf[group_id] = per_token_cast_to_fp8(
            a_ref_src[group_id], use_ue8m0=False, gran_k=a_gran_k
        )
        b_fp4[group_id], b_sf[group_id] = per_token_cast_to_fp4(
            b_ref_src[group_id], use_ue8m0=True, gran_k=b_gran_k, use_packed_ue8m0=use_packed_b_sf
        )
    a = (a_data, a_sf)
    b_w4 = (b_fp4, b_sf)

    assert a[1].shape == (groups, max_m, k // a_gran_k)
    assert b_w4[1].shape == (groups, n, b_sf_k)
    if b_gran_k == 128 and not use_packed_b_sf:
        assert b_w4[1].dtype == torch.float
        assert b_w4[1].shape == (groups, n, k // 128)


    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=a_gran_k)
    b_fp8_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // a_gran_k), device="cuda", dtype=torch.float)
    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        valid_m = int(masked_m[group_id].item())
        b_dequant = cast_back_from_fp4(
            b_w4[0][group_id], b_w4[1][group_id], gran_k=b_gran_k, use_packed_ue8m0=use_packed_b_sf
        )
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_dequant, use_ue8m0=False, gran_k=a_gran_k
        )
        if valid_m > 0:
            ref[group_id, :valid_m] = (a_dequant[group_id, :valid_m] @ b_dequant.t()).to(torch.bfloat16)
    b_fp8 = (b_fp8_data, b_fp8_sf)

    d_w4 = torch.empty_like(ref)

    def run_w4():
        sm90_masked_w4(
            a,
            b_w4,
            d_w4,
            masked_m,
            m_per_group,
            gran_k=a_gran_k,
            gran_k_a=a_gran_k,
            gran_k_b=b_gran_k,
        )

    run_w4()
    w4_diff = max(
        calc_diff(d_w4[group_id, :m_per_group], ref[group_id, :m_per_group])
        for group_id in range(groups)
    )
    w4_elapsed = _time_cuda(run_w4)

    d_fp8 = torch.empty_like(ref)

    def run_fp8():
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a,
            b_fp8,
            d_fp8,
            masked_m,
            m_per_group,
            recipe_a=(1, a_gran_k),
            recipe_b=(1, a_gran_k),
        )

    run_fp8()
    fp8_diff = max(
        calc_diff(d_fp8[group_id, :m_per_group], ref[group_id, :m_per_group])
        for group_id in range(groups)
    )
    fp8_elapsed = _time_cuda(run_fp8)

    #assert w4_diff < 0.015
    #assert fp8_diff < 0.05

    w4_bytes = _effective_bytes(groups, m_per_group, n, k, a_gran_k, fp8_b=False, b_gran_k=b_gran_k)
    fp8_bytes = _effective_bytes(groups, m_per_group, n, k, a_gran_k, fp8_b=True)
    return {
        "groups": groups,
        "m_per_group": m_per_group,
        "n": n,
        "k": k,
        "w4_us": w4_elapsed * 1e6,
        "w4_gbps": w4_bytes / w4_elapsed / 1e9,
        "w4_diff": w4_diff,
        "fp8_us": fp8_elapsed * 1e6,
        "fp8_gbps": fp8_bytes / fp8_elapsed / 1e9,
        "fp8_diff": fp8_diff,
        "speedup": fp8_elapsed / w4_elapsed,
    }


def _masked_skew_benchmark_case(
    name: str,
    masked_m_values: list[int],
    expected_m: int,
    n: int,
    k: int,
    max_m: int = 1024,
    a_gran_k: int = 128,
    b_gran_k: int = 32,
    pass_hints: bool = True,
) -> dict[str, float | int | str]:
    sm90_masked_w4 = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma", None)
    if sm90_masked_w4 is None:
        raise RuntimeError("SM90 FP8xFP4 masked fused kernel is not exposed yet.")

    groups = len(masked_m_values)
    assert groups > 0
    assert max(masked_m_values) <= max_m
    masked_m_max_hint = max(masked_m_values) if pass_hints else None
    active_groups_hint = (
        sum(1 for v in masked_m_values if v > 0) if pass_hints else None
    )
    masked_m = torch.tensor(masked_m_values, device="cuda", dtype=torch.int32)

    a_ref_src = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, k // a_gran_k), device="cuda", dtype=torch.float)
    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    use_packed_b_sf = bool(int(os.getenv("DG_W4_FUSE_SCALE_B_DECODE", "0")))
    b_sf_k = k // (b_gran_k * (4 if use_packed_b_sf else 1))
    block_m_override = int(os.getenv("DG_W4_BLOCK_M_OVERRIDE", "0")) or None
    block_n_override = int(os.getenv("DG_W4_BLOCK_N_OVERRIDE", "0")) or None
    bm32_skew_fast_path = (
        b_gran_k == 32
        and masked_m_max_hint is not None
        and masked_m_max_hint > 16
        and os.getenv("DG_W4_PATHB_FUSE_DECODE", "0") == "0"
        and os.getenv("DG_W4_PATHB_FAST_PATH", "1") != "0"
        and os.getenv("DG_W4_PATHB_BM64", "0") == "0"
        and (block_m_override is None or block_m_override == 32)
        and (block_n_override is None or block_n_override in (128, 256))
        and max_m >= 1024
        and groups == 32
        and n in (4096, 7168)
        and k in (2048, 3072, 4096, 7168)
    )
    scale_b_direct_load = b_gran_k == 32 and (
        expected_m <= 16 or bm32_skew_fast_path
    )
    scale_b_dtype_fast_path = (
        scale_b_direct_load
        and os.getenv("DG_W4_K32_QUAD_SCALE_B_PREFETCH", "0") == "0"
        and os.getenv("DG_W4_SCALE_B_POW2_PROMOTE", "0") == "0"
    )
    # bf16 sfb：path-A (gran_k_b=128) 与 path-B fast-path (gran_k_b=32) 都支持，
    # 体积砍半。按 MN-major + tma_aligned_mn=align(N, 8) 直接构造，避开 host 端
    # fp32-only 的 transpose 路径。**默认开启**，DG_W4_SCALE_B_BF16=0 时回退 fp32。
    use_bf16_b_sf = (
        ((b_gran_k == 32 and scale_b_dtype_fast_path) or b_gran_k == 128)
        and not use_packed_b_sf
        and bool(int(os.getenv("DG_W4_SCALE_B_BF16", "1")))
    )
    # E8M0 sfb（uint8）：仅 path-B (gran_k_b=32)。每元素 = fp32 pow2 scale 的指数位。
    # per_token_cast_to_fp4(..., use_ue8m0=True) 已保证 sf 是严格 pow2，因此抽指数无损。
    # **默认开启**，DG_W4_SCALE_B_E8M0=0 时回退。优先级高于 bf16（互斥）。
    use_e8m0_b_sf = (
        b_gran_k == 32
        and scale_b_dtype_fast_path
        and not use_packed_b_sf
        and bool(int(os.getenv("DG_W4_SCALE_B_E8M0", "1")))
    )
    if use_e8m0_b_sf:
        use_bf16_b_sf = False  # e8m0 优先
    if use_e8m0_b_sf:
        # uint8: tma_aligned_mn = ceil(N, 16)，要求 N % 16 == 0。
        assert n % 16 == 0
        tma_aligned_n = (n + 15) // 16 * 16
        b_sf = torch.empty_strided(
            (groups, n, b_sf_k),
            (tma_aligned_n * b_sf_k, 1, tma_aligned_n),
            device="cuda",
            dtype=torch.uint8,
        )
        b_sf_fp32 = torch.empty((groups, n, b_sf_k), device="cuda", dtype=torch.float)
    elif use_bf16_b_sf:
        # bf16: tma_aligned_mn = ceil(N, 8)，要求 N % 8 == 0。
        assert n % 8 == 0
        tma_aligned_n = (n + 7) // 8 * 8
        b_sf = torch.empty_strided(
            (groups, n, b_sf_k),
            (tma_aligned_n * b_sf_k, 1, tma_aligned_n),
            device="cuda",
            dtype=torch.bfloat16,
        )
        b_sf_fp32 = torch.empty((groups, n, b_sf_k), device="cuda", dtype=torch.float)
    else:
        b_sf = torch.empty((groups, n, b_sf_k), device="cuda", dtype=torch.int if use_packed_b_sf else torch.float)
        b_sf_fp32 = b_sf
    for group_id in range(groups):
        a_data[group_id], a_sf[group_id] = per_token_cast_to_fp8(
            a_ref_src[group_id], use_ue8m0=False, gran_k=a_gran_k
        )
        b_fp4[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp4(
            b_ref_src[group_id], use_ue8m0=True, gran_k=b_gran_k, use_packed_ue8m0=use_packed_b_sf
        )
    if use_e8m0_b_sf:
        # b_sf_fp32 已是严格 pow2（per_token_cast_to_fp4 use_ue8m0=True）。
        # 抽 fp32 指数位 = ((bits >> 23) & 0xff)：sign=0、mantissa=0 时 fp32 == 2^(e-127)。
        b_sf_bits = b_sf_fp32.view(torch.int32)
        b_sf_e8m0 = ((b_sf_bits >> 23) & 0xff).to(torch.uint8)
        b_sf.copy_(b_sf_e8m0)
    elif use_bf16_b_sf:
        # 写入 MN-major bf16 buffer：fp32 → bf16 round-down，再按目标 stride copy。
        b_sf.copy_(b_sf_fp32.to(torch.bfloat16))
    a = (a_data, a_sf)
    b_w4 = (b_fp4, b_sf)

    # caller hint：把 hot group 的真实大小传给 host，host 据此选 BM。
    # path-A 通过 masked_m_max_hint 接收 hint；FP8 baseline 没有 hint API，
    # 但它的 expected_m 可以直接传 max（FP8 路径无 expected_m<=8 fast-path 副作用），
    # 这样两条路径都按 hot 调度，speedup 反映的是算法差距而非 caller-side gap。
    gemm_expected_m = expected_m
    # pass_hints=False 时 max_hint=None：两边都用 expected_m（mirror 业务真实
    # caller 不传 hint 的状态，host 走没有 hint 的 small-m candidate 路径）。
    fp8_expected_m = (
        max(masked_m_max_hint, expected_m)
        if masked_m_max_hint is not None
        else expected_m
    )

    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=a_gran_k)
    b_fp8_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // a_gran_k), device="cuda", dtype=torch.float)
    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    # ref 反量化必须用 fp32 scale（cast_back_from_fp4 直接做 sf 乘法，不会解码 e8m0/bf16）。
    # use_e8m0_b_sf / use_bf16_b_sf 时 b_w4[1] 是 uint8/bf16 编码，需走 b_sf_fp32 兜底。
    b_sf_for_ref = b_sf_fp32 if (use_e8m0_b_sf or use_bf16_b_sf) else b_w4[1]
    for group_id, valid_m in enumerate(masked_m_values):
        b_dequant = cast_back_from_fp4(
            b_w4[0][group_id], b_sf_for_ref[group_id], gran_k=b_gran_k, use_packed_ue8m0=use_packed_b_sf
        )
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_dequant, use_ue8m0=False, gran_k=a_gran_k
        )
        if valid_m > 0:
            ref[group_id, :valid_m] = (a_dequant[group_id, :valid_m] @ b_dequant.t()).to(torch.bfloat16)
    b_fp8 = (b_fp8_data, b_fp8_sf)

    d_w4 = torch.empty_like(ref)

    def run_w4():
        sm90_masked_w4(
            a,
            b_w4,
            d_w4,
            masked_m,
            gemm_expected_m,
            gran_k=a_gran_k,
            gran_k_a=a_gran_k,
            gran_k_b=b_gran_k,
            masked_m_max_hint=masked_m_max_hint,
            active_groups_hint=active_groups_hint,
        )

    run_w4()
    w4_diff = max(
        calc_diff(d_w4[group_id, :valid_m], ref[group_id, :valid_m]) if valid_m > 0 else 0.0
        for group_id, valid_m in enumerate(masked_m_values)
    )
    w4_elapsed = _time_cuda(run_w4)

    d_fp8 = torch.empty_like(ref)

    def run_fp8():
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a,
            b_fp8,
            d_fp8,
            masked_m,
            fp8_expected_m,
            recipe_a=(1, a_gran_k),
            recipe_b=(1, a_gran_k),
        )

    run_fp8()
    fp8_diff = max(
        calc_diff(d_fp8[group_id, :valid_m], ref[group_id, :valid_m]) if valid_m > 0 else 0.0
        for group_id, valid_m in enumerate(masked_m_values)
    )
    fp8_elapsed = _time_cuda(run_fp8)

    w4_bytes = _effective_masked_bytes(groups, masked_m_values, n, k, a_gran_k, fp8_b=False, b_gran_k=b_gran_k)
    fp8_bytes = _effective_masked_bytes(groups, masked_m_values, n, k, a_gran_k, fp8_b=True)
    return {
        "case": name,
        "groups": groups,
        "expected_m": expected_m,
        "masked_m_hint": masked_m_max_hint,
        "b_gran_k": b_gran_k,
        "max_m": max_m,
        "sum_m": sum(masked_m_values),
        "masked_max": max(masked_m_values),
        "active_groups": sum(1 for value in masked_m_values if value > 0),
        "n": n,
        "k": k,
        "w4_us": w4_elapsed * 1e6,
        "w4_gbps": w4_bytes / w4_elapsed / 1e9,
        "w4_diff": w4_diff,
        "fp8_us": fp8_elapsed * 1e6,
        "fp8_gbps": fp8_bytes / fp8_elapsed / 1e9,
        "fp8_diff": fp8_diff,
        "speedup": fp8_elapsed / w4_elapsed,
    }


def _print_skew_table(rows: list[dict[str, float | int | str]]) -> None:
    print("case | groups | b_gran_k | exp_m_avg | hint | sum_m | max_m | active | n | k | "
          "W4 us | W4 GB/s | W4 diff | FP8 us | FP8 GB/s | FP8 diff | Speedup")
    print("-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --")
    for row in rows:
        print(
            f"{row['case']} | {row['groups']} | {row['b_gran_k']} | {row['expected_m']} | "
            f"{row['masked_m_hint']} | {row['sum_m']} | "
            f"{row['masked_max']} | {row['active_groups']} | {row['n']} | {row['k']} | "
            f"{row['w4_us']:.0f} | {row['w4_gbps']:.0f} | {row['w4_diff']:.4f} | "
            f"{row['fp8_us']:.0f} | {row['fp8_gbps']:.0f} | {row['fp8_diff']:.4f} | "
            f"{row['speedup']:.2f}x"
        )


def _accuracy_case(
    groups: int,
    m_per_group: int,
    n: int,
    k: int,
    gran_k: int = 128,
    *,
    block_m: int = 128,
    block_n: int = 128,
) -> tuple[float, float]:
    m, group_starts, group_ends, grouped_layout = _build_grouped_layout(groups, m_per_group)
    a_ref_src = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref_src, use_ue8m0=False, gran_k=gran_k)
    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    for group_id in range(groups):
        b_fp4[group_id], b_sf[group_id] = per_token_cast_to_fp4(
            b_ref_src[group_id], use_ue8m0=True, gran_k=gran_k
        )
    b_w4 = (b_fp4, b_sf)

    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=gran_k)
    b_fp8_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        b_dequant = cast_back_from_fp4(b_w4[0][group_id], b_w4[1][group_id], gran_k=gran_k)
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_dequant, use_ue8m0=False, gran_k=gran_k
        )
        start = group_starts[group_id]
        end = group_ends[group_id]
        if start != end:
            ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)
    b_fp8 = (b_fp8_data, b_fp8_sf)

    d_w4 = torch.empty_like(ref)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
        a,
        b_w4,
        d_w4,
        grouped_layout,
        gran_k=gran_k,
        compiled_dims="nk",
        use_psum_layout=False,
        block_m_override=block_m,
        block_n_override=block_n,
    )

    d_fp8 = torch.empty_like(ref)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a,
        b_fp8,
        d_fp8,
        grouped_layout,
        recipe_a=(1, gran_k),
        recipe_b=(1, gran_k),
        use_psum_layout=False,
    )

    return calc_diff(d_w4, ref), calc_diff(d_fp8, ref)


def test_sm90_fp8_fp4_contiguous() -> None:
    _require_sm90()
    torch.manual_seed(0)

    rows = []
    for groups in (8, 16, 24, 32):
        for m_per_group in (128, 256, 512, 1024):
            rows.append(_benchmark_case(groups, m_per_group, n=4096, k=7168))
    _print_markdown_table(rows)


def test_sm90_fp8_fp4_masked() -> None:
    _require_sm90()
    torch.manual_seed(2)

    print("direct E8M0 B scale case: b.second shape = [groups, N, K/32]")

    rows = []
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(8, m_per_group, n=4096, k=7168))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(8, m_per_group, n=7168, k=2048))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(16, m_per_group, n=4096, k=7168))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(16, m_per_group, n=7168, k=2048))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(32, m_per_group, n=4096, k=7168))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(32, m_per_group, n=7168, k=2048))
    _print_markdown_table(rows)

def test_sm90_fp8_fp4_masked_direct_fp32_scale() -> None:
    _require_sm90()
    torch.manual_seed(3)

    print("direct FP32 B scale case: b.second shape = [groups, N, K/128]")
    rows = []
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(8, m_per_group, n=4096, k=7168, b_gran_k=128))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(8, m_per_group, n=7168, k=2048, b_gran_k=128))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(16, m_per_group, n=4096, k=7168, b_gran_k=128))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(16, m_per_group, n=7168, k=2048, b_gran_k=128))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(32, m_per_group, n=4096, k=7168, b_gran_k=128))
    for m_per_group in (1, 4, 8, 16, 32):
        rows.append(_masked_benchmark_case(32, m_per_group, n=7168, k=2048, b_gran_k=128))
    _print_markdown_table(rows)


def test_sm90_fp8_fp4_masked_skew_cases() -> None:
    _require_sm90()
    torch.manual_seed(4)

    def skew_values(
        total: int, hot: int, active: int = 8, groups: int = 32
    ) -> list[int]:
        assert active >= 1
        assert active <= groups
        assert total >= hot
        values = [0] * groups
        values[0] = hot
        remaining = total - hot
        for idx in range(1, active):
            share = (remaining + active - idx - 1) // (active - idx)
            values[idx] = share
            remaining -= share
        assert sum(values) == total
        return values

    def values_from_active(active_values: list[int], groups: int = 32) -> list[int]:
        assert len(active_values) <= groups
        assert all(value >= 0 for value in active_values)
        return active_values + [0] * (groups - len(active_values))

    def values_from_repeated(value: int, active: int, groups: int = 32) -> list[int]:
        assert active <= groups
        return values_from_active([value] * active, groups=groups)

    def long_tail_values(
        hot: int, tail_start: int, active: int, groups: int = 32
    ) -> list[int]:
        values = [hot]
        next_value = tail_start
        for _ in range(active - 1):
            values.append(max(1, next_value))
            next_value = max(1, next_value // 2)
        return values_from_active(values, groups=groups)

    print("skewed masked case: shape mirrors DSV4 MTP verify; "
          "groups=32, b_gran_k=32 walks path-B (k32 fast path) "
          "with b.second shape [groups, N, K/32]")
    rows = []
    shapes = [
        ("gateup", 4096, 4096),
        ("down", 4096, 2048),
        # Non-DSV4 dimensions keep the same group/masked pattern but stress
        # different N/K ratios that can expose scheduler or scale-load cliffs.
        ("wide_n", 7168, 2048),
        ("wide_k", 4096, 7168),
    ]
    distributions = [
        # Uniform small-M cases validate that hint-aware BM32 selection does not
        # regress the common non-skew path.
        ("uniform_1", [1] * 32, 1),
        ("uniform_2", [2] * 32, 2),
        ("uniform_4", [4] * 32, 4),
        ("uniform_8", [8] * 32, 8),
        ("uniform_16", [16] * 32, 16),
        ("uniform_32", [32] * 32, 32),
        # Around the BM16 -> BM32 hot threshold.
        ("one_hot_15", values_from_active([15]), 1),
        ("one_hot_16", values_from_active([16]), 1),
        ("one_hot_17", values_from_active([17]), 1),
        ("one_hot_24", values_from_active([24]), 1),
        ("one_hot_32", values_from_active([32]), 1),
        ("one_hot_48", values_from_active([48]), 2),
        ("one_hot_64", values_from_active([64]), 2),
        ("one_hot_128", values_from_active([128]), 4),
        ("one_hot_214", values_from_active([214]), 1),
        ("one_hot_384", values_from_active([384]), 12),
        # Original MTP verify distributions copied from observed DP logs.
        ("mtp_dp2", skew_values(total=144, hot=50), 7),
        ("mtp_dp0", skew_values(total=195, hot=160), 7),
        ("mtp_dp4", skew_values(total=290, hot=214), 7),
        # Same total token count but different active/hotness patterns.
        ("mtp_144_hot96_a4", skew_values(total=144, hot=96, active=4), 7),
        ("mtp_144_hot96_a8", skew_values(total=144, hot=96, active=8), 7),
        ("mtp_195_hot96_a16", skew_values(total=195, hot=96, active=16), 7),
        ("mtp_290_hot160_a16", skew_values(total=290, hot=160, active=16), 7),
        ("mtp_384_hot256_a8", skew_values(total=384, hot=256, active=8), 12),
        ("mtp_512_hot384_a8", skew_values(total=512, hot=384, active=8), 16),
        # Multi-hot cases mimic router concentration on a few experts rather
        # than a single dominant expert.
        ("two_hot_64_64", values_from_active([64, 64]), 4),
        ("two_hot_128_64", values_from_active([128, 64]), 6),
        ("two_hot_160_96", values_from_active([160, 96]), 8),
        ("four_hot_32", values_from_repeated(32, active=4), 4),
        ("four_hot_64", values_from_repeated(64, active=4), 8),
        ("eight_hot_32", values_from_repeated(32, active=8), 8),
        ("eight_hot_64", values_from_repeated(64, active=8), 16),
        # Long tails stress compact masked scheduling and active-group scanning.
        ("longtail_128_a8", long_tail_values(hot=128, tail_start=64, active=8), 8),
        ("longtail_214_a8", long_tail_values(hot=214, tail_start=48, active=8), 7),
        ("longtail_256_a16", long_tail_values(hot=256, tail_start=64, active=16), 12),
        # Dense active but skewed cases can happen when all experts receive a
        # few tokens and one or two experts still become hot.
        (
            "dense_tail_hot64",
            values_from_active([64, 32, 16, 8] + [4] * 28),
            8,
        ),
        (
            "dense_tail_hot128",
            values_from_active([128, 64, 32, 16] + [4] * 28),
            8,
        ),
        (
            "dense_tail_hot214",
            values_from_active([214, 64, 32, 16] + [4] * 28),
            8,
        ),
    ]
    # for shape_name, n, k in shapes:
    #     for dist_name, masked_m_values, expected_m in distributions:
    #         rows.append(
    #             _masked_skew_benchmark_case(
    #                 f"{shape_name}_{dist_name}",
    #                 masked_m_values,
    #                 expected_m=expected_m,
    #                 n=n,
    #                 k=k,
    #                 max_m=1024,
    #                 b_gran_k=32,
    #             )
    #         )

    group24_shapes = [
        ("g24_m4096_n6144_k7168", 6144, 7168, 4096),
        ("g24_m4096_n7168_k3072", 7168, 3072, 4096),
    ]
    group24_distributions = [
        # Reproduce the warmup cliff that showed up as:
        # m=17, max_m=4096, n=6144, k=7168, num_groups=24.
        ("uniform_1", values_from_repeated(1, active=24, groups=24), 1),
        ("uniform_8", values_from_repeated(8, active=24, groups=24), 8),
        ("uniform_16", values_from_repeated(16, active=24, groups=24), 16),
        ("uniform_17", values_from_repeated(17, active=24, groups=24), 17),
        ("uniform_24", values_from_repeated(24, active=24, groups=24), 24),
        ("uniform_32", values_from_repeated(32, active=24, groups=24), 32),
        ("uniform_64", values_from_repeated(64, active=24, groups=24), 64),
        # Around the E8M0 direct-load threshold and BM16/BM32 transition.
        ("one_hot_16", values_from_active([16], groups=24), 16),
        ("one_hot_17", values_from_active([17], groups=24), 17),
        ("one_hot_32", values_from_active([32], groups=24), 32),
        ("one_hot_64", values_from_active([64], groups=24), 64),
        ("one_hot_128", values_from_active([128], groups=24), 128),
        ("one_hot_256", values_from_active([256], groups=24), 256),
        ("one_hot_512", values_from_active([512], groups=24), 512),
        ("two_hot_17", values_from_active([17, 17], groups=24), 17),
        ("two_hot_64_64", values_from_active([64, 64], groups=24), 64),
        ("two_hot_128_64", values_from_active([128, 64], groups=24), 128),
        ("four_hot_17", values_from_repeated(17, active=4, groups=24), 17),
        ("four_hot_32", values_from_repeated(32, active=4, groups=24), 32),
        ("four_hot_64", values_from_repeated(64, active=4, groups=24), 64),
        ("eight_hot_17", values_from_repeated(17, active=8, groups=24), 17),
        ("eight_hot_32", values_from_repeated(32, active=8, groups=24), 32),
        ("eight_hot_64", values_from_repeated(64, active=8, groups=24), 64),
        # MTP-like skew for 24 local groups, including the m=17 average.
        ("mtp_408_hot160_a8", skew_values(408, 160, active=8, groups=24), 17),
        ("mtp_408_hot256_a8", skew_values(408, 256, active=8, groups=24), 17),
        ("mtp_576_hot384_a8", skew_values(576, 384, active=8, groups=24), 24),
        ("mtp_768_hot512_a8", skew_values(768, 512, active=8, groups=24), 32),
        # Long tails and dense active tails stress compact scheduling when
        # active_groups is neither tiny nor fully uniform.
        (
            "longtail_128_a8",
            long_tail_values(128, tail_start=64, active=8, groups=24),
            17,
        ),
        (
            "longtail_256_a12",
            long_tail_values(256, tail_start=96, active=12, groups=24),
            24,
        ),
        (
            "dense_tail_hot17",
            values_from_active([17, 16, 8, 4] + [1] * 20, groups=24),
            17,
        ),
        (
            "dense_tail_hot64",
            values_from_active([64, 32, 16, 8] + [4] * 20, groups=24),
            17,
        ),
        (
            "dense_tail_hot128",
            values_from_active([128, 64, 32, 16] + [4] * 20, groups=24),
            24,
        ),
        (
            "dense_tail_hot256",
            values_from_active([256, 128, 64, 32] + [8] * 20, groups=24),
            32,
        ),
    ]
    for shape_name, n, k, max_m in group24_shapes:
        for dist_name, masked_m_values, expected_m in group24_distributions:
            rows.append(
                _masked_skew_benchmark_case(
                    f"{shape_name}_{dist_name}",
                    masked_m_values,
                    expected_m=expected_m,
                    n=n,
                    k=k,
                    max_m=max_m,
                    b_gran_k=32,
                )
            )

    # DSV4 EP 真实业务 shape mirror：m=256 + expected_m∈{1,2,3} + max_hint=None
    # （caller 没传 hint，business dispatch_output 不携带 max_hint/active_hint）。
    # 三组 shape：gateup(g24, n=6144, k=7168) / down(g24, n=7168, k=3072)
    # 业务比例：expected_m=2 占 71%, expected_m=3 占 18%, expected_m=1 占 11%。
    # 物理 m=256 但 caller 不知道每个 group 的真实 masked_m，masked_m_values 设
    # expected_m * groups 模拟 uniform 分布，用 pass_hints=False 让 host 走没有
    # hint 的 small-m candidate + simple_sched 路径。
    business_shapes = [
        ("biz_gateup_g24_m256_n6144_k7168", 6144, 7168, 256),
        ("biz_down_g24_m256_n7168_k3072", 7168, 3072, 256),
    ]
    business_distributions = [
        ("expected_m_1", values_from_repeated(1, active=24, groups=24), 1),
        ("expected_m_2", values_from_repeated(2, active=24, groups=24), 2),
        ("expected_m_3", values_from_repeated(3, active=24, groups=24), 3),
    ]
    for shape_name, n, k, max_m in business_shapes:
        for dist_name, masked_m_values, expected_m in business_distributions:
            rows.append(
                _masked_skew_benchmark_case(
                    f"{shape_name}_{dist_name}",
                    masked_m_values,
                    expected_m=expected_m,
                    n=n,
                    k=k,
                    max_m=max_m,
                    b_gran_k=32,
                    pass_hints=False,
                )
            )
    _print_skew_table(rows)

    # print("\nworst W4/FP8 speedup cases")
    # _print_skew_table(sorted(rows, key=lambda row: float(row["speedup"]))[:12])


if __name__ == "__main__":
    start_time = time.time()
    # if os.getenv("DG_W4_CONTIGUOUS_DIRECT_FP32_SCALE", "0") not in ("", "0"):
    #     test_sm90_fp8_fp4_contiguous()
    if os.getenv("DG_W4_MASKED_SKEW_CASES", "0") not in ("", "0"):
        test_sm90_fp8_fp4_masked_skew_cases()
    elif os.getenv("DG_W4_MASKED_DIRECT_FP32_SCALE", "0") not in ("", "0"):
        test_sm90_fp8_fp4_masked_direct_fp32_scale()
    else:
        test_sm90_fp8_fp4_masked()
    print(f"done in {time.time() - start_time:.2f}s")
