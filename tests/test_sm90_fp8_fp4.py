import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import cast_back_from_fp4, per_token_cast_to_fp4, per_token_cast_to_fp8


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


def _effective_bytes(groups: int, m_per_group: int, n: int, k: int, gran_k: int, *, fp8_b: bool) -> int:
    logical_m = groups * m_per_group
    scale_k = (k + gran_k - 1) // gran_k
    a_bytes = logical_m * k + logical_m * scale_k * 4
    b_data_bytes = groups * n * k if fp8_b else groups * n * (k // 2)
    b_scale_bytes = groups * n * scale_k * 4
    d_bytes = logical_m * n * 2
    return a_bytes + b_data_bytes + b_scale_bytes + d_bytes


def _sfb_alias_fits(block_m: int, k: int, gran_k: int) -> bool:
    # The 1d2d kernel aliases the SFB cache onto smem_d. This mirrors the
    # kernel-side capacity check and avoids device traps during autotune.
    return (k // gran_k) * 4 <= block_m * 2


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

    best_w4_elapsed = float("inf")
    best_w4_diff = float("inf")
    best_block_m = None
    best_block_n = None
    for block_m in (64, 128, 256):
        if m_per_group < block_m:
            continue
        if not _sfb_alias_fits(block_m, k, gran_k):
            continue
        for block_n in (64, 128, 256):
            d_w4 = torch.empty_like(ref)

            def run_w4(
                block_m: int = block_m,
                block_n: int = block_n,
                out: torch.Tensor = d_w4,
            ):
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
                    a,
                    b_w4,
                    out,
                    grouped_layout,
                    gran_k=gran_k,
                    compiled_dims="nk",
                    use_psum_layout=False,
                    block_m_override=block_m,
                    block_n_override=block_n,
                )

            try:
                run_w4()
            except RuntimeError:
                continue
            w4_diff = calc_diff(d_w4, ref)
            w4_elapsed = _time_cuda(run_w4)
            if w4_diff < 0.015 and w4_elapsed < best_w4_elapsed:
                best_w4_elapsed = w4_elapsed
                best_w4_diff = w4_diff
                best_block_m = block_m
                best_block_n = block_n

    if best_block_m is None or best_block_n is None:
        raise RuntimeError(f"No valid W4 candidate for groups={groups}, m/group={m_per_group}, n={n}, k={k}")

    fp8_threshold = 0.05
    assert best_w4_diff < 0.015
    assert fp8_diff < fp8_threshold

    w4_bytes = _effective_bytes(groups, m_per_group, n, k, gran_k, fp8_b=False)
    fp8_bytes = _effective_bytes(groups, m_per_group, n, k, gran_k, fp8_b=True)
    return {
        "groups": groups,
        "m_per_group": m_per_group,
        "n": n,
        "k": k,
        "w4_us": best_w4_elapsed * 1e6,
        "w4_gbps": w4_bytes / best_w4_elapsed / 1e9,
        "w4_diff": best_w4_diff,
        "fp8_us": fp8_elapsed * 1e6,
        "fp8_gbps": fp8_bytes / fp8_elapsed / 1e9,
        "fp8_diff": fp8_diff,
        "speedup": fp8_elapsed / best_w4_elapsed,
    }


def _print_markdown_table(rows: list[dict[str, float | int]]) -> None:
    print("groups | m/group | n | k | W4 us | W4 GB/s | W4 diff | FP8 us | FP8 GB/s | FP8 diff | Speedup")
    print("-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --")
    for row in rows:
        print(
            f"{row['groups']} | {row['m_per_group']} | {row['n']} | {row['k']} | "
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


def test_sm90_fp8_fp4_fallback() -> None:
    _require_sm90()
    torch.manual_seed(0)

    rows = []
    for groups in (8, 16, 24, 32):
        for m_per_group in (128, 256, 512, 1024):
            rows.append(_benchmark_case(groups, m_per_group, n=4096, k=7168))
    _print_markdown_table(rows)


def test_sm90_fp8_fp4_fallback_accuracy() -> None:
    _require_sm90()
    torch.manual_seed(1)

    cases = (
        (8, 128, 1024, 1024),
        (16, 128, 2048, 2048),
        (24, 256, 4096, 7168),
        (32, 512, 4096, 7168),
    )
    for groups, m_per_group, n, k in cases:
        w4_diff, fp8_diff = _accuracy_case(groups, m_per_group, n, k)
        assert w4_diff < 0.015, (
            f"W4 accuracy check failed for groups={groups}, m/group={m_per_group}, "
            f"n={n}, k={k}: diff={w4_diff}"
        )
        assert fp8_diff < 0.05, (
            f"FP8 accuracy check failed for groups={groups}, m/group={m_per_group}, "
            f"n={n}, k={k}: diff={fp8_diff}"
        )


if __name__ == "__main__":
    start_time = time.time()
    test_sm90_fp8_fp4_fallback()
    test_sm90_fp8_fp4_fallback_accuracy()
    print(f"done in {time.time() - start_time:.2f}s")
