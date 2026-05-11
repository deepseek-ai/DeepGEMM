import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import cast_back_from_fp4, per_token_cast_to_fp4, per_token_cast_to_fp8


def _align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


def _cast_back_from_fp8_1d(x: torch.Tensor, sf: torch.Tensor, gran_k: int = 128) -> torch.Tensor:
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def _require_sm90() -> None:
    assert torch.cuda.is_available()
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(f"This fallback test is intended for SM90, got sm_{major}x")


def _time_cuda(fn, warmup: int = 5, iters: int = 20) -> float:
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


def _normal_case(m: int, n: int, k: int, gran_k: int = 128) -> None:
    a_ref_src = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref_src, use_ue8m0=False, gran_k=gran_k)
    b = per_token_cast_to_fp4(b_ref_src, use_ue8m0=True, gran_k=gran_k)
    c = torch.zeros((m, n), device="cuda", dtype=torch.float)
    d = torch.empty((m, n), device="cuda", dtype=torch.float)

    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=gran_k)
    b_dequant = cast_back_from_fp4(b[0], b[1], gran_k=gran_k)
    b_fp8 = per_token_cast_to_fp8(b_dequant, use_ue8m0=False, gran_k=gran_k)
    ref = (a_dequant @ b_dequant.t()).to(torch.bfloat16)

    def run():
        deep_gemm.fp8_gemm_nt(
            a,
            b_fp8,
            d,
            c=c,
            recipe_a=(1, gran_k),
            recipe_b=(1, gran_k),
        )

    d_fp8 = torch.empty_like(d)
    d_fused = torch.empty_like(d)

    def run_fp8():
        deep_gemm.fp8_gemm_nt(
            a,
            b_fp8,
            d_fp8,
            c=c,
            recipe_a=(1, gran_k),
            recipe_b=(1, gran_k),
        )

    d_fused = torch.empty_like(d)

    def run_fused():
        deep_gemm.fp8_fp4_gemm_nt_sm90_cuda_core(
            a,
            b,
            d_fused,
            c=c,
            gran_k=gran_k,
        )

    d_fused_wgmma = torch.empty_like(d)

    def run_fused_wgmma():
        deep_gemm.fp8_fp4_gemm_nt_sm90_fused_wgmma(
            a,
            b,
            d_fused_wgmma,
            c=c,
            gran_k=gran_k,
        )

    run()
    run_fp8()
    if hasattr(deep_gemm, "fp8_fp4_gemm_nt_sm90_cuda_core"):
        run_fused()
    if hasattr(deep_gemm, "fp8_fp4_gemm_nt_sm90_fused_wgmma"):
        run_fused_wgmma()
    fallback_diff = calc_diff(d, ref)
    fp8_diff = calc_diff(d_fp8, ref)
    fallback_vs_fp8_diff = calc_diff(d, d_fp8)
    fused_diff = calc_diff(d_fused, ref) if hasattr(deep_gemm, "fp8_fp4_gemm_nt_sm90_cuda_core") else None
    fused_wgmma_diff = (
        calc_diff(d_fused_wgmma, ref)
        if hasattr(deep_gemm, "fp8_fp4_gemm_nt_sm90_fused_wgmma")
        else None
    )
    fallback_elapsed = _time_cuda(run)
    fp8_elapsed = _time_cuda(run_fp8)
    fused_elapsed = (
        _time_cuda(run_fused)
        if hasattr(deep_gemm, "fp8_fp4_gemm_nt_sm90_cuda_core")
        else None
    )
    fused_wgmma_elapsed = (
        _time_cuda(run_fused_wgmma)
        if hasattr(deep_gemm, "fp8_fp4_gemm_nt_sm90_fused_wgmma")
        else None
    )
    fallback_tflops = 2 * m * n * k / fallback_elapsed / 1e12
    fp8_tflops = 2 * m * n * k / fp8_elapsed / 1e12
    fused_tflops = (
        2 * m * n * k / fused_elapsed / 1e12 if fused_elapsed is not None else None
    )
    fused_wgmma_tflops = (
        2 * m * n * k / fused_wgmma_elapsed / 1e12
        if fused_wgmma_elapsed is not None
        else None
    )
    slowdown = fallback_elapsed / fp8_elapsed
    message = (
        f"normal m={m} n={n} k={k}:\n"
        f"  fallback end2end: diff={fallback_diff:.6f}, "
        f"time={fallback_elapsed * 1e6:.1f} us, {fallback_tflops:.2f} TFLOPS\n"
        f"  pure fp8 gemm:    diff={fp8_diff:.6f}, "
        f"time={fp8_elapsed * 1e6:.1f} us, {fp8_tflops:.2f} TFLOPS\n"
        f"  fallback vs fp8:  diff={fallback_vs_fp8_diff:.6f}\n"
        f"  slowdown:         {slowdown:.2f}x"
    )
    if fused_elapsed is not None:
        message += (
            f"\n  cuda-core fused:  diff={fused_diff:.6f}, "
            f"time={fused_elapsed * 1e6:.1f} us, {fused_tflops:.2f} TFLOPS, "
            f"slowdown_vs_fp8={fused_elapsed / fp8_elapsed:.2f}x"
        )
    if fused_wgmma_elapsed is not None:
        message += (
            f"\n  fused wgmma:      diff={fused_wgmma_diff:.6f}, "
            f"time={fused_wgmma_elapsed * 1e6:.1f} us, {fused_wgmma_tflops:.2f} TFLOPS, "
            f"slowdown_vs_fp8={fused_wgmma_elapsed / fp8_elapsed:.2f}x"
        )
    print(message)
    assert fallback_diff < 0.015
    assert fp8_diff < 0.015
    assert fallback_vs_fp8_diff < 1e-6
    if fused_diff is not None:
        assert fused_diff < 0.015
    if fused_wgmma_diff is not None:
        assert fused_wgmma_diff < 0.015


def _m_grouped_contiguous_case(
    num_groups: int,
    m_per_group: int | None,
    n: int,
    k: int,
    gran_k: int = 128,
    group_sizes: list[int] | None = None,
    use_psum_layout: bool = False,
    autotune_fused: bool = True,
) -> None:
    if group_sizes is None:
        assert m_per_group is not None
        group_sizes = [m_per_group] * num_groups
    assert len(group_sizes) == num_groups
    print(
        f"starting m_grouped groups={num_groups} n={n} k={k} "
        f"layout={'psum' if use_psum_layout else 'per-row'} "
        f"sizes={group_sizes} autotune={autotune_fused}",
        flush=True,
    )

    block_m_alignment = 128
    if use_psum_layout:
        group_starts: list[int] = []
        group_ends: list[int] = []
        cursor = 0
        for size in group_sizes:
            group_starts.append(cursor)
            cursor += size
            group_ends.append(cursor)
            cursor = _align_up(cursor, block_m_alignment)
        m = group_ends[-1] if group_ends else 0
        grouped_layout = torch.tensor(group_ends, device="cuda", dtype=torch.int32)
    else:
        assert all(size % block_m_alignment == 0 for size in group_sizes)
        m = sum(group_sizes)
        group_starts = []
        cursor = 0
        layout_chunks = []
        for group_id, size in enumerate(group_sizes):
            group_starts.append(cursor)
            cursor += size
            if size > 0:
                layout_chunks.append(
                    torch.full((size,), group_id, device="cuda", dtype=torch.int32)
                )
        grouped_layout = torch.cat(layout_chunks) if layout_chunks else torch.empty((0,), device="cuda", dtype=torch.int32)

    a_ref_src = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref_src, use_ue8m0=False, gran_k=gran_k)
    b_fp4 = torch.empty((num_groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_sf = torch.empty((num_groups, n, k // gran_k), device="cuda", dtype=torch.float)
    for group_id in range(num_groups):
        b_fp4[group_id], b_sf[group_id] = per_token_cast_to_fp4(
            b_ref_src[group_id], use_ue8m0=True, gran_k=gran_k
        )
    b = (b_fp4, b_sf)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=gran_k)
    b_fp8_data = torch.empty((num_groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((num_groups, n, k // gran_k), device="cuda", dtype=torch.float)
    ref = torch.empty_like(d)
    for group_id in range(num_groups):
        b_dequant = cast_back_from_fp4(b[0][group_id], b[1][group_id], gran_k=gran_k)
        b_fp8_data[group_id], b_fp8_sf[group_id] = per_token_cast_to_fp8(
            b_dequant, use_ue8m0=False, gran_k=gran_k
        )
        start = group_starts[group_id]
        end = (
            _align_up(group_ends[group_id], block_m_alignment)
            if use_psum_layout and group_id + 1 < num_groups
            else group_ends[group_id]
        ) if use_psum_layout else start + group_sizes[group_id]
        end = min(end, m)
        if start == end:
            continue
        ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)
    b_fp8 = (b_fp8_data, b_fp8_sf)

    def run():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a,
            b_fp8,
            d,
            grouped_layout,
            recipe_a=(1, gran_k),
            recipe_b=(1, gran_k),
            use_psum_layout=use_psum_layout,
            expected_m_for_psum_layout=m if use_psum_layout else None,
        )

    d_fp8 = torch.empty_like(d)
    d_fused = torch.empty_like(d)

    def run_fp8():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a,
            b_fp8,
            d_fp8,
            grouped_layout,
            recipe_a=(1, gran_k),
            recipe_b=(1, gran_k),
            use_psum_layout=use_psum_layout,
            expected_m_for_psum_layout=m if use_psum_layout else None,
        )

    def run_fused_wgmma(
        block_m_override: int | None = None,
        block_n_override: int | None = None,
    ):
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
            a,
            b,
            d_fused,
            grouped_layout,
            gran_k=gran_k,
            compiled_dims="nk",
            use_psum_layout=use_psum_layout,
            expected_m_for_psum_layout=m if use_psum_layout else None,
            block_m_override=block_m_override,
            block_n_override=block_n_override,
        )

    print("  running fallback...", flush=True)
    run()
    print("  running pure fp8...", flush=True)
    run_fp8()
    print("  running fused wgmma...", flush=True)
    run_fused_wgmma()
    fallback_diff = calc_diff(d, ref)
    fp8_diff = calc_diff(d_fp8, ref)
    fused_diff = calc_diff(d_fused, ref)
    fallback_vs_fp8_diff = calc_diff(d, d_fp8)
    fused_vs_fp8_diff = calc_diff(d_fused, d_fp8)
    fallback_elapsed = _time_cuda(run)
    fp8_elapsed = _time_cuda(run_fp8)
    fused_elapsed = _time_cuda(run_fused_wgmma)
    best_fused = (fused_elapsed, None, fused_diff)
    if autotune_fused:
        # Psum grouped_layout is encoded with 128-row alignment, so only BLOCK_M=128
        # is semantically valid. Sweep BLOCK_N only for psum.
        block_m_candidates = (128,) if use_psum_layout else (64, 128, 256)
        block_n_candidates = (64, 128, 256)
        for block_m in block_m_candidates:
            if m < block_m:
                continue
            if any(size > 0 and size < block_m for size in group_sizes) and not use_psum_layout:
                continue
            for block_n in block_n_candidates:
                d_candidate = torch.empty_like(d)

                def run_candidate(
                    block_m: int = block_m,
                    block_n: int = block_n,
                    out: torch.Tensor = d_candidate,
                ):
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
                        a,
                        b,
                        out,
                        grouped_layout,
                        gran_k=gran_k,
                        compiled_dims="nk",
                        use_psum_layout=use_psum_layout,
                        expected_m_for_psum_layout=m if use_psum_layout else None,
                        block_m_override=block_m,
                        block_n_override=block_n,
                    )

                try:
                    run_candidate()
                except RuntimeError as exc:
                    print(
                        f"  fused autotune block_m={block_m} block_n={block_n}: "
                        f"skipped ({exc})"
                    )
                    continue
                candidate_diff = calc_diff(d_candidate, ref)
                candidate_elapsed = _time_cuda(run_candidate)
                candidate_tflops = 2 * m * n * k / candidate_elapsed / 1e12
                print(
                    f"  fused autotune block_m={block_m} block_n={block_n}: "
                    f"diff={candidate_diff:.6f}, "
                    f"time={candidate_elapsed * 1e6:.1f} us, {candidate_tflops:.2f} TFLOPS"
                )
                if candidate_diff < 0.015 and candidate_elapsed < best_fused[0]:
                    best_fused = (candidate_elapsed, (block_m, block_n), candidate_diff)
                    d_fused.copy_(d_candidate)
        fused_elapsed, best_tile, fused_diff = best_fused
        fused_vs_fp8_diff = calc_diff(d_fused, d_fp8)
    else:
        best_tile = None
    fallback_tflops = 2 * m * n * k / fallback_elapsed / 1e12
    fp8_tflops = 2 * m * n * k / fp8_elapsed / 1e12
    fused_tflops = 2 * m * n * k / fused_elapsed / 1e12
    slowdown = fallback_elapsed / fp8_elapsed
    fused_slowdown = fused_elapsed / fp8_elapsed
    print(
        f"m_grouped groups={num_groups} m={m} n={n} k={k} "
        f"layout={'psum' if use_psum_layout else 'per-row'} sizes={group_sizes}:\n"
        f"  fallback end2end: diff={fallback_diff:.6f}, "
        f"time={fallback_elapsed * 1e6:.1f} us, {fallback_tflops:.2f} TFLOPS\n"
        f"  pure fp8 gemm:    diff={fp8_diff:.6f}, "
        f"time={fp8_elapsed * 1e6:.1f} us, {fp8_tflops:.2f} TFLOPS\n"
        f"  fused wgmma:      diff={fused_diff:.6f}, "
        f"time={fused_elapsed * 1e6:.1f} us, {fused_tflops:.2f} TFLOPS, "
        f"slowdown_vs_fp8={fused_slowdown:.2f}x"
        f"{'' if best_tile is None else f', best_tile={best_tile}'}\n"
        f"  fallback vs fp8:  diff={fallback_vs_fp8_diff:.6f}\n"
        f"  fused vs fp8:     diff={fused_vs_fp8_diff:.6f} "
        f"(expected: fused keeps original FP4 scales)\n"
        f"  slowdown:         {slowdown:.2f}x"
    )
    # Psum edge cases with few effective rows can amplify the FP4->FP8 requant
    # baseline error. Keep fused correctness strict; relax only the requantized
    # fallback/pure-FP8 baseline sanity threshold.
    requant_threshold = 0.25 if use_psum_layout else 0.05
    assert fallback_diff < requant_threshold
    assert fp8_diff < requant_threshold
    assert fused_diff < 0.015
    assert fallback_vs_fp8_diff < 1e-6


def test_sm90_fp8_fp4_fallback() -> None:
    _require_sm90()
    torch.manual_seed(0)

    # Small enough for quick CI/local validation, large enough to exercise SM90 FP8 GEMM.
    _normal_case(m=128, n=1024, k=1024)
    _m_grouped_contiguous_case(num_groups=1, m_per_group=128, n=1024, k=1024)
    _m_grouped_contiguous_case(num_groups=4, m_per_group=128, n=1024, k=1024)
    _m_grouped_contiguous_case(num_groups=8, m_per_group=128, n=512, k=1024)
    _m_grouped_contiguous_case(
        num_groups=4, m_per_group=256, n=2048, k=1024, autotune_fused=True
    )
    _m_grouped_contiguous_case(
        num_groups=4,
        m_per_group=None,
        n=1024,
        k=1024,
        group_sizes=[96, 0, 160, 64],
        use_psum_layout=True,
    )
    _m_grouped_contiguous_case(
        num_groups=5,
        m_per_group=None,
        n=512,
        k=2048,
        group_sizes=[0, 128, 32, 256, 48],
        use_psum_layout=True,
        autotune_fused=True,
    )
    # Psum stability coverage:
    # - aligned groups: sanity baseline where psum should behave like regular contiguous groups
    # - leading/trailing zero groups: scheduler fallthrough and empty group handling
    # - tiny partial groups: mixed valid/invalid rows inside a BLOCK_M tile
    # - larger K/N: exercise the 1d2d decode pipeline and smem SFB cache over more K blocks
    _m_grouped_contiguous_case(
        num_groups=3,
        m_per_group=None,
        n=512,
        k=1024,
        group_sizes=[128, 128, 128],
        use_psum_layout=True,
    )
    _m_grouped_contiguous_case(
        num_groups=4,
        m_per_group=None,
        n=512,
        k=1024,
        group_sizes=[0, 64, 128, 0],
        use_psum_layout=True,
    )
    _m_grouped_contiguous_case(
        num_groups=4,
        m_per_group=None,
        n=768,
        k=1024,
        group_sizes=[1, 127, 129, 1],
        use_psum_layout=True,
    )
    _m_grouped_contiguous_case(
        num_groups=6,
        m_per_group=None,
        n=1024,
        k=2048,
        group_sizes=[64, 0, 64, 0, 192, 32],
        use_psum_layout=True,
    )
    _m_grouped_contiguous_case(
        num_groups=4,
        m_per_group=None,
        n=1024,
        k=1024,
        group_sizes=[96, 160, 0, 64],
        use_psum_layout=True,
        autotune_fused=True,
    )


if __name__ == "__main__":
    start_time = time.time()
    test_sm90_fp8_fp4_fallback()
    print(f"done in {time.time() - start_time:.2f}s")
