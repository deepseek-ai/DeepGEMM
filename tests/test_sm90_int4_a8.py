"""Accuracy + benchmark for INT4-A8 (B = INT4 sym, A = FP8 e4m3) masked GEMM
on SM90, exercising the dedicated INT4-sym device path.

Pipeline:
  - B is symmetric INT4 (signed [-8, 7]) packed two nibbles per byte. The
    wire format is byte-identical to packed FP4 (kPackedFP4 == kInt8).
  - A is FP8 e4m3 with per-(row, K-block=128) fp32 scales.
  - SFB is per-(row, K-block=128) fp32 (path-A).
  - The kernel decodes B nibbles in registers via int4_symx4_to_e4m3x4
    instead of fp4x4_to_e4m3x4. This is selected via the new
    `b_is_int4_sym=True` argument on
    `m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma`.

Verification
------------
INT4-sym 的 16 个码点在 E4M3 中**无损可表示**，因此 INT4 kernel 与 fp32 ref
的差距应当极小（与 bf16 cast + fp32 累加噪声同量级，即 cos > 0.999）。这是
本测试的硬正确性指标 —— **cos_abs**。

作为对照，我们还跑一条 FP8(b_dequant) baseline：把 INT4 反量化回 fp32 后
再 re-quantize 成 fp8 喂进常规 FP8 kernel。这条路径自身要走一次 fp8 取整
（3-bit mantissa），相对 fp32 ref 会落到 ~cos 0.98 的 FP8 误差墙；INT4 与
它的差距正好反映这层 re-quant 噪声 —— **cos_eq**，仅做信息性记录，不参与
pass/fail 判定。

  cos_abs = cos(INT4-kernel out, fp32 INT4-dequant ref)   > 0.99    [硬指标]
  cos_eq  = cos(INT4-kernel out, FP8(b_dequant)-kernel)   info only

性能列与 test_sm90_fp8_fp4.py 一致（GB/s 用 effective bytes 模型）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.utils.math import (
    cast_back_from_int4_sym,
    per_token_cast_to_fp8,
    per_token_cast_to_int4_sym,
)


COS_ABS_THRESHOLD = 0.99   # INT4-sym is lossless in E4M3, must hit ~1.0
COS_EQ_INFO_ONLY = True    # cos vs FP8(b_dequant) baseline is informative,
                           # not a pass/fail signal: the FP8 baseline is the
                           # one with re-quant noise.


def _require_sm90() -> None:
    assert torch.cuda.is_available()
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(f"This test is intended for SM90, got sm_{major}x")


def _resolve_kernel():
    fn = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma", None)
    if fn is None:
        raise RuntimeError(
            "deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma not found. "
            "Rebuild the C++ extension after the INT4-sym wiring."
        )
    return fn


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


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(F.cosine_similarity(a, b, dim=0))


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _build_int4_a8_inputs(groups: int, m_per_group: int, max_m: int,
                          n: int, k: int, *, gran_k: int = 128):
    """Build one INT4-A8 problem with two parallel B representations.

    INT4 path:
      - b_int4: packed signed [-8, 7] nibbles, dtype int8, shape (G, n, k/2)
      - b_int4_sf: per-(n, K-block=128) fp32 scale, shape (G, n, k/128)

    FP8 baseline path (for equivalence check):
      - b_fp8: round-tripped fp8_e4m3, shape (G, n, k)
      - b_fp8_sf: per-(n, K-block=128) fp32 scale, shape (G, n, k/128)
        Built by dequantising INT4 to fp32 and re-quantising via
        per_token_cast_to_fp8 (use_ue8m0=False).
    """
    masked_m = torch.full((groups,), m_per_group, device="cuda", dtype=torch.int32)

    # ---- A: bf16 -> per-(row, 128-block) fp32-scale fp8_e4m3 ----
    a_ref = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    a_fp8 = torch.empty_like(a_ref, dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, k // gran_k), device="cuda", dtype=torch.float)
    for g in range(groups):
        a_fp8[g], a_sf[g] = per_token_cast_to_fp8(a_ref[g], use_ue8m0=False, gran_k=gran_k)

    # ---- B: bf16 -> per-(row, 128-block) INT4 sym ----
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
    b_int4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_int4_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    for g in range(groups):
        b_int4[g], b_int4_sf[g] = per_token_cast_to_int4_sym(b_ref[g], gran_k=gran_k)

    # ---- fp32 dequant references (for ground-truth matmul) ----
    group_idx_a = torch.arange(k, device="cuda") // gran_k
    a_dequant = a_fp8.float() * a_sf[..., group_idx_a]
    b_dequant = torch.empty((groups, n, k), device="cuda", dtype=torch.float)
    for g in range(groups):
        b_dequant[g] = cast_back_from_int4_sym(b_int4[g], b_int4_sf[g], gran_k=gran_k)

    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    for g in range(groups):
        valid_m = int(masked_m[g].item())
        if valid_m == 0:
            continue
        ref[g, :valid_m] = (a_dequant[g, :valid_m] @ b_dequant[g].t()).to(torch.bfloat16)

    # ---- FP8(b_dequant) baseline path (equivalence reference) ----
    b_fp8 = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    for g in range(groups):
        b_fp8[g], b_fp8_sf[g] = per_token_cast_to_fp8(
            b_dequant[g].to(torch.bfloat16), use_ue8m0=False, gran_k=gran_k,
        )

    return dict(
        masked_m=masked_m,
        a_fp8=a_fp8, a_sf=a_sf,
        b_int4=b_int4, b_int4_sf=b_int4_sf,
        b_fp8=b_fp8, b_fp8_sf=b_fp8_sf,
        ref=ref,
    )


def _run_int4_kernel(fn, case, m_per_group, gran_k=128):
    """Drive the FP4 masked entry with INT4-sym packed B + b_is_int4_sym=True."""
    a = (case["a_fp8"], case["a_sf"])
    # The kernel's B-first tensor is dtype-checked as kPackedFP4 (== kInt8);
    # our int8-packed INT4 fits the wire format directly.
    b = (case["b_int4"], case["b_int4_sf"])
    d = torch.empty_like(case["ref"])
    fn(
        a, b, d, case["masked_m"], m_per_group,
        gran_k=gran_k, gran_k_a=gran_k, gran_k_b=gran_k,
        b_is_int4_sym=True,
    )
    return d


def _run_fp8_baseline(case, m_per_group, gran_k=128):
    """Equivalence reference: FP8 fused kernel on (A_fp8, fp8(b_dequant))."""
    a = (case["a_fp8"], case["a_sf"])
    b = (case["b_fp8"], case["b_fp8_sf"])
    d = torch.empty_like(case["ref"])
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        a, b, d, case["masked_m"], m_per_group,
        recipe_a=(1, gran_k), recipe_b=(1, gran_k),
    )
    return d


def _per_group_cos_min(d, target, m_per_group):
    return min(
        _cos_sim(d[g, :m_per_group], target[g, :m_per_group])
        for g in range(d.shape[0])
    )


def _per_group_mae_max(d, target, m_per_group):
    return max(
        _max_abs_err(d[g, :m_per_group], target[g, :m_per_group])
        for g in range(d.shape[0])
    )


def _effective_masked_bytes(groups, m_per_group, n, k, a_gran_k, *,
                            fp8_b: bool, b_gran_k=None):
    """Bandwidth model identical to test_sm90_fp8_fp4._effective_masked_bytes
    (each masked group treats its valid m rows as the logical_m payload)."""
    b_gran_k = a_gran_k if b_gran_k is None else b_gran_k
    logical_m = groups * m_per_group
    a_scale_k = (k + a_gran_k - 1) // a_gran_k
    b_scale_k = (k + b_gran_k - 1) // b_gran_k
    a_bytes = logical_m * k + logical_m * a_scale_k * 4
    b_data_bytes = groups * n * k if fp8_b else groups * n * (k // 2)
    b_scale_bytes = groups * n * b_scale_k * 4
    d_bytes = logical_m * n * 2
    return a_bytes + b_data_bytes + b_scale_bytes + d_bytes


def _accuracy_case(fn, groups, m_per_group, n, k, *, gran_k=128, seed=0):
    torch.manual_seed(seed)
    max_m = max(m_per_group, 128)
    case = _build_int4_a8_inputs(groups, m_per_group, max_m, n, k, gran_k=gran_k)

    d_int4 = _run_int4_kernel(fn, case, m_per_group, gran_k=gran_k)
    d_fp8 = _run_fp8_baseline(case, m_per_group, gran_k=gran_k)

    cos_eq = _per_group_cos_min(d_int4, d_fp8, m_per_group)
    cos_abs = _per_group_cos_min(d_int4, case["ref"], m_per_group)
    mae_abs = _per_group_mae_max(d_int4, case["ref"], m_per_group)
    int4_us = _time_cuda(lambda: _run_int4_kernel(fn, case, m_per_group, gran_k=gran_k))
    fp8_us = _time_cuda(lambda: _run_fp8_baseline(case, m_per_group, gran_k=gran_k))

    int4_bytes = _effective_masked_bytes(groups, m_per_group, n, k, gran_k, fp8_b=False)
    fp8_bytes = _effective_masked_bytes(groups, m_per_group, n, k, gran_k, fp8_b=True)
    return dict(
        groups=groups, m_per_group=m_per_group, n=n, k=k,
        cos_eq=cos_eq, cos_abs=cos_abs, max_abs_err=mae_abs,
        int4_us=int4_us * 1e6, fp8_us=fp8_us * 1e6,
        int4_gbps=int4_bytes / int4_us / 1e9,
        fp8_gbps=fp8_bytes / fp8_us / 1e9,
        speedup=fp8_us / int4_us,
    )


def test_int4_a8_masked_accuracy() -> None:
    _require_sm90()
    fn = _resolve_kernel()
    print("INT4-A8 masked accuracy + perf vs FP8 (masked m_grouped)")
    print(f"  cos_abs = cos(INT4-kernel, fp32-ref)              > {COS_ABS_THRESHOLD:.3f}  -- pass/fail")
    print(f"  cos_eq  = cos(INT4-kernel, FP8(b_dequant)-kernel)             -- info only")
    print()
    print("groups | m/group | n | k | cos_abs | cos_eq | "
          "INT4 us | INT4 GB/s | FP8 us | FP8 GB/s | Speedup")
    print("-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --")

    pass_count = 0
    fail_count = 0
    failed_rows = []
    for groups in (8, 16, 32):
        for m_per_group in (1, 4, 8, 16, 32):
            for (n, k) in [(4096, 7168), (7168, 2048), (4096, 4096)]:
                row = _accuracy_case(fn, groups, m_per_group, n, k)
                ok_abs = row["cos_abs"] > COS_ABS_THRESHOLD
                ok = ok_abs
                pass_count += int(ok)
                fail_count += int(not ok)
                marker = "" if ok else f"  ## FAIL (cos_abs<={COS_ABS_THRESHOLD})"
                print(
                    f"{row['groups']} | {row['m_per_group']} | {row['n']} | "
                    f"{row['k']} | {row['cos_abs']:.4f} | {row['cos_eq']:.4f} | "
                    f"{row['int4_us']:.0f} | {row['int4_gbps']:.0f} | "
                    f"{row['fp8_us']:.0f} | {row['fp8_gbps']:.0f} | "
                    f"{row['speedup']:.2f}x{marker}"
                )
                if not ok:
                    failed_rows.append(row)
    print(f"\n{pass_count} passed, {fail_count} failed")
    if fail_count > 0:
        details = "; ".join(
            f"g={r['groups']} m={r['m_per_group']} n={r['n']} k={r['k']} "
            f"cos_abs={r['cos_abs']:.4f}"
            for r in failed_rows[:5]
        )
        raise AssertionError(
            f"{fail_count} INT4-sym cases failed cos_abs > {COS_ABS_THRESHOLD}: "
            + details + (" ..." if len(failed_rows) > 5 else "")
        )


# ---------------------------------------------------------------------------
# Skewed (uneven) masked-m benchmark — mirrors test_sm90_fp8_fp4 layout
# ---------------------------------------------------------------------------
# 在不均匀 mask 模式下度量 INT4 vs FP8 的 speedup。每条 case 用一组 per-group
# 的 masked_m_values（长度 == 32 个 group 槽，未激活槽 mask=0）。INT4 路径仍
# 走 b_is_int4_sym=True；FP8 baseline 用 cast_back_from_int4_sym 反量化后
# re-quant 到 fp8。bandwidth model 用每 group 的有效 m。
def _build_int4_a8_skew_inputs(masked_m_values, max_m, n, k, *, gran_k=128):
    groups = len(masked_m_values)
    masked_m = torch.tensor(masked_m_values, device="cuda", dtype=torch.int32)

    a_ref = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    a_fp8 = torch.empty_like(a_ref, dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, k // gran_k), device="cuda", dtype=torch.float)
    for g in range(groups):
        a_fp8[g], a_sf[g] = per_token_cast_to_fp8(a_ref[g], use_ue8m0=False, gran_k=gran_k)

    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
    b_int4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    # path-A bf16 sfb：与 fp8_fp4 测试同款，按 MN-major + tma_aligned_mn=align(N,8) 直构造。
    # **默认开启**，DG_W4_SCALE_B_BF16=0 时回退 fp32。
    use_bf16_b_sf = bool(int(os.getenv("DG_W4_SCALE_B_BF16", "1")))
    if use_bf16_b_sf:
        assert n % 8 == 0
        tma_aligned_n = (n + 7) // 8 * 8
        b_int4_sf = torch.empty_strided(
            (groups, n, k // gran_k),
            (tma_aligned_n * (k // gran_k), 1, tma_aligned_n),
            device="cuda",
            dtype=torch.bfloat16,
        )
        b_int4_sf_fp32 = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    else:
        b_int4_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
        b_int4_sf_fp32 = b_int4_sf
    for g in range(groups):
        b_int4[g], b_int4_sf_fp32[g] = per_token_cast_to_int4_sym(b_ref[g], gran_k=gran_k)
    if use_bf16_b_sf:
        b_int4_sf.copy_(b_int4_sf_fp32.to(torch.bfloat16))

    group_idx_a = torch.arange(k, device="cuda") // gran_k
    a_dequant = a_fp8.float() * a_sf[..., group_idx_a]
    b_dequant = torch.empty((groups, n, k), device="cuda", dtype=torch.float)
    for g in range(groups):
        b_dequant[g] = cast_back_from_int4_sym(b_int4[g], b_int4_sf_fp32[g], gran_k=gran_k)

    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    for g, valid_m in enumerate(masked_m_values):
        if valid_m > 0:
            ref[g, :valid_m] = (a_dequant[g, :valid_m] @ b_dequant[g].t()).to(torch.bfloat16)

    b_fp8 = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    for g in range(groups):
        b_fp8[g], b_fp8_sf[g] = per_token_cast_to_fp8(
            b_dequant[g].to(torch.bfloat16), use_ue8m0=False, gran_k=gran_k,
        )

    return dict(
        masked_m=masked_m,
        masked_m_values=masked_m_values,
        a_fp8=a_fp8, a_sf=a_sf,
        b_int4=b_int4, b_int4_sf=b_int4_sf,
        b_fp8=b_fp8, b_fp8_sf=b_fp8_sf,
        ref=ref,
    )


def _run_int4_kernel_skew(fn, case, expected_m, gran_k=128, masked_m_max_hint=None):
    a = (case["a_fp8"], case["a_sf"])
    b = (case["b_int4"], case["b_int4_sf"])
    d = torch.empty_like(case["ref"])
    kwargs = dict(
        gran_k=gran_k, gran_k_a=gran_k, gran_k_b=gran_k,
        b_is_int4_sym=True,
    )
    if masked_m_max_hint is not None:
        kwargs["masked_m_max_hint"] = masked_m_max_hint
    fn(a, b, d, case["masked_m"], expected_m, **kwargs)
    return d


def _run_fp8_baseline_skew(case, expected_m, gran_k=128):
    a = (case["a_fp8"], case["a_sf"])
    b = (case["b_fp8"], case["b_fp8_sf"])
    d = torch.empty_like(case["ref"])
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        a, b, d, case["masked_m"], expected_m,
        recipe_a=(1, gran_k), recipe_b=(1, gran_k),
    )
    return d


def _per_group_cos_min_skew(d, target, masked_m_values):
    cos = []
    for g, valid_m in enumerate(masked_m_values):
        if valid_m > 0:
            cos.append(_cos_sim(d[g, :valid_m], target[g, :valid_m]))
    return min(cos) if cos else 1.0


def _effective_masked_bytes_skew(masked_m_values, n, k, a_gran_k, *,
                                 fp8_b: bool, b_gran_k=None):
    """Bandwidth model identical to test_sm90_fp8_fp4 skew path: each active
    group contributes its valid_m rows to A/D and its full B+SFB tensor."""
    b_gran_k = a_gran_k if b_gran_k is None else b_gran_k
    sum_m = sum(masked_m_values)
    active_groups = sum(1 for v in masked_m_values if v > 0)
    a_scale_k = (k + a_gran_k - 1) // a_gran_k
    b_scale_k = (k + b_gran_k - 1) // b_gran_k
    a_bytes = sum_m * k + sum_m * a_scale_k * 4
    b_data_bytes = active_groups * n * (k if fp8_b else k // 2)
    b_scale_bytes = active_groups * n * b_scale_k * 4
    d_bytes = sum_m * n * 2
    return a_bytes + b_data_bytes + b_scale_bytes + d_bytes


def _masked_skew_benchmark_case(name, masked_m_values, expected_m, n, k, *,
                                max_m=1024, gran_k=128):
    fn = _resolve_kernel()
    assert max(masked_m_values) <= max_m
    case = _build_int4_a8_skew_inputs(masked_m_values, max_m, n, k, gran_k=gran_k)

    # 透传 masked_m_max_hint：caller 告知 hot group 大小，host 据此选 BM。
    # expected_m 保持原 distribution 平均值（与生产语义一致）。FP8 baseline
    # 没有 hint API 但 expected_m 接受任意值，让它走 max 以反映"两条路径都
    # 按 hot 调度"下的真实算法差距。
    masked_m_max_hint = max(masked_m_values)
    gemm_expected_m = expected_m
    fp8_expected_m = max(masked_m_max_hint, expected_m)

    d_int4 = _run_int4_kernel_skew(fn, case, gemm_expected_m, gran_k=gran_k,
                                    masked_m_max_hint=masked_m_max_hint)
    d_fp8 = _run_fp8_baseline_skew(case, fp8_expected_m, gran_k=gran_k)

    cos_abs = _per_group_cos_min_skew(d_int4, case["ref"], masked_m_values)
    cos_eq = _per_group_cos_min_skew(d_int4, d_fp8, masked_m_values)
    int4_us = _time_cuda(lambda: _run_int4_kernel_skew(
        fn, case, gemm_expected_m, gran_k=gran_k,
        masked_m_max_hint=masked_m_max_hint))
    fp8_us = _time_cuda(lambda: _run_fp8_baseline_skew(case, fp8_expected_m, gran_k=gran_k))

    int4_bytes = _effective_masked_bytes_skew(masked_m_values, n, k, gran_k, fp8_b=False)
    fp8_bytes = _effective_masked_bytes_skew(masked_m_values, n, k, gran_k, fp8_b=True)
    return dict(
        case=name,
        groups=len(masked_m_values),
        expected_m=expected_m,
        masked_m_hint=masked_m_max_hint,
        sum_m=sum(masked_m_values),
        masked_max=max(masked_m_values),
        active_groups=sum(1 for v in masked_m_values if v > 0),
        n=n, k=k,
        cos_abs=cos_abs, cos_eq=cos_eq,
        int4_us=int4_us * 1e6, fp8_us=fp8_us * 1e6,
        int4_gbps=int4_bytes / int4_us / 1e9,
        fp8_gbps=fp8_bytes / fp8_us / 1e9,
        speedup=fp8_us / int4_us,
    )


def _print_skew_table(rows):
    print("case | groups | exp_m_avg | hint | sum_m | max_m | active | n | k | "
          "cos_abs | cos_eq | INT4 us | INT4 GB/s | FP8 us | FP8 GB/s | Speedup")
    print("-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --")
    for r in rows:
        print(
            f"{r['case']} | {r['groups']} | {r['expected_m']} | {r['masked_m_hint']} | {r['sum_m']} | "
            f"{r['masked_max']} | {r['active_groups']} | {r['n']} | {r['k']} | "
            f"{r['cos_abs']:.4f} | {r['cos_eq']:.4f} | "
            f"{r['int4_us']:.0f} | {r['int4_gbps']:.0f} | "
            f"{r['fp8_us']:.0f} | {r['fp8_gbps']:.0f} | "
            f"{r['speedup']:.2f}x"
        )


def test_int4_a8_masked_skew_cases() -> None:
    _require_sm90()
    torch.manual_seed(4)

    def skew_values(total: int, hot: int, active: int = 8) -> list[int]:
        # 32 个槽位，前 active 个非零（第 0 个为 hot，其余分摊 total-hot），
        # 剩余 32-active 个槽 mask=0。
        assert active >= 1 and total >= hot
        values = [0] * 32
        values[0] = hot
        remaining = total - hot
        for idx in range(1, active):
            share = (remaining + active - idx - 1) // (active - idx)
            values[idx] = share
            remaining -= share
        assert sum(values) == total
        return values

    print("INT4-A8 masked SKEW perf vs FP8 (uneven masked_m distribution)")
    print(f"  cos_abs > {COS_ABS_THRESHOLD:.3f}  -- pass/fail")
    print()
    rows = []
    shapes = [
        ("gateup", 4096, 4096),
        ("down", 4096, 2048),
    ]
    distributions = [
        ("uniform_1", [1] * 32, 1),
        ("uniform_8", [8] * 32, 8),
        ("mtp_dp2", skew_values(total=144, hot=50), 7),
        ("mtp_dp0", skew_values(total=195, hot=160), 7),
        ("mtp_dp4", skew_values(total=290, hot=214), 7),
        ("one_hot_214", [214] + [0] * 31, 1),
    ]
    pass_count = 0
    fail_count = 0
    for shape_name, n, k in shapes:
        for dist_name, masked_m_values, expected_m in distributions:
            row = _masked_skew_benchmark_case(
                f"{shape_name}_{dist_name}",
                masked_m_values,
                expected_m=expected_m,
                n=n, k=k, max_m=1024,
            )
            ok = row["cos_abs"] > COS_ABS_THRESHOLD
            pass_count += int(ok)
            fail_count += int(not ok)
            rows.append(row)
    _print_skew_table(rows)
    print(f"\n{pass_count} passed, {fail_count} failed")
    if fail_count > 0:
        raise AssertionError(f"{fail_count} skew cases failed cos_abs > {COS_ABS_THRESHOLD}")


if __name__ == "__main__":
    start_time = time.time()
    _require_sm90()
    if os.getenv("DG_W4_MASKED_SKEW_CASES", "0") not in ("", "0"):
        test_int4_a8_masked_skew_cases()
    else:
        test_int4_a8_masked_accuracy()
    print(f"\ndone in {time.time() - start_time:.2f}s")
