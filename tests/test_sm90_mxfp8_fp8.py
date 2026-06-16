import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import per_token_cast_to_fp8


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SM90 MXFP8FP8 tests")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"SM90 MXFP8FP8 tests require sm_90, got sm_{major}x")


def _cast_back_from_fp8_1d(x: torch.Tensor, sf: torch.Tensor, gran_k: int) -> torch.Tensor:
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def _e8m0_from_fp32_pow2(sf: torch.Tensor) -> torch.Tensor:
    assert sf.dtype == torch.float32 or sf.dtype == torch.float
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _make_contiguous_case(groups: int, m_per_group: int, n: int, k: int):
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a = per_token_cast_to_fp8(a_ref, use_ue8m0=False, gran_k=128)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)
    a_dequant = _cast_back_from_fp8_1d(a[0], a[1], gran_k=128)
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for group_id in range(groups):
        start = group_id * m_per_group
        end = start + m_per_group
        b_dequant = _cast_back_from_fp8_1d(b_data[group_id], b_sf_fp32[group_id], gran_k=32)
        ref[start:end] = (a_dequant[start:end] @ b_dequant.t()).to(torch.bfloat16)
    return a, (b_data, _e8m0_from_fp32_pow2(b_sf_fp32)), grouped_layout, ref


def test_m_grouped_mxfp8_fp8_contiguous_e8m0_scale_accuracy():
    _require_sm90()
    groups, m_per_group, n, k = 2, 17, 48, 64
    a, b, grouped_layout, ref = _make_contiguous_case(groups, m_per_group, n, k)
    d = torch.empty_like(ref)

    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(a, b, d, grouped_layout)
    diff = calc_diff(d, ref)
    assert diff < 0.03


def test_m_grouped_mxfp8_fp8_masked_e8m0_scale_accuracy():
    _require_sm90()
    groups, max_m, n, k = 2, 32, 48, 64
    masked_m = torch.tensor([7, 19], device="cuda", dtype=torch.int32)
    a_ref = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, 1), device="cuda", dtype=torch.float32)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for group_id in range(groups):
        a_data[group_id], a_sf[group_id] = per_token_cast_to_fp8(
            a_ref[group_id], use_ue8m0=False, gran_k=128
        )
        b_data[group_id], b_sf_fp32[group_id] = per_token_cast_to_fp8(
            b_ref[group_id], use_ue8m0=True, gran_k=32
        )

    a = (a_data, a_sf)
    b = (b_data, _e8m0_from_fp32_pow2(b_sf_fp32))
    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf, gran_k=128)
    ref = torch.zeros((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    for group_id, valid_m in enumerate(masked_m.tolist()):
        b_dequant = _cast_back_from_fp8_1d(b_data[group_id], b_sf_fp32[group_id], gran_k=32)
        ref[group_id, :valid_m] = (a_dequant[group_id, :valid_m] @ b_dequant.t()).to(torch.bfloat16)

    d = torch.empty_like(ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m=max_m)
    diff = max(
        calc_diff(d[group_id, :valid_m], ref[group_id, :valid_m])
        for group_id, valid_m in enumerate(masked_m.tolist())
    )
    assert diff < 0.03
