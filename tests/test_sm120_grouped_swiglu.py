import pytest
import torch

import deep_gemm


@pytest.mark.parametrize("valid_m,n", [
    ((4, 4, 4, 112), 128),
    ((31, 31, 31, 31), 65),
    ((0, 7, 23, 64), 128),
])
def test_sm120_grouped_bf16_swiglu_masked(valid_m, n):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU required")

    groups, m, k = 4, 128, 256
    torch.manual_seed(42)
    a = torch.randn((groups, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((groups, 2 * n, k), device="cuda", dtype=torch.bfloat16) / k**0.5
    masked_m = torch.tensor(valid_m, device="cuda", dtype=torch.int32)
    l1 = torch.empty((groups, m, 2 * n), device="cuda", dtype=torch.bfloat16)
    actual = torch.full((groups, m, n), float("nan"), device="cuda", dtype=torch.bfloat16)

    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, l1, masked_m, max(valid_m))
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, masked_m, max(valid_m))

    for group, rows in enumerate(valid_m):
        if rows:
            expected = (torch.nn.functional.silu(l1[group, :rows, :n].float()) *
                        l1[group, :rows, n:].float()).to(torch.bfloat16)
            torch.testing.assert_close(actual[group, :rows], expected, rtol=1e-2, atol=2e-3)
        assert torch.isnan(actual[group, rows:]).all()


def test_sm120_grouped_bf16_swiglu_rejects_wrong_weight_shape():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU required")

    a = torch.empty((2, 32, 128), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((2, 127, 128), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((2, 32, 64), device="cuda", dtype=torch.bfloat16)
    masked_m = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, d, masked_m, 8)
