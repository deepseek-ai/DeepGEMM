import sys
from pathlib import Path
from unittest import mock
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Mock compiled C extension modules so pure-Python utility tests can run
mock_c = mock.MagicMock()
mock_loc = mock.MagicMock()
mock_loc.get_num_locality_domains.return_value = 2
mock_c.locality_domain = mock_loc

sys.modules["deep_gemm._C"] = mock_c
sys.modules["deep_gemm._C.locality_domain"] = mock_loc

import torch

from deep_gemm.utils.math import (
    ceil_div,
    align,
    ceil_to_ue8m0,
    pack_ue8m0_to_int,
    unpack_ue8m0_from_int,
    per_custom_dims_cast_to_fp8,
    cast_back_from_fp8,
    cast_back_from_fp4,
)
from deep_gemm.locality_domain import localize


def test_ceil_div_and_align():
    assert ceil_div(10, 3) == 4
    assert ceil_div(9, 3) == 3
    assert ceil_div(0, 5) == 0
    assert align(10, 4) == 12
    assert align(12, 4) == 12
    assert align(1, 128) == 128


def test_ceil_to_ue8m0():
    # UE8M0 rounds up to nearest power-of-two exponent (mantissa 0)
    x = torch.tensor([1.25, 0.75, 2.0, 3.5], dtype=torch.float32)
    ue8m0 = ceil_to_ue8m0(x)
    # 1.25 -> 2.0; 0.75 -> 1.0; 2.0 -> 2.0; 3.5 -> 4.0
    expected = torch.tensor([2.0, 1.0, 2.0, 4.0], dtype=torch.float32)
    assert torch.allclose(ue8m0, expected)


def test_pack_and_unpack_ue8m0_contiguous_and_non_contiguous():
    # Valid UE8M0 floats (powers of two, mantissa is 0)
    # Shape (2, 4), non-contiguous
    raw = torch.tensor([[1.0, 2.0, 4.0, 8.0], [0.5, 1.0, 2.0, 4.0]], dtype=torch.float32)
    # Create non-contiguous sliced tensor
    sliced = torch.cat([raw, raw], dim=1)[:, :4]
    assert not sliced.is_contiguous() or sliced.stride(0) != 4

    packed = pack_ue8m0_to_int(sliced)
    assert packed.shape == (2, 1)
    assert packed.dtype == torch.int32

    unpacked = unpack_ue8m0_from_int(packed)
    assert unpacked.shape == (2, 4)
    assert torch.allclose(unpacked, raw)


def test_per_custom_dims_cast_to_fp8_preserves_size_one_dim():
    # Tensor with shape (1, 128), requesting scale factor along dim 0
    x = torch.randn(1, 128, dtype=torch.float32)
    x_fp8, sf = per_custom_dims_cast_to_fp8(x, dims=(0,), use_ue8m0=False)
    # sf should retain shape (1,), NOT collapsed to scalar ()
    assert sf.shape == (1,)
    assert x_fp8.shape == (1, 128)

    # Tensor with shape (2, 1, 128), requesting dims (0, 1)
    x3d = torch.randn(2, 1, 128, dtype=torch.float32)
    x3d_fp8, sf3d = per_custom_dims_cast_to_fp8(x3d, dims=(0, 1), use_ue8m0=False)
    assert sf3d.shape == (2, 1)
    assert x3d_fp8.shape == (2, 1, 128)


def test_cast_back_from_fp8_1d_and_2d_sf():
    x = torch.ones(2, 128, dtype=torch.float32)
    x_fp8 = x.to(torch.float8_e4m3fn)
    # 2D sf
    sf_2d = torch.full((2, 1), 2.0, dtype=torch.float32)
    res_2d = cast_back_from_fp8(x_fp8, sf_2d, gran_k=128)
    assert res_2d.shape == (2, 128)
    assert torch.allclose(res_2d, torch.full((2, 128), 2.0))

    # 1D sf (e.g. per-channel or single row flattened)
    sf_1d = torch.full((1,), 3.0, dtype=torch.float32)
    res_1d = cast_back_from_fp8(x_fp8, sf_1d, gran_k=128)
    assert res_1d.shape == (2, 128)
    assert torch.allclose(res_1d, torch.full((2, 128), 3.0))


def test_localize_zero_dim_assert():
    scalar_tensor = torch.tensor(1.0)
    with pytest.raises(AssertionError, match="Input tensor must have at least 1 dimension"):
        localize(scalar_tensor)
