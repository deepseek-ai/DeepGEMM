import torch
from typing import Tuple


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    assert (x.view(torch.int) & ((1 << 23) - 1) == 0).all()
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    return x_fp8, pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


def per_channel_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(0) % gran_k == 0
    m, n = x.shape
    x_view = x.view(-1, gran_k, n)
    x_amax = x_view.abs().float().amax(dim=1).view(-1, n).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((align(m, gran_k), align(n, gran_k)), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple, use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries) 
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(x: torch.Tensor, use_ue8m0: bool, gran_k: int = 128,
                          use_packed_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    assert n % 2 == 0
    assert not use_packed_ue8m0 or use_ue8m0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)  # int8, (m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)  # int8
    return packed[:, :n // 2].contiguous(), pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf


def transpose_packed_fp4(a: torch.Tensor) -> torch.Tensor:
    assert a.dtype == torch.int8
    assert a.dim() == 2
    m, n2 = a.shape
    n = n2 * 2
    assert (m % 2) == 0
    lo = a & 0x0F
    hi = (a >> 4) & 0x0F
    codes = torch.empty((m, n), device=a.device, dtype=torch.int8)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    codes_t = codes.transpose(0, 1).contiguous()
    codes2 = codes_t.view(n, m // 2, 2)
    out = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return out.contiguous()


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float)
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def unpack_ue8m0_from_int(packed_sf: torch.Tensor) -> torch.Tensor:
    return (packed_sf.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def cast_back_from_fp4(packed: torch.Tensor, sf: torch.Tensor, gran_k: int = 128,
                       use_packed_ue8m0: bool = False) -> torch.Tensor:
    m, n2 = packed.shape
    n = n2 * 2
    if use_packed_ue8m0:
        sf = unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x_dequantized = _dequantize_from_fp4_e2m1(unpacked)
    group_idx = torch.arange(n, device=packed.device) // gran_k
    x_restored = x_dequantized * sf[:, group_idx]
    return x_restored


# ---------------------------------------------------------------------------
# INT4 symmetric (signed [-8, 7]) helpers used by sm90 INT4-A8 path.
# 4-bit two's-complement codes: 0..7 -> 0..7, 8..15 -> -8..-1.
# Two nibbles are packed into one int8 byte, low nibble first.
#
# All 16 INT4 sym values are exactly representable in FP8 E4M3, so we provide
# a lossless INT4 -> packed E4M3 byte converter that lets us drive the existing
# FP8xFP8 fused kernel with INT4-quantised B for end-to-end accuracy testing
# while a dedicated INT4-A8 fused kernel is being landed.
# ---------------------------------------------------------------------------

def _int4_sym_decode(nibble: torch.Tensor) -> torch.Tensor:
    """Map low-4-bit two's-complement nibble to signed int in [-8, 7]."""
    nib = (nibble & 0x0F).to(torch.int32)
    return torch.where(nib >= 8, nib - 16, nib)


def per_token_cast_to_int4_sym(x: torch.Tensor, gran_k: int = 128
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-(row, K-block) symmetric INT4 quantisation.

    Returns (packed, sf):
      packed: int8 tensor of shape (m, n // 2), two nibbles per byte (low first).
      sf:     fp32 tensor of shape (m, n // gran_k), absmax / 7.
    """
    assert x.dim() == 2
    m, n = x.shape
    assert n % 2 == 0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 7.0
    x_scaled = x_view.float() * (1.0 / sf.unsqueeze(2))
    # Round-half-to-even and clamp to int4 sym range.
    q = torch.round(x_scaled).clamp_(-8, 7).to(torch.int8).view(m, padded_n)
    # Pack two nibbles per byte; mask with 0x0F first to drop the sign extension.
    q2 = q.view(m, padded_n // 2, 2)
    packed = ((q2[:, :, 0] & 0x0F) | ((q2[:, :, 1] & 0x0F) << 4)).to(torch.int8)
    return packed[:, : n // 2].contiguous(), sf


def cast_back_from_int4_sym(packed: torch.Tensor, sf: torch.Tensor,
                            gran_k: int = 128) -> torch.Tensor:
    """Dequantise INT4-sym packed weights back to fp32."""
    m, n2 = packed.shape
    n = n2 * 2
    lo = _int4_sym_decode(packed)
    hi = _int4_sym_decode(packed >> 4)
    unpacked = torch.empty((m, n), dtype=torch.float, device=packed.device)
    unpacked[:, 0::2] = lo.float()
    unpacked[:, 1::2] = hi.float()
    group_idx = torch.arange(n, device=packed.device) // gran_k
    return unpacked * sf[:, group_idx]


# Lossless 16-entry LUT: INT4 sym (4-bit two's-complement) -> FP8 E4M3 byte.
# Verified: every signed int in [-8, 7] is exactly representable in E4M3, so
# casting q.float().to(torch.float8_e4m3fn) round-trips identically.
def int4_sym_to_e4m3_bytes(packed_int4: torch.Tensor) -> torch.Tensor:
    """Convert packed INT4-sym (2 nibbles / byte) to unpacked E4M3 bytes.

    Args:
        packed_int4: int8 tensor (..., n // 2). Low nibble is element 2*i,
                     high nibble is element 2*i+1.
    Returns:
        torch.float8_e4m3fn tensor (..., n) with byte-identical encoding to
        a fresh fp32 -> fp8_e4m3 cast of the dequantised integer values.
    """
    assert packed_int4.dtype == torch.int8
    *prefix, n2 = packed_int4.shape
    n = n2 * 2
    lo = _int4_sym_decode(packed_int4)
    hi = _int4_sym_decode(packed_int4 >> 4)
    unpacked = torch.empty((*prefix, n), dtype=torch.int32, device=packed_int4.device)
    unpacked[..., 0::2] = lo
    unpacked[..., 1::2] = hi
    return unpacked.float().to(torch.float8_e4m3fn)
