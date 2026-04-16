from typing import Any
from tilelang import language as T


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_sf_shape(
    num_tokens: int,
    hidden: int,
    num_per_channels: int,
    use_ue8m0: bool,
    use_col_major_sf: bool,
) -> tuple[int, int]:
    num_scales = ceil_div(hidden, num_per_channels)
    num_scales = ceil_div(num_scales, 4) if use_ue8m0 else num_scales

    # For col-major SF, TMA must be aligned into 16 bytes
    # For UE8M0, we must use col-major SF, and 4 UE8M0 are expanded into the inner dim (token)
    num_sf_tokens = num_tokens
    if use_col_major_sf:
        num_sf_tokens = align(num_tokens, 4)
        num_sf_tokens = num_sf_tokens * 4 if use_ue8m0 else num_sf_tokens

    return (num_scales, num_sf_tokens) if use_col_major_sf else (num_sf_tokens, num_scales)


def get_sf_and_inv(amax: float, round_sf: bool, use_ue8m0: bool) -> tuple[Any, Any]:
    sf = amax / 448.0
    if not round_sf:
        return sf, 448.0 / amax

    # Round into 2's power
    bits = T.reinterpret("uint32", sf)
    exp = (bits >> 23) & 0xFF
    man_bits = bits & ((1 << 23) - 1)
    exp_scale = T.reinterpret("int32", exp - 127 + (man_bits != 0))
    if use_ue8m0:  # noqa: SIM108
        sf = T.Cast("uint8", exp_scale + 127)
    else:
        sf = T.reinterpret("float", (127 + exp_scale) << 23)
    return sf, T.reinterpret("float", (127 - exp_scale) << 23)