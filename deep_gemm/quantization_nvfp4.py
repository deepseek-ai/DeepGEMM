"""Offline NVFP4 quantization for SM90 fused MegaMoE."""
import torch


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
FP4_MAX = 6.0

UE4M3_MIN_DENORM = 2.0 ** -9
UE4M3_MAX_DENORM = 7.0 * UE4M3_MIN_DENORM
UE4M3_MIN_NORMAL = 2.0 ** -6
UE4M3_MAX_FINITE = 448.0


def fp32_to_fp4_nibble(x: torch.Tensor) -> torch.Tensor:
    sign = (x < 0).to(torch.uint8) << 3
    mag = x.abs().clamp_max(FP4_MAX)
    # Midpoints for nearest E2M1 values {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    # This avoids materializing an extra trailing dimension of size 8.
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=torch.float32,
    )
    nibble_idx = torch.bucketize(mag.to(torch.float32), boundaries).to(torch.uint8)
    return sign | nibble_idx


def fp32_to_ue4m3_ceil(x: torch.Tensor) -> torch.Tensor:
    """Encode non-negative scales to the smallest finite UE4M3 value >= x."""
    x = x.to(torch.float32).clamp(min=UE4M3_MIN_DENORM, max=UE4M3_MAX_FINITE)

    denorm_code = torch.ceil(x / UE4M3_MIN_DENORM).to(torch.int32).clamp(1, 7)

    x_norm = torch.clamp(x, min=UE4M3_MIN_NORMAL)
    exp_unbiased = torch.floor(torch.log2(x_norm))
    exp_bits = (exp_unbiased + 7).to(torch.int32)
    base = torch.exp2(exp_unbiased)
    mant = torch.ceil((x_norm / base - 1.0) * 8.0 - 1e-6).to(torch.int32)
    overflow = mant > 7
    exp_bits = torch.where(overflow, exp_bits + 1, exp_bits)
    mant = torch.where(overflow, torch.zeros_like(mant), mant).clamp(0, 7)
    normal_code = (exp_bits * 8 + mant).clamp(8, 0x7E)

    code = torch.where(x <= UE4M3_MAX_DENORM, denorm_code, normal_code)
    return code.to(torch.uint8)


def ue4m3_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    code = scale.to(torch.int32) & 0x7F
    exp_bits = code >> 3
    mant = code & 0x7
    denorm = mant.to(torch.float32) * UE4M3_MIN_DENORM
    normal = (1.0 + mant.to(torch.float32) * 0.125) * torch.exp2((exp_bits - 7).to(torch.float32))
    value = torch.where(exp_bits == 0, denorm, normal)
    value = torch.where(code == 0x7F, torch.full_like(value, UE4M3_MAX_FINITE), value)
    return value


def quantize_to_nvfp4(weight: torch.Tensor, group_size: int = 16):
    """Quantize real-valued weights to packed E2M1 FP4 plus per-16 UE4M3 scale."""
    assert weight.is_floating_point() or weight.dtype == torch.float8_e4m3fn
    *outer_shape, K = weight.shape
    assert K % group_size == 0
    G = K // group_size
    w = weight.to(torch.float32).view(*outer_shape, G, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    desired_scale = max_abs / FP4_MAX
    scale_ue4m3 = fp32_to_ue4m3_ceil(desired_scale.squeeze(-1))
    scale = ue4m3_to_fp32(scale_ue4m3).unsqueeze(-1)
    w_normalized = w / scale
    nibbles = fp32_to_fp4_nibble(w_normalized.clamp(-FP4_MAX, FP4_MAX))
    nibbles = nibbles.view(*outer_shape, K)
    # ModelOpt canonical packing stores consecutive K values low nibble first.
    assert K % 2 == 0
    pairs = nibbles.view(*outer_shape, K // 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8).contiguous()
    return packed, scale_ue4m3.contiguous()


def nvfp4_scale_to_tile_major(
    scale_ue4m3: torch.Tensor,
    block_n: int = 128,
    block_k: int = 128,
    group_size: int = 16,
) -> torch.Tensor:
    """Repack row-major ``(E, N, K/16)`` UE4M3 scales for SM90 tile-local loads.

    The kernel consumes scales as ``(E, N/block_n, K/block_k, block_n, block_k/16)``.
    This makes the 128 row-local scale vectors touched by a math warpgroup
    contiguous instead of striding by the full K scale dimension.
    """
    assert scale_ue4m3.dtype == torch.uint8
    assert scale_ue4m3.dim() == 3
    assert block_k % group_size == 0
    groups_per_k_block = block_k // group_size
    E, N, G = scale_ue4m3.shape
    assert N % block_n == 0
    assert G % groups_per_k_block == 0
    return (
        scale_ue4m3.view(E, N // block_n, block_n, G // groups_per_k_block, groups_per_k_block)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def nvfp4_fuse_packed_with_scale_tile_major(
    packed: torch.Tensor,
    scale_tile_major: torch.Tensor,
    block_k: int = 128,
) -> torch.Tensor:
    """Pack each BK128 NVFP4 row as ``64B FP4 + 8B UE4M3 scale + 8B padding``.

    The SM90 NVFP4 bridge still keeps compact FP4 weights, but this layout lets
    the B TMA load bring packed values and their row-local scale bytes together.
    The returned tensor keeps a 3D public weight shape ``(E, N, K/128*80)`` so
    the normal K-major TMA descriptor path can be reused.
    """
    assert packed.dtype == torch.uint8
    assert scale_tile_major.dtype == torch.uint8
    assert packed.dim() == 3
    assert scale_tile_major.dim() == 5
    E, N, K_half = packed.shape
    E_s, n_blocks, k_blocks, block_n, groups_per_k_block = scale_tile_major.shape
    fused_row_bytes = block_k // 2 + 16
    scale_offset = block_k // 2
    assert E == E_s
    assert N == n_blocks * block_n
    assert K_half == k_blocks * (block_k // 2)
    assert groups_per_k_block == block_k // 16
    packed_tile = (
        packed.view(E, n_blocks, block_n, k_blocks, block_k // 2)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    # Keep the final 8-byte padding in every BK128 row deterministic.
    fused = torch.zeros(
        (E, n_blocks, k_blocks, block_n, fused_row_bytes),
        dtype=torch.uint8,
        device=packed.device,
    )
    fused[..., :scale_offset] = packed_tile
    fused[..., scale_offset : scale_offset + groups_per_k_block] = scale_tile_major
    return (
        fused.permute(0, 1, 3, 2, 4)
        .reshape(E, N, k_blocks * fused_row_bytes)
        .contiguous()
    )


def dequantize_nvfp4_to_fp32(packed: torch.Tensor, scale_ue4m3: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    if scale_ue4m3.dim() == 5:
        E, n_blocks, k_blocks, block_n, groups_per_k_block = scale_ue4m3.shape
        fused_row_bytes = 80
        fused_k = k_blocks * fused_row_bytes
        if packed.dim() == 3 and packed.shape == (E, n_blocks * block_n, fused_k):
            packed = (
                packed.view(E, n_blocks, block_n, k_blocks, fused_row_bytes)
                .permute(0, 1, 3, 2, 4)[..., :64]
                .permute(0, 1, 3, 2, 4)
                .reshape(E, n_blocks * block_n, k_blocks * 64)
                .contiguous()
            )
        scale_ue4m3 = (
            scale_ue4m3.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, n_blocks * block_n, k_blocks * groups_per_k_block)
        )
    *outer_shape, K_half = packed.shape
    K = K_half * 2
    G = K // group_size
    # Inverse ModelOpt packing: low nibble precedes high nibble along K.
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(
        *outer_shape, K
    )
    sign_bit = (nibbles >> 3) & 0x1
    mag_idx = (nibbles & 0x7).to(torch.long)
    fp4_values = FP4_VALUES.to(packed.device)
    mag = fp4_values[mag_idx]
    values = torch.where(sign_bit.bool(), -mag, mag)
    scale = ue4m3_to_fp32(scale_ue4m3)
    scale_expanded = scale.unsqueeze(-1).expand(*outer_shape, G, group_size).reshape(*outer_shape, K)
    return values * scale_expanded
