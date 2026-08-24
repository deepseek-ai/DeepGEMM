"""SM90 weight preparation for Humming-compatible MXFP4 MegaMoE.

The public transforms in this module intentionally do not depend on CUDA or
the DeepGEMM extension.  Weight conversion is a model-load-time operation and
can therefore be contract-tested on CPU.
"""

from typing import Optional, Tuple

import torch


MXFP4CheckpointWeights = Tuple[torch.Tensor, torch.Tensor]
MXFP4ProcessedWeights = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_SM90_MXFP4_COALESCED_SCALE_MAX_HIDDEN = 8192


def _is_valid_sm90_mxfp4_hidden_size(hidden: int) -> bool:
    """Match the vectorized SM90 combine-chunk divisibility contract."""
    return hidden % 512 == 0 and (hidden <= 8192 or hidden % 1024 == 0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalize_mxfp4_ue8m0(sf: torch.Tensor) -> torch.Tensor:
    """Return natural-layout UE8M0 exponent bytes.

    Checkpoints normally expose UE8M0 as ``uint8`` or
    ``float8_e8m0fnu``.  FP32 powers of two are also accepted because
    DeepGEMM's quantization utilities use that representation.  UE8M0 code 0
    is the finite endpoint ``2**-127``; zero, negative values, non-powers of
    two, and infinities are rejected.
    """
    if not isinstance(sf, torch.Tensor):
        raise TypeError('MXFP4 scale must be a torch.Tensor')
    if sf.dtype == torch.uint8:
        return sf.contiguous()

    e8m0_dtype = getattr(torch, 'float8_e8m0fnu', None)
    if e8m0_dtype is not None and sf.dtype == e8m0_dtype:
        return sf.contiguous().view(torch.uint8)

    if sf.dtype != torch.float32:
        raise TypeError(
            'MXFP4 scale must have dtype uint8, float8_e8m0fnu, or float32, '
            f'got {sf.dtype}')

    bits = sf.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xff
    mantissa = bits & ((1 << 23) - 1)
    is_min_subnormal = bits == (1 << 22)  # UE8M0 code 0: 2**-127.
    is_normal_power_of_two = (
        ((bits >> 31) == 0) &
        (mantissa == 0) &
        (exponent >= 1) &
        (exponent <= 254)
    )
    _require(
        bool((is_min_subnormal | is_normal_power_of_two).all().item()),
        'FP32 MXFP4 scales must be finite positive UE8M0 powers of two')
    return torch.where(
        is_min_subnormal, torch.zeros_like(exponent), exponent).to(torch.uint8)


def _normalize_mxfp4_packed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return packed E2M1 bytes using DeepGEMM's signed-byte tensor view."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError('MXFP4 weight must be a torch.Tensor')
    if weight.dtype not in (torch.uint8, torch.int8):
        raise TypeError(
            f'packed MXFP4 weight must have dtype uint8 or int8, got {weight.dtype}')
    weight = weight.contiguous()
    return weight if weight.dtype == torch.int8 else weight.view(torch.int8)


def _validate_checkpoint_mxfp4_weights(
    weights: MXFP4CheckpointWeights,
    name: str,
) -> MXFP4CheckpointWeights:
    if not isinstance(weights, tuple) or len(weights) != 2:
        raise TypeError(f'{name} must be a (packed_weight, ue8m0_scale) tuple')
    weight = _normalize_mxfp4_packed_weight(weights[0])
    sf = _normalize_mxfp4_ue8m0(weights[1])
    _require(weight.dim() == 3, f'{name} weight must have shape [E, N, K/2]')
    _require(sf.dim() == 3, f'{name} scale must have shape [E, N, K/32]')
    _require(weight.device == sf.device, f'{name} weight and scale must be on the same device')
    _require(weight.size(0) > 0 and weight.size(1) > 0 and weight.size(2) > 0,
             f'{name} dimensions must be nonzero')
    _require(weight.shape[:2] == sf.shape[:2],
             f'{name} weight and scale must have matching [E, N] dimensions')
    _require(weight.size(2) == sf.size(2) * 16,
             f'{name} expects packed K/2 weights and K/32 scales')
    return weight, sf


def _validate_mxfp4_layer_pair(
    l1_weights: MXFP4CheckpointWeights,
    l2_weights: MXFP4CheckpointWeights,
) -> None:
    l1_w, _ = l1_weights
    l2_w, _ = l2_weights
    _require(l1_w.device == l2_w.device,
             'L1 and L2 MXFP4 weights must be on the same device')
    _require(l1_w.size(0) == l2_w.size(0),
             'L1 and L2 MXFP4 weights must contain the same number of experts')
    _require(l1_w.size(1) % 16 == 0,
             'L1 gate/up rows must be divisible by 16 for granularity-8 interleave')
    # L1 is [E, 2I, H/2] and L2 is [E, H, I/2].
    _require(l1_w.size(1) == l2_w.size(2) * 4,
             'incompatible L1/L2 shapes: expected L1 N == 4 * L2 packed-K')
    _require(l2_w.size(1) == l1_w.size(2) * 2,
             'incompatible L1/L2 shapes: expected L2 N == 2 * L1 packed-K')
    hidden = l2_w.size(1)
    intermediate_hidden = l1_w.size(1) // 2
    _require(
        _is_valid_sm90_mxfp4_hidden_size(hidden),
        'SM90 MXFP4 MegaMoE hidden size must be divisible by 512 and, '
        'when greater than 8192, divisible by 1024')
    _require(intermediate_hidden % 256 == 0,
             'SM90 MXFP4 MegaMoE intermediate hidden size must be divisible by 256')


def _interleave_mxfp4_rows(tensor: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    """Convert ``[gate | up]`` rows to granularity-8 gate/up interleave."""
    _require(tensor.dim() == 3, 'MXFP4 expert tensor must be three-dimensional')
    num_experts, num_rows, *tail = tensor.shape
    _require(num_rows % (2 * granularity) == 0,
             f'gate/up row count must be divisible by {2 * granularity}')
    half = num_rows // 2
    gate = tensor[:, :half].reshape(
        num_experts, half // granularity, granularity, *tail)
    up = tensor[:, half:].reshape(
        num_experts, half // granularity, granularity, *tail)
    result = torch.empty_like(tensor)
    result.view(
        num_experts, half // granularity, 2, granularity, *tail
    ).copy_(torch.stack((gate, up), dim=2))
    return result


def _transpose_mxfp4_scales_for_sm90(tensor: torch.Tensor) -> torch.Tensor:
    """Store K128 scale words contiguously across N for coalesced SM90 loads.

    The returned tensor deliberately retains the public ``[E, N, K/32]``
    shape. Its opaque processed payload is physically ordered as
    ``[E, K/128, N, 4]`` and consumed only by the SM90 Humming kernel.
    """
    _require(tensor.dim() == 3,
             'MXFP4 scale tensor must be three-dimensional')
    num_experts, num_rows, num_k32_groups = tensor.shape
    _require(num_k32_groups % 4 == 0,
             'SM90 MXFP4 scale K/32 dimension must be divisible by 4')
    return tensor.reshape(
        num_experts, num_rows, num_k32_groups // 4, 4
    ).permute(0, 2, 1, 3).contiguous().view(tensor.shape)


def _uses_coalesced_mxfp4_scales_sm90(hidden: int) -> bool:
    """Match the compile-time scale-load specialization in the SM90 kernel."""
    return hidden <= _SM90_MXFP4_COALESCED_SCALE_MAX_HIDDEN


def _restore_mxfp4_scales_from_sm90(tensor: torch.Tensor) -> torch.Tensor:
    """Restore an opaque SM90 scale payload to logical ``[E, N, K/32]``."""
    _require(tensor.dim() == 3,
             'MXFP4 scale tensor must be three-dimensional')
    num_experts, num_rows, num_k32_groups = tensor.shape
    _require(num_k32_groups % 4 == 0,
             'SM90 MXFP4 scale K/32 dimension must be divisible by 4')
    return tensor.view(
        num_experts, num_k32_groups // 4, num_rows, 4
    ).permute(0, 2, 1, 3).contiguous().view(tensor.shape)


def _reorder_mxfp4_sign_bits_for_sm90(weight: torch.Tensor) -> torch.Tensor:
    """Reorder sign bits in each packed 32-bit word for the SM90 decoder.

    Magnitude nibbles remain in checkpoint order.  Sign bits change from
    ``[s0,s1,s2,s3,s4,s5,s6,s7]`` to
    ``[s0,s4,s1,s5,s2,s6,s3,s7]`` so the device decoder can form the two FP8
    output words without additional sign permutations.
    """
    if weight.dtype not in (torch.uint8, torch.int8):
        raise TypeError('SM90 MXFP4 sign reordering requires uint8 or int8 bytes')
    _require(weight.size(-1) % 4 == 0,
             'SM90 MXFP4 sign reordering requires K/2 bytes divisible by 4')
    original_dtype = weight.dtype
    source = weight.contiguous().view(torch.uint8).reshape(
        *weight.shape[:-1], -1, 4)
    reordered = source & 0x77
    reordered[..., 0] |= (source[..., 0] & 0x08) | ((source[..., 2] & 0x08) << 4)
    reordered[..., 1] |= ((source[..., 0] & 0x80) >> 4) | (source[..., 2] & 0x80)
    reordered[..., 2] |= (source[..., 1] & 0x08) | ((source[..., 3] & 0x08) << 4)
    reordered[..., 3] |= ((source[..., 1] & 0x80) >> 4) | (source[..., 3] & 0x80)
    return reordered.reshape(weight.shape).view(original_dtype)


def _restore_mxfp4_sign_bits_from_sm90(weight: torch.Tensor) -> torch.Tensor:
    """Restore standard consecutive-nibble signs from the SM90 layout."""
    if weight.dtype not in (torch.uint8, torch.int8):
        raise TypeError('SM90 MXFP4 sign restoration requires uint8 or int8 bytes')
    _require(weight.size(-1) % 4 == 0,
             'SM90 MXFP4 sign restoration requires K/2 bytes divisible by 4')
    original_dtype = weight.dtype
    source = weight.contiguous().view(torch.uint8).reshape(
        *weight.shape[:-1], -1, 4)
    restored = source & 0x77
    restored[..., 0] |= (source[..., 0] & 0x08) | ((source[..., 1] & 0x08) << 4)
    restored[..., 1] |= (source[..., 2] & 0x08) | ((source[..., 3] & 0x08) << 4)
    restored[..., 2] |= ((source[..., 0] & 0x80) >> 4) | (source[..., 1] & 0x80)
    restored[..., 3] |= ((source[..., 2] & 0x80) >> 4) | (source[..., 3] & 0x80)
    return restored.reshape(weight.shape).view(original_dtype)


def _process_mxfp4_e8m0(
    weight: torch.Tensor,
    sf: torch.Tensor,
    interleave_rows: bool = False,
) -> MXFP4ProcessedWeights:
    """Convert raw UE8M0 scales to bounded per-expert exponent offsets.

    The retained relative exponent range is 11, matching Humming's fused-E8M0
    weight contract.  Groups below that range are requantized in packed E2M1;
    the returned FP32 secondary scale restores the common expert exponent in
    the MegaMoE accumulation path.
    """
    weight, raw_sf = _validate_checkpoint_mxfp4_weights((weight, sf), 'MXFP4')
    _require(not bool((raw_sf == 255).any().item()),
             'processed MXFP4 does not accept UE8M0 NaN code 255')
    if interleave_rows:
        _require(weight.size(1) % 16 == 0,
                 'processed L1 rows must be divisible by 16')

    sf_i16 = raw_sf.to(torch.int16)
    max_exp = sf_i16.flatten(1).amax(dim=1)
    min_exp = sf_i16.flatten(1).amin(dim=1)
    base_exp = max_exp - torch.minimum(
        max_exp - min_exp, torch.full_like(max_exp, 11))

    base_view = base_exp.view(-1, 1, 1)
    clamped = torch.maximum(sf_i16, base_view)
    delta = clamped - sf_i16
    # Humming's unsigned exponent construction wraps for delta >= 128.
    _require(not bool((delta >= 128).any().item()),
             'processed MXFP4 exponent delta must be less than 128')
    offsets = (clamped - base_view + 1).to(torch.uint8)
    secondary = torch.exp2(base_exp.float() - 128.0).contiguous()

    # Exact E2M1 requantization lookup for exponent deltas 0..5.  Larger
    # accepted deltas map all finite magnitudes to zero and reuse row 5.
    nibble_lut = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 1, 2, 2, 3, 4, 5, 0, 9, 9, 10, 10, 11, 12, 13,
        0, 0, 1, 1, 1, 2, 2, 3, 0, 8, 9, 9, 9, 10, 10, 11,
        0, 0, 0, 0, 1, 1, 1, 2, 0, 8, 8, 8, 9, 9, 9, 10,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 8, 8, 8, 8, 8, 9, 9,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8,
    ], dtype=torch.uint8, device=weight.device).view(6, 16)
    packed_values = torch.arange(256, dtype=torch.long, device=weight.device)
    packed_lut = (
        nibble_lut[:, packed_values & 0x0f] |
        (nibble_lut[:, packed_values >> 4] << 4)
    ).reshape(-1)

    packed = weight.view(torch.uint8)
    rewritten = torch.empty_like(packed)
    if interleave_rows:
        half_rows = weight.size(1) // 2
        rewritten_view = rewritten.view(
            weight.size(0), half_rows // 8, 2, 8, weight.size(2))

    num_k_groups = raw_sf.size(2)
    num_rows_per_chunk = 1024
    for expert_idx in range(weight.size(0)):
        row_regions = ((0, weight.size(1), -1),) if not interleave_rows else (
            (0, half_rows, 0), (half_rows, weight.size(1), 1))
        for region_start, region_end, gate_up_idx in row_regions:
            for row_start in range(region_start, region_end, num_rows_per_chunk):
                row_end = min(row_start + num_rows_per_chunk, region_end)
                packed_chunk = packed[expert_idx, row_start:row_end].view(
                    row_end - row_start, num_k_groups, 16)
                delta_chunk = delta[
                    expert_idx, row_start:row_end].clamp_max(5).unsqueeze(-1)
                lut_idx = delta_chunk.to(torch.long) * 256 + packed_chunk.to(torch.long)
                rewritten_chunk = packed_lut[lut_idx].reshape(row_end - row_start, -1)
                rewritten_chunk = _reorder_mxfp4_sign_bits_for_sm90(rewritten_chunk)
                if interleave_rows:
                    relative_start = row_start - region_start
                    relative_end = row_end - region_start
                    rewritten_view[
                        expert_idx,
                        relative_start // 8:relative_end // 8,
                        gate_up_idx,
                    ].copy_(rewritten_chunk.view(-1, 8, weight.size(2)))
                else:
                    rewritten[expert_idx, row_start:row_end].copy_(rewritten_chunk)

    return rewritten.view(torch.int8), offsets.contiguous(), secondary


def _apply_weight_scale_2(
    scale: torch.Tensor,
    secondary: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f'{name}_weight_scale_2 must be a torch.Tensor')
    if scale.dtype != torch.float32:
        raise TypeError(f'{name}_weight_scale_2 must have dtype float32')
    _require(scale.dim() == 1 and scale.numel() == secondary.numel(),
             f'{name}_weight_scale_2 must have shape [E]')
    _require(scale.device == secondary.device,
             f'{name}_weight_scale_2 must be on the weight device')
    _require(bool(torch.isfinite(scale).all().item()),
             f'{name}_weight_scale_2 must contain only finite values')
    _require(bool((scale > 0).all().item()),
             f'{name}_weight_scale_2 must contain only positive values')
    return (secondary * scale).contiguous()


def transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
    l1_weights: MXFP4CheckpointWeights,
    l2_weights: MXFP4CheckpointWeights,
    l1_weight_scale_2: Optional[torch.Tensor] = None,
    l2_weight_scale_2: Optional[torch.Tensor] = None,
) -> Tuple[MXFP4ProcessedWeights, MXFP4ProcessedWeights]:
    """Prepare Humming-compatible MXFP4 triples for Hopper MegaMoE.

    Each result is ``(processed_e2m1, relative_ue8m0, weight_scale_2)``.
    ``weight_scale_2`` is FP32 ``[E]`` and includes the optional Humming
    checkpoint secondary scale. For hidden sizes up to 8192, the scale tensor
    keeps its public ``[E, N, K/32]`` shape but stores an opaque physical
    ``[E, K/128, N, 4]`` payload for coalesced producer-warp loads. Larger
    hidden sizes retain natural row-major storage. This is the only routed
    MXFP4 weight contract accepted by the SM90 runtime.
    """
    l1 = _validate_checkpoint_mxfp4_weights(l1_weights, 'L1 MXFP4')
    l2 = _validate_checkpoint_mxfp4_weights(l2_weights, 'L2 MXFP4')
    _validate_mxfp4_layer_pair(l1, l2)

    _require((l1_weight_scale_2 is None) == (l2_weight_scale_2 is None),
             'L1 and L2 weight_scale_2 must either both be provided or both be omitted')

    l1_w, l1_sf, l1_secondary = _process_mxfp4_e8m0(
        *l1, interleave_rows=True)
    l2_w, l2_sf, l2_secondary = _process_mxfp4_e8m0(*l2)
    if l1_weight_scale_2 is not None:
        l1_secondary = _apply_weight_scale_2(
            l1_weight_scale_2, l1_secondary, 'l1')
        l2_secondary = _apply_weight_scale_2(
            l2_weight_scale_2, l2_secondary, 'l2')

    hidden = l2_w.size(1)
    l1_sf = _interleave_mxfp4_rows(l1_sf)
    if _uses_coalesced_mxfp4_scales_sm90(hidden):
        l1_sf = _transpose_mxfp4_scales_for_sm90(l1_sf)
        l2_sf = _transpose_mxfp4_scales_for_sm90(l2_sf)

    return (
        l1_w,
        l1_sf,
        l1_secondary,
    ), (
        l2_w.contiguous(),
        l2_sf.contiguous(),
        l2_secondary,
    )


def _validate_processed_mxfp4_kernel_weights(
    l1_weights: MXFP4ProcessedWeights,
    l2_weights: MXFP4ProcessedWeights,
) -> None:
    """Validate transformed Humming triples for the SM90 wrapper."""
    if not isinstance(l1_weights, tuple) or not isinstance(l2_weights, tuple):
        raise TypeError('SM90 MXFP4 weights must be tuples')
    if len(l1_weights) != 3 or len(l2_weights) != 3:
        raise ValueError(
            'L1/L2 SM90 MXFP4 weights must be Humming processed triples; '
            'call transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90 first')

    def validate_payload(
        weights: MXFP4ProcessedWeights,
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight, sf = weights[:2]
        if not isinstance(weight, torch.Tensor) or not isinstance(sf, torch.Tensor):
            raise TypeError(f'{name} payload entries must be torch.Tensor objects')
        if weight.dtype != torch.int8:
            raise TypeError(
                f'{name} packed weight must be transformed to int8 before launch')
        if sf.dtype != torch.uint8:
            raise TypeError(
                f'{name} scale must be transformed to uint8 before launch')
        _require(weight.is_contiguous() and sf.is_contiguous(),
                 f'{name} weight and scale must be contiguous before launch')
        _require(weight.dim() == 3 and sf.dim() == 3,
                 f'{name} must have [E, N, K/2] weight and [E, N, K/32] scale')
        _require(weight.device == sf.device,
                 f'{name} weight and scale must be on the same device')
        _require(weight.size(0) > 0 and weight.size(1) > 0 and weight.size(2) > 0,
                 f'{name} dimensions must be nonzero')
        _require(weight.shape[:2] == sf.shape[:2] and weight.size(2) == sf.size(2) * 16,
                 f'{name} expects packed K/2 weights and K/32 scales')
        return weight, sf

    l1 = validate_payload(l1_weights, 'transformed L1 MXFP4')
    l2 = validate_payload(l2_weights, 'transformed L2 MXFP4')
    _validate_mxfp4_layer_pair(l1, l2)
    for weights, payload, name in (
        (l1_weights, l1, 'l1'),
        (l2_weights, l2, 'l2'),
    ):
        scale = weights[2]
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f'{name}_weight_scale_2 must be a torch.Tensor')
        if scale.dtype != torch.float32:
            raise TypeError(f'{name}_weight_scale_2 must have dtype float32')
        _require(scale.dim() == 1 and scale.numel() == payload[0].size(0),
                 f'{name}_weight_scale_2 must have shape [E]')
        _require(scale.device == payload[0].device,
                 f'{name}_weight_scale_2 must be on the weight device')
        _require(scale.is_contiguous(),
                 f'{name}_weight_scale_2 must be contiguous before launch')
