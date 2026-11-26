import warnings
from typing import Tuple, Optional, Union

import torch
import torch.distributed as dist

from .. import _C
from ..utils.math import align
from .symmetric_buffer import SymmetricBuffer
from .utils import _transpose_sf_for_utccp


class MegaMoEInferenceSymmetricBuffer(SymmetricBuffer):
    """Inference-specific tensor views over shared symmetric storage."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int, num_topk: int,
        hidden: int, intermediate_hidden: int,
        num_shared_experts: int = 0,
        mma_type: str = 'fp8xfp4',
        activation: str = 'swiglu',
        base: Optional[SymmetricBuffer] = None
    ) -> None:
        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe_inference())

        # Init
        assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe_inference(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        super().__init__(group, num_bytes, base=base)

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self) -> None:
        self.x = self.x_sf = None
        self.topk_idx = self.topk_weights = None
        self.shared_l1_acts = self.shared_l1_acts_sf = None
        self.shared_l2_acts = self.shared_l2_acts_sf = None
        self.l1_acts = self.l1_acts_sf = None
        self.l2_acts = self.l2_acts_sf = None
        super().destroy()


def get_symm_buffer_for_mega_moe_inference(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int, num_topk: int,
    hidden: int, intermediate_hidden: int,
    num_shared_experts: int = 0,
    use_fp8_dispatch: Optional[bool] = None,
    mma_type: str = 'fp8xfp4',
    activation: str = 'swiglu',
    base: Optional[SymmetricBuffer] = None
) -> MegaMoEInferenceSymmetricBuffer:
    # Check the legacy dispatch flag against the explicit MMA type.
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return MegaMoEInferenceSymmetricBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation, base=base
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe_inference(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed


def fp8_fp4_mega_moe_inference(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoEInferenceSymmetricBuffer,
    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = 'swiglu',
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    rms_weights: Optional[torch.Tensor] = None,
    rms_eps: Optional[float] = None
) -> None:
    """Run fused Mega MoE inference with optional RMSNorm.

    ``y`` may be BF16 or FP32. With RMSNorm, ``rms_eps`` must be supplied
    explicitly and ``rms_weights`` must match ``y.dtype``: BF16 mode normalizes
    the BF16-rounded combine result, while FP32 mode normalizes the FP32
    combine accumulator directly (both compute RMSNorm in FP32). Without
    RMSNorm, ``rms_eps`` is ignored and FP32 ``y`` writes the FP32 combine
    accumulator directly.
    """
    _C.fp8_fp4_mega_moe_inference(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math, rms_weights, rms_eps
    )


def bf16_mega_moe_inference(
    y: torch.Tensor,
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
    sym_buffer: MegaMoEInferenceSymmetricBuffer,
    shared_l1_weights: Optional[torch.Tensor] = None,
    shared_l2_weights: Optional[torch.Tensor] = None,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation: str = 'swiglu',
    activation_clamp: Optional[float] = None,
    fast_math: bool = True
) -> None:
    _C.bf16_mega_moe_inference(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )
