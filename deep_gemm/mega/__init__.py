import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align
from .mxfp4 import (
    MXFP4ProcessedWeights as _MXFP4ProcessedWeights,
    _normalize_mxfp4_ue8m0,
    _process_mxfp4_e8m0,
    _reorder_mxfp4_sign_bits_for_sm90,
    _restore_mxfp4_sign_bits_from_sm90,
    transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90,
    _validate_processed_mxfp4_kernel_weights,
)

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu'):
        assert activation == 'swiglu' or (mma_type == 'fp8xfp4' and activation == 'situ'), \
            f'Only FP8xFP4 MegaMoE supports `situ`, got mma_type={mma_type!r}, activation={activation!r}'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


class SM90SymmBuffer:
    """Symmetric buffer for the SM90 persistent MegaMoE backend.

    The buffer describes the FP8 activation/communication workspace shared by
    SM90 routed-weight backends. Routed-weight formats such as MXFP4 or WinT4
    are launch-API concerns and do not change these twelve views.
    """
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu',
                 num_shared_experts: int = 0):
        if num_shared_experts < 0:
            raise ValueError('num_shared_experts must be non-negative')
        if not use_fp8_dispatch:
            raise ValueError('SM90 MegaMoE requires FP8 dispatch')
        if activation != 'swiglu':
            raise ValueError(
                f'Only `swiglu` activation is supported, got `{activation}`')

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts

        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_sm90_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation, num_shared_experts,
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        for name in (
            'x', 'x_sf', 'topk_idx', 'topk_weights',
            'shared_l1_acts', 'shared_l1_acts_sf',
            'shared_l2_acts', 'shared_l2_acts_sf',
            'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf',
        ):
            setattr(self, name, None)
        self.buffer = None
        self.group = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: str = 'fp8xfp4',
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Align token count
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def get_symm_buffer_for_sm90_mega_moe(group: dist.ProcessGroup,
                                      num_experts: int,
                                      num_max_tokens_per_rank: int, num_topk: int,
                                      hidden: int, intermediate_hidden: int,
                                      use_fp8_dispatch: bool = True,
                                      activation: str = 'swiglu',
                                      num_shared_experts: int = 0) -> SM90SymmBuffer:
    """Allocate the twelve-view SM90 live-ring MegaMoE symmetric buffer.

    ``num_experts`` is global. Each rank supplies ``num_experts / group.size()``
    weights for its contiguous global expert-ID interval.
    """
    num_max_tokens_per_rank = align(
        num_max_tokens_per_rank, _C.get_token_alignment_for_sm90_mega_moe())
    return SM90SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch=use_fp8_dispatch,
        activation=activation,
        num_shared_experts=num_shared_experts,
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


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation in ('swiglu', 'situ'), f'Unsupported activation: {activation!r}'
    assert activation != 'situ' or (isinstance(l1_weights, tuple) and isinstance(l2_weights, tuple)), \
        '`situ` requires FP8xFP4 `(weight, sf)` tuples for both L1 and L2 weights'
    if isinstance(l1_weights, tuple):
        # FP8: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
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


def transform_shared_weights_for_fp8_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    activation: str = 'swiglu',
) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare replicated SM90 shared-expert FP8 weights.

    Input scales are natural row-major FP32 block-(128, 128) tensors. Only the
    L1 FP8 rows are gate/up-interleaved at granularity 8; unlike the SM100
    helper, neither scale tensor is UTCCP-transposed or interleaved. This
    transform is intentionally named after the shared-weight format, not the
    routed-weight backend, so MXFP4 and future WinT4 routed experts can share
    the same replicated FP8 expert representation.
    """
    if activation != 'swiglu':
        raise ValueError(
            f'Only `swiglu` activation is supported, got `{activation}`')
    if not (isinstance(l1_weights, tuple) and len(l1_weights) == 2 and
            isinstance(l2_weights, tuple) and len(l2_weights) == 2):
        raise TypeError('shared SM90 weights must be FP8/FP32-scale pairs')
    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    if l1_weight.dtype != torch.float8_e4m3fn or \
            l2_weight.dtype != torch.float8_e4m3fn:
        raise TypeError('shared SM90 weights must use torch.float8_e4m3fn')
    if l1_scale.dtype != torch.float32 or l2_scale.dtype != torch.float32:
        raise TypeError('shared SM90 weight scales must use torch.float32')
    if l1_weight.dim() != 2 or l2_weight.dim() != 2:
        raise ValueError('shared SM90 weights must be two-dimensional')
    shared_intermediate_hidden = l2_weight.size(1)
    hidden = l2_weight.size(0)
    if tuple(l1_weight.shape) != (2 * shared_intermediate_hidden, hidden):
        raise ValueError(
            'shared L1/L2 weight shapes must be [2*S*I,H] and [H,S*I]')
    if hidden % 128 != 0 or shared_intermediate_hidden % 128 != 0:
        raise ValueError('shared SM90 H and S*I must be divisible by 128')
    if tuple(l1_scale.shape) != (
            2 * shared_intermediate_hidden // 128, hidden // 128) or \
            tuple(l2_scale.shape) != (
                hidden // 128, shared_intermediate_hidden // 128):
        raise ValueError(
            'shared SM90 scales must use natural block-(128,128) shapes')
    if not all(tensor.is_contiguous() for tensor in (
            l1_weight, l1_scale, l2_weight, l2_scale)):
        raise ValueError('shared SM90 weights and scales must be contiguous')
    return (_interleave_weights(l1_weight), l1_scale), (l2_weight, l2_scale)



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     situ_beta: Optional[float] = None,
                     situ_linear_beta: Optional[float] = None):
    _C.fp8_fp4_mega_moe(
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
        fast_math,
        situ_beta, situ_linear_beta
    )


def fp8_mxfp4_mega_moe(y: torch.Tensor,
                       l1_weights: _MXFP4ProcessedWeights,
                       l2_weights: _MXFP4ProcessedWeights,
                       sym_buffer: SM90SymmBuffer,
                       shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                       recipe: Tuple[int, int, int] = (1, 1, 32),
                       activation: str = 'swiglu',
                       activation_clamp: Optional[float] = None,
                       fast_math: bool = True):
    """Run the SM90 Humming-compatible MXFP4 MegaMoE path.

    Routed weights must be processed triples
    ``(processed_e2m1, relative_ue8m0, weight_scale_2)`` returned by
    :func:`transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90`. Keeping this
    explicit entry point avoids architecture-dependent Python dispatch while
    the common ``fp8_fp4_mega_moe`` facade remains backward compatible with
    the SM100 implementation.

    When shared experts are enabled, prepare their FP8/FP32 pairs with
    :func:`transform_shared_weights_for_fp8_mega_moe_sm90` and copy the
    input K128 FP32 scales into ``sym_buffer.shared_l1_acts_sf`` before launch.
    ``shared_l1_acts`` itself aliases ``sym_buffer.x``.
    """
    if not isinstance(sym_buffer, SM90SymmBuffer):
        raise TypeError(
            'fp8_mxfp4_mega_moe requires an SM90SymmBuffer allocated by '
            'get_symm_buffer_for_sm90_mega_moe')
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError(
            'shared_l1_weights and shared_l2_weights must be provided together')
    num_shared_experts = getattr(sym_buffer, 'num_shared_experts', 0)
    if num_shared_experts == 0 and shared_l1_weights is not None:
        raise ValueError(
            'shared weights require an SM90SymmBuffer allocated with '
            'num_shared_experts > 0')
    if num_shared_experts > 0 and shared_l1_weights is None:
        raise ValueError(
            'an SM90SymmBuffer with shared experts requires both shared weight tuples')
    _validate_processed_mxfp4_kernel_weights(l1_weights, l2_weights)
    num_ranks = sym_buffer.group.size()
    if sym_buffer.num_experts % num_ranks != 0:
        raise ValueError('global num_experts must be divisible by the number of ranks')
    expected_local_experts = sym_buffer.num_experts // num_ranks
    if l1_weights[0].size(0) != expected_local_experts:
        raise ValueError(
            'SM90 MXFP4 weights must contain the contiguous local expert shard: '
            f'expected E_local={expected_local_experts}, got {l1_weights[0].size(0)}')
    expected_l1_shape = (
        expected_local_experts,
        2 * sym_buffer.intermediate_hidden,
        sym_buffer.hidden // 2,
    )
    expected_l2_shape = (
        expected_local_experts,
        sym_buffer.hidden,
        sym_buffer.intermediate_hidden // 2,
    )
    if tuple(l1_weights[0].shape) != expected_l1_shape or \
            tuple(l2_weights[0].shape) != expected_l2_shape:
        raise ValueError(
            'SM90 routed MXFP4 weights do not match the symmetric-buffer '
            f'H/I contract: expected {expected_l1_shape} and '
            f'{expected_l2_shape}, got {tuple(l1_weights[0].shape)} and '
            f'{tuple(l2_weights[0].shape)}')
    if num_shared_experts > 0:
        shared_intermediate_hidden = (
            num_shared_experts * sym_buffer.intermediate_hidden)
        expected_shared_l1_shape = (
            2 * shared_intermediate_hidden, sym_buffer.hidden)
        expected_shared_l2_shape = (
            sym_buffer.hidden, shared_intermediate_hidden)
        if tuple(shared_l1_weights[0].shape) != expected_shared_l1_shape or \
                tuple(shared_l2_weights[0].shape) != expected_shared_l2_shape:
            raise ValueError(
                'SM90 shared FP8 weights do not match the symmetric-buffer '
                f'shared-expert contract: expected {expected_shared_l1_shape} '
                f'and {expected_shared_l2_shape}, got '
                f'{tuple(shared_l1_weights[0].shape)} and '
                f'{tuple(shared_l2_weights[0].shape)}')
    try:
        op = _C.fp8_mxfp4_mega_moe
    except AttributeError as exception:
        raise RuntimeError(
            'DeepGEMM was built without the SM90 MXFP4 MegaMoE binding '
            '`fp8_mxfp4_mega_moe`; '
            'rebuild the extension after enabling the SM90 backend') from exception
    op(
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
        fast_math
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    _C.bf16_mega_moe(
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
