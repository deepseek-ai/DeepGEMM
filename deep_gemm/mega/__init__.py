import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

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
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

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


class Sm90SymmBuffer:
    """Symmetric buffer for the SM90 (Hopper) FP8 MegaMoE path (`fp8_mega_moe_sm90`).

    Unlike `SymmBuffer` (SM100), this always uses FP8xFP8 with float scale factors, a pool-based
    (non-ring) workspace, and does not support shared experts or the `situ` activation.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 activation: str = 'swiglu'):
        assert activation == 'swiglu', f'SM90 MegaMoE only supports `swiglu`, got activation={activation!r}'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_sm90_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            True, activation
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
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_sm90_mega_moe(group: dist.ProcessGroup,
                                      num_experts: int,
                                      num_max_tokens_per_rank: int, num_topk: int,
                                      hidden: int, intermediate_hidden: int,
                                      activation: str = 'swiglu') -> Sm90SymmBuffer:
    # Align token count (SM90 MegaMoE shares the same token alignment requirement as SM100)
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return Sm90SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        activation=activation
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
        # Scaled FP8xFP4/NVFP4: both formats use the same MN-major SF layout.
        # Interleave gate/up for weight and SF, then transpose L1 SF for UTCCP.
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



def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Weight preprocessing for `fp8_mega_moe_sm90`.

    Unlike `transform_weights_for_mega_moe` (SM100), the SM90 kernel consumes weight SF as plain
    float in block-(128, 128) layout (no UE8M0 packing, no UTCCP transpose), so only the L1
    gate/up interleave is needed; L2 weights and both SF tensors pass through unchanged.
    """
    l1_fp8, l1_sf = l1_weights
    l1_transformed = (_interleave_weights(l1_fp8), l1_sf)
    return l1_transformed, l2_weights


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

def fp4_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[torch.Tensor] = None,
                     shared_l2_weights: Optional[torch.Tensor] = None,
                     x_bf16: Optional[torch.Tensor] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 16),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     l1_alphas: Optional[torch.Tensor] = None,
                     l2_alphas: Optional[torch.Tensor] = None,
                     a2_scales: Optional[torch.Tensor] = None,
                     routed_scaling_factor: float = 1.0):
    """Run NVFP4 routed experts, optionally fused with BF16 shared experts.

    NVFP4 operands use packed E2M1 values and per-16-element E4M3 scales.
    ``l1_alphas`` and ``l2_alphas`` carry optional per-expert model scales.
    The caller must fold the FC1 input global scale into ``l1_alphas``;
    ``a2_scales`` is separate because the FC2 input is produced and requantized
    inside this kernel; when provided, every entry must be finite and strictly
    positive. Shared weights and ``x_bf16`` must be provided together;
    ``routed_scaling_factor`` is applied before adding their BF16 output.

    CUDA Graph replay must reuse the captured ``x_bf16`` allocation and token
    count because both are encoded in the captured shared-input tensor map.
    """
    if sym_buffer.mma_type != 'fp4xfp4':
        raise ValueError(
            f'NVFP4 MegaMoE requires an fp4xfp4 symmetric buffer, got {sym_buffer.mma_type!r}')
    if sym_buffer.activation != activation:
        raise ValueError(
            f'Activation mismatch: buffer={sym_buffer.activation!r}, call={activation!r}')
    if not ((shared_l1_weights is None) == (shared_l2_weights is None) == (x_bf16 is None)):
        raise ValueError('Shared L1 weights, shared L2 weights, and x_bf16 must be provided together')
    num_shared_experts = 0 if shared_l2_weights is None else \
        shared_l2_weights.size(1) // sym_buffer.intermediate_hidden
    if sym_buffer.num_shared_experts != num_shared_experts:
        raise ValueError(
            f'Shared-expert layout mismatch: buffer={sym_buffer.num_shared_experts}, call={num_shared_experts}')
    _C.fp4_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights, x_bf16,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        l1_alphas, l2_alphas, a2_scales, routed_scaling_factor
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


def fp8_mega_moe_sm90(y: torch.Tensor,
                      l1_weights: Tuple[torch.Tensor, torch.Tensor],
                      l2_weights: Tuple[torch.Tensor, torch.Tensor],
                      sym_buffer: Sm90SymmBuffer,
                      cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                      recipe: Tuple[int, int, int] = (128, 128, 128),
                      activation: str = 'swiglu',
                      activation_clamp: Optional[float] = None,
                      fast_math: bool = True):
    """SM90 (Hopper) FP8 MegaMoE fused GEMM, ported from upstream PR #36.

    Unlike `fp8_fp4_mega_moe` (SM100), both L1/L2 weights are FP8 e4m3 with block-(128, 128)
    float scale factors (not FP4/UE8M0), and there is no shared-expert or `situ` support.
    """
    _C.fp8_mega_moe_sm90(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )
