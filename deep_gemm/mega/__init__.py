import torch
import types
from typing import Tuple, Optional
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
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
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
            use_fp8_dispatch, activation
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


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Token count must be aligned to block sizes
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    # L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    # L2: only transpose SF for UTCCP.
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed



def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
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


def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """SM90 (Hopper) variant of `transform_weights_for_mega_moe`.

    SM90 has no TMEM / UTCCP path, so the SF tensors are consumed directly by
    WGMMA promote and don't need the 4x32 transpose. With block (128, 128)
    weight quantization, weight SFs are read by the math warpgroup directly
    from global memory in their natural ``(E, N/128, K/128)`` MN-major layout
    and require no transformation. Only L1's gate/up FP8 weight interleave is
    preserved.
    """
    l1_fp8, l1_sf = l1_weights
    # Reuse the gran-8 N interleave on the FP8 weight only; the block SF stays
    # in its natural ``(E, 2*IH/128, H/128)`` layout (gate then up along N).
    def _interleave_one(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    return (_interleave_one(l1_fp8), l1_sf), l2_weights


def fp8_mega_moe(y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SymmBuffer,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True):
    """SM90 (Hopper) MegaMoE entry point.

    Expects FP8 e4m3 weights and block-(128, 128) float scale factors. Backed by
    the single N-split cooperative kernel (BLOCK_M=64, BLOCK_N=256): two math
    warpgroups split the 256-wide N tile into two 128-wide halves, share one
    A-tile load, and quantize the L2-input activations at per-128 K to match the
    standard DeepEP runner. (The former pingpong kernel and token-count routing
    were removed.)
    """
    _C.fp8_mega_moe(
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


def fp8_mega_moe_cooperative(y: torch.Tensor,
                             l1_weights: Tuple[torch.Tensor, torch.Tensor],
                             l2_weights: Tuple[torch.Tensor, torch.Tensor],
                             sym_buffer: SymmBuffer,
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                             recipe: Tuple[int, int, int] = (128, 128, 128),
                             activation: str = 'swiglu',
                             activation_clamp: Optional[float] = None,
                             fast_math: bool = True):
    """SM90 MegaMoE — explicit cooperative kernel entry (BLOCK_M=64, BLOCK_N=256,
    two warpgroups N-split one tile and share the activation load). Identical to
    :func:`fp8_mega_moe`; kept as a named entry for A/B tests.
    """
    _C.fp8_mega_moe_cooperative(
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
