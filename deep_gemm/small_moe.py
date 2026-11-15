"""Opt-in, pre-routed BF16 MoE workspace for SM120 inference.

Import explicitly from deep_gemm.small_moe; existing GEMM/MoE defaults do not
change. Route planning and workspace allocation happen outside the timed path.
"""
import torch
import triton
import triton.language as tl

from . import _C
from .small_moe_policy import select_config


@triton.jit
def _dispatch_slots(X, PACKED, SLOTS, D: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    route = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    slot = tl.load(SLOTS + route).to(tl.int64)
    value = tl.load(X + (route // K) * D + col, col < D, 0)
    tl.store(PACKED + slot * D + col, value, col < D)


@triton.jit
def _swiglu(PROJECTION, H, SLOTS, I: tl.constexpr, B: tl.constexpr):
    slot = tl.load(SLOTS + tl.program_id(0)).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    g = tl.load(PROJECTION + slot * (2 * I) + col, col < I, 0).to(tl.float32)
    u = tl.load(PROJECTION + slot * (2 * I) + I + col, col < I, 0).to(tl.float32)
    tl.store(H + slot * I + col, g * tl.sigmoid(g) * u, col < I)


@triton.jit
def _combine(EXPERT, SLOTS, WEIGHTS, Y, D: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for r in tl.static_range(K):
        slot = tl.load(SLOTS + token * K + r).to(tl.int64)
        weight = tl.load(WEIGHTS + token * K + r)
        value = tl.load(EXPERT + slot * D + col, col < D, 0).to(tl.float32)
        total = total + value * weight
    tl.store(Y + token * D + col, total, col < D)


class SmallMoE:
    """Own reusable buffers and choose bounded fusion or ordinary fallback.

    gate_up: BF16 [E, 2*I, D], gate rows followed by up rows.
    down: BF16 [E, D, I]. forward inputs: X [T,D], counts int32 [E],
    slots int64 [T,k], route weights FP32 [T,k], all contiguous on one GPU.

    slots identifies expert*capacity+local_row. Routes must be unique per
    token, slots globally unique, and the rows per expert exactly 0..count-1.
    Call validate_routes outside capture when this contract is not guaranteed.
    No input value readback, sorting, allocation or floating atomics in forward.
    Warm up before Graph capture. Output aliases this workspace. One workspace
    cannot be used concurrently on independent streams. Inference only.
    """

    def __init__(self, tokens, gate_up, down, topk, *, capacity=None,
                 enable_fusion=True, fallback_m_tile=None):
        if gate_up.ndim != 3 or down.ndim != 3:
            raise ValueError('Expected rank-three gate_up and down weights')
        self.e, twice_i, self.d = gate_up.shape
        self.i, self.t, self.k = twice_i // 2, tokens, topk
        if not isinstance(tokens, int) or not isinstance(topk, int) or tokens < 1 or not 1 <= topk <= self.e:
            raise ValueError('Positive tokens and 1 <= topk <= experts required')
        if (self.e <= 0 or self.d <= 0 or self.i <= 0 or twice_i % 2
                or down.shape != (self.e, self.d, self.i) or self.d % 128 or self.i % 128):
            raise ValueError('Incompatible weights or D/I not divisible by 128')
        self.device = gate_up.device
        if not gate_up.is_cuda or torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError('SM120 CUDA device required')
        for name, tensor in (('gate_up', gate_up), ('down', down)):
            self._check_tensor(name, tensor, torch.bfloat16, tensor.shape)
            if tensor.requires_grad:
                raise ValueError('SmallMoE is an inference workspace')
        self.capacity = ((tokens + 127) // 128) * 128 if capacity is None else capacity
        if not isinstance(self.capacity, int) or self.capacity < tokens or self.capacity % 128:
            raise ValueError('Capacity must cover tokens and be divisible by 128')
        if self.e * self.capacity > 2**31 - 1 or self.e * max(self.d, 2*self.i) > 2**31 - 1:
            raise ValueError('Packed dimensions exceed int32 descriptor range')
        self.gate_up, self.down = gate_up, down
        self.config = select_config(tokens, self.e, topk, self.d, self.i) if enable_fusion else None
        self.fallback_m_tile = (32 if tokens <= 4 else 0) if fallback_m_tile is None else fallback_m_tile
        if self.fallback_m_tile not in (0, 32, 64, 128):
            raise ValueError('Invalid ordinary M tile')
        options = dict(device=self.device, dtype=torch.bfloat16)
        # Ordinary TMA tiles include padded rows; initialize their inputs once.
        self.packed = torch.zeros((self.e, self.capacity, self.d), **options)
        self.projection = torch.empty((self.e, self.capacity, 2*self.i), **options)
        self.h = torch.zeros((self.e, self.capacity, self.i), **options)
        self.expert = torch.empty((self.e, self.capacity, self.d), **options)
        self.output = torch.empty((self.t, self.d), **options)

    def _check_tensor(self, name, tensor, dtype, shape):
        if tensor.device != self.device or tensor.dtype != dtype or tensor.shape != shape or not tensor.is_contiguous():
            raise ValueError(f'{name}: expected contiguous {dtype} {tuple(shape)} on {self.device}')
        if tensor.data_ptr() % 16:
            raise ValueError(f'{name}: expected 16-byte aligned storage')

    def _check_routes_metadata(self, counts, slots, weights):
        self._check_tensor('counts', counts, torch.int32, (self.e,))
        self._check_tensor('slots', slots, torch.int64, (self.t, self.k))
        self._check_tensor('weights', weights, torch.float32, (self.t, self.k))
        if weights.requires_grad:
            raise ValueError('SmallMoE does not support autograd')

    def validate_routes(self, counts, slots, weights):
        """Synchronizing value validation; never call inside CUDA Graph capture."""
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('validate_routes must run outside Graph capture')
            self._check_routes_metadata(counts, slots, weights)
            c, s, w = counts.cpu(), slots.cpu(), weights.cpu()
        flat = s.flatten()
        if not bool(torch.isfinite(w).all()) or bool(((c < 0) | (c > self.capacity)).any()):
            raise ValueError('Nonfinite route weights or out-of-range counts')
        if bool(((flat < 0) | (flat >= self.e*self.capacity)).any()) or flat.unique().numel() != flat.numel():
            raise ValueError('Out-of-range or overlapping route slots')
        ids = s // self.capacity
        if self.k > 1 and bool((ids.sort(dim=1).values[:, 1:] == ids.sort(dim=1).values[:, :-1]).any()):
            raise ValueError('Duplicate experts within a token')
        for e in range(self.e):
            rows = (flat[flat // self.capacity == e] % self.capacity).sort().values
            if not torch.equal(rows, torch.arange(int(c[e]), dtype=torch.int64)):
                raise ValueError('slots must exactly cover counts and dense expert rows')

    @torch.no_grad()
    def forward(self, x, counts, slots, weights):
        self._check_tensor('X', x, torch.bfloat16, (self.t, self.d))
        self._check_routes_metadata(counts, slots, weights)
        if x.requires_grad:
            raise ValueError('SmallMoE does not support autograd')
        internal = {v.untyped_storage().data_ptr() for v in
                    (self.packed, self.projection, self.h, self.expert, self.output)}
        if any(v.untyped_storage().data_ptr() in internal for v in (x, counts, slots, weights)):
            raise ValueError('Inputs must not alias workspace buffers')
        with torch.cuda.device(self.device):
            config = self.config
            if config is not None and config.l1 is not None:
                _C._sm120_moe_small(x, self.gate_up, slots, weights, self.h, True, *config.l1)
            else:
                _dispatch_slots[(self.t*self.k, triton.cdiv(self.d, 256))](
                    x, self.packed, slots, self.d, self.k, 256, num_warps=4)
                _C.m_grouped_bf16_gemm_nt_masked(self.packed, self.gate_up, self.projection, counts,
                    triton.cdiv(self.t*self.k, self.e), sm120_m_tile=self.fallback_m_tile)
                _swiglu[(self.t*self.k, triton.cdiv(self.i, 128))](
                    self.projection, self.h, slots, self.i, 128, num_warps=4)
            if config is not None and config.l2 is not None:
                _C._sm120_moe_small(self.h, self.down, slots, weights, self.output, False, *config.l2)
            else:
                _C.m_grouped_bf16_gemm_nt_masked(self.h, self.down, self.expert, counts,
                    triton.cdiv(self.t*self.k, self.e), sm120_m_tile=self.fallback_m_tile)
                _combine[(self.t, triton.cdiv(self.d, 256))](self.expert, slots, weights,
                    self.output, self.d, self.k, 256, num_warps=4, enable_fp_fusion=False)
        return self.output
