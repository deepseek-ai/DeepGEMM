"""Single-SM100 MegaMoE lifecycle, shared-expert and main-API regressions."""

import os
import unittest

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.mega import SymmBuffer
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from test_mega_moe_situ import (
    CASES, HIDDEN, INTERMEDIATE, NUM_EXPERTS, NUM_TOKENS,
    _assert_rejected, _cast_weights_to_fp4, _decode_fp4, _decode_ue8m0,
    _find_free_port, _reference, _validate_oracle,
)


def _cast_fp8(tensor):
    squeeze = tensor.dim() == 2
    source = tensor.unsqueeze(0) if squeeze else tensor
    groups, n, k = source.shape
    values = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scales = torch.empty((groups, n, k // 32), dtype=torch.float32, device='cuda')
    for index in range(groups):
        values[index], scales[index] = per_token_cast_to_fp8(source[index], True, gran_k=32)
    scales = deep_gemm.transform_sf_into_required_layout(scales, n, k, (1, 32), groups)
    return (values.squeeze(0), scales.squeeze(0)) if squeeze else (values, scales)


def _decode_fp8(pair):
    values, scales = pair
    return values.cpu().float() * _decode_ue8m0(scales).repeat_interleave(32, dim=-1)


def _shared_input_scales(buffer, scales, num_tokens):
    block_m = deep_gemm.get_block_m_for_mega_moe(
        1, NUM_EXPERTS, buffer.num_max_tokens_per_rank, num_tokens,
        buffer.num_topk, buffer.mma_type)
    rows = torch.arange(num_tokens, device='cuda')
    local = rows % block_m
    transposed = local // 128 * 128 + local % 32 * 4 + local % 128 // 32
    destination = rows // block_m * ((block_m + 127) // 128 * 128) + transposed
    result = torch.zeros_like(buffer.shared_l1_acts_sf)
    result[destination] = scales[:num_tokens]
    return result


class _Fixture:
    def __init__(self, mma_type, shared_count, seed=20260730):
        self.mma_type = mma_type
        self.shared_count = shared_count
        self.bf16 = mma_type == 'bf16xbf16'
        generator = torch.Generator(device='cuda').manual_seed(seed)

        def randn(shape, scale):
            return torch.randn(shape, dtype=torch.bfloat16, device='cuda', generator=generator).mul_(scale)

        x = randn((NUM_TOKENS, HIDDEN), 0.5)
        l1 = randn((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN), 0.05)
        l2 = randn((NUM_EXPERTS, HIDDEN, INTERMEDIATE), 0.05)
        if self.bf16:
            self.x, self.x_sf, self.decoded_x = x, None, x.cpu().float()
            self.decoded_l1, self.decoded_l2 = l1.cpu().float(), l2.cpu().float()
        else:
            self.x, self.x_sf = per_token_cast_to_fp8(x, True, gran_k=32, use_packed_ue8m0=True)
            self.decoded_x = _decode_fp8((self.x, self.x_sf))
            cast = _cast_weights_to_fp4 if mma_type == 'fp8xfp4' else _cast_fp8
            decode = _decode_fp4 if mma_type == 'fp8xfp4' else _decode_fp8
            l1, l2 = cast(l1), cast(l2)
            self.decoded_l1, self.decoded_l2 = decode(l1), decode(l2)
        self.l1, self.l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
        self.shared_l1 = self.shared_l2 = self.decoded_shared = None
        if shared_count:
            shared_l1 = randn((2 * INTERMEDIATE * shared_count, HIDDEN), 0.05)
            shared_l2 = randn((HIDDEN, INTERMEDIATE * shared_count), 0.05)
            if self.bf16:
                self.decoded_shared = (shared_l1.cpu().float(), shared_l2.cpu().float())
            else:
                shared_l1, shared_l2 = _cast_fp8(shared_l1), _cast_fp8(shared_l2)
                self.decoded_shared = (_decode_fp8(shared_l1), _decode_fp8(shared_l2))
            self.shared_l1, self.shared_l2 = deep_gemm.transform_weights_for_mega_moe(shared_l1, shared_l2)

    def reference(self, routes, weights, case, clamp=None):
        return _reference(
            self.decoded_x[:routes.shape[0]], self.decoded_l1, self.decoded_l2,
            routes, weights, case, bf16=self.bf16, shared=self.decoded_shared,
            activation_clamp=clamp)

    def copy(self, buffer, routes, weights):
        n = routes.shape[0]
        buffer.x[:n].copy_(self.x[:n])
        if not self.bf16:
            buffer.x_sf[:n].copy_(self.x_sf[:n])
            if self.shared_count:
                buffer.shared_l1_acts_sf.copy_(_shared_input_scales(buffer, self.x_sf, n))
        buffer.topk_idx[:n].copy_(routes)
        buffer.topk_weights[:n].copy_(weights)

    def launch(self, buffer, y, stats, case, clamp=None):
        kwargs = dict(
            y=y, l1_weights=self.l1, l2_weights=self.l2, sym_buffer=buffer,
            shared_l1_weights=self.shared_l1, shared_l2_weights=self.shared_l2,
            cumulative_local_expert_recv_stats=stats,
            activation=case.activation, activation_clamp=clamp, fast_math=False)
        if case.activation == 'situ':
            kwargs.update(situ_beta=case.situ_beta, situ_linear_beta=case.situ_linear_beta)
        (deep_gemm.bf16_mega_moe if self.bf16 else deep_gemm.fp8_fp4_mega_moe)(**kwargs)


def _routes(n=NUM_TOKENS):
    rows = torch.arange(n, device='cuda')
    routes = torch.stack((rows % NUM_EXPERTS, (rows + 1) % NUM_EXPERTS), dim=-1)
    weights = torch.stack((torch.linspace(-1.25, 1.75, n, device='cuda'),
                           torch.full((n,), 0.375, device='cuda')), dim=-1)
    routes[::7, 1] = -1
    return routes, weights


def _eager(fixture, buffer, routes, weights, case, clamp=None):
    fixture.copy(buffer, routes, weights)
    y = torch.full((routes.shape[0], HIDDEN), float('nan'), dtype=torch.bfloat16, device='cuda')
    stats = torch.arange(NUM_EXPERTS, dtype=torch.int, device='cuda')
    before = stats.clone()
    fixture.launch(buffer, y, stats, case, clamp)
    torch.cuda.synchronize()
    counts = torch.bincount(routes[routes >= 0], minlength=NUM_EXPERTS)
    assert torch.equal(stats, before + counts)
    _validate_oracle(y, fixture.reference(routes, weights, case, clamp), 'eager')
    return y


def _graph_lifecycle(fixture, buffer, case, clamp=None):
    routes, weights = _routes()
    masked = torch.full_like(routes, -1)
    references = [fixture.reference(r, weights, case, clamp) for r in (routes, masked, routes)]
    fixture.copy(buffer, routes, weights)
    y = torch.full((NUM_TOKENS, HIDDEN), float('nan'), dtype=torch.bfloat16, device='cuda')
    stats = torch.arange(NUM_EXPERTS, dtype=torch.int, device='cuda')
    counts = torch.bincount(routes[routes >= 0], minlength=NUM_EXPERTS)
    initial = stats.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fixture.launch(buffer, y, stats, case, clamp)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.equal(stats, initial + 2 * counts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fixture.launch(buffer, y, stats, case, clamp)
    torch.cuda.synchronize()
    assert torch.equal(stats, initial + 2 * counts)
    for index, (current_routes, reference) in enumerate(zip((routes, masked, routes), references)):
        fixture.copy(buffer, current_routes, weights)
        y.fill_(float('nan'))
        before = stats.clone()
        graph.replay()
        torch.cuda.synchronize()
        increment = torch.bincount(current_routes[current_routes >= 0], minlength=NUM_EXPERTS)
        assert torch.equal(stats, before + increment)
        _validate_oracle(y, reference, f'graph/{fixture.mma_type}/{fixture.shared_count}/{index}')


def _base_reuse(group):
    alignment = deep_gemm._C.get_token_alignment_for_mega_moe()
    base = normal = smaller = other_group = None
    try:
        fixture = _Fixture('fp8xfp4', 0)
        base = SymmBuffer(group, NUM_EXPERTS, 2 * alignment, 2, HIDDEN, INTERMEDIATE,
                          mma_type='fp8xfp4', activation='situ')
        smaller = SymmBuffer(group, NUM_EXPERTS, alignment, 2, HIDDEN, INTERMEDIATE,
                             mma_type='fp8xfp4', activation='swiglu', base=base)
        assert smaller.buffer is base.buffer and smaller.handle is base.handle
        assert smaller.buffer.data_ptr() == base.buffer.data_ptr()
        normal = SymmBuffer(group, NUM_EXPERTS, 2 * alignment, 2, HIDDEN, INTERMEDIATE,
                            mma_type='fp8xfp4', activation='swiglu')
        routes, weights = _routes()
        expected = _eager(fixture, normal, routes, weights, CASES['swiglu'])
        first = _eager(fixture, base, routes, weights, CASES['swiglu'])
        assert torch.equal(expected, first)
        _eager(fixture, smaller, routes[:32], weights[:32], CASES['swiglu'])
        last = _eager(fixture, base, routes, weights, CASES['swiglu'])
        assert torch.equal(first, last)

        def reuse(candidate, process_group=group, capacity=alignment, mma_type='fp8xfp4'):
            view = SymmBuffer(process_group, NUM_EXPERTS, capacity, 2, HIDDEN, INTERMEDIATE,
                              mma_type=mma_type, base=candidate)
            view.destroy()

        _assert_rejected(lambda: reuse(base, capacity=128 * alignment), 'undersized base')
        other_group = dist.new_group([0])
        _assert_rejected(lambda: reuse(base, process_group=other_group), 'wrong group base')
        _assert_rejected(lambda: reuse(base, mma_type='fp4xfp4'), 'MX base for NVFP4')
        nv_base = SymmBuffer(group, NUM_EXPERTS, 2 * alignment, 2, HIDDEN, INTERMEDIATE,
                            mma_type='fp4xfp4')
        try:
            _assert_rejected(lambda: reuse(nv_base), 'NVFP4 base for MX')
        finally:
            nv_base.destroy()
        smaller.destroy()
        _assert_rejected(lambda: reuse(smaller), 'destroyed base')
    finally:
        for buffer in (smaller, normal, base):
            if buffer is not None:
                buffer.destroy()
        if other_group is not None:
            dist.destroy_process_group(other_group)


def _worker(local_rank, port):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE='1', RANK='0',
                      DG_COMM_KERNEL_DEBUG='0')
    try:
        _, _, group = init_dist(local_rank, 1)
        torch.set_default_device('cpu')
        for mma_type, shared_count, case in (
            ('bf16xbf16', 1, CASES['swiglu']),
            ('fp8xfp8', 1, CASES['swiglu']),
            ('fp8xfp4', 0, CASES['swiglu']),
            ('fp8xfp4', 0, CASES['situ_gate_low']),
            ('fp8xfp4', 1, CASES['situ_gate_low']),
            ('fp8xfp4', 2, CASES['situ_linear_low']),
        ):
            buffer = SymmBuffer(group, NUM_EXPERTS, NUM_TOKENS, 2, HIDDEN, INTERMEDIATE,
                                num_shared_experts=shared_count, mma_type=mma_type,
                                activation=case.activation)
            try:
                fixture = _Fixture(mma_type, shared_count)
                clamp = 1.0 if case.activation == 'swiglu' else None
                _graph_lifecycle(fixture, buffer, case, clamp)
            finally:
                buffer.destroy()
        _base_reuse(group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_main_contracts():
    if torch.cuda.device_count() < 1 or torch.cuda.get_device_capability(0)[0] != 10:
        raise unittest.SkipTest('requires one SM100-family CUDA device')
    torch.multiprocessing.spawn(_worker, args=(_find_free_port(),), nprocs=1, join=True)


if __name__ == '__main__':
    test_main_contracts()
