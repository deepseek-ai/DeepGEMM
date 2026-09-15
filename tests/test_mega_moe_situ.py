"""Deterministic single-GPU metamorphic and decoded-oracle regression for MegaMoE SiTU."""

import os
import socket
import unittest
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


NUM_RANKS = 1
NUM_EXPERTS = 4
NUM_TOPK = 1
NUM_TOKENS = 64
HIDDEN = 1024
INTERMEDIATE = 512


class Case(NamedTuple):
    activation: str
    situ_beta: Optional[float]
    situ_linear_beta: Optional[float]


CASES = {
    'swiglu': Case('swiglu', None, None),
    'situ_hi': Case('situ', 4096.0, 4096.0),
    'situ_gate_low': Case('situ', 0.25, 4096.0),
    'situ_linear_low': Case('situ', 4096.0, 0.5),
    'situ_tiny': Case('situ', 1e-40, 1e-40),
}


def _cast_weights_to_fp4(
        bf16_weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = bf16_weights.shape
    w = torch.empty(
        (num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    w_sf = torch.empty(
        (num_groups, n, k // 32), device='cuda', dtype=torch.float)
    for group_idx in range(num_groups):
        w[group_idx], w_sf[group_idx] = per_token_cast_to_fp4(
            bf16_weights[group_idx], use_ue8m0=True, gran_k=32)
    w_sf = deep_gemm.transform_sf_into_required_layout(
        w_sf, n, k, (1, 32), num_groups)
    return w, w_sf


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_f = actual.float()
    reference_f = reference.float()
    denominator = max(
        torch.linalg.vector_norm(reference_f).item(), 1e-12)
    return (
        torch.linalg.vector_norm(actual_f - reference_f).item()
        / denominator)


def _validate_suite(
        results: Dict[str, torch.Tensor], fast_math: bool
) -> None:
    mode = f'fast_math={fast_math}'
    for case_name, y in results.items():
        if not bool(torch.isfinite(y.float()).all().item()):
            raise AssertionError(f'{mode}/{case_name} contains NaN or Inf')

    swiglu = results['swiglu']
    if torch.linalg.vector_norm(swiglu.float()).item() <= 1e-6:
        raise AssertionError(f'{mode}/swiglu is identically zero or degenerate')
    hi = _relative_l2(results['situ_hi'], swiglu)
    gate_low = _relative_l2(results['situ_gate_low'], swiglu)
    linear_low = _relative_l2(results['situ_linear_low'], swiglu)
    low_vs_low = _relative_l2(
        results['situ_gate_low'], results['situ_linear_low'])

    low_floor = max(0.05, 4.0 * hi)
    low_pair_floor = max(0.025, 2.0 * hi)
    if hi >= 0.05:
        raise AssertionError(
            f'{mode}: SiTU(4096,4096) is not close to SwiGLU '
            f'(relative L2={hi:.6f})')
    if gate_low <= low_floor:
        raise AssertionError(
            f'{mode}: SiTU(.25,4096) is not significantly different '
            f'from SwiGLU ({gate_low:.6f} <= {low_floor:.6f})')
    if linear_low <= low_floor:
        raise AssertionError(
            f'{mode}: SiTU(4096,.5) is not significantly different '
            f'from SwiGLU ({linear_low:.6f} <= {low_floor:.6f})')
    if low_vs_low <= low_pair_floor:
        raise AssertionError(
            f'{mode}: the two low-beta controls are not distinct '
            f'({low_vs_low:.6f} <= {low_pair_floor:.6f})')


def _decode_ue8m0(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.cpu().to(torch.int64)
    exponents = torch.stack(
        [(packed >> shift) & 255 for shift in (0, 8, 16, 24)], dim=-1)
    return torch.pow(2.0, exponents.flatten(-2).double() - 127).float()


def _decode_fp4(pair: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    packed, sf = pair
    packed = packed.cpu().to(torch.int64)
    codes = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1).flatten(-2)
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device='cpu')
    values = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return values * _decode_ue8m0(sf).repeat_interleave(32, dim=-1)


def _requantize_fp8(x: torch.Tensor) -> torch.Tensor:
    groups = x.reshape(x.shape[0], -1, 32)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax.double() / 448.0))).float()
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(x)


def _reference(
        x: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor,
        topk_idx: torch.Tensor, topk_weights: torch.Tensor, case: Case,
        *, bf16: bool = False, shared: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        activation_clamp: Optional[float] = None
) -> torch.Tensor:
    topk_idx, topk_weights = topk_idx.cpu(), topk_weights.cpu()
    output = torch.zeros((x.shape[0], l2.shape[1]), dtype=torch.float32, device='cpu')
    for expert_idx in range(l1.shape[0]):
        rows, slots = torch.where(topk_idx == expert_idx)
        if rows.numel() == 0:
            continue
        projected = (x[rows].double() @ l1[expert_idx].double().T).float()
        gate, up = projected.to(torch.bfloat16).float().chunk(2, dim=-1)
        if activation_clamp is not None:
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(-activation_clamp, activation_clamp)
        if case.activation == 'situ':
            gated = torch.sigmoid(gate) * (case.situ_beta * torch.tanh(gate / case.situ_beta))
            linear = case.situ_linear_beta * torch.tanh(up / case.situ_linear_beta)
        else:
            gated, linear = torch.nn.functional.silu(gate), up
        activated = (gated * linear) * topk_weights[rows, slots, None]
        quantized = activated.to(torch.bfloat16).float() if bf16 else _requantize_fp8(activated)
        down = (quantized.double() @ l2[expert_idx].double().T).float().to(torch.bfloat16).float()
        output.index_add_(0, rows, down)
    if shared is not None:
        shared_idx = torch.zeros((x.shape[0], 1), dtype=torch.long, device='cpu')
        shared_weights = torch.ones((x.shape[0], 1), device='cpu')
        shared_output = _reference(
            x, shared[0].unsqueeze(0), shared[1].unsqueeze(0),
            shared_idx, shared_weights, case, bf16=bf16,
            activation_clamp=activation_clamp)
        output += shared_output.float()
    return output.to(torch.bfloat16)


def _validate_oracle(actual: torch.Tensor, reference: torch.Tensor, label: str) -> None:
    actual = actual.cpu()
    assert torch.isfinite(actual.float()).all(), label
    assert torch.isfinite(reference.float()).all(), label
    if torch.count_nonzero(reference) == 0:
        assert torch.count_nonzero(actual) == 0, label
    else:
        error = _relative_l2(actual, reference)
        assert error < 0.02, f'{label}: decoded reference relative L2={error:.6f}'


def _assert_rejected(call, label: str) -> None:
    try:
        call()
    except (AssertionError, RuntimeError, ValueError):
        return
    raise AssertionError(f'Invalid SiTU contract accepted: {label}')


def _worker(local_rank: int, master_port: int) -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['DG_COMM_KERNEL_DEBUG'] = '0'

    buffer = None
    try:
        _, _, group = init_dist(local_rank, NUM_RANKS)
        torch.set_default_device('cpu')
        generator = torch.Generator(device='cuda')
        generator.manual_seed(20260730)

        def fixed_randn(
                shape: Tuple[int, ...], scale: float
        ) -> torch.Tensor:
            value = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device='cuda',
                generator=generator)
            return value.mul_(scale)

        x_bf16 = fixed_randn((NUM_TOKENS, HIDDEN), 0.5)
        l1_bf16 = fixed_randn(
            (NUM_EXPERTS, INTERMEDIATE * 2, HIDDEN), 0.05)
        l1_bf16[0].zero_()
        l2_bf16 = fixed_randn(
            (NUM_EXPERTS, HIDDEN, INTERMEDIATE), 0.05)
        topk_idx = (
            torch.arange(NUM_TOKENS, device='cuda', dtype=torch.long)
            % NUM_EXPERTS
        ).view(NUM_TOKENS, NUM_TOPK)
        topk_weights = torch.ones(
            (NUM_TOKENS, NUM_TOPK), device='cuda', dtype=torch.float)

        x_fp8, x_sf = per_token_cast_to_fp8(
            x_bf16,
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True)
        l1_fp4, l2_fp4 = _cast_weights_to_fp4(l1_bf16), _cast_weights_to_fp4(l2_bf16)
        decoded_x = x_fp8.cpu().float() * _decode_ue8m0(x_sf).repeat_interleave(32, dim=-1)
        decoded_l1, decoded_l2 = _decode_fp4(l1_fp4), _decode_fp4(l2_fp4)
        transformed_l1, transformed_l2 = (
            deep_gemm.transform_weights_for_mega_moe(
                l1_fp4, l2_fp4,
                activation='situ'))
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            NUM_EXPERTS,
            NUM_TOKENS,
            NUM_TOPK,
            HIDDEN,
            INTERMEDIATE,
            mma_type='fp8xfp4',
            activation='situ')

        stats = torch.arange(NUM_EXPERTS, dtype=torch.int, device='cuda')

        def run_case(case: Case, fast_math: bool, **overrides) -> torch.Tensor:
            before_stats = stats.clone()
            buffer.x[:NUM_TOKENS].copy_(x_fp8)
            buffer.x_sf[:NUM_TOKENS].copy_(x_sf)
            buffer.topk_idx[:NUM_TOKENS].copy_(topk_idx)
            buffer.topk_weights[:NUM_TOKENS].copy_(topk_weights)

            y = torch.full(
                (NUM_TOKENS, HIDDEN),
                float('nan'),
                dtype=torch.bfloat16,
                device='cuda')
            kernel_kwargs = {
                'y': y,
                'l1_weights': transformed_l1,
                'l2_weights': transformed_l2,
                'sym_buffer': buffer,
                'activation': case.activation,
                'fast_math': fast_math,
                'cumulative_local_expert_recv_stats': stats,
            }
            if case.activation == 'situ':
                kernel_kwargs.update(
                    situ_beta=case.situ_beta,
                    situ_linear_beta=case.situ_linear_beta)
            kernel_kwargs.update(overrides)
            deep_gemm.fp8_fp4_mega_moe(**kernel_kwargs)
            torch.cuda.synchronize()
            counts = torch.bincount(topk_idx[topk_idx >= 0], minlength=NUM_EXPERTS)
            assert torch.equal(stats, before_stats + counts)
            return y

        references = {
            name: _reference(decoded_x, decoded_l1, decoded_l2, topk_idx, topk_weights, case)
            for name, case in CASES.items()
        }
        for fast_math in (True, False):
            results = {
                case_name: run_case(case, fast_math)
                for case_name, case in CASES.items()
            }
            _validate_suite(results, fast_math)
            for name, result in results.items():
                _validate_oracle(result, references[name], f'{fast_math=}/{name}')

        for parameter in ('situ_beta', 'situ_linear_beta'):
            for value in (None, 0.0, -1.0, float('nan'), float('inf'), -float('inf')):
                _assert_rejected(
                    lambda: run_case(CASES['situ_hi'], True, **{parameter: value}),
                    f'{parameter}={value}')
            _assert_rejected(
                lambda: run_case(CASES['swiglu'], True, **{parameter: 1.0}),
                f'SwiGLU with {parameter}')
        for clamp in (0.0, 1.0, float('inf')):
            _assert_rejected(
                lambda: run_case(CASES['situ_hi'], True, activation_clamp=clamp),
                f'explicit clamp {clamp}')
        fp8_l1 = (l1_bf16.to(torch.float8_e4m3fn), transformed_l1[1])
        fp8_l2 = (l2_bf16.to(torch.float8_e4m3fn), transformed_l2[1])
        _assert_rejected(
            lambda: run_case(CASES['situ_hi'], True, l1_weights=fp8_l1, l2_weights=fp8_l2),
            'FP8xFP8 weights')

        topk_weights[:, 0] = torch.linspace(-1.25, 1.75, NUM_TOKENS, device='cuda')
        topk_idx[::7] = -1
        for fast_math in (True, False):
            for name in ('situ_gate_low', 'situ_linear_low'):
                case = CASES[name]
                reference = _reference(decoded_x, decoded_l1, decoded_l2, topk_idx, topk_weights, case)
                _validate_oracle(run_case(case, fast_math), reference, f'routed/{fast_math=}/{name}')
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def test_situ_metamorphic() -> None:
    if torch.cuda.device_count() < NUM_RANKS:
        raise unittest.SkipTest('requires one CUDA device')
    if torch.cuda.get_device_capability(0)[0] != 10:
        raise unittest.SkipTest('requires an SM100 CUDA device')
    torch.multiprocessing.spawn(
        _worker,
        args=(_find_free_port(),),
        nprocs=NUM_RANKS,
        join=True)


if __name__ == '__main__':
    test_situ_metamorphic()
