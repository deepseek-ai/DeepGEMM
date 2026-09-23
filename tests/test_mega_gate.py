from typing import Any

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    bench_kineto,
    calc_diff,
    count_bytes,
    get_arch_major,
    test_filter,
)
from tile_kernels.moe import moe_topk_gate

BASELINE_KERNELS = ('nvjet', 'splitKreduce', 'cutlass', 'moe_topk_gate')

NUM_TOPK = 6
NUM_SHARED_EXPERTS = 1
ROUTED_SCALING_FACTOR = 1.5
VLLM_NUM_EXTRA_EXPERTS = 32

GATE_CONFIGS = (
    (64, 2, 6), (72, 1, 6), (96, 2, 6), (108, 2, 6), (128, 2, 6), (144, 2, 6),
    (256, 1, 6), (256, 2, 8), (384, 1, 6), (384, 1, 9), (512, 1, 6),
)


def enumerate_perf_cases():
    yield from ((num_tokens, hidden, 384)
                for num_tokens in (1, 3, 16, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768)
                for hidden in (4096, 5120, 7168))
    yield from (
        (512, 4096, 128), (1024, 4096, 128), (4096, 5120, 128), (32768, 7168, 128),
        (128, 4096, 256), (8192, 4352, 264), (4200, 7168, 256), (8192, 7168, 256),
        (128, 7168, 512), (8192, 4096, 512), (32768, 7168, 512),
    )


def make_physical_map(num_logical_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    to_physical_map = torch.full(
        (num_logical_experts, VLLM_NUM_EXTRA_EXPERTS + 1), -1, dtype=torch.int32, device='cuda')
    logical_count = torch.ones((num_logical_experts,), dtype=torch.int32, device='cuda')
    to_physical_map[:, 0] = torch.arange(num_logical_experts, dtype=torch.int32, device='cuda')
    num_duplicated_experts = min(num_logical_experts, VLLM_NUM_EXTRA_EXPERTS)
    to_physical_map[:num_duplicated_experts, 1] = (
        num_logical_experts + torch.arange(num_duplicated_experts, dtype=torch.int32, device='cuda'))
    logical_count[:num_duplicated_experts] = 2
    return to_physical_map, logical_count


def make_inputs(num_tokens: int, hidden: int, num_routed_experts: int,
                use_shared_as_routed: bool = False, with_bias: bool = True,
                with_image_bias: bool = False, with_mask: bool = False,
                with_physical_map: bool = True,
                num_shared_experts: int = NUM_SHARED_EXPERTS) -> dict[str, Any]:
    num_logical_experts = num_routed_experts + (num_shared_experts if use_shared_as_routed else 0)
    mask = torch.rand((num_tokens,), device='cuda') >= 0.25 if with_mask else None
    if mask is not None:
        mask[0] = True
    to_physical_map, logical_count = make_physical_map(num_logical_experts) if with_physical_map else (None, None)
    return dict(
        x=torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda'),
        weight=torch.randn((num_routed_experts, hidden), dtype=torch.bfloat16, device='cuda').mul_(hidden ** -0.5),
        mask=mask,
        bias=torch.randn((num_routed_experts,), dtype=torch.float32, device='cuda') if with_bias else None,
        image_bias=torch.randn((num_routed_experts,), dtype=torch.float32, device='cuda') if with_image_bias else None,
        image_token_mask=torch.rand((num_tokens,), device='cuda') < 0.5 if with_image_bias else None,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
    )


def make_outputs(inputs: dict[str, Any], use_shared_as_routed: bool = False,
                 with_unmapped_topk_idx: bool = False, num_topk: int = NUM_TOPK,
                 num_shared_experts: int = NUM_SHARED_EXPERTS) -> dict[str, torch.Tensor | None]:
    shape = (inputs['x'].shape[0], num_topk + (num_shared_experts if use_shared_as_routed else 0))
    return dict(
        topk_idx=torch.empty(shape, dtype=torch.int64, device='cuda'),
        topk_weights=torch.empty(shape, dtype=torch.float32, device='cuda'),
        unmapped_topk_idx=(torch.empty((shape[0], num_topk), dtype=torch.int64, device='cuda')
                           if with_unmapped_topk_idx else None),
    )


def run_mega_gate(inputs: dict[str, Any], outputs: dict[str, torch.Tensor | None], scoring_func: str,
                  use_shared_as_routed: bool = False,
                  num_topk: int = NUM_TOPK, num_shared_experts: int = NUM_SHARED_EXPERTS,
                  **overrides: Any) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {**inputs, **overrides}
    x, weight = kwargs.pop('x'), kwargs.pop('weight')
    unmapped_topk_idx = kwargs.pop('unmapped_topk_idx', outputs['unmapped_topk_idx'])
    return deep_gemm.bf16_mega_gate(
        x, weight, num_topk, use_shared_as_routed, num_shared_experts, ROUTED_SCALING_FACTOR, 0,
        scoring_func=scoring_func, unmapped_topk_idx=unmapped_topk_idx,
        out=(outputs['topk_idx'], outputs['topk_weights']), **kwargs)


def run_reference_gemm(inputs: dict[str, Any]) -> torch.Tensor:
    logits = torch.empty(
        (inputs['x'].shape[0], inputs['weight'].shape[0]), dtype=torch.float32, device='cuda')
    deep_gemm.cublaslt_gemm_nt(inputs['x'], inputs['weight'], logits)
    return logits


def apply_scoring(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    assert scoring_func in ('sigmoid', 'sqrtsoftplus', 'identity')
    return (torch.sigmoid(logits) if scoring_func == 'sigmoid' else
            F.softplus(logits).sqrt() if scoring_func == 'sqrtsoftplus' else logits)


def check_topk_weights(inputs: dict[str, Any], outputs: dict[str, torch.Tensor | None], logits: torch.Tensor,
                       scoring_func: str,
                       mask: torch.Tensor | None = None,
                       force_random: torch.Tensor | None = None,
                       fix_routing_mask: torch.Tensor | None = None,
                       num_topk: int = NUM_TOPK) -> float:
    fused_idx, fused_weights = outputs['topk_idx'], outputs['topk_weights']
    num_tokens, num_routed_experts = logits.shape
    active = mask if mask is not None else torch.ones((num_tokens,), dtype=torch.bool, device='cuda')
    score_based = active if force_random is None else active & ~force_random
    weight_diff = 0.0

    if score_based.any():
        scores = apply_scoring(logits, scoring_func)
        logical_topk_idx = outputs['unmapped_topk_idx']
        if logical_topk_idx is None:
            logical_topk_idx = fused_idx[:, :num_topk]
            to_physical_map = inputs['to_physical_map']
            if to_physical_map is not None:
                logical_topk_idx = torch.where(
                    logical_topk_idx < to_physical_map.shape[0],
                    logical_topk_idx, logical_topk_idx - to_physical_map.shape[0])

        selected_idx = logical_topk_idx[score_based]
        assert torch.all((selected_idx >= 0) & (selected_idx < num_routed_experts)).item()

        ranking_bias = (inputs['bias'] if inputs['bias'] is not None
                        else torch.zeros((num_routed_experts,), dtype=torch.float32, device='cuda'))
        if inputs['image_token_mask'] is not None:
            ranking_bias = torch.where(
                inputs['image_token_mask'][:, None], inputs['image_bias'], ranking_bias)
        ranking = scores + ranking_bias

        def reference_weights(row_mask: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
            topk_scores = scores[row_mask].gather(1, topk_idx)
            return topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20) * ROUTED_SCALING_FACTOR

        weight_diff = float(calc_diff(fused_weights[score_based, :num_topk],
                                      reference_weights(score_based, selected_idx)))
        assert weight_diff < 1e-11, f'{weight_diff=}'

        ranked = score_based if fix_routing_mask is None else score_based & ~fix_routing_mask
        if ranked.any():
            ranked_idx = logical_topk_idx[ranked]
            sorted_idx = ranked_idx.sort(dim=1).values
            assert torch.all(sorted_idx[:, 1:] != sorted_idx[:, :-1]).item()

            selected_ranking = ranking[ranked].gather(1, ranked_idx)
            unselected_max = ranking[ranked].scatter(1, ranked_idx, float('-inf')).max(dim=1).values
            crossing = unselected_max - selected_ranking.min(dim=1).values
            tolerance = 1e-5 + 1e-5 * ranking[ranked].abs().amax(dim=1)
            assert torch.all(crossing <= tolerance).item(), \
                f'top-k selection beyond numerical ties: {crossing.max().item()=}'

            ref_weights = reference_weights(ranked, ranking[ranked].topk(num_topk, dim=1).indices)
            ranked_diff = float(calc_diff(fused_weights[ranked, :num_topk].sort(dim=1).values,
                                          ref_weights.sort(dim=1).values))
            assert ranked_diff < 1e-11, f'{ranked_diff=}'
            weight_diff = max(weight_diff, ranked_diff)

        if fused_weights.shape[1] > num_topk:
            assert torch.all(fused_weights[score_based, num_topk:] == 1.0).item()

    if (~active).any():
        assert torch.all(fused_idx[~active] == -1).item()
        assert torch.all(fused_weights[~active] == 0).item()
        if outputs['unmapped_topk_idx'] is not None:
            assert torch.all(outputs['unmapped_topk_idx'][~active] == -1).item()

    return weight_diff


def check_correctness(inputs: dict[str, Any], outputs: dict[str, torch.Tensor | None], scoring_func: str,
                      use_shared_as_routed: bool = False,
                      num_topk: int = NUM_TOPK,
                      num_shared_experts: int = NUM_SHARED_EXPERTS) -> float:
    logits = run_reference_gemm(inputs)
    run_mega_gate(
        inputs, outputs, scoring_func, use_shared_as_routed,
        num_topk=num_topk, num_shared_experts=num_shared_experts)
    return check_topk_weights(
        inputs, outputs, logits, scoring_func,
        inputs['mask'], num_topk=num_topk)


def check_case(num_tokens: int, hidden: int, num_routed_experts: int,
               scoring_func: str = 'sqrtsoftplus', use_shared_as_routed: bool = False,
               num_topk: int = NUM_TOPK, num_shared_experts: int = NUM_SHARED_EXPERTS,
               with_unmapped_topk_idx: bool = False, **input_options: Any) -> None:
    inputs = make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed,
                         num_shared_experts=num_shared_experts, **input_options)
    outputs = make_outputs(inputs, use_shared_as_routed, with_unmapped_topk_idx, num_topk, num_shared_experts)
    check_correctness(inputs, outputs, scoring_func, use_shared_as_routed, num_topk, num_shared_experts)


def bench_baseline_gate(inputs: dict[str, Any]) -> float:
    outputs = make_outputs(inputs)
    logits = torch.empty(
        (inputs['x'].shape[0], inputs['weight'].shape[0]), dtype=torch.float32, device='cuda')
    gate_kwargs = {name: inputs[name] for name in
                   ('mask', 'bias', 'image_bias', 'image_token_mask', 'to_physical_map', 'logical_count')}

    def run_baseline():
        deep_gemm.cublaslt_gemm_nt(inputs['x'], inputs['weight'], logits)
        return moe_topk_gate(
            logits, NUM_TOPK, False, NUM_SHARED_EXPERTS, ROUTED_SCALING_FACTOR, 0,
            scoring_func='sqrtsoftplus', out=(outputs['topk_idx'], outputs['topk_weights']),
            unmapped_topk_idx=None, **gate_kwargs)

    return sum(bench_kineto(run_baseline, BASELINE_KERNELS, suppress_kineto_output=True))


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_mega_gate() -> None:
    torch.manual_seed(0)

    print('Testing BF16 GEMM + MoE top-k gate:')
    for num_tokens, hidden, num_routed_experts in enumerate_perf_cases():
        inputs = make_inputs(num_tokens, hidden, num_routed_experts)
        outputs = make_outputs(inputs)
        weight_diff = check_correctness(inputs, outputs, 'sqrtsoftplus')

        fused_t = bench_kineto(
            lambda: run_mega_gate(inputs, outputs, 'sqrtsoftplus'),
            'mega_gate', suppress_kineto_output=True)
        baseline_t = bench_baseline_gate(inputs)
        num_flops = 2 * num_tokens * num_routed_experts * hidden
        num_bytes = count_bytes(*inputs.values(), *outputs.values())
        print(f' > Perf (m={num_tokens:6}, n={num_routed_experts:4}, k={hidden:4}, topk={NUM_TOPK}, '
              f'sqrtsoftplus): '
              f'{fused_t * 1e6:7.1f} us | '
              f'{num_flops / fused_t / 1e12:4.0f} TFLOPS | '
              f'{num_bytes / fused_t / 1e9:4.0f} GB/s | '
              f'baseline {baseline_t * 1e6:7.1f} us, {baseline_t / fused_t:5.2f}x | '
              f'diff {weight_diff:.1e}')
    print()


def check_routing_controls() -> None:
    torch.manual_seed(1)
    num_tokens, hidden, num_routed_experts = 512, 4096, 384
    inputs = make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed=True, with_image_bias=True)
    outputs = make_outputs(inputs, use_shared_as_routed=True, with_unmapped_topk_idx=True)
    logits = run_reference_gemm(inputs)

    token_idx = torch.arange(num_tokens, device='cuda')
    mask, fix_routing_mask, force_random = token_idx % 11 != 0, token_idx % 3 == 0, token_idx % 5 == 0
    fixed_topk_idx = torch.randint(
        0, num_routed_experts, (num_tokens, NUM_TOPK), dtype=torch.int64, device='cuda')
    unmapped_topk_idx = outputs['unmapped_topk_idx']
    unmapped_topk_idx.copy_(fixed_topk_idx)

    fused_idx, fused_weights = run_mega_gate(
        inputs, outputs, 'sqrtsoftplus', True, mask=mask,
        fix_routing_mask=fix_routing_mask, force_random=force_random)
    check_topk_weights(
        inputs, outputs, logits, 'sqrtsoftplus', mask, force_random, fix_routing_mask)

    active_force_random = force_random & mask
    fixed = fix_routing_mask & mask & ~force_random
    assert torch.equal(unmapped_topk_idx[fixed], fixed_topk_idx[fixed])
    random_idx, random_weights = fused_idx[active_force_random], fused_weights[active_force_random]
    assert torch.all((random_idx >= 0) & (random_idx < inputs['logical_count'].sum())).item()
    assert torch.all(random_weights > 0).item()
    sorted_random_idx = random_idx.sort(dim=1).values
    assert torch.all(sorted_random_idx[:, 1:] != sorted_random_idx[:, :-1]).item()
    assert torch.all(unmapped_topk_idx[active_force_random] == -1).item()


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_mega_gate_api_contract() -> None:
    torch.manual_seed(2)

    for num_routed_experts, num_shared_experts, num_topk in GATE_CONFIGS:
        can_route_shared = (
            num_topk % num_shared_experts == 0 and
            num_routed_experts % (num_topk // num_shared_experts) == 0)
        for use_shared_as_routed in ((False, True) if can_route_shared else (False,)):
            check_case(16, 4096, num_routed_experts, use_shared_as_routed=use_shared_as_routed,
                       num_topk=num_topk, num_shared_experts=num_shared_experts)

    check_case(16, 4096, 128, 'identity', with_bias=False, with_physical_map=False)
    check_case(129, 4352, 260, 'sigmoid', with_image_bias=True, with_mask=True, with_unmapped_topk_idx=True)

    inputs = make_inputs(128, 4096, 384, with_image_bias=True)
    inputs['bias'].fill_(1e8)
    inputs['image_bias'].fill_(-1e8)
    check_correctness(inputs, make_outputs(inputs), 'sqrtsoftplus')

    check_routing_controls()

    num_topk = 8
    inputs = make_inputs(8, 4096, 128, with_bias=False, with_physical_map=False)
    inputs['x'].zero_()
    inputs['weight'].zero_()
    topk_idx, _ = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk), 'identity', num_topk=num_topk)
    assert torch.equal(topk_idx, torch.arange(num_topk, dtype=torch.int64, device='cuda').expand(8, -1))

    inputs = make_inputs(16, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    unmapped_storage = torch.empty((16, num_topk + 3), dtype=torch.int64, device='cuda')
    outputs['unmapped_topk_idx'] = unmapped_storage[:, :num_topk]
    assert outputs['unmapped_topk_idx'].stride() == (num_topk + 3, 1)
    check_correctness(inputs, outputs, 'sqrtsoftplus', num_topk=num_topk)

    inputs = make_inputs(32, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    force_random = torch.ones((32,), dtype=torch.bool, device='cuda')
    topk_idx, topk_weights = run_mega_gate(
        inputs, outputs, 'sqrtsoftplus', num_topk=num_topk, force_random=force_random)
    assert torch.all((topk_idx >= 0) & (topk_idx < 128)).item()
    assert torch.all(topk_weights > 0).item()
    sorted_topk_idx = topk_idx.sort(dim=1).values
    assert torch.all(sorted_topk_idx[:, 1:] != sorted_topk_idx[:, :-1]).item()
    assert torch.all(outputs['unmapped_topk_idx'] == -1).item()

    inputs = make_inputs(0, 4096, 128, with_physical_map=False)
    topk_idx, topk_weights = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk),
                                           'sqrtsoftplus', num_topk=num_topk)
    assert topk_idx.shape == (0, num_topk) and topk_weights.shape == (0, num_topk)

    inputs = make_inputs(16, 4096, 256)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
    deep_gemm.use_deterministic_algorithms(True)
    try:
        assert deep_gemm.get_bf16_mega_gate_config(16, 4096, 256, NUM_TOPK)['num_split_k'] == 1
        run_mega_gate(inputs, outputs, 'sqrtsoftplus')
        expected = {name: tensor.clone() for name, tensor in outputs.items()}
        for _ in range(100):
            run_mega_gate(inputs, outputs, 'sqrtsoftplus')
            for name, tensor in outputs.items():
                assert_bitwise_equal(tensor, expected[name], f'Deterministic invocation: {name}')
    finally:
        deep_gemm.use_deterministic_algorithms(False)


if __name__ == '__main__':
    torch.manual_seed(0)
    test_mega_gate_api_contract()
    test_mega_gate()
