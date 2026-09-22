import os
import sys

import torch

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    bench_kineto,
    calc_diff,
    count_bytes,
    get_arch_major,
    test_filter,
)
from deep_gemm.utils import align

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third-party'))
from tilelang_ops import ref_mhc  # noqa: E402


def enumerate_cases():
    for sf_layout in ('col', 'extra'):
        for hidden in (4096, 7168):
            for num_tokens in (1, 64, 65, 200, 1025, 4096, 32768):
                yield sf_layout, num_tokens, hidden


def make_inputs(num_tokens: int, hidden: int, seed: int, is_shifted: bool):
    hc_mult = 4
    num_hc_outputs = hc_mult * (hc_mult + 2)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    def randn(shape, dtype):
        return torch.randn(shape, dtype=dtype, device='cuda', generator=generator)

    inputs = dict(
        x=randn((num_tokens, hidden), torch.bfloat16),
        residual=randn((num_tokens, hc_mult, hidden), torch.bfloat16),
        post_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
        comb_res_mix=torch.rand((num_tokens, hc_mult, hc_mult), device='cuda', generator=generator),
        shifted_prev_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid() if is_shifted else None,
        fn=randn((num_hc_outputs, hc_mult * hidden), torch.float).mul_(1e-2),
        mix_scales=randn((3,), torch.float).mul_(0.1),
        mix_bases=randn((num_hc_outputs,), torch.float).mul_(0.1),
        rmsnorm_weight=randn((hidden,), torch.float).mul_(0.1).add_(1).bfloat16(),
        hc_mult=hc_mult,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        num_sinkhorn_iters=10,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=1.25,
    )
    for _ in range(3):
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-1, keepdim=True)
        inputs['comb_res_mix'] /= inputs['comb_res_mix'].sum(-2, keepdim=True)
    duplicates = tuple(i for i in (63, 64, 511, 512, num_tokens - 1) if 0 < i < num_tokens)
    for name in ('x', 'residual', 'post_mix', 'comb_res_mix', 'shifted_prev_mix'):
        if inputs[name] is not None:
            inputs[name][list(set(duplicates))] = inputs[name][0].clone()
    return inputs, duplicates


def make_outputs(inputs, sf_layout: str, shared_sf_block_m: int = 224):
    num_tokens, hidden = inputs['x'].shape
    is_shifted = inputs['shifted_prev_mix'] is not None
    outputs = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=torch.empty_like(inputs['shifted_prev_mix']) if is_shifted else None,
        new_post_mix=torch.empty_like(inputs['post_mix']),
        new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
        y_bf16=torch.empty_like(inputs['x']),
    )
    if sf_layout == 'bf16':
        return outputs
    sf_shape = (num_tokens, hidden // 128)
    outputs['y_fp8'] = torch.empty((num_tokens, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    if sf_layout == 'col':
        outputs['y_gemm_sf'] = torch.empty_strided(
            sf_shape, (1, align(num_tokens, 4)), dtype=torch.int32, device='cuda')
    else:
        outputs['y_routed_sf'] = torch.empty(sf_shape, dtype=torch.int32, device='cuda')
        num_rows = (num_tokens + shared_sf_block_m - 1) // shared_sf_block_m * align(shared_sf_block_m, 128)
        storage = torch.empty_strided(
            (num_rows, hidden // 128), (1, num_rows), dtype=torch.int32, device='cuda')
        outputs.update(
            y_shared_sf=storage[:num_tokens],
            y_shared_sf_storage=storage,
            shared_sf_block_m=shared_sf_block_m,
        )
    return outputs


def run_mega_mhc(inputs, outputs, **overrides):
    kwargs = {**inputs, **outputs}
    kwargs.pop('y_shared_sf_storage', None)
    kwargs.update(overrides)
    deep_gemm.mega_mhc(**kwargs)


def logical_outputs(result, shared_sf_block_m=None):
    names = ('new_residual', 'new_post_mix', 'new_comb_res_mix', 'y_bf16')
    names += ('new_prev_mix',) if result.get('new_prev_mix') is not None else ()
    names += tuple(name for name in ('y_fp8', 'y_gemm_sf', 'y_routed_sf') if name in result)
    values = {name: result[name] for name in names}
    if 'y_shared_sf_storage' in result:
        y_bf16 = result['y_bf16']
        rows = ref_mhc.extra_sf_rows(y_bf16, result.get('shared_sf_block_m', shared_sf_block_m))
        values['y_shared_sf'] = result['y_shared_sf_storage'][rows]
    return values


def comparable(tensor: torch.Tensor):
    return tensor.contiguous().view(torch.uint8) if tensor.dtype == torch.int32 else tensor


def check_correctness(actual, pytorch_ref, baseline_ref, case):
    max_diffs = [0.0, 0.0, 0.0]
    for name in actual:
        tensors = tuple(comparable(result[name]) for result in (actual, pytorch_ref, baseline_ref))
        diffs = tuple(float(calc_diff(tensors[i], tensors[j])) for i, j in ((0, 1), (0, 2), (2, 1)))
        pytorch_limit = 2e-4 if name == 'y_fp8' else 5e-5
        baseline_limit = 1e-5 if case[0] == 'normal' and name == 'y_fp8' else 1e-6
        limits = (pytorch_limit, baseline_limit, pytorch_limit)
        assert all(diff < limit for diff, limit in zip(diffs, limits)), \
            f'{case}, {name=}, {diffs=}'
        max_diffs = [max(old, new) for old, new in zip(max_diffs, diffs)]
    return max_diffs


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_mega_mhc_api_contract() -> None:
    num_tokens, hidden = 64, 4096
    inputs, _ = make_inputs(num_tokens, hidden, 2026, False)

    # Deterministic mode selects fixed Split-K independently for every invocation
    outputs = make_outputs(inputs, 'bf16')
    deep_gemm.use_deterministic_algorithms(True)
    try:
        run_mega_mhc(inputs, outputs)
        expected = {name: tensor.clone() for name, tensor in logical_outputs(outputs).items()}
        run_mega_mhc(inputs, outputs)
        for name, tensor in logical_outputs(outputs).items():
            assert_bitwise_equal(tensor, expected[name], f'Deterministic invocation: {name}')
    finally:
        deep_gemm.use_deterministic_algorithms(False)

    # A stream must initialize its split barriers before CUDA Graph capture
    expected_outputs = make_outputs(inputs, 'bf16')
    run_mega_mhc(inputs, expected_outputs)
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graph_outputs = make_outputs(inputs, 'bf16')
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=capture_stream):
            graph_outputs['y_bf16'].zero_()
            run_mega_mhc(inputs, graph_outputs)
    except RuntimeError as error:
        assert 'CaptureStatus::None' in str(error)
    else:
        raise AssertionError('Mega mHC must be warmed up before CUDA Graph capture')

    with torch.cuda.stream(capture_stream):
        run_mega_mhc(inputs, graph_outputs)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        run_mega_mhc(inputs, graph_outputs)
    graph.replay()
    capture_stream.synchronize()
    for name, tensor in logical_outputs(graph_outputs).items():
        assert_bitwise_equal(tensor, expected_outputs[name], f'CUDA Graph: {name}')

    # FP8-only calls remain bitwise stable after a different-sized invocation
    for is_shifted in (False, True):
        inputs, _ = make_inputs(num_tokens, hidden, 2026, is_shifted)
        small_inputs, _ = make_inputs(1, hidden, 2027, is_shifted)
        outputs = make_outputs(inputs, 'extra')
        run_mega_mhc(inputs, outputs)
        names = ('y_fp8', 'y_routed_sf', 'y_shared_sf_storage')
        expected = {name: outputs[name].clone() for name in names}
        run_mega_mhc(small_inputs, make_outputs(small_inputs, 'extra'))
        run_mega_mhc(inputs, outputs, y_bf16=None)
        for name in names:
            assert_bitwise_equal(outputs[name], expected[name], f'FP8-only: {name}')

    # Concurrent kernels on different streams use independent split barriers
    inputs, _ = make_inputs(num_tokens, hidden, 2028, False)
    expected_outputs = make_outputs(inputs, 'bf16')
    run_mega_mhc(inputs, expected_outputs)
    torch.cuda.synchronize()

    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    stream_outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            outputs = make_outputs(inputs, 'bf16')
            run_mega_mhc(inputs, outputs)
            stream_outputs.append(outputs)
    for stream in streams:
        stream.synchronize()
    for stream_idx, outputs in enumerate(stream_outputs):
        for name, tensor in logical_outputs(outputs).items():
            assert_bitwise_equal(tensor, expected_outputs[name], f'Multi-stream {stream_idx}: {name}')


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_mega_mhc() -> None:
    if not ref_mhc.has_baseline():
        return
    for is_shifted in (True, False):
        mode = 'shifted' if is_shifted else 'normal'
        baseline_kernels = ref_mhc.SHIFTED_KERNELS if is_shifted else ref_mhc.NORMAL_KERNELS
        print(f'Testing Mega {mode} mHC:')
        for case_idx, (sf_layout, num_tokens, hidden) in enumerate(enumerate_cases()):
            inputs, duplicates = make_inputs(
                num_tokens, hidden, 2026 + case_idx + (0 if is_shifted else 1000), is_shifted)
            outputs = make_outputs(inputs, sf_layout)
            ref_kwargs = dict(
                **inputs, sf_layout=sf_layout,
                shared_sf_block_m=outputs.get('shared_sf_block_m', 0))

            def run_baseline():
                return ref_mhc.mhc_baseline(**ref_kwargs)

            pytorch_ref, baseline_ref = ref_mhc.mhc_reference(**ref_kwargs), run_baseline()
            run_mega_mhc(inputs, outputs)
            actual = logical_outputs(outputs)
            refs = tuple(logical_outputs(ref, outputs.get('shared_sf_block_m'))
                         for ref in (pytorch_ref, baseline_ref))
            diffs = check_correctness(actual, *refs, (mode, sf_layout, num_tokens, hidden))

            expected = {name: tensor.clone() for name, tensor in actual.items()}
            for _ in range(30):
                run_mega_mhc(inputs, outputs)
                actual = logical_outputs(outputs)
                for name in actual:
                    assert_bitwise_equal(actual[name], expected[name], name)
            for token_idx in duplicates:
                for name, tensor in actual.items():
                    assert_bitwise_equal(tensor[token_idx], tensor[0], f'{name}, token={token_idx}')

            kernel_t = bench_kineto(lambda: run_mega_mhc(inputs, outputs),
                                    'sm100_mega_mhc', suppress_kineto_output=True)
            baseline_times = bench_kineto(run_baseline, baseline_kernels, suppress_kineto_output=True)
            baseline_t = sum(baseline_times)
            logical_inputs = [tensor for tensor in inputs.values() if isinstance(tensor, torch.Tensor)]
            logical_bytes = count_bytes(*logical_inputs, *actual.values())
            # Materialized Norm/Cast boundary, plus the normal-only Res reread
            intermediate_io_bytes = 2 * count_bytes(outputs['y_bf16'])
            if not is_shifted:
                intermediate_io_bytes += count_bytes(outputs['new_residual'])
            print(f' > {sf_layout:5} T={num_tokens:5}, H={hidden:4}: {kernel_t * 1e6:6.1f} us, '
                  f'{logical_bytes / kernel_t / 1e9:4.0f} GB/s '
                  f'({(logical_bytes + intermediate_io_bytes) / kernel_t / 1e9:4.0f} incl. Scratch I/O) | '
                  f'baseline {baseline_t * 1e6:6.1f} us, {baseline_t / kernel_t:5.2f}x | '
                  f'diff K/P={diffs[0]:.1e}, K/B={diffs[1]:.1e}, B/P={diffs[2]:.1e}')
    print()


if __name__ == '__main__':
    test_mega_mhc_api_contract()
    test_mega_mhc()
