import torch

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    assert_check,
    bench_kineto,
    calc_diff,
    count_bytes,
    get_arch_major,
    test_filter,
)
from deep_gemm.utils import align, per_token_cast_to_fp8


def enumerate_cases():
    for sf_layout in ('col', 'extra'):
        for hidden in (4096, 7168):
            for num_tokens in (1, 64, 65, 200, 256, 1025, 4096, 32768):
                yield sf_layout, num_tokens, hidden


def make_inputs(num_tokens: int, hidden: int, seed: int):
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
        shifted_prev_mix=randn((num_tokens, hc_mult, 1), torch.float).sigmoid(),
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
        inputs[name][list(set(duplicates))] = inputs[name][0].clone()
    return inputs, duplicates


def make_outputs(inputs, sf_layout: str, shared_sf_block_m: int = 224):
    num_tokens, hidden = inputs['x'].shape
    outputs = dict(
        new_residual=torch.empty_like(inputs['residual']),
        new_prev_mix=torch.empty_like(inputs['shifted_prev_mix']),
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


def extra_sf_rows(x: torch.Tensor, block_m: int) -> torch.Tensor:
    token_idx = torch.arange(x.size(0), device=x.device)
    index_in_block = token_idx % block_m
    return (
        token_idx // block_m * align(block_m, 128)
        + index_in_block // 128 * 128
        + index_in_block % 32 * 4
        + index_in_block % 128 // 32
    )


def _allocate_extra_sf(x: torch.Tensor, block_m: int) -> torch.Tensor:
    num_tokens, hidden = x.shape
    num_rows = (num_tokens + block_m - 1) // block_m * align(block_m, 128)
    return torch.empty_strided(
        (num_rows, hidden // 128), (1, num_rows), dtype=torch.int32, device=x.device)


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


@torch.no_grad()
def mhc_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    shifted_prev_mix: torch.Tensor,
    sf_layout: str = 'bf16',
    shared_sf_block_m: int = 0,
) -> dict[str, torch.Tensor]:
    new_residual = x.float().unsqueeze(1) * post_mix
    new_residual += torch.einsum(
        'tij,tih->tjh', comb_res_mix, residual.float())
    new_residual = new_residual.bfloat16()
    norm_input = (new_residual.float() * shifted_prev_mix).sum(dim=1).bfloat16()

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = new_residual.flatten(1).float() @ fn.mT
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    hc_sqr_sum = new_residual.float().square().sum((1, 2))
    mixes *= torch.rsqrt(
        hc_sqr_sum / (hc_mult * x.size(1)) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((
        mix_scales[0].expand(hc_mult),
        mix_scales[1].expand(hc_mult),
        mix_scales[2].expand(hc_mult * hc_mult),
    ))
    mixes = mixes * scales + mix_bases
    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (
        mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale
    ).unsqueeze(2)
    new_comb_res_mix = _sinkhorn_reference(
        mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult),
        num_sinkhorn_iters,
        sinkhorn_eps,
    )
    norm_input_float = norm_input.float()
    y_bf16 = (
        norm_input_float
        * torch.rsqrt(norm_input_float.square().mean(1) + rmsnorm_eps).unsqueeze(1)
        * (rmsnorm_weight.float() * rmsnorm_scale)
    ).bfloat16()
    result = {
        'new_residual': new_residual,
        'new_prev_mix': new_prev_mix,
        'new_post_mix': new_post_mix,
        'new_comb_res_mix': new_comb_res_mix,
        'y_bf16': y_bf16,
    }
    if sf_layout == 'bf16':
        return result

    y_fp8, y_fp8_sf = per_token_cast_to_fp8(
        y_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    if sf_layout == 'col':
        col_major_sf = torch.empty_strided(
            y_fp8_sf.shape, (1, align(x.size(0), 4)),
            dtype=torch.int32, device=x.device)
        col_major_sf.copy_(y_fp8_sf)
        y_fp8_sf = col_major_sf
    sf_name = 'y_gemm_sf' if sf_layout == 'col' else 'y_routed_sf'
    result.update(y_fp8=y_fp8, **{sf_name: y_fp8_sf})
    if sf_layout == 'extra':
        storage = _allocate_extra_sf(x, shared_sf_block_m)
        storage[extra_sf_rows(x, shared_sf_block_m)] = y_fp8_sf
        result['y_shared_sf_storage'] = storage
    return result


def logical_outputs(result, shared_sf_block_m=None):
    names = ('new_residual', 'new_prev_mix', 'new_post_mix', 'new_comb_res_mix', 'y_bf16')
    names += tuple(name for name in ('y_fp8', 'y_gemm_sf', 'y_routed_sf') if name in result)
    values = {name: result[name] for name in names}
    if 'y_shared_sf_storage' in result:
        y_bf16 = result['y_bf16']
        rows = extra_sf_rows(y_bf16, result.get('shared_sf_block_m', shared_sf_block_m))
        values['y_shared_sf'] = result['y_shared_sf_storage'][rows]
    return values


def comparable(tensor: torch.Tensor):
    return tensor.contiguous().view(torch.uint8) if tensor.dtype == torch.int32 else tensor


def check_correctness(actual, reference, case):
    max_diffs = []
    for name in actual:
        actual_tensor = comparable(actual[name])
        reference_tensor = comparable(reference[name])
        diff = float(calc_diff(actual_tensor, reference_tensor))
        limit = 2e-4 if name == 'y_fp8' else 5e-5
        assert_check(diff < limit, f'{case}, {name=}, {diff=}')
        max_diffs.append(diff)
    return max_diffs


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_mega_mhc_api_contract() -> None:
    def check_outputs(outputs, expected, label):
        for name, tensor in logical_outputs(outputs).items():
            assert_bitwise_equal(tensor, expected[name], f'{label}: {name}')

    num_tokens, hidden = 64, 4096
    inputs, _ = make_inputs(num_tokens, hidden, 2026)

    # Deterministic mode selects fixed Split-K independently for every invocation
    outputs = make_outputs(inputs, 'bf16')
    deep_gemm.use_deterministic_algorithms(True)
    try:
        run_mega_mhc(inputs, outputs)
        expected = {name: tensor.clone() for name, tensor in logical_outputs(outputs).items()}
        run_mega_mhc(inputs, outputs)
        check_outputs(outputs, expected, 'Deterministic invocation')
    finally:
        deep_gemm.use_deterministic_algorithms(False)

    # FP8-only calls remain bitwise stable after a different-sized invocation
    small_inputs, _ = make_inputs(1, hidden, 2027)
    outputs = make_outputs(inputs, 'extra')
    run_mega_mhc(inputs, outputs)
    names = ('y_fp8', 'y_routed_sf', 'y_shared_sf_storage')
    expected = {name: outputs[name].clone() for name in names}
    run_mega_mhc(small_inputs, make_outputs(small_inputs, 'extra'))
    run_mega_mhc(inputs, outputs, y_bf16=None)
    for name in names:
        assert_bitwise_equal(outputs[name], expected[name], f'FP8-only: {name}')

    # A stream must initialize its split barriers before CUDA Graph capture
    outputs = make_outputs(inputs, 'bf16')
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=torch.cuda.Stream()):
            run_mega_mhc(inputs, outputs)
    except RuntimeError as error:
        assert 'CaptureStatus::None' in str(error)
    else:
        raise AssertionError('Mega mHC must be warmed up before CUDA Graph capture')

    # Concurrent kernels on different streams use independent split barriers
    eager_inputs = inputs
    eager_outputs = make_outputs(eager_inputs, 'bf16')
    run_mega_mhc(eager_inputs, eager_outputs)
    eager_expected = {name: tensor.clone() for name, tensor in logical_outputs(eager_outputs).items()}

    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    stream_outputs = [make_outputs(inputs, 'bf16') for _ in streams]
    for stream, outputs in zip(streams, stream_outputs):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_mega_mhc(inputs, outputs)
    torch.cuda.synchronize()

    # Fork both cooperative grids from one graph so repeated launches can contend for SMs.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_stream = torch.cuda.current_stream()
        for stream, outputs in zip(streams, stream_outputs):
            stream.wait_stream(capture_stream)
            with torch.cuda.stream(stream):
                run_mega_mhc(inputs, outputs)
        for stream in streams:
            capture_stream.wait_stream(stream)
    for _ in range(20):
        graph.replay()
    torch.cuda.current_stream().synchronize()

    for stream_idx, outputs in enumerate(stream_outputs):
        check_outputs(outputs, eager_expected, f'Multi-stream {stream_idx}')

    # Share a stream while changing token count, split count and output layout
    stream = streams[0]
    cases = []
    for tokens, hidden, sf_layout in ((1, 4096, 'bf16'), (65, 4096, 'col'), (200, 7168, 'extra')):
        inputs, _ = make_inputs(tokens, hidden, 2026)
        outputs = make_outputs(inputs, sf_layout)
        x_versions = (inputs['x'].clone(), -inputs['x'])
        expected = []
        for x in x_versions:
            inputs['x'].copy_(x)
            run_mega_mhc(inputs, outputs)
            expected.append({name: tensor.clone() for name, tensor in logical_outputs(outputs).items()})
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_mega_mhc(inputs, outputs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run_mega_mhc(inputs, outputs)
        cases.append((graph, inputs, outputs, x_versions, expected))

    # Alternate inputs and interleave graphs and eager calls on the same split barriers.
    for replay_idx in range(20):
        for graph, inputs, outputs, x_versions, expected in reversed(cases):
            version = replay_idx % 2
            with torch.cuda.stream(stream):
                inputs['x'].copy_(x_versions[version])
                if version == 1:
                    run_mega_mhc(eager_inputs, eager_outputs)
                graph.replay()
            stream.synchronize()
            check_outputs(outputs, expected[version], f'Graph replay {replay_idx}: {inputs["x"].shape}')
            if version == 1:
                check_outputs(eager_outputs, eager_expected, f'Eager between replays: {replay_idx}')


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_mega_mhc() -> None:
    print('Testing Mega shifted mHC:')
    for case_idx, (sf_layout, num_tokens, hidden) in enumerate(enumerate_cases()):
        inputs, duplicates = make_inputs(
            num_tokens, hidden, 2026 + case_idx)
        outputs = make_outputs(inputs, sf_layout)
        ref_kwargs = dict(
            **inputs, sf_layout=sf_layout,
            shared_sf_block_m=outputs.get('shared_sf_block_m', 0))
        reference = mhc_reference(**ref_kwargs)
        run_mega_mhc(inputs, outputs)
        actual = logical_outputs(outputs)
        ref = logical_outputs(reference, outputs.get('shared_sf_block_m'))
        diffs = check_correctness(actual, ref, (sf_layout, num_tokens, hidden))

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
        logical_inputs = [tensor for tensor in inputs.values() if isinstance(tensor, torch.Tensor)]
        logical_bytes = count_bytes(*logical_inputs, *actual.values())
        intermediate_io_bytes = 2 * count_bytes(outputs['y_bf16'])
        print(f' > {sf_layout:5} T={num_tokens:5}, H={hidden:4}: {kernel_t * 1e6:6.1f} us, '
              f'{logical_bytes / kernel_t / 1e9:4.0f} GB/s '
              f'({(logical_bytes + intermediate_io_bytes) / kernel_t / 1e9:4.0f} incl. Scratch I/O) | '
              f'diff {max(diffs):.1e}')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_mega_mhc_api_contract()
    test_mega_mhc()
