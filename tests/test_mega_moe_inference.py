import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Optional, Tuple

import deep_gemm
from deep_gemm.utils import align, per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


def _to_shared_mega_moe_inference_sf_layout(sf: torch.Tensor, block_m: int, num_max_sf_tokens: int) -> torch.Tensor:
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = align(block_m, 128)
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype, device=sf.device)
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, num_tokens - block_idx * block_m)
        for m_idx in range(num_block_tokens):
            transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            result[block_idx * aligned_block_m + transposed_m_idx].copy_(sf[block_idx * block_m + m_idx])
    return result


def _cast_fp8_for_mega_moe_inference(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, align(mn, 4)), dtype=x_sf.dtype, device=x_sf.device)
    x_sf_tma.copy_(x_sf)
    return x_fp8, x_sf, x_sf_tma


def _copy_fp8_sf(dst: torch.Tensor, src: torch.Tensor, num_tokens: int) -> None:
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


def _rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    num_tokens, hidden = x.shape
    # Match the combine layout selected by the FP8/FP4 heuristic with eight epilogue warps.
    num_chunks = 1 if hidden <= 4096 else 2
    x = x.float()
    values = x.reshape(num_tokens, num_chunks, hidden // (num_chunks * 256), 32, 4, 2)
    sum_sq = torch.zeros_like(values[:, :, 0])
    for value in values.unbind(2):
        sum_sq = torch.addcmul(sum_sq, value, value)
    sum_sq = (sum_sq[..., 0, :] + sum_sq[..., 1, :]) + (sum_sq[..., 2, :] + sum_sq[..., 3, :])
    sum_sq = sum_sq[..., 0] + sum_sq[..., 1]
    # Reproduce lane 0 of the XOR warp reduction, then combine the chunks.
    for offset in (16, 8, 4, 2, 1):
        sum_sq = sum_sq[..., :offset] + sum_sq[..., offset:]
    sum_sq = sum_sq[:, :, 0]
    sum_sq = sum_sq[:, 0] if num_chunks == 1 else sum_sq[:, 0] + sum_sq[:, 1]
    # Match the FP32 reciprocal and fused multiply-add used before CUDA rsqrtf.
    inv_rms = torch.rsqrt(torch.addcmul(torch.full_like(sum_sq, eps), sum_sq, torch.full_like(sum_sq, 1.0 / hidden)))
    return (x * inv_rms[:, None] * weight.float()).to(weight.dtype)


def _fp32_combine_reference(x: torch.Tensor, handle, shared: Optional[torch.Tensor],
                            group: dist.ProcessGroup) -> torch.Tensor:
    """Reverse expanded dispatch without rounding the sum to BF16 (correctness only)."""
    assert handle.do_expand
    num_tokens, num_topk = handle.topk_idx.shape
    capacity, hidden = handle.num_max_tokens_per_rank, x.size(1)
    # Metadata holds the global source token and one expanded L2 row per top-k slot.
    # The remaining allocated rows and expert padding must never be read.
    counts = getattr(handle, 'psum_num_recv_tokens_per_scaleup_rank', None)
    if counts is None:
        counts = handle.psum_num_recv_tokens_per_rank
    metadata = handle.recv_src_metadata[:int(counts[-1].item())]
    assert metadata.size(1) == num_topk + 2
    combined = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x.device)
    if shared is not None:
        combined.copy_(shared.float())

    send = torch.empty((group.size() * capacity, hidden), dtype=x.dtype, device=x.device)
    received = torch.empty((capacity, hidden), dtype=x.dtype, device=x.device)
    for topk_slot_idx in range(num_topk):
        rows = metadata[:, 2 + topk_slot_idx].long()
        valid = rows >= 0
        send.zero_()
        send.index_copy_(0, metadata[valid, 0].long(), x[rows[valid]])
        # A token/slot has exactly one owner, so this transports BF16 values exactly.
        dist.reduce_scatter_tensor(received, send, group=group)
        combined.add_(received[:num_tokens].float())
    return combined


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    deep_gemm.use_deterministic_algorithms(True)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    is_bf16xbf16 = args.mma_type == 'bf16xbf16'
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    num_shared_experts = args.num_shared_experts
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts
    assert num_tokens <= num_max_tokens_per_rank
    assert not (args.rms_norm and is_bf16xbf16), 'RMSNorm requires FP8/FP4 Mega MoE'
    assert not args.rms_norm or (args.rms_eps is not None and args.rms_eps > 0)
    output_dtype = torch.float32 if args.output_dtype == 'fp32' else torch.bfloat16
    fp32_output = output_dtype == torch.float32
    assert not (fp32_output and is_bf16xbf16), 'FP32 output requires FP8/FP4 Mega MoE'

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe_inference(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared_experts,
        mma_type=args.mma_type
    )

    # Cast routed weights into FP4 or FP8
    def _cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    def _cast_weights_to_fp8(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k), device='cuda', dtype=torch.float8_e4m3fn)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp8(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, shared_x, shared_l1_x_sf, topk_idx, topk_weights, l1_weights, l2_weights
        global transformed_l1_weights, transformed_l2_weights, rms_weights
        global shared_l1_weights, shared_l2_weights, transformed_shared_l1_weights, transformed_shared_l2_weights
        global cumulative_local_expert_recv_stats_fused, cumulative_local_expert_recv_stats_baseline
        global initial_cumulative_local_expert_recv_stats_fused, initial_cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_fused = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_baseline.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        if num_shared_experts > 0:
            shared_l1_weights = torch.randn(
                (shared_intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
            shared_l2_weights = torch.randn(
                (hidden, shared_intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        else:
            shared_l1_weights = shared_l2_weights = None

        if not is_bf16xbf16:
            # FP8 path: cast inputs and weights with per-32 UE8M0 SF
            assert hidden % 128 == 0 and intermediate_hidden % 128 == 0 and shared_intermediate_hidden % 128 == 0
            block_m = deep_gemm.get_block_m_for_mega_moe_inference(
                num_ranks, num_experts, buffer.num_max_tokens_per_rank, num_tokens, num_topk, args.mma_type)
            x_fp8, x_sf, x_sf_tma = _cast_fp8_for_mega_moe_inference(x)
            x = (x_fp8, x_sf)
            shared_x = (x_fp8, x_sf_tma)
            if num_shared_experts > 0:
                shared_l1_x_sf = _to_shared_mega_moe_inference_sf_layout(x_sf, block_m, buffer.shared_l1_acts_sf.shape[0])
            cast_weights = _cast_weights_to_fp8 if args.mma_type == 'fp8xfp8' else _cast_weights_to_fp4
            l1_weights = cast_weights(l1_weights)
            l2_weights = cast_weights(l2_weights)
            if num_shared_experts > 0:
                shared_l1_weights = _cast_fp8_for_mega_moe_inference(shared_l1_weights)[0::2]
                shared_l2_weights = _cast_fp8_for_mega_moe_inference(shared_l2_weights)[0::2]

        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe_inference(l1_weights, l2_weights))
        if num_shared_experts > 0:
            transformed_shared_l1_weights, transformed_shared_l2_weights = (
                deep_gemm.transform_weights_for_mega_moe_inference(shared_l1_weights, shared_l2_weights))
        else:
            transformed_shared_l1_weights = transformed_shared_l2_weights = None

        if args.use_localization:
            def localize(transformed):
                if isinstance(transformed, tuple):
                    return deep_gemm.localize(transformed[0]), transformed[1]
                return deep_gemm.localize(transformed)
            transformed_l1_weights, transformed_l2_weights = localize(transformed_l1_weights), localize(transformed_l2_weights)
            if num_shared_experts > 0:
                transformed_shared_l1_weights, transformed_shared_l2_weights = localize(transformed_shared_l1_weights), localize(transformed_shared_l2_weights)

        # Generate in both modes to keep the same inputs across ON/OFF runs.
        rms_weights = torch.randn((hidden, ), dtype=output_dtype, device='cuda')

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def copy_inputs_to_buffer():
        if is_bf16xbf16:
            buffer.x[:num_tokens].copy_(x)
        else:
            buffer.x[:num_tokens].copy_(x[0])
            buffer.x_sf[:num_tokens].copy_(x[1])
            if num_shared_experts > 0:
                _copy_fp8_sf(buffer.shared_l1_acts_sf, shared_l1_x_sf, num_tokens)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def run_fused():
        cumulative_local_expert_recv_stats_fused.copy_(initial_cumulative_local_expert_recv_stats_fused)
        copy_inputs_to_buffer()

        y = torch.empty((num_tokens, hidden), dtype=output_dtype, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math))
        if not is_bf16xbf16:
            kernel_kwargs.update(rms_weights=rms_weights if args.rms_norm else None, rms_eps=args.rms_eps)
        if num_shared_experts > 0:
            kernel_kwargs.update(
                shared_l1_weights=transformed_shared_l1_weights,
                shared_l2_weights=transformed_shared_l2_weights
            )
        (deep_gemm.bf16_mega_moe_inference if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe_inference)(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Output: {args.output_dtype}, RMSNorm: {bool(args.rms_norm)}, eps: {args.rms_eps}', once_in_node=True)
    dist_print(f' > Localized weights: {args.use_localization}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Shared experts: {num_shared_experts}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        deep_gemm.destroy_localizer()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    ep_buffer_type = (getattr(deep_ep, 'EPBuffer', None) or deep_ep.ElasticBuffer) if is_legacy_loaded else None
    ep_buffer = ep_buffer_type(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=not is_bf16xbf16,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
    ) if is_legacy_loaded else None

    # Baseline params differ by mma type
    run_baseline = None
    if is_legacy_loaded:
        if is_bf16xbf16:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False, 'do_expand': True}
            gemm_fn = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
            gemm_kwargs = {'compiled_dims': '', 'use_psum_layout': True}
            swiglu_kwargs = {'round_scale': False, 'ue8m0_scale': False, 'output_bf16': True}
            get_num_tokens = lambda recv_x: recv_x.size(0)
        else:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False,
                               'do_expand': True, 'use_tma_aligned_col_major_sf': True}
            gemm_fn = deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous
            gemm_kwargs = {'use_psum_layout': True, 'recipe': (1, 1, 32)}
            swiglu_kwargs = {'round_scale': True, 'ue8m0_scale': True, 'output_bf16': False}
            get_num_tokens = lambda recv_x: recv_x[0].size(0)

        def get_baseline_shared_bias() -> Optional[torch.Tensor]:
            if num_shared_experts == 0:
                return None

            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            if is_bf16xbf16:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.bf16_gemm_nt(x, shared_l1_weights, l1_out)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=128, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    round_scale=False, ue8m0_scale=False, output_bf16=True)[-1]
                deep_gemm.bf16_gemm_nt(l2_in, shared_l2_weights, y)
            else:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.fp8_gemm_nt(shared_x, shared_l1_weights, l1_out, recipe=(1, 1, 32), disable_ue8m0_cast=True)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=32, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    round_scale=True, ue8m0_scale=True, output_bf16=False)
                deep_gemm.fp8_gemm_nt(l2_in, shared_l2_weights, y, recipe=(1, 1, 32), disable_ue8m0_cast=True)
            return y

        def run_baseline(with_rms_norm: bool = True):
            cumulative_local_expert_recv_stats_baseline.copy_(initial_cumulative_local_expert_recv_stats_baseline)
            # Dispatch
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                num_experts=num_experts, expert_alignment=alignment,
                **dispatch_kwargs)
            num_recv_tokens = get_num_tokens(recv_x)

            # L1 GEMM
            l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
            gemm_fn(recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # SwiGLU
            swiglu_result = tilelang_ops.swiglu_apply_weight_to_fp8(
                x=l1_y, topk_weights=recv_topk_weights,
                avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
                num_per_channels=32, use_col_major_scales=True,
                clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                **swiglu_kwargs)
            l1_y = swiglu_result[-1] if is_bf16xbf16 else swiglu_result

            # L2 GEMM
            l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            gemm_fn(l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # Combine
            shared_bias = get_baseline_shared_bias()
            y = _fp32_combine_reference(l2_y, handle, shared_bias, group) if fp32_output else \
                ep_buffer.combine(l2_y, handle=handle, bias=shared_bias)[0]
            if args.rms_norm and with_rms_norm:
                y = torch.nn.functional.rms_norm(
                    y.float(), (hidden, ), weight=rms_weights.float(), eps=args.rms_eps).to(output_dtype)
            return y, cumulative_local_expert_recv_stats_baseline

    # Check correctness
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, fused_stats = run_fused()
            baseline_y, baseline_stats = run_baseline(with_rms_norm=False)
            assert torch.equal(fused_stats, baseline_stats)
            if args.rms_norm:
                baseline_y = _rms_norm_reference(baseline_y, rms_weights, args.rms_eps)
            if fp32_output and args.rms_norm:
                torch.testing.assert_close(fused_y, baseline_y, rtol=1e-5, atol=1e-6)
            else:
                assert torch.equal(fused_y, baseline_y), f'calc_diff={calc_diff(fused_y, baseline_y)}'
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    barrier_kwargs = {'wait_comm_stream': False} if hasattr(deep_ep, 'EPBuffer') else {'use_comm_stream': False}
    barrier_fn = lambda: ep_buffer.barrier(**barrier_kwargs) if ep_buffer else dist.all_reduce(torch.empty(1, device='cuda'))
    trace_path = None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_inference_rank{rank_idx}.json'
    t_fused = bench_kineto(run_fused, 'mega_moe_inference', barrier=barrier_fn, trace_path=trace_path)
    # The FP32 reference reverses routing with PyTorch collectives and CPU metadata
    # reads; it is for correctness, not a comparable CUDA-graph performance baseline.
    t_baseline = tilelang_bench(
        run_baseline, _n_warmup=5, _n_repeat=1,
        backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded and not fp32_output else 0

    # TFLOPS: routed + shared L1/L2, each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    num_routed_flops = 2 * num_recv_tokens * hidden * intermediate_hidden * 3
    num_shared_flops = 2 * num_tokens * hidden * shared_intermediate_hidden * 3
    num_total_flops = num_routed_flops + num_shared_flops

    # HBM bytes: weights + activations + output
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()
    act_elem_size = 2 if is_bf16xbf16 else 1
    routed_weight_elem_size = 2 if is_bf16xbf16 else (1 if args.mma_type == 'fp8xfp8' else 0.5)
    shared_weight_elem_size = 2 if is_bf16xbf16 else 1
    num_routed_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * routed_weight_elem_size      # L1 weights
        + num_touched_experts * hidden * intermediate_hidden * routed_weight_elem_size        # L2 weights
        + num_recv_tokens * hidden * act_elem_size                                            # L1 acts read
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L1 output write
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L2 acts read
        + num_recv_tokens * hidden * 2                                                        # L2 output write (always BF16)
    )
    num_shared_hbm_bytes = 0 if num_shared_experts == 0 else (
        shared_intermediate_hidden * 2 * hidden * shared_weight_elem_size      # Shared L1 weights
        + hidden * shared_intermediate_hidden * shared_weight_elem_size        # Shared L2 weights
        + num_tokens * hidden * act_elem_size                                  # Shared L1 acts read
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L1 output write
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L2 acts read
        + num_tokens * hidden * 2                                              # Shared L2 output write
    )
    num_hbm_bytes = num_routed_hbm_bytes + num_shared_hbm_bytes

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    def print_perf(elapsed: float, ref_time: float, ref_label: str):
        tflops = safe_div(num_total_flops / 1e12, elapsed)
        hbm_gbs = safe_div(num_hbm_bytes / 1e9, elapsed)
        nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, elapsed)
        approx_factor = safe_div(elapsed, elapsed - t_reduction)
        comparison = f'{safe_div(ref_time, elapsed):.2f}x {ref_label}' if ref_time else 'no comparable baseline'
        dist_print(f' > EP {rank_idx:2}/{num_ranks} | '
                   f'{tflops:4.0f} TFLOPS | '
                   f'overlap: '
                   f'{tflops * approx_factor:4.0f} TFLOPS, '
                   f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
                   f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
                   f'{elapsed * 1e6:4.0f} us, '
                   f'reduction: {t_reduction * 1e6:4.1f} us | '
                   f'{comparison}')

    dist_print(f'Performance (w/{"" if num_shared_experts else "o"} shared):', once_in_node=True)
    print_perf(t_fused, t_baseline, f'legacy{"+shared" if num_shared_experts else ""}')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    deep_gemm.destroy_localizer()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--num-shared-experts', type=int, default=1, help='Number of shared experts (use 0 to disable)')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--rms-norm', type=int, default=0, choices=(0, 1), help='Enable RMSNorm after combine (0 or 1, default: 0)')
    parser.add_argument('--rms-eps', type=float, default=None, help='RMSNorm epsilon (required when --rms-norm=1)')
    parser.add_argument('--output-dtype', choices=('bf16', 'fp32'), default='bf16', help='Output y (and RMSNorm weight) dtype')
    parser.add_argument('--mma-type', type=str, default='fp8xfp4', choices=('fp8xfp4', 'fp8xfp8', 'bf16xbf16'))
    parser.add_argument('--use-localization', action='store_true', help='Localize the weights into the locality domains (requires MLOPart)')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
