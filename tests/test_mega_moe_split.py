"""Split-kernel MegaMoE test: correctness (bitwise vs the fused megakernel) + performance.

The split pipeline runs three kernels wired into one CUDA graph via green contexts:
  * dispatch_l1_swiglu (K1): gather routed tokens + Linear1 + SwiGLU + FP8-quant to the pool.
  * l2_combine        (K2): Linear2 + NVLink combine-scatter; runs CONCURRENTLY with K1 on a
                            disjoint SM partition, consuming K1's pool blocks via an arrival mask.
  * combine_reduce    (K3): reduce the top-k combine partials into the final output.

It reproduces the fused `fp8_fp4_mega_moe` arithmetic exactly (same MMA order), so the output is
expected to be bitwise identical. Performance is usually slightly better than the fused kernel.
"""
import argparse
import random
import torch
import torch.distributed as dist
from typing import Callable, List, Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather


def _bench_replay(replay: Callable, reset: Callable, barrier: Callable,
                  num_warmups: int, num_tests: int, flush_mb: int) -> float:
    """Best-of-N wall-clock (seconds) of `replay`, resetting state and flushing L2 each iter."""
    flush = torch.empty(max(1, flush_mb * 1024 * 1024 // 4), dtype=torch.int, device='cuda')

    def prepare():
        reset()
        flush.zero_()
        torch.cuda.synchronize()
        barrier()

    for _ in range(num_warmups):
        prepare()
        replay()
        torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float('inf')
    for _ in range(num_tests):
        prepare()
        start.record()
        replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3)
    return best


def _capture(fn: Callable) -> torch.cuda.CUDAGraph:
    """Warm up `fn` on a side stream, then capture it into a replayable CUDA graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0

    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    kernel1_sms = args.kernel1_sms
    kernel2_sms = args.kernel2_sms
    reduce_sms = args.reduce_sms
    assert kernel1_sms + kernel2_sms <= prop.multi_processor_count

    # Symmetric buffers: fused (reference) and split (route-based dispatch layout).
    fused_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
    split_buffer = deep_gemm.get_symm_buffer_for_mega_moe_split(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)

    def cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    # Inputs (identical for fused and split)
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_weights = torch.randn((num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0
    x = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    l1_weights = cast_weights_to_fp4(l1_weights)
    l2_weights = cast_weights_to_fp4(l2_weights)
    l1w, l2w = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Routing stats for this rank's local experts (identical for fused and split: same topk_idx).
    # Mirrors test_mega_moe.py so TFLOPS/HBM/NVLink are computed the same way.
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) |
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()

    def fill(buffer):
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

    fused_stats = torch.randint(0, 100, (num_experts_per_rank,), dtype=torch.int, device='cuda')
    split_stats = fused_stats.clone()
    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    y_split = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    state = torch.zeros((16,), dtype=torch.int, device='cuda')

    def reset_fused():
        fused_buffer.buffer.zero_()
        fused_stats.zero_()
        fill(fused_buffer)

    def reset_split():
        split_buffer.buffer.zero_()
        split_stats.zero_()
        state.zero_()
        fill(split_buffer)

    def run_fused():
        deep_gemm.fp8_fp4_mega_moe(
            y_fused, l1w, l2w, fused_buffer,
            cumulative_local_expert_recv_stats=fused_stats,
            activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))

    split_graph = deep_gemm.SM100FP8FP4MegaMoESplitGraph(
        [state], [y_split], [split_buffer], [l1w], [l2w], [split_stats],
        num_tokens, args.activation_clamp, bool(args.fast_math),
        kernel1_sms, kernel2_sms, reduce_sms, 0, 0)

    dist_print('Config:', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}, Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}, GPU SMs: {prop.multi_processor_count}', once_in_node=True)
    dist_print(f' > Split SMs: K1(dispatch_l1_swiglu)={kernel1_sms}, K2(l2_combine)={kernel2_sms}, '
               f'K3(combine_reduce)={reduce_sms}', once_in_node=True)
    dist_print(f' > Split green context ids: {split_graph.get_green_context_ids()}', once_in_node=True)
    dist_print(once_in_node=True)

    # Correctness: split output must be bitwise identical to the fused megakernel.
    reset_fused()
    run_fused()
    torch.cuda.synchronize()
    dist.barrier()
    reset_split()
    torch.cuda.synchronize()
    dist.barrier()  # all ranks must finish zeroing before any K1 NVLink dispatch reads a peer buffer
    split_graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    diff = (y_split.float() - y_fused.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / y_fused.float().abs().clamp_min(1e-6)).max().item()
    is_bitwise = torch.equal(y_split, y_fused)
    dist_print(f'Correctness: bitwise={is_bitwise} (max_abs={max_abs:.6g}, max_rel={max_rel:.6g})',
               once_in_node=True)
    assert is_bitwise, f'split output is NOT bitwise identical to fused (max_abs={max_abs:.6g})'

    # Performance: wall-clock best-of-N for the fused megakernel vs the split graph.
    fused_graph = _capture(run_fused)
    t_fused = _bench_replay(fused_graph.replay, reset_fused, lambda: dist.barrier(),
                            args.num_warmups, args.num_tests, args.flush_l2_mb)
    t_split = _bench_replay(split_graph.replay, reset_split, lambda: dist.barrier(),
                            args.num_warmups, args.num_tests, args.flush_l2_mb)
    safe_div = lambda a, b: float('nan') if b == 0 else a / b

    def perf_metrics(t):
        # 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
        tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t)
        # HBM bytes: weights (FP4 = 0.5 B) + activations + output
        num_hbm_bytes = (
            num_touched_experts * intermediate_hidden * 2 * hidden * 0.5    # L1 weights
            + num_touched_experts * hidden * intermediate_hidden * 0.5      # L2 weights
            + num_recv_tokens * hidden                                      # L1 acts read
            + num_recv_tokens * intermediate_hidden                         # L1 output write
            + num_recv_tokens * intermediate_hidden                         # L2 acts read
            + num_recv_tokens * hidden * 2                                  # L2 output write (BF16)
        )
        hbm_gbs = safe_div(num_hbm_bytes / 1e9, t)
        # NVLink bytes: dispatch pull + combine write-back
        nvlink_gbs = safe_div(num_recv_tokens * hidden * 3 / 1e9, t)
        return tflops, hbm_gbs, nvlink_gbs

    tf_f, hbm_f, nvl_f = perf_metrics(t_fused)
    tf_s, hbm_s, nvl_s = perf_metrics(t_split)
    dist_print(f'Routing: recv_tokens={num_recv_tokens}, touched_experts={num_touched_experts} (rank 0)',
               once_in_node=True)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > Fused megakernel : {t_fused * 1e6:7.1f} us | {tf_f:5.0f} TFLOPS | '
               f'HBM {hbm_f:5.0f} GB/s | NVL {nvl_f:4.0f} GB/s', once_in_node=True)
    dist_print(f' > Split pipeline   : {t_split * 1e6:7.1f} us | {tf_s:5.0f} TFLOPS | '
               f'HBM {hbm_s:5.0f} GB/s | NVL {nvl_s:4.0f} GB/s', once_in_node=True)
    dist_print(f' > Split / Fused    : {safe_div(t_split, t_fused):.3f}x '
               f'({"faster" if t_split < t_fused else "slower"})', once_in_node=True)
    # Machine-parseable summary line for sweep collection.
    dist_print(f'SWEEP_RESULT tokens={num_tokens} hidden={hidden} inter={intermediate_hidden} '
               f'procs={num_ranks} recv={num_recv_tokens} experts={num_touched_experts} '
               f'fused_us={t_fused * 1e6:.1f} split_us={t_split * 1e6:.1f} '
               f'ratio={safe_div(t_split, t_fused):.4f} '
               f'fused_tflops={tf_f:.1f} split_tflops={tf_s:.1f} '
               f'fused_hbm={hbm_f:.1f} split_hbm={hbm_s:.1f} '
               f'fused_nvl={nvl_f:.1f} split_nvl={nvl_s:.1f}', once_in_node=True)

    dist.barrier()
    fused_buffer.destroy()
    split_buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split-kernel MegaMoE: bitwise vs fused + perf')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--num-tokens', type=int, default=8192)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=3072)
    parser.add_argument('--activation-clamp', type=float, default=10)
    parser.add_argument('--num-experts', type=int, default=384)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--kernel1-sms', type=int, default=96)
    parser.add_argument('--kernel2-sms', type=int, default=52)
    parser.add_argument('--reduce-sms', type=int, default=148)
    parser.add_argument('--num-warmups', type=int, default=3)
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--flush-l2-mb', type=int, default=2048)
    parser.add_argument('--local-rank-idx', type=int, default=None)
    args = parser.parse_args()

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
