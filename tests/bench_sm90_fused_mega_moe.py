"""Interleaved-comparison building block for SM90 fused MegaMoE.

CUDA-event measurements use a 256 MiB L2 flush, NCCL start alignment, and
a device pad before each replay. median_us is the median of per-iteration
minima over ranks; max_rank_us includes rank start skew. Neither is host wall time.
MXFP4 checkpoint fixtures use a fixed round-to-nearest-even quantizer here,
so changing DG_ROOT does not change weights through historical helper versions.
"""
import argparse, os, sys, statistics, time, torch
import torch.distributed as dist
from pathlib import Path


def quantize_mxfp4_fixture(w):
    """Canonical E2M1/K32 fixture, independent of the selected DeepGEMM tree."""
    e, n, k = w.shape
    assert k % 32 == 0
    groups = w.view(e, n, k // 32, 32)
    sf = groups.abs().float().amax(dim=-1).clamp_min(1e-4) / 6.0
    bits = sf.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    sf = (exponent.clamp(1, 254) << 23).view(torch.float32)
    scaled = groups * (1.0 / sf.unsqueeze(-1))
    ax = scaled.abs()
    code = torch.zeros_like(scaled, dtype=torch.uint8)
    for boundary, round_up in zip((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
                                  (False, True, False, True, False, True, False)):
        code += (ax >= boundary if round_up else ax > boundary).to(torch.uint8)
    code |= (((scaled < 0) & (code != 0)).to(torch.uint8) << 3)
    pairs = code.view(e, n, k // 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).view(torch.int8)
    return packed.contiguous(), sf


def worker(local_rank, num_local_ranks, a):
    sys.path.insert(0, os.environ.get('DG_ROOT', str(Path(__file__).resolve().parents[1])))
    import deep_gemm
    deep_gemm.set_pdl(bool(a.pdl))
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist
    rank, nr, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank)
    H, I, E, K = a.hidden, a.ih, a.epr, a.topk
    l1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device='cuda') * 0.05
    l2 = torch.randn(E, H, I, dtype=torch.bfloat16, device='cuda') * 0.05
    fp8 = a.impl in ('pr440_fp8', 'pr383_fp8')
    if fp8:
        sys.path.insert(0, os.path.join(os.environ.get('DG_ROOT', str(Path(__file__).resolve().parents[1])), 'tests'))
        import test_mega_moe_sm90 as T
        w1, w2 = deep_gemm.transform_weights_for_mega_moe_sm90(T._quantize_grouped_fp8_block_128_128(l1), T._quantize_grouped_fp8_block_128_128(l2))
    else:
        w1, w2 = deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(quantize_mxfp4_fixture(l1), quantize_mxfp4_fixture(l2), **({} if a.impl == 'pr411_mxfp4' else dict(layout=a.layout)))
    del l1, l2
    torch.cuda.empty_cache()
    flush = torch.empty(256 << 20, dtype=torch.int8, device='cuda')
    tok = torch.zeros(1, dtype=torch.int, device='cuda')
    fused = a.impl in ('pr440_fp8', 'pr440_mxfp4')
    for mi, mg in enumerate(a.m):
        m = -(-mg // nr) if a.global_m else mg
        cap = a.cap if a.cap else max(128, (m + 127) // 128 * 128)
        if a.buf_num_sms:   # SM count the buffer (ring capacity) is sized with; --num-sms then applies to the launches only
            deep_gemm.set_num_sms(a.buf_num_sms)
        buf = deep_gemm.get_symm_buffer_for_sm90_mega_moe(group, E * nr, cap, K, H, I, **(dict(fused=True) if fused else {}))
        if a.num_sms:
            deep_gemm.set_num_sms(a.num_sms)
        if a.global_m:   # one global batch (seed 1000003 * B on every rank, as bench_humming.py), rank r owns rows r*m .. r*m+m-1
            torch.manual_seed(1000003 * mg + a.seed * 1009)
            xg = torch.randn(mg, H, dtype=torch.bfloat16, device='cuda')
            twg, tig = torch.topk(torch.randn(mg, E * nr, device='cuda'), K, dim=-1, sorted=False)
            lo, hi = min(rank * m, mg), min(rank * m + m, mg)
            xb = torch.zeros(m, H, dtype=torch.bfloat16, device='cuda'); xb[:hi - lo] = xg[lo:hi]
            ti = torch.full((m, K), -1, dtype=tig.dtype, device='cuda'); ti[:hi - lo] = tig[lo:hi]   # rows past B: SP padding
            tw = torch.zeros(m, K, device='cuda'); tw[:hi - lo] = twg[lo:hi]
            x, xs = per_token_cast_to_fp8(xb, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
        else:
            torch.manual_seed(1000003 * m + rank + a.seed * 1009)
            x, xs = per_token_cast_to_fp8(torch.randn(m, H, dtype=torch.bfloat16, device='cuda'), use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
            tw, ti = torch.topk(torch.randn(m, E * nr, device='cuda'), K, dim=-1, sorted=False)
            xb = torch.randn(m, H, dtype=torch.bfloat16, device='cuda')
        if a.mask_all:   # every row masked on every rank (topk -1, weight 0): the kernel skeleton without any tile (single-token decode)
            ti = torch.full_like(ti, -1); tw = torch.zeros_like(tw)
        buf.x[:m].copy_(x); buf.x_sf[:m].copy_(xs); buf.topk_idx[:m].copy_(ti); buf.topk_weights[:m].copy_(tw)
        y = torch.empty(m, H, dtype=torch.bfloat16, device='cuda')
        kw = dict(num_tokens_bound=m) if (a.bound and fused) else {}
        if a.split_k and a.impl == 'pr440_mxfp4' and mg == 1:   # split-K hint: one global token (vLLM decode b1; single-token decode)
            kw['split_k'] = True
        k = (lambda: deep_gemm.fp8_mega_moe(y, w1, w2, buf, activation_clamp=10.0, **kw)) if fp8 else \
            (lambda: deep_gemm.fp8_mxfp4_mega_moe(y, w1, w2, buf, activation_clamp=10.0, **kw))
        f = k
        for _ in range(3):
            f()
        torch.cuda.synchronize(); dist.barrier(group=group)
        graph = None
        if a.graph:   # capture the operator only; checkpoint preprocessing and input staging are outside the timed region
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                f()
            torch.cuda.synchronize(); dist.barrier(group=group)
            f = graph.replay
            for _ in range(3):
                f()
            torch.cuda.synchronize(); dist.barrier(group=group)
        for rep in range(a.reps):   # --reps windows of --iters launches each on the same buffer
          ts, hs, ls = [], [], []
          for _ in range(a.iters):
              flush.zero_()
              dist.all_reduce(tok, group=group)
              # r -> s: the pad when the host issued `s` in time, longer when it was late
              r = torch.cuda.Event(enable_timing=True); r.record()
              # keep the stream busy while the host issues the events and the MegaMoE launch, so `s` fires right before the kernel
              # (without it `s` fires when the all-reduce completes and the interval holds the host launch latency)
              torch.cuda._sleep(a.pad_cycles)
              s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
              h0 = time.perf_counter()
              s.record(); f(); e.record()
              hs.append((time.perf_counter() - h0) * 1e6)   # host time of the timed launch (wrapper + launch)
              torch.cuda.synchronize()
              ts.append(s.elapsed_time(e) * 1000); ls.append(r.elapsed_time(s) * 1000)
          t = torch.tensor([ts, hs, ls], device='cuda', dtype=torch.float64)
          g = [torch.zeros_like(t) for _ in range(nr)]
          dist.all_gather(g, t, group=group)
          G, HT, LT = torch.stack(g).cpu().unbind(1)   # [rank, iter] each: kernel, host launch, pad
          mn, mx = sorted(G.min(0).values.tolist()), sorted(G.max(0).values.tolist())
          cpu = int(open('/proc/self/stat').read().rsplit(')', 1)[1].split()[36])   # the core this rank's main thread last ran on
          cpus = [0] * nr
          dist.all_gather_object(cpus, cpu, group=group)
          if rank == 0:
              hmax = HT.max(0).values
              print(f'RESULT impl={a.impl} np={nr} mi={mi} m={mg} rows={m} rep={rep} buf=0x{buf.buffer.data_ptr():x} median_us={statistics.median(mn):.1f} max_rank_us={statistics.median(mx):.1f} '
                    f'rank0_us={statistics.median(G[0].tolist()):.1f} p10={mn[len(mn) // 10]:.1f} p90={mn[len(mn) * 9 // 10]:.1f} n={len(mn)} '
                    f'host_med_us={hmax.median().item():.1f} host_max_us={hmax.max().item():.1f} '
                    f'pad_us={LT.median().item():.1f} cpus={",".join(map(str, cpus))}', flush=True)
              if a.dump:
                  torch.save(dict(G=G, H=HT, L=LT, cpus=cpus), f'{a.dump}.i{mi}.m{m}.r{rep}.pt')
        del graph, f
        buf.destroy(); dist.barrier(group=group)
    dist.barrier(group=group); dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--impl', choices=['pr440_fp8', 'pr440_mxfp4', 'pr411_mxfp4', 'pr383_fp8'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--np', type=int, default=8)
    p.add_argument('--m', type=int, nargs='+', default=[8, 64, 128, 256, 1024])
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--hidden', type=int, default=4096); p.add_argument('--ih', type=int, default=2048)
    p.add_argument('--epr', type=int, default=32); p.add_argument('--topk', type=int, default=6)
    p.add_argument('--cap', type=int, default=0)   # symm buffer capacity (tokens per rank); 0 = max(128, align(m, 128))
    p.add_argument('--layout', default='row', choices=['row', 'tiled'])
    p.add_argument('--pad-cycles', type=int, default=2000000)   # torch.cuda._sleep before the timed launch (~1 ms); 0 = off (ep8b used 400000)
    p.add_argument('--reps', type=int, default=1)   # timing windows per buffer (each M entry allocates a new buffer)
    p.add_argument('--global-m', action='store_true')   # --m are global token counts; cdiv(B, np) rows per rank, the tail masked
    p.add_argument('--bound', action='store_true')   # num_tokens_bound = rows per rank (fused buffers only)
    p.add_argument('--dump', default='')   # per-(rank, iter) kernel / host / pad times -> <dump>.i<M index>.m<M>.r<rep>.pt
    p.add_argument('--num-sms', type=int, default=0)   # deep_gemm.set_num_sms for the MegaMoE launches (0 = all SMs; buffer sizing)
    p.add_argument('--buf-num-sms', type=int, default=0)   # SM count at buffer creation (0 = same as the launches)
    p.add_argument('--split-k', action='store_true')   # pass split_k=True to fp8_mxfp4_mega_moe when B (global) == 1 (needs --global-m)
    p.add_argument('--pdl', type=int, choices=[0, 1], default=1)   # runtime PDL enable flag
    p.add_argument('--mask-all', action='store_true')   # all rows masked on all ranks: no tile runs (kernel skeleton)
    p.add_argument('--graph', action='store_true')   # time CUDA-graph replays of the region (vLLM graph decode; CUDA Graph replay)
    a = p.parse_args()
    torch.multiprocessing.spawn(worker, args=(a.np, a), nprocs=a.np)
