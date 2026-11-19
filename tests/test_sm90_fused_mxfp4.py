"""PR440 fused MegaMoE + opt-in MXFP4 routed weights: correctness against an independent PyTorch reference.

Weights follow PR #411's contract: per_token_cast_to_fp4(gran_k=32, ue8m0) -> transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90.
The reference dequantizes the *transformed* triple back to logical [gate | up] rows (sign restore, deinterleave, scale layout
restore, secondary * 2^offset), so it checks the kernel's TMA / decode / scale chain without reusing kernel code.
"""
import argparse, math, os, sys, torch
import torch.distributed as dist
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
from deep_gemm.utils.dist import init_dist, dist_print, uneven_all_gather
from deep_gemm.testing import calc_diff
from deep_gemm.mega.mxfp4 import _restore_mxfp4_sign_bits_from_sm90

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def quantize_mxfp4(w):
    e, n, k = w.shape
    packed = torch.empty((e, n, k // 2), dtype=torch.int8, device=w.device)
    sf = torch.empty((e, n, k // 32), dtype=torch.float32, device=w.device)
    for i in range(e):
        packed[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
    return packed, sf

def dequant_mxfp4(packed, sf):
    *pre, n, pk = packed.shape
    codes = torch.empty((*pre, n, pk * 2), dtype=torch.int8, device=packed.device)
    codes[..., 0::2] = packed & 0x0f
    codes[..., 1::2] = (packed >> 4) & 0x0f
    mag = torch.tensor(E2M1, device=packed.device)[(codes & 7).long()]
    val = torch.where((codes & 8) != 0, -mag, mag)
    return (val.view(*pre, n, pk * 2 // 32, 32) * sf.unsqueeze(-1)).view(*pre, n, pk * 2)

def deinterleave_l1(t, g=8):
    e, n, *tail = t.shape
    h = n // 2
    v = t.view(e, h // g, 2, g, *tail)
    return torch.cat([v[:, :, 0].reshape(e, h, *tail), v[:, :, 1].reshape(e, h, *tail)], dim=1)

def restore_scale_layout(t):
    e, n, k32 = t.shape
    return t.view(e, k32 // 4, n, 4).permute(0, 2, 1, 3).contiguous().view(t.shape)

def logical_weights(tr, is_l1):
    w, sfb, sec = tr
    w = _restore_mxfp4_sign_bits_from_sm90(deinterleave_l1(w) if is_l1 else w)
    off = restore_scale_layout(sfb)
    off = deinterleave_l1(off) if is_l1 else off
    return w, sec[:, None, None] * torch.exp2(off.float())

def swiglu(x, clamp):
    h = x.size(-1) // 2
    g, u = x[..., :h], x[..., h:]
    if math.isfinite(clamp):
        g = g.clamp(max=clamp); u = u.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(g) * u

def reference(x_fp8, x_sf, topk_idx, topk_w, l1, l2, rank, nr, group, num_experts, hidden, ih, clamp, gran):
    sizes = [x_fp8.size(0)] if nr == 1 else None
    if nr == 1:
        X, XS, TI, TW = x_fp8, x_sf, topk_idx, topk_w
    else:
        X, XS = uneven_all_gather(x_fp8, group=group), uneven_all_gather(x_sf, group=group)
        TI, TW = uneven_all_gather(topk_idx, group=group), uneven_all_gather(topk_w, group=group)
        st = torch.tensor([x_fp8.size(0)], device='cuda'); al = torch.empty(nr, dtype=torch.long, device='cuda')
        dist.all_gather_into_tensor(al, st, group=group); sizes = al.tolist()
    mg = X.size(0)
    x = (X.float().view(mg, hidden // 128, 128) * XS.unsqueeze(-1)).view(mg, hidden)
    combine = torch.zeros(mg, TI.size(1), hidden, dtype=torch.float32, device='cuda')
    epr = num_experts // nr
    for e in torch.unique(TI[(TI >= rank * epr) & (TI < (rank + 1) * epr)]).tolist():
        r = (TI == e).nonzero(as_tuple=False); sel, slot = r[:, 0], r[:, 1]
        le = e - rank * epr
        w1 = dequant_mxfp4(l1[0][le], l1[1][le])
        y1 = swiglu(x[sel] @ w1.t(), clamp) * TW[sel, slot].unsqueeze(-1)
        v = y1.view(-1, ih // gran, gran)
        s = v.abs().amax(-1).clamp(1e-4) / 448.0
        y1 = ((v / s.unsqueeze(-1)).to(torch.float8_e4m3fn).float() * s.unsqueeze(-1)).view(-1, ih)
        w2 = dequant_mxfp4(l2[0][le], l2[1][le])
        combine[sel, slot] = (y1 @ w2.t()).to(torch.bfloat16).float()
    if nr > 1:
        dist.all_reduce(combine, group=group)
    out = combine.to(torch.bfloat16).sum(1).to(torch.bfloat16)
    st = sum(sizes[:rank])
    return out[st:st + sizes[rank]].contiguous()

def run_case(name, m, fm, args, rank, nr, group, seed):
    hidden, ih, topk = args.hidden, args.ih, args.topk
    num_experts = args.experts_per_rank * nr
    epr = args.experts_per_rank
    torch.manual_seed(seed + rank * 1009)
    x_fp8, x_sf = per_token_cast_to_fp8(torch.randn(m, hidden, dtype=torch.bfloat16, device='cuda') * args.x_scale,
                                        use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    l1q = quantize_mxfp4(torch.randn(epr, 2 * ih, hidden, dtype=torch.bfloat16, device='cuda') * 0.05)
    l2q = quantize_mxfp4(torch.randn(epr, hidden, ih, dtype=torch.bfloat16, device='cuda') * 0.05)
    scores = torch.randn(m, num_experts, device='cuda')
    if args.route == 'skew':   # concentrate traffic: half the tokens go to experts 0..topk-1 -> odd/long tile streams
        scores[: m // 2, :topk] += 100.0
    topk_w, topk_idx = torch.topk(scores, topk, dim=-1, sorted=False)
    if args.masked > 0:
        msk = torch.rand_like(topk_w) < args.masked
        topk_idx.masked_fill_(msk, -1); topk_w.masked_fill_(msk, 0)
    s2 = None
    if args.scale2:   # exercise the per-expert secondary path with a checkpoint weight_scale_2
        s2 = (torch.rand(epr, device='cuda') * 3 + 0.25, torch.rand(epr, device='cuda') * 3 + 0.25)
    if args.wide_scale2:
        exponents = torch.linspace(-24, 24, epr, device='cuda')
        s2 = (torch.exp2(exponents), torch.exp2(-exponents))
    # Reuse exact quantized inputs across Python/JIT implementations; an RNG
    # seed alone does not guarantee identical quantization in two environments.
    if args.input_root:
        from pathlib import Path
        fixture = Path(args.input_root) / f'{name}.rank{rank}.pt'
        if fixture.exists():
            saved = torch.load(fixture, map_location='cuda', weights_only=True)
            x_fp8, x_sf = saved['x'], saved['xs']
            l1q, l2q = tuple(saved['l1q']), tuple(saved['l2q'])
            topk_idx, topk_w, s2 = saved['idx'], saved['weights'], saved['scale2']
        else:
            fixture.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(x=x_fp8.cpu(), xs=x_sf.cpu(),
                            l1q=[v.cpu() for v in l1q], l2q=[v.cpu() for v in l2q],
                            idx=topk_idx.cpu(), weights=topk_w.cpu(),
                            scale2=None if s2 is None else [v.cpu() for v in s2]), fixture)
    tl1, tl2 = deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
        l1q, l2q, *(s2 if s2 is not None else (None, None)), layout=args.layout)
    if args.layout == 'tiled':   # the reference reads the row layout back from the tiled bytes (independent inverse)
        from deep_gemm.mega.mxfp4 import _untile_mxfp4_weights_from_sm90 as untile
        row1, row2 = (untile(tl1[0]),) + tl1[1:], (untile(tl2[0]),) + tl2[1:]
        tr1, tr2 = deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(l1q, l2q, *(s2 if s2 is not None else (None, None)))
        assert torch.equal(row1[0], tr1[0]) and torch.equal(row2[0], tr2[0]), 'tiled layout does not round-trip'
    else:
        row1, row2 = tl1, tl2
    ref_l1, ref_l2 = logical_weights(row1, True), logical_weights(row2, False)

    buf_kwargs = dict(fused=True)
    if args.gran:
        buf_kwargs['l2_act_sf_gran_k'] = args.gran
    buf = deep_gemm.get_symm_buffer_for_sm90_mega_moe(group, num_experts, max(m, args.capacity), topk, hidden, ih, **buf_kwargs)
    gran = buf.l2_act_sf_gran_k
    buf.x[:m].copy_(x_fp8); buf.x_sf[:m].copy_(x_sf)
    buf.topk_idx[:m].copy_(topk_idx); buf.topk_weights[:m].copy_(topk_w)
    stats = torch.zeros(epr, dtype=torch.int, device='cuda')
    outs = []
    if args.graph:
        y = torch.empty(m, hidden, dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_mxfp4_mega_moe(y, tl1, tl2, buf, cumulative_local_expert_recv_stats=None,
                                     activation_clamp=args.clamp, fast_math=bool(fm),
                                     **(dict(num_tokens_bound=m) if args.bound else {}),
                                     **(dict(split_k=True) if args.split_k else {}))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            deep_gemm.fp8_mxfp4_mega_moe(y, tl1, tl2, buf, cumulative_local_expert_recv_stats=stats,
                                         activation_clamp=args.clamp, fast_math=bool(fm),
                                         **(dict(num_tokens_bound=m) if args.bound else {}),
                                         **(dict(split_k=True) if args.split_k else {}))
        for _ in range(args.repeat):
            graph.replay()
            outs.append(y.clone())
        torch.cuda.synchronize()
    else:
        for _ in range(args.repeat):
            y = torch.empty(m, hidden, dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_mxfp4_mega_moe(y, tl1, tl2, buf, cumulative_local_expert_recv_stats=stats,
                                         activation_clamp=args.clamp, fast_math=bool(fm),
                                         **(dict(num_tokens_bound=m) if args.bound else {}),
                                         **(dict(split_k=True) if args.split_k else {}))
            torch.cuda.synchronize()
            outs.append(y)

    y_ref = reference(x_fp8, x_sf, topk_idx, topk_w, ref_l1, ref_l2, rank, nr, group, num_experts, hidden, ih, args.clamp, gran)
    diff = calc_diff(outs[0], y_ref) if m > 0 and y_ref.abs().sum() > 0 else float((outs[0].float() - y_ref.float()).abs().max()) if m > 0 else 0.0
    bitwise = all(torch.equal(outs[0], o) for o in outs[1:])
    nan = any(bool(torch.isnan(o).any()) for o in outs)
    finite = all(bool(torch.isfinite(o).all()) for o in outs)
    # recv stats: every launch counts this rank's received routes
    gi = topk_idx if nr == 1 else uneven_all_gather(topk_idx, group=group)
    exp_stats = torch.bincount(gi[gi >= 0].long(), minlength=num_experts)[rank * epr:(rank + 1) * epr].int() * args.repeat
    stats_ok = torch.equal(stats, exp_stats)
    ok = diff < args.tol and bitwise and finite and stats_ok
    f = torch.tensor([0 if ok else 1], device='cuda'); dist.all_reduce(f, op=dist.ReduceOp.MAX, group=group)
    dist_print(f'  [{name:<24}] gran={gran} diff={diff:.6f} bitwise_repeat={bitwise} nan={nan} finite={finite} stats={stats_ok} '
               f'{"OK" if f.item() == 0 else "FAIL"}', once_in_node=True)
    if args.dump_y:   # output only (small): bitwise comparisons between kernel variants
        torch.save(outs[0].cpu(), os.path.join(args.dump_y, f'{name}.rank{rank}.pt'))
    buf.destroy(); dist.barrier()
    return f.item() == 0

def worker(local_rank, nlr, args):
    rank, nr, group = init_dist(local_rank, nlr)
    if args.rank_decode_modes:   # per-rank DG_SM90_MXFP4_DECODE_MODE (cross-rank mixed kernels; read by the host at every call)
        os.environ['DG_SM90_MXFP4_DECODE_MODE'] = str(args.rank_decode_modes[rank % len(args.rank_decode_modes)])
    if args.sms:
        deep_gemm.set_num_sms(args.sms)
    fails = []
    for m in args.m:
        for fm in args.fm:
            name = f'mxfp4.{args.route}.m{m}.fm{fm}'
            if not run_case(name, m, fm, args, rank, nr, group, seed=m * 7 + fm):
                fails.append(name)
    dist_print(f'SUMMARY total={len(args.m) * len(args.fm)} fails={fails}', once_in_node=True)
    dist.barrier(); dist.destroy_process_group()
    assert not fails, f'Correctness failures: {fails}'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sms', type=int, default=0)
    p.add_argument('--graph', action='store_true')
    p.add_argument('--np', type=int, default=1)
    p.add_argument('--m', type=int, nargs='+', default=[8, 64, 128, 256, 1024])
    p.add_argument('--fm', type=int, nargs='+', default=[1, 0])
    p.add_argument('--hidden', type=int, default=4096)
    p.add_argument('--ih', type=int, default=2048)
    p.add_argument('--experts-per-rank', type=int, default=32)
    p.add_argument('--topk', type=int, default=6)
    p.add_argument('--gran', type=int, default=0)
    p.add_argument('--route', default='rand', choices=['rand', 'skew'])
    p.add_argument('--layout', default='tiled', choices=['row', 'tiled'])
    p.add_argument('--masked', type=float, default=0.0)
    p.add_argument('--clamp', type=float, default=10.0)
    p.add_argument('--scale2', action='store_true')
    p.add_argument('--wide-scale2', action='store_true')
    p.add_argument('--x-scale', type=float, default=1.0)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--tol', type=float, default=0.002)
    p.add_argument('--dump', default='')
    p.add_argument('--dump-y', default='')
    p.add_argument('--input-root', default='')
    p.add_argument("--capacity", type=int, default=1)
    p.add_argument('--rank-decode-modes', type=int, nargs='*', default=[])
    p.add_argument('--split-k', action='store_true')   # Optional performance hint for single-token decode.
    p.add_argument('--bound', action='store_true')   # All ranks in this test use the same local M.
    a = p.parse_args()
    torch.multiprocessing.spawn(worker, args=(a.np, a), nprocs=a.np)
