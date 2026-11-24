"""Repeated timing against the original SM120 GEMM (no alternate GEMM baseline).

Example: python tests/bench_sm120_bf16_swiglu.py --output results.json
Use --baseline-only with an independently installed unmodified upstream build.
Weights are synthetic; this measures the projection+activation scope, not a model.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
import deep_gemm


def activation(p):
    g, u = p.chunk(2, -1)
    return (F.silu(g.float()) * u.float()).to(torch.bfloat16)


def activation_valid(p, slots, out):
    out.flatten(0, 1).index_copy_(0, slots, activation(p.flatten(0, 1).index_select(0, slots)))
    return out


def capture(fn, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            fn()
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('qwen3-30b', 'qwen3-235b', 'mixtral', 'wide'), default='qwen3-30b')
    parser.add_argument('--tokens', type=int, default=1)
    parser.add_argument('--distribution', choices=('uniform', 'skew75'), default='uniform')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=60)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    assert args.rounds >= 5 and args.repeats > 0 and args.tokens > 0
    d, i, e, topk = {'qwen3-30b': (2048, 768, 128, 8), 'qwen3-235b': (4096, 1536, 128, 8),
                     'mixtral': (4096, 14336, 8, 2), 'wide': (2048, 1408, 64, 4)}[args.model]
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    scores = torch.rand(args.tokens, e, generator=gen)
    if args.distribution == 'skew75':
        scores[:math.ceil(args.tokens * .75), max(topk, math.ceil(e / 4)):] = -1
    routes = scores.topk(topk, dim=1).indices
    counts_cpu = torch.bincount(routes.flatten(), minlength=e)
    m = math.ceil(int(counts_cpu.max()) / 128) * 128
    expected = max(1, math.ceil(args.tokens * topk / e))
    counts = counts_cpu.cuda().int()
    slots = torch.cat([torch.arange(int(n)) + expert * m for expert, n in enumerate(counts_cpu) if n]).cuda()
    a = torch.randn(e, m, d, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(e, 2 * i, d, device='cuda', dtype=torch.bfloat16) / math.sqrt(d)
    p = torch.zeros(e, m, 2 * i, device='cuda', dtype=torch.bfloat16)
    h = torch.zeros(e, m, i, device='cuda', dtype=torch.bfloat16)
    act = torch.compile(activation, fullgraph=True)
    act_valid = torch.compile(activation_valid, fullgraph=True)
    def original_l1():
        deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, p, counts, expected)
    def split():
        original_l1()
        return act(p)
    def split_valid():
        original_l1()
        return act_valid(p, slots, h)
    original_l1()
    ref = act(p).clone()
    functions = {'original_l1': original_l1, 'activation_padded': lambda: act(p),
                 'activation_valid': lambda: act_valid(p, slots, h),
                 'split_padded': split, 'split_valid': split_valid}
    errors = None
    if not args.baseline_only:
        b8, _ = deep_gemm.transform_weights_for_mega_moe(b, h)
        def fused():
            deep_gemm.m_grouped_bf16_gemm_swiglu_nt_masked(a, b8, h, counts, expected)
        functions['fused'] = fused
        h.fill_(19)
        fused()
        valid = torch.arange(m, device='cuda')[None, :] < counts[:, None]
        torch.testing.assert_close(h[valid], ref[valid], rtol=0.02, atol=0.003)
        delta = h[valid].float() - ref[valid].float()
        errors = {'max_abs': float(delta.abs().max()), 'unequal': int((h[valid] != ref[valid]).sum()),
                  'relative_l2': float(delta.norm() / ref[valid].float().norm().clamp_min(1e-12))}
        assert errors['relative_l2'] < 0.005
        assert bool((h[~valid] == 19).all())
    graphs = {name: capture(fn, args.repeats) for name, fn in functions.items()}
    samples = {name: [] for name in graphs}
    for round_idx in range(args.rounds):
        names = list(graphs)
        if round_idx % 2:
            names.reverse()
        for name in names:
            graphs[name].replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); graphs[name].replay(); end.record(); end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.repeats)
    records = {name: {'mean_us': statistics.mean(values), 'std_us': statistics.stdev(values),
                      'median_us': statistics.median(values), 'samples_us': values}
               for name, values in samples.items()}
    result = {'model': args.model, 'tokens': args.tokens, 'distribution': args.distribution,
              'seed': args.seed, 'capacity': m, 'expected_m': expected, 'rounds': args.rounds,
              'repeats': args.repeats, 'torch': torch.__version__, 'module': deep_gemm.__file__,
              'counts': counts_cpu.tolist(), 'validation': errors, 'timings': records}
    if not args.baseline_only:
        result['speedup'] = min(records[n]['mean_us'] for n in ('split_padded', 'split_valid')) / records['fused']['mean_us']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
