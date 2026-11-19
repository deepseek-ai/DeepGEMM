"""Revalidate offline T1 candidates on the PR's repaired public implementation.

Calibration and holdout are separate invocations. No historical-driver kernel
is loaded. Existing public SwiGLU plus an ordinary tail is also a control.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
import triton

from deep_gemm import _C
from deep_gemm.small_moe import SmallMoE, _dispatch_slots, _swiglu, _combine
from deep_gemm.small_moe_policy import SINGLE_TOKEN_CANDIDATES, select_config
from tests.bench_sm120_small_moe import compare, independent_oracle, telemetry
from tests.test_sm120_small_moe_gpu import make_routes

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'qwen30': (2048, 768, 128, 8),
    'qwen235': (4096, 1536, 128, 8),
    'qwen15': (2048, 1408, 60, 4),
    'qwen2': (3584, 2560, 64, 8),
    'deepseek': (2048, 1408, 64, 6),
}


def source_hashes():
    names = ('deep_gemm/small_moe.py', 'deep_gemm/small_moe_policy.py',
             'csrc/apis/sm120_moe_small.hpp',
             'deep_gemm/include/deep_gemm/mma/sm120.cuh',
             'deep_gemm/include/deep_gemm/impls/sm120_bf16_moe_small.cuh',
             'deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh',
             'tests/bench_sm120_single_token.py')
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def paths(rt):
    configs = {'candidate': SINGLE_TOKEN_CANDIDATES[(rt.d, rt.i, rt.e, rt.k)]}
    old = select_config(1, rt.e, rt.k, rt.d, rt.i)
    if old.l1 is not None:
        configs['legacy'] = old.l1, old.l2
    labels = [f'split-{a}-{b}' for a in (0, 32, 64, 128) for b in (0, 32, 64, 128)]
    labels += [f'public-{b}' for b in (0, 32, 64, 128)]
    return labels + list(configs), configs


def launch(rt, x, counts, slots, weights, label, configs):
    if label in configs:
        l1, l2 = configs[label]
        _C._sm120_moe_small(x, rt.gate_up, slots, weights, rt.h, True, *l1)
        _C._sm120_moe_small(rt.h, rt.down, slots, weights, rt.output, False, *l2)
        return
    _dispatch_slots[(rt.k, triton.cdiv(rt.d, 256))](
        x, rt.packed, slots, rt.d, rt.k, 256, num_warps=4)
    if label.startswith('public-'):
        b = int(label.split('-')[1])
        _C.m_grouped_bf16_swiglu_nt_masked(rt.packed, rt.gate_up, rt.h, counts, 1)
    else:
        _, a, b = label.split('-'); a, b = int(a), int(b)
        _C.m_grouped_bf16_gemm_nt_masked(rt.packed, rt.gate_up, rt.projection,
                                      counts, 1, sm120_m_tile=a)
        _swiglu[(rt.k, triton.cdiv(rt.i, 128))](
            rt.projection, rt.h, slots, rt.i, 128, num_warps=4)
    _C.m_grouped_bf16_gemm_nt_masked(rt.h, rt.down, rt.expert, counts, 1, sm120_m_tile=b)
    _combine[(1, triton.cdiv(rt.d, 256))](rt.expert, slots, weights, rt.output,
        rt.d, rt.k, 256, num_warps=4, enable_fp_fusion=False)


def measure(calls, rounds, seed):
    tables = {}
    for condition in ('warm', 'pressure512'):
        repeat = 10 if condition == 'warm' else 1
        graphs = {}
        for label, call in calls.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(repeat): call()
            for _ in range(3): g.replay()
            graphs[label] = g
        torch.cuda.synchronize()
        flush = torch.empty(512*1024*1024//4, device='cuda', dtype=torch.int32) if repeat == 1 else None
        samples = {label: [] for label in calls}; rng = random.Random(seed)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(rounds):
            order = list(calls); rng.shuffle(order)
            for label in order:
                if flush is not None: flush.zero_()
                start.record(); graphs[label].replay(); end.record(); end.synchronize()
                samples[label].append(start.elapsed_time(end)*1000/repeat)
        tables[condition] = {label: dict(median_us=statistics.median(v), samples_us=v)
                             for label, v in samples.items()}
        del graphs, flush
    return tables


@torch.no_grad()
def run(model, seed, rounds):
    d, i, e, k = MODELS[model]
    torch.manual_seed(seed)
    opts = dict(device='cuda', dtype=torch.bfloat16)
    gu = torch.randn((e, 2*i, d), **opts).div_(d**.5)
    down = torch.randn((e, d, i), **opts).div_(i**.5)
    rt = SmallMoE(1, gu, down, k, fusion_policy='single_token')
    x = torch.randn((1, d), **opts)
    route = make_routes(1, e, k, rt.capacity, seed)
    rt.validate_routes(*route)
    owners = []
    for name in ('h', 'output'):
        old = getattr(rt, name)
        owner = torch.full((old.numel()+32,), 17., **opts)
        setattr(rt, name, owner[16:-16].view(old.shape)); owners.append(owner)
    labels, configs = paths(rt)
    calls = {label: lambda label=label: launch(rt, x, *route, label, configs) for label in labels}
    checks = []; graphs = {}
    for changed in (False, True):
        if changed:
            new = make_routes(1, e, k, rt.capacity, seed+19, 'edge')
            for dst, src in zip(route, new): dst.copy_(src)
            route[2][0, -1].neg_(); x.neg_().add_(.125)
            rt.validate_routes(*route)
        flat = route[1].flatten()
        oracle_h, oracle_y = independent_oracle(rt, x, route[1], route[2])
        rt.projection.fill_(float('nan')); rt.expert.fill_(float('nan'))
        calls['split-32-32'](); torch.cuda.synchronize()
        reference_h = rt.h.view(-1, i)[flat].clone(); reference_y = rt.output.clone()
        compare(reference_h, oracle_h, exact=False); compare(reference_y, oracle_y, exact=False)
        for label, call in calls.items():
            rt.h.fill_(float('nan')); rt.output.fill_(float('nan'))
            rt.projection.fill_(float('nan')); rt.expert.fill_(float('nan'))
            if not changed:
                call(); torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): call()
                graphs[label] = graph
            graphs[label].replay(); torch.cuda.synchronize()
            h = rt.h.view(-1, i)[flat]
            exact = not label.startswith('public-')
            compare(h, reference_h, exact=exact); compare(rt.output, reference_y, exact=exact)
            hc = compare(h, oracle_h, exact=False); yc = compare(rt.output, oracle_y, exact=False)
            assert all(bool((o[:16] == 17).all() and (o[-16:] == 17).all()) for o in owners)
            checks.append(dict(label=label, changed_after_capture=changed, H=hc, Y=yc))
    tables = measure(calls, rounds, seed)
    result = dict(model=model, key=[d, i, e, k], seed=seed, configs=configs,
                  checks=checks, tables=tables, telemetry=telemetry())
    del graphs, calls, rt, owners, gu, down
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    p.add_argument('--rounds', type=int, default=15)
    a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, source_hashes=source_hashes(), rows=[],
                  extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
                  boundary='X + existing GPU routes -> Y, allocation/JIT/router excluded',
                  repaired_arithmetic=True, ncu_used=False)
    try:
        for seed in a.seeds:
            for model in a.models:
                print('RUN', model, seed, flush=True)
                report['rows'].append(run(model, seed, a.rounds))
                a.output.write_text(json.dumps(report, indent=2)+'\n')
        assert report['source_hashes'] == source_hashes()
        report['complete'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
