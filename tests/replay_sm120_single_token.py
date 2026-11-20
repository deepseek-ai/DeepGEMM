"""Revision-matched public-entry replay of sealed real Qwen3 layer traces.

The trace is teacher-forced HF prefix data, not a serving benchmark. Weight
and trace identities are checked before use. No model download is performed.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from deep_gemm.small_moe import SmallMoE
from tests.bench_sm120_single_token import launch, measure, paths, source_hashes
from tests.bench_sm120_small_moe import compare, digest, independent_oracle, telemetry

TRACE_SHA256 = '3b534da5ff7fa08ec728050c0498a36931ead73e6eced42510ff946ce9636dd3'


def load_weights(layer, manifest):
    prefix = f'model.layers.{layer}.mlp.experts.'
    selected = {n: p for n, p in manifest['weight_files'].items() if n.startswith(prefix)}
    state = {}; receipts = {}
    for path in sorted(set(selected.values())):
        with safe_open(path, framework='pt', device='cpu') as f:
            for name, source in selected.items():
                if source != path: continue
                value = f.get_tensor(name)
                assert value.dtype == torch.bfloat16, name
                actual = digest(value)
                assert actual == manifest['layer_weight_sha256'][str(layer)][name], name
                state[name] = value; receipts[name] = actual
    key = lambda e, role: f'{prefix}{e}.{role}_proj.weight'
    gate_up = torch.stack([torch.cat((state[key(e, 'gate')], state[key(e, 'up')]))
                          for e in range(128)]).cuda()
    down = torch.stack([state[key(e, 'down')] for e in range(128)]).cuda()
    return gate_up, down, receipts


def pack_routes(ids, weights, capacity):
    # T1: distinct expert ids imply one row per selected expert, no histogram
    # readback. Preparation is outside the measured expert-branch boundary.
    assert ids.ndim == 1 and ids.unique().numel() == ids.numel()
    counts = torch.zeros(128, dtype=torch.int32)
    counts[ids.long()] = 1
    return counts.cuda(), (ids.long()[None, :]*capacity).cuda(), weights.float()[None, :].cuda()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=31)
    a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = a.data/'inputs.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete']
    assert manifest['trace_sha256'] == TRACE_SHA256
    assert hashlib.sha256((a.data/'trace.pt').read_bytes()).hexdigest() == TRACE_SHA256
    trace = torch.load(a.data/'trace.pt', weights_only=True, map_location='cpu')
    plan = json.loads(a.plan.read_text())
    assert plan['complete'] and 'qwen30' in plan['plans']
    frozen_selected = plan['plans']['qwen30']['selected']
    selected = 'entry'
    controls = plan['plans']['qwen30']['controls']
    labels = list(dict.fromkeys([selected, frozen_selected, *controls]))
    report = dict(complete=False, source_hashes=source_hashes(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        trace_sha256=TRACE_SHA256, scope=manifest['scope'],
        selected=selected, frozen_selected=frozen_selected, labels=labels, weights={}, checks=[], timing=[],
        telemetry_before=telemetry(), ncu_used=False,
        steady_boundary='X + existing GPU counts/slots/weights -> Y; no router/planning/JIT/allocation',
        sequence_boundary='32 recorded calls; GPU copies into stable X/routes + expert branch; per-call average')
    save = lambda: a.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for layer in (0, 4, 7):
            print('LOAD LAYER', layer, flush=True)
            gu, down, receipts = load_weights(layer, manifest)
            report['weights'][str(layer)] = receipts
            rt = SmallMoE(1, gu, down, 8, fusion_policy='single_token')
            assert (rt.config.l1 is not None) == plan['plans']['qwen30']['fusion_accepted']
            x = torch.empty((1, 2048), device='cuda', dtype=torch.bfloat16)
            counts = torch.empty(128, device='cuda', dtype=torch.int32)
            slots = torch.empty((1, 8), device='cuda', dtype=torch.int64)
            weights = torch.empty((1, 8), device='cuda', dtype=torch.float32)
            _, configs = paths(rt)
            calls = {label: lambda label=label: launch(rt, x, counts, slots, weights, label, configs)
                     for label in labels}
            calls['entry'] = lambda: rt.forward(x, counts, slots, weights)
            for call in calls.values():
                # Initialize first so warmup never reads arbitrary slot values.
                counts.zero_(); counts[:8] = 1
                slots.copy_((torch.arange(8, device='cuda')*rt.capacity)[None, :])
                weights.fill_(1/8); x.zero_(); call()
            graphs = {}
            for label, call in calls.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): call()
                graphs[label] = graph
            for prompt in range(8):
                held = []
                for step in range(32):
                    data = trace[layer]
                    record_x = data['hf_x'][step, prompt][None, :].contiguous().cuda()
                    route = pack_routes(data['hf_ids'][step, prompt], data['hf_weights'][step, prompt], rt.capacity)
                    held.append((record_x, *route))
                    for dst, src in zip((x, counts, slots, weights), held[-1]): dst.copy_(src)
                    rt.validate_routes(counts, slots, weights)
                    oracle = independent_oracle(rt, x, slots, weights) if prompt in (0, 3, 7) and step in (0, 15, 31) else None
                    reference_h = reference_y = None
                    # First frozen ordinary split is the bitwise reference.
                    split = next(label for label in labels if label.startswith('split-'))
                    for label in [split]+[v for v in labels if v != split]:
                        rt.h.fill_(float('nan')); rt.output.fill_(float('nan'))
                        rt.projection.fill_(float('nan')); rt.expert.fill_(float('nan'))
                        graphs[label].replay(); torch.cuda.synchronize()
                        h = rt.h.view(-1, 768)[slots.flatten()]
                        assert bool(torch.isfinite(h).all()) and bool(torch.isfinite(rt.output).all())
                        if reference_h is None:
                            reference_h, reference_y = h.clone(), rt.output.clone()
                        exact = not label.startswith('public-')
                        hc = compare(h, reference_h, exact=exact)
                        yc = compare(rt.output, reference_y, exact=exact)
                        row = dict(layer=layer, prompt=prompt, step=step, label=label,
                                   H=hc, Y=yc, changed_route_graph=True)
                        if oracle:
                            row['oracle_H'] = compare(h, oracle[0], exact=False)
                            row['oracle_Y'] = compare(rt.output, oracle[1], exact=False)
                        # HF is diagnostic only: its materialization/reduction differs.
                        if label == selected:
                            expected = data['hf_y'][step, prompt].cuda().float()
                            delta = (rt.output[0].float()-expected).abs()
                            row['historical_HF'] = dict(max_abs=float(delta.max()),
                                outside_original_tolerance=int((delta > .002+.01*expected.abs()).sum()))
                        report['checks'].append(row)
                    if prompt in (0, 3, 7) and step in (0, 15, 31):
                        tables = measure(calls, a.rounds, 25107+layer*100+prompt*32+step)
                        report['timing'].append(dict(kind='steady', layer=layer, prompt=prompt, step=step, tables=tables))
                def sequence(label):
                    for record in held:
                        for dst, src in zip((x, counts, slots, weights), record): dst.copy_(src)
                        calls[label]()
                sequences = {label: lambda label=label: sequence(label) for label in labels}
                tables = measure(sequences, a.rounds, 25223+layer*10+prompt)
                for condition in tables.values():
                    for row in condition.values():
                        row['samples_us'] = [v/32 for v in row['samples_us']]
                        row['median_us'] /= 32
                report['timing'].append(dict(kind='sequence_average', layer=layer, prompt=prompt, tables=tables))
                print('REPLAY', layer, prompt, 'checks', len(report['checks']), flush=True); save()
                del sequences, held
            del graphs, calls, rt, gu, down, configs
            gc.collect(); torch.cuda.empty_cache()
        assert report['source_hashes'] == source_hashes()
        report['complete'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['telemetry_after'] = telemetry(); save()


if __name__ == '__main__': main()
