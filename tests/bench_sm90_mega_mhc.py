"""Single-GPU shifted Mega mHC correctness and CUDA Graph latency.

Standalone (synthetic inputs and weights, no SGLang dependency):
  CUDA_VISIBLE_DEVICES=0 python tests/bench_sm90_mega_mhc.py --tokens 4096 32

Optional SGLang post + mix_stats + combine/RMSNorm comparison:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/sglang/python python \
    tests/bench_sm90_mega_mhc.py --sglang-baseline --tokens 4096 32 \
    --model /path/to/checkpoint --layer 0 --sublayer ffn

The optional baseline needs SGLang's deepseek_v4_mhc module. Checkpoint loading
needs safetensors. Activations remain synthetic; this measures one shifted mHC
boundary, including all outputs, not model inference or distributed collectives.
Decode joins SGLang's mix_stats side stream before timing ends. Graphs contain
16 calls; timing alternates baseline/optimized order and reports median latency.
"""

import argparse
import json
from pathlib import Path
import statistics

import torch
import deep_gemm

from test_mega_mhc import (
    check_correctness, logical_outputs, make_inputs, make_outputs, mhc_reference, run_mega_mhc,
)


def load_weights(args):
    from safetensors import safe_open

    config = json.loads((args.model / 'config.json').read_text())
    config = config.get('text_config', config)
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'layers.{args.layer}.'
    names = dict(fn=prefix + f'hc_{args.sublayer}_fn',
                 mix_scales=prefix + f'hc_{args.sublayer}_scale',
                 mix_bases=prefix + f'hc_{args.sublayer}_base',
                 rmsnorm_weight=prefix + f'{args.sublayer}_norm.weight')
    weights = {}
    for name, key in names.items():
        with safe_open(str(args.model / index[key]), framework='pt', device='cpu') as f:
            value = f.get_tensor(key)
        dtype = torch.bfloat16 if name == 'rmsnorm_weight' else torch.float32
        weights[name] = value.to(device='cuda', dtype=dtype).contiguous().squeeze()
    weights['fn'] = weights['fn'].view(24, -1)
    assert config['hc_mult'] == 4
    return config, weights


def time_graphs(runs, repetitions, rounds):
    graphs = {}
    for name, run in runs.items():
        for _ in range(4):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repetitions):
                run()
        graphs[name] = graph
        for _ in range(2):
            graph.replay()
    torch.cuda.synchronize()
    times = {name: [] for name in runs}
    for i in range(rounds):
        order = list(runs) if i % 2 == 0 else list(reversed(runs))
        for name in order:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / repetitions)
    return {name: statistics.median(values) for name, values in times.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', type=int, nargs='+', default=[4096, 32])
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--sublayer', choices=('attn', 'ffn'), default='ffn')
    parser.add_argument('--sglang-baseline', action='store_true')
    parser.add_argument('--repetitions', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=8)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability()[0] == 9, 'requires SM90'
    assert args.repetitions > 0 and args.rounds > 0
    torch.manual_seed(2026)
    weights = {}
    hidden, eps, rms_eps, iters = args.hidden, 1e-6, 1e-6, 20
    if args.model:
        config, weights = load_weights(args)
        hidden, eps, rms_eps, iters = (config[k] for k in
                                      ('hidden_size', 'hc_eps', 'rms_norm_eps', 'hc_sinkhorn_iters'))
    if args.sglang_baseline:
        from sglang.srt.models import deepseek_v4_mhc as mhc
        from sglang.kernels.ops.layernorm.mhc import split_bf16_hc_weight
        from sglang.srt.runtime_context import get_parallel

        get_parallel().override_permanently(attn_cp_size=1)
        print(f'SGLang baseline: {mhc.__file__}')
    print(f'{torch.cuda.get_device_name()}, SMs={deep_gemm.get_num_sms()}, '
          f'PyTorch={torch.__version__}, CUDA={torch.version.cuda}, H={hidden}, HC=4, '
          f'Sinkhorn iterations={iters}, checkpoint weights={bool(args.model)}')
    print(f'DeepGEMM: {deep_gemm.__file__}')
    for tokens in args.tokens:
        inputs, _ = make_inputs(tokens, hidden, 2026, True)
        inputs.update(weights)
        inputs.update(hc_norm_eps=rms_eps, hc_pre_eps=eps, hc_post_scale=2.,
                      sinkhorn_eps=eps, num_sinkhorn_iters=iters, rmsnorm_eps=rms_eps, rmsnorm_scale=1.)
        outputs = make_outputs(inputs, 'bf16')
        runs = {}
        if args.sglang_baseline:
            cfg = mhc.HcConfig(4, iters, eps, rms_eps, hidden, True, False)
            norm = mhc.RMSNorm(hidden, eps=rms_eps, weight_dtype=torch.bfloat16).cuda()
            norm.weight.copy_(inputs['rmsnorm_weight'])
            hc = mhc.HcSubLayer(cfg, inputs['fn'], inputs['mix_scales'], inputs['mix_bases'], norm,
                               bf16_parts=split_bf16_hc_weight(inputs['fn']))
            pre, post, comb = mhc.mix_stats(hc, inputs['residual'])
            inputs.update(shifted_prev_mix=pre.unsqueeze(-1), post_mix=post.unsqueeze(-1), comb_res_mix=comb)
            stream = torch.cuda.Stream() if tokens == 32 else None

            def baseline():
                updated = mhc.post(cfg, inputs['x'], inputs['residual'], post, comb)
                if stream is not None:
                    stream.wait_stream(torch.cuda.current_stream())
                next_pre, next_post, next_comb = mhc.mix_stats(hc, updated, stream)
                normalized = mhc.combine(hc, mhc.HcState(updated, pre))
                if stream is not None:
                    torch.cuda.current_stream().wait_stream(stream)
                return dict(new_residual=updated, y_bf16=normalized,
                            new_prev_mix=next_pre.unsqueeze(-1), new_post_mix=next_post.unsqueeze(-1),
                            new_comb_res_mix=next_comb)

            runs['SGLang'] = baseline

        def mega():
            run_mega_mhc(inputs, outputs)
            return logical_outputs(outputs)

        runs['MegaMHC'] = mega
        actual = mega()
        reference = logical_outputs(mhc_reference(**inputs, sf_layout='bf16'))
        check_correctness(actual, reference, ('MegaMHC', tokens, hidden))
        if args.sglang_baseline:
            check_correctness(baseline(), reference, ('SGLang', tokens, hidden))
            check_correctness(actual, baseline(), ('MegaMHC vs SGLang', tokens, hidden))
        timing = time_graphs(runs, args.repetitions, args.rounds)
        summary = ', '.join(f'{name}={us:.3f} us' for name, us in timing.items())
        if args.sglang_baseline:
            summary += f", speedup={timing['SGLang'] / timing['MegaMHC']:.3f}x"
        print(f'T={tokens}: {summary}', flush=True)


if __name__ == '__main__':
    main()
