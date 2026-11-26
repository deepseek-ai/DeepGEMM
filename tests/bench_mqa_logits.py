#!/usr/bin/env python3
"""Reproduce SM90 non-paged MQA API latency on each branch separately.

Run with the branch's built deep_gemm on PYTHONPATH, for example:
  PYTHONPATH=. python tests/bench_mqa_logits.py --output main.csv
Repeat unchanged on the PR branch with --output pr.csv, on the same idle GPU.

Inputs are natural-layout E4M3 Q, per-token E4M3 K plus FP32 scales, and FP32
weights (32 heads, dimension 128). Input generation/quantization, range setup,
JIT compilation, and warmup are outside timing. Each timed call is the entire
public API, including allocation and any internal layout conversion; CUDA
events span that call's device timeline, including submission gaps. Wall time
also includes recording the end event and waiting for completion. Neither is
a single-kernel timing or a serving/TTFT measurement. Outputs are not copied
to CPU. Raw timings are saved beside the CSV as .samples.json. Compare
medians/p90, not only best samples.

CP cases model one rank's interleaved query positions, without communication.
The 4096-query cases use CP=1: one has no prefix; one has 155904 cached tokens.
Use --short --cuda-graph to measure short rows without Python submission gaps.
Graph results are per captured call, with replay overhead amortized over the
calls in each graph; they exclude the API's host allocation/submission cost.
"""
import argparse
import csv
import json
import math
import statistics
import time

import torch
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--calls-per-graph', type=int, default=100)
    args = parser.parse_args()
    assert args.iters >= 10 and args.warmup > 0 and args.calls_per_graph > 0
    assert torch.cuda.get_device_capability()[0] == 9, 'This benchmark targets SM90.'
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                          torch_cuda=torch.version.cuda, deep_gemm=deep_gemm.__file__,
                          deep_gemm_version=getattr(deep_gemm, '__version__', 'unknown'),
                          seed=args.seed, warmup=args.warmup, samples=args.iters,
                          cuda_graph=args.cuda_graph)))
    # (query rows on this rank, total K rows, CP size, CP rank)
    cases = [(32, 4096, 1, 0), (256, 32768, 1, 0),
             (1000, 160000, 8, 0), (1000, 160000, 8, 7),
             (1250, 210000, 1, 0), (4096, 4096, 1, 0), (4096, 160000, 1, 0)]
    if args.short:
        cases = [(nq, nk, 1, 0) for nq in (1, 3, 17, 32)
                 for nk in (4, 256, 257, 768, 769, 1024, 4096) if nq <= nk]
    fields = ['query_rows', 'key_rows', 'cp_size', 'cp_rank', 'new_request_tokens',
              'cached_prefix_tokens', 'heads', 'head_dim', 'samples', 'calls_per_sample',
              'gpu_median_ms', 'gpu_p90_ms', 'wall_median_ms', 'wall_p90_ms']
    raw_samples = []
    with open(args.output, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for nq, nk, cp_size, cp_rank in cases:
            heads, dim = 32, 128
            generator = torch.Generator(device='cuda').manual_seed(args.seed)
            q = torch.randn(nq, heads, dim, device='cuda', dtype=torch.bfloat16, generator=generator)
            k = torch.randn(nk, dim, device='cuda', dtype=torch.bfloat16, generator=generator)
            weights = torch.randn(nq, heads, device='cuda', dtype=torch.float, generator=generator) * heads ** -0.5
            q_in = q.to(torch.float8_e4m3fn), None
            k_in = per_custom_dims_cast_to_fp8(k, (0,), False)
            ks = torch.zeros(nq, device='cuda', dtype=torch.int)
            ke = nk - nq * cp_size + cp_rank + 1 + cp_size * torch.arange(nq, device='cuda', dtype=torch.int)
            max_seqlen_k = int(ke[-1])

            def run():
                return deep_gemm.fp8_fp4_mqa_logits(
                    q=q_in, kv=k_in, weights=weights,
                    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=max_seqlen_k)

            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()
            timed_run, calls_per_sample = run, 1
            if args.cuda_graph:
                expected = run().clone()
                valid = torch.arange(max_seqlen_k, device='cuda')[None, :] < ke[:, None]
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.calls_per_graph):
                        graph_output = run()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(graph_output[valid], expected[valid])

                def run_graph():
                    graph.replay()
                    return graph_output

                timed_run, calls_per_sample = run_graph, args.calls_per_graph
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            # Initialize the event objects before collecting samples.
            begin.record()
            end.record()
            end.synchronize()
            gpu_ms, wall_ms = [], []
            for _ in range(args.iters):
                begin.record()
                before = time.perf_counter()
                result = timed_run()
                end.record()
                end.synchronize()
                wall_ms.append((time.perf_counter() - before) * 1e3 / calls_per_sample)
                gpu_ms.append(begin.elapsed_time(end) / calls_per_sample)
                del result
            row = dict(query_rows=nq, key_rows=nk, cp_size=cp_size, cp_rank=cp_rank,
                       new_request_tokens=nq * cp_size, cached_prefix_tokens=nk - nq * cp_size,
                       heads=heads, head_dim=dim, samples=args.iters, calls_per_sample=calls_per_sample)
            for name, samples in (('gpu', gpu_ms), ('wall', wall_ms)):
                row[name + '_median_ms'] = statistics.median(samples)
                row[name + '_p90_ms'] = sorted(samples)[math.ceil(0.9 * len(samples)) - 1]
            writer.writerow(row)
            output_file.flush()
            raw_samples.append(dict(case=row, gpu_ms=gpu_ms, wall_ms=wall_ms))
            with open(args.output + '.samples.json', 'w') as samples_file:
                json.dump(raw_samples, samples_file)
            print(json.dumps(row), flush=True)
            if args.cuda_graph:
                assert torch.equal(graph_output[valid], expected[valid])
                del graph, graph_output, expected, valid
            del q, k, weights, q_in, k_in, ks, ke
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
