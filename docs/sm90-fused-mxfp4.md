# Fused SM90 MegaMoE with MXFP4 weights

This path runs dispatch, L1 GEMM, SwiGLU and activation quantization, L2 GEMM,
and NVLink combine in one persistent kernel. Activations use FP8 E4M3 with
per-128 FP32 scales. Routed weights use packed E2M1, K32 UE8M0 scales, and one
FP32 secondary scale per local expert.

## Provenance

The fused SM90 FP8 infrastructure is derived from
[PR #440](https://github.com/deepseek-ai/DeepGEMM/pull/440), specifically
`8cf8180fe6f97eb65ee6638a14a21497cc1cb5e0`, and adapted to the current `nv_dev`
DeepJIT runtime. This change includes the required infrastructure and targets
`nv_dev` directly. It does not depend on #440 merging first.

The MXFP4 format handling and software decoder build on
[PR #411](https://github.com/deepseek-ai/DeepGEMM/pull/411), specifically
`5d32216686b982a39eabccca9419af430a60cfc2`, whose SM90 implementation references
Humming. The PRMT/LUT decoder and the idea of small-M swapAB are prior work;
the tiled register-operand pipeline and specialized split-M tails described
below extend those foundations in the #440-derived fused kernel.
Processed tensors are private kernel payloads: reprocess the
checkpoint weights with this implementation rather than assuming another
implementation's processed triples are interchangeable.

The small-M source-metadata optimization was developed in collaboration with
a colleague: CTA0 handles metadata for sufficiently small bounded workloads,
avoiding the corresponding multi-CTA collection synchronization.

## API

Existing SM90 non-fused APIs keep their defaults. Select the fused buffer
explicitly, and request tiled weights for the optimized MXFP4 path:

```python
import torch
import deep_gemm

# group: initialized torch.distributed ProcessGroup
# packed_l1: [local_experts, 2 * intermediate, hidden // 2], int8/uint8
# scales_l1: [local_experts, 2 * intermediate, hidden // 32], UE8M0
# packed_l2: [local_experts, hidden, intermediate // 2], int8/uint8
# scales_l2: [local_experts, hidden, intermediate // 32], UE8M0
w1, w2 = deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
    (packed_l1, scales_l1), (packed_l2, scales_l2), layout='tiled')
buf = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
    group, num_experts, capacity_per_rank, topk, hidden, intermediate,
    fused=True)
# Before each call, populate buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights.
# y.shape[0] defines this rank's active row count.
y = torch.empty((local_rows, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_mxfp4_mega_moe(y, w1, w2, buf,
    num_tokens_bound=common_rows_bound)
# Complete outstanding work before releasing the collective buffer.
torch.cuda.synchronize()
buf.destroy()
```

`num_experts` is global and divisible by the process-group size. `capacity_per_rank`
is allocation capacity. `num_tokens_bound` is an optional common upper bound
covering the active row count on **every** rank; omitting it uses capacity.
A rank's row count cannot by itself establish a valid peer bound. All ranks
must use compatible allocation and schedule parameters.

The optional keyword `split_k=True` is a performance hint for one global decode
token, applied consistently on every rank. The kernel checks routing-dependent
eligibility. Leave it off for other workloads.

FP8 weights use the existing `transform_weights_for_mega_moe_sm90` and
`fp8_mega_moe` APIs with a fused buffer. The default buffer remains non-fused.
Fused buffers have a separate workspace contract, including split-K partial
storage, and cannot replace already allocated non-fused buffers.

## Weight layout and numerical contract

The transform returns a triple per layer: processed packed values, relative
scale bytes, and FP32 secondary scales. It accepts checkpoint UE8M0 as `uint8`,
`float8_e8m0fnu`, or FP32 positive powers of two. Optional FP32 per-expert
checkpoint secondary scales must be provided for both layers together.

For `layout='tiled'`, packed values have shape
`[E, N/256, K/128, 16384]`; each stage is a contiguous 16 KiB region.
L1 gate/up rows are interleaved in groups of eight. Scale tensors retain logical
`[E, N, K/32]` shapes, with a coalesced physical layout where applicable.
The transform also changes packed sign-bit organization and byte order.
Do not reshape a checkpoint directly into this layout or apply the transform
to an already processed tensor.

Relative exponents are bounded to an 11-exponent range. Smaller groups are
requantized to E2M1. Reference correctness therefore compares against the
**processed weights**, rather than the original unquantized checkpoint.
The device places its x64 compensation in the secondary scale where representable,
and uses the activation scale for experts whose secondary scale would overflow.

## Scheduling and tuning

Small expert tiles can decode packed values into registers and use swapAB
register/shared WGMMA. Larger workloads use split-M and specialized tail
handling. The math warpgroups execute the GEMMs, epilogues, and combine; there
is no separate epilogue/combine warp pool.

The CTA0 metadata path is narrowly enabled for the validated EP8 Flash geometry:
H4096/I2048, 256 experts, topk6, capacity4096, 78 selected SMs, local row counts
1/2/4/8 with a positive matching common bound, mode4 swapAB, and the matching
tile/SF/schedule settings. `DG_SM90_ROUTED_HEADONLY=0` disables it for A/B.
Other geometries retain the general metadata path. These thresholds are H20
measurements, not a claim of optimal tuning on every SM90 GPU.

## Two main optimizations

**Register decoding for small expert tiles.** Packed weights are transformed
into contiguous 16 KiB stages. Swapping WGMMA operands puts weight rows on its
fixed M dimension and valid token rows on its variable N dimension. The math
warpgroup decodes E2M1 directly into register A fragments, then issues RS WGMMA
against the shared-memory activations. Compared with expanding the weights into
shared memory, this removes the expanded-tile store/read and the associated
per-k-block proxy fence and warpgroup synchronization on this path.
Compile-time modes select N8/16/24/32/40/48 buckets and double buffering;
runtime valid-row counts select the actual bucket. Eligible swap accumulators
feed SwiGLU and activation quantization directly. Larger tiles retain a regular
fallback. Keeping modes separate also limits code-size and register-allocation
effects on the smallest loop.

**Split-M with specialized tails for larger workloads.** The MXFP4 path uses
128x256 tiles where the two math warpgroups cover disjoint 64-row activation
halves and share the decoded weight tile. This adapts the split-M structure of
the fused FP8 implementation to packed weights. Partial tiles select separate
compiled loops: no WGMMA for an empty half, RS swap buckets for up to 40 rows,
and a dual-warpgroup N96 swap path for 65--96 rows. Dynamic tail work is issued
in passes ordered by valid rows. These paths preserve the full tile's scaling
chain so changing token slot order does not change repeated output bits.
The default host heuristic selects split-M above 256 rows per rank when the
tiled/per-128 activation-scale configuration permits it. Forced inlining
prevents the tile callback from spilling its accumulators into outlined calls.

## Reproducing checks

Build the extension with the repository's normal build procedure. A C++20-capable
compiler is required; GCC 13 was used for the DeepJIT port.

```bash
python tests/test_sm90_mxfp4_weights.py
python tests/test_sm90_fused_mxfp4.py --np 1 \
  --m 1 8 64 128 256 1024 2048 --fm 1 0 --bound --capacity 4096 --graph
python tests/test_sm90_fused_mxfp4.py --np 8 \
  --m 1 2 4 8 64 256 --fm 1 0 --bound --split-k --capacity 4096 --graph
python tests/bench_sm90_fused_mega_moe.py --impl pr440_mxfp4 --np 8 \
  --m 1 2 4 8 16 32 64 --global-m --bound --split-k --cap 4096 \
  --layout tiled --graph --iters 50 --pad-cycles 2000000 --num-sms 78
```

The correctness script reports `calc_diff`, deterministic replay, finite output,
and cumulative expert receive counts. Its tolerance is 0.002; this is not a
promise of bitwise agreement with PyTorch GEMMs. Default/off comparisons use
identical quantized inputs and require bitwise agreement.

Benchmark `--m` means rows per rank unless `--global-m` is supplied. The benchmark
uses a 256 MiB L2 flush and CUDA events after a synchronized device pad.
`median_us` is the median of per-iteration minimum times over ranks;
`max_rank_us` reports the corresponding maximum-rank statistic, which includes
start skew. Neither statistic is service latency or host wall time. Use separate
`DG_JIT_CACHE_DIR` values and three interleaved rounds for each variant.

Validation scope: H20-3e (SM90a); H100 and vLLM end-to-end overlap are unverified.

Four-way Graph measurements against frozen #383/#411/#440 commits, including
M=1--8192 for Flash and Pro, are documented in
[the benchmark report](sm90-fused-mxfp4-benchmarks.md).
