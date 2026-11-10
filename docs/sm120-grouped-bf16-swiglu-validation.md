# RTX 5090 validation: grouped BF16 SwiGLU

Measured on NVIDIA GeForce RTX 5090, PyTorch `2.13.0+cu130` (PyTorch reports CUDA
13.0). The measured source is commit
`6cfdc45ac28d7a4ed9347f5ae9dd62eb6c6a2419` plus the paired-output-store change
and its four alignment/tail tests. The tested kernel and test-file SHA256 hashes
are recorded below. Subsequent documentation changes do not change those files.

This patch implements the gate/up projection and SwiGLU, with a separate down
projection. Results exclude routing, communication, down GEMM and serving latency.

## Validation results

- The original scalar implementation passed 37/38 focused tests. The paired-store
  version passed 41/42; both fail the same seed-0 random-input FP64 test. These are
  retained failures, not passing or skipped tests.
- Both versions passed 50 ordinary BF16 regression cases each. The paired version
  passed 11 memcheck and 11 synccheck focused cases. Additional racecheck and
  initialized 48-expert memcheck/initcheck runs passed.
- 413 controlled scenarios, two seeds each, passed materialized-contract checks in
  both versions. All 826 complete valid-output hashes matched across versions.
  Scenarios cover E=4/8/16/32/48 and mean tokens=16/32/64/128/256/512/1024, with
  uniform, empty-expert and nonempty skew patterns. Matched patterns preserve the
  total token count and capacity; exact empty fractions require an integral
  number of experts.
- 56 supplemental boundary executions per version passed, with equal valid-output
  hashes. They include odd/small N, K tails, non-tile-aligned capacities, and
  aligned/two-byte-offset outputs.
- 12 Nsight Systems captures confirmed two kernels for materialized GEMM plus
  activation and one for fused SwiGLU. Direct-call profiler gaps are not used as
  CUDA Graph speedup measurements.
- Nsight Compute reports and raw CSV were retained. Resource information and SASS
  were exported for all 416 indexed CUBINs, including ordinary and helper kernels.
  Missing profiler/disassembler tools in the first run were installed and the
  missing evidence was recovered separately; the initial failed statuses remain
  in the original logs.

## CUDA Graph measurements

K=896, logical output N=4864, reused weights. For uniform cases below, values are
medians of six measurements per path, alternating ordinary/fused timing order.
The ordinary control is from the paired-store process. Scalar and paired runs
are separate processes; sub-percent gains are not a claim of stable improvement.

| Uniform case | Ordinary GEMM + activation, us | Scalar fused, us | Paired fused, us | Ordinary / paired |
|---|---:|---:|---:|---:|
| E4, 16 tokens/expert | 26.971 | 25.679 | 25.701 | 1.0494x |
| E4, 128 tokens/expert | 48.880 | 53.993 | 48.491 | 1.0080x |
| E16, 128 tokens/expert | 230.854 | 232.303 | 218.271 | 1.0576x |
| E48, 128 tokens/expert | 762.293 | 630.439 | 603.623 | 1.2629x |

Across 35 uniform cases, measured speedups over the ordinary path ranged from 1.0076x
to 1.2629x. Fusion also has regressions: an E16, mean-64 nonempty heavy-skew case
reproduced at 249.104 us fused versus 244.454 us ordinary (0.9813x, 12 samples).
This case has twelve 16-token experts and four 208-token experts, permuted, and
capacity 256. It has the same 1024 total tokens as the uniform E16/64 control.
No universal speedup, cold-cache or cross-GPU claim is made.

The E4/128 NCU comparison provides a mechanism for the paired-store improvement:
warp-level global-store instructions fell from 77,824 to 38,912 and store-sector
requests from 622,592 to 311,296. Launch registers remained 168/thread, and tensor
activity rose from 65.43% to 73.53%. Sector requests count accesses per instruction;
they are not unique cache lines or measured off-chip bytes.

## Known numerical question

The original random-input FP64 test remains unchanged and fails for both ordinary
materialized and fused paths. With E48/M128/K896/N4864, supplemental all-expert
diagnostics found 35 strict-threshold failures for seed 0 and 34 for seed 1 among
29,884,416 outputs per seed. Ordinary and fused outputs were bitwise identical,
and scalar and paired-store versions also matched.

Each of these 69 outputs involves one gate/up projection whose native BF16 value
and independently rounded exact-dot value are adjacent BF16 numbers. Exact
rational dot products agree with FP64 at the failing coordinates. The exact dots
lie within 3.208e-6 of a BF16 rounding midpoint. This supports a shared projection
rounding explanation, but the native pre-rounding FP32 accumulator was not
observed, so a specific Tensor Core reduction order is not claimed as measured.

FP32 accumulation followed by BF16 rounding need not equal BF16 rounding of an
exact real dot near a quantization boundary. The remaining decision is to specify
and independently validate the intended FP32 accumulation error contract, or to
change the implementation if exact-dot rounding is required. Matching the ordinary
path alone does not replace independent mathematical validation. Increasing a
fixed tolerance or silently removing this failure is not a numerical fix.

## Known BK32 spills and rejected experiment

At expected M >= 2048, the heuristic can select BM128/BN128/BK32 with four shared
pipeline stages. Both scalar and paired fused specializations show static spill
loads/stores in compiler logs and LDL/STL instructions in SASS. Actual NCU
captures confirm local traffic; the ordinary GEMM has zero local sectors in the
same two cases.

| Half-empty case, active experts have 2048 tokens | Paired fused local load sectors | Paired fused local store sectors |
|---|---:|---:|
| E4 (2 active), capacity 3328 | 183,616 | 183,616 |
| E48 (24 active), capacity 3328 | 2,203,392 | 2,203,392 |

Source array size estimates depend on the actual warp layout. These launches use
kNWarps=1, kMWarps=8, kMTilesPerWarp=1 and kNTilesPerWarp=16. Dual accumulator
arrays occupy 128 float slots; two operand fragment buffers occupy 136 u32 slots.
These 264 source slots are not the compiled physical register count.

An isolated experiment reduced register fragments to one buffer only in fused
BK32, keeping the shared TMA pipeline and per-output MMA order. It passed six
exact-dot/Graph/boundary cases, ordinary regressions, sanitizer checks and 48
controlled output-hash comparisons. However, LDL/STL instruction counts did not
decrease, and local sectors increased by about 0.66%. Four affected Graph timing
cases improved by only 0.11%-0.41%; two unchanged-path controls varied by
+0.57%/+0.11%. The experiment does not establish a useful spill reduction and is
excluded from this patch. The validated paired-store schedule is retained.

## Reproduction and review status

```bash
python -m pytest -q -ra tests/test_sm120_grouped_swiglu.py
python -m pytest -q -ra tests/test_sm120_bf16_gemm.py tests/test_sm120_bf16_odd_n.py tests/test_sm120_native_rejections.py -k 'not k_grouped_small_stride and not test_filter'
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest -q -ra tests/test_sm120_grouped_swiglu.py -k 'exact_dot_product or graph_replay or runtime_alignment or capacity_tail or output_pair'
compute-sanitizer --tool synccheck --error-exitcode 99 python -m pytest -q -ra tests/test_sm120_grouped_swiglu.py -k 'exact_dot_product or graph_replay or runtime_alignment or capacity_tail or output_pair'
python experiments/sm120_grouped_swiglu_native.py --rows 128 128 128 128 --max-m 128 --repeats 12 --output timing.json
```

The first command is expected to retain the documented seed-0 failure on the
tested environment. This contribution remains Draft while the numerical question
is unresolved. Performance and profiler evidence do not waive that failure.

Tested file identities (SHA256):

- `deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh`: `bcc75a436bb6af663a24c02668693759b988b2b7395706c607d26d0931c93ac9`
- `tests/test_sm120_grouped_swiglu.py`: `442ec81f9a830041235838556032f73b6741bc1591059fb47e2cbd2a8d7a54ff`
