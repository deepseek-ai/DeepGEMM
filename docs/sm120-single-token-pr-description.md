Adds masked grouped BF16 GEMM+SwiGLU on SM120, repairs projection rounding failures in the original FP64-reference tests, and provides opt-in, bounded decode MoE fusion. Existing API and policy defaults are preserved.

### Implementation and numerical contract

The small decode path uses two kernels:

1. Direct-X gate/up GEMM + SwiGLU, writing packed BF16 H.
2. Down GEMM + CTA-local ordered weighted combine, writing Y.

Each math warp owns an expert; four producer warps feed the staged TMA pipeline. L1 shares the input row across its expert bundle. L2 keeps the rounded expert contributions in CTA shared memory and combines them in route order. H remains in global memory. There are no global work queues or grid barriers; stage ownership synchronization remains necessary.

A projection near a BF16 midpoint could previously round to the wrong neighbor after FP32 accumulation and exceed the existing SwiGLU tolerance. The repair uses zero-C K16 MMA, explicit FP32 accumulation, and warp-cooperative TwoSum refinement near BF16 midpoints. Production uses no FP64. Original reference calculations, BF16 materialization boundaries, seeds, and `rtol=0.01, atol=0.002` are retained. This is tested numerical coverage, not a universal correct-rounding guarantee. Performance comparisons use repaired arithmetic.

Both public masked BF16 GEMM and SwiGLU accept keyword-only `sm120_m_tile=0/32/64/128`; zero preserves the existing heuristic. BM32 requires N>32. Generated stubs preserve keyword-only defaults. Private `_sm120_moe_small` adds explicit gate/up order and direct expert-ID inputs without changing existing callers.

### Support and fallback

`SmallMoE` is an explicitly imported inference workspace consuming precomputed routing. Its existing default `fusion_policy='validated'` remains unchanged (E64/E128):

| D / I | T | fused top-k |
|---|---:|---|
| 2048 / 768 | 1 | 1, 2, 4, 8 |
| 2048 / 768 | 2 | 1, 2, 4 |
| 2048 / 768 | 4 | 1, 2 |
| 4096 / 1536 | 1 | 1, 2, 4, 8 |
| 4096 / 1536 | 2 | 1, 2 |

The separate opt-in `fusion_policy='single_token'` requires T=1 and RTX 5090 / 170 SMs:

| D / I / E / top-k | L1 BN / stages / experts per CTA | L2 BN / stages / experts per CTA |
|---|---|---|
| 2048 / 768 / 128 / 8 | 32 / 3 / 2 | 16 / 4 / 8 |
| 4096 / 1536 / 128 / 8 | 16 / 2 / 8 | 32 / 2 / 8 |
| 2048 / 1408 / 60 / 4 | 16 / 3 / 4 | 16 / 4 / 4 |
| 2048 / 1408 / 64 / 6 | 16 / 3 / 2 | 16 / 4 / 6 |

Multi-token requests, unknown keys and other hardware fall back under this policy. The D3584/I2560/E64/k8 pressure-regression candidate remains rejected. `enable_fusion=False` forces the ordinary path. Shape-family labels are synthetic dimension coverage; full-model BF16 results below cover one checkpoint.

Forward selection reads metadata only: no device readback, online tuning or allocation. Routing values must meet the documented unique-expert/dense-slot contract; `validate_routes` checks them outside capture. One workspace cannot be shared concurrently across independent streams. Current streams, changed contents during CUDA Graph replay and fallback are covered by the recorded tests.

### Experimental framework integration

The example adapter requires exactly vLLM 0.32.0 and is disabled unless `SM120_VLLM_ENABLE=1`. It preserves native routing, weight conversion, shared-expert order and native fallback. It rejects unsupported scheduling, DP/EP, DBO/EPLB, quantization, activation/scaling, bias/LoRA and weight layouts.

PP2 uses local I=768 and the public T1 plan. TP2 additionally requires `SM120_VLLM_TP_ENABLE=1`, exact local shape `(T,E,k,D,I)=(1,128,8,2048,384)` and unpadded shards. TP partitions each expert's gate/up intermediate channels together; L2 returns the rank-local result to the existing vLLM outer all-reduce. This adds no TP communication algorithm and does not expand the public policy.

The measured TP2 configuration explicitly sets `SM120_VLLM_DIRECT_IDS=1`, `SM120_TP_L1=16,4,2` and `SM120_TP_L2=16,2,8`. The adapter's unset-variable defaults are different and must not be substituted for this measured configuration. PP2 uses slot preparation (`SM120_VLLM_DIRECT_IDS=0`). See `examples/vllm_sm120/README.md` for the exact boundary.

### Validation and measured effect

Earlier core validation retained in the repository includes the original FP64-reference checks, saved rounding/cancellation operands, ordinary-vs-fused BF16 boundaries, policy holdouts and real-weight layer replay. The recorded final single-token revision passed 30 public correctness tests, 42 original contract tests, four numerical regression methods and all four sanitizers. Later measured-snapshot checks additionally cover gate/up order, direct IDs, TP local I=384, two-rank outer sum and four sanitizers. Counts refer to their individual recorded revisions; they are not summed into one run.

The full-model snapshot is identified by package-manifest SHA256 `676eb83a3784082fbf03cc9f7cd2508c34aaeb55b43fd60462e1fda4b35f012a` and extension SHA256 `e5274e4a8c9aac8747258e7ed8fa1abca84986c3885bf56424d6cdb37437d2d6`.

On two RTX 5090s, raw BF16 Qwen3-Coder-30B-A3B-Instruct, vLLM 0.32.0, compilation mode 0, decode Graph and one active request:

- The 12 equal-output full-model timing scenes reduced request latency by 2.27–7.22% for PP2 and 4.82–8.30% for TP2 against each mode's native auto-selected FlashInfer CUTLASS path.
- HTTP timing has 72 rows: 56 equal-output rows show 1.00–3.00% PP2 and 2.23–5.84% TP2 latency reductions. All 16 differing-output rows are disclosed and excluded from equal-output speedup claims.
- Each implementation used two independent processes in the same host session. These measurements do not establish cross-session stability, throughput or batch>1 advantages. The final quality run below adds no performance percentages.

Final preregistered quality acceptance completed 11/11 tasks with `PASS_SCOPED_NLL`. It used 48 independent sources per domain (English/SQuAD dev, Chinese/CMRC dev, code/MBPP test), 32 fixed-prefix decode positions per source, and 4608 scored positions per parallel mode. Paired source-cluster bootstrap used 50,000 resamples with a six-family Bonferroni adjustment; the engineering margin was fixed at log(1.01)=0.00995033 nat before collection.

| Parallel / domain | Mean NLL difference, fused − native (nat) | Adjusted interval |
|---|---:|---|
| PP2 / English | −0.00114408 | [−0.00775824, 0.00519780] |
| PP2 / Chinese | −0.00107341 | [−0.00678663, 0.00462963] |
| PP2 / code | +0.00197745 | [−0.00388146, 0.00873573] |
| TP2 / English | +0.00206673 | [−0.00361240, 0.00769698] |
| TP2 / Chinese | +0.00060420 | [−0.00508645, 0.00644499] |
| TP2 / code | −0.00267468 | [−0.00648276, 0.00098186] |

All six upper bounds are below the preset margin and all nine hard controls passed. Independent local reanalysis of raw logits reproduced the decision. The intervals have approximate bootstrap coverage.

This is bounded NLL evidence, not semantic equivalence or task-accuracy evidence. Top-1 agreement is 98.50% for PP2 and 98.39% for TP2; natural generation can diverge. Earlier numerical/strict-gate failures remain documented. Same-input localization found ordinary BM32 and fused BF16 H/local Y bitwise equal at the observed first global MoE layer; native local Y differs, but its internal projection/accumulation and the outer TP reduction were not observed, so this does not identify the precise internal cause.

### Review and release boundary

Review the core numerical/API/policy changes separately from private framework inputs and the default-disabled adapter. The finite support table and default opt-in status remain unchanged. Large-load scheduling, training, quantization, arbitrary models, batched serving and default framework enablement are outside this change.

`docs/sm120-final-review-scope.md` lists the final source identities, evidence locations, preserved failures and remaining limitations. Historical detailed calibration and timing records remain in `docs/sm120-single-token-results.md`, `docs/sm120-small-moe.md` and `docs/validation/`. GPU evidence belongs to the recorded measured snapshot; the local review preparation checks patch applicability, source content and CPU contracts, and does not claim a fresh Linux build or GPU run after LF normalization.
