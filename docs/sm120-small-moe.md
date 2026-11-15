# Opt-in BF16 decode fusion on SM120

`from deep_gemm.small_moe import SmallMoE` provides a pre-routed, inference-only
MoE workspace. It does not replace existing GEMM APIs or enable fusion globally.
The native building block uses DeepJIT and the current CUDA stream; production
code does not import research runtimes, generate patched CUDA source, or use
ctypes launchers.

## Supported input contract

- SM120, contiguous CUDA BF16 `gate_up[E,2I,D]`, `down[E,D,I]`, and `X[T,D]`.
  Gate rows precede up rows. D and I must be positive and divisible by 128.
- A precomputed plan with int32 `counts[E]`, int64 `slots[T,k]`, and FP32
  `weights[T,k]`, contiguous and on the same device. Routing-plan construction
  and validation are outside the forward path.
- `slot = expert * capacity + local_row`. Experts are distinct within a token,
  slots are globally unique, and each expert uses exactly rows `0..count-1`.
  Capacity is T rounded up to 128 by default; custom capacities must cover T
  and be divisible by 128. Weights must be finite; normalization is optional.
- Input tensors must not alias workspace allocations. Weights are fixed
  allocations, although their contents may change between invocations.
  All supplied tensors must have 16-byte aligned storage addresses.

Call `workspace.validate_routes(counts, slots, weights)` outside Graph capture
if the plan is not already guaranteed by its producer. This synchronizes to
check values. `forward` checks metadata but trusts device-side plan values;
invalid unchecked slots are not a supported input.

The returned output aliases workspace storage and is overwritten by subsequent
calls. Clone it when ownership must outlive the next call. A workspace cannot
be used concurrently on independent streams. Autograd is unsupported. Warm up
before CUDA Graph capture to compile all kernels, then keep tensor addresses
stable during replay. The kernels read current input, routing, and weight
contents; capture does not freeze their values.

```python
from deep_gemm.small_moe import SmallMoE

workspace = SmallMoE(tokens, gate_up, down, topk)
workspace.validate_routes(counts, slots, weights)  # outside capture
y = workspace.forward(x, counts, slots, weights)
```

## Selection and fallback

Fusion is limited by a fixed shape table in `small_moe_policy.py`; it never
benchmarks or reads counts back to choose a kernel. The table does not assume
uniform routing or shared experts across tokens. Shapes outside the table use
ordinary dispatch, masked BF16 gate/up GEMM, routed SwiGLU, masked BF16 down
GEMM, and ordered combine. `enable_fusion=False` forces that same fallback.
Unsupported device, dtype, layout, and invalid metadata raise errors rather
than selecting an unrelated implementation.

Both expert counts E=64 and E=128 use this same table:

| D / I | T | Fused top-k |
|---|---:|---|
| 2048 / 768 | 1 | 1, 2, 4, 8 |
| 2048 / 768 | 2 | 1, 2, 4 |
| 2048 / 768 | 4 | 1, 2 |
| 4096 / 1536 | 1 | 1, 2, 4, 8 |
| 4096 / 1536 | 2 | 1, 2 |

All other shapes fall back for both segments. In particular D2048 T2/k8 and
T4/k4 were rejected in the earlier study under concentrated routing. D4096
T4/k1 had too little earlier calibration margin. This revision rechecks their
ordinary fallback; it does not promote them based on new fused measurements. These shapes remain outside fusion
even if a uniform-routing measurement alone looks favorable. A retained shape
needed at least a 3% calibration advantage in every measured expert-count and
routing-distribution combination before independent-seed validation.

The table is intentionally narrower than the native kernel's mechanical
capability. Its performance evidence applies to RTX 5090 and the recorded
CUDA/PyTorch environment, not every SM120 product or deployment workload.
See the accompanying validation report for the final measured ranges and
adjacent fallback cases. No original API's dispatch policy is changed.

## Architecture and arithmetic

L1 uses a CTA for one token, one N tile, and a small group of experts. Four
producer warps issue TMA; each math warp owns one expert's gate/up projection.
The experts share the original X row, eliminating packed-X dispatch and
materialized gate/up output. H retains the existing packed expert layout.

L2 uses a CTA for one token and one output N tile. Each math warp computes one
expert's down projection, rounds it to BF16 in shared memory, and the CTA
reduces top-k contributions in a fixed order. It writes Y directly without a
global partial-output buffer, ready counters, queues, or grid barriers.

Threads per CTA are `(experts_per_CTA + 4) * 32`. The L1 grid is
`T * (I / BN) * (k / experts_per_CTA)`; L2 is `T * (D / BN)`. BN, stages, and
expert grouping are frozen in the table. Full/empty K-stage barriers transfer
data ownership and are retained even when producer roles are inactive. The B
ring has its own 1 KiB aligned base so top-k 1/2 does not violate TMA swizzle
alignment. L2 adds a CTA barrier before reading BF16 expert contributions.

The numerical contract retains BF16 gate/up projection rounding, FP32 SwiGLU
with the routed Triton-compatible approximate sigmoid, BF16 H, BF16 down-result rounding,
separate FP32 multiply/add in top-k order, and BF16 Y. It does not move route
weights across nonlinearities or change the reduction tree. Correctness is
checked against the separate path, including bitwise H/Y, changed routes and
inputs after Graph capture, output guards, and a controlled dyadic oracle.
The shared MMA and both small fused epilogues inherit the accuracy repair
validated in revision `1c31df4`: zero-C K16 dots with explicit FP32 accumulation,
plus warp-cooperative TwoSum refinement near BF16 midpoints. Saved historical
L1 operands and independent short L2 midpoint/cancellation operands exercise
this path. Production arithmetic uses no FP64. This is measured numerical
validation, not a proof of universal correctly rounded GEMM.

The full-chain benchmark independently constructs gate/up and down dots in
FP64, rounds through FP32 to BF16 at the materialized boundaries, and checks H
and Y at the unchanged `rtol=0.01, atol=0.002`. Matching the separate kernel alone
is not used as an independent accuracy oracle.

## Reproduce checks

Build the extension from this revision and use a fresh or revision-matched JIT
cache. The validation runner records source and extension hashes, GPU identity,
commands, exit codes, and full logs. It rejects a skipped GPU run as evidence.

```bash
python -m tests.run_sm120_small_moe_validation --output results/small-moe-validation
python -m tests.bench_sm120_small_moe --jobs jobs.json --output results/chain.json --seeds 277 281
```

The four sanitizers exercise every tile/barrier configuration using a compact
E=8 fixture plus odd-shape fallback, custom capacity, and a nondefault stream.
The ordinary GPU suite and full-chain benchmark separately check E=64/128.
For native TMA-store scratch, the fixture poisons outputs with NaN before a
call and requires finite, fully overwritten results. A controlled dyadic
probe found that this toolchain's initcheck reports every read from a freshly
allocated native TMA-store output despite all 2048 columns matching the exact
oracle; initializing the same allocation to NaN or a sentinel removes that
tracking report and the output still matches. The fused kernels themselves
are checked with poisoned H/Y and guard regions. This fixture treatment adds
no initialization to production `forward` or performance measurements.
The benchmark times the whole data path with precomputed routing, not a sum of
segment timings. Allocation, compilation, and route planning are excluded
equally. Each native L1 BM is paired with each native L2 BM (0/32/64/128).
The benchmark also measures dispatch plus the public PR501 fused L1 and an
ordinary down/combine tail, and small fused L1 with an ordinary tail. All
controls use the same repaired MMA and midpoint epilogue. Earlier research
donor timings with different arithmetic are excluded from these results.
The fastest measured separate combination is a conservative comparison,
not a claim of a globally optimal baseline. GPU clocks and caches are unlocked.

## Repaired-arithmetic study, 2026-11-15

The fixed table was rechecked after the accuracy repair, using calibration seed
269 and independent holdout seeds 277/281. The calibration contains 144 scenes:
two D/I models, T=1/2/4, k=1/2/4/8, E=64/128, and uniform/concentrated/skew75
routing. Holdout order is shuffled and contains all 90 supported combinations
per seed, nine adjacent fallback combinations and three additional fallback
contracts (odd T/k, unknown D/I, and E=32).

All 15 predeclared supported D/I/T/k shapes retained at least 3% advantage in
calibration. Across 180 fused holdouts the geometric mean speedup over the
fastest of 16 repaired ordinary L1/L2 BM combinations is 1.937x, with a
1.049x minimum and 3.328x maximum. The ordinary controls cross BM0/32/64/128
independently for both layers. Public fused L1 and small fused L1 each get four
ordinary tail BM controls as additional attribution measurements.

The weakest retained range is D4096/I1536, T1/k8. Its L2/combine incremental
benefit over small fused L1 with the best ordinary tail falls to 1.009x in one
holdout; the full two-kernel chain is still 1.049x faster than the strongest
ordinary control. No broader fusion policy is inferred from this result.

The 24 fallback holdouts are 0.9979-1.0029x versus the identical fixed ordinary
configuration. Relative to each scene's fastest observed BM choice the range
is 0.9662-1.0029x. The fixed fallback is not claimed to choose every timing
winner, and no online autotuning is performed.

The original 42 grouped SwiGLU tests, 50 native tests (five historical
non-numerical deselections) and five expanded accuracy methods were rerun on
the new extension. The final small workspace suite has 13 methods, and each
of four sanitizers covers six compact methods. Saved historical L1 midpoint
operands and independently constructed L2 midpoint/cancellation dots pass.
A constructor-only guard against zero-size weights was added after holdouts;
its preceding source is retained. Timed forward/Triton AST and GPU sources
are unchanged, and final-source correctness, four sanitizers and seed-283
full-chain boundary checks are recorded separately.

See [the validation receipt](validation/sm120-small-moe-accuracy-20261115.json)
for source/extension hashes, per-scene timing controls and raw samples for the
selected and strongest comparison paths. Complete unrounded raw results are
sealed in the external results archive identified by SHA256 there. No NCU
counters were collected for this study. Measurements exclude allocation,
compilation and routing-plan construction, use reused weights and CUDA Graphs,
and keep GPU clocks unlocked; they are not end-to-end serving benchmarks.
