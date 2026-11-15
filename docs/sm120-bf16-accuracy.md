# SM120 BF16 projection accuracy

The original strict test checks a BF16 materialized gate/up projection followed
by FP32 SwiGLU against independently computed FP64 dots, with the existing
BF16 conversion between projection and activation. Its tolerances remain
`rtol=1e-2, atol=2e-3`. The original seeds and selected experts are unchanged.

## Failure mechanism

A projection just above or below a BF16 rounding midpoint can round to the
opposite BF16 neighbor after a small FP32 accumulation error. SwiGLU can
amplify that one-step projection difference enough to exceed the original
tolerance. The failure was present in the materialized implementation as well
as the fused implementation; moving the activation into the epilogue did not
create the initial projection discrepancy.

Two saved historical dots have exact values `1.0664084367454052` and
`1.1601590558548196`, respectively about `2.19e-6` and `2.81e-6` above BF16
rounding midpoints. The old outputs chose the lower BF16 values. Computing
short tensor-core dots into zero and accumulating them explicitly in FP32
fixes these cases and the original seeds, but an expanded seed sweep found
another midpoint failure. Ordinary compensated FP32 summation alone did not
remove it.

The original PyTorch reference's conversion also matters: on the tested
environment, FP64 tensors convert to BF16 through FP32. For example the exact
dot `68943869 / 67108864 = 1.0273437052965164` produces BF16 `1.03125` under
that conversion. A direct single-round conversion to BF16 would produce
`1.0234375`. This repair preserves the existing conversion, rather than
silently redefining the oracle.

## Implementation

The BF16 MMA helper computes each 16-wide dot with zero C, then adds the
partial result using explicit `__fadd_rn`. The ordinary masked grouped BF16
and grouped SwiGLU epilogues additionally inspect the FP32 result's distance
from a BF16 rounding midpoint. A value within 32 FP32 representable steps of
the midpoint is recomputed from the current original BF16 operands using two
FP32 components: a running sum and its rounding residual, maintained with
explicit TwoSum operations.
All 32 lanes in a math warp cooperate on that rare dot: K loads are coalesced,
the sum/residual pairs are reduced with warp shuffles, and the owning lane
receives their FP32 sum before the existing BF16 conversion. BF16 products
are exactly representable in FP32 when their exponent is in the normal FP32
range. Retaining each addition's residual substantially reduces summation
error without using FP64 in production. Normal outputs keep the tensor-core
result.

The midpoint test stays inline, while the rare recomputation body is a
`__noinline__` device helper shared by its call sites. Its call is guarded by
the full-warp ballot: either all 32 lanes call with the same nonzero pending
mask, or all lanes skip it. The TwoSum operations, reduction order, midpoint
window and projection conversion are unchanged. This reduces repeated code
in the epilogue without adding a kernel launch or synchronization protocol.

The refinement is part of the same kernel and uses no scratch allocation,
host readback, extra launch, CTA barrier, ready counter, or global queue. Every
math lane participates in the ballots and shuffles, including lanes whose
output positions are outside the valid row or column range. Original input
row strides are passed explicitly for ordinary grouped GEMMs.

This is a practical accuracy refinement for the measured workloads. The
32-step window is not a mathematical guarantee of correctly rounded GEMM for
arbitrary K, exponent ranges, cancellation patterns, or nonfinite operands.
The residual sum itself is FP32; this is not an arbitrary-precision expansion.
FP64 remains an independent reference. An earlier local FP64 refinement passed
the numerical and sanitizer checks but cost more; its receipts are retained
separately. Historical evidence is preserved and must not be relabeled as
having passed.

## Reproduction

Rebuild the extension after the host launch-argument change. An incremental
`setup.py build_ext` can reuse an object compiled before an included C++ header
changed; use `python setup.py build_ext --inplace --force` for this revision.
Use a revision-matched JIT cache.

```bash
python -m unittest tests.test_sm120_bf16_accumulation -v
python -m pytest tests/test_sm120_grouped_swiglu.py -v
python -m unittest tests.test_sm120_bf16_all_experts -v
python -m tests.run_sm120_bf16_accuracy_validation --output results/accuracy-validation
```

The added regression preserves the saved midpoint operands, checks expected-M
hints 1/33/65,
ordinary FP32 output and Graph replay, tests the discovered seed-2 case, and
extends the unchanged original 48-expert strict comparison to seeds 2 through
31. The original two seeds are run separately. The validation runner runs the
unchanged original suite separately from the
expanded seed tests, and runs memcheck, synccheck, racecheck, and initcheck on
compact fixtures that exercise midpoint refinement, odd shapes, output guards,
and changed input/counts under Graph replay on a nondefault stream. A separate
all-expert test extends the original seed-0/1 oracle to all 48 experts for both
ordinary and fused paths, retaining the generator and tolerance.
