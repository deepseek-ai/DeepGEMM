# SM120 masked grouped BF16 GEMM with SwiGLU

`m_grouped_bf16_swiglu_nt_masked(a, b, d, masked_m, expected_m)` computes the
gate and up projections using the existing SM120 Tensor Core pipeline and applies
SwiGLU before writing the output. It eliminates the materialized gate/up tensor
and a separate activation launch. A later down projection remains a separate GEMM.

## Contract

- SM120; contiguous CUDA BF16 inputs/output on one device.
- `a`: `[E, M, K]`; `b`: `[E, 2 * N, K]`, in `[gate | up]` order;
  `d`: `[E, M, N]`.
- `masked_m`: contiguous one-dimensional CUDA int32 tensor with E entries.
  The caller must keep every count in `[0, M]`; counts are not read back by the host.
- `expected_m` is a positive scheduling hint no larger than M. For all-empty
  experts, pass a positive hint such as 1. It does not replace device counts.
- A/B retain the native SM120 TMA layout/alignment requirements. Counts and input
  shapes are fixed-size buffers; their values can change between Graph replays.
- As with native grouped GEMM, runtime M/K alignment 32 requires output N > 32;
  use alignment 64 or 128 for smaller N.
- Output rows at or above each expert's count are left untouched. No accumulation,
  routing, weighted combine, communication, or training backward is provided.
- Input, weights, output, and count storage must not overlap.

For each valid row, the numerical contract is:

```python
gate = (a.float() @ gate_weight.float().T).bfloat16().float()
up = (a.float() @ up_weight.float().T).bfloat16().float()
d = (torch.nn.functional.silu(gate) * up).bfloat16()
```

The explicit BF16 rounding of the projections preserves the materialized BF16
pipeline's semantics. This does not promise bitwise equality to every GEMM
specialization or a real-number matrix product. Tests compare to the materialized
pipeline and include an independently computed, exactly representable dot-product
fixture. The random-input FP64 test currently fails for both ordinary and fused
paths on RTX 5090. Independent exact-dot diagnostics locate projection differences
near BF16 rounding boundaries. This failure is retained; no test threshold has
been relaxed. See the [validation results](sm120-grouped-bf16-swiglu-validation.md)
for the evidence and the remaining numerical-contract question.

## Validation

```bash
python -m pytest -q tests/test_sm120_grouped_swiglu.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest -q tests/test_sm120_grouped_swiglu.py -k 'materialized or graph'
python experiments/sm120_grouped_swiglu_native.py --rows 16 16 16 16
python experiments/sm120_grouped_swiglu_native.py --rows 128 128 128 128
python -m pytest -q tests/test_sm120_bf16_gemm.py tests/test_sm120_bf16_odd_n.py
```

The benchmark reports L1 plus activation only, using CUDA Graphs and reused
weights. It does not measure down projection, routing, communication, or serving
latency. Ordinary GEMM has the existing default specialization; the new template
argument is emitted only for fused calls.
Use `--output timing.json` to retain full precision timing samples. For Nsight
Compute, `--profile-path grouped_fused` (or `grouped_plus_swiglu`) brackets one
validated invocation with the CUDA profiler API; pass `--profile-from-start off`
to Nsight Compute. These captures do not produce ordinary timing samples.

The fused epilogue packs two adjacent BF16 results into a 32-bit store when the
output address is four-byte aligned and both columns are valid. Odd-N tails and
two-byte-offset output views use scalar stores. Projection rounding and activation
are identical in these branches.

Status: compiled and measured on RTX 5090, with a retained random-input FP64
failure. This is still a draft, not an all-tests-passing contribution. Detailed
test counts, measured speedups, regressions and known BK32 spilling are recorded
in the [validation results](sm120-grouped-bf16-swiglu-validation.md). Measurements
from RTX PRO 6000 or the broader research prototype are not used as validation
of this patch.
