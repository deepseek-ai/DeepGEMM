# Single-token integration and revision-matched acceptance

This extension is based on PR501 head
`87bb4fcc15d4ced85b91649392bc45b727355cf1`. It retains the zero-C K16
Tensor Core dots, explicit FP32 accumulation, cooperative midpoint refinement,
BF16 projection/H/partial boundaries, and ordered FP32 multiply/add combine.
Production uses no FP64; tests use FP64 reference dots at the original
`rtol=0.01, atol=0.002`. This is numerical coverage, not a universal
correct-rounding proof.

## Public entry

```python
from deep_gemm.small_moe import SmallMoE

workspace = SmallMoE(
    1, gate_up, down, topk,
    fusion_policy='single_token',
)
workspace.validate_routes(counts, slots, weights)  # optional value audit, outside capture
y = workspace.forward(x, counts, slots, weights)
print(workspace.config)  # frozen configuration or the reason for fallback
```

The existing default `fusion_policy='validated'` is unchanged. The new policy
uses a separately audited single-token table, requires RTX5090/170 SMs, and
falls back for multi-token requests and unknown dimension/expert/top-k keys.
`enable_fusion=False` forces the ordinary implementation. Unsupported dtype,
layout, device architecture and malformed metadata retain the existing errors.
No online timing, host route readback or route-value inspection is added to
forward. Compilation/allocation are outside Graph capture as before.

Each token must select distinct experts. If A is the active expert count and
rho=T*k/A is the mean rows per active expert, at T1 A=k and rho=1 are known
without a histogram readback. The policy does not extrapolate that property
to T>1. Counts, slots and weights must satisfy the existing workspace
contract; unchecked malformed values are unsupported.

The independently accepted table is:

| Shape label | D | I | E | top-k | L1 BN/stages/bundle | L2 BN/stages/bundle |
|---|---:|---:|---:|---:|---|---|
| Qwen3-30B-A3B | 2048 | 768 | 128 | 8 | 32/3/2 | 16/4/8 |
| Qwen3-235B-A22B | 4096 | 1536 | 128 | 8 | 16/2/8 | 32/2/8 |
| Qwen1.5-MoE | 2048 | 1408 | 60 | 4 | 16/3/4 | 16/4/4 |
| DeepSeek-V2-Lite | 2048 | 1408 | 64 | 6 | 16/3/2 | 16/4/6 |

These are dimension labels, not evidence of full-model execution. Qwen2's
3584/2560/E64/k8 candidate failed pressure acceptance and stays ordinary.
This opt-in extension does not supersede the old fused configuration in
every condition: Qwen3-30B's legacy fusion was faster in the warm pilot,
while the new plan was faster under pressure.

For either layer, one math warp owns one expert and four producer warps
feed the TMA ring. Let b be bundle, s stages and n BN. Threads per CTA are
32(b+4). L1 grid is T*(I/n)*(k/b); L2 grid is T*(D/n), because L2 requires
b=k to keep all ordered contributions within one CTA. More CTAs than SMs
are legal and execute in waves; grid size is not an occupancy guarantee.
The K tile is 64 BF16 elements; the instruction dot is K16.

The requested shared-memory bytes follow the launch code exactly:

```text
A = 128*s*(1 for L1, b for L2)
offset_B = round_up(A, 1024)
smem = offset_B + 128*s*b*n*(2 for L1, 1 for L2) + 16*s
       + (0 for L1, 2*k*n for L2)
```

| Shape label | L1 CTA count/threads/smem bytes | L2 CTA count/threads/smem bytes |
|---|---|---|
| Qwen3-30B | 96 / 192 / 50224 | 128 / 384 / 69952 |
| Qwen3-235B | 96 / 384 / 66592 | 128 / 384 / 68128 |
| Qwen1.5 | 88 / 256 / 50224 | 128 / 256 / 35008 |
| DeepSeek-V2-Lite | 264 / 192 / 25648 | 128 / 320 / 52480 |

Registers and active CTAs depend on compilation and cannot be inferred from
this shared-memory formula alone. Selection combines legal resource geometry
with measured acceptance; it is not a universal analytic latency predictor.

## Why policy24 timings cannot establish this extension's speed

The policy24 experiment used a historical small-kernel body without the PR's
midpoint refinement. Its choices are parameter candidates, not the acceptance
receipts for this implementation. The repaired public C++ API and JIT headers
are rebuilt and tested here. The original arithmetic is never swapped out to
recover those historical numbers. Both the inherited and new study are kept.

## Calibration and independent validation

The proposed BN/stage/expert-group configurations are fixed before this
study. Calibration uses seeds25011/25029, 15 randomized interleaved rounds,
warm repeated Graph and 512MiB pressure conditions. Controls include all
16 ordinary M-tile pairs and public grouped SwiGLU with four ordinary tails.
All are the same repaired revision. The original configured small path is
also measured where supported.

A candidate must beat the fastest observed ordinary/public control by at
least3% in both conditions and both calibration seeds; the paired-round
bootstrap one-sided95% ratio upper bound must be below0.97. Selection and
source hashes are frozen before seeds25107/25223, each with31 rounds.
The independent seeds audit all controls again and can reject, never add,
the calibration support set. Family-wise multiple-comparison control is not
claimed. Cross-driver/day repeatability is outside this single-card session.

An added test adapter measures actual `SmallMoE.forward`, not just native
launches. A final revision then reruns inherited grouped SwiGLU contracts,
midpoint regressions, Graph/guard tests, and all four sanitizers. New
geometries include stage4 and top-k6.

## Real-data replay

Reuse the sealed Qwen3-30B-A3B capture: layers0/4/7, eight texts,32
teacher-forced decode positions. Trace SHA256:
`3b534da5ff7fa08ec728050c0498a36931ead73e6eced42510ff946ce9636dd3`.
Every required BF16 gate/up/down tensor is verified against the recorded
tensor hash before use. Actual post-attention inputs, selected experts and
route weights are replayed without resampling.

The entry and ordinary materialized path must agree bitwise for H/Y. Selected
samples additionally use independent FP64 dots with the materialization
contract. Historical HF outputs are recorded separately because activation,
reduction and projection implementation can differ; those differences must
not be hidden by relabeling the contract.

Two timing boundaries are reported separately:

- Steady single-call X+existing GPU routing toY, excluding router/planning,
  allocation and JIT. Report raw samples, median andP95 of sample latency.
- Recorded32-call sequence with GPU copies into stable input/route buffers
  plus the expert branch. Values are per-call averages of the sequence;
  their percentiles are not individual-request P95.

This is real single-layer expert replay, not a full-model or online serving
throughput claim. It does not validate new real-model families.

## Reproduction

Required test dependencies include PyTorch, Triton, pytest and safetensors.
The real replay used safetensors 0.6.2 in an isolated `replay-deps` directory.
Check these imports before starting GPU work. Native sources and the compiled
extension are hashed; `record_single_token_provenance.py` verifies that a reused
extension's native source tree is byte-identical to its built donor tree.

```bash
python -m tests.bench_sm120_single_token \
  --seeds 25011 25029 --rounds 15 --output calibration.json
python scripts/freeze_single_token_plan.py \
  --input calibration.json --output frozen-plan.json
python -m tests.bench_sm120_single_token \
  --seeds 25107 25223 --rounds 31 --output holdout.json
python scripts/audit_single_token_holdout.py \
  --plan frozen-plan.json --holdout holdout.json --output accepted-plan.json
python -m tests.run_sm120_single_token_validation --output final-validation
python -m tests.bench_sm120_single_token_entry \
  --seeds 25301 25309 --rounds 31 --output final-entry.json
python -m tests.replay_sm120_single_token \
  --data /root/sm120-work/expert-pair11 --plan frozen-plan.json \
  --rounds 31 --output real-replay.json
```

The support table was populated after the independent acceptance gate.
Final public-entry validation and real replay remain separate required gates;
their revision-specific outcomes belong in the collected report.

The final1 run initially stopped before loading real data because safetensors
was absent. Its nonzero launcher exit and original log are retained. The
validation and final-entry stages had already passed. Only real replay was
restarted after installing the pinned dependency into an isolated directory;
the production, correctness, benchmark and replay source bodies were unchanged.
