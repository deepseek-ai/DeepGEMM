# PR501 follow-up description

Single-token MoE inputs have one useful row per selected expert, while the
ordinary materialized path dispatches padded inputs and launches separate
activation/combine kernels. This adds an explicit `fusion_policy='single_token'`
to the existing inference workspace, reusing the repaired Tensor Core L1/SwiGLU
and L2/combine building blocks with a bounded configuration table. It does not
change the existing default policy.

The new policy is limited to RTX 5090 with 170 SMs and four audited D/I/E/top-k
keys. Unknown keys, multi-token requests, and a rejected Qwen2 pressure case
use ordinary kernels. Selection inspects shape metadata during construction;
forward adds no route readback, online tuning or allocation. Native shape
support expands to the candidate geometries without changing the repaired
MMA, midpoint refinement or ordered combine arithmetic.

Calibration and independent seeds compare the same repaired revision against
16 ordinary tile pairs and four public grouped-SwiGLU tails, under warm Graph
and 512 MiB interference conditions. Historical policy24 timings came from a
kernel without the PR's midpoint refinement and are not acceptance evidence
for this change. New configurations also do not universally beat legacy
fusion; the warm Qwen3-30B pilot favored the legacy configuration.

The final revision passed 30 public correctness tests, 42 inherited contract
tests, four accuracy regressions and all four sanitizers (eight compact test
methods per tool, racecheck with zero warnings). Two additional seeds measured
the actual public forward against all controls: four accepted shapes reduced
warm latency by 13–49% and interference latency by 8–33%.

Sealed Qwen3 layers 0/4/7 replayed 768 actual single-token inputs. All public
H/Y outputs matched ordinary materialization bitwise; 27 selected inputs passed
the independent FP64 oracle at the unchanged tolerances. All 768 outputs also
met the original tolerance against the recorded historical HF local outputs.
This is one real model's single-layer evidence, not full-model or serving
validation. See [the exact results](sm120-single-token-results.md) for controls,
warm/pressure differences, legacy comparison and reproducibility limits.
