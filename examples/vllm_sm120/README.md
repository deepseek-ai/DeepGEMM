# Opt-in SM120 single-token vLLM adapter

This is an experimental out-of-tree plugin for **vLLM 0.32.0**, not a replacement default backend. It uses the measured DeepGEMM private BF16 APIs. The entry point does nothing unless `SM120_VLLM_ENABLE=1`.

## Scope

One active decode token, RTX 5090 with 170 SMs, contiguous BF16 unquantized weights, supported SiLU gate/up layout and unique valid top-k experts. Routing and shared-expert order stay in vLLM. Unsupported batch/shape/layout/scheduling retains the native path. Quantization, asynchronous scheduling, DP/EP, DBO/EPLB, LoRA/bias, custom activation/scaling and independent concurrent forwards are outside the measured scope.

L1 directly reads X and computes gate/up GEMM + SwiGLU into packed BF16 H. L2 computes down GEMM + ordered CTA-local combine. TP2 shards the intermediate channels of gate and up consistently; down consumes the matching shard. vLLM owns the outer TP all-reduce. No custom distributed communication is added.

## Installation and measured settings

First build this candidate DeepGEMM source with its pinned submodules in a compatible Linux CUDA environment; use the repository's build instructions and force a source build if a cached wheel could be selected. The review source export does not contain submodule content or the measured binary. Then install the plugin in the same environment:

```bash
python -m pip install --no-deps --no-build-isolation ./examples/vllm_sm120
```

For the measured **PP2** run (TP1, local I=768):

```bash
export VLLM_PLUGINS=deepgemm_sm120
export SM120_VLLM_ENABLE=1
export SM120_VLLM_TP_ENABLE=0
export SM120_VLLM_DIRECT_IDS=0
```

The selected public plan is L1 `(32,3,2)`, L2 `(16,4,8)` for D2048/I768/E128/k8.

For the measured **TP2** run (PP1, local I=384):

```bash
export VLLM_PLUGINS=deepgemm_sm120
export SM120_VLLM_ENABLE=1
export SM120_VLLM_TP_ENABLE=1
export SM120_VLLM_DIRECT_IDS=1
export SM120_TP_L1=16,4,2
export SM120_TP_L2=16,2,8
```

Triples are BN/stages/experts-per-CTA. **Set both TP triples explicitly**: unset-variable source defaults are `(16,3,2)` and `(16,3,8)`, which are not the measured plan above. Selection is a private exact-shape override for `(T,E,k,D,I,tp)=(1,128,8,2048,384,2)` and leaves the public single-token table unchanged. Arbitrary environment overrides passing resource checks are not performance-validated plans.

For native controls, start a new process with `SM120_VLLM_ENABLE=0`. Keep all other runtime/model settings matched. The measurements used raw BF16 Qwen3-Coder-30B-A3B-Instruct, two RTX 5090s, serial requests, `async_scheduling=False`, compilation mode 0 and `FULL_DECODE_ONLY` Graph with capture size 1. Use the sealed harness for full reproduction, including model identity, warmup and native controls; these exports alone are not a complete benchmark command.

## Workspace and routes

Each layer owns H `[E,128,I]` and Y `[1,D]`; output aliases its reusable workspace. Slot mode launches one preparation kernel plus two fused kernels. Direct-ID mode launches two fused kernels. Direct IDs can be int32 or int64; valid experts must be unique. Invalid IDs use reserved padding rows and their contributions are skipped, including sentinel weights. This is a caller contract, not host-side validation of every route value. Python counters measure dispatch/capture, not Graph replay counts.

## Evidence and limits

Core numerical and sanitizer evidence, equal-output timing, and final three-domain NLL evidence are separate checks. The preregistered final run passed its log(1.01) NLL engineering bound, but top-1 agreement is 98.50% PP2 / 98.39% TP2 and free generation can diverge. No semantic-equivalence, task-accuracy, throughput, batch>1 or universal superiority claim is made. The adapter remains opt-in.

See `../../docs/sm120-final-review-scope.md` for exact source identities, evidence paths, preserved failures and the current review boundary. `sm120_vllm_validation.py` is a diagnostic helper; it is not a replacement for the sealed full-model acceptance harness.
