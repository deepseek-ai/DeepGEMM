# GPU Architecture Support Matrix and SM89 Porting Notes

DeepGEMM natively targets **SM89 (Ada Lovelace)**, **SM90 (Hopper)** and **SM100 (Blackwell)**.

Ada Lovelace is the **first NVIDIA architecture with FP8 Tensor Cores**
(`mma.sync.aligned.m16n8k32...e4m3`, which lowers to `QMMA.16832.F32.E4M3.E4M3`). SM89 therefore
gets a native FP8 GEMM, while Ampere (SM80) — which only has BF16 Tensor Cores — is **not** a
supported target for this library. BF16 GEMMs on SM89 are served by cuBLASLt, because Ada has no
WGMMA/UMMA and DeepGEMM does not ship an Ampere/Ada BF16 tensor-core kernel.

---

## Architecture Support Matrix

| Operation / Feature | SM80 (Ampere) | SM89 (Ada Lovelace) | SM90 (Hopper) | SM100 (Blackwell) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense FP8 GEMM (`fp8_gemm_*`)** | Unsupported (no FP8 Tensor Cores) | **Native SM89 Kernel** (1D1D + 1D2D) | Native SM90 Kernel (1D1D / 1D2D) | Native SM100 Kernel (FP8 / FP4) |
| **Dense BF16 GEMM (`bf16_gemm_*`)** | cuBLASLt (not a supported target) | cuBLASLt Fallback | Native SM90 Kernel / cuBLASLt | Native SM100 Kernel / cuBLASLt |
| **Microscaled FP4 GEMM (`fp4_gemm_*`)** | Unsupported (Blackwell Only) | Unsupported (Blackwell Only) | Unsupported (Blackwell Only) | Native SM100 FP4 Kernel |
| **Grouped FP8 GEMM (`m_grouped_fp8_*`)** | Unsupported | Unsupported (needs SM90+) | Native SM90 Kernel | Native SM100 Kernel |
| **K-grouped FP8 GEMM (`k_grouped_fp8_*`)** | Unsupported | Unsupported (needs SM90+) | Native SM90 Kernel | Native SM100 Kernel |
| **Grouped BF16 GEMM (`m_grouped_bf16_*`)** | Unsupported | Unsupported (needs SM90+) | Native SM90 Kernel | Native SM100 Kernel |
| **V3.2 MQA Logits (`fp8_fp4_mqa_logits`)** | Unsupported | Unsupported (needs SM90+) | Native SM90 Kernel | Native SM100 Kernel |
| **Mega MoE (`fp8_fp4_mega_moe`)** | Unsupported | Unsupported (needs SM90+) | Native SM90 Kernel | Native SM100 Kernel |
| **HyperConnection & Mega Gate / mHC** | Unsupported | Unsupported (needs SM90+) | Native SM90 Kernel | Native SM100 Kernel |
| **Programmatic Dependent Launch (PDL)** | Unsupported | Unsupported (needs `griddepcontrol`) | Supported | Supported |
| **FP32 scaling-factor layout kernels** | Supported | Supported | Supported | Supported |

Every unsupported combination raises an explicit `RuntimeError` naming the minimum supported
architecture, rather than silently producing wrong results. SM80 is deliberately not part of the
contract: only the SF layout helpers (pure CUDA, no Tensor Cores) happen to run there.

---

## SM89 (Ada Lovelace) Implementation Notes

Ada sits between Ampere and Hopper: it has **FP8 Tensor Cores** but none of the Hopper-only
data-movement machinery. The SM89 FP8 kernel is therefore built from the Ampere-era primitives
that Ada still exposes.

### What SM89 has, and what it does not

| Feature | SM89 | Consequence for DeepGEMM |
| :--- | :---: | :--- |
| `mma.sync.aligned.m16n8k32...e4m3` (`QMMA.16832`) | yes | FP8 Tensor Core GEMM is possible (first architecture with it) |
| `ldmatrix.sync.aligned.m8n8.x4` | yes | A/B fragments are fed from shared memory with `ldmatrix` |
| `cp.async` (`ca` / `cg`, with byte-granular zero fill) | yes | multi-stage shared-memory pipeline |
| `bar.sync` | yes | pipeline barriers (no async `mbarrier.try_wait`) |
| TMA (`cp.async.bulk.tensor`) | **no** | A/B are staged with `cp.async`; C/D is written with plain global stores |
| WGMMA / `tcgen05` / thread-block clusters | **no** | math uses warp-level `mma.sync` with register accumulators |
| `mbarrier.try_wait.parity` | **no** | `bar.sync` is used instead of async transaction barriers |
| `griddepcontrol` (PDL) | **no** | PDL is rejected on the host and compiled out on the device |
| `tensormap.replace` | **no** | no in-kernel tensor-map updates (K-grouping is unsupported anyway) |

### Kernel structure

* `BLOCK_K` is pinned to **128**, which is also the FP8 scaling granularity, so every MMA
  accumulator can be promoted with a single per-`(128-wide N block, 128-wide K block)` scale pair.
* Two SF recipes are supported, matching SM90 exactly:
  * **1D1D** — `(1, 1, 128)`: per-token scales for A *and* B, both staged in shared memory.
  * **1D2D** — `(1, 128, 128)` (the default whenever FP32 scales are used, same as SM90):
    per-token A scales staged in shared memory plus one uniform B scale per 128-wide N block
    read from global memory. Any `BLOCK_N` dividing 128 is legal.
* Shared memory uses a **128-byte XOR swizzle** (`chunk ^ (row & 7)`) over `BLOCK_M`/`BLOCK_N`
  rows of 128 bytes. This is exactly the layout `ldmatrix.x4` wants: for one 8-row matrix the
  eight 16-byte chunks land in eight distinct bank groups, so every fragment load is conflict free.
* Ragged tiles are handled by `cp.async`'s `src-size` operand: any of the 16 destination bytes
  may be dropped and is zero filled, which covers partial N rows, partial M rows and a K tail
  that is not a multiple of 128 without a separate epilogue pass.
* Shared memory is limited to 99 KiB per block on SM89 (`sharedMemPerBlockOptin == 101376`);
  `SM89ArchSpec` sizes the pipeline to that budget and requires at least two stages.
* Warps are tiled by `get_mma_sync_warps_n()`, a `constexpr` helper shared by the host
  heuristics (`csrc/jit_kernels/heuristics/sm89.hpp`) and the JIT'd device code
  (`deep_gemm/include/deep_gemm/impls/sm89_fp8_gemm.cuh`), so both always agree on the split.
  It keeps the per-warp accumulator at 64 registers and prefers the most square warp tile.
* The block scheduler is the same `sched::Scheduler` persistent loop used by SM90/SM100, so
  results are bitwise deterministic regardless of how blocks are distributed over SMs.
* The next stage's `cp.async` copies are issued *after* the MMA block so their address
  registers are not live across the math; this keeps the kernel under the 255-register
  ceiling with only minor spilling.

### Source map

| File | Contents |
| :--- | :--- |
| `deep_gemm/include/deep_gemm/mma/sm89.cuh` | `mma.sync.aligned.m16n8k32` FP8 Tensor Core wrapper + fragment layout documentation |
| `deep_gemm/include/deep_gemm/impls/sm89_fp8_gemm.cuh` | The SM89 FP8 GEMM kernel |
| `csrc/jit_kernels/heuristics/sm89.hpp` | `SM89ArchSpec`: block sizes, pipeline depth, warp split, cost model |
| `csrc/jit_kernels/impls/sm89_fp8_gemm.hpp` | JIT host wrapper (SF layout checks, launch configuration) |
| `csrc/apis/gemm.hpp` | dispatch of `fp8_fp4_gemm_nt` into the SM89 kernel |
| `deep_gemm/testing/utils.py` | `is_sm89()` / `get_arch_pair()` helpers used by the test generators |

### Requirements and limitations

* A and B must both be **K-major** (the NT layout), exactly like on SM90. `fp8_fp4_gemm_nn`,
  `fp8_fp4_gemm_tn` and `fp8_fp4_gemm_tt` therefore raise an assertion for FP8 operands.
* `k` must be a multiple of 16 so that `cp.async` can move 16 bytes per row.
* M and N are unconstrained: ragged tiles are zero filled and the epilogue is bounds checked.
* Accumulation (`D = C + A @ B`) is supported for both FP32 and BF16 outputs and follows the
  library's usual contract, i.e. the add happens in the dtype of D.
* `alpha` scaling and custom epilogues (`deep_gemm.epilogue`) require SM100.
* Only dense (non-grouped) FP8 GEMMs are implemented; MoE/grouped kernels need SM90+.

### Measured performance

On an RTX 4060 Laptop (SM89, 24 SMs, 99 KiB shared memory per SM), FP8 GEMM with per-128-channel
scales:

| Shape (M x N x K) | DeepGEMM `sm89_fp8_gemm` | cuBLASLt FP8 |
| :--- | ---: | ---: |
| 128 x 4096 x 4096 | ~135 us (32 TFLOPS) | ~110 us (39 TFLOPS) |
| 1024 x 4096 x 4096 | ~960 us (36 TFLOPS) | ~780 us (44 TFLOPS) |
| 4096 x 4096 x 4096 | ~3.7 ms (37 TFLOPS) | ~3.1 ms (45 TFLOPS) |
| 4096 x 8192 x 4096 | ~6.4 ms (43 TFLOPS) | ~5.3 ms (52 TFLOPS) |

i.e. ~85 % of cuBLASLt, which is a reasonable target given that Ada has no TMA (no asynchronous
tensor copies, no `tensormap.replace`) while cuBLASLt ships hand-tuned SASS for the same
architecture.

---

## SM80 (Ampere)

SM80 is **not** a supported target: it has no FP8 Tensor Cores, so none of the low-precision
kernels can run, and DeepGEMM does not ship an Ampere/Ada BF16 tensor-core kernel either. The
generic `arch_major == 8` branches that remain (for example the cuBLASLt BF16 fallback) are
there so that library code degrades gracefully instead of crashing, but nothing about SM80 is
tested or documented as supported.

> [!NOTE]
> The cuBLASLt path selects `CUBLAS_COMPUTE_32F` (never `CUBLAS_COMPUTE_32F_FAST_TF32`) for
> BF16/FP8 operands, so the fallback does not silently run at TF32 precision.

---

## Running the Validation Test

```bash
python3 -m unittest tests/test_sm89.py
```

`tests/test_sm89.py` skips itself on non-SM89 devices, so it is safe to run from a
cross-architecture CI matrix. It covers:

1. GPU compute capability detection.
2. Native FP8 GEMM correctness against an FP32 reference for both SF recipes, both output
   dtypes (BF16/FP32), with and without accumulation, and across eight shapes including
   ragged M and N, decode-sized M and multi-wave sizes.
3. Explicit `recipe`, `recipe_a`/`recipe_b`, `compiled_dims` and `fp8`/`fp4` alias handling.
4. Bitwise determinism across repeated launches.
5. `D = C + A @ B` consistency with the same workload accumulated from a zero FP32 `C`.
6. BF16 GEMMs in NT/NN/TN/TT layouts and the cuBLASLt fallback paths.
7. The FP32 scaling-factor layout kernels.
8. Explicit, descriptive errors for FP4, grouped GEMMs, MN-major FP8 operands, PDL and
   MQA logits.

The shapes in the suite are deliberately small enough to fit in 8 GB of VRAM while still
spanning more than one wave of the 24 SMs of the reference device.