# DeepGEMM

DeepGEMM is a unified, high-performance tensor core kernel library that brings together the key computation primitives of modern large language models — GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more — into a single, cohesive CUDA codebase. All kernels are compiled at runtime through DeepJIT, requiring no CUDA compilation during installation.

DeepGEMM leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), but avoids heavy reliance on their templates or algebras. The library is designed for simplicity, with only a limited number of core kernel functions, making it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## News

- 2026.09.10: Sparse Indexer, Mega Gate, Mega mHC, DeepJIT, MoE and Indexer optimizations and more.
    - Please see [#432](https://github.com/deepseek-ai/DeepGEMM/pull/432) for more details.
- 2026.04.16: Mega MoE, FP8xFP4 GEMM, FP4 Indexer, PDL, faster JIT compilation and more.
    - Please see [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) for more details.
    - For Mega MoE benchmarks, refer to [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316).
- 2025.09.28: DeepGEMM now supports scoring kernels (weighted ReLU MQA logits) for the lightning indexer for DeepSeek v3.2.
    - Please see [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200) for more details.
- 2025.07.20: DeepGEMM now supports both SM90/SM100, and has a full refactor with a low-CPU-overhead JIT CPP module.
    - As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be no longer supported.
    - Please see [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112) for more details.
- 2025.05.14: DeepGEMM now offers weight gradient kernels for dense and MoE backward! See [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95) for details.
- 2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800! See [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74), [#78](https://github.com/deepseek-ai/DeepGEMM/pull/78), [#81](https://github.com/deepseek-ai/DeepGEMM/pull/81), [#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) and [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526) for details.

## Quick start

### Requirements

- NVIDIA SM90, SM100, or SM120 architecture GPU (kernel coverage differs; see below)
- Python 3.8 or higher
- Compilers and standard libraries with C++20 `<format>` support
- CUDA Toolkit 12.9 or higher; SM120 requires a compiler supporting the `sm_120f` target
- PyTorch 2.3 or higher
- CUTLASS 4.0 or higher (could be cloned by Git submodule)

### SM120 coverage

SM120 uses the current Python APIs and DeepJIT, targeting `compute_120f` / `sm_120f`. Execution has been tested on SM120 with CUDA 13.2; SM121 and other hardware are not validated by those results.

- **GEMM:** dense BF16, FP8, FP4 and mixed FP8/FP4; M-grouped contiguous and masked variants; K-grouped BF16 and FP8. Dense layout aliases may copy inputs into the native K-major layout. K-grouped FP4 NT is not implemented.
- **Einsum:** all three FP8 expressions (`bhr,hdr->bhd`, `bhd,hdr->bhr`, `bhd,bhr->hdr`) produce BF16 or FP32, not quantized FP8 output. BF16 has native `bhr,hdr->bhd` and `bhd,hdr->bhr` paths, plus the `bmk,bnk->mn` batch reduction. The first two may use cuBLASLt when available unless deterministic algorithms are requested; BF16 `bhd,bhr->hdr` remains a cuBLASLt path.
- **HyperConnection:** `tf32_hc_prenorm_gemm` accepts BF16 A and FP32 B, with FP32 output and square sums; `N <= 128`, `N % 8 == 0`, and `K % 64 == 0`.
- **Dense and paged MQA:** FP8 with `q_sf=None`, or MXFP4 with packed UE8M0 Q/KV scales; 16, 32 or 64 heads. FP8 head dimensions are 32, 64 or 128; MXFP4 requires 128. Weights are FP32 and logits may be BF16 or FP32. FP8 page sizes are 64, 128 or 256; MXFP4 also supports 32. Non-paged logits can be cleaned; paged `clean_logits=True` is unsupported. SM100-only dense scheduling metadata and dense/paged MXFP8 mode are not supported.
- **Sparse MQA:** MXFP8 and MXFP4, including paged variants, use the current sparse metadata APIs with packed UE8M0 scales and BF16 weights/logits. These are distinct from dense MQA; see the [scaling-factor contract](docs/scaling-factor-format.md).

This coverage does not imply SM120 support for Mega MoE, Mega Gate or Mega mHC. Shape, alignment and layout checks still apply to each API.

Run the focused suites through pytest, rather than relying on the original scripts' `__main__` blocks to execute every test:

```bash
python -m pytest -q tests/test_sm120_gemm.py tests/test_sm120_attention.py
python -m pytest -q tests/test_sm120_*.py tests/test_bf16.py tests/test_fp8_fp4.py tests/test_attention.py -k sm120
python -m pytest -q tests/test_pack_ue8m0.py
```

### Development

```bash
# Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Link some essential includes and build the C++ extension
cat develop.sh
./develop.sh
```

### Installation

```bash
cat install.sh
./install.sh
```

Then, import `deep_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices

This library provides optimized GEMM kernels for NVIDIA GPUs with a naming convention: `D = C + A @ B`. The input shape layout is NT (non-transposed A, transposed B). While the SM90 implementation supports only the NT memory layout (row-major, col-major), the SM100 implementation supports all memory layouts (NT, TN, NN, TT). For example, `fp8_gemm_nt` will do a `D = C + A @ B.T`

The native GEMM scaling-factor layout is TMA-aligned and MN-major, but the compute format differs by architecture:

- SM90 uses scaling factors in FP32 format.
- SM100 and SM120 use packed [UE8M0](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats), which packs 4 UE8M0 values into a single `torch.int`. APIs can transform untransformed FP32 scales automatically; see [Scaling Factor Format](docs/scaling-factor-format.md) for recipes and pre-packed layout requirements.

FP8 casting and input-layout preparation are generally the caller's responsibility and can be fused into prior kernels. SM120 layout aliases and some einsum paths can transpose or copy inputs internally, but those copies add overhead; supplying the native layout avoids them where supported. The library's simple PyTorch utilities may also be slower than fused preparation; our primary focus is optimizing the GEMM kernels themselves.

#### Normal dense GEMMs (non-grouped)

To perform a basic non-grouped FP8 GEMM, call the `fp8_gemm_{nt, nn, tn, tt}` function. For more details, please refer to the function documentation.

#### Grouped GEMMs (contiguous layout)

Unlike traditional grouped GEMMs in CUTLASS, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape. For training forward passes or inference prefilling, where each expert may process a varying number of tokens, we concatenate these tokens into a single tensor, referred to as the "contiguous" layout. Note that each expert segment must be aligned to the GEMM M block size (`get_mk_alignment_for_contiguous_layout()`).  For more information, please refer to the `m_grouped_fp8_gemm_{nt, nn}_contiguous` function documentation.

We also provide a K-axis-grouped API for MoE weight backward (with M and N must remain fixed), please refer to `k_grouped_fp8_gemm_tn_contiguous` for more information.

#### Grouped GEMMs (masked layout)

During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions.

Use `m_grouped_fp8_gemm_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.

#### V3.2 MQA kernels for the indexer

The kernel family has two versions, non-paged (for prefilling) and paged (for decoding).
Take the non-paged version `fp8_mqa_logits` as an example. It has 6 inputs:

- `q`, E4M3 tensor with shape `[seq_len, num_heads, head_dim]`
- `kv`, E4M3 tensor (shaped as `[seq_len_kv, head_dim]`) with float SF (shaped as `[seq_len_kv]`)
- `weights`, float tensor with shape `[seq_len, num_heads]`
- `cu_seq_len_k_start` and `cu_seq_len_k_end`, int tensor with shape `[seq_len]`
- `clean_logits`, whether to clean the unfilled logits into `-inf`

The output tensor is shaped as `[seq_len, seq_len_kv]`, indicating token-to-token logits.
For each token `i` in `q`, it will iterate all tokens `j` from `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])`,
and calculate the logit `out[i, j]` as:

```python
kv_j = kv[0][j, :] * kv[1][j].unsqueeze(1)  # [head_dim]
out_ij = q[i, :, :] @ kv_j  # [num_heads]
out_ij = out_ij.relu() * weights[i, :]  # [num_heads]
out_ij = out_ij.sum()  # Scalar
```

For more details and the paged version `fp8_paged_mqa_logits`, please refer to `tests/test_attention.py`.

The five restored features below (SM100 FP16-weight MQA, SM90 paged MQA, SM90 FP8 Mega MoE, SM100 NVFP4 Mega MoE and SiTU) target `nv_dev`, not `main`. They use the current public Python APIs and DeepJIT, not a separate legacy runtime:

- **SM100 dense MQA with FP16 weights:** `fp8_fp4_mqa_logits` accepts `torch.float16` weights with FP8 Q/KV, `q_sf=None` and one FP32 KV scale per token. `seq_len` must be positive and divisible by 4, `seq_len_kv` positive, `num_heads` one of 4/8/16/32/64/128, and `head_dim` 32/64/128. The runtime SM count must be even. Logits may be FP32 or BF16. This path rejects `schedule_meta` and does not extend paged MQA to FP16 weights. `max_seqlen_k >= 0`; compressed output (`max_seqlen_k > 0`) requires `clean_logits=False`, while full-output cleaning uses a separate pass.
- **SM90 paged MQA:** physical KV pages may contain 32 or 64 tokens, with `next_n` 1/2/4, 32 or 64 heads and head dimension 32/64/128. Q/KV are FP8, weights are FP32 and cache scales are FP32. `indices` and paged cleaning are unsupported. For `next_n=4`, the scheduler uses two CTAs per work item: pass `deep_gemm.get_num_sms() // 2` as the `num_sms` argument to `get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)` and keep the runtime SM count even. For `next_n=1/2`, pass the full SM count. The helper's argument counts scheduler slots, not physical SMs; the two-CTA implementation issues independent KV copies, not a KV-multicast guarantee.

#### Mega MoE

The SM100 `fp8_fp4_mega_moe` path fuses EP dispatch, linear 1 and linear 2 (FP8xFP4 or FP8xFP8), gated activation, and EP combine, overlapping NVLink communication and tensor core computation. A process group is required; multi-rank execution uses symmetric memory, while a world-size-one group uses local CUDA storage without symmetric-memory rendezvous. The symmetric-memory setup requires PyTorch >= 2.9. Usage:

```python
buffer = deep_gemm.SymmBuffer(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
    mma_type='fp8xfp4',  # Use 'fp8xfp8' for FP8 routed-expert weights
)

transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

`get_symm_buffer_for_mega_moe` remains a compatibility factory. Inputs and their scales are caller-produced; when using FP8 shared experts, also populate `shared_l1_acts_sf` in the shared-input layout. See [Scaling Factor Format](docs/scaling-factor-format.md#63-mega-moe-fp8_fp4_mega_moe) and `tests/test_mega_moe.py` for preparation and multi-process examples.

**SM90 FP8 Mega MoE:** use `SM90SymmBuffer` (or the capacity-aligning factory `get_symm_buffer_for_sm90_mega_moe`), `transform_weights_for_mega_moe_sm90`, and `fp8_mega_moe`. This is a split L1/L2 implementation, not the SM100 single-kernel path. It supports FP8 dispatch and SwiGLU only, with no shared experts or SiTU. Routed weights are contiguous E4M3 with natural contiguous FP32 block-128-by-128 scales and `recipe=(128, 128, 128)`. The SM90 transform interleaves only the L1 gate/up weight rows; it does not interleave or pack the scales. `hidden` must be a positive multiple of 256 and `intermediate_hidden` a positive multiple of 128, subject to the selected launch configuration's combine alignment. Direct `SM90SymmBuffer` construction requires a positive capacity divisible by 128; the factory rounds capacity up. Output is contiguous BF16 `[num_tokens, hidden]`. See `tests/test_mega_moe_sm90.py` and [Section 6.5](docs/scaling-factor-format.md#65-sm90-fp8-mega-moe).

**SM100 NVFP4 Mega MoE:** call `fp4_fp4_mega_moe` with `SymmBuffer(..., mma_type='fp4xfp4')`. NVFP4 means packed E2M1 values plus per-16-element E4M3 scales, **not MXFP4/UE8M0**; `recipe=(1, 1, 16)` and SwiGLU are required. Both hidden dimensions must be positive multiples of 512, with an even runtime SM count of at least 2. Prepare routed weights with `transform_weights_for_mega_moe` after packing the E4M3 scales. Optional shared experts use BF16 weights and a separate contiguous `x_bf16`; provide both shared weights and `x_bf16` together, matching the buffer's `num_shared_experts`. Transform the BF16 shared weights with the same helper. The top-k budget is `num_topk + (num_shared_experts > 0) <= 32`.

Optional contiguous CUDA FP32 model scales are `l1_alphas[E, 2]` (gate, up), `l2_alphas[E]` and `a2_scales[E]`, with `E` the local expert count. Fold the FC1 input global scale into `l1_alphas`; `a2_scales` separately normalizes the internally requantized FC2 input and must be finite and strictly positive. Routing weights apply after L2, not before intermediate quantization. Routed partials are summed in FP32 and rounded to BF16, then multiplied by finite `routed_scaling_factor` and rounded to BF16 before adding the BF16 shared result. CUDA Graph replay must retain the captured `x_bf16` storage and token count. See `tests/test_nvfp4_mega_moe.py` and [Section 6.6](docs/scaling-factor-format.md#66-sm100-nvfp4-mega-moe).

NVFP4 `base=` reuse requires the same live process group and identical buffer layout: expert count, aligned token capacity, top-k, hidden dimensions, shared-expert count, MMA type, activation and runtime SM count. Do not change `num_sms` between allocation and execution or reuse storage across NVFP4 and the main FP8/BF16 protocol. The main FP8/BF16 `base=` capacity-reuse behavior is unchanged; the exact-layout restriction is NVFP4-specific.

**SM100 SiTU:** select `activation='situ'` consistently in `SymmBuffer`, `transform_weights_for_mega_moe` and `fp8_fp4_mega_moe`, using `mma_type='fp8xfp4'` (FP8 dispatch with FP4 routed weights). Supply finite, strictly positive `situ_beta` and `situ_linear_beta`, and leave `activation_clamp=None` (even explicit infinity is rejected). SiTU applies to **both routed and shared experts** through the unified L1 epilogue; optional shared weights remain FP8. It is not supported for FP8 routed weights, BF16 Mega MoE, NVFP4 or SM90. With BF16-rounded gate `g` and up `u`, the activation is `sigmoid(g) * beta * tanh(g / beta) * linear_beta * tanh(u / linear_beta)`. See `tests/test_mega_moe_situ.py`.

#### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms` / `get_num_sms`: set/get the maximum SM count to use
- `deep_gemm.set_tc_util` / `get_tc_util`: set/get an approximated tensor core utilization ratio
- `deep_gemm.set_pdl` / `get_pdl`: enable/disable Programmatic Dependent Launch (PDL)
- `deep_gemm.use_deterministic_algorithms`: enable/disable deterministic algorithms
- `deep_gemm.set_mk_alignment_for_contiguous_layout` / `get_mk_alignment_for_contiguous_layout`: set/get the group-level M/K alignment for contiguous layout
- `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout`: get the theoretical minimum M/K alignment
- `deep_gemm.set_ignore_compile_dims`: configure dimensions to ignore during JIT compilation
- `deep_gemm.set_block_size_multiple_of`: constrain block sizes to be multiples of a given value
- `deep_gemm.transform_sf_into_required_layout`: transform scaling factors into the required layout
- `deep_gemm.get_tma_aligned_size`: get the required TMA alignment size
- `deep_gemm.get_mn_major_tma_aligned_tensor`: get a MN-major TMA-aligned tensor
- `deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor`: get a MN-major TMA-aligned tensor (with packing FP32 into UE8M0)
- `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`: K-grouped GEMM packing kernel

The library also provides some environment variables, which may be useful:

Each `DG_JIT_*` variable falls back to the corresponding global `DJ_JIT_*` variable when unset.

- General
    - `DG_JIT_DEBUG`: `0` or `1`, enable JIT debugging features, including compiler command and PTXAS output, load-time reporting, line info, and PTX/SASS dumps; `0` by default
    - `DG_PRINT_CONFIGS`: `0` or `1`, print selected configs for each shape, `0` by default
- JIT cache
    - `DG_JIT_CACHE_DIR`: string, cache directory (or a `:`-separated list of directories) for compiled kernels; lookup searches all paths front-to-back (first hit wins) and a cache miss compiles into the first path, `$HOME/.dj` by default
- Compiler selection
    - `DG_JIT_NVCC_COMPILER`: string, NVCC compiler path; otherwise CUDA is found through `CUDA_HOME`, `CUDA_PATH`, `which nvcc`, then `/usr/local/cuda`
    - `DG_JIT_CPP_STANDARD`: integer, C++ standard version, `20` by default
- Compiler output
    - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print compilation commands, `0` by default
    - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS output, `0` by default
    - `DG_JIT_CHECK_NO_SPILLS`: `0` or `1`, assert no register spills in compiled kernels, `0` by default
    - `DG_JIT_CHECK_NO_LOCAL_MEMORY`: `0` or `1`, assert no local memory usage in compiled kernels, `0` by default
    - `DG_JIT_PRINT_LOAD_TIME`: `0` or `1`, print kernel load time, `0` by default
- Debug and profiling
    - `DG_JIT_WITH_LINEINFO`: `0` or `1`, embed source line info for profiling tools, `0` by default
    - `DG_JIT_DUMP_ASM`: `0` or `1`, dump both PTX and SASS, `0` by default
    - `DG_JIT_DUMP_PTX`: `0` or `1`, dump PTX output, `0` by default
    - `DG_JIT_DUMP_SASS`: `0` or `1`, dump SASS output, `0` by default
    - `DG_COMM_KERNEL_DEBUG`: `0` or `1`, zero the entire symmetric buffer after each Mega MoE call for debugging; refill caller-produced inputs before the next call, `0` by default
    - `DG_USE_NVIDIA_TOOLS`: `0` or `1`, skip internal profiling when running under external NVIDIA tools, `0` by default
- Build options
    - `DG_SKIP_CUDA_BUILD`: `0` or `1`, skip CUDA extension build during installation, `0` by default
    - `DG_FORCE_BUILD`: `0` or `1`, force local build instead of downloading pre-built wheels, `0` by default

For additional examples and details, please refer to [the test code](tests) or review the corresponding Python documentation.

## Acknowledgement

DeepGEMM is inspired by the [CUTLASS](https://github.com/nvidia/cutlass) project. Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient BLAS kernel library on GPU}, 
      author={Chenggang Zhao and Zhean Xu and Liang Zhao and Jiashi Li and Chenhao Xu and Anyi Xu and Shengyu Liu and Kexing Zhou and Kuai Yu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
