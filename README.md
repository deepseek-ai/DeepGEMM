# DeepGEMM

DeepGEMM is a unified, high-performance tensor core kernel library that brings together the key computation primitives of modern large language models — GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more — into a single, cohesive CUDA codebase. All kernels are compiled at runtime through DeepJIT, requiring no CUDA compilation during installation.

DeepGEMM leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), but avoids heavy reliance on their templates or algebras. The library is designed for simplicity, with only a limited number of core kernel functions, making it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## News

- 2026.10.15
  - Added support for NVIDIA **SM89 (Ada Lovelace)**, the first architecture with FP8 Tensor
    Cores: **native FP8 GEMMs** (`fp8_gemm_*`, both the 1D1D and 1D2D scaling recipes) built on
    `mma.sync.m16n8k32` + `ldmatrix` + `cp.async`, reaching ~85% of cuBLASLt FP8 on Ada, plus
    cuBLASLt-backed BF16 GEMMs. See
    [docs/architecture_support.md](docs/architecture_support.md) for the porting notes.
  - Fixed the cuBLASLt fallback silently running BF16 GEMMs at TF32 precision
    (`CUBLAS_COMPUTE_32F_FAST_TF32` is now only used for genuine FP32 inputs).
  - Note: NVIDIA **SM80 (Ampere) has no FP8 Tensor Cores** and is not a supported target.
- 2026.09.30
  - DeepGEMM Ascend is available! Check [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend/) for more details
  - Add more optimizations, including locality domain features, check [#462](https://github.com/deepseek-ai/DeepGEMM/pull/462) for more details
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
- 2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800! See [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74), [#78](https://github.com/deepseek-ai/DeepGEMM/pull/78), [#81](https://github.com/deepseek-ai/DeepGEMM/pull/81), [#86](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526) for details.

## Quick start

### Requirements

- NVIDIA GPU architecture support:
  - **SM89 (Ada Lovelace)**: Native FP8 GEMM (`fp8_gemm_*`, 1D1D and 1D2D recipes) and BF16 GEMM via cuBLASLt.
  - **SM90 (Hopper)**: Native FP8 and BF16 GEMMs.
  - **SM100 (Blackwell)**: Native FP8, FP4, and BF16 GEMMs.
  - **SM80 (Ampere)** is **not** supported: it predates the FP8 Tensor Cores (introduced in SM89).

  Operations a given architecture cannot run raise an explicit error naming the minimum
  supported architecture; see the full [support matrix](docs/architecture_support.md).
- Python 3.8 or higher
- Compilers and standard libraries with C++20 `<format>` support
- CUDA Toolkit 12.9 or higher
- PyTorch 2.3 or higher
- CUTLASS 4.0 or higher (could be cloned by Git submodule)

### Development & Installation Guide

```bash
# Clone the repository with submodules
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Setup C++ development extension
./develop.sh

# Install DeepGEMM into python environment
./install.sh
```

### Running Hardware Validation Tests

To verify the DeepGEMM installation and the architecture dispatch on your GPU:

```bash
python3 -m unittest tests/test_sm89.py
```

The suite covers native FP8 GEMMs (both scaling recipes, both output dtypes, accumulation,
ragged shapes, determinism), BF16 GEMMs in every operand layout, the FP32 scaling-factor layout
kernels, and the explicit error paths for operations SM89 cannot perform. It skips itself
automatically on other architectures, so it is safe to run from a cross-architecture CI matrix.
For the detailed architecture matrix and the SM89 porting notes, refer to
[docs/architecture_support.md](docs/architecture_support.md).

## Interfaces

#### Notices

This library provides optimized GEMM kernels for NVIDIA GPUs with a naming convention: `D = C + A @ B`. The input shape layout is NT (non-transposed A, transposed B). While the SM89 and SM90 implementations support only the NT memory layout (row-major, col-major), the SM100 implementation supports all memory layouts (NT, TN, NN, TT). For example, `fp8_gemm_nt` will do a `D = C + A @ B.T`

The LHS scaling factor is required to have a TMA-aligned and transposed layout. And the data format for the scaling factor of SM89, SM90 and SM100 is different:

- SM89 and SM90 require scaling factors in FP32 format, with a K granularity of 128.
- SM100 requires scaling factors in packed [UE8M0](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats) format, which packs 4 UE8M0 into a single `torch.int`.

On SM89 and SM90 both scaling recipes are available: `(1, 1, 128)` (1D1D, per-token scales for
A and B) and `(1, 128, 128)` (1D2D, per-token A scales and one scale per 128-wide N block for
B, which is also the default when FP32 scales are used).

Please note that operations like input transposition or FP8 casting must be handled separately by the user, please implement or fuse them into prior kernels independently. While the library provides some simple PyTorch utility functions, these may result in slower performance, but our primary focus is on optimizing the GEMM kernels themselves.

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
Take the non-paged version `fp8_fp4_mqa_logits` as an example. Its main inputs are:

- `q`, a `(q_data, q_sf)` tuple; SM100 accepts MXFP4/MXFP8 data with packed UE8M0 scales
- `kv`, a `(kv_data, kv_sf)` tuple with shape `[seq_len_kv, head_dim]` logically
- `weights`, tensor with shape `[seq_len, num_heads]` (BF16 on SM100)
- `cu_seq_len_k_start` and `cu_seq_len_k_end`, int tensor with shape `[seq_len]`
- `max_seqlen_k`, the maximum valid KV span of any query row

The output is compressed to `[seq_len, max_seqlen_k]`; row `i` stores its valid KV span starting at column zero.
For each token `i` in `q`, it will iterate all tokens `j` from `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])`,
and calculate the corresponding compressed logit as:

```python
kv_j = kv[0][j, :] * kv[1][j].unsqueeze(1)  # [head_dim]
out_ij = q[i, :, :] @ kv_j  # [num_heads]
out_ij = out_ij.relu() * weights[i, :]  # [num_heads]
out_ij = out_ij.sum()  # Scalar
```

For more details and the paged version `fp8_fp4_paged_mqa_logits`, please refer to `tests/test_attention.py`.

#### Mega MoE

Mega MoE fuses and overlaps EP dispatch, linear 1 and linear 2 (FP8xFP4 or FP8xFP8), SwiGLU, and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory. Usage:

```python
# Allocate symmetric memory buffer
# NOTES: requires PyTorch >= 2.9
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
    mma_type='fp8xfp4',  # Use 'fp8xfp8' for FP8 routed-expert weights
)

# Transform weights (FP4 or FP8 with UE8M0 SF) into the required layout
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

# (Optional) Localize weights into locality domains
transformed_l1 = (deep_gemm.localize(transformed_l1[0]), transformed_l1[1])
transformed_l2 = (deep_gemm.localize(transformed_l2[0]), transformed_l2[1])
deep_gemm.destroy_localizer()

# Copy inputs into the buffer before each call
# You may fuse these into previous kernels
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# Run the fused mega MoE kernel
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

For the full example with multi-process setup and benchmarking, please refer to `tests/test_mega_moe.py`.

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
    - `DG_COMM_KERNEL_DEBUG`: `0` or `1`, zero symmetric buffer before each Mega MoE call for debugging, `0` by default
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
