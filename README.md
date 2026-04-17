# DeepGEMM

DeepGEMM is a unified, high-performance tensor core kernel library that brings together the key computation primitives of modern large language models — GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more — into a single, cohesive CUDA codebase. All kernels are compiled at runtime via a lightweight Just-In-Time (JIT) module, requiring no CUDA compilation during installation.

DeepGEMM leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), but avoids heavy reliance on their templates or algebras. The library is designed for simplicity, with only a limited number of core kernel functions, making it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## News

- 2026.04.16: Mega MoE, FP8xFP4 GEMM, FP4 Indexer, PDL, faster JIT compilation and more.
  - Performance comparison will be posted later.
  - Please see [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) for more details.
- 2025.09.28: DeepGEMM now supports scoring kernels (weighted ReLU MQA logits) for the lightning indexer for DeepSeek v3.2.
  - Please see [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200) for more details.
- 2025.07.20: DeepGEMM now supports both SM90/SM100, and has a full refactor with a low-CPU-overhead JIT CPP module.
  - NVRTC and post-compilation SASS optimization are all disabled.
  - NVRTC will be supported later.
  - As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be no longer supported.
  - Please see [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112) for more details.
- 2025.05.14: DeepGEMM now offers weight gradient kernels for dense and MoE backward! See [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95) for details.
- 2025.05.07: DeepGEMM now supports NVRTC with up to 10x compilation speedup! See [#94](https://github.com/deepseek-ai/DeepGEMM/pull/94) for details. Please use `DG_JIT_USE_NVRTC=1` to enable it (may have performance loss with some cases).
- 2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800! See [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74), [#78](https://github.com/deepseek-ai/DeepGEMM/pull/78), [#81](https://github.com/deepseek-ai/DeepGEMM/pull/81), [#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) and [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526) for details.

## Quick start

### Requirements

- NVIDIA SM90 or SM100 architecture GPU
- Python 3.8 or higher
- Compilers with C++20 support
- CUDA Toolkit:
  - CUDA 12.3 or higher for SM90
    - **We highly recommend 12.9 or higher for the best performance**
  - CUDA 12.9 or higher for SM100
- PyTorch 2.1 or higher
- CUTLASS 4.0 or higher (could be cloned by Git submodule)
- `{fmt}` library (could be cloned by Git submodule)

### Development

```bash
# Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Link some essential includes and build the CPP JIT module
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

For both architectures, the LHS scaling factor is required to have a TMA-aligned and transposed layout. And the data format for the scaling factor of SM90 and SM100 is different:

- SM90 requires scaling factors in FP32 format.
- SM100 requires scaling factors in packed [UE8M0](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats) format, which packs 4 UE8M0 into a single `torch.int`.

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

#### Mega MoE

Mega MoE fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory. Usage:

```python
# Allocate symmetric memory buffer
# NOTES: requires PyTorch >= 2.9
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
)

# Transform weights (FP4 with UE8M0 SF) into the required layout
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

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

- General
  - `DG_JIT_DEBUG`: `0` or `1`, print JIT debugging information, `0` by default
  - `DG_PRINT_CONFIGS`: `0` or `1`, print selected configs for each shape, `0` by default
- JIT cache
  - `DG_JIT_CACHE_DIR`: string, cache directory for compiled kernels, `$HOME/.deep_gemm` by default
- Compiler selection
  - `DG_JIT_USE_NVRTC`: `0` or `1`, use NVRTC instead of NVCC (faster compilation, may have lower performance for some cases), `0` by default
  - `DG_JIT_NVCC_COMPILER`: string, NVCC compiler path; defaults to `torch.utils.cpp_extension.CUDA_HOME`
  - `DG_JIT_CPP_STANDARD`: integer, C++ standard version, `20` by default
- Compiler output
  - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print compilation commands, `0` by default
  - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS output, `0` by default
  - `DG_JIT_PTXAS_CHECK`: `0` or `1`, assert no local memory usage in compiled kernels, `0` by default
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
  - `DG_JIT_USE_RUNTIME_API`: `0` or `1`, use CUDA Runtime API for kernel loading (requires CUDA runtime >= 12.8), `0` by default

For additional examples and details, please refer to [the test code](tests/test_core.py) or review the corresponding Python documentation.

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
