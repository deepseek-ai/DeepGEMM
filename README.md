# DeepGEMM

DeepGEMM is a library designed for clean and efficient FP8 General Matrix Multiplications (GEMMs) with fine-grained scaling, as proposed in [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3). It supports both normal and Mix-of-Experts (MoE) grouped GEMMs. Written in CUDA, the library has no compilation need during installation, by compiling all kernels at runtime using a lightweight Just-In-Time (JIT) module.

Currently, DeepGEMM exclusively supports NVIDIA Hopper tensor cores. To address the imprecise FP8 tensor core accumulation, it employs CUDA-core two-level accumulation (promotion). While it leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), it avoids heavy reliance on their templates or algebras. Instead, the library is designed for simplicity, with only one core kernel function. This makes it a clean and accessible resource for learning Hopper FP8 matrix multiplication and optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## News

- 2025.05.14: DeepGEMM now offers weight gradient kernels for dense and MoE backward! See [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95) for details.
- 2025.05.07: DeepGEMM now supports NVRTC with up to 10x compilation speedup! See [#94](https://github.com/deepseek-ai/DeepGEMM/pull/94) for details. Please use `DG_JIT_USE_NVRTC=1` to enable it (may have performance loss with some cases).
- 2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800! See [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74), [#78](https://github.com/deepseek-ai/DeepGEMM/pull/78), [#81](https://github.com/deepseek-ai/DeepGEMM/pull/81), [#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) and [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526) for details.

## Roadmap

- [x] More correctness tests for grouped-contiguous layout
- [x] Shared memory swizzling for output
- [ ] Larger block size on N (up to 256)
- [x] MoE scheduler with TMA multicast compatibility
- [x] Fix TMA multicast compatibility for indivisible shapes
- [x] Skip useless computation on M
- [x] NVRTC as a faster compiler
- [ ] Stolen JIT cache
- [ ] Sanitizer for testing
- [x] Weight gradient kernels for dense models
- [x] Weight gradient kernels for MoE models
- [ ] Better `get_best_configs` modeling
- [ ] Utility kernels for MoE models (maybe with [tile-lang](https://github.com/tile-ai/tilelang))
- [ ] CUDA PDL support
- [ ] More scaling granularity support via templates
- [ ] Larger TMA multicast size for some shapes
- [x] MMA template refactor with CUTLASS
- [ ] Optimizations for power efficiency
- [x] Remove shape limitations on N and K
- [ ] BF16 kernels
- [ ] Split/stream-k optimizations

## Quick start

### Requirements

- Hopper architecture GPUs, `sm_90a` must be supported
- Python 3.8 or above
- CUDA 12.3 or above
  - **But we highly recommend 12.8 or above for the best performance**
- PyTorch 2.1 or above
- CUTLASS 3.6 or above (could be cloned by Git submodule)

### Development

```bash
# Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git

# Make symbolic links for third-party (CUTLASS and CuTe) include directories
python setup.py develop

# Test JIT compilation
python tests/test_jit.py

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
python tests/test_core.py
```

### Installation

```bash
python setup.py install
```

Then, import `deep_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices

This library exclusively contains GEMM kernels. It requires the LHS scaling factor to be TMA-aligned and transposed, and it only supports the NT format (non-transposed LHS and transposed RHS). For transposition or other FP8 casting operations, please implement or fuse them into prior kernels independently. While the library provides some simple PyTorch utility functions, these may result in slower performance, but our primary focus is on optimizing the GEMM kernels themselves.

#### Normal dense GEMMs (non-grouped)

To perform a basic non-grouped FP8 GEMM, call the `deep_gemm.gemm_fp8_fp8_bf16_nt` function. For more details, please refer to the function documentation.

#### Grouped GEMMs (contiguous layout)

Unlike traditional grouped GEMMs in CUTLASS, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape.

For training forward passes or inference prefilling, where each expert may process a varying number of tokens, we concatenate these tokens into a single tensor, referred to as the "contiguous" layout. Note that each expert segment must be aligned to the GEMM M block size (`get_m_alignment_for_contiguous_layout()`).

For more information, please refer to the `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` function documentation.

#### Grouped GEMMs (masked layout)

During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions.

Use `m_grouped_gemm_fp8_fp8_bf16_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.

#### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms`: set the maximum SM count to use
- `deep_gemm.get_num_sms`: get the current SM maximum count
- `deep_gemm.get_m_alignment_for_contiguous_layout`: get the group-level alignment requirement for grouped contiguous layout
- `deep_gemm.get_tma_aligned_size`: get the required TMA alignment size
- `deep_gemm.get_col_major_tma_aligned_tensor`: get a column-major TMA-aligned tensor

The library also provides some environment variables, which may be useful:

- General
  - `DG_JIT_DEBUG`: `0` or `1`, print more JIT debugging information, `0` by default
- JIT cache related
  - `DG_JIT_CACHE_DIR`: string, the cache directory to store compiled kernels, `$HOME/.deep_gemm` by default
  - `DG_JIT_DISABLE_CACHE`: `0` or `1`, disable the use of cache directory, `0` by default
- NVCC/NVRTC selections
  - `DG_JIT_USE_NVRTC`: `0` or `1`, use NVRTC instead of NVCC, faster compilation but maybe have lower performance for some cases, `0` by default
  - `DG_JIT_NVCC_COMPILER`: string, specified NVCC compiler path; will find in `torch.utils.cpp_extension.CUDA_HOME` by default
- Compiler options
  - `DG_JIT_OVERRIDE_CPP_STANDARD`: integer (e.g., `20`), support for some old version GCC compiler, `20` by default
  - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS compiler output, `0` by default
  - `DG_JIT_PRINT_REG_REUSE`: `0` or `1`, print FFMA-interleaving details, `0` by default
  - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print NVCC compilation command, `0` by default
- Post optimization
  - `DG_JIT_DISABLE_FFMA_INTERLEAVE`: `0` or `1`, disable FFMA-interleaving optimization, `0` by default
- Heuristic selection
  - `DG_PRINT_CONFIGS`: `0` or `1`, print selected configs for each shape, `0` by default
- Testing
  - `DG_NSYS_PROFILING`: `0` or `1`, Nsight-system compatible testing, `0` by default

For additional examples and details, please refer to [the test code](tests/test_core.py) or review the corresponding Python documentation.

## Optimizations

We indicate the techniques excluded from CUTLASS with 🐳.

#### Persistent warp-specialization

Following the CUTLASS design, the kernels in DeepGEMM are warp-specialized, enabling overlapping data movement, tensor-core MMA instructions, and CUDA-core promotion. A simplified figure illustrating this process is shown below:

![design](figures/design.png)

#### Hopper TMA features

The [Tensor Memory Accelerator](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator) (TMA) is a new hardware feature introduced by the Hopper architecture, designed for faster and asynchronous data movement. Specifically, we utilize TMA for:

- TMA load for LHS, LHS scaling factors, and RHS matrices
- TMA store for the output matrix
- TMA multicast (automatically decide LHS or RHS to broadcast)
- TMA descriptor prefetching

#### Common detail optimizations

- Utilization of the [`stmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix) PTX instruction
- [Register count control](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) tailored for different warpgroups
- Less bank conflicts via 3D TMA or swizzling
- Larger block sizes (up to 256x128 🐳)
- Overlapping as much as possible, e.g., overlapping TMA store and non-TMA RHS scaling factor load 🐳

#### A unified and optimized block scheduler

- [One scheduler](deep_gemm/include/deep_gemm/scheduler.cuh) for all non-grouped and grouped kernels
- [Rasterization](https://github.com/NVIDIA/cutlass/blob/eefa171318b79cbe2e78514d4cce5cd0fe919d0c/media/docs/efficient_gemm.md#threadblock-rasterization) to enhance L2 cache reuse

#### Fully JIT design 🐳

DeepGEMM employs a fully [Just-In-Time](deep_gemm/jit) (JIT) design, with no compilation required at installation. All kernels are compiled at runtime using a lightweight JIT implementation. This approach offers several advantages:

- GEMM shapes, block sizes, and the number of pipeline stages are treated as compile-time constants
  - Saving registers
  - Compilers may do more optimizations
- Automatic selection of block sizes, number of warpgroups, optimal pipeline stages, and TMA cluster size
  - But without auto-tuning, the optimal one is deterministically selected
- Full unrolling of the MMA pipelines, providing compilers with more optimization opportunities
  - Very important for small shapes 
  - Refer to `launch_k_iterations` in [the kernel file](deep_gemm/include/deep_gemm/fp8_gemm.cuh) for details

Overall, JIT significantly improves performance for small shapes, similar to the approach of the [Triton](https://github.com/triton-lang/triton/) compiler.

#### Unaligned block sizes 🐳

For certain shapes, block sizes aligned to powers of 2 can lead to underutilized SMs. For instance, with `M=256, N=7168`, a typical block size assignment of `BLOCK_M=128, BLOCK_N=128` results in only `(256 / 128) * (7168 / 128) = 112` out of 132 SMs being utilized. To address this, we support unaligned block sizes like 112, enabling `(256 / 128) * (7168 / 112) = 128` SMs to work in such scenarios. Implementing this technique alongside fine-grained scaling requires careful optimization but ultimately delivers performance gains.

#### FFMA SASS interleaving 🐳

We observe a performance improvement in [the CUTLASS FP8 kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/54_hopper_fp8_warp_specialized_gemm) between NVCC 12.2 and 12.3. By comparing the compiled SASS, we discover that one bit in [a series of `FADD` instructions](https://github.com/NVIDIA/cutlass/blob/eefa171318b79cbe2e78514d4cce5cd0fe919d0c/include/cutlass/gemm/collective/fp8_accumulation.hpp#L73) is flipped in an interleaving pattern.
After referencing some open-source [CUDA assembler](https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuControlCode.py#L46) implementations, we identified that this bit controls `yield`, which may enhance warp-level parallelism (just a guess, yielding the current warp and let other warps work).

To leverage this, we develop [a similar script](deep_gemm/jit/interleave_ffma.py) to modify the `FFMA` instructions in the compiled binary. Besides simply modifying the `yield` bit, we also flip the `reuse` bit (registers cannot be reused if the warp is yielded). This adjustment improves performance (10%+ in some cases) for fine-grained scaling FP8 GEMMs by creating more opportunities to overlap MMA instructions with promotion `FFMA` instructions.

## Acknowledgement

DeepGEMM is inspired by the [CUTLASS](https://github.com/nvidia/cutlass) project. Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling}, 
      author={Chenggang Zhao and Liang Zhao and Jiashi Li and Zhean Xu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
