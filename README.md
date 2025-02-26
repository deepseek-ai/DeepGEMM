# DeepGEMM

DeepGEMM is a library designed for clean and efficient FP8 General Matrix Multiplications (GEMMs) with fine-grained scaling, as proposed in [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3). It supports both normal and Mix-of-Experts (MoE) grouped GEMMs. Written in CUDA, the library has no compilation need during installation, by compiling all kernels at runtime using a lightweight Just-In-Time (JIT) module.

Currently, DeepGEMM exclusively supports NVIDIA Hopper tensor cores. To address the imprecise FP8 tensor core accumulation, it employs CUDA-core two-level accumulation (promotion). While it leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), it avoids heavy reliance on their templates or algebras. Instead, the library is designed for simplicity, with only one core kernel function comprising around **~300 lines of code**. This makes it a clean and accessible resource for learning Hopper FP8 matrix multiplication and optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## Performance

We test all shapes potentially used in DeepSeek-V3/R1 inference (including both prefilling and decoding, but without tensor parallelism) on H800 with NVCC 12.8. All speedup metrics are calculated in comparison to our internally and carefully optimized implementation based on CUTLASS 3.6.

DeepGEMM does not behave very well on some shapes, optimization PRs are welcomed if you are interested.

### Normal GEMMs for dense models

|  M   |   N   |   K   | Computation | Memory bandwidth | Speedup |
|:----:|:-----:|:-----:|:-----------:|:----------------:|:-------:|
|  64  | 2112  | 7168  | 206 TFLOPS  |    1688 GB/s     |  2.7x   |
|  64  | 24576 | 1536  | 289 TFLOPS  |    2455 GB/s     |  1.7x   |
|  64  | 32768 |  512  | 219 TFLOPS  |    2143 GB/s     |  1.8x   |
|  64  | 7168  | 16384 | 336 TFLOPS  |    2668 GB/s     |  1.4x   |
|  64  | 4096  | 7168  | 287 TFLOPS  |    2320 GB/s     |  1.4x   |
|  64  | 7168  | 2048  | 295 TFLOPS  |    2470 GB/s     |  1.7x   |
| 128  | 2112  | 7168  | 352 TFLOPS  |    1509 GB/s     |  2.4x   |
| 128  | 24576 | 1536  | 535 TFLOPS  |    2448 GB/s     |  1.6x   |
| 128  | 32768 |  512  | 358 TFLOPS  |    2103 GB/s     |  1.5x   |
| 128  | 7168  | 16384 | 645 TFLOPS  |    2604 GB/s     |  1.4x   |
| 128  | 4096  | 7168  | 533 TFLOPS  |    2221 GB/s     |  2.0x   |
| 128  | 7168  | 2048  | 510 TFLOPS  |    2277 GB/s     |  1.7x   |
| 4096 | 2112  | 7168  | 1058 TFLOPS |     527 GB/s     |  1.1x   |
| 4096 | 24576 | 1536  | 990 TFLOPS  |     786 GB/s     |  1.0x   |
| 4096 | 32768 |  512  | 590 TFLOPS  |    1232 GB/s     |  1.0x   |
| 4096 | 7168  | 16384 | 1358 TFLOPS |     343 GB/s     |  1.2x   |
| 4096 | 4096  | 7168  | 1304 TFLOPS |     500 GB/s     |  1.1x   |
| 4096 | 7168  | 2048  | 1025 TFLOPS |     697 GB/s     |  1.1x   |

### Grouped GEMMs for MoE models (contiguous layout)

| #Groups | M per group |  N   |  K   | Computation | Memory bandwidth | Speedup |
|:-------:|:-----------:|:----:|:----:|:-----------:|:----------------:|:-------:|
|    4    |    8192     | 4096 | 7168 | 1297 TFLOPS |     418 GB/s     |  1.2x   |
|    4    |    8192     | 7168 | 2048 | 1099 TFLOPS |     681 GB/s     |  1.2x   |
|    8    |    4096     | 4096 | 7168 | 1288 TFLOPS |     494 GB/s     |  1.2x   |
|    8    |    4096     | 7168 | 2048 | 1093 TFLOPS |     743 GB/s     |  1.1x   |

### Grouped GEMMs for MoE models (masked layout)

| #Groups | M per group |  N   |  K   | Computation | Memory bandwidth | Speedup |
|:-------:|:-----------:|:----:|:----:|:-----------:|:----------------:|:-------:|
|    1    |    1024     | 4096 | 7168 | 1233 TFLOPS |     924 GB/s     |  1.2x   |
|    1    |    1024     | 7168 | 2048 | 925 TFLOPS  |     968 GB/s     |  1.2x   |
|    2    |     512     | 4096 | 7168 | 1040 TFLOPS |    1288 GB/s     |  1.2x   |
|    2    |     512     | 7168 | 2048 | 916 TFLOPS  |    1405 GB/s     |  1.2x   |
|    4    |     256     | 4096 | 7168 | 932 TFLOPS  |    2064 GB/s     |  1.1x   |
|    4    |     256     | 7168 | 2048 | 815 TFLOPS  |    2047 GB/s     |  1.2x   |

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

- `DG_CACHE_DIR`: string, the cache directory to store compiled kernels, `$HOME/.deep_gemm` by default
- `DG_NVCC_COMPILER`: string, specified NVCC compiler path; will find in `from torch.utils.cpp_extension.CUDA_HOME` by default
- `DG_DISABLE_FFMA_INTERLEAVE`: 0 or 1, disable FFMA-interleaving optimization
- `DG_PTXAS_VERBOSE`: 0 or 1, show detailed PTXAS compiler output
- `DG_PRINT_REG_REUSE`: 0 or 1, print FFMA-interleaving details
- `DG_JIT_PRINT_NVCC_COMMAND`: 0 or 1, print NVCC compilation command
- `DG_JIT_DEBUG`: 0 or 1, print more debugging information

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
- TMA multicast (exclusive to the LHS matrix)
- TMA descriptor prefetching

#### Common detail optimizations

- Utilization of the [`stmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix) PTX instruction
- [Register count control](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) tailored for different warpgroups
- Overlapping as much as possible, e.g. overlapping TMA store and non-TMA RHS scaling factor load 🐳

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
After referencing some open-source [CUDA assembler](https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuControlCode.py#L46) implementations, we identified that this bit controls `yield`, which may enhance warp-level parallelism (just a guess, yielding the current warp and let other warps work).

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
