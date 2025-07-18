# DeepGEMM

DeepGEMM is a library designed for clean and efficient General Matrix Multiplications (GEMMs). It supports FP8 and BF16 (working in progress) for both normal and Mix-of-Experts (MoE) grouped scenarios. Written in CUDA, the library has no kernel compilation need during installation, by compiling all kernels at runtime using a lightweight Just-In-Time (JIT) module.

DeepGEMM leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), it avoids heavy reliance on their templates or algebras. Instead, the library is designed for simplicity, with only a limited number of core kernel functions. This makes it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## News

- 2025.07.20: DeepGEMM now supports both SM90/SM100, and has a full refactor with a low-CPU-overhead JIT CPP module.
  - NVRTC and post-compilation SASS optimization are all disabled
  - NVRTC will be supported later
  - As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be no longer supported
  - Please see [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112) for more details
- 2025.05.14: DeepGEMM now offers weight gradient kernels for dense and MoE backward! See [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95) for details.
- 2025.05.07: DeepGEMM now supports NVRTC with up to 10x compilation speedup! See [#94](https://github.com/deepseek-ai/DeepGEMM/pull/94) for details. Please use `DG_JIT_USE_NVRTC=1` to enable it (may have performance loss with some cases).
- 2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800! See [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74), [#78](https://github.com/deepseek-ai/DeepGEMM/pull/78), [#81](https://github.com/deepseek-ai/DeepGEMM/pull/81), [#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) and [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526) for details.

## Roadmap

- [x] More correctness tests for grouped-contiguous layout
- [x] Shared memory swizzling for output
- [x] MoE scheduler with TMA multicast compatibility
- [x] Fix TMA multicast compatibility for indivisible shapes
- [x] Skip useless computation on M
- [ ] NVRTC as a faster compiler
- [ ] Sanitizer for testing
- [x] Weight gradient kernels for dense models
- [x] Weight gradient kernels for MoE models
- [ ] Better `get_best_configs` modeling
- [ ] CUDA PDL support
- [ ] Larger TMA multicast size for some shapes
- [x] MMA template refactor with CUTLASS
- [x] Remove shape limitations on N and K
- [ ] BF16 kernels
- [ ] Split/stream-k optimizations
- [ ] Ampere kernels
- [ ] Polish docs

## Quick start

### Requirements

- NVIDIA SM90 or SM100 architecture GPU
- Python 3.8 or higher
- Compilers with C++20 support
- CUDA Toolkit:
    - Currently, CUDA 12.8 or higher is required, but support for older versions may be added in the future
    - CUDA 12.8 or higher for SM90
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

# Test all GEMM implements
python tests/test_layout.py
python tests/test_core.py
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

Use `fp8_m_grouped_gemm_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.

#### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms`: set the maximum SM count to use
- `deep_gemm.get_num_sms`: get the current SM maximum count (return the device SM count if not set)
- `deep_gemm.transform_sf_into_required_layout`: transform scaling factors into required layout
- `deep_gemm.get_tma_aligned_size`: get the required TMA alignment size
- `deep_gemm.get_mk_alignment_for_contiguous_layout`: get the group-level alignment requirement for grouped contiguous layout
- `deep_gemm.get_mn_major_tma_aligned_tensor`: get a MN-major TMA-aligned tensor
- `deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor`: get a MN-major TMA-aligned tensor (with packing FP32 into UE8M0)
- `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`: K-grouped GEMM packing kernel

The library also provides some environment variables, which may be useful:

- General
    - `DG_JIT_DEBUG`: `0` or `1`, print more JIT debugging information, `0` by default
- JIT cache related
    - `DG_JIT_CACHE_DIR`: string, the cache directory to store compiled kernels, `$HOME/.deep_gemm` by default
- NVCC/NVRTC selections
    - `DG_JIT_USE_NVRTC`: `0` or `1`, use NVRTC instead of NVCC, faster compilation but maybe have lower performance for some cases, `0` by default
    - `DG_JIT_NVCC_COMPILER`: string, specified NVCC compiler path; will find in `torch.utils.cpp_extension.CUDA_HOME` by default
- Compiler options
    - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS compiler output, `0` by default
    - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print NVCC compilation command, `0` by default
- Heuristic selection
    - `DG_PRINT_CONFIGS`: `0` or `1`, print selected configs for each shape, `0` by default

For additional examples and details, please refer to [the test code](tests/test_core.py) or review the corresponding Python documentation.

## Acknowledgement

DeepGEMM is inspired by the [CUTLASS](https://github.com/nvidia/cutlass) project. Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).
