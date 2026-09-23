#pragma once

#include <torch/version.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <deep_gemm/common/exception.cuh>

DG_STATIC_ASSERT(TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3),
                 "DeepGEMM requires PyTorch 2.3 or newer");
DG_STATIC_ASSERT(CUDA_VERSION >= 12090, "DeepGEMM requires CUDA Driver API 12.9 or newer");
DG_STATIC_ASSERT(CUDART_VERSION >= 12090, "DeepGEMM requires CUDA Runtime API 12.9 or newer");
