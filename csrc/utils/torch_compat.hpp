#pragma once

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/tvm_ffi.h>

#include "exception.hpp"

namespace deep_gemm {

// ---------------------------------------------------------------------------
// Scalar-type → CUDA / cuBLAS / TensorMap conversions
// ---------------------------------------------------------------------------
inline std::string dtype_to_cuda_type_string(at::ScalarType dtype) {
    if (dtype == at::kInt)             return "int";
    if (dtype == at::kFloat)           return "float";
    if (dtype == at::kBFloat16)        return "cutlass::bfloat16_t";
    if (dtype == at::kFloat8_e4m3fn)   return "cutlass::float_e4m3_t";
    if (dtype == at::kByte)            return "cutlass::detail::float_e2m1_unpacksmem_t";
    DG_HOST_UNREACHABLE("Unsupported dtype for CUDA type string");
}

inline cudaDataType_t dtype_to_cublas(at::ScalarType dtype) {
    if (dtype == at::kFloat)           return CUDA_R_32F;
    if (dtype == at::kHalf)            return CUDA_R_16F;
    if (dtype == at::kBFloat16)        return CUDA_R_16BF;
    if (dtype == at::kFloat8_e4m3fn)   return CUDA_R_8F_E4M3;
    if (dtype == at::kInt)             return CUDA_R_32I;
    DG_HOST_UNREACHABLE("Unsupported dtype for cuBLAS");
}

inline CUtensorMapDataType dtype_to_tensormap(at::ScalarType dtype, bool allow_tf32 = false) {
    if (allow_tf32 && dtype == at::kFloat) return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
    if (dtype == at::kInt)             return CU_TENSOR_MAP_DATA_TYPE_INT32;
    if (dtype == at::kFloat)           return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    if (dtype == at::kBFloat16)        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    if (dtype == at::kFloat8_e4m3fn)   return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if (dtype == at::kByte)            return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
    DG_HOST_UNREACHABLE("Unsupported dtype for TensorMap");
}

inline std::string dtype_to_string(at::ScalarType dtype) {
    return std::string(c10::toString(dtype));
}

inline at::ScalarType string_to_dtype(const std::string& dtype_str) {
   auto it = c10::getStringToDtypeMap().find(dtype_str);
   if (it == c10::getStringToDtypeMap().end()) {
       DG_HOST_UNREACHABLE("Unsupported dtype string");
   }
   return it->second;
}

// ---------------------------------------------------------------------------
// CUDA stream helper
// ---------------------------------------------------------------------------
inline cudaStream_t get_current_cuda_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

// ---------------------------------------------------------------------------
// Create torch::Tensor from raw device pointer (non-owning view)
// ---------------------------------------------------------------------------
inline torch::Tensor tensor_from_ptr(void* data, at::ScalarType dtype,
                                     std::initializer_list<int64_t> shape,
                                     int device_id = 0) {
    auto opts = torch::TensorOptions().dtype(dtype)
                    .device(torch::kCUDA, device_id)
                    .requires_grad(false);
    auto sizes = std::vector<int64_t>(shape);
    return torch::from_blob(data, sizes, opts);
}

inline torch::Tensor tensor_from_ptr_strided(void* data, at::ScalarType dtype,
                                             std::initializer_list<int64_t> shape,
                                             std::initializer_list<int64_t> strides,
                                             int device_id = 0) {
    auto opts = torch::TensorOptions().dtype(dtype)
                    .device(torch::kCUDA, device_id)
                    .requires_grad(false);
    auto sizes = std::vector<int64_t>(shape);
    auto str   = std::vector<int64_t>(strides);
    return torch::from_blob(data, sizes, str, opts);
}

// ---------------------------------------------------------------------------
// DLTensor* → torch::Tensor (non-owning, for tvm-ffi boundary)
// ---------------------------------------------------------------------------
inline at::ScalarType dl_dtype_to_torch(DLDataType dtype) {
    if (dtype.lanes != 1) DG_HOST_UNREACHABLE("Unsupported DLDataType lanes");
    switch (dtype.code) {
        case kDLFloat:
            if (dtype.bits == 64) return at::kDouble;
            if (dtype.bits == 32) return at::kFloat;
            if (dtype.bits == 16) return at::kHalf;
            break;
        case kDLBfloat:
            if (dtype.bits == 16) return at::kBFloat16;
            break;
        case kDLInt:
            if (dtype.bits == 64) return at::kLong;
            if (dtype.bits == 32) return at::kInt;
            if (dtype.bits == 16) return at::kShort;
            if (dtype.bits ==  8) return at::kChar;
            break;
        case kDLUInt:
            if (dtype.bits == 8) return at::kByte;
            break;
        case kDLFloat8_e4m3fn: // kDLFloat8_e4m3fn
            return at::kFloat8_e4m3fn;
        default:
            break;
    }
    DG_HOST_UNREACHABLE("Unsupported DLDataType for torch conversion: " + tvm::ffi::DLDataTypeToString(dtype));
}

inline torch::Tensor convert_to_torch_tensor(tvm::ffi::TensorView tensor) {
    auto scalar_type = dl_dtype_to_torch(tensor.dtype());
    int device_id = tensor.device().device_id;
    void* data = static_cast<char*>(tensor.data_ptr()) + tensor.byte_offset();

    auto sizes = std::vector<int64_t>(tensor.shape().begin(), tensor.shape().end());
    auto opts = torch::TensorOptions().dtype(scalar_type)
                    .device(torch::kCUDA, device_id)
                    .requires_grad(false);

    if (tensor.strides().data()) {
        auto strides = std::vector<int64_t>(tensor.strides().begin(), tensor.strides().end());
        return torch::from_blob(data, sizes, strides, opts);
    }
    return torch::from_blob(data, sizes, opts);
}

} // namespace deep_gemm
