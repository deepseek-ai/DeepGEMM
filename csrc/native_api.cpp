#include "native_api.h"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

struct DeviceState {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
};

std::mutex g_mutex;
std::unordered_map<int, DeviceState> g_devices;
thread_local std::string g_last_error;

const char* CublasStatus(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
        default: return "unknown";
    }
}

void CheckCuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << expression << ": " << cudaGetErrorName(status) << " ("
                << cudaGetErrorString(status) << ")";
        throw std::runtime_error(message.str());
    }
}

void CheckCublas(cublasStatus_t status, const char* expression) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream message;
        message << expression << ": CUBLAS_STATUS_" << CublasStatus(status);
        throw std::runtime_error(message.str());
    }
}

cudaDataType_t CudaType(deep_gemm_native_dtype dtype) {
    switch (dtype) {
        case DEEP_GEMM_NATIVE_FLOAT16: return CUDA_R_16F;
        case DEEP_GEMM_NATIVE_BFLOAT16: return CUDA_R_16BF;
        case DEEP_GEMM_NATIVE_FLOAT32: return CUDA_R_32F;
        default: throw std::runtime_error("unsupported DeepGEMM native dtype");
    }
}

DeviceState& GetDeviceState(int device) {
    auto [it, inserted] = g_devices.try_emplace(device);
    if (!inserted) return it->second;
    CheckCublas(cublasLtCreate(&it->second.handle), "cublasLtCreate");
    try {
        CheckCuda(cudaMalloc(&it->second.workspace, kWorkspaceBytes), "cudaMalloc(workspace)");
    } catch (...) {
        cublasLtDestroy(it->second.handle);
        g_devices.erase(it);
        throw;
    }
    return it->second;
}

void DestroyStates() {
    for (auto& [device, state] : g_devices) {
        (void)device;
        if (state.workspace != nullptr) cudaFree(state.workspace);
        if (state.handle != nullptr) cublasLtDestroy(state.handle);
    }
}

struct StateCleanup {
    ~StateCleanup() { DestroyStates(); }
};
StateCleanup g_cleanup;

void Validate(const deep_gemm_native_tensor_view* tensor, const char* name) {
    if (tensor == nullptr || tensor->data == nullptr) {
        throw std::runtime_error(std::string(name) + " is null");
    }
    if (tensor->rows < 0 || tensor->cols < 0 || tensor->row_stride < tensor->cols) {
        throw std::runtime_error(std::string(name) + " has invalid row-major shape/stride");
    }
}

void Run(const deep_gemm_native_tensor_view& a,
         const deep_gemm_native_tensor_view& b,
         const deep_gemm_native_tensor_view& d,
         void* stream,
         bool accumulate) {
    Validate(&a, "A");
    Validate(&b, "B");
    Validate(&d, "D");
    if (a.cols != b.cols || d.rows != a.rows || d.cols != b.rows) {
        throw std::runtime_error("DeepGEMM native GEMM shape mismatch");
    }
    if (a.dtype != b.dtype || a.dtype != d.dtype) {
        throw std::runtime_error("DeepGEMM native GEMM dtype mismatch");
    }
    if (a.rows == 0 || b.rows == 0) return;

    int device = 0;
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceState& state = GetDeviceState(device);

    const int64_t m = d.rows;
    const int64_t n = d.cols;
    const int64_t k = a.cols;
    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_d = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    try {
        const cudaDataType_t type = CudaType(a.dtype);
        // Row-major A[M,K], B[N,K], D[M,N] are represented as column-major
        // KxM, KxN, NxM.  Compute (N,K) @ (K,M) -> (N,M).
        CheckCublas(cublasLtMatrixLayoutCreate(&layout_a, type, k, n, b.row_stride), "layout B");
        CheckCublas(cublasLtMatrixLayoutCreate(&layout_b, type, k, m, a.row_stride), "layout A");
        CheckCublas(cublasLtMatrixLayoutCreate(&layout_d, type, n, m, d.row_stride), "layout D");
        const cublasOperation_t trans_a = CUBLAS_OP_T;
        const cublasOperation_t trans_b = CUBLAS_OP_N;
        const cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;
        const cudaDataType_t scale = CUDA_R_32F;
        CheckCublas(cublasLtMatmulDescCreate(&desc, compute, scale), "matmul desc");
        CheckCublas(cublasLtMatmulDescSetAttribute(
            desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)), "set trans A");
        CheckCublas(cublasLtMatmulDescSetAttribute(
            desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)), "set trans B");
        uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        CheckCublas(cublasLtMatmulPreferenceCreate(&preference), "preference");
        CheckCublas(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceBytes, sizeof(kWorkspaceBytes)), "set workspace");
        CheckCublas(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &reduction, sizeof(reduction)), "set reduction");
        cublasLtMatmulHeuristicResult_t heuristic{};
        int count = 0;
        CheckCublas(cublasLtMatmulAlgoGetHeuristic(
            state.handle, desc, layout_a, layout_b, layout_d, layout_d,
            preference, 1, &heuristic, &count), "heuristic");
        if (count != 1) throw std::runtime_error("cuBLASLt returned no GEMM algorithm");
        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        CheckCublas(cublasLtMatmul(
            state.handle, desc, &alpha, b.data, layout_a, a.data, layout_b,
            &beta, d.data, layout_d, d.data, layout_d, &heuristic.algo,
            state.workspace, kWorkspaceBytes,
            reinterpret_cast<cudaStream_t>(stream)), "cublasLtMatmul");
    } catch (...) {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (layout_d) cublasLtMatrixLayoutDestroy(layout_d);
        if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
        if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
        throw;
    }
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layout_d);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
}

}  // namespace

extern "C" int deep_gemm_native_cublaslt_gemm_nn(
    const deep_gemm_native_tensor_view* a,
    const deep_gemm_native_tensor_view* b,
    const deep_gemm_native_tensor_view* d,
    void* stream,
    int accumulate) {
    try {
        g_last_error.clear();
        if (a == nullptr || b == nullptr || d == nullptr) {
            throw std::runtime_error("DeepGEMM native GEMM tensor view is null");
        }
        Run(*a, *b, *d, stream, accumulate != 0);
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    } catch (...) {
        g_last_error = "unknown DeepGEMM native error";
        return -1;
    }
}

extern "C" const char* deep_gemm_native_last_error(void) {
    return g_last_error.c_str();
}
