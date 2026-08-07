#include "native_api.h"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace {

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

struct StreamState {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    std::mutex mutex;
};

struct StreamKey {
    uintptr_t stream;
    size_t thread;

    bool operator==(const StreamKey& other) const {
        return stream == other.stream && thread == other.thread;
    }
};

struct StreamKeyHash {
    size_t operator()(const StreamKey& key) const {
        return std::hash<uintptr_t>{}(key.stream) ^
               (std::hash<size_t>{}(key.thread) << 1);
    }
};

struct DeviceState {
    std::unordered_map<StreamKey, std::unique_ptr<StreamState>, StreamKeyHash> streams;
};

constexpr size_t kMaxStreamStatesPerDevice = 8;
std::mutex g_devices_mutex;
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

StreamKey MakeStreamKey(cudaStream_t stream) {
    // cudaStreamPerThread has one logical stream per host thread even though
    // the opaque handle value is shared. Explicit streams are process-wide.
    const size_t thread = stream == cudaStreamPerThread
        ? std::hash<std::thread::id>{}(std::this_thread::get_id()) : 0;
    return {reinterpret_cast<uintptr_t>(stream), thread};
}

StreamState& GetStreamState(int device, cudaStream_t stream) {
    auto [device_it, inserted] = g_devices.try_emplace(device);
    (void)inserted;
    DeviceState& device_state = device_it->second;
    const StreamKey stream_key = MakeStreamKey(stream);
    auto stream_it = device_state.streams.find(stream_key);
    if (stream_it != device_state.streams.end()) return *stream_it->second;
    if (device_state.streams.size() >= kMaxStreamStatesPerDevice) {
        throw std::runtime_error(
            "DeepGEMM native stream-state limit reached; call "
            "deep_gemm_native_release_stream before creating more streams");
    }

    auto state = std::make_unique<StreamState>();
    CheckCublas(cublasLtCreate(&state->handle), "cublasLtCreate");
    try {
        CheckCuda(cudaMalloc(&state->workspace, kWorkspaceBytes), "cudaMalloc(workspace)");
    } catch (...) {
        cublasLtDestroy(state->handle);
        throw;
    }
    StreamState* result = state.get();
    device_state.streams.emplace(stream_key, std::move(state));
    return *result;
}

void DestroyStates() {
    std::lock_guard<std::mutex> devices_lock(g_devices_mutex);
    for (auto& [device, state] : g_devices) {
        (void)device;
        for (auto& [stream, stream_state] : state.streams) {
            (void)stream;
            std::lock_guard<std::mutex> stream_lock(stream_state->mutex);
            if (stream_state->workspace != nullptr) cudaFree(stream_state->workspace);
            if (stream_state->handle != nullptr) cublasLtDestroy(stream_state->handle);
        }
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
         bool accumulate,
         bool b_transposed) {
    Validate(&a, "A");
    Validate(&b, "B");
    Validate(&d, "D");
    const int64_t k = b_transposed ? b.cols : b.rows;
    const int64_t n = b_transposed ? b.rows : b.cols;
    if (a.cols != k || d.rows != a.rows || d.cols != n) {
        throw std::runtime_error("DeepGEMM native GEMM shape mismatch");
    }
    if (a.dtype != b.dtype || a.dtype != d.dtype) {
        throw std::runtime_error("DeepGEMM native GEMM dtype mismatch");
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (d.rows == 0 || d.cols == 0) return;
    if (k == 0) {
        if (!accumulate) {
            const size_t element_bytes = a.dtype == DEEP_GEMM_NATIVE_FLOAT32 ? 4 : 2;
            CheckCuda(cudaMemset2DAsync(
                d.data, d.row_stride * element_bytes, 0,
                d.cols * element_bytes, d.rows, cuda_stream),
                "cudaMemset2DAsync(empty GEMM)");
        }
        return;
    }

    int device = 0;
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    // Keep the map locked until the per-stream lock is held. This prevents a
    // concurrent release call from erasing the map entry between pointer
    // lookup and lock acquisition.
    std::unique_lock<std::mutex> devices_lock(g_devices_mutex);
    StreamState* state = &GetStreamState(device, cuda_stream);
    std::unique_lock<std::mutex> stream_lock(state->mutex);
    devices_lock.unlock();

    const int64_t m = d.rows;
    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_d = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    try {
        const cudaDataType_t type = CudaType(a.dtype);
        // Row-major matrices are represented as column-major transposes. The
        // public ABI uses the historical B[N,K] row-major layout and computes
        // A @ B.T. The internal non-transposed branch remains available only
        // to keep the implementation straightforward for future versioned
        // APIs.
        const int64_t b_layout_rows = b_transposed ? k : n;
        const int64_t b_layout_cols = b_transposed ? n : k;
        CheckCublas(cublasLtMatrixLayoutCreate(
            &layout_a, type, b_layout_rows, b_layout_cols, b.row_stride), "layout B");
        CheckCublas(cublasLtMatrixLayoutCreate(&layout_b, type, k, m, a.row_stride), "layout A");
        CheckCublas(cublasLtMatrixLayoutCreate(&layout_d, type, n, m, d.row_stride), "layout D");
        const cublasOperation_t trans_a = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t trans_b = CUBLAS_OP_N;
        const cublasComputeType_t compute =
            a.dtype == DEEP_GEMM_NATIVE_FLOAT32
                ? CUBLAS_COMPUTE_32F_FAST_TF32
                : CUBLAS_COMPUTE_32F;
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
            state->handle, desc, layout_a, layout_b, layout_d, layout_d,
            preference, 1, &heuristic, &count), "heuristic");
        if (count != 1) throw std::runtime_error("cuBLASLt returned no GEMM algorithm");
        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        CheckCublas(cublasLtMatmul(
            state->handle, desc, &alpha, b.data, layout_a, a.data, layout_b,
            &beta, d.data, layout_d, d.data, layout_d, &heuristic.algo,
            state->workspace, kWorkspaceBytes, cuda_stream), "cublasLtMatmul");
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
        Run(*a, *b, *d, stream, accumulate != 0, true);
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    } catch (...) {
        g_last_error = "unknown DeepGEMM native error";
        return -1;
    }
}

extern "C" int deep_gemm_native_release_stream(void* stream) {
    try {
        g_last_error.clear();
        int device = 0;
        CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
        const StreamKey key = MakeStreamKey(reinterpret_cast<cudaStream_t>(stream));
        std::lock_guard<std::mutex> devices_lock(g_devices_mutex);
        auto device_it = g_devices.find(device);
        if (device_it == g_devices.end()) return 0;
        auto stream_it = device_it->second.streams.find(key);
        if (stream_it == device_it->second.streams.end()) return 0;
        StreamState& state = *stream_it->second;
        std::lock_guard<std::mutex> stream_lock(state.mutex);
        CheckCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                  "cudaStreamSynchronize(release stream)");
        if (state.workspace != nullptr) {
            CheckCuda(cudaFree(state.workspace), "cudaFree(workspace)");
            state.workspace = nullptr;
        }
        if (state.handle != nullptr) {
            CheckCublas(cublasLtDestroy(state.handle), "cublasLtDestroy");
            state.handle = nullptr;
        }
        device_it->second.streams.erase(stream_it);
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
