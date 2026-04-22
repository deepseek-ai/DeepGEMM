#pragma once

#include <cublasLt.h>
#include <torch/version.h>
#include <ATen/cuda/CUDAContext.h>

#include "../utils/exception.hpp"
#include "../utils/lazy_init.hpp"

#define PYTORCH_SUPPORTS_GET_CUBLASLT_HANDLE (TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3))

namespace deep_gemm {

class DeviceRuntime {
    int num_sms = 0, tc_util = 0;
    bool enable_pdl = false;
    std::shared_ptr<cudaDeviceProp> cached_prop;

    // cuBLASLt utils
    static constexpr size_t kCublasLtWorkspaceSize = 32 * 1024 * 1024;

public:
    // Create the cuBLASLt handle ourselves
    cublasLtHandle_t cublaslt_handle;
    torch::Tensor cublaslt_workspace;
    bool use_pytorch_managed_cublaslt_handle;
    bool use_temp_cublaslt_workspace;

    explicit DeviceRuntime() {

        // Whether to use PyTorch cuBLASLt
        // By default, we don't use it,
        // as `at::cuda::getCurrentCUDABlasLtHandle` has large CPU overhead with some PyTorch versions
        use_pytorch_managed_cublaslt_handle = get_env<int>("DG_USE_PYTORCH_CUBLASLT_HANDLE", 0) > 0;
#if not PYTORCH_SUPPORTS_GET_CUBLASLT_HANDLE
        DG_HOST_ASSERT(not use_pytorch_managed_cublaslt_handle and "PyTorch does not support to get cuBLASLt handle");
#endif

        // Whether to create workspace tensor on each call instead of holding one.
        // Enabled by compute-sanitizer tests, which trigger `cudaErrorCudartUnloading`
        // when the workspace tensor is destructed after CUDA driver shutdown.
        use_temp_cublaslt_workspace = get_env<int>("DG_USE_TEMP_CUBLASLT_WORKSPACE", 0) > 0;

        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));

        if (not use_temp_cublaslt_workspace)
            cublaslt_workspace = torch::empty({kCublasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA));
    }

    ~DeviceRuntime() noexcept(false) {
        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    }

    cublasLtHandle_t get_cublaslt_handle() const {
#if PYTORCH_SUPPORTS_GET_CUBLASLT_HANDLE
        if (use_pytorch_managed_cublaslt_handle)
            return at::cuda::getCurrentCUDABlasLtHandle();
#endif

        // Self-managed handle
        return cublaslt_handle;
    }

    torch::Tensor get_cublaslt_workspace() const {
        if (use_temp_cublaslt_workspace)
            return torch::empty({kCublasLtWorkspaceSize}, dtype(torch::kByte).device(at::kCUDA));
        return cublaslt_workspace;
    }

    std::shared_ptr<cudaDeviceProp> get_prop() {
        if (cached_prop == nullptr) {
            int device_idx;
            cudaDeviceProp prop;
            DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
            DG_CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_idx));
            cached_prop = std::make_shared<cudaDeviceProp>(prop);
        }
        return cached_prop;
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch(const bool& number_only = false,
                         const bool& support_arch_family = false) {
        const auto [major, minor] = get_arch_pair();
        if (major == 10 and minor != 1) {
            if (number_only)
                return "100";
            return support_arch_family ? "100f" : "100a";
        }
        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");
    }

    int get_arch_major() {
        return get_arch_pair().first;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 <= new_num_sms and new_num_sms <= get_prop()->multiProcessorCount);
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = get_prop()->multiProcessorCount;
        return num_sms;
    }

    int get_l2_cache_size() {
        return get_prop()->l2CacheSize;
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

    void set_pdl(const bool& new_enable_pdl) {
        enable_pdl = new_enable_pdl;
    }

    bool get_pdl() const {
        return enable_pdl;
    }
};

static auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });

} // namespace deep_gemm
