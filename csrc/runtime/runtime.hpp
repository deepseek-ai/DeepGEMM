#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/lazy.hpp>

#include "../utils/exception.hpp"
#include "jit.hpp"

namespace deep_gemm {

class Runtime {
public:
    int num_sms = 0, tc_util = 0;

    // cuBLASLt utils
    // cuBLAS will select worse heuristics when workspace > 16 MiB with shape, e.g. m=128, n=7168, k=16384
    static constexpr size_t kCublasLtWorkspaceSize = 16 * 1024 * 1024;
    // cuBLASLt may use the workspace asynchronously, so concurrent streams cannot share it.
    std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> cublaslt_workspaces;

    // Create the cuBLASLt handle ourselves
    cublasLtHandle_t cublaslt_handle;
    bool use_pytorch_managed_cublaslt_handle;
    bool use_temp_cublaslt_workspace;

    explicit Runtime() {
        // Whether to use PyTorch cuBLASLt
        // By default, we don't use it,
        // as `at::cuda::getCurrentCUDABlasLtHandle` has large CPU overhead with some PyTorch versions
        use_pytorch_managed_cublaslt_handle = deep_jit::get_env<int>("DG_USE_PYTORCH_CUBLASLT_HANDLE", 0) > 0;
        // Whether to create workspace tensor on each call instead of holding one.
        // Enabled by compute-sanitizer tests, which trigger `cudaErrorCudartUnloading`
        // when the workspace tensor is destructed after CUDA driver shutdown.
        use_temp_cublaslt_workspace = deep_jit::get_env<int>("DG_USE_TEMP_CUBLASLT_WORKSPACE", 0) > 0;

        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
    }

    ~Runtime() noexcept(false) {
        if (not use_pytorch_managed_cublaslt_handle)
            DG_CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));
    }

    cublasLtHandle_t get_cublaslt_handle() const {
        if (use_pytorch_managed_cublaslt_handle)
            return at::cuda::getCurrentCUDABlasLtHandle();

        // Self-managed handle
        return cublaslt_handle;
    }

    torch::Tensor get_cublaslt_workspace(const c10::cuda::CUDAStream& stream) {
        const auto options = dtype(torch::kByte).device(stream.device());
        if (use_temp_cublaslt_workspace)
            return torch::empty({kCublasLtWorkspaceSize}, options);

        auto& workspace = cublaslt_workspaces[stream];
        if (not workspace.defined())
            workspace = torch::empty({kCublasLtWorkspaceSize}, options);
        return workspace;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 < new_num_sms and new_num_sms <= jit->device.get_num_sms());
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = jit->device.get_num_sms();
        return num_sms;
    }

    bool is_cublaslt_available() {
        return get_num_sms() == jit->device.get_num_sms();
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }

};

inline auto runtime = deep_jit::LazyInit<Runtime>([](){ return std::make_shared<Runtime>(); });

}  // namespace deep_gemm
