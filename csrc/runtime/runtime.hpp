#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <cublasLt.h>
#include <torch/python.h>

#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/lazy.hpp>

#include "../utils/exception.hpp"
#include "jit.hpp"
#include "locality_domain.hpp"

namespace deep_gemm {

class Runtime {
public:
    int num_sms = 0, tc_util = 0;

    // cuBLASLt utils
    // cuBLAS will select worse heuristics when workspace > 16 MiB with shape, e.g. m=128, n=7168, k=16384
    static constexpr size_t kCublasLtWorkspaceSize = 16 * 1024 * 1024;
    // Bound the per-stream cache. cuBLASLt may use the workspace asynchronously, so concurrent
    // streams cannot share one tensor, but an unbounded map pins a caching-allocator block for
    // every stream that ever called into cuBLASLt.
    static constexpr size_t kMaxCachedCublasLtWorkspaces = 16;
    // `c10::cuda::CUDAStream` is a non-owning {device, id} value; storing it as a map key does not
    // keep the underlying `cudaStream_t` alive, and destroyed streams cannot be queried safely.
    // Key by the raw stream handle and evict LRU entries so dropped/ephemeral streams can release.
    struct CachedCublasLtWorkspace {
        torch::Tensor tensor;
        std::list<cudaStream_t>::iterator lru_it;
    };
    std::list<cudaStream_t> cublaslt_workspace_lru;
    std::unordered_map<cudaStream_t, CachedCublasLtWorkspace> cublaslt_workspaces;

    // Create the cuBLASLt handle ourselves
    cublasLtHandle_t cublaslt_handle;
    bool use_pytorch_managed_cublaslt_handle;
    bool use_temp_cublaslt_workspace;

    // The locality domain of every SM, probed lazily
    torch::Tensor sm_locality_domains;
    // We need to balance the SMs across locality domains to provide simpler load balancing
    torch::Tensor balanced_sm_locality_domains;

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

        // Capture streams are often created per graph and discarded. A cached tensor would also be
        // captured by pointer, so put scratch in the graph mempool instead of inserting a cache entry.
        if (c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None)
            return torch::empty({kCublasLtWorkspaceSize}, options);

        const cudaStream_t raw = stream.stream();
        if (const auto cached = cublaslt_workspaces.find(raw); cached != cublaslt_workspaces.end()) {
            cublaslt_workspace_lru.splice(
                cublaslt_workspace_lru.begin(), cublaslt_workspace_lru, cached->second.lru_it);
            return cached->second.tensor;
        }

        if (cublaslt_workspaces.size() >= kMaxCachedCublasLtWorkspaces) {
            cublaslt_workspaces.erase(cublaslt_workspace_lru.back());
            cublaslt_workspace_lru.pop_back();
        }

        auto workspace = torch::empty({kCublasLtWorkspaceSize}, options);
        cublaslt_workspace_lru.push_front(raw);
        cublaslt_workspaces.emplace(raw, CachedCublasLtWorkspace{workspace, cublaslt_workspace_lru.begin()});
        return workspace;
    }

    size_t num_cached_cublaslt_workspaces() const {
        return cublaslt_workspaces.size();
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
