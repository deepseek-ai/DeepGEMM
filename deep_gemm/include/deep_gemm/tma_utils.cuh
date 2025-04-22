#pragma once

#ifndef NVRTC_JIT_COMPILATION
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#endif

#include <cuda/barrier>

#include "utils.cuh"

namespace deep_gemm {

template <uint32_t kNumTMAMulticast = 1>
__device__ __forceinline__ void
tma_copy(void const* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr,
         int32_t const& crd_0, int32_t const& crd_1) {
    constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    if constexpr (kNumTMAMulticast == 1) {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
    } else if (cute::block_rank_in_cluster() == 0) {
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr, (1 << kNumTMAMulticast) - 1, cache_hint, smem_ptr, crd_0, crd_1);
    }
}

}  // namespace deep_gemm
