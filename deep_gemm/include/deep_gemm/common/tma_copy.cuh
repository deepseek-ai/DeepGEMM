#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/packing.cuh>

namespace deep_gemm::tma {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, uint32_t kPackFactor, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER / kPackFactor : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t kNumDimensions> struct LoadOps;
template <> struct LoadOps<2> {
    using Load = cute::SM90_TMA_LOAD_2D;
    using Load2SM = cute::SM100_TMA_2SM_LOAD_2D;
    using LoadMulticast = cute::SM90_TMA_LOAD_MULTICAST_2D;
};
template <> struct LoadOps<3> {
    using Load = cute::SM90_TMA_LOAD_3D;
    using Load2SM = cute::SM100_TMA_2SM_LOAD_3D;
    using LoadMulticast = cute::SM90_TMA_LOAD_MULTICAST_3D;
};
template <> struct LoadOps<4> {
    using Load = cute::SM90_TMA_LOAD_4D;
    using Load2SM = cute::SM100_TMA_2SM_LOAD_4D;
    using LoadMulticast = cute::SM90_TMA_LOAD_MULTICAST_4D;
};
template <> struct LoadOps<5> {
    using Load = cute::SM90_TMA_LOAD_5D;
    using Load2SM = cute::SM100_TMA_2SM_LOAD_5D;
    using LoadMulticast = cute::SM90_TMA_LOAD_MULTICAST_5D;
};

// Load a `BLOCK_OUTER x BLOCK_INNER` box of a `kNumDimensions`-D (2 to 5) tensor map at `(inner_idx, outer_coords...)`
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, uint32_t kNumDimensions,
          uint64_t kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
          typename... coord_ts>
CUTLASS_DEVICE void
copy_nd(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
        dtype_t* smem_ptr, const uint32_t& num_tma_multicast,
        const uint32_t& inner_idx, const coord_ts&... outer_coords) {
    DG_STATIC_ASSERT(sizeof...(coord_ts) + 1 == kNumDimensions, "Invalid coordinates");
    DG_STATIC_ASSERT(static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL) ==
                     static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), "Invalid cache hint");
    using Ops = LoadOps<kNumDimensions>;
    constexpr uint32_t kPackFactor = get_smem_pack_factor<dtype_t>();
    // NOTES: for sub-byte packed types, the SMEM atom stride stays in
    //        bytes (`sizeof(dtype_t) == 1`), while the GMEM index and loop count run over logical elements
    DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1, "Packing expects a 1-byte storage type");
    constexpr uint32_t BLOCK_INNER_ATOM_STORAGE = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, kPackFactor, dtype_t>();
    constexpr uint32_t BLOCK_INNER_ATOM_LOGICAL = BLOCK_INNER_ATOM_STORAGE * kPackFactor;
    DG_STATIC_ASSERT(BLOCK_INNER % BLOCK_INNER_ATOM_LOGICAL == 0, "TMA inner block must contain whole atoms");

    if (num_tma_multicast == 1) {
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
            Ops::Load::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr), kCacheHint,
                            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                            inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_coords...);
        }
    } else {
        #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
            // 2-CTA function will send signals to the leader CTA only
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                Ops::Load2SM::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr), kCacheHint,
                                   smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                   inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_coords...);
            }
        #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
            if (cute::block_rank_in_cluster() == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL; ++ i) {
                    Ops::LoadMulticast::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                             (1 << num_tma_multicast) - 1, kCacheHint,
                                             smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM_STORAGE,
                                             inner_idx + i * BLOCK_INNER_ATOM_LOGICAL, outer_coords...);
                }
            }
        #endif
    }
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false,
          uint64_t kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL)>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    if constexpr (kIs3DTMA) {
        copy_nd<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype_t, 3, kCacheHint>(desc_ptr, barrier_ptr, smem_ptr, num_tma_multicast, inner_idx, outer_idx, batch_idx);
    } else {
        copy_nd<BLOCK_INNER, BLOCK_OUTER, kSwizzleMode, dtype_t, 2, kCacheHint>(desc_ptr, barrier_ptr, smem_ptr, num_tma_multicast, inner_idx, outer_idx);
    }
}

} // namespace deep_gemm::tma
