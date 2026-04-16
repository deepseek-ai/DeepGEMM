#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_cd(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
               const uint32_t& tmem_base_addr,
               const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
               const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
               const cute::TmaDescriptor& tensor_map_cd) {
    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    // Share store pipeline between blocks
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over M waves
    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        // Issue every swizzled atom and pipeline STSM and TMA store
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            // Wait shared memory to be released
            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            // The pipeline stage
            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_type_t::apply_index_n<STORE_BLOCK_N>(base_n_idx + s * STORE_BLOCK_N);

            // Store into shared memory
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                // Calculate the index of the bank group to be written in the atom
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                // Reshape the atom in another view and swizzle
                //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
                //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
                // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                // Source and destination memory address
                uint32_t tmem_addr = tmem_base_addr +                                       // Accumulator offset
                                     w * BLOCK_N +                                          // Wave offset
                                     s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
                auto smem_ptr = smem_base_ptr +                                             // Base pointer
                                epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp offset
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

                // Load from tensor memory, store into shared memory
                uint32_t values[kNumElemsPerBankGroup];
                if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                    // For FP32 output, read and store
                    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3]);
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                } else {
                    // For BF16 output, read, cast and store
                    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                    cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::st_shared(
                        smem_ptr,
                        math::cast_into_bf16_and_pack(values[0], values[1]),
                        math::cast_into_bf16_and_pack(values[2], values[3]),
                        math::cast_into_bf16_and_pack(values[4], values[5]),
                        math::cast_into_bf16_and_pack(values[6], values[7])
                    );
                }
            }

            // Notify tensor memory empty (only at the leader CTA) arrival ASAP
            // NOTES: only the last stage needs to do this
            if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            // Synchronize all threads and issue TMA
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

} // namespace deep_gemm::epilogue
