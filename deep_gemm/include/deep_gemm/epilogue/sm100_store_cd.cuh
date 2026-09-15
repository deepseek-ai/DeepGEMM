#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          uint32_t kNumOverlappedTmemCols,
          GemmType kGemmType, bool kWithAccumulation,
          typename epilogue_op_t,
          typename smem_t>
CUTLASS_DEVICE void
sm100_store_cd(smem_t& smem, uint32_t& tma_stage_idx,
               const uint32_t& tmem_base_addr,
               const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
               const bool& is_empty_group,
               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
               const epilogue_op_t& epilogue_op,
               const bool& reverse_store_order,
               const cutlass::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
               const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
               const cute::TmaDescriptor& tensor_map_cd) {
    using cd_dtype_t = typename smem_t::cd_dtype;
    // Whether to cast D into FP8 with dynamic per-`kSFGranN` UE8M0 SFs (see `EpilogueDynamicScaledFP8`)
    constexpr bool kWithOutputSF = cute::is_same_v<epilogue_op_t, transform::EpilogueDynamicScaledFP8>;
    constexpr uint32_t kSFGranN = transform::EpilogueDynamicScaledFP8::kSFGranN;
    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(not kWithOutputSF or cute::is_same_v<cd_dtype_t, cutlass::float_e4m3_t>, "FP8 output requires an E4M3 D");
    DG_STATIC_ASSERT(not kWithOutputSF or STORE_BLOCK_N % kSFGranN == 0, "A store must cover complete SF groups");

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
            const auto store_idx = reverse_store_order ? kNumStores - 1 - s : s;
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]);

            // Swizzled shared memory address of the `bank_group_idx`-th bank group in this
            // warp's atom, reshaping the atom in another view:
            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
            const auto get_swizzled_smem_ptr = [&](const uint32_t& bank_group_idx) {
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                const auto shifted_idx = bank_group_idx + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                auto row = kHasShortcut ? (bank_group_idx / 8 + lane_idx) : (shifted_idx / 8);
                auto col = kHasShortcut ? bank_group_idx : (shifted_idx % 8);
                col ^= row % (kSwizzleCDMode / 16);
                return smem_base_ptr +                                              // Base pointer
                       epilogue_warp_idx * 32 * kSwizzleCDMode +                    // Warp offset
                       row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;   // In-atom offset
            };

            // Wait shared memory to be released
            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            // The pipeline stage
            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_op_t::apply_index_n<STORE_BLOCK_N>(base_n_idx + store_idx * STORE_BLOCK_N);

            // Iterate over TMEM loads: each covers one swizzled bank group per lane, or one
            // complete SF group per lane (one row, `kSFGranN` columns) with FP8 output
            constexpr uint32_t kNumElemsPerLoad = kWithOutputSF ? kSFGranN : kNumElemsPerBankGroup;
            constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerLoad;
            constexpr auto kNumOverlapLoads = math::constexpr_ceil_div(kNumOverlappedTmemCols, kNumElemsPerLoad);
            DG_STATIC_ASSERT(kNumOverlapLoads <= kNumLoads, "The first store block must cover all overlapped columns");

            // A store covering a whole packed word (with word-aligned batches) writes all 4 SF bytes at once
            const auto row_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
            const bool store_whole_sf_word = kWithOutputSF and kNumLoads == 4 and epilogue_op.shape_n % (4 * kSFGranN) == 0;
            uint32_t sf_word = 0;

            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                const auto load_idx = reverse_store_order ? kNumLoads - 1 - i : i;
                const auto tmem_addr = tmem_base_addr +                                     // Accumulator offset
                                       w * BLOCK_N +                                        // Wave offset
                                       store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerLoad; // In-block offset

                // Read from tensor memory into registers
                // NOTES: empty groups also read the accumulator (safe, as at least one UMMA is
                //        issued) and select zeros afterwards, keeping the loads branchless
                uint32_t values[kNumElemsPerLoad];
                DG_STATIC_ASSERT(kNumElemsPerLoad == 4 or kNumElemsPerLoad == 8 or kNumElemsPerLoad == 32, "Invalid load width");
                using tmem_load_t = cute::conditional_t<kNumElemsPerLoad == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                                    cute::conditional_t<kNumElemsPerLoad ==  8, cute::SM100_TMEM_LOAD_32dp32b8x,
                                                                                cute::SM100_TMEM_LOAD_32dp32b4x>>;
                [&]<size_t... Is>(cute::index_sequence<Is...>) {
                    tmem_load_t::copy(tmem_addr, values[Is]...);
                }(cute::make_index_sequence<kNumElemsPerLoad>{});
                cutlass::arch::fence_view_async_tmem_load();
                epilogue_op.apply_values(values);
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < kNumElemsPerLoad; ++ value_idx)
                    values[value_idx] = is_empty_group ? 0u : values[value_idx];

                // Notify the MMA warp once all overlapped TMEM columns have been read,
                // before any of the cast/store work
                if constexpr (kNumOverlapLoads > 0) {
                    if (w == 0 and s == 0 and i + 1 == kNumOverlapLoads) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_overlap_barrier->arrive(0u);
                    }
                }
                // Notify tensor memory empty (only at the leader CTA) as soon as all loads are in registers
                if (w == kNumMWaves - 1 and s == kNumStores - 1 and i == kNumLoads - 1) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barrier->arrive(0u);
                }

                // Store into shared memory
                if constexpr (kWithOutputSF) {
                    // Round the accumulator into BF16 pairs first, so the output bitwise matches
                    // a BF16 D followed by the standalone cast (power-of-two scaling of BF16 is exact)
                    nv_bfloat162 values_bf16x2[kNumElemsPerLoad / 2];
                    #pragma unroll
                    for (uint32_t pair_idx = 0; pair_idx < kNumElemsPerLoad / 2; ++ pair_idx)
                        values_bf16x2[pair_idx] = math::cast_into_bf16x2(values[pair_idx * 2], values[pair_idx * 2 + 1]);

                    // Reduce the group amax and compute the SF (the SF exponent floor covers the amax clamp),
                    // then splat the BF16 sf_inv into both packed halves
                    const auto sf_exp = math::get_ue8m0_sf_exp<cd_dtype_t>(math::get_packed_bf16_amax(values_bf16x2));
                    const auto sf_inv = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp));

                    // Cast into FP8 and store into shared memory
                    constexpr uint32_t kNumBankGroupsPerLoad = kNumElemsPerLoad / kNumElemsPerBankGroup;
                    #pragma unroll
                    for (uint32_t bank_group_iter_idx = 0; bank_group_iter_idx < kNumBankGroupsPerLoad; ++ bank_group_iter_idx) {
                        const auto smem_ptr = get_swizzled_smem_ptr(load_idx * kNumBankGroupsPerLoad + bank_group_iter_idx);
                        const auto pair_idx = bank_group_iter_idx * (kNumElemsPerBankGroup / 2);
                        ptx::st_shared(smem_ptr,
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 0], values_bf16x2[pair_idx + 1], sf_inv, sf_inv),
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 2], values_bf16x2[pair_idx + 3], sf_inv, sf_inv),
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 4], values_bf16x2[pair_idx + 5], sf_inv, sf_inv),
                                       math::scale_bf16x2_into_fp8x4(values_bf16x2[pair_idx + 6], values_bf16x2[pair_idx + 7], sf_inv, sf_inv));
                    }

                    // Accumulate the SF byte into the packed word, or store it directly
                    if (store_whole_sf_word)
                        sf_word |= sf_exp << (load_idx * 8);
                    else
                        epilogue_op.store_sf(row_idx, n_idx + load_idx * kSFGranN, batch_idx, static_cast<uint8_t>(sf_exp));
                } else if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                    ptx::st_shared(get_swizzled_smem_ptr(load_idx), values[0], values[1], values[2], values[3]);
                } else {
                    ptx::st_shared(
                        get_swizzled_smem_ptr(load_idx),
                        math::cast_into_bf16_and_pack(values[0], values[1]),
                        math::cast_into_bf16_and_pack(values[2], values[3]),
                        math::cast_into_bf16_and_pack(values[4], values[5]),
                        math::cast_into_bf16_and_pack(values[6], values[7])
                    );
                }
            }
            if constexpr (kWithOutputSF) {
                if (store_whole_sf_word)
                    epilogue_op.store_sf(row_idx, n_idx, batch_idx, sf_word);
            }

            // Synchronize all threads and issue TMA
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched or is_k_grouped_contiguous(kGemmType)) {
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
