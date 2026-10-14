#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/operators.cuh>
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
          typename epilogue_operator_t,
          typename smem_t>
CUTLASS_DEVICE void
sm100_store_cd_swap_ab(smem_t& smem, uint32_t& tma_stage_idx,
                       const uint32_t& tmem_base_addr,
                       const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                       const bool& is_empty_group,
                       const uint32_t& effective_m,
                       const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                       epilogue_operator_t& epilogue_operator,
                       const bool& reverse_store_order,
                       const cutlass::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
                       const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                       const cute::TmaDescriptor& tensor_map_cd) {
    using cd_dtype_t = typename smem_t::cd_dtype;
    // NOTES: The epilogue requires a full warpgroup to read all 128 TMEM rows,
    //          implying STORE_BLOCK_N must be 128.
    DG_STATIC_ASSERT(STORE_BLOCK_N == 128, "STORE_BLOCK_N must be 128 to match TMEM rows");

    // Whether to cast D into FP8 with dynamic per-`kSFGranN` UE8M0 SFs (see `CastToFP8WithClamp`)
    constexpr bool kIsFP8CastWithClamp = epilogue_operator_t::kIsFP8CastWithClamp;
    constexpr uint32_t kSFGranN = operators::CastToFP8WithClamp::kSFGranN;

    // TMA checks
    constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    DG_STATIC_ASSERT(kSwizzleCDMode == 128, "TMA D must be 128B swizzled");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(not kIsFP8CastWithClamp or (STORE_BLOCK_M == 16 and kNumUMMAStoreThreads == 128),
                     "Swap-AB FP8 output requires one warpgroup and a 16x128 store shape");
    DG_STATIC_ASSERT(not epilogue_operator_t::kWithStochasticRounding or
                     (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t> and not kWithAccumulation),
                     "Stochastic rounding only supports direct BF16 output");

    // Iterate over M blocks. The scheduler skips empty blocks and aligns dynamic effective M to STORE_BLOCK_M.
    const auto num_stores = effective_m / STORE_BLOCK_M;
    for (uint32_t s = 0; s < num_stores; ++ s) {
        const auto store_idx = reverse_store_order ? num_stores - 1 - s : s;
        // Wait shared memory to be released
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<kNumTMAStoreStages - 1>();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

        // Warps cooperatively write an atomic block to shared memory.
        DG_STATIC_ASSERT(STORE_BLOCK_N_ATOM % 32 == 0, "Invalid block sizes");
        constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
        const auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]) +
                                   (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;

        // Prepare all SFs before casting the fragments, allowing independent reductions to overlap.
        constexpr uint32_t kNumTmemLoads = STORE_BLOCK_M / kNumSwizzleAtomRows;
        nv_bfloat162 values_bf16x2[kNumTmemLoads][kNumSwizzleAtomRows / 2];
        uint2 sf_exps[kNumTmemLoads];
        #pragma unroll
        for (uint32_t load_iter_idx = 0; load_iter_idx < kNumTmemLoads; ++ load_iter_idx) {
            const auto load_idx = reverse_store_order ? kNumTmemLoads - 1 - load_iter_idx : load_iter_idx;
            uint32_t tmem_addr = tmem_base_addr +
                                 store_idx * STORE_BLOCK_M +    // Store stage offset
                                 load_idx * kNumSwizzleAtomRows; // In-block offset
            uint32_t values[kNumSwizzleAtomRows];

            const auto smem_atom_ptr = smem_base_ptr + load_idx * kNumSwizzleAtomRows * kSwizzleCDMode;

            // Read from tensor memory into registers
            // NOTES: empty groups also read the accumulator (safe, as at least one UMMA is
            //        issued) and select zeros afterwards, keeping the loads branchless
            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                // NOTES: the FP32 store does not use STSM, so the plain `.32x32b` layout works
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, values[0], values[1], values[2], values[3],
                                                                 values[4], values[5], values[6], values[7]);
            } else {
                // Load from TMEM using `.16x256b` shape to satisfy the STSM layout requirements
                // (`.b16` for the BF16 store, `.b8` for the FP8 store): each lane receives
                // 2 rows (TMEM columns `2 * (lane % 4)` onwards) x 2 columns (dps `lane / 4` and `lane / 4 + 8`)
                // Start from lane index 0
                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                       values[0], values[1], values[2], values[3]);
                // Start from lane index 16
                cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                       values[4], values[5], values[6], values[7]);
            }
            cutlass::arch::fence_view_async_tmem_load();
            epilogue_operator.apply_values(values);
            #pragma unroll
            for (uint32_t value_idx = 0; value_idx < kNumSwizzleAtomRows; ++ value_idx)
                values[value_idx] = is_empty_group ? 0u : values[value_idx];

            // Notify the MMA warp once all overlapped TMEM columns have been read,
            // before any of the cast/store work
            constexpr auto kNumOverlapLoads = math::constexpr_ceil_div(kNumOverlappedTmemCols, kNumSwizzleAtomRows);
            if constexpr (kNumOverlapLoads > 0) {
                // NOTES: dynamic UMMA N writes only `effective_m` columns, so compile-time overlap
                //        columns beyond the available TMEM loads were never produced
                const auto num_loads_before_arrive = cute::min<uint32_t>(kNumOverlapLoads, num_stores * kNumTmemLoads);
                if (s * kNumTmemLoads + load_iter_idx + 1 == num_loads_before_arrive) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_overlap_barrier->arrive(0u);
                }
            }
            // Notify tensor memory empty (only at the leader CTA) as soon as all loads are in registers
            if (s == num_stores - 1 and load_iter_idx == kNumTmemLoads - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            // Store into shared memory
            if constexpr (kIsFP8CastWithClamp) {
                const auto row_idx = base_m_idx + store_idx * STORE_BLOCK_M + load_idx * kNumSwizzleAtomRows +
                                     2 * (lane_idx % 4);
                const auto col_idx = base_n_idx + epilogue_warp_idx * 32 + lane_idx / 16 * 4 + lane_idx / 4 % 4;
                epilogue_operator.template prepare_fp8<true>(values, values_bf16x2[load_iter_idx], sf_exps[load_iter_idx], row_idx, col_idx);
            } else if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                uint32_t col = lane_idx / 4;

                #pragma unroll
                for (uint32_t row = 0; row < kNumSwizzleAtomRows; ++ row) {
                    auto smem_ptr = smem_atom_ptr + row * (kNumBankGroupBytes * 8)
                                                  + (col ^ row) * kNumBankGroupBytes
                                                  + (lane_idx % 4) * sizeof(float);
                    ptx::st_shared(reinterpret_cast<uint32_t*>(smem_ptr), values[row]);
                }
            } else {
                // Destination shared memory address
                uint32_t row = lane_idx % 8;
                uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                auto smem_ptr = smem_atom_ptr + row * (kNumBankGroupBytes * 8)
                                              + (col ^ row) * kNumBankGroupBytes;

                // Cast and store matrix with transposition
                uint32_t packed[4];
                epilogue_operator.template cast_bf16<true>(values, packed, base_n_idx + epilogue_warp_idx * 32, lane_idx);
                ptx::SM90_U32x4_STSM_T<uint32_t>::copy(packed[0], packed[1], packed[2], packed[3], smem_ptr);
            }
        }

        if constexpr (kIsFP8CastWithClamp) {
            #pragma unroll
            for (uint32_t load_iter_idx = 0; load_iter_idx < kNumTmemLoads; ++ load_iter_idx) {
                const auto load_idx = reverse_store_order ? kNumTmemLoads - 1 - load_iter_idx : load_iter_idx;
                const auto smem_atom_ptr = smem_base_ptr + load_idx * kNumSwizzleAtomRows * kSwizzleCDMode;
                const auto row_idx = base_m_idx + store_idx * STORE_BLOCK_M + load_idx * kNumSwizzleAtomRows +
                                     2 * (lane_idx % 4);
                uint32_t packed[kNumSwizzleAtomRows / 4];
                epilogue_operator.cast_fp8(values_bf16x2[load_iter_idx], packed, sf_exps[load_iter_idx]);

                // Transpose the packed FP8 fragment into the TMA shared-memory layout.
                const auto smem_ptr = smem_atom_ptr + (lane_idx % 8) * kSwizzleCDMode +
                                      ((epilogue_warp_idx * 2 + lane_idx / 8) ^ (lane_idx % 8)) * kNumBankGroupBytes;
                ptx::SM100_U8x8_STSM_T<uint32_t>::copy(packed[0], packed[1], smem_ptr);

                // Lanes 0-3 store the even row's SF; lanes 4-7 store the odd row's SF.
                if (lane_idx < kNumSwizzleAtomRows)
                    epilogue_operator.store_sf(row_idx + lane_idx / 4,
                                               base_n_idx + epilogue_warp_idx * kSFGranN, batch_idx,
                                               static_cast<uint8_t>(lane_idx < 4 ? sf_exps[load_iter_idx].x : sf_exps[load_iter_idx].y));
            }
        }

        // Synchronize all threads and issue TMA
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ i) {
                auto smem_ptr = smem.cd[tma_stage_idx] + i * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                uint32_t m_idx = base_m_idx + store_idx * STORE_BLOCK_M;
                uint32_t n_idx = base_n_idx + i * STORE_BLOCK_N_ATOM;

                // Issue 2D or 3D TMA store
                if constexpr (kGemmType == GemmType::Batched or is_k_grouped_contiguous(kGemmType)) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                                           cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                                           cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx);
                }
            }
            cute::tma_store_arrive();
        }
        __syncwarp();
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    }
}

} // namespace deep_gemm::epilogue
