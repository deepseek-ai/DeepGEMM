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
sm100_store_cd_swap_ab(smem_t& smem, uint32_t& tma_stage_idx,
                       const uint32_t& tmem_base_addr,
                       const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                       const bool& is_empty_group,
                       const uint32_t& effective_m,
                       const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                       const epilogue_op_t& epilogue_op,
                       const bool& reverse_store_order,
                       const cutlass::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
                       const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                       const cute::TmaDescriptor& tensor_map_cd) {
    using cd_dtype_t = typename smem_t::cd_dtype;
    // NOTES: The epilogue requires a full warpgroup to read all 128 TMEM rows,
    //          implying STORE_BLOCK_N must be 128.
    DG_STATIC_ASSERT(STORE_BLOCK_N == 128, "STORE_BLOCK_N must be 128 to match TMEM rows");

    // Whether to cast D into FP8 with dynamic per-`kSFGranN` UE8M0 SFs (see `EpilogueDynamicScaledFP8`)
    constexpr bool kWithOutputSF = cute::is_same_v<epilogue_op_t, transform::EpilogueDynamicScaledFP8>;
    constexpr uint32_t kSFGranN = transform::EpilogueDynamicScaledFP8::kSFGranN;

    // TMA checks
    constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    DG_STATIC_ASSERT(kSwizzleCDMode == 128, "TMA D must be 128B swizzled");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(not kWithOutputSF or cute::is_same_v<cd_dtype_t, cutlass::float_e4m3_t>, "FP8 output requires an E4M3 D");
    DG_STATIC_ASSERT(not kWithOutputSF or (STORE_BLOCK_M == 16 and kNumUMMAStoreThreads == 128),
                     "Swap-AB FP8 output requires one warpgroup and a 16x128 store shape");

    // Share store pipeline between blocks
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over M blocks. The scheduler skips empty blocks and aligns dynamic effective M to STORE_BLOCK_M.
    const auto num_stores = effective_m / STORE_BLOCK_M;
    for (uint32_t s = 0; s < num_stores; ++ s, advance_store_pipeline()) {
        const auto store_idx = reverse_store_order ? num_stores - 1 - s : s;
        // Wait shared memory to be released
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<kNumTMAStoreStages - 1>();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

        // Store into shared memory
        constexpr uint32_t kNumTmemLoads = STORE_BLOCK_M / kNumSwizzleAtomRows;
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemLoads; ++ i) {
            const auto load_idx = reverse_store_order ? kNumTmemLoads - 1 - i : i;
            uint32_t tmem_addr = tmem_base_addr +
                                 store_idx * STORE_BLOCK_M +    // Store stage offset
                                 load_idx * kNumSwizzleAtomRows; // In-block offset
            uint32_t values[kNumSwizzleAtomRows];

            // Warps cooperatively write an atomic block to shared memory
            DG_STATIC_ASSERT(STORE_BLOCK_N_ATOM % 32 == 0, "Invalid block sizes");
            constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
            uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
            uint32_t inner_atom_offset = load_idx * kNumSwizzleAtomRows * kSwizzleCDMode;
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]) + outer_atom_offset + inner_atom_offset;

            // Read from tensor memory into registers
            // NOTES: empty groups also read the accumulator (safe, as at least one UMMA is
            //        issued) and select zeros afterwards, keeping the loads branchless
            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                // NOTES: the FP32 store does not use STSM, so the plain `.32x32b` layout works
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, values[0], values[1], values[2], values[3],
                                                                 values[4], values[5], values[6], values[7]);
                cutlass::arch::fence_view_async_tmem_load();
                epilogue_op.apply_values(values);
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
                cutlass::arch::fence_view_async_tmem_load();
                epilogue_op.apply_values(values);
            }
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
                if (s * kNumTmemLoads + i + 1 == num_loads_before_arrive) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_overlap_barrier->arrive(0u);
                }
            }
            // Notify tensor memory empty (only at the leader CTA) as soon as all loads are in registers
            if (s == num_stores - 1 and i == kNumTmemLoads - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            // Store into shared memory
            if constexpr (kWithOutputSF) {
                // Round the accumulator into BF16 pairs first, so the output bitwise matches
                // a BF16 D followed by the standalone cast (power-of-two scaling of BF16 is exact)
                // NOTES: with the `.16x256b` load, every pair holds this lane's row pair at one
                //        of its 4 columns, so all packed multiplications share one per-half sf_inv
                nv_bfloat162 values_bf16x2[kNumSwizzleAtomRows / 2];
                #pragma unroll
                for (uint32_t pair_idx = 0; pair_idx < kNumSwizzleAtomRows / 2; ++ pair_idx)
                    values_bf16x2[pair_idx] = math::cast_into_bf16x2(values[pair_idx * 2], values[pair_idx * 2 + 1]);

                // Reduce the row pair's amax and compute the SFs (the SF exponent floor covers
                // the amax clamp): the 8 lanes sharing `lane_idx % 4` cover the pair's two SF
                // groups, so a packed tree plus a cross-group partial reduction is enough
                const auto amax_pair = math::warp_reduce_max<4, true>(
                    __hmax2(__hmax2(__habs2(values_bf16x2[0]), __habs2(values_bf16x2[1])),
                            __hmax2(__habs2(values_bf16x2[2]), __habs2(values_bf16x2[3]))));
                const auto sf_exp_lower = math::get_ue8m0_sf_exp<cd_dtype_t>(amax_pair.x);
                const auto sf_exp_upper = math::get_ue8m0_sf_exp<cd_dtype_t>(amax_pair.y);
                const auto sf_inv = nv_bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp_lower),
                                                 math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp_upper));

                // Cast into FP8 and store with a transposing STSM: the loaded fragment matches
                // `stmatrix.m16n8.trans.b8`, with one 8x16 matrix per 16 columns
                const auto smem_ptr = smem_base_ptr + (lane_idx % 8) * kSwizzleCDMode +
                                      ((epilogue_warp_idx * 2 + lane_idx / 8) ^ (lane_idx % 8)) * kNumBankGroupBytes;
                ptx::SM100_U8x8_STSM_T<uint32_t>::copy(
                    math::scale_bf16x2_into_fp8x4(values_bf16x2[0], values_bf16x2[1], sf_inv, sf_inv),
                    math::scale_bf16x2_into_fp8x4(values_bf16x2[2], values_bf16x2[3], sf_inv, sf_inv),
                    smem_ptr);

                // Store the SF byte into the packed word
                // NOTES: lanes 0-3 own their row pair's even row, lanes 4-7 the odd row
                if (lane_idx < kNumSwizzleAtomRows)
                    epilogue_op.store_sf(base_m_idx + store_idx * STORE_BLOCK_M + load_idx * kNumSwizzleAtomRows +
                                          2 * (lane_idx % 4) + lane_idx / 4,
                                      base_n_idx + epilogue_warp_idx * kSFGranN, batch_idx,
                                      static_cast<uint8_t>(lane_idx < 4 ? sf_exp_lower : sf_exp_upper));
            } else if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                uint32_t col = lane_idx / 4;

                #pragma unroll
                for (uint32_t row = 0; row < kNumSwizzleAtomRows; ++ row) {
                    auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                                  + (col ^ row) * kNumBankGroupBytes
                                                  + (lane_idx % 4) * sizeof(float);
                    ptx::st_shared(reinterpret_cast<uint32_t*>(smem_ptr), values[row]);
                }
            } else {
                // Destination shared memory address
                uint32_t row = lane_idx % 8;
                uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                              + (col ^ row) * kNumBankGroupBytes;

                // Store matrix with transposition
                ptx::SM90_U32x4_STSM_T<int>::copy(math::cast_into_bf16_and_pack(values[0], values[1]),
                                                  math::cast_into_bf16_and_pack(values[2], values[3]),
                                                  math::cast_into_bf16_and_pack(values[4], values[5]),
                                                  math::cast_into_bf16_and_pack(values[6], values[7]),
                                                  smem_ptr);
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
                uint32_t n_idx = epilogue_op_t::apply_index_n<STORE_BLOCK_N_ATOM>(base_n_idx + i * STORE_BLOCK_N_ATOM);

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
    }
}

} // namespace deep_gemm::epilogue
