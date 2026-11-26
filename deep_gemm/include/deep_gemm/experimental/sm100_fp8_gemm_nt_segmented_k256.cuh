#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/operators.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/layout/gemm.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm::experimental {

// Engram: A [M, S, 256] with 128B-aligned segment pitch, B [N, S * 256].
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumTMAStoreStages,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, bool kSwapAB>
CUTLASS_GLOBAL void __launch_bounds__(256, 1)
sm100_fp8_gemm_nt_segmented_k256_impl(uint32_t shape_m,
                                      const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                      const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                      const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                                      const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                                      const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using ab_dtype_t = cutlass::float_e4m3_t;
    using cd_dtype_t = cutlass::bfloat16_t;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    constexpr uint32_t kSegmentA = 256, BLOCK_K = 128, kNumKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t kSwizzleABMode = 128;
    constexpr uint32_t kNumNonEpilogueThreads = 128, kNumEpilogueThreads = 128;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(SHAPE_N > 0 and SHAPE_K > 0 and SHAPE_K % kSegmentA == 0, "Invalid Engram shape");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // Four per-32 UE8M0 scales per K tile, packed into one uint32_t.
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);

    // Double-buffer accumulators; TMA stores use one or two stages.
    constexpr uint32_t kNumEpilogueStages = 2;
    DG_STATIC_ASSERT(kNumTMAStoreStages == 1 or kNumTMAStoreStages == 2, "Invalid number of TMA store stages");
    // Each store covers all of BLOCK_N for swap-AB, or BLOCK_M otherwise.
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(ab_dtype_t);

    // Tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumSFTmemCols = kNumSFATmemCols + kNumSFBTmemCols;
    constexpr uint32_t kNumOverlappedTmemCols = cute::max<uint32_t>(kNumAccumTmemCols + kNumSFTmemCols, 512) - 512;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFTmemCols - kNumOverlappedTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols - kNumOverlappedTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kTmemStartColOfSFA + kNumSFATmemCols;
    DG_STATIC_ASSERT(kNumOverlappedTmemCols <= UMMA_N, "Invalid overlapped tensor memory columns");
    DG_STATIC_ASSERT(kSwapAB or kNumOverlappedTmemCols == 0 or kNumOverlappedTmemCols <= STORE_BLOCK_N,
                     "Non-swap overlapped tensor memory columns must fit in the first epilogue store");

    // reviwed
    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : void();

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    using SharedStorage = layout::SM100FP8FP4GemmSharedStorage<
        kNumStages, kNumEpilogueStages, kNumTMAStoreStages, LOAD_BLOCK_M, LOAD_BLOCK_N, BLOCK_K,
        STORE_BLOCK_M, STORE_BLOCK_N, SF_BLOCK_M, SF_BLOCK_N, 1,
        ab_dtype_t, ab_dtype_t, cd_dtype_t>;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= sizeof(smem.a[0]) + sizeof(smem.b), "Memory out of bound for UMMA");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            smem.sf_full_barriers[i].init(1);
            smem.empty_barriers[i].init(1);
            // TMA completion plus one SF-transpose warp per CTA, arriving at the leader.
            smem.full_barriers[i].init(kNumMulticast * (1 + 32));
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            smem.tmem_full_barriers[i].init(1);
            // Arrive only at the leader CTA
            smem.tmem_empty_barriers[i].init(kNumMulticast * kNumUMMAStoreThreads);
            smem.tmem_overlap_barriers[i].init(kNumMulticast * kNumUMMAStoreThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &smem.tmem_ptr);
    }
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;

    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, 1, kNumMulticast, kIsMulticastOnA, kNumSMs>(
        shape_m, SHAPE_N, SHAPE_K);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        constexpr uint32_t kNumTMABytesPerStage = (LOAD_BLOCK_M + LOAD_BLOCK_N) * BLOCK_K;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                smem.empty_barriers[stage_idx].wait(phase ^ 1);
                uint32_t m_idx = m_block_idx * BLOCK_M, n_idx = n_block_idx * BLOCK_N;
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Load SF first so that its transpose overlaps the A/B transfer.
                tma::copy<SF_BLOCK_M, 1, 0>(
                    &tensor_map_sfa, &smem.sf_full_barriers[stage_idx], smem.sfa[stage_idx], m_block_idx * BLOCK_M, k_block_idx);
                tma::copy<SF_BLOCK_N, 1, 0>(
                    &tensor_map_sfb, &smem.sf_full_barriers[stage_idx], smem.sfb[stage_idx], n_block_idx * BLOCK_N, k_block_idx);
                smem.sf_full_barriers[stage_idx].arrive_and_expect_tx(sizeof(smem.sfa[0]) + sizeof(smem.sfb[0]));

                tma::copy_nd<BLOCK_K, LOAD_BLOCK_M, kSwizzleABMode, ab_dtype_t, 3>(
                    &tensor_map_a, &smem.full_barriers[stage_idx], smem.a[stage_idx], kNumMulticast,
                    k_idx % kSegmentA, k_idx / kSegmentA, m_idx);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleABMode, ab_dtype_t>(
                    &tensor_map_b, &smem.full_barriers[stage_idx], smem.b[stage_idx], k_idx, n_idx, kNumMulticast);

                if (is_leader_cta) {
                    smem.full_barriers[stage_idx].arrive_and_expect_tx(kNumTMABytesPerStage * kNumMulticast);
                } else {
                    smem.full_barriers[stage_idx].arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // Only the leader CTA issues MMA instructions.
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<ab_dtype_t, ab_dtype_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleABMode>(smem.a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleABMode>(smem.b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(smem.a[0]) / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(smem.b[0]) / 16 : 0u;

        DG_STATIC_ASSERT(UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256, "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait until this accumulator stage is fully reusable.
            // The overlap barrier is deferred to the first K block so that
            // the SF UTCCP can be issued while the preceding epilogue drains.
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            smem.tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    if constexpr (kNumMulticast == 1) {
                        cutlass::arch::umma_arrive(barrier);
                    } else {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Launch MMAs
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);

                // Wait A/B TMA and SF-transpose arrival
                smem.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();

                if (cute::elect_one_sync()) {
                    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                                                             cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                        const auto smem_ptr = smem.sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                        const auto smem_ptr = smem.sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                    }
                }
                __syncwarp();

                // The overlap barrier is waited after SF UTCCP but before the
                // first UMMA: UTCCP can overlap the preceding epilogue, and
                // subsequent K blocks share the same accumulator columns.
                if constexpr (kNumOverlappedTmemCols > 0) {
                    if (k_block_idx == 0 and scheduler.current_iter > 0) {
                        ptx::tcgen05_before_thread_sync();
                        const auto preceding_iter_idx = scheduler.current_iter - 1;
                        const auto preceding_stage_idx = preceding_iter_idx % kNumEpilogueStages;
                        const auto preceding_phase_idx = (preceding_iter_idx / kNumEpilogueStages) & 1;
                        smem.tmem_overlap_barriers[preceding_stage_idx].wait(preceding_phase_idx);
                        ptx::tcgen05_after_thread_sync();
                    }
                }

                if (cute::elect_one_sync()) {
                    // Issue UMMA
                    using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_MXF8F6F4_SS, ptx::SM100_MMA_MXF8F6F4_2x1SM_SS>;
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        const uint32_t offset = umma_k_idx * UMMA_K;
                        const auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, umma_k_idx, umma_k_idx);
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleABMode, ab_dtype_t>(a_desc_base_lo, 0, offset);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleABMode, ab_dtype_t>(b_desc_base_lo, 0, offset);
                        if constexpr (kSwapAB) {
                            mma_t::fma(b_desc, a_desc,
                                       accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols),
                                       umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc,
                                       kTmemStartColOfSFB, kTmemStartColOfSFA);
                        } else {
                            mma_t::fma(a_desc, b_desc,
                                       accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols),
                                       umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc,
                                       kTmemStartColOfSFA, kTmemStartColOfSFB);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == kNumKBlocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            smem.tmem_empty_barriers[iter_idx % kNumEpilogueStages].wait(accum_phase_idx);
        }
    } else if (warp_idx == 2) {
        // Transpose SF in shared memory into the UTCCP layout.
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                smem.sf_full_barriers[stage_idx].wait(phase);
                #pragma unroll
                for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                    utccp_required_smem_warp_transpose(smem.sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                #pragma unroll
                for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                    utccp_required_smem_warp_transpose(smem.sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();
                smem.full_barriers[stage_idx].arrive(0u);
            }
        }
    } else if (epilogue_warp_idx < kNumUMMAStoreThreads / 32) {
        const auto epilogue_args = EpilogueOperatorArgs();
        epilogue::operators::Identity epilogue_operator(epilogue_args);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            smem.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols);
            const bool reverse_store_order = kNumOverlappedTmemCols > 0 and accum_stage_idx == 0;
            const auto base_m_idx = m_block_idx * BLOCK_M;
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                epilogue::sm100_store_cd_swap_ab<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kNumOverlappedTmemCols,
                    GemmType::Normal, false>
                (smem, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, 0, false,
                 BLOCK_M,
                 epilogue_warp_idx, lane_idx,
                 epilogue_operator,
                 reverse_store_order,
                 &smem.tmem_overlap_barriers[accum_stage_idx],
                 &smem.tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                epilogue::sm100_store_cd<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kNumOverlappedTmemCols,
                    GemmType::Normal, false>
                (smem, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, 0, false,
                 epilogue_warp_idx, lane_idx,
                 epilogue_operator,
                 reverse_store_order,
                 &smem.tmem_overlap_barriers[accum_stage_idx],
                 &smem.tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm::experimental

#pragma clang diagnostic pop
