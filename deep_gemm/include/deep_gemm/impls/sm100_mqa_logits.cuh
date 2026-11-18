#pragma once

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

// Shared SM100 MQA logits core with contiguous and paged entries using the same TMA signature

namespace deep_gemm {

// Convert runtime valid-token count to `cute::Int` so token loops stay compile-time constant
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) {
        fn(cute::Int<1>{});
    } else if (num_block_tokens >= kCandidate) {
        fn(cute::Int<kCandidate>{});
    } else {
        dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn));
    }
}

// Visit one token's heads in 16 / 8 / 4-wide TMEM chunks
template <uint32_t kNumHeads, uint32_t kHeadBase = 0, typename Fn>
CUTLASS_DEVICE void visit_tmem_head_chunks(const uint32_t& tmem_col, float* accum, Fn&& fn) {
    constexpr uint32_t kNumRemaining = kNumHeads - kHeadBase;
    if constexpr (kNumRemaining > 0) {
        constexpr uint32_t kNumChunkHeads = kNumRemaining >= 16 ? 16 : (kNumRemaining >= 8 ? 8 : 4);
        ptx::tmem_load_32dp32b<kNumChunkHeads>(tmem_col + kHeadBase, reinterpret_cast<uint32_t*>(accum));
        cutlass::arch::fence_view_async_tmem_load();
        fn(cute::Int<kHeadBase>{}, cute::Int<kNumChunkHeads>{});
        visit_tmem_head_chunks<kNumHeads, kHeadBase + kNumChunkHeads>(tmem_col, accum, static_cast<Fn&&>(fn));
    }
}

// Shared BF16-output device core parameterized by Q/K dtype and scheduler geometry/addressing
template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t UMMA_N,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_DEVICE void sm100_mqa_logits_core_impl(const uint32_t logits_stride,
                                               nv_bfloat16* logits,
                                               const cute::TmaDescriptor& tensor_map_q,
                                               const cute::TmaDescriptor& tensor_map_sf_q,
                                               const cute::TmaDescriptor& tensor_map_kv,
                                               const cute::TmaDescriptor& tensor_map_sf_kv,
                                               const cute::TmaDescriptor& tensor_map_weights,
                                               const MakeScheduler& make_scheduler) {
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;
    using Scheduler = decltype(make_scheduler(0u));
    constexpr bool kIsPaged = Scheduler::kIsPaged;

    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, BLOCK_Q, SPLIT_KV, UMMA_N,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    static constexpr uint32_t kNumUTCCPAlignedElems = SharedStorage::kNumUTCCPAlignedElems;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t BLOCK_QH = SharedStorage::BLOCK_QH;
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ = SharedStorage::kNumSFQ;
    static constexpr uint32_t kNumSFKV = SharedStorage::kNumSFKV;
    static constexpr uint32_t kNumQKBytesPerToken = SharedStorage::kNumQKBytesPerToken;
    static constexpr uint32_t kNumWeightElementsPerRow = SharedStorage::kNumWeightElementsPerRow;
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");
    DG_STATIC_ASSERT(kNumTmemStages == kNumMathWarpGroups, "TMEM stages must match math warp groups");

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
    constexpr uint32_t kNumSFQColsPerStage = kNumSFQ / 32;
    constexpr uint32_t kNumSFKVColsPerStage = kNumSFKV / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumQStages * kNumSFQColsPerStage +
                                                                       kNumKVStages * kNumSFKVColsPerStage>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumQStages * kNumSFQColsPerStage;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q_barriers[i].init(2);
            smem.full_sf_q_barriers[i].init(1);
            smem.empty_q_barriers[i].init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(2);
            smem.full_sf_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++ i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
    __syncthreads();

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumWarpGroups = kNumMathWarpGroups + kNumSpecializedThreads / 128;
    constexpr uint32_t kNumEntryRegisters = (512 / kNumWarpGroups / 8) * 8;
    constexpr uint32_t kNumMathRegisters = ((kNumEntryRegisters * kNumWarpGroups - kNumSpecializedRegisters * (kNumSpecializedThreads / 128))
                                           / kNumMathWarpGroups / 8) * 8;
    DG_STATIC_ASSERT(kNumMathRegisters * kNumMathWarpGroups + kNumSpecializedRegisters * (kNumSpecializedThreads / 128) <=
                     kNumEntryRegisters * kNumWarpGroups, "Register reconfiguration exceeds the CTA entry pool");

    cudaGridDependencySynchronize();

    // Reuse a full KV ring when consecutive tasks share the same splits
    const auto reuses_kv_stages = [&](const sched::MQALogitsTask& task) {
        return task.kv_shared_with_prev and task.num_kv_splits == kNumKVStages;
    };

    // Shared KV/SF producer loop; the SF producer also drives Q
    const auto produce_kv = [&]<bool kIsSFProducer>(cute::bool_constant<kIsSFProducer>) {
        auto scheduler = make_scheduler(sm_idx);
        auto& full_barriers = kIsSFProducer ? smem.full_sf_kv_barriers : smem.full_kv_barriers;
        constexpr uint32_t kNumFullTxBytes = kIsSFProducer
            ? kNumSFKV * sizeof(uint32_t) : SharedStorage::kNumKVBytesPerStage;

        // Contiguous KV: the split is one or two dense TMA tiles of each operand
        const auto issue_contiguous_split = [&](const uint32_t& kv_stage_idx, const uint32_t& kv_token_offset) {
            constexpr uint32_t kNumKVTokensPerTMA = SPLIT_KV % 256 == 0 ? 256 : 128;
            DG_STATIC_ASSERT(SPLIT_KV % kNumKVTokensPerTMA == 0, "KV split must contain whole TMA tiles");
            #pragma unroll
            for (uint32_t token_offset = 0; token_offset < SPLIT_KV; token_offset += kNumKVTokensPerTMA) {
                if constexpr (kIsSFProducer) {
                    tma::copy<kNumKVTokensPerTMA, 1, 0>(
                        &tensor_map_sf_kv, &smem.full_sf_kv_barriers[kv_stage_idx],
                        smem.smem_sf_kv[kv_stage_idx] + token_offset,
                        kv_token_offset + token_offset, 0);
                } else {
                    tma::copy<kHeadDim, kNumKVTokensPerTMA, 0>(
                        &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                        smem.smem_kv[kv_stage_idx] + token_offset * kNumQKBytesPerToken,
                        0, kv_token_offset + token_offset);
                }
            }
            full_barriers[kv_stage_idx].arrive_and_expect_tx(kNumFullTxBytes);
        };

        // Gather SF pages in fours where possible and share KV loads between producers
        const auto issue_paged_split = [&]<uint32_t kNumPagesPerSplit>(const uint32_t& kv_stage_idx,
                                                                       const int (&page_coords)[kNumPagesPerSplit]) {
            constexpr uint32_t kPageKV = SPLIT_KV / kNumPagesPerSplit;
            DG_STATIC_ASSERT(kPageKV * kNumPagesPerSplit == SPLIT_KV, "KV split must contain whole pages");
            constexpr uint32_t kNumGatherPages = kNumPagesPerSplit / 4 * 4;
            constexpr uint32_t kNumSFTMAs = kNumGatherPages / 4 + kNumPagesPerSplit - kNumGatherPages;
            constexpr uint32_t kNumKVPagesFromSFProducer = kNumPagesPerSplit > 2 * kNumSFTMAs ? (kNumPagesPerSplit - kNumSFTMAs) / 2 : 0;
            constexpr uint32_t kv_page_begin = kIsSFProducer ? 0 : kNumKVPagesFromSFProducer;
            constexpr uint32_t kv_page_end = kIsSFProducer ? kNumKVPagesFromSFProducer : kNumPagesPerSplit;

            if constexpr (kIsSFProducer) {
                #pragma unroll
                for (uint32_t page_idx = 0; page_idx < kNumGatherPages; page_idx += 4) {
                    const int4 page_coord_vec = make_int4(page_coords[page_idx], page_coords[page_idx + 1],
                                                          page_coords[page_idx + 2], page_coords[page_idx + 3]);
                    ptx::tma_gather4(
                        &tensor_map_sf_kv, smem.full_sf_kv_barriers[kv_stage_idx],
                        smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                        0, page_coord_vec,
                        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL));
                }
                #pragma unroll
                for (uint32_t page_idx = kNumGatherPages; page_idx < kNumPagesPerSplit; ++ page_idx) {
                    tma::copy<kPageKV, 1, 0>(
                        &tensor_map_sf_kv, &smem.full_sf_kv_barriers[kv_stage_idx],
                        smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                        0, page_coords[page_idx]);
                }
            }
            full_barriers[kv_stage_idx].arrive_and_expect_tx(kNumFullTxBytes);
            #pragma unroll
            for (uint32_t page_idx = kv_page_begin; page_idx < kv_page_end; ++ page_idx) {
                tma::copy<kHeadDim, kPageKV, 0, qk_dtype_t, true>(
                    &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                    smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                    0, 0, 1, page_coords[page_idx]);
            }
        };

        sched::MQALogitsTask task;
        while (scheduler.next_q_block(task)) {
            const bool reuse_kv = reuses_kv_stages(task);
            if constexpr (kIsSFProducer) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                if (cute::elect_one_sync())
                    smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);
                __syncwarp();

                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_Q * kNumHeads, 1, 0>(
                        &tensor_map_sf_q, &smem.full_sf_q_barriers[q_stage_idx],
                        smem.smem_sf_q[q_stage_idx], 0, task.q_token_base);
                    smem.full_sf_q_barriers[q_stage_idx].arrive_and_expect_tx(BLOCK_QH * sizeof(uint32_t));
                    tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
                        &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                        smem.smem_q[q_stage_idx], 0, task.q_token_base * kNumHeads);
                    tma::copy<kNumWeightElementsPerRow, BLOCK_Q, 0>(
                        &tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
                        smem.smem_weights[q_stage_idx], 0, task.q_token_base);
                    // TMA bytes exclude padding in the Q and weight stages
                    smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(
                        BLOCK_QH * kNumQKBytesPerToken + BLOCK_Q * SharedStorage::kNumWeightBytesPerRow);
                }
            }

            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                if (reuse_kv) {
                    if (cute::elect_one_sync()) {
                        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                        full_barriers[kv_stage_idx].arrive();
                    }
                    __syncwarp();
                    continue;
                }

                if constexpr (kIsPaged) {
                    // Resolve page coordinates while the stage may still be busy
                    int page_coords[Scheduler::kNumPagesPerSplit];
                    scheduler.get_kv_page_coords(task, kv_split_idx, page_coords);
                    if (cute::elect_one_sync()) {
                        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                        issue_paged_split(kv_stage_idx, page_coords);
                    }
                } else if (cute::elect_one_sync()) {
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                    issue_contiguous_split(kv_stage_idx, task.kv_token_base + kv_split_idx * SPLIT_KV);
                }
                __syncwarp();
            }
        }
    };

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        produce_kv(cute::true_type{});
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        produce_kv(cute::false_type{});
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto scheduler = make_scheduler(sm_idx);
        sched::MQALogitsTask task;
        while (scheduler.next_q_block(task)) {
            const bool reuse_kv = reuses_kv_stages(task);
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_sf_q_barriers[q_stage_idx].wait(q_phase);
            // UTCCP overwrites TMEM columns read by the previous MMA using this stage
            ptx::tcgen05_after_thread_sync();

            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                utccp_required_smem_warp_transpose(smem_ptr);
            }
            // Every lane fences its own stores before the elected lane issues the async-proxy read
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                if (cute::elect_one_sync())
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                        sf_desc, kTmemStartColOfSFQ + q_stage_idx * kNumSFQColsPerStage + i * 4);
                __syncwarp();
            }
            if (cute::elect_one_sync()) {
                ptx::tcgen05_before_thread_sync();
                smem.full_q_barriers[q_stage_idx].arrive();
            }
            for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_sf_kv_barriers[kv_stage_idx].wait(kv_phase);
                ptx::tcgen05_after_thread_sync();

                if (reuse_kv) {
                    // SF remains in TMEM; do not transpose the already-transposed SMEM again.
                    if (cute::elect_one_sync()) {
                        ptx::tcgen05_before_thread_sync();
                        smem.full_kv_barriers[kv_stage_idx].arrive();
                    }
                    continue;
                }

                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                }
                cutlass::arch::fence_view_async_shared();
                __syncwarp();

                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                            sf_desc, kTmemStartColOfSFKV + kv_stage_idx * kNumSFKVColsPerStage + i * 4);
                    }
                    ptx::tcgen05_before_thread_sync();
                    smem.full_kv_barriers[kv_stage_idx].arrive();
                }
            }
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Load the allocated TMEM base to keep derived addresses in uniform registers
        const uint32_t tmem_base = ptx::ld_shared(&smem.tmem_ptr_in_smem);
        DG_TRAP_ONLY_DEVICE_ASSERT(tmem_base == 0);
        if (cute::elect_one_sync()) {
            using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
            DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or kHeadDim == 64 or kHeadDim == 128, "Invalid head dim");
            constexpr uint32_t kPackFactor = get_smem_pack_factor<qk_dtype_t>();
            constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;
            constexpr uint32_t kNumUMMAK = kHeadDim / UMMA_K;
            // Advance descriptor low words by each stage's fixed offset
            const auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(smem.smem_kv[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(smem.smem_q[0], 0, 0);
            const uint32_t a_desc_lo = a_desc.lo, b_desc_lo = b_desc.lo;
            constexpr uint32_t kNumKVDescPerStage = sizeof(smem.smem_kv[0]) / 16;
            constexpr uint32_t kNumQDescPerStage = sizeof(smem.smem_q[0]) / 16;
            uint64_t runtime_instr_descs[kNumUMMAK];
            #pragma unroll
            for (uint32_t k = 0; k < kNumUMMAK; ++ k)
                runtime_instr_descs[k] = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * kPackFactor, k * kPackFactor);
            // With one TMEM stage per math warpgroup the ring is walked exactly once per KV split
            uint32_t tmem_phase = 0;

            auto scheduler = make_scheduler(sm_idx);
            sched::MQALogitsTask task;
            while (scheduler.next_q_block(task)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.full_q_barriers[q_stage_idx].wait(q_phase);
                ptx::tcgen05_after_thread_sync();
                const uint32_t b_desc_stage_lo = b_desc_lo + q_stage_idx * kNumQDescPerStage;
                const uint32_t tmem_sfb = tmem_base + kTmemStartColOfSFQ + q_stage_idx * kNumSFQColsPerStage;
                for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++ kv_split_idx) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    // The fence after the first empty-TMEM wait also orders this data
                    smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                    const uint32_t a_desc_stage_lo = a_desc_lo + kv_stage_idx * kNumKVDescPerStage;
                    const uint32_t tmem_sfa = tmem_base + kTmemStartColOfSFKV + kv_stage_idx * kNumSFKVColsPerStage;
                    #pragma unroll
                    for (uint32_t tmem_stage_idx = 0; tmem_stage_idx < kNumMathWarpGroups; ++ tmem_stage_idx) {
                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        #pragma unroll
                        for (uint32_t k = 0; k < kNumUMMAK; ++ k) {
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, 0, kQKSwizzleMode, qk_dtype_t>(
                                a_desc_stage_lo, tmem_stage_idx * UMMA_M * kHeadDim, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, 0, kQKSwizzleMode, qk_dtype_t>(
                                b_desc_stage_lo, 0, k * UMMA_K);
                            mma_op_t::fma(a_desc, b_desc, tmem_base + tmem_stage_idx * UMMA_N, k, runtime_instr_descs[k],
                                          tmem_sfa + tmem_stage_idx * 4, tmem_sfb);
                        }
                        ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
                    }
                    tmem_phase ^= 1;
                    ptx::umma_arrive_no_elect(smem.empty_kv_barriers[kv_stage_idx]);
                }
                // Count the MMA warp here; math threads release Q after all MMAs complete
                ptx::mbarrier_arrive_count(smem.empty_q_barriers[q_stage_idx], 32);
            }
        }
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        auto scheduler = make_scheduler(sm_idx);
        uint32_t seq_k_start[BLOCK_Q];
        uint32_t seq_k_end[BLOCK_Q];
        const auto math_warpgroup_idx = warp_idx / 4;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");
        tmem_pipeline.advance(math_warpgroup_idx);

        DG_STATIC_ASSERT(kNumHeads % 4 == 0, "Head count must be a multiple of 4");
        DG_STATIC_ASSERT(8 <= UMMA_N and UMMA_N <= 256, "Invalid UMMA_N for MMA");
        // Every token's weights stay in registers, so one TMEM chunk buffer is all that fits
        nv_bfloat162 weights[BLOCK_Q][kNumHeads / 2];
        float accum[kNumHeads < 16 ? kNumHeads : 16];

        sched::MQALogitsTask task;
        while (scheduler.next_q_block(task, seq_k_start, seq_k_end)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;
                // Offset warp-uniform output bases by -seq_k_start[i] for shared KV indexing
                nv_bfloat16* output_bases[BLOCK_Q];
                #pragma unroll
                for (uint32_t i = 0; i < kNumValidTokens; ++ i)
                    output_bases[i] = logits + (task.q_token_base + i) * static_cast<uint64_t>(logits_stride) - seq_k_start[i];

                #pragma unroll
                for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                    const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow;
                    // Load two bf16 weights at a time as one packed shared u32
                    const auto packed_row = reinterpret_cast<const uint32_t*>(smem_weights_row);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 2; ++ j) {
                        const auto packed = ptx::ld_shared(packed_row + j);
                        weights[i][j] = *reinterpret_cast<const nv_bfloat162*>(&packed);
                    }
                }

                // Reduce both bf16x2 halves and store the low half only for valid KV columns
                const auto store_token = [&](const uint32_t& i, const uint32_t& kv_offset,
                                             const nv_bfloat162& sum_0, const nv_bfloat162& sum_1) {
                    const auto sum = __hadd2_rn(sum_0, sum_1);
                    const auto result = __hadd2_rn(__low2bfloat162(sum), __high2bfloat162(sum));
                    if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i])
                        ptx::st_global_low_bf16(output_bases[i] + kv_offset, result);
                };
                const auto reduce_pairs = [&](const float* chunk, const nv_bfloat162* chunk_weights, const uint32_t& num_heads,
                                              nv_bfloat162& sum_0, nv_bfloat162& sum_1) {
                    #pragma unroll
                    for (uint32_t head_offset = 0; head_offset < num_heads; head_offset += 4) {
                        const auto accum_pair_0 = make_float2(chunk[head_offset], chunk[head_offset + 1]);
                        const auto accum_pair_1 = make_float2(chunk[head_offset + 2], chunk[head_offset + 3]);
                        sum_0 = __hfma2(ptx::cvt_relu_bf16x2_f32(accum_pair_0), chunk_weights[head_offset / 2], sum_0);
                        sum_1 = __hfma2(ptx::cvt_relu_bf16x2_f32(accum_pair_1), chunk_weights[head_offset / 2 + 1], sum_1);
                    }
                };

                // Overlap loads and reduction by unrolling two splits, except in paged mode
                #pragma unroll (kIsPaged ? 1 : 2)
                for (uint32_t kv_split_idx = 0; kv_split_idx < task.num_kv_splits; ++ kv_split_idx) {
                    const auto kv_offset = task.kv_token_base + kv_split_idx * SPLIT_KV + math_thread_idx;
                    CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                        auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                        auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                        const auto reduce_head_chunk = [&](auto head_base_t, auto num_chunk_heads_t) {
                            constexpr uint32_t kHeadBase = decltype(head_base_t)::value;
                            constexpr uint32_t kNumChunkHeads = decltype(num_chunk_heads_t)::value;
                            // Release TMEM after its last load to overlap MMA with reduction
                            if constexpr (kHeadBase + kNumChunkHeads == kNumHeads) {
                                if (i == kNumValidTokens - 1) {
                                    ptx::tcgen05_before_thread_sync();
                                    smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                                }
                            }
                            reduce_pairs(accum, weights[i] + kHeadBase / 2, kNumChunkHeads, sum_0, sum_1);
                        };
                        visit_tmem_head_chunks<kNumHeads>(tmem_stage_idx * UMMA_N + i * kNumHeads, accum, reduce_head_chunk);
                        store_token(i, kv_offset, sum_0, sum_1);
                    }
                }
            };

            if constexpr (kIsPaged)
                dispatch_num_block_tokens<BLOCK_Q>(task.num_q_tokens, process_q_block);
            else
                process_q_block(cute::Int<BLOCK_Q>{});

            cutlass::arch::fence_view_async_shared();
            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

// Unified contiguous-KV entry for MXFP4 / MXFP8.
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kUseSchedule,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t UMMA_N,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_mqa_logits(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                      const uint32_t logits_stride,
                      const uint32_t* cu_seq_len_k_start,
                      const uint32_t* cu_seq_len_k_end,
                      const uint32_t* schedule_meta,
                      nv_bfloat16* logits,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs, kUseSchedule>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, schedule_meta);
    };

    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim,
                               BLOCK_Q, SPLIT_KV,
                               UMMA_N, kNumQStages, kNumKVStages, kNumTmemStages,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t,
                               decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// Unified MXFP4 / MXFP8 paged entry scheduling (Q-block, chunk) tasks
template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t PAGE_KV,
          uint32_t BLOCK_Q, uint32_t UMMA_N,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_paged_mqa_logits(const uint32_t num_q_tokens_total,
                            const uint32_t logits_stride, const uint32_t block_table_stride,
                            const uint32_t* context_lens, nv_bfloat16* logits,
                            const uint32_t* block_table, const uint32_t* indices,
                            const uint32_t* schedule_meta,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    DG_STATIC_ASSERT(UMMA_N == math::constexpr_align(BLOCK_Q * kNumHeads, 8u), "Invalid Q tile shape");

    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        return sched::SM100PagedMQALogitsScheduler<kNumHeads, BLOCK_Q, SPLIT_KV, PAGE_KV, kSplitsPerChunk>(
            sm_idx, num_q_tokens_total, context_lens, indices, block_table, block_table_stride, schedule_meta);
    };

    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim,
                               BLOCK_Q, SPLIT_KV,
                               UMMA_N, kNumQStages, kNumKVStages, kNumTmemStages,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t,
                               decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

} // namespace deep_gemm
