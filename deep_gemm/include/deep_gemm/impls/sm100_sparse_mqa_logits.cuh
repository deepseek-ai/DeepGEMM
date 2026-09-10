#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/sparse_mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

namespace sparse_mqa_detail {

using namespace layout::sparse_mqa_logits;

template <uint32_t kSwizzleMode>
CUTLASS_DEVICE uint32_t get_swizzled_kv_chunk_idx(const uint32_t logical_chunk_idx) {
    constexpr uint32_t kMask = kSwizzleMode / 16 - 1;
    return (logical_chunk_idx & ~kMask) | ((logical_chunk_idx & kMask) ^ ((logical_chunk_idx >> 3u) & kMask));
}

template <uint32_t SPARSE_BLOCK_KV, typename qk_dtype_t>
struct ContiguousSparseKVAccessor {
    static constexpr bool kSupportsContiguousTMA = true;
    static constexpr uint32_t kNumQKBytesPerToken = kHeadDim / get_smem_pack_factor<qk_dtype_t>();
    using KVBlockRef = uint32_t;

    const uint8_t* kv;
    const uint32_t* sf_kv;
    const cute::TmaDescriptor* tensor_map_kv;
    const cute::TmaDescriptor* tensor_map_sf_kv;

    CUTLASS_HOST_DEVICE ContiguousSparseKVAccessor(const uint8_t* kv, const uint32_t* sf_kv,
                                                  const cute::TmaDescriptor* tensor_map_kv,
                                                  const cute::TmaDescriptor* tensor_map_sf_kv)
        : kv(kv), sf_kv(sf_kv), tensor_map_kv(tensor_map_kv), tensor_map_sf_kv(tensor_map_sf_kv) {}

    CUTLASS_DEVICE void prefetch_tma_descriptors() const {
        cute::prefetch_tma_descriptor(tensor_map_kv);
        cute::prefetch_tma_descriptor(tensor_map_sf_kv);
    }

    template <uint32_t SPLIT_KV>
    CUTLASS_DEVICE void copy_contiguous_kv_split(cutlass::arch::ClusterTransactionBarrier& kv_barrier,
                                                 cutlass::arch::ClusterTransactionBarrier& sf_barrier,
                                                 void* smem_kv, void* smem_sf_kv,
                                                 const uint32_t kv_token_start) const {
        DG_STATIC_ASSERT(SPLIT_KV % kNumKVTokensPerTMA == 0, "KV split must contain whole TMA tiles");
        // Publish the small SF transfer first so its transpose can overlap the bulk KV transfer.
        #pragma unroll
        for (uint32_t token_offset = 0; token_offset < SPLIT_KV; token_offset += kNumKVTokensPerTMA) {
            tma::copy<kNumKVTokensPerTMA, 1, 0>(tensor_map_sf_kv, &sf_barrier,
                static_cast<uint32_t*>(smem_sf_kv) + token_offset, kv_token_start + token_offset, 0);
        }
        sf_barrier.arrive_and_expect_tx(SPLIT_KV * sizeof(uint32_t));
        #pragma unroll
        for (uint32_t token_offset = 0; token_offset < SPLIT_KV; token_offset += kNumKVTokensPerTMA) {
            tma::copy<kHeadDim, kNumKVTokensPerTMA, 0>(tensor_map_kv, &kv_barrier,
                static_cast<uint8_t*>(smem_kv) + token_offset * kNumQKBytesPerToken, 0, kv_token_start + token_offset);
        }
        kv_barrier.arrive_and_expect_tx(SPLIT_KV * kNumQKBytesPerToken);
    }

    CUTLASS_DEVICE KVBlockRef resolve_kv_block(const uint32_t physical_kv_block_idx) const {
        return physical_kv_block_idx;
    }

    CUTLASS_DEVICE const uint8_t* get_kv_block(const KVBlockRef kv_block_ref,
                                               const uint32_t src_lane_idx) const {
        const uint32_t kv_token_start = ptx::exchange(kv_block_ref, src_lane_idx);
        // Tail reads rely on PyTorch allocator padding; extra logits are ignored
        return kv + static_cast<uint64_t>(kv_token_start) * kNumQKBytesPerToken;
    }

    CUTLASS_DEVICE const uint32_t* get_sf_kv_block(const KVBlockRef kv_block_ref,
                                                   const uint32_t src_lane_idx) const {
        return sf_kv + ptx::exchange(kv_block_ref, src_lane_idx);
    }
};

template <uint32_t PAGE_KV, uint32_t SPARSE_BLOCK_KV, typename qk_dtype_t>
struct PagedSparseKVAccessor {
    static constexpr bool kSupportsContiguousTMA = false;
    static constexpr uint32_t kNumQKBytesPerToken = kHeadDim / get_smem_pack_factor<qk_dtype_t>();

    struct KVBlockRef {
        const uint8_t* kv;
        const uint32_t* sf_kv;

        CUTLASS_DEVICE KVBlockRef(const uint8_t* kv, const uint32_t* sf_kv): kv(kv), sf_kv(sf_kv) {}
    };

    const uint8_t* fused_kv_cache;
    uint32_t kv_page_stride_bytes;

    CUTLASS_HOST_DEVICE PagedSparseKVAccessor(const uint8_t* fused_kv_cache,
                                             const uint32_t kv_page_stride_bytes)
        : fused_kv_cache(fused_kv_cache), kv_page_stride_bytes(kv_page_stride_bytes) {}

    CUTLASS_DEVICE KVBlockRef resolve_kv_block(const uint32_t physical_kv_block_idx) const {
        constexpr uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
        const uint32_t physical_page_idx = physical_kv_block_idx / kNumKVBlocksPerPage;
        const uint32_t kv_block_idx_in_page = physical_kv_block_idx % kNumKVBlocksPerPage;
        const auto page = fused_kv_cache + static_cast<uint64_t>(physical_page_idx) * kv_page_stride_bytes;
        const uint32_t token_idx_in_page = kv_block_idx_in_page * SPARSE_BLOCK_KV;
        return KVBlockRef(
            page + token_idx_in_page * kNumQKBytesPerToken,
            reinterpret_cast<const uint32_t*>(page + PAGE_KV * kNumQKBytesPerToken) + token_idx_in_page
        );
    }

    CUTLASS_DEVICE const uint8_t* get_kv_block(const KVBlockRef& kv_block_ref,
                                               const uint32_t src_lane_idx) const {
        return ptx::exchange(kv_block_ref.kv, src_lane_idx);
    }

    CUTLASS_DEVICE const uint32_t* get_sf_kv_block(const KVBlockRef& kv_block_ref,
                                                   const uint32_t src_lane_idx) const {
        return ptx::exchange(kv_block_ref.sf_kv, src_lane_idx);
    }
};

} // namespace sparse_mqa_detail

template <uint32_t SPARSE_BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages, uint32_t kNumMathWarpGroups,
          uint32_t kNumSMs, uint32_t BLOCK_Q, bool kUseUnalignedKs,
          typename qk_dtype_t, typename KVAccessor>
CUTLASS_DEVICE void sm100_sparse_mqa_logits_core_impl(const uint32_t logits_stride, nv_bfloat16* logits,
        const uint8_t* metadata, const cute::TmaDescriptor& tensor_map_q,
        const cute::TmaDescriptor& tensor_map_sf_q, const cute::TmaDescriptor& tensor_map_weights,
        const KVAccessor& kv_accessor) {
    using namespace sparse_mqa_detail;
    using namespace layout::sparse_mqa_logits;

    // MMA configs
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;
    constexpr uint32_t kPackFactor = get_smem_pack_factor<qk_dtype_t>();
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    constexpr uint32_t SPLIT_KV = kNumMathWarpGroups * UMMA_M;

    // Thread and register configs
    constexpr uint32_t kNumThreads = get_num_threads(kNumMathWarpGroups);
    constexpr uint32_t kNumWarpGroups = kNumThreads / 128;
    constexpr uint32_t kNumMathThreads = kNumMathWarpGroups * 128;
    constexpr uint32_t kNumKVCopyThreads = kNumMathWarpGroups * 32;
    constexpr uint32_t kNumEntryRegisters = (512 / kNumWarpGroups / 8) * 8;
    constexpr uint32_t kNumControlRegisters = 64;
    constexpr uint32_t kNumKVCopyRegisters = 64;
    constexpr uint32_t kNumMathRegisters = cute::min(120u, ((kNumEntryRegisters * kNumWarpGroups - kNumControlRegisters -
        kNumKVCopyRegisters * (kNumKVCopyThreads / 128)) / kNumMathWarpGroups / 8) * 8);

    // Memory configs
    using smem_t = SharedStorage<BLOCK_Q, SPARSE_BLOCK_KV, SPLIT_KV,
                                 kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t>;
    constexpr uint32_t kNumKVBlocksPerSplit = smem_t::kNumKVBlocksPerSplit;
    constexpr uint32_t kNumSFQ = smem_t::kNumSFQ;
    constexpr uint32_t kNumSFQCols = kNumSFQ / 32;
    constexpr uint32_t kNumSFKVColsPerStage = SPLIT_KV / 32;
    constexpr uint32_t kTmemStartColOfSFQ = UMMA_N * kNumTmemStages;
    constexpr uint32_t kTmemStartColOfSFKV = kTmemStartColOfSFQ + kNumQStages * kNumSFQCols;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kTmemStartColOfSFKV + kNumKVStages * kNumSFKVColsPerStage>();

    // Template checks
    DG_STATIC_ASSERT(BLOCK_Q == 2 and UMMA_N == 64, "Sparse MQA requires BLOCK_Q=2");
    DG_STATIC_ASSERT(kNumMathWarpGroups == 4 or kNumMathWarpGroups == 5,
                     "Invalid number of math warpgroups");
    DG_STATIC_ASSERT(kNumTmemStages >= kNumMathWarpGroups, "Invalid TMEM stage count");
    DG_STATIC_ASSERT(kNumTmemCols <= 512 and kNumThreads <= 1024, "Sparse MQA resource overflow");
    DG_STATIC_ASSERT(kNumMathRegisters * kNumMathWarpGroups + kNumControlRegisters +
                     kNumKVCopyRegisters * (kNumKVCopyThreads / 128) <= kNumEntryRegisters * kNumWarpGroups,
                     "Register reconfiguration exceeds the CTA entry pool");

    // Thread indices
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warpgroup_idx = warp_idx / 4;
    constexpr uint32_t kQAndMetadataWarpIdx = kNumMathWarpGroups * 4;
    constexpr uint32_t kSFTransposeWarpIdx = kQAndMetadataWarpIdx + 1;

    // Shared memory
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<smem_t*>(smem_buffer);
    const auto kv_splits = reinterpret_cast<const KVSplit<kNumKVBlocksPerSplit>*>(
        metadata + sizeof(MetadataHeader));

    // Initialization
    if (warp_idx == kQAndMetadataWarpIdx and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        if constexpr (KVAccessor::kSupportsContiguousTMA)
            kv_accessor.prefetch_tma_descriptors();
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumQStages; ++ stage_idx) {
            smem.full_q_barriers[stage_idx].init(1);
            smem.full_sf_q_barriers[stage_idx].init(1);
            // Released by the producer warp, UMMA warp, and math warpgroups
            smem.empty_q_barriers[stage_idx].init(kNumMathThreads + 64);
        }
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++ stage_idx) {
            smem.full_metadata_barriers[stage_idx].init(32);
            smem.full_sf_copy_barriers[stage_idx].init(kNumKVCopyThreads);
            // The SF transpose warp contributes the final arrival after its UTCCP completes.
            smem.full_kv_barriers[stage_idx].init(kNumKVCopyThreads + 1);
            smem.empty_kv_barriers[stage_idx].init(kNumMathThreads + 1);
        }
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumTmemStages; ++ stage_idx) {
            smem.full_tmem_barriers[stage_idx].init(1);
            smem.empty_tmem_barriers[stage_idx].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSFTransposeWarpIdx)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);

    // Zero padded Q scales for UTCCP
    constexpr uint32_t kNumSFQValues = BLOCK_Q * kNumHeads;
    constexpr uint32_t kNumSFQPaddingValues = kNumSFQ - kNumSFQValues;
    for (uint32_t padding_idx = threadIdx.x; padding_idx < kNumQStages * kNumSFQPaddingValues;
         padding_idx += kNumThreads) {
        const uint32_t q_stage_idx = padding_idx / kNumSFQPaddingValues;
        const uint32_t stage_padding_idx = padding_idx % kNumSFQPaddingValues;
        smem.sf_q[q_stage_idx][kNumSFQValues + stage_padding_idx] = 0;
    }
    __syncthreads();

    DG_DEVICE_ASSERT(blockDim.x == kNumThreads and gridDim.x == kNumSMs);
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);
    cudaGridDependencySynchronize();

    constexpr uint32_t kUMMAWarpIdx = kQAndMetadataWarpIdx + 2;
    constexpr uint32_t kFirstKVCopyWarpIdx = kNumThreads / 32 - kNumMathWarpGroups;
    if (warp_idx == kQAndMetadataWarpIdx) {
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
        RingPipeline<kNumQStages> q_pipeline;
        RingPipeline<kNumKVStages> kv_pipeline;

        const auto header = reinterpret_cast<const MetadataHeader*>(metadata);
        DG_DEVICE_ASSERT(header->use_unaligned_ks == kUseUnalignedKs);
        const uint32_t num_waves = header->num_waves;
        const auto schedule_entries = reinterpret_cast<const ScheduleEntry*>(
            kv_splits + header->num_kv_splits);
        for (uint32_t wave_idx = 0; wave_idx < num_waves; ++ wave_idx) {
            const auto entry = schedule_entries[wave_idx * kNumSMs + blockIdx.x];
            if (entry.kv_split_begin == entry.kv_split_end)
                continue;

            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1u);
            if (cute::elect_one_sync()) {
                smem.q_blocks[q_stage_idx] = QBlock(
                    entry.q_token_base, entry.num_q_tokens, entry.kv_split_end - entry.kv_split_begin);
                tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(&tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                                                            smem.q[q_stage_idx][0], 0, entry.q_token_base * kNumHeads);
                tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx],
                                                     smem.sf_q[q_stage_idx], 0, entry.q_token_base);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
                                                 smem.weights[q_stage_idx], 0, entry.q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(
                    BLOCK_Q * kNumHeads * (kHeadDim / kPackFactor + sizeof(uint32_t) + sizeof(nv_bfloat16)));
            }

            for (uint32_t kv_split_idx = entry.kv_split_begin; kv_split_idx < entry.kv_split_end; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1u);
                if (cute::elect_one_sync())
                    ptx::cp_async_cg<256>(reinterpret_cast<const uint4*>(&kv_splits[kv_split_idx].header),
                                          reinterpret_cast<uint4*>(&smem.kv_split_headers[kv_stage_idx]));
                #pragma unroll
                for (uint32_t chunk_idx = lane_idx; chunk_idx < kNumKVBlocksPerSplit / 2; chunk_idx += 32) {
                    ptx::cp_async_cg<256>(reinterpret_cast<const uint4*>(kv_splits[kv_split_idx].kv_block_infos) + chunk_idx,
                                          reinterpret_cast<uint4*>(smem.kv_block_infos[kv_stage_idx]) + chunk_idx);
                }
                cutlass::arch::cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t*>(&smem.full_metadata_barriers[kv_stage_idx]));
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1u);
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1u);
        if (cute::elect_one_sync()) {
            smem.q_blocks[q_stage_idx].num_q_tokens = 0;
            smem.kv_split_headers[kv_stage_idx].packed_num_kv_blocks = 0;
            // Publish the generic sentinel with a regular release arrival
            ptx::mbarrier_arrive_count(smem.full_metadata_barriers[kv_stage_idx], 32);
            smem.full_q_barriers[q_stage_idx].arrive();
        }
    } else if (warp_idx == kSFTransposeWarpIdx) {
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
        const auto transpose_sf = [&](uint32_t* smem_ptr, const uint32_t num_sf) {
            for (uint32_t sf_base = 0; sf_base < num_sf; sf_base += kNumUTCCPAlignedElems) {
                uint32_t values[4];
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++ i)
                    values[i] = ptx::ld_shared(smem_ptr + sf_base + i * 32 + lane_idx);
                __syncwarp();
                ptx::st_shared(smem_ptr + sf_base + lane_idx * 4, values[0], values[1], values[2], values[3]);
            }
        };
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);
        RingPipeline<kNumQStages> q_pipeline;
        RingPipeline<kNumKVStages> kv_pipeline;
        while (true) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);
            if (ptx::ld_shared(&smem.q_blocks[q_stage_idx].num_q_tokens) == 0)
                break;
            const uint32_t num_kv_splits = ptx::ld_shared(&smem.q_blocks[q_stage_idx].num_kv_splits);
            transpose_sf(smem.sf_q[q_stage_idx], kNumSFQ);
            cutlass::arch::fence_view_async_shared();
            if (cute::elect_one_sync()) {
                mma::sm100::replace_smem_desc_addr(sf_desc, smem.sf_q[q_stage_idx]);
                cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + q_stage_idx * kNumSFQCols);
                ptx::tcgen05_before_thread_sync();
                smem.full_sf_q_barriers[q_stage_idx].arrive();
            }
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                // The copy warps only signal this after consuming the corresponding metadata.
                smem.full_sf_copy_barriers[kv_stage_idx].wait(kv_phase);
                transpose_sf(smem.sf_kv[kv_stage_idx], SPLIT_KV);
                cutlass::arch::fence_view_async_shared();
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t sf_idx = 0; sf_idx < SPLIT_KV; sf_idx += kNumUTCCPAlignedElems) {
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem.sf_kv[kv_stage_idx] + sf_idx);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc,
                            kTmemStartColOfSFKV + kv_stage_idx * kNumSFKVColsPerStage + sf_idx / 32);
                    }
                    ptx::tcgen05_before_thread_sync();
                    smem.full_kv_barriers[kv_stage_idx].arrive();
                }
            }
        }
    } else if (warp_idx == kUMMAWarpIdx) {
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
        if (cute::elect_one_sync()) {
            using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
            const auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<qk_dtype_t, qk_dtype_t, float,
                cutlass::float_ue8m0_t, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
            RingPipeline<kNumQStages> q_pipeline;
            RingPipeline<kNumKVStages> kv_pipeline;
            RingPipeline<kNumTmemStages> tmem_pipeline;
            while (true) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.full_q_barriers[q_stage_idx].wait(q_phase);
                if (ptx::ld_shared(&smem.q_blocks[q_stage_idx].num_q_tokens) == 0)
                    break;
                const uint32_t num_kv_splits = ptx::ld_shared(&smem.q_blocks[q_stage_idx].num_kv_splits);
                smem.full_sf_q_barriers[q_stage_idx].wait(q_phase);
                ptx::tcgen05_after_thread_sync();
                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                    cutlass::arch::fence_view_async_shared();
                    #pragma unroll
                    for (uint32_t m_idx = 0; m_idx < kNumMathWarpGroups; ++ m_idx) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        const uint32_t tmem_addr = tmem_stage_idx * UMMA_N;
                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1u);
                        ptx::tcgen05_after_thread_sync();
                        #pragma unroll
                        for (uint32_t k_idx = 0; k_idx < kHeadDim / UMMA_K; ++ k_idx) {
                            const uint32_t sf_id = k_idx * kPackFactor;
                            const auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sf_id, sf_id);
                            const auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim / kPackFactor>(
                                                                           smem.kv[kv_stage_idx][0], m_idx * UMMA_M, k_idx * UMMA_K);
                            const auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim / kPackFactor>(
                                                                           smem.q[q_stage_idx][0], 0, k_idx * UMMA_K);
                            mma_op_t::fma(a_desc, b_desc, tmem_addr, k_idx, runtime_instr_desc,
                                          kTmemStartColOfSFKV + kv_stage_idx * kNumSFKVColsPerStage + m_idx * 4,
                                          kTmemStartColOfSFQ + q_stage_idx * kNumSFQCols);
                        }
                        ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
                    }
                    ptx::umma_arrive_no_elect(smem.empty_kv_barriers[kv_stage_idx]);
                }
                ptx::mbarrier_arrive_count(smem.empty_q_barriers[q_stage_idx], 32);
            }
        }
    } else if (warp_idx >= kFirstKVCopyWarpIdx) {
        if (warpgroup_idx == kNumMathWarpGroups)
            cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
        else
            cutlass::arch::warpgroup_reg_dealloc<kNumKVCopyRegisters>();
        constexpr uint32_t kNumKVBlocksPerWarp = UMMA_M / SPARSE_BLOCK_KV;
        constexpr uint32_t kNumChunksPerKVBlock = SPARSE_BLOCK_KV * (kHeadDim / kPackFactor) / 16;
        constexpr uint32_t kNumSFChunksPerKVBlock = SPARSE_BLOCK_KV * sizeof(uint32_t) / 16;
        DG_STATIC_ASSERT(kNumKVBlocksPerWarp % 2 == 0, "Each KV copy warp must process pairs of sparse KV blocks");
        DG_STATIC_ASSERT(kNumKVBlocksPerWarp * kNumSFChunksPerKVBlock == 32, "Each KV copy warp must issue exactly one full-warp SF copy");
        const uint32_t copy_warp_idx = warp_idx - kFirstKVCopyWarpIdx;
        RingPipeline<kNumKVStages> kv_pipeline;
        while (true) {
            CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
            smem.full_metadata_barriers[kv_stage_idx].wait(kv_phase);
            const uint32_t packed_num_kv_blocks = ptx::ld_shared(&smem.kv_split_headers[kv_stage_idx].packed_num_kv_blocks);
            const uint32_t num_kv_blocks = KVSplitHeader::get_num_kv_blocks(packed_num_kv_blocks);
            if (num_kv_blocks == 0)
                break;

            if constexpr (KVAccessor::kSupportsContiguousTMA) {
                if (KVSplitHeader::is_contiguous(packed_num_kv_blocks)) {
                    // One TMA lane replaces all generic copy warps for a contiguous split
                    if (copy_warp_idx == 0 and cute::elect_one_sync()) {
                        const uint32_t kv_token_start = ptx::ld_shared(
                            &smem.kv_block_infos[kv_stage_idx][0].physical_kv_block_idx);
                        kv_accessor.template copy_contiguous_kv_split<SPLIT_KV>(
                            smem.full_kv_barriers[kv_stage_idx], smem.full_sf_copy_barriers[kv_stage_idx],
                            smem.kv[kv_stage_idx][0], smem.sf_kv[kv_stage_idx],
                            kv_token_start);
                    }
                    __syncwarp();
                    // Each copy warp releases its share only after consuming the header
                    const uint32_t num_arrivals = copy_warp_idx == 0 ? 31 : 32;
                    // Predication keeps the sparse copy path from being duplicated
                    const bool is_elected = cute::elect_one_sync();
                    ptx::mbarrier_arrive_count_pred(
                        smem.full_kv_barriers[kv_stage_idx], num_arrivals, is_elected);
                    ptx::mbarrier_arrive_count_pred(
                        smem.full_sf_copy_barriers[kv_stage_idx], num_arrivals, is_elected);
                    continue;
                }
            }

            const uint32_t kv_block_base_in_split = copy_warp_idx * kNumKVBlocksPerWarp;
            if (kv_block_base_in_split < num_kv_blocks) {
                const auto kv_block_ref = kv_accessor.resolve_kv_block(lane_idx < kNumKVBlocksPerWarp ?
                    smem.kv_block_infos[kv_stage_idx][kv_block_base_in_split + lane_idx].physical_kv_block_idx : 0);

                // Copy SF first
                if constexpr (kUseUnalignedKs) {
                    // NOTES: unaligned ks keeps KV rows aligned, but may only 4-byte align SF rows
                    constexpr uint32_t kNumSFBlocksPerIteration = 32 / SPARSE_BLOCK_KV;
                    #pragma unroll
                    for (uint32_t sf_kv_block_base = 0; sf_kv_block_base < kNumKVBlocksPerWarp; sf_kv_block_base += kNumSFBlocksPerIteration) {
                        const uint32_t sf_kv_block_offset = sf_kv_block_base + lane_idx / SPARSE_BLOCK_KV;
                        const uint32_t token_in_kv_block = lane_idx % SPARSE_BLOCK_KV;
                        const auto sf_kv_block = kv_accessor.get_sf_kv_block(kv_block_ref, sf_kv_block_offset);
                        cute::SM80_CP_ASYNC_CACHEALWAYS<uint32_t>::copy(sf_kv_block[token_in_kv_block],
                            smem.sf_kv[kv_stage_idx][(kv_block_base_in_split + sf_kv_block_offset) * SPARSE_BLOCK_KV + token_in_kv_block]);
                    }
                } else {
                    const uint32_t sf_kv_block_offset = lane_idx / kNumSFChunksPerKVBlock;
                    const uint32_t chunk_idx = lane_idx % kNumSFChunksPerKVBlock;
                    const auto sf_kv_block = kv_accessor.get_sf_kv_block(kv_block_ref, sf_kv_block_offset);
                    ptx::cp_async_cg<64>(reinterpret_cast<const uint4*>(sf_kv_block) + chunk_idx,
                                         reinterpret_cast<uint4*>(smem.sf_kv[kv_stage_idx]
                                         + (kv_block_base_in_split + sf_kv_block_offset) * SPARSE_BLOCK_KV) + chunk_idx);
                }
                cutlass::arch::cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t*>(&smem.full_sf_copy_barriers[kv_stage_idx]));

                // Copy sparse KV blocks
                #pragma unroll
                for (uint32_t kv_block_pair_offset = 0; kv_block_pair_offset < kNumKVBlocksPerWarp; kv_block_pair_offset += 2) {
                    const uint32_t kv_block_offset = kv_block_pair_offset + lane_idx / 16;
                    const auto kv_block = kv_accessor.get_kv_block(kv_block_ref, kv_block_offset);
                    #pragma unroll
                    for (uint32_t chunk_base = 0; chunk_base < kNumChunksPerKVBlock; chunk_base += 16) {
                        const uint32_t chunk_idx = chunk_base + lane_idx % 16;
                        ptx::cp_async_cg<256>(reinterpret_cast<const uint4*>(kv_block) + chunk_idx,
                                              reinterpret_cast<uint4*>(smem.kv[kv_stage_idx][0])
                                              + (kv_block_base_in_split + kv_block_offset) * kNumChunksPerKVBlock +
                                                get_swizzled_kv_chunk_idx<kHeadDim / kPackFactor>(chunk_idx));
                    }
                }
            } else {
                cutlass::arch::cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t*>(&smem.full_sf_copy_barriers[kv_stage_idx]));
            }
            cutlass::arch::cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t*>(&smem.full_kv_barriers[kv_stage_idx]));
        }
    } else if (warpgroup_idx < kNumMathWarpGroups) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const uint32_t math_thread_idx = (warp_idx % 4) * 32 + lane_idx;
        RingPipeline<kNumQStages> q_pipeline;
        RingPipeline<kNumKVStages> kv_pipeline;
        RingPipeline<kNumTmemStages> tmem_pipeline;
        tmem_pipeline.advance(warpgroup_idx);

        while (true) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);
            const uint32_t q_token_base = ptx::ld_shared(&smem.q_blocks[q_stage_idx].q_token_base);
            const uint32_t num_q_tokens = ptx::ld_shared(&smem.q_blocks[q_stage_idx].num_q_tokens);
            const uint32_t num_kv_splits = ptx::ld_shared(&smem.q_blocks[q_stage_idx].num_kv_splits);
            if (num_q_tokens == 0)
                break;

            nv_bfloat16* output_rows[BLOCK_Q];
            nv_bfloat162 weights[BLOCK_Q][kNumHeads / 2];
            float accum[kNumHeads / 2];

            #pragma unroll
            for (uint32_t q_token_offset = 0; q_token_offset < BLOCK_Q; ++ q_token_offset) {
                output_rows[q_token_offset] = logits + (q_token_base + q_token_offset) * static_cast<uint64_t>(logits_stride);
                if (q_token_offset >= num_q_tokens)
                    continue;
                const auto packed_weights = reinterpret_cast<const nv_bfloat162*>(smem.weights[q_stage_idx] + q_token_offset * kNumHeads);
                #pragma unroll
                for (uint32_t i = 0; i < kNumHeads / 2; ++ i)
                    weights[q_token_offset][i] = packed_weights[i];
            }

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);

                // One 128-token tile per math warpgroup
                const uint32_t token_in_kv_split = warpgroup_idx * UMMA_M + math_thread_idx;
                const uint32_t kv_block_idx_in_split = token_in_kv_split / SPARSE_BLOCK_KV;
                const uint32_t token_in_kv_block = token_in_kv_split - kv_block_idx_in_split * SPARSE_BLOCK_KV;

                smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                ptx::tcgen05_after_thread_sync();
                // Metadata stays in the generic proxy, so no proxy fence is needed before releasing the stage
                const uint32_t q0_slot_base = ptx::ld_shared(&smem.kv_split_headers[kv_stage_idx].q0_slot_base);
                const uint32_t q1_slot_base = ptx::ld_shared(&smem.kv_split_headers[kv_stage_idx].q1_slot_base);
                const uint32_t packed_slot_offsets = ptx::ld_shared(&smem.kv_block_infos[kv_stage_idx][kv_block_idx_in_split].packed_slot_offsets);
                smem.empty_kv_barriers[kv_stage_idx].arrive();

                #pragma unroll
                for (uint32_t q_token_offset = 0; q_token_offset < BLOCK_Q; ++ q_token_offset) {
                    if (q_token_offset >= num_q_tokens)
                        continue;
                    const uint32_t tmem_addr = tmem_stage_idx * UMMA_N + q_token_offset * kNumHeads;
                    auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                    auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                    #pragma unroll
                    for (uint32_t head_base = 0; head_base < kNumHeads; head_base += kNumHeads / 2) {
                        ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr + head_base,
                                                              reinterpret_cast<uint32_t*>(accum));
                        cutlass::arch::fence_view_async_tmem_load();
                        if (q_token_offset + 1 == num_q_tokens and head_base == kNumHeads / 2) {
                            ptx::tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                        }
                        #pragma unroll
                        for (uint32_t head_offset = 0; head_offset < kNumHeads / 2; head_offset += 4) {
                            const auto accum_pair_0 = make_float2(accum[head_offset], accum[head_offset + 1]);
                            const auto accum_pair_1 = make_float2(accum[head_offset + 2], accum[head_offset + 3]);
                            sum_0 = __hfma2(ptx::cvt_relu_bf16x2_f32(accum_pair_0), weights[q_token_offset][(head_base + head_offset) / 2], sum_0);
                            sum_1 = __hfma2(ptx::cvt_relu_bf16x2_f32(accum_pair_1), weights[q_token_offset][(head_base + head_offset + 2) / 2], sum_1);
                        }
                    }
                    const auto sum = __hadd2_rn(sum_0, sum_1);
                    const nv_bfloat16 reduced = __hadd_rn(sum.x, sum.y);

                    // Map split-local slots back to compressed-logits columns
                    const uint32_t q_slot_offset = (packed_slot_offsets >> (q_token_offset * kNumSparseSlotBits)) & kInvalidSparseSlot;
                    if (q_slot_offset != kInvalidSparseSlot) {
                        const uint32_t q_slot_base = q_token_offset == 0 ? q0_slot_base : q1_slot_base;
                        const uint32_t output_col_idx = (q_slot_base + q_slot_offset) * SPARSE_BLOCK_KV + token_in_kv_block;
                        output_rows[q_token_offset][output_col_idx] = reduced;
                    }
                }
            }
            cutlass::arch::fence_view_async_shared();
            smem.empty_q_barriers[q_stage_idx].arrive();
        }
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    } else {
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
    }
}

template <uint32_t SPARSE_BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages, uint32_t kNumMathWarpGroups, uint32_t kNumSMs, uint32_t BLOCK_Q,
          bool kUseUnalignedKs, bool kIsMXFP4>
CUTLASS_GLOBAL __launch_bounds__(layout::sparse_mqa_logits::get_num_threads(kNumMathWarpGroups), 1)
void sm100_sparse_mqa_logits(const uint32_t logits_stride, nv_bfloat16* logits,
                             const uint8_t* kv, const uint32_t* sf_kv,
                             const uint8_t* metadata, const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_weights,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv) {
    // Keep the CUTLASS dtype out of the kernel template signature to avoid ptxas register spills
    using qk_dtype_t = cute::conditional_t<kIsMXFP4, cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
    const auto kv_accessor = sparse_mqa_detail::ContiguousSparseKVAccessor<SPARSE_BLOCK_KV, qk_dtype_t>(
        kv, sf_kv, &tensor_map_kv, &tensor_map_sf_kv);
    sm100_sparse_mqa_logits_core_impl<SPARSE_BLOCK_KV, kNumQStages, kNumKVStages, kNumTmemStages,
        kNumMathWarpGroups, kNumSMs, BLOCK_Q, kUseUnalignedKs, qk_dtype_t>(
            logits_stride, logits, metadata, tensor_map_q, tensor_map_sf_q, tensor_map_weights, kv_accessor);
}

template <uint32_t PAGE_KV, uint32_t SPARSE_BLOCK_KV, uint32_t kNumQStages,
          uint32_t kNumKVStages, uint32_t kNumTmemStages, uint32_t kNumMathWarpGroups,
          uint32_t kNumSMs, uint32_t BLOCK_Q, bool kIsMXFP4>
CUTLASS_GLOBAL __launch_bounds__(layout::sparse_mqa_logits::get_num_threads(kNumMathWarpGroups), 1)
void sm100_paged_sparse_mqa_logits(const uint32_t logits_stride, const uint32_t kv_page_stride_bytes,
                                   nv_bfloat16* logits,
                                   const uint8_t* fused_kv_cache, const uint8_t* metadata,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // Keep the CUTLASS dtype out of the kernel template signature to avoid ptxas register spills
    using qk_dtype_t = cute::conditional_t<kIsMXFP4, cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
    DG_STATIC_ASSERT(PAGE_KV % SPARSE_BLOCK_KV == 0, "Sparse KV blocks must not cross pages");
    DG_DEVICE_ASSERT(kv_page_stride_bytes % 512 == 0);
    const auto kv_accessor = sparse_mqa_detail::PagedSparseKVAccessor<PAGE_KV, SPARSE_BLOCK_KV, qk_dtype_t>(
        fused_kv_cache, kv_page_stride_bytes);
    sm100_sparse_mqa_logits_core_impl<SPARSE_BLOCK_KV, kNumQStages, kNumKVStages, kNumTmemStages,
        kNumMathWarpGroups, kNumSMs, BLOCK_Q, false, qk_dtype_t>(
            logits_stride, logits, metadata, tensor_map_q, tensor_map_sf_q, tensor_map_weights, kv_accessor);
}

} // namespace deep_gemm
