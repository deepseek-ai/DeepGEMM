#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::layout {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t UMMA_N,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages,
          typename qk_dtype_t>
struct MQALogitsSharedStorage {
    static constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t kQKBytesPerElem = sizeof(qk_dtype_t);
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
    // Align to one 8-row Q/K swizzle tile: FP4 uses head_dim / 2 bytes, FP8 use head_dim bytes per token
    static constexpr uint32_t kSwizzleAlignment = 8 * kNumQKBytesPerToken;
    static constexpr uint32_t kNumSFQ = math::constexpr_align(UMMA_N, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumQBytesPerStage = UMMA_N * kNumQKBytesPerToken;
    static constexpr uint32_t kNumKVBytesPerStage = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t kNumQElementsPerStage = kNumQBytesPerStage / kQKBytesPerElem;
    static constexpr uint32_t kNumKVElementsPerStage = kNumKVBytesPerStage / kQKBytesPerElem;
    // TMA destinations in shared memory must be 128-byte aligned.
    static constexpr uint32_t kTmaAlignment = 128;
    static constexpr uint32_t kNumWeightBytesPerRow = math::constexpr_align(
        kNumHeads * static_cast<uint32_t>(sizeof(nv_bfloat16)), 16u);
    static constexpr uint32_t kNumWeightElementsPerRow =
        kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(nv_bfloat16));
    static constexpr uint32_t kNumWeightBytesPerStage = math::constexpr_align(
        BLOCK_Q * kNumWeightBytesPerRow, kTmaAlignment);
    static constexpr uint32_t kNumWeightElementsPerStage =
        kNumWeightBytesPerStage / static_cast<uint32_t>(sizeof(nv_bfloat16));

    DG_STATIC_ASSERT(UMMA_N == math::constexpr_align(BLOCK_QH, 8u), "Invalid Q tile shape");
    DG_STATIC_ASSERT(kNumQBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kNumKVBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kSwizzleAlignment % 128 == 0, "TMA destination must be 128-byte aligned");
    DG_STATIC_ASSERT(kTmaAlignment % 128 == 0, "TMA destination must be 128-byte aligned");
    DG_STATIC_ASSERT(kNumWeightBytesPerStage % kTmaAlignment == 0, "Unaligned weight TMA stage");

    alignas(kSwizzleAlignment) qk_dtype_t smem_q[kNumQStages][kNumQElementsPerStage];
    alignas(kSwizzleAlignment) qk_dtype_t smem_kv[kNumKVStages][kNumKVElementsPerStage];
    alignas(kTmaAlignment) uint32_t smem_sf_q[kNumQStages][kNumSFQ];
    alignas(kTmaAlignment) uint32_t smem_sf_kv[kNumKVStages][kNumSFKV];
    alignas(kTmaAlignment) nv_bfloat16 smem_weights[kNumQStages][kNumWeightElementsPerStage];
    // Barriers require 8-byte alignment, already guaranteed by the preceding TMA-aligned arrays.
    Barrier full_q_barriers[kNumQStages];
    Barrier full_sf_q_barriers[kNumQStages];
    Barrier empty_q_barriers[kNumQStages];
    Barrier full_kv_barriers[kNumKVStages];
    Barrier full_sf_kv_barriers[kNumKVStages];
    Barrier empty_kv_barriers[kNumKVStages];
    Barrier full_tmem_barriers[kNumTmemStages];
    Barrier empty_tmem_barriers[kNumTmemStages];
    uint32_t tmem_ptr_in_smem;
};

} // namespace deep_gemm::layout
