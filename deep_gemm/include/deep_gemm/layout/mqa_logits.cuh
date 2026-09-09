#pragma once

#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::layout {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages,
          typename qk_dtype_t, typename reduce_dtype_t = float>
struct MQALogitsSharedStorage {
    static constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using sf_dtype_t = cute::conditional_t<kIsMXSF, uint32_t, float>;

    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t kQKBytesPerElem = sizeof(qk_dtype_t);
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    // Align to one 8-row Q/K swizzle tile: FP4 uses head_dim / 2 bytes, FP8 use head_dim bytes per token
    static constexpr uint32_t kSwizzleAlignment = 8 * kNumQKBytesPerToken;
    static constexpr uint32_t kNumSFQ = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumQBytesPerStage = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t kNumKVBytesPerStage = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t kNumQElementsPerStage = kNumQBytesPerStage / kQKBytesPerElem;
    static constexpr uint32_t kNumKVElementsPerStage = kNumKVBytesPerStage / kQKBytesPerElem;
    // MX SF formats store per-block scale factors; FP8 stores one per-KV scale and no Q scale
    static constexpr uint32_t kNumScaleQ = kIsMXSF ? kNumSFQ : 1;
    static constexpr uint32_t kNumScaleKV = kIsMXSF ? kNumSFKV : SPLIT_KV;
    // TMA destinations in shared memory must be 128-byte aligned.
    static constexpr uint32_t kTmaAlignment = 128;

    DG_STATIC_ASSERT(kNumQBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kNumKVBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kSwizzleAlignment % 128 == 0, "TMA destination must be 128-byte aligned");
    DG_STATIC_ASSERT(kTmaAlignment % 128 == 0, "TMA destination must be 128-byte aligned");

    alignas(kSwizzleAlignment) qk_dtype_t smem_q[kNumQStages][kNumQElementsPerStage];
    alignas(kSwizzleAlignment) qk_dtype_t smem_kv[kNumKVStages][kNumKVElementsPerStage];
    alignas(kTmaAlignment) sf_dtype_t smem_sf_q[kNumQStages][kNumScaleQ];
    alignas(kTmaAlignment) sf_dtype_t smem_sf_kv[kNumKVStages][kNumScaleKV];
    alignas(kTmaAlignment) reduce_dtype_t smem_weights[kNumQStages][BLOCK_Q * kNumHeads];
    // Barriers require 8-byte alignment, already guaranteed by the preceding TMA-aligned arrays.
    Barrier full_q_barriers[kNumQStages];
    Barrier empty_q_barriers[kNumQStages];
    Barrier full_kv_barriers[kNumKVStages];
    Barrier empty_kv_barriers[kNumKVStages];
    Barrier full_tmem_barriers[kNumTmemStages];
    Barrier empty_tmem_barriers[kNumTmemStages];
    uint32_t tmem_ptr_in_smem;
};

} // namespace deep_gemm::layout
