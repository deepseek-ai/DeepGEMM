#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::layout {

template <bool kIsFP4, uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages,
          typename reduce_dtype_t = float>
struct MQALogitsSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using qk_dtype_t = cute::conditional_t<kIsFP4, uint8_t, __nv_fp8_e4m3>;
    using sf_dtype_t = cute::conditional_t<kIsFP4, uint32_t, float>;

    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t kQKBytesPerElem = kIsFP4 ? 1 : sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    // FP4 needs `8 * (head_dim / 2)` swizzle bytes; FP8 uses fixed 512B swizzle alignment
    static constexpr uint32_t kSwizzleAlignment = kIsFP4 ? (8 * kNumQKBytesPerToken) : 512;
    static constexpr uint32_t kNumSFQ = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumQBytesPerStage = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t kNumKVBytesPerStage = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t kNumQElementsPerStage = kNumQBytesPerStage / kQKBytesPerElem;
    static constexpr uint32_t kNumKVElementsPerStage = kNumKVBytesPerStage / kQKBytesPerElem;
    // FP4 stores per-block scale factors; FP8 stores one per-KV scale and no Q scale
    static constexpr uint32_t kNumScaleQ = kIsFP4 ? kNumSFQ : 1;
    static constexpr uint32_t kNumScaleKV = kIsFP4 ? kNumSFKV : SPLIT_KV;
    // FP4 SF arrays need 16B alignment; FP8 KV scales need 512B TMA alignment
    static constexpr uint32_t kScaleAlignment = kIsFP4 ? 16 : 512;

    // Pad weights stages to 128B so each TMA stage start is aligned
    // bf16 stages are otherwise only half the float byte size
    static constexpr uint32_t kWeightsTmaAlignment = 128;
    static constexpr uint32_t kNumWeightsElementsPerStage =
        math::constexpr_align(BLOCK_Q * kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), kWeightsTmaAlignment)
        / static_cast<uint32_t>(sizeof(reduce_dtype_t));

    DG_STATIC_ASSERT(kNumQBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kNumKVBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    alignas(kSwizzleAlignment) qk_dtype_t smem_q[kNumQStages][kNumQElementsPerStage];
    alignas(kSwizzleAlignment) qk_dtype_t smem_kv[kNumKVStages][kNumKVElementsPerStage];
    alignas(kScaleAlignment) sf_dtype_t smem_sf_q[kNumQStages][kNumScaleQ];
    alignas(kScaleAlignment) sf_dtype_t smem_sf_kv[kNumKVStages][kNumScaleKV];
    alignas(kWeightsTmaAlignment) reduce_dtype_t smem_weights[kNumQStages][kNumWeightsElementsPerStage];
    Barrier full_q_barriers[kNumQStages];
    Barrier empty_q_barriers[kNumQStages];
    Barrier full_kv_barriers[kNumKVStages];
    Barrier empty_kv_barriers[kNumKVStages];
    Barrier full_tmem_barriers[kNumTmemStages];
    Barrier empty_tmem_barriers[kNumTmemStages];
    uint32_t tmem_ptr_in_smem;
};

} // namespace deep_gemm::layout
