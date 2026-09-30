#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::layout::mega_gate {

static constexpr uint32_t BLOCK_K = 64;
static constexpr uint32_t kSwizzleMode = 128;
static constexpr uint32_t kNumNonEpilogueThreads = 128;
static constexpr uint32_t kNumEpilogueStages = 2;
static constexpr uint32_t kNumMaxBlockTokens = 256;
static constexpr uint32_t kExpertAlignment = 128;
static constexpr uint32_t kSharedMemoryAlignment = 1024;

static constexpr uint32_t kNumMinBlockTokens = 16;
static constexpr uint32_t kNumMaxTokens = 1u << 20;
static constexpr uint32_t kNumMaxTokenBlocks = math::constexpr_ceil_div(kNumMaxTokens, kNumMinBlockTokens);

static constexpr uint32_t kScoreBarrierLineBytes = 128;
static constexpr uint32_t kNumMaxLogicalCtas = 64;
static constexpr uint32_t kNumMaxMetadataCacheBytes = 4 * 1024;

// Top-k geometry: each lane holds `kNumExpertsPerLaneVector` consecutive experts of every 128-expert wave
static constexpr uint32_t kNumExpertsPerLaneVector = 4;
static constexpr uint32_t kNumExpertsPerWave = 32 * kNumExpertsPerLaneVector;
DG_STATIC_ASSERT(kNumExpertsPerWave == kExpertAlignment, "Expert waves must match the expert alignment");

struct RoutingArgs {
    const int* to_physical_map;
    const int* logical_count;
    int64_t* topk_idx;
    int64_t* unmapped_topk_idx;
    float* topk_weights;
    int64_t unmapped_topk_idx_stride;
    uint32_t num_routed_experts;
    uint32_t num_shared_experts;
    uint32_t num_duplicate_experts;
    uint32_t rank_idx;
    float routed_scaling_factor;
};

// Bias, image bias and logical count share the 4 KiB cache in that order, the count only when it fits
CUTLASS_HOST_DEVICE constexpr bool caches_logical_count(const uint32_t num_routed_experts, const bool has_bias,
                                                        const bool has_image_token_mask, const bool has_physical_map) {
    const auto num_bias_cache_bytes = num_routed_experts * ((has_bias ? 1u : 0u) + (has_image_token_mask ? 1u : 0u)) *
                                      static_cast<uint32_t>(sizeof(float));
    return has_physical_map and num_bias_cache_bytes + num_routed_experts * static_cast<uint32_t>(sizeof(int)) <= kNumMaxMetadataCacheBytes;
}

static constexpr uint32_t kNumMaxStages = 32;

// Fixed SMEM header, followed by `num_stages` x stages then `num_stages` weight stages (runtime strides)
struct SharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    alignas(Barrier) Barrier full_barriers[kNumMaxStages];
    alignas(Barrier) Barrier empty_barriers[kNumMaxStages];
    alignas(Barrier) Barrier tmem_full_barriers[kNumEpilogueStages];
    alignas(Barrier) Barrier tmem_empty_barriers[kNumEpilogueStages];
    alignas(uint32_t) uint32_t tmem_ptr;
    alignas(4 * sizeof(float)) float metadata_cache[kNumMaxMetadataCacheBytes / sizeof(float)];
};

static constexpr uint32_t kStageDataOffset = math::constexpr_align(static_cast<uint32_t>(sizeof(SharedStorage)), kSharedMemoryAlignment);

CUTLASS_DEVICE cutlass::bfloat16_t* get_x_stage_ptr(uint8_t* smem_buffer, const uint32_t& stage_idx, const uint32_t& x_stage_bytes) {
    return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + kStageDataOffset + stage_idx * x_stage_bytes);
}

CUTLASS_DEVICE cutlass::bfloat16_t* get_weight_stage_ptr(uint8_t* smem_buffer, const uint32_t& stage_idx, const uint32_t& num_stages,
                                                         const uint32_t& x_stage_bytes, const uint32_t& w_stage_bytes) {
    return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + kStageDataOffset + num_stages * x_stage_bytes + stage_idx * w_stage_bytes);
}

// View over global score scratch: [num_token_blocks * block_tokens, kNumSplits, kNumExperts]
template <uint32_t kNumSplits = 0, uint32_t kNumExperts = 0>
struct Workspace {
    static constexpr uint32_t kTokenStride = kNumSplits * kNumExperts;

    void* gmem_scratch;
    void* gmem_score_barriers;

    CUTLASS_HOST_DEVICE
    Workspace(void* gmem_scratch, void* gmem_score_barriers):
        gmem_scratch(gmem_scratch), gmem_score_barriers(gmem_score_barriers) {}

    CUTLASS_HOST_DEVICE
    static constexpr uint64_t get_num_scratch_bytes(const uint32_t num_token_blocks,
                                                    const uint32_t num_splits,
                                                    const uint32_t block_tokens,
                                                    const uint32_t num_experts) {
        return static_cast<uint64_t>(num_token_blocks) * num_splits * block_tokens * num_experts * sizeof(float);
    }

    CUTLASS_HOST_DEVICE
    float* get_score_ptr(const uint32_t token_idx, const uint32_t split_idx = 0) const {
        return reinterpret_cast<float*>(gmem_scratch) +
               static_cast<uint64_t>(token_idx) * kTokenStride + split_idx * kNumExperts;
    }

    CUTLASS_HOST_DEVICE uint64_t* get_score_barrier_ptr(const uint32_t token_block_idx) const {
        return reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(gmem_score_barriers) +
                                           token_block_idx * kScoreBarrierLineBytes);
    }
};

} // namespace deep_gemm::layout::mega_gate
