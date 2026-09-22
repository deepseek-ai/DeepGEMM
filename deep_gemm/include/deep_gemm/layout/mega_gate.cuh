#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::layout::mega_gate {

// Kernel geometry and alignments
static constexpr uint32_t BLOCK_K = 64;
static constexpr uint32_t kNumNonEpilogueThreads = 128;
static constexpr uint32_t kNumEpilogueStages = 2;
static constexpr uint32_t kNumMaxBlockTokens = 256;
static constexpr uint32_t kExpertAlignment = 128;
static constexpr uint32_t kSharedMemoryAlignment = 1024;

// Token bounds
static constexpr uint32_t kNumMinBlockTokens = 16;
static constexpr uint32_t kNumMaxTokens = 1u << 20;
static constexpr uint32_t kNumMaxTokenBlocks = math::constexpr_ceil_div(kNumMaxTokens, kNumMinBlockTokens);

// Score-barrier workspace geometry
static constexpr uint32_t kScoreBarrierLineBytes = 128;
static constexpr uint32_t kNumMaxLogicalCtas = 64;
static constexpr uint32_t kNumMaxMetadataCacheBytes = 4 * 1024;

enum class ScoringType : uint32_t {
    Sigmoid = 0,
    SqrtSoftplus = 1,
    Identity = 3,
};

CUTLASS_HOST_DEVICE constexpr uint32_t get_num_bias_cache_bytes(const uint32_t num_routed_experts,
                                                                const bool has_bias,
                                                                const bool has_image_token_mask) {
    return num_routed_experts * ((has_bias ? 1u : 0u) + (has_image_token_mask ? 1u : 0u)) *
           static_cast<uint32_t>(sizeof(float));
}

CUTLASS_HOST_DEVICE constexpr bool caches_logical_count(const uint32_t num_routed_experts,
                                                        const bool has_bias, const bool has_image_token_mask,
                                                        const bool to_physical_map_exists) {
    return to_physical_map_exists and
           get_num_bias_cache_bytes(num_routed_experts, has_bias, has_image_token_mask) +
           num_routed_experts * static_cast<uint32_t>(sizeof(int)) <= kNumMaxMetadataCacheBytes;
}

CUTLASS_HOST_DEVICE constexpr uint32_t get_num_metadata_cache_bytes(const uint32_t num_routed_experts,
                                                                    const bool has_bias,
                                                                    const bool has_image_token_mask,
                                                                    const bool to_physical_map_exists) {
    return get_num_bias_cache_bytes(num_routed_experts, has_bias, has_image_token_mask) +
           (caches_logical_count(num_routed_experts, has_bias, has_image_token_mask, to_physical_map_exists)
                ? num_routed_experts * static_cast<uint32_t>(sizeof(int)) : 0u);
}

template <uint32_t kNumStages, uint32_t kNumEpilogueStages,
          uint32_t kNumRoutedExperts,
          uint32_t LOAD_BLOCK_M, uint32_t LOAD_BLOCK_N,
          uint32_t BLOCK_K_,
          bool kHasBias, bool kHasImageTokenMask,
          bool kToPhysicalMapExists>
struct SharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using x_stage_t = cutlass::bfloat16_t[LOAD_BLOCK_M * BLOCK_K_];
    using weight_stage_t = cutlass::bfloat16_t[LOAD_BLOCK_N * BLOCK_K_];

    static constexpr bool kCacheLogicalCount = caches_logical_count(kNumRoutedExperts, kHasBias,
                                                                    kHasImageTokenMask, kToPhysicalMapExists);
    static constexpr bool kHasMetadataCache = kHasBias or kHasImageTokenMask or kCacheLogicalCount;
    static constexpr uint32_t kNumMetadataCacheElems = get_num_metadata_cache_bytes(kNumRoutedExperts, kHasBias,
                                                                                    kHasImageTokenMask, kToPhysicalMapExists) / sizeof(float);

    alignas(kSharedMemoryAlignment) x_stage_t x[kNumStages];
    alignas(kSharedMemoryAlignment) weight_stage_t weight[kNumStages];

    alignas(Barrier) Barrier full_barriers[kNumStages];
    alignas(Barrier) Barrier empty_barriers[kNumStages];
    alignas(Barrier) Barrier tmem_full_barriers[kNumEpilogueStages];
    alignas(Barrier) Barrier tmem_empty_barriers[kNumEpilogueStages];
    alignas(Barrier) Barrier metadata_ready_barrier;
    alignas(uint32_t) uint32_t tmem_ptr;

    alignas(4 * sizeof(float)) float metadata_cache[kNumMetadataCacheElems > 0 ? kNumMetadataCacheElems : 1];

    CUTLASS_DEVICE float* get_bias_ptr() {
        return metadata_cache;
    }

    CUTLASS_DEVICE float* get_image_bias_ptr() {
        return metadata_cache + (kHasBias ? kNumRoutedExperts : 0);
    }

    CUTLASS_DEVICE int* get_logical_count_ptr() {
        return reinterpret_cast<int*>(get_image_bias_ptr() + (kHasImageTokenMask ? kNumRoutedExperts : 0));
    }
};

// View over global score scratch: [num_token_blocks, kNumSplits, kBlockTokens, kNumExperts]
template <uint32_t kNumSplits = 0, uint32_t kBlockTokens = 0, uint32_t kNumExperts = 0>
struct Workspace {
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
    float* get_score_ptr(const uint32_t token_block_idx, const uint32_t split_idx = 0) const {
        return reinterpret_cast<float*>(gmem_scratch) +
               (static_cast<uint64_t>(token_block_idx) * kNumSplits + split_idx) * kBlockTokens * kNumExperts;
    }

    CUTLASS_HOST_DEVICE uint64_t* get_score_barrier_ptr(const uint32_t token_block_idx) const {
        return reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(gmem_score_barriers) +
                                           token_block_idx * kScoreBarrierLineBytes);
    }
};

} // namespace deep_gemm::layout::mega_gate
