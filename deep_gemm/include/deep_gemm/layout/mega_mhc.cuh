#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::layout::mega_mhc {

// Model geometry
static constexpr uint32_t kNumRoutes = 4;
static constexpr uint32_t kNumHCOutputs = kNumRoutes * (kNumRoutes + 2);

// FP8 scale-factor layout
static constexpr uint32_t kHiddenPerSF = 32;
static constexpr uint32_t kHiddenPerSFWord = 128;

// Kernel tile and pipeline geometry
static constexpr uint32_t BLOCK_M = 64;
static constexpr uint32_t BLOCK_N = kNumHCOutputs;
static constexpr uint32_t BLOCK_K = 64;
static constexpr uint32_t kSwizzleMode = 128;
static constexpr uint32_t kFnAtomK = kSwizzleMode / sizeof(float);
static constexpr uint32_t kSwizzleAlignment = 8 * kSwizzleMode;
static constexpr uint32_t kNumIOStages = 4;
static constexpr uint32_t kNumFnStages = 2;
static constexpr uint32_t kNumAStages = 4;
static constexpr uint32_t kNumAccumStages = 2;
static constexpr uint32_t kNumCoeffStages = 2;
static constexpr uint32_t kNumWarpsPerWG = 4;
static constexpr uint32_t kNumThreads = 6 * kNumWarpsPerWG * 32;

// Split-K workspace geometry
static constexpr uint32_t kNumGemmPartialElementsPerTask = BLOCK_M * kNumHCOutputs;
static constexpr uint32_t kNumSqrSumPartialElementsPerTask = BLOCK_M;
static constexpr uint32_t kDefaultNumSplits = 16;
static constexpr uint32_t kNumMaxSplits = 64;
static constexpr uint32_t kNumSplitBarriers = 2;
static constexpr uint64_t kSplitBarrierLineBytes = 128;
static constexpr uint32_t kNumMaxTokens = 1u << 20;
static constexpr uint32_t kNumMaxMBlocks = math::constexpr_ceil_div(kNumMaxTokens, BLOCK_M);

// Host/device ABI payloads for the two consumer domains. Tensor maps remain
// standalone kernel parameters so TMA descriptors keep direct grid-constant access.
struct MixArgs {
    const float* scales;
    const float* bases;
    float* new_prev_mix;
    float* new_post_mix;
    float* new_comb_res_mix;
    float hc_norm_eps;
    float hc_pre_eps;
    float hc_post_scale;
    float sinkhorn_eps;
    uint32_t num_sinkhorn_iters;
};

struct NormArgs {
    uint32_t num_tokens;
    const nv_bfloat16* weight;
    const nv_bfloat16* new_residual;
    float eps;
    float scale;
    nv_bfloat16* y_bf16;
    __nv_fp8_e4m3* y_fp8;
    uint32_t* y_primary_sf;
    int64_t y_primary_sf_stride_token;
    int64_t y_primary_sf_stride_word;
    uint32_t* y_shared_sf;
    int64_t y_shared_sf_stride_word;
};

// View over global split partial scratch.
//
// Data layout:
//   gemm_partials            [num_m_blocks, kNumSplits, BLOCK_M, kNumHCOutputs]
//   hc_norm_sqr_sum_partials [num_m_blocks, kNumSplits, BLOCK_M]
//   x1_sqr_sum_partials      [num_m_blocks, kNumSplits, BLOCK_M]
// where task_idx = m_block_idx * kNumSplits + k_split_idx.
template <uint32_t kNumSplits = kDefaultNumSplits>
struct Workspace {
    void* gmem_scratch;
    uint32_t num_m_blocks;

    DG_STATIC_ASSERT(kNumSplits > 0, "The number of splits must be positive");
    DG_STATIC_ASSERT(kNumSplits <= kNumMaxSplits, "The number of splits exceeds the supported maximum");

    CUTLASS_HOST_DEVICE
    Workspace(void* gmem_scratch, const uint32_t num_m_blocks):
        gmem_scratch(gmem_scratch), num_m_blocks(num_m_blocks) {}

    CUTLASS_HOST_DEVICE
    static constexpr uint32_t get_num_m_blocks(const uint32_t num_tokens) {
        return math::constexpr_ceil_div<uint32_t>(num_tokens, BLOCK_M);
    }

    CUTLASS_HOST_DEVICE
    static constexpr uint64_t get_num_scratch_bytes(const uint32_t num_m_blocks, const uint32_t num_splits = kNumSplits) {
        const auto num_tasks = static_cast<uint64_t>(num_m_blocks) * num_splits;
        return num_tasks * (kNumGemmPartialElementsPerTask + 2 * kNumSqrSumPartialElementsPerTask) * sizeof(float);
    }

    CUTLASS_HOST_DEVICE
    float* get_gemm_partial_ptr(const uint32_t task_idx = 0) const {
        return reinterpret_cast<float*>(gmem_scratch) + static_cast<uint64_t>(task_idx) * kNumGemmPartialElementsPerTask;
    }

    CUTLASS_HOST_DEVICE
    float* get_hc_norm_sqr_sum_partial_ptr(const uint32_t task_idx = 0) const {
        const auto base = get_gemm_partial_ptr(num_m_blocks * kNumSplits);
        return base + static_cast<uint64_t>(task_idx) * kNumSqrSumPartialElementsPerTask;
    }

    CUTLASS_HOST_DEVICE
    float* get_x1_sqr_sum_partial_ptr(const uint32_t task_idx = 0) const {
        const auto base = get_hc_norm_sqr_sum_partial_ptr(num_m_blocks * kNumSplits);
        return base + static_cast<uint64_t>(task_idx) * kNumSqrSumPartialElementsPerTask;
    }

};

struct SharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Route-major TMA layouts keep every 64 x 64 destination contiguous.
    alignas(kSwizzleAlignment) nv_bfloat162 residual[kNumIOStages][kNumRoutes][BLOCK_M * BLOCK_K / 2];
    alignas(kSwizzleAlignment) nv_bfloat162 x[kNumIOStages][BLOCK_M * BLOCK_K / 2];

    // Each Fn TMA fills all routes for one 32-element TF32 swizzle atom.
    alignas(kSwizzleAlignment) float fn[kNumFnStages][BLOCK_K / kFnAtomK][kNumRoutes][kNumHCOutputs * kFnAtomK];

    // Coeff tensors retain their TMA-native per-token layouts.
    alignas(kSwizzleMode) float pre_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes];
    alignas(kSwizzleMode) float post_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes];
    alignas(kSwizzleMode) float comb_coeff[kNumCoeffStages][BLOCK_M][kNumRoutes * kNumRoutes];

    // Post workers own independent partials reduced before the Mix/Norm barrier arrivals.
    alignas(kSwizzleMode) float hc_norm_sqr_sums[2][BLOCK_M];
    alignas(kSwizzleMode) float x1_sqr_sums[2][BLOCK_M];
    // Fused Normal workers use dedicated scratch instead of aliasing Shifted statistics.
    alignas(kSwizzleMode) float normal_scratch[BLOCK_M];

    // Full/empty pairs connect static roles without CTA-wide block-boundary synchronization.
    alignas(Barrier) Barrier full_io_barriers[kNumIOStages];
    alignas(Barrier) Barrier empty_io_barriers[kNumIOStages];
    alignas(Barrier) Barrier store_ready_io_barriers[kNumIOStages];

    alignas(Barrier) Barrier full_fn_barriers[kNumFnStages];
    alignas(Barrier) Barrier empty_fn_barriers[kNumFnStages];

    alignas(Barrier) Barrier full_a_barriers[kNumAStages];
    alignas(Barrier) Barrier empty_a_barriers[kNumAStages];

    alignas(Barrier) Barrier full_accum_barriers[kNumAccumStages];
    alignas(Barrier) Barrier empty_accum_barriers[kNumAccumStages];

    alignas(Barrier) Barrier full_coeff_barriers[kNumCoeffStages];
    alignas(Barrier) Barrier empty_coeff_barriers[kNumCoeffStages];

    alignas(Barrier) Barrier full_stats_barrier;
    alignas(Barrier) Barrier empty_stats_barrier;

    // Gate each split's Mix arrival on all WG3 warps and, in Normal, the output-store worker.
    alignas(Barrier) Barrier mix_arrival_barrier;
    alignas(uint32_t) uint32_t tmem_base;
    // Shifted Norm and fused Normal workers share this mutually exclusive task queue.
    alignas(uint32_t) uint32_t next_norm_task_ticket;
};

} // namespace deep_gemm::layout::mega_mhc
