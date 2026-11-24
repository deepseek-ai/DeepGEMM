#pragma once
#include <deep_gemm/layout/mega_mhc.cuh>

namespace deep_gemm::layout::sm90_mega_mhc {
using namespace deep_gemm::layout::mega_mhc;
// Three IO stages leave space for the four-stage shared A ring on Hopper.
static constexpr uint32_t kNumIOStages = 3;
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
    // Normal workers and shifted decode Mix are mutually exclusive. The larger
    // view fits the padding before A, keeping the total shared allocation fixed.
    union {
        alignas(kSwizzleMode) float normal_scratch[BLOCK_M];
        alignas(kSwizzleMode) float mix_scratch[kNumWarpsPerWG][32];
    };

    // Four TF32 A atoms, each in a WGMMA-compatible 32-byte swizzle.
    alignas(1024) float a[kNumAStages][kNumRoutes][BLOCK_M * 8];

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
    // Shifted Norm and fused Normal workers share this mutually exclusive task queue.
    alignas(uint32_t) uint32_t next_norm_task_ticket;
};


} // namespace deep_gemm::layout::sm90_mega_mhc
