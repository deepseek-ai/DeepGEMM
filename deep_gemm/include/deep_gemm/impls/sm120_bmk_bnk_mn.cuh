#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>

#include <cute/int_tuple.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm120_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSplitFactor,
          uint32_t kSwizzleABMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm120_bmn_bnk_mn_gemm_impl(const uint32_t shape_s,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                           float *d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    namespace sm120_mma = mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = 16;
    static constexpr uint32_t MMA_N = 8;
    static constexpr uint32_t MMA_K = sm120_mma::BF16_MMA_K;
    static constexpr uint32_t MMA_ACCUM = 4;

    DG_STATIC_ASSERT(BLOCK_M == 128, "Invalid block M");
    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Invalid number of TMA threads");
    DG_STATIC_ASSERT(kNumMathThreads == 256, "Invalid number of math threads");
    DG_STATIC_ASSERT(BLOCK_M % MMA_M == 0 and BLOCK_N % MMA_N == 0 and BLOCK_K % MMA_K == 0, "Invalid block dims");

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kMTilesPerWarp = BLOCK_M / kNumMathWarps / MMA_M;
    static constexpr uint32_t kNTiles = BLOCK_N / MMA_N;
    static constexpr uint32_t kKSteps = BLOCK_K / MMA_K;
    static constexpr uint32_t kAccumPerWarp = kMTilesPerWarp * kNTiles * MMA_ACCUM;

    DG_STATIC_ASSERT(BLOCK_M == kNumMathWarps * kMTilesPerWarp * MMA_M, "M tiles must divide evenly");
    DG_STATIC_ASSERT(kNTiles % 2 == 0, "kNTiles must be even for ldmatrix.x2 B loading");

    static constexpr uint32_t kSMEMKBytes = BLOCK_K * 2;
    static constexpr uint32_t SMEM_A = BLOCK_M * kSMEMKBytes;
    static constexpr uint32_t SMEM_B = BLOCK_N * kSMEMKBytes;
    static constexpr uint32_t SMEM_TMA_BYTES = SMEM_A + SMEM_B;
    static constexpr uint32_t kLdmK = MMA_K * 2;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_a = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + s * SMEM_A);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + kNumStages * SMEM_A + s * SMEM_B);
    });

    constexpr uint32_t BAR_BASE = kNumStages * (SMEM_A + SMEM_B);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + s * sizeof(Barrier));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + (kNumStages + s) * sizeof(Barrier));
    });

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Grid scheduling
    const uint32_t num_n_blocks = math::ceil_div(SHAPE_N, BLOCK_N);
    const uint32_t num_mn_blocks = num_n_blocks * math::ceil_div(SHAPE_M, BLOCK_M);
    const uint32_t mn_block_idx = blockIdx.x % num_mn_blocks;
    const uint32_t sk_block_idx = blockIdx.x / num_mn_blocks;
    const uint32_t n_block_idx = mn_block_idx % num_n_blocks;
    const uint32_t m_block_idx = mn_block_idx / num_n_blocks;
    const uint32_t num_total_stages = cute::min(kSplitFactor, shape_s * (SHAPE_K / BLOCK_K) - sk_block_idx * kSplitFactor);

    cudaGridDependencySynchronize();

    // PRODUCER WARP GROUP (TMA warps, 40 regs)
    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<40>();

        if (warp_idx == kNumMathWarps and lane_idx == 0) {
            for (uint32_t s = 0; s < num_total_stages; ++s) {
                const auto stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

                const uint32_t sk_idx = (sk_block_idx * kSplitFactor + s) * BLOCK_K;
                const uint32_t k_idx = sk_idx % SHAPE_K;
                const uint32_t s_idx = sk_idx / SHAPE_K;

                auto& full_barrier = *full_barriers[stage_idx];
                tma::copy<kSMEMKBytes, BLOCK_M, kSwizzleABMode, char>(
                    &tensor_map_a, &full_barrier, smem_a[stage_idx],
                    k_idx, m_block_idx * BLOCK_M + s_idx * SHAPE_M, 1);
                tma::copy<kSMEMKBytes, BLOCK_N, kSwizzleABMode, char>(
                    &tensor_map_b, &full_barrier, smem_b[stage_idx],
                    k_idx, n_block_idx * BLOCK_N + s_idx * SHAPE_N, 1);
                full_barrier.arrive_and_expect_tx(SMEM_TMA_BYTES);
            }
        }
    }
    // CONSUMER WARP GROUPS (math warps, 232 regs)
    else {
        cutlass::arch::warpgroup_reg_alloc<232>();

        const uint32_t math_warp_idx = warp_idx;
        const uint32_t group_id = lane_idx / 4;
        const uint32_t thread_id = lane_idx % 4;
        const uint32_t m_tile_base = math_warp_idx * kMTilesPerWarp;

        float accum[kAccumPerWarp];
        #pragma unroll
        for (uint32_t i = 0; i < kAccumPerWarp; ++i) accum[i] = 0.f;

        for (uint32_t s = 0; s < num_total_stages; ++s) {
            const auto stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            // Pre-compute SwizzleContexts
            sm120::SwizzleContext<kSwizzleABMode> a_ctx[kMTilesPerWarp];
            #pragma unroll
            for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                int a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + (m_tile_base + mt) * 16;
                a_ctx[mt].init(a_row, kSMEMKBytes);
            }

            sm120::SwizzleContext<kSwizzleABMode> b_ctx[kNTiles];
            #pragma unroll
            for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                int b_row = (lane_idx & 7) + nt * 8;
                b_ctx[nt].init(b_row, kSMEMKBytes);
            }

            uint32_t a_frag[2][kMTilesPerWarp][4];
            uint32_t b_tile_frag[2][kNTiles][2];

            auto load_kstep = [&](int buf, uint32_t ks) {
                #pragma unroll
                for (uint32_t nt = 0; nt < kNTiles; ++nt)
                    sm120::load_b_fragment_x2(b_tile_frag[buf][nt], smem_b[stage_idx], b_ctx[nt], lane_idx, ks, kLdmK);
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                    sm120::load_a_fragment(a_frag[buf][mt], smem_a[stage_idx], a_ctx[mt], lane_idx, ks, kLdmK);
            };

            auto compute_kstep = [&](int buf) {
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                        float (&accum_d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTiles + nt) * MMA_ACCUM]);
                        sm120_mma::bf16_mma(accum_d, a_frag[buf][mt], b_tile_frag[buf][nt]);
                    }
                }
            };

            load_kstep(0, 0);
            #pragma unroll
            for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                int cur = ks & 1;
                int nxt = (ks + 1) & 1;
                if (ks < kKSteps - 1)
                    load_kstep(nxt, ks + 1);
                compute_kstep(cur);
            }

            if (lane_idx == 0)
                empty_barriers[stage_idx]->arrive();
        }

        // Epilogue: atomicAdd to FP32 output
        #pragma unroll
        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
            #pragma unroll
            for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                const uint32_t ai = (mt * kNTiles + nt) * MMA_ACCUM;
                const uint32_t row0 = m_block_idx * BLOCK_M + (m_tile_base + mt) * MMA_M + group_id;
                const uint32_t row1 = row0 + 8;
                const uint32_t col = n_block_idx * BLOCK_N + nt * MMA_N + thread_id * 2;

                if (col + 1 < SHAPE_N) {
                    if (row0 < SHAPE_M) {
                        atomicAdd(reinterpret_cast<float2*>(d + row0 * SHAPE_N + col),
                                  make_float2(accum[ai + 0], accum[ai + 1]));
                    }
                    if (row1 < SHAPE_M) {
                        atomicAdd(reinterpret_cast<float2*>(d + row1 * SHAPE_N + col),
                                  make_float2(accum[ai + 2], accum[ai + 3]));
                    }
                }
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

}  // namespace deep_gemm

#pragma clang diagnostic pop
