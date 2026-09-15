#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm120_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kNumStages,
          uint32_t kNumMathThreads, uint32_t kNumTMAThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
sm120_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                float* gmem_d, float* sqr_sum) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    namespace sm120_mma = mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = sm120_mma::TF32_MMA_M;  // 16
    static constexpr uint32_t MMA_N = sm120_mma::TF32_MMA_N;  // 8
    static constexpr uint32_t MMA_K = sm120_mma::TF32_MMA_K;  // 8

    DG_STATIC_ASSERT(BLOCK_M % MMA_M == 0 and BLOCK_N % MMA_N == 0 and BLOCK_K % MMA_K == 0, "Invalid block dims");
    DG_STATIC_ASSERT(kNumMathThreads % 32 == 0, "Invalid math threads");
    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K");

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kMTilesPerWarp = BLOCK_M / kNumMathWarps / MMA_M;
    static constexpr uint32_t kNTiles = BLOCK_N / MMA_N;
    static constexpr uint32_t kKSteps = BLOCK_K / MMA_K;  // 8
    static constexpr uint32_t kAccumPerWarp = kMTilesPerWarp * kNTiles * sm120_mma::TF32_MMA_ACCUM;

    DG_STATIC_ASSERT(BLOCK_M == kNumMathWarps * kMTilesPerWarp * MMA_M, "M tiles must divide evenly");
    DG_STATIC_ASSERT(kMTilesPerWarp == 1, "sqr_sum assumes one M-tile per warp for per-row accumulation");

    static constexpr uint32_t kTMARegisters = 40;
    static constexpr uint32_t kMMARegisters = 232;

    // A (BF16): row stride = BLOCK_K * 2 bytes
    static constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * 2u, 128u);  // B128
    static constexpr uint32_t kSMEMKBytesA = BLOCK_K * 2;
    static constexpr uint32_t SMEM_A_PER_STAGE = BLOCK_M * kSMEMKBytesA;

    // B (FP32): TMA loads as K-major [BLOCK_N rows, BLOCK_K cols].
    // Row stride = BLOCK_K * 4 = 256 bytes. B128 swizzle → 2 atoms of 32 FP32 each.
    static constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * 4u, 128u);  // B128
    static constexpr uint32_t kBAtomK = kSwizzleBMode / 4u;                   // 32 FP32 elements per atom
    static constexpr uint32_t kBAtomBytes = kBAtomK * 4u;                     // 128 bytes per atom row
    static constexpr uint32_t SMEM_B_PER_STAGE = BLOCK_N * BLOCK_K * 4;

    static constexpr uint32_t SMEM_TMA_BYTES = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE;

    // Split-K
    static constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    static constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_a = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + s * SMEM_A_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + kNumStages * SMEM_A_PER_STAGE + s * SMEM_B_PER_STAGE);
    });
    constexpr uint32_t BAR_BASE = kNumStages * (SMEM_A_PER_STAGE + SMEM_B_PER_STAGE);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + s * sizeof(Barrier));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + (kNumStages + s) * sizeof(Barrier));
    });

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }
    __syncwarp();

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

    cudaGridDependencySynchronize();

    // PRODUCER (TMA warps)
    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kTMARegisters>();

        if (warp_idx == kNumMathWarps and lane_idx == 0) {
            for (uint32_t s = 0; s < num_total_stages; ++s) {
                const auto stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

                uint32_t m_idx = m_block_idx * BLOCK_M;
                uint32_t k_idx = k_offset + s * BLOCK_K;

                // A TMA: inner=K(BF16), outer=M. char dtype (1 byte = 1 bf16 is wrong, but
                // kSMEMKBytesA=128 = swizzle=128 → single atom, so loop=1 and it works)
                tma::copy<kSMEMKBytesA, BLOCK_M, kSwizzleAMode>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx);
                // B TMA: inner=K(FP32), outer=N. float dtype for correct multi-atom iteration.
                tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, float>(
                    &tensor_map_b, full_barriers[stage_idx],
                    reinterpret_cast<float*>(smem_b[stage_idx]), k_idx, 0);

                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_TMA_BYTES);
            }

            for (uint32_t s = num_total_stages; s < num_total_stages + kNumStages; ++s) {
                const auto stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);
            }
        }
    }
    // CONSUMER (math warps)
    else {
        cutlass::arch::warpgroup_reg_alloc<kMMARegisters>();

        const uint32_t math_warp_idx = warp_idx;
        const uint32_t group_id = lane_idx / 4;
        const uint32_t thread_id = lane_idx % 4;
        const uint32_t m_tile_base = math_warp_idx * kMTilesPerWarp;

        float accum[kAccumPerWarp];
        #pragma unroll
        for (uint32_t i = 0; i < kAccumPerWarp; ++i) accum[i] = 0.f;

        float sqr_sum_acc_0 = 0.f;
        float sqr_sum_acc_1 = 0.f;

        #pragma unroll kNumStages < 8 ? kNumStages : kNumStages / 2
        for (uint32_t s = 0; s < num_total_stages; ++s) {
            const auto stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            // A SwizzleContexts: row=M, stride=kSMEMKBytesA=128
            sm120::SwizzleContext<kSwizzleAMode> a_ctx_0[kMTilesPerWarp];
            sm120::SwizzleContext<kSwizzleAMode> a_ctx_8[kMTilesPerWarp];
            #pragma unroll
            for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                a_ctx_0[mt].init((m_tile_base + mt) * MMA_M + group_id, kSMEMKBytesA);
                a_ctx_8[mt].init((m_tile_base + mt) * MMA_M + group_id + 8, kSMEMKBytesA);
            }

            #pragma unroll
            for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                // ---- Load A: BF16 → FP32 → TF32, fuse sqr_sum ----
                // TF32 A: a0=A[g,t], a1=A[g+8,t], a2=A[g,t+4], a3=A[g+8,t+4]
                uint32_t a_frag[kMTilesPerWarp][4];
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    uint32_t col0_bytes = (ks * MMA_K + thread_id) * 2;
                    uint32_t col4_bytes = (ks * MMA_K + thread_id + 4) * 2;

                    float fa0 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(a_ctx_0[mt].addr(smem_a[stage_idx], col0_bytes)));
                    float fa1 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(a_ctx_8[mt].addr(smem_a[stage_idx], col0_bytes)));
                    float fa2 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(a_ctx_0[mt].addr(smem_a[stage_idx], col4_bytes)));
                    float fa3 = __bfloat162float(*reinterpret_cast<nv_bfloat16*>(a_ctx_8[mt].addr(smem_a[stage_idx], col4_bytes)));

                    sqr_sum_acc_0 += fa0 * fa0 + fa2 * fa2;
                    sqr_sum_acc_1 += fa1 * fa1 + fa3 * fa3;

                    a_frag[mt][0] = __float_as_uint(fa0);
                    a_frag[mt][1] = __float_as_uint(fa1);
                    a_frag[mt][2] = __float_as_uint(fa2);
                    a_frag[mt][3] = __float_as_uint(fa3);
                }

                // ---- Load B + MMA ----
                // B SMEM is K-major [BLOCK_N, BLOCK_K] with B128 swizzle → 2 atoms of kBAtomK FP32 each.
                // TF32 B: b0=B[t, g], b1=B[t+4, g] where t=thread_id, g=group_id.
                // B_smem[n, k] is at: atom(k/kBAtomK) * BLOCK_N * kBAtomBytes + swizzle(n * kBAtomBytes + k_local * 4)
                uint32_t k_b0 = ks * MMA_K + thread_id;
                uint32_t k_b1 = k_b0 + 4;
                uint32_t atom_b0 = k_b0 / kBAtomK;
                uint32_t atom_b1 = k_b1 / kBAtomK;
                uint32_t k_local_b0 = k_b0 % kBAtomK;
                uint32_t k_local_b1 = k_b1 % kBAtomK;

                #pragma unroll
                for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                    uint32_t n = nt * MMA_N + group_id;
                    sm120::SwizzleContext<kSwizzleBMode> b_ctx;
                    b_ctx.init(n, kBAtomBytes);

                    char* base_b0 = smem_b[stage_idx] + atom_b0 * BLOCK_N * kBAtomBytes;
                    char* base_b1 = smem_b[stage_idx] + atom_b1 * BLOCK_N * kBAtomBytes;
                    float fb0 = *reinterpret_cast<float*>(b_ctx.addr(base_b0, k_local_b0 * 4));
                    float fb1 = *reinterpret_cast<float*>(b_ctx.addr(base_b1, k_local_b1 * 4));
                    uint32_t b[2] = {__float_as_uint(fb0), __float_as_uint(fb1)};

                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTiles + nt) * 4]);
                        sm120_mma::tf32_mma(d, a_frag[mt], b);
                    }
                }
            }

            if (lane_idx == 0)
                empty_barriers[stage_idx]->arrive();
        }

        // sqr_sum reduction and write
        const float reduced_sum_0 = math::warp_reduce_sum<4>(sqr_sum_acc_0);
        const float reduced_sum_1 = math::warp_reduce_sum<4>(sqr_sum_acc_1);

        #pragma unroll
        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
            const uint32_t m_idx_0 = m_block_idx * BLOCK_M + (m_tile_base + mt) * MMA_M + group_id;
            const uint32_t m_idx_1 = m_idx_0 + 8;
            if (thread_id == 0) {
                if (m_idx_0 < shape_m)
                    sqr_sum[m_offset + m_idx_0] = reduced_sum_0;
                if (m_idx_1 < shape_m)
                    sqr_sum[m_offset + m_idx_1] = reduced_sum_1;
            }
        }

        // D epilogue: direct store FP32
        // D fragment: d0=D[g, t*2], d1=D[g, t*2+1], d2=D[g+8, t*2], d3=D[g+8, t*2+1]
        const int64_t d_split_offset = static_cast<int64_t>(k_split_idx) * shape_m * SHAPE_N;

        #pragma unroll
        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
            #pragma unroll
            for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                const uint32_t ai = (mt * kNTiles + nt) * 4;
                const uint32_t row0 = m_block_idx * BLOCK_M + (m_tile_base + mt) * MMA_M + group_id;
                const uint32_t row1 = row0 + 8;
                const uint32_t col = nt * MMA_N + thread_id * 2;

                if (row0 < shape_m and col + 1 < SHAPE_N) {
                    auto idx = d_split_offset + static_cast<int64_t>(row0) * SHAPE_N + col;
                    *reinterpret_cast<float2*>(&gmem_d[idx]) = make_float2(accum[ai + 0], accum[ai + 1]);
                }
                if (row1 < shape_m and col + 1 < SHAPE_N) {
                    auto idx = d_split_offset + static_cast<int64_t>(row1) * SHAPE_N + col;
                    *reinterpret_cast<float2*>(&gmem_d[idx]) = make_float2(accum[ai + 2], accum[ai + 3]);
                }
            }
        }
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
