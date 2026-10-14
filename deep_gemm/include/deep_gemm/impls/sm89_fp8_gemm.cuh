#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm89.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

// Ada Lovelace (SM89) FP8 GEMM.
//
// Ada is the first architecture with FP8 Tensor Cores, but it has no TMA, no WGMMA and no
// `mbarrier.try_wait`. This kernel is therefore built from the Ampere-era primitives that Ada still
// exposes: `cp.async` staging into shared memory, `bar.sync` for the pipeline and `ldmatrix` +
// `mma.sync.aligned.m16n8k32` (`QMMA`) for the math.
//
// Shared memory layout (per pipeline stage):
//   A: `BLOCK_M` rows x `BLOCK_K` (128) bytes, 128-byte XOR swizzle (`chunk ^ (row & 7)`)
//   B: `BLOCK_N` rows x `BLOCK_K` (128) bytes, same swizzle
//   SFA: `BLOCK_M` FP32 per-token/per-block scales (MN-major, TMA-aligned row stride)
//   SFB: `BLOCK_N` FP32 scales (1D1D), unused for 1D2D where a single scale covers a 128-wide N block
//
// The 128-byte swizzle is exactly what `ldmatrix.x4` wants: for one 8-row matrix the eight
// 16-byte chunks land in eight distinct 16-byte bank groups, so every load is conflict free.
namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages, uint32_t kNumThreads, uint32_t kNumSMs,
          bool kWithAccumulation, bool kIsBatchedSFB,
          typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm89_fp8_gemm_impl(const __nv_fp8_e4m3* __restrict__ gmem_a,
                   const __nv_fp8_e4m3* __restrict__ gmem_b,
                   const cd_dtype_t* __restrict__ gmem_c,
                   cd_dtype_t* __restrict__ gmem_d,
                   const float* __restrict__ gmem_sfa,
                   const float* __restrict__ gmem_sfb,
                   uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                   // K-direction strides (in elements) of the MN-major, TMA-aligned SFA/SFB
                   uint32_t sfa_k_stride, uint32_t sfb_k_stride,
                   // Only used by the 1D2D recipe, where SFB is `[ceil(N / 128), ceil(K / 128)]`
                   uint32_t sfb_n_stride, uint32_t sfb_n_block_stride) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 890) and (__CUDA_ARCH__ < 900)) or defined(__CLION_IDE__)
    using MMA = mma::sm89::FP8MMAM16N8K32;
    // N granularity of the 1D2D recipe: all columns inside one 128-wide group share a B scale
    static constexpr uint32_t kNScaleGroup = 128;

    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only per-128-channel FP8 scaling is supported");
    DG_STATIC_ASSERT(kNumThreads % 32 == 0 and kNumThreads >= 64, "Invalid number of threads");
    DG_STATIC_ASSERT(kNumStages >= 2, "The software pipeline needs at least two stages");

    // Warp tiling (must agree with the host-side heuristics)
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    constexpr uint32_t kNumWarpsN = get_mma_sync_warps_n(kNumWarps, BLOCK_M, BLOCK_N);
    DG_STATIC_ASSERT(kNumWarpsN > 0, "Invalid warp tiling");
    constexpr uint32_t kNumWarpsM = kNumWarps / kNumWarpsN;
    constexpr uint32_t kWarpM = BLOCK_M / kNumWarpsM;
    constexpr uint32_t kWarpN = BLOCK_N / kNumWarpsN;
    DG_STATIC_ASSERT(kWarpM % 16 == 0 and kWarpN % 8 == 0, "Invalid warp tiling");

    constexpr uint32_t kNumMTiles = kWarpM / MMA::M;
    constexpr uint32_t kNumNTiles = kWarpN / MMA::N;
    constexpr uint32_t kNumKSlices = BLOCK_K / MMA::K;
    DG_STATIC_ASSERT(kNumKSlices % 2 == 0, "`ldmatrix.x4` for B covers two K slices");
    constexpr uint32_t kNumAccums = kNumMTiles * kNumNTiles * MMA::kNumAccum;

    // Each thread stages a whole number of 16-byte chunks of A/B, so that all copy addresses
    // stay compile-time offsets from the stage base.
    DG_STATIC_ASSERT(BLOCK_M * 8 % kNumThreads == 0, "Invalid A staging");
    DG_STATIC_ASSERT(BLOCK_N * 8 % kNumThreads == 0, "Invalid B staging");
    constexpr uint32_t kNumChunksA = BLOCK_M * 8 / kNumThreads;
    constexpr uint32_t kNumChunksB = BLOCK_N * 8 / kNumThreads;
    constexpr uint32_t kNumScalarCopiesSFA = math::constexpr_ceil_div(BLOCK_M, kNumThreads);
    constexpr uint32_t kNumScalarCopiesSFB = math::constexpr_ceil_div(BLOCK_N, kNumThreads);

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    DG_TRAP_ONLY_DEVICE_ASSERT(shape_k % 16 == 0 and shape_m > 0 and shape_n > 0 and shape_k > 0);

    // Shared memory sizes (per pipeline stage)
    static constexpr uint32_t kSmemA = BLOCK_M * BLOCK_K;
    static constexpr uint32_t kSmemB = BLOCK_N * BLOCK_K;
    static constexpr uint32_t kSmemSFA = BLOCK_M * static_cast<uint32_t>(sizeof(float));
    static constexpr uint32_t kSmemSFB = kIsBatchedSFB ? 0u : BLOCK_N * static_cast<uint32_t>(sizeof(float));
    static constexpr uint32_t kSmemPerStage = kSmemA + kSmemB + kSmemSFA + kSmemSFB;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Thread indices
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_idx = tid / 32;
    const uint32_t lane_idx = tid % 32;
    const uint32_t warp_m_idx = warp_idx / kNumWarpsN;
    const uint32_t warp_n_idx = warp_idx % kNumWarpsN;
    const uint32_t row_in_mma = lane_idx / 4, col_in_mma = (lane_idx % 4) * 2;

    // 128-byte XOR swizzle: `chunk` is a 16-byte chunk index inside a `BLOCK_K`-byte row
    const auto get_ab_offset = [](const uint32_t& row, const uint32_t& chunk) {
        return row * BLOCK_K + ((chunk ^ (row & 7u)) << 4);
    };

    const uint32_t num_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t num_chunks_k = BLOCK_K / 16;

    // Stage one A/B/SF tile with `cp.async`, zero-filling out-of-bounds elements
    const auto load_stage = [&](const uint32_t& stage, const uint32_t& m_base, const uint32_t& n_base,
                                const uint32_t& k_block) {
        auto smem_a = smem_buffer + stage * kSmemPerStage;
        auto smem_b = smem_a + kSmemA;
        const uint32_t k_byte_offset = k_block * BLOCK_K;
        const uint32_t num_valid_k_bytes = cute::min(BLOCK_K, shape_k > k_byte_offset ? shape_k - k_byte_offset : 0u);

        #pragma unroll
        for (uint32_t j = 0; j < kNumChunksA; ++ j) {
            const uint32_t idx = tid + j * kNumThreads;
            const uint32_t row = idx / num_chunks_k, chunk = idx % num_chunks_k;
            const uint32_t g_row = m_base + row;
            const uint32_t chunk_byte = chunk * 16;
            const uint32_t num_src_bytes = (g_row < shape_m and chunk_byte < num_valid_k_bytes) ?
                cute::min(16u, num_valid_k_bytes - chunk_byte) : 0u;
            const uint32_t safe_row = g_row < shape_m ? g_row : 0;
            ptx::cp_async_zfill16(
                gmem_a + static_cast<uint64_t>(safe_row) * shape_k + k_byte_offset + chunk_byte,
                smem_a + get_ab_offset(row, chunk), num_src_bytes);
        }

        #pragma unroll
        for (uint32_t j = 0; j < kNumChunksB; ++ j) {
            const uint32_t idx = tid + j * kNumThreads;
            const uint32_t row = idx / num_chunks_k, chunk = idx % num_chunks_k;
            const uint32_t g_row = n_base + row;
            const uint32_t chunk_byte = chunk * 16;
            const uint32_t num_src_bytes = (g_row < shape_n and chunk_byte < num_valid_k_bytes) ?
                cute::min(16u, num_valid_k_bytes - chunk_byte) : 0u;
            const uint32_t safe_row = g_row < shape_n ? g_row : 0;
            ptx::cp_async_zfill16(
                gmem_b + static_cast<uint64_t>(safe_row) * shape_k + k_byte_offset + chunk_byte,
                smem_b + get_ab_offset(row, chunk), num_src_bytes);
        }

        auto smem_sfa = reinterpret_cast<float*>(smem_b + kSmemB);
        #pragma unroll
        for (uint32_t j = 0; j < kNumScalarCopiesSFA; ++ j) {
            const uint32_t row = tid + j * kNumThreads;
            if (row < BLOCK_M) {
                const uint32_t g_row = m_base + row;
                const uint32_t num_src_bytes = g_row < shape_m ? sizeof(float) : 0u;
                ptx::cp_async_ca_zfill(
                    reinterpret_cast<const uint32_t*>(gmem_sfa + g_row + static_cast<uint64_t>(k_block) * sfa_k_stride),
                    reinterpret_cast<uint32_t*>(smem_sfa + row), num_src_bytes);
            }
        }

        // For the 1D2D recipe every N column in a 128-wide group shares one B scale, so it is read
        // straight from global memory (a single uniform load) instead of being staged.
        if constexpr (not kIsBatchedSFB) {
            auto smem_sfb = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_sfa) + kSmemSFA);
            #pragma unroll
            for (uint32_t j = 0; j < kNumScalarCopiesSFB; ++ j) {
                const uint32_t row = tid + j * kNumThreads;
                if (row < BLOCK_N) {
                    const uint32_t g_row = n_base + row;
                    const uint32_t num_src_bytes = g_row < shape_n ? sizeof(float) : 0u;
                    ptx::cp_async_ca_zfill(
                        reinterpret_cast<const uint32_t*>(gmem_sfb + g_row + static_cast<uint64_t>(k_block) * sfb_k_stride),
                        reinterpret_cast<uint32_t*>(smem_sfb + row), num_src_bytes);
                }
            }
        }
    };

    // Persistent block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, 1, 1, false, kNumSMs>(
        shape_m, shape_n, shape_k);
    if (not scheduler.get_next_block(m_block_idx, n_block_idx))
        return;

    uint32_t next_m_block_idx, next_n_block_idx;
    auto has_next_block = scheduler.get_next_block(next_m_block_idx, next_n_block_idx);

    // Software pipeline prologue. Exactly `kNumStages - 1` groups are committed so that the
    // `cp_async_wait<kNumStages - 2>` inside the K loop always releases the group holding stage
    // `k_block_idx`; stages past `num_k_blocks` are committed empty to keep that count fixed.
    const auto prime_pipeline = [&](const uint32_t& m_base, const uint32_t& n_base) {
        #pragma unroll
        for (uint32_t p = 0; p < kNumStages - 1; ++ p) {
            if (p < num_k_blocks)
                load_stage(p, m_base, n_base, p);
            ptx::cp_async_commit();
        }
    };

    prime_pipeline(m_block_idx * BLOCK_M, n_block_idx * BLOCK_N);

    float accum[kNumAccums], final_accum[kNumAccums];

    while (true) {
        const auto m_base = m_block_idx * BLOCK_M;
        const auto n_base = n_block_idx * BLOCK_N;

        #pragma unroll
        for (uint32_t i = 0; i < kNumAccums; ++ i)
            final_accum[i] = 0;

        #pragma unroll 4
        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
            // Wait until this stage's `cp.async` group landed; this also orders the previous
            // iteration's readers of the stage we are about to refill.
            ptx::cp_async_wait<kNumStages - 2>();
            __syncthreads();

            // Compute on the ready stage
            const uint32_t stage = k_block_idx % kNumStages;
            auto smem_a = smem_buffer + stage * kSmemPerStage;
            auto smem_b = smem_a + kSmemA;
            const auto smem_sfa = reinterpret_cast<const float*>(smem_b + kSmemB);
            const auto smem_sfb = reinterpret_cast<const float*>(
                reinterpret_cast<const uint8_t*>(smem_sfa) + kSmemSFA);

            #pragma unroll
            for (uint32_t i = 0; i < kNumAccums; ++ i)
                accum[i] = 0;

            #pragma unroll
            for (uint32_t slice_pair = 0; slice_pair < kNumKSlices / 2; ++ slice_pair) {
                // A: one `ldmatrix.x4` per (M tile, K slice) -> registers a0/a1/a2/a3
                uint32_t a_frag[2][kNumMTiles][MMA::kNumRegsA];
                #pragma unroll
                for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                    const uint32_t row = warp_m_idx * kWarpM + i * MMA::M + (lane_idx % 8) + 8 * ((lane_idx / 8) % 2);
                    #pragma unroll
                    for (uint32_t s = 0; s < 2; ++ s) {
                        const uint32_t chunk = (slice_pair * 2 + s) * 2 + (lane_idx / 16);
                        ptx::SM90_U32x4_LDSM_N::copy(
                            a_frag[s][i][0], a_frag[s][i][1], a_frag[s][i][2], a_frag[s][i][3],
                            smem_a + get_ab_offset(row, chunk));
                    }
                }

                // B: one `ldmatrix.x4` per (N tile, K slice pair) -> r0/r1 (slice 0), r2/r3 (slice 1)
                uint32_t b_frag[2][kNumNTiles][MMA::kNumRegsB];
                #pragma unroll
                for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                    const uint32_t row = warp_n_idx * kWarpN + j * MMA::N + (lane_idx % 8);
                    const uint32_t chunk = slice_pair * 4 + (lane_idx / 8);
                    uint32_t b4[4];
                    ptx::SM90_U32x4_LDSM_N::copy(b4[0], b4[1], b4[2], b4[3], smem_b + get_ab_offset(row, chunk));
                    b_frag[0][j][0] = b4[0], b_frag[0][j][1] = b4[1];
                    b_frag[1][j][0] = b4[2], b_frag[1][j][1] = b4[3];
                }

                #pragma unroll
                for (uint32_t s = 0; s < 2; ++ s) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                            const auto base = (i * kNumNTiles + j) * MMA::kNumAccum;
                            MMA::mma(accum[base + 0], accum[base + 1], accum[base + 2], accum[base + 3],
                                     a_frag[s][i], b_frag[s][j]);
                        }
                    }
                }
            }

            // Promote the unscaled MMA accumulator with the per-128-channel scales
            float scale_a[kNumMTiles][2];
            #pragma unroll
            for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                scale_a[i][0] = ptx::ld_shared(smem_sfa + warp_m_idx * kWarpM + i * MMA::M + row_in_mma);
                scale_a[i][1] = ptx::ld_shared(smem_sfa + warp_m_idx * kWarpM + i * MMA::M + row_in_mma + 8);
            }
            if constexpr (kIsBatchedSFB) {
                // One shared FP32 scale per (128-wide N block, 128-wide K block)
                const float scale_b = gmem_sfb[static_cast<uint64_t>(n_base / kNScaleGroup) * sfb_n_stride +
                                              k_block_idx * sfb_n_block_stride];
                #pragma unroll
                for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                        const auto base = (i * kNumNTiles + j) * MMA::kNumAccum;
                        const auto scale_0 = scale_a[i][0] * scale_b, scale_1 = scale_a[i][1] * scale_b;
                        final_accum[base + 0] += scale_0 * accum[base + 0];
                        final_accum[base + 1] += scale_0 * accum[base + 1];
                        final_accum[base + 2] += scale_1 * accum[base + 2];
                        final_accum[base + 3] += scale_1 * accum[base + 3];
                    }
                }
            } else {
                float scale_b[kNumNTiles][2];
                #pragma unroll
                for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                    const auto values = ptx::ld_shared(reinterpret_cast<const float2*>(
                        smem_sfb + warp_n_idx * kWarpN + j * MMA::N + col_in_mma));
                    scale_b[j][0] = values.x, scale_b[j][1] = values.y;
                }
                #pragma unroll
                for (uint32_t i = 0; i < kNumMTiles; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                        const auto base = (i * kNumNTiles + j) * MMA::kNumAccum;
                        final_accum[base + 0] += scale_a[i][0] * scale_b[j][0] * accum[base + 0];
                        final_accum[base + 1] += scale_a[i][0] * scale_b[j][1] * accum[base + 1];
                        final_accum[base + 2] += scale_a[i][1] * scale_b[j][0] * accum[base + 2];
                        final_accum[base + 3] += scale_a[i][1] * scale_b[j][1] * accum[base + 3];
                    }
                }
            }

            // Issue the next stage's copies only after the math is issued: the `__syncthreads()`
            // above already guarantees every thread is done with the stage being refilled, and
            // keeping these addresses out of the math block saves ~20 registers.
            const uint32_t target = k_block_idx + kNumStages - 1;
            if (target < num_k_blocks)
                load_stage(target % kNumStages, m_base, n_base, target);
            ptx::cp_async_commit();
        }

        // Epilogue: `D = C + sum_k (A * SFA) * (B * SFB)`
        // NOTES: SM89 has no TMA store, so the accumulators go straight to global memory. Each lane
        // owns a pair of adjacent columns, which is only one vector store when the pair does not
        // cross the end of a row and the row pitch keeps it naturally aligned.
        const auto store_element = [&](const uint32_t& accum_idx, const uint64_t& offset) {
            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                float value = final_accum[accum_idx];
                if constexpr (kWithAccumulation)
                    value += gmem_c[offset];
                gmem_d[offset] = value;
            } else {
                cutlass::bfloat16_t value = cutlass::bfloat16_t(final_accum[accum_idx]);
                if constexpr (kWithAccumulation)
                    value = value + gmem_c[offset];
                gmem_d[offset] = value;
            }
        };

        #pragma unroll
        for (uint32_t i = 0; i < kNumMTiles; ++ i) {
            #pragma unroll
            for (uint32_t j = 0; j < kNumNTiles; ++ j) {
                #pragma unroll
                for (uint32_t h = 0; h < 2; ++ h) {
                    const uint32_t row = m_base + warp_m_idx * kWarpM + i * MMA::M + row_in_mma + h * 8;
                    const uint32_t col = n_base + warp_n_idx * kWarpN + j * MMA::N + col_in_mma;
                    if (row >= shape_m or col >= shape_n)
                        continue;

                    const auto idx = (i * kNumNTiles + j) * MMA::kNumAccum + h * 2;
                    const auto cd_offset = static_cast<uint64_t>(row) * shape_n + col;
                    const bool has_next_col = col + 1 < shape_n;
                    // A two-element store needs both columns in range and an even element offset
                    // (for an odd `shape_n`, odd rows have an odd offset)
                    const bool is_pair_store = has_next_col and (cd_offset & 1) == 0;

                    if (is_pair_store) {
                        if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                            float2 value = {final_accum[idx], final_accum[idx + 1]};
                            if constexpr (kWithAccumulation) {
                                const auto c_value = *reinterpret_cast<const float2*>(gmem_c + cd_offset);
                                value.x += c_value.x, value.y += c_value.y;
                            }
                            *reinterpret_cast<float2*>(gmem_d + cd_offset) = value;
                        } else {
                            DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                                             "Invalid C/D data dtype");
                            auto value = __floats2bfloat162_rn(final_accum[idx], final_accum[idx + 1]);
                            if constexpr (kWithAccumulation) {
                                const auto c_value = *reinterpret_cast<const nv_bfloat162*>(gmem_c + cd_offset);
                                value = __hadd2(value, c_value);
                            }
                            *reinterpret_cast<nv_bfloat162*>(gmem_d + cd_offset) =
                                *reinterpret_cast<const nv_bfloat162*>(&value);
                        }
                    } else {
                        store_element(idx, cd_offset);
                        if (has_next_col)
                            store_element(idx + 1, cd_offset + 1);
                    }
                }
            }
        }

        if (not has_next_block)
            break;
        m_block_idx = next_m_block_idx, n_block_idx = next_n_block_idx;
        has_next_block = scheduler.get_next_block(next_m_block_idx, next_n_block_idx);

        // Re-prime the pipeline for the next tile. The K loop always consumes prologue stages
        // `0 .. num_k_blocks - 1`, but a tile's tail can only pre-load the stages that are
        // already free of the current tile, which is not enough of them whenever
        // `num_k_blocks < kNumStages - 1`; leaving those stages stale makes the next tile read
        // whatever the previous one left in shared memory. Priming here is unconditional.
        // Every outstanding group must land first, because the stages are about to be reused and
        // `cp.async` completion is per-thread; the barrier then guarantees that no thread is
        // still reading them.
        ptx::cp_async_wait<0>();
        __syncthreads();
        prime_pipeline(m_block_idx * BLOCK_M, n_block_idx * BLOCK_N);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_89");
#endif
}

}  // namespace deep_gemm