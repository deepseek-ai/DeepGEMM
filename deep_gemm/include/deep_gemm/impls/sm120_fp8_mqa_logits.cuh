#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/sm120_utils.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm120_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t max_seqlen_k, const uint32_t stride_logits,
                          uint32_t* cu_seq_len_k_start,
                          uint32_t* cu_seq_len_k_end,
                          logits_dtype_t* logits,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    // Warp-specialized mma.sync m16n8k32 FP8 (no block_scale).
    // 8 math warps (256 threads) + 4 TMA warps (128 threads) = 384 total.
    // Each math warp: 1 M-tile (16 KV rows) × all N-tiles, in-warp 2-shfl reduction.
    // Global stores are fire-and-forget on SM120a — no epilogue warps needed.

    using namespace mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = FP8_MMA_M;
    static constexpr uint32_t MMA_N = FP8_MMA_N;
    static constexpr uint32_t MMA_K = FP8_MMA_K;
    static constexpr uint32_t kKSteps = kHeadDim / MMA_K;
    static constexpr uint32_t kNTilesPerQ = kNumHeads / MMA_N;
    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;

    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads == 256, "Expected 256 math + 128 TMA");
    DG_STATIC_ASSERT(BLOCK_KV == kNumMathWarps * MMA_M, "BLOCK_KV = warps × MMA_M");
    DG_STATIC_ASSERT(kHeadDim % MMA_K == 0 and kNumHeads % MMA_N == 0, "Alignment");

    static constexpr uint32_t kSwizzleMode = 128;
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t kSMEMKBytes = kHeadDim;

    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);

    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    // Prefetch TMA descriptors
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // SMEM layout
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // Barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });

    // Init barriers
    const bool is_tma_load_warp = kNumMathThreads <= threadIdx.x and threadIdx.x < kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfig
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Persistent scheduler
    const auto sm_idx = blockIdx.x;
    // Split-KV: gridDim.y blocks cooperatively cover one q-block's KV range, each
    // writing a disjoint kv-subrange (no reduction over KV → no combine needed).
    // Fills idle SMs when num_q_blocks < num_sms (small-S dense). gridDim.y == 1
    // reproduces the original single-block-per-q-block behavior exactly.
    const uint32_t kv_split_idx = blockIdx.y;
    const uint32_t kv_splits = gridDim.y;
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        start = start / 4 * 4;
        // Partition this q-block's KV blocks across the kv_splits cooperating blocks.
        const uint32_t total_kv_blocks = math::ceil_div(end - start, BLOCK_KV);
        const uint32_t chunk = math::ceil_div(total_kv_blocks, kv_splits);
        const uint32_t kv_block_begin = kv_split_idx * chunk;
        const uint32_t split_count = kv_block_begin < total_kv_blocks
            ? min(chunk, total_kv_blocks - kv_block_begin) : 0u;
        const uint32_t split_start = start + kv_block_begin * BLOCK_KV;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1,
                split_start, split_count};
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1
        };
    };

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warps
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (not is_tma_load_warp)
            return;

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(
                &tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx],
                0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(
                &tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx],
                0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(
                        &tensor_map_kv, full_kv_barriers[kv_stage_idx],
                        smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(
                        &tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                        smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else {
        // Math warps
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& lane_idx = ptx::get_lane_idx();
        const uint32_t g = lane_idx / 4;
        const uint32_t t = lane_idx % 4;
        const uint32_t a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                const float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_idx * MMA_M + g);
                const float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_idx * MMA_M + g + 8);

                sm120::SwizzleContext<kSwizzleMode> a_ctx;
                a_ctx.init(a_row + warp_idx * MMA_M, kSMEMKBytes);

                #pragma unroll
                for (uint32_t qi = 0; qi < BLOCK_Q; ++ qi) {
                    float partial_0 = 0.0f, partial_1 = 0.0f;

                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerQ; ++ nt) {
                        const uint32_t n_tile = qi * kNTilesPerQ + nt;
                        float d[4] = {0, 0, 0, 0};

                        sm120::SwizzleContext<kSwizzleMode> b_ctx;
                        b_ctx.init((lane_idx & 7) + n_tile * MMA_N, kSMEMKBytes);

                        #pragma unroll
                        for (uint32_t k = 0; k < kKSteps; ++ k) {
                            uint32_t a_frag[4];
                            sm120::load_a_fragment<kSwizzleMode>(
                                a_frag, reinterpret_cast<char*>(smem_kv[kv_stage_idx]),
                                a_ctx, lane_idx, k, MMA_K);

                            uint32_t b_frag[2];
                            sm120::load_b_fragment_x2<kSwizzleMode>(
                                b_frag, reinterpret_cast<char*>(smem_q[q_stage_idx]),
                                b_ctx, lane_idx, k, MMA_K);

                            fp8_mma(d, a_frag, b_frag);
                        }

                        const uint32_t h0 = nt * MMA_N + t * 2;
                        const float w0 = ptx::ld_shared(smem_weights[q_stage_idx] + qi * kNumHeads + h0);
                        const float w1 = ptx::ld_shared(smem_weights[q_stage_idx] + qi * kNumHeads + h0 + 1);
                        partial_0 += fmaxf(d[0], 0.0f) * w0 + fmaxf(d[1], 0.0f) * w1;
                        partial_1 += fmaxf(d[2], 0.0f) * w0 + fmaxf(d[3], 0.0f) * w1;
                    }

                    // 2-shfl reduction + fire-and-forget global store
                    float v_0 = partial_0 * scale_kv_0;
                    float v_1 = partial_1 * scale_kv_1;

                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    const auto kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_idx * MMA_M;
                    const auto q_offset = (block_q_idx * BLOCK_Q + qi) * static_cast<uint64_t>(stride_logits);
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[qi] <= kv_offset + g and kv_offset + g < seq_k_end[qi])
                            logits[q_offset + kv_offset + g - seq_k_start[qi]] = static_cast<logits_dtype_t>(v_0);
                        if (seq_k_start[qi] <= kv_offset + g + 8 and kv_offset + g + 8 < seq_k_end[qi])
                            logits[q_offset + kv_offset + g + 8 - seq_k_start[qi]] = static_cast<logits_dtype_t>(v_1);
                    } else {
                        logits[q_offset + kv_offset + g] = static_cast<logits_dtype_t>(v_0);
                        logits[q_offset + kv_offset + g + 8] = static_cast<logits_dtype_t>(v_1);
                    }
                }

                empty_kv_barriers[kv_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;

            empty_q_barriers[q_stage_idx]->arrive();
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
