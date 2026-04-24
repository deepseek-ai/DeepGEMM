#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumEpiStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads + 128, 1)
void sm90_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                         const uint32_t max_seqlen_k, const uint32_t stride_logits,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         logits_dtype_t* logits,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    // Types — N=64 split for wait<1> overlap
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using WGMMA_HALF = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads / 2>::type;
    static constexpr uint32_t kBlockQPerBank = BLOCK_Q / 2;
    static constexpr uint32_t kNumEpiThreads = 128;
    static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(kNumEpiStages >= 2, "Need double-buffer for epi pipeline");
    DG_STATIC_ASSERT(kNumEpiStages >= kNumKVStages, "Epi stages must cover KV pipeline depth");
    DG_STATIC_ASSERT(kHeadDim % WGMMA_HALF::K == 0, "Invalid head dim");
    DG_STATIC_ASSERT(WGMMA_HALF::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
    DG_STATIC_ASSERT(WGMMA_HALF::kNumAccum / kNumAccumPerReduce == kBlockQPerBank, "Invalid accumulation");
    DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

    // Thread layout: TC[0..511] + Epi[512..639] + TMA[640..767]
    static constexpr uint32_t kEpiStart = kNumMathThreads;
    static constexpr uint32_t kTMAStart = kNumMathThreads + kNumEpiThreads;

    // Prefetch TMA descriptors
    if (threadIdx.x / 32 == kTMAStart / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t kEpiStride = BLOCK_Q * 4 + 1;
    static constexpr uint32_t SMEM_EPI_SIZE_PER_STAGE = BLOCK_KV * kEpiStride * sizeof(float);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto smem_epi = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages +
            SMEM_EPI_SIZE_PER_STAGE * i);
    });

    // Barriers: Q(full/empty) + KV(full/empty) + Epi(full/empty)
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_epi[kNumEpiStages]);
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    auto full_epi_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + i; });
    auto empty_epi_barriers= utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumEpiStages + i; });

    // Initialize barriers
    const bool is_tma_load_warp = kTMAStart <= threadIdx.x and threadIdx.x < kTMAStart + 32;
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
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpiStages; ++ i) {
            full_epi_barriers[i]->init(kNumMathThreads);
            empty_epi_barriers[i]->init(kNumEpiThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Block scheduler (shared by TC and Epi, computed independently)
    const auto sm_idx = blockIdx.x;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // ═══════════════════════════════════════════════════════════════════
    // TMA WG: threadIdx.x in [kTMAStart..kTMAStart+127]
    // ═══════════════════════════════════════════════════════════════════
    if (threadIdx.x >= kTMAStart) {
        cutlass::arch::warpgroup_reg_dealloc<32>();
        if (not is_tma_load_warp)
            return;

        uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
        uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
        uint32_t num_total_kv_blocks = 0;

        const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
            return {block_q_idx + kNumSMs, q_iter_idx + 1};
        };
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
            return {(q_iter_idx + q_iter_offset) % kNumQStages,
                    ((q_iter_idx + q_iter_offset) / kNumQStages) & 1,
                    start, math::ceil_div(end - start, BLOCK_KV)};
        };
        const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
            return {
                (num_total_kv_blocks + kv_block_idx) % kNumKVStages,
                ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1
            };
        };

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& blk_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, blk_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, blk_idx * BLOCK_Q);
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

                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }

    // ═══════════════════════════════════════════════════════════════════
    // Epi WG: threadIdx.x in [kEpiStart..kEpiStart+127]
    // ═══════════════════════════════════════════════════════════════════
    } else if (threadIdx.x >= kEpiStart) {
        cutlass::arch::warpgroup_reg_dealloc<32>();
        const uint32_t epi_tid = threadIdx.x - kEpiStart;
        const uint32_t kv_col_lo = epi_tid;
        const uint32_t kv_col_hi = epi_tid + 128;

        uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
        uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
        uint32_t num_total_kv_blocks = 0;

        const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
            return {block_q_idx + kNumSMs, q_iter_idx + 1};
        };
        const auto load_schedule = [&]() -> cute::tuple<uint32_t, uint32_t> {
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
            return {start, math::ceil_div(end - start, BLOCK_KV)};
        };
        const auto get_epi_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
            return {
                (num_total_kv_blocks + kv_block_idx) % kNumEpiStages,
                ((num_total_kv_blocks + kv_block_idx) / kNumEpiStages) & 1
            };
        };

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), kv_start, num_kv_blocks);

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_epi_pipeline(kv_block_idx), epi_stage_idx, epi_phase);

                full_epi_barriers[epi_stage_idx]->wait(epi_phase);

                const float* epi_buf = smem_epi[epi_stage_idx];

                #pragma unroll
                for (uint32_t col_iter = 0; col_iter < 2; ++ col_iter) {
                    const uint32_t kv_col = (col_iter == 0) ? kv_col_lo : kv_col_hi;
                    const uint32_t actual_kv = kv_start + kv_block_idx * BLOCK_KV + kv_col;

                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const float* slot = epi_buf + kv_col * kEpiStride + i * 4;
                        float p0 = ptx::ld_shared(slot + 0);
                        float p1 = ptx::ld_shared(slot + 1);
                        float p2 = ptx::ld_shared(slot + 2);
                        float p3 = ptx::ld_shared(slot + 3);
                        float result = (p0 + p1) + (p2 + p3);

                        const auto q_offset = (block_q_idx * BLOCK_Q + i) * static_cast<uint64_t>(stride_logits);
                        if constexpr (kIsCompressedLogits) {
                            if (seq_k_start[i] <= actual_kv and actual_kv < seq_k_end[i])
                                logits[q_offset + actual_kv - seq_k_start[i]] = static_cast<logits_dtype_t>(result);
                        } else {
                            logits[q_offset + actual_kv] = static_cast<logits_dtype_t>(result);
                        }
                    }
                }

                empty_epi_barriers[epi_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;

            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }

    // ═══════════════════════════════════════════════════════════════════
    // TC WGs (4 warpgroups): threadIdx.x in [0..511]
    // ═══════════════════════════════════════════════════════════════════
    } else {
        cutlass::arch::warpgroup_reg_alloc<104>();
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = ptx::get_lane_idx();
        float accum_A[WGMMA_HALF::kNumAccum], accum_B[WGMMA_HALF::kNumAccum];
        float weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
        uint32_t num_total_kv_blocks = 0;

        const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
            return {block_q_idx + kNumSMs, q_iter_idx + 1};
        };
        const auto load_schedule = [&]() -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
            uint32_t start = cute::numeric_limits<uint32_t>::max();
            uint32_t end = cute::numeric_limits<uint32_t>::min();
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
                const uint32_t sk_start = cu_seq_len_k_start[q_idx];
                const uint32_t sk_end = cu_seq_len_k_end[q_idx];
                start = min(start, min(sk_start, seq_len_kv));
                end = max(end, min(sk_end, seq_len_kv));
            }
            start = start / 4 * 4;
            return {q_iter_idx % kNumQStages,
                    (q_iter_idx / kNumQStages) & 1,
                    start, math::ceil_div(end - start, BLOCK_KV)};
        };
        const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
            return {
                (num_total_kv_blocks + kv_block_idx) % kNumKVStages,
                ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1
            };
        };
        const auto get_epi_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
            return {
                (num_total_kv_blocks + kv_block_idx) % kNumEpiStages,
                ((num_total_kv_blocks + kv_block_idx) / kNumEpiStages) & 1
            };
        };


        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            full_q_barriers[q_stage_idx]->wait(q_phase);

            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Issue WGMMA — bank A (q-rows 0..kBlockQPerBank-1)
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA_HALF::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum_A[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA_HALF::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA_HALF::M) * kHeadDim + k * WGMMA_HALF::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[q_stage_idx] + k * WGMMA_HALF::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA_HALF::wgmma(desc_a, desc_b, accum_A, k);
                }
                ptx::warpgroup_commit_batch();

                // Issue WGMMA — bank B (q-rows kBlockQPerBank..BLOCK_Q-1)
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA_HALF::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum_B[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA_HALF::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA_HALF::M) * kHeadDim + k * WGMMA_HALF::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[q_stage_idx] + WGMMA_HALF::N * kHeadDim + k * WGMMA_HALF::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA_HALF::wgmma(desc_a, desc_b, accum_B, k);
                }
                ptx::warpgroup_commit_batch();

                // Wait epi-empty (producer acquire — skip on first kv-block)
                CUTE_TIE_DECL(get_epi_pipeline(kv_block_idx), epi_stage_idx, epi_phase);
                if ((num_total_kv_blocks | kv_block_idx) != 0)
                    empty_epi_barriers[epi_stage_idx]->wait(epi_phase ^ 1);

                // Load KV scales between epi-wait and DEPBAR — gives LDS time to complete
                float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                // Drain bank A — bank B continues in background
                ptx::warpgroup_wait<1>();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA_HALF::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum_A[i]);

                // Partial reduce bank A → STS to epi_buf
                float* epi_buf = smem_epi[epi_stage_idx];

                #define DO_PARTIAL_REDUCE(ACCUM, Q_BASE, Q_COUNT)                                   \
                    _Pragma("unroll")                                                                \
                    for (uint32_t i = 0; i < Q_COUNT; ++ i) {                                      \
                        auto shifted_accum = ACCUM + i * kNumAccumPerReduce;                        \
                        const auto transform = [&](const uint32_t& j) {                            \
                            return fmaxf(shifted_accum[j], 0) *                                    \
                                   weights[(Q_BASE) + i][(j / 4) * 2 + (j & 1)];                  \
                        };                                                                          \
                        float sum[4] = {transform(0), transform(1),                                \
                                        transform(2), transform(3)};                               \
                        _Pragma("unroll")                                                           \
                        for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {                            \
                            _Pragma("unroll")                                                       \
                            for (uint32_t kk = 0; kk < 4; kk ++)                                  \
                                sum[kk] += transform(j * 4 + kk);                                 \
                        }                                                                           \
                        float v_0 = (sum[0] + sum[1]) * scale_kv_0;                                \
                        float v_1 = (sum[2] + sum[3]) * scale_kv_1;                                \
                        const uint32_t kv_col_0 = warp_offset + v_0_offset;                        \
                        const uint32_t kv_col_1 = warp_offset + v_1_offset;                        \
                        const uint32_t shfl_lane = lane_idx % 4;                                   \
                        ptx::st_shared(epi_buf + kv_col_0 * kEpiStride                             \
                                       + ((Q_BASE) + i) * 4 + shfl_lane, v_0);                    \
                        ptx::st_shared(epi_buf + kv_col_1 * kEpiStride                             \
                                       + ((Q_BASE) + i) * 4 + shfl_lane, v_1);                    \
                    }

                DO_PARTIAL_REDUCE(accum_A, 0, kBlockQPerBank)

                // Drain bank B — must complete before releasing KV SMEM
                ptx::warpgroup_wait<0>();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA_HALF::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum_B[i]);

                // Release KV empty — both banks fully drained
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Partial reduce bank B → STS to epi_buf
                DO_PARTIAL_REDUCE(accum_B, kBlockQPerBank, kBlockQPerBank)

                #undef DO_PARTIAL_REDUCE

                // Fence STS visibility + signal Epi
                cutlass::arch::fence_view_async_shared();
                full_epi_barriers[epi_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
}

} // namespace deep_gemm
