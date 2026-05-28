#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_paged_mqa_logits(const uint32_t batch_size,
                               const uint32_t logits_stride, const uint32_t block_table_stride,
                               const uint32_t* context_lens, logits_dtype_t* logits,
                               const uint32_t* block_table, const uint32_t* indices,
                               const uint32_t* schedule_meta,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    DG_STATIC_ASSERT(not kIsVarlen, "Varlen is not supported for SM90 paged MQA logits");

    // Types
    static constexpr bool kPadOddN = (not kIsVarlen) and (kNextN % 2 == 1) and (kNextN >= 3);
    static constexpr uint32_t kNextNAtom = (kIsVarlen or kNextN >= 2) ? 2 : 1;
    static constexpr uint32_t kNumNextNAtoms = kPadOddN ? 1 : math::constexpr_ceil_div(kNextN, kNextNAtom);
    using WGMMA = typename mma::sm90::FP8MMASelector<kNextNAtom * kNumHeads>::type;
    using TailWGMMA = typename mma::sm90::FP8MMASelector<kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors
    static constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumMathWarpGroups, "Invalid `SPLIT_KV`");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextNAtom * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = math::constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 math::constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
                                                  math::constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q data and barriers on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });

    // Separate math warpgroups and tma load warps into KV groups
    // Each math warpgroup corresponds to a tma load warp
    const auto kv_group_idx = __shfl_sync(0xffffffff, threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);

    // Per group KV data and barriers on shared memory
    const auto smem_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + i; });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + kNumKVStages + i; });

    // Initialize barriers
    if (warp_idx >= kNumMathThreads / 32 and cute::elect_one_sync()) {
        if (kv_group_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumQStages; ++ i) {
                full_q_barriers[i]->init(1);
                empty_q_barriers[i]->init(kNumMathThreads);
            }
        }
        if (kv_group_idx < kNumMathWarpGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++ i) {
                full_kv_barriers[i]->init(1);
                empty_kv_barriers[i]->init(128);
            }
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 64;
    constexpr uint32_t kNumMathRegisters = 104;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Scheduler
    using Scheduler = sched::PagedMQALogitsScheduler<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumMathWarpGroups, kNumNextNAtoms>;
    auto scheduler = Scheduler(blockIdx.x, batch_size, context_lens, schedule_meta, indices);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    // Q and KV pipeline
    const auto get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (kv_group_idx >= kNumMathWarpGroups)
            return;

        const auto issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_atom_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                const auto q_token_idx = Scheduler::atom_to_token_idx(q_atom_idx);
                tma::copy<kHeadDim, kNextNAtom * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_token_idx * kNumHeads);
                tma::copy<kNextNAtom * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_token_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };
        const auto issue_tma_q_pair = [&](const uint32_t& stage_idx, const uint32_t& tail_stage_idx, const uint32_t& q_atom_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                const auto q_token_idx = Scheduler::atom_to_token_idx(q_atom_idx);
                tma::copy<kHeadDim, kNextNAtom * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_token_idx * kNumHeads);
                tma::copy<kNextNAtom * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_token_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
                tma::copy<kHeadDim, kNextNAtom * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[tail_stage_idx], smem_q[tail_stage_idx], 0, (q_token_idx + 2) * kNumHeads);
                tma::copy<kNextNAtom * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[tail_stage_idx], smem_weights[tail_stage_idx], 0, q_token_idx + 2);
                full_q_barriers[tail_stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize `q_atom_idx` outside the valid range to indicate it was none
        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx, num_kv;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q atom
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv))) {
            if constexpr (kPadOddN)
                issue_tma_q_pair(0, 1, next_q_atom_idx), q_iter_idx = 2;
            else
                issue_tma_q(0, next_q_atom_idx), q_iter_idx = 1;
        }

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when current atom changes
            const auto next_advance = scheduler.get_atom_advance(next_q_atom_idx, batch_size * kNumNextNAtoms);
            bool prefetch_q = (q_atom_idx != next_q_atom_idx and scheduler.exist_q_atom_idx(next_q_atom_idx + next_advance));
            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                if constexpr (kPadOddN) {
                    CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                    CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_tail_stage_idx, q_tail_phase);
                    empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                    empty_q_barriers[q_tail_stage_idx]->wait(q_tail_phase ^ 1);
                    issue_tma_q_pair(q_stage_idx, q_tail_stage_idx, q_atom_idx + next_advance);
                } else {
                    CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                    empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                    issue_tma_q(q_stage_idx, q_atom_idx + next_advance);
                }
            }

            // Read KV block index
            // TODO: deal with `-1`?
            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                const auto block_table_offset = Scheduler::atom_to_block_table_row(q_atom_idx) * static_cast<uint64_t>(block_table_stride);
                kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups < num_kv ?
                    block_table[block_table_offset + (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)] : 0);
            }
            const auto kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            // Issue TMA KV
            if (cute::elect_one_sync()) {
                tma::copy<kHeadDim, BLOCK_KV, 0, __nv_fp8_e4m3, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                                      smem_kv[kv_stage_idx], 0, 0, 1, kv_block_idx);
                tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                          smem_kv_scales[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv);
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        float accum[WGMMA::kNumAccum], weights[kNextNAtom][kNumHeads / 4], tail_weights[kNextNAtom][kNumHeads / 4];
        const auto sub_warp_offset = (warp_idx % 4) * 16;
        const auto v_0_offset = lane_idx / 4 + 0;
        const auto v_1_offset = lane_idx / 4 + 8;

        // Initialize `q_atom_idx` outside the valid range to indicate it was none
        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase, q_tail_stage_idx, q_tail_phase;

        while (scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv)) {
            // Current Q atom changes
            if (q_atom_idx != next_q_atom_idx) {
                // Release Last Q empty
                if (q_iter_idx > 0) {
                    if constexpr (kPadOddN) {
                        empty_q_barriers[(q_iter_idx - 2) % kNumQStages]->arrive();
                        empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();
                    } else {
                        empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();
                    }
                }

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);
                if constexpr (kPadOddN) {
                    CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_tail_stage_idx, q_tail_phase);
                    full_q_barriers[q_tail_stage_idx]->wait(q_tail_phase);
                }

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextNAtom; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
                if constexpr (kPadOddN) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNextNAtom; ++ i) {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                            tail_weights[i][j] = ptx::ld_shared(smem_weights[q_tail_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                    }
                }
            }

            // Get current Q atom and KV index
            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = Scheduler::atom_to_token_idx(q_atom_idx) * static_cast<uint64_t>(logits_stride) + ((kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset);

            // Compute `[kNextNAtom * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [kNextNAtom, BLOCK_KV]`
            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");
            DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
            DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == kNextNAtom, "Invalid accumulation");
            DG_STATIC_ASSERT(TailWGMMA::kNumAccum == kNumAccumPerReduce, "Invalid tail accumulation");
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

            const auto issue_wgmma = [&](const uint32_t& active_q_stage_idx) {
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[active_q_stage_idx] + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
            };
            const auto issue_tail_wgmma = [&](const uint32_t& active_q_stage_idx) {
                #pragma unroll
                for (uint32_t i = 0; i < TailWGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / TailWGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + k * TailWGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[active_q_stage_idx] + k * TailWGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    TailWGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < TailWGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
            };

            // Read per-KV scales
            float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_0_offset);
            float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_1_offset);

            const auto reduce_and_store = [&](auto num_iters_c, const auto& active_weights, const uint64_t& active_kv_offset) {
                constexpr uint32_t kNumIters = decltype(num_iters_c)::value;
                #pragma unroll
                for (uint32_t i = 0; i < kNumIters; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto transform = [&](const uint32_t& j) {
                        return fmaxf(shifted_accum[j], 0) * active_weights[i][(j / 4) * 2 + (j & 1)];
                    };

                    // Intra-thread reduction
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k ++)
                            sum[k] += transform(j * 4 + k);
                    }
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    logits[active_kv_offset + i * static_cast<uint64_t>(logits_stride) + v_0_offset] = static_cast<logits_dtype_t>(v_0);
                    logits[active_kv_offset + i * static_cast<uint64_t>(logits_stride) + v_1_offset] = static_cast<logits_dtype_t>(v_1);
                }
            };

            if constexpr (kPadOddN) {
                issue_wgmma(q_stage_idx);
                ptx::warpgroup_wait<0>();
                reduce_and_store(cute::Int<kNextNAtom>{}, weights, kv_offset);

                issue_tail_wgmma(q_tail_stage_idx);
                ptx::warpgroup_wait<0>();
                empty_kv_barriers[kv_stage_idx]->arrive();
                reduce_and_store(cute::Int<1>{}, tail_weights, kv_offset + 2 * static_cast<uint64_t>(logits_stride));
            } else {
                issue_wgmma(q_stage_idx);
                ptx::warpgroup_wait<0>();
                empty_kv_barriers[kv_stage_idx]->arrive();
                reduce_and_store(cute::Int<kNextNAtom>{}, weights, kv_offset);
            }
        }
    }
}

} // namespace deep_gemm
