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
void sm120_fp4_paged_mqa_logits(const uint32_t batch_size,
                                const uint32_t logits_stride, const uint32_t block_table_stride,
                                const uint32_t* context_lens, logits_dtype_t* logits,
                                const uint32_t* block_table, const uint32_t* indices,
                                const uint32_t* schedule_meta,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    using namespace mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = FP4_MMA_M;
    static constexpr uint32_t MMA_N = FP4_MMA_N;
    static constexpr uint32_t MMA_K = FP4_MMA_K;
    static constexpr uint32_t kKSteps = kHeadDim / MMA_K;
    static constexpr uint32_t kNTilesPerQ = kNumHeads / MMA_N;
    static constexpr uint32_t kLdmK = MMA_K / 2;

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kWarpsPerGroup = BLOCK_KV / MMA_M;
    static constexpr uint32_t kNumGroups = kNumMathWarps / kWarpsPerGroup;
    static constexpr uint32_t kSwizzleMode = 64;
    static constexpr uint32_t kSMEMKBytes = kHeadDim / 2;

    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Expected 128 TMA threads");
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumGroups, "SPLIT_KV = BLOCK_KV * groups");
    DG_STATIC_ASSERT(BLOCK_KV % MMA_M == 0 and kNumMathWarps == kWarpsPerGroup * kNumGroups, "Warp grouping error");
    DG_STATIC_ASSERT(kHeadDim == 128, "FP4 MQA only supports head_dim=128");

    // SMEM sizes (FP4 packed: half data, int32 SF)
    static constexpr uint32_t kNextNAtom = (kIsVarlen or kNextN >= 2) ? 2 : 1;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextNAtom * kNumHeads * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(uint32_t);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = BLOCK_KV * sizeof(uint32_t);

    static constexpr uint32_t kSwizzleAlignment = (kHeadDim / 2) * 8;

    // SMEM layout: Q pipe (shared) + per-group KV pipes
    static constexpr uint32_t ALIGNED_SMEM_SF_Q_SIZE_PER_STAGE = math::constexpr_align(SMEM_SF_Q_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = math::constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_SF_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 math::constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);
    static constexpr uint32_t ALIGNED_SMEM_SF_KV_SIZE_PER_STAGE = math::constexpr_align(SMEM_SF_KV_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_SF_KV_SIZE_PER_STAGE) +
                                                  math::constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    const auto warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto lane_idx = ptx::get_lane_idx();

    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    // Q pipe (shared across groups)
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_SF_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages +
            ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers = utils::PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });

    // Per-group KV pipe
    const auto kv_group_idx = __shfl_sync(0xffffffff,
        threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warp_idx / kWarpsPerGroup, 0);
    const auto smem_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + smem_offset +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_SF_KV_SIZE_PER_STAGE * i);
    });
    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_sf_kv[kNumKVStages]);
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
        if (kv_group_idx < kNumGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++ i) {
                full_kv_barriers[i]->init(1);
                empty_kv_barriers[i]->init(kWarpsPerGroup * 32);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    cudaGridDependencySynchronize();

    // Scheduler
    static constexpr bool kPadOddN = (not kIsVarlen) and (kNextN % 2 == 1) and (kNextN >= 3);
    static constexpr uint32_t kNumNextNAtoms = math::constexpr_ceil_div(kNextN, kNextNAtom);
    auto scheduler = sched::PagedMQALogitsScheduler<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumGroups, kNumNextNAtoms>(
        blockIdx.x, batch_size, context_lens, schedule_meta, indices);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    const auto get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1};
    };
    const auto get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1};
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warps
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (kv_group_idx >= kNumGroups)
            return;

        using Scheduler = decltype(scheduler);
        const auto issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                const auto q_token_idx = Scheduler::atom_to_token_idx(q_idx);
                tma::copy<kHeadDim, kNextNAtom * kNumHeads, 0>(
                    &tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx],
                    0, q_token_idx * kNumHeads);
                tma::copy<kNextNAtom * kNumHeads, 1, 0>(
                    &tensor_map_sf_q, full_q_barriers[stage_idx], smem_sf_q[stage_idx],
                    0, q_token_idx);
                tma::copy<kNextNAtom * kNumHeads, 1, 0>(
                    &tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx],
                    0, q_token_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(
                    SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        uint32_t q_idx = batch_size * kNumNextNAtoms, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            const auto next_advance = scheduler.get_last_advance();
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_atom_idx(next_q_idx + next_advance));

            if (q_idx != next_q_idx)
                kv_block_idx_ptr = 32;

            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + next_advance);
            }

            if (kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * kNumGroups < num_kv ?
                    block_table[scheduler.atom_to_block_table_row(q_idx) * static_cast<uint64_t>(block_table_stride) +
                                (kv_idx + kv_group_idx + lane_idx * kNumGroups)] : 0);
            }
            const auto kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            if (cute::elect_one_sync()) {
                tma::copy<kHeadDim, BLOCK_KV, 0, uint8_t, true>(
                    &tensor_map_kv, full_kv_barriers[kv_stage_idx],
                    smem_kv[kv_stage_idx], 0, 0, 1, kv_block_idx);
                tma::copy<BLOCK_KV, 1, 0>(
                    &tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
                    smem_sf_kv[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
                    SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
            }

            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else {
        // Math warps
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const uint32_t warp_in_group = warp_idx % kWarpsPerGroup;
        const uint32_t g = lane_idx / 4;
        const uint32_t t = lane_idx % 4;
        const uint32_t a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8;

        using Scheduler = decltype(scheduler);
        uint32_t q_idx = batch_size * kNumNextNAtoms, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        bool is_paired_atom = false;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            if (q_idx != next_q_idx) {
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                if constexpr (kIsVarlen) {
                    is_paired_atom = (scheduler.get_last_advance() == 2);
                }
            }

            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            auto kv_offset = scheduler.atom_to_token_idx(q_idx) * static_cast<uint64_t>(logits_stride) +
                             ((kv_idx + kv_group_idx) * BLOCK_KV + warp_in_group * MMA_M);

            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Load KV SF
            const uint32_t sf_kv_packed = sm120::load_sf(
                reinterpret_cast<const char*>(smem_sf_kv[kv_stage_idx]),
                warp_in_group * MMA_M + g + (t & 1) * 8);

            sm120::SwizzleContext<kSwizzleMode> a_ctx;
            a_ctx.init(a_row + warp_in_group * MMA_M, kSMEMKBytes);

            const auto compute_and_store = [&](auto kNumQTokens) {
                #pragma unroll
                for (uint32_t qi = 0; qi < decltype(kNumQTokens)::value; ++ qi) {
                    float partial_0 = 0.0f, partial_1 = 0.0f;

                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerQ; ++ nt) {
                        const uint32_t n_tile = qi * kNTilesPerQ + nt;
                        float d[4] = {0, 0, 0, 0};

                        sm120::SwizzleContext<kSwizzleMode> b_ctx;
                        b_ctx.init((lane_idx & 7) + n_tile * MMA_N, kSMEMKBytes);

                        const uint32_t sf_q_packed = sm120::load_sf(
                            reinterpret_cast<const char*>(smem_sf_q[q_stage_idx]),
                            qi * kNumHeads + nt * MMA_N + g);

                        #pragma unroll
                        for (uint32_t k = 0; k < kKSteps; ++ k) {
                            uint32_t a_frag[4];
                            sm120::load_a_fragment<kSwizzleMode>(
                                a_frag, reinterpret_cast<char*>(smem_kv[kv_stage_idx]),
                                a_ctx, lane_idx, k, kLdmK);

                            uint32_t b_frag[2];
                            sm120::load_b_fragment_x2<kSwizzleMode>(
                                b_frag, reinterpret_cast<char*>(smem_q[q_stage_idx]),
                                b_ctx, lane_idx, k, kLdmK);

                            uint16_t sfa = extract_sf_pair(sf_kv_packed, k * 2);
                            uint16_t sfb = extract_sf_pair(sf_q_packed, k * 2);

                            fp4_mma_block_scaled(d, a_frag, b_frag, sfa, sfb);
                        }

                        const uint32_t h0 = nt * MMA_N + t * 2;
                        const float w0 = ptx::ld_shared(smem_weights[q_stage_idx] + qi * kNumHeads + h0);
                        const float w1 = ptx::ld_shared(smem_weights[q_stage_idx] + qi * kNumHeads + h0 + 1);
                        partial_0 += fmaxf(d[0], 0.0f) * w0 + fmaxf(d[1], 0.0f) * w1;
                        partial_1 += fmaxf(d[2], 0.0f) * w0 + fmaxf(d[3], 0.0f) * w1;
                    }

                    float v_0 = partial_0;
                    float v_1 = partial_1;

                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    logits[kv_offset + qi * static_cast<uint64_t>(logits_stride) + g] = static_cast<logits_dtype_t>(v_0);
                    logits[kv_offset + qi * static_cast<uint64_t>(logits_stride) + g + 8] = static_cast<logits_dtype_t>(v_1);
                }
            };

            if constexpr (kIsVarlen) {
                if (is_paired_atom)
                    compute_and_store(cute::Int<kNextNAtom>{});
                else
                    compute_and_store(cute::Int<1>{});
            } else if constexpr (kPadOddN) {
                if (q_idx % kNumNextNAtoms == kNumNextNAtoms - 1)
                    compute_and_store(cute::Int<1>{});
                else
                    compute_and_store(cute::Int<kNextNAtom>{});
            } else {
                compute_and_store(cute::Int<kNextNAtom>{});
            }

            empty_kv_barriers[kv_stage_idx]->arrive();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
