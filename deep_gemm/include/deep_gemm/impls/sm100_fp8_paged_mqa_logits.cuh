#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;
using namespace deep_gemm::sm100;

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads>
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads + 128, 1)
void sm100_fp8_paged_mqa_logits(const uint32_t batch_size,
                                const uint64_t logits_stride, const uint64_t block_table_stride,
                                const uint32_t* context_lens, float* logits,
                                const uint32_t* block_table, const uint32_t* schedule_meta,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    // Prefetch TMA descriptors
    static constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
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
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
                                                  constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_UMMA_SIZE = kNumMathWarpGroups * 2 * 8 + static_cast<uint32_t>(sizeof(uint32_t));

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q data and barriers on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers    = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });

    // Separate math warpgroups and tma load warps into KV groups
    // Each math warpgroup corresponds to a tma load warp
    const auto& kv_group_idx = __shfl_sync(0xffffffff, threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);

    // Per group KV data and barriers on shared memory
    const auto& smem_kv_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + smem_kv_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales =  PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_kv_offset + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + i; });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + kNumKVStages + i; });

    // UMMA barriers and TMEM pointer on shared memroy 
    auto umma_barrier_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kNumMathWarpGroups);
    auto full_umma_barriers  = PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + i; });
    auto empty_umma_barriers = PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + kNumMathWarpGroups + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(umma_barrier_ptr + kNumMathWarpGroups * 2);

    constexpr uint32_t kNumTmemCols = kNextN * kNumHeads * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");
    const bool& is_math_warp = (warp_idx < (kNumMathThreads / 32));         // 0 ～ 16
    const bool& is_tma_load_warp = (warp_idx >= (kNumMathThreads / 32) and warp_idx < (kNumMathThreads / 32 + 4));  // 16 ～ 20
    const bool& is_umma_warp = (warp_idx == (kNumMathThreads / 32 + 4));    // 20

    // Initialize barriers
    if (is_tma_load_warp and cute::elect_one_sync()) {
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
        cutlass::arch::fence_barrier_init();
    }
    if (is_umma_warp) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 32;
    constexpr uint32_t kNumMathRegisters = 104;

    // Scheduler
    auto scheduler = PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups>(batch_size, blockIdx.x, context_lens, schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    // Q and KV pipeline
    const auto& get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto& get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    // UMMA settings
    // Construct instruction with layout F
    constexpr uint32_t UMMA_M = 64;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = kNextN * kNumHeads;

    if (is_tma_load_warp) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (kv_group_idx >= kNumMathWarpGroups)
            return;

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, q_idx * kNextN * kNumHeads);
                tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, q_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when current Q changes
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            // Read KV block index
            // TODO: deal with `-1`?
            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + kv_group_idx + + lane_idx * kNumMathWarpGroups < num_kv ?
                    __ldg(block_table + q_idx * block_table_stride + (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)) : 0);
            }
            const auto& kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            // Issue TMA KV
            if (cute::elect_one_sync()) {
                tma_3d_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                            smem_kv[kv_stage_idx], 0, 0, kv_block_idx);
                tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                            smem_kv_scales[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else if (is_umma_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 0;

        auto smem_kv = PatternVisitor([&](const uint32_t& stage_idx) {
            return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_PIPE_SIZE + SMEM_KV_SIZE_PER_STAGE * stage_idx);
        });

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            if (q_idx != next_q_idx) {
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
            }

            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);

            DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");
            DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
        
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                empty_umma_barriers[i]->wait(umma_phase & 1);    
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                    auto a_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_kv[kv_stage_idx] + i * SMEM_KV_PIPE_SIZE, 0, k * UMMA_K);
                    auto b_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_q[q_stage_idx], 0, k * UMMA_K);
                    cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
            }
            umma_phase ^= 1;
        }
    } else if (is_math_warp) {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Offsets
        const auto& tmem_start = __shfl_sync(0xffffffff, warpgroup_idx * UMMA_N, 0);
        float weights[kNextN][kNumHeads / 4];
        const auto& sub_warp_offset = (warp_idx % 4) * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 0;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            // Current Q changes
            if (q_idx != next_q_idx) {
                // Release Last Q empty
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
            }

            // Get current Q and KV index
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = q_idx * kNextN * logits_stride + ((kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset);

            // Compute `[kNextN * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [kNextN, BLOCK_KV]`
            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Read per-KV scales
            auto scale_kv = make_float2(ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_0_offset),
                                        ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_1_offset));

            empty_umma_barriers[warpgroup_idx]->arrive();
            // Wait UMMA arrival
            full_umma_barriers[warpgroup_idx]->wait(umma_phase & 1);
            umma_phase ^= 1;

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store
            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
                // Load from the tensor memory
                constexpr uint32_t kNumLDTMElems = UMMA_M * kNumHeads / 128;
                uint32_t shifted_accum[kNumLDTMElems];
                DG_STATIC_ASSERT(kNumLDTMElems == 16 or kNumLDTMElems == 32 or kNumLDTMElems == 64, "Invalid LDTM");
                auto tmem_load = [&](auto... Is) {
                    if constexpr (kNumLDTMElems == 16) {
                        cute::SM100_TMEM_LOAD_16dp256b4x::copy(tmem_start + i * kNumHeads, shifted_accum[Is]...);
                    } else if constexpr (kNumLDTMElems == 32) {
                        cute::SM100_TMEM_LOAD_16dp256b8x::copy(tmem_start + i * kNumHeads, shifted_accum[Is]...);
                    } else if constexpr (kNumLDTMElems == 64) {
                        cute::SM100_TMEM_LOAD_16dp256b16x::copy(tmem_start + i * kNumHeads, shifted_accum[Is]...);
                    }
                };
                [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load(Is...); }(cute::make_index_sequence<kNumLDTMElems>{});
                cutlass::arch::fence_view_async_tmem_load();

                // Transform
                const auto& transform_2 = [&](const uint32_t& j, const uint32_t& k, const float2& sum) {
                    auto a = make_float2(fmaxf(*reinterpret_cast<float*>(&shifted_accum[j * 4 + k]), 0),
                                            fmaxf(*reinterpret_cast<float*>(&shifted_accum[j * 4 + k + 2]), 0));
                    auto b = make_float2(weights[i][j * 2 + k], weights[i][j * 2 + k]);
                    return __ffma2_rn(a, b, sum);
                };

                // Intra-thread reduction
                auto sum_0 = make_float2(0, 0);
                auto sum_1 = make_float2(0, 0);
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 8; ++ j) {
                    sum_0 = transform_2(j, 0, sum_0);
                    sum_1 = transform_2(j, 1, sum_1);
                }
                auto v = __fmul2_rn(__fadd2_rn(sum_0, sum_1), scale_kv);

                // Inter-thread reduction
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const auto& offset = 1u << j;
                    v.x += __shfl_xor_sync(0xffffffffu, v.x, offset);
                    v.y += __shfl_xor_sync(0xffffffffu, v.y, offset);
                }
                // Store into the global memory
                // NOTES: we have redundant writes here, consider more carefully
                logits[kv_offset + i * logits_stride + v_0_offset] = v.x;
                logits[kv_offset + i * logits_stride + v_1_offset] = v.y;
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    }

    // Free tensor memory
    __syncthreads();
    if (is_umma_warp)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
}

} // namespace deep_gemm
