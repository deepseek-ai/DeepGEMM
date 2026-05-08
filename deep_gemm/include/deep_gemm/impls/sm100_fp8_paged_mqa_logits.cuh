#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp8_paged_mqa_logits(const uint32_t batch_size,
                                const uint32_t logits_stride, const uint32_t block_table_stride,
                                const uint32_t* context_lens, logits_dtype_t* logits,
                                const uint32_t* block_table, const uint32_t* indices,
                                const uint32_t* schedule_meta,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    // For non-varlen odd kNextN >= 3, pad to even using TMA OOB zero-fill.
    static constexpr bool kPadOddN = (not kIsVarlen) and (kNextN % 2 == 1) and (kNextN >= 3);
    static constexpr uint32_t kNextNAtom = (kIsVarlen or kNextN >= 2) ? 2 : 1;
    static constexpr uint32_t kNumNextNAtoms = math::constexpr_ceil_div(kNextN, kNextNAtom);

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextNAtom * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = SPLIT_KV * sizeof(float);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextNAtom * kNumHeads * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q and KV data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i);
    });
    constexpr auto smem_offset = SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages;
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });

    // Barriers and TMEM pointer on shared memory
    const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    const auto umma_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + i; });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + kNumMathWarpGroups + i; });
    auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(umma_barrier_ptr + kNumMathWarpGroups * 2);

    constexpr uint32_t kNumTmemCols = kNextNAtom * kNumHeads * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Initialize barriers
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 1) {
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
    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Scheduler
    constexpr uint32_t kNumBlocksPerSplit = SPLIT_KV / BLOCK_KV;
    using Scheduler = sched::PagedMQALogitsScheduler<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumBlocksPerSplit, kNumNextNAtoms>;
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumBlocksPerSplit, "Invalid `SPLIT_KV`");

    // Q and KV pipeline
    const auto get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };

    // UMMA settings
    // Construct instruction with layout D
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = kNextNAtom * kNumHeads;
    DG_STATIC_ASSERT(SPLIT_KV == UMMA_M * kNumMathWarpGroups, "Invalid `SPLIT_KV`");

    if (warp_idx == kSpecWarpStart) {
        // TMA warp for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        auto scheduler = Scheduler(sm_idx, batch_size, context_lens, schedule_meta, indices);
        uint32_t q_iter_idx = 0, kv_iter_idx = 0;

        const auto issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& tma_q_atom_idx) {
            if (cute::elect_one_sync()) {
                const auto q_token_idx = Scheduler::atom_to_token_idx(tma_q_atom_idx);
                tma::copy<kHeadDim, kNextNAtom * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_token_idx * kNumHeads);
                tma::copy<kNextNAtom * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_token_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize outside valid range to indicate no previous task
        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx, num_kv;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_atom_idx), q_iter_idx = 1;

        uint32_t kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when (q, atom) changes
            const auto next_advance = scheduler.get_atom_advance(next_q_atom_idx, batch_size);
            bool prefetch_q = (q_atom_idx != next_q_atom_idx) and scheduler.exist_q_atom_idx(next_q_atom_idx + next_advance);

            if (q_atom_idx != next_q_atom_idx)
                kv_block_idx_ptr = 32;

            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Read KV block index
            // TODO(xuzhean): consider -1
            if (kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                const auto block_table_offset = Scheduler::atom_to_block_table_row(q_atom_idx) * static_cast<uint64_t>(block_table_stride);
                kv_block_idx_storage = (kv_idx + lane_idx < num_kv)
                    ? block_table[block_table_offset + kv_idx + lane_idx] : 0;
            }
            __syncwarp();
            DG_STATIC_ASSERT(32 % kNumBlocksPerSplit == 0, "Invalid `UMMA_M`");

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_atom_idx + next_advance);
            }

            uint32_t kv_block_idx[kNumBlocksPerSplit];
            #pragma unroll
            for (uint32_t i = 0; i < kNumBlocksPerSplit; ++ i)
                kv_block_idx[i] = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr + i);
            kv_block_idx_ptr += kNumBlocksPerSplit;

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumBlocksPerSplit; ++ i) {
                    tma::copy<kHeadDim, BLOCK_KV, 0, __nv_fp8_e4m3, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                                          smem_kv[kv_stage_idx] + (BLOCK_KV * kHeadDim) * i,
                                                                          0, 0, 1, kv_block_idx[i]);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                              smem_kv_scales[kv_stage_idx] + BLOCK_KV * i,
                                              0, kv_block_idx[i]);
                }
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv);
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        auto scheduler = Scheduler(sm_idx, batch_size, context_lens, schedule_meta, indices);
        uint32_t q_iter_idx = 0, kv_iter_idx = 0;

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 1;

        while (scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv)) {
            if (q_atom_idx != next_q_atom_idx) {
                // Release previous Q empty (UMMA warp must participate to prevent
                // running ahead of math warps in the Q pipeline)
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);
            }

            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;

            // Wait KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                empty_umma_barriers[i]->wait(umma_phase);    
                ptx::tcgen05_after_thread_sync();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                    auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                    auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_q[q_stage_idx], 0, k * UMMA_K);
                    cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
            }
            umma_phase ^= 1;
        }
    } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        // Math warpgroups for reduce
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        auto scheduler = Scheduler(sm_idx, batch_size, context_lens, schedule_meta, indices);
        uint32_t q_iter_idx = 0, kv_iter_idx = 0;

        // Offsets
        const auto math_warpgroup_idx = warpgroup_idx;
        const auto tmem_start = math_warpgroup_idx * UMMA_N;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Local register buffers
        float weights[kNextNAtom][kNumHeads];

        // Initialize outside valid range to indicate no previous task
        uint32_t q_atom_idx = batch_size * kNumNextNAtoms, kv_idx;
        uint32_t next_q_atom_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 0;
        bool is_paired_atom = false;

        while (scheduler.fetch_next_task(next_q_atom_idx, next_kv_idx, next_num_kv)) {
            // Q or atom changes
            if (q_atom_idx != next_q_atom_idx) {
                // Release last Q empty
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextNAtom; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; ++ j)
                        weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
                }

                if constexpr (kIsVarlen) {
                    is_paired_atom = (scheduler.get_atom_advance(next_q_atom_idx, batch_size) == 2);
                }
            }

            // Get current task indices
            q_atom_idx = next_q_atom_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = Scheduler::atom_to_token_idx(q_atom_idx) * static_cast<uint64_t>(logits_stride) + kv_idx * BLOCK_KV;

            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Read per-KV scales
            float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

            // Wait UMMA arrival
            full_umma_barriers[math_warpgroup_idx]->wait(umma_phase);
            ptx::tcgen05_after_thread_sync();
            umma_phase ^= 1;

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

            const auto reduce_and_store = [&](auto num_iters_c) {
                constexpr uint32_t kNumIters = decltype(num_iters_c)::value;
                float accum[kNumHeads];

                #pragma unroll
                for (uint32_t i = 0; i < kNumIters; ++ i) {
                    // Load accumulator from TMEM
                    tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads, accum);

                    // Accumulate weighted ReLU in parallel
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);

                    const auto transform = [&](const uint32_t& j, const float2& sum) {
                        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                        auto b = make_float2(weights[i][j], weights[i][j + 1]);
                        return __ffma2_rn(a, b, sum);
                    };

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        sum_0 = transform(j, sum_0);
                        sum_1 = transform(j + 2, sum_1);
                    }

                    auto sum = __fadd2_rn(sum_0, sum_1);
                    auto result = static_cast<logits_dtype_t>(scale_kv * (sum.x + sum.y));

                    // Store into the global memory
                    logits[kv_offset + i * static_cast<uint64_t>(logits_stride) + math_thread_idx] = result;
                    __syncwarp();
                }

                // Release TMEM empty
                ptx::tcgen05_before_thread_sync();
                empty_umma_barriers[math_warpgroup_idx]->arrive();
            };

            if constexpr (kIsVarlen) {
                if (is_paired_atom)
                    reduce_and_store(cute::Int<kNextNAtom>{});
                else
                    reduce_and_store(cute::Int<1>{});
            } else if constexpr (kPadOddN) {
                if (q_atom_idx % kNumNextNAtoms == kNumNextNAtoms - 1)
                    reduce_and_store(cute::Int<1>{});
                else
                    reduce_and_store(cute::Int<kNextNAtom>{});
            } else {
                reduce_and_store(cute::Int<kNextNAtom>{});
            }
        }

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
