#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/mega_mhc.cuh>
#include <deep_gemm/epilogue/sm100_mega_mhc.cuh>

namespace deep_gemm {

using namespace layout::mega_mhc;

namespace mega_mhc {

// A task is one (m-block, K-split) pair, subdivided into BLOCK_K blocks.
// UMMA_K is the per-route hidden width of one TF32 UMMA instruction.
constexpr uint32_t UMMA_K = 32 / sizeof(float);
constexpr uint32_t kNumTokenRowsPerLane = 2;

} // namespace mega_mhc

// Persistent mHC with static grid-stride roles and TMA-native layouts.
// Norm receives one arrival only after Shifted X1 stores and statistics are committed.
template <uint32_t kHidden, uint32_t kNumSplits, uint32_t kNumSMs,
          bool kIsShifted, bool kStoreBF16, bool kStoreFP8, uint32_t SF_BLOCK_M>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
    sm100_mega_mhc_impl(const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_fn,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_post_mix,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_comb_res_mix,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_shifted_prev_mix,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_new_residual,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_y_bf16,
                        const __grid_constant__ MixArgs mix_args,
                        const __grid_constant__ NormArgs norm_args,
                        void* gmem_scratch,
                        uint64_t* gmem_split_barriers) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using namespace mega_mhc;
    using Allocator = cute::TMEM::Allocator1Sm;
    // Each Post worker occupies exactly one WG; the role dispatch hardwires two of them.
    constexpr uint32_t kNumPostWGs = 2;

    // Template checks
    DG_STATIC_ASSERT(kHidden > 0 and kHidden % BLOCK_K == 0 and kNumSplits <= kHidden / BLOCK_K,
                     "Splits must contain nonempty, aligned K blocks");
    DG_STATIC_ASSERT(SF_BLOCK_M == 0 or kStoreFP8, "Shared SF requires FP8 output");

    // MMA and tensor memory configs
    constexpr uint32_t kNumTmemColumnsPerAStage = kNumRoutes * UMMA_K;
    constexpr uint32_t kAccumTmemStartColumn = kNumAStages * kNumTmemColumnsPerAStage;
    constexpr uint32_t kNumTmemCols =
        utils::get_num_aligned_tmem_cols<kAccumTmemStartColumn + kNumAccumStages * BLOCK_N>();
    // Task geometry
    constexpr uint32_t kNumKBlocksPerSplit = kHidden / BLOCK_K / kNumSplits;
    constexpr uint32_t kNumLongSplits = kHidden / BLOCK_K % kNumSplits;
    const auto get_task_k_begin = [&](const uint32_t split_idx) {
        return split_idx * (kNumKBlocksPerSplit * BLOCK_K) + cute::min(split_idx, kNumLongSplits) * BLOCK_K;
    };
    const auto get_num_task_k_blocks = [&](const uint32_t split_idx) {
        return kNumKBlocksPerSplit + (split_idx < kNumLongSplits);
    };

    // CTA role assignments
    constexpr uint32_t kFirstPostWGIdx = 1;
    constexpr uint32_t kWorkspaceEpilogueWGIdx = kFirstPostWGIdx + kNumPostWGs;
    constexpr uint32_t kMixWGIdx = kWorkspaceEpilogueWGIdx + 1;
    constexpr uint32_t kNormWGIdx = kMixWGIdx + 1;
    constexpr uint32_t kNumWGsPerCTA = kNormWGIdx + 1;
    constexpr uint32_t kNumPostWarps = kNumPostWGs * kNumWarpsPerWG;
    DG_STATIC_ASSERT(kNumWGsPerCTA == 6, "Unexpected CTA topology");

    // Register reconfigurations
    constexpr uint32_t kNumControlRegisters = 32;
    constexpr uint32_t kNumPostRegisters = kIsShifted ? 136 : 112;
    constexpr uint32_t kNumLightweightRegisters = 40;
    constexpr uint32_t kNumShiftedNormRegisters = 96;
    constexpr uint32_t kNumNormalRegisters = 80;

    // WG3 epilogue and WG4 Mix use the same lightweight allocation.
    constexpr uint32_t kStaticRoleRegisterBudget = kNumControlRegisters + kNumPostRegisters * kNumPostWGs +
                                                   2 * kNumLightweightRegisters + kNumShiftedNormRegisters;
    constexpr uint32_t kMaxRegisterBudget = cute::max(kStaticRoleRegisterBudget, kNumNormalRegisters * kNumWGsPerCTA);
    DG_STATIC_ASSERT(kMaxRegisterBudget * 128 <= 64512, "Register reconfiguration exceeds the SM budget");
    DG_STATIC_ASSERT(kMaxRegisterBudget <= 80 * kNumWGsPerCTA, "Register reconfiguration exceeds the CTA entry pool");

    // Thread indices
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Post and workspace-epilogue warps share this mapping: each lane owns two rows and one column pair.
    // NOTES: runtime indices are recomputed per role branch so no value stays live across the
    // register reconfigurations (which would spill to local memory).
    constexpr uint32_t kNumRowsPerWarp = BLOCK_M / kNumWarpsPerWG;
    constexpr uint32_t kNumRowGroupsPerWarp = kNumRowsPerWarp / kNumTokenRowsPerLane;
    constexpr uint32_t kNumLanesPerRowGroup = 32 / kNumRowGroupsPerWarp;
    const auto get_first_row_idx = [&] {
        return warp_idx % kNumWarpsPerWG * kNumRowsPerWarp + lane_idx / kNumLanesPerRowGroup;
    };

    // Runtime state
    const uint32_t num_m_blocks = Workspace<kNumSplits>::get_num_m_blocks(norm_args.num_tokens);
    const uint32_t num_tasks = num_m_blocks * kNumSplits;
    Workspace<kNumSplits> workspace(gmem_scratch, num_m_blocks);

    // Shared memory
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    const auto init_barriers = [](auto& barriers, const uint32_t count) {
        #pragma unroll
        for (auto& barrier: barriers)
            barrier.init(count);
    };

    // Descriptor cache hints and barriers are initialized once per persistent CTA.
    if (warp_idx == 0 and cute::elect_one_sync()) {
        smem.next_norm_task_ticket = 0;

        // Normal aliases the shifted-only descriptors to valid maps, so prefetch unconditionally.
        for (const auto& tensor_map: {&tensor_map_residual, &tensor_map_x, &tensor_map_fn,
                                      &tensor_map_post_mix, &tensor_map_comb_res_mix,
                                      &tensor_map_shifted_prev_mix, &tensor_map_new_residual,
                                      &tensor_map_y_bf16})
            cute::prefetch_tma_descriptor(tensor_map);

        init_barriers(smem.full_io_barriers, 1);
        init_barriers(smem.empty_io_barriers, 1);
        init_barriers(smem.store_ready_io_barriers, kNumPostWarps);
        init_barriers(smem.full_coeff_barriers, 1);
        init_barriers(smem.empty_coeff_barriers, kNumPostWarps);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 1 and cute::elect_one_sync()) {
        // Fn load pipeline.
        init_barriers(smem.full_fn_barriers, 1);
        init_barriers(smem.empty_fn_barriers, 1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2 and cute::elect_one_sync()) {
        // MMA, Post, and workspace-epilogue pipelines.
        init_barriers(smem.full_a_barriers, 128);
        init_barriers(smem.empty_a_barriers, 1);
        init_barriers(smem.full_accum_barriers, 1);
        init_barriers(smem.empty_accum_barriers, kNumWarpsPerWG);

        smem.full_stats_barrier.init(kNumPostWarps);
        smem.empty_stats_barrier.init(kNumWarpsPerWG + (kIsShifted ? 1 : 0));
        smem.mix_arrival_barrier.init(kNumWarpsPerWG + (kIsShifted ? 0 : 1));
        cutlass::arch::fence_barrier_init();
    }

    // Warp 3 owns the CTA-wide TMEM allocation.
    if (warp_idx == 3)
        Allocator().allocate(kNumTmemCols, &smem.tmem_base);

    // Local initialization can overlap the preceding grid, but persistent split barriers cannot.
    cudaGridDependencySynchronize();
    if (warp_idx == kFirstPostWGIdx * kNumWarpsPerWG + 1 and cute::elect_one_sync()) {
        for (uint32_t m_block_idx = blockIdx.x; m_block_idx < num_m_blocks; m_block_idx += kNumSMs)
            sched::mega_mhc::Mix::init(gmem_split_barriers, m_block_idx);
    }
    if constexpr (kIsShifted) {
        if (warp_idx == kFirstPostWGIdx * kNumWarpsPerWG and cute::elect_one_sync()) {
            for (uint32_t m_block_idx = blockIdx.x; m_block_idx < num_m_blocks; m_block_idx += kNumSMs)
                sched::mega_mhc::Norm::init(gmem_split_barriers, m_block_idx);
        }
    }

    __syncthreads();
    const uint32_t tmem_base = ptx::ld_shared(&smem.tmem_base);

    // Static roles follow the execution graph: WG0 TMA/MMA, WG1-2 Post, WG3 workspace epilogue, WG4 Mix, and WG5 Norm.
    if (warp_idx < kFirstPostWGIdx * kNumWarpsPerWG) {
        // WG0: producer pipelines
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();

        // IO and Fn warp workers follow the same task traversal; only their hidden-block copies differ.
        const auto run_tma_load_worker = [&](auto& empty_barriers, const auto& load_hidden_block) {
            constexpr uint32_t kNumStages = sizeof(empty_barriers) / sizeof(empty_barriers[0]);
            RingPipeline<kNumStages> pipeline;

            if (cute::elect_one_sync()) {
                for (uint32_t task_idx = blockIdx.x; task_idx < num_tasks; task_idx += kNumSMs) {
                    const uint32_t split_idx = task_idx % kNumSplits;
                    const uint32_t num_task_k_blocks = get_num_task_k_blocks(split_idx);
                    const uint32_t m_idx = task_idx / kNumSplits * BLOCK_M;

                    #pragma unroll 1
                    for (uint32_t k_block_idx = 0; k_block_idx < num_task_k_blocks; ++ k_block_idx) {
                        CUTE_TIE_DECL(pipeline.advance(), stage_idx, stage_phase);
                        empty_barriers[stage_idx].wait(stage_phase ^ 1u);
                        load_hidden_block(stage_idx, get_task_k_begin(split_idx) + k_block_idx * BLOCK_K, m_idx);
                    }
                }
            }
            __syncwarp();
        };

        if (warp_idx == 0) {
            // Warp 0: residual and X TMA-load worker
            run_tma_load_worker(smem.empty_io_barriers,
                [&](const uint32_t& stage_idx, const uint32_t& k_idx, const uint32_t& m_idx) {
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleMode, nv_bfloat16, true>(
                        &tensor_map_residual, &smem.full_io_barriers[stage_idx],
                        reinterpret_cast<nv_bfloat16*>(smem.residual[stage_idx][0]), k_idx, m_idx);
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleMode, nv_bfloat16>(
                        &tensor_map_x, &smem.full_io_barriers[stage_idx],
                        reinterpret_cast<nv_bfloat16*>(smem.x[stage_idx]), k_idx, m_idx);
                    smem.full_io_barriers[stage_idx].arrive_and_expect_tx(sizeof(smem.residual[0]) + sizeof(smem.x[0]));
                });
        } else if (warp_idx == 1) {
            // Warp 1: Fn TMA-load worker
            run_tma_load_worker(smem.empty_fn_barriers,
                [&](const uint32_t& stage_idx, const uint32_t& k_idx, const uint32_t&) {
                    tma::copy<BLOCK_K, kNumRoutes * kNumHCOutputs, kSwizzleMode, float, true>(
                        &tensor_map_fn, &smem.full_fn_barriers[stage_idx],
                        smem.fn[stage_idx][0][0], k_idx, 0u);
                    smem.full_fn_barriers[stage_idx].arrive_and_expect_tx(sizeof(smem.fn[0]));
                });
        } else if (warp_idx == 2) {
            // Warp 2: HC MMA worker
            using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float, BLOCK_M, BLOCK_N,
                                                   cute::UMMA::Major::K, cute::UMMA::Major::K>;
            const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float, BLOCK_M, BLOCK_N,
                                                                                cute::UMMA::Major::K, cute::UMMA::Major::K>();

            RingPipeline<kNumFnStages> fn_pipeline;
            RingPipeline<kNumAStages> a_pipeline;
            RingPipeline<kNumAccumStages> accum_pipeline;

            for (uint32_t task_idx = blockIdx.x; task_idx < num_tasks; task_idx += kNumSMs) {
                const uint32_t split_idx = task_idx % kNumSplits;
                const uint32_t num_task_k_blocks = get_num_task_k_blocks(split_idx);

                // Wait workspace-epilogue accumulator release
                CUTE_TIE_DECL(accum_pipeline.advance(), accum_stage_idx, accum_phase);
                smem.empty_accum_barriers[accum_stage_idx].wait(accum_phase ^ 1u);
                ptx::tcgen05_after_thread_sync();

                #pragma unroll 1
                for (uint32_t k_block_idx = 0; k_block_idx < num_task_k_blocks; ++ k_block_idx) {
                    // Wait TMA Fn arrival
                    CUTE_TIE_DECL(fn_pipeline.advance(), fn_stage_idx, fn_phase);
                    smem.full_fn_barriers[fn_stage_idx].wait(fn_phase);
                    ptx::tcgen05_after_thread_sync();

                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        // Wait Post A arrival
                        CUTE_TIE_DECL(a_pipeline.advance(), a_stage_idx, a_phase);
                        smem.full_a_barriers[a_stage_idx].wait(a_phase);
                        ptx::tcgen05_after_thread_sync();

                        // Hidden blocks, UMMA K atoms, and routes accumulate into one HC output block in fixed order.
                        #pragma unroll
                        for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
                            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_N, kFnAtomK, kSwizzleMode>(
                                            smem.fn[fn_stage_idx][umma_k_idx * UMMA_K / kFnAtomK][route_idx],
                                            0, umma_k_idx * UMMA_K % kFnAtomK);
                            umma_t::fma(tmem_base + a_stage_idx * kNumTmemColumnsPerAStage + route_idx * UMMA_K,
                                        b_desc, tmem_base + kAccumTmemStartColumn + accum_stage_idx * BLOCK_N,
                                        k_block_idx > 0 or umma_k_idx > 0 or route_idx > 0, runtime_instr_desc);
                        }
                        // Release the A stage
                        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_a_barriers[a_stage_idx]));
                    }
                    // Release the Fn stage
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_fn_barriers[fn_stage_idx]));
                }
                // Commit this split's accumulator
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.full_accum_barriers[accum_stage_idx]));
            }
        } else {
            // Warp 3: output-store, coefficient-prefetch, and Norm-arrival worker
            RingPipeline<kNumIOStages> io_pipeline;
            RingPipeline<kNumCoeffStages> coeff_pipeline;

            const auto issue_coeff_tma = [&](const uint32_t& m_idx, const uint32_t& stage_idx) {
                tma::copy<kNumRoutes, BLOCK_M, 0, float>(&tensor_map_post_mix, &smem.full_coeff_barriers[stage_idx],
                                                        smem.post_coeff[stage_idx][0], 0u, m_idx);
                tma::copy<kNumRoutes * kNumRoutes, BLOCK_M, kSwizzleMode / 2, float>(
                    &tensor_map_comb_res_mix, &smem.full_coeff_barriers[stage_idx], smem.comb_coeff[stage_idx][0], 0u, m_idx);
                if constexpr (kIsShifted)
                    tma::copy<kNumRoutes, BLOCK_M, 0, float>(&tensor_map_shifted_prev_mix, &smem.full_coeff_barriers[stage_idx],
                                                             smem.pre_coeff[stage_idx][0], 0u, m_idx);
                smem.full_coeff_barriers[stage_idx].arrive_and_expect_tx(sizeof(smem.post_coeff[0]) + sizeof(smem.comb_coeff[0]) +
                                                                         (kIsShifted ? sizeof(smem.pre_coeff[0]) : 0));
            };

            // The coefficient ring advances once per task, independently of the per-block IO ring.
            auto advance_coeff_pipeline = [&]() {
                CUTE_TIE_DECL(coeff_pipeline.advance(), stage_idx, phase);
                smem.empty_coeff_barriers[stage_idx].wait(phase ^ 1u);
                return stage_idx;
            };

            // The first task has no preceding store-drain window, so seed its coefficients here.
            if (blockIdx.x < num_tasks) {
                const uint32_t coeff_stage_idx = advance_coeff_pipeline();
                if (cute::elect_one_sync())
                    issue_coeff_tma(blockIdx.x / kNumSplits * BLOCK_M, coeff_stage_idx);
                __syncwarp();
            }

            for (uint32_t task_idx = blockIdx.x, task_phase = 0; task_idx < num_tasks; task_idx += kNumSMs, task_phase ^= 1u) {
                const uint32_t split_idx = task_idx % kNumSplits;
                const uint32_t num_task_k_blocks = get_num_task_k_blocks(split_idx);
                const uint32_t m_block_idx = task_idx / kNumSplits;
                if constexpr (kIsShifted)
                    sched::mega_mhc::Norm::wait_init<kNumSplits>(gmem_split_barriers, m_block_idx);
                uint32_t next_coeff_stage_idx = 0;

                #pragma unroll 1
                for (uint32_t k_block_idx = 0; k_block_idx < num_task_k_blocks; ++ k_block_idx) {
                    // Wait for all Post route stores of this hidden block
                    CUTE_TIE_DECL(io_pipeline.advance(), io_stage_idx, io_phase);
                    smem.store_ready_io_barriers[io_stage_idx].wait(io_phase);

                    const uint32_t k_idx = get_task_k_begin(split_idx) + k_block_idx * BLOCK_K;
                    const uint32_t m_idx = m_block_idx * BLOCK_M;
                    const bool should_prefetch_next_coeff =
                        k_block_idx == num_task_k_blocks - 1 and task_idx + kNumSMs < num_tasks;

                    // Wait for a reusable slot before entering the final store-drain overlap window.
                    if (should_prefetch_next_coeff)
                        next_coeff_stage_idx = advance_coeff_pipeline();

                    if (cute::elect_one_sync()) {
                        // New residual is re-read by the fused Normal workers, so it stays in L2
                        // (EVICT_LAST); Shifted consumes it only in the next layer.
                        ptx::tma_store_3d(&tensor_map_new_residual, smem.residual[io_stage_idx][0],
                                          k_idx, m_idx, 0u,
                                          kIsShifted ? cute::TMA::CacheHintSm100::EVICT_NORMAL
                                                     : cute::TMA::CacheHintSm100::EVICT_LAST);
                        if constexpr (kIsShifted)
                            ptx::tma_store_2d(&tensor_map_y_bf16, smem.x[io_stage_idx], k_idx,
                                              m_idx, cute::TMA::CacheHintSm100::EVICT_LAST);

                        cute::tma_store_arrive();
                        // Issue next-task coefficients here to avoid competing with the task's main TMA wave.
                        if (should_prefetch_next_coeff)
                            issue_coeff_tma((task_idx + kNumSMs) / kNumSplits * BLOCK_M, next_coeff_stage_idx);

                        // Complete the destination writes before releasing this IO stage.
                        ptx::tma_store_wait<0>();
                        smem.empty_io_barriers[io_stage_idx].arrive();
                    }
                }

                if constexpr (kIsShifted) {
                    // Arrive at Norm only after both Post workers have committed their row statistics.
                    smem.full_stats_barrier.wait(task_phase);
                    const auto x1_sqr_sum_partial = workspace.get_x1_sqr_sum_partial_ptr(task_idx);
                    float2 row_sqr_sums = {};
                    #pragma unroll
                    for (uint32_t post_worker_idx = 0; post_worker_idx < kNumPostWGs; ++ post_worker_idx) {
                        row_sqr_sums.x += smem.x1_sqr_sums[post_worker_idx][lane_idx];
                        row_sqr_sums.y += smem.x1_sqr_sums[post_worker_idx][lane_idx + 32];
                    }

                    x1_sqr_sum_partial[lane_idx] = row_sqr_sums.x;
                    x1_sqr_sum_partial[lane_idx + 32] = row_sqr_sums.y;
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        smem.empty_stats_barrier.arrive();
                        sched::mega_mhc::Norm::arrive<kNumSplits>(gmem_split_barriers, m_block_idx);
                    }
                } else if (cute::elect_one_sync()) {
                    smem.mix_arrival_barrier.arrive();
                }
            }
        }
    } else if (warp_idx < kWorkspaceEpilogueWGIdx * kNumWarpsPerWG) {
        // WG1-2: Post workers, one per WG
        const auto get_bf16x2_offset = [](const uint32_t& row_idx, const uint32_t& col_idx) {
            constexpr uint32_t kNumColsPerRow = kSwizzleMode / sizeof(nv_bfloat162);
            constexpr uint32_t kNumColsPerBank = 16 / sizeof(nv_bfloat162);
            const uint32_t bank_group_idx = (col_idx / kNumColsPerBank) ^ (row_idx & (kSwizzleMode / 16 - 1));
            return row_idx * kNumColsPerRow + bank_group_idx * kNumColsPerBank + col_idx % kNumColsPerBank;
        };

        // Comb rows use a separate 64B swizzle to distribute each four-float group across banks.
        const auto load_comb_coeff = [](const float* row, const uint32_t& row_idx, const uint32_t& coeff_idx) {
            const uint32_t bank_group_idx = (coeff_idx / 4) ^ ((row_idx / 2) & 3);
            return row[bank_group_idx * 4 + coeff_idx % 4];
        };

        DG_STATIC_ASSERT(UMMA_K % kNumLanesPerRowGroup == 0, "Invalid Post geometry");

        const auto run_post_worker = [&](auto post_worker) {
            constexpr uint32_t kPostWorkerIdx = decltype(post_worker)::value;
            cutlass::arch::warpgroup_reg_alloc<kNumPostRegisters>();
            const uint32_t col_pair_idx = lane_idx & (kNumLanesPerRowGroup - 1);
            const uint32_t first_row_idx = get_first_row_idx();
            const uint32_t row_indices[kNumTokenRowsPerLane] = {first_row_idx, first_row_idx + kNumRowGroupsPerWarp};

            RingPipeline<kNumIOStages> io_pipeline;
            RingPipeline<kNumAStages> a_pipeline;
            RingPipeline<kNumCoeffStages> coeff_pipeline;

            // Each Post worker maps to one WG; offset them so their A stages interleave.
            a_pipeline.advance(kPostWorkerIdx);

            for (uint32_t task_idx = blockIdx.x, task_phase = 0; task_idx < num_tasks; task_idx += kNumSMs, task_phase ^= 1u) {
                const uint32_t split_idx = task_idx % kNumSplits;
                const uint32_t num_task_k_blocks = get_num_task_k_blocks(split_idx);

                CUTE_TIE_DECL(coeff_pipeline.advance(), coeff_stage_idx, coeff_phase);
                float2 hc_norm_sqr_sums[kNumTokenRowsPerLane] = {};
                float2 x1_sqr_sums[kNumTokenRowsPerLane] = {};

                // Coefficients stay in registers for every hidden block, so release their SMEM stage immediately.
                smem.full_coeff_barriers[coeff_stage_idx].wait(coeff_phase);
                float2 row_pre_coeff[kNumRoutes];
                float2 row_post_coeff[kNumRoutes];
                #pragma unroll
                for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
                    if constexpr (kIsShifted)
                        row_pre_coeff[route_idx] = {smem.pre_coeff[coeff_stage_idx][row_indices[0]][route_idx],
                                                    smem.pre_coeff[coeff_stage_idx][row_indices[1]][route_idx]};
                    row_post_coeff[route_idx] = {smem.post_coeff[coeff_stage_idx][row_indices[0]][route_idx],
                                                 smem.post_coeff[coeff_stage_idx][row_indices[1]][route_idx]};
                }

                float2 row_comb_coeff[kNumRoutes * kNumRoutes];
                #pragma unroll
                for (uint32_t coeff_idx = 0; coeff_idx < kNumRoutes * kNumRoutes; ++ coeff_idx)
                    row_comb_coeff[coeff_idx] = {load_comb_coeff(smem.comb_coeff[coeff_stage_idx][row_indices[0]], row_indices[0], coeff_idx),
                                                 load_comb_coeff(smem.comb_coeff[coeff_stage_idx][row_indices[1]], row_indices[1], coeff_idx)};

                __syncwarp();
                if (cute::elect_one_sync()) {
                    cutlass::arch::fence_view_async_shared();
                    smem.empty_coeff_barriers[coeff_stage_idx].arrive();
                }

                // Round-robin UMMA atoms across Post workers and retain statistics until the split ends.
                #pragma unroll 1
                for (uint32_t k_block_idx = 0; k_block_idx < num_task_k_blocks; ++ k_block_idx) {
                    // Wait TMA residual and X arrival
                    CUTE_TIE_DECL(io_pipeline.advance(), io_stage_idx, io_phase);
                    smem.full_io_barriers[io_stage_idx].wait(io_phase);

                    #pragma unroll
                    for (uint32_t umma_k_idx = kPostWorkerIdx; umma_k_idx < BLOCK_K / UMMA_K;
                         umma_k_idx += kNumPostWGs) {
                        const uint32_t hidden_pair_idx = umma_k_idx * (UMMA_K / 2) +
                                                         col_pair_idx * (UMMA_K / kNumLanesPerRowGroup / 2);
                        float2 x[kNumTokenRowsPerLane];
                        float2 residual[kNumTokenRowsPerLane][kNumRoutes];
                        #pragma unroll
                        for (uint32_t row_idx = 0; row_idx < kNumTokenRowsPerLane; ++ row_idx) {
                            x[row_idx] = __bfloat1622float2(
                                smem.x[io_stage_idx][get_bf16x2_offset(row_indices[row_idx], hidden_pair_idx)]);
                            #pragma unroll
                            for (uint32_t input_route_idx = 0; input_route_idx < kNumRoutes; ++ input_route_idx)
                                residual[row_idx][input_route_idx] = __bfloat1622float2(
                                    smem.residual[io_stage_idx][input_route_idx][get_bf16x2_offset(row_indices[row_idx], hidden_pair_idx)]);
                        }

                        // Wait MMA release of this worker's next A stage
                        CUTE_TIE_DECL(a_pipeline.advance(kNumPostWGs), a_stage_idx, a_phase);
                        smem.empty_a_barriers[a_stage_idx].wait(a_phase ^ 1u);
                        ptx::tcgen05_after_thread_sync();

                        float2 x1[kNumTokenRowsPerLane] = {};
                        float2 post_values[kNumRoutes][kNumTokenRowsPerLane];
                        #pragma unroll
                        for (uint32_t row_idx = 0; row_idx < kNumTokenRowsPerLane; ++ row_idx) {
                            #pragma unroll
                            for (uint32_t output_route_idx = 0; output_route_idx < kNumRoutes; ++ output_route_idx) {
                                const float2 post = row_post_coeff[output_route_idx];
                                const float post_coeff = row_idx == 0 ? post.x : post.y;
                                auto& post_value = post_values[output_route_idx][row_idx];
                                post_value = __fmul2_rn(x[row_idx], {post_coeff, post_coeff});
                                #pragma unroll
                                for (uint32_t input_route_idx = 0; input_route_idx < kNumRoutes; ++ input_route_idx) {
                                    const uint32_t comb_idx = input_route_idx * kNumRoutes + output_route_idx;
                                    const float2 comb = row_comb_coeff[comb_idx];
                                    const float coeff = row_idx == 0 ? comb.x : comb.y;
                                    post_value = __ffma2_rn(residual[row_idx][input_route_idx], {coeff, coeff}, post_value);
                                }
                                // NOTES: this BF16 round is the shared numerical boundary for all consumers.
                                const nv_bfloat162 post_value_bf16 = __float22bfloat162_rn(post_value);
                                post_value = __bfloat1622float2(post_value_bf16);
                                smem.residual[io_stage_idx][output_route_idx][
                                    get_bf16x2_offset(row_indices[row_idx], hidden_pair_idx)] = post_value_bf16;
                                hc_norm_sqr_sums[row_idx] = __ffma2_rn(post_value, post_value, hc_norm_sqr_sums[row_idx]);
                                if constexpr (kIsShifted) {
                                    const float2 pre = row_pre_coeff[output_route_idx];
                                    const float coeff = row_idx == 0 ? pre.x : pre.y;
                                    x1[row_idx] = __ffma2_rn(post_value, {coeff, coeff}, x1[row_idx]);
                                }
                            }
                        }

                        const auto tmem_values = reinterpret_cast<const uint32_t(*)[4]>(post_values);
                        const uint32_t tmem_addr = tmem_base + a_stage_idx * kNumTmemColumnsPerAStage;
                        cute::SM100_TMEM_STORE_16dp256b4x::copy(tmem_values[0][0], tmem_values[0][1], tmem_values[0][2], tmem_values[0][3],
                                                                tmem_values[1][0], tmem_values[1][1], tmem_values[1][2], tmem_values[1][3],
                                                                tmem_values[2][0], tmem_values[2][1], tmem_values[2][2], tmem_values[2][3],
                                                                tmem_values[3][0], tmem_values[3][1], tmem_values[3][2], tmem_values[3][3], tmem_addr);

                        if constexpr (kIsShifted) {
                            #pragma unroll
                            for (uint32_t row_idx = 0; row_idx < kNumTokenRowsPerLane; ++ row_idx) {
                                const nv_bfloat162 x1_bf16 = __float22bfloat162_rn(x1[row_idx]);
                                // NOTES: X1 is already rounded to BF16; accumulate without unpacking it back to FP32.
                                ptx::accumulate_square(x1_sqr_sums[row_idx], x1_bf16);
                                smem.x[io_stage_idx][get_bf16x2_offset(row_indices[row_idx], hidden_pair_idx)] = x1_bf16;
                            }
                        }

                        // All four warps have issued their route stores before A becomes visible to the MMA warp.
                        cutlass::arch::fence_view_async_tmem_store();
                        ptx::tcgen05_before_thread_sync();
                        smem.full_a_barriers[a_stage_idx].arrive();
                    }

                    // Make generic-proxy writes visible before the output-store warp reads through the TMA async proxy.
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        smem.store_ready_io_barriers[io_stage_idx].arrive();
                    }
                }

                // Wait only when register-held statistics are committed to shared rows reused by the next task.
                smem.empty_stats_barrier.wait(task_phase ^ 1u);

                // Four lanes own both column pairs of a row, independent of token batch size and SM scheduling.
                const float reduced_hc_norm_0 = math::warp_reduce_sum<kNumLanesPerRowGroup>(hc_norm_sqr_sums[0].x + hc_norm_sqr_sums[0].y);
                const float reduced_hc_norm_1 = math::warp_reduce_sum<kNumLanesPerRowGroup>(hc_norm_sqr_sums[1].x + hc_norm_sqr_sums[1].y);
                float reduced_x1_0, reduced_x1_1;
                if constexpr (kIsShifted) {
                    reduced_x1_0 = math::warp_reduce_sum<kNumLanesPerRowGroup>(x1_sqr_sums[0].x + x1_sqr_sums[0].y);
                    reduced_x1_1 = math::warp_reduce_sum<kNumLanesPerRowGroup>(x1_sqr_sums[1].x + x1_sqr_sums[1].y);
                }

                smem.hc_norm_sqr_sums[kPostWorkerIdx][row_indices[0]] = reduced_hc_norm_0;
                smem.hc_norm_sqr_sums[kPostWorkerIdx][row_indices[1]] = reduced_hc_norm_1;
                if constexpr (kIsShifted) {
                    smem.x1_sqr_sums[kPostWorkerIdx][row_indices[0]] = reduced_x1_0;
                    smem.x1_sqr_sums[kPostWorkerIdx][row_indices[1]] = reduced_x1_1;
                }
                __syncwarp();

                if (cute::elect_one_sync())
                    smem.full_stats_barrier.arrive();
                __syncwarp();
            }
        };
        warp_idx < (kFirstPostWGIdx + 1) * kNumWarpsPerWG ?
            run_post_worker(cute::Int<0>{}) : run_post_worker(cute::Int<1>{});
        cutlass::arch::warpgroup_reg_dealloc<kNumLightweightRegisters>();
    } else if (warp_idx < kMixWGIdx * kNumWarpsPerWG) {
        // WG3: workspace-epilogue worker, draining one split accumulator block per grid-stride task
        cutlass::arch::warpgroup_reg_dealloc<kNumLightweightRegisters>();

        const uint32_t warp_idx_in_wg = warp_idx % kNumWarpsPerWG;
        const uint32_t col_pair_idx = lane_idx & (kNumLanesPerRowGroup - 1);
        const uint32_t first_row_idx = get_first_row_idx();
        const uint32_t row_indices[kNumTokenRowsPerLane] = {first_row_idx, first_row_idx + kNumRowGroupsPerWarp};

        RingPipeline<kNumAccumStages> accum_pipeline;

        for (uint32_t task_idx = blockIdx.x, task_phase = 0; task_idx < num_tasks; task_idx += kNumSMs, task_phase ^= 1u) {
            const uint32_t m_block_idx = task_idx / kNumSplits;
            if (warp_idx_in_wg == 0)
                sched::mega_mhc::Mix::wait_init<kNumSplits>(gmem_split_barriers, m_block_idx);

            const uint32_t token_indices[kNumTokenRowsPerLane] = {m_block_idx * BLOCK_M + row_indices[0],
                                                                  m_block_idx * BLOCK_M + row_indices[1]};
            const auto gemm_partial = workspace.get_gemm_partial_ptr(task_idx);

            // Wait this split's accumulated HC outputs
            CUTE_TIE_DECL(accum_pipeline.advance(), accum_stage_idx, accum_phase);
            smem.full_accum_barriers[accum_stage_idx].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            // Each group covers eight FP32 outputs per row; the 2x and 1x loads fetch all three groups.
            constexpr uint32_t kNumOutputValuesPerLoad = 32 / sizeof(float);
            constexpr uint32_t kNumOutputValuesPerLane = kNumOutputValuesPerLoad / kNumLanesPerRowGroup;
            DG_STATIC_ASSERT(kNumOutputValuesPerLoad % kNumLanesPerRowGroup == 0,
                             "Invalid workspace epilogue geometry");

            DG_STATIC_ASSERT(BLOCK_N == 3 * kNumOutputValuesPerLoad, "Unsupported workspace output width");
            uint32_t values[3 * kNumLanesPerRowGroup];
            const uint32_t tmem_addr = tmem_base + kAccumTmemStartColumn + accum_stage_idx * BLOCK_N;
            cute::SM100_TMEM_LOAD_16dp256b2x::copy(tmem_addr, values[0], values[1], values[2], values[3],
                                                              values[4], values[5], values[6], values[7]);
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr + 2 * kNumOutputValuesPerLoad,
                                                   values[8], values[9], values[10], values[11]);
            cutlass::arch::fence_view_async_tmem_load();

            #pragma unroll
            for (uint32_t output_idx = 0; output_idx < BLOCK_N; output_idx += kNumOutputValuesPerLoad) {
                const uint32_t value_idx = output_idx / kNumOutputValuesPerLane;
                float2 row_values[kNumTokenRowsPerLane] = {
                    {__uint_as_float(values[value_idx]), __uint_as_float(values[value_idx + 1])},
                    {__uint_as_float(values[value_idx + 2]), __uint_as_float(values[value_idx + 3])}};
                #pragma unroll
                for (uint32_t row_idx = 0; row_idx < kNumTokenRowsPerLane; ++ row_idx) {
                    if (token_indices[row_idx] >= norm_args.num_tokens)
                        row_values[row_idx] = {0.0f, 0.0f};
                }

                *reinterpret_cast<float2*>(
                    gemm_partial + row_indices[0] * BLOCK_N + output_idx + col_pair_idx * kNumOutputValuesPerLane) = row_values[0];
                *reinterpret_cast<float2*>(
                    gemm_partial + row_indices[1] * BLOCK_N + output_idx + col_pair_idx * kNumOutputValuesPerLane) = row_values[1];
            }

            ptx::tcgen05_before_thread_sync();
            if (cute::elect_one_sync())
                smem.empty_accum_barriers[accum_stage_idx].arrive();

            // Gate this split's Mix arrival on all four epilogue warps and, in Normal, the output-store warp.
            smem.full_stats_barrier.wait(task_phase);

            if (lane_idx < kNumRowsPerWarp) {
                const uint32_t row_idx = warp_idx_in_wg * kNumRowsPerWarp + lane_idx;
                const auto hc_norm_sqr_sum_partial = workspace.get_hc_norm_sqr_sum_partial_ptr(task_idx);
                float row_hc_norm_sqr_sum = 0.0f;
                #pragma unroll
                for (uint32_t post_worker_idx = 0; post_worker_idx < kNumPostWGs; ++ post_worker_idx)
                    row_hc_norm_sqr_sum += smem.hc_norm_sqr_sums[post_worker_idx][row_idx];
                hc_norm_sqr_sum_partial[row_idx] = row_hc_norm_sqr_sum;
            }

            __syncwarp();
            if (cute::elect_one_sync()) {
                smem.empty_stats_barrier.arrive();
                smem.mix_arrival_barrier.arrive();
            }

            if (warp_idx_in_wg == 0) {
                smem.mix_arrival_barrier.wait(task_phase);
                if (cute::elect_one_sync())
                    sched::mega_mhc::Mix::arrive<kNumSplits>(gmem_split_barriers, m_block_idx);
            }
        }
    } else if (warp_idx < kNormWGIdx * kNumWarpsPerWG) {
        // WG4: four shifted Mix workers, one per warp
        if constexpr (kIsShifted) {
            cutlass::arch::warpgroup_reg_dealloc<kNumLightweightRegisters>();
            const uint32_t warp_idx_in_mix_wg = warp_idx - kMixWGIdx * kNumWarpsPerWG;
            for (uint32_t token_idx = blockIdx.x * kNumWarpsPerWG + warp_idx_in_mix_wg;
                 token_idx < norm_args.num_tokens; token_idx += kNumSMs * kNumWarpsPerWG)
                epilogue::mega_mhc::run_mix_task<kHidden, true>(
                    workspace, gmem_split_barriers, mix_args, token_idx, lane_idx);
        }
    }

    if constexpr (not kIsShifted) {
        // Normal repurposes WG4-5 immediately and each producer WG as soon as its static role retires.
        if (warp_idx >= kMixWGIdx * kNumWarpsPerWG)
            cutlass::arch::warpgroup_reg_dealloc<kNumLightweightRegisters>();

        const uint32_t wg_idx_in_cta = warp_idx / kNumWarpsPerWG;

        // Each Normal worker owns a distinct named barrier.
        cutlass::arch::NamedBarrier::sync(128, 1 + wg_idx_in_cta);
        cutlass::arch::warpgroup_reg_alloc<kNumNormalRegisters>();

        epilogue::mega_mhc::run_normal_worker<kHidden, kNumSMs, kStoreBF16, kStoreFP8, SF_BLOCK_M>(
            smem, workspace, wg_idx_in_cta, warp_idx % kNumWarpsPerWG, lane_idx,
            gmem_split_barriers, mix_args, norm_args);
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
    } else {
        // WG5 starts immediately; retired Post workers join the same Norm task queue.
        const uint32_t wg_idx = warp_idx / kNumWarpsPerWG;
        if ((wg_idx >= kFirstPostWGIdx and wg_idx < kWorkspaceEpilogueWGIdx) or wg_idx == kNormWGIdx) {
            epilogue::mega_mhc::run_shifted_norm_worker<kHidden, kNumSMs, kNumShiftedNormRegisters,
                                                        kStoreBF16, kStoreFP8, SF_BLOCK_M>(
                smem, workspace, gmem_split_barriers, norm_args, lane_idx);
        }
    }

    // All roles must retire before the allocating warp releases TMEM.
    __syncthreads();
    if (warp_idx == 3)
        Allocator().free(tmem_base, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm
#pragma clang diagnostic pop
