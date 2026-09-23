#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>

#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_mega_gate.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/mega_gate.cuh>

namespace deep_gemm {

namespace mega_gate_layout = layout::mega_gate;

CUTLASS_HOST_DEVICE constexpr uint32_t get_num_stages_per_merge(const uint32_t num_stages,
                                                                const uint32_t num_k_atoms_per_slice) {
    return num_stages / 3 >= 8 and num_k_atoms_per_slice % 3 == 0 ? 3u :
           num_stages / 2 >= 8 and num_k_atoms_per_slice % 2 == 0 ? 2u : 1u;
}

template <uint32_t SHAPE_K, uint32_t kNumRoutedExperts,
          uint32_t BLOCK_TOKENS,
          uint32_t kNumStages_,
          uint32_t kNumGateThreads,
          uint32_t kNumMmaCtas, uint32_t kNumSplitK, uint32_t kNumExpertGroups,
          uint32_t kNumSMs,
          uint32_t kNumTopk, uint32_t kScoringType,
          bool kFullExpertTile, bool kMaskExists, bool kUnmappedTopkIdxExists, bool kToPhysicalMapExists,
          bool kHasBias, bool kHasImageTokenMask, bool kFixRoutingMaskExists, bool kForceRandomExists>
CUTLASS_GLOBAL void __launch_bounds__(mega_gate_layout::kNumNonEpilogueThreads + kNumGateThreads, 1)
sm100_bf16_mega_gate_impl(const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weight,
                          const float* bias, const float* image_bias, const bool* image_token_mask,
                          const bool* mask, const int* to_physical_map, const int* logical_count,
                          int64_t* topk_idx, int64_t* unmapped_topk_idx, float* topk_weights,
                          void* gmem_scratch, void* gmem_score_barriers,
                          uint32_t num_tokens, uint32_t num_routed_experts,
                          uint32_t num_shared_experts, uint32_t num_duplicate_experts,
                          float routed_scaling_factor, int ep_rank, int64_t unmapped_topk_idx_stride,
                          const bool* fix_routing_mask, const bool* force_random) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Allocator = cute::conditional_t<kNumMmaCtas == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    constexpr uint32_t kNumNonEpilogueWarps = mega_gate_layout::kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumGateWarps = kNumGateThreads / 32;
    constexpr uint32_t BLOCK_ATOM_K = mega_gate_layout::BLOCK_K;
    constexpr uint32_t kNumExpertsPerGroup = kNumRoutedExperts / kNumExpertGroups;

    constexpr uint32_t UMMA_M = kNumExpertsPerGroup;
    constexpr uint32_t UMMA_N = BLOCK_TOKENS;
    constexpr uint32_t kNumLogicalCtas = kNumMmaCtas * kNumExpertGroups * kNumSplitK;

    constexpr uint32_t kNumStagesPerMerge = get_num_stages_per_merge(kNumStages_, (SHAPE_K / BLOCK_ATOM_K) / kNumSplitK);
    constexpr uint32_t BLOCK_K = BLOCK_ATOM_K * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;
    constexpr uint32_t LOAD_BLOCK_M = UMMA_N / kNumMmaCtas;
    constexpr uint32_t LOAD_BLOCK_N = UMMA_M / kNumMmaCtas;
    DG_STATIC_ASSERT(kNumExpertsPerGroup <= 256, "An expert group must be a single UMMA tile");
    constexpr uint32_t kNumEpilogueStages = mega_gate_layout::kNumEpilogueStages;
    DG_STATIC_ASSERT(UMMA_N <= mega_gate_layout::kNumMaxBlockTokens, "Block tokens exceed a single UMMA tile");
    using SharedStorage = mega_gate_layout::SharedStorage<kNumStages, kNumEpilogueStages, kNumRoutedExperts,
                                                          LOAD_BLOCK_M, LOAD_BLOCK_N, BLOCK_K,
                                                          kHasBias, kHasImageTokenMask, kToPhysicalMapExists>;
    using Barrier = typename SharedStorage::Barrier;

    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumEpilogueStages * UMMA_N>();

    const auto mma_cta_rank = cute::block_rank_in_cluster();
    const auto mma_cluster_idx = blockIdx.x / kNumMmaCtas;
    const auto slice_idx = mma_cluster_idx % (kNumExpertGroups * kNumSplitK);
    const auto expert_group_idx = slice_idx % kNumExpertGroups;
    const auto split_k_idx = slice_idx / kNumExpertGroups;
    const auto logical_cta_rank = slice_idx * kNumMmaCtas + mma_cta_rank;
    constexpr uint32_t kNumKBlocksPerSlice = (SHAPE_K / BLOCK_K) / kNumSplitK;
    const auto split_k_offset = split_k_idx * kNumKBlocksPerSlice * BLOCK_K;
    const auto is_leader_cta = mma_cta_rank == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_weight);
    }

    extern __shared__ __align__(mega_gate_layout::kSharedMemoryAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const auto smem_x = smem.x;
    const auto smem_weight = smem.weight;
    const auto smem_bias = smem.get_bias_ptr();
    const auto smem_image_bias = smem.get_image_bias_ptr();
    const auto smem_logical_count = smem.get_logical_count_ptr();
    const auto routed_logical_count = SharedStorage::kCacheLogicalCount ? smem_logical_count : logical_count;

    if constexpr (kNumMmaCtas > 1)
        comm::cluster_sync_with_relaxed_arrive();

    if (warp_idx < 4 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t stage_idx = warp_idx; stage_idx < kNumStages; stage_idx += 4) {
            smem.full_barriers[stage_idx].init(kNumMmaCtas);
            smem.empty_barriers[stage_idx].init(1);
        }
        if (warp_idx == 0) {
            #pragma unroll
            for (uint32_t accum_stage_idx = 0; accum_stage_idx < kNumEpilogueStages; ++ accum_stage_idx) {
                smem.tmem_full_barriers[accum_stage_idx].init(1);
                smem.tmem_empty_barriers[accum_stage_idx].init(kNumMmaCtas);
            }
            if constexpr (SharedStorage::kHasMetadataCache)
                smem.metadata_ready_barrier.init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();
    if (warp_idx == 2)
        Allocator().allocate(kNumTmemCols, &smem.tmem_ptr);
    comm::cluster_sync_with_relaxed_arrive();

    cudaGridDependencySynchronize();

    const auto get_tmem_base_addr = [](const uint32_t& accum_stage_idx) {
        return accum_stage_idx * UMMA_N;
    };
    const sched::mega_gate::Scheduler<kNumSMs, kNumLogicalCtas, BLOCK_TOKENS> scheduler(num_tokens);
    const mega_gate_layout::Workspace<kNumSplitK, BLOCK_TOKENS, kNumRoutedExperts> workspace(gmem_scratch,
                                                                                           gmem_score_barriers);
    const auto get_effective_umma_n = [&](const uint32_t& token_block_idx) {
        return math::align(cute::min(BLOCK_TOKENS, num_tokens - token_block_idx * BLOCK_TOKENS), 16u);
    };

    const auto produce_tma = [&](const uint32_t& token_block_idx, const uint32_t& iter_idx) {
        const auto first_k_block_idx = iter_idx * kNumKBlocksPerSlice;
        const auto expert_base_idx = expert_group_idx * kNumExpertsPerGroup + mma_cta_rank * LOAD_BLOCK_N;
        const auto token_base_idx = token_block_idx * BLOCK_TOKENS;
        const auto effective_umma_n = get_effective_umma_n(token_block_idx);
        const auto x_token_idx = token_base_idx + mma_cta_rank * (effective_umma_n / kNumMmaCtas);
        if (cute::elect_one_sync()) {
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocksPerSlice; ++ k_block_idx) {
                const auto pipeline_idx = first_k_block_idx + k_block_idx;
                const auto stage_idx = pipeline_idx % kNumStages;
                const auto phase = pipeline_idx / kNumStages & 1;
                smem.empty_barriers[stage_idx].wait(phase ^ 1);

                const auto k_idx = split_k_offset + k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, 128, cutlass::bfloat16_t, false,
                          static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST)>(&tensor_map_x,
                                                                                         &smem.full_barriers[stage_idx],
                                                                                         smem_x[stage_idx], k_idx,
                                                                                         x_token_idx, kNumMmaCtas);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, 128, cutlass::bfloat16_t>(&tensor_map_weight,
                                                                           &smem.full_barriers[stage_idx],
                                                                           smem_weight[stage_idx], k_idx,
                                                                           expert_base_idx, kNumMmaCtas);
                if (is_leader_cta) {
                    constexpr uint32_t kNumArrivalBytes = sizeof(smem_x[0]) + sizeof(smem_weight[0]);
                    smem.full_barriers[stage_idx].arrive_and_expect_tx(kNumArrivalBytes * kNumMmaCtas);
                } else {
                    smem.full_barriers[stage_idx].arrive(0u);
                }
            }
        }
    };

    const auto produce_mma = [&](const uint32_t& token_block_idx, const uint32_t& iter_idx) {
        const auto first_k_block_idx = iter_idx * kNumKBlocksPerSlice;
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto x_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_ATOM_K, 128>(smem_x[0], 0, 0);
        auto weight_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_ATOM_K, 128>(smem_weight[0], 0, 0);

        const auto x_desc_lo = lane_idx < kNumStages ? x_desc.lo + lane_idx * sizeof(smem_x[0]) / 16 : 0u;
        const auto weight_desc_lo = lane_idx < kNumStages ? weight_desc.lo + lane_idx * sizeof(smem_weight[0]) / 16 : 0u;

        auto umma_arrive = [&](const Barrier* barrier) {
            if constexpr (kNumMmaCtas == 1) {
                cutlass::arch::umma_arrive(reinterpret_cast<const uint64_t*>(barrier));
            } else {
                constexpr auto pair_cta_mask = static_cast<uint16_t>((1u << kNumMmaCtas) - 1u);
                cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<const uint64_t*>(barrier), pair_cta_mask);
            }
        };

        const auto effective_umma_n = get_effective_umma_n(token_block_idx);
        mma::sm100::update_instr_desc_with_umma_n(instr_desc, effective_umma_n);

        const auto accum_stage_idx = iter_idx % kNumEpilogueStages;
        const auto accum_phase_idx = iter_idx / kNumEpilogueStages & 1;
        smem.tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
        ptx::tcgen05_after_thread_sync();

        for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocksPerSlice; ++ k_block_idx) {
            const auto pipeline_idx = first_k_block_idx + k_block_idx;
            const auto stage_idx = pipeline_idx % kNumStages;
            const auto phase = pipeline_idx / kNumStages & 1;
            const auto x_desc_base_lo = ptx::exchange(x_desc_lo, stage_idx);
            const auto weight_desc_base_lo = ptx::exchange(weight_desc_lo, stage_idx);
            smem.full_barriers[stage_idx].wait(phase);
            ptx::tcgen05_after_thread_sync();
            const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / 16; ++ umma_k_idx) {
                    const auto atom_k_idx = umma_k_idx * 16 / BLOCK_ATOM_K;
                    const auto inner_k_idx = umma_k_idx * 16 % BLOCK_ATOM_K;
                    using mma_t = cute::conditional_t<kNumMmaCtas == 1,
                                                      ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                    x_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, 128, cutlass::bfloat16_t>(
                                    x_desc_base_lo, atom_k_idx * LOAD_BLOCK_M * BLOCK_ATOM_K, inner_k_idx);
                    weight_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, 128, cutlass::bfloat16_t>(
                                        weight_desc_base_lo, atom_k_idx * LOAD_BLOCK_N * BLOCK_ATOM_K,
                                                                                    inner_k_idx);
                    mma_t::fma(weight_desc, x_desc, get_tmem_base_addr(accum_stage_idx),
                               umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc);
                }
            }
            __syncwarp();

            if (k_block_idx == kNumKBlocksPerSlice - 1)
                umma_arrive(&smem.tmem_full_barriers[accum_stage_idx]);
            __syncwarp();

            umma_arrive(&smem.empty_barriers[stage_idx]);
        }
    };

    if constexpr (SharedStorage::kHasMetadataCache) {
        if (warp_idx == kNumNonEpilogueWarps - 1) {
            #pragma unroll
            for (uint32_t cache_round_idx = 0; cache_round_idx < kNumRoutedExperts / 32; ++ cache_round_idx) {
                const auto cache_idx = cache_round_idx * 32 + lane_idx;
                const auto is_valid_expert = kFullExpertTile or cache_idx < num_routed_experts;
                if constexpr (kHasBias)
                    smem_bias[cache_idx] = is_valid_expert ? bias[cache_idx] : 0.0f;
                if constexpr (kHasImageTokenMask)
                    smem_image_bias[cache_idx] = is_valid_expert ? image_bias[cache_idx] : 0.0f;
                if constexpr (SharedStorage::kCacheLogicalCount)
                    smem_logical_count[cache_idx] = is_valid_expert ? logical_count[cache_idx] : 0;
            }
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (lane_idx == 0)
                smem.metadata_ready_barrier.arrive();
        }
    }

    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);

    const auto gate_warp_idx = warp_idx - kNumNonEpilogueWarps;
    const auto run_gate = [&](const uint32_t& token_block_idx, const uint32_t& current_iter_idx) {
        const auto token_base_idx = token_block_idx * BLOCK_TOKENS;
        const auto num_valid_tokens = cute::min(BLOCK_TOKENS, num_tokens - token_block_idx * BLOCK_TOKENS);
        const auto effective_umma_n = get_effective_umma_n(token_block_idx);
        const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
        const auto accum_phase_idx = (current_iter_idx / kNumEpilogueStages) & 1;
        const auto score_barrier_ptr = workspace.get_score_barrier_ptr(token_block_idx);

        if (gate_warp_idx == kNumGateWarps - 1 and lane_idx == 31 and kNumLogicalCtas > 1) {
            if (logical_cta_rank == 0)
                sched::mega_gate::ScoreBarrier<kNumLogicalCtas>::init(score_barrier_ptr);
            else
                sched::mega_gate::ScoreBarrier<kNumLogicalCtas>::wait_init(score_barrier_ptr);
        }

        __syncwarp();
        smem.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
        ptx::tcgen05_after_thread_sync();

        const auto global_score_tile = workspace.get_score_ptr(token_block_idx);
        const auto split_score_tile = workspace.get_score_ptr(token_block_idx, split_k_idx);
        const auto expert_group_base_idx = expert_group_idx * kNumExpertsPerGroup;
        constexpr bool kHasScoringBias = kHasBias or kHasImageTokenMask;
        constexpr bool kScoreOnStore = kHasScoringBias and kNumSplitK == 1;
        auto gate_is_valid = epilogue::mega_gate::sm100_store_tmem_scores_to_global<
                                kScoreOnStore ? kScoringType : static_cast<uint32_t>(mega_gate_layout::ScoringType::Identity),
                                kNumRoutedExperts, UMMA_M, kNumMmaCtas, kNumGateWarps>(
                                    split_score_tile,
                                    get_tmem_base_addr(accum_stage_idx),
                                    effective_umma_n, expert_group_base_idx,
                                    mma_cta_rank, gate_warp_idx
                                );
        cutlass::arch::NamedBarrier::sync(kNumGateWarps * 32, 0);
        if (gate_warp_idx == kNumGateWarps - 1 and lane_idx == 31) {
            ptx::tcgen05_before_thread_sync();
            smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
            if constexpr (kNumLogicalCtas > 1)
                sched::mega_gate::ScoreBarrier<kNumLogicalCtas>::arrive(score_barrier_ptr);
        }
        if (gate_warp_idx == 0 and lane_idx == 0 and kNumLogicalCtas > 1)
            sched::mega_gate::ScoreBarrier<kNumLogicalCtas>::wait(score_barrier_ptr);
        cutlass::arch::NamedBarrier::sync(kNumGateWarps * 32, 0);
        const auto load_scores = [&](const uint32_t& token_idx_in_block, const uint32_t& global_expert_idx, auto& scores) {
            constexpr uint32_t kNumValues = sizeof(scores) / sizeof(float);
            using vec_t = cute::conditional_t<kNumValues == 1, float, float4>;
            const auto score_ptr = global_score_tile +
                static_cast<uint64_t>(token_idx_in_block) * kNumRoutedExperts + global_expert_idx;
            const auto vec_score_ptr = reinterpret_cast<const vec_t*>(score_ptr);
            scores = ptx::ld_global(vec_score_ptr);
            #pragma unroll
            for (uint32_t split_idx = 1; split_idx < kNumSplitK; ++ split_idx) {
                const auto partial_score_ptr = score_ptr + split_idx * UMMA_N * kNumRoutedExperts;
                const auto partial_scores = ptx::ld_global(reinterpret_cast<const vec_t*>(partial_score_ptr));
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
                    reinterpret_cast<float*>(&scores)[value_idx] += reinterpret_cast<const float*>(&partial_scores)[value_idx];
            }
        };
        const auto load_reduced_score = [&](const uint32_t& token_idx_in_block, const uint32_t& global_expert_idx) {
            auto score = 0.0f;
            load_scores(token_idx_in_block, global_expert_idx, score);
            return score;
        };
        for (uint32_t token_idx_in_block = gate_warp_idx * kNumLogicalCtas + logical_cta_rank;
             token_idx_in_block < num_valid_tokens;
             token_idx_in_block += kNumGateWarps * kNumLogicalCtas) {
            const auto global_token_idx = token_base_idx + token_idx_in_block;
            if (kMaskExists and not mask[global_token_idx]) {
                const auto num_physical_topk = kNumTopk + num_shared_experts;
                if (lane_idx < num_physical_topk) {
                    const auto output_idx = global_token_idx * num_physical_topk + lane_idx;
                    topk_idx[output_idx] = -1;
                    topk_weights[output_idx] = 0.0f;
                }
                if constexpr (kUnmappedTopkIdxExists) {
                    if (lane_idx < kNumTopk)
                        unmapped_topk_idx[global_token_idx * unmapped_topk_idx_stride + lane_idx] = -1;
                }
                continue;
            }
            if constexpr (kForceRandomExists) {
                if (force_random[global_token_idx]) {
                    epilogue::mega_gate::store_random_topk_token<kNumTopk, kUnmappedTopkIdxExists, kToPhysicalMapExists>(
                        global_token_idx, lane_idx,
                        num_routed_experts, num_shared_experts,
                        logical_count, ep_rank,
                        unmapped_topk_idx_stride, topk_idx,
                        unmapped_topk_idx, topk_weights
                    );
                    continue;
                }
            }
            if constexpr (kFixRoutingMaskExists) {
                if (fix_routing_mask[global_token_idx]) {
                    int selected_expert_idx = -1;
                    auto selected_unbiased_score = 0.0f;
                    if (lane_idx < kNumTopk) {
                        const auto fixed_expert_idx = unmapped_topk_idx[global_token_idx * unmapped_topk_idx_stride + lane_idx];
                        DG_TRAP_ONLY_DEVICE_ASSERT(fixed_expert_idx >= 0 and fixed_expert_idx < num_routed_experts);
                        selected_expert_idx = static_cast<int>(fixed_expert_idx);
                        selected_unbiased_score = load_reduced_score(token_idx_in_block, static_cast<uint32_t>(selected_expert_idx));
                        if constexpr (kNumSplitK > 1 or not kHasScoringBias)
                            selected_unbiased_score = epilogue::mega_gate::apply_scoring<kScoringType>(selected_unbiased_score);
                    }
                    epilogue::mega_gate::store_topk_token<
                        kNumTopk, kUnmappedTopkIdxExists, kToPhysicalMapExists>(global_token_idx, lane_idx,
                                                                               num_routed_experts,
                                                                               num_shared_experts,
                                                                               num_duplicate_experts,
                                                                               routed_scaling_factor, ep_rank,
                                                                               unmapped_topk_idx_stride,
                                                                               to_physical_map, logical_count,
                                                                               routed_logical_count, topk_idx,
                                                                               unmapped_topk_idx, topk_weights,
                                                                               selected_expert_idx,
                                                                               selected_unbiased_score);
                    continue;
                }
            }
            constexpr uint32_t kNumVectorElements = 4;
            constexpr uint32_t kNumExpertsPerWave = 32 * kNumVectorElements;
            constexpr uint32_t kNumExpertWaves = kNumRoutedExperts / kNumExpertsPerWave;
            constexpr uint32_t kNumExpertsPerLane = kNumRoutedExperts / 32;
            float unbiased_scores_local[kNumExpertsPerLane];
            float scores_local[kNumExpertsPerLane];
            const auto is_image_token = kHasImageTokenMask and image_token_mask[global_token_idx];
            #pragma unroll
            for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
                const auto global_expert_base_idx = expert_wave_idx * kNumExpertsPerWave +
                                                    lane_idx * kNumVectorElements;
                const auto local_expert_base_idx = expert_wave_idx * kNumVectorElements;
                auto& score_values = *reinterpret_cast<float4*>(scores_local + local_expert_base_idx);
                load_scores(token_idx_in_block, global_expert_base_idx, score_values);
            }
            #pragma unroll
            for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
                const auto global_expert_base_idx = expert_wave_idx * kNumExpertsPerWave + lane_idx * kNumVectorElements;
                const auto local_expert_base_idx = expert_wave_idx * kNumVectorElements;
                auto& score_values = *reinterpret_cast<float (*)[kNumVectorElements]>(scores_local + local_expert_base_idx);
                if constexpr (kHasScoringBias and kNumSplitK > 1)
                    epilogue::mega_gate::apply_scoring<kScoringType>(score_values);
                const float* bias_ptr = nullptr;
                if constexpr (kHasBias)
                    bias_ptr = smem_bias;
                if constexpr (kHasImageTokenMask)
                    bias_ptr = is_image_token ? smem_image_bias : bias_ptr;
                const auto bias_values = bias_ptr
                                         ? ptx::ld_shared(reinterpret_cast<const float4*>(bias_ptr + global_expert_base_idx))
                                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const auto bias_local = reinterpret_cast<const float*>(&bias_values);
                const auto is_valid_expert = kFullExpertTile or
                                             expert_wave_idx + 1 < kNumExpertWaves or
                                             global_expert_base_idx < num_routed_experts;
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < kNumVectorElements; ++ value_idx) {
                    const auto unbiased_score = score_values[value_idx];
                    if constexpr (kNumSplitK > 1)
                        unbiased_scores_local[local_expert_base_idx + value_idx] = unbiased_score;
                    scores_local[local_expert_base_idx + value_idx] = is_valid_expert
                                                                          ? unbiased_score + bias_local[value_idx]
                                                                          : -cute::numeric_limits<float>::infinity();
                }
            }

            int selected_expert_idx = -1;
            auto selected_unbiased_score = 0.0f;
            epilogue::mega_gate::select_warp_topk<kNumExpertWaves, kNumTopk>(scores_local, 0, selected_expert_idx);
            gate_is_valid = gate_is_valid and (lane_idx >= kNumTopk or
                (selected_expert_idx >= 0 and selected_expert_idx < static_cast<int>(num_routed_experts)));
            selected_expert_idx = lane_idx < kNumTopk
                ? cute::min(cute::max(selected_expert_idx, 0), static_cast<int>(num_routed_experts) - 1)
                : selected_expert_idx;
            if constexpr (kNumSplitK == 1) {
                if (lane_idx < kNumTopk)
                    selected_unbiased_score = load_reduced_score(token_idx_in_block, static_cast<uint32_t>(selected_expert_idx));
            } else {
                selected_unbiased_score = epilogue::mega_gate::select_warp_expert_value<kNumExpertWaves>(unbiased_scores_local, 0, selected_expert_idx);
            }
            if constexpr (not kHasScoringBias) {
                if (lane_idx < kNumTopk)
                    selected_unbiased_score = epilogue::mega_gate::apply_scoring<kScoringType>(selected_unbiased_score);
            }

            epilogue::mega_gate::store_topk_token<
                kNumTopk, kUnmappedTopkIdxExists, kToPhysicalMapExists>(global_token_idx, lane_idx,
                                                                       num_routed_experts, num_shared_experts,
                                                                       num_duplicate_experts, routed_scaling_factor,
                                                                       ep_rank, unmapped_topk_idx_stride,
                                                                       to_physical_map, logical_count,
                                                                       routed_logical_count, topk_idx,
                                                                       unmapped_topk_idx, topk_weights,
                                                                       selected_expert_idx,
                                                                       selected_unbiased_score);
        }

        DG_TRAP_ONLY_DEVICE_ASSERT(gate_is_valid and "Mega gate scores must be finite and top-k indices valid");
    };

    if (warp_idx == 0)
        scheduler.run(produce_tma);
    else if (warp_idx == 1 and is_leader_cta)
        scheduler.run(produce_mma);

    if (warp_idx >= kNumNonEpilogueWarps) {
        if constexpr (SharedStorage::kHasMetadataCache)
            smem.metadata_ready_barrier.wait(0);
        scheduler.run(run_gate);
    }

    comm::cluster_sync_with_relaxed_arrive();
    __syncwarp();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports SM100-family GPUs");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
