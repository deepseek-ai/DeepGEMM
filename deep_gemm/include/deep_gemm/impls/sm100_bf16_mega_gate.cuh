#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>

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

template <uint32_t SHAPE_K, uint32_t kNumRoutedExperts,
          uint32_t kNumGateThreads,
          uint32_t kNumMmaCtas, uint32_t kNumSplitK, uint32_t kNumExpertGroups,
          uint32_t kNumTopk,
          bool kHasPaddedExperts, bool kHasBias, bool kHasImageTokenMask,
          bool kHasMask, bool kHasFixRoutingMask, bool kHasForceRandom,
          bool kHasPhysicalMap, bool kHasUnmappedTopkIdx>
CUTLASS_GLOBAL void __launch_bounds__(mega_gate_layout::kNumNonEpilogueThreads + kNumGateThreads, 1)
sm100_bf16_mega_gate_impl(const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weight,
                          const float* bias, const float* image_bias, const bool* image_token_mask,
                          const bool* mask, const bool* fix_routing_mask, const bool* force_random,
                          const __grid_constant__ mega_gate_layout::RoutingArgs routing_args,
                          void* gmem_scratch, void* gmem_score_barriers,
                          uint32_t num_tokens, uint32_t block_tokens, uint32_t num_stages,
                          uint32_t num_worker_groups, uint32_t num_token_blocks) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Allocator = cute::conditional_t<kNumMmaCtas == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    constexpr uint32_t kNumNonEpilogueWarps = mega_gate_layout::kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumGateWarps = kNumGateThreads / 32;
    DG_STATIC_ASSERT(kNumGateThreads % 128 == 0, "Gate warps must be whole warpgroups");
    constexpr uint32_t BLOCK_K = mega_gate_layout::BLOCK_K;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t kSwizzleMode = mega_gate_layout::kSwizzleMode;
    constexpr uint32_t kNumExpertsPerGroup = kNumRoutedExperts / kNumExpertGroups;

    // NOTES: the SMEM/TMEM capacity is `kNumMaxBlockTokens`, the actual UMMA N (`block_tokens`) is a runtime value
    constexpr uint32_t UMMA_M = kNumExpertsPerGroup;
    constexpr uint32_t UMMA_N = mega_gate_layout::kNumMaxBlockTokens;
    constexpr uint32_t kNumLogicalCtas = kNumMmaCtas * kNumExpertGroups * kNumSplitK;
    using ScoreBarrier = sched::mega_gate::ScoreBarrier<kNumLogicalCtas>;
    constexpr uint32_t LOAD_BLOCK_M = UMMA_N / kNumMmaCtas;
    constexpr uint32_t LOAD_BLOCK_N = UMMA_M / kNumMmaCtas;
    DG_STATIC_ASSERT((kNumMmaCtas == 1 and UMMA_M == 128) or (kNumMmaCtas == 2 and (UMMA_M == 128 or UMMA_M == 256)),
                     "Invalid MMA instruction shape");
    constexpr uint32_t kNumEpilogueStages = mega_gate_layout::kNumEpilogueStages;
    using SharedStorage = mega_gate_layout::SharedStorage;
    constexpr bool kCacheLogicalCount = mega_gate_layout::caches_logical_count(kNumRoutedExperts, kHasBias, kHasImageTokenMask, kHasPhysicalMap);
    constexpr bool kHasMetadataCache = kHasBias or kHasImageTokenMask or kCacheLogicalCount;

    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumEpilogueStages * UMMA_N>();

    constexpr uint32_t kNumWeightStageBytes = LOAD_BLOCK_N * BLOCK_K * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t));

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
    const auto smem_bias = smem.metadata_cache;
    const auto smem_image_bias = smem_bias + (kHasBias ? kNumRoutedExperts : 0);
    const auto smem_logical_count = reinterpret_cast<int*>(smem_image_bias + (kHasImageTokenMask ? kNumRoutedExperts : 0));
    const auto routed_logical_count = kCacheLogicalCount ? smem_logical_count : routing_args.logical_count;

    if constexpr (kNumMmaCtas > 1)
        comm::cluster_sync_with_relaxed_arrive();

    // Initialize barriers
    if (warp_idx < 4 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = warp_idx; i < mega_gate_layout::kNumMaxStages; i += 4) {
            smem.full_barriers[i].init(kNumMmaCtas);
            smem.empty_barriers[i].init(1);
        }
        if (warp_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                smem.tmem_full_barriers[i].init(1);
                smem.tmem_empty_barriers[i].init(kNumMmaCtas);
            }
        }

        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == 2)
        Allocator().allocate(kNumTmemCols, &smem.tmem_ptr);
    kNumMmaCtas > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    cudaGridDependencySynchronize();

    using Workspace = mega_gate_layout::Workspace<kNumSplitK, kNumRoutedExperts>;
    const Workspace workspace(gmem_scratch, gmem_score_barriers);
    const auto get_effective_umma_n = [&](const uint32_t& token_block_idx) {
        return math::align(cute::min(block_tokens, num_tokens - token_block_idx * block_tokens), 16u);
    };
    // Persistent schedule: the `kNumLogicalCtas` CTAs of a worker group stride over the token blocks together
    const auto worker_group_idx = blockIdx.x / kNumLogicalCtas;

    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);
    DG_TRAP_ONLY_DEVICE_ASSERT(num_stages <= mega_gate_layout::kNumMaxStages);

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        const auto num_x_stage_bytes = (block_tokens / kNumMmaCtas) * BLOCK_K * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t));
        // A CTA pair splits both the experts (UMMA M) and the tokens (UMMA N) between its two CTAs
        const auto expert_base_idx = expert_group_idx * kNumExpertsPerGroup + mma_cta_rank * LOAD_BLOCK_N;
        uint32_t stage_idx = 0, phase = 0;
        for (auto token_block_idx = worker_group_idx; token_block_idx < num_token_blocks; token_block_idx += num_worker_groups) {
            const auto effective_umma_n = get_effective_umma_n(token_block_idx);
            const auto x_token_idx = token_block_idx * block_tokens + mma_cta_rank * (effective_umma_n / kNumMmaCtas);
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocksPerSlice; ++ k_block_idx) {
                smem.empty_barriers[stage_idx].wait(phase ^ 1);

                // Issue TMAs
                const auto k_idx = split_k_offset + k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleMode, cutlass::bfloat16_t, false,
                          static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST)>(
                    &tensor_map_x, &smem.full_barriers[stage_idx],
                    mega_gate_layout::get_x_stage_ptr(smem_buffer, stage_idx, num_x_stage_bytes),
                    k_idx, x_token_idx, kNumMmaCtas);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleMode, cutlass::bfloat16_t>(
                    &tensor_map_weight, &smem.full_barriers[stage_idx],
                    mega_gate_layout::get_weight_stage_ptr(smem_buffer, stage_idx, num_stages, num_x_stage_bytes, kNumWeightStageBytes),
                    k_idx, expert_base_idx, kNumMmaCtas);

                if (is_leader_cta) {
                    smem.full_barriers[stage_idx].arrive_and_expect_tx((num_x_stage_bytes + kNumWeightStageBytes) * kNumMmaCtas);
                } else {
                    smem.full_barriers[stage_idx].arrive(0u);
                }
                if (++ stage_idx == num_stages)
                    stage_idx = 0, phase ^= 1;
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        const auto num_x_stage_bytes = (block_tokens / kNumMmaCtas) * BLOCK_K * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t));
        auto x_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleMode>(
            mega_gate_layout::get_x_stage_ptr(smem_buffer, 0, num_x_stage_bytes), 0, 0);
        auto weight_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleMode>(
            mega_gate_layout::get_weight_stage_ptr(smem_buffer, 0, num_stages, num_x_stage_bytes, kNumWeightStageBytes), 0, 0);

        const auto x_desc_lo = x_desc.lo, weight_desc_lo = weight_desc.lo;
        auto x_desc_base_lo = x_desc_lo, weight_desc_base_lo = weight_desc_lo;

        uint32_t stage_idx = 0, phase = 0, iter_idx = 0;
        for (auto token_block_idx = worker_group_idx; token_block_idx < num_token_blocks;
             token_block_idx += num_worker_groups, ++ iter_idx) {
            mma::sm100::update_instr_desc_with_umma_n(instr_desc, get_effective_umma_n(token_block_idx));

            const auto accum_stage_idx = iter_idx % kNumEpilogueStages;
            const auto accum_phase_idx = iter_idx / kNumEpilogueStages & 1;
            smem.tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMmaCtas == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMmaCtas) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_barriers[stage_idx]));

                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocksPerSlice; ++ k_block_idx) {
                smem.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

                // Issue UMMA
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        using mma_t = cute::conditional_t<kNumMmaCtas == 1,
                                                          ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                        x_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleMode, cutlass::bfloat16_t>(
                                        x_desc_base_lo, 0, umma_k_idx * UMMA_K);
                        weight_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleMode, cutlass::bfloat16_t>(
                                            weight_desc_base_lo, 0, umma_k_idx * UMMA_K);
                        // NOTES: the weights are the A operand, so a TMEM lane holds one expert and a column one token
                        mma_t::fma(weight_desc, x_desc, accum_stage_idx * UMMA_N,
                                   umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc);
                    }
                }
                __syncwarp();

                empty_barrier_arrive(k_block_idx == kNumKBlocksPerSlice - 1);
                if (++ stage_idx == num_stages) {
                    stage_idx = 0, phase ^= 1;
                    x_desc_base_lo = x_desc_lo, weight_desc_base_lo = weight_desc_lo;
                } else {
                    x_desc_base_lo += num_x_stage_bytes / 16, weight_desc_base_lo += kNumWeightStageBytes / 16;
                }
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        // The follower's remote arrivals are not ordered by the relaxed tail cluster sync
        if constexpr (kNumMmaCtas > 1) {
            const auto last_iter_idx = iter_idx - 1;
            smem.tmem_empty_barriers[last_iter_idx % kNumEpilogueStages].wait(last_iter_idx / kNumEpilogueStages & 1);
        }
    } else if (warp_idx >= kNumNonEpilogueWarps) {
        // Gate warps
        const auto gate_warp_idx = warp_idx - kNumNonEpilogueWarps;
        if constexpr (kHasMetadataCache) {
            const auto gate_thread_idx = threadIdx.x - mega_gate_layout::kNumNonEpilogueThreads;
            #pragma unroll
            for (uint32_t cache_idx = gate_thread_idx; cache_idx < kNumRoutedExperts; cache_idx += kNumGateThreads) {
                const auto is_valid_expert = not kHasPaddedExperts or cache_idx < routing_args.num_routed_experts;
                if constexpr (kHasBias)
                    smem_bias[cache_idx] = is_valid_expert ? bias[cache_idx] : 0.0f;
                if constexpr (kHasImageTokenMask)
                    smem_image_bias[cache_idx] = is_valid_expert ? image_bias[cache_idx] : 0.0f;
                if constexpr (kCacheLogicalCount)
                    smem_logical_count[cache_idx] = is_valid_expert ? routing_args.logical_count[cache_idx] : 0;
            }
        }
        for (auto token_block_idx = worker_group_idx, iter_idx = 0u; token_block_idx < num_token_blocks;
             token_block_idx += num_worker_groups, ++ iter_idx) {
            const auto token_base_idx = token_block_idx * block_tokens;
            const auto token_end_idx = cute::min(token_base_idx + block_tokens, num_tokens);
            const auto effective_umma_n = get_effective_umma_n(token_block_idx);
            const auto accum_stage_idx = iter_idx % kNumEpilogueStages;
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            const auto score_barrier_ptr = workspace.get_score_barrier_ptr(token_block_idx);

            // The first logical CTA re-arms the score barrier of this block for the current grid
            if (gate_warp_idx == kNumGateWarps - 1 and kNumLogicalCtas > 1 and cute::elect_one_sync()) {
                if (logical_cta_rank == 0)
                    ScoreBarrier::init(score_barrier_ptr);
                else
                    ScoreBarrier::wait_init(score_barrier_ptr);
            }

            __syncwarp();
            smem.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto expert_group_base_idx = expert_group_idx * kNumExpertsPerGroup;
            constexpr bool kHasScoringBias = kHasBias or kHasImageTokenMask;
            // sqrt-softplus is not linear: with split-K it is applied after summing the partials in the top-k loop,
            // with a single split it is applied on store and the selected score is re-read
            constexpr bool kScoreOnStore = kHasScoringBias and kNumSplitK == 1;
            auto gate_is_valid = epilogue::mega_gate::sm100_store_tmem_scores_to_global<
                                    kScoreOnStore, Workspace::kTokenStride, UMMA_M, kNumMmaCtas, kNumGateWarps>(
                                        workspace.get_score_ptr(token_base_idx, split_k_idx),
                                        accum_stage_idx * UMMA_N,
                                        effective_umma_n, expert_group_base_idx,
                                        mma_cta_rank, gate_warp_idx
                                    );
            cutlass::arch::NamedBarrier::sync(kNumGateThreads, 0);
            if (gate_warp_idx == kNumGateWarps - 1 and cute::elect_one_sync()) {
                ptx::tcgen05_before_thread_sync();
                smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                if constexpr (kNumLogicalCtas > 1)
                    ScoreBarrier::arrive(score_barrier_ptr);
            }
            if (gate_warp_idx == 0 and kNumLogicalCtas > 1 and cute::elect_one_sync())
                ScoreBarrier::wait(score_barrier_ptr);
            cutlass::arch::NamedBarrier::sync(kNumGateThreads, 0);
            const auto load_scores = [&](const uint32_t& global_token_idx, const uint32_t& global_expert_idx, auto& scores) {
                constexpr uint32_t kNumValues = sizeof(scores) / sizeof(float);
                using vec_t = cute::conditional_t<kNumValues == 1, float, float4>;
                const auto score_ptr = workspace.get_score_ptr(global_token_idx) + global_expert_idx;
                scores = *reinterpret_cast<const vec_t*>(score_ptr);
                #pragma unroll
                for (uint32_t split_idx = 1; split_idx < kNumSplitK; ++ split_idx) {
                    const auto partial_scores = *reinterpret_cast<const vec_t*>(score_ptr + split_idx * kNumRoutedExperts);
                    #pragma unroll
                    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
                        reinterpret_cast<float*>(&scores)[value_idx] += reinterpret_cast<const float*>(&partial_scores)[value_idx];
                }
            };
            const auto load_reduced_score = [&](const uint32_t& global_token_idx, const uint32_t& global_expert_idx) {
                auto score = 0.0f;
                load_scores(global_token_idx, global_expert_idx, score);
                return score;
            };
            // Every logical CTA of the block ranks a share of its tokens once all the splits are visible
            for (uint32_t global_token_idx = token_base_idx + gate_warp_idx * kNumLogicalCtas + logical_cta_rank;
                 global_token_idx < token_end_idx;
                 global_token_idx += kNumGateWarps * kNumLogicalCtas) {
                const auto is_masked = kHasMask and not mask[global_token_idx];
                const auto is_force_random = kHasForceRandom and force_random[global_token_idx];
                const auto has_fixed_routing = kHasFixRoutingMask and fix_routing_mask[global_token_idx];
                const auto is_image_token = kHasImageTokenMask and image_token_mask[global_token_idx];

                if (is_masked) {
                    epilogue::mega_gate::store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(
                        routing_args, global_token_idx, lane_idx, -1, 0.0f, -1);
                    continue;
                }
                if (is_force_random) {
                    epilogue::mega_gate::store_random_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(
                        routing_args, global_token_idx, lane_idx);
                    continue;
                }
                if constexpr (kHasFixRoutingMask) {
                    if (has_fixed_routing) {
                        int selected_expert_idx = -1;
                        auto selected_unbiased_score = 0.0f;
                        if (lane_idx < kNumTopk) {
                            const auto fixed_expert_idx = routing_args.unmapped_topk_idx[
                                global_token_idx * routing_args.unmapped_topk_idx_stride + lane_idx];
                            DG_TRAP_ONLY_DEVICE_ASSERT(fixed_expert_idx >= 0 and fixed_expert_idx < routing_args.num_routed_experts);
                            selected_expert_idx = static_cast<int>(fixed_expert_idx);
                            selected_unbiased_score = load_reduced_score(global_token_idx, static_cast<uint32_t>(selected_expert_idx));
                            if constexpr (not kScoreOnStore)
                                selected_unbiased_score = epilogue::mega_gate::sqrt_softplus(selected_unbiased_score);
                        }
                        epilogue::mega_gate::store_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(
                            routing_args, routed_logical_count, global_token_idx, lane_idx, selected_expert_idx, selected_unbiased_score);
                        continue;
                    }
                }
                constexpr uint32_t kNumExpertsPerLaneVector = mega_gate_layout::kNumExpertsPerLaneVector;
                constexpr uint32_t kNumExpertsPerWave = mega_gate_layout::kNumExpertsPerWave;
                constexpr uint32_t kNumExpertWaves = kNumRoutedExperts / kNumExpertsPerWave;
                float unbiased_scores_local[kNumExpertWaves * kNumExpertsPerLaneVector];
                float scores_local[kNumExpertWaves * kNumExpertsPerLaneVector];
                #pragma unroll
                for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
                    const auto global_expert_base_idx = expert_wave_idx * kNumExpertsPerWave +
                                                        lane_idx * kNumExpertsPerLaneVector;
                    const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
                    auto& score_values = *reinterpret_cast<float4*>(scores_local + local_expert_base_idx);
                    load_scores(global_token_idx, global_expert_base_idx, score_values);
                }
                #pragma unroll
                for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
                    const auto global_expert_base_idx = expert_wave_idx * kNumExpertsPerWave + lane_idx * kNumExpertsPerLaneVector;
                    const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
                    auto& score_values = *reinterpret_cast<float (*)[kNumExpertsPerLaneVector]>(scores_local + local_expert_base_idx);
                    if constexpr (kHasScoringBias and kNumSplitK > 1)
                        epilogue::mega_gate::sqrt_softplus(score_values);
                    const float* bias_ptr = nullptr;
                    if constexpr (kHasBias)
                        bias_ptr = smem_bias;
                    if constexpr (kHasImageTokenMask)
                        bias_ptr = is_image_token ? smem_image_bias : bias_ptr;
                    const auto bias_values = bias_ptr ?
                        ptx::ld_shared(reinterpret_cast<const float4*>(bias_ptr + global_expert_base_idx)) :
                        make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    const auto bias_local = reinterpret_cast<const float*>(&bias_values);
                    // Only the last wave can hold padded experts
                    const auto is_valid_expert = not kHasPaddedExperts or
                                                 expert_wave_idx + 1 < kNumExpertWaves or
                                                 global_expert_base_idx < routing_args.num_routed_experts;
                    #pragma unroll
                    for (uint32_t value_idx = 0; value_idx < kNumExpertsPerLaneVector; ++ value_idx) {
                        const auto unbiased_score = score_values[value_idx];
                        if constexpr (kNumSplitK > 1)
                            unbiased_scores_local[local_expert_base_idx + value_idx] = unbiased_score;
                        scores_local[local_expert_base_idx + value_idx] = is_valid_expert ?
                            unbiased_score + bias_local[value_idx] : -cute::numeric_limits<float>::infinity();
                    }
                }

                int selected_expert_idx = -1;
                auto selected_unbiased_score = 0.0f;
                epilogue::mega_gate::select_warp_topk<kNumExpertWaves, kNumTopk>(scores_local, selected_expert_idx);
                gate_is_valid = gate_is_valid and (lane_idx >= kNumTopk or
                                                   (selected_expert_idx >= 0 and
                                                    selected_expert_idx < static_cast<int>(routing_args.num_routed_experts)));
                // Keep the re-read in bounds for an invalid selection, the trap assert below reports it
                selected_expert_idx = lane_idx < kNumTopk ?
                    cute::min(cute::max(selected_expert_idx, 0), static_cast<int>(routing_args.num_routed_experts) - 1) :
                    selected_expert_idx;
                if constexpr (kNumSplitK == 1) {
                    if (lane_idx < kNumTopk)
                        selected_unbiased_score = load_reduced_score(global_token_idx, static_cast<uint32_t>(selected_expert_idx));
                    __syncwarp();
                } else {
                    selected_unbiased_score = epilogue::mega_gate::select_warp_expert_value<kNumExpertWaves>(
                        unbiased_scores_local, selected_expert_idx);
                }
                // Without a bias the ranking is on raw scores (sqrt-softplus is monotonic), only the selected ones get activated
                if constexpr (not kHasScoringBias) {
                    if (lane_idx < kNumTopk)
                        selected_unbiased_score = epilogue::mega_gate::sqrt_softplus(selected_unbiased_score);
                }

                epilogue::mega_gate::store_topk_token<kNumTopk, kHasUnmappedTopkIdx, kHasPhysicalMap>(
                    routing_args, routed_logical_count, global_token_idx, lane_idx, selected_expert_idx, selected_unbiased_score);
            }

            DG_TRAP_ONLY_DEVICE_ASSERT(gate_is_valid);
        }
    }

    // TODO: Remove redundant synchronization
    kNumMmaCtas > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
