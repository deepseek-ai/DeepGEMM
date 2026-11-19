#pragma once

#include <cmath>
#include <cstdint>

#include <cute/atom/copy_traits_sm100.hpp>
#include <curand_kernel.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::epilogue::mega_gate {

using namespace layout::mega_gate;

CUTLASS_DEVICE float sqrt_softplus(const float& value) {
    const auto softplus_precise = log1pf(expf(value));
    const auto softplus = value > 20.0f ? value : softplus_precise;
    return sqrtf(softplus);
}

template <uint32_t kNumValues>
CUTLASS_DEVICE void sqrt_softplus(float (&values)[kNumValues]) {
    float exponentials[kNumValues], softplus[kNumValues];
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        exponentials[value_idx] = expf(values[value_idx]);
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        softplus[value_idx] = log1pf(exponentials[value_idx]);
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
        values[value_idx] = sqrtf(values[value_idx] > 20.0f ? values[value_idx] : softplus[value_idx]);
}

template <bool kScoreOnStore, uint32_t kScoreTokenStride,
          uint32_t UMMA_M, uint32_t kNumMmaCtas, uint32_t kNumGateWarps>
CUTLASS_DEVICE bool sm100_store_tmem_scores_to_global(float* scores, const uint32_t& tmem_base_addr,
                                                      const uint32_t& effective_umma_n,
                                                      const uint32_t& expert_base_idx,
                                                      const uint32_t& mma_cta_rank,
                                                      const uint32_t& gate_warp_idx) {
    // NOTES: with `UMMA_M == 128` on a CTA pair, the 64 experts of a CTA are spread over the two lane halves,
    //        so warps 0/1 and 2/3 of a warpgroup see the same experts for the two token halves
    constexpr uint32_t kNumTmemFragmentRows = 8;
    constexpr bool kPartitionedTokenPaths = UMMA_M == 128 and kNumMmaCtas == 2;
    constexpr uint32_t kNumExpertsPerCta = UMMA_M / kNumMmaCtas;
    const auto warp_idx_in_wg = gate_warp_idx % 4;
    const auto wg_idx = gate_warp_idx / 4;
    constexpr uint32_t kTokenStep = kNumGateWarps / 4 * kNumTmemFragmentRows;
    const auto num_token_cols = kPartitionedTokenPaths ? effective_umma_n / 2 : effective_umma_n;
    const auto token_base_idx = kPartitionedTokenPaths ? warp_idx_in_wg / 2 * num_token_cols : 0u;
    const auto expert_atom_idx = kPartitionedTokenPaths ? warp_idx_in_wg % 2 : warp_idx_in_wg;
    const auto global_expert_idx = expert_base_idx + mma_cta_rank * kNumExpertsPerCta +
                                   expert_atom_idx * 32 + ptx::get_lane_idx();

    // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits
    auto scores_are_finite = true;
    for (auto token_idx = wg_idx * kNumTmemFragmentRows; token_idx < num_token_cols; token_idx += kTokenStep) {
        uint32_t values[kNumTmemFragmentRows];
        ptx::tmem_load_32dp32b<kNumTmemFragmentRows>(tmem_base_addr + token_idx, values);
        cutlass::arch::fence_view_async_tmem_load();
        const auto row_base_ptr = scores + static_cast<uint64_t>(token_base_idx + token_idx) * kScoreTokenStride +
                                  global_expert_idx;
        #pragma unroll
        for (uint32_t row_idx = 0; row_idx < kNumTmemFragmentRows; ++ row_idx) {
            auto score = __uint_as_float(values[row_idx]);
            if constexpr (kScoreOnStore)
                score = sqrt_softplus(score);
            scores_are_finite = scores_are_finite and isfinite(score);
            row_base_ptr[row_idx * kScoreTokenStride] = score;
        }
    }
    return scores_are_finite;
}

CUTLASS_DEVICE void warp_reduce_best(const float& score, int& expert_idx) {
    const auto best_score = ptx::reduce_max_sync(score);
    // Ties resolve to the smallest expert index
    const auto tied_expert_idx = score == best_score and expert_idx >= 0 ?
        static_cast<uint32_t>(expert_idx) : 0xffffffffu;
    expert_idx = static_cast<int>(__reduce_min_sync(0xffffffff, tied_expert_idx));
}

CUTLASS_DEVICE void compare_swap_best_first(float& lhs_score, int& lhs_expert_idx, float& rhs_score, int& rhs_expert_idx) {
    if (rhs_score > lhs_score or (rhs_score == lhs_score and rhs_expert_idx < lhs_expert_idx)) {
        const auto score = lhs_score;
        const auto expert_idx = lhs_expert_idx;
        lhs_score = rhs_score, lhs_expert_idx = rhs_expert_idx;
        rhs_score = score, rhs_expert_idx = expert_idx;
    }
}

CUTLASS_DEVICE uint32_t sort4_best_first(float& score_0, float& score_1, float& score_2, float& score_3) {
    int idx_0 = 0, idx_1 = 1, idx_2 = 2, idx_3 = 3;
    compare_swap_best_first(score_0, idx_0, score_1, idx_1);
    compare_swap_best_first(score_2, idx_2, score_3, idx_3);
    compare_swap_best_first(score_0, idx_0, score_2, idx_2);
    compare_swap_best_first(score_1, idx_1, score_3, idx_3);
    compare_swap_best_first(score_1, idx_1, score_2, idx_2);
    return static_cast<uint32_t>(idx_0 | idx_1 << 2 | idx_2 << 4 | idx_3 << 6);
}

template <uint32_t kNumExpertWaves, uint32_t kNumTopk>
CUTLASS_DEVICE void select_warp_topk(float (&scores)[kNumExpertWaves * kNumExpertsPerLaneVector], int& selected_expert_idx) {
    DG_STATIC_ASSERT(kNumExpertsPerLaneVector == 4, "The lane vector is sorted with a 4-element network");
    uint32_t permutations[kNumExpertWaves];
    #pragma unroll
    for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
        const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
        permutations[expert_wave_idx] = sort4_best_first(scores[local_expert_base_idx + 0],
                                                         scores[local_expert_base_idx + 1],
                                                         scores[local_expert_base_idx + 2],
                                                         scores[local_expert_base_idx + 3]);
    }

    // Every round, each lane offers the best unconsumed candidate of each wave (a 3-bit cursor into the
    // 2-bit permutation), the warp picks the best one and only the winning lane advances that cursor
    const auto lane_idx = ptx::get_lane_idx();
    uint32_t packed_cursors = 0;
    #pragma unroll
    for (uint32_t output_idx = 0; output_idx < kNumTopk; ++ output_idx) {
        auto best_score = -cute::numeric_limits<float>::infinity();
        int best_expert_idx = -1;
        uint32_t best_expert_wave_idx = 0;
        #pragma unroll
        for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
            const auto cursor = packed_cursors >> (expert_wave_idx * 3) & 7;
            const auto local_expert_base_idx = expert_wave_idx * kNumExpertsPerLaneVector;
            const auto candidate_score = cursor == 4 ? -cute::numeric_limits<float>::infinity() :
                                         cursor & 2 ? (cursor & 1 ? scores[local_expert_base_idx + 3] : scores[local_expert_base_idx + 2]) :
                                                      (cursor & 1 ? scores[local_expert_base_idx + 1] : scores[local_expert_base_idx + 0]);
            const auto candidate_offset = permutations[expert_wave_idx] >> (cursor * 2) & 3;
            const auto candidate_expert_idx = static_cast<int>(expert_wave_idx * kNumExpertsPerWave +
                                                               lane_idx * kNumExpertsPerLaneVector + candidate_offset);
            if (candidate_score > best_score) {
                best_score = candidate_score;
                best_expert_idx = candidate_expert_idx;
                best_expert_wave_idx = expert_wave_idx;
            }
        }
        const auto local_best_expert_idx = best_expert_idx;
        warp_reduce_best(best_score, best_expert_idx);
        if (local_best_expert_idx == best_expert_idx)
            packed_cursors += 1u << (best_expert_wave_idx * 3);
        if (lane_idx == output_idx)
            selected_expert_idx = best_expert_idx;
    }
}

template <uint32_t kNumExpertWaves>
CUTLASS_DEVICE float select_warp_expert_value(const float (&values)[kNumExpertWaves * kNumExpertsPerLaneVector],
                                              const int& expert_idx) {
    const auto selected_value_idx = expert_idx >= 0 ?
        expert_idx / static_cast<int>(kNumExpertsPerWave) * static_cast<int>(kNumExpertsPerLaneVector) +
        expert_idx % static_cast<int>(kNumExpertsPerLaneVector) : -1;
    const auto source_lane_idx = expert_idx >= 0 ?
        static_cast<uint32_t>(expert_idx % static_cast<int>(kNumExpertsPerWave) / static_cast<int>(kNumExpertsPerLaneVector)) : 0u;
    auto selected_value = 0.0f;
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumExpertWaves * kNumExpertsPerLaneVector; ++ value_idx) {
        const auto exchanged = ptx::exchange(values[value_idx], source_lane_idx);
        if (selected_value_idx == static_cast<int>(value_idx))
            selected_value = exchanged;
    }
    return selected_value;
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx>
CUTLASS_DEVICE void store_token_outputs(const RoutingArgs& args, const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                        const int64_t& physical_expert_idx, const float& weight, const int64_t& logical_expert_idx) {
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    if (lane_idx < num_physical_topk) {
        const auto output_idx = global_token_idx * num_physical_topk + lane_idx;
        args.topk_idx[output_idx] = physical_expert_idx;
        args.topk_weights[output_idx] = weight;
    }
    if constexpr (kHasUnmappedTopkIdx) {
        if (lane_idx < kNumTopk)
            args.unmapped_topk_idx[global_token_idx * args.unmapped_topk_idx_stride + lane_idx] = logical_expert_idx;
    }
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx, bool kHasPhysicalMap>
CUTLASS_DEVICE void store_topk_token(const RoutingArgs& args, const int* routed_logical_count,
                                     const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                     int selected_expert_idx, const float& selected_unbiased_score) {
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k slots exceed a warp");
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    // The shared experts follow the routed ones and take the slots after the top-k
    if (lane_idx >= kNumTopk and lane_idx < num_physical_topk)
        selected_expert_idx = static_cast<int>(lane_idx + args.num_routed_experts - kNumTopk);

    auto physical_expert_idx = selected_expert_idx;
    if constexpr (kHasPhysicalMap) {
        if (lane_idx < num_physical_topk) {
            const auto logical_expert_idx = static_cast<uint32_t>(selected_expert_idx);
            // The SMEM cache only covers the routed experts
            const auto num_duplicates = static_cast<uint32_t>((lane_idx < kNumTopk ? routed_logical_count : args.logical_count)[logical_expert_idx]);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_duplicates > 0 and num_duplicates <= args.num_duplicate_experts);
            // Spread the tokens of a logical expert over its duplicates, differently on every rank
            const auto duplicate_idx = (args.rank_idx + global_token_idx * 23333u) % num_duplicates;
            physical_expert_idx = args.to_physical_map[logical_expert_idx * args.num_duplicate_experts + duplicate_idx];
        }
    }

    // The smallest shuffle width covering the routed lanes, the other lanes hold zero
    constexpr uint32_t kSumWidth = kNumTopk <= 8 ? 8 : (kNumTopk <= 16 ? 16 : 32);
    const auto topk_sum = math::warp_reduce_sum<kSumWidth>(selected_unbiased_score) + 1e-20f;
    const auto selected_weight = lane_idx < kNumTopk ? selected_unbiased_score / topk_sum * args.routed_scaling_factor : 1.0f;
    store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(args, global_token_idx, lane_idx,
                                                       physical_expert_idx, selected_weight, selected_expert_idx);
}

template <uint32_t kNumTopk, bool kHasUnmappedTopkIdx, bool kHasPhysicalMap>
CUTLASS_DEVICE void store_random_topk_token(const RoutingArgs& args, const uint32_t& global_token_idx, const uint32_t& lane_idx) {
    const auto num_physical_topk = kNumTopk + args.num_shared_experts;
    auto num_physical_experts = args.num_routed_experts + args.num_shared_experts;
    if constexpr (kHasPhysicalMap) {
        auto local_count = 0u;
        for (uint32_t logical_expert_idx = lane_idx; logical_expert_idx < num_physical_experts; logical_expert_idx += 32)
            local_count += static_cast<uint32_t>(args.logical_count[logical_expert_idx]);
        num_physical_experts = __reduce_add_sync(0xffffffff, local_count);
    }
    DG_TRAP_ONLY_DEVICE_ASSERT(num_physical_experts >= num_physical_topk);

    // Sample `num_physical_topk` distinct experts: lane `i` draws from `n - i`, then ranks below every earlier draw
    // that is not larger shift up by one
    curandStatePhilox4_32_10_t rng_state;
    curand_init(args.rank_idx, global_token_idx * 32 + lane_idx, 0, &rng_state);
    const auto random_weight = curand_uniform(&rng_state);
    constexpr uint32_t kInvalidIdx = 0xffffffffu;
    auto candidate_idx = lane_idx < num_physical_topk ?
        curand(&rng_state) % (num_physical_experts - lane_idx) : kInvalidIdx;
    auto selected_expert_idx = kInvalidIdx;
    for (uint32_t output_idx = 0; output_idx < num_physical_topk; ++ output_idx) {
        const auto min_candidate_idx = __reduce_min_sync(0xffffffff, candidate_idx);
        const auto min_lane_idx = __reduce_min_sync(0xffffffff, candidate_idx == min_candidate_idx ? lane_idx : kInvalidIdx);
        if (candidate_idx != kInvalidIdx) {
            if (min_lane_idx == lane_idx) {
                selected_expert_idx = candidate_idx;
                candidate_idx = kInvalidIdx;
            } else if (candidate_idx >= min_candidate_idx and min_lane_idx < lane_idx) {
                ++ candidate_idx;
            }
        }
    }
    store_token_outputs<kNumTopk, kHasUnmappedTopkIdx>(args, global_token_idx, lane_idx,
                                                       static_cast<int64_t>(selected_expert_idx), random_weight, -1);
}

} // namespace deep_gemm::epilogue::mega_gate
