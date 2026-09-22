#pragma once

#include <cmath>
#include <cstdint>

#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::epilogue::mega_gate {

template <uint32_t kScoringType> CUTLASS_DEVICE float apply_scoring(const float& value) {
    using ScoringType = layout::mega_gate::ScoringType;
    if constexpr (kScoringType == static_cast<uint32_t>(ScoringType::Sigmoid)) {
        return 1.0f / (1.0f + expf(-value));
    } else if constexpr (kScoringType == static_cast<uint32_t>(ScoringType::SqrtSoftplus)) {
        const auto softplus_precise = log1pf(expf(value));
        const auto softplus = value > 20.0f ? value : softplus_precise;
        return sqrtf(softplus);
    } else {
        return value;
    }
}

template <uint32_t kScoringType, uint32_t kNumValues>
CUTLASS_DEVICE void apply_scoring(float (&values)[kNumValues]) {
    using ScoringType = layout::mega_gate::ScoringType;
    if constexpr (kScoringType == static_cast<uint32_t>(ScoringType::Sigmoid)) {
        float exponentials[kNumValues];
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
            exponentials[value_idx] = expf(-values[value_idx]);
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
            values[value_idx] = 1.0f / (1.0f + exponentials[value_idx]);
    } else if constexpr (kScoringType == static_cast<uint32_t>(ScoringType::SqrtSoftplus)) {
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
}

template <uint32_t kStoreScoringType, uint32_t kNumRoutedExperts,
          uint32_t UMMA_M, uint32_t kNumMmaCtas, uint32_t kNumGateWarps>
CUTLASS_DEVICE bool sm100_store_tmem_scores_to_global(float* scores, const uint32_t& tmem_base_addr,
                                                      const uint32_t& effective_umma_n,
                                                      const uint32_t& expert_base_idx,
                                                      const uint32_t& mma_cta_rank,
                                                      const uint32_t& gate_warp_idx) {
    constexpr uint32_t kNumTmemFragmentRows = 8;
    constexpr bool kPartitionedTokenPaths = UMMA_M == 128 and kNumMmaCtas == 2;
    constexpr uint32_t kNumExpertsPerCta = UMMA_M / kNumMmaCtas;
    const auto subpartition_idx = gate_warp_idx % 4;
    const auto warp_idx_in_subpartition = gate_warp_idx / 4;
    constexpr uint32_t kTokenStride = kNumGateWarps / 4 * kNumTmemFragmentRows;
    const auto num_token_cols = kPartitionedTokenPaths ? effective_umma_n / 2 : effective_umma_n;
    const auto token_base_idx = kPartitionedTokenPaths ? subpartition_idx / 2 * num_token_cols : 0u;
    const auto expert_atom_idx = kPartitionedTokenPaths ? subpartition_idx % 2 : subpartition_idx;
    const auto global_expert_idx = expert_base_idx + mma_cta_rank * kNumExpertsPerCta +
                                   expert_atom_idx * 32 + ptx::get_lane_idx();
    const auto tmem_addr = tmem_base_addr + (subpartition_idx * 32 << 16);

    auto scores_are_finite = true;
    for (auto token_idx = warp_idx_in_subpartition * kNumTmemFragmentRows;
         token_idx < num_token_cols; token_idx += kTokenStride) {
        uint32_t values[kNumTmemFragmentRows];
        cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr + token_idx,
                                              values[0], values[1], values[2], values[3],
                                              values[4], values[5], values[6], values[7]);
        cutlass::arch::fence_view_async_tmem_load();
        #pragma unroll
        for (uint32_t row_idx = 0; row_idx < kNumTmemFragmentRows; ++ row_idx) {
            const auto dst_token_idx = token_base_idx + token_idx + row_idx;
            const auto score = apply_scoring<kStoreScoringType>(__uint_as_float(values[row_idx]));
            scores_are_finite = scores_are_finite and isfinite(score);
            ptx::st_global(scores + static_cast<uint64_t>(dst_token_idx) * kNumRoutedExperts +
                           global_expert_idx, score);
        }
    }
    return scores_are_finite;
}

CUTLASS_DEVICE void warp_reduce_best(const float& score, int& expert_idx) {
    const auto best_score = ptx::reduce_max_sync(score);
    const auto tied_expert_idx = score == best_score and expert_idx >= 0
                                     ? static_cast<uint32_t>(expert_idx) : UINT32_MAX;
    expert_idx = static_cast<int>(__reduce_min_sync(0xffffffffu, tied_expert_idx));
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
CUTLASS_DEVICE void select_warp_topk(float (&scores)[kNumExpertWaves * 4], const int& expert_base_idx,
                                     int& selected_expert_idx) {
    uint32_t permutations[kNumExpertWaves];
    #pragma unroll
    for (uint32_t expert_wave_idx = 0; expert_wave_idx < kNumExpertWaves; ++ expert_wave_idx) {
        const auto local_expert_base_idx = expert_wave_idx * 4;
        permutations[expert_wave_idx] = sort4_best_first(scores[local_expert_base_idx + 0],
                                                         scores[local_expert_base_idx + 1],
                                                         scores[local_expert_base_idx + 2],
                                                         scores[local_expert_base_idx + 3]);
    }

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
            const auto local_expert_base_idx = expert_wave_idx * 4;
            const auto candidate_score = cursor == 4 ? -cute::numeric_limits<float>::infinity()
                                       : cursor & 2 ? cursor & 1 ? scores[local_expert_base_idx + 3]
                                                                 : scores[local_expert_base_idx + 2]
                                                    : cursor & 1 ? scores[local_expert_base_idx + 1]
                                                                 : scores[local_expert_base_idx + 0];
            const auto candidate_offset = permutations[expert_wave_idx] >> (cursor * 2) & 3;
            const auto candidate_expert_idx = expert_base_idx + static_cast<int>(expert_wave_idx * 128 + lane_idx * 4 + candidate_offset);
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
CUTLASS_DEVICE float select_warp_expert_value(const float (&values)[kNumExpertWaves * 4],
                                               const int& expert_base_idx, const int& selected_expert_idx) {
    const auto relative_expert_idx = selected_expert_idx - expert_base_idx;
    const auto selected_value_idx = relative_expert_idx >= 0
                                        ? relative_expert_idx / 128 * 4 + relative_expert_idx % 4 : -1;
    const auto source_lane_idx = static_cast<uint32_t>(relative_expert_idx >= 0 ? relative_expert_idx % 128 / 4 : 0);
    auto selected_value = 0.0f;
    #pragma unroll
    for (uint32_t value_idx = 0; value_idx < kNumExpertWaves * 4; ++ value_idx) {
        const auto exchanged = ptx::exchange(values[value_idx], source_lane_idx);
        if (selected_value_idx == static_cast<int>(value_idx))
            selected_value = exchanged;
    }
    return selected_value;
}

template <uint32_t kNumTopk, bool kUnmappedTopkIdxExists, bool kToPhysicalMapExists>
CUTLASS_DEVICE void store_topk_token(const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                     const uint32_t& num_routed_experts, const uint32_t& num_shared_experts,
                                     const uint32_t& num_duplicate_experts, const float& routed_scaling_factor,
                                     const int& ep_rank, const int64_t& unmapped_topk_idx_stride,
                                     const int* to_physical_map, const int* logical_count,
                                     const int* routed_logical_count,
                                     int64_t* topk_idx, int64_t* unmapped_topk_idx, float* topk_weights,
                                     int selected_expert_idx, const float& selected_unbiased_score) {
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k slots exceed a warp");
    const auto num_physical_topk = kNumTopk + num_shared_experts;
    if (lane_idx >= kNumTopk and lane_idx < num_physical_topk)
        selected_expert_idx = static_cast<int>(lane_idx + num_routed_experts - kNumTopk);

    auto physical_expert_idx = selected_expert_idx;
    if constexpr (kToPhysicalMapExists) {
        if (lane_idx < num_physical_topk) {
            const auto logical_expert_idx = static_cast<uint32_t>(selected_expert_idx);
            const auto num_duplicates = static_cast<uint32_t>((lane_idx < kNumTopk ? routed_logical_count : logical_count)[logical_expert_idx]);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_duplicates > 0 and num_duplicates <= num_duplicate_experts);
            const auto duplicate_idx = (static_cast<uint32_t>(ep_rank) + global_token_idx * 23333u) % num_duplicates;
            physical_expert_idx = to_physical_map[logical_expert_idx * num_duplicate_experts + duplicate_idx];
        }
    }

    constexpr uint32_t kSumWidth = kNumTopk <= 8 ? 8 : (kNumTopk <= 16 ? 16 : 32);
    const auto topk_sum = math::warp_reduce_sum<kSumWidth>(selected_unbiased_score) + 1e-20f;

    auto selected_weight = selected_unbiased_score;
    if (lane_idx < kNumTopk) {
        selected_weight = selected_weight / topk_sum * routed_scaling_factor;
        if constexpr (kUnmappedTopkIdxExists)
            unmapped_topk_idx[global_token_idx * unmapped_topk_idx_stride + lane_idx] = selected_expert_idx;
    } else if (lane_idx < num_physical_topk) {
        selected_weight = 1.0f;
    }

    if (lane_idx < num_physical_topk) {
        const auto output_idx = global_token_idx * num_physical_topk + lane_idx;
        topk_idx[output_idx] = physical_expert_idx;
        topk_weights[output_idx] = selected_weight;
    }
}

CUTLASS_DEVICE uint32_t random_hash(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ value >> 16;
}

template <uint32_t kNumTopk, bool kUnmappedTopkIdxExists, bool kToPhysicalMapExists>
CUTLASS_DEVICE void store_random_topk_token(const uint32_t& global_token_idx, const uint32_t& lane_idx,
                                            const uint32_t& num_routed_experts, const uint32_t& num_shared_experts,
                                            const int* logical_count, const int& ep_rank,
                                            const int64_t& unmapped_topk_idx_stride, int64_t* topk_idx,
                                            int64_t* unmapped_topk_idx, float* topk_weights) {
    const auto num_physical_topk = kNumTopk + num_shared_experts;
    auto num_physical_experts = num_routed_experts + num_shared_experts;
    if constexpr (kToPhysicalMapExists) {
        auto local_count = 0u;
        for (uint32_t logical_expert_idx = lane_idx; logical_expert_idx < num_physical_experts; logical_expert_idx += 32)
            local_count += static_cast<uint32_t>(logical_count[logical_expert_idx]);
        num_physical_experts = __reduce_add_sync(0xffffffffu, local_count);
    }
    DG_TRAP_ONLY_DEVICE_ASSERT(num_physical_experts >= num_physical_topk);

    const auto seed = random_hash(static_cast<uint32_t>(ep_rank) ^ global_token_idx * 0x9e3779b9u ^ lane_idx * 0x85ebca6bu);
    const auto random_weight = __uint_as_float(0x3f800000u | random_hash(seed ^ 0xc2b2ae35u) >> 9) - 1.0f + 0x1p-24f;
    constexpr uint32_t kInvalidRandomIdx = INT32_MAX;
    auto candidate_idx = lane_idx < num_physical_topk
                             ? random_hash(seed) % (num_physical_experts - lane_idx)
                             : kInvalidRandomIdx;
    auto selected_expert_idx = kInvalidRandomIdx;
    for (uint32_t output_idx = 0; output_idx < num_physical_topk; ++ output_idx) {
        const auto min_candidate_idx = __reduce_min_sync(0xffffffffu, candidate_idx);
        const auto min_lane_idx = __reduce_min_sync(0xffffffffu, candidate_idx == min_candidate_idx ? lane_idx : kInvalidRandomIdx);
        if (selected_expert_idx == kInvalidRandomIdx) {
            if (candidate_idx >= min_candidate_idx and min_lane_idx < lane_idx) {
                ++ candidate_idx;
            } else if (min_lane_idx == lane_idx) {
                selected_expert_idx = candidate_idx;
                candidate_idx = kInvalidRandomIdx;
            }
        }
    }

    if (lane_idx < num_physical_topk) {
        const auto output_idx = global_token_idx * num_physical_topk + lane_idx;
        topk_idx[output_idx] = static_cast<int64_t>(selected_expert_idx);
        topk_weights[output_idx] = random_weight;
        if constexpr (kUnmappedTopkIdxExists) {
            if (lane_idx < kNumTopk)
                unmapped_topk_idx[global_token_idx * unmapped_topk_idx_stride + lane_idx] = -1;
        }
    }
}

} // namespace deep_gemm::epilogue::mega_gate
