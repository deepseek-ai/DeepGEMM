#pragma once

#include <algorithm>
#include <vector>

#include <deep_gemm/layout/mega_gate.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"

namespace deep_gemm {

namespace mega_gate_layout = layout::mega_gate;

constexpr int kNumMaxGateWarpgroups = 7;
constexpr int kNumTopkTokensPerWarpgroup = 4;
constexpr int kNumTransposeColsPerWarpgroup = 8;
constexpr int kMegaGateMaxSplitK = 8;
constexpr int kNumMaxStages = 32;
constexpr int kNumMaxRoutedExperts = 512;
constexpr int kNumStageBarrierBytes = 2 * static_cast<int>(sizeof(uint64_t));
constexpr int kNumFixedBarrierBytes = (2 * static_cast<int>(mega_gate_layout::kNumEpilogueStages) + 1) * static_cast<int>(sizeof(uint64_t));

static int get_num_fixed_smem_bytes(const int& num_routed_experts, const bool& has_bias,
                                    const bool& has_image_token_mask, const bool& to_physical_map_exists) {
    return align(kNumFixedBarrierBytes + static_cast<int>(sizeof(uint32_t)) +
                 static_cast<int>(mega_gate_layout::get_num_metadata_cache_bytes(num_routed_experts, has_bias, has_image_token_mask, to_physical_map_exists)),
                 static_cast<int>(mega_gate_layout::kSharedMemoryAlignment));
}

static int get_num_stage_bytes(const int& load_block_m, const int& load_block_n) {
    return (load_block_m + load_block_n) * static_cast<int>(mega_gate_layout::BLOCK_K) *
           static_cast<int>(sizeof(cutlass::bfloat16_t)) + kNumStageBarrierBytes;
}

static int get_num_stages(const int& smem_capacity, const int& num_fixed_smem_bytes,
                         const int& load_block_m, const int& load_block_n) {
    return std::min(kNumMaxStages,
                    (smem_capacity - num_fixed_smem_bytes) / get_num_stage_bytes(load_block_m, load_block_n));
}

static int get_num_smem_bytes(const int& num_stages, const int& num_fixed_smem_bytes,
                              const int& load_block_m, const int& load_block_n) {
    return align(num_fixed_smem_bytes + num_stages * get_num_stage_bytes(load_block_m, load_block_n),
                 static_cast<int>(mega_gate_layout::kSharedMemoryAlignment));
}

struct SM100BF16MegaGateConfig {
    int block_tokens;
    int num_mma_ctas;
    int num_split_k;
    int num_expert_groups;
    int num_stages;
    int smem_size;
    int num_gate_warpgroups;
    int num_launch_sms;
};

static int get_num_worker_groups(const int& num_task_ctas, const int& num_sms) {
    return num_sms / num_task_ctas;
}

static int select_wave_tile(const int& num_tokens, const int& num_workers) {
    const auto num_waves = ceil_div(ceil_div(num_tokens,
                                             static_cast<int>(mega_gate_layout::kNumMaxBlockTokens)),
                                    num_workers);
    return std::min(static_cast<int>(mega_gate_layout::kNumMaxBlockTokens),
                    align(ceil_div(num_tokens, num_waves * num_workers), static_cast<int>(mega_gate_layout::kNumMinBlockTokens)));
}

static int get_num_token_cols(const int& block_tokens, const int& num_experts_per_group, const int& num_mma_ctas) {
    return num_experts_per_group == static_cast<int>(mega_gate_layout::kExpertAlignment)
           ? block_tokens / num_mma_ctas : block_tokens;
}

struct MegaGateConfig {
    int num_expert_groups;
    int num_mma_ctas;
    int num_split_k;
    int block_tokens;
    int num_token_cols;
    int num_launch_sms;
    int num_waves;
};

static int get_num_waves(const int& num_tokens, const int& num_workers) {
    return ceil_div(ceil_div(num_tokens, select_wave_tile(num_tokens, num_workers)), num_workers);
}

static bool is_task_doubling_free(const int& num_tokens, const int& num_sms, const MegaGateConfig& candidate) {
    const auto num_task_ctas = candidate.num_expert_groups * candidate.num_mma_ctas * candidate.num_split_k;
    const auto num_doubled_task_ctas = num_task_ctas * 2;
    if (num_doubled_task_ctas > num_sms or
        num_doubled_task_ctas >= static_cast<int>(mega_gate_layout::kNumMaxLogicalCtas))
        return false;

    const auto num_doubled_workers = get_num_worker_groups(num_doubled_task_ctas, num_sms);
    if (candidate.num_mma_ctas == 1 and candidate.num_split_k != kMegaGateMaxSplitK)
        return get_num_waves(num_tokens, num_doubled_workers) <= candidate.num_waves + 2;
    if (candidate.num_mma_ctas == 2 and candidate.num_split_k == 1 and candidate.num_waves == 1)
        return select_wave_tile(num_tokens, num_doubled_workers) == candidate.block_tokens;
    return false;
}

static std::vector<MegaGateConfig> get_mega_gate_candidates(const int& num_tokens, const int& hidden,
                                                            const int& num_routed_experts, const int& num_sms) {
    constexpr int kExpertAlignment = mega_gate_layout::kExpertAlignment;
    std::vector<MegaGateConfig> candidates;
    for (int num_expert_groups = 1; num_expert_groups <= num_routed_experts / kExpertAlignment; ++ num_expert_groups) {
        if (num_routed_experts % (kExpertAlignment * num_expert_groups) != 0)
            continue;

        const auto num_experts_per_group = num_routed_experts / num_expert_groups;
        if (num_experts_per_group > 2 * kExpertAlignment)
            continue;

        for (int num_mma_ctas = 1; num_mma_ctas <= 2; ++ num_mma_ctas) {
            if (num_experts_per_group > kExpertAlignment and num_mma_ctas != 2)
                continue;

            for (int num_split_k = 1; num_split_k <= kMegaGateMaxSplitK; num_split_k *= 2) {
                if (hidden % (num_split_k * static_cast<int>(mega_gate_layout::BLOCK_K)) != 0)
                    continue;

                if (num_mma_ctas == 2 and num_split_k == kMegaGateMaxSplitK)
                    continue;

                const auto num_task_ctas = num_expert_groups * num_mma_ctas * num_split_k;
                if (num_task_ctas > num_sms)
                    continue;

                if (num_task_ctas >= static_cast<int>(mega_gate_layout::kNumMaxLogicalCtas))
                    continue;

                const auto num_workers = get_num_worker_groups(num_task_ctas, num_sms);
                const auto block_tokens = select_wave_tile(num_tokens, num_workers);
                const auto num_token_blocks = ceil_div(num_tokens, block_tokens);
                const auto num_token_cols = std::min(get_num_token_cols(block_tokens, num_experts_per_group, num_mma_ctas),
                                                     kNumMaxGateWarpgroups * kNumTransposeColsPerWarpgroup);
                const MegaGateConfig candidate = {num_expert_groups, num_mma_ctas, num_split_k,
                                                  block_tokens, num_token_cols,
                                                  std::min(num_workers, num_token_blocks) * num_task_ctas,
                                                  ceil_div(num_token_blocks, num_workers)};
                if (is_task_doubling_free(num_tokens, num_sms, candidate))
                    continue;

                candidates.push_back(candidate);
            }
        }
    }

    DG_HOST_ASSERT(not candidates.empty());
    return candidates;
}

static bool compare_mega_gate(const MegaGateConfig& a, const MegaGateConfig& b) {
    if (a.num_token_cols != b.num_token_cols)
        return a.num_token_cols > b.num_token_cols;

    if (a.num_launch_sms != b.num_launch_sms)
        return a.num_launch_sms > b.num_launch_sms;

    if (a.num_mma_ctas != b.num_mma_ctas)
        return a.num_mma_ctas > b.num_mma_ctas;

    if (a.num_split_k != b.num_split_k)
        return a.num_split_k < b.num_split_k;

    return a.num_waves < b.num_waves;
}

static MegaGateConfig select_mega_gate_config(const int& num_tokens, const int& hidden,
                                              const int& num_routed_experts, const int& num_sms) {
    const auto candidates = get_mega_gate_candidates(num_tokens, hidden, num_routed_experts, num_sms);
    return *std::min_element(candidates.begin(), candidates.end(), compare_mega_gate);
}

static int get_num_balanced_warpgroups(const int& num_units) {
    return ceil_div(num_units, ceil_div(num_units, kNumMaxGateWarpgroups));
}

static int select_gate_warpgroups(const MegaGateConfig& config, const int& num_experts_per_group) {
    if (config.num_waves > 1)
        return kNumMaxGateWarpgroups;

    const auto num_task_ctas = config.num_expert_groups * config.num_mma_ctas * config.num_split_k;
    const auto num_transpose_units = ceil_div(get_num_token_cols(config.block_tokens, num_experts_per_group, config.num_mma_ctas),
                                              kNumTransposeColsPerWarpgroup);
    const auto num_topk_units = ceil_div(config.block_tokens, kNumTopkTokensPerWarpgroup * num_task_ctas);
    return std::max(get_num_balanced_warpgroups(num_transpose_units),
                    get_num_balanced_warpgroups(num_topk_units));
}

static SM100BF16MegaGateConfig get_sm100_bf16_mega_gate_config(const int& num_tokens, const int& hidden,
                                                               const int& num_actual_routed_experts,
                                                               const int& num_sms, const bool& has_bias,
                                                               const bool& has_image_token_mask,
                                                               const bool& to_physical_map_exists) {
    DG_HOST_ASSERT(num_tokens > 0 and num_tokens <= static_cast<int>(mega_gate_layout::kNumMaxTokens) and num_sms > 0);
    DG_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    DG_HOST_ASSERT(num_actual_routed_experts > 0 and num_actual_routed_experts <= kNumMaxRoutedExperts and
                   num_actual_routed_experts % 4 == 0);
    const auto num_routed_experts = align(num_actual_routed_experts, static_cast<int>(mega_gate_layout::kExpertAlignment));
    auto config = select_mega_gate_config(num_tokens, hidden, num_routed_experts, num_sms);
    if (heuristics_runtime->get_deterministic_algorithms())
        config.num_split_k = 1;
    const auto token_mma = config.block_tokens;

    const auto experts_per_group = num_routed_experts / config.num_expert_groups;
    const auto expert_mma = std::min(experts_per_group, 2 * static_cast<int>(mega_gate_layout::kExpertAlignment));
    DG_HOST_ASSERT(expert_mma == static_cast<int>(mega_gate_layout::kExpertAlignment) or config.num_mma_ctas == 2);
    const auto load_block_m = token_mma / config.num_mma_ctas;
    const auto load_block_n = expert_mma / config.num_mma_ctas;
    const auto num_gate_warpgroups = select_gate_warpgroups(config, experts_per_group);
    const auto num_fixed_smem_bytes = get_num_fixed_smem_bytes(num_routed_experts, has_bias,
                                                               has_image_token_mask, to_physical_map_exists);
    const auto num_stages = get_num_stages(SM100ArchSpec::smem_capacity, num_fixed_smem_bytes,
                                           load_block_m, load_block_n);
    const auto smem_size = get_num_smem_bytes(num_stages, num_fixed_smem_bytes,
                                              load_block_m, load_block_n);
    DG_HOST_ASSERT(num_stages >= 2 and smem_size <= SM100ArchSpec::smem_capacity);

    return SM100BF16MegaGateConfig{
        .block_tokens = token_mma,
        .num_mma_ctas = config.num_mma_ctas,
        .num_split_k = config.num_split_k,
        .num_expert_groups = config.num_expert_groups,
        .num_stages = num_stages,
        .smem_size = smem_size,
        .num_gate_warpgroups = num_gate_warpgroups,
        .num_launch_sms = config.num_launch_sms,
    };
}

} // namespace deep_gemm
