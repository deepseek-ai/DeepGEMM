#pragma once

#include <algorithm>
#include <format>
#include <iostream>
#include <unordered_set>
#include <vector>

#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"

namespace deep_gemm {

namespace mega_gate_layout = layout::mega_gate;

struct MegaGateConfig {
    // Task decomposition: a token block is computed by `num_logical_ctas = num_expert_groups * num_mma_ctas * num_split_k` CTAs
    int num_expert_groups, num_mma_ctas, num_split_k;
    int num_logical_ctas;

    int block_tokens;
    int load_block_m, load_block_n;

    int num_worker_groups, num_launch_sms, num_waves;

    int num_token_cols;
    int num_gate_threads;

    int num_stages, smem_size;

    friend std::ostream& operator << (std::ostream& os, const MegaGateConfig& config) {
        os << "MegaGateConfig("
           << "num_expert_groups=" << config.num_expert_groups << ", num_mma_ctas=" << config.num_mma_ctas
           << ", num_split_k=" << config.num_split_k << ", num_logical_ctas=" << config.num_logical_ctas
           << ", block_tokens=" << config.block_tokens
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", num_worker_groups=" << config.num_worker_groups << ", num_launch_sms=" << config.num_launch_sms
           << ", num_waves=" << config.num_waves
           << ", num_token_cols=" << config.num_token_cols << ", num_gate_threads=" << config.num_gate_threads
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size << ")";
        return os;
    }
};

// Gate warpgroup budget: at most 7 (1024 threads with the 4 non-epilogue warps), each transposes 8 TMEM
// columns per round and ranks 4 tokens per round across the logical CTAs
constexpr int kNumMaxGateWarpgroups = 7;
constexpr int kNumTransposeColsPerWarpgroup = 8;
constexpr int kNumTopkTokensPerWarpgroup = 4;
constexpr int kNumMaxSplitK = 8;
// Doubling the task CTAs of a candidate is preferred if it costs at most this many extra waves
constexpr int kNumMaxExtraWavesForDoubling = 2;

static int select_wave_block_tokens(const int& num_tokens, const int& num_worker_groups) {
    // The smallest 16-aligned token block that keeps the wave count of the largest one
    const auto num_waves = ceil_div(ceil_div(num_tokens, static_cast<int>(mega_gate_layout::kNumMaxBlockTokens)), num_worker_groups);
    return std::min(static_cast<int>(mega_gate_layout::kNumMaxBlockTokens),
                    align(ceil_div(num_tokens, num_waves * num_worker_groups), static_cast<int>(mega_gate_layout::kNumMinBlockTokens)));
}

static int get_num_waves(const int& num_tokens, const int& num_worker_groups) {
    return ceil_div(ceil_div(num_tokens, select_wave_block_tokens(num_tokens, num_worker_groups)), num_worker_groups);
}

// TMEM columns per gate warpgroup: a 128-expert group on a CTA pair splits the tokens over the two lane halves
static int get_num_token_cols(const int& block_tokens, const int& num_experts_per_group, const int& num_mma_ctas) {
    return num_experts_per_group == static_cast<int>(mega_gate_layout::kExpertAlignment) ? block_tokens / num_mma_ctas : block_tokens;
}

static bool is_task_doubling_free(const int& num_tokens, const int& num_sms, const bool& deterministic, const MegaGateConfig& candidate) {
    const auto num_doubled_logical_ctas = candidate.num_logical_ctas * 2;
    if (num_doubled_logical_ctas > num_sms or num_doubled_logical_ctas >= static_cast<int>(mega_gate_layout::kNumMaxLogicalCtas))
        return false;

    // A single-CTA MMA can be doubled into a CTA pair (the split stays)
    const auto num_doubled_worker_groups = num_sms / num_doubled_logical_ctas;
    if (candidate.num_mma_ctas == 1 and candidate.num_split_k != kNumMaxSplitK)
        return get_num_waves(num_tokens, num_doubled_worker_groups) <= candidate.num_waves + kNumMaxExtraWavesForDoubling;
    // A single-wave CTA pair can only be doubled by splitting K, which deterministic mode forbids
    if (candidate.num_mma_ctas == 2 and candidate.num_split_k == 1 and candidate.num_waves == 1 and not deterministic)
        return select_wave_block_tokens(num_tokens, num_doubled_worker_groups) == candidate.block_tokens;
    return false;
}

static bool compare_mega_gate(const MegaGateConfig& a, const MegaGateConfig& b) {
    // Prefer longer TMEM transposes, then more SMs, then CTA pairs, then fewer splits, then fewer waves
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

static int get_num_balanced_warpgroups(const int& num_units) {
    return ceil_div(num_units, ceil_div(num_units, kNumMaxGateWarpgroups));
}

static MegaGateConfig get_mega_gate_config(const int& num_tokens, const int& hidden, const int& num_routed_experts, const int& num_sms) {
    constexpr int kExpertAlignment = mega_gate_layout::kExpertAlignment;
    constexpr int BLOCK_K = mega_gate_layout::BLOCK_K;
    const auto num_aligned_experts = align(num_routed_experts, kExpertAlignment);
    // NOTES: deterministic mode is also batch invariant, so K is never split
    const bool deterministic = heuristics_runtime->get_deterministic_algorithms();

    std::vector<MegaGateConfig> candidates;
    for (int num_expert_groups = 1; num_expert_groups <= num_aligned_experts / kExpertAlignment; ++ num_expert_groups) {
        // An expert group is a single UMMA M: 128 experts, or 256 on a CTA pair
        if (num_aligned_experts % (kExpertAlignment * num_expert_groups) != 0)
            continue;
        const auto num_experts_per_group = num_aligned_experts / num_expert_groups;
        if (num_experts_per_group > 2 * kExpertAlignment)
            continue;

        for (int num_mma_ctas = 1; num_mma_ctas <= 2; ++ num_mma_ctas) {
            if (num_experts_per_group > kExpertAlignment and num_mma_ctas != 2)
                continue;

            for (int num_split_k = 1; num_split_k <= (deterministic ? 1 : kNumMaxSplitK); num_split_k *= 2) {
                if (hidden % (num_split_k * BLOCK_K) != 0)
                    continue;
                if (num_mma_ctas == 2 and num_split_k == kNumMaxSplitK)
                    continue;

                // All the logical CTAs of a token block must be co-resident and fit the score barrier
                const auto num_logical_ctas = num_expert_groups * num_mma_ctas * num_split_k;
                if (num_logical_ctas > num_sms or num_logical_ctas >= static_cast<int>(mega_gate_layout::kNumMaxLogicalCtas))
                    continue;

                const auto num_worker_groups = num_sms / num_logical_ctas;
                const auto block_tokens = select_wave_block_tokens(num_tokens, num_worker_groups);
                const auto num_token_blocks = ceil_div(num_tokens, block_tokens);
                const auto candidate = MegaGateConfig {
                    .num_expert_groups = num_expert_groups, .num_mma_ctas = num_mma_ctas, .num_split_k = num_split_k,
                    .num_logical_ctas = num_logical_ctas,
                    .block_tokens = block_tokens,
                    .load_block_m = block_tokens / num_mma_ctas, .load_block_n = num_experts_per_group / num_mma_ctas,
                    .num_worker_groups = std::min(num_worker_groups, num_token_blocks),
                    .num_launch_sms = std::min(num_worker_groups, num_token_blocks) * num_logical_ctas,
                    .num_waves = ceil_div(num_token_blocks, num_worker_groups),
                    .num_token_cols = std::min(get_num_token_cols(block_tokens, num_experts_per_group, num_mma_ctas),
                                               kNumMaxGateWarpgroups * kNumTransposeColsPerWarpgroup),
                };
                if (not is_task_doubling_free(num_tokens, num_sms, deterministic, candidate))
                    candidates.push_back(candidate);
            }
        }
    }
    DG_HOST_ASSERT(not candidates.empty());
    auto config = *std::min_element(candidates.begin(), candidates.end(), compare_mega_gate);

    // Gate warpgroups: enough to balance both the TMEM transpose and the top-k rounds, all of them for multiple waves
    const auto num_experts_per_group = num_aligned_experts / config.num_expert_groups;
    const auto num_transpose_units = ceil_div(get_num_token_cols(config.block_tokens, num_experts_per_group, config.num_mma_ctas),
                                              kNumTransposeColsPerWarpgroup);
    const auto num_topk_units = ceil_div(config.block_tokens, kNumTopkTokensPerWarpgroup * config.num_logical_ctas);
    const auto num_gate_warpgroups = config.num_waves > 1 ? kNumMaxGateWarpgroups :
        std::max(get_num_balanced_warpgroups(num_transpose_units), get_num_balanced_warpgroups(num_topk_units));
    config.num_gate_threads = num_gate_warpgroups * 128;

    constexpr int kNumFixedSmemBytes = mega_gate_layout::kStageDataOffset;
    const auto num_stage_bytes = (config.load_block_m + config.load_block_n) * BLOCK_K * static_cast<int>(sizeof(cutlass::bfloat16_t));
    config.num_stages = std::min(static_cast<int>(mega_gate_layout::kNumMaxStages),
                                 (SM100ArchSpec::smem_capacity - kNumFixedSmemBytes) / num_stage_bytes);
    config.smem_size = align(kNumFixedSmemBytes + config.num_stages * num_stage_bytes,
                             static_cast<int>(mega_gate_layout::kSharedMemoryAlignment));
    DG_HOST_ASSERT(config.num_stages >= 2);

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format("MegaGateConfig(num_tokens={}, hidden={}, num_routed_experts={}, deterministic={})",
                                     num_tokens, hidden, num_routed_experts, deterministic);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
