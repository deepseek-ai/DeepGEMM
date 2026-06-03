#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm100.hpp"
#include "sm90.hpp"

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Pool capacity and SF-padded token count
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static std::tuple<int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    const auto& [cluster_size, block_m, store_block_m, num_epilogue_warpgroups] = [&]() -> std::tuple<int, int, int, int> {
        float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
        if (num_expected_tokens_per_expert <= 8.5) {
            // Really small token-per-expert (e.g. RL long-tail rollout), use the smallest block_m
            return {2, 16, 8, 2};
        } else if (num_expected_tokens_per_expert <= 16.5) {
            // Small batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 128
            return {2, 32, 16, 2};
        } else if (num_expected_tokens_per_expert <= 32.5) {
            // Medium batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 256
            return {2, 64, 32, 1};
        } else if (num_expected_tokens_per_expert <= 64.5) {
            // Large batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 512
            return {2, 96, 16, 2};
        } else if (num_expected_tokens_per_expert <= 96.5) {
            // Medium batch size, Medium EP, decoding, e.g. 6/384 experts, EP16, bsz 256, or EP32, bsz128
            return {2, 128, 32, 2};
        } else {
            // Prefill, or large EP decoding
            return {2, 192, 32, 2};
        }
    }();

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    // Return configs
    return {cluster_size, block_m, store_block_m, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {

    float expected_tokens_per_expert = static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (expected_tokens_per_expert < 1) {
        // Most experts don't have tokens, calculate all experts at once
        return num_experts_per_rank;
    }

    // Reduce per-expert block count by this factor since uneven routing leaves some experts with fewer tokens
    constexpr int kImbalanceFactor = 2;

    // Count L1 blocks per expert assuming tokens are evenly spread across experts
    const int num_m_blocks = ceil_div(static_cast<int>(std::ceil(expected_tokens_per_expert)), block_m);
    const int num_n_blocks = (2 * intermediate_hidden) / block_n;
    const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;

    // Pick the smallest value whose total blocks (after imbalance reduction) can keep all SMs busy
    int num_experts_per_wave = num_l1_blocks_per_expert > 0
        ? ceil_div(kImbalanceFactor * num_sms, num_l1_blocks_per_expert) : 1;
    num_experts_per_wave = std::min(num_experts_per_wave, num_experts_per_rank);

    // Round up to the nearest divisor of num_experts_per_rank so every wave processes the same count
    while (num_experts_per_wave < num_experts_per_rank and num_experts_per_rank % num_experts_per_wave != 0)
        ++ num_experts_per_wave;

    return num_experts_per_wave;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (2 TMA stages, BLOCK_N/2 post-SwiGLU) and L2 BF16 (1 stage)
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = std::max(smem_cd_l1, smem_cd_l2);

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8;

    // Amax reduction
    const int smem_amax_reduction = store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = sf_block_m * 4;
    const int smem_sfb_per_stage = sf_block_n * 4;

    // Per-stage: A tile + B tile + SFA tile + SFB tile + full/empty barriers
    const int smem_per_stage = load_block_m * block_k + block_n * block_k + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;

    // Select maximum num_stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    // Block config
    const auto [cluster_size, block_m, store_block_m, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int block_n = 128;
    const int block_k = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4);
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // Waves
    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, store_block_m,
        sf_block_m, sf_block_n,
        num_dispatch_threads / 32, num_epilogue_threads / 32);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

// ============================================================================
// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3, scales are per-128 channel float.
//   - No 2-CTA cluster MMA: TMA multicast cluster=2 may still be used.
//   - SF for activations is float (not UE8M0 int) and per-128 (not per-32).
// The kernel is in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`; this config is
// what the host runtime reads when instantiating a shape-specialized variant.
// ============================================================================

struct MegaMoESM90Config {
    // Block tiling (no STORE_BLOCK_M / SF_BLOCK_M concept on SM90)
    int block_m, block_n, block_k;

    // Cluster size for TMA multicast (1 or 2). Multicast is on A.
    int cluster_size;

    // Pool capacity and SF-padded token count (SF is per-128 float on SM90)
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // Swizzle modes for TMA descriptors (acts/weights). Both are 128B on FP8 K-major.
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout: dispatch + non-epilogue (TMA) + epilogue (math)
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Chosen scheduler / epilogue modes.  Keeping these in the config makes the
    // SM90 path follow the same single-source-of-truth style as regular GEMM
    // configs: the selector chooses a complete candidate, then launch consumes it.
    bool direct_l2_scatter, l2_nmajor_schedule, one_warp_cleanup;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", direct_l2_scatter=" << config.direct_l2_scatter
           << ", l2_nmajor_schedule=" << config.l2_nmajor_schedule
           << ", one_warp_cleanup=" << config.one_warp_cleanup << ")";
        return os;
    }
};

enum class Sm90MoeRuntimeProfile {
    Generic,
    LowSm,
    HighSm
};

static std::string get_sm90_moe_lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static Sm90MoeRuntimeProfile get_sm90_moe_runtime_profile() {
    const auto forced = get_sm90_moe_lowercase(
        get_env<std::string>("DG_SM90_MOE_DEVICE_PROFILE", ""));
    if (not forced.empty() and forced != "auto") {
        DG_HOST_ASSERT(forced == "generic" or forced == "low_sm" or forced == "high_sm");
        if (forced == "low_sm")
            return Sm90MoeRuntimeProfile::LowSm;
        if (forced == "high_sm")
            return Sm90MoeRuntimeProfile::HighSm;
        return Sm90MoeRuntimeProfile::Generic;
    }

    const int num_sms = device_runtime->get_num_sms();
    if (num_sms <= 80)
        return Sm90MoeRuntimeProfile::LowSm;
    if (num_sms >= 100)
        return Sm90MoeRuntimeProfile::HighSm;
    return Sm90MoeRuntimeProfile::Generic;
}

struct Sm90MoeProfileConfig {
    // A zero wave count means "use the generic SM-count based computation".
    int num_experts_per_wave, num_stages;
    bool direct_l2_scatter, l2_nmajor_schedule, one_warp_cleanup;
};

struct Sm90MoeHeuristicPolicy {
    Sm90MoeRuntimeProfile runtime_profile;
    int num_experts_per_rank, num_topk, intermediate_hidden;
    int block_m, block_n;
    float expected_tokens_per_expert;

    template <typename... Values>
    bool expected_is_one_of(const Values&... values) const {
        return ((expected_tokens_per_expert == static_cast<float>(values)) or ...);
    }

    bool expected_is_between(const float& low, const float& high) const {
        return expected_tokens_per_expert >= low and expected_tokens_per_expert <= high;
    }

    bool uses_bn256_main_tile() const {
        return block_m == 64 and block_n == 256;
    }

    bool is_main_topk8() const {
        return num_experts_per_rank == 32 and num_topk == 8 and intermediate_hidden == 2048;
    }

    bool is_hopper_topk6() const {
        return num_experts_per_rank == 48 and num_topk == 6 and intermediate_hidden == 3072;
    }

    bool low_sm_main_topk8_profile_config(Sm90MoeProfileConfig& config,
                                          const bool& direct_l2_scatter_enabled,
                                          const bool& eplb_hint,
                                          const bool& skew_hint,
                                          const bool& masked_hint) const {
        int wave_override = 0;
        if (expected_tokens_per_expert == 128.0f or
            (expected_tokens_per_expert >= 256.0f and expected_tokens_per_expert < 512.0f)) {
            wave_override = 16;
        }

        const bool direct_l2_scatter_enabled_by_profile =
            expected_is_one_of(2, 4, 8, 16, 32, 64, 76, 80, 88, 128) or
            expected_is_between(64.0f, 80.0f) or
            expected_is_between(96.0f, 120.0f) or
            expected_tokens_per_expert >= 144.0f;

        const bool l2_nmajor_schedule_enabled = [&]() {
            if (expected_tokens_per_expert == 256.0f and eplb_hint)
                return false;
            if (expected_tokens_per_expert >= 256.0f and skew_hint)
                return false;
            return expected_tokens_per_expert >= 256.0f;
        }();

        const bool one_warp_cleanup_enabled = expected_tokens_per_expert <= 80.0f;
        const bool stage5_pipeline_enabled = [&]() {
            if (not direct_l2_scatter_enabled)
                return false;
            const bool hinted_m64 =
                (eplb_hint or skew_hint or masked_hint) and expected_tokens_per_expert == 64.0f;
            return expected_is_one_of(2, 4, 16, 32, 128) or
                   hinted_m64 or
                   expected_tokens_per_expert >= 192.0f;
        }();

        config = {
            wave_override,
            stage5_pipeline_enabled ? 5 : 4,
            direct_l2_scatter_enabled_by_profile,
            l2_nmajor_schedule_enabled,
            one_warp_cleanup_enabled
        };
        return true;
    }

    bool high_sm_main_topk8_profile_config(Sm90MoeProfileConfig& config) const {
        // Profile buckets keyed by expected_tokens_per_expert.
        if (expected_tokens_per_expert <= 3.0f) {
            config = {32, 4, true,  true,  false};
        } else if (expected_tokens_per_expert <= 6.0f) {
            config = {32, 4, false, true,  true};
        } else if (expected_tokens_per_expert <= 12.0f) {
            config = {32, 4, true,  false, true};
        } else if (expected_tokens_per_expert <= 24.0f) {
            config = {32, 4, false, true,  true};
        } else if (expected_tokens_per_expert <= 48.0f) {
            config = {32, 4, true,  false, true};
        } else if (expected_tokens_per_expert <= 64.5f) {
            config = {32, 4, false, true,  true};
        } else if (expected_tokens_per_expert <= 160.0f) {
            config = {32, 4, false, true,  false};
        } else if (expected_tokens_per_expert <= 240.0f) {
            config = {32, 4, false, true,  false};
        } else if (expected_tokens_per_expert <= 384.0f) {
            config = {16, 4, false, true,  false};
        } else if (expected_tokens_per_expert <= 640.0f) {
            config = {32, 4, false, true,  true};
        } else if (expected_tokens_per_expert <= 896.0f) {
            config = {32, 4, false, true,  false};
        } else if (expected_tokens_per_expert <= 1536.0f) {
            config = {32, 4, false, true,  true};
        } else {
            config = {32, 4, false, true,  false};
        }
        return true;
    }

    bool device_profile_config(Sm90MoeProfileConfig& config,
                               const bool& direct_l2_scatter_enabled = false,
                               const bool& eplb_hint = false,
                               const bool& skew_hint = false,
                               const bool& masked_hint = false) const {
        if (not uses_bn256_main_tile() or not is_main_topk8())
            return false;

        if (runtime_profile == Sm90MoeRuntimeProfile::LowSm) {
            return low_sm_main_topk8_profile_config(
                config, direct_l2_scatter_enabled, eplb_hint, skew_hint, masked_hint);
        }
        if (runtime_profile == Sm90MoeRuntimeProfile::HighSm)
            return high_sm_main_topk8_profile_config(config);
        return false;
    }

    int experts_per_wave_override() const {
        if (not (block_m == 64 and block_n == 256))
            return 0;
        Sm90MoeProfileConfig profile_config;
        if (device_profile_config(profile_config))
            return profile_config.num_experts_per_wave;
        if (is_hopper_topk6() and expected_tokens_per_expert >= 8.0f and expected_tokens_per_expert <= 32.0f)
            return 16;
        if (is_main_topk8() and expected_tokens_per_expert == 128.0f)
            return 16;
        if (is_main_topk8() and expected_tokens_per_expert >= 256.0f and expected_tokens_per_expert < 512.0f)
            return 16;
        return 0;
    }

    bool direct_l2_scatter() const {
        if (not uses_bn256_main_tile())
            return false;
        Sm90MoeProfileConfig profile_config;
        if (device_profile_config(profile_config))
            return profile_config.direct_l2_scatter;
        if (is_main_topk8()) {
            return expected_is_one_of(2, 4, 8, 16, 32, 64, 76, 80, 88, 128) or
                   expected_is_between(64.0f, 80.0f) or
                   expected_is_between(96.0f, 120.0f) or
                   expected_tokens_per_expert >= 144.0f;
        }
        if (is_hopper_topk6()) {
            return expected_is_between(61.0f, 62.0f) or
                   expected_tokens_per_expert >= 64.0f;
        }
        return false;
    }

    bool l2_nmajor_schedule(const bool& eplb_hint, const bool& skew_hint) const {
        if (not uses_bn256_main_tile() or not is_main_topk8())
            return false;
        Sm90MoeProfileConfig profile_config;
        if (device_profile_config(profile_config, false, eplb_hint, skew_hint))
            return profile_config.l2_nmajor_schedule;
        if (expected_tokens_per_expert == 256.0f and eplb_hint)
            return false;
        if (expected_tokens_per_expert >= 256.0f and skew_hint)
            return false;
        return expected_tokens_per_expert >= 256.0f;
    }

    bool one_warp_cleanup(const bool& masked_hint) const {
        if (not uses_bn256_main_tile())
            return false;
        Sm90MoeProfileConfig profile_config;
        if (device_profile_config(profile_config, false, false, false, masked_hint))
            return profile_config.one_warp_cleanup;
        if (is_main_topk8() and expected_tokens_per_expert <= 80.0f)
            return true;
        if (is_hopper_topk6() and masked_hint and expected_tokens_per_expert == 64.0f)
            return true;
        return is_hopper_topk6() and expected_is_one_of(80, 128);
    }

    bool stage5_pipeline(const bool& direct_l2_scatter_enabled,
                         const bool& eplb_hint,
                         const bool& skew_hint,
                         const bool& masked_hint) const {
        Sm90MoeProfileConfig profile_config;
        if (device_profile_config(
                profile_config, direct_l2_scatter_enabled, eplb_hint, skew_hint, masked_hint))
            return profile_config.num_stages == 5;
        if (not direct_l2_scatter_enabled)
            return false;
        if (is_main_topk8()) {
            const bool hinted_m64 = (eplb_hint or skew_hint or masked_hint) and expected_tokens_per_expert == 64.0f;
            return expected_is_one_of(2, 4, 16, 32, 128) or
                   hinted_m64 or
                   expected_tokens_per_expert >= 192.0f;
        }
        if (is_hopper_topk6()) {
            return expected_tokens_per_expert == 64.0f or
                   expected_is_between(76.0f, 96.0f) or
                   (expected_tokens_per_expert >= 128.0f and expected_tokens_per_expert < 240.0f) or
                   expected_tokens_per_expert >= 384.0f;
        }
        return false;
    }
};

static Sm90MoeHeuristicPolicy get_sm90_moe_heuristic_policy(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n) {
    return {
        get_sm90_moe_runtime_profile(),
        num_experts_per_rank,
        num_topk,
        intermediate_hidden,
        block_m,
        block_n,
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank
    };
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    if (const int forced = get_env<int>("DG_SM90_MOE_EXPERTS_PER_WAVE"); forced > 0) {
        DG_HOST_ASSERT(forced <= num_experts_per_rank);
        DG_HOST_ASSERT(num_experts_per_rank % forced == 0);
        return forced;
    }

    const auto policy = get_sm90_moe_heuristic_policy(
        num_experts_per_rank, num_tokens, num_topk, intermediate_hidden, block_m, block_n);
    if (const int wave_override = policy.experts_per_wave_override(); wave_override > 0)
        return wave_override;
    if (block_m == 64 and
        (policy.expected_tokens_per_expert < 1.0f or policy.expected_tokens_per_expert > 4.0f)) {
        return num_experts_per_rank;
    }
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& direct_l2_scatter_enabled = false,
    const int& default_num_stages = 0) {
    constexpr int kSmemAlignment = 1024;

    // Dispatch region (same as SM100)
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (single-buffered, BLOCK_N/2 post-SwiGLU)
    // and L2 BF16, then 1024-byte aligned (matches kernel's SMEM_CD_SIZE).
    // The mma.sync decode path additionally stages one BLOCK_M x BLOCK_N FP32
    // accumulator tile in SMEM for logical-row epilogue mapping.
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const bool split_n_warpgroups = block_m == 64 and block_n == 256 and num_epilogue_warpgroups == 2;
    const bool serial_n_warpgroups = false;
    const int wg_block_m = split_n_warpgroups ? block_m : block_m / num_epilogue_warpgroups;
    const int wg_block_n = (split_n_warpgroups or serial_n_warpgroups) ? block_n / 2 : block_n;
    const int smem_cd_accum = (block_m == 16 or block_m == 32) ? align(block_m * block_n * static_cast<int>(sizeof(float)), kSmemAlignment) : 0;
    const int smem_cd_l1 = num_epilogue_warpgroups * wg_block_m * (wg_block_n / 2);  // 1 byte/elem (FP8)
    const bool direct_l2_scatter = direct_l2_scatter_enabled and
                                   block_m != 16 and block_m != 32 and
                                   not serial_n_warpgroups and wg_block_n == 128;
    const bool async_l1_tma_store = get_env<int>("DG_SM90_MOE_ASYNC_L1_STORE", 0) != 0 and
                                    block_m != 16 and block_m != 32 and
                                    not split_n_warpgroups and num_epilogue_warpgroups == 1;
    const int smem_cd_l2 = direct_l2_scatter ? 0 :
        num_epilogue_warpgroups * wg_block_m * wg_block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_l1_async = async_l1_tma_store ?
        2 * num_epilogue_warpgroups * wg_block_m * (block_n / 2) : 0;
    const int smem_cd = smem_cd_accum + align(std::max(std::max(smem_cd_l1, smem_cd_l2), smem_cd_l1_async), kSmemAlignment);

    // SF on SM90:
    //   * SFA per stage must hold the larger of L1 (BLOCK_M floats, per-128 K)
    //     and L2 (2 * BLOCK_M floats, per-64 K), aligned to 128 bytes
    //   * SFB is loaded directly from global by the math warpgroup (block-(128,128)
    //     weight quantization), so no SMEM is reserved for it.
    const int smem_sfa_half_stride_bytes = align(block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfa_per_stage = 2 * smem_sfa_half_stride_bytes;
    const int smem_sfb_per_stage = 0;

    // Per-stage: A tile + B tile + SFA tile + SFB tile
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    // Barriers (8 bytes each):
    //   * dispatch: num_dispatch_warps
    //   * GEMM full + empty: 2 * num_stages
    //   * combine: 2 * num_epilogue_warps
    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed;

    // Select the retained stage count for the current shape.
    const int max_num_stages = (smem_capacity - smem_fixed) /
                               (smem_per_stage + smem_barriers_per_stage);
    const bool prefer_bn256_n_tile = block_n == 256;
    const int preferred_num_stages = default_num_stages > 0
        ? std::min(default_num_stages, max_num_stages)
        : (prefer_bn256_n_tile ? std::min(4, max_num_stages) : 0);
    const int forced_num_stages = get_env<int>("DG_SM90_MOE_NUM_STAGES");
    const int num_stages = forced_num_stages > 0
        ? std::min(forced_num_stages, max_num_stages)
        : (preferred_num_stages > 0 ? preferred_num_stages : max_num_stages);
    DG_HOST_ASSERT(num_stages >= 2 and num_stages <= max_num_stages);
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
}

template <typename T>
static void append_unique_moe_candidate(std::vector<T>& values, const T& value) {
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.emplace_back(value);
}

static std::vector<int> get_sm90_moe_bool_candidates(
    const std::string& env_name,
    const bool& default_value) {
    const int forced = get_env<int>(env_name, -1);
    DG_HOST_ASSERT(forced == -1 or forced == 0 or forced == 1);
    std::vector<int> values;
    if (forced != -1) {
        values.emplace_back(forced);
        return values;
    }
    append_unique_moe_candidate(values, default_value ? 1 : 0);
    return values;
}

struct Sm90MoeConfigInfo {
    int64_t score;
    int num_blocks, num_waves, last_wave_util;
    int empirical_penalty;
    MegaMoESM90Config config;

    friend std::ostream& operator << (std::ostream& os, const Sm90MoeConfigInfo& info) {
        os << "Sm90MoeConfigInfo(score=" << info.score
           << ", num_blocks=" << info.num_blocks
           << ", num_waves=" << info.num_waves
           << ", last_wave_util=" << info.last_wave_util
           << ", empirical_penalty=" << info.empirical_penalty
           << ", config=" << info.config << ")";
        return os;
    }
};

static Sm90MoeConfigInfo get_sm90_moe_config_info(
    const MegaMoESM90Config& config,
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const int& num_sms,
    const bool& empirical_direct_l2_scatter,
    const bool& empirical_l2_nmajor_schedule,
    const bool& empirical_one_warp_cleanup,
    const int& empirical_num_stages,
    const int& empirical_num_experts_per_wave) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    const int expected_tokens_ceil =
        std::max(1, static_cast<int>(std::ceil(expected_tokens_per_expert)));
    const int num_m_blocks = ceil_div(expected_tokens_ceil, config.block_m);
    const int num_l1_n_blocks = ceil_div(2 * intermediate_hidden, config.block_n);
    const int num_l2_n_blocks = ceil_div(hidden, config.block_n);
    const int num_blocks = num_experts_per_rank * num_m_blocks *
                           (num_l1_n_blocks + num_l2_n_blocks);
    const int num_waves = ceil_div(num_blocks, num_sms);
    const int num_last_blocks = num_blocks % num_sms;
    const int last_wave_util = num_last_blocks == 0 ? num_sms : num_last_blocks;

    // Rank legal selector candidates with cheap shape-derived estimates.
    int empirical_penalty = 0;
    if (config.direct_l2_scatter != empirical_direct_l2_scatter)
        empirical_penalty += 1000000;
    if (config.l2_nmajor_schedule != empirical_l2_nmajor_schedule)
        empirical_penalty += 500000;
    if (config.one_warp_cleanup != empirical_one_warp_cleanup)
        empirical_penalty += 250000;
    if (config.num_stages != empirical_num_stages)
        empirical_penalty += 500000;
    if (config.num_experts_per_wave != empirical_num_experts_per_wave)
        empirical_penalty += 250000;

    int64_t score = 0;
    score += static_cast<int64_t>(num_waves) * 100000;
    score -= static_cast<int64_t>(last_wave_util) * 100;
    score += static_cast<int64_t>(num_blocks);
    score += static_cast<int64_t>(config.smem_size / 1024);
    score += empirical_penalty;

    // Prefer the compact split frontend when the calibrated modes tie.
    if (config.block_m == 64 and config.block_n == 256 and
        config.num_dispatch_threads == 64 and config.num_non_epilogue_threads == 64)
        score -= 1000;

    return {score, num_blocks, num_waves, last_wave_util, empirical_penalty, config};
}

static std::vector<MegaMoESM90Config> get_mega_moe_config_candidates_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;

    const int requested_mma_m = get_env<int>("DG_SM90_MOE_MMA_SYNC_M") > 0
        ? get_env<int>("DG_SM90_MOE_MMA_SYNC_M")
        : (get_env<int>("DG_SM90_MOE_MMA_SYNC") != 0 ? 16 : 0);
    DG_HOST_ASSERT(requested_mma_m == 0 or requested_mma_m == 16 or requested_mma_m == 32);
    const int forced_block_m = get_env<int>("DG_SM90_MOE_FORCE_BLOCK_M");
    const int forced_epilogue_warpgroups = get_env<int>("DG_SM90_MOE_FORCE_EPILOGUE_WG");
    DG_HOST_ASSERT(forced_block_m == 0 or forced_block_m == 64 or forced_block_m == 128);
    DG_HOST_ASSERT(forced_epilogue_warpgroups == 0 or
                   forced_epilogue_warpgroups == 1 or
                   forced_epilogue_warpgroups == 2);

    const bool use_mma_sync_decode =
        requested_mma_m > 0 and expected_tokens_per_expert <= static_cast<float>(requested_mma_m);
    const bool use_b_stationary_2wg =
        get_env<int>("DG_SM90_MOE_B_STATIONARY_2WG") != 0 and not use_mma_sync_decode;
    const bool use_bn256_split_n_env =
        get_env<int>("DG_SM90_MOE_BN256_2WG", 1) != 0 and
        forced_block_m != 128 and not use_mma_sync_decode;
    DG_HOST_ASSERT(not (use_b_stationary_2wg and use_bn256_split_n_env));

    std::vector<int> block_m_candidates;
    if (forced_block_m > 0) {
        append_unique_moe_candidate(block_m_candidates, forced_block_m);
    } else if (use_b_stationary_2wg) {
        append_unique_moe_candidate(block_m_candidates, 128);
    } else if (use_mma_sync_decode) {
        append_unique_moe_candidate(block_m_candidates, requested_mma_m);
    } else {
        append_unique_moe_candidate(block_m_candidates, 64);
    }

    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int block_k = 128;
    const bool split_sfa_tma = get_env<int>("DG_SM90_MOE_SPLIT_SFA_TMA", 0) != 0;
    const bool use_cluster_bcast_b = get_env<int>("DG_SM90_MOE_CLUSTER_BCAST_B") != 0 or use_b_stationary_2wg;
    const int num_sms = device_runtime->get_num_sms();

    std::vector<MegaMoESM90Config> candidates;
    for (const int& block_m: block_m_candidates) {
        DG_HOST_ASSERT(std::any_of(
            layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
            [=](const auto& candidate) { return candidate == block_m; })
        );

        std::vector<int> block_n_candidates;
        if (block_m == 64 and not use_mma_sync_decode and
            use_bn256_split_n_env) {
            append_unique_moe_candidate(block_n_candidates, 256);
        } else {
            append_unique_moe_candidate(block_n_candidates, 128);
        }

        for (const int& block_n: block_n_candidates) {
            std::vector<int> epilogue_wg_candidates;
            if (forced_epilogue_warpgroups > 0) {
                append_unique_moe_candidate(epilogue_wg_candidates, forced_epilogue_warpgroups);
            } else {
                append_unique_moe_candidate(epilogue_wg_candidates,
                    (block_m == 128 or use_b_stationary_2wg or block_n == 256) ? 2 : 1);
            }

            for (const int& num_epilogue_warpgroups: epilogue_wg_candidates) {
                if (block_m % num_epilogue_warpgroups != 0)
                    continue;
                if ((block_m == 16 or block_m == 32) and num_epilogue_warpgroups != 1)
                    continue;
                if (block_m == 128 and num_epilogue_warpgroups != 2)
                    continue;
                if (block_m == 64 and block_n == 256 and num_epilogue_warpgroups != 2)
                    continue;
                const int num_epilogue_threads = num_epilogue_warpgroups * 128;

                if (use_cluster_bcast_b and
                    not ((block_m == 64 and block_n == 128 and num_epilogue_threads == 128) or
                         (block_m == 128 and block_n == 128 and num_epilogue_threads == 256)))
                    continue;
                const int cluster_size = use_cluster_bcast_b ? 2 : 1;
                const int swizzle_acts_mode = (block_m == 16 or block_m == 32) ? 0 : 128;
                const int swizzle_weights_mode = (block_m == 16 or block_m == 32) ? 0 : 128;

                const bool prefer_compact_frontend =
                    block_n == 256 and not split_sfa_tma;
                const bool compact_frontend = get_env<int>("DG_SM90_MOE_COMPACT_FRONTEND",
                                                           prefer_compact_frontend ? 1 : 0) != 0;
                const int forced_dispatch_warps = get_env<int>("DG_SM90_MOE_DISPATCH_WARPS", -1);
                DG_HOST_ASSERT(forced_dispatch_warps == -1 or forced_dispatch_warps == 0 or
                               forced_dispatch_warps == 2 or forced_dispatch_warps == 4 or
                               forced_dispatch_warps == 8);
                std::vector<int> dispatch_warp_candidates;
                if (forced_dispatch_warps > 0) {
                    append_unique_moe_candidate(dispatch_warp_candidates, forced_dispatch_warps);
                } else {
                    append_unique_moe_candidate(dispatch_warp_candidates, compact_frontend ? 2 : 4);
                }

                for (const int& num_dispatch_warps: dispatch_warp_candidates) {
                    if (compact_frontend and num_dispatch_warps != 2)
                        continue;
                    if (split_sfa_tma and compact_frontend)
                        continue;
                    const int num_dispatch_threads = num_dispatch_warps * 32;
                    const int num_non_epilogue_threads = compact_frontend ? 64 : 128;
                    if ((num_dispatch_threads + num_non_epilogue_threads) % 128 != 0)
                        continue;

                    const auto policy = get_sm90_moe_heuristic_policy(
                        num_experts_per_rank, num_tokens, num_topk,
                        intermediate_hidden, block_m, block_n);
                    const bool direct_l2_scatter_default = policy.direct_l2_scatter();
                    const bool l2_nmajor_schedule_default = policy.l2_nmajor_schedule(
                        get_env<int>("DG_SM90_MOE_EPLB_HINT", 0) != 0,
                        get_env<int>("DG_SM90_MOE_SKEW_HINT", 0) != 0);
                    const bool one_warp_cleanup_default = policy.one_warp_cleanup(
                        get_env<int>("DG_SM90_MOE_MASKED_HINT", 0) != 0);
                    const bool direct_l2_scatter_legal =
                        block_m != 16 and block_m != 32 and
                        ((block_m == 64 and block_n == 256 and num_epilogue_warpgroups == 2) or
                         block_n == 128);

                    auto direct_candidates = get_sm90_moe_bool_candidates(
                        "DG_SM90_MOE_DIRECT_L2_SCATTER",
                        direct_l2_scatter_default and direct_l2_scatter_legal);
                    auto l2_nmajor_candidates = get_sm90_moe_bool_candidates(
                        "DG_SM90_MOE_L2_NMAJOR",
                        l2_nmajor_schedule_default);
                    auto cleanup_candidates = get_sm90_moe_bool_candidates(
                        "DG_SM90_MOE_ONE_WARP_CLEANUP",
                        one_warp_cleanup_default);

                    const int default_epw = get_num_experts_per_wave_for_mega_moe_sm90(
                        num_experts_per_rank, num_tokens, num_topk,
                        intermediate_hidden, block_m, block_n, num_sms);
                    std::vector<int> experts_per_wave_candidates;
                    append_unique_moe_candidate(experts_per_wave_candidates, default_epw);

                    for (const int& direct_value: direct_candidates) {
                        const bool direct_l2_scatter = direct_value != 0;
                        if (direct_l2_scatter and not direct_l2_scatter_legal)
                            continue;
                        const int empirical_stage = policy.stage5_pipeline(
                            direct_l2_scatter,
                            get_env<int>("DG_SM90_MOE_EPLB_HINT", 0) != 0,
                            get_env<int>("DG_SM90_MOE_SKEW_HINT", 0) != 0,
                            get_env<int>("DG_SM90_MOE_MASKED_HINT", 0) != 0) ? 5 : 4;
                        const int forced_num_stages = get_env<int>("DG_SM90_MOE_NUM_STAGES");
                        std::vector<int> stage_candidates;
                        if (forced_num_stages > 0) {
                            append_unique_moe_candidate(stage_candidates, forced_num_stages);
                        } else {
                            append_unique_moe_candidate(stage_candidates, empirical_stage);
                        }

                        for (const int& requested_num_stages: stage_candidates) {
                            const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
                                SM90ArchSpec::smem_capacity,
                                num_experts, hidden,
                                block_m, block_n, block_k,
                                num_dispatch_threads / 32, num_epilogue_threads / 32,
                                direct_l2_scatter,
                                requested_num_stages);
                            for (const int& l2_nmajor_value: l2_nmajor_candidates) {
                                for (const int& cleanup_value: cleanup_candidates) {
                                    for (const int& num_experts_per_wave: experts_per_wave_candidates) {
                                        if (num_experts_per_wave <= 0 or
                                            num_experts_per_wave > num_experts_per_rank or
                                            num_experts_per_rank % num_experts_per_wave != 0)
                                            continue;
                                        candidates.emplace_back(MegaMoESM90Config {
                                            block_m, block_n, block_k,
                                            cluster_size,
                                            num_max_pool_tokens, num_padded_sf_pool_tokens,
                                            swizzle_acts_mode, swizzle_weights_mode,
                                            num_experts_per_wave,
                                            num_stages, smem_size,
                                            num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
                                            direct_l2_scatter, l2_nmajor_value != 0, cleanup_value != 0
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    DG_HOST_ASSERT(not candidates.empty());
    return candidates;
}

static Sm90MoeConfigInfo get_best_mega_moe_config_info_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto candidates = get_mega_moe_config_candidates_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);
    const int num_sms = device_runtime->get_num_sms();

    Sm90MoeConfigInfo best {
        std::numeric_limits<int64_t>::max(), 0, 0, 0, 0, candidates[0]
    };
    for (const auto& candidate: candidates) {
        const auto policy = get_sm90_moe_heuristic_policy(
            num_experts_per_rank, num_tokens, num_topk,
            intermediate_hidden, candidate.block_m, candidate.block_n);
        const bool empirical_direct_l2_scatter = policy.direct_l2_scatter();
        const bool empirical_l2_nmajor_schedule = policy.l2_nmajor_schedule(
            get_env<int>("DG_SM90_MOE_EPLB_HINT", 0) != 0,
            get_env<int>("DG_SM90_MOE_SKEW_HINT", 0) != 0);
        const bool empirical_one_warp_cleanup = policy.one_warp_cleanup(
            get_env<int>("DG_SM90_MOE_MASKED_HINT", 0) != 0);
        const int empirical_num_stages = policy.stage5_pipeline(
            candidate.direct_l2_scatter,
            get_env<int>("DG_SM90_MOE_EPLB_HINT", 0) != 0,
            get_env<int>("DG_SM90_MOE_SKEW_HINT", 0) != 0,
            get_env<int>("DG_SM90_MOE_MASKED_HINT", 0) != 0) ? 5 : 4;
        const int empirical_num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
            num_experts_per_rank, num_tokens, num_topk,
            intermediate_hidden, candidate.block_m, candidate.block_n, num_sms);
        auto info = get_sm90_moe_config_info(
            candidate,
            num_experts_per_rank, num_tokens, num_topk,
            hidden, intermediate_hidden, num_sms,
            empirical_direct_l2_scatter,
            empirical_l2_nmajor_schedule,
            empirical_one_warp_cleanup,
            empirical_num_stages,
            empirical_num_experts_per_wave);
        if (info.score < best.score)
            best = info;
    }
    return best;
}

static MegaMoESM90Config get_mega_moe_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto config_info = get_best_mega_moe_config_info_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);
    const auto config = config_info.config;

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
