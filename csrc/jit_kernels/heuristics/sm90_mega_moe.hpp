#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>
#include <unordered_set>

#include "../../utils/exception.hpp"

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../utils/math.hpp"
#include "sm90.hpp"

namespace deep_gemm {

// SM90 uses register-resident WGMMA accumulators and FP8 weights with per-128
// float scale factors; it has no TMEM, FP4, cluster MMA, or TMA multicast.

struct MegaMoESM90Config {
    // Block tiling (no STORE_BLOCK_M / SF_BLOCK_M concept on SM90)
    int block_m, block_n, block_k;

    // Pool capacity, allocated SF capacity, and per-config logical SF stride.
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens, sf_pool_stride_tokens;

    int num_experts_per_wave;

    int num_sms;

    int num_stages, smem_size;

    // Thread layout: dispatch + non-epilogue (TMA) + epilogue (math)
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Scheduler and epilogue modes selected for this phase.
    bool direct_l2_scatter, nmajor_schedule, one_warp_cleanup, swap_ab;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", sf_pool_stride_tokens=" << config.sf_pool_stride_tokens
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_sms=" << config.num_sms
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", direct_l2_scatter=" << config.direct_l2_scatter
           << ", nmajor_schedule=" << config.nmajor_schedule
           << ", one_warp_cleanup=" << config.one_warp_cleanup
           << ", swap_ab=" << config.swap_ab << ")";
        return os;
    }
};

enum class Sm90MoeHardwareProfile {
    LowSm,
    HighSm
};

struct Sm90MoeHeuristicInput {
    int launch_num_sms;

    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;
};

struct Sm90MoeNumericalConfig {
    bool bf16_scaled_accum = false;
};

struct Sm90MoeLaunchConfig {
    MegaMoESM90Config l1;
    MegaMoESM90Config l2;
    Sm90MoeNumericalConfig numerical;
};

static Sm90MoeHardwareProfile classify_sm90_moe_hardware(
    const int launch_num_sms) {
    DG_HOST_ASSERT(launch_num_sms > 0);
    return launch_num_sms < 100
        ? Sm90MoeHardwareProfile::LowSm
        : Sm90MoeHardwareProfile::HighSm;
}

enum class Sm90MoeShapeFamily {
    Generic,
    Compact,
    Wide
};

struct Sm90MoeLoad {
    int64_t routed_tokens;
    int64_t local_experts;

    bool valid() const {
        return routed_tokens >= 0 and local_experts > 0;
    }

    bool less_than(const int64_t value) const {
        return routed_tokens < value * local_experts;
    }

    bool less_equal(const int64_t value) const {
        return routed_tokens <= value * local_experts;
    }

    bool greater_than(const int64_t value) const {
        return routed_tokens > value * local_experts;
    }

    bool greater_equal(const int64_t value) const {
        return routed_tokens >= value * local_experts;
    }

    bool in_open_closed(const int64_t low, const int64_t high) const {
        return greater_than(low) and less_equal(high);
    }

    bool in_closed(const int64_t low, const int64_t high) const {
        return greater_equal(low) and less_equal(high);
    }
};

static Sm90MoeLoad get_sm90_moe_load(const Sm90MoeHeuristicInput& input) {
    const Sm90MoeLoad load {
        static_cast<int64_t>(input.num_tokens) * input.num_topk,
        input.num_experts_per_rank,
    };
    DG_HOST_ASSERT(load.valid());
    return load;
}

static Sm90MoeShapeFamily classify_sm90_moe_shape(
    const int hidden, const int intermediate_hidden) {
    if (hidden >= 3072 and hidden < 5120 and
        intermediate_hidden >= 1536 and intermediate_hidden < 2560)
        return Sm90MoeShapeFamily::Compact;
    if (hidden >= 5120 and hidden <= 8192 and
        intermediate_hidden >= 2560 and intermediate_hidden <= 4096)
        return Sm90MoeShapeFamily::Wide;
    return Sm90MoeShapeFamily::Generic;
}

static bool should_use_swap_ab_for_mega_moe_sm90(
    const Sm90MoeLoad& load,
    const Sm90MoeShapeFamily shape_family,
    const int block_m,
    const int num_epilogue_threads) {
    const int max_load = shape_family == Sm90MoeShapeFamily::Compact ? 24 : 16;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    return decode_split_n_path and load.greater_than(0) and
           load.less_equal(max_load);
}

static int get_generic_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n,
    const int& num_sms) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (expected_tokens_per_expert < 1)
        return num_experts_per_rank;

    constexpr int kImbalanceFactor = 2;
    const int num_m_blocks = ceil_div(
        static_cast<int>(std::ceil(expected_tokens_per_expert)), block_m);
    const int num_n_blocks = (2 * intermediate_hidden) / block_n;
    const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;
    const int min_num_experts_per_wave = num_l1_blocks_per_expert > 0
        ? ceil_div(kImbalanceFactor * num_sms, num_l1_blocks_per_expert) : 1;
    if (min_num_experts_per_wave >= num_experts_per_rank)
        return num_experts_per_rank;
    if (num_l1_blocks_per_expert >= num_sms)
        return min_num_experts_per_wave;

    const int max_num_experts_per_wave = std::min(
        num_experts_per_rank, min_num_experts_per_wave * 2);
    int best_num_experts_per_wave = min_num_experts_per_wave;
    float best_tail_ratio = -1.0f;
    for (int num_experts_per_wave = min_num_experts_per_wave;
         num_experts_per_wave <= max_num_experts_per_wave;
         ++num_experts_per_wave) {
        const int remainder = num_experts_per_rank % num_experts_per_wave;
        const float tail_ratio = remainder == 0
            ? 1.0f : static_cast<float>(remainder) / num_experts_per_wave;
        if (tail_ratio > best_tail_ratio) {
            best_tail_ratio = tail_ratio;
            best_num_experts_per_wave = num_experts_per_wave;
        }
    }
    return best_num_experts_per_wave;
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const Sm90MoeLoad& load,
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    if (block_m == 64 and (load.less_than(1) or load.greater_than(4))) {
        return num_experts_per_rank;
    }
    return get_generic_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

static int normalize_num_experts_per_wave_for_mega_moe_sm90(
    const int num_experts_per_rank,
    const int requested_num_experts_per_wave) {
    DG_HOST_ASSERT(num_experts_per_rank > 0);
    DG_HOST_ASSERT(requested_num_experts_per_wave > 0 and
                   requested_num_experts_per_wave <= num_experts_per_rank);
    if (num_experts_per_rank % requested_num_experts_per_wave == 0)
        return requested_num_experts_per_wave;

    // The tuned schedules use exact divisors. For a generic, previously unseen
    // expert count, round upward to the first divisor so the fallback remains
    // legal without reducing the amount of work available in a wave.
    for (int candidate = requested_num_experts_per_wave + 1;
         candidate < num_experts_per_rank; ++ candidate) {
        if (num_experts_per_rank % candidate == 0)
            return candidate;
    }
    return num_experts_per_rank;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& direct_l2_scatter_enabled = false,
    const int& default_num_stages = 0,
    const bool& swap_ab = false,
    const bool& require_exact_default_stages = false) {
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
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        block_m, block_n, num_epilogue_warpgroups);
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    const int smem_cd_l1 = num_epilogue_warpgroups * wg_block_m * (wg_block_n / 2);  // 1 byte/elem (FP8)
    const bool direct_l2_scatter =
        direct_l2_scatter_enabled and not swap_ab and wg_block_n == 128;
    const int smem_cd_l2 = direct_l2_scatter ? 0 :
        num_epilogue_warpgroups * wg_block_m * wg_block_n *
            static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_swap_l1 = swap_ab
        ? block_m * (block_n / 2) *
              (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int smem_cd = align(
        std::max(std::max(smem_cd_l1, smem_cd_l2), smem_cd_swap_l1),
        kSmemAlignment);

    // SF on SM90:
    //   * SFA per stage must hold one aligned BLOCK_M-float vector for every
    //     per-64-K L2 scale group (two for BK128, four for BK256)
    //   * SFB is loaded directly from global by the math warpgroup (block-(128,128)
    //     weight quantization), so no SMEM is reserved for it.
    const int smem_sfa_half_stride_bytes = align(block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfa_per_stage =
        (block_k / 64) * smem_sfa_half_stride_bytes;
    // Per-stage: A tile + B tile + SFA tile. Weight SF is loaded directly.
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage;

    // Barriers (8 bytes each):
    //   * dispatch: num_dispatch_warps
    //   * GEMM full + empty: 2 * num_stages
    //   * combine: 2 * num_epilogue_warps
    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = 2 * 8;

    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed;

    const int max_num_stages = (smem_capacity - smem_fixed) /
                               (smem_per_stage + smem_barriers_per_stage);
    if (max_num_stages < 2)
        return {0, 0};
    const bool prefer_bn256_n_tile = block_n == 256;
    if (require_exact_default_stages and default_num_stages > max_num_stages)
        return {0, 0};
    const int preferred_num_stages = default_num_stages > 0
        ? std::min(default_num_stages, max_num_stages)
        : (prefer_bn256_n_tile ? std::min(4, max_num_stages) : 0);
    const int num_stages = preferred_num_stages > 0 ?
        preferred_num_stages : max_num_stages;
    if (num_stages < 2 or num_stages > max_num_stages)
        return {0, 0};
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
}

// Mirror the device kernel's pre-barrier alias region so invalid combine
// vectorization is rejected before NVRTC sees a shape specialization.
static uint32_t get_sm90_moe_pre_barrier_smem_size_for_combine(
    const Sm90MoeHeuristicInput& input,
    const MegaMoESM90Config& config) {
    constexpr int kSmemAlignment = 1024;
    const int num_dispatch_warps = config.num_dispatch_threads / 32;
    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);

    const int smem_expert_count = align(
        input.num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers = align(
        static_cast<int>(layout::Buffer(
            layout::Data(input.hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_cd_l2 = config.direct_l2_scatter ? 0 :
        num_epilogue_warpgroups * static_cast<int>(wg_layout.block_m) *
        static_cast<int>(wg_layout.block_n) * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_swap_l1 = config.swap_ab ?
        config.block_m * (config.block_n / 2) *
            (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t))) : 0;
    const int smem_cd = align(
        std::max(smem_cd_l2, smem_cd_swap_l1), kSmemAlignment);
    const int smem_gemm = config.num_stages *
        (config.block_m * config.block_k + config.block_n * config.block_k);
    return static_cast<uint32_t>(
        smem_expert_count + smem_send_buffers + smem_cd + smem_gemm);
}

static bool supports_sm90_moe_bf16_scaled_accum(
    const MegaMoESM90Config& config) {
    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    if (config.swap_ab) {
        return config.block_m == 64 and config.block_n == 128 and
               config.block_k == 128 and num_epilogue_warpgroups == 2;
    }
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    return config.block_m == 64 and
           (wg_block_n == 128 or wg_block_n == 256) and
           (config.block_k == 128 or config.block_k == 256);
}

static bool is_sm90_moe_phase_config_legal(
    const Sm90MoeHeuristicInput& input,
    const MegaMoESM90Config& config,
    const int phase_n,
    const bool require_exact_pipeline) {
    if ((config.block_m != 64 and config.block_m != 128) or
        (config.block_n != 128 and config.block_n != 256 and config.block_n != 512) or
        (config.block_k != 128 and config.block_k != 256))
        return false;
    if (phase_n % config.block_n != 0 or
        input.hidden % config.block_k != 0 or
        input.intermediate_hidden % config.block_k != 0)
        return false;
    if (config.num_sms <= 0 or config.num_sms > input.launch_num_sms)
        return false;
    if (config.num_experts_per_wave <= 0 or
        config.num_experts_per_wave > input.num_experts_per_rank or
        input.num_experts_per_rank % config.num_experts_per_wave != 0)
        return false;
    if ((config.num_dispatch_threads != 64 and config.num_dispatch_threads % 128 != 0) or
        (config.num_non_epilogue_threads != 64 and config.num_non_epilogue_threads != 128) or
        config.num_epilogue_threads <= 0 or config.num_epilogue_threads % 128 != 0 or
        config.num_dispatch_threads + config.num_non_epilogue_threads +
            config.num_epilogue_threads > 1024)
        return false;

    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);
    const int split_m = static_cast<int>(wg_layout.split_m);
    const int split_n_count = static_cast<int>(wg_layout.split_n_count);
    if (split_m <= 0 or split_n_count <= 0 or
        config.block_m % split_m != 0 or config.block_n % split_n_count != 0)
        return false;
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    if (wg_block_m != 64 or
        (wg_block_n != 64 and wg_block_n != 128 and wg_block_n != 256))
        return false;
    // WG_BLOCK_N=64 is supported only by the swap-AB path.
    if (wg_block_n == 64 and not config.swap_ab)
        return false;
    if (config.block_m == 64 and config.block_n == 256 and
        num_epilogue_warpgroups != 2)
        return false;
    if (config.swap_ab and not (
            config.block_m == 64 and config.block_n == 128 and
            config.block_k == 128 and num_epilogue_warpgroups == 2))
        return false;
    if (config.direct_l2_scatter and (config.swap_ab or wg_block_n != 128))
        return false;
    if (config.smem_size <= 0 or config.smem_size > SM90ArchSpec::smem_capacity)
        return false;
    if (not require_exact_pipeline)
        return true;

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        config.block_m, config.block_n, config.block_k,
        config.num_dispatch_threads / 32, config.num_epilogue_threads / 32,
        config.direct_l2_scatter,
        config.num_stages,
        config.swap_ab,
        true);
    return num_stages == config.num_stages and smem_size == config.smem_size;
}

static bool is_sm90_moe_launch_config_legal(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLaunchConfig& config,
    const bool require_exact_pipeline = true) {
    if (config.l1.num_sms != input.launch_num_sms or
        config.l2.num_sms != input.launch_num_sms)
        return false;
    if (not is_sm90_moe_phase_config_legal(
            input, config.l1, 2 * input.intermediate_hidden,
            require_exact_pipeline) or
        not is_sm90_moe_phase_config_legal(
            input, config.l2, input.hidden, require_exact_pipeline))
        return false;
    // The four-math-warpgroup BN512 L1 path relies on packed BF16 partial
    // accumulation to stay within the CTA register budget.
    if (config.l1.block_n == 512 and
        not config.numerical.bf16_scaled_accum)
        return false;
    return not config.numerical.bf16_scaled_accum or
           (supports_sm90_moe_bf16_scaled_accum(config.l1) and
            supports_sm90_moe_bf16_scaled_accum(config.l2));
}

static MegaMoESM90Config make_generic_mega_moe_config_sm90(
    const Sm90MoeHeuristicInput& input,
    const bool conservative = false) {
    const int block_m = 64;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        input.num_ranks, input.num_max_tokens_per_rank,
        input.num_topk, input.num_experts_per_rank);
    const int block_k = 128;
    const auto load = get_sm90_moe_load(input);
    const auto shape_family = classify_sm90_moe_shape(
        input.hidden, input.intermediate_hidden);

    const bool prefer_swap_ab_block =
        not conservative and
        should_use_swap_ab_for_mega_moe_sm90(
            load, shape_family, block_m, 256);
    const int block_n = prefer_swap_ab_block ? 128 :
        (conservative ? 128 : 256);
    const bool prefer_swap_ab_shape = prefer_swap_ab_block and block_n == 128;
    const int num_epilogue_warpgroups =
        (block_n == 256 or prefer_swap_ab_shape) ? 2 : 1;
    DG_HOST_ASSERT(block_m % num_epilogue_warpgroups == 0);
    DG_HOST_ASSERT(block_m != 128 or
                   (block_n == 128 and num_epilogue_warpgroups == 2));
    DG_HOST_ASSERT(block_m != 64 or block_n != 256 or
                   num_epilogue_warpgroups == 2);
    const int num_epilogue_threads = num_epilogue_warpgroups * 128;
    const bool swap_ab =
        not conservative and block_n == 128 and
        should_use_swap_ab_for_mega_moe_sm90(
            load, shape_family, block_m, num_epilogue_threads);

    const bool compact_frontend = block_n >= 256 or swap_ab;
    const int num_dispatch_warps = compact_frontend ? 2 : 4;
    DG_HOST_ASSERT(not compact_frontend or num_dispatch_warps == 2);
    const int num_dispatch_threads = num_dispatch_warps * 32;
    const int num_non_epilogue_threads = compact_frontend ? 64 : 128;
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    constexpr bool direct_l2_scatter = false;
    constexpr bool l2_nmajor_schedule = false;
    constexpr bool one_warp_cleanup = false;

    const int requested_num_experts_per_wave = conservative ?
        get_generic_num_experts_per_wave_for_mega_moe_sm90(
            input.num_experts_per_rank, input.num_tokens, input.num_topk,
            input.intermediate_hidden, block_m, block_n, input.launch_num_sms) :
        get_num_experts_per_wave_for_mega_moe_sm90(
            load,
            input.num_experts_per_rank, input.num_tokens, input.num_topk,
            input.intermediate_hidden, block_m, block_n, input.launch_num_sms);
    const int num_experts_per_wave =
        normalize_num_experts_per_wave_for_mega_moe_sm90(
            input.num_experts_per_rank, requested_num_experts_per_wave);
    DG_HOST_ASSERT(num_experts_per_wave > 0 and
                   num_experts_per_wave <= input.num_experts_per_rank and
                   input.num_experts_per_rank % num_experts_per_wave == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        direct_l2_scatter,
        0,
        swap_ab);
    DG_HOST_ASSERT(num_stages >= 2 and smem_size > 0);
    const int sf_pool_stride_tokens =
        layout::get_num_sf_ring_tokens(num_max_pool_tokens, block_m);
    return {
        block_m, block_n, block_k,
        num_max_pool_tokens, input.num_padded_sf_pool_tokens, sf_pool_stride_tokens,
        num_experts_per_wave,
        input.launch_num_sms,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        direct_l2_scatter, l2_nmajor_schedule, one_warp_cleanup, swap_ab
    };
}

static bool select_sm90_moe_high_sm_bf16_scaled_accum(
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load) {
    if (shape_family == Sm90MoeShapeFamily::Compact) {
        return load.less_equal(3) or
               (load.greater_than(6) and load.less_equal(384));
    }
    if (shape_family == Sm90MoeShapeFamily::Wide) {
        return load.less_equal(4) or load.greater_than(8);
    }
    return false;
}

struct Sm90MoePhaseTuning {
    int block_m, block_n, block_k;
    int num_experts_per_wave, num_stages;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;
    bool direct_l2_scatter, nmajor_schedule, one_warp_cleanup, swap_ab;
};

struct Sm90MoeScheduleTuning {
    bool selected;
    Sm90MoePhaseTuning l1, l2;
};

static Sm90MoePhaseTuning make_sm90_moe_phase_tuning(
    const MegaMoESM90Config& config) {
    return {
        config.block_m, config.block_n, config.block_k,
        config.num_experts_per_wave, config.num_stages,
        config.num_dispatch_threads, config.num_non_epilogue_threads,
        config.num_epilogue_threads,
        config.direct_l2_scatter, config.nmajor_schedule,
        config.one_warp_cleanup, config.swap_ab,
    };
}

static void set_sm90_moe_specialized_phase_tuning(
    Sm90MoePhaseTuning& tuning,
    const int block_n,
    const int block_k,
    const int num_experts_per_wave,
    const int num_stages) {
    tuning.block_m = 64;
    tuning.block_n = block_n;
    tuning.block_k = block_k;
    tuning.num_experts_per_wave = num_experts_per_wave;
    tuning.num_stages = num_stages;
    tuning.num_dispatch_threads = block_n >= 256 ? 64 : 128;
    tuning.num_non_epilogue_threads = block_n >= 256 ? 64 : 128;
    tuning.num_epilogue_threads = block_n >= 256 ? block_n : 128;
    tuning.direct_l2_scatter = false;
    tuning.nmajor_schedule = false;
    tuning.one_warp_cleanup = false;
    tuning.swap_ab = false;
}

static int derive_sm90_moe_tuned_epw(
    const int num_experts_per_rank,
    const int target_num_experts_per_wave) {
    const int requested = std::min(
        num_experts_per_rank, target_num_experts_per_wave);
    return normalize_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, requested);
}

static Sm90MoeScheduleTuning make_sm90_moe_schedule_tuning(
    const Sm90MoeLaunchConfig& generic) {
    return {
        false,
        make_sm90_moe_phase_tuning(generic.l1),
        make_sm90_moe_phase_tuning(generic.l2),
    };
}

static bool uses_sm90_moe_bn256_frontend(
    const Sm90MoePhaseTuning& tuning) {
    return tuning.block_m == 64 and tuning.block_n == 256;
}

static Sm90MoeScheduleTuning select_sm90_moe_low_sm_tuning(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load,
    Sm90MoeScheduleTuning tuning) {
    if (shape_family == Sm90MoeShapeFamily::Compact and
        load.greater_than(8) and load.less_than(16)) {
        const int tuned_epw = derive_sm90_moe_tuned_epw(
            input.num_experts_per_rank, 16);
        tuning.selected = true;
        tuning.l1.num_experts_per_wave = tuned_epw;
        tuning.l2.num_experts_per_wave = tuned_epw;
    }

    if (not uses_sm90_moe_bn256_frontend(tuning.l1) or
        not uses_sm90_moe_bn256_frontend(tuning.l2))
        return tuning;

    int target_epw = 0;
    int num_stages = 4;
    bool direct_l2_scatter = false;
    bool l2_nmajor = false;
    bool one_warp_cleanup = false;
    bool range_selected = false;

    if (shape_family == Sm90MoeShapeFamily::Compact) {
        if (load.in_open_closed(16, 32)) {
            range_selected = true;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = true;
        } else if (load.greater_than(48) and load.less_than(96)) {
            range_selected = true;
            direct_l2_scatter = true;
            num_stages = 4;
        } else if (load.in_closed(96, 160)) {
            range_selected = true;
            target_epw = 8;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = true;
        } else if (load.in_open_closed(160, 384)) {
            range_selected = true;
            target_epw = 16;
            num_stages = 5;
            direct_l2_scatter = true;
            l2_nmajor = load.greater_equal(256);
        } else if (load.greater_than(384)) {
            range_selected = true;
            target_epw = 32;
            num_stages = 5;
            direct_l2_scatter = true;
            l2_nmajor = true;
        }
    } else if (shape_family == Sm90MoeShapeFamily::Wide) {
        if (load.in_open_closed(16, 48)) {
            range_selected = true;
            target_epw = 16;
        } else if (load.in_open_closed(48, 96)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = false;
        } else if (load.in_open_closed(96, 192)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = true;
        } else if (load.in_open_closed(192, 384)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            direct_l2_scatter = true;
        } else if (load.greater_than(384)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            num_stages = 5;
            direct_l2_scatter = true;
        }
    }

    if (not range_selected)
        return tuning;

    tuning.selected = true;
    if (target_epw > 0) {
        const int tuned_epw = derive_sm90_moe_tuned_epw(
            input.num_experts_per_rank, target_epw);
        tuning.l1.num_experts_per_wave = tuned_epw;
        tuning.l2.num_experts_per_wave = tuned_epw;
    }
    tuning.l1.num_stages = tuning.l2.num_stages = num_stages;
    tuning.l1.direct_l2_scatter = tuning.l2.direct_l2_scatter =
        direct_l2_scatter;
    tuning.l1.nmajor_schedule = false;
    tuning.l2.nmajor_schedule = l2_nmajor;
    tuning.l1.one_warp_cleanup = tuning.l2.one_warp_cleanup =
        one_warp_cleanup;
    return tuning;
}

static Sm90MoeScheduleTuning select_sm90_moe_high_sm_tuning(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load,
    Sm90MoeScheduleTuning tuning) {
    int l1_block_n = 256, l1_block_k = 128, l1_epw = 0, l1_stages = 0;
    int l2_block_n = 256, l2_block_k = 128, l2_epw = 0, l2_stages = 0;
    bool l1_nmajor = false, one_warp_cleanup = false;
    bool specialized = false;

    if (shape_family == Sm90MoeShapeFamily::Compact) {
        if (load.in_open_closed(12, 32)) {
            specialized = true;
            l1_epw = l2_epw = 4;
            l1_stages = l2_stages = 3;
        } else if (load.in_open_closed(128, 256)) {
            specialized = true;
            l1_epw = l2_epw = 32;
            l1_stages = l2_stages = 4;
        } else if (load.greater_than(1024)) {
            specialized = true;
            l1_epw = l2_epw = 32;
            l1_stages = l2_stages = 4;
        }
    } else if (shape_family == Sm90MoeShapeFamily::Wide) {
        if (load.in_open_closed(8, 24)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
        } else if (load.in_open_closed(24, 48)) {
            specialized = true;
            l1_block_k = 256;
            l1_epw = 8;
            l2_epw = 48;
            l1_stages = 2;
            l2_stages = 3;
        } else if (load.in_open_closed(48, 96)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
            l1_nmajor = true;
        } else if (load.in_open_closed(96, 192)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
            l1_nmajor = true;
            one_warp_cleanup = true;
        } else if (load.greater_than(192)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
            one_warp_cleanup = true;
        }
    }
    if (not specialized)
        return tuning;

    l1_epw = derive_sm90_moe_tuned_epw(
        input.num_experts_per_rank, l1_epw);
    l2_epw = derive_sm90_moe_tuned_epw(
        input.num_experts_per_rank, l2_epw);
    tuning.selected = true;
    set_sm90_moe_specialized_phase_tuning(
        tuning.l1, l1_block_n, l1_block_k, l1_epw, l1_stages);
    set_sm90_moe_specialized_phase_tuning(
        tuning.l2, l2_block_n, l2_block_k, l2_epw, l2_stages);
    tuning.l1.nmajor_schedule = l1_nmajor;
    tuning.l2.nmajor_schedule = true;
    tuning.l2.one_warp_cleanup = one_warp_cleanup;
    return tuning;
}

static Sm90MoeScheduleTuning select_sm90_moe_tuning(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeHardwareProfile hardware_profile,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load,
    const Sm90MoeLaunchConfig& generic) {
    auto tuning = make_sm90_moe_schedule_tuning(generic);
    if (hardware_profile == Sm90MoeHardwareProfile::LowSm) {
        return select_sm90_moe_low_sm_tuning(
            input, shape_family, load, tuning);
    }
    if (hardware_profile == Sm90MoeHardwareProfile::HighSm) {
        return select_sm90_moe_high_sm_tuning(
            input, shape_family, load, tuning);
    }
    return tuning;
}

static bool try_materialize_sm90_moe_phase_tuning(
    const Sm90MoeHeuristicInput& input,
    MegaMoESM90Config& config,
    const Sm90MoePhaseTuning& tuning) {
    config.block_m = tuning.block_m;
    config.block_n = tuning.block_n;
    config.block_k = tuning.block_k;
    config.num_experts_per_wave = tuning.num_experts_per_wave;
    config.num_sms = input.launch_num_sms;
    config.num_dispatch_threads = tuning.num_dispatch_threads;
    config.num_non_epilogue_threads = tuning.num_non_epilogue_threads;
    config.num_epilogue_threads = tuning.num_epilogue_threads;
    config.direct_l2_scatter = tuning.direct_l2_scatter;
    config.nmajor_schedule = tuning.nmajor_schedule;
    config.one_warp_cleanup = tuning.one_warp_cleanup;
    config.swap_ab = tuning.swap_ab;
    config.sf_pool_stride_tokens = layout::get_num_sf_ring_tokens(
        config.num_max_pool_tokens, config.block_m);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        config.block_m, config.block_n, config.block_k,
        config.num_dispatch_threads / 32, config.num_epilogue_threads / 32,
        config.direct_l2_scatter,
        tuning.num_stages,
        config.swap_ab,
        true);
    if (num_stages == 0 or smem_size == 0)
        return false;
    config.num_stages = num_stages;
    config.smem_size = smem_size;
    return true;
}

static bool try_apply_sm90_moe_tuning(
    const Sm90MoeHeuristicInput& input,
    Sm90MoeScheduleTuning tuning,
    Sm90MoeLaunchConfig& config) {
    if (not tuning.selected)
        return true;
    return try_materialize_sm90_moe_phase_tuning(input, config.l1, tuning.l1) and
           try_materialize_sm90_moe_phase_tuning(input, config.l2, tuning.l2);
}

static Sm90MoeLaunchConfig select_mega_moe_sm90(
    const Sm90MoeHeuristicInput& input) {
    const auto hardware_profile = classify_sm90_moe_hardware(input.launch_num_sms);
    const auto shape_family = classify_sm90_moe_shape(
        input.hidden, input.intermediate_hidden);
    const auto load = get_sm90_moe_load(input);

    auto generic = make_generic_mega_moe_config_sm90(input);
    Sm90MoeLaunchConfig result {
        generic, generic,
        {}
    };
    result.l1.nmajor_schedule = false;

    // Shapes that cannot use the BN256 frontend fall back to the conservative
    // BN128 route before hardware-specific range tuning is applied.
    if (not is_sm90_moe_launch_config_legal(input, result, false)) {
        generic = make_generic_mega_moe_config_sm90(input, true);
        result.l1 = result.l2 = generic;
        result.l1.nmajor_schedule = false;
    }
    DG_HOST_ASSERT(is_sm90_moe_launch_config_legal(input, result, false));

    auto candidate = result;
    candidate.numerical.bf16_scaled_accum =
        hardware_profile == Sm90MoeHardwareProfile::HighSm and
        select_sm90_moe_high_sm_bf16_scaled_accum(shape_family, load);
    const auto tuning = select_sm90_moe_tuning(
        input, hardware_profile, shape_family, load, result);
    if (try_apply_sm90_moe_tuning(input, tuning, candidate) and
        is_sm90_moe_launch_config_legal(input, candidate, tuning.selected)) {
        result = candidate;
    }
    if (deep_jit::get_env<int>("DG_JIT_DEBUG") or deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "Sm90MoeLaunchConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            input.num_ranks, input.num_experts, input.hidden,
            input.intermediate_hidden, input.num_max_tokens_per_rank,
            input.num_tokens, input.num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": profile=" << static_cast<int>(hardware_profile)
                      << ", shape_family=" << static_cast<int>(shape_family)
                      << ", l1=" << result.l1
                      << ", l2=" << result.l2
                      << ", bf16_scaled_accum=" << result.numerical.bf16_scaled_accum
                      << std::endl;
            printed.insert(key);
        }
    }
    return result;
}

} // namespace deep_gemm
