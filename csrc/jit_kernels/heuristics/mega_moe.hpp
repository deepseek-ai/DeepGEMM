#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/types.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm100.hpp"

namespace deep_gemm {

constexpr int kMegaMoESmallTokenBlockM = 16;
// Packet weights are physically grouped as pairs of 128-wide N tiles.
// Keep the Python packer and host validation in sync if this changes.
constexpr int kMegaMoEPacketBlockN = 128;

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Ring capacity and SF ring token count
    int num_ring_tokens;
    int num_sf_ring_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_ring_tokens=" << config.num_ring_tokens
           << ", num_sf_ring_tokens=" << config.num_sf_ring_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull << ")";
        return os;
    }
};

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    DG_HOST_ASSERT(mma_type_str == "fp8xfp4");
    return MmaKind::MXFP8FP4;
}

static int get_num_mma_elem_bytes(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::BF16 ? 2 : 1;
}

static bool is_mma_with_sf(const MmaKind& mma_kind) {
    return mma_kind == MmaKind::MXFP8FP4;
}

static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens,
    const MmaKind& mma_kind,
    const bool tile_packed_weights = false) {
    auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups] = [&]() -> std::tuple<int, int, int, int, int> {
        float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
        if (num_expected_tokens_per_expert <= 8.5) {
            // Really small token-per-expert (e.g. RL long-tail rollout), use the smallest block_m and larger BLOCK_K for less synchronization
            return {
                2, kMegaMoESmallTokenBlockM, 8, 256,
                tile_packed_weights ? 1 : 2};
        } else if (num_expected_tokens_per_expert <= 16.5) {
            // Small batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 128
            return {2, 32, 16, 128, 2};
        } else if (num_expected_tokens_per_expert <= 32.5) {
            // Medium batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 256
            return {2, 64, 32, 128, 1};
        } else if (num_expected_tokens_per_expert <= 64.5) {
            // Large batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 512
            return {2, 96, 16, 128, 2};
        } else if (num_expected_tokens_per_expert <= 96.5) {
            // Medium batch size, Medium EP, decoding, e.g. 6/384 experts, EP16, bsz 256, or EP32, bsz128
            return {2, 128, 32, 128, 2};
        } else {
            // Prefill, or large EP decoding
            return {2, 192, 32, 128, 2};
        }
    }();
    block_k /= get_num_mma_elem_bytes(mma_kind);

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    // Return configs
    return {cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, 
    const int& num_bytes_per_pull, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const MmaKind& mma_kind) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;
    const int num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 output staging and L2 BF16 staging.
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages * get_num_mma_elem_bytes(mma_kind);
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Schedule task payloads
    constexpr int kNumScheduleStages = 2;
    const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    // + schedule task publish full/empty barriers.
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 + kNumScheduleStages * 2) * 8;

    // Amax warp-pair reduction buffer for SwiGLU's cross-warp amax exchange.
    const int smem_amax_reduction = is_mma_with_sf(mma_kind) ?
        store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float)) : 0;

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = is_mma_with_sf(mma_kind) ? sf_block_m * (block_k / gran_k) : 0;
    const int smem_sfb_per_stage = is_mma_with_sf(mma_kind) ? sf_block_n * (block_k / gran_k) : 0;

    // Per-stage: A tile + B tile + optional SF tiles + full/empty barriers.
    const int smem_a_size_per_stage = load_block_m * block_k * num_mma_elem_bytes;
    const int smem_b_size_per_stage = block_n * block_k * num_mma_elem_bytes;
    DG_HOST_ASSERT(smem_a_size_per_stage % kSmemAlignment == 0);
    DG_HOST_ASSERT(smem_b_size_per_stage % kSmemAlignment == 0);
    const int smem_stage_barriers = 2 * 8;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers +
        smem_task_info + smem_tmem_ptr;

    // Select maximum number of stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind,
    const bool tile_packed_weights = false) {

    // Block config
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
            num_tokens, mma_kind, tile_packed_weights);
    const int block_n = kMegaMoEPacketBlockN;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = is_mma_with_sf(mma_kind) ?
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4) : std::pair(0, 0);
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = 32;

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pull: divide token bytes by 2 until <= kPullThreshold
    constexpr int kPullThreshold = 4096;
    int num_bytes_per_pull = hidden * get_num_mma_elem_bytes(mma_kind);
    while (num_bytes_per_pull > kPullThreshold) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        mma_kind);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_ring_tokens, is_mma_with_sf(mma_kind) ? num_sf_ring_tokens : 0,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull
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

} // namespace deep_gemm
