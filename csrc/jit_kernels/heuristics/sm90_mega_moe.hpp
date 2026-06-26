#pragma once

#include <algorithm>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm90.hpp"

// SM90 MegaMoE heuristics live in their own translation-unit header so the
// SM100 path (`mega_moe.hpp`) stays untouched. We include `mega_moe.hpp` only
// to reuse the shared `get_num_experts_per_wave_for_mega_moe` wave search.
#include "mega_moe.hpp"

namespace deep_gemm {

// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3, scales are per-128 channel float.
//   - No 2-CTA cluster MMA: TMA multicast cluster=2 may still be used.
//   - SF for activations is float (not UE8M0 int) and per-128 (not per-32).
// `get_mega_moe_cooperative_config_sm90` drives the single SM90 mega-MoE kernel
// (the N-split cooperative kernel: BLOCK_M=64, BLOCK_N=256, two math warpgroups
// split the 256-wide N tile into two 128-wide halves and share one A-tile load);
// this config is what the SM90 host runtime reads. (The pingpong kernel and its
// config were removed.)
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

    // L2 GEMM uses N-major block scheduling (weight stays resident in L2 while
    // sweeping m) — enabled at large tokens-per-expert to cut weight L2 thrash.
    bool l2_nmajor_schedule;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout: dispatch + non-epilogue (TMA) + epilogue (math)
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", l2_nmajor_schedule=" << config.l2_nmajor_schedule
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static std::tuple<int, int> get_block_config_for_mega_moe_cooperative_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    // N-split cooperative: fixed block_m=64 (one m64 WGMMA covers all rows) with
    // exactly 2 math warpgroups. The two WGs process the SAME tile, splitting its
    // BLOCK_N=256 columns into two WG_BLOCK_N=128 halves (each an m64n128 WGMMA,
    // 64 accum-floats/thread — no register spill) and sharing one A-tile load.
    //
    //   12 warps = 384 threads:
    //     HW WG0: 2 dispatch + TMA A + TMA B  → dealloc<48>
    //     HW WG1: Math WG0 (cols 0..127)      → alloc<224>
    //     HW WG2: Math WG1 (cols 128..255)    → alloc<224>
    //     Register budget: 128×48 + 256×224 = 6144 + 57344 = 63488 ≤ 64512
    constexpr int block_m                = 64;
    constexpr int num_epilogue_warpgroups = 2;

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    // Reuse SM100 logic; the block-shape units are different but the wave-balancing
    // intent is identical.
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const int& cd_stages = 1) {
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
    // The tile covers the full BLOCK_M rows; each warpgroup writes its own
    // WG_BLOCK_M-row slice within a single staging tile.
    const int smem_cd_l1 = block_m * (block_n / 2);  // 1 byte/elem (FP8)
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    // `cd_stages` buffers (cooperative double-buffers CD to overlap the L1 store
    // with the next tile's compute). Must match the kernel's
    // SMEM_CD_SIZE = kNumCDStages * aligned(max(L1,L2)).
    const int smem_cd = cd_stages * align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Cross-warpgroup per-row amax exchange (N-split): each of the 2 math WGs
    // writes its 64-col-half per-row amax (BLOCK_M floats) so the other WG can
    // take the max → one shared per-128 SF over the full 128 post-SwiGLU cols.
    // Single-buffered (written then read within one tile's barrier-protected
    // epilogue). Must match the kernel's SMEM_AMAX_SIZE.
    const int smem_amax = align(2 * block_m * static_cast<int>(sizeof(float)), kSmemAlignment);

    // SF on SM90:
    //   * SFA per stage holds BLOCK_M floats: both L1 and L2 are per-128 K, so a
    //     single BLOCK_K=128 tile maps to exactly one SF group per row.
    //   * SFB is loaded directly from global by the math warpgroup (block-(128,128)
    //     weight quantization), so no SMEM is reserved for it.
    const int smem_sfa_per_stage = align(block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage = 0;

    // Per-stage: A tile + B tile + SFA tile + SFB tile
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    // Barriers (8 bytes each):
    //   * dispatch: num_dispatch_warps
    //   * GEMM full + empty: 2 * num_stages
    //   * combine: 2 * num_epilogue_warps
    //   * pingpong order: 4 (OrderedSequenceBarrier<2,2>)
    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps + 4) * 8;
    const int smem_barriers_per_stage = 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax + smem_barriers_fixed;

    // Select max num_stages
    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
}

// SM90 mega-MoE config: the single N-split cooperative kernel (block_m=64,
// block_n=256). CD is double-buffered (cd_stages=2) to overlap the L1 store with
// the next tile's compute.
static MegaMoESM90Config get_mega_moe_cooperative_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_cooperative_sm90(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    // N-split: BLOCK_N=256 so one L1 block emits 128 post-SwiGLU columns (= one
    // per-128 SF group). The two math warpgroups split it into two 128-wide
    // halves (each an m64n128 WGMMA).
    const int block_n = 256;
    const int block_k = 128;
    // cluster_size=1: single-CTA. The per-128 SF amax is reduced across the two
    // math warpgroups within this CTA (see the kernel L1 epilogue), so no
    // cross-CTA synchronisation is needed.
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);

    const int num_dispatch_threads = 64;
    const int num_non_epilogue_threads = 64;

    // L2 N-major scheduling: enable at large tokens-per-expert, where the expert
    // weight (L2 B operand) is large and low-reuse and the M-major order thrashes
    // L2 (measured 47% hit / 97% L2 busy). N-major keeps each weight N-column
    // resident while sweeping m. Threshold matches the megamoe_sm90 branch.
    const float tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    // DIAGNOSTIC: DG_SM90_MOE_NMAJOR overrides the N-major heuristic (-1=auto, 0=off, 1=on).
    const int nmajor_override = get_env<int>("DG_SM90_MOE_NMAJOR", -1);
    const bool l2_nmajor_schedule = nmajor_override < 0
                                        ? (tokens_per_expert >= 256.0f)
                                        : (nmajor_override != 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        /*cd_stages=*/2);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        l2_nmajor_schedule,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90Config(cooperative, num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
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
