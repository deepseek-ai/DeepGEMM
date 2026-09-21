#pragma once

#include <algorithm>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm90.hpp"

namespace deep_gemm {

static constexpr int kSM90NVFP4BStoragePerKBlock = 80;

struct SM90NVFP4FusedConfig {
    static constexpr int kBlockK = 128;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumEpilogueThreads = 256;
    static constexpr int kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    int block_m, block_n;
    int num_ring_tokens;
    int num_padded_sf_pool_tokens;
    int num_stages, smem_size;
};

struct SM90NVFP4FusedShape {
    static constexpr int kMaxNumRanks = 72;
    // The expert-count SMEM slice has the same aligned size as the original
    // 384-expert tuning through 512 experts.
    static constexpr int kMaxNumExperts = 512;
    static constexpr int kMinDimensionAlignment = 512;
    static constexpr int kMaxTopk = 32;

    int num_sms;
    int num_ranks;
    int num_experts;
    int num_topk;
    int hidden;
    int intermediate_hidden;

    static constexpr bool is_supported_batch(const int num_tokens) noexcept {
        return num_tokens > 0;
    }

    constexpr bool is_supported_sm90_shape() const noexcept {
        return num_sms > 1 &&
            num_ranks > 0 && num_ranks <= kMaxNumRanks &&
            num_experts >= num_topk && num_experts <= kMaxNumExperts &&
            num_experts % num_ranks == 0 &&
            num_topk > 0 && num_topk <= kMaxTopk &&
            hidden >= kMinDimensionAlignment &&
            hidden % kMinDimensionAlignment == 0 &&
            intermediate_hidden >= kMinDimensionAlignment &&
            intermediate_hidden % kMinDimensionAlignment == 0;
    }

    // Ring-buffer capacity for this kernel's L1/L2 activation buffers.
    // Mirrors the generic MegaMoE buffer sizing's use of
    // `sched::get_num_max_live_pool_blocks` (see `get_symm_buffer_size_for_mega_moe`
    // in `mega.hpp`), but iterates this kernel's own BLOCK_M candidates --
    // {8, 16, 24, 64, 128}, from the tuning table in `select_sm90_nvfp4_fused`
    // below -- instead of the generic `layout::kCandidateBlockM` set, which
    // omits 24. `mega.hpp` (buffer creation) and `select_sm90_nvfp4_fused`
    // (kernel launch, TMA descriptor extents) both call this single function so
    // they cannot silently drift apart, the way the ported full-pool token count
    // could (see the buffer-layout fix in commit `f178178`).
    static int get_num_ring_tokens(
            int num_ranks, int num_max_tokens_per_rank,
            int num_topk, int num_experts_per_rank, int num_sms,
            int hidden, int intermediate_hidden) {
        static constexpr int kRingCandidateBlockM[] = {8, 16, 24, 64, 128};
        const int num_active_topk = std::min(num_topk, num_experts_per_rank);
        const int num_max_routed_tokens =
            num_max_tokens_per_rank * num_ranks * num_active_topk;
        int num_ring_tokens = 0;
        for (const auto block_m : kRingCandidateBlockM) {
            const auto num_pool_blocks =
                ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
            const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
                num_pool_blocks, num_sms, hidden, intermediate_hidden);
            num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
        }
        // All ring candidates divide `layout::kLCMCandidateBlockM` (384), so
        // this keeps `kNumRingTokens / BLOCK_M` exact for every BLOCK_M this
        // kernel's tuning table can pick across the buffer's shared lifetime.
        return align(num_ring_tokens, layout::kLCMCandidateBlockM);
    }
};

struct SM90NVFP4FusedInput {
    int num_sms;
    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;

    SM90NVFP4FusedShape shape() const noexcept {
        return {
            num_sms, num_ranks, num_experts, num_topk,
            hidden, intermediate_hidden};
    }
};

struct SM90NVFP4FusedPlan {
    SM90NVFP4FusedConfig config;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
    bool use_interleaved_scheduler;
};

static SM90NVFP4FusedPlan
select_sm90_nvfp4_fused(
        const SM90NVFP4FusedInput& input) {
    DG_HOST_ASSERT(input.shape().is_supported_sm90_shape());
    DG_HOST_ASSERT(input.num_experts_per_rank > 0);
    DG_HOST_ASSERT(input.num_experts ==
                   input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(
        SM90NVFP4FusedShape::is_supported_batch(input.num_tokens));
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    struct Tuning {
        int block_m, block_n;
        int num_stages;
        int smem_size;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
    } tuning {};

    if (input.num_tokens <= 1)
        tuning = {8, 256, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 8)
        tuning = {8, 256, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 16)
        tuning = {8, 256, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 32)
        tuning = {16, 256, 3, SM90ArchSpec::smem_capacity,
                  true, true, false};
    else if (input.num_tokens <= 64)
        tuning = {24, 256, 3, 229312,
                  true, false, true};
    // BM64N256 is not deterministic when an expert spans multiple M blocks.
    // Use the stable split-M plan until that path's race is resolved.
    else if (input.num_tokens <= 256)
        tuning = {128, 128, 6, SM90ArchSpec::smem_capacity,
                  false, true, false};
    else
        tuning = {128, 128, 6, SM90ArchSpec::smem_capacity,
                  false, true, false};

    DG_HOST_ASSERT(tuning.smem_size <= SM90ArchSpec::smem_capacity);
    return {
        {
            tuning.block_m,
            tuning.block_n,
            SM90NVFP4FusedShape::get_num_ring_tokens(
                input.num_ranks, input.num_max_tokens_per_rank,
                input.num_topk, input.num_experts_per_rank, input.num_sms,
                input.hidden, input.intermediate_hidden),
            input.num_padded_sf_pool_tokens,
            tuning.num_stages,
            cute::min(tuning.smem_size +
                          layout::kSM90InterleavedSchedulerSMEMBytes,
                      SM90ArchSpec::smem_capacity),
        },
        tuning.swap_ab,
        tuning.use_mode2_row_decoder,
        tuning.single_active_dispatch_warp,
        true,
    };
}

}  // namespace deep_gemm
