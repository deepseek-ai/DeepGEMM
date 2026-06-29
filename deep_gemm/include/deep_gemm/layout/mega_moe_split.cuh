#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

// Split-kernel MegaMoE workspace.
//
// The split pipeline (dispatch_l1_swiglu / l2_combine / combine_reduce) shares the fused-megamoe
// `Workspace` bookkeeping region (barriers, expert counts, L1/L2 arrival masks) but needs a
// route-based ("token pull shared") dispatch-metadata sub-layout for the K1 dispatch pull: a
// per-(expert, rank) source-token-topk slot sized by `num_max_tokens_per_rank`, plus per-token
// route-count and multi-route-entry regions.
//
// It derives from `Workspace` so it binds to the `const layout::Workspace&` parameters of the
// shared scheduler / comm helpers (which only touch the identical region), while the split
// kernels call the overridden dispatch accessors on the `SplitWorkspace` object directly. Only
// the dispatch-region accessors plus `get_num_bytes` / `get_end_ptr` (the buffer size, hence the
// pool base) are overridden; everything else (and the fused `Workspace`) is left untouched.
struct SplitWorkspace : public Workspace {
    uint32_t num_topk;

    CUTLASS_HOST_DEVICE
    SplitWorkspace(void* base,
                   const uint32_t& num_ranks,
                   const uint32_t& num_experts,
                   const uint32_t& num_max_tokens_per_rank,
                   const uint32_t& num_topk):
        Workspace(base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk),
        num_topk(num_topk) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count (padded to even entry count for `uint64_t` alignment of L2 mask)
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_tokens_per_rank * sizeof(int);

        // Dispatch pulling per-token route counts
        num_bytes += num_ranks * num_max_tokens_per_rank * sizeof(int);

        // Dispatch pulling multi-route entries
        num_bytes += num_ranks * num_max_tokens_per_rank * num_topk * sizeof(int);

        // Combine push source indices
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            (expert_idx * num_ranks + rank_idx) * num_max_tokens_per_rank + token_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_src_route_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        return get_src_token_topk_idx_ptr(num_experts_per_rank) +
            rank_idx * num_max_tokens_per_rank + token_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_src_route_entry_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0, const uint32_t& topk_idx = 0) const {
        return get_src_route_count_ptr(num_ranks) +
            (rank_idx * num_max_tokens_per_rank + token_idx) * num_topk + topk_idx;
    }

    // For combine usages
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(
            get_src_route_entry_ptr(num_ranks, 0, 0));
        return base + pool_token_idx;
    }
};

} // namespace deep_gemm::layout
