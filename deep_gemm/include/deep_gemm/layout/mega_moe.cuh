#pragma once

#include <cstddef>
#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 8;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192, 240};
static constexpr int kMaxCandidateBlockM = 240;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 1920;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;

    TokenSrcMetadata() = default;

    CUTLASS_DEVICE TokenSrcMetadata(const uint32_t& rank_idx, const uint32_t& token_idx, const uint32_t& topk_idx):
        rank_idx(rank_idx), token_idx(token_idx), topk_idx(topk_idx) {}
};

struct alignas(128) MegaMoESignals {
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;
    static constexpr uint32_t kNumMaxExperts = 2048;
    static constexpr uint32_t kNumMaxExpertsPerRank = 512;
    static constexpr uint32_t kNumMaxRingBlocks = 1u << 20;  // 20 MiB for the ring signals
    static constexpr uint32_t kNumMaxSharedL2Blocks = 1u << 15;  // 128 KiB

    // Grid and NVLink synchronization
    uint32_t grid_sync_count[kNumMaxGridSyncCounters];
    uint32_t nvl_barrier_counter;
    int nvl_barrier_signals[2];

    // Task scheduling
    uint32_t l1_task_count;
    uint32_t l2_task_count;
    uint32_t shared_l1_task_count;
    uint32_t shared_l2_task_count;

    // Combine readiness: `combine_ready_grid_idx[peer] == own grid index` means the peer's L2 writes into this
    // rank are done; grid indices are unique per launch, so no reset is needed. Peers push theirs during dispatch.
    alignas(128) uint64_t combine_ready_grid_idx[kNumMaxRanks];
    alignas(128) uint64_t peer_grid_idx[kNumMaxRanks];

    // Expert token counts
    alignas(128) uint64_t expert_send_count[kNumMaxExperts];
    uint64_t expert_recv_count[kNumMaxExperts];
    uint64_t expert_recv_count_sum[kNumMaxExpertsPerRank];

    // Routed-expert ring signals, `l2_full_mask` has one bit per L1 N block
    uint32_t l1_full_count[kNumMaxRingBlocks];
    uint32_t l1_empty_count[kNumMaxRingBlocks];
    uint64_t l2_full_mask[kNumMaxRingBlocks];
    uint32_t l2_empty_count[kNumMaxRingBlocks];

    // Shared-expert signals
    uint32_t shared_l2_full_count[kNumMaxSharedL2Blocks];
};
DG_STATIC_ASSERT(offsetof(MegaMoESignals, combine_ready_grid_idx) == 128, "Invalid control signal layout");

struct Workspace {
    MegaMoESignals* signals;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_shared_l2_pool_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    Workspace() = default;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        signals(static_cast<MegaMoESignals*>(base)),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);

        DG_UNIFIED_ASSERT(num_ranks > 0);
        DG_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        DG_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        DG_UNIFIED_ASSERT(num_experts <= MegaMoESignals::kNumMaxExperts);
        DG_UNIFIED_ASSERT(num_experts_per_rank <= MegaMoESignals::kNumMaxExpertsPerRank);
        DG_UNIFIED_ASSERT(num_ring_tokens <= MegaMoESignals::kNumMaxRingBlocks * kMinCandidateBlockM);
        DG_UNIFIED_ASSERT(num_shared_l2_pool_blocks <= MegaMoESignals::kNumMaxSharedL2Blocks);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Fixed-position global and expert signals
        num_bytes += sizeof(MegaMoESignals);

        // Source token-topk: [local expert][source rank][token], with distinct experts per token
        num_bytes += num_experts * num_max_tokens_per_rank * sizeof(uint32_t);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(signals, get_num_bytes());
    }

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < MegaMoESignals::kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return signals->grid_sync_count + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return &signals->nvl_barrier_counter;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return signals->nvl_barrier_signals + phase;
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_ready_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->combine_ready_grid_idx + peer_rank_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_peer_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->peer_grid_idx + peer_rank_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return &signals->l1_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return &signals->l2_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return &signals->shared_l1_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return &signals->shared_l2_task_count;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_send_count + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count_sum + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_full_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_empty_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_full_mask_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_full_mask + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_empty_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        return signals->shared_l2_full_count + block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto offset = (expert_idx * num_ranks + rank_idx) * num_max_tokens_per_rank + token_idx;
        return math::advance_ptr<uint32_t>(signals, sizeof(MegaMoESignals)) + offset;
    }

    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_topk_weights_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const bool& with_sf,
                  const uint32_t& num_shared_experts = 0) {
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto num_max_shared_sf_tokens = with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t num_mma_elem_bytes = with_sf ? 1 : 2;
        const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
        const auto shared_intermediate_token_layout = layout::Data(shared_intermediate_hidden * num_mma_elem_bytes);
        const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0, false);
        const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0, false);
        const auto shared_intermediate_sf_layout = layout::Data(with_sf ? shared_intermediate_hidden / 32 : 0, false);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                (with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
               - reinterpret_cast<uint8_t*>(workspace.signals);
    }
};

} // namespace deep_gemm::layout
