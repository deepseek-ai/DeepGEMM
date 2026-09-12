#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

// SM90 (`impls/sm90_fp8_fused_mega_moe.cuh`) only instantiates BLOCK_M in {64, 128}, so its pool/SF sizing uses this set
static constexpr int kNumSM90CandidateBlockMs = 2;
static constexpr int kSM90FusedCandidateBlockM[kNumSM90CandidateBlockMs] = {64, 128};
static constexpr int kSM90FusedMaxCandidateBlockM = 128;
static constexpr int kSM90FusedLCMBlockM = 128;
static constexpr uint32_t kSM90FusedLagUnitM = 8;

// Combine staging ladder of `impls/sm90_fp8_fused_mega_moe.cuh`, shared with the host so a hidden size the
// vectorization cannot serve is reported before NVRTC sees the specialization. Three whole-row slots per combine
// warp; the row is combined in chunks when they do not fit the shared memory in front of the barriers.
static constexpr uint32_t kSM90FusedCombineChunkSlots = 3;
static constexpr uint32_t kSM90FusedCombineMaxRegistersPerLane = 128;
static constexpr uint32_t kSM90FusedCombineElemBytes = 2;   // bf16

CUTLASS_HOST_DEVICE constexpr uint32_t get_sm90_fused_moe_combine_num_chunks(
    const uint32_t hidden, const uint32_t num_combine_warps,
    const uint32_t smem_before_barriers, const bool split_mn_warpgroups) {
    // hidden = 7 * 1024 is combined in 7 chunks, which keeps the 32-lane uint4 mapping
    if (split_mn_warpgroups)
        return hidden % 7 == 0 ? 7u : (hidden >= 1024 ? 4u : 1u);
    const uint64_t one_chunk_bytes = static_cast<uint64_t>(kSM90FusedCombineChunkSlots) *
                                     num_combine_warps * hidden * kSM90FusedCombineElemBytes;
    // 2 chunks do not always fit either: hidden 7168 at 3 stages needs 3 * 8 * 14336 / 2 = 172032 B of staging and
    // the region in front of the barriers is short of that, so the ladder goes on to 4
    const uint32_t num_chunks_if_chunked = one_chunk_bytes / 2 <= smem_before_barriers ? 2u : 4u;
    return (one_chunk_bytes <= smem_before_barriers and
            hidden <= 32u * kSM90FusedCombineMaxRegistersPerLane) ? 1u : num_chunks_if_chunked;
}

CUTLASS_HOST_DEVICE constexpr bool is_sm90_fused_moe_combine_vectorization_legal(
    const uint32_t hidden, const uint32_t num_combine_warps,
    const uint32_t smem_before_barriers, const bool split_mn_warpgroups) {
    const auto num_chunks = get_sm90_fused_moe_combine_num_chunks(
        hidden, num_combine_warps, smem_before_barriers, split_mn_warpgroups);
    const uint64_t selected_chunk_bytes = static_cast<uint64_t>(kSM90FusedCombineChunkSlots) *
        num_combine_warps * hidden * kSM90FusedCombineElemBytes / num_chunks;
    // every lane moves whole 16-byte elements, so a chunk must be a whole number of 32-lane rounds
    return hidden > 0u and hidden % num_chunks == 0u and
           selected_chunk_bytes <= smem_before_barriers and
           (hidden * kSM90FusedCombineElemBytes / num_chunks) % (32u * sizeof(uint4)) == 0u;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens_sm90(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                             T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kSM90FusedMaxCandidateBlockM) - 1),
        static_cast<T>(kSM90FusedLCMBlockM));
}

// SM90 tile table: at most ceil(total tiles / num_sms) entries per CTA plus the end marker, 16 bytes each (host smem sizing and the kernel layout both use this)
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_sm90_tile_table_entries(T num_max_pool_tokens, T block_m, T num_l1_block_ns,
                                                            T num_l2_block_ns, T num_sms) {
    return math::constexpr_ceil_div((num_max_pool_tokens / block_m) * (num_l1_block_ns + num_l2_block_ns), num_sms) + 1;
}

// Compact table (8-byte entries): one spare entry more than the round-robin share, so the CTAs together absorb the whole bounded dynamic tail
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_sm90_tile_table_entries_compact(T num_max_pool_tokens, T block_m, T num_l1_block_ns,
                                                                    T num_l2_block_ns, T num_sms) {
    return get_sm90_tile_table_entries(num_max_pool_tokens, block_m, num_l1_block_ns, num_l2_block_ns, num_sms) + 1;
}

// SM90 MegaMoE predates the reusable ring workspace used by the SM100 kernels.
// Keep its compact layout explicit so Hopper codegen and buffer slicing stay
// identical to the tuned SM90 implementation while SM100 can use Workspace.
// The words remote ranks write into this workspace are double-buffered by launch parity (`t2_bank`); the per-expert recv counts are plain stores + one release flag per source rank, summed by the receivers.
struct SM90FusedWorkspace {
    static constexpr bool kHeadLL = true;
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // Ring-buffer capacity (defaults to the full pool): only the reusable
    // full/empty semaphores are sized by it, sized conservatively at
    // `kMinCandidateBlockM` granularity so any compiled BLOCK_M fits.
    uint32_t num_ring_tokens;
    uint32_t num_ring_blocks;

    // bank of the words remote ranks write into THIS rank's workspace (recv counts, done flags, L2 tile counts): launch g
    // uses bank g & 1 and zeroes bank (g + 1) & 1 after its tag-1 barrier, so nothing of generation g is zeroed while a remote rank may still write it
    uint32_t t2_bank = 0;

    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    SM90FusedWorkspace(void* base,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_max_pool_tokens,
                  const uint32_t& num_ring_tokens = 0):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_max_pool_tokens(num_max_pool_tokens),
        num_ring_tokens(num_ring_tokens) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
        num_ring_blocks = (num_ring_tokens == 0 ? num_max_pool_tokens : num_ring_tokens) / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        num_bytes += num_experts * sizeof(uint64_t) * 2;
        num_bytes += num_experts_per_rank * sizeof(uint64_t);
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);
        // Ring full/empty semaphores (alias the arrival arrays for `*_full`)
        num_bytes += num_ring_blocks * sizeof(uint32_t) * 2;
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);
        // early-combine flags, tile counts and the launch-parity second bank; padded to 1 KiB so the data pools that follow keep their alignment
        num_bytes += math::align<uint64_t>(get_t2_bank1_bytes() + get_t2_bank1_offset_in_flags(), 1024);
        return math::align<uint64_t>(num_bytes, 16);
    }

    // Bank geometry (bank 1 = the copy behind bank 0, 8-byte aligned): u64 part [recv_count x num_experts | unused x num_experts_per_rank],
    // u32 part [done_flag x num_experts | l2_tile_done_count x num_experts_per_rank | head count flags x num_ranks]
    CUTLASS_HOST_DEVICE
    uint32_t get_t2_bank_flag_words() const {
        return num_experts + num_experts_per_rank + num_ranks;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_t2_bank1_offset_in_flags() const {
        return math::align<uint64_t>(get_t2_bank_flag_words() * sizeof(uint32_t), 8);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_t2_bank1_bytes() const {
        return (num_experts + num_experts_per_rank) * sizeof(uint64_t) + get_t2_bank_flag_words() * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(
            base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    // Launch parity word (last word of the 32-byte signal block; flipped by SM0 in the tail of every launch, read after the PDL wait)
    CUTLASS_DEVICE
    uint32_t* get_t2_parity_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters + 3;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_bank0_u64_end_ptr() const {
        return get_expert_send_count_ptr(num_experts * 2) + num_experts_per_rank;
    }

    CUTLASS_DEVICE
    uint64_t* get_t2_recv_count_bank_ptr(const uint32_t& bank) const {
        return bank ? get_t2_bank1_recv_count_ptr() : get_expert_send_count_ptr(num_experts);
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_t2_recv_count_bank_ptr(t2_bank) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        return reinterpret_cast<uint32_t*>(get_bank0_u64_end_ptr()) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    // Ring-mode counting semaphores. `*_full` alias the legacy arrival arrays
    // (u32 view for L2), `*_empty` live in the dedicated ring area.
    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return get_l1_arrival_count_ptr(ring_block_idx);
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return reinterpret_cast<uint32_t*>(get_l2_arrival_mask_ptr(ring_block_idx));
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = reinterpret_cast<uint32_t*>(
            get_l2_arrival_mask_ptr(num_max_pool_blocks));
        return base + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return get_l1_empty_count_ptr(num_ring_blocks) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }

    // early-combine done flags: set by the owning rank once all L2 tiles of a global expert have scattered. Warps 1.. combine
    // a token as soon as its top-k experts' flags are all set, overlapping that with the same all-rank barrier the other
    // mode also takes; tokens whose flags are not all set are combined after it
    CUTLASS_DEVICE
    uint32_t* get_expert_done_flag_ptr_bank0(const uint32_t& expert_idx = 0) const {
        return reinterpret_cast<uint32_t*>(get_token_src_metadata_ptr(num_max_pool_tokens)) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_t2_bank1_recv_count_ptr() const {
        return reinterpret_cast<uint64_t*>(
            reinterpret_cast<uint8_t*>(get_expert_done_flag_ptr_bank0()) + get_t2_bank1_offset_in_flags());
    }

    CUTLASS_DEVICE
    uint32_t* get_t2_flag_bank_ptr(const uint32_t& bank) const {
        return bank ? reinterpret_cast<uint32_t*>(get_t2_bank1_recv_count_ptr() + num_experts + num_experts_per_rank)
                    : get_expert_done_flag_ptr_bank0();
    }

    CUTLASS_DEVICE
    uint32_t* get_expert_done_flag_ptr(const uint32_t& expert_idx = 0) const {
        return get_t2_flag_bank_ptr(t2_bank) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_tile_done_count_ptr(const uint32_t& local_expert_idx = 0) const {
        return get_t2_flag_bank_ptr(t2_bank) + num_experts + local_expert_idx;
    }

    // per source rank "my recv counts for you are stored" flag (st.release.sys by the source's SM0 after its count stores; receivers poll)
    CUTLASS_DEVICE
    uint32_t* get_hll_count_flag_ptr(const uint32_t& src_rank_idx = 0) const {
        return get_t2_flag_bank_ptr(t2_bank) + num_experts + num_experts_per_rank + src_rank_idx;
    }
};

} // namespace deep_gemm::layout
