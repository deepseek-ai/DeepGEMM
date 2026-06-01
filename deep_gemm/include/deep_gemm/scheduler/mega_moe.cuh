#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

// Computation phase for the current block
enum class BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kClusterSize = 2,
          bool kL2NMajorSchedule = false,
          bool kL1NMajorSchedule = false,
          uint32_t kL2MSwizzleGroup = 0,
          uint32_t kL1MSwizzleGroup = 0,
          uint32_t kExpertRangeStart = 0,
          uint32_t kExpertRangeEnd = kNumExpertsPerRank,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");
    DG_STATIC_ASSERT(kExpertRangeStart <= kExpertRangeEnd, "Invalid expert range");
    DG_STATIC_ASSERT(kExpertRangeEnd <= kNumExpertsPerRank, "Expert range exceeds local experts");

    // For 2-CTA clusters, neighbour CTAs process adjacent M blocks for the
    // same N tile so the B tile can be multicast. Odd M tails fall back to
    // independent unicast work, matching DeepGEMM SM90 grouped GEMM behavior.
    DG_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2, "Invalid cluster size");
    DG_STATIC_ASSERT(kClusterSize == 1 or kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kClusterSize == 1 or kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kClusterSize == 1 or kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kL1MSwizzleGroup == 0 or kL1MSwizzleGroup == 4 or kL1MSwizzleGroup == 8 or kL1MSwizzleGroup == 16,
                     "Invalid L1 M-swizzle group");
    DG_STATIC_ASSERT(kL2MSwizzleGroup == 0 or kL2MSwizzleGroup == 4 or kL2MSwizzleGroup == 8 or kL2MSwizzleGroup == 16,
                     "Invalid L2 M-swizzle group");

    // Arrival counts
    const layout::Workspace& workspace;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t current_token_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;
    bool is_peer_cta_alive = true;
    bool is_a_multicast_valid = true;
    bool is_b_multicast_valid = true;

    // Pre-cached per-expert token counts (filled during `for_each_block` init)
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        return cute::min<uint32_t>(math::align(current_local_expert_idx + 1, kNumExpertsPerWave), kExpertRangeEnd);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    template <uint32_t kPackedAlignment = 128>
    CUTLASS_DEVICE uint32_t get_packed_l2_token_offset(const uint32_t& expert_idx) {
        uint32_t num_tokens = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_tokens += math::align(stored_num_tokens_per_expert[i], kPackedAlignment);
        }
        return __reduce_add_sync(0xffffffff, num_tokens);
    }

    template <uint32_t kPackedAlignment = 128>
    CUTLASS_DEVICE uint32_t get_packed_l2_block_offset(const uint32_t& expert_idx) {
        DG_STATIC_ASSERT(kPackedAlignment % BLOCK_M == 0, "Packed L2 alignment must be a multiple of BLOCK_M");
        return get_packed_l2_token_offset<kPackedAlignment>(expert_idx) / BLOCK_M;
    }

    template <uint32_t kPoolBlockM>
    CUTLASS_DEVICE uint32_t get_aligned_pool_token_offset(const uint32_t& expert_idx) {
        uint32_t num_tokens = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_tokens += math::ceil_div(stored_num_tokens_per_expert[i], kPoolBlockM) * kPoolBlockM;
        }
        return __reduce_add_sync(0xffffffff, num_tokens);
    }

    CUTLASS_DEVICE uint32_t get_token_offset(const uint32_t& expert_idx) {
        uint32_t num_tokens = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_tokens += stored_num_tokens_per_expert[i];
        }
        return __reduce_add_sync(0xffffffff, num_tokens);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_token_offset += current_num_tokens;
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
        current_token_offset = get_token_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_token_offset() const {
        return current_token_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m_start = m_block_idx * BLOCK_M;
        if (m_start >= current_num_tokens)
            return 0;
        const auto m = cute::min(current_num_tokens - m_start, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    template <uint32_t kNumBlockNs, bool kNMajorSchedule, uint32_t kMSwizzleGroup>
    CUTLASS_DEVICE void map_expert_block_idx(const uint32_t& local_block_idx,
                                             const uint32_t& num_m_units,
                                             uint32_t& m_unit_idx,
                                             uint32_t& n_idx) const {
        if constexpr (kMSwizzleGroup > 0) {
            const auto group_start = (local_block_idx / (kMSwizzleGroup * kNumBlockNs)) * kMSwizzleGroup;
            const auto in_group_idx = local_block_idx % (kMSwizzleGroup * kNumBlockNs);
            const auto num_m_in_group = cute::min<uint32_t>(kMSwizzleGroup, num_m_units - group_start);
            n_idx = in_group_idx / num_m_in_group;
            m_unit_idx = group_start + in_group_idx - n_idx * num_m_in_group;
        } else if constexpr (kNMajorSchedule) {
            n_idx = local_block_idx / num_m_units;
            m_unit_idx = local_block_idx - n_idx * num_m_units;
        } else {
            m_unit_idx = local_block_idx / kNumBlockNs;
            n_idx = local_block_idx % kNumBlockNs;
        }
    }

    CUTLASS_DEVICE void update_peer_cta_alive(const uint32_t& num_m_blocks) {
        if constexpr (kClusterSize == 1) {
            is_peer_cta_alive = true;
            is_a_multicast_valid = true;
            is_b_multicast_valid = true;
        } else {
            const auto peer_m_block_idx = (m_block_idx ^ 1u);
            is_peer_cta_alive = m_block_idx < num_m_blocks and
                                peer_m_block_idx < num_m_blocks;
            is_a_multicast_valid = false;
            is_b_multicast_valid = is_peer_cta_alive;
        }
    }

    template <uint32_t kNumBlockNs, uint32_t kMSwizzleGroup = 0>
    CUTLASS_DEVICE void map_cluster_bcast_b_block_idx(const uint32_t& local_block_idx,
                                                       const uint32_t& num_m_blocks,
                                                       uint32_t& m_idx,
                                                       uint32_t& n_idx) {
        DG_STATIC_ASSERT(kClusterSize == 2, "Cluster B multicast mapping requires 2 CTAs");
        DG_STATIC_ASSERT(kMSwizzleGroup == 0 or kMSwizzleGroup == 4 or kMSwizzleGroup == 8 or kMSwizzleGroup == 16,
                         "Invalid cluster B multicast M-swizzle group");

        if constexpr (kMSwizzleGroup == 0) {
            const auto even_m_blocks = num_m_blocks & ~1u;
            const auto even_region_blocks = even_m_blocks * kNumBlockNs;
            if (local_block_idx < even_region_blocks) {
                n_idx = local_block_idx / even_m_blocks;
                m_idx = local_block_idx - n_idx * even_m_blocks;
                is_peer_cta_alive = true;
                is_a_multicast_valid = false;
                is_b_multicast_valid = true;
            } else {
                const auto tail_idx = local_block_idx - even_region_blocks;
                m_idx = even_m_blocks;
                n_idx = tail_idx;
                // Odd-M tails pair CTAs across adjacent N blocks. The peer CTA is
                // still alive for remote empty-barrier arrivals, but it does not
                // consume the same B tile, so B multicast must be disabled.
                is_peer_cta_alive = (tail_idx ^ 1u) < kNumBlockNs;
                is_a_multicast_valid = false;
                is_b_multicast_valid = false;
            }
        } else {
            const auto num_blocks_per_group = kMSwizzleGroup * kNumBlockNs;
            const auto group_idx = local_block_idx / num_blocks_per_group;
            const auto first_m_idx = group_idx * kMSwizzleGroup;
            auto in_group_idx = local_block_idx - group_idx * num_blocks_per_group;
            auto num_m_in_group = cute::min<uint32_t>(kMSwizzleGroup, num_m_blocks - first_m_idx);

            if (num_m_in_group % 2 != 0) {
                const auto even_m_in_group = num_m_in_group ^ 1u;
                const auto even_region_blocks = even_m_in_group * kNumBlockNs;
                if (in_group_idx < even_region_blocks) {
                    num_m_in_group = even_m_in_group;
                } else {
                    in_group_idx -= even_region_blocks;
                    m_idx = first_m_idx + even_m_in_group;
                    n_idx = in_group_idx;
                    is_peer_cta_alive = (in_group_idx ^ 1u) < kNumBlockNs;
                    is_a_multicast_valid = false;
                    is_b_multicast_valid = false;
                    return;
                }
            }

            m_idx = first_m_idx + in_group_idx % num_m_in_group;
            n_idx = in_group_idx / num_m_in_group;
            is_peer_cta_alive = true;
            is_a_multicast_valid = false;
            is_b_multicast_valid = true;
        }
    }

    template <uint32_t kNumBlockNs>
    CUTLASS_DEVICE void map_cluster_bcast_a_block_idx(const uint32_t& local_block_idx,
                                                       const uint32_t& num_m_blocks,
                                                       uint32_t& m_idx,
                                                       uint32_t& n_idx) {
        (void)num_m_blocks;
        DG_STATIC_ASSERT(kClusterSize == 2, "Cluster A multicast mapping requires 2 CTAs");
        DG_STATIC_ASSERT(kNumBlockNs % 2 == 0, "N block count must be even for 2-CTA A multicast");
        m_idx = local_block_idx / kNumBlockNs;
        n_idx = local_block_idx - m_idx * kNumBlockNs;
        is_peer_cta_alive = true;
        is_a_multicast_valid = true;
        is_b_multicast_valid = false;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            const auto num_blocks = num_m_blocks * kNumL1BlockNs;
            if (block_idx < num_blocks) {
                if constexpr (kClusterSize == 1) {
                    map_expert_block_idx<kNumL1BlockNs, kL1NMajorSchedule, kL1MSwizzleGroup>(
                        block_idx, num_m_blocks, m_block_idx, n_block_idx);
                    update_peer_cta_alive(num_m_blocks);
                } else {
                    map_cluster_bcast_b_block_idx<kNumL1BlockNs, kL1MSwizzleGroup>(
                        block_idx, num_m_blocks, m_block_idx, n_block_idx);
                }
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_blocks;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            const auto num_blocks = num_m_blocks * kNumL2BlockNs;
            if (block_idx < num_blocks) {
                if constexpr (kClusterSize == 1) {
                    map_expert_block_idx<kNumL2BlockNs, kL2NMajorSchedule, kL2MSwizzleGroup>(
                        block_idx, num_m_blocks, m_block_idx, n_block_idx);
                    update_peer_cta_alive(num_m_blocks);
                } else {
                    if constexpr (kL2NMajorSchedule) {
                        map_cluster_bcast_a_block_idx<kNumL2BlockNs>(
                            block_idx, num_m_blocks, m_block_idx, n_block_idx);
                    } else {
                        map_cluster_bcast_b_block_idx<kNumL2BlockNs, kL2MSwizzleGroup>(
                            block_idx, num_m_blocks, m_block_idx, n_block_idx);
                    }
                }
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_blocks;
            advance_expert_idx();
        }
        return false;
    }


    // Core state machine: assigns the next block
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kExpertRangeEnd)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    // Found a new L1 block
                    // Jump to next persistent CTA work item
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // L1 for the current wave is complete, transition to L2
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }
            } else {
                if (fetch_next_l2_block()) {
                    // Found a new L2 block
                    // Jump to next persistent CTA work item
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // Move to L1 of the next wave
                    next_phase = BlockPhase::Linear1;
                }
            }
        }

        // All waves and experts are fully processed
        return {BlockPhase::None, 0, 0, 0};
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    CUTLASS_DEVICE void fetch_packed_l2_metadata_count() {
        // Split K2 starts after split K1 globally completes. When K1 built the
        // metadata prefix, the per-expert actual-M values are already final, so
        // K2 can initialize scheduler counts without volatile polling.
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint32_t value = 0;
            if (expert_idx < kNumExpertsPerRank)
                value = *workspace.get_packed_l2_expert_m_ptr(expert_idx);
            stored_num_tokens_per_expert[i] = value;
        }
        __syncwarp();
    }

    template <uint32_t kPackedAlignment = 128>
    CUTLASS_DEVICE void build_packed_l2_metadata_prefix(const bool& is_leader_thread) const {
        DG_STATIC_ASSERT(kPackedAlignment == 128, "Only 128-row packed L2 metadata is currently supported");
        if (blockIdx.x != 0 or !is_leader_thread)
            return;

        uint32_t packed_offset = 0;
        #pragma unroll
        for (uint32_t expert_idx = 0; expert_idx < kNumExpertsPerRank; ++expert_idx) {
            uint64_t value = 0;
            do {
                value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
            } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);

            const uint32_t actual_m = static_cast<uint32_t>(value);
            *workspace.get_packed_l2_expert_start_ptr(expert_idx) = packed_offset;
            *workspace.get_packed_l2_expert_m_ptr(expert_idx) = actual_m;
            packed_offset += math::align(actual_m, kPackedAlignment);
        }
        *workspace.get_packed_l2_expert_start_ptr(kNumExpertsPerRank) = packed_offset;
    }

    template <uint32_t kPackedAlignment = 128>
    CUTLASS_DEVICE void fill_packed_l2_metadata_rows(const uint32_t& thread_idx,
                                                     const uint32_t& num_threads) const {
        DG_STATIC_ASSERT(kPackedAlignment == 128, "Only 128-row packed L2 metadata is currently supported");
        for (uint32_t expert_idx = blockIdx.x; expert_idx < kNumExpertsPerRank; expert_idx += kNumSMs) {
            const uint32_t packed_start = *workspace.get_packed_l2_expert_start_ptr(expert_idx);
            const uint32_t packed_end = *workspace.get_packed_l2_expert_start_ptr(expert_idx + 1);
            const uint32_t actual_m = *workspace.get_packed_l2_expert_m_ptr(expert_idx);
            const uint32_t packed_m = packed_end - packed_start;

            for (uint32_t row = thread_idx; row < packed_m; row += num_threads) {
                *workspace.get_packed_l2_row_to_expert_ptr(packed_start + row) =
                    row < actual_m ? static_cast<int32_t>(expert_idx) : -1;
            }
        }
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // Wait for all expert counters to be finalized
        fetch_expert_recv_count();

        // Initialize current expert with 0
        set_expert_idx(kExpertRangeStart);

        // Iterate over all blocks
        // TODO: add swizzle within expert waves for better L2 cache utilization
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_linear1_block(Func&& func) {
        // Split-kernel mode: K1 owns only dispatch + Linear1. Unlike
        // for_each_block(), do not burn scheduler iterations on Linear2 blocks.
        fetch_expert_recv_count();
        set_expert_idx(kExpertRangeStart);
        while (current_local_expert_idx < kExpertRangeEnd) {
            if (fetch_next_l1_block()) {
                block_idx += kNumSMs;
                func(current_local_expert_idx, kNumL1BlockKs, m_block_idx, n_block_idx);
            } else if (current_local_expert_idx >= kExpertRangeEnd) {
                break;
            }
        }
    }

    template <bool kUsePackedL2MetadataCounts = false, typename Func>
    CUTLASS_DEVICE void for_each_linear2_block(Func&& func) {
        // Split-kernel mode: K2 starts after K1 has completed globally, so all
        // L2-ready masks are already final. Schedule Linear2 blocks directly
        // from the phase start, like a standalone grouped GEMM.
        if constexpr (kUsePackedL2MetadataCounts)
            fetch_packed_l2_metadata_count();
        else
            fetch_expert_recv_count();
        set_expert_idx(kExpertRangeStart);
        while (current_local_expert_idx < kExpertRangeEnd) {
            if (fetch_next_l2_block()) {
                block_idx += kNumSMs;
                func(current_local_expert_idx, kNumL2BlockKs, m_block_idx, n_block_idx);
            } else if (current_local_expert_idx >= kExpertRangeEnd) {
                break;
            }
        }
    }
};

} // namespace deep_gemm::sched
