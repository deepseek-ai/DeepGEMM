#pragma once

#include <type_traits>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_fused_mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

namespace deep_gemm::sched {

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          // m-inner block order: adjacent blocks (and thus the two CTAs of a
          // cluster pair) share n_block_idx so B (weights) can be multicast;
          // the default n-inner order pairs on m for A multicast instead.
          bool kMulticastOnB = false,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K,
          typename WorkspaceT = layout::Workspace,
          // > 0: L2-lag interleaved schedule (see get_next_block_lag); 0: wave schedule
          uint32_t kL2LagUnits = 0,
          // bounds of the bit-packed tile-table entry fields (pool blocks of the rank, tokens one expert can receive); tile-table users (SM90) only
          uint32_t kNumMaxPoolBlocks = 0,
          uint32_t kNumMaxTokensPerExpert = 0,
          // CTAs per cluster: with 2 the leader publishes the partner's dynamic-tail entries over DSMEM, so the tag loads
          // acquire at cluster scope
          uint32_t kTileTableClusterSize = 1>
struct SM90FusedMegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerWave > 0 and kNumExpertsPerWave <= kNumExpertsPerRank, "Invalid wave config");

    // NOTES: with a 2-CTA cluster the N block counts must be even so that 2 adjacent CTAs always land on
    // the same m_block_idx with n_block_idx differing by 1; a shape (or SM count) that cannot be paired runs
    // 1-CTA clusters instead, which the host selects
    DG_STATIC_ASSERT(kTileTableClusterSize == 1 or kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kTileTableClusterSize == 1 or kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kTileTableClusterSize == 1 or kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // Arrival counts
    const WorkspaceT& workspace;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Pre-cached per-expert token counts (filled during `for_each_block` init)
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    // L2-lag schedule state (kL2LagUnits > 0). A unit is up to layout::kSM90FusedLagUnitM consecutive m-blocks of one expert
    // times all N blocks; inside a unit the order is m-inner and pair validity is decided per position (lag_decode).
    // The global sequence is
    //   L1(U0) .. L1(U_lag)  L2(U0)  L1(U_lag+1)  L2(U1)  ...  then the remaining L2 units,
    // i.e. L2 of a pool block is issued kL2LagUnits units after its L1 and the whole rank is one wave. L2(p) only ever
    // waits on L1(p), which precedes it and never waits on any L2, so the smallest pending L2 always makes progress.
    uint32_t l1_expert = 0, l1_num_tokens = 0, l1_pool_offset = 0, l1_m0 = 0;
    uint32_t l2_expert = 0, l2_num_tokens = 0, l2_pool_offset = 0, l2_m0 = 0;
    uint32_t num_l1_units_done = 0;
    uint32_t num_l2_units_pending = 0;   // L2 units still to emit in the current group
    bool current_pair_valid = true;
    // kL2LagUnits encodes lag + 1000 * group: after every `group` L1 units past the lag the same number of L2 units is emitted back to back
    static constexpr uint32_t kLagUnitsOnly = kL2LagUnits % 1000u;
    static constexpr uint32_t kLagGroupUnits = (kL2LagUnits / 1000u) > 0 ? (kL2LagUnits / 1000u) : 1u;

    CUTLASS_DEVICE explicit SM90FusedMegaMoEScheduler(const WorkspaceT& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        // Align up to wave boundary, clamped for the last partial wave
        const auto aligned = math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
        return cute::min(aligned, kNumExpertsPerRank);
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

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
        if constexpr (kL2LagUnits > 0) {
            l1_expert = l2_expert = expert_idx;
            l1_num_tokens = l2_num_tokens = current_num_tokens;
            l1_pool_offset = l2_pool_offset = current_pool_block_offset;
            l1_m0 = l2_m0 = 0;
            num_l1_units_done = 0;
            num_l2_units_pending = 0;
            lag_skip_empty(l1_expert, l1_num_tokens, l1_pool_offset, l1_m0);
            lag_skip_empty(l2_expert, l2_num_tokens, l2_pool_offset, l2_m0);
        }
    }

    // Cluster pairing: positions 2j (leader) and 2j+1 (partner) must share n and differ in m by 1 for B multicast. Wave
    // schedule: local check (chunks are even-sized). Lag schedule: decided per unit so both CTAs agree on 1-wide tail units.
    CUTLASS_DEVICE bool is_pair_valid(const bool& is_leader) const {
        if constexpr (kL2LagUnits > 0) {
            return current_pair_valid;
        } else {
            const auto num_m_blocks = get_current_num_m_blocks();
            return is_leader ? (m_block_idx + 1 < num_m_blocks) : (m_block_idx > 0);
        }
    }

    // Advance a lag cursor past experts that have no m-blocks left
    CUTLASS_DEVICE void lag_skip_empty(uint32_t& expert, uint32_t& num_tokens, uint32_t& pool_offset, uint32_t& m0) {
        while (expert < kNumExpertsPerRank and m0 >= math::ceil_div(num_tokens, BLOCK_M)) {
            pool_offset += math::ceil_div(num_tokens, BLOCK_M);
            expert += 1;
            m0 = 0;
            num_tokens = expert < kNumExpertsPerRank ? get_num_tokens(expert) : 0u;
        }
    }

    // Decode the CTA's offset inside the current unit chunk into (m, n)
    CUTLASS_DEVICE void lag_decode(const uint32_t& q, const uint32_t& um, const uint32_t& num_n_blocks, const uint32_t& m0) {
        if constexpr (kMulticastOnB) {
            const uint32_t m_in_unit = q % um;
            m_block_idx = m0 + m_in_unit;
            n_block_idx = q / um;
            current_pair_valid = (q % 2 == 0) ? (m_in_unit + 1 < um) : (m_in_unit > 0);
        } else {
            m_block_idx = m0 + q / num_n_blocks;
            n_block_idx = q % num_n_blocks;
            current_pair_valid = false;
        }
    }

    // Dynamic tail (kStopAtTail): the walk stops at the trailing L2-only units with `tail_reached` set and the L2 cursor on the first tail unit
    bool tail_reached = false;

    template <bool kStopAtTail = false>
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block_lag() {
        constexpr uint32_t kUnitM = layout::kSM90FusedLagUnitM;
        while (true) {
            if (l2_expert >= kNumExpertsPerRank)
                return {BlockPhase::None, 0, 0, 0};
            const bool walk_l2 = (num_l2_units_pending > 0) or (l1_expert >= kNumExpertsPerRank);
            if (not walk_l2) {
                const uint32_t um = cute::min(kUnitM, math::ceil_div(l1_num_tokens, BLOCK_M) - l1_m0);
                const uint32_t chunk = um * kNumL1BlockNs;
                if (block_idx < chunk) {
                    current_local_expert_idx = l1_expert;
                    current_num_tokens = l1_num_tokens;
                    current_pool_block_offset = l1_pool_offset;
                    lag_decode(block_idx, um, kNumL1BlockNs, l1_m0);
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, l1_expert, m_block_idx, n_block_idx};
                }
                block_idx -= chunk;
                l1_m0 += um;
                lag_skip_empty(l1_expert, l1_num_tokens, l1_pool_offset, l1_m0);
                num_l1_units_done += 1;
                if (num_l1_units_done > kLagUnitsOnly and (num_l1_units_done - kLagUnitsOnly) % kLagGroupUnits == 0)
                    num_l2_units_pending = kLagGroupUnits;
            } else {
                if constexpr (kStopAtTail) {
                    if (l1_expert >= kNumExpertsPerRank) {
                        tail_reached = true;
                        return {BlockPhase::None, 0, 0, 0};
                    }
                }
                const uint32_t um = cute::min(kUnitM, math::ceil_div(l2_num_tokens, BLOCK_M) - l2_m0);
                const uint32_t chunk = um * kNumL2BlockNs;
                if (block_idx < chunk) {
                    current_local_expert_idx = l2_expert;
                    current_num_tokens = l2_num_tokens;
                    current_pool_block_offset = l2_pool_offset;
                    lag_decode(block_idx, um, kNumL2BlockNs, l2_m0);
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, l2_expert, m_block_idx, n_block_idx};
                }
                block_idx -= chunk;
                l2_m0 += um;
                lag_skip_empty(l2_expert, l2_num_tokens, l2_pool_offset, l2_m0);
                if (num_l2_units_pending > 0)
                    num_l2_units_pending -= 1;
            }
        }
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if constexpr (kMulticastOnB) {
                if (block_idx < num_m_blocks * kNumL1BlockNs) {
                    m_block_idx = block_idx % num_m_blocks;
                    n_block_idx = block_idx / num_m_blocks;
                    return true;
                }
            } else {
                m_block_idx = block_idx / kNumL1BlockNs;
                if (m_block_idx < num_m_blocks) {
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    return true;
                }
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                if constexpr (kMulticastOnB) {
                    m_block_idx = block_idx % num_m_blocks;
                    n_block_idx = block_idx / num_m_blocks;
                } else {
                    m_block_idx = block_idx / kNumL2BlockNs;
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                }
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // Core state machine. kStopAtTail: the wave schedule stops when the L2 phase of the last wave begins, the lag schedule at its trailing L2-only units
    template <bool kStopAtTail = false>
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        if constexpr (kL2LagUnits > 0)
            return get_next_block_lag<kStopAtTail>();
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    // Found a new L1 block (m/n set by the fetcher); jump to next
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // L1 for the current wave is complete, transition to L2
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                    if constexpr (kStopAtTail) {
                        if (get_wave_expert_end_idx() >= kNumExpertsPerRank) {
                            tail_reached = true;
                            return {BlockPhase::None, 0, 0, 0};
                        }
                    }
                }
            } else {
                if (fetch_next_l2_block()) {
                    // Found a new L2 block (m/n set by the fetcher); jump to next
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

    // does the workspace publish the recv counts through per-source flags (SM90FusedWorkspace) or through the summed count (layout::Workspace)?
    template <typename T, typename = void> struct HasHeadLL : std::false_type {};
    template <typename T> struct HasHeadLL<T, std::void_t<decltype(T::kHeadLL)>> : std::bool_constant<T::kHeadLL> {};
    static constexpr bool kHeadLL = HasHeadLL<WorkspaceT>::value;

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        if constexpr (kHeadLL) {
            // low-latency head: lane r waits for source rank r's release flag (its count stores are visible once the acquire
            // returns), the __syncwarp orders every lane behind all the acquires, then each lane sums its experts' counts over the ranks
            DG_STATIC_ASSERT(kNumRanks <= 32, "low-latency head: one flag per lane");
            if (ptx::get_lane_idx() < kNumRanks)
                while (ptx::ld_acq_sys(workspace.get_hll_count_flag_ptr(ptx::get_lane_idx())) == 0u);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                const auto expert_idx = i * 32 + ptx::get_lane_idx();
                uint32_t sum = 0;
                if (expert_idx < kNumExpertsPerRank) {
                    #pragma unroll
                    for (uint32_t r = 0; r < kNumRanks; ++ r)
                        sum += static_cast<uint32_t>(ptx::ld_volatile(workspace.get_expert_recv_count_ptr(r, expert_idx)));
                }
                stored_num_tokens_per_expert[i] = sum;
            }
        } else {
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
        }
        __syncwarp();
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // Wait for all expert counters to be finalized
        fetch_expert_recv_count();

        // Initialize current expert with 0
        set_expert_idx(0);

        // Iterate over all blocks
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }

    // Tile table (SM90): one role walks the schedule with `for_each_block_publish` and stores every tile as a shared-memory
    // entry; the other roles replay the entries with `for_each_block_replay`. The publisher must run at least one tile ahead
    // of every replayer (the B loader does: it only waits on pipeline slots, never on data), and the table must hold
    // ceil(total tiles / kNumSMs) + 1 entries, all initialised to kTileTableNotReady before any replay starts.
    // 8-byte entry, bit-packed with widths derived from the template constants (static_asserts below):
    //   x = expert | m_block << kEBits | n_block << (kEBits + kMBits) | pair_valid << 29 | tag << 30
    //       (tag 0: end of schedule, 1: Linear1, 2: Linear2, 3: not ready)
    //   y = num_tokens | pool_block_offset << kTBits                              (kTBits + kPBits <= 32)
    //   y = valid_m | last_m_block << kVBits | pool_block_offset << kVBits + 1   (otherwise, see kTilePayloadByRows)
    // Publish protocol: payload words first, then the tag word with a release store (CTA scope locally, cluster scope for
    // the DSMEM publish of the dynamic tail); replayers spin on the 32-bit tag word with an acquire load of the matching
    // scope and read the payload afterwards. A single 16 B store / 16 B load pair is not single-copy atomic, hence the
    // two-step protocol.
    static constexpr uint32_t kTileTableNotReady = 0xffffffffu;
    using TileEntry = uint2;

    // Bit widths of the compact entry (the value ranges are template constants)
    CUTLASS_HOST_DEVICE static constexpr uint32_t bits_for(uint32_t max_value) {
        uint32_t bits = 1;
        while (bits < 32 and (max_value >> bits) != 0)
            ++ bits;
        return bits;
    }
    static constexpr uint32_t kEBits = bits_for(kNumExpertsPerRank);
    static constexpr uint32_t kMBits = bits_for(kNumMaxPoolBlocks);
    static constexpr uint32_t kNBits = bits_for(kNumL1BlockNs > kNumL2BlockNs ? kNumL1BlockNs : kNumL2BlockNs);
    static constexpr uint32_t kTBits = bits_for(kNumMaxTokensPerExpert);
    static constexpr uint32_t kPBits = bits_for(kNumMaxPoolBlocks);
    DG_STATIC_ASSERT(kNumMaxPoolBlocks == 0 or (kEBits + kMBits + kNBits <= 29u), "tile table: tag word overflow");
    // Payload word format: the expert's token count and its pool-block offset share the word when they fit; otherwise
    // (kTilePayloadByRows) the entry carries the tile's valid row count, a "last m-block of the expert" flag and the pool offset,
    // and the replayer rebuilds a token count that yields the same `get_valid_m` and the same pair validity as the expert's count.
    // Only those two derived quantities and the pool offset are read after a replay.
    static constexpr bool kTilePayloadByRows = kNumMaxTokensPerExpert != 0 and (kTBits + kPBits > 32u);
    static constexpr uint32_t kVBits = bits_for(BLOCK_M);
    DG_STATIC_ASSERT(not kTilePayloadByRows or (kVBits + 1u + kPBits <= 32u), "tile table: payload word overflow");

    CUTLASS_DEVICE static uint32_t ld_acquire_tile_tag(const TileEntry* entry) {
        uint32_t ret;
        if constexpr (kTileTableClusterSize > 1) {
            asm volatile("ld.acquire.cluster.shared::cta.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(entry)) : "memory");
        } else {
            asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(entry)) : "memory");
        }
        return ret;
    }

    CUTLASS_DEVICE static void st_release_tile_tag(const TileEntry* entry, const uint32_t& tag_word) {
        asm volatile("st.release.cta.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(entry)), "r"(tag_word) : "memory");
    }

    // The same two stores into the shared memory of another CTA of the cluster
    CUTLASS_DEVICE static void st_remote_tile_entry(const TileEntry* entry, const uint32_t& cta_rank,
                                                    const uint32_t& tag_word, const uint32_t& y) {
        const auto local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(entry));
        uint32_t remote_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(local_addr), "r"(cta_rank));
        asm volatile("st.shared::cluster.v2.u32 [%0], {%1, %2};"
                     :: "r"(remote_addr), "r"(kTileTableNotReady), "r"(y) : "memory");
        asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;" :: "r"(remote_addr), "r"(tag_word) : "memory");
    }

    CUTLASS_DEVICE static uint32_t make_compact_tag_word(const uint32_t& expert, const uint32_t& m_block, const uint32_t& n_block,
                                                         const bool& pair_valid, const uint32_t& tag) {
        return expert | (m_block << kEBits) | (n_block << (kEBits + kMBits)) |
               (static_cast<uint32_t>(pair_valid) << 29) | (tag << 30);
    }
    CUTLASS_DEVICE static uint32_t make_compact_payload_word(const uint32_t& num_tokens, const uint32_t& pool_block_offset,
                                                             const uint32_t& m_block) {
        if constexpr (kTilePayloadByRows) {
            const uint32_t valid_m = cute::min(num_tokens - m_block * BLOCK_M, BLOCK_M);
            const uint32_t last_m_block = (m_block + 1 == math::ceil_div(num_tokens, BLOCK_M)) ? 1u : 0u;
            return valid_m | (last_m_block << kVBits) | (pool_block_offset << (kVBits + 1u));
        } else {
            return num_tokens | (pool_block_offset << kTBits);
        }
    }

    CUTLASS_DEVICE void publish_tile_entry(TileEntry* entry, const uint32_t& tag) const {
        ptx::st_shared(entry, kTileTableNotReady, make_compact_payload_word(current_num_tokens, current_pool_block_offset, m_block_idx));
        st_release_tile_tag(entry, make_compact_tag_word(current_local_expert_idx, m_block_idx, n_block_idx, current_pair_valid, tag));
    }

    // Decodes a tile-table entry into the scheduler state; returns the tag
    CUTLASS_DEVICE uint32_t load_tile_entry(const TileEntry* entry) {
        uint32_t tag_word = ld_acquire_tile_tag(entry);
        while ((tag_word >> 30) == 3u)
            tag_word = ld_acquire_tile_tag(entry);
        const uint32_t tag = tag_word >> 30;
        if (tag == 0u)
            return 0u;
        const uint32_t payload = ld_tile_table_payload(entry);
        current_local_expert_idx = tag_word & ((1u << kEBits) - 1u);
        m_block_idx = (tag_word >> kEBits) & ((1u << kMBits) - 1u);
        n_block_idx = (tag_word >> (kEBits + kMBits)) & ((1u << kNBits) - 1u);
        current_pair_valid = ((tag_word >> 29) & 1u) != 0;
        if constexpr (kTilePayloadByRows) {
            const uint32_t valid_m = payload & ((1u << kVBits) - 1u);
            const bool last_m_block = ((payload >> kVBits) & 1u) != 0;
            current_num_tokens = m_block_idx * BLOCK_M + valid_m + (last_m_block ? 0u : BLOCK_M);
            current_pool_block_offset = payload >> (kVBits + 1u);
        } else {
            current_num_tokens = payload & ((1u << kTBits) - 1u);
            current_pool_block_offset = payload >> kTBits;
        }
        return tag;
    }

    CUTLASS_DEVICE static uint32_t ld_tile_table_payload(const uint2* entry) {
        uint32_t ret;
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(ret) : "l"(__cvta_generic_to_shared(entry) + 4) : "memory");
        return ret;
    }

    // Lag schedule with a dynamic tail. The static prefix is walked round robin as in `for_each_block`; the trailing L2-only
    // units are handed out in cluster pairs through `counter`, a per-rank global atomic that the dispatch warps zero after the
    // pre-combine barrier. Only the cluster leader fetches; it publishes its own entry and the partner's (into the partner's table
    // over DSMEM, same index: the static prefix gives both CTAs of a pair the same tile count). Tail tiles wait only on L1
    // outputs, all issued in the static prefix, so any order of tail tiles is deadlock-free. Wave schedule: the tail is the L2
    // phase of the last wave. Table bound: the leader stops fetching tickets once its table cannot hold one more tile plus the
    // end marker and the other clusters take the rest (together they hold kNumSMs * (capacity - 1) - P >= T - P tail slots).
    template <uint32_t kClusterSize, typename Func>
    CUTLASS_DEVICE void for_each_block_publish_dynamic_tail(TileEntry* table, uint32_t* counter, const uint32_t& cta_rank_in_cluster,
                                                            const uint32_t& capacity, Func&& func) {
        DG_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2, "dynamic tail: 1- or 2-CTA clusters");
        DG_STATIC_ASSERT(kClusterSize == kTileTableClusterSize, "dynamic tail: the tag-load scope must cover the publishing CTA");
        constexpr uint32_t kUnitM = layout::kSM90FusedLagUnitM;
        fetch_expert_recv_count();
        set_expert_idx(0);
        uint32_t num_published = 0;
        while (true) {
            CUTE_TIE_DECL(get_next_block<true>(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;
            if (ptx::get_lane_idx() == 0)
                publish_tile_entry(table + num_published, block_phase == BlockPhase::Linear2 ? 2u : 1u);
            ++ num_published;
            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
        if (not tail_reached) {
            if (ptx::get_lane_idx() == 0)
                publish_tile_entry(table + num_published, 0u);
            return;
        }
        if (cta_rank_in_cluster != 0) {
            // partner: the leader publishes our tail entries
            while (true) {
                const uint32_t tag = load_tile_entry(table + num_published);
                if (tag == 0u)
                    return;
                ++ num_published;
                func(BlockPhase::Linear2, current_local_expert_idx, kNumL2BlockKs, m_block_idx, n_block_idx);
            }
        }
        // leader: the cursor is on the first tail unit (lag) / the last wave's first expert (wave)
        uint32_t chunk_start = 0;   // tail-relative position of the current chunk's first tile
        while (true) {
            // table bound (both entry formats): stop before fetching a ticket the table cannot hold together with the end marker
            if (num_published + 2 > capacity) {
                if (ptx::get_lane_idx() == 0) {
                    publish_tile_entry(table + num_published, 0u);
                    if constexpr (kClusterSize > 1)
                        st_remote_tile_entry(table + num_published, 1u, 0u, 0u);   // end marker: tag word 0
                }
                return;
            }
            uint32_t fetched = 0;
            if (ptx::get_lane_idx() == 0)
                fetched = atomicAdd(counter, 1u);
            fetched = __shfl_sync(0xffffffffu, fetched, 0);
            const uint32_t g = fetched * kClusterSize;
            // advance the cursor to the chunk holding g; `um` = m-blocks of that chunk, `m0` its first m-block
            uint32_t um = 0, m0 = 0;
            bool at_end;
            if constexpr (kL2LagUnits > 0) {
                while (l2_expert < kNumExpertsPerRank) {
                    um = cute::min(kUnitM, math::ceil_div(l2_num_tokens, BLOCK_M) - l2_m0);
                    if (g < chunk_start + um * kNumL2BlockNs)
                        break;
                    chunk_start += um * kNumL2BlockNs;
                    l2_m0 += um;
                    lag_skip_empty(l2_expert, l2_num_tokens, l2_pool_offset, l2_m0);
                }
                at_end = l2_expert >= kNumExpertsPerRank;
                current_local_expert_idx = l2_expert;
                current_num_tokens = l2_num_tokens;
                current_pool_block_offset = l2_pool_offset;
                m0 = l2_m0;
            } else {
                while (current_local_expert_idx < kNumExpertsPerRank) {
                    um = get_current_num_m_blocks();
                    if (g < chunk_start + um * kNumL2BlockNs)
                        break;
                    chunk_start += um * kNumL2BlockNs;
                    advance_expert_idx();
                }
                at_end = current_local_expert_idx >= kNumExpertsPerRank;
                m0 = 0;
            }
            if (at_end) {
                if (ptx::get_lane_idx() == 0) {
                    publish_tile_entry(table + num_published, 0u);
                    if constexpr (kClusterSize > 1)
                        st_remote_tile_entry(table + num_published, 1u, 0u, 0u);   // end marker: tag word 0
                }
                return;
            }
            const uint32_t q = g - chunk_start;
            // own tile (position q, even) and the partner's (q + 1), decoded with the chunk's m-inner / n-inner rule
            uint32_t partner_m, partner_n;
            bool partner_pair_valid;
            if constexpr (kMulticastOnB) {
                m_block_idx = m0 + q % um;
                n_block_idx = q / um;
                current_pair_valid = (q % um) + 1 < um;
                partner_m = m0 + (q + 1) % um;
                partner_n = (q + 1) / um;
                partner_pair_valid = ((q + 1) % um) > 0;
            } else {
                m_block_idx = m0 + q / kNumL2BlockNs;
                n_block_idx = q % kNumL2BlockNs;
                current_pair_valid = false;
                partner_m = m0 + (q + 1) / kNumL2BlockNs;
                partner_n = (q + 1) % kNumL2BlockNs;
                partner_pair_valid = false;
            }
            if (ptx::get_lane_idx() == 0) {
                publish_tile_entry(table + num_published, 2u);
                if constexpr (kClusterSize > 1) {
                    st_remote_tile_entry(table + num_published, 1u,
                                         make_compact_tag_word(current_local_expert_idx, partner_m, partner_n, partner_pair_valid, 2u),
                                         make_compact_payload_word(current_num_tokens, current_pool_block_offset, partner_m));
                }
            }
            ++ num_published;
            func(BlockPhase::Linear2, current_local_expert_idx, kNumL2BlockKs, m_block_idx, n_block_idx);
        }
    }

    // Replays the published sequence: `l1_func(expert, num_k_blocks, m_block, n_block)` / `l2_func(...)` for every tile in order,
    // with the getters valid inside. Returns the number of tiles replayed; `replay_next_idx` = the index of the entry after the current tile's.
    uint32_t replay_next_idx = 0;

    template <typename L1Func, typename L2Func>
    CUTLASS_DEVICE uint32_t for_each_block_replay(const TileEntry* table, L1Func&& l1_func, L2Func&& l2_func) {
        uint32_t i = 0;
        while (true) {
            const uint32_t tag = load_tile_entry(table + i);
            if (tag == 0u)
                return i;
            ++ i;
            replay_next_idx = i;
            if (tag == 2u) {
                l2_func(current_local_expert_idx, kNumL2BlockKs, m_block_idx, n_block_idx);
            } else {
                l1_func(current_local_expert_idx, kNumL1BlockKs, m_block_idx, n_block_idx);
            }
        }
    }
};

} // namespace deep_gemm::sched
