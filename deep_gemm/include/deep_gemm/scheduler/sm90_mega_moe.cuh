#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/nv_moe_workspace.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

enum class SM90BlockPhase {
    Linear1,
    Linear2
};

// SM90 uses one CTA per scheduled work item and may launch L1 and L2 as
// independent kernels with different grids and traversal orders. Keep those
// semantics out of MegaMoEScheduler, whose default mapping is the SM100
// two-CTA-cluster contract from nv_dev.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          bool kL2NMajorSchedule = false,
          bool kL1NMajorSchedule = false,
          typename WorkspaceT = layout::nv_moe::Workspace,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct SM90MegaMoESchedulerAdapter {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerWave > 0 and kNumExpertsPerWave <= kNumExpertsPerRank,
                     "Invalid wave config");
    DG_STATIC_ASSERT(kNumSMs > 0, "Invalid SM count");

    const WorkspaceT& workspace;

    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit SM90MegaMoESchedulerAdapter(const WorkspaceT& workspace):
        workspace(workspace), block_idx(blockIdx.x) {}

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        const auto aligned = math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
        return cute::min(aligned, kNumExpertsPerRank);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

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
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
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

    CUTLASS_DEVICE void fetch_expert_recv_count() {
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

    template <SM90BlockPhase kPhase>
    CUTLASS_DEVICE bool fetch_next_phase_block() {
        DG_STATIC_ASSERT(kPhase == SM90BlockPhase::Linear1 or kPhase == SM90BlockPhase::Linear2,
                         "Invalid MegaMoE scheduler phase");
        constexpr uint32_t kNumBlockNs =
            kPhase == SM90BlockPhase::Linear1 ? kNumL1BlockNs : kNumL2BlockNs;
        constexpr bool kNMajorSchedule =
            kPhase == SM90BlockPhase::Linear1 ? kL1NMajorSchedule : kL2NMajorSchedule;

        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumBlockNs) {
                if constexpr (kNMajorSchedule) {
                    n_block_idx = block_idx / num_m_blocks;
                    m_block_idx = block_idx - n_block_idx * num_m_blocks;
                } else {
                    m_block_idx = block_idx / kNumBlockNs;
                    n_block_idx = block_idx - m_block_idx * kNumBlockNs;
                }
                return true;
            }

            block_idx -= num_m_blocks * kNumBlockNs;
            advance_expert_idx();
        }
        return false;
    }

    template <SM90BlockPhase kPhase, typename Func>
    CUTLASS_DEVICE void for_each_phase_block(Func&& func) {
        fetch_expert_recv_count();
        set_expert_idx(0);
        while (current_local_expert_idx < kNumExpertsPerRank) {
            if (fetch_next_phase_block<kPhase>()) {
                block_idx += kNumSMs;
                constexpr uint32_t kNumPhaseBlockKs =
                    kPhase == SM90BlockPhase::Linear1 ? kNumL1BlockKs : kNumL2BlockKs;
                func(current_local_expert_idx, kNumPhaseBlockKs, m_block_idx, n_block_idx);
            } else if (current_local_expert_idx >= kNumExpertsPerRank) {
                break;
            }
        }
    }
};

} // namespace deep_gemm::sched
