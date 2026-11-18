#pragma once

#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched::mega_mhc {

using namespace layout::mega_mhc;

// Executions sharing a split barrier buffer are tagged by an epoch stored with the
// barrier lines, not by %gridid: a CUDA graph kernel node keeps one grid ID across
// replays, so a waiter could accept the previous replay's completed state and read
// its stale split partials. Every CTA reads the same epoch after
// cudaGridDependencySynchronize, and the last CTA to retire advances it.
struct Epoch {
    static CUTLASS_DEVICE uint64_t* get_epoch_ptr(uint64_t* gmem_split_barriers) {
        return math::advance_ptr<uint64_t>(gmem_split_barriers, kSplitBarrierEpochOffset);
    }

    static CUTLASS_DEVICE uint32_t* get_exit_count_ptr(uint64_t* gmem_split_barriers) {
        return math::advance_ptr<uint32_t>(gmem_split_barriers, kSplitBarrierExitCountOffset);
    }

    // NOTES: stable from cudaGridDependencySynchronize until every CTA retires.
    static CUTLASS_DEVICE uint64_t get(uint64_t* gmem_split_barriers) {
        return ptx::ld_acq_gpu(get_epoch_ptr(gmem_split_barriers));
    }

    // One thread per CTA, after the CTA's last barrier use.
    static CUTLASS_DEVICE void retire(uint64_t* gmem_split_barriers, const uint32_t& num_ctas) {
        const auto exit_count_ptr = get_exit_count_ptr(gmem_split_barriers);
        if (ptx::atomic_add_rel(exit_count_ptr, 1) == num_ctas - 1) {
            __threadfence();
            *exit_count_ptr = 0;
            const auto epoch_ptr = get_epoch_ptr(gmem_split_barriers);
            ptx::st_rel(epoch_ptr, *epoch_ptr + 1);
        }
    }
};

// Every split arrives once; wait observes all split writes through the
// terminal release sequence. wait_init prevents a producer from racing the epoch tag.
template <uint32_t kBarrierIndex>
struct SplitBarrier {
    static CUTLASS_DEVICE uint64_t* get_ptr(uint64_t* gmem_split_barriers, const uint32_t m_block_idx) {
        const auto line_idx = static_cast<uint64_t>(kBarrierIndex) * kNumMaxMBlocks + m_block_idx;
        return math::advance_ptr<uint64_t>(gmem_split_barriers, line_idx * kSplitBarrierLineBytes);
    }

    static CUTLASS_DEVICE uint64_t encode_state(const uint64_t epoch, const uint32_t num_arrived_splits = 0) {
        return (epoch + 1) * kNumSplitBarriers * kNumMaxSplits + num_arrived_splits;
    }

    static CUTLASS_DEVICE void init(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        ptx::st_rel(get_ptr(gmem_split_barriers, m_block_idx), encode_state(Epoch::get(gmem_split_barriers)));
    }

    template <uint32_t kNumSplits>
    static CUTLASS_DEVICE void wait_init(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        const auto current_barrier_base = encode_state(Epoch::get(gmem_split_barriers));
        const auto split_barrier_ptr = get_ptr(gmem_split_barriers, m_block_idx);
        // NOTES: unsigned distance rejects stale epoch tags but accepts any current-epoch arrival count.
        while (ptx::ld_acq_gpu(split_barrier_ptr) - current_barrier_base >= kNumSplits);
    }

    template <uint32_t kNumSplits>
    static CUTLASS_DEVICE void arrive(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        const auto state_ptr = get_ptr(gmem_split_barriers, m_block_idx);
        // NOTES: SM100 can retire the release reduction asynchronously while this CTA starts its next task.
        ptx::red_async_inc_rel(state_ptr);
    }

    template <uint32_t kNumSplits>
    static CUTLASS_DEVICE void wait(uint64_t* gmem_split_barriers, const uint32_t& m_block_idx) {
        const auto all_splits_arrived_state = encode_state(Epoch::get(gmem_split_barriers), kNumSplits);
        const auto split_barrier_ptr = get_ptr(gmem_split_barriers, m_block_idx);
        while (ptx::ld_acq_gpu(split_barrier_ptr) != all_splits_arrived_state);
    }
};

// Norm is used only by Shifted after materializing X1 and its statistics.
using Norm = SplitBarrier<0>;
using Mix = SplitBarrier<1>;

} // namespace deep_gemm::sched::mega_mhc
