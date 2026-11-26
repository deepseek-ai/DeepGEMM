#pragma once

#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::sched::mega_mhc {

using namespace layout::mega_mhc;

// Zero-initialized counters require cooperative launches and ordered reuse.
// After local registration, zero means all readers have registered and all splits have arrived.
template <uint32_t kBarrierIndex, uint32_t kTokensPerTask>
struct SplitBarrier {
    DG_STATIC_ASSERT(kBarrierIndex < kNumSplitBarriers, "Invalid split barrier index");

    static CUTLASS_DEVICE uint32_t* get_ptr(uint32_t* barriers, const uint32_t m_block_idx) {
        return barriers + (kBarrierIndex * kNumMaxMBlocks + m_block_idx) * (kSplitBarrierLineBytes / sizeof(uint32_t));
    }

    template <uint32_t kNumSplits, uint32_t kNumSMs>
    static CUTLASS_DEVICE void prepare(uint32_t* barriers, const uint32_t num_tokens,
                                     const uint32_t num_partitions = 1) {
        // Groups of four warp tasks are assigned round-robin to CTAs.
        const uint32_t num_task_groups_per_m_block = BLOCK_M / kTokensPerTask * num_partitions / kNumWarpsPerWG;
        const uint32_t num_task_groups = math::ceil_div(math::ceil_div(num_tokens, kTokensPerTask) * num_partitions, kNumWarpsPerWG);
        const uint32_t first_idx = blockIdx.x + threadIdx.x * kNumSMs;

        // One initializer per m-block; RMW preserves registrations and arrivals that raced ahead.
        for (uint32_t m_block_idx = first_idx; m_block_idx < math::ceil_div(num_tokens, BLOCK_M);
             m_block_idx += kNumSMs * kNumThreads) {
            const uint32_t num_readers = cute::min(kNumSMs, cute::min(num_task_groups_per_m_block,
                num_task_groups - m_block_idx * num_task_groups_per_m_block));
            ptx::atomic_add(get_ptr(barriers, m_block_idx), 0u - kNumSplits - num_readers);
        }

        // Register only the first task group this CTA owns in each m-block.
        for (uint32_t task_group_idx = first_idx; task_group_idx < num_task_groups; task_group_idx += kNumSMs * kNumThreads) {
            if (task_group_idx % num_task_groups_per_m_block < kNumSMs)
                ptx::atomic_add(get_ptr(barriers, task_group_idx / num_task_groups_per_m_block), 1u);
        }
    }

    static CUTLASS_DEVICE void arrive(uint32_t* barriers, const uint32_t m_block_idx) {
        ptx::red_async_inc_rel(get_ptr(barriers, m_block_idx));
    }

    static CUTLASS_DEVICE void wait(uint32_t* barriers, const uint32_t m_block_idx) {
        while (ptx::ld_acq(get_ptr(barriers, m_block_idx)) != 0);
    }
};

// Norm waits for materialized X1 and its statistics
using Norm = SplitBarrier<0, kNumTokensPerNormTask>;
using Mix = SplitBarrier<1, 1>;

} // namespace deep_gemm::sched::mega_mhc
