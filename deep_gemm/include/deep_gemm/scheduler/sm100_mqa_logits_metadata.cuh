#pragma once

#include <cutlass/arch/grid_dependency_control.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>

// SM100 scheduler for contiguous-KV logits emits per-SM start point (q_block, split_offset) and work counts
// Each Q block priced as kMQALogitsSplitCost * num_kv_splits + kMQALogitsSegmentCost
// both constants are 1 for now

namespace deep_gemm::sched {

inline constexpr uint32_t kNumLogitsMetadataThreads = 256;
inline constexpr uint32_t kNumLogitsMetadataWarps = kNumLogitsMetadataThreads / 32;
DG_STATIC_ASSERT(kNumLogitsMetadataWarps <= 32 and kNumLogitsMetadataThreads % 32 == 0, "Invalid metadata thread count");

// Number of prefix <= target, at most count
template <uint32_t kMaxLog2>
CUTLASS_DEVICE uint32_t mqa_logits_prefix_upper_bound(const uint32_t* prefix, const uint32_t& count,
                                                      const uint32_t& target) {
    uint32_t pos = 0;
    #pragma unroll
    for (uint32_t step = 1u << (kMaxLog2 - 1); step > 0; step >>= 1) {
        const uint32_t candidate = pos + step;
        if (candidate <= count and prefix[candidate - 1] <= target)
            pos = candidate;
    }
    return pos;
}

template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs,
          uint32_t kSegmentCost, uint32_t kSplitCost, uint32_t kMaxNumQBlocksLog2>
CUTLASS_GLOBAL __launch_bounds__(kNumLogitsMetadataThreads, 1)
void sm100_mqa_logits_metadata(const uint32_t num_q_tokens,
                               const uint32_t num_kv_tokens,
                               const uint32_t* cu_seq_len_k_start,
                               const uint32_t* cu_seq_len_k_end,
                               uint32_t* schedule_meta) {
    DG_STATIC_ASSERT(kSegmentCost > 0 and kSplitCost > 0, "Invalid costs");
    DG_STATIC_ASSERT(kNumSMs <= kNumLogitsMetadataThreads, "Invalid SM count");
    constexpr uint32_t kSpanWordOffset = SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs>::kMetaSpanWordOffset;

    const uint32_t thread_idx = threadIdx.x;
    DG_DEVICE_ASSERT(blockDim.x == kNumLogitsMetadataThreads);
    cudaGridDependencySynchronize();
    if (cutlass::canonical_warp_idx_sync() == 0 and cute::elect_one_sync())
        cutlass::arch::launch_dependent_grids();

    const uint32_t num_q_blocks = math::ceil_div(num_q_tokens, BLOCK_Q);
    extern __shared__ uint32_t metadata_storage[];
    const auto work_prefix = metadata_storage;                                                 // [num_q_blocks]
    const auto cost_prefix = work_prefix + num_q_blocks;                                       // [num_q_blocks]
    const auto kv_spans = reinterpret_cast<MQALogitsKVSpan*>(schedule_meta + kSpanWordOffset); // [num_q_blocks]

    for (uint32_t q_block_idx = thread_idx; q_block_idx < num_q_blocks; q_block_idx += kNumLogitsMetadataThreads) {
        const auto span = get_mqa_logits_kv_span<BLOCK_Q, SPLIT_KV>(q_block_idx, num_q_tokens, num_kv_tokens,
                                                                    cu_seq_len_k_start, cu_seq_len_k_end);
        kv_spans[q_block_idx] = span;
        work_prefix[q_block_idx] = span.num_kv_splits;
        cost_prefix[q_block_idx] = kSplitCost * span.num_kv_splits + (span.num_kv_splits == 0 ? 0 : kSegmentCost);
    }
    __syncthreads();

    __shared__ uint64_t warp_sums[kNumLogitsMetadataWarps];
    __shared__ uint64_t carry;
    if (thread_idx == 0)
        carry = 0;
    for (uint32_t base_q_block_idx = 0; base_q_block_idx < num_q_blocks; base_q_block_idx += kNumLogitsMetadataThreads) {
        const uint32_t q_block_idx = base_q_block_idx + thread_idx;
        const bool active = q_block_idx < num_q_blocks;
        // One scan for both prefixes, work in high half, cost in the low
        // both bounded to u32, will not interfere
        const uint64_t value = active ? (static_cast<uint64_t>(work_prefix[q_block_idx]) << 32) + cost_prefix[q_block_idx] : 0;
        // publish the carry, protect warp_sums
        __syncthreads();
        // no data race: read before the scan call, whose internal __syncthreads() separate this read from the next write
        const uint64_t previous_carry = carry;
        const uint64_t scanned = math::cta_exclusive_sum<kNumLogitsMetadataThreads>(value, warp_sums) + value + previous_carry;
        if (active) {
            work_prefix[q_block_idx] = static_cast<uint32_t>(scanned >> 32);
            cost_prefix[q_block_idx] = static_cast<uint32_t>(scanned);
        }
        if (thread_idx == kNumLogitsMetadataThreads - 1)
            carry = scanned;
        __syncthreads();
    }
    // num_q_blocks > 0 is asserted on the host side
    const uint32_t total_work = work_prefix[num_q_blocks - 1];
    const uint32_t total_cost = cost_prefix[num_q_blocks - 1];

    if (thread_idx >= kNumSMs)
        return;
    const uint32_t base = total_cost / kNumSMs;
    const uint32_t remainder = total_cost % kNumSMs;
    struct Boundary {
        uint2 start;
        uint32_t coordinate;
    };
    const auto locate = [&](const uint32_t& boundary_idx) -> Boundary {
        const uint32_t target = boundary_idx * base + cute::min(boundary_idx, remainder);
        if (target == total_cost)
            return {make_uint2(num_q_blocks, 0), total_work}; // Tail sentinel: one-past-the-end
        const uint32_t q_block_idx = mqa_logits_prefix_upper_bound<kMaxNumQBlocksLog2>(cost_prefix, num_q_blocks, target);
        const uint32_t cost_before = q_block_idx == 0 ? 0 : cost_prefix[q_block_idx - 1];
        const uint32_t work_before = q_block_idx == 0 ? 0 : work_prefix[q_block_idx - 1];
        // Targets inside a block land after its segment charge; the clamp
        // keeps the boundary before the block's last split
        const uint32_t split_offset = cute::min((cute::max(target - cost_before, kSegmentCost) - kSegmentCost) / kSplitCost,
                                                work_prefix[q_block_idx] - work_before - 1);
        return {make_uint2(q_block_idx, split_offset), work_before + split_offset};
    };
    const auto begin = locate(thread_idx);
    const auto end = locate(thread_idx + 1);
    reinterpret_cast<uint2*>(schedule_meta)[thread_idx] = begin.start;
    schedule_meta[2 * kNumSMs + thread_idx] = end.coordinate - begin.coordinate;
}

} // namespace deep_gemm::sched
