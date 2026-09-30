#pragma once

#include <cutlass/arch/grid_dependency_control.h>

#include <deep_gemm/common/math.cuh>

// SM100 contiguous-KV scheduling via grid-stride traversal or per-SM metadata

namespace deep_gemm::sched {

// A Q block and its KV splits, shared by contiguous and paged schedulers
struct MQALogitsTask {
    uint32_t q_token_base;      // First Q token and logits row
    uint32_t num_q_tokens;      // Valid tokens in the Q block (always BLOCK_Q for contiguous KV)
    uint32_t kv_token_base;     // KV base aligned to 4 tokens (SPLIT_KV for paged KV)
    uint32_t num_kv_splits;
    bool kv_shared_with_prev;   // Same KV as prior task; false for first task or contiguous KV
};

// KV range of one Q block: covers `[k_start, k_end)` of all its rows
struct MQALogitsKVSpan {
    uint32_t kv_token_base;     // `min k_start` over the rows, rounded down to 4 tokens
    uint32_t num_kv_splits;     // `ceil((max k_end - kv_token_base) / SPLIT_KV)`

    CUTLASS_HOST_DEVICE MQALogitsKVSpan() = default;

    CUTLASS_HOST_DEVICE MQALogitsKVSpan(const uint32_t& kv_token_base, const uint32_t& num_kv_splits):
        kv_token_base(kv_token_base), num_kv_splits(num_kv_splits) {}
};

// Metadata int32 words: per-SM starts (2), split counts (1), then 8-byte-aligned spans
CUTLASS_HOST_DEVICE constexpr uint32_t get_mqa_logits_metadata_span_word_offset(const uint32_t& num_sms) {
    return (3 * num_sms + 1) / 2 * 2;
}

// NOTES: out-of-range rows are clamped to the last valid row
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV>
CUTLASS_DEVICE MQALogitsKVSpan get_mqa_logits_kv_span(const uint32_t& q_block_idx,
                                                      const uint32_t& num_q_tokens, const uint32_t& num_kv_tokens,
                                                      const uint32_t* cu_seq_len_k_start,
                                                      const uint32_t* cu_seq_len_k_end) {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();
    #pragma unroll 8
    for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
        const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
        const auto k_start = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
        const auto k_end = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
        start = cute::min(start, k_start);
        end = cute::max(end, k_end);
    }
    const uint32_t kv_token_base = start / 4 * 4;
    return MQALogitsKVSpan(kv_token_base, math::ceil_div(end - kv_token_base, SPLIT_KV));
}

// CTA-wide inclusive prefix sum of `values[0, num_items)` in shared memory
template <uint32_t kNumThreads>
CUTLASS_DEVICE void mqa_logits_metadata_prefix_scan(const uint32_t& thread_idx, const uint32_t& num_items,
                                                    uint32_t* values, uint32_t* warp_sums) {
    const uint32_t num_items_per_thread = math::ceil_div(num_items, kNumThreads) | 1u;
    const uint32_t item_begin_idx = cute::min(thread_idx * num_items_per_thread, num_items);
    const uint32_t item_end_idx = cute::min(item_begin_idx + num_items_per_thread, num_items);
    uint32_t even_sum = 0, odd_sum = 0;
    uint32_t item_idx = item_begin_idx;
    for (; item_idx + 2 <= item_end_idx; item_idx += 2) {
        even_sum += values[item_idx];
        values[item_idx] = even_sum + odd_sum;
        odd_sum += values[item_idx + 1];
        values[item_idx + 1] = odd_sum + even_sum;
    }
    if (item_idx < item_end_idx) {
        even_sum += values[item_idx];
        values[item_idx] = even_sum + odd_sum;
    }
    const uint32_t thread_offset = math::cta_exclusive_sum<kNumThreads>(even_sum + odd_sum, warp_sums);
    for (item_idx = item_begin_idx; item_idx < item_end_idx; ++ item_idx)
        values[item_idx] += thread_offset;
    __syncthreads();
}

// Index of the first prefix > target, or count if none
CUTLASS_DEVICE uint32_t mqa_logits_metadata_upper_bound(const uint32_t* prefix, const uint32_t& count,
                                                        const uint32_t& target) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (prefix[mid] <= target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Build KV spans, work/cost prefix sums and per-SM starts in one CTA
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs,
          uint32_t kSegmentCost, uint32_t kSplitCost, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1)
void sm100_mqa_logits_metadata(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                               const uint32_t* cu_seq_len_k_start,
                               const uint32_t* cu_seq_len_k_end,
                               uint32_t* schedule_meta) {
    DG_STATIC_ASSERT(kNumThreads > 0 and kNumThreads <= 1024 and kNumThreads % 32 == 0, "Invalid thread count");
    DG_STATIC_ASSERT(kSegmentCost > 0 and kSplitCost > 0, "Invalid costs");
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    const uint32_t thread_idx = threadIdx.x;
    DG_DEVICE_ASSERT(blockDim.x == kNumThreads);
    cudaGridDependencySynchronize();
    if (cutlass::canonical_warp_idx_sync() == 0 and cute::elect_one_sync())
        cutlass::arch::launch_dependent_grids();

    // NOTES: `num_q_blocks > 0` is asserted on the host side
    const uint32_t num_q_blocks = math::ceil_div(num_q_tokens, BLOCK_Q);
    extern __shared__ uint32_t smem[];
    const auto work_prefix = smem;                              // [num_q_blocks]
    const auto cost_prefix = work_prefix + num_q_blocks;        // [num_q_blocks]
    const auto warp_sums = cost_prefix + num_q_blocks;          // [kNumWarps]
    const auto kv_spans = reinterpret_cast<MQALogitsKVSpan*>(
        schedule_meta + get_mqa_logits_metadata_span_word_offset(kNumSMs));

    // Per-block spans, work and cost
    for (uint32_t q_block_idx = thread_idx; q_block_idx < num_q_blocks; q_block_idx += kNumThreads) {
        const auto span = get_mqa_logits_kv_span<BLOCK_Q, SPLIT_KV>(q_block_idx, num_q_tokens, num_kv_tokens,
                                                                    cu_seq_len_k_start, cu_seq_len_k_end);
        kv_spans[q_block_idx] = span;
        work_prefix[q_block_idx] = span.num_kv_splits;
        cost_prefix[q_block_idx] = kSplitCost * span.num_kv_splits + (span.num_kv_splits == 0 ? 0 : kSegmentCost);
    }
    __syncthreads();
    mqa_logits_metadata_prefix_scan<kNumThreads>(thread_idx, num_q_blocks, work_prefix, warp_sums);
    mqa_logits_metadata_prefix_scan<kNumThreads>(thread_idx, num_q_blocks, cost_prefix, warp_sums);
    const uint32_t total_work = work_prefix[num_q_blocks - 1];
    const uint32_t total_cost = cost_prefix[num_q_blocks - 1];

    // Balance cost across SMs at KV split boundaries
    struct Boundary {
        uint2 start;            // (q_block_idx, split_offset)
        uint32_t work_offset;   // Splits before the boundary

        CUTLASS_DEVICE Boundary(const uint2& start, const uint32_t& work_offset): start(start), work_offset(work_offset) {}
    };
    const uint32_t cost_per_sm = total_cost / kNumSMs;
    const uint32_t cost_remainder = total_cost % kNumSMs;
    const auto locate = [&](const uint32_t& boundary_idx) -> Boundary {
        const uint32_t target = boundary_idx * cost_per_sm + cute::min(boundary_idx, cost_remainder);
        if (target == total_cost)
            return Boundary(make_uint2(num_q_blocks, 0), total_work);   // Tail sentinel: one-past-the-end
        const uint32_t q_block_idx = mqa_logits_metadata_upper_bound(cost_prefix, num_q_blocks, target);
        const uint32_t cost_before = q_block_idx == 0 ? 0 : cost_prefix[q_block_idx - 1];
        const uint32_t work_before = q_block_idx == 0 ? 0 : work_prefix[q_block_idx - 1];
        // Subtract Q-block startup cost and clamp to the block's last split
        const uint32_t split_offset = cute::min((cute::max(target - cost_before, kSegmentCost) - kSegmentCost) / kSplitCost,
                                                work_prefix[q_block_idx] - work_before - 1);
        return Boundary(make_uint2(q_block_idx, split_offset), work_before + split_offset);
    };
    for (uint32_t sm_idx = thread_idx; sm_idx < kNumSMs; sm_idx += kNumThreads) {
        const auto begin = locate(sm_idx);
        const auto end = locate(sm_idx + 1);
        reinterpret_cast<uint2*>(schedule_meta)[sm_idx] = begin.start;
        schedule_meta[2 * kNumSMs + sm_idx] = end.work_offset - begin.work_offset;
    }
}

// Emit tasks from this SM's grid-stride Q blocks or scheduled range
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs, bool kUseSchedule = false>
struct SM100MQALogitsScheduler {
    static constexpr bool kIsPaged = false;

    uint32_t num_q_blocks;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    const uint32_t* cu_seq_len_k_start;
    const uint32_t* cu_seq_len_k_end;
    const MQALogitsKVSpan* kv_spans = nullptr;      // Scheduled mode only

    // Q-block cursor, starting split and remaining scheduled work
    uint32_t current_q_block_idx;
    uint32_t current_split_offset = 0;
    uint32_t remaining_splits = 0;

    CUTLASS_DEVICE SM100MQALogitsScheduler(const uint32_t& sm_idx,
                                           const uint32_t& num_q_tokens,
                                           const uint32_t& num_kv_tokens,
                                           const uint32_t* cu_seq_len_k_start,
                                           const uint32_t* cu_seq_len_k_end,
                                           const uint32_t* schedule_meta = nullptr):
            num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
            num_q_tokens(num_q_tokens), num_kv_tokens(num_kv_tokens),
            cu_seq_len_k_start(cu_seq_len_k_start), cu_seq_len_k_end(cu_seq_len_k_end),
            current_q_block_idx(sm_idx) {
        DG_STATIC_ASSERT(kNumSMs > 0, "Invalid SM count");
        if constexpr (kUseSchedule) {
            const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
            current_q_block_idx = start.x;
            current_split_offset = start.y;
            remaining_splits = schedule_meta[2 * kNumSMs + sm_idx];
            kv_spans = reinterpret_cast<const MQALogitsKVSpan*>(
                schedule_meta + get_mqa_logits_metadata_span_word_offset(kNumSMs));
        }
    }

    // Emit the next Q-block task
    CUTLASS_DEVICE bool next_q_block(MQALogitsTask& task) {
        return advance<false>(task, nullptr, nullptr);
    }

    // Also return clamped KV bounds per token; tail rows reuse the last valid row
    CUTLASS_DEVICE bool next_q_block(MQALogitsTask& task, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return advance<true>(task, seq_k_start, seq_k_end);
    }

    template <bool kLoadSeqBounds>
    CUTLASS_DEVICE bool advance(MQALogitsTask& task, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        task.num_q_tokens = BLOCK_Q;
        task.kv_shared_with_prev = false;
        if constexpr (kUseSchedule) {
            while (remaining_splits > 0 and current_q_block_idx < num_q_blocks) {
                const auto span = kv_spans[current_q_block_idx];
                const auto split_offset = current_split_offset;
                current_split_offset = 0;
                if (split_offset < span.num_kv_splits) {
                    if constexpr (kLoadSeqBounds) {
                        #pragma unroll
                        for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
                            const auto row_idx = cute::min(current_q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
                            seq_k_start[token_idx] = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
                            seq_k_end[token_idx] = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
                        }
                    }
                    task.q_token_base = current_q_block_idx * BLOCK_Q;
                    task.kv_token_base = span.kv_token_base + split_offset * SPLIT_KV;
                    task.num_kv_splits = cute::min(span.num_kv_splits - split_offset, remaining_splits);
                    remaining_splits -= task.num_kv_splits;
                    ++ current_q_block_idx;
                    return true;
                }
                ++ current_q_block_idx;
            }
            return false;
        } else {
            if (current_q_block_idx >= num_q_blocks)
                return false;
            const auto q_block_idx = current_q_block_idx;
            current_q_block_idx += kNumSMs;
            task.q_token_base = q_block_idx * BLOCK_Q;
            if constexpr (kLoadSeqBounds) {
                // The bounds of the rows give the span, so load them only once
                uint32_t start = cute::numeric_limits<uint32_t>::max();
                uint32_t end = cute::numeric_limits<uint32_t>::min();
                #pragma unroll
                for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
                    const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
                    const auto k_start = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
                    const auto k_end = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
                    seq_k_start[token_idx] = k_start;
                    seq_k_end[token_idx] = k_end;
                    start = cute::min(start, k_start);
                    end = cute::max(end, k_end);
                }
                task.kv_token_base = start / 4 * 4;
                task.num_kv_splits = math::ceil_div(end - task.kv_token_base, SPLIT_KV);
            } else {
                const auto span = get_mqa_logits_kv_span<BLOCK_Q, SPLIT_KV>(q_block_idx, num_q_tokens, num_kv_tokens,
                                                                            cu_seq_len_k_start, cu_seq_len_k_end);
                task.kv_token_base = span.kv_token_base;
                task.num_kv_splits = span.num_kv_splits;
            }
            return true;
        }
    }
};

} // namespace deep_gemm::sched
