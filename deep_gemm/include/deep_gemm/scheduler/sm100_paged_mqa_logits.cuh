#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>

// SM100 paged-KV scheduler with per-SM ranges balanced by estimated split cost

namespace deep_gemm::sched {

// Geometry of one request
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV>
struct RequestInfo {
    uint32_t q_token_start;
    uint32_t num_q_tokens;
    uint32_t num_q_blocks;
    uint32_t num_kv_splits;
    uint32_t num_kv_pages;

    CUTLASS_DEVICE RequestInfo() = default;

    CUTLASS_DEVICE RequestInfo(const uint32_t& q_token_start, const uint32_t& num_q_tokens,
                               const uint32_t& context_len):
        q_token_start(q_token_start), num_q_tokens(num_q_tokens),
        num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
        num_kv_splits(math::ceil_div(context_len, SPLIT_KV)),
        num_kv_pages(math::ceil_div(context_len, PAGE_KV)) {}

    // Group consecutive equal indices; the last token has the longest context
    CUTLASS_DEVICE static RequestInfo from_q_token(const uint32_t& q_token_idx,
                                                   const uint32_t& num_q_tokens_total,
                                                   const uint32_t* context_lens,
                                                   const uint32_t* indices) {
        const uint32_t request_id = indices[q_token_idx];
        uint32_t q_token_end_idx = q_token_idx + 1;
        while (q_token_end_idx < num_q_tokens_total and indices[q_token_end_idx] == request_id)
            ++ q_token_end_idx;
        return RequestInfo(q_token_idx, q_token_end_idx - q_token_idx, context_lens[q_token_end_idx - 1]);
    }

    // Distribute request tokens evenly across Q blocks
    CUTLASS_DEVICE void get_q_block_span(const uint32_t& q_block_idx, uint32_t& token_offset, uint32_t& num_tokens) const {
        const uint32_t base = num_q_tokens / num_q_blocks, remainder = num_q_tokens % num_q_blocks;
        token_offset = q_block_idx * base + cute::min(q_block_idx, remainder);
        num_tokens = base + (q_block_idx < remainder ? 1 : 0);
    }
};

// Metadata builder, one CTA: request runs, work prefix sums, per-SM start points
template <uint32_t SPLIT_KV, uint32_t kNumSMs, uint32_t BLOCK_Q, uint32_t kNumThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1)
void sm100_paged_mqa_logits_metadata(const uint32_t num_q_tokens_total,
                                     const uint32_t* context_lens,
                                     const uint32_t* indices,
                                     uint32_t* schedule_meta) {
    DG_STATIC_ASSERT(kNumThreads > 0 and kNumThreads <= 1024 and kNumThreads % 32 == 0, "Invalid thread count");
    DG_STATIC_ASSERT(BLOCK_Q > 0, "Invalid Q block size");
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    const uint32_t thread_idx = threadIdx.x;
    DG_DEVICE_ASSERT(blockDim.x == kNumThreads);
    cudaGridDependencySynchronize();
    if (cutlass::canonical_warp_idx_sync() == 0 and cute::elect_one_sync())
        cutlass::arch::launch_dependent_grids();

    extern __shared__ uint32_t smem[];
    const auto request_q_token_start_idx = smem;                                        // [num_q_tokens_total]
    const auto request_work_prefix = request_q_token_start_idx + num_q_tokens_total;    // [num_q_tokens_total]
    const auto warp_sums = request_work_prefix + num_q_tokens_total;                     // [kNumWarps]
    const auto num_requests_shared = warp_sums + kNumWarps;

    // Scan changes in indices twice: count request starts, then write them
    DG_DEVICE_ASSERT(reinterpret_cast<uintptr_t>(indices) % 8 == 0);
    const auto indices_vec2 = reinterpret_cast<const uint2*>(indices);
    const uint32_t num_tokens_per_thread = math::ceil_div(num_q_tokens_total, kNumThreads * 2) * 2;
    const uint32_t token_begin_idx = cute::min(thread_idx * num_tokens_per_thread, num_q_tokens_total);
    const uint32_t token_end_idx = cute::min(token_begin_idx + num_tokens_per_thread, num_q_tokens_total);
    const auto scan_request_starts = [&](auto&& on_start) {
        uint32_t prev_id = token_begin_idx > 0 and token_begin_idx < token_end_idx ? indices[token_begin_idx - 1] : 0u;
        uint32_t token_idx = token_begin_idx;
        for (; token_idx + 2 <= token_end_idx; token_idx += 2) {
            const uint2 ids = indices_vec2[token_idx / 2];
            if (token_idx == 0 or ids.x != prev_id)
                on_start(token_idx);
            if (ids.y != ids.x)
                on_start(token_idx + 1);
            prev_id = ids.y;
        }
        if (token_idx < token_end_idx and (token_idx == 0 or indices[token_idx] != prev_id))
            on_start(token_idx);
    };
    uint32_t num_request_starts = 0;
    scan_request_starts([&](const uint32_t&) { ++ num_request_starts; });
    uint32_t request_idx = math::cta_exclusive_sum<kNumThreads>(num_request_starts, warp_sums);
    scan_request_starts([&](const uint32_t& token_idx) { request_q_token_start_idx[request_idx ++] = token_idx; });
    if (thread_idx == kNumThreads - 1)
        *num_requests_shared = request_idx;
    __syncthreads();
    const uint32_t num_requests = *num_requests_shared;

    const auto get_request_info = [&](const uint32_t& request_idx,
                                      uint32_t& q_token_start_idx, uint32_t& num_q_tokens, uint32_t& context_len) {
        q_token_start_idx = request_q_token_start_idx[request_idx];
        const uint32_t q_token_end_idx = request_idx + 1 < num_requests ?
                                             request_q_token_start_idx[request_idx + 1] : num_q_tokens_total;
        num_q_tokens = q_token_end_idx - q_token_start_idx;
        context_len = context_lens[q_token_end_idx - 1];
    };

    // KV/MMA cost scales with Q blocks; epilogue cost scales with valid tokens
    const auto get_split_cost = [&](const uint32_t& num_q_tokens) {
        return 2 * BLOCK_Q * math::ceil_div(num_q_tokens, BLOCK_Q) + num_q_tokens;
    };

    // Use the spare request-count bit to flag overflow and fall back to token cost
    constexpr uint32_t kCostOverflowFlag = 0x80000000u;
    bool cost_overflow = false;
    for (uint32_t request_idx = thread_idx; request_idx < num_requests; request_idx += kNumThreads) {
        uint32_t q_token_start_idx, num_q_tokens, context_len;
        get_request_info(request_idx, q_token_start_idx, num_q_tokens, context_len);
        const uint64_t cost = static_cast<uint64_t>(math::ceil_div(context_len, SPLIT_KV)) * get_split_cost(num_q_tokens);
        request_work_prefix[request_idx] = static_cast<uint32_t>(cost);
        cost_overflow |= cost > 0xffffffffu;
    }
    __syncthreads();
    if (num_requests > 0)
        mqa_logits_metadata_prefix_scan<kNumThreads>(thread_idx, num_requests, request_work_prefix, warp_sums);
    for (uint32_t request_idx = thread_idx + 1; request_idx < num_requests; request_idx += kNumThreads)
        cost_overflow |= request_work_prefix[request_idx] < request_work_prefix[request_idx - 1];
    if (cost_overflow)
        atomicOr(num_requests_shared, kCostOverflowFlag);
    __syncthreads();
    const bool use_token_cost = (*num_requests_shared & kCostOverflowFlag) != 0;
    if (use_token_cost) {
        for (uint32_t request_idx = thread_idx; request_idx < num_requests; request_idx += kNumThreads) {
            uint32_t q_token_start_idx, num_q_tokens, context_len;
            get_request_info(request_idx, q_token_start_idx, num_q_tokens, context_len);
            request_work_prefix[request_idx] = math::ceil_div(context_len, SPLIT_KV) * num_q_tokens;
        }
        __syncthreads();
        mqa_logits_metadata_prefix_scan<kNumThreads>(thread_idx, num_requests, request_work_prefix, warp_sums);
    }
    const uint32_t total_cost = num_requests > 0 ? request_work_prefix[num_requests - 1] : 0u;

    // Balance cost across SMs at request split boundaries
    const uint32_t cost_per_sm = total_cost / kNumSMs;
    const uint32_t cost_remainder = total_cost % kNumSMs;
    for (uint32_t sm_idx = thread_idx; sm_idx <= kNumSMs; sm_idx += kNumThreads) {
        const uint32_t target = sm_idx * cost_per_sm + cute::min(sm_idx, cost_remainder);
        const uint32_t request_idx = mqa_logits_metadata_upper_bound(request_work_prefix, num_requests, target);
        uint32_t q_token_idx = num_q_tokens_total, kv_split_idx = 0;    // Tail sentinel: one-past-the-end
        if (request_idx < num_requests) {
            const uint32_t cost_before = request_idx == 0 ? 0u : request_work_prefix[request_idx - 1];
            uint32_t num_q_tokens, context_len;
            get_request_info(request_idx, q_token_idx, num_q_tokens, context_len);
            kv_split_idx = (target - cost_before) / (use_token_cost ? num_q_tokens : get_split_cost(num_q_tokens));
        }
        reinterpret_cast<uint2*>(schedule_meta)[sm_idx] = make_uint2(q_token_idx, kv_split_idx);
    }
}

// Visit requests chunk by chunk, reusing each chunk's KV across its Q blocks
template <uint32_t kNumHeads, uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kSplitsPerChunk>
struct SM100PagedMQALogitsScheduler {
    static constexpr bool kIsPaged = true;
    static constexpr uint32_t kPageKV = PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    // Cache one page coordinate per lane
    static constexpr uint32_t kNumCachedPages = 32;
    DG_STATIC_ASSERT(BLOCK_Q * kNumHeads <= 128, "Invalid Q block shape");
    DG_STATIC_ASSERT(SPLIT_KV % PAGE_KV == 0 and kNumPagesPerSplit <= kNumCachedPages, "Invalid split shape");

    using Info = RequestInfo<BLOCK_Q, SPLIT_KV, PAGE_KV>;

    uint32_t num_q_tokens_total;
    const uint32_t* context_lens;
    const uint32_t* indices;
    const uint32_t* block_table;
    uint32_t block_table_stride;
    uint32_t end_q_token_idx, end_kv_split_idx;     // The next SM's start

    // Request/chunk/Q-block cursor; only current.q_token_start is valid after exhaustion
    Info current;
    uint32_t current_kv_split_base;
    uint32_t current_q_block_in_request;

    // Preserve the emitted task's page lookup state as the request cursor advances
    const uint32_t* task_block_table_row;
    uint32_t task_num_kv_pages;
    uint32_t cached_page_base;
    uint32_t cached_page_coord;

    CUTLASS_DEVICE SM100PagedMQALogitsScheduler(const uint32_t& sm_idx,
                                                const uint32_t& num_q_tokens_total,
                                                const uint32_t* context_lens,
                                                const uint32_t* indices,
                                                const uint32_t* block_table,
                                                const uint32_t& block_table_stride,
                                                const uint32_t* schedule_meta):
            num_q_tokens_total(num_q_tokens_total),
            context_lens(context_lens), indices(indices),
            block_table(block_table), block_table_stride(block_table_stride) {
        const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        end_q_token_idx = end.x;
        end_kv_split_idx = end.y;

        current.q_token_start = start.x;
        current_kv_split_base = start.y;
        current_q_block_in_request = 0;
        if (has_work())
            current = Info::from_q_token(start.x, num_q_tokens_total, context_lens, indices);
    }

    // Whether the current chunk is within this SM's assigned range
    CUTLASS_DEVICE bool has_work() const {
        return current.q_token_start < num_q_tokens_total and
               (current.q_token_start != end_q_token_idx or current_kv_split_base < end_kv_split_idx);
    }

    // Emit the next Q-block task
    CUTLASS_DEVICE bool next_q_block(MQALogitsTask& task) {
        if (not has_work())
            return false;

        // Exclusive split bound of the current request, clamped at the next SM's start
        const uint32_t upper = (current.q_token_start == end_q_token_idx) ? end_kv_split_idx : current.num_kv_splits;
        const uint32_t remaining = upper - current_kv_split_base;
        uint32_t q_block_token_offset;
        current.get_q_block_span(current_q_block_in_request, q_block_token_offset, task.num_q_tokens);
        task.q_token_base = current.q_token_start + q_block_token_offset;
        task.kv_token_base = current_kv_split_base * SPLIT_KV;
        task.num_kv_splits = (current.num_q_blocks == 1) ? remaining : cute::min(remaining, kSplitsPerChunk);
        task.kv_shared_with_prev = current_q_block_in_request > 0;
        task_block_table_row = block_table + current.q_token_start * static_cast<uint64_t>(block_table_stride);
        task_num_kv_pages = current.num_kv_pages;
        cached_page_base = cute::numeric_limits<uint32_t>::max();

        // Advance in Q-block, chunk, request order
        if (++ current_q_block_in_request == current.num_q_blocks) {
            current_q_block_in_request = 0;
            current_kv_split_base += task.num_kv_splits;
            if (current_kv_split_base >= upper and current.q_token_start != end_q_token_idx) {
                // Start the next request at split 0; the final request stops at the SM boundary
                current.q_token_start += current.num_q_tokens;
                current_kv_split_base = 0;
                if (has_work())
                    current = Info::from_q_token(current.q_token_start, num_q_tokens_total, context_lens, indices);
            }
        }
        return true;
    }

    // Also return per-token KV bounds [0, context_len)
    CUTLASS_DEVICE bool next_q_block(MQALogitsTask& task, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        if (not next_q_block(task))
            return false;
        const uint32_t lane_idx = ptx::get_lane_idx();
        const auto row_idx = cute::min(task.q_token_base + lane_idx, num_q_tokens_total - 1);
        const uint32_t lane_k_end = lane_idx < BLOCK_Q ? context_lens[row_idx] : 0;
        #pragma unroll
        for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
            seq_k_start[token_idx] = 0;
            seq_k_end[token_idx] = ptx::exchange(lane_k_end, token_idx);
        }
        return true;
    }

    // Warp-wide page lookup with a sliding cache; out-of-range pages map to zero
    CUTLASS_DEVICE void get_kv_page_coords(const MQALogitsTask& task, const uint32_t& kv_split_idx,
                                           int (&page_coords)[kNumPagesPerSplit]) {
        const uint32_t page_base = task.kv_token_base / PAGE_KV + kv_split_idx * kNumPagesPerSplit;
        if (page_base < cached_page_base or page_base + kNumPagesPerSplit > cached_page_base + kNumCachedPages) {
            const uint32_t page_offset = page_base + ptx::get_lane_idx();
            cached_page_base = page_base;
            cached_page_coord = page_offset < task_num_kv_pages ? task_block_table_row[page_offset] : 0;
        }
        #pragma unroll
        for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx)
            page_coords[page_idx] = ptx::exchange(cached_page_coord, page_base - cached_page_base + page_idx);
    }
};

} // namespace deep_gemm::sched
