#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/utils.cuh>

// SM100 paged scheduler: metadata emits per-SM (q_token_idx, kv_split_idx) starts
// Device traversal walks chunk-outer / Q-block-inner tasks

namespace deep_gemm::sched {

// Per-request geometry accessor; this is where varlen and non-varlen diverge
template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV>
struct RequestInfo {
    uint32_t q_token_start;       // request_q_token_start
    uint32_t num_q_tokens;        // request_num_q_tokens
    uint32_t num_q_blocks;        // request_num_q_blocks
    uint32_t num_kv_splits;       // request_num_kv_splits  = ceil(context_len / SPLIT_KV)
    uint32_t num_kv_pages;        // = ceil(context_len / PAGE_KV); page-level bound for the last partial split

    // Resolve the request that starts at `q_token_idx`
    CUTLASS_DEVICE static RequestInfo from_q_token(const uint32_t& q_token_idx,
                                                   const uint32_t& num_q_tokens_total,
                                                   const uint32_t* context_lens,
                                                   const uint32_t* indices) {
        RequestInfo info;
        info.q_token_start = q_token_idx;
        uint32_t context_len;
        if constexpr (kIsVarlen) {
            // Varlen request = maximal run of equal `indices`
            const uint32_t request_id = indices[q_token_idx];
            uint32_t t = q_token_idx;
            while (t + 1 < num_q_tokens_total and indices[t + 1] == request_id)
                ++ t;
            info.num_q_tokens = t - q_token_idx + 1;
            context_len = context_lens[t];
        } else {
            // Regular grid: request = q_token_idx / next_n, next_n tokens each
            const uint32_t request_id = q_token_idx / kNextN;
            info.num_q_tokens = kNextN;
            const uint32_t lens_idx = kIsContextLens2D ? request_id * kNextN + kNextN - 1 : request_id;
            context_len = context_lens[lens_idx];
        }
        info.num_q_blocks = math::ceil_div(info.num_q_tokens, BLOCK_Q);
        info.num_kv_splits = math::ceil_div(context_len, SPLIT_KV);
        info.num_kv_pages = math::ceil_div(context_len, PAGE_KV);
        return info;
    }

    // Average q-token partition across Q-blocks; returns both offset and count
    CUTLASS_DEVICE void get_q_block_span(const uint32_t& q, uint32_t& token_offset, uint32_t& num_tokens) const {
        const uint32_t base = num_q_tokens / num_q_blocks, rem = num_q_tokens % num_q_blocks;
        token_offset = q * base + (q < rem ? q : rem);
        num_tokens = base + (q < rem ? 1 : 0);
    }

    // block_table row for this request: request id for non-varlen, token row for varlen
    CUTLASS_DEVICE uint32_t get_block_table_row() const {
        if constexpr (kIsVarlen)
            return q_token_start;
        else
            return q_token_start / kNextN;
    }
};

// Metadata kernel balances work across SMs via a prefix sum over request work
template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(256, 1)
void sm100_paged_mqa_logits_metadata(const uint32_t num_requests,
                                     const uint32_t num_q_tokens_total,
                                     const uint32_t* context_lens,
                                     const uint32_t* indices,
                                     uint32_t* schedule_meta) {
    // PAGE_KV is unused for metadata; pass SPLIT_KV as a placeholder
    using Info = RequestInfo<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_Q, SPLIT_KV, SPLIT_KV>;

    cudaGridDependencySynchronize();  // wait for the primary kernel (CDP launch)

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t num_threads = blockDim.x;

    // smem: per-request work prefix sum + request start token
    extern __shared__ uint32_t smem[];
    uint32_t* prefix_work = smem;                       // [num_requests]
    uint32_t* request_q_token_start = smem + num_requests;  // [num_requests]

    // Logical requests are regular requests for non-varlen and index runs for varlen
    uint32_t num_logical_requests;
    if constexpr (kIsVarlen) {
        if (thread_idx == 0) {
            uint32_t r = 0, t = 0;
            while (t < num_q_tokens_total) {
                request_q_token_start[r] = t;
                const uint32_t request_id = indices[t];
                while (t < num_q_tokens_total and indices[t] == request_id)
                    ++ t;
                ++ r;
            }
            // Temporarily stash run count for broadcast
            prefix_work[0] = r;  // temporary: run count
        }
        __syncthreads();
        num_logical_requests = prefix_work[0];
        __syncthreads();
    } else {
        num_logical_requests = num_requests;
        for (uint32_t r = thread_idx; r < num_logical_requests; r += num_threads)
            request_q_token_start[r] = r * kNextN;
        __syncthreads();
    }

    // Work per request before prefix sum
    for (uint32_t r = thread_idx; r < num_logical_requests; r += num_threads) {
        const auto info = Info::from_q_token(request_q_token_start[r],
                                             num_q_tokens_total, context_lens, indices);
        prefix_work[r] = info.num_kv_splits * info.num_q_tokens;
    }
    __syncthreads();

    // Inclusive prefix sum by one warp
    if (warp_idx == 0) {
        uint32_t carry = 0;
        for (uint32_t base = 0; base < num_logical_requests; base += 32) {
            const uint32_t r = base + lane_idx;
            const uint32_t v = (r < num_logical_requests) ? prefix_work[r] : 0u;
            const uint32_t scanned = math::warp_inclusive_sum(v, lane_idx) + carry;
            if (r < num_logical_requests)
                prefix_work[r] = scanned;
            carry = __shfl_sync(0xffffffff, scanned, 31);
        }
    }
    __syncthreads();

    const uint32_t num_total_work = num_logical_requests > 0 ? prefix_work[num_logical_requests - 1] : 0u;

    // Each thread emits one SM start; remainder is assigned to earlier SMs
    const uint32_t q = num_total_work / kNumSMs, rem = num_total_work % kNumSMs;
    for (uint32_t sm_idx = thread_idx; sm_idx <= kNumSMs; sm_idx += num_threads) {
        const uint32_t w = sm_idx * q + (sm_idx < rem ? sm_idx : rem);
        // First request whose prefix_work owns work unit `w`
        uint32_t lo = 0, hi = num_logical_requests;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (prefix_work[mid] <= w) lo = mid + 1; else hi = mid;
        }
        const uint32_t request_idx = lo;
        uint32_t q_token_idx, kv_split_idx;
        if (request_idx < num_logical_requests) {
            const uint32_t work_before = (request_idx == 0) ? 0u : prefix_work[request_idx - 1];
            const uint32_t w_in_request = w - work_before;
            const auto info = Info::from_q_token(request_q_token_start[request_idx],
                                                 num_q_tokens_total, context_lens, indices);
            // Align SM starts to request/split boundaries
            q_token_idx = info.q_token_start;
            kv_split_idx = w_in_request / info.num_q_tokens;
        } else {
            // Tail sentinel: one-past-the-end
            q_token_idx = num_q_tokens_total;
            kv_split_idx = 0;
        }
        schedule_meta[sm_idx * 2] = q_token_idx;
        schedule_meta[sm_idx * 2 + 1] = kv_split_idx;
    }
}

// Device scheduler walks this SM's schedule range and implements SchedulerConcept
// All specialized warps instantiate it and advance through the same task sequence
template <bool kHasIndices> struct SM100IndicesStorage { const uint32_t* indices; };
template <> struct SM100IndicesStorage<false> {};

template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumHeads, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kSplitsPerChunk>
struct SM100PagedMQALogitsScheduler : SM100IndicesStorage<kIsVarlen> {
    // SchedulerConcept descriptors
    static constexpr bool kIsPaged = true;
    static constexpr bool kHasPartialBlock = true;
    static constexpr uint32_t kPageKV = PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;

    using Info = RequestInfo<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_Q, SPLIT_KV, PAGE_KV>;

    const uint32_t* context_lens;
    const uint32_t* block_table;
    uint32_t block_table_stride;
    uint32_t num_q_tokens_total;

    // Walk state
    Info cur;                           // current request geometry
    uint32_t cur_kv_split_base;         // current chunk start (request-internal split)
    uint32_t cur_q_block_in_request;    // current Q-block within the request
    uint32_t end_q_token_idx, end_kv_split_idx;
    bool done;

    // Geometry stashed by `next_q_block` for the accessors below
    uint32_t cur_block_table_row;       // request's block-table row
    uint32_t cur_q_block_token_base;    // global first-token row of this Q-block
    uint32_t cur_num_block_tokens;      // valid tokens in this Q-block
    uint32_t cur_request_num_kv_pages;  // ceil(context_len / PAGE_KV); bound for last partial split

    CUTLASS_DEVICE const uint32_t* get_indices() const {
        if constexpr (kIsVarlen)
            return this->indices;
        return nullptr;
    }

    CUTLASS_DEVICE SM100PagedMQALogitsScheduler(const uint32_t& sm_idx,
                                                const uint32_t* context_lens,
                                                const uint32_t* schedule_meta,
                                                const uint32_t* indices,
                                                const uint32_t* block_table,
                                                const uint32_t& block_table_stride,
                                                const uint32_t& num_q_tokens_total) {
        this->context_lens = context_lens;
        this->block_table = block_table;
        this->block_table_stride = block_table_stride;
        this->num_q_tokens_total = num_q_tokens_total;
        if constexpr (kIsVarlen)
            this->indices = indices;

        const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        end_q_token_idx = end.x;
        end_kv_split_idx = end.y;

        cur_kv_split_base = start.y;
        cur_q_block_in_request = 0;
        done = (start.x >= num_q_tokens_total) or
               (start.x == end_q_token_idx and start.y >= end_kv_split_idx);
        if (not done)
            cur = Info::from_q_token(start.x, num_q_tokens_total, context_lens, get_indices());

        cur_block_table_row = 0;
        cur_q_block_token_base = 0;
        cur_num_block_tokens = 1;
        cur_request_num_kv_pages = 0;
    }

    // Exclusive split bound for the current request, clamped at the next SM start
    CUTLASS_DEVICE uint32_t get_cur_kv_split_upper() const {
        return (cur.q_token_start == end_q_token_idx) ? end_kv_split_idx : cur.num_kv_splits;
    }

    // Emit the next (Q-block, chunk) task and stash its addressing geometry
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_split_base, uint32_t& num_kv_splits) {
        q_block_idx = 0;  // addressing uses stashed state
        if (done)
            return false;

        // Capture emitted task geometry before advancing state
        const uint32_t upper = get_cur_kv_split_upper();
        cur_block_table_row = cur.get_block_table_row();
        uint32_t q_block_token_offset, q_block_num_tokens;
        cur.get_q_block_span(cur_q_block_in_request, q_block_token_offset, q_block_num_tokens);
        cur_q_block_token_base = cur.q_token_start + q_block_token_offset;
        cur_num_block_tokens = q_block_num_tokens;
        cur_request_num_kv_pages = cur.num_kv_pages;
        kv_split_base = cur_kv_split_base;
        const uint32_t remaining = upper - cur_kv_split_base;   // upper > cur_kv_split_base (guarded by `done`)
        num_kv_splits = (cur.num_q_blocks == 1) ? remaining
                                               : (remaining < kSplitsPerChunk ? remaining : kSplitsPerChunk);

        // Advance in Q-block, chunk, request order
        ++ cur_q_block_in_request;
        if (cur_q_block_in_request == cur.num_q_blocks) {
            cur_q_block_in_request = 0;
            cur_kv_split_base += num_kv_splits;
            if (cur_kv_split_base >= upper) {
                if (cur.q_token_start == end_q_token_idx) {
                    // Reached the next SM's start
                    done = true;
                } else {
                    // Move to next request owned from split 0
                    const uint32_t next_q_token = cur.q_token_start + cur.num_q_tokens;
                    cur_kv_split_base = 0;
                    if (next_q_token >= num_q_tokens_total)
                        done = true;
                    else {
                        cur = Info::from_q_token(next_q_token, num_q_tokens_total, context_lens, get_indices());
                        // The new request may already be this SM's end
                        if (cur.q_token_start == end_q_token_idx and end_kv_split_idx == 0)
                            done = true;
                    }
                }
            }
        }
        return true;
    }

    CUTLASS_DEVICE uint32_t get_num_block_tokens(const uint32_t&) const {
        return cur_num_block_tokens;
    }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t&) const {
        return cur_q_block_token_base;
    }

    CUTLASS_DEVICE static uint32_t get_kv_tma_offset(const uint32_t& kv_split_base, const uint32_t& kv_split_idx) {
        return (kv_split_base + kv_split_idx) * SPLIT_KV;
    }

    CUTLASS_DEVICE uint32_t get_kv_page_coord_by_page_offset(const uint32_t& page_offset) const {
        if (page_offset >= cur_request_num_kv_pages)
            return 0;
        const auto block_table_offset = cur_block_table_row * static_cast<uint64_t>(block_table_stride);
        return block_table[block_table_offset + page_offset];
    }

    CUTLASS_DEVICE uint32_t get_logits_row(const uint32_t&, const uint32_t& token_idx) const {
        return cur_q_block_token_base + token_idx;
    }

    CUTLASS_DEVICE uint32_t get_logits_col(const uint32_t& kv_split_base,
                                           const uint32_t& kv_split_idx,
                                           const uint32_t& math_thread_idx) const {
        return (kv_split_base + kv_split_idx) * SPLIT_KV + math_thread_idx;
    }
};

} // namespace deep_gemm::sched
