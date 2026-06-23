#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

// SM100 contiguous-KV scheduler; translates row bounds into core task geometry

namespace deep_gemm::sched {

template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs>
struct SM100MQALogitsScheduler {
    static constexpr bool kIsPaged = false;
    static constexpr bool kHasPartialBlock = false;
    static constexpr uint32_t kPageKV = 0;

    uint32_t current_q_block_idx;
    uint32_t num_q_blocks;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    const uint32_t* cu_seq_len_k_start;
    const uint32_t* cu_seq_len_k_end;
    uint32_t* seq_k_start;
    uint32_t* seq_k_end;

    CUTLASS_DEVICE SM100MQALogitsScheduler(const uint32_t& sm_idx,
                                           const uint32_t& num_q_tokens,
                                           const uint32_t& num_kv_tokens,
                                           const uint32_t* cu_seq_len_k_start,
                                           const uint32_t* cu_seq_len_k_end,
                                           uint32_t* seq_k_start,
                                           uint32_t* seq_k_end):
            current_q_block_idx(sm_idx),
            num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
            num_q_tokens(num_q_tokens),
            num_kv_tokens(num_kv_tokens),
            cu_seq_len_k_start(cu_seq_len_k_start),
            cu_seq_len_k_end(cu_seq_len_k_end),
            seq_k_start(seq_k_start),
            seq_k_end(seq_k_end) {}

    // Contiguous KV uses absolute token offsets; align the base to 4 for compressed logits
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_token_base, uint32_t& num_kv_splits) {
        if (current_q_block_idx >= num_q_blocks)
            return false;

        q_block_idx = current_q_block_idx;
        current_q_block_idx += kNumSMs;

        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
            const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
            seq_k_start[token_idx] = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
            seq_k_end[token_idx] = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
            start = cute::min(start, seq_k_start[token_idx]);
            end = cute::max(end, seq_k_end[token_idx]);
        }

        kv_token_base = start / 4 * 4;
        num_kv_splits = math::ceil_div(end - kv_token_base, SPLIT_KV);
        return true;
    }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t& q_block_idx) const {
        return q_block_idx * BLOCK_Q;
    }

    CUTLASS_DEVICE static uint32_t get_kv_tma_offset(const uint32_t& kv_token_base, const uint32_t& kv_split_idx) {
        return kv_token_base + kv_split_idx * SPLIT_KV;
    }

    CUTLASS_DEVICE static uint32_t get_logits_row(const uint32_t& q_block_idx, const uint32_t& token_idx) {
        return q_block_idx * BLOCK_Q + token_idx;
    }

    CUTLASS_DEVICE static uint32_t get_logits_col(const uint32_t& kv_token_base,
                                                  const uint32_t& kv_split_idx,
                                                  const uint32_t& math_thread_idx) {
        return kv_token_base + kv_split_idx * SPLIT_KV + math_thread_idx;
    }
};

} // namespace deep_gemm::sched
