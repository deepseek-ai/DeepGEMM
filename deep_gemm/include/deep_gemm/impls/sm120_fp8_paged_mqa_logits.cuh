#pragma once

#include <cuda_fp8.h>

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/compile.cuh>

namespace deep_gemm {

template <uint32_t kHeadDim>
CUTLASS_DEVICE float sm120_fp8_dot_scaled(
    const __nv_fp8_e4m3* q,
    const __nv_fp8_e4m3* kv,
    const float kv_scale) {
    float score = 0.0f;
    #pragma unroll
    for (uint32_t dim_idx = 0; dim_idx < kHeadDim; dim_idx += 4) {
        const auto q_values = static_cast<float4>(
            *reinterpret_cast<const __nv_fp8x4_e4m3*>(q + dim_idx));
        const auto kv_values = static_cast<float4>(
            *reinterpret_cast<const __nv_fp8x4_e4m3*>(kv + dim_idx));
        score = fmaf(q_values.x, kv_values.x * kv_scale, score);
        score = fmaf(q_values.y, kv_values.y * kv_scale, score);
        score = fmaf(q_values.z, kv_values.z * kv_scale, score);
        score = fmaf(q_values.w, kv_values.w * kv_scale, score);
    }
    return score;
}

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D, uint32_t kTokensPerBlock,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kTokensPerBlock, 1)
void sm120_fp8_paged_mqa_logits_reference(
    const uint32_t batch_size,
    const uint32_t next_n,
    const uint32_t logits_stride,
    const uint32_t block_table_stride,
    const uint64_t q_batch_stride,
    const uint64_t q_next_stride,
    const uint64_t q_head_stride,
    const uint64_t kv_block_stride,
    const uint64_t kv_token_stride,
    const uint64_t kv_scale_block_stride,
    const uint64_t weights_row_stride,
    const __nv_fp8_e4m3* q,
    const __nv_fp8_e4m3* kv_cache,
    const float* kv_cache_scales,
    const float* weights,
    const uint32_t* context_lens,
    logits_dtype_t* logits,
    const uint32_t* block_table) {
    const auto row = blockIdx.x;
    const auto token_idx = blockIdx.y * kTokensPerBlock + threadIdx.x;
    if (row >= batch_size * next_n or token_idx >= logits_stride)
        return;

    const auto batch_idx = row / next_n;
    const auto next_idx = row - batch_idx * next_n;
    const auto context_lens_idx = kIsContextLens2D ? row : batch_idx;
    const auto context_len = context_lens[context_lens_idx];
    const auto output_offset = row * static_cast<uint64_t>(logits_stride) + token_idx;

    if (token_idx >= context_len) {
        logits[output_offset] = static_cast<logits_dtype_t>(-__uint_as_float(0x7f800000u));
        return;
    }

    const auto logical_block_idx = token_idx / BLOCK_KV;
    const auto token_in_block = token_idx - logical_block_idx * BLOCK_KV;
    const auto kv_block_idx = block_table[batch_idx * static_cast<uint64_t>(block_table_stride) + logical_block_idx];
    const auto q_base = q + batch_idx * q_batch_stride + next_idx * q_next_stride;
    const auto kv_base = kv_cache + kv_block_idx * kv_block_stride + token_in_block * kv_token_stride;
    const auto kv_scale = kv_cache_scales[kv_block_idx * kv_scale_block_stride + token_in_block];
    const auto weight_base = weights + row * weights_row_stride;

    float total = 0.0f;
    #pragma unroll
    for (uint32_t head_idx = 0; head_idx < kNumHeads; ++head_idx) {
        const auto q_head = q_base + head_idx * q_head_stride;
        const auto score = sm120_fp8_dot_scaled<kHeadDim>(q_head, kv_base, kv_scale);
        total += fmaxf(score, 0.0f) * weight_base[head_idx];
    }

    logits[output_offset] = static_cast<logits_dtype_t>(total);
}

} // namespace deep_gemm
