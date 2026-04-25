#pragma once

#include <cuda_fp8.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/tensor.hpp>

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

template <uint32_t BLOCK_KV, bool kIsContextLens2D, uint32_t kTokenGroups,
          bool kCacheQ,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(128, 1)
void sm120_fp8_paged_mqa_logits_mma_tiled(
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
    using Element = cute::float_e4m3_t;
    using Accumulator = float;

    constexpr uint32_t kNumHeads = 64;
    constexpr uint32_t kHeadDim = 128;
    constexpr uint32_t kHeadsPerWarp = 16;
    constexpr uint32_t kTokensPerTile = 8;
    constexpr uint32_t kTokensPerCta = kTokensPerTile * kTokenGroups;
    constexpr uint32_t kMmaK = 32;
    constexpr uint32_t kHeadTiles = kNumHeads / kHeadsPerWarp;

    extern __shared__ float shared_scores[];

    const auto row = blockIdx.x;
    const auto token_tile_start = blockIdx.y * kTokensPerCta;
    if (row >= batch_size * next_n)
        return;

    const auto batch_idx = row / next_n;
    const auto next_idx = row - batch_idx * next_n;
    const auto context_lens_idx = kIsContextLens2D ? row : batch_idx;
    const auto context_len = context_lens[context_lens_idx];

    if (token_tile_start >= context_len) {
        if (threadIdx.x < kTokensPerCta) {
            const auto token_idx = token_tile_start + threadIdx.x;
            if (token_idx < logits_stride) {
                const auto output_offset = row * static_cast<uint64_t>(logits_stride) + token_idx;
                logits[output_offset] = static_cast<logits_dtype_t>(-__uint_as_float(0x7f800000u));
            }
        }
        return;
    }

    const auto lane_idx = threadIdx.x % 32;
    const auto warp_idx = threadIdx.x / 32;
    const auto q_base = q + batch_idx * q_batch_stride + next_idx * q_next_stride;

    auto tiled_mma = cute::make_tiled_mma(
        cute::SM120_16x8x32_TN<Element, Element, Accumulator>{});
    auto thread_mma = tiled_mma.get_slice(lane_idx);

    if constexpr (kCacheQ and kTokenGroups == 4) {
        auto score_tile0 = cute::make_tensor(
            cute::make_smem_ptr(shared_scores + warp_idx * kHeadsPerWarp * kTokensPerCta),
            cute::make_shape(cute::Int<kHeadsPerWarp>{}, cute::Int<kTokensPerTile>{}),
            cute::make_stride(cute::Int<kTokensPerCta>{}, cute::Int<1>{}));
        auto score_tile1 = cute::make_tensor(
            cute::make_smem_ptr(shared_scores + warp_idx * kHeadsPerWarp * kTokensPerCta + kTokensPerTile),
            cute::make_shape(cute::Int<kHeadsPerWarp>{}, cute::Int<kTokensPerTile>{}),
            cute::make_stride(cute::Int<kTokensPerCta>{}, cute::Int<1>{}));
        auto score_tile2 = cute::make_tensor(
            cute::make_smem_ptr(shared_scores + warp_idx * kHeadsPerWarp * kTokensPerCta + 2 * kTokensPerTile),
            cute::make_shape(cute::Int<kHeadsPerWarp>{}, cute::Int<kTokensPerTile>{}),
            cute::make_stride(cute::Int<kTokensPerCta>{}, cute::Int<1>{}));
        auto score_tile3 = cute::make_tensor(
            cute::make_smem_ptr(shared_scores + warp_idx * kHeadsPerWarp * kTokensPerCta + 3 * kTokensPerTile),
            cute::make_shape(cute::Int<kHeadsPerWarp>{}, cute::Int<kTokensPerTile>{}),
            cute::make_stride(cute::Int<kTokensPerCta>{}, cute::Int<1>{}));
        auto tCgC0 = thread_mma.partition_C(score_tile0);
        auto tCgC1 = thread_mma.partition_C(score_tile1);
        auto tCgC2 = thread_mma.partition_C(score_tile2);
        auto tCgC3 = thread_mma.partition_C(score_tile3);
        auto tCrC0 = thread_mma.make_fragment_C(tCgC0);
        auto tCrC1 = thread_mma.make_fragment_C(tCgC1);
        auto tCrC2 = thread_mma.make_fragment_C(tCgC2);
        auto tCrC3 = thread_mma.make_fragment_C(tCgC3);
        cute::clear(tCrC0);
        cute::clear(tCrC1);
        cute::clear(tCrC2);
        cute::clear(tCrC3);

        #pragma unroll
        for (uint32_t dim_idx = 0; dim_idx < kHeadDim; dim_idx += kMmaK) {
            const auto q_tile = reinterpret_cast<const uint8_t*>(
                q_base + (warp_idx * kHeadsPerWarp) * q_head_stride + dim_idx);
            auto q_tensor = cute::make_tensor(
                cute::make_gmem_ptr(q_tile),
                cute::make_shape(cute::Int<kHeadsPerWarp>{}, cute::Int<kMmaK>{}),
                cute::make_stride(q_head_stride, cute::Int<1>{}));
            auto tAgA = thread_mma.partition_A(q_tensor);
            auto tArA = thread_mma.partition_fragment_A(q_tensor);
            cute::copy(tAgA, tArA);

            auto accumulate_group = [&](auto token_group, auto& tCrC) {
                constexpr uint32_t kGroup = decltype(token_group)::value;
                const auto group_token_start = token_tile_start + kGroup * kTokensPerTile;
                if (group_token_start >= context_len or group_token_start >= logits_stride)
                    return;

                const auto logical_block_idx = group_token_start / BLOCK_KV;
                const auto token_in_block = group_token_start - logical_block_idx * BLOCK_KV;
                const auto kv_block_idx = block_table[
                    batch_idx * static_cast<uint64_t>(block_table_stride) + logical_block_idx];
                const auto kv_base = kv_cache + kv_block_idx * kv_block_stride + token_in_block * kv_token_stride;
                const auto kv_tile = reinterpret_cast<const uint8_t*>(kv_base + dim_idx);
                auto kv_tensor = cute::make_tensor(
                    cute::make_gmem_ptr(kv_tile),
                    cute::make_shape(cute::Int<kTokensPerTile>{}, cute::Int<kMmaK>{}),
                    cute::make_stride(kv_token_stride, cute::Int<1>{}));
                auto tBgB = thread_mma.partition_B(kv_tensor);
                auto tBrB = thread_mma.partition_fragment_B(kv_tensor);
                cute::copy(tBgB, tBrB);
                cute::gemm(tiled_mma, tArA, tBrB, tCrC);
            };
            accumulate_group(cute::Int<0>{}, tCrC0);
            accumulate_group(cute::Int<1>{}, tCrC1);
            accumulate_group(cute::Int<2>{}, tCrC2);
            accumulate_group(cute::Int<3>{}, tCrC3);
        }

        cute::copy(tCrC0, tCgC0);
        cute::copy(tCrC1, tCgC1);
        cute::copy(tCrC2, tCgC2);
        cute::copy(tCrC3, tCgC3);
    } else {
        #pragma unroll
        for (uint32_t token_group = 0; token_group < kTokenGroups; ++token_group) {
            const auto group_token_start = token_tile_start + token_group * kTokensPerTile;
            if (group_token_start >= context_len or group_token_start >= logits_stride)
                continue;

            const auto logical_block_idx = group_token_start / BLOCK_KV;
            const auto token_in_block = group_token_start - logical_block_idx * BLOCK_KV;
            const auto kv_block_idx = block_table[
                batch_idx * static_cast<uint64_t>(block_table_stride) + logical_block_idx];
            const auto kv_base = kv_cache + kv_block_idx * kv_block_stride + token_in_block * kv_token_stride;

            auto score_tile = cute::make_tensor(
                cute::make_smem_ptr(
                    shared_scores +
                    warp_idx * kHeadsPerWarp * kTokensPerCta +
                    token_group * kTokensPerTile),
                cute::make_shape(cute::Int<kHeadsPerWarp>{}, cute::Int<kTokensPerTile>{}),
                cute::make_stride(cute::Int<kTokensPerCta>{}, cute::Int<1>{}));
            auto tCgC = thread_mma.partition_C(score_tile);
            auto tCrC = thread_mma.make_fragment_C(tCgC);
            cute::clear(tCrC);

            #pragma unroll
            for (uint32_t dim_idx = 0; dim_idx < kHeadDim; dim_idx += kMmaK) {
                const auto q_tile = reinterpret_cast<const uint8_t*>(
                    q_base + (warp_idx * kHeadsPerWarp) * q_head_stride + dim_idx);
                const auto kv_tile = reinterpret_cast<const uint8_t*>(kv_base + dim_idx);

                auto q_tensor = cute::make_tensor(
                    cute::make_gmem_ptr(q_tile),
                    cute::make_shape(cute::Int<kHeadsPerWarp>{}, cute::Int<kMmaK>{}),
                    cute::make_stride(q_head_stride, cute::Int<1>{}));
                auto kv_tensor = cute::make_tensor(
                    cute::make_gmem_ptr(kv_tile),
                    cute::make_shape(cute::Int<kTokensPerTile>{}, cute::Int<kMmaK>{}),
                    cute::make_stride(kv_token_stride, cute::Int<1>{}));

                auto tAgA = thread_mma.partition_A(q_tensor);
                auto tBgB = thread_mma.partition_B(kv_tensor);
                auto tArA = thread_mma.partition_fragment_A(q_tensor);
                auto tBrB = thread_mma.partition_fragment_B(kv_tensor);

                cute::copy(tAgA, tArA);
                cute::copy(tBgB, tBrB);
                cute::gemm(tiled_mma, tArA, tBrB, tCrC);
            }

            cute::copy(tCrC, tCgC);
        }
    }
    __syncthreads();

    const auto token_idx = token_tile_start + threadIdx.x;
    if (threadIdx.x >= kTokensPerCta or token_idx >= logits_stride)
        return;

    const auto output_offset = row * static_cast<uint64_t>(logits_stride) + token_idx;
    if (token_idx >= context_len) {
        logits[output_offset] = static_cast<logits_dtype_t>(-__uint_as_float(0x7f800000u));
        return;
    }

    const auto logical_block_idx = token_idx / BLOCK_KV;
    const auto token_in_block = token_idx - logical_block_idx * BLOCK_KV;
    const auto kv_block_idx = block_table[
        batch_idx * static_cast<uint64_t>(block_table_stride) + logical_block_idx];
    const auto kv_scale = kv_cache_scales[kv_block_idx * kv_scale_block_stride + token_in_block];
    const auto weight_base = weights + row * weights_row_stride;

    float total = 0.0f;
    #pragma unroll
    for (uint32_t head_tile = 0; head_tile < kHeadTiles; ++head_tile) {
        #pragma unroll
        for (uint32_t head = 0; head < kHeadsPerWarp; ++head) {
            const auto head_idx = head_tile * kHeadsPerWarp + head;
            const auto score = shared_scores[
                head_tile * kHeadsPerWarp * kTokensPerCta +
                head * kTokensPerCta + threadIdx.x] * kv_scale;
            total += fmaxf(score, 0.0f) * weight_base[head_idx];
        }
    }

    logits[output_offset] = static_cast<logits_dtype_t>(total);
}

} // namespace deep_gemm
