#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>

#include <deep_gemm/common/reduction.cuh>

namespace deep_gemm {

template <uint32_t kGroupSize, bool kUsePDL>
__launch_bounds__(1024, 2)
__global__ void sm90_mega_moe_pre_dispatch_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t*       __restrict__ topk_idx,
    const float*         __restrict__ topk_weights,
    __nv_fp8_e4m3*       __restrict__ buf_x,
    float*               __restrict__ buf_x_sf,
    int64_t*             __restrict__ buf_topk_idx,
    float*               __restrict__ buf_topk_weights,
    const uint32_t num_tokens,
    const uint32_t padded_max,
    const uint32_t hidden,
    const uint32_t num_groups,
    const uint32_t top_k,
    const float routed_scaling_factor) {
    static_assert(kGroupSize == 128, "SM90 mega-moe pre-dispatch requires per-128 SF");
    constexpr uint32_t kVecElems = 8;  // 16-byte BF16 load per thread
    static_assert(kGroupSize % kVecElems == 0, "kGroupSize must be a multiple of 8");
    constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;

    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if constexpr (kUsePDL) {
        cudaGridDependencySynchronize();
    }

    if (bid < num_tokens) {
        const uint32_t token_id = bid;
        const auto* token_in = x + static_cast<uint64_t>(token_id) * hidden;

        uint4 in_bits = reinterpret_cast<const uint4*>(token_in)[tid];
        const auto* bf16_pairs = reinterpret_cast<const __nv_bfloat162*>(&in_bits);

        float vals[kVecElems];
        float local_max = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < kVecElems / 2; ++i) {
            float2 fp = __bfloat1622float2(bf16_pairs[i]);
            vals[2 * i + 0] = fp.x;
            vals[2 * i + 1] = fp.y;
            local_max = fmaxf(local_max, fmaxf(fabsf(fp.x), fabsf(fp.y)));
        }

        local_max = warp_reduce<kThreadsPerGroup, false>(
            local_max, ReduceMax<float>{});

        const float absmax = fmaxf(local_max, 1e-10f);
        const float raw_scale = absmax / 448.0f;
        const float inv_scale = 1.0f / raw_scale;

        uint64_t packed = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kVecElems / 2; ++i) {
            const __nv_fp8x2_storage_t fp8x2 = __nv_cvt_float2_to_fp8x2(
                make_float2(vals[2 * i + 0] * inv_scale,
                            vals[2 * i + 1] * inv_scale),
                __NV_SATFINITE, __NV_E4M3);
            packed |= static_cast<uint64_t>(fp8x2) << (16u * i);
        }
        auto* row_out = reinterpret_cast<uint64_t*>(buf_x) +
                        static_cast<uint64_t>(token_id) * (hidden / 8u);
        row_out[tid] = packed;

        const uint32_t group_id = tid / kThreadsPerGroup;
        const uint32_t within_group_id = tid % kThreadsPerGroup;
        if (within_group_id == 0u && group_id < num_groups) {
            const uint32_t off = token_id * num_groups + group_id;
            buf_x_sf[off] = raw_scale;
        }

        if (tid < top_k) {
            const uint32_t off = token_id * top_k + tid;
            buf_topk_idx[off] = static_cast<int64_t>(topk_idx[off]);
            buf_topk_weights[off] = topk_weights[off] * routed_scaling_factor;
        }
    } else {
        const uint32_t copy_bid = bid - num_tokens;
        const uint32_t pad_base = num_tokens * top_k;
        const uint32_t slot = pad_base + copy_bid * blockDim.x + tid;
        const uint32_t total = padded_max * top_k;
        if (slot < total) {
            buf_topk_idx[slot] = static_cast<int64_t>(-1);
            buf_topk_weights[slot] = 0.0f;
        }
    }

    if constexpr (kUsePDL) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

}  // namespace deep_gemm
