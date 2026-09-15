#pragma once

#include <cutlass/bfloat16.h>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm {

template <typename dtype_t, bool kKGrouped, bool kPsum>
__global__ void sm120_clear_padding(dtype_t* d, const int* layout,
                                    uint32_t m, uint32_t n, uint32_t groups,
                                    uint32_t alignment, uint64_t stride_m) {
    cudaGridDependencySynchronize();
    if constexpr (kKGrouped) {
        const uint32_t group = blockIdx.x;
        const uint32_t start = kPsum and group > 0 ? math::align(static_cast<uint32_t>(layout[group - 1]), alignment) : 0;
        if (static_cast<uint32_t>(layout[group]) != start)
            return;
        for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(m) * n; idx += blockDim.x)
            d[static_cast<uint64_t>(group) * m * n + idx] = dtype_t(0.0f);
    } else if constexpr (kPsum) {
        const uint32_t group = blockIdx.x;
        const uint32_t end = layout[group];
        if (end >= m)
            return;
        const uint32_t padded_end = min(math::align(end, alignment), m);
        for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(padded_end - end) * n; idx += blockDim.x)
            d[(end + idx / n) * stride_m + idx % n] = dtype_t(0.0f);
    } else {
        const uint32_t row = blockIdx.x;
        if (layout[row] >= 0)
            return;
        for (uint32_t col = threadIdx.x; col < n; col += blockDim.x)
            d[static_cast<uint64_t>(row) * stride_m + col] = dtype_t(0.0f);
    }
}

template <bool kMasked, bool kPsum>
__global__ void sm120_copy_grouped_output(const cutlass::bfloat16_t* src, cutlass::bfloat16_t* d,
                                         const int* layout, uint32_t m, uint32_t n, uint32_t alignment,
                                         uint64_t src_stride_m, uint64_t src_stride_n, uint64_t src_stride_group,
                                         uint64_t dst_stride_m, uint64_t dst_stride_n, uint64_t dst_stride_group) {
    cudaGridDependencySynchronize();
    const uint32_t group = kMasked or kPsum ? blockIdx.x : 0;
    uint32_t begin = 0, end = m;
    if constexpr (kMasked) {
        end = min(static_cast<uint32_t>(max(layout[group], 0)), m);
    } else if constexpr (kPsum) {
        begin = group > 0 ? math::align(static_cast<uint32_t>(max(layout[group - 1], 0)), alignment) : 0;
        end = min(static_cast<uint32_t>(max(layout[group], 0)), m);
    } else {
        begin = blockIdx.x;
        if (begin >= m or layout[begin] < 0)
            return;
        end = begin + 1;
    }
    if (begin >= end)
        return;
    const uint64_t src_group_offset = kMasked ? static_cast<uint64_t>(group) * src_stride_group : 0;
    const uint64_t dst_group_offset = kMasked ? static_cast<uint64_t>(group) * dst_stride_group : 0;
    for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(end - begin) * n; idx += blockDim.x) {
        const uint64_t row = begin + idx / n;
        const uint64_t col = idx % n;
        d[dst_group_offset + row * dst_stride_m + col * dst_stride_n] =
            src[src_group_offset + row * src_stride_m + col * src_stride_n];
    }
}

} // namespace deep_gemm
