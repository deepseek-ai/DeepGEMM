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

} // namespace deep_gemm
