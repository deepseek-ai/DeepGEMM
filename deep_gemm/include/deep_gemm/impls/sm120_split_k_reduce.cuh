#pragma once

#include <cutlass/bfloat16.h>
#include <deep_gemm/epilogue/transform.cuh>

namespace deep_gemm {

template <typename cd_dtype_t, uint32_t kSplitKFactor,
          typename epilogue_type_t = epilogue::transform::EpilogueIdentity>
__global__ void sm120_split_k_reduce_impl(
    cd_dtype_t* gmem_d,
    const float* __restrict__ workspace,
    uint32_t shape_m, uint32_t shape_n,
    int stride_cd_m, int stride_cd_n,
    const cd_dtype_t* gmem_c, int stride_c_m, int stride_c_n,
    bool with_alpha, float alpha) {
    cudaGridDependencySynchronize();

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = shape_m * shape_n;
    if (idx >= total)
        return;

    const uint32_t row = idx / shape_n;
    const uint32_t col = epilogue_type_t::template apply_index_n<1>(idx % shape_n);
    const uint32_t ws_stride = shape_m * shape_n;

    float sum = workspace[idx];
    #pragma unroll
    for (uint32_t s = 1; s < kSplitKFactor; ++s)
        sum += workspace[s * ws_stride + idx];

    if (with_alpha)
        sum *= alpha;
    if (gmem_c != nullptr)
        sum += static_cast<float>(gmem_c[static_cast<int64_t>(row) * stride_c_m + static_cast<int64_t>(col) * stride_c_n]);
    gmem_d[static_cast<int64_t>(row) * stride_cd_m + static_cast<int64_t>(col) * stride_cd_n] = cd_dtype_t(sum);
}

} // namespace deep_gemm
