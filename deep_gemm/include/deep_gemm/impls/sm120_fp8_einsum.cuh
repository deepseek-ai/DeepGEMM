#pragma once

#include <cuda_fp8.h>

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/compile.cuh>

namespace deep_gemm {

template <typename output_t, uint32_t kOutputTileD>
CUTLASS_GLOBAL void sm120_fp8_bhr_hdr_bhd_reference(
    const uint32_t shape_b,
    const uint32_t shape_h,
    const uint32_t shape_d,
    const uint32_t shape_r,
    const uint64_t a_stride_b,
    const uint64_t a_stride_h,
    const uint64_t a_stride_r,
    const uint64_t sfa_stride_b,
    const uint64_t sfa_stride_h,
    const uint64_t sfa_stride_r,
    const uint64_t b_stride_h,
    const uint64_t b_stride_d,
    const uint64_t b_stride_r,
    const uint64_t sfb_stride_h,
    const uint64_t sfb_stride_d,
    const uint64_t sfb_stride_r,
    const uint64_t d_stride_b,
    const uint64_t d_stride_h,
    const uint64_t d_stride_d,
    const __nv_fp8_e4m3* a,
    const float* sfa,
    const __nv_fp8_e4m3* b,
    const float* sfb,
    output_t* d) {
    const uint32_t num_d_tiles = (shape_d + kOutputTileD - 1) / kOutputTileD;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t d_tile_idx = tile_idx % num_d_tiles;
    const uint32_t h_idx = (tile_idx / num_d_tiles) % shape_h;
    const uint32_t b_idx = tile_idx / (num_d_tiles * shape_h);
    const uint32_t d_idx = d_tile_idx * kOutputTileD + threadIdx.x;

    if (b_idx >= shape_b or d_idx >= shape_d)
        return;

    float accum = 0.0f;
    for (uint32_t r_idx = 0; r_idx < shape_r; ++r_idx) {
        const uint32_t r_scale_idx = r_idx / 128;
        const auto a_offset = b_idx * a_stride_b + h_idx * a_stride_h + r_idx * a_stride_r;
        const auto b_offset = h_idx * b_stride_h + d_idx * b_stride_d + r_idx * b_stride_r;
        const auto sfa_offset = b_idx * sfa_stride_b + h_idx * sfa_stride_h + r_scale_idx * sfa_stride_r;
        const auto sfb_offset = h_idx * sfb_stride_h + (d_idx / 128) * sfb_stride_d + r_scale_idx * sfb_stride_r;

        const float a_value = static_cast<float>(a[a_offset]) * sfa[sfa_offset];
        const float b_value = static_cast<float>(b[b_offset]) * sfb[sfb_offset];
        accum = fmaf(a_value, b_value, accum);
    }

    const auto d_offset = b_idx * d_stride_b + h_idx * d_stride_h + d_idx * d_stride_d;
    d[d_offset] = static_cast<output_t>(accum);
}

} // namespace deep_gemm
