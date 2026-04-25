#pragma once

#include <cuda_fp8.h>

#include <cutlass/numeric_types.h>

#include <deep_gemm/common/compile.cuh>

namespace deep_gemm {

CUTLASS_DEVICE float sm120_fp8_einsum_dot_scalar(
    const __nv_fp8_e4m3* a,
    const __nv_fp8_e4m3* b,
    const float* sfa,
    const float* sfb,
    const uint32_t shape_r,
    const uint64_t sfa_stride_r,
    const uint64_t sfb_stride_r) {
    float accum = 0.0f;
    for (uint32_t r_idx = 0; r_idx < shape_r; ++r_idx) {
        const uint32_t r_scale_idx = r_idx / 128;
        const float scale = sfa[r_scale_idx * sfa_stride_r] * sfb[r_scale_idx * sfb_stride_r];
        const float a_value = static_cast<float>(a[r_idx]);
        const float b_value = static_cast<float>(b[r_idx]);
        accum = fmaf(a_value, b_value * scale, accum);
    }
    return accum;
}

CUTLASS_DEVICE float sm120_fp8_einsum_dot_fp8x4(
    const __nv_fp8_e4m3* a,
    const __nv_fp8_e4m3* b,
    const float* sfa,
    const float* sfb,
    const uint32_t shape_r,
    const uint64_t sfa_stride_r,
    const uint64_t sfb_stride_r) {
    float accum = 0.0f;
    const uint32_t num_full_scale_blocks = shape_r / 128;
    for (uint32_t scale_idx = 0; scale_idx < num_full_scale_blocks; ++scale_idx) {
        const float scale = sfa[scale_idx * sfa_stride_r] * sfb[scale_idx * sfb_stride_r];
        const uint32_t r_start = scale_idx * 128;
        #pragma unroll
        for (uint32_t r_offset = 0; r_offset < 128; r_offset += 4) {
            const uint32_t r_idx = r_start + r_offset;
            const auto a_values = static_cast<float4>(
                *reinterpret_cast<const __nv_fp8x4_e4m3*>(a + r_idx));
            const auto b_values = static_cast<float4>(
                *reinterpret_cast<const __nv_fp8x4_e4m3*>(b + r_idx));
            accum = fmaf(a_values.x, b_values.x * scale, accum);
            accum = fmaf(a_values.y, b_values.y * scale, accum);
            accum = fmaf(a_values.z, b_values.z * scale, accum);
            accum = fmaf(a_values.w, b_values.w * scale, accum);
        }
    }
    const uint32_t tail_start = num_full_scale_blocks * 128;
    if (tail_start >= shape_r)
        return accum;

    const float tail_scale = sfa[num_full_scale_blocks * sfa_stride_r] *
                             sfb[num_full_scale_blocks * sfb_stride_r];
    for (uint32_t r_idx = tail_start; r_idx < shape_r; r_idx += 4) {
        const auto a_values = static_cast<float4>(
            *reinterpret_cast<const __nv_fp8x4_e4m3*>(a + r_idx));
        const auto b_values = static_cast<float4>(
            *reinterpret_cast<const __nv_fp8x4_e4m3*>(b + r_idx));
        accum = fmaf(a_values.x, b_values.x * tail_scale, accum);
        accum = fmaf(a_values.y, b_values.y * tail_scale, accum);
        accum = fmaf(a_values.z, b_values.z * tail_scale, accum);
        accum = fmaf(a_values.w, b_values.w * tail_scale, accum);
    }
    return accum;
}

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

    const auto a_base = b_idx * a_stride_b + h_idx * a_stride_h;
    const auto b_base = h_idx * b_stride_h + d_idx * b_stride_d;
    const auto sfa_base = b_idx * sfa_stride_b + h_idx * sfa_stride_h;
    const auto sfb_base = h_idx * sfb_stride_h + (d_idx / 128) * sfb_stride_d;
    const bool can_vectorize = (shape_r % 4 == 0) and (a_stride_r == 1) and (b_stride_r == 1);
    const float accum = can_vectorize ?
        sm120_fp8_einsum_dot_fp8x4(a + a_base, b + b_base, sfa + sfa_base, sfb + sfb_base,
                                   shape_r, sfa_stride_r, sfb_stride_r) :
        sm120_fp8_einsum_dot_scalar(a + a_base, b + b_base, sfa + sfa_base, sfb + sfb_base,
                                    shape_r, sfa_stride_r, sfb_stride_r);

    const auto d_offset = b_idx * d_stride_b + h_idx * d_stride_h + d_idx * d_stride_d;
    d[d_offset] = static_cast<output_t>(accum);
}

} // namespace deep_gemm
