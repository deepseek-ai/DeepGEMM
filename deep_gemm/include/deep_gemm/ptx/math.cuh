#pragma once

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include <deep_gemm/common/exception.cuh>

#if defined(DG_IN_CUDA_COMPILATION)
namespace deep_gemm::ptx {

// BF16 shares FP32's exponent: place each 16-bit value in the upper half of an FP32 word.
CUTLASS_DEVICE float2 cvt_bf16x2_to_fp32x2(const nv_bfloat162& value) {
    const auto packed = reinterpret_cast<const uint32_t&>(value);
    float2 unpacked;
    asm("shl.b32 %0, %2, 16;\n"
        "and.b32 %1, %2, 0xffff0000;\n"
        : "=&f"(unpacked.x), "=f"(unpacked.y) : "r"(packed));
    return unpacked;
}

CUTLASS_DEVICE float cvt_rn_tf32_f32(const float& value) {
    uint32_t result;
    asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return __uint_as_float(result);
}

CUTLASS_DEVICE uint32_t cvt_rs_bf16x2_f32(const float& lower, const float& upper,
                                          const uint32_t& random_bits) {
    uint32_t packed;
    asm volatile("cvt.rs.bf16x2.f32 %0, %1, %2, %3;\n"
                 : "=r"(packed) : "f"(upper), "f"(lower), "r"(random_bits));
    return packed;
}

CUTLASS_DEVICE uint32_t cvt_rs_e4m3x4_f32(const float& v0, const float& v1,
                                          const float& v2, const float& v3,
                                          const uint32_t& random_bits) {
    uint32_t packed;
    asm volatile("cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
                 : "=r"(packed) : "f"(v3), "f"(v2), "f"(v1), "f"(v0), "r"(random_bits));
    return packed;
}

// Symmetric clamp for a nonnegative bound in one min instruction.
CUTLASS_DEVICE float clamp_fp32(const float& value, const float& bound) {
    float result;
    asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "f"(bound));
    return result;
}

CUTLASS_DEVICE nv_bfloat162 clamp_bf16x2_no_nan(const nv_bfloat162& value, const nv_bfloat162& bound) {
    uint32_t packed;
    asm("min.xorsign.abs.bf16x2 %0, %1, %2;"
        : "=r"(packed)
        : "r"(reinterpret_cast<const uint32_t&>(value)),
          "r"(reinterpret_cast<const uint32_t&>(bound)));
    return reinterpret_cast<const nv_bfloat162&>(packed);
}

// Symmetric clamp for a nonnegative bound: retain the input sign and propagate NaNs.
CUTLASS_DEVICE nv_bfloat162 clamp_bf16x2(const nv_bfloat162& value, const nv_bfloat162& bound) {
    uint32_t packed;
    asm("min.NaN.xorsign.abs.bf16x2 %0, %1, %2;"
        : "=r"(packed)
        : "r"(reinterpret_cast<const uint32_t&>(value)),
          "r"(reinterpret_cast<const uint32_t&>(bound)));
    return reinterpret_cast<const nv_bfloat162&>(packed);
}

CUTLASS_DEVICE float reduce_max_sync(const float& value) {
    float ret;
    asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(ret) : "f"(value));
    return ret;
}

CUTLASS_DEVICE nv_bfloat162 cvt_relu_bf16x2_f32(const float2& v) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    uint32_t packed;
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(v.y), "f"(v.x));
    return *reinterpret_cast<const nv_bfloat162*>(&packed);
#else
    return __floats2bfloat162_rn(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f));
#endif
}

CUTLASS_DEVICE void accumulate(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const auto [x, y] = cvt_bf16x2_to_fp32x2(b);
    a.x += x, a.y += y;
#endif
}

CUTLASS_DEVICE void accumulate_square(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Accumulate BF16 squares directly into FP32 without separate conversion instructions.
    asm("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("fma.rn.f32.bf16 %0, %1, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const float2 b_f32 = cvt_bf16x2_to_fp32x2(b);
    a.x += b_f32.x * b_f32.x;
    a.y += b_f32.y * b_f32.y;
#endif
}

} // namespace deep_gemm::ptx
#endif
