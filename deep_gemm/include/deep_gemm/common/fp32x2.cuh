#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

namespace deep_gemm::math {

// Blackwell packed FP32 instructions have no Hopper equivalent. Keep the
// component-wise rounding boundaries identical when lowering to scalar SM90 ops.
CUTLASS_DEVICE float2 mul_fp32x2_rn(const float2 a, const float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return __fmul2_rn(a, b);
#else
    return {__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)};
#endif
}

CUTLASS_DEVICE float2 fma_fp32x2_rn(const float2 a, const float2 b, const float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return __ffma2_rn(a, b, c);
#else
    return {__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y)};
#endif
}

} // namespace deep_gemm::math
