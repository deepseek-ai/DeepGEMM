#pragma once

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::sm90_moe_math {

CUTLASS_DEVICE float fast_pow2(const int& x) {
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

CUTLASS_DEVICE int fast_log2_ceil(float x) {
    const auto bits = *reinterpret_cast<uint32_t*>(&x);
    const auto exp = bits >> 23;
    const auto man = bits & ((1 << 23) - 1);
    return exp - 127 + (man != 0);
}

template <typename T>
struct ReduceMax {
    CUTLASS_DEVICE T operator()(T a, T b) const { return a > b ? a : b; }
};

} // namespace deep_gemm::sm90_moe_math
