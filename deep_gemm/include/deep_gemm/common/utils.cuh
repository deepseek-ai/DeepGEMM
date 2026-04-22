#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    CUTLASS_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    CUTLASS_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

template <uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}

template <typename T>
__device__ __forceinline__ T shfl_sync(unsigned mask, T var, int srcLane, int width = 32) {

    using shfl_t = std::conditional_t<sizeof(T) == 4, int,
                   std::conditional_t<sizeof(T) == 8, long long, long long>>;

    T result;
    shfl_t* var_ptr = reinterpret_cast<shfl_t*>(&var);
    shfl_t* result_ptr = reinterpret_cast<shfl_t*>(&result);
    *result_ptr = __shfl_sync(mask, *var_ptr, srcLane, width);

    if constexpr (sizeof(T) == 16) {
        *(result_ptr + 1) = __shfl_sync(mask, *(var_ptr + 1), srcLane, width);
    }

    return result;
}

} // namespace deep_gemm::utils
