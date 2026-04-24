#pragma once

#include <deep_gemm/common/exception.cuh>
#include <cuda_runtime.h> // Required for __float_as_uint

namespace deep_gemm::mma::sm120 {

CUTLASS_DEVICE void mma_m16n8k8_f32_tf32accum(
    float& d0, float& d1, float& d2, float& d3,
    float a0, float a1, float a2, float a3,
    float b0, float b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        // tf32 multiplicands expect .b32 registers ("r")
        : "r"(__float_as_uint(a0)), "r"(__float_as_uint(a1)), "r"(__float_as_uint(a2)), "r"(__float_as_uint(a3)),
          "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1)),
        // f32 accumulators expect .f32 registers ("f")
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <int M, int N, int K>
struct TF32MMASync {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 8;
    static constexpr int kNumAccum = M * N / 32;

    static_assert(M == 16 and N == 8 and K == 8, "SM120 TF32 mma.sync atom is 16x8x8");

    CUTLASS_DEVICE static void fma(
        float& d0, float& d1, float& d2, float& d3,
        float a0, float a1, float a2, float a3,
        float b0, float b1,
        float c0, float c1, float c2, float c3) {
        mma_m16n8k8_f32_tf32accum(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    }
};

template <int N>
struct TF32MMASelector {
    static constexpr auto select_type() {
        static_assert(N == 8 or N == 16 or N == 32, "SM120 TF32 hc_prenorm supports N <= 32");
        return TF32MMASync<16, 8, 8>{};
    }

    using type = decltype(select_type());

    static constexpr int M = 16;
    static constexpr int K = 8;
    static constexpr int kNumAccumPerAtom = 4;
    
    static constexpr int MMA_N_per_atom() { return 8; }
    static constexpr int N_atoms = N / MMA_N_per_atom();
};

} // namespace deep_gemm::mma::sm120
