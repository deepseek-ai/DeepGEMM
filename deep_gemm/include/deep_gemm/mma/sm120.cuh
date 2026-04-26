#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cuda/std/cstdint>

namespace deep_gemm::mma::sm120 {

// Block-scaled FP8 MMA: m16n8k32
// byte_id selects which UE8M0 byte within the packed int32 SF operand.
// For gran_k=128 with BLOCK_K=128: byte_id=0 always.
// For gran_k=32 with MMA_K=32: byte_id = k_step % 4.
static constexpr int FP8_MMA_M = 16, FP8_MMA_N = 8, FP8_MMA_K = 32;
static constexpr int FP8_MMA_ACCUM = 4;

#define SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, byte_a, byte_b)        \
    asm volatile(                                                              \
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."          \
        "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "                           \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "                              \
        "{%0,%1,%2,%3}, %10, {" #byte_a ", 0}, %11, {" #byte_b ", 0};\n"       \
        : "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3])               \
        : "r"((a)[0]), "r"((a)[1]), "r"((a)[2]), "r"((a)[3]),                  \
          "r"((b)[0]), "r"((b)[1]),                                            \
          "r"(sfa), "r"(sfb)                                                   \
    )

__device__ __forceinline__ void fp8_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint32_t sfa, uint32_t sfb, uint32_t byte_a, uint32_t byte_b
) {
    // byte_a and byte_b must be compile-time constants (PTX immediates).
    // With #pragma unroll on the K-step loop, the compiler constant-folds them.
    // The switch is eliminated after unrolling.
    switch (byte_a * 4 + byte_b) {
        case  0: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 0, 0); break;
        case  1: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 0, 1); break;
        case  2: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 0, 2); break;
        case  3: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 0, 3); break;
        case  4: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 1, 0); break;
        case  5: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 1, 1); break;
        case  6: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 1, 2); break;
        case  7: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 1, 3); break;
        case  8: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 2, 0); break;
        case  9: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 2, 1); break;
        case 10: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 2, 2); break;
        case 11: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 2, 3); break;
        case 12: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 3, 0); break;
        case 13: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 3, 1); break;
        case 14: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 3, 2); break;
        case 15: SM120_FP8_MMA_BLOCK_SCALED(d, a, b, sfa, sfb, 3, 3); break;
    }
}

#undef SM120_FP8_MMA_BLOCK_SCALED

// FP4 Block-Scaled MMA: m16n8k64
static constexpr int FP4_MMA_M = 16, FP4_MMA_N = 8, FP4_MMA_K = 64;
static constexpr int FP4_MMA_ACCUM = 4;

__device__ __forceinline__ void fp4_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint32_t sfa, uint32_t sfb
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(sfa), "r"(sfb)
    );
}

// BF16 MMA: m16n8k16
static constexpr int BF16_MMA_M = 16, BF16_MMA_N = 8, BF16_MMA_K = 16;
static constexpr int BF16_MMA_ACCUM = 4;

__device__ __forceinline__ void bf16_mma(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

// TF32 MMA: m16n8k8
static constexpr int TF32_MMA_M = 16, TF32_MMA_N = 8, TF32_MMA_K = 8;
static constexpr int TF32_MMA_ACCUM = 4;

__device__ __forceinline__ void tf32_mma(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

} // namespace deep_gemm::mma::sm120

#endif
