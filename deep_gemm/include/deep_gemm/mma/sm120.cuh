#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cuda/std/cstdint>
#include <cute/arch/mma_sm120.hpp>

namespace deep_gemm::mma::sm120 {

// Block-scaled FP8 MMA: m16n8k32
// Uses CUTLASS SM120 MMA atom directly (cute/arch/mma_sm120.hpp).
// SF bytes must be pre-extracted before calling (CUTLASS approach: byte_id=0 always).
static constexpr int FP8_MMA_M = 16, FP8_MMA_N = 8, FP8_MMA_K = 32;
static constexpr int FP8_MMA_ACCUM = 4;

using FP8_MMA_Op = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t, 32>;

__device__ __forceinline__ uint8_t extract_sf_byte(uint32_t packed, uint32_t byte_idx) {
    return static_cast<uint8_t>((packed >> (byte_idx * 8)) & 0xFF);
}

// FP4 scale_vec::2X: extract 2 consecutive SF bytes into uint16_t
__device__ __forceinline__ uint16_t extract_sf_pair(uint32_t packed, uint32_t first_byte_idx) {
    return static_cast<uint16_t>((packed >> (first_byte_idx * 8)) & 0xFFFF);
}

__device__ __forceinline__ void fp8_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint8_t sfa, uint8_t sfb
) {
    FP8_MMA_Op::fma(d[0], d[1], d[2], d[3],
                     a[0], a[1], a[2], a[3],
                     b[0], b[1],
                     d[0], d[1], d[2], d[3],
                     sfa, sfb);
}

// FP4 Block-Scaled MMA: m16n8k64
static constexpr int FP4_MMA_M = 16, FP4_MMA_N = 8, FP4_MMA_K = 64;
static constexpr int FP4_MMA_ACCUM = 4;

using FP4_MMA_Op = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t, 32>;

__device__ __forceinline__ void fp4_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint16_t sfa, uint16_t sfb
) {
    FP4_MMA_Op::fma(d[0], d[1], d[2], d[3],
                     a[0], a[1], a[2], a[3],
                     b[0], b[1],
                     d[0], d[1], d[2], d[3],
                     sfa, sfb);
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
