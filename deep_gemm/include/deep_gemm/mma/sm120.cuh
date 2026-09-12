#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cuda/std/cstdint>

namespace deep_gemm::mma::sm120 {

// FP8 MMA: m16n8k32
static constexpr int FP8_MMA_M = 16, FP8_MMA_N = 8, FP8_MMA_K = 32;
static constexpr int FP8_MMA_ACCUM = 4;

// Non-block-scaled FP8 MMA — for kernels that apply scaling separately (e.g. MQA logits)
__device__ __forceinline__ void fp8_mma(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

// Block-scaled FP8 MMA: m16n8k32

__device__ __forceinline__ uint8_t extract_sf_byte(uint32_t packed, uint32_t byte_idx) {
    return static_cast<uint8_t>((packed >> (byte_idx * 8)) & 0xFF);
}

// FP4 scale_vec::2X: extract 2 consecutive SF bytes into uint16_t
__device__ __forceinline__ uint16_t extract_sf_pair(uint32_t packed, uint32_t first_byte_idx) {
    return static_cast<uint16_t>((packed >> (first_byte_idx * 8)) & 0xFFFF);
}

__device__ __forceinline__ void fp8_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
    uint8_t sfa, uint8_t sfb
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0), "r"(b1),
          "r"(static_cast<uint32_t>(sfa)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
          "r"(static_cast<uint32_t>(sfb)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0))
    );
}

__device__ __forceinline__ void fp8_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint8_t sfa, uint8_t sfb
) {
    fp8_mma_block_scaled(d, a, b[0], b[1], sfa, sfb);
}

// Mixed FP8_A × FP4_B block-scaled MMA: m16n8k32, scale_vec::1X
// Uses mxf8f6f4 with e4m3 A and e2m1 B. Same K=32 as pure FP8.
// B operand must be unpacked (1 byte/elem) and shifted << 2 before calling.
__device__ __forceinline__ void fp8_fp4_mixed_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint8_t sfa, uint8_t sfb
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(static_cast<uint32_t>(sfa)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
          "r"(static_cast<uint32_t>(sfb)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0))
    );
}

// Mixed FP4_A × FP8_B block-scaled MMA: m16n8k32, scale_vec::1X (swapAB of the
// fp8×fp4 mixed path). Uses mxf8f6f4 with e2m1 A and e4m3 B. Verified on RTX PRO
// 6000 (sm_120a). A is the fp4 operand (4 regs, unpacked 1-byte container, value
// << 2, same layout as the fp4 B in fp8_fp4_mixed); B is the fp8 operand (2 regs).
__device__ __forceinline__ void fp4_fp8_mixed_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint8_t sfa, uint8_t sfb
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(static_cast<uint32_t>(sfa)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
          "r"(static_cast<uint32_t>(sfb)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0))
    );
}

// FP4 Block-Scaled MMA: m16n8k64
static constexpr int FP4_MMA_M = 16, FP4_MMA_N = 8, FP4_MMA_K = 64;
static constexpr int FP4_MMA_ACCUM = 4;

__device__ __forceinline__ void fp4_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
    uint16_t sfa, uint16_t sfb
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0), "r"(b1),
          "r"(static_cast<uint32_t>(sfa)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
          "r"(static_cast<uint32_t>(sfb)), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0))
    );
}

__device__ __forceinline__ void fp4_mma_block_scaled(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    uint16_t sfa, uint16_t sfb
) {
    fp4_mma_block_scaled(d, a, b[0], b[1], sfa, sfb);
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
