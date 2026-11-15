#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

// Duplicate of the CUDA>=13 guard in `common/sm120_utils.cuh` -- see that file for the
// full rationale and for why it sits before the includes, inside the `__CUDA_ARCH__ >= 1200`
// block. It is repeated here because THIS is the file that actually emits the
// `block_scale` MMA PTX. Today the SM120 MMA kernels include both headers, so the
// duplicate adds no coverage; it makes coverage structural rather than incidental, so a
// future kernel that includes only this header is still guarded.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ < 13)
#error "DeepGEMM SM120 kernels require CUDA 13.0 or newer"
#endif

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

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
    // Accumulate the local tensor-core dot into zero, then explicitly round
    // the cross-instruction accumulation in FP32. Keeping a long reduction
    // inside MMA can move a projection across a BF16 rounding midpoint.
    float partial[4];
    const float zero = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%10,%10,%10};\n"
        : "=f"(partial[0]), "=f"(partial[1]), "=f"(partial[2]), "=f"(partial[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "f"(zero)
    );
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        d[i] = __fadd_rn(d[i], partial[i]);
}

// Rare BF16 midpoint refinement. FP32 accumulation can select the opposite
// BF16 neighbor from the FP64 reference. Keep this
// slow path outside the MMA loop; ordinary values retain the tensor-core path.
__device__ __forceinline__ float bf16_refine_midpoint(
    float value, const __nv_bfloat16* a, const __nv_bfloat16* b, uint32_t k, bool valid = true
) {
    const uint32_t bits = __float_as_uint(value);
    const int distance = static_cast<int>(bits & 0xffffu) - 0x8000;
    // This guard is a practical refinement window, not a proof of correctly
    // rounded arbitrary GEMMs. Infinities and NaNs keep their normal behavior.
    constexpr uint32_t mask = 0xffffffffu;
    const uint32_t lane = threadIdx.x % 32;
    uint32_t pending = __ballot_sync(mask, valid and (bits & 0x7f800000u) != 0x7f800000u
                                    and distance >= -32 and distance <= 32);
    // All lanes in a full math warp must call this helper, even for invalid
    // output positions. Cooperate on each rare dot to keep K loads coalesced
    // and avoid a long scalar tail on the decode critical path.
    while (pending) {
        const uint32_t owner = __ffs(pending) - 1;
        const auto* row_a = reinterpret_cast<const __nv_bfloat16*>(
            __shfl_sync(mask, reinterpret_cast<unsigned long long>(a), owner));
        const auto* row_b = reinterpret_cast<const __nv_bfloat16*>(
            __shfl_sync(mask, reinterpret_cast<unsigned long long>(b), owner));
        // Retain the low part of each FP32 addition with TwoSum. BF16
        // products are exactly representable within the normal FP32 range.
        float sum = 0.f, residual = 0.f;
        auto add = [&](float term) {
            const float next = __fadd_rn(sum, term);
            const float split = __fsub_rn(next, sum);
            const float error = __fadd_rn(__fsub_rn(sum, __fsub_rn(next, split)),
                                         __fsub_rn(term, split));
            residual = __fadd_rn(residual, error);
            sum = next;
        };
        for (uint32_t index = lane; index < k; index += 32)
            add(__fmul_rn(__bfloat162float(row_a[index]), __bfloat162float(row_b[index])));
        #pragma unroll
        for (uint32_t offset = 16; offset; offset >>= 1) {
            const float other_sum = __shfl_down_sync(mask, sum, offset);
            const float other_residual = __shfl_down_sync(mask, residual, offset);
            add(other_sum);
            residual = __fadd_rn(residual, other_residual);
        }
        const float refined = __shfl_sync(mask, __fadd_rn(sum, residual), 0);
        if (lane == owner) value = refined;
        pending &= pending - 1;
    }
    return value;
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
