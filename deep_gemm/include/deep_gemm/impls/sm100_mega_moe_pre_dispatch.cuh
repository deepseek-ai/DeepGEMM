#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/reduction.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// Fused BF16 → quant + topk copy + pad-fill kernel that produces the exact
// byte layout DeepGEMM's mega-MoE symmetric buffer expects in its `x`,
// `x_sf`, `topk_idx`, and `topk_weights` slots. Two variants:
//
//   - `kUseFp4Acts == false` → FP8 (E4M3) acts; per-row stride = `hidden`.
//   - `kUseFp4Acts == true`  → packed FP4 (E2M1) acts; per-row stride
//                              = `hidden / 2`. Layout: byte holds 2 nibbles,
//                              low nibble = even col, high nibble = odd col,
//                              matching `deep_gemm.utils.per_token_cast_to_fp4`.
//
// Both paths share the UE8M0 SF byte layout: `byte_off = token*num_groups +
// group`, with the contiguous `(P, num_groups/4)` int32 slot storing 4 bytes
// per int32 in row-major order.
//
// The FP4 quant matches `per_token_cast_to_fp4` (host helper) bytewise via
// explicit bucketize boundaries — PTX `cvt.rn.satfinite.e2m1x2.f32` rounds
// midpoints to-even, but the host helper rounds midpoints toward zero.

// ceil_to_ue8m0(raw_scale) — matches `deep_gemm.utils.math.ceil_to_ue8m0`:
// returns the UE8M0 exponent byte (in [1, 254]) such that 2^(exp-127) is the
// smallest power of 2 >= raw_scale.
__forceinline__ __device__ uint32_t pre_dispatch_cast_to_ue8m0(float raw_scale) {
    uint32_t bits = __float_as_uint(raw_scale);
    uint32_t exp = (bits >> 23u) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (mantissa != 0u) exp += 1u;
    if (exp < 1u)   exp = 1u;
    if (exp > 254u) exp = 254u;
    return exp;
}

// E2M1 (FP4) bucketize encode matching `deep_gemm.utils.math._quantize_to_fp4_e2m1`.
// Boundaries are midpoints between adjacent representable magnitudes; ties round
// toward zero (bucketize default), which differs from PTX `cvt.rn.satfinite`
// rounding ties to even.
__forceinline__ __device__ uint32_t pre_dispatch_e2m1_encode(float v) {
    float ax = fabsf(v);
    if (ax > 6.0f) ax = 6.0f;
    uint32_t idx = (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) +
                   (ax > 1.75f) + (ax > 2.5f)  + (ax > 3.5f)  + (ax > 5.0f);
    uint32_t code = idx;
    if ((v < 0.0f) && (idx != 0u))
        code |= 0x8u;
    return code;
}

template <uint32_t kGroupSize, bool kUseFp4Acts, bool kUsePDL>
__launch_bounds__(1024, 2)
__global__ void mega_moe_pre_dispatch_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t*       __restrict__ topk_idx,
    const float*         __restrict__ topk_weights,
    void*                __restrict__ buf_x,
    int32_t*             __restrict__ buf_x_sf,
    int64_t*             __restrict__ buf_topk_idx,
    float*               __restrict__ buf_topk_weights,
    const uint32_t num_tokens,
    const uint32_t padded_max,
    const uint32_t hidden,
    const uint32_t num_groups,
    const uint32_t top_k) {
    static_assert(kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128,
                  "kGroupSize must be 32, 64, or 128");
    constexpr uint32_t kVecElems = 8;  // 16-byte BF16 load per thread
    static_assert(kGroupSize % kVecElems == 0, "kGroupSize must be a multiple of 8");
    constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;

    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if constexpr (kUsePDL) {
        cudaGridDependencySynchronize();
    }

    if (bid < num_tokens) {
        // ---- Quantize path: one CTA per valid token ----
        const uint32_t token_id = bid;

        const auto* token_in = x + static_cast<uint64_t>(token_id) * hidden;
        // Coalesced 16-byte BF16 vector load. Threads cover columns
        // [tid*kVecElems, tid*kVecElems + kVecElems) — each thread owns
        // one contiguous slice of one token.
        uint4 in_bits = reinterpret_cast<const uint4*>(token_in)[tid];
        const auto* bf16_pairs = reinterpret_cast<const __nv_bfloat162*>(&in_bits);

        float vals[kVecElems];
        float local_max = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < kVecElems / 2; ++i) {
            float2 fp = __bfloat1622float2(bf16_pairs[i]);
            vals[2 * i + 0] = fp.x;
            vals[2 * i + 1] = fp.y;
            local_max = fmaxf(local_max, fmaxf(fabsf(fp.x), fabsf(fp.y)));
        }

        // Reduce absmax across the kThreadsPerGroup threads that cover one
        // group. Lanes outside the group keep their own value (different
        // group's max), so SF write below is gated to one thread per group.
        local_max = warp_reduce<kThreadsPerGroup, /*kIntergroupReduce=*/false>(
            local_max, ReduceMax<float>{});

        // Match host `per_token_cast_to_fp4/fp8`: clamp absmax to 1e-4
        // before dividing by the dtype's max representable value.
        const float absmax = fmaxf(local_max, 1e-4f);
        constexpr float kFinfoMax = kUseFp4Acts ? 6.0f : 448.0f;
        const float raw_scale = absmax / kFinfoMax;
        const uint32_t ue8m0_exp = pre_dispatch_cast_to_ue8m0(raw_scale);
        // 1 / 2^(ue8m0_exp - 127) = 2^(127 - ue8m0_exp); fp32 bits =
        // (127 - ue8m0_exp + 127) << 23 = (254 - ue8m0_exp) << 23.
        const float inv_scale = __uint_as_float((254u - ue8m0_exp) << 23u);

        if constexpr (kUseFp4Acts) {
            // 8 BF16 → 4 packed nibbles → 4 bytes (uint32_t). Output stride
            // per token is hidden/2; thread tid writes 4 bytes at offset
            // [tid*4, tid*4+4) in the output row. Pairing matches host
            // `per_token_cast_to_fp4`: byte b's low nibble is column 2b
            // (even), high nibble is column 2b+1 (odd).
            uint32_t packed = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kVecElems / 2; ++i) {
                const uint32_t lo = pre_dispatch_e2m1_encode(vals[2 * i + 0] * inv_scale);
                const uint32_t hi = pre_dispatch_e2m1_encode(vals[2 * i + 1] * inv_scale);
                packed |= ((lo & 0xFu) | ((hi & 0xFu) << 4u)) << (8u * i);
            }
            auto* row_out = static_cast<uint32_t*>(buf_x) +
                            static_cast<uint64_t>(token_id) * (hidden / 8u);
            row_out[tid] = packed;
        } else {
            // 8 BF16 → 4 fp8x2 = 8 FP8 bytes (uint64_t). Output stride per
            // token is `hidden` bytes. Use CUDA's saturating fp8 conversion
            // (RNE), matching PyTorch's `.to(torch.float8_e4m3fn)`.
            uint64_t packed = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kVecElems / 2; ++i) {
                const __nv_fp8x2_storage_t fp8x2 = __nv_cvt_float2_to_fp8x2(
                    make_float2(vals[2 * i + 0] * inv_scale, vals[2 * i + 1] * inv_scale),
                    __NV_SATFINITE, __NV_E4M3);
                packed |= static_cast<uint64_t>(fp8x2) << (16u * i);
            }
            auto* row_out = static_cast<uint64_t*>(buf_x) +
                            static_cast<uint64_t>(token_id) * (hidden / 8u);
            row_out[tid] = packed;
        }

        // One thread per group writes its UE8M0 exponent byte. Row-major
        // contiguous layout into `buf_x_sf` viewed as bytes:
        //   byte_off = token_id * num_groups + group_id.
        const uint32_t group_id = tid / kThreadsPerGroup;
        const uint32_t within_group_id = tid % kThreadsPerGroup;
        if (within_group_id == 0u && group_id < num_groups) {
            const uint32_t byte_off = token_id * num_groups + group_id;
            reinterpret_cast<uint8_t*>(buf_x_sf)[byte_off] =
                static_cast<uint8_t>(ue8m0_exp);
        }

        // Copy this token's topk row. top_k is small (≤ num_threads enforced
        // at host); each tid<top_k thread copies one entry.
        if (tid < top_k) {
            const uint32_t off = token_id * top_k + tid;
            buf_topk_idx[off]     = static_cast<int64_t>(topk_idx[off]);
            buf_topk_weights[off] = topk_weights[off];
        }
    } else {
        // ---- Pad path: trailing CTAs fill [num_tokens, padded_max) topk
        // slots with (-1, 0.0) so the dispatch sentinel matches an empty
        // expert assignment. blockDim.x slots per pad CTA.
        const uint32_t copy_bid = bid - num_tokens;
        const uint32_t pad_base = num_tokens * top_k;
        const uint32_t slot     = pad_base + copy_bid * blockDim.x + tid;
        const uint32_t total    = padded_max * top_k;
        if (slot < total) {
            buf_topk_idx[slot]     = static_cast<int64_t>(-1);
            buf_topk_weights[slot] = 0.0f;
        }
    }

    if constexpr (kUsePDL) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

}  // namespace deep_gemm
