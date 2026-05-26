// SPDX-License-Identifier: MIT
//
// MXFP4 (E2M1) → FP8 (E4M3) dequant helper for SM90 W4A8 fused MegaMoE.
//
// Ported from vLLM Marlin's `dequant<__nv_fp8x4_e4m3, kFE2M1f, true>` in
// `csrc/quantization/marlin/dequant.h` (Marlin / vLLM Apache 2.0). The bit
// pattern conversion is intentionally identical so that weight files prepared
// with Marlin / TensorRT-LLM `mxfp4_moe` preprocessing can be reused.

#pragma once

#include <cuda_fp8.h>
#include <cstdint>

namespace deep_gemm {
namespace w4a8 {

#define DG_W4A8_INLINE __device__ __forceinline__

// Convert one packed-FP4 dword (8 nibbles = 8 FP4 values) into eight FP8 E4M3
// bytes laid out as two `__nv_fp8x4_e4m3` fragments.
//
// IMPORTANT: this routine performs the bit-pattern conversion only; the per-32
// E8M0 group scale must be applied separately on the WGMMA accumulator
// (Marlin-style), not folded into the FP8 exponent (which is the Humming
// alternative we deliberately do not use here).
//
// Layout note inherited from Marlin: the upper half of `q` (q << 4 step)
// produces `frag_b[0]` and the lower half produces `frag_b[1]`. This reverse
// indexing matches the way Marlin permutes the packed-FP4 weight tile so the
// FP8 register layout aligns with the WGMMA m64n*k32 B-fragment layout.
DG_W4A8_INLINE void dequant_mxfp4_to_fp8(int q, __nv_fp8x4_e4m3* frag_b) {
    constexpr int FP4_EXPONENT = 2;
    constexpr int FP8_EXPONENT = 4;
    constexpr int RIGHT_SHIFT = FP8_EXPONENT - FP4_EXPONENT;  // = 2
    constexpr int MASK = 0x70707070;

    int Out1 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 4;
    int Out2 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);

    frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
    frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

// Convert an E8M0 byte (unsigned 8-bit exponent, bias = 127) into a float
// multiplicative scale. E8M0 stores `2 ** (e - 127)`. Used to apply the
// MXFP4 per-32 group scale to a WGMMA accumulator.
DG_W4A8_INLINE float e8m0_to_float(std::uint8_t e8m0_byte) {
    // 2 ** (e - 127). Build via float bit pattern: sign=0, exponent=e, mantissa=0.
    std::uint32_t bits = static_cast<std::uint32_t>(e8m0_byte) << 23;
    return *reinterpret_cast<const float*>(&bits);
}

#undef DG_W4A8_INLINE

} // namespace w4a8
} // namespace deep_gemm
