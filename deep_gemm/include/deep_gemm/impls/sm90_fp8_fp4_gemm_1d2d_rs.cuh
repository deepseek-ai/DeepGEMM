#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

// SM90 FP8 x FP4 GEMM, RS-mode variant.
//
// S2 stage: this kernel is a verbatim copy of `sm90_fp8_fp4_gemm_1d2d_impl`,
// renamed to `sm90_fp8_fp4_gemm_1d2d_rs_impl` so it gets its own JIT artifact.
// Functionally identical to the SS variant; later steps will replace the
// inner WGMMA loop with RS-mode (register A) WGMMA + register-resident FP4
// dequant (using `FP8MMASelectorRS`, ldmatrix, prmt+lop3) without touching
// the SS variant or the contiguous code path.

// =============================================================================
// S3 prep (M1): RS-mode utility helpers.
// These helpers are intentionally not yet called by `sm90_fp8_fp4_gemm_1d2d_rs_impl`;
// they are introduced as a self-contained, side-effect-free building block so
// the second-round kernel rewrite can plug them in without further header
// surgery. Numerical equivalence with the SS variant is preserved this round.
// =============================================================================
namespace fp4_rs_detail {

template <bool kBF16FinalAccum>
struct FinalAccumStorage {
    using type = float;
};

template <>
struct FinalAccumStorage<true> {
    using type = nv_bfloat162;
};

template <bool kBF16FinalAccum, typename dtype_t>
CUTLASS_DEVICE void final_accum_init(dtype_t* base, uint32_t idx) {
    if constexpr (kBF16FinalAccum) {
        base[idx] = __float22bfloat162_rn({0.0f, 0.0f});
    } else {
        base[idx] = 0.0f;
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair(dtype_t* base, uint32_t pair_idx,
                                             float scale_0, float value_0,
                                             float scale_1, float value_1) {
    if constexpr (kBF16PromoteMath) {
        const nv_bfloat162 scale = __float22bfloat162_rn({scale_0, scale_1});
        const nv_bfloat162 value = __float22bfloat162_rn({value_0, value_1});
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        const nv_bfloat162 out = __hfma2(scale, value, dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = out;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(out);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(scale_0, value_0, dst.x);
            dst.y = __fmaf_rn(scale_1, value_1, dst.y);
        } else {
            dst.x += scale_0 * value_0;
            dst.y += scale_1 * value_1;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        if constexpr (kFmaPromote) {
            base[idx + 0] = __fmaf_rn(scale_0, value_0, base[idx + 0]);
            base[idx + 1] = __fmaf_rn(scale_1, value_1, base[idx + 1]);
        } else {
            base[idx + 0] += scale_0 * value_0;
            base[idx + 1] += scale_1 * value_1;
        }
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair2(dtype_t* base, uint32_t pair_idx,
                                              float scale_0_a, float value_0_a,
                                              float scale_1_a, float value_1_a,
                                              float scale_0_b, float value_0_b,
                                              float scale_1_b, float value_1_b) {
    if constexpr (kBF16PromoteMath) {
        const nv_bfloat162 scale_a = __float22bfloat162_rn({scale_0_a, scale_1_a});
        const nv_bfloat162 value_a = __float22bfloat162_rn({value_0_a, value_1_a});
        const nv_bfloat162 scale_b = __float22bfloat162_rn({scale_0_b, scale_1_b});
        const nv_bfloat162 value_b = __float22bfloat162_rn({value_0_b, value_1_b});
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        dst = __hfma2(scale_a, value_a, dst);
        dst = __hfma2(scale_b, value_b, dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = dst;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(dst);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(scale_0_a, value_0_a, dst.x);
            dst.y = __fmaf_rn(scale_1_a, value_1_a, dst.y);
            dst.x = __fmaf_rn(scale_0_b, value_0_b, dst.x);
            dst.y = __fmaf_rn(scale_1_b, value_1_b, dst.y);
        } else {
            dst.x += scale_0_a * value_0_a;
            dst.y += scale_1_a * value_1_a;
            dst.x += scale_0_b * value_0_b;
            dst.y += scale_1_b * value_1_b;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        float dst_0 = base[idx + 0];
        float dst_1 = base[idx + 1];
        if constexpr (kFmaPromote) {
            dst_0 = __fmaf_rn(scale_0_a, value_0_a, dst_0);
            dst_1 = __fmaf_rn(scale_1_a, value_1_a, dst_1);
            dst_0 = __fmaf_rn(scale_0_b, value_0_b, dst_0);
            dst_1 = __fmaf_rn(scale_1_b, value_1_b, dst_1);
        } else {
            dst_0 += scale_0_a * value_0_a;
            dst_1 += scale_1_a * value_1_a;
            dst_0 += scale_0_b * value_0_b;
            dst_1 += scale_1_b * value_1_b;
        }
        base[idx + 0] = dst_0;
        base[idx + 1] = dst_1;
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair4(dtype_t* base, uint32_t pair_idx,
                                              float scale_0_a, float value_0_a,
                                              float scale_1_a, float value_1_a,
                                              float scale_0_b, float value_0_b,
                                              float scale_1_b, float value_1_b,
                                              float scale_0_c, float value_0_c,
                                              float scale_1_c, float value_1_c,
                                              float scale_0_d, float value_0_d,
                                              float scale_1_d, float value_1_d) {
    if constexpr (kBF16PromoteMath) {
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        dst = __hfma2(__float22bfloat162_rn({scale_0_a, scale_1_a}),
                      __float22bfloat162_rn({value_0_a, value_1_a}), dst);
        dst = __hfma2(__float22bfloat162_rn({scale_0_b, scale_1_b}),
                      __float22bfloat162_rn({value_0_b, value_1_b}), dst);
        dst = __hfma2(__float22bfloat162_rn({scale_0_c, scale_1_c}),
                      __float22bfloat162_rn({value_0_c, value_1_c}), dst);
        dst = __hfma2(__float22bfloat162_rn({scale_0_d, scale_1_d}),
                      __float22bfloat162_rn({value_0_d, value_1_d}), dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = dst;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(dst);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(scale_0_a, value_0_a, dst.x);
            dst.y = __fmaf_rn(scale_1_a, value_1_a, dst.y);
            dst.x = __fmaf_rn(scale_0_b, value_0_b, dst.x);
            dst.y = __fmaf_rn(scale_1_b, value_1_b, dst.y);
            dst.x = __fmaf_rn(scale_0_c, value_0_c, dst.x);
            dst.y = __fmaf_rn(scale_1_c, value_1_c, dst.y);
            dst.x = __fmaf_rn(scale_0_d, value_0_d, dst.x);
            dst.y = __fmaf_rn(scale_1_d, value_1_d, dst.y);
        } else {
            dst.x += scale_0_a * value_0_a;
            dst.y += scale_1_a * value_1_a;
            dst.x += scale_0_b * value_0_b;
            dst.y += scale_1_b * value_1_b;
            dst.x += scale_0_c * value_0_c;
            dst.y += scale_1_c * value_1_c;
            dst.x += scale_0_d * value_0_d;
            dst.y += scale_1_d * value_1_d;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        float dst_0 = base[idx + 0];
        float dst_1 = base[idx + 1];
        if constexpr (kFmaPromote) {
            dst_0 = __fmaf_rn(scale_0_a, value_0_a, dst_0);
            dst_1 = __fmaf_rn(scale_1_a, value_1_a, dst_1);
            dst_0 = __fmaf_rn(scale_0_b, value_0_b, dst_0);
            dst_1 = __fmaf_rn(scale_1_b, value_1_b, dst_1);
            dst_0 = __fmaf_rn(scale_0_c, value_0_c, dst_0);
            dst_1 = __fmaf_rn(scale_1_c, value_1_c, dst_1);
            dst_0 = __fmaf_rn(scale_0_d, value_0_d, dst_0);
            dst_1 = __fmaf_rn(scale_1_d, value_1_d, dst_1);
        } else {
            dst_0 += scale_0_a * value_0_a;
            dst_1 += scale_1_a * value_1_a;
            dst_0 += scale_0_b * value_0_b;
            dst_1 += scale_1_b * value_1_b;
            dst_0 += scale_0_c * value_0_c;
            dst_1 += scale_1_c * value_1_c;
            dst_0 += scale_0_d * value_0_d;
            dst_1 += scale_1_d * value_1_d;
        }
        base[idx + 0] = dst_0;
        base[idx + 1] = dst_1;
    }
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair4x2(dtype_t* base, uint32_t pair_idx,
                                                float scale_0_a, float value_0_a,
                                                float scale_1_a, float value_1_a,
                                                float scale_0_b, float value_0_b,
                                                float scale_1_b, float value_1_b,
                                                float scale_0_c, float value_0_c,
                                                float scale_1_c, float value_1_c,
                                                float scale_0_d, float value_0_d,
                                                float scale_1_d, float value_1_d,
                                                float scale_2_a, float value_2_a,
                                                float scale_3_a, float value_3_a,
                                                float scale_2_b, float value_2_b,
                                                float scale_3_b, float value_3_b,
                                                float scale_2_c, float value_2_c,
                                                float scale_3_c, float value_3_c,
                                                float scale_2_d, float value_2_d,
                                                float scale_3_d, float value_3_d) {
    final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
        base, pair_idx,
        scale_0_a, value_0_a, scale_1_a, value_1_a,
        scale_0_b, value_0_b, scale_1_b, value_1_b,
        scale_0_c, value_0_c, scale_1_c, value_1_c,
        scale_0_d, value_0_d, scale_1_d, value_1_d);
    final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
        base, pair_idx + 1,
        scale_2_a, value_2_a, scale_3_a, value_3_a,
        scale_2_b, value_2_b, scale_3_b, value_3_b,
        scale_2_c, value_2_c, scale_3_c, value_3_c,
        scale_2_d, value_2_d, scale_3_d, value_3_d);
}

CUTLASS_DEVICE void keep_float_live(float value) {
    asm volatile("" :: "f"(value) : "memory");
}

struct FP8MMAF16AccumM64N8K32RS {
    static constexpr uint32_t M = 64;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 32;
    static constexpr uint32_t kNumAccum = 2;

    template <typename GmmaDescriptor>
    CUTLASS_DEVICE static void wgmma(uint32_t const* a, GmmaDescriptor const& desc,
                                     uint32_t* d, bool scale_d) {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %7, 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3 "
            "  {%0, %1}, {%2, %3, %4, %5}, %6, p, 1, 1;\n"
            "}\n"
            : "+r"(d[0]), "+r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "l"(desc.desc_), "r"(static_cast<uint32_t>(scale_d)));
    }
};

CUTLASS_DEVICE void unpack_f16_accum_m64n8(uint32_t const* src, float* dst) {
    #pragma unroll
    for (uint32_t i = 0; i < FP8MMAF16AccumM64N8K32RS::kNumAccum; ++ i) {
        const half2 value_h = *reinterpret_cast<const half2*>(src + i);
        const float2 value_f = __half22float2(value_h);
        dst[i * 2 + 0] = value_f.x;
        dst[i * 2 + 1] = value_f.y;
    }
}

CUTLASS_DEVICE float scale_float_by_pow2(float value, float pow2_scale) {
    uint32_t value_bits = *reinterpret_cast<uint32_t*>(&value);
    const uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&pow2_scale);
    const int32_t exp_shift = static_cast<int32_t>((scale_bits >> 23) & 0xffu) - 127;
    value_bits = static_cast<uint32_t>(static_cast<int32_t>(value_bits) + (exp_shift << 23));
    return *reinterpret_cast<float*>(&value_bits);
}

template <bool kBF16FinalAccum, bool kFmaPromote, bool kBF16PromoteMath, bool kScaleBPow2Promote,
          typename dtype_t>
CUTLASS_DEVICE void final_accum_promote_pair4_split_scale(dtype_t* base, uint32_t pair_idx,
                                                          float scale_a_0, float scale_a_1,
                                                          float scale_b_a, float scale_b_b,
                                                          float scale_b_c, float scale_b_d,
                                                          float value_0_a, float value_1_a,
                                                          float value_0_b, float value_1_b,
                                                          float value_0_c, float value_1_c,
                                                          float value_0_d, float value_1_d) {
    auto prod_0 = [&](float scale_b) {
        return kScaleBPow2Promote ? scale_float_by_pow2(scale_a_0, scale_b) : scale_a_0 * scale_b;
    };
    auto prod_1 = [&](float scale_b) {
        return kScaleBPow2Promote ? scale_float_by_pow2(scale_a_1, scale_b) : scale_a_1 * scale_b;
    };
    if constexpr (kBF16PromoteMath) {
        nv_bfloat162 dst;
        if constexpr (kBF16FinalAccum) {
            dst = base[pair_idx];
        } else {
            const uint32_t idx = pair_idx * 2;
            dst = __float22bfloat162_rn({base[idx + 0], base[idx + 1]});
        }
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_a), prod_1(scale_b_a)}),
                      __float22bfloat162_rn({value_0_a, value_1_a}), dst);
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_b), prod_1(scale_b_b)}),
                      __float22bfloat162_rn({value_0_b, value_1_b}), dst);
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_c), prod_1(scale_b_c)}),
                      __float22bfloat162_rn({value_0_c, value_1_c}), dst);
        dst = __hfma2(__float22bfloat162_rn({prod_0(scale_b_d), prod_1(scale_b_d)}),
                      __float22bfloat162_rn({value_0_d, value_1_d}), dst);
        if constexpr (kBF16FinalAccum) {
            base[pair_idx] = dst;
        } else {
            const uint32_t idx = pair_idx * 2;
            const float2 out_f = __bfloat1622float2(dst);
            base[idx + 0] = out_f.x;
            base[idx + 1] = out_f.y;
        }
    } else if constexpr (kBF16FinalAccum) {
        float2 dst = __bfloat1622float2(base[pair_idx]);
        if constexpr (kFmaPromote) {
            dst.x = __fmaf_rn(prod_0(scale_b_a), value_0_a, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_a), value_1_a, dst.y);
            dst.x = __fmaf_rn(prod_0(scale_b_b), value_0_b, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_b), value_1_b, dst.y);
            dst.x = __fmaf_rn(prod_0(scale_b_c), value_0_c, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_c), value_1_c, dst.y);
            dst.x = __fmaf_rn(prod_0(scale_b_d), value_0_d, dst.x);
            dst.y = __fmaf_rn(prod_1(scale_b_d), value_1_d, dst.y);
        } else {
            dst.x += prod_0(scale_b_a) * value_0_a;
            dst.y += prod_1(scale_b_a) * value_1_a;
            dst.x += prod_0(scale_b_b) * value_0_b;
            dst.y += prod_1(scale_b_b) * value_1_b;
            dst.x += prod_0(scale_b_c) * value_0_c;
            dst.y += prod_1(scale_b_c) * value_1_c;
            dst.x += prod_0(scale_b_d) * value_0_d;
            dst.y += prod_1(scale_b_d) * value_1_d;
        }
        base[pair_idx] = __float22bfloat162_rn(dst);
    } else {
        const uint32_t idx = pair_idx * 2;
        float dst_0 = base[idx + 0];
        float dst_1 = base[idx + 1];
        if constexpr (kFmaPromote) {
            dst_0 = __fmaf_rn(prod_0(scale_b_a), value_0_a, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_a), value_1_a, dst_1);
            dst_0 = __fmaf_rn(prod_0(scale_b_b), value_0_b, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_b), value_1_b, dst_1);
            dst_0 = __fmaf_rn(prod_0(scale_b_c), value_0_c, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_c), value_1_c, dst_1);
            dst_0 = __fmaf_rn(prod_0(scale_b_d), value_0_d, dst_0);
            dst_1 = __fmaf_rn(prod_1(scale_b_d), value_1_d, dst_1);
        } else {
            dst_0 += prod_0(scale_b_a) * value_0_a;
            dst_1 += prod_1(scale_b_a) * value_1_a;
            dst_0 += prod_0(scale_b_b) * value_0_b;
            dst_1 += prod_1(scale_b_b) * value_1_b;
            dst_0 += prod_0(scale_b_c) * value_0_c;
            dst_1 += prod_1(scale_b_c) * value_1_c;
            dst_0 += prod_0(scale_b_d) * value_0_d;
            dst_1 += prod_1(scale_b_d) * value_1_d;
        }
        base[idx + 0] = dst_0;
        base[idx + 1] = dst_1;
    }
}

template <bool kBF16FinalAccum, typename dtype_t>
CUTLASS_DEVICE float final_accum_load_scalar(const dtype_t* base, uint32_t idx) {
    if constexpr (kBF16FinalAccum) {
        const float2 value = __bfloat1622float2(base[idx / 2]);
        return (idx % 2 == 0) ? value.x : value.y;
    } else {
        return base[idx];
    }
}

template <bool kBF16FinalAccum, typename dtype_t>
CUTLASS_DEVICE nv_bfloat162 final_accum_load_pair_bf16(const dtype_t* base, uint32_t pair_idx) {
    if constexpr (kBF16FinalAccum) {
        return base[pair_idx];
    } else {
        return __float22bfloat162_rn({base[pair_idx * 2 + 0], base[pair_idx * 2 + 1]});
    }
}

// `ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16`: load 4 8x8 b16 matrices
// from swizzled smem into 4 b32 registers per lane, transposed. The transpose
// option is the key to feeding RS-mode WGMMA's A operand layout when the
// source tile is laid out K-major in smem (which is the only layout the
// dequant step can produce cheaply from packed FP4 input).
//
// Note: WGMMA RS-mode A still consumes 8-bit (E4M3) pairs, so we phrase the
// granularity as b16 ldmatrix and let the upper layer treat each b32 register
// as 4 packed E4M3 lanes.
struct SM90_U32x4_LDSM_T {
    CUTLASS_DEVICE static void
    copy(uint32_t& dst_0, uint32_t& dst_1, uint32_t& dst_2, uint32_t& dst_3, void* smem_src) {
        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst_0), "=r"(dst_1), "=r"(dst_2), "=r"(dst_3)
            : "l"(__cvta_generic_to_shared(smem_src)));
    }
};

template <uint32_t COL_BYTES, uint32_t NV>
CUTLASS_DEVICE uint32_t permute_col(const uint32_t row, const uint32_t col) {
    constexpr uint32_t strd = 128 / (COL_BYTES < 128u ? COL_BYTES : 128u);
    return ((col / NV) ^ (row % 8 / strd)) * NV;
}

template <typename dtype_t>
struct SM90_U32x2_STSM_T {
    CUTLASS_DEVICE static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        DG_STATIC_ASSERT(sizeof(dtype_t) == sizeof(uint32_t), "Invalid dtype");
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16.trans [%0], {%1, %2};\n"
                     :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(src[0]), "r"(src[1]));
    }
};

// Decode a single FP4 (E2M1) code (low 4 bits) to its E4M3 byte
// representation. Same byte-LUT formulation as the scalar path inside the SS
// kernel, hoisted to namespace scope so it can also be called from
// register-level dequant routines.
//   mag {0..7} -> {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c}
//   sign bit copied from FP4 bit 3 to E4M3 bit 7 (FP4 -0 maps to E4M3 -0;
//   WGMMA treats -0 as 0, so this is numerically safe and saves a branch).
CUTLASS_DEVICE uint32_t fp4_to_e4m3_byte(uint32_t code) {
    constexpr uint32_t LUT_LO = 0x3c383000u;
    constexpr uint32_t LUT_HI = 0x4c484440u;
    const uint32_t mag      = code & 0x07u;
    const uint32_t mag_byte = __byte_perm(LUT_LO, LUT_HI, mag) & 0xffu;
    const uint32_t sign     = (code & 0x08u) << 4;  // 0 or 0x80
    return mag_byte | sign;
}

// Decode 8 packed FP4 codes (one 32-bit word holding 8 nibbles) into 8 E4M3
// bytes packed into a single 64-bit value (low byte = code 0).
//
// This is the register-resident analogue of the SS variant's
// `fp4_pair_to_e4m3_pair` chained over 4 bytes. It is the building block the
// second-round kernel will use after `ldmatrix.x4.trans` to materialise A
// operand registers for RS-mode WGMMA without round-tripping through smem.
//
// Implementation kept straightforward (LUT per nibble). PTX-level
// vectorisation via `prmt`/`lop3` is left for a follow-up micro-optimisation
// in round 2 once the rest of the data path is verified correct.
CUTLASS_DEVICE uint64_t fp4x8_to_e4m3x8(uint32_t packed) {
    uint64_t out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t nib = (packed >> (i * 4)) & 0x0fu;
        out |= static_cast<uint64_t>(fp4_to_e4m3_byte(nib)) << (i * 8);
    }
    return out;
}

CUTLASS_DEVICE void fast_fp4_to_e4m3_convert(uint32_t outputs[2], uint32_t input) {
    const uint64_t decoded = fp4x8_to_e4m3x8(input);
    outputs[0] = static_cast<uint32_t>(decoded);
    outputs[1] = static_cast<uint32_t>(decoded >> 32);
}

CUTLASS_DEVICE uint32_t fp4x4_to_e4m3x4(uint32_t packed) {
    constexpr uint32_t pos0 = 0x3c383000u;
    constexpr uint32_t pos1 = 0x4c484440u;
    const uint32_t lut_idx = packed & 0x7777u;
    const uint32_t sign_shifted = packed << 4;
    uint32_t mag_bytes, sign_bytes;
    asm volatile(
        "{\n"
        "  prmt .b32 %0, %3, %4, %2;\n"
        "  prmt .b32 %1, %5, %6, 0xd9c8;\n"
        "}\n"
        : "=r"(mag_bytes), "=r"(sign_bytes)
        : "r"(lut_idx), "r"(pos0), "r"(pos1), "r"(sign_shifted), "r"(packed));
    uint32_t out;
    asm volatile(
        "{\n"
        "  lop3.b32 %0, %1, %2, 0x80808080, 0xf8;\n"
        "}\n"
        : "=r"(out)
        : "r"(mag_bytes), "r"(sign_bytes));
    return out;
}

CUTLASS_DEVICE int32_t pow2_scale_to_exp_shift(float scale) {
    const uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&scale);
    return static_cast<int32_t>((scale_bits >> 23) & 0xffu) - 127;
}

struct ScaledE4M3Lut {
    uint32_t lo;
    uint32_t hi;
};

CUTLASS_DEVICE ScaledE4M3Lut make_scaled_e4m3_lut(uint32_t exp_offset) {
    // Random FP4 per-32 scales are overwhelmingly ceil-pow2(max(abs(x))/6)
    // in {2^-1, 1}, which maps to exp_offset {5, 6}. Fast-path those to
    // avoid the dynamic IMAD chain in the WGMMA issue loop.
    if (exp_offset == 5u)
        return {0x34302800u, 0x44403c38u};
    if (exp_offset == 6u)
        return {0x3c383000u, 0x4c484440u};
    const uint32_t exp_offset_buffer1 =
        exp_offset * 0x08080800u + (exp_offset ? 0xfffffc00u : 0u);
    const uint32_t exp_offset_buffer2 = exp_offset * 0x08080808u;
    constexpr uint32_t mantissa_lo = 0x0c080400u;
    constexpr uint32_t mantissa_hi = 0x1c181410u;
    return {mantissa_lo + exp_offset_buffer1, mantissa_hi + exp_offset_buffer2};
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_lut(uint32_t packed, ScaledE4M3Lut lut) {
    const uint32_t lut_idx = packed & 0x7777u;
    const uint32_t sign_shifted = packed << 4;
    uint32_t mantissa_bytes, sign_bytes;
    asm volatile(
        "{\n"
        "  prmt .b32 %0, %3, %4, %2;\n"
        "  prmt .b32 %1, %5, %6, 0xd9c8;\n"
        "}\n"
        : "=r"(mantissa_bytes), "=r"(sign_bytes)
        : "r"(lut_idx), "r"(lut.lo), "r"(lut.hi), "r"(sign_shifted), "r"(packed));
    return (sign_bytes & 0x80808080u) | mantissa_bytes;
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_offset(uint32_t packed, uint32_t exp_offset) {
    return fp4x4_to_scaled_e4m3x4_lut(packed, make_scaled_e4m3_lut(exp_offset));
}

CUTLASS_DEVICE uint32_t fp4x4_to_scaled_e4m3x4_humming(uint32_t packed_nibbles, uint32_t exp_offset) {
    const uint32_t packed =
        (packed_nibbles & 0x000fu) |
        ((packed_nibbles & 0x00f0u) << 4) |
        ((packed_nibbles & 0x0f00u) << 8) |
        ((packed_nibbles & 0xf000u) << 12);

    const uint32_t exp_offset_buffer1 = exp_offset * 0x08080800u + (exp_offset ? 0xfffffc00u : 0u);
    const uint32_t exp_offset_buffer2 = exp_offset * 0x08080808u;
    const uint32_t exp_offsets0 = __byte_perm(exp_offset_buffer1, exp_offset_buffer2, packed);
    const uint32_t exp_offsets1 = __byte_perm(exp_offset_buffer1, exp_offset_buffer2, packed >> 16);

    uint32_t scaled_mag;
    asm volatile(
        "{\n"
        "  lop3.b32 %0, %1, 0x80808080, %2, 0xf8;\n"
        "}\n"
        : "=r"(scaled_mag)
        : "r"(packed << 4), "r"((packed & 0x07070707u) << 2));
    return scaled_mag + __byte_perm(exp_offsets0, exp_offsets1, 0x6420);
}

}  // namespace fp4_rs_detail

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters_rs(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters_rs<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t,
          bool kDecodeStub = false,
          bool kScaleAStub = false,
          bool kScaleBStub = false,
          bool kScaleBMulStub = false,
          bool kScaleProductStub = false,
          bool kScaleBPow2Promote = false,
          bool kScaleBEarlyLoad = false,
          bool kScaleBEarlyProduct = false,
          bool kScaleBDirectLoad = false,
          bool kWeightStub = false,
          bool kWGMMAStub = false,
          bool kStoreStub = false,
          bool kDirectStore = false,
          bool kSTSMStub = false,
          bool kTMAStoreStub = false,
          bool kSTSMConvertOnly = false,
          bool kPromoteStub = false,
          bool kPromoteAccumStub = false,
          bool kPromoteFinalAccumStub = false,
          bool kPromoteMulStub = false,
          bool kOverlapPromote = false,
          bool kFusedPromote = false,
          bool kPromoteFromSmem = false,
          bool kFmaPromote = false,
          bool kLateScaleA = false,
          bool kBF16FinalAccum = true,
          bool kBF16PromoteMath = false,
          bool kScaleKGroupExact = false,
          bool kBLoadStub = false,
          bool kDecodePairShfl = false,
          bool kParallelNWaves = false,
          bool kK32Pingpong = false,
          bool kK32PairReduce = false,
          bool kK32QuadReduce = false,
          bool kK32QuadSplitPromote = false,
          bool kK32QuadScaleBInline = false,
          bool kK32QuadScaleBPrefetch = false,
          bool kK32QuadScaleBVec4 = false,
          bool kK32QuadPair4x2Promote = false,
          bool kWGMMAF16Accum = false,
          bool kFinalAccumScratch = false,
          bool kK32QuadPersistentScaleProduct = false,
          bool kK32QuadShortProductPromote = false,
          bool kSmallMSimpleSched = false,
          bool kFuseScaleBDecode = false,
          bool kFuseScaleBDecodeStub = false,
          bool kFuseScaleBHummingDecode = false,
          bool kFuseScaleBDecodeFastCommon = false,
          bool kFuseScaleBPredecode = false,
          bool kFuseScaleBPredecodePair = false,
          bool kFuseScaleBSharedStage = false,
          bool kFuseScaleBWSDecode = false,
          bool kFuseScaleBOnDemandLut = false,
          bool kFuseScaleBSlicePromote = false,
          uint32_t kFuseScaleBDecodeAssumeExp = 0,
          bool kScaleBPackedUE8M0 = false,
          uint32_t kScaleBGranK = 128,
          uint32_t kScaleKGroup = 1,
          uint32_t kLaunchBoundsMinBlocks = 1,
          uint32_t kMathRegCap = 0,
          bool kBIsInt4Sym = false,
          bool kScaleBBF16 = false,
          bool kScaleBE8M0 = false,
          bool kReorderMaskedByMaxM = false>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, kLaunchBoundsMinBlocks) void
sm90_fp8_fp4_gemm_1d2d_rs_impl(int8_t* gmem_b_ptr, float* sfb, int* grouped_layout,
                            nv_bfloat16* gmem_d_ptr,
                            uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(kScaleBGranK == 32 or kScaleBGranK == 128,
                     "Only support per-32 or per-128-channel FP4 scaling");
    DG_STATIC_ASSERT(not kScaleBPackedUE8M0 or (kFuseScaleBDecode and kScaleBGranK == 32),
                     "Packed UE8M0 SFB is only supported by fused per-32 scale_b decode");
    DG_STATIC_ASSERT(kScaleBGranK == 128 or not kOverlapPromote,
                     "DG_W4_OVERLAP_PROMOTE does not support per-32 FP4 scaling");
    DG_STATIC_ASSERT(kScaleBGranK == 128 or kScaleKGroup == 1,
                     "DG_W4_SCALE_K_GROUP does not support per-32 FP4 scaling");
    DG_STATIC_ASSERT(kScaleKGroup == 1 or kScaleKGroup == 2 or kScaleKGroup == 4,
                     "DG_W4_SCALE_K_GROUP only supports 1/2/4");
    DG_STATIC_ASSERT(not kBIsInt4Sym, "RS-mode FP8xFP4 kernel does not support INT4-sym B");
    DG_STATIC_ASSERT(not (kScaleBBF16 and kScaleBE8M0), "Scale-B cannot be both BF16 and E8M0");
    DG_STATIC_ASSERT((not kScaleBBF16 and not kScaleBE8M0) or (kScaleBDirectLoad and kScaleBGranK == 32),
                     "Compressed Scale-B dtypes are only supported by direct-load per-32 path");
    DG_STATIC_ASSERT(kFuseScaleBDecodeAssumeExp == 0 or kFuseScaleBDecodeAssumeExp == 5 or
                     kFuseScaleBDecodeAssumeExp == 6,
                     "DG_W4_FUSE_SCALE_B_DECODE_ASSUME_EXP only supports 0/5/6");
    DG_STATIC_ASSERT(
        math::constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or
        (math::constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");
    // Types
    using WGMMA = typename mma::sm90::FP8MMASelectorRS<BLOCK_M>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M <= 256, "Invalid RS-mode MMA N size");
    DG_STATIC_ASSERT(BLOCK_N % WGMMA::M == 0, "RS-mode swap_ab requires BLOCK_N to be tiled by 64-row WGMMA M");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_ROWS = BLOCK_M < WGMMA::M ? WGMMA::M : BLOCK_M;
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(SMEM_D_ROWS * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_TMA_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE =
        (BLOCK_M < WGMMA::M ? WGMMA::M : BLOCK_M) * BLOCK_K * sizeof(__nv_fp8_e4m3);
    // Packed FP4 B is loaded by TMA into a separate buffer; each row is BLOCK_K / 2 bytes.
    static constexpr uint32_t BLOCK_K_PACKED = BLOCK_K / 2;
    static constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE = BLOCK_N * BLOCK_K_PACKED;
    static constexpr uint32_t kNumRSMathWGs = kNumMathThreads / 128;
    static constexpr uint32_t SMEM_SFA_TMA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        (BLOCK_M < WGMMA::M ? WGMMA::M : BLOCK_M) * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    static constexpr uint32_t SCALE_B_ELEMENT_SIZE =
        kScaleBE8M0 ? static_cast<uint32_t>(sizeof(uint8_t)) :
        (kScaleBBF16 ? static_cast<uint32_t>(sizeof(nv_bfloat16)) :
                       static_cast<uint32_t>(sizeof(float)));
    const uint32_t shape_k_scales_a = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_k_scales_b = math::ceil_div(shape_k, kScaleBGranK);
    const uint32_t aligned_shape_n_sfb = math::align<uint32_t>(shape_n, 16u / SCALE_B_ELEMENT_SIZE);
    // SFB cache aliases smem_d when it fits. Small-M tiles may not have enough
    // smem_d capacity, so they fall back to a separate SFB region.
    const uint32_t smem_sfb_bytes = kScaleBGranK == 32 ?
        math::align<uint32_t>((BLOCK_K / kScaleBGranK) *
                              (kFuseScaleBDecode ? math::ceil_div(BLOCK_N, 4u) : BLOCK_N) *
                              sizeof(float), 16u) :
        math::align<uint32_t>(shape_k_scales_b * BLOCK_N * sizeof(float), 16u);
    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();
    constexpr uint32_t WAVE_BLOCK_M = BLOCK_N <= WGMMA::M ? BLOCK_N : WGMMA::M * 2;
    DG_STATIC_ASSERT(BLOCK_N % WAVE_BLOCK_M == 0, "Invalid block sizes");
    constexpr uint32_t WAVE_WGMMA = BLOCK_N / WAVE_BLOCK_M;
    constexpr uint32_t kWGsPerNWave = WAVE_BLOCK_M / WGMMA::M;
    constexpr bool kParallelNWavesEnabled =
        kParallelNWaves and WAVE_WGMMA == 2 and kWGsPerNWave == 2 and kNumMathThreads == 512;
    DG_STATIC_ASSERT(not kParallelNWaves or kParallelNWavesEnabled,
                     "DG_W4_PARALLEL_N_WAVES only supports BN256 with 512 math threads");
    constexpr uint32_t kBaseWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
    constexpr uint32_t kEmptyBarrierMathWarps =
        kParallelNWavesEnabled ? kBaseWGMMAStoreThreads / 32 : kNumMathThreads / 32;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_B_PACKED_OFFSET = SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE;
    auto smem_b_packed = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + SMEM_B_PACKED_OFFSET + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_B_PACKED_OFFSET + kNumStages * SMEM_B_PACKED_SIZE_PER_STAGE;
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SFB_OFFSET = SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE;
    // Prefer aliasing SFB onto smem_d. Small-M tiles usually have too little
    // smem_d, and SHAPE_K can be runtime-only, so choose the separate SFB region
    // dynamically to match the host-side smem_size calculation.
    const bool use_separate_sfb = smem_sfb_bytes > SMEM_D_SIZE;
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + (use_separate_sfb ? SMEM_SFB_OFFSET : 0));
    auto smem_sfb_exp = reinterpret_cast<uint32_t*>(smem_sfb);

    // Fill barriers.
    // After the A/packed-B barrier merge there is only one set of full/empty barriers.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_SFB_OFFSET + (use_separate_sfb ? smem_sfb_bytes : 0));
    auto full_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kEmptyBarrierMathWarps);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kDefaultMathRegisters = kNumMathThreads == 128 ? 248 : 232;
    constexpr uint32_t kNumMathRegisters = kMathRegCap == 0 ? kDefaultMathRegisters : kMathRegCap;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // `gmem_b_ptr` is no longer used: B is now loaded by TMA into `smem_b_packed`.
    (void)gmem_b_ptr;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);
    uint32_t simple_sched_linear_idx = blockIdx.x;
    constexpr bool kUseSmallMSimpleSched =
        kSmallMSimpleSched and kGemmType == GemmType::MGroupedMasked and BLOCK_M <= 8 and kNumTMAMulticast == 1;
    auto get_next_block = [&]() {
        if constexpr (kUseSmallMSimpleSched) {
            const uint32_t n_blocks = math::ceil_div(shape_n, BLOCK_N);
            const uint32_t total_blocks = n_blocks * kNumGroups;
            while (simple_sched_linear_idx < total_blocks) {
                scheduler.current_group_idx = simple_sched_linear_idx / n_blocks;
                n_block_idx = simple_sched_linear_idx - scheduler.current_group_idx * n_blocks;
                m_block_idx = 0;
                simple_sched_linear_idx += gridDim.x;
                if (scheduler.is_computation_valid(m_block_idx, 0))
                    return true;
            }
            return false;
        } else {
            return scheduler.get_next_block(m_block_idx, n_block_idx);
        }
    };
    auto get_current_group_idx = [&]() {
        if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return static_cast<uint32_t>(cute::max(0, grouped_layout[m_block_idx * BLOCK_M]));
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return scheduler.current_group_idx;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return scheduler.current_group_idx;
        } else {
            return 0u;
        }
    };

    // Pipeline and TMA phases (single shared pipeline for A/SFA/packed-B after the barrier merge)
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used.
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`.
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (get_next_block()) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release for the (now merged) A/SFA/packed-B slot
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA A
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.template get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                             num_tma_multicast_a, batch_idx);
                    if constexpr (not kScaleAStub) {
                        tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                                 smem_sfa[stage_idx], m_block_idx * BLOCK_M, scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales_a, 1, k_block_idx),
                                 num_tma_multicast_a);
                    }

                    if constexpr (not kWeightStub) {
                        // Issue TMA B (packed FP4 bytes loaded as raw uint8 via FP8 alias) on the same barrier.
                        const uint32_t k_idx_packed = k_block_idx * BLOCK_K_PACKED;
                        tma::copy<BLOCK_K_PACKED, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                                 reinterpret_cast<__nv_fp8_e4m3*>(smem_b_packed[stage_idx]),
                                 k_idx_packed,
                                 scheduler.template get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                                 num_tma_multicast_b, batch_idx);
                    }

                    constexpr uint32_t kExpectedTxBytes = SMEM_A_TMA_SIZE_PER_STAGE +
                                                          (kWeightStub ? 0 : SMEM_B_PACKED_SIZE_PER_STAGE) +
                                                          (kScaleAStub ? 0 : SMEM_SFA_TMA_SIZE_PER_STAGE);
                    full_barrier.arrive_and_expect_tx(kExpectedTxBytes);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; ++ i) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    phase ^= stage_idx == 0;
                }
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        const auto warp_in_wg = warp_idx % 4;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);

        constexpr uint32_t kLdmatrixVecBytes = 16 / sizeof(__nv_fp8_e4m3);

        // Persistently schedule over blocks
        while (get_next_block()) {
            const uint32_t current_group_idx = get_current_group_idx();

            if constexpr (kScaleBGranK == 128) {
                // Cooperatively prefetch the SFB tile for this block from gmem to smem.
                // Layout in smem: [shape_k_scales_b, BLOCK_N] (k outer, n inner).
                // Out-of-bound n is filled with 1.0f to keep `n_idx >= shape_n` neutral.
                //
                // Optimization: use cp.async to copy gmem->smem directly (no register
                // round-trip). For MN-major (n innermost in gmem) we can issue 16-byte
                // (float4) cp.async per thread, cutting #instructions by 4x. K-major
                // and the OOB tail use scalar 4-byte cp.async / st.shared.
                const uint32_t n_block_base = n_block_idx * BLOCK_N;
                if constexpr (kMajorSFB == cute::UMMA::Major::MN) {
                    constexpr uint32_t kVec = 4;
                    DG_STATIC_ASSERT(BLOCK_N % kVec == 0,
                                     "BLOCK_N must be a multiple of 4 for vectorized SFB load");
                    constexpr uint32_t kVecsPerRow = BLOCK_N / kVec;
                    const uint32_t total_vecs = shape_k_scales_b * kVecsPerRow;
                    const float* sfb_base = sfb +
                        current_group_idx * aligned_shape_n_sfb * shape_k_scales_b;
                    // Issue cp.async (16B per thread) for fully in-bounds vectors;
                    // fall back to scalar st.shared with 1.0f-fill for the tail.
                    for (uint32_t i = threadIdx.x; i < total_vecs; i += kNumMathThreads) {
                        const uint32_t k_idx = i / kVecsPerRow;
                        const uint32_t vec_n = i % kVecsPerRow;
                        const uint32_t n_off = vec_n * kVec;
                        const uint32_t n_idx = n_block_base + n_off;
                        float* smem_dst = smem_sfb + k_idx * BLOCK_N + n_off;
                        if (n_idx + kVec <= shape_n) {
                            const float* gmem_src = sfb_base +
                                k_idx * aligned_shape_n_sfb + n_idx;
                            ptx::cp_async_16(smem_dst, gmem_src);
                        } else {
                            // Tail: at least one element is OOB; use scalar st with 1.0f fill.
                            float4 vals;
                            const float* sfb_row = sfb_base + k_idx * aligned_shape_n_sfb + n_idx;
                            vals.x = (n_idx + 0 < shape_n) ? *(sfb_row + 0) : 1.0f;
                            vals.y = (n_idx + 1 < shape_n) ? *(sfb_row + 1) : 1.0f;
                            vals.z = (n_idx + 2 < shape_n) ? *(sfb_row + 2) : 1.0f;
                            vals.w = (n_idx + 3 < shape_n) ? *(sfb_row + 3) : 1.0f;
                            ptx::st_shared(reinterpret_cast<float4*>(smem_dst), vals);
                        }
                    }
                } else {
                    // K-major: sfb is strided along n; cannot easily vectorize across n.
                    // Use scalar 4B cp.async for in-bounds, st.shared with 1.0f for OOB.
                    const uint32_t total = shape_k_scales_b * BLOCK_N;
                    for (uint32_t i = threadIdx.x; i < total; i += kNumMathThreads) {
                        const uint32_t k_idx = i / BLOCK_N;
                        const uint32_t n_off = i % BLOCK_N;
                        const uint32_t n_idx = n_block_base + n_off;
                        float* smem_dst = smem_sfb + k_idx * BLOCK_N + n_off;
                        if (n_idx >= shape_n) {
                            ptx::st_shared(smem_dst, 1.0f);
                        } else {
                            const float* gmem_src = sfb +
                                current_group_idx * shape_n * shape_k_scales_b +
                                n_idx * shape_k_scales_b + k_idx;
                            ptx::cp_async_4(smem_dst, gmem_src);
                        }
                    }
                }
                // Commit and wait for all cp.async issued above; pair with the
                // existing NamedBarrier so the smem cache is visible to all
                // math warps before they enter the K loop.
                ptx::cp_async_commit_group();
                ptx::cp_async_wait_group<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
            }

            auto cache_sfb_k32 = [&](uint32_t k_block_idx) {
                if constexpr (kScaleBGranK == 32 and not kScaleBStub and not kFuseScaleBDecodeStub and not kScaleBDirectLoad) {
                    const uint32_t n_block_base = n_block_idx * BLOCK_N;
                    const uint32_t scale_k_base = k_block_idx * (BLOCK_K / kScaleBGranK);
                    constexpr uint32_t kScaleRows = BLOCK_K / kScaleBGranK;
                    if constexpr (kFuseScaleBDecode) {
                        DG_STATIC_ASSERT(not kScaleBPackedUE8M0 or kMajorSFB == cute::UMMA::Major::MN,
                                         "Packed UE8M0 SFB path expects MN-major transformed scale layout");
                        constexpr uint32_t kVec = 4;
                        DG_STATIC_ASSERT(BLOCK_N % kVec == 0,
                                         "BLOCK_N must be a multiple of 4 for packed K/32 SFB exp cache");
                        constexpr uint32_t kVecsPerRow = BLOCK_N / kVec;
                        const uint32_t total_vecs = kScaleRows * kVecsPerRow;
                        for (uint32_t i = threadIdx.x; i < total_vecs; i += kNumMathThreads) {
                            const uint32_t k_off = i / kVecsPerRow;
                            const uint32_t n_off = (i % kVecsPerRow) * kVec;
                            const uint32_t n_idx = n_block_base + n_off;
                            uint32_t packed_offsets = 0;
                            #pragma unroll
                            for (uint32_t j = 0; j < kVec; ++ j) {
                                uint32_t exp_offset = 6;
                                if constexpr (kScaleBPackedUE8M0) {
                                    if (n_idx + j < shape_n) {
                                        const uint32_t packed_shape_k_scales_b = math::ceil_div(shape_k_scales_b, 4u);
                                        const uint32_t packed_k_idx = (scale_k_base + k_off) / 4u;
                                        const uint32_t byte_idx = (scale_k_base + k_off) % 4u;
                                        const uint32_t* sfb_packed = reinterpret_cast<const uint32_t*>(sfb);
                                        const uint32_t packed_scale = sfb_packed[
                                            current_group_idx * aligned_shape_n_sfb * packed_shape_k_scales_b +
                                            packed_k_idx * aligned_shape_n_sfb + n_idx + j];
                                        const uint32_t e8m0_exp = (packed_scale >> (byte_idx * 8)) & 0xffu;
                                        exp_offset = (e8m0_exp > 121u) ? (e8m0_exp - 121u) : 0u;
                                    }
                                } else {
                                    float val = 1.0f;
                                    if (n_idx + j < shape_n) {
                                        if constexpr (kMajorSFB == cute::UMMA::Major::MN) {
                                            const float* sfb_base = sfb +
                                                current_group_idx * aligned_shape_n_sfb * shape_k_scales_b;
                                            const float* ptr = sfb_base +
                                                (scale_k_base + k_off) * aligned_shape_n_sfb + n_idx + j;
                                            val = *ptr;
                                        } else {
                                            const float* ptr = sfb +
                                                current_group_idx * shape_n * shape_k_scales_b +
                                                (n_idx + j) * shape_k_scales_b + scale_k_base + k_off;
                                            val = *ptr;
                                        }
                                    }
                                    const int32_t exp_shift = fp4_rs_detail::pow2_scale_to_exp_shift(val);
                                    exp_offset = static_cast<uint32_t>(exp_shift + 6) & 0xffu;
                                }
                                packed_offsets |= exp_offset << (j * 8);
                            }
                            ptx::st_shared(smem_sfb_exp + k_off * kVecsPerRow + n_off / kVec, packed_offsets);
                        }
                    } else if constexpr (kMajorSFB == cute::UMMA::Major::MN) {
                        constexpr uint32_t kVec = 4;
                        DG_STATIC_ASSERT(BLOCK_N % kVec == 0,
                                         "BLOCK_N must be a multiple of 4 for vectorized K/32 SFB load");
                        constexpr uint32_t kVecsPerRow = BLOCK_N / kVec;
                        const uint32_t total_vecs = kScaleRows * kVecsPerRow;
                        const float* sfb_base = sfb +
                            current_group_idx * aligned_shape_n_sfb * shape_k_scales_b;
                        for (uint32_t i = threadIdx.x; i < total_vecs; i += kNumMathThreads) {
                            const uint32_t k_off = i / kVecsPerRow;
                            const uint32_t vec_n = i % kVecsPerRow;
                            const uint32_t n_off = vec_n * kVec;
                            const uint32_t n_idx = n_block_base + n_off;
                            float* smem_dst = smem_sfb + k_off * BLOCK_N + n_off;
                            if (n_idx + kVec <= shape_n) {
                                const float* gmem_src = sfb_base +
                                    (scale_k_base + k_off) * aligned_shape_n_sfb + n_idx;
                                ptx::cp_async_16(smem_dst, gmem_src);
                            } else {
                                float4 vals;
                                const float* sfb_row = sfb_base +
                                    (scale_k_base + k_off) * aligned_shape_n_sfb + n_idx;
                                vals.x = (n_idx + 0 < shape_n) ? *(sfb_row + 0) : 1.0f;
                                vals.y = (n_idx + 1 < shape_n) ? *(sfb_row + 1) : 1.0f;
                                vals.z = (n_idx + 2 < shape_n) ? *(sfb_row + 2) : 1.0f;
                                vals.w = (n_idx + 3 < shape_n) ? *(sfb_row + 3) : 1.0f;
                                ptx::st_shared(reinterpret_cast<float4*>(smem_dst), vals);
                            }
                        }
                    } else {
                        const uint32_t total = kScaleRows * BLOCK_N;
                        for (uint32_t i = threadIdx.x; i < total; i += kNumMathThreads) {
                            const uint32_t k_off = i / BLOCK_N;
                            const uint32_t n_off = i % BLOCK_N;
                            const uint32_t n_idx = n_block_base + n_off;
                            float* smem_dst = smem_sfb + k_off * BLOCK_N + n_off;
                            if (n_idx >= shape_n) {
                                ptx::st_shared(smem_dst, 1.0f);
                            } else {
                                const float* gmem_src = sfb +
                                    current_group_idx * shape_n * shape_k_scales_b +
                                    n_idx * shape_k_scales_b + scale_k_base + k_off;
                                ptx::cp_async_4(smem_dst, gmem_src);
                            }
                        }
                    }
                    ptx::cp_async_commit_group();
                    ptx::cp_async_wait_group<0>();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
                }
            };

            auto load_sfb = [&](uint32_t n_idx, uint32_t k_block_idx) {
                if (n_idx >= shape_n)
                    return 1.0f;
                if constexpr (kScaleBDirectLoad and kScaleBGranK == 32) {
                    if constexpr (kMajorSFB == cute::UMMA::Major::MN) {
                        const uint32_t offset =
                            current_group_idx * aligned_shape_n_sfb * shape_k_scales_b +
                            k_block_idx * aligned_shape_n_sfb + n_idx;
                        if constexpr (kScaleBE8M0) {
                            const auto* sfb_u8 = reinterpret_cast<const uint8_t*>(sfb);
                            return __uint_as_float(static_cast<uint32_t>(sfb_u8[offset]) << 23);
                        } else if constexpr (kScaleBBF16) {
                            const auto* sfb_bf16 = reinterpret_cast<const nv_bfloat16*>(sfb);
                            return __bfloat162float(sfb_bf16[offset]);
                        } else {
                            return sfb[offset];
                        }
                    } else {
                        const uint32_t offset = current_group_idx * shape_n * shape_k_scales_b +
                                                n_idx * shape_k_scales_b + k_block_idx;
                        if constexpr (kScaleBE8M0) {
                            const auto* sfb_u8 = reinterpret_cast<const uint8_t*>(sfb);
                            return __uint_as_float(static_cast<uint32_t>(sfb_u8[offset]) << 23);
                        } else if constexpr (kScaleBBF16) {
                            const auto* sfb_bf16 = reinterpret_cast<const nv_bfloat16*>(sfb);
                            return __bfloat162float(sfb_bf16[offset]);
                        } else {
                            return sfb[offset];
                        }
                    }
                } else if constexpr (kScaleBGranK == 32) {
                    const uint32_t n_off = n_idx - n_block_idx * BLOCK_N;
                    const uint32_t k_off = k_block_idx % (BLOCK_K / kScaleBGranK);
                    return ptx::ld_shared(smem_sfb + k_off * BLOCK_N + n_off);
                } else {
                    // SFB has been staged into smem above; out-of-bound `n_idx` already
                    // resolves to 1.0f because we wrote 1.0f for those slots.
                    const uint32_t n_off = n_idx - n_block_idx * BLOCK_N;
                    return ptx::ld_shared(smem_sfb + k_block_idx * BLOCK_N + n_off);
                }
            };

            auto load_sfb_exp_offset = [&](uint32_t n_idx, uint32_t k_block_idx) {
                if (n_idx >= shape_n)
                    return uint32_t(6);
                constexpr uint32_t kVec = 4;
                constexpr uint32_t kVecsPerRow = BLOCK_N / kVec;
                const uint32_t n_off = n_idx - n_block_idx * BLOCK_N;
                const uint32_t k_off = k_block_idx % (BLOCK_K / kScaleBGranK);
                const uint32_t packed_offsets = ptx::ld_shared(smem_sfb_exp + k_off * kVecsPerRow + n_off / kVec);
                return (packed_offsets >> ((n_off % kVec) * 8)) & 0xffu;
            };

            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
            }

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_N <= WGMMA::M ? BLOCK_N : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_N % WAVE_BLOCK_M == 0, "Invalid block sizes");
            constexpr uint32_t WAVE_WGMMA = BLOCK_N / WAVE_BLOCK_M;
            constexpr uint32_t kWGsPerNWave = WAVE_BLOCK_M / WGMMA::M;
            constexpr bool kParallelNWavesEnabled =
                kParallelNWaves and WAVE_WGMMA == 2 and kWGsPerNWave == 2 and kNumMathThreads == 512;
            constexpr bool kUseScaleKGroup = (kScaleKGroup > 1 and (WAVE_WGMMA == 1 or WAVE_WGMMA == 2));
            constexpr bool kUseScaleKGroupExact =
                (kScaleKGroupExact and kUseScaleKGroup and kScaleKGroup == 2 and not kFusedPromote);
            DG_STATIC_ASSERT(kNumMathThreads % 128 == 0, "RS-mode math threads must be whole warpgroups");
            DG_STATIC_ASSERT(not kParallelNWaves or kParallelNWavesEnabled,
                             "DG_W4_PARALLEL_N_WAVES only supports BN256 with 512 math threads");
            const uint32_t wave_group_idx = kParallelNWavesEnabled ? math_wg_idx / kWGsPerNWave : 0;
            const uint32_t wave_mwg_idx = kParallelNWavesEnabled ? math_wg_idx % kWGsPerNWave : math_wg_idx;
            const uint32_t wave_warp_idx = wave_mwg_idx * 4 + warp_in_wg;
            const uint32_t r_0 = wave_warp_idx * 16 + row_idx;
            const uint32_t r_1 = r_0 + 8;
            using final_accum_t = typename fp4_rs_detail::FinalAccumStorage<kBF16FinalAccum>::type;
            constexpr uint32_t kNumAccumSets = kUseScaleKGroup ? WAVE_WGMMA : 1;
            constexpr uint32_t kFinalAccumStride =
                kBF16FinalAccum ? WGMMA::kNumAccum / 2 : WGMMA::kNumAccum;
            constexpr uint32_t kNumFinalAccumRegs = kFinalAccumStride * WAVE_WGMMA;
            constexpr bool kUseFinalAccumScratch =
                kFinalAccumScratch and kDirectStore and kGemmType == GemmType::MGroupedMasked and
                BLOCK_M < WGMMA::M and not kUseScaleKGroup;
            constexpr uint32_t kBaseWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            constexpr uint32_t kNumWGMMAStoreThreads =
                kParallelNWavesEnabled ? kBaseWGMMAStoreThreads * WAVE_WGMMA : kBaseWGMMAStoreThreads;
            constexpr uint32_t kFinalAccumScratchBytes =
                kNumMathThreads * kNumFinalAccumRegs * sizeof(final_accum_t);
            float accum_storage[WGMMA::kNumAccum * kNumAccumSets];
            final_accum_t final_accum_regs[kUseFinalAccumScratch ? 1 : kNumFinalAccumRegs];
            final_accum_t* final_accum = final_accum_regs;
            if constexpr (kUseFinalAccumScratch) {
                DG_STATIC_ASSERT(kFinalAccumScratchBytes <= SMEM_D_SIZE,
                                 "DG_W4_FINAL_ACCUM_SCRATCH needs more smem_d scratch space");
                final_accum = reinterpret_cast<final_accum_t*>(smem_d) +
                    (warp_idx * 32 + lane_idx) * kNumFinalAccumRegs;
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumFinalAccumRegs; ++ i)
                fp4_rs_detail::final_accum_init<kBF16FinalAccum>(final_accum, i);

            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_N >= 64, "RS-mode swap_ab requires at least one 64-row compute tile");
            const bool do_wgmma_store = warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive_stage = [&](uint32_t target_stage) {
                if constexpr (kParallelNWavesEnabled)
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 2);
                if constexpr (kNumTMAMulticast == 1) {
                    (lane_idx == 0 and (not kParallelNWavesEnabled or wave_group_idx == 0)) ?
                        empty_barriers[target_stage]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    (lane_idx < kNumTMAMulticast and (not kParallelNWavesEnabled or wave_group_idx == 0)) ?
                        empty_barriers[target_stage]->arrive(target_cta) : void();
                }
            };
            auto empty_barrier_arrive = [&]() {
                empty_barrier_arrive_stage(stage_idx);
            };

            // Skip useless computations
            const bool is_cta_computation_valid = scheduler.is_computation_valid(m_block_idx, 0);
            if (is_cta_computation_valid) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / math::constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = math::constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                auto wait_stage = [&](uint32_t s, uint32_t p) {
                    full_barriers[s]->wait(p);
                };


                // Dispatch `num_former_iters` and launch MMAs.
                dispatch_num_former_iters_rs<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    constexpr uint32_t kAccumScratchBytes = kNumWGMMAStoreThreads * WGMMA::kNumAccum * sizeof(float);
                    if constexpr (kOverlapPromote and WAVE_WGMMA == 1 and kNumStages >= 2 and
                                  not kWGMMAStub and kAccumScratchBytes <= SMEM_D_SIZE) {
                        auto accum = accum_storage;
                        auto smem_accum = reinterpret_cast<float*>(smem_d) +
                                          (warp_idx * 32 + lane_idx) * WGMMA::kNumAccum;
                        float scale_0_0_regs[WGMMA::kNumAccum / 4];
                        float scale_1_0_regs[WGMMA::kNumAccum / 4];
                        float scale_0_1_regs[WGMMA::kNumAccum / 4];
                        float scale_1_1_regs[WGMMA::kNumAccum / 4];
                        bool prev_valid = false;

                        auto snapshot_accum = [&]() {
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                smem_accum[i] = accum[i];
                            asm volatile("" ::: "memory");
                        };

                        auto promote_snapshot = [&]() {
                            if constexpr (not kPromoteStub) {
                                asm volatile("" ::: "memory");
                                auto shifted_accum = final_accum;
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                    fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                        shifted_accum, i * 2 + 0,
                                        scale_0_0_regs[i], smem_accum[i * 4 + 0],
                                        scale_1_0_regs[i], smem_accum[i * 4 + 1]);
                                    fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                        shifted_accum, i * 2 + 1,
                                        scale_0_1_regs[i], smem_accum[i * 4 + 2],
                                        scale_1_1_regs[i], smem_accum[i * 4 + 3]);
                                }
                            }
                        };

                        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;) {
                            const uint32_t cur_stage = stage_idx;
                            const uint32_t cur_phase = phase;
                            const auto a_desc_base_lo = a_desc_lo + cur_stage * (SMEM_A_SIZE_PER_STAGE / 16);
                            wait_stage(cur_stage, cur_phase);

                            if (not do_wgmma_store) {
                                empty_barrier_arrive_stage(cur_stage);
                                advance_pipeline(k_block_idx);
                                continue;
                            }

                            constexpr uint32_t local_idx = 0;
                            constexpr uint32_t m_offset = 0;
                            const uint32_t lane_row = lane_idx / 4;
                            const uint32_t lane_col_pair = lane_idx % 4;
                            const uint32_t packed_shift = (lane_col_pair & 1u) * 16u;
                            const uint32_t lane_pair_col = (lane_col_pair & 2u) * 2u;
                            uint32_t rs_row_offset[4];
                            uint32_t rs_raw_col[4];
                            uint32_t rs_swizzle_xor[4];
                            #pragma unroll
                            for (uint32_t mat = 0; mat < 4; ++mat) {
                                const uint32_t addr_lane = mat * 8 + lane_row;
                                const uint32_t addr_tid_g = (warp_idx % 4) * 32 + addr_lane;
                                const uint32_t addr_t_row = (addr_tid_g & 15) | ((addr_tid_g >> 5) << 4);
                                const uint32_t addr_t_col = ((addr_tid_g >> 4) & 1) * kLdmatrixVecBytes;
                                const uint32_t src_row = wave_mwg_idx * WGMMA::M + addr_t_row + m_offset;
                                rs_row_offset[mat] = src_row * BLOCK_K_PACKED;
                                rs_raw_col[mat] = addr_t_col / 2 + lane_pair_col;
                                rs_swizzle_xor[mat] = (addr_t_row & 7u) >> 1;
                            }
                            auto load_a_regs = [&](uint32_t k, uint32_t a_regs[4]) {
                                if constexpr (kWeightStub) {
                                    #pragma unroll
                                    for (uint32_t mat = 0; mat < 4; ++mat)
                                        a_regs[mat] = 0x38383838u;
                                    return;
                                }
                                DG_STATIC_ASSERT(BLOCK_K_PACKED == 64 and kLdmatrixVecBytes == 16 and WGMMA::K / 2 == 16,
                                                 "RS decode address fast path assumes 64-byte packed K tile");
                                #pragma unroll
                                for (uint32_t mat = 0; mat < 4; ++mat) {
                                    const uint32_t swizzled_col = ((k ^ rs_swizzle_xor[mat]) << 4) + rs_raw_col[mat];
                                    uint32_t packed_word = 0;
                                    if constexpr (kDecodePairShfl) {
                                        if ((lane_col_pair & 1u) == 0)
                                            packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                                smem_b_packed[cur_stage] + rs_row_offset[mat] + swizzled_col));
                                        packed_word = __shfl_sync(0xffffffff, packed_word, lane_idx & ~1u);
                                    } else {
                                        packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                            smem_b_packed[cur_stage] + rs_row_offset[mat] + swizzled_col));
                                    }
                                    const uint32_t packed_shifted = packed_word >> packed_shift;
                                    if constexpr (kDecodeStub) {
                                        a_regs[mat] = 0x38383838u;
                                    } else {
                                        a_regs[mat] = fp4_rs_detail::fp4x4_to_e4m3x4(packed_shifted);
                                    }
                                }
                            };

                            uint32_t a_regs[4];
                            load_a_regs(0, a_regs);
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                                WGMMA::wgmma(a_regs, a_desc, accum, k);
                                asm volatile("" ::: "memory");
                                if constexpr (BLOCK_K / WGMMA::K > 1) {
                                    if (k + 1 < BLOCK_K / WGMMA::K)
                                        load_a_regs(k + 1, a_regs);
                                }
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);

                            if (prev_valid) {
                                promote_snapshot();
                            }

                            const uint32_t compute_n_0 = n_block_idx * BLOCK_N + r_0;
                            const uint32_t compute_n_1 = n_block_idx * BLOCK_N + r_1;
                            float scale_b_0 = 1.0f;
                            float scale_b_1 = 1.0f;
                            if constexpr (not kScaleBStub) {
                                scale_b_0 = load_sfb(compute_n_0, k_block_idx);
                                scale_b_1 = load_sfb(compute_n_1, k_block_idx);
                            }
                            if constexpr (not kPromoteStub) {
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                    float scale_a_0 = 1.0f;
                                    float scale_a_1 = 1.0f;
                                    if constexpr (not kScaleAStub) {
                                        const uint32_t m_idx = i * 8 + col_idx * 2;
                                        scale_a_0 = ptx::ld_shared(smem_sfa[cur_stage] + m_idx);
                                        scale_a_1 = ptx::ld_shared(smem_sfa[cur_stage] + m_idx + 1);
                                    }
                                    auto scale_product = [&](float scale_a, float scale_b) {
                                        if constexpr (kScaleProductStub) {
                                            fp4_rs_detail::keep_float_live(scale_a + scale_b);
                                            return 1.0f;
                                        } else {
                                            return scale_a * scale_b;
                                        }
                                    };
                                    scale_0_0_regs[i] = scale_product(scale_a_0, scale_b_0);
                                    scale_1_0_regs[i] = scale_product(scale_a_1, scale_b_0);
                                    scale_0_1_regs[i] = scale_product(scale_a_0, scale_b_1);
                                    scale_1_1_regs[i] = scale_product(scale_a_1, scale_b_1);
                                }
                            }

                            ptx::warpgroup_wait<0>();
                            empty_barrier_arrive_stage(cur_stage);
                            snapshot_accum();
                            prev_valid = true;
                            advance_pipeline(k_block_idx);
                        }

                        if (prev_valid) {
                            promote_snapshot();
                        }
                    } else {
                    constexpr uint32_t kScaleSumStride = WGMMA::kNumAccum / 4;
                    float scale_0_0_sum[(kUseScaleKGroup ? WAVE_WGMMA : 1) * kScaleSumStride];
                    float scale_1_0_sum[(kUseScaleKGroup ? WAVE_WGMMA : 1) * kScaleSumStride];
                    float scale_0_1_sum[(kUseScaleKGroup ? WAVE_WGMMA : 1) * kScaleSumStride];
                    float scale_1_1_sum[(kUseScaleKGroup ? WAVE_WGMMA : 1) * kScaleSumStride];
                    float accum_first_storage[kUseScaleKGroupExact ? WAVE_WGMMA * WGMMA::kNumAccum : 1];
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        wait_stage(stage_idx, phase);
                        cache_sfb_k32(k_block_idx);

                        // Small-N/BM=32 experiments may launch more math WGs than
                        // the current B tile has 64-row RS slices. Inactive WGs must
                        // still release the pipeline slot, but must not read B rows
                        // beyond BLOCK_N.
                        if (not do_wgmma_store) {
                            empty_barrier_arrive();
                            continue;
                        }

                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = kParallelNWavesEnabled ? wave_group_idx : 0;
                             local_idx < WAVE_WGMMA;
                             local_idx += kParallelNWavesEnabled ? WAVE_WGMMA : 1) {
                            auto accum = accum_storage + (kUseScaleKGroup ? local_idx * WGMMA::kNumAccum : 0);
                            auto m_offset = local_idx * WAVE_BLOCK_M;
                            const uint32_t scale_group_pos =
                                kUseScaleKGroup ? (k_block_idx % kScaleKGroup) : 0;
                            const bool should_promote_group =
                                (not kUseScaleKGroup) or (scale_group_pos == kScaleKGroup - 1) or
                                (k_block_idx + 1 == num_total_k_blocks);

                            // Read scales before `warpgroup_arrive` so the next CTA cannot pollute shared memory.
                            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                            const uint32_t compute_n_0 = n_block_idx * BLOCK_N + m_offset + r_0;
                            const uint32_t compute_n_1 = n_block_idx * BLOCK_N + m_offset + r_1;
                            float scale_b_0 = 0.0f;
                            float scale_b_1 = 0.0f;
                            if constexpr (kScaleBGranK == 128) {
                                if (do_wgmma_store and (should_promote_group or kUseScaleKGroup)) {
                                    if constexpr (kScaleBStub) {
                                        scale_b_0 = 1.0f;
                                        scale_b_1 = 1.0f;
                                    } else {
                                        scale_b_0 = load_sfb(compute_n_0, k_block_idx);
                                        scale_b_1 = load_sfb(compute_n_1, k_block_idx);
                                    }
                                }
                            }
                            float scale_a_0_regs[kLateScaleA ? 1 : WGMMA::kNumAccum / 4];
                            float scale_a_1_regs[kLateScaleA ? 1 : WGMMA::kNumAccum / 4];
                            if constexpr (kLateScaleA) {
                                // Keep SFA live in shared memory and load it in the promotion loop.
                            } else if constexpr (kScaleAStub) {
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                    scale_a_0_regs[i] = 1.0f;
                                    scale_a_1_regs[i] = 1.0f;
                                }
                            } else if (do_wgmma_store and (should_promote_group or kUseScaleKGroup)) {
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                    const uint32_t m_idx = i * 8 + col_idx * 2;
                                    scale_a_0_regs[i] = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                    scale_a_1_regs[i] = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                }
                            }

                            const uint32_t lane_row = lane_idx / 4;
                            const uint32_t lane_col_pair = lane_idx % 4;
                            const uint32_t packed_shift = (lane_col_pair & 1u) * 16u;
                            const uint32_t lane_pair_col = (lane_col_pair & 2u) * 2u;
                            uint32_t rs_row_offset[4];
                            uint32_t rs_raw_col[4];
                            uint32_t rs_swizzle_xor[4];
                            #pragma unroll
                            for (uint32_t mat = 0; mat < 4; ++mat) {
                                const uint32_t addr_lane = mat * 8 + lane_row;
                                const uint32_t addr_tid_g = (warp_idx % 4) * 32 + addr_lane;
                                const uint32_t addr_t_row = (addr_tid_g & 15) | ((addr_tid_g >> 5) << 4);
                                const uint32_t addr_t_col = ((addr_tid_g >> 4) & 1) * kLdmatrixVecBytes;
                                const uint32_t src_row = wave_mwg_idx * WGMMA::M + addr_t_row + m_offset;
                                rs_row_offset[mat] = src_row * BLOCK_K_PACKED;
                                rs_raw_col[mat] = addr_t_col / 2 + lane_pair_col;
                                rs_swizzle_xor[mat] = (addr_t_row & 7u) >> 1;
                            }
                            auto load_a_regs = [&](uint32_t k, uint32_t a_regs[4]) {
                                if constexpr (kWeightStub) {
                                    #pragma unroll
                                    for (uint32_t mat = 0; mat < 4; ++mat)
                                        a_regs[mat] = 0x38383838u;
                                    return;
                                }
                                DG_STATIC_ASSERT(BLOCK_K_PACKED == 64 and kLdmatrixVecBytes == 16 and WGMMA::K / 2 == 16,
                                                 "RS decode address fast path assumes 64-byte packed K tile");
                                #pragma unroll
                                for (uint32_t mat = 0; mat < 4; ++mat) {
                                    const uint32_t swizzled_col = ((k ^ rs_swizzle_xor[mat]) << 4) + rs_raw_col[mat];
                                    uint32_t packed_word = 0;
                                    if constexpr (kDecodePairShfl) {
                                        if ((lane_col_pair & 1u) == 0)
                                            packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                                smem_b_packed[stage_idx] + rs_row_offset[mat] + swizzled_col));
                                        packed_word = __shfl_sync(0xffffffff, packed_word, lane_idx & ~1u);
                                    } else {
                                        packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                            smem_b_packed[stage_idx] + rs_row_offset[mat] + swizzled_col));
                                    }
                                    const uint32_t packed_shifted = packed_word >> packed_shift;
                                    if constexpr (kDecodeStub) {
                                        // Four E4M3 1.0 values. Keeps the LDS/address path but removes FP4 decode.
                                        a_regs[mat] = 0x38383838u;
                                    } else {
                                        a_regs[mat] = fp4_rs_detail::fp4x4_to_e4m3x4(packed_shifted);
                                    }
                                }
                            };
                            auto load_a_regs_scaled_b = [&](uint32_t k,
                                                            const uint32_t exp_offsets[2],
                                                            const fp4_rs_detail::ScaledE4M3Lut scaled_luts[2],
                                                            uint32_t a_regs[4]) {
                                if constexpr (kWeightStub) {
                                    #pragma unroll
                                    for (uint32_t mat = 0; mat < 4; ++mat)
                                        a_regs[mat] = 0x38383838u;
                                    return;
                                }
                                DG_STATIC_ASSERT(BLOCK_K_PACKED == 64 and kLdmatrixVecBytes == 16 and WGMMA::K / 2 == 16,
                                                 "RS decode address fast path assumes 64-byte packed K tile");
                                auto load_packed_word = [&](uint32_t mat) {
                                    const uint32_t swizzled_col = ((k ^ rs_swizzle_xor[mat]) << 4) + rs_raw_col[mat];
                                    uint32_t packed_word = 0;
                                    if constexpr (kDecodePairShfl) {
                                        if ((lane_col_pair & 1u) == 0)
                                            packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                                smem_b_packed[stage_idx] + rs_row_offset[mat] + swizzled_col));
                                        packed_word = __shfl_sync(0xffffffff, packed_word, lane_idx & ~1u);
                                    } else {
                                        packed_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                            smem_b_packed[stage_idx] + rs_row_offset[mat] + swizzled_col));
                                    }
                                    return packed_word;
                                };
                                if constexpr (kDecodeStub) {
                                    #pragma unroll
                                    for (uint32_t mat = 0; mat < 4; ++mat) {
                                        static_cast<void>(load_packed_word(mat));
                                        a_regs[mat] = 0x38383838u;
                                    }
                                } else if constexpr (kBLoadStub) {
                                    #pragma unroll
                                    for (uint32_t mat = 0; mat < 4; ++mat)
                                        a_regs[mat] = load_packed_word(mat);
                                } else if constexpr (kFuseScaleBDecodeAssumeExp == 6) {
                                    #pragma unroll
                                    for (uint32_t mat = 0; mat < 4; ++mat) {
                                        const uint32_t packed_shifted = load_packed_word(mat) >> packed_shift;
                                        a_regs[mat] = fp4_rs_detail::fp4x4_to_e4m3x4(packed_shifted);
                                    }
                                } else if constexpr (kFuseScaleBDecodeAssumeExp == 5) {
                                    constexpr fp4_rs_detail::ScaledE4M3Lut kExp5Lut{0x34302800u, 0x44403c38u};
                                    #pragma unroll
                                    for (uint32_t mat = 0; mat < 4; ++mat) {
                                        const uint32_t packed_shifted = load_packed_word(mat) >> packed_shift;
                                        a_regs[mat] = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                                            packed_shifted, kExp5Lut);
                                    }
                                } else {
                                    if constexpr (kFuseScaleBDecodeFastCommon) {
                                        const bool uniform_exp = exp_offsets[0] == exp_offsets[1];
                                        if (uniform_exp and exp_offsets[0] == 6u) {
                                            #pragma unroll
                                            for (uint32_t mat = 0; mat < 4; ++mat) {
                                                const uint32_t packed_shifted = load_packed_word(mat) >> packed_shift;
                                                a_regs[mat] = fp4_rs_detail::fp4x4_to_e4m3x4(packed_shifted);
                                            }
                                        } else if (uniform_exp and exp_offsets[0] == 5u) {
                                            constexpr fp4_rs_detail::ScaledE4M3Lut kExp5Lut{0x34302800u, 0x44403c38u};
                                            #pragma unroll
                                            for (uint32_t mat = 0; mat < 4; ++mat) {
                                                const uint32_t packed_shifted = load_packed_word(mat) >> packed_shift;
                                                a_regs[mat] = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                                                    packed_shifted, kExp5Lut);
                                            }
                                        } else {
                                            #pragma unroll
                                            for (uint32_t mat = 0; mat < 4; ++mat) {
                                                const uint32_t packed_shifted = load_packed_word(mat) >> packed_shift;
                                                if constexpr (kFuseScaleBHummingDecode) {
                                                    a_regs[mat] = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(
                                                        packed_shifted, exp_offsets[mat & 1u]);
                                                } else {
                                                    a_regs[mat] = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                                                        packed_shifted, scaled_luts[mat & 1u]);
                                                }
                                            }
                                        }
                                    } else {
                                        #pragma unroll
                                        for (uint32_t mat = 0; mat < 4; ++mat) {
                                            const uint32_t packed_shifted = load_packed_word(mat) >> packed_shift;
                                            if constexpr (kFuseScaleBHummingDecode) {
                                                a_regs[mat] = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_humming(
                                                    packed_shifted, exp_offsets[mat & 1u]);
                                            } else {
                                                a_regs[mat] = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                                                    packed_shifted, scaled_luts[mat & 1u]);
                                            }
                                        }
                                    }
                                }
                            };

                            // Decode the next RS-A fragment immediately after issuing the
                            // current WGMMA so decode/address work can overlap the async MMA.
                            uint32_t a_regs[4];
                            if constexpr (not kFuseScaleBDecode)
                                load_a_regs(0, a_regs);
                            if constexpr (kScaleBGranK == 32) {
                                DG_STATIC_ASSERT(WGMMA::K == 32,
                                                 "per-32 FP4 scale path assumes each WGMMA K slice is one scale group");
                                DG_STATIC_ASSERT(not kUseScaleKGroup,
                                                 "per-32 FP4 scale path does not support DG_W4_SCALE_K_GROUP");
                                auto shifted_accum = final_accum + kFinalAccumStride * local_idx;
                                if constexpr (kFuseScaleBDecode) {
                                    if constexpr (not kWGMMAStub) {
                                        uint32_t scale_b_exp_offsets[BLOCK_K / WGMMA::K][2];
                                        fp4_rs_detail::ScaledE4M3Lut scale_b_luts[kFuseScaleBOnDemandLut ? 1 : (BLOCK_K / WGMMA::K)][2];
                                        #pragma unroll
                                        for (uint32_t kk = 0; kk < BLOCK_K / WGMMA::K; ++ kk) {
                                            const uint32_t scale_b_k_idx =
                                                k_block_idx * (BLOCK_K / kScaleBGranK) + kk;
                                            #pragma unroll
                                            for (uint32_t pair = 0; pair < 2; ++ pair) {
                                                const uint32_t n_idx =
                                                    n_block_idx * BLOCK_N + rs_row_offset[pair] / BLOCK_K_PACKED;
                                                if constexpr (kScaleBStub)
                                                    scale_b_exp_offsets[kk][pair] = 6u;
                                                else
                                                    scale_b_exp_offsets[kk][pair] =
                                                        load_sfb_exp_offset(n_idx, scale_b_k_idx);
                                            }
                                        }
                                        if constexpr (not kFuseScaleBOnDemandLut and not kDecodeStub and not kBLoadStub) {
                                            #pragma unroll
                                            for (uint32_t kk = 0; kk < BLOCK_K / WGMMA::K; ++ kk) {
                                                #pragma unroll
                                                for (uint32_t pair = 0; pair < 2; ++ pair) {
                                                    scale_b_luts[kk][pair] =
                                                        fp4_rs_detail::make_scaled_e4m3_lut(scale_b_exp_offsets[kk][pair]);
                                                }
                                            }
                                        }
                                        uint32_t staged_a_regs[BLOCK_K / WGMMA::K][4];
                                        uint32_t staged_pair_a_regs[2][4];
                                        uint32_t* staged_smem_a_regs = reinterpret_cast<uint32_t*>(smem_d) +
                                            math::ceil_div(smem_sfb_bytes, 128u) * 32u +
                                            (warp_idx * 32 + lane_idx) * (BLOCK_K / WGMMA::K) * 4u;
                                        uint32_t* ws_decoded_regs = reinterpret_cast<uint32_t*>(smem_d) +
                                            math::ceil_div(smem_sfb_bytes, 128u) * 32u;
                                        auto load_a_regs_scaled_b_for_k = [&](uint32_t kk, uint32_t regs[4]) {
                                            if constexpr (kFuseScaleBOnDemandLut) {
                                                fp4_rs_detail::ScaledE4M3Lut on_demand_luts[2];
                                                #pragma unroll
                                                for (uint32_t pair = 0; pair < 2; ++ pair)
                                                    on_demand_luts[pair] =
                                                        fp4_rs_detail::make_scaled_e4m3_lut(scale_b_exp_offsets[kk][pair]);
                                                load_a_regs_scaled_b(kk, scale_b_exp_offsets[kk], on_demand_luts, regs);
                                            } else {
                                                load_a_regs_scaled_b(kk, scale_b_exp_offsets[kk], scale_b_luts[kk], regs);
                                            }
                                        };
                                        if constexpr (kFuseScaleBSlicePromote) {
                                            DG_STATIC_ASSERT(not kFuseScaleBPredecode and not kFuseScaleBPredecodePair and
                                                             not kFuseScaleBSharedStage and not kFuseScaleBWSDecode,
                                                             "slice-promote is only for the direct fused decode path");
                                            DG_STATIC_ASSERT(not kPromoteFromSmem,
                                                             "slice-promote does not use promote-from-smem");
                                            #pragma unroll
                                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                                load_a_regs_scaled_b_for_k(k, a_regs);
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                                    accum[i] = 0.0f;
                                                if constexpr (not kWGMMAStub) {
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                    ptx::warpgroup_arrive();
                                                    a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                                                    WGMMA::wgmma(a_regs, a_desc, accum, false);
                                                    asm volatile("" ::: "memory");
                                                    ptx::warpgroup_commit_batch();
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                    ptx::warpgroup_wait<0>();
                                                }
                                                if constexpr (not kPromoteStub) {
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                                        float scale_a_0 = 1.0f;
                                                        float scale_a_1 = 1.0f;
                                                        if constexpr (not kScaleBEarlyProduct) {
                                                            if constexpr (kLateScaleA) {
                                                                if constexpr (not kScaleAStub) {
                                                                    const uint32_t m_idx = i * 8 + col_idx * 2;
                                                                    scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                                    scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                                }
                                                            } else {
                                                                scale_a_0 = scale_a_0_regs[i];
                                                                scale_a_1 = scale_a_1_regs[i];
                                                            }
                                                        }
                                                        fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 0,
                                                            scale_a_0, accum[i * 4 + 0],
                                                            scale_a_1, accum[i * 4 + 1]);
                                                        fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 1,
                                                            scale_a_0, accum[i * 4 + 2],
                                                            scale_a_1, accum[i * 4 + 3]);
                                                    }
                                                }
                                            }
                                        } else {
                                        if constexpr (kFuseScaleBWSDecode) {
                                            DG_STATIC_ASSERT(BLOCK_N == 256 and kNumMathThreads == 256,
                                                             "DG_W4_FUSE_SCALE_B_WS_DECODE only targets BN256/256-thread experiments");
                                            DG_STATIC_ASSERT(SMEM_D_SIZE >= 1024 + kNumMathThreads *
                                                             (BLOCK_K / WGMMA::K) * 4 * sizeof(uint32_t),
                                                             "smem_d is too small for WS decoded FP4 staging");
                                            if (warp_idx == 0) {
                                                #pragma unroll
                                                for (uint32_t target_tid = lane_idx; target_tid < kNumMathThreads; target_tid += 32) {
                                                    const uint32_t target_warp_idx = target_tid / 32;
                                                    const uint32_t target_lane_idx = target_tid % 32;
                                                    const uint32_t target_math_wg_idx = target_tid / 128;
                                                    const uint32_t target_warp_in_wg = target_warp_idx % 4;
                                                    const uint32_t target_wave_mwg_idx =
                                                        kParallelNWavesEnabled ? target_math_wg_idx % kWGsPerNWave : target_math_wg_idx;
                                                    const uint32_t target_lane_row = target_lane_idx / 4;
                                                    const uint32_t target_lane_col_pair = target_lane_idx % 4;
                                                    const uint32_t target_packed_shift = (target_lane_col_pair & 1u) * 16u;
                                                    const uint32_t target_lane_pair_col = (target_lane_col_pair & 2u) * 2u;
                                                    uint32_t target_rs_row_offset[4];
                                                    uint32_t target_rs_raw_col[4];
                                                    uint32_t target_rs_swizzle_xor[4];
                                                    #pragma unroll
                                                    for (uint32_t mat = 0; mat < 4; ++mat) {
                                                        const uint32_t addr_lane = mat * 8 + target_lane_row;
                                                        const uint32_t addr_tid_g = (target_warp_idx % 4) * 32 + addr_lane;
                                                        const uint32_t addr_t_row = (addr_tid_g & 15) | ((addr_tid_g >> 5) << 4);
                                                        const uint32_t addr_t_col = ((addr_tid_g >> 4) & 1) * kLdmatrixVecBytes;
                                                        const uint32_t src_row =
                                                            target_wave_mwg_idx * WGMMA::M + addr_t_row + m_offset;
                                                        target_rs_row_offset[mat] = src_row * BLOCK_K_PACKED;
                                                        target_rs_raw_col[mat] = addr_t_col / 2 + target_lane_pair_col;
                                                        target_rs_swizzle_xor[mat] = (addr_t_row & 7u) >> 1;
                                                    }
                                                    #pragma unroll
                                                    for (uint32_t kk = 0; kk < BLOCK_K / WGMMA::K; ++ kk) {
                                                        uint32_t target_exp_offsets[2];
                                                        fp4_rs_detail::ScaledE4M3Lut target_luts[2];
                                                        const uint32_t scale_b_k_idx =
                                                            k_block_idx * (BLOCK_K / kScaleBGranK) + kk;
                                                        #pragma unroll
                                                        for (uint32_t pair = 0; pair < 2; ++ pair) {
                                                            const uint32_t n_idx =
                                                                n_block_idx * BLOCK_N +
                                                                target_rs_row_offset[pair] / BLOCK_K_PACKED;
                                                            if constexpr (kScaleBStub)
                                                                target_exp_offsets[pair] = 6u;
                                                            else
                                                                target_exp_offsets[pair] =
                                                                    load_sfb_exp_offset(n_idx, scale_b_k_idx);
                                                            target_luts[pair] =
                                                                fp4_rs_detail::make_scaled_e4m3_lut(target_exp_offsets[pair]);
                                                        }
                                                        #pragma unroll
                                                        for (uint32_t mat = 0; mat < 4; ++ mat) {
                                                            const uint32_t swizzled_col =
                                                                ((kk ^ target_rs_swizzle_xor[mat]) << 4) +
                                                                target_rs_raw_col[mat];
                                                            const uint32_t packed_word = ptx::ld_shared(
                                                                reinterpret_cast<const uint32_t*>(
                                                                    smem_b_packed[stage_idx] +
                                                                    target_rs_row_offset[mat] + swizzled_col));
                                                            const uint32_t packed_shifted = packed_word >> target_packed_shift;
                                                            uint32_t decoded = 0x38383838u;
                                                            if constexpr (not kDecodeStub) {
                                                                decoded = fp4_rs_detail::fp4x4_to_scaled_e4m3x4_lut(
                                                                    packed_shifted, target_luts[mat & 1u]);
                                                            }
                                                            ptx::st_shared(ws_decoded_regs +
                                                                            target_tid * (BLOCK_K / WGMMA::K) * 4u +
                                                                            kk * 4u + mat,
                                                                            decoded);
                                                        }
                                                    }
                                                }
                                            }
                                            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);
                                        } else if constexpr (kFuseScaleBSharedStage) {
                                            DG_STATIC_ASSERT(BLOCK_N == 256,
                                                             "DG_W4_FUSE_SCALE_B_SHARED_STAGE only targets BN256 experiments");
                                            DG_STATIC_ASSERT(SMEM_D_SIZE >= 1024 + kNumMathThreads *
                                                             (BLOCK_K / WGMMA::K) * 4 * sizeof(uint32_t),
                                                             "smem_d is too small for shared decoded FP4 staging");
                                            uint32_t tmp_a_regs[4];
                                            #pragma unroll
                                            for (uint32_t kk = 0; kk < BLOCK_K / WGMMA::K; ++ kk) {
                                                load_a_regs_scaled_b_for_k(kk, tmp_a_regs);
                                                #pragma unroll
                                                for (uint32_t mat = 0; mat < 4; ++ mat) {
                                                    ptx::st_shared(staged_smem_a_regs + kk * 4 + mat, tmp_a_regs[mat]);
                                                }
                                            }
                                        } else if constexpr (kFuseScaleBPredecode) {
                                            #pragma unroll
                                            for (uint32_t kk = 0; kk < BLOCK_K / WGMMA::K; ++ kk) {
                                                load_a_regs_scaled_b_for_k(kk, staged_a_regs[kk]);
                                            }
                                        } else if constexpr (kFuseScaleBPredecodePair) {
                                            DG_STATIC_ASSERT(BLOCK_K / WGMMA::K == 4,
                                                             "DG_W4_FUSE_SCALE_B_PREDECODE_PAIR assumes four K=32 slices");
                                            load_a_regs_scaled_b_for_k(0, staged_pair_a_regs[0]);
                                            load_a_regs_scaled_b_for_k(1, staged_pair_a_regs[1]);
                                        }
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                            ptx::warpgroup_fence_operand(accum[i]);
                                        ptx::warpgroup_arrive();
                                        if constexpr (not kFuseScaleBPredecode and not kFuseScaleBPredecodePair and
                                                      not kFuseScaleBSharedStage and not kFuseScaleBWSDecode) {
                                            load_a_regs_scaled_b_for_k(0, a_regs);
                                        }
                                        #pragma unroll
                                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                            a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                                            if constexpr (kFuseScaleBWSDecode) {
                                                uint32_t ws_a_regs[4];
                                                #pragma unroll
                                                for (uint32_t mat = 0; mat < 4; ++ mat) {
                                                    ws_a_regs[mat] = ptx::ld_shared(
                                                        ws_decoded_regs + threadIdx.x * (BLOCK_K / WGMMA::K) * 4u +
                                                        k * 4u + mat);
                                                }
                                                WGMMA::wgmma(ws_a_regs, a_desc, accum, static_cast<bool>(k));
                                            } else if constexpr (kFuseScaleBSharedStage) {
                                                uint32_t smem_a_regs[4];
                                                #pragma unroll
                                                for (uint32_t mat = 0; mat < 4; ++ mat)
                                                    smem_a_regs[mat] = ptx::ld_shared(staged_smem_a_regs + k * 4 + mat);
                                                WGMMA::wgmma(smem_a_regs, a_desc, accum, static_cast<bool>(k));
                                            } else if constexpr (kFuseScaleBPredecode)
                                                WGMMA::wgmma(staged_a_regs[k], a_desc, accum, static_cast<bool>(k));
                                            else if constexpr (kFuseScaleBPredecodePair)
                                                WGMMA::wgmma(staged_pair_a_regs[k & 1u], a_desc, accum,
                                                            static_cast<bool>(k));
                                            else
                                                WGMMA::wgmma(a_regs, a_desc, accum, static_cast<bool>(k));
                                            asm volatile("" ::: "memory");
                                            if constexpr (kFuseScaleBPredecodePair) {
                                                if (k + 2 < BLOCK_K / WGMMA::K) {
                                                    load_a_regs_scaled_b_for_k(k + 2, staged_pair_a_regs[k & 1u]);
                                                }
                                            } else if constexpr (not kFuseScaleBPredecode) {
                                                if constexpr (BLOCK_K / WGMMA::K > 1) {
                                                    if (k + 1 < BLOCK_K / WGMMA::K) {
                                                        load_a_regs_scaled_b_for_k(k + 1, a_regs);
                                                    }
                                                }
                                            }
                                        }
                                        ptx::warpgroup_commit_batch();
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                            ptx::warpgroup_fence_operand(accum[i]);
                                        ptx::warpgroup_wait<0>();
                                    }
                                    }
                                    if constexpr (kWGMMAStub) {
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                            accum[i] = 0.0f;
                                    }

                                    if constexpr (not kPromoteStub) {
                                        float* smem_promote_accum = reinterpret_cast<float*>(smem_d) +
                                            (warp_idx * 32 + lane_idx) * WGMMA::kNumAccum;
                                        if constexpr (kPromoteFromSmem) {
                                            DG_STATIC_ASSERT(BLOCK_N == 256 and kNumMathThreads == 256,
                                                             "DG_W4_PROMOTE_FROM_SMEM only targets BN256/256-thread experiments");
                                            DG_STATIC_ASSERT(kNumMathThreads * WGMMA::kNumAccum * sizeof(float) <= SMEM_D_SIZE,
                                                             "smem_d is too small for promote-from-smem accum scratch");
                                            #pragma unroll
                                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                                ptx::st_shared(smem_promote_accum + i, accum[i]);
                                            asm volatile("" ::: "memory");
                                        }
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                            float scale_a_0 = 1.0f;
                                            float scale_a_1 = 1.0f;
                                            if constexpr (kLateScaleA) {
                                                if constexpr (not kScaleAStub) {
                                                    const uint32_t m_idx = i * 8 + col_idx * 2;
                                                    scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                    scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                }
                                            } else {
                                                scale_a_0 = scale_a_0_regs[i];
                                                scale_a_1 = scale_a_1_regs[i];
                                            }
                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                shifted_accum, i * 2 + 0,
                                                scale_a_0, kPromoteFromSmem ? ptx::ld_shared(smem_promote_accum + i * 4 + 0) : accum[i * 4 + 0],
                                                scale_a_1, kPromoteFromSmem ? ptx::ld_shared(smem_promote_accum + i * 4 + 1) : accum[i * 4 + 1]);
                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                shifted_accum, i * 2 + 1,
                                                scale_a_0, kPromoteFromSmem ? ptx::ld_shared(smem_promote_accum + i * 4 + 2) : accum[i * 4 + 2],
                                                scale_a_1, kPromoteFromSmem ? ptx::ld_shared(smem_promote_accum + i * 4 + 3) : accum[i * 4 + 3]);
                                        }
                                    }
                                } else if constexpr (kFuseScaleBDecodeStub) {
                                    if constexpr (not kWGMMAStub) {
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                            ptx::warpgroup_fence_operand(accum[i]);
                                        ptx::warpgroup_arrive();
                                        #pragma unroll
                                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                            a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                                            WGMMA::wgmma(a_regs, a_desc, accum, static_cast<bool>(k));
                                            asm volatile("" ::: "memory");
                                            if constexpr (BLOCK_K / WGMMA::K > 1) {
                                                if (k + 1 < BLOCK_K / WGMMA::K)
                                                    load_a_regs(k + 1, a_regs);
                                            }
                                        }
                                        ptx::warpgroup_commit_batch();
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                            ptx::warpgroup_fence_operand(accum[i]);
                                        ptx::warpgroup_wait<0>();
                                    } else {
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                            accum[i] = 0.0f;
                                    }

                                    if constexpr (not kPromoteStub) {
                                        #pragma unroll
                                        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                            float scale_a_0 = 1.0f;
                                            float scale_a_1 = 1.0f;
                                            if constexpr (kLateScaleA) {
                                                if constexpr (not kScaleAStub) {
                                                    const uint32_t m_idx = i * 8 + col_idx * 2;
                                                    scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                    scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                }
                                            } else {
                                                scale_a_0 = scale_a_0_regs[i];
                                                scale_a_1 = scale_a_1_regs[i];
                                            }
                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                shifted_accum, i * 2 + 0,
                                                scale_a_0, accum[i * 4 + 0],
                                                scale_a_1, accum[i * 4 + 1]);
                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                shifted_accum, i * 2 + 1,
                                                scale_a_0, accum[i * 4 + 2],
                                                scale_a_1, accum[i * 4 + 3]);
                                        }
                                    }
                                } else {
                                    if constexpr (kK32Pingpong) {
                                        DG_STATIC_ASSERT(BLOCK_K / WGMMA::K == 4,
                                                         "DG_W4_K32_PINGPONG currently assumes four K=32 slices");
                                        float accum_alt[WGMMA::kNumAccum];
                                        auto promote_k32_slice = [&](float* slice_accum, uint32_t k_slice) {
                                            if constexpr (not kPromoteStub) {
                                                float slice_scale_b_0 = 1.0f;
                                                float slice_scale_b_1 = 1.0f;
                                                if constexpr (not kScaleBStub) {
                                                    const uint32_t scale_b_k_idx =
                                                        k_block_idx * (BLOCK_K / kScaleBGranK) + k_slice;
                                                    slice_scale_b_0 = load_sfb(compute_n_0, scale_b_k_idx);
                                                    slice_scale_b_1 = load_sfb(compute_n_1, scale_b_k_idx);
                                                }
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                                    float scale_a_0 = 1.0f;
                                                    float scale_a_1 = 1.0f;
                                                    if constexpr (kLateScaleA) {
                                                        if constexpr (not kScaleAStub) {
                                                            const uint32_t m_idx = i * 8 + col_idx * 2;
                                                            scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                            scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                        }
                                                    } else {
                                                        scale_a_0 = scale_a_0_regs[i];
                                                        scale_a_1 = scale_a_1_regs[i];
                                                    }
                                                if constexpr (kScaleBMulStub) {
                                                    fp4_rs_detail::keep_float_live(slice_scale_b_0);
                                                    fp4_rs_detail::keep_float_live(slice_scale_b_1);
                                                }
                                                auto scale_prod = [&](float scale_a, float scale_b) {
                                                    if constexpr (kScaleProductStub) {
                                                        fp4_rs_detail::keep_float_live(scale_a + scale_b);
                                                        return 1.0f;
                                                    } else if constexpr (kScaleBMulStub or kPromoteMulStub) {
                                                        return scale_a;
                                                    } else {
                                                        return kScaleBPow2Promote ?
                                                            fp4_rs_detail::scale_float_by_pow2(scale_a, scale_b) :
                                                            scale_a * scale_b;
                                                    }
                                                };
                                                const float prod_0_0 = scale_prod(scale_a_0, slice_scale_b_0);
                                                const float prod_1_0 = scale_prod(scale_a_1, slice_scale_b_0);
                                                const float prod_0_1 = scale_prod(scale_a_0, slice_scale_b_1);
                                                const float prod_1_1 = scale_prod(scale_a_1, slice_scale_b_1);
                                                if constexpr (kPromoteAccumStub or kPromoteFinalAccumStub) {
                                                    fp4_rs_detail::keep_float_live(prod_0_0 + slice_accum[i * 4 + 0]);
                                                    fp4_rs_detail::keep_float_live(prod_1_0 + slice_accum[i * 4 + 1]);
                                                    fp4_rs_detail::keep_float_live(prod_0_1 + slice_accum[i * 4 + 2]);
                                                    fp4_rs_detail::keep_float_live(prod_1_1 + slice_accum[i * 4 + 3]);
                                                } else if constexpr (kPromoteMulStub) {
                                                    fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                        shifted_accum, i * 2 + 0,
                                                        1.0f, slice_accum[i * 4 + 0],
                                                        1.0f, slice_accum[i * 4 + 1]);
                                                    fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                        shifted_accum, i * 2 + 1,
                                                        1.0f, slice_accum[i * 4 + 2],
                                                        1.0f, slice_accum[i * 4 + 3]);
                                                } else {
                                                    fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                        shifted_accum, i * 2 + 0,
                                                        prod_0_0, slice_accum[i * 4 + 0],
                                                        prod_1_0, slice_accum[i * 4 + 1]);
                                                    fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                        shifted_accum, i * 2 + 1,
                                                        prod_0_1, slice_accum[i * 4 + 2],
                                                        prod_1_1, slice_accum[i * 4 + 3]);
                                                }
                                                }
                                            }
                                        };
                                        auto promote_k32_pair = [&](float* slice_accum_0, float* slice_accum_1,
                                                                    uint32_t k_slice_0, uint32_t k_slice_1) {
                                            if constexpr (not kPromoteStub) {
                                                float slice0_scale_b_0 = 1.0f;
                                                float slice0_scale_b_1 = 1.0f;
                                                float slice1_scale_b_0 = 1.0f;
                                                float slice1_scale_b_1 = 1.0f;
                                                if constexpr (not kScaleBStub) {
                                                    const uint32_t scale_b_k_idx_0 =
                                                        k_block_idx * (BLOCK_K / kScaleBGranK) + k_slice_0;
                                                    const uint32_t scale_b_k_idx_1 =
                                                        k_block_idx * (BLOCK_K / kScaleBGranK) + k_slice_1;
                                                    slice0_scale_b_0 = load_sfb(compute_n_0, scale_b_k_idx_0);
                                                    slice0_scale_b_1 = load_sfb(compute_n_1, scale_b_k_idx_0);
                                                    slice1_scale_b_0 = load_sfb(compute_n_0, scale_b_k_idx_1);
                                                    slice1_scale_b_1 = load_sfb(compute_n_1, scale_b_k_idx_1);
                                                }
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                                    float scale_a_0 = 1.0f;
                                                    float scale_a_1 = 1.0f;
                                                    if constexpr (kLateScaleA) {
                                                        if constexpr (not kScaleAStub) {
                                                            const uint32_t m_idx = i * 8 + col_idx * 2;
                                                            scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                            scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                        }
                                                    } else {
                                                        scale_a_0 = scale_a_0_regs[i];
                                                        scale_a_1 = scale_a_1_regs[i];
                                                    }
                                                    if constexpr (kScaleBMulStub) {
                                                        fp4_rs_detail::keep_float_live(slice0_scale_b_0);
                                                        fp4_rs_detail::keep_float_live(slice0_scale_b_1);
                                                        fp4_rs_detail::keep_float_live(slice1_scale_b_0);
                                                        fp4_rs_detail::keep_float_live(slice1_scale_b_1);
                                                    }
                                                    auto scale_prod = [&](float scale_a, float scale_b) {
                                                        if constexpr (kScaleProductStub) {
                                                            fp4_rs_detail::keep_float_live(scale_a + scale_b);
                                                            return 1.0f;
                                                        } else if constexpr (kScaleBMulStub or kPromoteMulStub) {
                                                            return scale_a;
                                                        } else {
                                                            return kScaleBPow2Promote ?
                                                                fp4_rs_detail::scale_float_by_pow2(scale_a, scale_b) :
                                                                scale_a * scale_b;
                                                        }
                                                    };
                                                    const float prod0_0_0 = scale_prod(scale_a_0, slice0_scale_b_0);
                                                    const float prod0_1_0 = scale_prod(scale_a_1, slice0_scale_b_0);
                                                    const float prod0_0_1 = scale_prod(scale_a_0, slice0_scale_b_1);
                                                    const float prod0_1_1 = scale_prod(scale_a_1, slice0_scale_b_1);
                                                    const float prod1_0_0 = scale_prod(scale_a_0, slice1_scale_b_0);
                                                    const float prod1_1_0 = scale_prod(scale_a_1, slice1_scale_b_0);
                                                    const float prod1_0_1 = scale_prod(scale_a_0, slice1_scale_b_1);
                                                    const float prod1_1_1 = scale_prod(scale_a_1, slice1_scale_b_1);
                                                    if constexpr (kPromoteAccumStub or kPromoteFinalAccumStub) {
                                                        fp4_rs_detail::keep_float_live(prod0_0_0 + slice_accum_0[i * 4 + 0]);
                                                        fp4_rs_detail::keep_float_live(prod0_1_0 + slice_accum_0[i * 4 + 1]);
                                                        fp4_rs_detail::keep_float_live(prod0_0_1 + slice_accum_0[i * 4 + 2]);
                                                        fp4_rs_detail::keep_float_live(prod0_1_1 + slice_accum_0[i * 4 + 3]);
                                                        fp4_rs_detail::keep_float_live(prod1_0_0 + slice_accum_1[i * 4 + 0]);
                                                        fp4_rs_detail::keep_float_live(prod1_1_0 + slice_accum_1[i * 4 + 1]);
                                                        fp4_rs_detail::keep_float_live(prod1_0_1 + slice_accum_1[i * 4 + 2]);
                                                        fp4_rs_detail::keep_float_live(prod1_1_1 + slice_accum_1[i * 4 + 3]);
                                                    } else if constexpr (kPromoteMulStub) {
                                                        fp4_rs_detail::final_accum_promote_pair2<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 0,
                                                            1.0f, slice_accum_0[i * 4 + 0],
                                                            1.0f, slice_accum_0[i * 4 + 1],
                                                            1.0f, slice_accum_1[i * 4 + 0],
                                                            1.0f, slice_accum_1[i * 4 + 1]);
                                                        fp4_rs_detail::final_accum_promote_pair2<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 1,
                                                            1.0f, slice_accum_0[i * 4 + 2],
                                                            1.0f, slice_accum_0[i * 4 + 3],
                                                            1.0f, slice_accum_1[i * 4 + 2],
                                                            1.0f, slice_accum_1[i * 4 + 3]);
                                                    } else {
                                                        fp4_rs_detail::final_accum_promote_pair2<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 0,
                                                            prod0_0_0, slice_accum_0[i * 4 + 0],
                                                            prod0_1_0, slice_accum_0[i * 4 + 1],
                                                            prod1_0_0, slice_accum_1[i * 4 + 0],
                                                            prod1_1_0, slice_accum_1[i * 4 + 1]);
                                                        fp4_rs_detail::final_accum_promote_pair2<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 1,
                                                            prod0_0_1, slice_accum_0[i * 4 + 2],
                                                            prod0_1_1, slice_accum_0[i * 4 + 3],
                                                            prod1_0_1, slice_accum_1[i * 4 + 2],
                                                            prod1_1_1, slice_accum_1[i * 4 + 3]);
                                                    }
                                                }
                                            }
                                        };
                                        auto load_k32_quad_scale_b = [&](float* scale_b_0, float* scale_b_1) {
                                            #pragma unroll
                                            for (uint32_t kk = 0; kk < 4; ++ kk) {
                                                scale_b_0[kk] = 1.0f;
                                                scale_b_1[kk] = 1.0f;
                                            }
                                            if constexpr (not kScaleBStub) {
                                                if constexpr ((kK32QuadScaleBInline or kK32QuadScaleBVec4) and
                                                              kScaleBDirectLoad and kScaleBGranK == 32) {
                                                    const uint32_t scale_b_k_base =
                                                        k_block_idx * (BLOCK_K / kScaleBGranK);
                                                    if constexpr (kMajorSFB == cute::UMMA::Major::MN) {
                                                        const float* sfb_base = sfb +
                                                            current_group_idx * aligned_shape_n_sfb * shape_k_scales_b +
                                                            scale_b_k_base * aligned_shape_n_sfb;
                                                        #pragma unroll
                                                        for (uint32_t kk = 0; kk < 4; ++ kk) {
                                                            scale_b_0[kk] = compute_n_0 < shape_n ?
                                                                *(sfb_base + kk * aligned_shape_n_sfb + compute_n_0) : 1.0f;
                                                            scale_b_1[kk] = compute_n_1 < shape_n ?
                                                                *(sfb_base + kk * aligned_shape_n_sfb + compute_n_1) : 1.0f;
                                                        }
                                                    } else {
                                                        const float* sfb_base = sfb +
                                                            current_group_idx * shape_n * shape_k_scales_b + scale_b_k_base;
                                                        if constexpr (kK32QuadScaleBVec4) {
                                                            if (compute_n_0 < shape_n) {
                                                                const float4 v0 = *reinterpret_cast<const float4*>(
                                                                    sfb_base + compute_n_0 * shape_k_scales_b);
                                                                scale_b_0[0] = v0.x;
                                                                scale_b_0[1] = v0.y;
                                                                scale_b_0[2] = v0.z;
                                                                scale_b_0[3] = v0.w;
                                                            }
                                                            if (compute_n_1 < shape_n) {
                                                                const float4 v1 = *reinterpret_cast<const float4*>(
                                                                    sfb_base + compute_n_1 * shape_k_scales_b);
                                                                scale_b_1[0] = v1.x;
                                                                scale_b_1[1] = v1.y;
                                                                scale_b_1[2] = v1.z;
                                                                scale_b_1[3] = v1.w;
                                                            }
                                                        } else {
                                                            #pragma unroll
                                                            for (uint32_t kk = 0; kk < 4; ++ kk) {
                                                                scale_b_0[kk] = compute_n_0 < shape_n ?
                                                                    *(sfb_base + compute_n_0 * shape_k_scales_b + kk) : 1.0f;
                                                                scale_b_1[kk] = compute_n_1 < shape_n ?
                                                                    *(sfb_base + compute_n_1 * shape_k_scales_b + kk) : 1.0f;
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    #pragma unroll
                                                    for (uint32_t kk = 0; kk < 4; ++ kk) {
                                                        const uint32_t scale_b_k_idx =
                                                            k_block_idx * (BLOCK_K / kScaleBGranK) + kk;
                                                        scale_b_0[kk] = load_sfb(compute_n_0, scale_b_k_idx);
                                                        scale_b_1[kk] = load_sfb(compute_n_1, scale_b_k_idx);
                                                    }
                                                }
                                            }
                                        };
                                        auto promote_k32_quad = [&](float* accum_0, float* accum_1,
                                                                    float* accum_2, float* accum_3,
                                                                    float* scale_b_0, float* scale_b_1) {
                                            if constexpr (not kPromoteStub) {
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                                    float scale_a_0 = 1.0f;
                                                    float scale_a_1 = 1.0f;
                                                    if constexpr (kLateScaleA) {
                                                        if constexpr (not kScaleAStub) {
                                                            const uint32_t m_idx = i * 8 + col_idx * 2;
                                                            scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                            scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                        }
                                                    } else {
                                                        scale_a_0 = scale_a_0_regs[i];
                                                        scale_a_1 = scale_a_1_regs[i];
                                                    }
                                                    if constexpr (kScaleBMulStub) {
                                                        #pragma unroll
                                                        for (uint32_t kk = 0; kk < 4; ++ kk) {
                                                            fp4_rs_detail::keep_float_live(scale_b_0[kk]);
                                                            fp4_rs_detail::keep_float_live(scale_b_1[kk]);
                                                        }
                                                    }
                                                    auto scale_prod = [&](float scale_a, float scale_b) {
                                                        if constexpr (kScaleProductStub) {
                                                            fp4_rs_detail::keep_float_live(scale_a + scale_b);
                                                            return 1.0f;
                                                        } else if constexpr (kScaleBMulStub or kPromoteMulStub) {
                                                            return scale_a;
                                                        } else {
                                                            return kScaleBPow2Promote ?
                                                                fp4_rs_detail::scale_float_by_pow2(scale_a, scale_b) :
                                                                scale_a * scale_b;
                                                        }
                                                    };
                                                    if constexpr (kPromoteAccumStub or kPromoteFinalAccumStub) {
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_0[0]) + accum_0[i * 4 + 0]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_0[0]) + accum_0[i * 4 + 1]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_1[0]) + accum_0[i * 4 + 2]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_1[0]) + accum_0[i * 4 + 3]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_0[1]) + accum_1[i * 4 + 0]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_0[1]) + accum_1[i * 4 + 1]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_1[1]) + accum_1[i * 4 + 2]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_1[1]) + accum_1[i * 4 + 3]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_0[2]) + accum_2[i * 4 + 0]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_0[2]) + accum_2[i * 4 + 1]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_1[2]) + accum_2[i * 4 + 2]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_1[2]) + accum_2[i * 4 + 3]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_0[3]) + accum_3[i * 4 + 0]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_0[3]) + accum_3[i * 4 + 1]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_0, scale_b_1[3]) + accum_3[i * 4 + 2]);
                                                        fp4_rs_detail::keep_float_live(scale_prod(scale_a_1, scale_b_1[3]) + accum_3[i * 4 + 3]);
                                                    } else if constexpr (kPromoteMulStub) {
                                                        fp4_rs_detail::final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 0,
                                                            1.0f, accum_0[i * 4 + 0], 1.0f, accum_0[i * 4 + 1],
                                                            1.0f, accum_1[i * 4 + 0], 1.0f, accum_1[i * 4 + 1],
                                                            1.0f, accum_2[i * 4 + 0], 1.0f, accum_2[i * 4 + 1],
                                                            1.0f, accum_3[i * 4 + 0], 1.0f, accum_3[i * 4 + 1]);
                                                        fp4_rs_detail::final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 1,
                                                            1.0f, accum_0[i * 4 + 2], 1.0f, accum_0[i * 4 + 3],
                                                            1.0f, accum_1[i * 4 + 2], 1.0f, accum_1[i * 4 + 3],
                                                            1.0f, accum_2[i * 4 + 2], 1.0f, accum_2[i * 4 + 3],
                                                            1.0f, accum_3[i * 4 + 2], 1.0f, accum_3[i * 4 + 3]);
                                                    } else {
                                                        if constexpr (kK32QuadPersistentScaleProduct) {
                                                            float prod_0_0[4], prod_1_0[4];
                                                            float prod_0_1[4], prod_1_1[4];
                                                            #pragma unroll
                                                            for (uint32_t kk = 0; kk < 4; ++ kk) {
                                                                prod_0_0[kk] = scale_prod(scale_a_0, scale_b_0[kk]);
                                                                prod_1_0[kk] = scale_prod(scale_a_1, scale_b_0[kk]);
                                                                prod_0_1[kk] = scale_prod(scale_a_0, scale_b_1[kk]);
                                                                prod_1_1[kk] = scale_prod(scale_a_1, scale_b_1[kk]);
                                                            }
                                                            fp4_rs_detail::final_accum_promote_pair4<
                                                                kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                                shifted_accum, i * 2 + 0,
                                                                prod_0_0[0], accum_0[i * 4 + 0], prod_1_0[0], accum_0[i * 4 + 1],
                                                                prod_0_0[1], accum_1[i * 4 + 0], prod_1_0[1], accum_1[i * 4 + 1],
                                                                prod_0_0[2], accum_2[i * 4 + 0], prod_1_0[2], accum_2[i * 4 + 1],
                                                                prod_0_0[3], accum_3[i * 4 + 0], prod_1_0[3], accum_3[i * 4 + 1]);
                                                            fp4_rs_detail::final_accum_promote_pair4<
                                                                kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                                shifted_accum, i * 2 + 1,
                                                                prod_0_1[0], accum_0[i * 4 + 2], prod_1_1[0], accum_0[i * 4 + 3],
                                                                prod_0_1[1], accum_1[i * 4 + 2], prod_1_1[1], accum_1[i * 4 + 3],
                                                                prod_0_1[2], accum_2[i * 4 + 2], prod_1_1[2], accum_2[i * 4 + 3],
                                                                prod_0_1[3], accum_3[i * 4 + 2], prod_1_1[3], accum_3[i * 4 + 3]);
                                                        } else if constexpr (kK32QuadSplitPromote or kK32QuadShortProductPromote) {
                                                            fp4_rs_detail::final_accum_promote_pair4_split_scale<
                                                                kBF16FinalAccum, kFmaPromote, kBF16PromoteMath, kScaleBPow2Promote>(
                                                                shifted_accum, i * 2 + 0, scale_a_0, scale_a_1,
                                                                scale_b_0[0], scale_b_0[1], scale_b_0[2], scale_b_0[3],
                                                                accum_0[i * 4 + 0], accum_0[i * 4 + 1],
                                                                accum_1[i * 4 + 0], accum_1[i * 4 + 1],
                                                                accum_2[i * 4 + 0], accum_2[i * 4 + 1],
                                                                accum_3[i * 4 + 0], accum_3[i * 4 + 1]);
                                                            fp4_rs_detail::final_accum_promote_pair4_split_scale<
                                                                kBF16FinalAccum, kFmaPromote, kBF16PromoteMath, kScaleBPow2Promote>(
                                                                shifted_accum, i * 2 + 1, scale_a_0, scale_a_1,
                                                                scale_b_1[0], scale_b_1[1], scale_b_1[2], scale_b_1[3],
                                                                accum_0[i * 4 + 2], accum_0[i * 4 + 3],
                                                                accum_1[i * 4 + 2], accum_1[i * 4 + 3],
                                                                accum_2[i * 4 + 2], accum_2[i * 4 + 3],
                                                                accum_3[i * 4 + 2], accum_3[i * 4 + 3]);
                                                        } else {
                                                            if constexpr (kK32QuadPair4x2Promote) {
                                                                fp4_rs_detail::final_accum_promote_pair4x2<
                                                                    kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                                    shifted_accum, i * 2,
                                                                    scale_prod(scale_a_0, scale_b_0[0]), accum_0[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[0]), accum_0[i * 4 + 1],
                                                                    scale_prod(scale_a_0, scale_b_0[1]), accum_1[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[1]), accum_1[i * 4 + 1],
                                                                    scale_prod(scale_a_0, scale_b_0[2]), accum_2[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[2]), accum_2[i * 4 + 1],
                                                                    scale_prod(scale_a_0, scale_b_0[3]), accum_3[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[3]), accum_3[i * 4 + 1],
                                                                    scale_prod(scale_a_0, scale_b_1[0]), accum_0[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[0]), accum_0[i * 4 + 3],
                                                                    scale_prod(scale_a_0, scale_b_1[1]), accum_1[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[1]), accum_1[i * 4 + 3],
                                                                    scale_prod(scale_a_0, scale_b_1[2]), accum_2[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[2]), accum_2[i * 4 + 3],
                                                                    scale_prod(scale_a_0, scale_b_1[3]), accum_3[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[3]), accum_3[i * 4 + 3]);
                                                            } else {
                                                                fp4_rs_detail::final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                                    shifted_accum, i * 2 + 0,
                                                                    scale_prod(scale_a_0, scale_b_0[0]), accum_0[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[0]), accum_0[i * 4 + 1],
                                                                    scale_prod(scale_a_0, scale_b_0[1]), accum_1[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[1]), accum_1[i * 4 + 1],
                                                                    scale_prod(scale_a_0, scale_b_0[2]), accum_2[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[2]), accum_2[i * 4 + 1],
                                                                    scale_prod(scale_a_0, scale_b_0[3]), accum_3[i * 4 + 0],
                                                                    scale_prod(scale_a_1, scale_b_0[3]), accum_3[i * 4 + 1]);
                                                                fp4_rs_detail::final_accum_promote_pair4<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                                    shifted_accum, i * 2 + 1,
                                                                    scale_prod(scale_a_0, scale_b_1[0]), accum_0[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[0]), accum_0[i * 4 + 3],
                                                                    scale_prod(scale_a_0, scale_b_1[1]), accum_1[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[1]), accum_1[i * 4 + 3],
                                                                    scale_prod(scale_a_0, scale_b_1[2]), accum_2[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[2]), accum_2[i * 4 + 3],
                                                                    scale_prod(scale_a_0, scale_b_1[3]), accum_3[i * 4 + 2],
                                                                    scale_prod(scale_a_1, scale_b_1[3]), accum_3[i * 4 + 3]);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        };

                                        if constexpr (kK32QuadReduce) {
                                            DG_STATIC_ASSERT(BLOCK_K / WGMMA::K == 4,
                                                             "DG_W4_K32_QUAD_REDUCE assumes four K=32 slices");
                                            float scale_b_0[4];
                                            float scale_b_1[4];
                                            if constexpr (kWGMMAF16Accum) {
                                                DG_STATIC_ASSERT(BLOCK_M == 8,
                                                                 "DG_W4_WGMMA_F16_ACCUM currently targets BM8 small-M path");
                                                DG_STATIC_ASSERT(WGMMA::kNumAccum == 4,
                                                                 "FP16 accumulator unpack assumes m64n8 f32 accumulator layout");
                                                using WGMMAF16 = fp4_rs_detail::FP8MMAF16AccumM64N8K32RS;
                                                uint32_t accum_h0[WGMMAF16::kNumAccum];
                                                uint32_t accum_h1[WGMMAF16::kNumAccum];
                                                uint32_t accum_h2[WGMMAF16::kNumAccum];
                                                uint32_t accum_h3[WGMMAF16::kNumAccum];
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMAF16::kNumAccum; ++ i) {
                                                    accum_h0[i] = 0;
                                                    accum_h1[i] = 0;
                                                    accum_h2[i] = 0;
                                                    accum_h3[i] = 0;
                                                }
                                                if constexpr (not kWGMMAStub) {
                                                    auto fence_u32 = [](uint32_t& value) {
                                                        asm volatile("" : "+r"(value) :: "memory");
                                                    };
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMAF16::kNumAccum; ++ i) {
                                                        fence_u32(accum_h0[i]);
                                                        fence_u32(accum_h1[i]);
                                                        fence_u32(accum_h2[i]);
                                                        fence_u32(accum_h3[i]);
                                                    }
                                                    ptx::warpgroup_arrive();
                                                    a_desc.reg32_[0] = a_desc_base_lo;
                                                    WGMMAF16::wgmma(a_regs, a_desc, accum_h0, false);
                                                    asm volatile("" ::: "memory");
                                                    load_a_regs(1, a_regs);
                                                    a_desc.reg32_[0] = a_desc_base_lo + WGMMAF16::K / 16;
                                                    WGMMAF16::wgmma(a_regs, a_desc, accum_h1, false);
                                                    asm volatile("" ::: "memory");
                                                    load_a_regs(2, a_regs);
                                                    a_desc.reg32_[0] = a_desc_base_lo + 2 * WGMMAF16::K / 16;
                                                    WGMMAF16::wgmma(a_regs, a_desc, accum_h2, false);
                                                    asm volatile("" ::: "memory");
                                                    load_a_regs(3, a_regs);
                                                    a_desc.reg32_[0] = a_desc_base_lo + 3 * WGMMAF16::K / 16;
                                                    WGMMAF16::wgmma(a_regs, a_desc, accum_h3, false);
                                                    asm volatile("" ::: "memory");
                                                    ptx::warpgroup_commit_batch();
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMAF16::kNumAccum; ++ i) {
                                                        fence_u32(accum_h0[i]);
                                                        fence_u32(accum_h1[i]);
                                                        fence_u32(accum_h2[i]);
                                                        fence_u32(accum_h3[i]);
                                                    }
                                                    if constexpr (kK32QuadScaleBPrefetch) {
                                                        load_k32_quad_scale_b(scale_b_0, scale_b_1);
                                                    }
                                                    ptx::warpgroup_wait<0>();
                                                }
                                                if constexpr (kK32QuadScaleBPrefetch) {
                                                    // Already loaded while WGMMA was in flight.
                                                } else {
                                                    load_k32_quad_scale_b(scale_b_0, scale_b_1);
                                                }
                                                float accum_f0[WGMMA::kNumAccum];
                                                float accum_f1[WGMMA::kNumAccum];
                                                float accum_f2[WGMMA::kNumAccum];
                                                float accum_f3[WGMMA::kNumAccum];
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i) {
                                                    accum_f0[i] = 0.0f;
                                                    accum_f1[i] = 0.0f;
                                                    accum_f2[i] = 0.0f;
                                                    accum_f3[i] = 0.0f;
                                                }
                                                if constexpr (not kWGMMAStub) {
                                                    fp4_rs_detail::unpack_f16_accum_m64n8(accum_h0, accum_f0);
                                                    fp4_rs_detail::unpack_f16_accum_m64n8(accum_h1, accum_f1);
                                                    fp4_rs_detail::unpack_f16_accum_m64n8(accum_h2, accum_f2);
                                                    fp4_rs_detail::unpack_f16_accum_m64n8(accum_h3, accum_f3);
                                                }
                                                promote_k32_quad(accum_f0, accum_f1, accum_f2, accum_f3, scale_b_0, scale_b_1);
                                            } else {
                                                float accum_2[WGMMA::kNumAccum];
                                                float accum_3[WGMMA::kNumAccum];
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i) {
                                                    accum[i] = 0.0f;
                                                    accum_alt[i] = 0.0f;
                                                    accum_2[i] = 0.0f;
                                                    accum_3[i] = 0.0f;
                                                }
                                                if constexpr (not kWGMMAStub) {
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i) {
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                        ptx::warpgroup_fence_operand(accum_alt[i]);
                                                        ptx::warpgroup_fence_operand(accum_2[i]);
                                                        ptx::warpgroup_fence_operand(accum_3[i]);
                                                    }
                                                    ptx::warpgroup_arrive();
                                                    a_desc.reg32_[0] = a_desc_base_lo;
                                                    WGMMA::wgmma(a_regs, a_desc, accum, false);
                                                    asm volatile("" ::: "memory");
                                                    load_a_regs(1, a_regs);
                                                    a_desc.reg32_[0] = a_desc_base_lo + WGMMA::K / 16;
                                                    WGMMA::wgmma(a_regs, a_desc, accum_alt, false);
                                                    asm volatile("" ::: "memory");
                                                    load_a_regs(2, a_regs);
                                                    a_desc.reg32_[0] = a_desc_base_lo + 2 * WGMMA::K / 16;
                                                    WGMMA::wgmma(a_regs, a_desc, accum_2, false);
                                                    asm volatile("" ::: "memory");
                                                    load_a_regs(3, a_regs);
                                                    a_desc.reg32_[0] = a_desc_base_lo + 3 * WGMMA::K / 16;
                                                    WGMMA::wgmma(a_regs, a_desc, accum_3, false);
                                                    asm volatile("" ::: "memory");
                                                    ptx::warpgroup_commit_batch();
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i) {
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                        ptx::warpgroup_fence_operand(accum_alt[i]);
                                                        ptx::warpgroup_fence_operand(accum_2[i]);
                                                        ptx::warpgroup_fence_operand(accum_3[i]);
                                                    }
                                                    if constexpr (kK32QuadScaleBPrefetch) {
                                                        load_k32_quad_scale_b(scale_b_0, scale_b_1);
                                                    }
                                                    ptx::warpgroup_wait<0>();
                                                }
                                                if constexpr (not kK32QuadScaleBPrefetch) {
                                                    load_k32_quad_scale_b(scale_b_0, scale_b_1);
                                                }
                                                promote_k32_quad(accum, accum_alt, accum_2, accum_3, scale_b_0, scale_b_1);
                                            }
                                        } else {
                                            #pragma unroll
                                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; k += 2) {
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i) {
                                                    accum[i] = 0.0f;
                                                    accum_alt[i] = 0.0f;
                                                }
                                                if constexpr (not kWGMMAStub) {
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i) {
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                        ptx::warpgroup_fence_operand(accum_alt[i]);
                                                    }
                                                    ptx::warpgroup_arrive();
                                                    a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                                                    WGMMA::wgmma(a_regs, a_desc, accum, false);
                                                    asm volatile("" ::: "memory");
                                                }
                                                load_a_regs(k + 1, a_regs);
                                                if constexpr (not kWGMMAStub) {
                                                    a_desc.reg32_[0] = a_desc_base_lo + (k + 1) * WGMMA::K / 16;
                                                    WGMMA::wgmma(a_regs, a_desc, accum_alt, false);
                                                    asm volatile("" ::: "memory");
                                                    ptx::warpgroup_commit_batch();
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i) {
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                        ptx::warpgroup_fence_operand(accum_alt[i]);
                                                    }
                                                    ptx::warpgroup_wait<0>();
                                                }

                                                if constexpr (kK32PairReduce) {
                                                    promote_k32_pair(accum, accum_alt, k, k + 1);
                                                } else {
                                                    promote_k32_slice(accum, k);
                                                    promote_k32_slice(accum_alt, k + 1);
                                                }

                                                if constexpr (BLOCK_K / WGMMA::K > 2) {
                                                    if (k + 2 < BLOCK_K / WGMMA::K)
                                                        load_a_regs(k + 2, a_regs);
                                                }
                                            }
                                        }
                                    } else {
                                            #pragma unroll
                                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                                #pragma unroll
                                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                                    accum[i] = 0.0f;
                                                float slice_scale_b_0 = 1.0f;
                                                float slice_scale_b_1 = 1.0f;
                                                if constexpr ((kScaleBEarlyLoad or kScaleBEarlyProduct) and not kScaleBStub) {
                                                    const uint32_t scale_b_k_idx =
                                                        k_block_idx * (BLOCK_K / kScaleBGranK) + k;
                                                    slice_scale_b_0 = load_sfb(compute_n_0, scale_b_k_idx);
                                                    slice_scale_b_1 = load_sfb(compute_n_1, scale_b_k_idx);
                                                }
                                                float early_prod_0_0[WGMMA::kNumAccum / 4];
                                                float early_prod_1_0[WGMMA::kNumAccum / 4];
                                                float early_prod_0_1[WGMMA::kNumAccum / 4];
                                                float early_prod_1_1[WGMMA::kNumAccum / 4];
                                                if constexpr (kScaleBEarlyProduct and not kPromoteStub) {
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                                        float scale_a_0 = 1.0f;
                                                        float scale_a_1 = 1.0f;
                                                        if constexpr (kLateScaleA) {
                                                            if constexpr (not kScaleAStub) {
                                                                const uint32_t m_idx = i * 8 + col_idx * 2;
                                                                scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                                scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                            }
                                                        } else {
                                                            scale_a_0 = scale_a_0_regs[i];
                                                            scale_a_1 = scale_a_1_regs[i];
                                                        }
                                                        if constexpr (kScaleBMulStub) {
                                                            fp4_rs_detail::keep_float_live(slice_scale_b_0);
                                                            fp4_rs_detail::keep_float_live(slice_scale_b_1);
                                                        }
                                                        early_prod_0_0[i] = kScaleBMulStub ? scale_a_0 :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_0, slice_scale_b_0) :
                                                             scale_a_0 * slice_scale_b_0);
                                                        early_prod_1_0[i] = kScaleBMulStub ? scale_a_1 :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_1, slice_scale_b_0) :
                                                             scale_a_1 * slice_scale_b_0);
                                                        early_prod_0_1[i] = kScaleBMulStub ? scale_a_0 :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_0, slice_scale_b_1) :
                                                             scale_a_0 * slice_scale_b_1);
                                                        early_prod_1_1[i] = kScaleBMulStub ? scale_a_1 :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_1, slice_scale_b_1) :
                                                             scale_a_1 * slice_scale_b_1);
                                                    }
                                                }
                                                if constexpr (not kWGMMAStub) {
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                    ptx::warpgroup_arrive();
                                                    a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                                                    WGMMA::wgmma(a_regs, a_desc, accum, false);
                                                    asm volatile("" ::: "memory");
                                                    ptx::warpgroup_commit_batch();
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                                        ptx::warpgroup_fence_operand(accum[i]);
                                                    ptx::warpgroup_wait<0>();
                                                }

                                                if constexpr (not kPromoteStub) {
                                                    if constexpr (not kScaleBEarlyLoad and not kScaleBEarlyProduct and not kScaleBStub) {
                                                        const uint32_t scale_b_k_idx =
                                                            k_block_idx * (BLOCK_K / kScaleBGranK) + k;
                                                        slice_scale_b_0 = load_sfb(compute_n_0, scale_b_k_idx);
                                                        slice_scale_b_1 = load_sfb(compute_n_1, scale_b_k_idx);
                                                    }
                                                    #pragma unroll
                                                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                                        float scale_a_0 = 1.0f;
                                                        float scale_a_1 = 1.0f;
                                                        if constexpr (kLateScaleA) {
                                                            if constexpr (not kScaleAStub) {
                                                                const uint32_t m_idx = i * 8 + col_idx * 2;
                                                                scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                                scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                                            }
                                                        } else {
                                                            scale_a_0 = scale_a_0_regs[i];
                                                            scale_a_1 = scale_a_1_regs[i];
                                                        }
                                                        if constexpr (kScaleBEarlyProduct and not kPromoteMulStub) {
                                                            // Product was computed before WGMMA wait to test latency hiding.
                                                        } else if constexpr (kScaleBMulStub) {
                                                            fp4_rs_detail::keep_float_live(slice_scale_b_0);
                                                            fp4_rs_detail::keep_float_live(slice_scale_b_1);
                                                        }
                                                        const float prod_0_0 = (kPromoteMulStub or kScaleBMulStub) ? scale_a_0 :
                                                            (kScaleBEarlyProduct ? early_prod_0_0[i] :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_0, slice_scale_b_0) :
                                                             scale_a_0 * slice_scale_b_0));
                                                        const float prod_1_0 = (kPromoteMulStub or kScaleBMulStub) ? scale_a_1 :
                                                            (kScaleBEarlyProduct ? early_prod_1_0[i] :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_1, slice_scale_b_0) :
                                                             scale_a_1 * slice_scale_b_0));
                                                        const float prod_0_1 = (kPromoteMulStub or kScaleBMulStub) ? scale_a_0 :
                                                            (kScaleBEarlyProduct ? early_prod_0_1[i] :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_0, slice_scale_b_1) :
                                                             scale_a_0 * slice_scale_b_1));
                                                        const float prod_1_1 = (kPromoteMulStub or kScaleBMulStub) ? scale_a_1 :
                                                            (kScaleBEarlyProduct ? early_prod_1_1[i] :
                                                            (kScaleBPow2Promote ? fp4_rs_detail::scale_float_by_pow2(scale_a_1, slice_scale_b_1) :
                                                             scale_a_1 * slice_scale_b_1));
                                                        if constexpr (kPromoteAccumStub or kPromoteFinalAccumStub) {
                                                            fp4_rs_detail::keep_float_live(prod_0_0 + accum[i * 4 + 0]);
                                                            fp4_rs_detail::keep_float_live(prod_1_0 + accum[i * 4 + 1]);
                                                            fp4_rs_detail::keep_float_live(prod_0_1 + accum[i * 4 + 2]);
                                                            fp4_rs_detail::keep_float_live(prod_1_1 + accum[i * 4 + 3]);
                                                        } else if constexpr (kPromoteMulStub) {
                                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 0,
                                                            1.0f, accum[i * 4 + 0],
                                                            1.0f, accum[i * 4 + 1]);
                                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 1,
                                                            1.0f, accum[i * 4 + 2],
                                                            1.0f, accum[i * 4 + 3]);
                                                        } else {
                                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 0,
                                                            prod_0_0, accum[i * 4 + 0],
                                                            prod_1_0, accum[i * 4 + 1]);
                                                            fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                            shifted_accum, i * 2 + 1,
                                                            prod_0_1, accum[i * 4 + 2],
                                                            prod_1_1, accum[i * 4 + 3]);
                                                        }
                                                    }
                                                }

                                                if constexpr (BLOCK_K / WGMMA::K > 1) {
                                                    if (k + 1 < BLOCK_K / WGMMA::K)
                                                        load_a_regs(k + 1, a_regs);
                                                }
                                            }
                                        }
                                }

                                const bool is_last_wave = (local_idx == WAVE_WGMMA - 1);
                                if (kParallelNWavesEnabled or is_last_wave)
                                    empty_barrier_arrive();
                                continue;
                            }
                            if constexpr (not kWGMMAStub) {
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                            }
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                if constexpr (not kWGMMAStub) {
                                    a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                                    const bool scale_d = kUseScaleKGroup ? (scale_group_pos != 0 or k != 0) : static_cast<bool>(k);
                                    WGMMA::wgmma(a_regs, a_desc, accum, scale_d);
                                    asm volatile("" ::: "memory");
                                }
                                if constexpr (BLOCK_K / WGMMA::K > 1) {
                                    if (k + 1 < BLOCK_K / WGMMA::K)
                                        load_a_regs(k + 1, a_regs);
                                }
                            }
                            float scale_0_0_regs[kFusedPromote ? 1 : WGMMA::kNumAccum / 4];
                            float scale_1_0_regs[kFusedPromote ? 1 : WGMMA::kNumAccum / 4];
                            float scale_0_1_regs[kFusedPromote ? 1 : WGMMA::kNumAccum / 4];
                            float scale_1_1_regs[kFusedPromote ? 1 : WGMMA::kNumAccum / 4];
                            if constexpr (not kPromoteStub and not kFusedPromote) {
                                if (do_wgmma_store and (should_promote_group or kUseScaleKGroup)) {
                                    const uint32_t scale_sum_offset = local_idx * kScaleSumStride;
                                    if constexpr (kUseScaleKGroup) {
                                        if (scale_group_pos == 0) {
                                            #pragma unroll
                                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                                scale_0_0_sum[scale_sum_offset + i] = 0.0f;
                                                scale_1_0_sum[scale_sum_offset + i] = 0.0f;
                                                scale_0_1_sum[scale_sum_offset + i] = 0.0f;
                                                scale_1_1_sum[scale_sum_offset + i] = 0.0f;
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                        const float prod_0_0 = scale_a_0_regs[i] * scale_b_0;
                                        const float prod_1_0 = scale_a_1_regs[i] * scale_b_0;
                                        const float prod_0_1 = scale_a_0_regs[i] * scale_b_1;
                                        const float prod_1_1 = scale_a_1_regs[i] * scale_b_1;
                                        if constexpr (kUseScaleKGroupExact) {
                                            if (scale_group_pos == 0) {
                                                scale_0_0_sum[scale_sum_offset + i] = prod_0_0;
                                                scale_1_0_sum[scale_sum_offset + i] = prod_1_0;
                                                scale_0_1_sum[scale_sum_offset + i] = prod_0_1;
                                                scale_1_1_sum[scale_sum_offset + i] = prod_1_1;
                                            }
                                            if (should_promote_group) {
                                                scale_0_0_regs[i] = prod_0_0;
                                                scale_1_0_regs[i] = prod_1_0;
                                                scale_0_1_regs[i] = prod_0_1;
                                                scale_1_1_regs[i] = prod_1_1;
                                            }
                                        } else if constexpr (kUseScaleKGroup) {
                                            scale_0_0_sum[scale_sum_offset + i] += prod_0_0;
                                            scale_1_0_sum[scale_sum_offset + i] += prod_1_0;
                                            scale_0_1_sum[scale_sum_offset + i] += prod_0_1;
                                            scale_1_1_sum[scale_sum_offset + i] += prod_1_1;
                                            if (should_promote_group) {
                                                const float inv_group = 1.0f / static_cast<float>(scale_group_pos + 1);
                                                scale_0_0_regs[i] = scale_0_0_sum[scale_sum_offset + i] * inv_group;
                                                scale_1_0_regs[i] = scale_1_0_sum[scale_sum_offset + i] * inv_group;
                                                scale_0_1_regs[i] = scale_0_1_sum[scale_sum_offset + i] * inv_group;
                                                scale_1_1_regs[i] = scale_1_1_sum[scale_sum_offset + i] * inv_group;
                                            }
                                        } else {
                                            scale_0_0_regs[i] = prod_0_0;
                                            scale_1_0_regs[i] = prod_1_0;
                                            scale_0_1_regs[i] = prod_0_1;
                                            scale_1_1_regs[i] = prod_1_1;
                                        }
                                    }
                                }
                            }
                            if constexpr (not kWGMMAStub) {
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                            }

                            const bool is_last_wave = (local_idx == WAVE_WGMMA - 1);
                            if constexpr (not kWGMMAStub) {
                                ptx::warpgroup_wait<0>();
                            } else {
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    accum[i] = 0.0f;
                            }
                            if constexpr (kUseScaleKGroupExact) {
                                if (scale_group_pos == 0 and not should_promote_group) {
                                    auto accum_first = accum_first_storage + local_idx * WGMMA::kNumAccum;
                                    #pragma unroll
                                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                        accum_first[i] = accum[i];
                                }
                            }

                            // Notify barrier arrival at the last warpgroup wave
                            if ((kParallelNWavesEnabled or is_last_wave) and
                                (not kLateScaleA or not do_wgmma_store or not should_promote_group))
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store or not should_promote_group)
                                continue;

                            // Promote with scales
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            auto shifted_accum = final_accum + kFinalAccumStride * local_idx;
                            if constexpr (not kPromoteStub) {
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                    if constexpr (kFusedPromote) {
                                        float scale_a_0 = 1.0f;
                                        float scale_a_1 = 1.0f;
                                        if constexpr (kLateScaleA) {
                                            if constexpr (not kScaleAStub) {
                                                const uint32_t m_idx = i * 8 + col_idx * 2;
                                                scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx);
                                                scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m_idx + 1);
                                            }
                                        } else {
                                            scale_a_0 = scale_a_0_regs[i];
                                            scale_a_1 = scale_a_1_regs[i];
                                        }
                                        fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                            shifted_accum, i * 2 + 0,
                                            scale_a_0 * scale_b_0, accum[i * 4 + 0],
                                            scale_a_1 * scale_b_0, accum[i * 4 + 1]);
                                        fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                            shifted_accum, i * 2 + 1,
                                            scale_a_0 * scale_b_1, accum[i * 4 + 2],
                                            scale_a_1 * scale_b_1, accum[i * 4 + 3]);
                                    } else {
                                        fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                            shifted_accum, i * 2 + 0,
                                            scale_0_0_regs[i], accum[i * 4 + 0],
                                            scale_1_0_regs[i], accum[i * 4 + 1]);
                                        fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                            shifted_accum, i * 2 + 1,
                                            scale_0_1_regs[i], accum[i * 4 + 2],
                                            scale_1_1_regs[i], accum[i * 4 + 3]);
                                        if constexpr (kUseScaleKGroupExact) {
                                            if (scale_group_pos != 0) {
                                                const uint32_t scale_sum_offset = local_idx * kScaleSumStride;
                                                auto accum_first = accum_first_storage + local_idx * WGMMA::kNumAccum;
                                                fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                    shifted_accum, i * 2 + 0,
                                                    scale_0_0_sum[scale_sum_offset + i] - scale_0_0_regs[i],
                                                    accum_first[i * 4 + 0],
                                                    scale_1_0_sum[scale_sum_offset + i] - scale_1_0_regs[i],
                                                    accum_first[i * 4 + 1]);
                                                fp4_rs_detail::final_accum_promote_pair<kBF16FinalAccum, kFmaPromote, kBF16PromoteMath>(
                                                    shifted_accum, i * 2 + 1,
                                                    scale_0_1_sum[scale_sum_offset + i] - scale_0_1_regs[i],
                                                    accum_first[i * 4 + 2],
                                                    scale_1_1_sum[scale_sum_offset + i] - scale_1_1_regs[i],
                                                    accum_first[i * 4 + 3]);
                                            }
                                        }
                                    }
                                }
                            }
                            if constexpr (kLateScaleA) {
                                if (kParallelNWavesEnabled or is_last_wave)
                                    empty_barrier_arrive();
                            }
                        }
                    }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            if constexpr (kStoreStub) {
                __syncwarp();
                continue;
            }

            if constexpr (kDirectStore and kGemmType == GemmType::MGroupedMasked and BLOCK_M < WGMMA::M) {
                const uint32_t global_m_base = scheduler.template get_global_idx<true>(shape_m, BLOCK_M, m_block_idx);
                #pragma unroll
                for (uint32_t local_idx = kParallelNWavesEnabled ? wave_group_idx : 0;
                     local_idx < WAVE_WGMMA;
                     local_idx += kParallelNWavesEnabled ? WAVE_WGMMA : 1) {
                    const uint32_t n_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = final_accum + kFinalAccumStride * local_idx;
                    constexpr uint32_t kDirectStoreIters =
                        BLOCK_M < WGMMA::M ? math::constexpr_ceil_div(BLOCK_M, 8u) : WGMMA::kNumAccum / 4;
                    #pragma unroll
                    for (uint32_t i = 0; i < kDirectStoreIters; ++ i) {
                        const uint32_t local_m_0 = i * 8 + col_idx * 2;
                        const uint32_t local_m_1 = local_m_0 + 1;
                        const uint32_t col_0 = epilogue_type_t::template apply_index_n<1>(
                            n_block_idx * BLOCK_N + n_offset + r_0);
                        const uint32_t col_1 = epilogue_type_t::template apply_index_n<1>(
                            n_block_idx * BLOCK_N + n_offset + r_1);
                        const bool row_0_valid = scheduler.is_computation_valid(m_block_idx, local_m_0);
                        const bool row_1_valid = scheduler.is_computation_valid(m_block_idx, local_m_1);
                        auto direct_store = [&](bool row_valid, uint32_t local_m, uint32_t col, uint32_t accum_idx) {
                            if (row_valid and col < shape_n) {
                                const nv_bfloat16 out = __float2bfloat16_rn(
                                    fp4_rs_detail::final_accum_load_scalar<kBF16FinalAccum>(shifted_accum, accum_idx));
                                if constexpr (kTMAStoreStub) {
                                    const uint32_t sink = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&out));
                                    asm volatile("" :: "r"(sink) : "memory");
                                } else {
                                    gmem_d_ptr[(global_m_base + local_m) * shape_n + col] = out;
                                }
                            }
                        };
                        direct_store(row_0_valid, local_m_0, col_0, i * 4 + 0);
                        direct_store(row_1_valid, local_m_1, col_0, i * 4 + 1);
                        direct_store(row_0_valid, local_m_0, col_1, i * 4 + 2);
                        direct_store(row_1_valid, local_m_1, col_1, i * 4 + 3);
                    }
                }
                __syncwarp();
                continue;
            }

            // Psum layout can have a final partial M tile. TMA store writes a
            // full BLOCK_M tile and may go out of the tensor-map bounds, so use
            // a guarded scalar store for this layout.
            if constexpr (kGemmType == GemmType::MGroupedMasked and BLOCK_M < WGMMA::M) {
                // Full small-BM tiles can use the normal STSM+TMA store path below.
                // Guarded scalar copy-back is only needed for partial masked tiles.
                if (not scheduler.is_computation_valid(m_block_idx, BLOCK_M - 1)) {
                    const uint32_t global_m_base = scheduler.template get_global_idx<true>(shape_m, BLOCK_M, m_block_idx);
                    #pragma unroll
                    for (uint32_t local_idx = kParallelNWavesEnabled ? wave_group_idx : 0;
                         local_idx < WAVE_WGMMA;
                         local_idx += kParallelNWavesEnabled ? WAVE_WGMMA : 1) {
                        const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                        auto shifted_accum = final_accum + kFinalAccumStride * local_idx;
                        constexpr uint32_t kStoreIters =
                            BLOCK_M < WGMMA::M ? math::constexpr_ceil_div(BLOCK_M, 8u) : WGMMA::kNumAccum / 4;
                        #pragma unroll
                        for (auto i = 0; i < kStoreIters; ++ i) {
                            uint8_t* smem_ptr = nullptr;
                            if constexpr (kSwizzleDMode > 0) {
                                constexpr uint32_t kNumBankGroupBytes = 16;
                                const uint32_t row = i * 8 + lane_idx % 8;
                                uint32_t col = warp_in_wg * 2 + lane_idx / 8;
                                col ^= row % (kSwizzleDMode / 16);
                                const uint32_t n_atom_idx = m_offset / WGMMA::M + wave_mwg_idx;
                                smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                                    n_atom_idx * SMEM_D_ROWS * kSwizzleDMode +
                                    row * (kNumBankGroupBytes * 8) +
                                    col * kNumBankGroupBytes;
                            } else {
                                const uint32_t row = i * 8 + lane_idx % 8;
                                const uint32_t col = m_offset + wave_mwg_idx * WGMMA::M + warp_in_wg * 2 + lane_idx / 8;
                                smem_ptr = reinterpret_cast<uint8_t*>(smem_d + row * BLOCK_N + col);
                            }

                            if constexpr (not kSTSMStub) {
                                const nv_bfloat162 out_0 =
                                    fp4_rs_detail::final_accum_load_pair_bf16<kBF16FinalAccum>(shifted_accum, i * 2 + 0);
                                const nv_bfloat162 out_1 =
                                    fp4_rs_detail::final_accum_load_pair_bf16<kBF16FinalAccum>(shifted_accum, i * 2 + 1);
                                if constexpr (kSTSMConvertOnly) {
                                    const uint32_t sink_0 = *reinterpret_cast<const uint32_t*>(&out_0);
                                    const uint32_t sink_1 = *reinterpret_cast<const uint32_t*>(&out_1);
                                    asm volatile("" :: "r"(sink_0), "r"(sink_1) : "memory");
                                } else {
                                    fp4_rs_detail::SM90_U32x2_STSM_T<nv_bfloat162>::copy(out_0, out_1, smem_ptr);
                                }
                            }
                        }
                    }
                    cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

                    if constexpr (not kTMAStoreStub) {
                        const uint32_t group_m = __ldg(grouped_layout + current_group_idx);
                        const uint32_t valid_rows = group_m - m_block_idx * BLOCK_M;
                        const uint32_t store_elems = valid_rows * BLOCK_N;
                        for (uint32_t idx = threadIdx.x; idx < store_elems; idx += kNumWGMMAStoreThreads) {
                            const uint32_t row = idx / BLOCK_N;
                            const uint32_t col = idx % BLOCK_N;
                            const uint32_t global_col = epilogue_type_t::template apply_index_n<1>(n_block_idx * BLOCK_N + col);
                            if (global_col < shape_n) {
                                nv_bfloat16* smem_src = nullptr;
                                if constexpr (kSwizzleDMode > 0) {
                                    constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
                                    constexpr uint32_t kNumBankGroupBytes = 16;
                                    constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode / kNumElemBytes;
                                    const uint32_t n_atom_idx = col / TMA_D_BLOCK_N;
                                    const uint32_t in_atom_col = col % TMA_D_BLOCK_N;
                                    uint32_t bank_col = in_atom_col / (kNumBankGroupBytes / kNumElemBytes);
                                    bank_col ^= row % (kSwizzleDMode / kNumBankGroupBytes);
                                    const uint32_t bank_offset = in_atom_col % (kNumBankGroupBytes / kNumElemBytes);
                                    smem_src = reinterpret_cast<nv_bfloat16*>(
                                        reinterpret_cast<uint8_t*>(smem_d) +
                                        n_atom_idx * SMEM_D_ROWS * kSwizzleDMode +
                                        row * (kNumBankGroupBytes * 8) +
                                        bank_col * kNumBankGroupBytes +
                                        bank_offset * kNumElemBytes);
                                } else {
                                    smem_src = smem_d + row * BLOCK_N + col;
                                }
                                gmem_d_ptr[(global_m_base + row) * shape_n + global_col] = *smem_src;
                            }
                        }
                    }
                    __syncwarp();
                    continue;
                }
            }

            if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
                const uint32_t psum_store_end =
                    kGemmType == GemmType::MGroupedContiguousWithPsumLayout ?
                    (scheduler.current_group_idx + 1 < kNumGroups ?
                        math::align(scheduler.current_psum_m, BLOCK_M) : scheduler.current_psum_m) :
                    shape_m;
                constexpr uint32_t kStoreWaves =
                    BLOCK_M < WAVE_BLOCK_M ? 1 : BLOCK_M / WAVE_BLOCK_M;
                #pragma unroll
                for (uint32_t local_idx = kParallelNWavesEnabled ? wave_group_idx : 0;
                     local_idx < kStoreWaves;
                     local_idx += kParallelNWavesEnabled ? WAVE_WGMMA : 1) {
                    const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = final_accum + kFinalAccumStride * local_idx;
                    constexpr bool kStoreWithGroupOffset = kGemmType == GemmType::MGroupedMasked;
                    const uint32_t row_0 = scheduler.template get_global_idx<kStoreWithGroupOffset>(
                        shape_m, BLOCK_M, m_block_idx) + m_offset + r_0;
                    const uint32_t row_1 = scheduler.template get_global_idx<kStoreWithGroupOffset>(
                        shape_m, BLOCK_M, m_block_idx) + m_offset + r_1;
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        const uint32_t base_col = epilogue_type_t::template apply_index_n<8>(
                            n_block_idx * BLOCK_N + i * 8);
                        const uint32_t col = base_col + col_idx * 2;
                        const bool row_0_valid = kGemmType == GemmType::MGroupedContiguousWithPsumLayout ?
                            (row_0 >= scheduler.last_psum_m and row_0 < psum_store_end) :
                            scheduler.is_computation_valid(m_block_idx, m_offset + r_0);
                        const bool row_1_valid = kGemmType == GemmType::MGroupedContiguousWithPsumLayout ?
                            (row_1 >= scheduler.last_psum_m and row_1 < psum_store_end) :
                            scheduler.is_computation_valid(m_block_idx, m_offset + r_1);
                        if (row_0_valid and col < shape_n)
                            gmem_d_ptr[row_0 * shape_n + col] =
                                __float2bfloat16_rn(fp4_rs_detail::final_accum_load_scalar<kBF16FinalAccum>(shifted_accum, i * 4 + 0));
                        if (row_0_valid and col + 1 < shape_n)
                            gmem_d_ptr[row_0 * shape_n + col + 1] =
                                __float2bfloat16_rn(fp4_rs_detail::final_accum_load_scalar<kBF16FinalAccum>(shifted_accum, i * 4 + 1));
                        if (row_1_valid and col < shape_n)
                            gmem_d_ptr[row_1 * shape_n + col] =
                                __float2bfloat16_rn(fp4_rs_detail::final_accum_load_scalar<kBF16FinalAccum>(shifted_accum, i * 4 + 2));
                        if (row_1_valid and col + 1 < shape_n)
                            gmem_d_ptr[row_1 * shape_n + col + 1] =
                                __float2bfloat16_rn(fp4_rs_detail::final_accum_load_scalar<kBF16FinalAccum>(shifted_accum, i * 4 + 3));
                    }
                }
                __syncwarp();
                continue;
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = kParallelNWavesEnabled ? wave_group_idx : 0;
                 local_idx < WAVE_WGMMA;
                 local_idx += kParallelNWavesEnabled ? WAVE_WGMMA : 1) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + kFinalAccumStride * local_idx;
                constexpr uint32_t kStoreIters =
                    BLOCK_M < WGMMA::M ? math::constexpr_ceil_div(BLOCK_M, 8u) : WGMMA::kNumAccum / 4;
                #pragma unroll
                for (auto i = 0; i < kStoreIters; ++ i) {
                    // RS swap_ab accumulates an original-N x original-M tile.
                    // Store with stmatrix.trans so smem_d remains original-M x original-N for TMA store.
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        const uint32_t row = i * 8 + lane_idx % 8;
                        uint32_t col = warp_in_wg * 2 + lane_idx / 8;
                        col ^= row % (kSwizzleDMode / 16);
                        const uint32_t n_atom_idx = m_offset / WGMMA::M + wave_mwg_idx;
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                            n_atom_idx * BLOCK_M * kSwizzleDMode +
                            row * (kNumBankGroupBytes * 8) +
                            col * kNumBankGroupBytes;
                    } else {
                        // No swizzling, just padding
                        const uint32_t row = i * 8 + lane_idx % 8;
                        const uint32_t col = m_offset + wave_mwg_idx * WGMMA::M + warp_in_wg * 2 + lane_idx / 8;
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + row * BLOCK_N + col);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    if constexpr (not kSTSMStub) {
                        const nv_bfloat162 out_0 =
                            fp4_rs_detail::final_accum_load_pair_bf16<kBF16FinalAccum>(shifted_accum, i * 2 + 0);
                        const nv_bfloat162 out_1 =
                            fp4_rs_detail::final_accum_load_pair_bf16<kBF16FinalAccum>(shifted_accum, i * 2 + 1);
                        if constexpr (kSTSMConvertOnly) {
                            const uint32_t sink_0 = *reinterpret_cast<const uint32_t*>(&out_0);
                            const uint32_t sink_1 = *reinterpret_cast<const uint32_t*>(&out_1);
                            asm volatile("" :: "r"(sink_0), "r"(sink_1) : "memory");
                        } else {
                            fp4_rs_detail::SM90_U32x2_STSM_T<nv_bfloat162>::copy(out_0, out_1, smem_ptr);
                        }
                    }
                }
            }
            if constexpr (not kTMAStoreStub) {
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);
            }

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if constexpr (not kTMAStoreStub) {
                if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                    auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                    auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                    auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                    auto m_idx = scheduler.template get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                    if constexpr (kGemmType == GemmType::Batched) {
                        cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                      n_idx, m_idx, scheduler.current_group_idx);
                    } else {
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                    }
                    cute::tma_store_arrive();
                }
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
