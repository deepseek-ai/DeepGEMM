#pragma once

#include <deep_gemm/impls/sm120_bf16_gemm.cuh>

namespace deep_gemm {

// One math warp owns one expert. L1 shares the original X row between
// experts; L2 retains the rounded expert contributions until CTA-local combine.
// The ring barriers transfer data ownership, not scheduler commands.
template <uint32_t D, uint32_t I, uint32_t TOPK, uint32_t BN,
          uint32_t STAGES, uint32_t BUNDLE, bool L1, bool GATE_UP_SWAPPED = false,
          uint32_t EXPERTS = 0, typename IndexT = int64_t>
CUTLASS_GLOBAL __launch_bounds__((BUNDLE + 4) * 32, 2)
void sm120_bf16_moe_small(
    const IndexT* slots, const float* weights, cutlass::bfloat16_t* output,
    uint32_t capacity, const __nv_bfloat16* original_a, const __nv_bfloat16* original_b,
    const __grid_constant__ cute::TmaDescriptor map_a,
    const __grid_constant__ cute::TmaDescriptor map_b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    static_assert(TOPK >= 1 && TOPK <= 8 && TOPK % BUNDLE == 0);
    static_assert(L1 || BUNDLE == TOPK);
    static_assert(L1 || !GATE_UP_SWAPPED);
    static_assert(BN == 8 || BN == 16 || BN == 32);
    static_assert(D % BN == 0 && I % 64 == 0 && STAGES >= 2 && STAGES <= 4);
    constexpr uint32_t GROUPS = TOPK / BUNDLE;
    constexpr uint32_t A_BYTES = (L1 ? 1 : BUNDLE) * 128;
    constexpr uint32_t B_BYTES = BUNDLE * (L1 ? 2 : 1) * BN * 128;
    // Each swizzled B tile needs a separately 1 KiB aligned base, including
    // for TOPK=1/2. The unswizzled A ring size need not have this alignment.
    constexpr uint32_t B_OFFSET = (STAGES * A_BYTES + 1023u) & ~1023u;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    extern __shared__ __align__(1024) char buffer[];
    auto* full = reinterpret_cast<Barrier*>(buffer + B_OFFSET + STAGES * B_BYTES);
    auto* empty = full + STAGES;
    auto* partial = reinterpret_cast<__nv_bfloat16*>(empty + STAGES);
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t token = blockIdx.x / (D / BN) / GROUPS;
    const uint32_t nb = blockIdx.x % (D / BN);
    const uint32_t route_base = blockIdx.x / (D / BN) % GROUPS * BUNDLE;
    if (threadIdx.x == 0) {
        for (uint32_t s = 0; s < STAGES; ++s) {
            full[s].init(4);
            empty[s].init(BUNDLE);
        }
        cutlass::arch::fence_barrier_init();
        cute::prefetch_tma_descriptor(&map_a);
        cute::prefetch_tma_descriptor(&map_b);
    }
    __syncthreads();
    if (warp >= BUNDLE) {
        if (lane == 0) {
            for (uint32_t kb = 0; kb < I / 64; ++kb) {
                const uint32_t s = kb % STAGES, phase = (kb / STAGES) & 1;
                empty[s].wait(phase ^ 1);
                uint32_t bytes = 0;
                for (uint32_t r = warp - BUNDLE; r < BUNDLE; r += 4) {
                    uint64_t slot;
                    uint32_t expert;
                    if constexpr (EXPERTS > 0) {
                        const int64_t id = slots[route_base + r];
                        const bool valid = id >= 0 && id < int64_t(EXPERTS);
                        expert = valid ? uint32_t(id) : 0;
                        slot = valid ? uint64_t(expert) * capacity : route_base + r + 1;
                    } else {
                        slot = slots[token * TOPK + route_base + r];
                        expert = slot / capacity;
                    }
                    if constexpr (L1) {
                        if (r == 0) {
                            tma::copy<128, 1, 0, char>(&map_a, &full[s],
                                buffer + s * A_BYTES, kb * 64, token, 1);
                            bytes += 128;
                        }
                    } else {
                        tma::copy<128, 1, 0, char>(&map_a, &full[s],
                            buffer + s * A_BYTES + r * 128, kb * 64, slot, 1);
                        bytes += 128;
                    }
                    tma::copy<128, BN, 128, char>(&map_b, &full[s],
                        buffer + B_OFFSET + s * B_BYTES + r * (L1 ? 2 : 1) * BN * 128,
                        kb * 64, expert * (L1 ? 2 : 1) * D +
                            (GATE_UP_SWAPPED ? D : 0) + nb * BN, 1);
                    bytes += BN * 128;
                    if constexpr (L1) {
                        tma::copy<128, BN, 128, char>(&map_b, &full[s],
                            buffer + B_OFFSET + s * B_BYTES + (r * 2 + 1) * BN * 128,
                            kb * 64, expert * 2 * D + (GATE_UP_SWAPPED ? 0 : D) + nb * BN, 1);
                        bytes += BN * 128;
                    }
                }
                // All four producer lanes arrive, including inactive roles.
                full[s].arrive_and_expect_tx(bytes);
            }
        }
    } else {
        float accum[BN / 8][4] = {}, accum_up[BN / 8][4] = {};
        sm120::SwizzleContext<128> b_ctx[BN / 8];
        #pragma unroll
        for (uint32_t nt = 0; nt < BN / 8; ++nt)
            b_ctx[nt].init((lane & 7) + nt * 8, 128);
        for (uint32_t kb = 0; kb < I / 64; ++kb) {
            const uint32_t s = kb % STAGES, phase = (kb / STAGES) & 1;
            full[s].wait(phase);
            #pragma unroll
            for (uint32_t ks = 0; ks < 4; ++ks) {
                uint32_t a[4];
                // Broadcast one row to all logical MMA rows. Only row zero
                // is stored; this avoids a padded M tile for decode workloads.
                sm120::ldmatrix_x4(a[0], a[1], a[2], a[3],
                    buffer + s * A_BYTES + (L1 ? 0 : warp * 128) + (lane >> 4) * 16 + ks * 32);
                #pragma unroll
                for (uint32_t nt = 0; nt < BN / 8; ++nt) {
                    uint32_t b[2];
                    sm120::load_b_fragment_x2(b,
                        buffer + B_OFFSET + s * B_BYTES + warp * (L1 ? 2 : 1) * BN * 128,
                        b_ctx[nt], lane, ks, 32);
                    mma::sm120::bf16_mma(accum[nt], a, b);
                    if constexpr (L1) {
                        sm120::load_b_fragment_x2(b,
                            buffer + B_OFFSET + s * B_BYTES + (warp * 2 + 1) * BN * 128,
                            b_ctx[nt], lane, ks, 32);
                        mma::sm120::bf16_mma(accum_up[nt], a, b);
                    }
                }
            }
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (lane == 0) empty[s].arrive();
        }
        uint64_t slot, expert;
        if constexpr (EXPERTS > 0) {
            const int64_t id = slots[route_base + warp];
            const bool valid = id >= 0 && id < int64_t(EXPERTS);
            expert = valid ? uint64_t(id) : 0;
            slot = valid ? expert * capacity : route_base + warp + 1;
        } else {
            slot = slots[token * TOPK + route_base + warp];
            expert = slot / capacity;
        }
        const auto* row_a = original_a + (L1 ? token : slot) * I;
        // Every math lane participates in the midpoint ballot and cooperative
        // refinement. Only lanes 0..3 own the logical row that is stored.
        #pragma unroll
        for (uint32_t nt = 0; nt < BN / 8; ++nt) {
            #pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                const uint32_t col = nb * BN + nt * 8 + (lane < 4 ? lane * 2 + j : 0);
                const auto* row_b = original_b +
                    (expert * (L1 ? 2 : 1) * D + (GATE_UP_SWAPPED ? D : 0) + col) * I;
                accum[nt][j] = mma::sm120::bf16_refine_midpoint(accum[nt][j], row_a, row_b, I, lane < 4);
                if constexpr (L1)
                    accum_up[nt][j] = mma::sm120::bf16_refine_midpoint(
                        accum_up[nt][j], row_a,
                        GATE_UP_SWAPPED ? row_b - uint64_t(D) * I : row_b + uint64_t(D) * I,
                        I, lane < 4);
            }
        }
        if (lane < 4) {
            #pragma unroll
            for (uint32_t nt = 0; nt < BN / 8; ++nt) {
                const uint32_t col = nt * 8 + lane * 2;
                if constexpr (L1) {
                    float value[2];
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++j) {
                        // Preserve materialized BF16 gate/up boundaries and
                        // the standalone Triton sigmoid's exp2/reciprocal path.
                        const float gate = __bfloat162float(__float2bfloat16_rn(accum[nt][j]));
                        const float up = __bfloat162float(__float2bfloat16_rn(accum_up[nt][j]));
                        const float scaled = __fmul_rn(gate, -1.4426950408889634f);
                        float exponential, sigmoid;
                        asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(scaled));
                        const float denom = __fadd_rn(1.f, exponential);
                        asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(sigmoid) : "f"(denom));
                        value[j] = __fmul_rn(__fmul_rn(gate, sigmoid), up);
                    }
                    *reinterpret_cast<__nv_bfloat162*>(output + slot * D + nb * BN + col) =
                        __floats2bfloat162_rn(value[0], value[1]);
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(partial + warp * BN + col) =
                        __floats2bfloat162_rn(accum[nt][0], accum[nt][1]);
                }
            }
        }
    }
    if constexpr (!L1) {
        __syncthreads();
        if (threadIdx.x < BN / 2) {
            const uint32_t col = threadIdx.x * 2;
            float x = 0.f, y = 0.f;
            #pragma unroll
            for (uint32_t r = 0; r < TOPK; ++r) {
                if constexpr (EXPERTS > 0) {
                    const int64_t id = slots[r];
                    // Do not consume padding weights or contributions. This
                    // also masks NaN sentinels without relying on 0 * NaN.
                    if (id < 0 || id >= int64_t(EXPERTS)) continue;
                }
                const float2 v = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(partial + r * BN + col));
                const float weight = weights[token * TOPK + r];
                x = __fadd_rn(x, __fmul_rn(v.x, weight));
                y = __fadd_rn(y, __fmul_rn(v.y, weight));
            }
            *reinterpret_cast<__nv_bfloat162*>(output + uint64_t(token) * D + nb * BN + col) =
                __floats2bfloat162_rn(x, y);
        }
    }
#endif
}

} // namespace deep_gemm
