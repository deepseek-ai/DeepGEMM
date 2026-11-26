#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::epilogue::operators {

// Identity also provides shared parameters, lifecycle hooks and default BF16 conversion.
struct Identity {
    const EpilogueOperatorArgs& args;

    static constexpr bool kIsFP8Output = false;
    static constexpr bool kWithStochasticRounding = false;
    static constexpr bool kWithBias = false;

    CUTLASS_DEVICE explicit Identity(const EpilogueOperatorArgs& args): args(args) {}

    template <bool kSwapAB, bool kReverseStore, typename scheduler_t>
    CUTLASS_DEVICE void init(const scheduler_t&, const uint32_t&, const uint32_t&) {}

    template <bool kSwapAB, bool kReverseStore, typename scheduler_t>
    CUTLASS_DEVICE void prefetch(const scheduler_t&) {}

    CUTLASS_DEVICE void finalize() const {}

    template <bool kSwapAB, uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&)[kNumValues], const uint32_t&) const {}

    template <bool kSwapAB>
    CUTLASS_DEVICE void cast_bf16(const uint32_t (&values)[8], uint32_t (&packed)[4],
                                  const uint32_t&, const uint32_t&) const {
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < 4; ++ pair_idx)
            packed[pair_idx] = math::cast_into_bf16_and_pack(values[pair_idx * 2], values[pair_idx * 2 + 1]);
    }
};

// Scale only the product term of a BLAS-style linear combination. Supporting an
// arbitrary beta requires a separate C load/initialization path and does not belong
// in this value-only operator.
struct ScaleByAlpha: Identity {
    using Identity::Identity;

    template <bool kSwapAB, uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&values)[kNumValues], const uint32_t&) const {
        DG_STATIC_ASSERT(kNumValues % 2 == 0, "Alpha scaling requires float2 alignment");
        const auto values_f32x2 = reinterpret_cast<float2*>(values);
        const auto alpha_f32x2 = make_float2(args.alpha, args.alpha);
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues / 2; ++ value_idx)
            values_f32x2[value_idx] = __fmul2_rn(values_f32x2[value_idx], alpha_f32x2);
    }
};

template <typename cd_dtype_t, uint32_t BLOCK_N = 0>
struct AddBias: Identity {
    using Identity::Identity;
    static constexpr bool kWithBias = true;

private:
    uint32_t bias_offset, reverse_bias_offset;
    static constexpr uint32_t kNumPacked = sizeof(uint32_t) / sizeof(cd_dtype_t);
    static constexpr uint32_t kNumRegisters = (BLOCK_N + 32 * kNumPacked - 1) / (32 * kNumPacked);
    uint32_t bias_registers[kNumRegisters < 4 ? 4 : kNumRegisters];
    uint32_t next_bias_registers[kNumRegisters < 4 ? 4 : kNumRegisters];

public:
    template <bool kSwapAB, bool kReverseStore, typename scheduler_t>
    CUTLASS_DEVICE void init(const scheduler_t& scheduler, const uint32_t&, const uint32_t& epilogue_thread_idx) {
        const auto warp_idx = epilogue_thread_idx / 32;
        const auto lane_idx = epilogue_thread_idx % 32;
        if constexpr (kSwapAB) {
            bias_offset = warp_idx * 32 + (cute::is_same_v<cd_dtype_t, float> ? lane_idx : lane_idx / 4);
        } else {
            constexpr uint32_t kNumElemsPerLoad = 16 / sizeof(cd_dtype_t);
            bias_offset = lane_idx * kNumPacked;
            reverse_bias_offset = (BLOCK_N / kNumElemsPerLoad - 1 - bias_offset / kNumElemsPerLoad) * kNumElemsPerLoad +
                                  bias_offset % kNumElemsPerLoad;
        }
        prefetch_next<kSwapAB, kReverseStore>(scheduler);
    }

    template <bool kSwapAB, bool kReverseStore, typename scheduler_t>
    CUTLASS_DEVICE void prefetch(const scheduler_t& scheduler) {
        constexpr bool kFP32 = cute::is_same_v<cd_dtype_t, float>;
        constexpr uint32_t kNumValues = kSwapAB ? (kFP32 ? 1 : 4) : kNumRegisters;
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx)
            bias_registers[value_idx] = kSwapAB and not kFP32 ? next_bias_registers[value_idx] << 16 : next_bias_registers[value_idx];
        prefetch_next<kSwapAB, kReverseStore>(scheduler);
    }

private:
    template <bool kSwapAB, bool kReverseStore, typename scheduler_t>
    CUTLASS_DEVICE void prefetch_next(const scheduler_t& scheduler) {
        auto next_scheduler = scheduler;
        uint32_t next_m_block_idx, next_n_block_idx;
        if (next_scheduler.get_next_block(next_m_block_idx, next_n_block_idx))
            load<kSwapAB>(next_n_block_idx * BLOCK_N, kReverseStore and next_scheduler.current_iter % 2 == 0);
    }

    template <bool kSwapAB>
    CUTLASS_DEVICE void load(const uint32_t& n_idx, const bool& reverse) {
        const auto bias_ptr = static_cast<const cd_dtype_t*>(args.bias) + n_idx;
        constexpr bool kFP32 = cute::is_same_v<cd_dtype_t, float>;
        constexpr uint32_t kNumValues = kSwapAB ? (kFP32 ? 1 : 4) : kNumRegisters;
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues; ++ value_idx) {
            if constexpr (kSwapAB) {
                const auto col_idx = bias_offset + (kFP32 ? 0 : value_idx * 8);
                if constexpr (kFP32)
                    next_bias_registers[value_idx] = n_idx + col_idx < args.shape_n ?
                        reinterpret_cast<const uint32_t*>(bias_ptr)[col_idx] : 0u;
                else
                    next_bias_registers[value_idx] = n_idx + col_idx < args.shape_n ?
                        reinterpret_cast<const uint16_t*>(bias_ptr)[col_idx] : 0u;
            } else {
                const auto col_idx = reverse ? reverse_bias_offset - value_idx * 32 * kNumPacked :
                                               bias_offset + value_idx * 32 * kNumPacked;
                uint32_t packed = 0;
                if (col_idx < BLOCK_N and n_idx + col_idx + kNumPacked <= args.shape_n and reinterpret_cast<uintptr_t>(bias_ptr) % 4 == 0) {
                    packed = reinterpret_cast<const uint32_t*>(bias_ptr)[col_idx / kNumPacked];
                } else if (col_idx < BLOCK_N) {
                    #pragma unroll
                    for (uint32_t packed_idx = 0; packed_idx < kNumPacked; ++ packed_idx) {
                        if (n_idx + col_idx + packed_idx < args.shape_n) {
                            if constexpr (kFP32)
                                packed = reinterpret_cast<const uint32_t*>(bias_ptr)[col_idx];
                            else
                                packed |= static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(bias_ptr)[col_idx + packed_idx]) << (packed_idx * 16);
                        }
                    }
                }
                next_bias_registers[value_idx] = packed;
            }
        }
    }

public:
    template <bool kSwapAB, uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&values)[kNumValues], const uint32_t& n_idx) const {
        uint4 bias_values;
        constexpr bool kFP32 = cute::is_same_v<cd_dtype_t, float>;
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < 4; ++ value_idx) {
            if constexpr (kSwapAB)
                reinterpret_cast<uint32_t*>(&bias_values)[value_idx] = bias_registers[kFP32 ? 0 : value_idx];
            else {
                const auto offset = n_idx / (kFP32 ? 1 : 2);
                reinterpret_cast<uint32_t*>(&bias_values)[value_idx] = ptx::exchange(bias_registers[offset / 32], (offset + value_idx) % 32);
            }
        }
        const auto values_f32x2 = reinterpret_cast<float2*>(values);
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumValues / 2; ++ pair_idx) {
            if constexpr (kSwapAB) {
                const auto bias_value = reinterpret_cast<const float*>(&bias_values)[pair_idx];
                values_f32x2[pair_idx] = __fadd2_rn(values_f32x2[pair_idx], make_float2(bias_value, bias_value));
            } else if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                values_f32x2[pair_idx] = __fadd2_rn(values_f32x2[pair_idx], reinterpret_cast<const float2*>(&bias_values)[pair_idx]);
            } else {
                ptx::accumulate(values_f32x2[pair_idx], reinterpret_cast<const nv_bfloat162*>(&bias_values)[pair_idx]);
            }
        }
    }
};

template <typename epilogue_operator_t, uint32_t BLOCK_N>
struct OperatorForBlock {
    using type = epilogue_operator_t;
};

template <typename cd_dtype_t, uint32_t BLOCK_N>
struct OperatorForBlock<AddBias<cd_dtype_t>, BLOCK_N> {
    using type = AddBias<cd_dtype_t, BLOCK_N>;
};

struct StochasticRoundToBF16: Identity {
    using Identity::Identity;
    static constexpr bool kWithStochasticRounding = true;

    // N-quartets are lane-local normally, or span lanes spaced by four with swap-AB.
    template <bool kSwapAB>
    CUTLASS_DEVICE void cast_bf16(const uint32_t (&values)[8], uint32_t (&packed)[4],
                                  const uint32_t& n_idx, const uint32_t& lane_idx) const {
        const auto quartet_offset = lane_idx / 4 % 4;
        uint32_t hashes[8];
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < 8; ++ value_idx)
            hashes[value_idx] = math::partial_hash(values[value_idx], kSwapAB ? quartet_offset : value_idx % 4);
        if constexpr (kSwapAB) {
            #pragma unroll
            for (uint32_t xor_mask = 4; xor_mask <= 8; xor_mask <<= 1) {
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < 8; ++ value_idx)
                    hashes[value_idx] += __shfl_xor_sync(0xffffffffu, hashes[value_idx], xor_mask);
            }
        }
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < 8; value_idx += 2) {
            auto hash_lower = hashes[value_idx], hash_upper = hashes[value_idx + 1];
            if constexpr (not kSwapAB) {
                const auto quartet_idx = value_idx / 4 * 4;
                hash_lower = hash_upper = hashes[quartet_idx] + hashes[quartet_idx + 1] +
                                          hashes[quartet_idx + 2] + hashes[quartet_idx + 3];
            }
            const auto quartet_n_idx = n_idx / 4 + (kSwapAB ? value_idx + lane_idx / 16 : value_idx / 4);
            const auto offset = kSwapAB ? quartet_offset : value_idx % 4;
            packed[value_idx / 2] = ptx::cvt_rs_bf16x2_f32(
                *reinterpret_cast<const float*>(&values[value_idx]),
                *reinterpret_cast<const float*>(&values[value_idx + 1]),
                math::select_random_bits(hash_lower + quartet_n_idx, offset) |
                math::select_random_bits(hash_upper + quartet_n_idx, offset + (kSwapAB ? 0 : 1)) << 16);
        }
    }
};

// Quantize D into FP8 with dynamic per-row, per-32-column UE8M0 SFs, packed into `uint32_t`
// words in a TMA-aligned MN-major layout (the same layout accepted for SFA)
struct CastToFP8Base: Identity {
    static constexpr bool kIsFP8Output = true;
    static constexpr uint32_t kSFGranN = 32;

    using Identity::Identity;

    // Store one SF byte (`uint8_t`), or all four SF bytes of one packed word at once
    // (`uint32_t`, with word-aligned `group_n_idx` and `shape_n % 128 == 0`, so that a word
    // never crosses a batch or shape boundary)
    // NOTES: batched GEMMs flatten the SF columns of all batches along `(batch_idx, n)`,
    //        matching a D viewed by the consumer as a `(shape_m, num_batches * shape_n)` 2D tensor
    template <typename sf_t>
    CUTLASS_DEVICE void store_sf(const uint32_t& row_idx, const uint32_t& group_n_idx,
                                 const uint32_t& batch_idx, const sf_t& sf) const {
        if (row_idx >= args.shape_m or group_n_idx >= args.shape_n)
            return;
        const auto sf_idx = (batch_idx * args.shape_n + group_n_idx) / kSFGranN;
        const auto sf_word_ptr = args.sfd + (sf_idx / 4) * args.sfd_stride + row_idx;
        const auto sf_ptr = reinterpret_cast<uint8_t*>(sf_word_ptr) + sf_idx % 4;
        *reinterpret_cast<sf_t*>(sf_ptr) = sf;
    }
};

// NOTES: the accumulator is rounded into BF16 before the amax/scale/cast steps, so the
//        output bitwise matches a BF16 D followed by the standalone per-token cast kernel
struct CastToFP8WithClamp: CastToFP8Base {
private:
    uint32_t num_clamped = 0;
    bool count_enabled = false;

public:
    using CastToFP8Base::CastToFP8Base;

    // Reset while the MMA pipeline starts; idle CTAs need no counter work.
    template <bool kSwapAB, bool kReverseStore, typename scheduler_t>
    CUTLASS_DEVICE void init(const scheduler_t&, const uint32_t& num_blocks, const uint32_t& epilogue_thread_idx) {
        count_enabled = args.clamped_count != nullptr and blockIdx.x < num_blocks;
        if (not count_enabled or epilogue_thread_idx != 0)
            return;
        const auto grid_tag = ptx::get_grid_idx() + 1;
        const auto ready_ptr = reinterpret_cast<decltype(grid_tag)*>(args.clamp_count_ready);
        if (blockIdx.x == 0) {
            *args.clamped_count = 0;
            ptx::st_rel(ready_ptr, grid_tag);
        } else {
            while (ptx::ld_acq_gpu(ready_ptr) != grid_tag) {}
        }
    }

    // Round and count BF16 values, then compute SFs for the chosen TMEM layout.
    template <bool kSwapAB, uint32_t kNumValues>
    CUTLASS_DEVICE void prepare_fp8(const uint32_t (&values)[kNumValues],
                                    nv_bfloat162 (&values_bf16x2)[kNumValues / 2], uint2& sf_exp,
                                    const uint32_t& row_idx, const uint32_t& col_idx) {
        // An infinite bound makes the unclamped path use the same conversion loop.
        const auto bound = __float2bfloat162_rn(
            args.clamp_value != 0.0f ? args.clamp_value : cute::numeric_limits<float>::infinity());
        // Count while rounding to avoid a runtime branch in the unrolled loop.
        uint32_t packed_count = 0;
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumValues / 2; ++ pair_idx) {
            const auto pair = math::cast_into_bf16x2(values[pair_idx * 2], values[pair_idx * 2 + 1]);
            values_bf16x2[pair_idx] = pair;
            packed_count += __hgtu2_mask(__habs2(pair), bound) & 0x00010001u;
        }

        // Two independent 16-bit counts, one per BF16 component. N is SF-group aligned.
        DG_STATIC_ASSERT(kNumValues / 2 < 65536, "Packed component counts must not overflow");
        if (args.clamp_value != 0.0f and col_idx < args.shape_n) {
            num_clamped += row_idx < args.shape_m ? packed_count & 0xffffu : 0u;
            num_clamped += row_idx + (kSwapAB ? 1 : 0) < args.shape_m ? packed_count >> 16 : 0u;
        }

        nv_bfloat162 amax;
        if constexpr (kSwapAB) {
            amax = math::warp_reduce_max<4, true>(
                __hmax2(__hmax2(__habs2(values_bf16x2[0]), __habs2(values_bf16x2[1])),
                        __hmax2(__habs2(values_bf16x2[2]), __habs2(values_bf16x2[3]))));
        } else {
            amax = __bfloat162bfloat162(math::get_packed_bf16_amax(values_bf16x2));
        }
        // Clamp amax once; defer element clamping until packing so the reduction can start earlier.
        amax = __hmin2_nan(amax, bound);
        sf_exp.x = math::get_ue8m0_sf_exp(amax.x);
        sf_exp.y = kSwapAB ? math::get_ue8m0_sf_exp(amax.y) : sf_exp.x;
    }

    // Clamp, scale and pack a prepared BF16 fragment.
    template <uint32_t kNumPairs>
    CUTLASS_DEVICE void cast_fp8(const nv_bfloat162 (&values_bf16x2)[kNumPairs],
                                 uint32_t (&packed)[kNumPairs / 2], const uint2& sf_exp) const {
        const auto bound = __float2bfloat162_rn(
            args.clamp_value != 0.0f ? args.clamp_value : cute::numeric_limits<float>::infinity());
        const auto sf_inv = nv_bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.x),
                                         math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp.y));
        #pragma unroll
        for (uint32_t packed_idx = 0; packed_idx < kNumPairs / 2; ++ packed_idx) {
            const auto lower = ptx::clamp_bf16x2(values_bf16x2[packed_idx * 2], bound);
            const auto upper = ptx::clamp_bf16x2(values_bf16x2[packed_idx * 2 + 1], bound);
            packed[packed_idx] = math::scale_bf16x2_into_fp8x4(lower, upper, sf_inv, sf_inv);
        }
    }

    // One non-returning atomic per warp.
    CUTLASS_DEVICE void finalize() const {
        if (count_enabled) {
            const auto warp_count = __reduce_add_sync(0xffffffffu, num_clamped);
            if (ptx::get_lane_idx() == 0)
                ptx::red_add(args.clamped_count, warp_count);
        }
    }
};

// Quantize FP32 accumulators into FP8 with dynamic per-`kSFGranN` UE8M0 SFs and stochastic rounding
struct StochasticRoundToFP8: CastToFP8Base {
    static constexpr bool kWithStochasticRounding = true;

    using CastToFP8Base::CastToFP8Base;

    template <uint32_t kNumValues>
    CUTLASS_DEVICE void cast_fp8(const uint32_t (&values)[kNumValues], uint32_t (&packed)[kNumValues / 4],
                                 uint32_t& sf_exp, const uint32_t& col_idx) const {
        const auto values_f32x2 = reinterpret_cast<const float2*>(values);
        float amax = fabsf(values_f32x2[0].x);
        amax = fmaxf(amax, fabsf(values_f32x2[0].y));
        #pragma unroll
        for (uint32_t pair_idx = 1; pair_idx < kNumValues / 2; ++ pair_idx) {
            amax = fmaxf(amax, fabsf(values_f32x2[pair_idx].x));
            amax = fmaxf(amax, fabsf(values_f32x2[pair_idx].y));
        }
        sf_exp = math::get_ue8m0_sf_exp<cutlass::float_e4m3_t>(amax);
        const auto sf_inv_scalar = math::get_ue8m0_sf_inv<float>(sf_exp);
        const auto sf_inv = make_float2(sf_inv_scalar, sf_inv_scalar);
        const auto quartet_n_idx = col_idx / 4;
        #pragma unroll
        for (uint32_t quartet_idx = 0; quartet_idx < kNumValues / 4; ++ quartet_idx) {
            const auto lower = __fmul2_rn(values_f32x2[quartet_idx * 2], sf_inv);
            const auto upper = __fmul2_rn(values_f32x2[quartet_idx * 2 + 1], sf_inv);
            const auto h = math::partial_hash(__float_as_uint(lower.x), 0) +
                           math::partial_hash(__float_as_uint(lower.y), 1) +
                           math::partial_hash(__float_as_uint(upper.x), 2) +
                           math::partial_hash(__float_as_uint(upper.y), 3) +
                           quartet_n_idx + quartet_idx;
            packed[quartet_idx] = ptx::cvt_rs_e4m3x4_f32(lower.x, lower.y, upper.x, upper.y,
                                                         math::select_random_bits<false>(h));
        }
    }

    template <uint32_t kNumValues>
    CUTLASS_DEVICE void prepare_fp8_swap_ab(const uint32_t (&values)[kNumValues],
                                            float2 (&values_f32x2)[kNumValues / 2], uint2& sf_exp) const {
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumValues / 2; ++ pair_idx)
            values_f32x2[pair_idx] = *reinterpret_cast<const float2*>(&values[pair_idx * 2]);

        float amax_lower = fabsf(values_f32x2[0].x);
        float amax_upper = fabsf(values_f32x2[0].y);
        #pragma unroll
        for (uint32_t pair_idx = 1; pair_idx < kNumValues / 2; ++ pair_idx) {
            amax_lower = fmaxf(amax_lower, fabsf(values_f32x2[pair_idx].x));
            amax_upper = fmaxf(amax_upper, fabsf(values_f32x2[pair_idx].y));
        }
        sf_exp.x = math::get_ue8m0_sf_exp<cutlass::float_e4m3_t>(math::warp_reduce_max<4, true>(amax_lower));
        sf_exp.y = math::get_ue8m0_sf_exp<cutlass::float_e4m3_t>(math::warp_reduce_max<4, true>(amax_upper));
    }

    template <uint32_t kNumPairs>
    CUTLASS_DEVICE void cast_fp8_swap_ab(const float2 (&values_f32x2)[kNumPairs], uint32_t (&packed)[kNumPairs / 2],
                                         const uint2& sf_exp, const uint32_t& col_idx, const uint32_t& lane_idx) const {
        // Pack the matching 16-bit hash halves with `rbits_perm` and select the `cvt.rs` slots with `out_perm`
        const auto sf_inv = make_float2(math::get_ue8m0_sf_inv<float>(sf_exp.x),
                                        math::get_ue8m0_sf_inv<float>(sf_exp.y));
        const auto quartet_offset = lane_idx / 4 % 4;
        const auto quartet_n_idx = col_idx / 4 + lane_idx / 16;
        const uint32_t rbits_perm = quartet_offset < 2 ? 0x5410u : 0x7632u;
        const uint32_t out_perm = (quartet_offset & 1u) ? 0x7351u : 0x6240u;
        float2 scaled[kNumPairs];
        uint32_t hashes[kNumPairs * 2];
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            scaled[pair_idx] = __fmul2_rn(values_f32x2[pair_idx], sf_inv);
            hashes[pair_idx * 2] = math::partial_hash(__float_as_uint(scaled[pair_idx].x), quartet_offset);
            hashes[pair_idx * 2 + 1] = math::partial_hash(__float_as_uint(scaled[pair_idx].y), quartet_offset);
        }
        #pragma unroll
        for (uint32_t xor_mask = 4; xor_mask <= 8; xor_mask <<= 1) {
            #pragma unroll
            for (uint32_t value_idx = 0; value_idx < kNumPairs * 2; ++ value_idx)
                hashes[value_idx] += __shfl_xor_sync(0xffffffffu, hashes[value_idx], xor_mask);
        }
        #pragma unroll
        for (uint32_t packed_idx = 0; packed_idx < kNumPairs / 2; ++ packed_idx) {
            const auto base_idx = packed_idx * 4;
            const auto s0 = scaled[packed_idx * 2], s1 = scaled[packed_idx * 2 + 1];
            const auto rbits_even = __byte_perm(
                math::select_random_bits<false>(hashes[base_idx + 0] + quartet_n_idx + base_idx),
                math::select_random_bits<false>(hashes[base_idx + 2] + quartet_n_idx + base_idx + 2),
                rbits_perm);
            const auto rbits_odd = __byte_perm(
                math::select_random_bits<false>(hashes[base_idx + 1] + quartet_n_idx + base_idx),
                math::select_random_bits<false>(hashes[base_idx + 3] + quartet_n_idx + base_idx + 2),
                rbits_perm);
            const auto even = ptx::cvt_rs_e4m3x4_f32(s0.x, s0.x, s1.x, s1.x, rbits_even);
            const auto odd = ptx::cvt_rs_e4m3x4_f32(s0.y, s0.y, s1.y, s1.y, rbits_odd);
            packed[packed_idx] = __byte_perm(even, odd, out_perm);
        }
    }
};

} // namespace deep_gemm::epilogue::operators
