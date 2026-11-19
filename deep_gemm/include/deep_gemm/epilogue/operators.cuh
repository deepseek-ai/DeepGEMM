#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::epilogue::operators {

// Identity also provides shared parameters, lifecycle hooks and default BF16 conversion.
// NOTES: the operators do not compose with each other
struct Identity {
    const EpilogueOperatorArgs& args;

    static constexpr bool kIsFP8CastWithClamp = false;
    static constexpr bool kWithStochasticRounding = false;

    CUTLASS_DEVICE explicit Identity(const EpilogueOperatorArgs& args): args(args) {}

    CUTLASS_DEVICE void init(const uint32_t&, const uint32_t&) {}

    CUTLASS_DEVICE void finalize() const {}

    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&)[kNumValues]) const {}

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

    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&values)[kNumValues]) const {
        DG_STATIC_ASSERT(kNumValues % 2 == 0, "Alpha scaling requires float2 alignment");
        const auto values_f32x2 = reinterpret_cast<float2*>(values);
        const auto alpha_f32x2 = make_float2(args.alpha, args.alpha);
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues / 2; ++ value_idx)
            values_f32x2[value_idx] = __fmul2_rn(values_f32x2[value_idx], alpha_f32x2);
    }
};

struct StochasticRoundToBF16: Identity {
private:
    // The quartet hash is the sum of the four members' partial terms plus `quartet_n_idx`;
    // each member then finalizes the sum and selects its own 16-bit random half
    CUTLASS_DEVICE static uint32_t partial_hash(const uint32_t& value, const uint32_t& quartet_offset) {
        return value * (quartet_offset == 0 ? 0x5671d42bu :
                        quartet_offset == 1 ? 0x9995e499u :
                        quartet_offset == 2 ? 0xace1b8a5u : 0xe153538du);
    }

    CUTLASS_DEVICE static uint32_t select_random_bits(uint32_t h, const uint32_t& quartet_offset) {
        h ^= h >> 23;
        h *= 0x7feb352du;
        h ^= h >> 16;
        h *= quartet_offset < 2 ? 0x846ca68bu : 0xd35a2d97u;
        h ^= h >> 11;
        return (h >> (quartet_offset % 2 * 16)) & 0xffffu;
    }

public:
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
            hashes[value_idx] = partial_hash(values[value_idx], kSwapAB ? quartet_offset : value_idx % 4);
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
                select_random_bits(hash_lower + quartet_n_idx, offset) |
                select_random_bits(hash_upper + quartet_n_idx, offset + (kSwapAB ? 0 : 1)) << 16);
        }
    }
};

// Quantize D into FP8 with dynamic per-row, per-32-column UE8M0 SFs, packed into `uint32_t`
// words in a TMA-aligned MN-major layout (the same layout accepted for SFA)
// NOTES: the accumulator is rounded into BF16 before the amax/scale/cast steps, so the
//        output bitwise matches a BF16 D followed by the standalone per-token cast kernel
struct CastToFP8WithClamp: Identity {
private:
    uint32_t num_clamped = 0;
    bool count_enabled = false;

public:
    static constexpr bool kIsFP8CastWithClamp = true;
    static constexpr uint32_t kSFGranN = 32;

    using Identity::Identity;

    // Reset while the MMA pipeline starts; idle CTAs need no counter work.
    CUTLASS_DEVICE void init(const uint32_t& num_blocks, const uint32_t& epilogue_thread_idx) {
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

} // namespace deep_gemm::epilogue::operators
