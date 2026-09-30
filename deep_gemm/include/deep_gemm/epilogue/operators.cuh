#pragma once

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::epilogue::operators {

// Epilogue operators add behavior (never state) to `EpilogueOperatorArgs`, so kernels take
// the host-constructed operator directly as a kernel argument
// NOTES: the operators do not compose with each other
struct Identity: EpilogueOperatorArgs {
    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&)[kNumValues]) const {}
};

// Scale only the product term of a BLAS-style linear combination. Supporting an
// arbitrary beta requires a separate C load/initialization path and does not belong
// in this value-only operator.
struct ScaleByAlpha: Identity {
    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&values)[kNumValues]) const {
        DG_STATIC_ASSERT(kNumValues % 2 == 0, "Alpha scaling requires float2 alignment");
        const auto values_f32x2 = reinterpret_cast<float2*>(values);
        const auto alpha_f32x2 = make_float2(alpha, alpha);
        #pragma unroll
        for (uint32_t value_idx = 0; value_idx < kNumValues / 2; ++ value_idx)
            values_f32x2[value_idx] = __fmul2_rn(values_f32x2[value_idx], alpha_f32x2);
    }
};

struct StochasticRoundToBF16: Identity {
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

    CUTLASS_DEVICE static void cast_quartet(const uint32_t& a, const uint32_t& b,
                                            const uint32_t& c, const uint32_t& d,
                                            const uint32_t& quartet_n_idx,
                                            uint32_t& packed_ab, uint32_t& packed_cd) {
        const auto h = partial_hash(a, 0) + partial_hash(b, 1) +
                       partial_hash(c, 2) + partial_hash(d, 3) + quartet_n_idx;
        packed_ab = ptx::cvt_rs_bf16x2_f32(*reinterpret_cast<const float*>(&a),
                                           *reinterpret_cast<const float*>(&b),
                                           select_random_bits(h, 0) | select_random_bits(h, 1) << 16);
        packed_cd = ptx::cvt_rs_bf16x2_f32(*reinterpret_cast<const float*>(&c),
                                           *reinterpret_cast<const float*>(&d),
                                           select_random_bits(h, 2) | select_random_bits(h, 3) << 16);
    }
};

// Quantize D into FP8 with dynamic per-row, per-32-column UE8M0 SFs, packed into `uint32_t`
// words in a TMA-aligned MN-major layout (the same layout accepted for SFA)
// NOTES: the accumulator is rounded into BF16 before the amax/scale/cast steps, so the
//        output bitwise matches a BF16 D followed by the standalone per-token cast kernel
struct QuantizeToFP8: Identity {
    static constexpr uint32_t kSFGranN = 32;

    // Store one SF byte (`uint8_t`), or all four SF bytes of one packed word at once
    // (`uint32_t`, fully coalesced; only valid when `shape_n % 128 == 0`, so that a word
    // never crosses a batch or shape boundary)
    // NOTES: batched GEMMs flatten the SF columns of all batches along `(batch_idx, n)`,
    //        matching a D viewed by the consumer as a `(shape_m, num_batches * shape_n)` 2D tensor
    template <typename sf_t>
    CUTLASS_DEVICE void store_sf(const uint32_t& row_idx, const uint32_t& group_n_idx,
                                 const uint32_t& batch_idx, const sf_t& sf) const {
        DG_STATIC_ASSERT(cute::is_same_v<sf_t, uint8_t> or cute::is_same_v<sf_t, uint32_t>,
                         "The SF must be a single byte or a whole packed word");
        if (row_idx >= shape_m or group_n_idx >= shape_n)
            return;
        const auto sf_idx = (batch_idx * shape_n + group_n_idx) / kSFGranN;
        const auto sf_word_ptr = sfd + (sf_idx / 4) * sfd_stride + row_idx;
        if constexpr (cute::is_same_v<sf_t, uint32_t>) {
            *sf_word_ptr = sf;
        } else {
            reinterpret_cast<uint8_t*>(sf_word_ptr)[sf_idx % 4] = sf;
        }
    }
};

} // namespace deep_gemm::epilogue::operators
