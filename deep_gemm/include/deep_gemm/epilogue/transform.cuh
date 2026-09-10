#pragma once

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::epilogue::transform {

// Epilogue operators add behavior (never state) to `EpilogueArgs`, so kernels take
// the host-constructed operator directly as a kernel argument
// NOTES: the operators do not compose with each other
struct EpilogueIdentity: EpilogueArgs {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }

    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&)[kNumValues]) const {}
};

// Scale only the product term of a BLAS-style linear combination. Supporting an
// arbitrary beta requires a separate C load/initialization path and does not belong
// in this value-only transform.
struct EpilogueWithAlpha: EpilogueIdentity {
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

template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits: EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        DG_STATIC_ASSERT(kLeft % STORE_BLOCK_N == 0 and kMid % STORE_BLOCK_N == 0 and
                         kRight % STORE_BLOCK_N == 0, "Invalid head splits config");
        return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
    }
};

// Cast D into FP8 with dynamic per-row, per-32-column UE8M0 SFs, packed into `uint32_t`
// words in a TMA-aligned MN-major layout (the same layout accepted for SFA)
// NOTES: the accumulator is rounded into BF16 before the amax/scale/cast steps, so the
//        output bitwise matches a BF16 D followed by the standalone per-token cast kernel
struct EpilogueDynamicScaledFP8: EpilogueIdentity {
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

} // namespace deep_gemm::epilogue::transform
