#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    CUTLASS_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    CUTLASS_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

template <uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}

template <typename T>
__device__ __forceinline__ T shfl_sync(unsigned mask, T var, int srcLane, int width = 32) {

    using shfl_t = std::conditional_t<sizeof(T) == 4, int,
                   std::conditional_t<sizeof(T) == 8, long long, long long>>;

    T result;
    shfl_t* var_ptr = reinterpret_cast<shfl_t*>(&var);
    shfl_t* result_ptr = reinterpret_cast<shfl_t*>(&result);
    *result_ptr = __shfl_sync(mask, *var_ptr, srcLane, width);

    if constexpr (sizeof(T) == 16) {
        *(result_ptr + 1) = __shfl_sync(mask, *(var_ptr + 1), srcLane, width);
    }

    return result;
}


// --- Round-robin "peel" indexing over per-rank token counts (shared by the split-kernel
// dispatch pull). Maps a flat pool slot <-> (rank, token-in-rank) for the contiguous
// expert token pool. ---
template <uint32_t kNumRanks_>
struct PeelIter {
    static constexpr uint32_t kNumRanks = kNumRanks_;
    static constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);

    uint32_t remaining[kNumRanksPerLane];
    uint32_t slot_base;
    uint32_t row_base;
    uint32_t num_active_ranks;
    uint32_t round_depth;

    CUTLASS_DEVICE explicit PeelIter(const uint32_t (&counts)[kNumRanksPerLane])
        : slot_base(0), row_base(0)
    {
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
            remaining[i] = counts[i];
        compute_round();
    }

    CUTLASS_DEVICE void compute_round() {
        uint32_t in_lane_actives = 0;
        uint32_t in_lane_min = 0xffffffffu;
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
            in_lane_actives += remaining[i] > 0;
            if (remaining[i] > 0)
                in_lane_min = remaining[i] < in_lane_min ? remaining[i] : in_lane_min;
        }
        num_active_ranks = __reduce_add_sync(0xffffffffu, in_lane_actives);
        round_depth = __reduce_min_sync(0xffffffffu, in_lane_min);
    }

    CUTLASS_DEVICE void advance() {
        slot_base += round_depth * num_active_ranks;
        row_base += round_depth;
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
            remaining[i] -= remaining[i] < round_depth ? remaining[i] : round_depth;
        compute_round();
    }

    CUTLASS_DEVICE uint32_t select_active_rank(const uint32_t& active_rank_idx) const {
        uint32_t rank = 0;
        uint32_t seen = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
            const uint32_t mask = __ballot_sync(0xffffffffu, remaining[i] > 0);
            const uint32_t active_lanes = __popc(mask);
            if (active_rank_idx >= seen and active_rank_idx < seen + active_lanes)
                rank = i * 32 + __fns(mask, 0, active_rank_idx - seen + 1);
            seen += active_lanes;
        }
        return rank;
    }

    CUTLASS_DEVICE uint32_t rank_order_of(const uint32_t& target_rank) const {
        uint32_t order = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
            const uint32_t mask = __ballot_sync(0xffffffffu, remaining[i] > 0);
            if (target_rank >= i * 32 + 32) {
                order += __popc(mask);
            } else if (target_rank >= i * 32) {
                order += __popc(mask & ((1u << (target_rank - i * 32)) - 1u));
            }
        }
        return order;
    }
};

template <uint32_t kNumRanks>
CUTLASS_DEVICE void peel_forward(
    const uint32_t (&counts)[math::constexpr_ceil_div(kNumRanks, 32u)],
    const uint32_t& slot_idx,
    uint32_t& out_rank,
    uint32_t& out_token_idx_in_rank)
{
    PeelIter<kNumRanks> it(counts);
    while (slot_idx >= it.slot_base + it.round_depth * it.num_active_ranks)
        it.advance();
    const uint32_t local_slot_idx = slot_idx - it.slot_base;
    out_rank = it.select_active_rank(local_slot_idx % it.num_active_ranks);
    out_token_idx_in_rank = it.row_base + local_slot_idx / it.num_active_ranks;
}

template <uint32_t kNumRanks>
CUTLASS_DEVICE uint32_t peel_inverse(
    const uint32_t (&counts)[math::constexpr_ceil_div(kNumRanks, 32u)],
    const uint32_t& target_rank,
    const uint32_t& target_token_idx_in_rank)
{
    PeelIter<kNumRanks> it(counts);
    while (target_token_idx_in_rank >= it.row_base + it.round_depth)
        it.advance();
    return it.slot_base +
           (target_token_idx_in_rank - it.row_base) * it.num_active_ranks +
           it.rank_order_of(target_rank);
}

} // namespace deep_gemm::utils
