#pragma once

#include <deep_gemm/epilogue/sm100_mega_mhc.cuh>

namespace deep_gemm::epilogue::sm90_mega_mhc {

using namespace layout::mega_mhc;

// One WG owns one decode token. Four warps reduce disjoint split residues;
// warp 0 then runs the unchanged affine/Sinkhorn epilogue.
template <uint32_t kHidden, uint32_t kNumSplits, typename Storage>
CUTLASS_DEVICE void run_shifted_mix_task_wg(
    Storage& smem, const Workspace<kNumSplits>& workspace, uint64_t* barriers,
    const MixArgs& mix_args, const uint32_t token_idx,
    const uint32_t warp_idx_in_wg, const uint32_t lane_idx) {
    constexpr uint32_t kMixBarrier = 5;
    const uint32_t m_block_idx = token_idx / BLOCK_M;
    const uint32_t row_idx = token_idx % BLOCK_M;
    if (warp_idx_in_wg == 0)
        sched::mega_mhc::Mix::wait<kNumSplits>(barriers, m_block_idx);
    // Transfer the acquired producer writes to all participating warps.
    cutlass::arch::NamedBarrier::sync(128, kMixBarrier);

    const uint32_t first_task_idx = m_block_idx * kNumSplits;
    const auto stats = workspace.get_hc_norm_sqr_sum_partial_ptr(first_task_idx);
    float hc_norm_sqr_sum = 0.f;
    for (uint32_t k = warp_idx_in_wg * 32 + lane_idx; k < kNumSplits; k += 128)
        hc_norm_sqr_sum += stats[k * kNumSqrSumPartialElementsPerTask + row_idx];
    hc_norm_sqr_sum = math::warp_reduce_sum(hc_norm_sqr_sum);

    const bool is_output_lane = lane_idx < kNumHCOutputs;
    const uint32_t output_idx = cute::min(lane_idx, kNumHCOutputs - 1);
    const auto partials = workspace.get_gemm_partial_ptr(first_task_idx);
    float hc_output = 0.f;
    #pragma unroll
    for (uint32_t round = 0; round < (kNumSplits + 3) / 4; ++round) {
        const uint32_t k = round * 4 + warp_idx_in_wg;
        if (k < kNumSplits)
            hc_output += partials[k * kNumGemmPartialElementsPerTask + row_idx * kNumHCOutputs + output_idx];
    }
    if (is_output_lane)
        smem.mix_scratch[warp_idx_in_wg][lane_idx] = hc_output;
    if (lane_idx == 0)
        smem.mix_scratch[warp_idx_in_wg][24] = hc_norm_sqr_sum;
    cutlass::arch::NamedBarrier::sync(128, kMixBarrier);

    if (warp_idx_in_wg == 0) {
        hc_output = hc_norm_sqr_sum = 0.f;
        #pragma unroll
        for (uint32_t w = 0; w < kNumWarpsPerWG; ++w) {
            if (is_output_lane)
                hc_output += smem.mix_scratch[w][lane_idx];
            hc_norm_sqr_sum += smem.mix_scratch[w][24];
        }
    }
    if (warp_idx_in_wg == 0)
        mega_mhc::finish_mix<kHidden, true>(hc_output, hc_norm_sqr_sum, mix_args, token_idx, lane_idx);

    // The next token may reuse scratch only after warp 0 finishes reading it.
    cutlass::arch::NamedBarrier::sync(128, kMixBarrier);
}

} // namespace deep_gemm::epilogue::sm90_mega_mhc
