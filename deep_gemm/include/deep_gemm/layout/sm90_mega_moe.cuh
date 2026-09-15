#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/compile.cuh>

namespace deep_gemm::layout {

// Shared by host validation and the device combine implementation so shape
// errors are reported before JIT compilation.
inline constexpr uint32_t kSM90MoeCombineElemsPerLane = 8;
inline constexpr uint32_t kSM90MoeCombineMaxRegistersPerLane = 128;

CUTLASS_HOST_DEVICE constexpr uint32_t get_sm90_moe_combine_num_chunks(
    const uint32_t hidden,
    const uint32_t num_epilogue_warps,
    const uint32_t smem_before_barriers,
    const uint32_t input_element_bytes,
    const uint32_t output_element_bytes) {
    const uint64_t one_chunk_bytes =
        static_cast<uint64_t>(num_epilogue_warps) * hidden *
        (2u * input_element_bytes + output_element_bytes);
    const bool one_chunk_fits =
        one_chunk_bytes <= smem_before_barriers and
        hidden <= 32u * kSM90MoeCombineMaxRegistersPerLane;
    const bool two_chunks_fit =
        hidden % 2u == 0u and one_chunk_bytes / 2u <= smem_before_barriers and
        hidden <= 2u * 32u * kSM90MoeCombineMaxRegistersPerLane;
    return one_chunk_fits ? 1u : (two_chunks_fit ? 2u : 4u);
}

CUTLASS_HOST_DEVICE constexpr bool is_sm90_moe_combine_vectorization_legal(
    const uint32_t hidden,
    const uint32_t num_epilogue_warps,
    const uint32_t smem_before_barriers,
    const uint32_t input_element_bytes,
    const uint32_t output_element_bytes) {
    const auto num_chunks = get_sm90_moe_combine_num_chunks(
        hidden, num_epilogue_warps, smem_before_barriers,
        input_element_bytes, output_element_bytes);
    const uint64_t selected_chunk_bytes =
        static_cast<uint64_t>(num_epilogue_warps) * hidden *
        (2u * input_element_bytes + output_element_bytes) / num_chunks;
    return hidden > 0u and selected_chunk_bytes <= smem_before_barriers and
           hidden % num_chunks == 0u and
           (hidden / num_chunks) % (32u * kSM90MoeCombineElemsPerLane) == 0u;
}

struct Sm90MoeWarpgroupLayout {
    bool split_n;
    uint32_t split_m, split_n_count;
    uint32_t block_m, block_n;
};

CUTLASS_HOST_DEVICE constexpr Sm90MoeWarpgroupLayout get_sm90_moe_warpgroup_layout(
    const uint32_t cta_block_m,
    const uint32_t cta_block_n,
    const uint32_t num_epilogue_warpgroups) {
    const bool split_n =
        cta_block_m == 64 and num_epilogue_warpgroups > 1 and
        cta_block_n % num_epilogue_warpgroups == 0 and
        (cta_block_n / num_epilogue_warpgroups == 64 or
         cta_block_n / num_epilogue_warpgroups == 128);
    const uint32_t split_m = split_n ? 1 : num_epilogue_warpgroups;
    const uint32_t split_n_count = split_n ? num_epilogue_warpgroups : 1;
    return {
        split_n,
        split_m,
        split_n_count,
        split_m == 0 ? 0 : cta_block_m / split_m,
        split_n_count == 0 ? 0 : cta_block_n / split_n_count,
    };
}

} // namespace deep_gemm::layout
