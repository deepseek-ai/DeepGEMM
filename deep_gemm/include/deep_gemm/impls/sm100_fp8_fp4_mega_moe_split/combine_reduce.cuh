#pragma once

#include <cuda_bf16.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_split/common.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_split.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::mega_moe_split {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden,
    uint32_t kIntermediateHidden,
    uint32_t kNumExperts,
    uint32_t kNumTopk,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumRanks,
    uint32_t kNumThreads = 256
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_split_combine_reduce_impl(
    void* y,
    uint32_t* state,
    const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer
) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kNumThreads % 32 == 0, "K3 thread count must be warp-aligned");
    DG_STATIC_ASSERT(kHidden % 8 == 0, "Hidden must be divisible by one uint4 of BF16");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in one warp");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid expert/rank shape");

    constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kNumHiddenVec = kNumHiddenBytes / sizeof(ptx::longlong4_t);
    constexpr uint32_t kNumElemsPerVec = sizeof(ptx::longlong4_t) / sizeof(nv_bfloat162);
    DG_STATIC_ASSERT(kNumHiddenBytes % sizeof(ptx::longlong4_t) == 0,
                     "Hidden must be divisible by one 256-bit vector of BF16");

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    const auto workspace = layout::SplitWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    const auto fp8_token_layout = layout::Data(kHidden);
    const auto fp8_sf_layout = layout::Data(kHidden / 32);
    const auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
    constexpr uint32_t kNumMaxPoolTokens = layout::get_num_max_pool_tokens(
        kNumRanks, kNumMaxTokensPerRank, kNumTopk, kNumExpertsPerRank);
    const auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    const auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    const auto bf16_token_layout = layout::Data(kNumHiddenBytes);

    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr());
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr());

    __shared__ int32_t topk_idx_smem[kNumTopk];
    const auto topk_idx_ptr = input_topk_idx_buffer.get_data_buffer(token_idx)
                                  .get_base_ptr<const int64_t>();
    if (thread_idx < kNumTopk)
        topk_idx_smem[thread_idx] = static_cast<int32_t>(topk_idx_ptr[thread_idx]);
    __syncthreads();

    for (uint32_t vec_idx = thread_idx; vec_idx < kNumHiddenVec; vec_idx += kNumThreads) {
        float2 reduced[kNumElemsPerVec];
        #pragma unroll
        for (uint32_t elem_idx = 0; elem_idx < kNumElemsPerVec; ++elem_idx)
            reduced[elem_idx] = make_float2(0.0f, 0.0f);

        #pragma unroll
        for (uint32_t topk_slot_idx = 0; topk_slot_idx < kNumTopk; ++topk_slot_idx) {
            const auto valid_topk = static_cast<int>(topk_idx_smem[topk_slot_idx]);
            const auto src_ptr = combine_token_buffer.get_rank_buffer(topk_slot_idx)
                                     .get_data_buffer(token_idx)
                                     .get_base_ptr<const ptx::longlong4_t>();
            const auto vec_values = ptx::ld_gez_pred(src_ptr + vec_idx, valid_topk);
            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&vec_values);
            #pragma unroll
            for (uint32_t elem_idx = 0; elem_idx < kNumElemsPerVec; ++elem_idx)
                ptx::accumulate(reduced[elem_idx], bf16_values[elem_idx]);
        }

        ptx::longlong4_t casted;
        const auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
        #pragma unroll
        for (uint32_t elem_idx = 0; elem_idx < kNumElemsPerVec; ++elem_idx)
            casted_bf16[elem_idx] = __float22bfloat162_rn(reduced[elem_idx]);

        const auto dst_ptr = math::advance_ptr<uint4>(
            y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes);
        const auto casted_uint4 = reinterpret_cast<const uint4*>(&casted);
        ptx::st_global_v4_u32(dst_ptr + vec_idx * 2u, casted_uint4[0]);
        ptx::st_global_v4_u32(dst_ptr + vec_idx * 2u + 1u, casted_uint4[1]);
    }

    if (thread_idx == 0)
        atomicAdd(state + get_state_offset(SplitStateOffset::K3DoneElements), 1u);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm::mega_moe_split
