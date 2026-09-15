#pragma once

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/nv_moe_workspace.cuh>

namespace deep_gemm::layout::nvfp4 {

using nv_moe::Workspace;
using nv_moe::TokenSrcMetadata;
using nv_moe::kCandidateBlockM;
using nv_moe::kNumCandidateBlockMs;
using nv_moe::kLCMCandidateBlockM;
using nv_moe::get_num_sf_ring_tokens;

struct MegaMoEBuffer {
    Workspace workspace;
    Buffer input_token_buffer, input_sf_buffer,
           input_topk_idx_buffer, input_topk_weights_buffer;
    Buffer shared_l2_token_buffer;
    Buffer l1_token_buffer, l1_sf_buffer, l2_token_buffer, l2_sf_buffer;
    Buffer combine_token_buffer, routed_topk_weights_buffer;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const uint32_t& num_shared_experts = 0) {
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto input_token_layout = Data(hidden / 2);
        const auto intermediate_token_layout = Data(intermediate_hidden / 2);
        const auto input_sf_layout = Data(hidden / 16);
        const auto intermediate_sf_layout = Data(intermediate_hidden / 16);

        routed_topk_weights_buffer = Buffer(
            Data(sizeof(float), false), 1, workspace.num_max_pool_tokens,
            workspace.get_end_ptr());
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            routed_topk_weights_buffer.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            Data(num_topk * sizeof(int64_t), false), 1, num_max_tokens_per_rank,
            input_sf_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            Data(num_topk * sizeof(float), false), 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        shared_l2_token_buffer = Buffer(
            Data(shared_intermediate_hidden * 2), 1,
            num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            input_topk_weights_buffer.get_end_ptr());
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            shared_l2_token_buffer.get_end_ptr());
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_sf_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());
        combine_token_buffer = Buffer(
            Data(hidden * 2), num_topk + (num_shared_experts > 0 ? 1u : 0u),
            num_max_tokens_per_rank, l2_sf_buffer.get_end_ptr());
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
               - static_cast<uint8_t*>(workspace.base);
    }
};

} // namespace deep_gemm::layout::nvfp4
