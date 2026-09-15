#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/nvfp4_mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/nvfp4_mega_moe.hpp"

namespace deep_gemm::nvfp4 {

static void sm100_fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    // BF16 shared expert tensors: `shared_l1_acts` is the caller-provided BF16 `x`,
    // `shared_l2_acts` is the BF16 intermediate view sliced from the symmetric buffer
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l2_acts,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const void* l1_alphas, const void* l2_alphas, const void* a2_scales,
    const float& routed_scaling_factor
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens);

    // View all packed-FP4 tensors as raw bytes: TMA moves plain byte rows, so the
    // K dimensions below are all `elements / 2`
    constexpr int kGranK = 16;
    const auto l1_weights_bytes = l1_weights.view(torch::kUInt8);
    const auto l2_weights_bytes = l2_weights.view(torch::kUInt8);
    const int block_k_bytes = config.block_k / 2;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);

    // Make tensormap
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden / 2, config.num_ring_tokens,
                                                     block_k_bytes, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights_bytes,
                                                        hidden / 2, num_experts_per_rank * intermediate_hidden * 2,
                                                        block_k_bytes, config.load_block_n,
                                                        static_cast<int>(l1_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` elements = `BLOCK_N / 4`
    // bytes per input tile); the store box is small, so no swizzling is applied
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden / 2, config.num_ring_tokens,
                                                       config.block_n / 4, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden / 2, config.num_ring_tokens,
                                                     block_k_bytes, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights_bytes,
                                                        intermediate_hidden / 2, num_experts_per_rank * hidden,
                                                        block_k_bytes, config.load_block_n,
                                                        static_cast<int>(l2_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim);

    // BF16 shared expert descriptors: same 128B tiles hold `block_k / 4` BF16 elements.
    // With no shared work they fall back to routed maps so an empty local-token tensor
    // never passes a null base pointer to cuTensorMapEncodeTiled.
    const int shared_block_k = config.block_k / 4;
    const bool has_shared_work = num_shared_experts > 0 and num_tokens > 0;
    const auto tensor_map_shared_l1_acts = has_shared_work ? make_tma_2d_desc(
        shared_l1_acts,
        // `shared_l1_acts` is the caller's current-token tensor, not a
        // capacity-sized symmetric buffer.  Describe its real row extent so
        // TMA zero-fills the partial M tile instead of reading past storage.
        hidden, num_tokens,
        shared_block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_weights = has_shared_work ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        shared_block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = has_shared_work ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = has_shared_work ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        shared_block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_weights = has_shared_work ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        shared_block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = runtime->get_num_sms();
    const auto kernel = jit->compile("sm100_fp4_fp4_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh>

using namespace deep_gemm::nvfp4;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m,
        config.sf_block_m, config.sf_block_n,
        config.num_ring_tokens,
        config.num_sf_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false"));

    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        routed_scaling_factor,
        l1_alphas, l2_alphas, a2_scales,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_weights);
}

} // namespace deep_gemm::nvfp4
