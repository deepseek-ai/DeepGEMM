#pragma once

#include <format>
#include <torch/python.h>

#include <deep_gemm/layout/mega_gate.cuh>

#include "../../runtime/jit.hpp"
#include "../../runtime/runtime.hpp"
#include "../heuristics/mega_gate.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm100_bf16_mega_gate(const torch::Tensor& x, const torch::Tensor& weight,
                                 const std::optional<torch::Tensor>& bias,
                                 const std::optional<torch::Tensor>& image_bias,
                                 const std::optional<torch::Tensor>& image_token_mask,
                                 const std::optional<torch::Tensor>& mask,
                                 const std::optional<torch::Tensor>& fix_routing_mask,
                                 const std::optional<torch::Tensor>& force_random,
                                 const mega_gate_layout::RoutingArgs& routing_args,
                                 const torch::Tensor& score_barriers,
                                 const int& num_tokens, const int& hidden,
                                 const int& num_routed_experts, const int& num_topk) {
    const auto num_aligned_experts = align(num_routed_experts, static_cast<int>(mega_gate_layout::kExpertAlignment));

    const auto config = get_mega_gate_config(num_tokens, hidden, num_routed_experts, runtime->get_num_sms());

    // Score scratch: every split writes its slice of `[num_token_blocks * block_tokens, num_aligned_experts]`
    const auto num_token_blocks = ceil_div(num_tokens, config.block_tokens);
    const auto num_scratch_bytes = mega_gate_layout::Workspace<>::get_num_scratch_bytes(
        num_token_blocks, config.num_split_k, config.block_tokens, num_aligned_experts);
    const auto scratch = torch::empty({static_cast<int64_t>(num_scratch_bytes)}, x.options().dtype(torch::kByte));

    const auto tensor_map_x = make_tma_a_desc(cute::UMMA::Major::K, x, num_tokens, hidden,
                                              config.load_block_m, mega_gate_layout::BLOCK_K,
                                              static_cast<int>(x.stride(0)), 1, mega_gate_layout::kSwizzleMode);
    const auto tensor_map_weight = make_tma_b_desc(cute::UMMA::Major::K, weight, num_routed_experts, hidden,
                                                   config.load_block_n, mega_gate_layout::BLOCK_K,
                                                   static_cast<int>(weight.stride(0)), 1, mega_gate_layout::kSwizzleMode);

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mega_gate", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_gate.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)",
        hidden, num_aligned_experts,
        config.num_gate_threads,
        config.num_mma_ctas, config.num_split_k, config.num_expert_groups,
        num_topk,
        num_routed_experts != num_aligned_experts, bias.has_value(), image_token_mask.has_value(),
        mask.has_value(), fix_routing_mask.has_value(), force_random.has_value(),
        routing_args.to_physical_map != nullptr, routing_args.unmapped_topk_idx != nullptr));

    // Launch
    // The token count only reaches the kernel variants through the gate warpgroup count
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(config.num_launch_sms, 1, 1),
            .block_dim = dim3(mega_gate_layout::kNumNonEpilogueThreads + config.num_gate_threads, 1, 1),
            .cluster_dim = dim3(config.num_mma_ctas, 1, 1),
        },
        tensor_map_x, tensor_map_weight,
        bias ? bias->data_ptr() : nullptr,
        image_bias ? image_bias->data_ptr() : nullptr,
        image_token_mask ? image_token_mask->data_ptr() : nullptr,
        mask ? mask->data_ptr() : nullptr,
        fix_routing_mask ? fix_routing_mask->data_ptr() : nullptr,
        force_random ? force_random->data_ptr() : nullptr,
        routing_args,
        scratch.data_ptr(), score_barriers.data_ptr(),
        static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(config.block_tokens),
        static_cast<uint32_t>(config.num_stages),
        static_cast<uint32_t>(config.num_worker_groups),
        static_cast<uint32_t>(num_token_blocks));
}

} // namespace deep_gemm
