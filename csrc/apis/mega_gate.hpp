#pragma once

#include <cmath>
#include <unordered_map>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include "../utils/compatibility.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_gate.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::mega_gate {

static const torch::Tensor& get_score_barriers(const torch::TensorOptions& options) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    DG_HOST_ASSERT(options.device() == stream.device());
    static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> score_barriers_by_stream;
    auto& score_barriers = score_barriers_by_stream[stream];
    if (not score_barriers.defined()) {
        // Warm up each stream before capture so one-time zeroing is not replayed with the graph.
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        score_barriers = torch::zeros(
            {mega_gate_layout::kNumMaxTokenBlocks, mega_gate_layout::kScoreBarrierLineBytes},
            options.dtype(torch::kByte));
    }
    return score_barriers;
}

/**
 * Fuse the BF16 gate GEMM with the MoE top-k gate.
 *
 * Args:
 *     x                              [T,H]               BF16
 *     weight                         [E,H]               BF16
 *     num_topk                       int
 *     use_shared_as_routed           bool
 *     num_shared_experts             int
 *     routed_scaling_factor          float
 *     ep_rank                        int
 *     scoring_func                   "sqrtsoftplus"
 *     mask                           [T] bool | None
 *     bias                           [E] FP32 | None
 *     image_bias                     [E] FP32 | None
 *     image_token_mask               [T] bool | None
 *     fix_routing_mask               [T] bool | None
 *     to_physical_map                [E+S, dup] int32 | None
 *     logical_count                  [E+S] int32 | None
 *     unmapped_topk_idx              [T,K] int64 | None
 *     force_random                   [T] bool | None
 *     out                            ([T,K'] int64, [T,K'] FP32) | None
 *
 * Returns:
 *     (topk_idx, topk_weights)
 *
 * Notes:
 *     S = num_shared_experts when use_shared_as_routed, else 0; K' = K + S.
 *     The kernel ranks experts on score + bias, while the emitted weights are the
 *     unbiased scores normalized over the top-k sum and scaled by routed_scaling_factor.
 *     unmapped_topk_idx receives the logical top-k indices, and for tokens with fix_routing_mask
 *     set it is also read as the routing to keep (so fix_routing_mask requires it).
 */
static std::tuple<torch::Tensor, torch::Tensor>
bf16_mega_gate(const torch::Tensor& x,
               const torch::Tensor& weight,
               const int& num_topk,
               const bool& use_shared_as_routed,
               const int& num_shared_experts,
               const float& routed_scaling_factor,
               const int& ep_rank,
               const std::string& scoring_func,
               const std::optional<torch::Tensor>& mask,
               const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& image_bias,
               const std::optional<torch::Tensor>& image_token_mask,
               const std::optional<torch::Tensor>& fix_routing_mask,
               const std::optional<torch::Tensor>& to_physical_map,
               const std::optional<torch::Tensor>& logical_count,
               const std::optional<torch::Tensor>& unmapped_topk_idx,
               const std::optional<torch::Tensor>& force_random,
               const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& out) {
    constexpr int kHiddenAlignment = 256, kNumMaxRoutedExperts = 512, kNumMaxTopk = 32;
    const auto [num_tokens, hidden] = get_shape<2>(x);
    const auto num_routed_experts = static_cast<int>(weight.size(0));
    DG_HOST_ASSERT(hidden > 0 and hidden % kHiddenAlignment == 0);
    DG_HOST_ASSERT(num_routed_experts <= kNumMaxRoutedExperts and
                   num_routed_experts % static_cast<int>(mega_gate_layout::kNumExpertsPerLaneVector) == 0);

    const auto device = x.device();
    const auto check_dense = [&](const torch::Tensor& tensor, const at::IntArrayRef& shape, const torch::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.sizes() == shape and tensor.scalar_type() == dtype);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(tensor.device() == device);
    };
    check_dense(x, {num_tokens, hidden}, torch::kBFloat16);
    check_dense(weight, {num_routed_experts, hidden}, torch::kBFloat16);

    DG_HOST_ASSERT(num_topk > 0 and num_topk <= num_routed_experts);
    DG_HOST_ASSERT(ep_rank >= 0);
    DG_HOST_ASSERT(std::isfinite(routed_scaling_factor));
    DG_HOST_ASSERT(scoring_func == "sqrtsoftplus");

    int effective_num_shared_experts = 0;
    if (use_shared_as_routed) {
        DG_HOST_ASSERT(num_shared_experts == 1 or num_shared_experts == 2);
        DG_HOST_ASSERT(num_topk % num_shared_experts == 0);
        DG_HOST_ASSERT(num_routed_experts % (num_topk / num_shared_experts) == 0);
        effective_num_shared_experts = num_shared_experts;
    }
    const auto num_physical_topk = num_topk + effective_num_shared_experts;
    DG_HOST_ASSERT(num_physical_topk <= kNumMaxTopk);

    if (mask.has_value())
        check_dense(mask.value(), {num_tokens}, torch::kBool);
    if (bias.has_value())
        check_dense(bias.value(), {num_routed_experts}, torch::kFloat);

    DG_HOST_ASSERT(image_bias.has_value() == image_token_mask.has_value());
    if (image_bias.has_value()) {
        check_dense(image_bias.value(), {num_routed_experts}, torch::kFloat);
        check_dense(image_token_mask.value(), {num_tokens}, torch::kBool);
    }

    DG_HOST_ASSERT(to_physical_map.has_value() == logical_count.has_value());
    const auto num_logical_experts = num_routed_experts + effective_num_shared_experts;
    if (to_physical_map.has_value()) {
        const auto& physical_map = to_physical_map.value();
        DG_HOST_ASSERT(physical_map.dim() == 2 and physical_map.size(1) > 0);
        check_dense(physical_map, {num_logical_experts, physical_map.size(1)}, torch::kInt);
        check_dense(logical_count.value(), {num_logical_experts}, torch::kInt);
    }

    if (unmapped_topk_idx.has_value()) {
        // Rows may be strided (a view into a wider buffer), the slots are contiguous
        const auto& unmapped = unmapped_topk_idx.value();
        DG_HOST_ASSERT(unmapped.sizes() == at::IntArrayRef({num_tokens, num_topk}) and unmapped.scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(unmapped.stride(1) == 1 and unmapped.device() == device);
    }
    if (fix_routing_mask.has_value()) {
        DG_HOST_ASSERT(unmapped_topk_idx.has_value());
        check_dense(fix_routing_mask.value(), {num_tokens}, torch::kBool);
    }
    if (force_random.has_value())
        check_dense(force_random.value(), {num_tokens}, torch::kBool);

    torch::Tensor topk_idx, topk_weights;
    if (out.has_value()) {
        std::tie(topk_idx, topk_weights) = out.value();
        check_dense(topk_idx, {num_tokens, num_physical_topk}, torch::kInt64);
        check_dense(topk_weights, {num_tokens, num_physical_topk}, torch::kFloat);
    } else {
        topk_idx = torch::empty({num_tokens, num_physical_topk}, x.options().dtype(torch::kInt64));
        topk_weights = torch::empty({num_tokens, num_physical_topk}, x.options().dtype(torch::kFloat));
    }

    if (num_tokens == 0)
        return {topk_idx, topk_weights};
    DG_HOST_ASSERT(num_tokens <= static_cast<int>(mega_gate_layout::kNumMaxTokens));

    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    const mega_gate_layout::RoutingArgs routing_args = {
        .to_physical_map = to_physical_map ? to_physical_map->data_ptr<int>() : nullptr,
        .logical_count = logical_count ? logical_count->data_ptr<int>() : nullptr,
        .topk_idx = topk_idx.data_ptr<int64_t>(),
        .unmapped_topk_idx = unmapped_topk_idx ? unmapped_topk_idx->data_ptr<int64_t>() : nullptr,
        .topk_weights = topk_weights.data_ptr<float>(),
        .unmapped_topk_idx_stride = unmapped_topk_idx ? unmapped_topk_idx->stride(0) : 0,
        .num_routed_experts = static_cast<uint32_t>(num_routed_experts),
        .num_shared_experts = static_cast<uint32_t>(effective_num_shared_experts),
        .num_duplicate_experts = to_physical_map ? static_cast<uint32_t>(to_physical_map->size(1)) : 0u,
        .rank_idx = static_cast<uint32_t>(ep_rank),
        .routed_scaling_factor = routed_scaling_factor,
    };
    sm100_bf16_mega_gate(x, weight, bias, image_bias, image_token_mask, mask, fix_routing_mask, force_random,
                         routing_args, get_score_barriers(x.options()), num_tokens, hidden, num_routed_experts, num_topk);
    return {topk_idx, topk_weights};
}

static void register_apis(pybind11::module_& m) {
    m.def("bf16_mega_gate", &bf16_mega_gate,
          py::arg("x"), py::arg("weight"), py::arg("num_topk"),
          py::arg("use_shared_as_routed"), py::arg("num_shared_experts"),
          py::arg("routed_scaling_factor"), py::arg("ep_rank"),
          py::arg("scoring_func") = "sqrtsoftplus",
          py::arg("mask") = std::nullopt,
          py::arg("bias") = std::nullopt,
          py::arg("image_bias") = std::nullopt,
          py::arg("image_token_mask") = std::nullopt,
          py::arg("fix_routing_mask") = std::nullopt,
          py::arg("to_physical_map") = std::nullopt,
          py::arg("logical_count") = std::nullopt,
          py::arg("unmapped_topk_idx") = std::nullopt,
          py::arg("force_random") = std::nullopt,
          py::arg("out") = std::nullopt);
}

} // namespace deep_gemm::mega_gate
