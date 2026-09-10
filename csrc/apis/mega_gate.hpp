#pragma once

#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/python.h>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_gate.hpp"
#include "../utils/exception.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::mega_gate {

static int get_scoring_type(const std::string& scoring_func) {
    if (scoring_func == "sigmoid")
        return static_cast<int>(mega_gate_layout::ScoringType::Sigmoid);
    if (scoring_func == "sqrtsoftplus")
        return static_cast<int>(mega_gate_layout::ScoringType::SqrtSoftplus);
    if (scoring_func == "identity")
        return static_cast<int>(mega_gate_layout::ScoringType::Identity);
    DG_HOST_UNREACHABLE("Unsupported MoE scoring function");
}

static const torch::Tensor& get_score_barriers(const torch::TensorOptions& options) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    DG_HOST_ASSERT(options.device() == stream.device());
    static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> score_barriers_by_stream;
    auto& score_barriers = score_barriers_by_stream[stream];
    if (not score_barriers.defined()) {
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        score_barriers = torch::zeros({mega_gate_layout::kNumMaxTokenBlocks, mega_gate_layout::kScoreBarrierLineBytes},
                                      options.dtype(torch::kByte));
    }
    return score_barriers;
}

static void check_same_cuda_device(const torch::Tensor& tensor, const torch::Tensor& reference) {
    DG_HOST_ASSERT(tensor.is_cuda() and tensor.get_device() == reference.get_device());
}

static void check_tensor(const torch::Tensor& tensor, const torch::Tensor& reference,
                         const at::IntArrayRef& shape, const torch::ScalarType& scalar_type,
                         const bool& contiguous = true) {
    check_same_cuda_device(tensor, reference);
    DG_HOST_ASSERT(tensor.sizes() == shape and tensor.scalar_type() == scalar_type and
                   (not contiguous or tensor.is_contiguous()));
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
 *     scoring_func                   "sigmoid" | "sqrtsoftplus" | "identity"
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
    const auto [num_tokens, hidden] = get_shape<2>(x);
    const auto [num_routed_experts, weight_hidden] = get_shape<2>(weight);
    DG_HOST_ASSERT(hidden == weight_hidden);
    DG_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    DG_HOST_ASSERT(num_routed_experts > 0 and num_routed_experts <= 512 and num_routed_experts % 4 == 0);
    DG_HOST_ASSERT(x.scalar_type() == torch::kBFloat16 and weight.scalar_type() == torch::kBFloat16 and
                   x.is_contiguous() and weight.is_contiguous());
    check_same_cuda_device(weight, x);

    DG_HOST_ASSERT(num_topk > 0 and num_topk <= num_routed_experts and num_topk <= 32);
    DG_HOST_ASSERT(ep_rank >= 0);
    DG_HOST_ASSERT(std::isfinite(routed_scaling_factor));
    const auto scoring_type = get_scoring_type(scoring_func);

    int effective_num_shared_experts = 0;
    if (use_shared_as_routed) {
        DG_HOST_ASSERT(num_shared_experts == 1 or num_shared_experts == 2);
        DG_HOST_ASSERT(num_topk % num_shared_experts == 0);
        DG_HOST_ASSERT(num_routed_experts % (num_topk / num_shared_experts) == 0);
        effective_num_shared_experts = num_shared_experts;
    }
    const auto num_physical_topk = num_topk + effective_num_shared_experts;
    DG_HOST_ASSERT(num_physical_topk <= 32);

    if (mask.has_value())
        check_tensor(mask.value(), x, {num_tokens}, torch::kBool);
    if (bias.has_value())
        check_tensor(bias.value(), x, {num_routed_experts}, torch::kFloat32);

    DG_HOST_ASSERT(image_bias.has_value() == image_token_mask.has_value());
    if (image_bias.has_value()) {
        check_tensor(image_bias.value(), x, {num_routed_experts}, torch::kFloat32);
        check_tensor(image_token_mask.value(), x, {num_tokens}, torch::kBool);
    }

    DG_HOST_ASSERT(to_physical_map.has_value() == logical_count.has_value());
    const auto num_logical_experts = num_routed_experts + effective_num_shared_experts;
    if (to_physical_map.has_value()) {
        const auto& physical_map = to_physical_map.value();
        DG_HOST_ASSERT(physical_map.dim() == 2);
        DG_HOST_ASSERT(physical_map.size(1) > 0);
        check_tensor(physical_map, x, {num_logical_experts, physical_map.size(1)}, torch::kInt32);
        check_tensor(logical_count.value(), x, {num_logical_experts}, torch::kInt32);
    }

    if (unmapped_topk_idx.has_value()) {
        const auto& unmapped = unmapped_topk_idx.value();
        check_tensor(unmapped, x, {num_tokens, num_topk}, torch::kInt64, false);
        DG_HOST_ASSERT(unmapped.stride(1) == 1);
    }
    if (fix_routing_mask.has_value()) {
        DG_HOST_ASSERT(unmapped_topk_idx.has_value());
        check_tensor(fix_routing_mask.value(), x, {num_tokens}, torch::kBool);
    }
    if (force_random.has_value())
        check_tensor(force_random.value(), x, {num_tokens}, torch::kBool);

    torch::Tensor topk_idx, topk_weights;
    if (out.has_value()) {
        std::tie(topk_idx, topk_weights) = out.value();
        check_tensor(topk_idx, x, {num_tokens, num_physical_topk}, torch::kInt64);
        check_tensor(topk_weights, x, {num_tokens, num_physical_topk}, torch::kFloat32);
    } else {
        topk_idx = torch::empty({num_tokens, num_physical_topk}, x.options().dtype(torch::kInt64));
        topk_weights = torch::empty({num_tokens, num_physical_topk}, x.options().dtype(torch::kFloat32));
    }

    if (num_tokens == 0)
        return {topk_idx, topk_weights};
    DG_HOST_ASSERT(num_tokens <= static_cast<int>(layout::mega_gate::kNumMaxTokens));

    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    const auto config = get_sm100_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts,
                                                        runtime->get_num_sms(), bias.has_value(),
                                                        image_token_mask.has_value(), to_physical_map.has_value());
    const auto num_aligned_experts = align(num_routed_experts, static_cast<int>(mega_gate_layout::kExpertAlignment));
    const auto num_token_blocks = ceil_div(num_tokens, config.block_tokens);
    const auto num_scratch_bytes = mega_gate_layout::Workspace<>::get_num_scratch_bytes(num_token_blocks, config.num_split_k,
                                                                                        config.block_tokens, num_aligned_experts);
    const auto scratch = torch::empty({static_cast<int64_t>(num_scratch_bytes)}, x.options().dtype(torch::kByte));
    const auto& score_barriers = get_score_barriers(x.options());
    sm100_bf16_mega_gate(x, weight, bias, image_bias, image_token_mask, mask, fix_routing_mask,
                         to_physical_map, logical_count, topk_idx, unmapped_topk_idx,
                         topk_weights, force_random, num_tokens, hidden, num_routed_experts,
                         num_topk, effective_num_shared_experts, routed_scaling_factor,
                         ep_rank, scoring_type, config, scratch, score_barriers);
    return {topk_idx, topk_weights};
}

static pybind11::dict get_bf16_mega_gate_config(const int& num_tokens, const int& hidden,
                                                const int& num_routed_experts, const int& num_topk) {
    DG_HOST_ASSERT(num_tokens > 0 and num_tokens <= static_cast<int>(layout::mega_gate::kNumMaxTokens));
    DG_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    DG_HOST_ASSERT(num_routed_experts > 0 and num_routed_experts <= 512 and num_routed_experts % 4 == 0);
    DG_HOST_ASSERT(num_topk > 0 and num_topk <= num_routed_experts and num_topk <= 32);
    const auto num_device_sms = runtime->get_num_sms();
    const auto config = get_sm100_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts,
                                                        num_device_sms, true, true, true);
    pybind11::dict result;
    result["block_tokens"] = config.block_tokens;
    result["num_mma_ctas"] = config.num_mma_ctas;
    result["num_split_k"] = config.num_split_k;
    result["num_expert_groups"] = config.num_expert_groups;
    result["num_gate_warpgroups"] = config.num_gate_warpgroups;
    result["num_sms"] = config.num_launch_sms;
    return result;
}

static void register_apis(pybind11::module_& m) {
    m.def("get_bf16_mega_gate_config", &get_bf16_mega_gate_config,
          py::arg("num_tokens"), py::arg("hidden"),
          py::arg("num_routed_experts"), py::arg("num_topk"));
    m.def("bf16_mega_gate", &bf16_mega_gate,
          py::arg("x"), py::arg("weight"), py::arg("num_topk"),
          py::arg("use_shared_as_routed"), py::arg("num_shared_experts"),
          py::arg("routed_scaling_factor"), py::arg("ep_rank"),
          py::arg("scoring_func") = "identity",
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
