#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <pybind11/functional.h>

#include <deep_gemm/scheduler/nvfp4_mega_moe.cuh>

#include "../runtime/runtime.hpp"
#include "../utils/layout.hpp"
#include "../jit_kernels/impls/sm100_fp4_fp4_mega_moe.hpp"

namespace deep_gemm::nvfp4 {

static int get_token_alignment_for_mega_moe() {
    return layout::nvfp4::kLCMCandidateBlockM;
}

static int get_block_m_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const std::string& mma_type) {
    DG_HOST_ASSERT(mma_type == "fp4xfp4");
    DG_HOST_ASSERT(num_ranks > 0 and num_experts > 0 and num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_tokens >= 0 and num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    return block_m;
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0) {
    DG_HOST_ASSERT(mma_type == "fp4xfp4" and activation == "swiglu");
    DG_HOST_ASSERT(num_ranks > 0 and num_experts > 0 and num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and num_shared_experts >= 0);
    DG_HOST_ASSERT(num_topk > 0 and num_topk + (num_shared_experts > 0 ? 1 : 0) <= 32);
    DG_HOST_ASSERT(hidden > 0 and intermediate_hidden > 0);
    DG_HOST_ASSERT(hidden % 512 == 0 and intermediate_hidden % 512 == 0);

    const auto num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms >= 2 and num_sms % 2 == 0);
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;
    const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    int num_ring_tokens = 0;
    for (const auto& block_m: layout::nvfp4::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::nvfp4::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::nvfp4::kLCMCandidateBlockM);

    int num_sf_ring_tokens = 0;
    for (auto block_m: layout::nvfp4::kCandidateBlockM) {
        num_sf_ring_tokens = std::max(
            num_sf_ring_tokens,
            layout::nvfp4::get_num_sf_ring_tokens(num_ring_tokens, block_m));
    }
    const auto mega_buffer = layout::nvfp4::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens, num_shared_experts);
    DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);

    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        DG_HOST_ASSERT(buffer.is_cuda() and buffer.is_contiguous());
        DG_HOST_ASSERT(buffer.nbytes() >= static_cast<size_t>(mega_buffer.get_num_bytes()));
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden / 2},
            torch::TensorOptions().dtype(torch::kUInt8).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 64},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, hidden / 2},
            torch::TensorOptions().dtype(torch::kUInt8).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden / 64}, {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden / 2},
            torch::TensorOptions().dtype(torch::kUInt8).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden / 64}, {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               torch::Tensor(), torch::Tensor(), shared_l2_acts, torch::Tensor(),
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
}

static void fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt,
    const std::optional<torch::Tensor>& x_bf16_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const std::optional<torch::Tensor>& l1_alphas_opt,
    const std::optional<torch::Tensor>& l2_alphas_opt,
    const std::optional<torch::Tensor>& a2_scales_opt,
    const float& routed_scaling_factor
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;
    DG_HOST_ASSERT(y.dim() == 2);
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT((rm == 1 and rn == 1 and rk == 16) and
                   "NVFP4 MegaMoE currently supports only recipe (1, 1, 16)");
    DG_HOST_ASSERT(activation == "swiglu" and
                   "NVFP4 MegaMoE currently supports only SwiGLU");
    DG_HOST_ASSERT(std::isfinite(routed_scaling_factor));
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == x_bf16_opt.has_value());
    const auto activation_clamp = activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4 and l2_weights.scalar_type() == kPackedFP4);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_ and intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(hidden > 0 and intermediate_hidden > 0);
    DG_HOST_ASSERT(hidden % 512 == 0 and intermediate_hidden % 512 == 0);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(y.is_cuda() and y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DG_HOST_ASSERT(y.size(1) == hidden);

    const auto output_device = y.device();
    const auto is_local_cuda_tensor = [&output_device](const torch::Tensor& tensor) {
        return tensor.is_cuda() and tensor.device() == output_device;
    };
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights) and is_local_cuda_tensor(l2_weights));
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights_sf) and is_local_cuda_tensor(l2_weights_sf));
    DG_HOST_ASSERT(is_local_cuda_tensor(sym_buffer) and sym_buffer.is_contiguous());
    constexpr int kGranMN = 1, kGranK = 16;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
        DG_HOST_ASSERT(is_local_cuda_tensor(cumulative_local_expert_recv_stats.value()));
    }

    int num_shared_experts = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights, x_bf16;
    if (shared_l1_weights_opt.has_value()) {
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        x_bf16 = x_bf16_opt.value();
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        const auto shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;
        DG_HOST_ASSERT(num_shared_experts > 0);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden and shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(x_bf16.scalar_type() == torch::kBFloat16 and x_bf16.is_contiguous());
        DG_HOST_ASSERT(x_bf16.dim() == 2 and x_bf16.size(0) >= num_tokens and x_bf16.size(1) == hidden);
        DG_HOST_ASSERT(is_local_cuda_tensor(shared_l1_weights) and is_local_cuda_tensor(shared_l2_weights));
        DG_HOST_ASSERT(is_local_cuda_tensor(x_bf16));
    }

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(sym_buffer_ptrs[rank_idx] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));
    DG_HOST_ASSERT(num_experts == num_experts_per_rank * num_ranks);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, "fp4xfp4", activation, num_shared_experts);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));

    const void* l1_alphas_ptr = nullptr;
    const void* l2_alphas_ptr = nullptr;
    if (l1_alphas_opt.has_value()) {
        const auto& l1_alphas = l1_alphas_opt.value();
        DG_HOST_ASSERT(l1_alphas.scalar_type() == torch::kFloat and l1_alphas.is_contiguous());
        DG_HOST_ASSERT(l1_alphas.dim() == 2 and l1_alphas.size(0) == num_experts_per_rank and l1_alphas.size(1) == 2);
        DG_HOST_ASSERT(is_local_cuda_tensor(l1_alphas));
        l1_alphas_ptr = l1_alphas.data_ptr();
    }
    if (l2_alphas_opt.has_value()) {
        const auto& l2_alphas = l2_alphas_opt.value();
        DG_HOST_ASSERT(l2_alphas.scalar_type() == torch::kFloat and l2_alphas.is_contiguous());
        DG_HOST_ASSERT(l2_alphas.dim() == 1 and l2_alphas.size(0) == num_experts_per_rank);
        DG_HOST_ASSERT(is_local_cuda_tensor(l2_alphas));
        l2_alphas_ptr = l2_alphas.data_ptr();
    }
    const void* a2_scales_ptr = nullptr;
    if (a2_scales_opt.has_value()) {
        const auto& a2_scales = a2_scales_opt.value();
        DG_HOST_ASSERT(a2_scales.scalar_type() == torch::kFloat and a2_scales.is_contiguous());
        DG_HOST_ASSERT(a2_scales.dim() == 1 and a2_scales.size(0) == num_experts_per_rank);
        DG_HOST_ASSERT(is_local_cuda_tensor(a2_scales));
        a2_scales_ptr = a2_scales.data_ptr();
    }

    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);
    sm100_fp4_fp4_mega_moe(y,
                           l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
                           num_shared_experts > 0 ? x_bf16 : l1_acts,
                           num_shared_experts > 0 ? shared_l2_acts : l2_acts,
                           num_shared_experts > 0 ? shared_l1_weights : l1_weights,
                           num_shared_experts > 0 ? shared_l2_weights : l2_weights,
                           l1_weights, l2_weights, l1_weights_sf, l2_weights_sf,
                           cumulative_local_expert_recv_stats, sym_buffer_ptrs,
                           rank_idx, num_max_tokens_per_rank, num_experts_per_rank,
                           num_shared_experts, num_tokens, num_topk, hidden, intermediate_hidden,
                           activation_clamp, fast_math, l1_alphas_ptr, l2_alphas_ptr, a2_scales_ptr,
                           routed_scaling_factor);
    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

} // namespace deep_gemm::nvfp4
