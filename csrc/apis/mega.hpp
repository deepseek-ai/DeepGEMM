#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <pybind11/functional.h>

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp4_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static int get_block_m_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const std::string& mma_type) {
    DG_HOST_ASSERT(num_tokens >= 0);
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind);
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
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(activation == "swiglu" or (mma_type == "fp8xfp4" and activation == "situ"));
    DG_HOST_ASSERT(num_shared_experts >= 0);

    // Ring capacity: worst-case live pool blocks over all candidate BLOCK_M; mirrors the kernel assert.
    // TODO: we temporarily assume the SM count is consistent with the runtime value
    const auto num_sms = device_runtime->get_num_sms();
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;

    // Shared
    const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Iterate all block candidates to get the maximum ring size
    int num_ring_tokens = 0;
    for (const auto& block_m: layout::kCandidateBlockM) {
        const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
        const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden);
        num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
    }
    num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);

    // Parse MMA type
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto num_mma_elem_bits = get_num_mma_elem_bits(mma_kind);
    const auto with_sf = is_mma_with_sf(mma_kind);
    const auto sf_gran_k = get_mma_sf_gran_k(mma_kind);

    // Compute num_sf_ring_tokens (max across all candidate block sizes)
    int num_sf_ring_tokens = 0;
    if (with_sf) {
        for (auto block_m: layout::kCandidateBlockM) {
            num_sf_ring_tokens = std::max(
                num_sf_ring_tokens,
                layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
        }
    }

    // All buffers
    // NOTES: NVFP4 shared experts run in BF16 (16-bit, SF-free)
    const auto shared_num_mma_elem_bits = mma_kind == MmaKind::NVFP4 ? 16 : 0;
    const auto mega_buffer = layout::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens, with_sf,
        num_shared_experts, num_mma_elem_bits, sf_gran_k,
        shared_num_mma_elem_bits
    );

    // Check SF buffer requirements
    if (with_sf) {
        // The smallest-token NVFP4 heuristic uses a 512-element K tile; enforce
        // the strongest candidate constraint here instead of failing in NVRTC.
        // Keep the existing MXFP8FP4 contract at 128 elements.
        const int sf_alignment = mma_kind == MmaKind::NVFP4 ? 512 : 128;
        DG_HOST_ASSERT(hidden % sf_alignment == 0 and intermediate_hidden % sf_alignment == 0);
        DG_HOST_ASSERT(shared_intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    // Slice function: creates tensor views from the raw buffer.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    // NOTES: for NVFP4, token views are packed E2M1 bytes (2 elements each) and SF
    // views pack 4 E4M3 bytes per `int`
    const auto is_fp4 = mma_kind == MmaKind::NVFP4;
    const auto token_dtype = is_fp4 ? torch::kUInt8 : (with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    const auto hidden_cols = is_fp4 ? hidden / 2 : hidden;
    const auto intermediate_cols = is_fp4 ? intermediate_hidden / 2 : intermediate_hidden;
    const auto hidden_sf_cols = hidden / (sf_gran_k * 4);
    const auto intermediate_sf_cols = intermediate_hidden / (sf_gran_k * 4);
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden_cols},
            torch::TensorOptions().dtype(token_dtype).device(buffer.device()));
        auto x_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden_sf_cols},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        // NOTES: NVFP4 shared experts are BF16 and SF-free; their L1 activations come from
        // a caller-provided BF16 tensor, so `shared_l1_acts`/SF views stay undefined
        const bool shared_with_sf = with_sf and not is_fp4;
        auto shared_l1_acts = is_fp4 ? torch::Tensor() : x;
        auto shared_l1_acts_sf = (shared_with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden},
            torch::TensorOptions().dtype(shared_with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = (shared_with_sf and num_shared_experts > 0) ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base)),
            {layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank), shared_intermediate_hidden / 128},
            {1, layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank)},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, hidden_cols},
            torch::TensorOptions().dtype(token_dtype).device(buffer.device()));
        auto l1_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden_sf_cols},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, intermediate_cols},
            torch::TensorOptions().dtype(token_dtype).device(buffer.device()));
        auto l2_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_sf_cols},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const std::optional<float>& situ_beta_opt,
    const std::optional<float>& situ_linear_beta_opt
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu" or activation == "situ");
    DG_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == shared_l2_weights_tuple_opt.has_value());

    // Activation checks
    const auto use_situ = activation == "situ";
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);
    const auto situ_beta = situ_beta_opt.value_or(0.0f);
    const auto situ_linear_beta = situ_linear_beta_opt.value_or(0.0f);
    DG_HOST_ASSERT(not use_situ or not activation_clamp_opt.has_value());
    DG_HOST_ASSERT(not use_situ or
                   (std::isfinite(situ_beta) and situ_beta > 0 and
                    std::isfinite(situ_linear_beta) and situ_linear_beta > 0));
    DG_HOST_ASSERT(use_situ or
                   (not situ_beta_opt.has_value() and not situ_linear_beta_opt.has_value()));

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf) = shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf) = shared_l2_weights_tuple_opt.value();
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        check_sf_layout(shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
        check_sf_layout(shared_l2_weights_sf, hidden, shared_intermediate_hidden, kGranMN, kGranK,
                        std::nullopt, true, false, torch::kInt);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               shared_l1_acts, shared_l1_acts_sf,
                               shared_l2_acts, shared_l2_acts_sf,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               shared_l1_weights, shared_l2_weights,
                               shared_l1_weights_sf, shared_l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_shared_experts,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, fast_math,
                               use_situ, situ_beta, situ_linear_beta);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    // BF16 shared expert (optional): plain BF16 weights (no SF) and the BF16 input
    // activations of the local tokens (the routed `x` in the buffer is packed FP4)
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

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT((rm == 1 and rn == 1 and rk == 16) and
                   "NVFP4 MegaMoE currently supports only recipe (1, 1, 16)");
    DG_HOST_ASSERT(activation == "swiglu" and
                   "NVFP4 MegaMoE currently supports only SwiGLU");
    DG_HOST_ASSERT(std::isfinite(routed_scaling_factor));
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == x_bf16_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4 and l2_weights.scalar_type() == kPackedFP4);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(y.is_cuda() and y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DG_HOST_ASSERT(y.dim() == 2 and y.size(1) == hidden);

    const auto output_device = y.device();
    const auto is_local_cuda_tensor = [&output_device](const torch::Tensor& tensor) {
        return tensor.is_cuda() and tensor.device() == output_device;
    };
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights) and is_local_cuda_tensor(l2_weights));
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights_sf) and is_local_cuda_tensor(l2_weights_sf));
    DG_HOST_ASSERT(is_local_cuda_tensor(sym_buffer));

    // Check weight SF layout for E4M3 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 16;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
        DG_HOST_ASSERT(is_local_cuda_tensor(cumulative_local_expert_recv_stats.value()));
    }

    // Check the optional BF16 shared expert tensors
    // NOTES: multiple shared experts are folded into the weights (L1 concatenated on N,
    // L2 concatenated on K), so `num_shared_experts = shared_intermediate_hidden / intermediate_hidden`
    int num_shared_experts = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights, x_bf16;
    if (shared_l1_weights_opt.has_value()) {
        DG_HOST_ASSERT(shared_l2_weights_opt.has_value() and x_bf16_opt.has_value());
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        x_bf16 = x_bf16_opt.value();

        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        const auto shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;
        DG_HOST_ASSERT(num_shared_experts > 0);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(x_bf16.scalar_type() == torch::kBFloat16 and x_bf16.is_contiguous());
        // The leading dimension may be over-allocated for CUDA Graphs, but
        // replay must preserve both this allocation and the captured token count
        // because they are encoded in the shared-input TMA descriptor.
        DG_HOST_ASSERT(x_bf16.dim() == 2 and x_bf16.size(0) >= num_tokens and x_bf16.size(1) == hidden);
        DG_HOST_ASSERT(is_local_cuda_tensor(shared_l1_weights) and is_local_cuda_tensor(shared_l2_weights));
        DG_HOST_ASSERT(is_local_cuda_tensor(x_bf16));
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp4xfp4", activation, num_shared_experts);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Check the optional per-local-expert scales (e.g. modelopt's `weight_scale_2`)
    // NOTES: L1 has separate gate/up factors, so its scales come as `(E, 2)` pairs
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

    // Per-local-expert down-proj input scale (modelopt's `input_scale`): normalizes the
    // in-kernel intermediate NVFP4 requant and is folded into the L2 alpha
    const void* a2_scales_ptr = nullptr;
    if (a2_scales_opt.has_value()) {
        const auto& a2_scales = a2_scales_opt.value();
        DG_HOST_ASSERT(a2_scales.scalar_type() == torch::kFloat and a2_scales.is_contiguous());
        DG_HOST_ASSERT(a2_scales.dim() == 1 and a2_scales.size(0) == num_experts_per_rank);
        DG_HOST_ASSERT(is_local_cuda_tensor(a2_scales));
        a2_scales_ptr = a2_scales.data_ptr();
    }

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        // With no shared expert, these descriptor placeholders alias routed
        // tensors. Their dtype/layout is intentionally irrelevant because the
        // kHasShared=false kernel specialization never dereferences them.
        sm100_fp4_fp4_mega_moe(y,
                               l1_acts, l1_acts_sf,
                               l2_acts, l2_acts_sf,
                               num_shared_experts > 0 ? x_bf16 : l1_acts,
                               num_shared_experts > 0 ? shared_l2_acts : l2_acts,
                               num_shared_experts > 0 ? shared_l1_weights : l1_weights,
                               num_shared_experts > 0 ? shared_l2_weights : l2_weights,
                               l1_weights, l2_weights,
                               l1_weights_sf, l2_weights_sf,
                               cumulative_local_expert_recv_stats,
                               sym_buffer_ptrs,
                               rank_idx, num_max_tokens_per_rank,
                               num_experts_per_rank,
                               num_shared_experts,
                               num_tokens, num_topk,
                               hidden, intermediate_hidden,
                               activation_clamp, fast_math,
                               l1_alphas_ptr, l2_alphas_ptr, a2_scales_ptr,
                               routed_scaling_factor);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}


// ============================================================================
// SM90 (Hopper) FP8 MegaMoE
// ----------------------------------------------------------------------------
// Ported from upstream PR https://github.com/sgl-project/DeepGEMM/pull/36
// (`csrc/apis/sm90_mega.hpp` on the old `dev` branch).
//
// Unlike `fp8_fp4_mega_moe` (SM100, FP4 weights, UE8M0 SF, ring-buffer workspace, optional
// shared experts), this path is FP8-only (both activations and weights), uses float SF at
// per-128 (L1)/per-64 (L2) K granularity, and uses a simpler pool-based workspace/scheduler
// (`layout::MegaMoESM90Workspace` / `sched::MegaMoESM90Scheduler`). It does not support shared
// experts. The symmetric buffer layout is therefore also different and is *not*
// interchangeable with `get_symm_buffer_size_for_mega_moe`'s buffer.
// ============================================================================

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(use_fp8_dispatch);
    DG_HOST_ASSERT(activation == "swiglu");
    // `get_mega_moe_config_sm90` may pick `num_dispatch_threads == 64`, and the SM90
    // `nvlink_barrier` only has one signaling thread per rank (`thread_idx < kNumRanks`),
    // so more than 64 ranks either fails the kernel's `kNumRanks <= kNumThreads`
    // static_assert at JIT time or, if that were relaxed, would hang on missed signals.
    DG_HOST_ASSERT(num_ranks <= 64);

    const auto workspace = layout::MegaMoESM90Workspace(nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);

    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden / 16);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    const auto num_max_pool_tokens = static_cast<int>(workspace.num_max_pool_tokens);
    // Unlike SM100 (which can select any of `layout::kCandidateBlockM`), SM90's
    // `get_block_config_for_mega_moe_sm90` only ever picks block_m in {64, 128}. Sizing the
    // SF pool against the full shared candidate set (which includes block_m=8) would
    // over-allocate the SF pool by ~8x.
    constexpr int kSm90CandidateBlockM[] = {64, 128};
    int num_max_padded_sf_pool_tokens = 0;
    for (int block_m: kSm90CandidateBlockM) {
        num_max_padded_sf_pool_tokens = std::max(
            num_max_padded_sf_pool_tokens,
            layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m)
        );
    }

    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    // Kept in sync with the stricter check in `fp8_mega_moe_sm90` (see comment there):
    // hidden must be a multiple of 256 for the scheduler's BLOCK_N=256 case to compile.
    DG_HOST_ASSERT(hidden % 256 == 0 and intermediate_hidden % 128 == 0);

    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_max_pool_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, intermediate_hidden / 64},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()), slice_input_buffers};
}

static void fp8_mega_moe_sm90(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 128 and rn == 128 and rk == 128);
    DG_HOST_ASSERT(activation == "swiglu");

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    // `get_mega_moe_config_sm90` may pick BLOCK_N=256 for either the L1 (2 * intermediate_hidden)
    // or L2 (hidden) GEMM depending on the runtime token distribution, and the scheduler
    // requires L1_SHAPE_N/L2_SHAPE_N to be an exact multiple of BLOCK_N or the JIT compile
    // fails. Require hidden % 256 == 0 so this holds regardless of which BLOCK_N is chosen;
    // intermediate_hidden % 128 == 0 already implies (2 * intermediate_hidden) % 256 == 0.
    DG_HOST_ASSERT(hidden % 256 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    constexpr int kGranMN = 128, kGranK = 128;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_sm90_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    sm90_fp8_mega_moe(y,
                     l1_acts, l1_acts_sf,
                     l2_acts, l2_acts_sf,
                     l1_weights, l2_weights,
                     l1_weights_sf, l2_weights_sf,
                     cumulative_local_expert_recv_stats,
                     sym_buffer_ptrs,
                     rank_idx, num_max_tokens_per_rank,
                     num_experts_per_rank,
                     num_tokens, num_topk,
                     hidden, intermediate_hidden,
                     activation_clamp, fast_math);

    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& shared_l1_weights_opt,
    const std::optional<torch::Tensor>& shared_l2_weights_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");
    DG_HOST_ASSERT(shared_l1_weights_opt.has_value() == shared_l2_weights_opt.has_value());

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    int num_shared_experts = 0, shared_intermediate_hidden = 0;
    torch::Tensor shared_l1_weights, shared_l2_weights;
    if (shared_l1_weights_opt.has_value()) {
        shared_l1_weights = shared_l1_weights_opt.value();
        shared_l2_weights = shared_l2_weights_opt.value();
        shared_intermediate_hidden = static_cast<int>(shared_l2_weights.size(1));
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;

        DG_HOST_ASSERT(shared_intermediate_hidden % intermediate_hidden == 0);
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2);
        DG_HOST_ASSERT(shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l2_weights.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights,
                shared_l1_acts, _shared_l1_acts_sf, shared_l2_acts, _shared_l2_acts_sf,
                l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            l1_acts, l2_acts,
                            shared_l1_acts, shared_l2_acts,
                            l1_weights, l2_weights,
                            shared_l1_weights, shared_l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_shared_experts,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_block_m_for_mega_moe", &get_block_m_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
    m.def("bf16_mega_moe", &bf16_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_mega_moe", &get_symm_buffer_size_for_sm90_mega_moe);
    m.def("fp8_mega_moe_sm90", &fp8_mega_moe_sm90);
#endif
}

} // namespace deep_gemm::mega
