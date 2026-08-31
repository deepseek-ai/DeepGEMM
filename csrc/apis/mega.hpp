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

static void fp8_fp4_mega_moe_bf16_shared(
    const torch::Tensor& y,
    const std::optional<torch::Tensor>& shared_y_opt,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const torch::Tensor& shared_x,
    const torch::Tensor& shared_l2_acts,
    const torch::Tensor& shared_l1_weights,
    const torch::Tensor& shared_l2_weights,
    const torch::Tensor& rms_weight,
    const float& rms_epsilon,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const bool& fast_math,
    const std::optional<float>& situ_beta_opt,
    const std::optional<float>& situ_linear_beta_opt,
    const std::optional<torch::Tensor>& shared_rs_workspace_opt,
    const std::optional<torch::Tensor>& shared_rs_flags_opt,
    const std::optional<torch::Tensor>& shared_rs_peer_ptrs_opt,
    const bool& publish_shared_sp_rs
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto num_shared_tokens = static_cast<int>(shared_x.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "situ");
    DG_HOST_ASSERT(std::isfinite(rms_epsilon) and rms_epsilon > 0);
    const auto situ_beta = situ_beta_opt.value_or(0.0f);
    const auto situ_linear_beta = situ_linear_beta_opt.value_or(0.0f);
    DG_HOST_ASSERT(std::isfinite(situ_beta) and situ_beta > 0);
    DG_HOST_ASSERT(std::isfinite(situ_linear_beta) and situ_linear_beta > 0);

    // Routed FP8 x FP4 tensors.
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
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    // Independent BF16 shared expert. Its input/output width intentionally does
    // not have to match the routed latent width.
    DG_HOST_ASSERT(y.is_cuda() and y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DG_HOST_ASSERT(y.dim() == 2 and y.size(1) == hidden);
    DG_HOST_ASSERT(shared_x.is_cuda() and shared_x.scalar_type() == torch::kBFloat16 and shared_x.is_contiguous());
    DG_HOST_ASSERT(shared_x.dim() == 2 and shared_x.size(0) >= num_tokens);
    const auto shared_hidden = static_cast<int>(shared_x.size(1));
    DG_HOST_ASSERT(num_shared_tokens <= num_max_tokens_per_rank);
    const auto publish_shared_rs = shared_rs_workspace_opt.has_value();
    DG_HOST_ASSERT(publish_shared_rs == shared_rs_flags_opt.has_value());
    DG_HOST_ASSERT(publish_shared_rs == shared_rs_peer_ptrs_opt.has_value());
    DG_HOST_ASSERT(publish_shared_rs != shared_y_opt.has_value());
    torch::Tensor shared_y;
    torch::Tensor shared_rs_workspace;
    torch::Tensor shared_rs_flags;
    torch::Tensor shared_rs_peer_ptrs;
    if (publish_shared_rs) {
        shared_rs_workspace = shared_rs_workspace_opt.value();
        shared_rs_flags = shared_rs_flags_opt.value();
        shared_rs_peer_ptrs = shared_rs_peer_ptrs_opt.value();
        const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
        DG_HOST_ASSERT(not publish_shared_sp_rs or num_shared_tokens % num_ranks == 0);
        DG_HOST_ASSERT(publish_shared_sp_rs or shared_hidden % num_ranks == 0);
        DG_HOST_ASSERT(shared_rs_workspace.is_cuda());
        DG_HOST_ASSERT(shared_rs_workspace.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(shared_rs_workspace.is_contiguous());
        DG_HOST_ASSERT(shared_rs_workspace.dim() == 4);
        DG_HOST_ASSERT(shared_rs_workspace.size(0) >=
                       (publish_shared_sp_rs ? 1 : 3));
        DG_HOST_ASSERT(shared_rs_workspace.size(1) >=
                       (publish_shared_sp_rs ? num_shared_tokens / num_ranks : num_shared_tokens));
        DG_HOST_ASSERT(shared_rs_workspace.size(2) == static_cast<int64_t>(sym_buffer_ptrs.size()));
        DG_HOST_ASSERT(publish_shared_sp_rs ?
            shared_rs_workspace.size(3) == shared_hidden :
            shared_rs_workspace.size(3) * shared_rs_workspace.size(2) == shared_hidden);
        // Cross-repository ABI with the ReduceScatter consumer: flags[0] is
        // the generation index and flags[2] is the byte stride per generation.
        DG_HOST_ASSERT(shared_rs_flags.is_cuda());
        DG_HOST_ASSERT(shared_rs_flags.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(shared_rs_flags.is_contiguous() and shared_rs_flags.numel() >= 9);
        DG_HOST_ASSERT(shared_rs_peer_ptrs.is_cuda());
        DG_HOST_ASSERT(shared_rs_peer_ptrs.scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(shared_rs_peer_ptrs.is_contiguous());
        DG_HOST_ASSERT(shared_rs_peer_ptrs.dim() == 1);
        DG_HOST_ASSERT(shared_rs_peer_ptrs.numel() == static_cast<int64_t>(sym_buffer_ptrs.size()));
    } else {
        shared_y = shared_y_opt.value();
        DG_HOST_ASSERT(shared_y.is_cuda() and shared_y.scalar_type() == torch::kBFloat16 and shared_y.is_contiguous());
        DG_HOST_ASSERT(shared_y.dim() == 2 and shared_y.size(0) == num_shared_tokens and shared_y.size(1) == shared_hidden);
    }
    DG_HOST_ASSERT(shared_l2_acts.is_cuda() and shared_l2_acts.scalar_type() == torch::kBFloat16 and shared_l2_acts.is_contiguous());
    DG_HOST_ASSERT(shared_l2_acts.dim() == 2 and shared_l2_acts.size(0) >= num_shared_tokens);
    const auto shared_intermediate_hidden = static_cast<int>(shared_l2_acts.size(1));
    DG_HOST_ASSERT(shared_intermediate_hidden > 0);
    DG_HOST_ASSERT(shared_l1_weights.is_cuda() and shared_l1_weights.scalar_type() == torch::kBFloat16 and shared_l1_weights.is_contiguous());
    DG_HOST_ASSERT(shared_l2_weights.is_cuda() and shared_l2_weights.scalar_type() == torch::kBFloat16 and shared_l2_weights.is_contiguous());
    DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l1_weights.size(0) == shared_intermediate_hidden * 2 and shared_l1_weights.size(1) == shared_hidden);
    DG_HOST_ASSERT(shared_l2_weights.dim() == 2 and shared_l2_weights.size(0) == shared_hidden and shared_l2_weights.size(1) == shared_intermediate_hidden);
    DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(rms_weight.is_cuda() and rms_weight.scalar_type() == torch::kBFloat16 and rms_weight.is_contiguous());
    DG_HOST_ASSERT(rms_weight.dim() == 1 and rms_weight.size(0) == hidden);

    const auto output_device = y.device();
    const auto is_local_cuda_tensor = [&output_device](const torch::Tensor& tensor) {
        return tensor.is_cuda() and tensor.device() == output_device;
    };
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights) and is_local_cuda_tensor(l2_weights));
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights_sf) and is_local_cuda_tensor(l2_weights_sf));
    DG_HOST_ASSERT(is_local_cuda_tensor(shared_x));
    DG_HOST_ASSERT(publish_shared_rs ?
        (is_local_cuda_tensor(shared_rs_workspace) and is_local_cuda_tensor(shared_rs_flags) and
         is_local_cuda_tensor(shared_rs_peer_ptrs)) :
        is_local_cuda_tensor(shared_y));
    DG_HOST_ASSERT(is_local_cuda_tensor(shared_l2_acts));
    DG_HOST_ASSERT(is_local_cuda_tensor(shared_l1_weights) and is_local_cuda_tensor(shared_l2_weights));
    DG_HOST_ASSERT(is_local_cuda_tensor(rms_weight) and is_local_cuda_tensor(sym_buffer));

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
        DG_HOST_ASSERT(is_local_cuda_tensor(cumulative_local_expert_recv_stats.value()));
    }

    // The BF16 shared intermediate and output are external, so the symmetric
    // communication buffer only needs routed slots.
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, 0);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_per_rank * num_ranks);
    const auto [x, x_sf, topk_idx, topk_weights,
                unused_shared_l1_acts, unused_shared_l1_acts_sf,
                unused_shared_l2_acts, unused_shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(
            y,
            l1_acts, l1_acts_sf,
            l2_acts, l2_acts_sf,
            unused_shared_l1_acts, unused_shared_l1_acts_sf,
            unused_shared_l2_acts, unused_shared_l2_acts_sf,
            l1_weights, l2_weights,
            l1_weights_sf, l2_weights_sf,
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            cumulative_local_expert_recv_stats,
            sym_buffer_ptrs,
            rank_idx, num_max_tokens_per_rank,
            num_experts_per_rank,
            /*num_shared_experts=*/ 1,
            num_tokens, num_topk,
            hidden, intermediate_hidden,
            std::numeric_limits<float>::infinity(), fast_math,
            /*use_situ=*/ true, situ_beta, situ_linear_beta,
            shared_x, shared_l2_acts,
            shared_l1_weights, shared_l2_weights,
            shared_y, rms_weight, rms_epsilon,
            /*use_bf16_shared=*/ true,
            /*apply_rms_norm=*/ true,
            shared_rs_flags, shared_rs_peer_ptrs,
            publish_shared_rs, publish_shared_sp_rs);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

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
    m.def("fp8_fp4_mega_moe_bf16_shared", &fp8_fp4_mega_moe_bf16_shared);
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
    m.def("bf16_mega_moe", &bf16_mega_moe);
#endif
}

} // namespace deep_gemm::mega
