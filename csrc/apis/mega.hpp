#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include <torch/all.h>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include "../utils/math.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp4_fp4_mega_moe.hpp"
#include <torch/library.h>
#include "../torch_library_utils.hpp"

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

struct SymmBufferLayoutInfo {
    int64_t num_bytes = 0;
    int64_t input_token_base = 0;
    int64_t input_sf_base = 0;
    int64_t input_topk_idx_base = 0;
    int64_t input_topk_weights_base = 0;
    int64_t shared_l1_sf_base = 0;
    int64_t shared_l2_token_base = 0;
    int64_t shared_l2_sf_base = 0;
    int64_t l1_token_base = 0;
    int64_t l1_sf_base = 0;
    int64_t l2_token_base = 0;
    int64_t l2_sf_base = 0;
    MmaKind mma_kind = MmaKind::BF16;
    bool with_sf = false;
    int sf_gran_k = 0;
    int num_max_tokens_per_rank = 0;
    int num_topk = 0;
    int hidden = 0;
    int intermediate_hidden = 0;
    int num_shared_experts = 0;
    int shared_intermediate_hidden = 0;
    int num_ring_tokens = 0;
    int num_sf_ring_tokens = 0;

    // Flatten into a plain `int[]` so it can cross the TORCH_LIBRARY boundary
    std::vector<int64_t> to_int_list() const {
        return {
            num_bytes, input_token_base, input_sf_base, input_topk_idx_base, input_topk_weights_base,
            shared_l1_sf_base, shared_l2_token_base, shared_l2_sf_base,
            l1_token_base, l1_sf_base, l2_token_base, l2_sf_base,
            static_cast<int64_t>(mma_kind), static_cast<int64_t>(with_sf), sf_gran_k,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, num_shared_experts, shared_intermediate_hidden,
            num_ring_tokens, num_sf_ring_tokens,
        };
    }

    static SymmBufferLayoutInfo from_int_list(const std::vector<int64_t>& values) {
        DG_HOST_ASSERT(static_cast<int64_t>(values.size()) == 23);
        SymmBufferLayoutInfo info;
        info.num_bytes = values[0];
        info.input_token_base = values[1];
        info.input_sf_base = values[2];
        info.input_topk_idx_base = values[3];
        info.input_topk_weights_base = values[4];
        info.shared_l1_sf_base = values[5];
        info.shared_l2_token_base = values[6];
        info.shared_l2_sf_base = values[7];
        info.l1_token_base = values[8];
        info.l1_sf_base = values[9];
        info.l2_token_base = values[10];
        info.l2_sf_base = values[11];
        info.mma_kind = static_cast<MmaKind>(values[12]);
        // `with_sf` is a bool, encoded as 0/1 since the list is all `int64_t`.
        info.with_sf = values[13] != 0;
        info.sf_gran_k = static_cast<int>(values[14]);
        info.num_max_tokens_per_rank = static_cast<int>(values[15]);
        info.num_topk = static_cast<int>(values[16]);
        info.hidden = static_cast<int>(values[17]);
        info.intermediate_hidden = static_cast<int>(values[18]);
        info.num_shared_experts = static_cast<int>(values[19]);
        info.shared_intermediate_hidden = static_cast<int>(values[20]);
        info.num_ring_tokens = static_cast<int>(values[21]);
        info.num_sf_ring_tokens = static_cast<int>(values[22]);
        return info;
    }
};

static SymmBufferLayoutInfo build_symm_buffer_layout(
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

    SymmBufferLayoutInfo layout_info;
    layout_info.num_bytes = mega_buffer.get_num_bytes();
    layout_info.input_token_base = reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base);
    layout_info.input_sf_base = reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base);
    layout_info.input_topk_idx_base = reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base);
    layout_info.input_topk_weights_base = reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base);
    layout_info.shared_l1_sf_base = reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base);
    layout_info.shared_l2_token_base = reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base);
    layout_info.shared_l2_sf_base = reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base);
    layout_info.l1_token_base = reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base);
    layout_info.l1_sf_base = reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base);
    layout_info.l2_token_base = reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base);
    layout_info.l2_sf_base = reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base);
    layout_info.mma_kind = mma_kind;
    layout_info.with_sf = with_sf;
    layout_info.sf_gran_k = sf_gran_k;
    layout_info.num_max_tokens_per_rank = num_max_tokens_per_rank;
    layout_info.num_topk = num_topk;
    layout_info.hidden = hidden;
    layout_info.intermediate_hidden = intermediate_hidden;
    layout_info.num_shared_experts = num_shared_experts;
    layout_info.shared_intermediate_hidden = shared_intermediate_hidden;
    layout_info.num_ring_tokens = num_ring_tokens;
    layout_info.num_sf_ring_tokens = num_sf_ring_tokens;
    return layout_info;
}

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_shared_experts = 0) {
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, mma_type, activation, num_shared_experts);
    return std::make_tuple(layout_info.num_bytes, layout_info.to_int_list());
}

using SymmBufferSlice = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                   at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                   at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

static SymmBufferSlice slice_symm_buffer_from_layout(
    const torch::Tensor& buffer, const SymmBufferLayoutInfo& layout_info) {
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    // NOTES: for NVFP4, token views are packed E2M1 bytes (2 elements each) and SF
    // views pack 4 E4M3 bytes per `int`
    const bool is_fp4 = layout_info.mma_kind == MmaKind::NVFP4;
    const auto token_dtype = is_fp4
        ? torch::kUInt8
        : (layout_info.with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16);
    const auto hidden_cols = is_fp4 ? layout_info.hidden / 2 : layout_info.hidden;
    const auto intermediate_cols = is_fp4
        ? layout_info.intermediate_hidden / 2
        : layout_info.intermediate_hidden;
    const auto hidden_sf_cols = layout_info.with_sf
        ? layout_info.hidden / (layout_info.sf_gran_k * 4) : 0;
    const auto intermediate_sf_cols = layout_info.with_sf
        ? layout_info.intermediate_hidden / (layout_info.sf_gran_k * 4) : 0;
    auto x = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_token_base),
        {layout_info.num_max_tokens_per_rank, hidden_cols},
        torch::TensorOptions().dtype(token_dtype).device(buffer.device()));
    auto x_sf = layout_info.with_sf ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_sf_base),
        {layout_info.num_max_tokens_per_rank, hidden_sf_cols},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
    auto topk_idx = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_idx_base),
        {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
        torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
    auto topk_weights = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.input_topk_weights_base),
        {layout_info.num_max_tokens_per_rank, layout_info.num_topk},
        torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

    // NVFP4 shared experts consume caller-provided BF16 inputs and do not use SFs.
    const bool shared_with_sf = layout_info.with_sf and not is_fp4;
    auto shared_l1_acts = is_fp4 ? torch::Tensor() : x;
    auto shared_l1_acts_sf = (shared_with_sf and layout_info.num_shared_experts > 0) ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.shared_l1_sf_base),
        {layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank), layout_info.hidden / 128},
        {1, layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank)},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
    auto shared_l2_acts = layout_info.num_shared_experts > 0 ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.shared_l2_token_base),
        {layout_info.num_max_tokens_per_rank, layout_info.shared_intermediate_hidden},
        torch::TensorOptions().dtype(shared_with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device())) : torch::Tensor();
    auto shared_l2_acts_sf = (shared_with_sf and layout_info.num_shared_experts > 0) ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.shared_l2_sf_base),
        {layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank), layout_info.shared_intermediate_hidden / 128},
        {1, layout::get_num_max_shared_sf_tokens(layout_info.num_max_tokens_per_rank)},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();

    auto l1_acts = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l1_token_base),
        {layout_info.num_ring_tokens, hidden_cols},
        torch::TensorOptions().dtype(token_dtype).device(buffer.device()));
    auto l1_acts_sf = layout_info.with_sf ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l1_sf_base),
        {layout_info.num_sf_ring_tokens, hidden_sf_cols},
        {1, layout_info.num_sf_ring_tokens},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
    auto l2_acts = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l2_token_base),
        {layout_info.num_ring_tokens, intermediate_cols},
        torch::TensorOptions().dtype(token_dtype).device(buffer.device()));
    auto l2_acts_sf = layout_info.with_sf ? torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), layout_info.l2_sf_base),
        {layout_info.num_sf_ring_tokens, intermediate_sf_cols},
        {1, layout_info.num_sf_ring_tokens},
        torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
    return {x, x_sf, topk_idx, topk_weights,
            shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
            l1_acts, l1_acts_sf, l2_acts, l2_acts_sf};
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

    // Check buffer bytes and slice views from one shared layout plan.
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] =
        slice_symm_buffer_from_layout(sym_buffer, layout_info);

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

    // Check buffer bytes and slice views from one shared layout plan.
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp4xfp4", activation, num_shared_experts);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
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

    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] =
        slice_symm_buffer_from_layout(sym_buffer, layout_info);

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

    // Check buffer bytes and slice views from one shared layout plan.
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_shared_experts
    );
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    const auto [x, _x_sf, topk_idx, topk_weights,
                shared_l1_acts, _shared_l1_acts_sf, shared_l2_acts, _shared_l2_acts_sf,
                l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf] =
        slice_symm_buffer_from_layout(sym_buffer, layout_info);

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

}  // namespace deep_gemm::mega

namespace deep_gemm::torch_registration {

using namespace deep_gemm::torch_utils;

static int64_t get_token_alignment_for_mega_moe() {
    return static_cast<int64_t>(mega::get_token_alignment_for_mega_moe());
}

static int64_t get_block_m_for_mega_moe(
    const int64_t& num_ranks, const int64_t& num_experts,
    const int64_t& num_max_tokens_per_rank, const int64_t& num_tokens,
    const int64_t& num_topk, const std::string& mma_type) {
    return static_cast<int64_t>(mega::get_block_m_for_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_tokens),
        static_cast<int>(num_topk), mma_type));
}

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_mega_moe(
    const int64_t& num_ranks, const int64_t& num_experts,
    const int64_t& num_max_tokens_per_rank, const int64_t& num_topk,
    const int64_t& hidden, const int64_t& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int64_t& num_shared_experts) {
    return mega::get_symm_buffer_size_for_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        mma_type, activation, static_cast<int>(num_shared_experts));
}

static mega::SymmBufferSlice _slice_symm_buffer_for_mega_moe(
    const torch::Tensor& buffer,
    const std::vector<int64_t>& layout_info) {
    return mega::slice_symm_buffer_from_layout(
        buffer, mega::SymmBufferLayoutInfo::from_int_list(layout_info));
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const c10::optional<torch::Tensor>& shared_l1_weights,
    const c10::optional<torch::Tensor>& shared_l1_weights_sf,
    const c10::optional<torch::Tensor>& shared_l2_weights,
    const c10::optional<torch::Tensor>& shared_l2_weights_sf,
    const c10::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts, const int64_t& num_topk,
    const std::vector<int64_t>& recipe,
    const std::string& activation,
    const c10::optional<double>& activation_clamp,
    const bool& fast_math,
    const c10::optional<double>& situ_beta,
    const c10::optional<double>& situ_linear_beta) {
    std::optional<std::tuple<torch::Tensor, torch::Tensor>> shared_l1_opt = std::nullopt;
    std::optional<std::tuple<torch::Tensor, torch::Tensor>> shared_l2_opt = std::nullopt;
    if (shared_l1_weights.has_value()) {
        DG_HOST_ASSERT(shared_l1_weights_sf.has_value() and shared_l2_weights.has_value() and shared_l2_weights_sf.has_value());
        shared_l1_opt = std::make_tuple(shared_l1_weights.value(), shared_l1_weights_sf.value());
        shared_l2_opt = std::make_tuple(shared_l2_weights.value(), shared_l2_weights_sf.value());
    } else {
        DG_HOST_ASSERT(not shared_l1_weights_sf.has_value() and not shared_l2_weights.has_value() and not shared_l2_weights_sf.has_value());
    }

    mega::fp8_fp4_mega_moe(
        y,
        std::make_tuple(l1_weights, l1_weights_sf),
        std::make_tuple(l2_weights, l2_weights_sf),
        shared_l1_opt,
        shared_l2_opt,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        sym_buffer_ptrs,
        static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        list_to_tuple3(recipe),
        activation,
        activation_clamp.has_value()
            ? std::make_optional(static_cast<float>(activation_clamp.value()))
            : std::nullopt,
        fast_math,
        situ_beta.has_value()
            ? std::make_optional(static_cast<float>(situ_beta.value()))
            : std::nullopt,
        situ_linear_beta.has_value()
            ? std::make_optional(static_cast<float>(situ_linear_beta.value()))
            : std::nullopt);
}

static void fp4_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const c10::optional<torch::Tensor>& shared_l1_weights,
    const c10::optional<torch::Tensor>& shared_l2_weights,
    const c10::optional<torch::Tensor>& x_bf16,
    const c10::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts, const int64_t& num_topk,
    const std::vector<int64_t>& recipe,
    const std::string& activation,
    const c10::optional<double>& activation_clamp,
    const bool& fast_math,
    const c10::optional<torch::Tensor>& l1_alphas,
    const c10::optional<torch::Tensor>& l2_alphas,
    const c10::optional<torch::Tensor>& a2_scales,
    const double& routed_scaling_factor) {
    mega::fp4_fp4_mega_moe(
        y,
        std::make_tuple(l1_weights, l1_weights_sf),
        std::make_tuple(l2_weights, l2_weights_sf),
        shared_l1_weights, shared_l2_weights, x_bf16,
        cumulative_local_expert_recv_stats,
        sym_buffer, sym_buffer_ptrs,
        static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        list_to_tuple3(recipe), activation,
        activation_clamp.has_value()
            ? std::make_optional(static_cast<float>(activation_clamp.value()))
            : std::nullopt,
        fast_math, l1_alphas, l2_alphas, a2_scales,
        static_cast<float>(routed_scaling_factor));
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const c10::optional<torch::Tensor>& shared_l1_weights,
    const c10::optional<torch::Tensor>& shared_l2_weights,
    const c10::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts, const int64_t& num_topk,
    const std::string& activation,
    const c10::optional<double>& activation_clamp,
    const bool& fast_math) {
    mega::bf16_mega_moe(
        y, l1_weights, l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        sym_buffer_ptrs,
        static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        activation,
        activation_clamp.has_value()
            ? std::make_optional(static_cast<float>(activation_clamp.value()))
            : std::nullopt,
        fast_math);
}

}  // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def(
        "get_token_alignment_for_mega_moe() -> int",
        TORCH_FN(deep_gemm::torch_registration::get_token_alignment_for_mega_moe));
    m.def(
        "get_block_m_for_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_tokens, int num_topk, str mma_type) -> int",
        TORCH_FN(deep_gemm::torch_registration::get_block_m_for_mega_moe));
    m.def(
        "get_symm_buffer_size_for_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, str mma_type, str activation, int num_shared_experts=0) -> (int, int[])",
        TORCH_FN(deep_gemm::torch_registration::get_symm_buffer_size_for_mega_moe));
    m.def(
        "_slice_symm_buffer_for_mega_moe(Tensor buffer, int[] layout_info) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "fp8_fp4_mega_moe(Tensor(y!) y, Tensor l1_weights, Tensor l1_weights_sf, Tensor l2_weights, Tensor l2_weights_sf, Tensor? shared_l1_weights, Tensor? shared_l1_weights_sf, Tensor? shared_l2_weights, Tensor? shared_l2_weights_sf, Tensor(cumulative_local_expert_recv_stats!)? cumulative_local_expert_recv_stats, Tensor(sym_buffer!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, int[3] recipe, str activation, float? activation_clamp, bool fast_math, float? situ_beta=None, float? situ_linear_beta=None) -> ()");
    m.def(
        "fp4_fp4_mega_moe(Tensor(y!) y, Tensor l1_weights, Tensor l1_weights_sf, Tensor l2_weights, Tensor l2_weights_sf, Tensor? shared_l1_weights, Tensor? shared_l2_weights, Tensor? x_bf16, Tensor(cumulative_local_expert_recv_stats!)? cumulative_local_expert_recv_stats, Tensor(sym_buffer!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, int[3] recipe, str activation, float? activation_clamp, bool fast_math, Tensor? l1_alphas, Tensor? l2_alphas, Tensor? a2_scales, float routed_scaling_factor) -> ()");
    m.def(
        "bf16_mega_moe(Tensor(y!) y, Tensor l1_weights, Tensor l2_weights, Tensor? shared_l1_weights, Tensor? shared_l2_weights, Tensor(cumulative_local_expert_recv_stats!)? cumulative_local_expert_recv_stats, Tensor(sym_buffer!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, str activation, float? activation_clamp, bool fast_math) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::torch_registration;

#if DG_TENSORMAP_COMPATIBLE
    m.impl("_slice_symm_buffer_for_mega_moe", TORCH_FN(_slice_symm_buffer_for_mega_moe));
    m.impl("fp8_fp4_mega_moe", TORCH_FN(fp8_fp4_mega_moe));
    m.impl("fp4_fp4_mega_moe", TORCH_FN(fp4_fp4_mega_moe));
    m.impl("bf16_mega_moe", TORCH_FN(bf16_mega_moe));
#endif
}
