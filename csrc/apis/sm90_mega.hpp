#pragma once

#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <torch/all.h>
#include <torch/library.h>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"
#include "../torch_library_utils.hpp"
#include "../utils/layout.hpp"
#include "../utils/system.hpp"

namespace deep_gemm::mega {

static constexpr int kSM90MegaMoETokenAlignment = 128;

static int get_token_alignment_for_sm90_mega_moe() {
    return kSM90MegaMoETokenAlignment;
}

struct SM90SymmBufferLayoutInfo {
    int64_t num_bytes = 0;
    int64_t input_token_base = 0;
    int64_t input_sf_base = 0;
    int64_t input_topk_idx_base = 0;
    int64_t input_topk_weights_base = 0;
    int64_t l1_token_base = 0;
    int64_t l1_sf_base = 0;
    int64_t l2_token_base = 0;
    int64_t l2_sf_base = 0;
    int num_max_tokens_per_rank = 0;
    int num_topk = 0;
    int hidden = 0;
    int intermediate_hidden = 0;
    int num_max_pool_tokens = 0;
    int num_max_padded_sf_pool_tokens = 0;

    std::vector<int64_t> to_int_list() const {
        return {
            num_bytes,
            input_token_base, input_sf_base,
            input_topk_idx_base, input_topk_weights_base,
            l1_token_base, l1_sf_base,
            l2_token_base, l2_sf_base,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            num_max_pool_tokens, num_max_padded_sf_pool_tokens,
        };
    }

    static SM90SymmBufferLayoutInfo from_int_list(const std::vector<int64_t>& values) {
        DG_HOST_ASSERT(static_cast<int64_t>(values.size()) == 15);
        SM90SymmBufferLayoutInfo info;
        info.num_bytes = values[0];
        info.input_token_base = values[1];
        info.input_sf_base = values[2];
        info.input_topk_idx_base = values[3];
        info.input_topk_weights_base = values[4];
        info.l1_token_base = values[5];
        info.l1_sf_base = values[6];
        info.l2_token_base = values[7];
        info.l2_sf_base = values[8];
        info.num_max_tokens_per_rank = static_cast<int>(values[9]);
        info.num_topk = static_cast<int>(values[10]);
        info.hidden = static_cast<int>(values[11]);
        info.intermediate_hidden = static_cast<int>(values[12]);
        info.num_max_pool_tokens = static_cast<int>(values[13]);
        info.num_max_padded_sf_pool_tokens = static_cast<int>(values[14]);
        return info;
    }
};

using SM90SymmBufferSlice = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

static SM90SymmBufferLayoutInfo build_sm90_symm_buffer_layout(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    if (not use_fp8_dispatch)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE currently supports FP8 dispatch only");
    if (activation != "swiglu")
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE currently supports the swiglu activation only");
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and
                   num_max_tokens_per_rank % kSM90MegaMoETokenAlignment == 0);
    if (hidden <= 0 or hidden % 256 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires hidden to be a positive multiple of 256 for combine vectorization");
    if (intermediate_hidden <= 0 or intermediate_hidden % 128 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires intermediate_hidden to be a positive multiple of 128");

    // Workspace bytes
    const auto num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts / num_ranks);
    const auto workspace = layout::Workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        num_max_pool_tokens);

    // Layouts
    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    // Pool SFs are MN-major; their 128-token outer stride remains TMA-aligned
    // even when one logical token's SF byte count is not.
    const auto fp8_sf_layout = layout::Data(hidden / 32, false);
    const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden / 16, false);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
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

    // BLOCK_M=64 is the worst case for the allocated SF pool capacity.
    const auto num_max_padded_sf_pool_tokens =
        layout::get_num_sf_ring_tokens(num_max_pool_tokens, 64);

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    SM90SymmBufferLayoutInfo info;
    info.num_bytes = reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr());
    info.input_token_base = reinterpret_cast<int64_t>(input_token_buffer.base);
    info.input_sf_base = reinterpret_cast<int64_t>(input_sf_buffer.base);
    info.input_topk_idx_base = reinterpret_cast<int64_t>(input_topk_idx_buffer.base);
    info.input_topk_weights_base = reinterpret_cast<int64_t>(input_topk_weights_buffer.base);
    info.l1_token_base = reinterpret_cast<int64_t>(l1_token_buffer.base);
    info.l1_sf_base = reinterpret_cast<int64_t>(l1_sf_buffer.base);
    info.l2_token_base = reinterpret_cast<int64_t>(l2_token_buffer.base);
    info.l2_sf_base = reinterpret_cast<int64_t>(l2_sf_buffer.base);
    info.num_max_tokens_per_rank = num_max_tokens_per_rank;
    info.num_topk = num_topk;
    info.hidden = hidden;
    info.intermediate_hidden = intermediate_hidden;
    info.num_max_pool_tokens = num_max_pool_tokens;
    info.num_max_padded_sf_pool_tokens = num_max_padded_sf_pool_tokens;
    return info;
}

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    const auto info = build_sm90_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, use_fp8_dispatch, activation);
    return {info.num_bytes, info.to_int_list()};
}

static SM90SymmBufferSlice slice_sm90_symm_buffer_from_layout(
    const torch::Tensor& buffer, const SM90SymmBufferLayoutInfo& info) {
    DG_HOST_ASSERT(buffer.nbytes() >= static_cast<size_t>(info.num_bytes));

    // `x_sf` is K-major; pool scale factors are M-major.
    auto x = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.input_token_base),
        {info.num_max_tokens_per_rank, info.hidden},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
    auto x_sf = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.input_sf_base),
        {info.num_max_tokens_per_rank, info.hidden / 128},
        torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
    auto topk_idx = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.input_topk_idx_base),
        {info.num_max_tokens_per_rank, info.num_topk},
        torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
    auto topk_weights = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.input_topk_weights_base),
        {info.num_max_tokens_per_rank, info.num_topk},
        torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
    auto l1_acts = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.l1_token_base),
        {info.num_max_pool_tokens, info.hidden},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
    auto l1_acts_sf = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.l1_sf_base),
        {info.num_max_padded_sf_pool_tokens, info.hidden / 128},
        {1, info.num_max_padded_sf_pool_tokens},
        torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
    auto l2_acts = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.l2_token_base),
        {info.num_max_pool_tokens, info.intermediate_hidden},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
    auto l2_acts_sf = torch::from_blob(
        math::advance_ptr(buffer.data_ptr(), info.l2_sf_base),
        {info.num_max_padded_sf_pool_tokens, info.intermediate_hidden / 64},
        {1, info.num_max_padded_sf_pool_tokens},
        torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
    return {x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf};
}

// SM90 (Hopper) FP8 MegaMoE entry point.
//
// Mirrors `fp8_fp4_mega_moe` but expects FP8 (e4m3) weights with per-128 channel
// float scale factors. Top-level routing (which entry to call) is the caller's
// responsibility (see `deep_gemm/mega/__init__.py`).
static void fp8_mega_moe(
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

    // Architecture check
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major != 9)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires a compute capability 9.x GPU");

    // Config checks: SM90 uses block (128, 128) float SF for weights,
    // per-token per-128-K float SF for activations.
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    if (rm != 128 or rn != 128 or rk != 128)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires recipe=(128, 128, 128)");
    if (activation != "swiglu")
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE currently supports the swiglu activation only");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks: SM90 weights must be FP8 e4m3, K-major
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

    // The combine warp consumes eight values per lane, so hidden must satisfy
    // at least its base 256-element vector width. The selected launch config is
    // checked against the exact one/two/four-chunk layout before JIT compilation.
    if (hidden <= 0 or hidden % 256 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires hidden to be a positive multiple of 256 for combine vectorization");
    if (intermediate_hidden <= 0 or intermediate_hidden % 128 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires intermediate_hidden to be a positive multiple of 128");

    // Weight SFs are raw global-memory loads in natural MN-major order.
    constexpr int kGranMN = 128, kGranK = 128;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    if (not l1_weights_sf.is_contiguous() or not l2_weights_sf.is_contiguous())
        DG_HOST_UNREACHABLE(
            "SM90 FP8 MegaMoE weight scale factors must use contiguous natural layouts");

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() ==
                       num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto layout_info = build_sm90_symm_buffer_layout(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(layout_info.num_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] =
        slice_sm90_symm_buffer_from_layout(sym_buffer, layout_info);

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

} // namespace deep_gemm::mega

namespace deep_gemm::torch_registration {

using namespace deep_gemm::torch_utils;

static int64_t get_token_alignment_for_sm90_mega_moe() {
    return static_cast<int64_t>(mega::get_token_alignment_for_sm90_mega_moe());
}

static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_sm90_mega_moe(
    const int64_t& num_ranks, const int64_t& num_experts,
    const int64_t& num_max_tokens_per_rank, const int64_t& num_topk,
    const int64_t& hidden, const int64_t& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    return mega::get_symm_buffer_size_for_sm90_mega_moe(
        static_cast<int>(num_ranks), static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
        static_cast<int>(hidden), static_cast<int>(intermediate_hidden),
        use_fp8_dispatch, activation);
}

static mega::SM90SymmBufferSlice _slice_symm_buffer_for_sm90_mega_moe(
    const torch::Tensor& buffer, const std::vector<int64_t>& layout_info) {
    return mega::slice_sm90_symm_buffer_from_layout(
        buffer, mega::SM90SymmBufferLayoutInfo::from_int_list(layout_info));
}

static void fp8_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const c10::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int64_t& rank_idx,
    const int64_t& num_max_tokens_per_rank,
    const int64_t& num_experts, const int64_t& num_topk,
    const std::vector<int64_t>& recipe,
    const std::string& activation,
    const c10::optional<double>& activation_clamp,
    const bool& fast_math) {
    mega::fp8_mega_moe(
        y,
        std::make_tuple(l1_weights, l1_weights_sf),
        std::make_tuple(l2_weights, l2_weights_sf),
        cumulative_local_expert_recv_stats,
        sym_buffer, sym_buffer_ptrs,
        static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_experts), static_cast<int>(num_topk),
        list_to_tuple3(recipe), activation,
        activation_clamp.has_value()
            ? std::make_optional(static_cast<float>(activation_clamp.value()))
            : std::nullopt,
        fast_math);
}

} // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def(
        "get_token_alignment_for_sm90_mega_moe() -> int",
        TORCH_FN(deep_gemm::torch_registration::get_token_alignment_for_sm90_mega_moe));
    m.def(
        "get_symm_buffer_size_for_sm90_mega_moe(int num_ranks, int num_experts, int num_max_tokens_per_rank, int num_topk, int hidden, int intermediate_hidden, bool use_fp8_dispatch, str activation) -> (int, int[])",
        TORCH_FN(deep_gemm::torch_registration::get_symm_buffer_size_for_sm90_mega_moe));
    m.def(
        "_slice_symm_buffer_for_sm90_mega_moe(Tensor buffer, int[] layout_info) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "fp8_mega_moe(Tensor(y!) y, Tensor l1_weights, Tensor l1_weights_sf, Tensor l2_weights, Tensor l2_weights_sf, Tensor(cumulative_local_expert_recv_stats!)? cumulative_local_expert_recv_stats, Tensor(sym_buffer!) sym_buffer, int[] sym_buffer_ptrs, int rank_idx, int num_max_tokens_per_rank, int num_experts, int num_topk, int[3] recipe, str activation, float? activation_clamp, bool fast_math) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::torch_registration;

#if DG_TENSORMAP_COMPATIBLE
    m.impl(
        "_slice_symm_buffer_for_sm90_mega_moe",
        TORCH_FN(_slice_symm_buffer_for_sm90_mega_moe));
    m.impl("fp8_mega_moe", TORCH_FN(fp8_mega_moe));
#endif
}
