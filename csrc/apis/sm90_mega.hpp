#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "mega.hpp"
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_fp8_fp4_mega_moe.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"
#include "../utils/layout.hpp"
#include "../utils/system.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_sm90_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static bool is_packed_fp4_storage_sm90(const torch::Tensor& t) {
    return t.scalar_type() == kPackedFP4 or t.scalar_type() == torch::kByte;
}

static std::tuple<int, int, int> check_grouped_ab_sm90_fp4_mega_moe(const torch::Tensor& ab) {
    const auto [num_groups, mn, packed_k] = get_shape<3>(ab);
    DG_HOST_ASSERT(is_packed_fp4_storage_sm90(ab));
    DG_HOST_ASSERT(get_major_type_ab(ab) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(packed_k > 0 and packed_k % 64 == 0);
    return {num_groups, mn, packed_k * 2};
}

static void check_sm90_fp4_sfb_layout(const torch::Tensor& sf,
                                      const int& mn, const int& k,
                                      const int& num_groups) {
    DG_HOST_ASSERT(sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(sf.dim() == 3);
    DG_HOST_ASSERT(sf.size(0) == num_groups);
    DG_HOST_ASSERT(sf.size(1) == mn);
    DG_HOST_ASSERT(sf.size(2) == ceil_div(k, 128));
    DG_HOST_ASSERT(sf.is_contiguous());
}

struct FP4SM90APIDefaults {
    bool math_wg_participates_in_decode;
    int num_math_wg_decode_warps;
    int first_decode_assist_warp;
    bool wide_load_decode;
    bool early_b_decode;
    bool decode_done_mbarrier;
    bool l2_arrival_counter;
    bool ss_nsplit;
    bool swap_ab;
    bool swap_ab_fast_amax;
};

static FP4SM90APIDefaults get_fp4_sm90_api_defaults(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    // Shape bands exclude kernel tile/thread constraints; JIT heuristics add those as kernel bands.
    const bool fp4_flash_shape = intermediate_hidden <= 2048;
    const bool fp4_pro_shape = intermediate_hidden >= 3072;
    const bool fp4_middle_shape = !fp4_flash_shape and !fp4_pro_shape;
    const bool fp4_decode_lookahead_shape_band =
        expected_tokens_per_expert >= 3.0f and expected_tokens_per_expert <= 6.0f;
    const bool fp4_bigband_lookahead_shape_band =
        expected_tokens_per_expert >= 12.0f and expected_tokens_per_expert <= 24.0f;
    const bool fp4_b4_skip_decode_shape_band =
        expected_tokens_per_expert >= 0.5f and expected_tokens_per_expert < 1.0f;
    const bool fp4_pro_single_token_per_expert_shape_band =
        fp4_pro_shape and
        expected_tokens_per_expert >= 1.0f and expected_tokens_per_expert < 1.5f and
        num_experts_per_rank % 8 == 0;
    const bool fp4_pro_split_n_mbarrier_shape_band =
        fp4_pro_shape and
        expected_tokens_per_expert > 0.0f and expected_tokens_per_expert < 64.0f;
    const bool fp4_pro_two_tokens_per_expert_shape_band =
        fp4_pro_shape and
        expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert < 3.0f;
    const bool fp4_pro_mid_decode_assist_shape_band =
        fp4_pro_shape and
        expected_tokens_per_expert >= 6.0f and expected_tokens_per_expert < 12.0f;
    const bool fp4_pro_large_decode_assist_shape_band =
        fp4_pro_shape and
        expected_tokens_per_expert >= 24.0f and expected_tokens_per_expert < 64.0f;
    const bool fp4_flash_two_tokens_per_expert_shape_band =
        fp4_flash_shape and
        expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert < 2.0f;
    const bool fp4_flash_half_token_per_expert_shape_band =
        fp4_flash_shape and
        expected_tokens_per_expert >= 0.375f and expected_tokens_per_expert < 0.5f;
    const bool fp4_flash_decode_lookahead_shape_band =
        fp4_flash_shape and
        expected_tokens_per_expert >= 3.0f and expected_tokens_per_expert < 6.0f;
    const bool fp4_flash_wide_load_decode_shape_band =
        fp4_flash_shape and
        expected_tokens_per_expert >= 6.0f and expected_tokens_per_expert < 64.0f;
    const bool fp4_pro_wide_load_decode_shape_band =
        fp4_pro_shape and
        expected_tokens_per_expert > 0.0f and expected_tokens_per_expert < 64.0f;
    const bool fp4_flash_split_n_mbarrier_shape_band =
        fp4_flash_shape and
        expected_tokens_per_expert >= 0.75f and expected_tokens_per_expert < 64.0f;
    const bool fp4_flash_small_mbarrier_shape_band =
        fp4_flash_shape and
        expected_tokens_per_expert > 0.0f and expected_tokens_per_expert < 0.5f;
    const bool fp4_2wg_decode_offload_shape_band =
        expected_tokens_per_expert >= 64.0f;
    const bool fp4_shared_decode_assist_shape_band =
        ((expected_tokens_per_expert > 0.0f and expected_tokens_per_expert < 0.375f) or
         fp4_flash_half_token_per_expert_shape_band or
         fp4_b4_skip_decode_shape_band or fp4_decode_lookahead_shape_band or
         fp4_flash_split_n_mbarrier_shape_band or
         fp4_pro_mid_decode_assist_shape_band or fp4_pro_large_decode_assist_shape_band or
         fp4_bigband_lookahead_shape_band or fp4_2wg_decode_offload_shape_band);
    const bool default_math_wg_decode =
        fp4_shared_decode_assist_shape_band or
        (expected_tokens_per_expert >= 1.0f and expected_tokens_per_expert < 2.0f) or
        fp4_pro_two_tokens_per_expert_shape_band;
    const bool math_wg_participates_in_decode =
        !default_math_wg_decode;
    const bool default_skip_loader_decode_assist =
        fp4_shared_decode_assist_shape_band or
        fp4_pro_single_token_per_expert_shape_band or
        (expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert < 3.0f);
    const bool default_wide_load_decode =
        fp4_pro_wide_load_decode_shape_band or
        fp4_flash_half_token_per_expert_shape_band or
        fp4_flash_two_tokens_per_expert_shape_band or
        fp4_flash_wide_load_decode_shape_band;
    const bool default_ss_early_b_decode =
        ((expected_tokens_per_expert >= 1.5f and expected_tokens_per_expert <= 3.0f and
          !fp4_pro_two_tokens_per_expert_shape_band and
          !fp4_flash_two_tokens_per_expert_shape_band and
          !fp4_flash_decode_lookahead_shape_band) or
         fp4_2wg_decode_offload_shape_band);
    const bool fp4_middle_decode_lookahead_mbarrier_shape_band =
        fp4_middle_shape and fp4_decode_lookahead_shape_band;
    const bool fp4_middle_bigband_mbarrier_shape_band =
        fp4_middle_shape and fp4_bigband_lookahead_shape_band;
    const bool default_decode_done_mbarrier =
        fp4_pro_split_n_mbarrier_shape_band or
        fp4_flash_split_n_mbarrier_shape_band or
        fp4_flash_small_mbarrier_shape_band or
        fp4_middle_decode_lookahead_mbarrier_shape_band or
        fp4_middle_bigband_mbarrier_shape_band or
        fp4_2wg_decode_offload_shape_band;
    const bool default_l2_arrival_counter =
        ((fp4_flash_shape and
          expected_tokens_per_expert >= 0.375f and expected_tokens_per_expert < 0.75f) or
         (fp4_pro_shape and
          expected_tokens_per_expert >= 0.25f and expected_tokens_per_expert < 0.375f));
    const bool default_swap_ab =
        (fp4_flash_shape or fp4_pro_shape) and
        expected_tokens_per_expert > 0.0f and expected_tokens_per_expert <= 24.0f;
    const bool default_swap_ab_fast_amax =
        fp4_pro_shape and
        expected_tokens_per_expert >= 12.0f and expected_tokens_per_expert <= 24.0f;
    return {
        math_wg_participates_in_decode,
        math_wg_participates_in_decode ? 4 : 0,
        default_skip_loader_decode_assist ? 2 : 0,
        default_wide_load_decode,
        default_ss_early_b_decode,
        default_decode_done_mbarrier,
        default_l2_arrival_counter,
        expected_tokens_per_expert >= 64.0f,
        default_swap_ab,
        default_swap_ab_fast_amax
    };
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(use_fp8_dispatch);
    DG_HOST_ASSERT(activation == "swiglu");

    const auto workspace = layout::SM90Workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);

    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    const int sm90_l2_act_sf_gran_k = 64;
    const auto fp8_intermediate_sf_layout =
        layout::Data(intermediate_hidden * static_cast<int>(sizeof(float)) / sm90_l2_act_sf_gran_k);
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
    int num_max_padded_sf_pool_tokens = 0;
    for (int block_m: layout::kCandidateBlockM) {
        num_max_padded_sf_pool_tokens = std::max(
            num_max_padded_sf_pool_tokens,
            layout::get_num_sf_ring_tokens(num_max_pool_tokens, block_m)
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

    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);

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
            {num_max_padded_sf_pool_tokens, intermediate_hidden / sm90_l2_act_sf_gran_k},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()), slice_input_buffers};
}

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
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
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

static void fp8_fp4_mega_moe_sm90(
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
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_sm90_fp4_mega_moe(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_sm90_fp4_mega_moe(l2_weights);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    check_sm90_fp4_sfb_layout(l1_weights_sf, intermediate_hidden * 2, hidden,
                              num_experts_per_rank);
    check_sm90_fp4_sfb_layout(l2_weights_sf, hidden, intermediate_hidden,
                              num_experts_per_rank);

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
    (void)x;
    (void)x_sf;
    (void)topk_idx;
    (void)topk_weights;

    DG_HOST_ASSERT(get_env<int>("DG_USE_FP4_ACTS") == 0);
    DG_HOST_ASSERT(get_env<int>("DG_USE_FP8_COMBINE") == 0);

    const auto fp4_defaults = get_fp4_sm90_api_defaults(
        num_experts_per_rank, num_tokens, num_topk, intermediate_hidden);
    sm90_fp8_fp4_mega_moe(y,
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
                          activation_clamp, fast_math,
                          fp4_defaults.math_wg_participates_in_decode,
                          fp4_defaults.num_math_wg_decode_warps,
                          fp4_defaults.first_decode_assist_warp,
                          fp4_defaults.wide_load_decode,
                          fp4_defaults.early_b_decode,
                          fp4_defaults.decode_done_mbarrier,
                          fp4_defaults.l2_arrival_counter,
                          fp4_defaults.ss_nsplit,
                          fp4_defaults.swap_ab,
                          fp4_defaults.swap_ab_fast_amax);

    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_sm90_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_sm90_mega_moe", &get_token_alignment_for_sm90_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_mega_moe", &get_symm_buffer_size_for_sm90_mega_moe);
    m.def("fp8_fp4_mega_moe_sm90", &fp8_fp4_mega_moe_sm90);
    m.def("fp8_mega_moe", &fp8_mega_moe);
#endif
}

} // namespace deep_gemm::mega
