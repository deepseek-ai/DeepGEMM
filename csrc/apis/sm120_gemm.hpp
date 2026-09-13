#pragma once

#include "layout.hpp"
#include "../jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp"

namespace deep_gemm::gemm {

static torch::Tensor sm120_to_k_major(const torch::Tensor& t, const cute::UMMA::Major& major,
                                      const int& logical_mn) {
    if (major == cute::UMMA::Major::K)
        return t;
    if (t.scalar_type() != kPackedFP4)
        return t.contiguous();
    const int ndim = t.dim();
    DG_HOST_ASSERT(ndim == 2 or ndim == 3);
    const int mn_packed = t.size(-2);
    const int k = t.size(-1);
    DG_HOST_ASSERT(mn_packed * 2 == logical_mn and k % 2 == 0);
    auto lo = t.bitwise_and(0x0F);
    auto hi = t.to(torch::kByte).bitwise_right_shift(4).to(torch::kInt8).bitwise_and(0x0F);
    auto shape_full = t.sizes().vec();
    shape_full[ndim - 2] = logical_mn;
    auto codes = torch::empty(shape_full, t.options());
    using S = torch::indexing::Slice;
    codes.index_put_({torch::indexing::Ellipsis, S(0, torch::indexing::None, 2), S()}, lo);
    codes.index_put_({torch::indexing::Ellipsis, S(1, torch::indexing::None, 2), S()}, hi);
    auto shape_view = shape_full;
    shape_view[ndim - 1] = k / 2;
    shape_view.push_back(2);
    auto codes2 = codes.view(shape_view);
    auto result = codes2.select(-1, 0).bitwise_and(0x0F)
                  .bitwise_or(codes2.select(-1, 1).bitwise_and(0x0F).to(torch::kByte)
                              .bitwise_left_shift(4).to(torch::kInt8));
    return result.contiguous();
}

static void fp8_fp4_gemm_nt_sm120(const std::pair<torch::Tensor, torch::Tensor>& a,
                                  const std::pair<torch::Tensor, torch::Tensor>& b,
                                  const torch::Tensor& d,
                                  const std::optional<torch::Tensor>& c,
                                  const std::optional<std::tuple<int, int, int>>& recipe,
                                  const std::optional<std::tuple<int, int>>& recipe_a,
                                  const std::optional<std::tuple<int, int>>& recipe_b,
                                  const std::string& compiled_dims,
                                  const bool& disable_ue8m0_cast,
                                  const cute::UMMA::Major& major_a,
                                  const cute::UMMA::Major& major_b,
                                  const int& m, const int& n, const int& k,
                                  const std::optional<float>& alpha) {
    const auto a_data = sm120_to_k_major(a.first, major_a, m);
    const auto b_data = sm120_to_k_major(b.first, major_b, n);
    constexpr auto k_major = cute::UMMA::Major::K;
    const bool is_mixed_fp4 = (a_data.scalar_type() != b_data.scalar_type()) and
                              (a_data.scalar_type() == kPackedFP4 or b_data.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(!is_mixed_fp4 or k % 128 == 0);
    constexpr int kSwapAbMMax = 16;
    const bool swap_ab = (m >= 1 and m <= kSwapAbMMax
        and d.stride(-1) == 1 and !is_mixed_fp4 and !c.has_value());

    auto effective_recipe = recipe;
    auto effective_recipe_a = recipe_a;
    auto effective_recipe_b = recipe_b;
    if (swap_ab) {
        if (recipe_a.has_value()) {
            effective_recipe_a = recipe_b;
            effective_recipe_b = recipe_a;
        } else {
            const auto [ga, gb, gk] = recipe.value_or(get_default_recipe(a.second.scalar_type(), b.second.scalar_type()));
            effective_recipe.reset();
            effective_recipe_a = std::make_tuple(gb, gk);
            effective_recipe_b = std::make_tuple(ga, gk);
        }
    }
    const auto& sf_a_raw = swap_ab ? b.second : a.second;
    const auto& sf_b_raw = swap_ab ? a.second : b.second;
    const int eff_m = swap_ab ? n : m;
    const int eff_n = swap_ab ? m : n;
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        sf_a_raw, sf_b_raw, eff_m, eff_n, k, effective_recipe,
        effective_recipe_a, effective_recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);
    sm120_fp8_fp4_gemm_1d1d(swap_ab ? b_data : a_data, sfa, swap_ab ? a_data : b_data, sfb,
                            c, d, eff_m, eff_n, k, gran_k_a, gran_k_b,
                            k_major, k_major, compiled_dims, std::nullopt, swap_ab, alpha);
}

} // namespace deep_gemm::gemm
