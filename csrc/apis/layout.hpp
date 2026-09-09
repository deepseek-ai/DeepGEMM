#pragma once

#include "../jit_kernels/heuristics/runtime.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/smxx_layout.hpp"
#endif
#include <torch/library.h>
#include "../torch_library_utils.hpp"

namespace deep_gemm::layout {

#if DG_TENSORMAP_COMPATIBLE
static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::variant<std::tuple<int, int, int>,
                                                                          std::tuple<int, int>>& recipe,
                                                       const std::optional<int>& num_groups,
                                                       const std::optional<bool>& is_sfa,
                                                       const bool& disable_ue8m0_cast,
                                                       const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    const auto arch_major = device_runtime->get_arch_major();

    // Get granularity MN/K from recipe
    int gran_mn, gran_k;
    if (auto p = std::get_if<std::tuple<int, int, int>>(&recipe)) {
        DG_HOST_ASSERT(is_sfa.has_value());
        gran_mn = is_sfa.value() ? std::get<0>(*p) : std::get<1>(*p);
        gran_k = std::get<2>(*p);
    } else if (auto p = std::get_if<std::tuple<int, int>>(&recipe)) {
        DG_HOST_ASSERT(not is_sfa.has_value());
        std::tie(gran_mn, gran_k) = *p;
    } else {
        DG_HOST_UNREACHABLE("Invalid recipe");
    }

    // Pre-transform checks
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 128, 128) on SM90: no need to transform, check SFB requirements
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, x, gran_k) on SM100/SM120: transform to (INT, 1, gran_k), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12)) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto broadcasted = gran_mn == 1 ? sf :
                                 sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(gran_mn));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted, psum_layout);
    }

    // (INT, 1, gran_k) on SM100/SM120: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static std::tuple<torch::Tensor, torch::Tensor, int, int> transform_sf_pair_into_required_layout(
        const torch::Tensor& sfa, const torch::Tensor& sfb,
        const int& m, const int& n, const int& k,
        std::optional<std::tuple<int, int, int>>& recipe,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b,
        const std::optional<int>& num_groups_a,
        const std::optional<int>& num_groups_b,
        const bool& disable_ue8m0_cast = false,
        // PSUM layout only applies to SFA (B is weights with no gaps)
        const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    // Use default recipe, if none is specified
    if (not recipe_a.has_value() and not recipe.has_value())
        recipe = get_default_recipe(sfa.scalar_type(), sfb.scalar_type());

    // Must be either 'recipe' or the 'recipe_a' + 'recipe_b' pair.
    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    DG_HOST_ASSERT(recipe_a.has_value() != recipe.has_value());

    // Transform SFA and SFB layout (PSUM only for SFA)
    const auto transformed_sfa = recipe.has_value() ? transform_sf_into_required_layout(sfa, m, k, recipe.value(), num_groups_a, true, disable_ue8m0_cast, psum_layout)
                                                    : transform_sf_into_required_layout(sfa, m, k, recipe_a.value(), num_groups_a, std::nullopt, disable_ue8m0_cast, psum_layout);
    const auto transformed_sfb = recipe.has_value() ? transform_sf_into_required_layout(sfb, n, k, recipe.value(), num_groups_b, false, disable_ue8m0_cast)
                                                    : transform_sf_into_required_layout(sfb, n, k, recipe_b.value(), num_groups_b, std::nullopt, disable_ue8m0_cast);
    const int gran_k_a = recipe_a.has_value() ? std::get<1>(recipe_a.value()) : std::get<2>(recipe.value());
    const int gran_k_b = recipe_b.has_value() ? std::get<1>(recipe_b.value()) : std::get<2>(recipe.value());
    return std::make_tuple(transformed_sfa, transformed_sfb, gran_k_a, gran_k_b);
}

static torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
                                                                 const std::optional<std::vector<int>>& ks_cpu,
                                                                 const torch::Tensor& grouped_layout,
                                                                 const std::tuple<int, int, int>& recipe,
                                                                 const int& k_alignment,
                                                                 const bool& use_psum_layout) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);

    const int gran_k = std::get<2>(recipe);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    DG_HOST_ASSERT(k_alignment % 32 == 0);

    const auto arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (sf.scalar_type() == torch::kFloat and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);

    // FP32 on SM100/SM120
    if (sf.scalar_type() == torch::kFloat and (arch_major == 10 or arch_major == 12)) {
        auto sf_input = sf;
        // SM120 also accepts K-major operands. Their SF tensor is [mn, sf_k],
        // while the common packer consumes [sf_k, mn].
        if (arch_major == 12) {
            const auto sf_contiguous = sf.is_contiguous() ? sf : sf.contiguous();
            if (ks_cpu.has_value() and not ks_cpu.value().empty()) {
                int expected_sf_k = 0;
                for (const auto k: ks_cpu.value())
                    expected_sf_k += ceil_div(k, gran_k);
                sf_input = sf_contiguous.size(0) == expected_sf_k ? sf_contiguous : sf_contiguous.t().contiguous();
            } else {
                sf_input = sf_contiguous;
            }
        }
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            sf_input, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout);
    }

    // INT (already packed UE8M0) on SM100/SM120
    if (sf.scalar_type() == torch::kInt and (arch_major == 10 or arch_major == 12))
        return check_k_grouped_packed_ue8m0_tensor(sf, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout);

    DG_HOST_UNREACHABLE("Unknown cases");
}

#endif

}  // namespace deep_gemm::layout

namespace deep_gemm::torch_registration {

using namespace deep_gemm::torch_utils;

#if DG_TENSORMAP_COMPATIBLE
static torch::Tensor transform_sf_into_required_layout(
    const torch::Tensor& sf, const int64_t& mn, const int64_t& k,
    const std::vector<int64_t>& recipe,
    const c10::optional<int64_t>& num_groups,
    const c10::optional<bool>& is_sfa,
    const bool& disable_ue8m0_cast,
    const c10::optional<torch::Tensor>& psum_layout) {
    return layout::transform_sf_into_required_layout(
        sf, static_cast<int>(mn), static_cast<int>(k),
        list_to_recipe_variant(recipe),
        num_groups.has_value() ? std::make_optional(static_cast<int>(num_groups.value())) : std::nullopt,
        is_sfa,
        disable_ue8m0_cast,
        psum_layout);
}

static torch::Tensor get_mn_major_tma_aligned_tensor(const torch::Tensor& sf) {
    return ::deep_gemm::get_mn_major_tma_aligned_tensor(sf);
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor(
    const torch::Tensor& sf, const c10::optional<torch::Tensor>& psum_layout) {
    return ::deep_gemm::get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, psum_layout);
}

static torch::Tensor get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    const torch::Tensor& sf, const torch::Tensor& grouped_layout,
    const c10::optional<std::vector<int64_t>>& ks_cpu,
    const int64_t& gran_k, const int64_t& k_alignment,
    const bool& use_psum_layout) {
    return ::deep_gemm::get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        sf, grouped_layout,
        list_to_optional_vector_int(ks_cpu),
        static_cast<int>(gran_k), static_cast<int>(k_alignment),
        use_psum_layout);
}
#endif

static int64_t get_tma_aligned_size(const int64_t& x, const int64_t& element_size) {
    return ::deep_gemm::get_tma_aligned_size(static_cast<int>(x), static_cast<int>(element_size));
}

static void set_mk_alignment_for_contiguous_layout(const int64_t& new_value) {
    heuristics_runtime->set_mk_alignment_for_contiguous_layout(static_cast<int>(new_value));
}

static int64_t get_mk_alignment_for_contiguous_layout() {
    return heuristics_runtime->get_mk_alignment_for_contiguous_layout();
}

static int64_t get_theoretical_mk_alignment_for_contiguous_layout(
    const c10::optional<int64_t>& expected_m,
    const c10::optional<int64_t>& num_groups) {
    return HeuristicsRuntime::get_theoretical_mk_alignment_for_contiguous_layout(
        expected_m.has_value() ? std::make_optional(static_cast<int>(expected_m.value())) : std::nullopt,
        num_groups.has_value() ? std::make_optional(static_cast<int>(num_groups.value())) : std::nullopt);
}

}  // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def(
        "transform_sf_into_required_layout(Tensor sf, int mn, int k, int[] recipe, int? num_groups=None, bool? is_sfa=None, bool disable_ue8m0_cast=False, Tensor? psum_layout=None) -> Tensor");
    m.def("get_tma_aligned_size(int x, int element_size) -> int", TORCH_FN(deep_gemm::torch_registration::get_tma_aligned_size));
    m.def("get_mn_major_tma_aligned_tensor(Tensor sf) -> Tensor");
    m.def(
        "get_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor sf, Tensor? psum_layout=None) -> Tensor");
    m.def(
        "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor sf, Tensor grouped_layout, int[]? ks_cpu, int gran_k, int k_alignment, bool use_psum_layout=False) -> Tensor");
#endif

    m.def("set_mk_alignment_for_contiguous_layout(int new_value) -> ()",
          TORCH_FN(deep_gemm::torch_registration::set_mk_alignment_for_contiguous_layout));
    m.def("get_mk_alignment_for_contiguous_layout() -> int",
          TORCH_FN(deep_gemm::torch_registration::get_mk_alignment_for_contiguous_layout));
    m.def("get_theoretical_mk_alignment_for_contiguous_layout(int? expected_m=None, int? num_groups=None) -> int",
          TORCH_FN(deep_gemm::torch_registration::get_theoretical_mk_alignment_for_contiguous_layout));
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::torch_registration;

#if DG_TENSORMAP_COMPATIBLE
    m.impl("transform_sf_into_required_layout",
           TORCH_FN(transform_sf_into_required_layout));
    m.impl("get_mn_major_tma_aligned_tensor",
           TORCH_FN(get_mn_major_tma_aligned_tensor));
    m.impl("get_mn_major_tma_aligned_packed_ue8m0_tensor",
           TORCH_FN(get_mn_major_tma_aligned_packed_ue8m0_tensor));
    m.impl("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
           TORCH_FN(get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor));
#endif
}
