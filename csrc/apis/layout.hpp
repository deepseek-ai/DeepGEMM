#pragma once

#include "../jit_kernels/heuristics/runtime.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/smxx_layout.hpp"
#endif

namespace deep_gemm::layout {

#if DG_TENSORMAP_COMPATIBLE
static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::variant<std::tuple<int, int, int>,
                                                                          std::tuple<int, int>>& recipe,
                                                       const std::optional<int>& num_groups,
                                                       const std::optional<bool>& is_sfa,
                                                       const bool& disable_ue8m0_cast) {
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

    // (BF16, 1, 32/128) on SM90 SFB fast path: path-A (k128) 与 path-B fast-path (k32)
    // 都支持 bf16 sfb，体积砍半。tensor 已由调用方按 MN-major + tma_aligned_mn=align(N,8)
    // 构造，这里跳过 fp32 only 的 transpose，直接复用 align 路径。
    if (sf.scalar_type() == torch::kBFloat16 and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);
    // (UInt8/E8M0, 1, 32) on SM90 SFB fast path：path-B 专用，每元素 1B 即 fp32 的
    // 8 位指数。tensor 已由调用方按 MN-major + tma_aligned_mn=align(N,16) 构造。
    if (sf.scalar_type() == torch::kUInt8 and gran_mn == 1 and gran_k == 32 and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 1, 32/128) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and
        (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (INT packed UE8M0, 1, 32/128) on SM90: transform to TMA-aligned and MN-major.
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 128, 128) on SM90: no need to transform, check SFB requirements
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, x, gran_k) on SM100: transform to (INT, 1, gran_k), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto broadcasted = gran_mn == 1 ? sf :
                                 sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(gran_mn));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
    }

    // (INT, 1, gran_k) on SM100: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 10)
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
        const bool& disable_ue8m0_cast = false) {
    // Use default recipe, if none is specified
    if (not recipe_a.has_value() and not recipe.has_value())
        recipe = get_default_recipe(sfa.scalar_type(), sfb.scalar_type());

    // Must be either 'recipe' or the 'recipe_a' + 'recipe_b' pair.
    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    DG_HOST_ASSERT(recipe_a.has_value() != recipe.has_value());

    // Transform SFA and SFB layout
    const auto transformed_sfa = recipe.has_value() ? transform_sf_into_required_layout(sfa, m, k, recipe.value(), num_groups_a, true, disable_ue8m0_cast)
                                                    : transform_sf_into_required_layout(sfa, m, k, recipe_a.value(), num_groups_a, std::nullopt, disable_ue8m0_cast);
    const auto transformed_sfb = recipe.has_value() ? transform_sf_into_required_layout(sfb, n, k, recipe.value(), num_groups_b, false, disable_ue8m0_cast)
                                                    : transform_sf_into_required_layout(sfb, n, k, recipe_b.value(), num_groups_b, std::nullopt, disable_ue8m0_cast);
    const int gran_k_a = recipe_a.has_value() ? std::get<1>(recipe_a.value()) : std::get<2>(recipe.value());
    const int gran_k_b = recipe_b.has_value() ? std::get<1>(recipe_b.value()) : std::get<2>(recipe.value());
    return std::make_tuple(transformed_sfa, transformed_sfb, gran_k_a, gran_k_b);
}

static torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
                                                                 const std::vector<int>& ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);

    const int gran_k = std::get<2>(recipe);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);

    const auto arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (sf.scalar_type() == torch::kFloat and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);

    // FP32 on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks, gran_k);

    // INT on SM100
    if (sf.scalar_type() == torch::kInt and arch_major == 10)
        DG_HOST_UNREACHABLE("Unimplemented");

    DG_HOST_UNREACHABLE("Unknown cases");
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
      py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"),
      py::arg("num_groups") = std::nullopt,
      py::arg("is_sfa") = std::nullopt,
      py::arg("disable_ue8m0_cast") = false);

    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
#endif

    m.def("set_mk_alignment_for_contiguous_layout", [&](const int& new_value) {
        heuristics_runtime->set_mk_alignment_for_contiguous_layout(new_value);
    });
    m.def("get_mk_alignment_for_contiguous_layout", [&]() {
        return heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    });
    m.def("get_theoretical_mk_alignment_for_contiguous_layout", [&](const std::optional<int>& expected_m) {
        return heuristics_runtime->get_theoretical_mk_alignment_for_contiguous_layout(expected_m);
    }, py::arg("expected_m") = std::nullopt);
}

} // namespace deep_gemm::layout
