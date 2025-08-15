#pragma once

#include "../utils/layout.hpp"
#include "../jit_kernels/impls/smxx_layout.hpp"

namespace deep_gemm::layout {

static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::tuple<int, int, int>& recipe,
                                                       const std::optional<int>& num_groups,
                                                       const bool& is_sfa,
                                                       const bool& disable_ue8m0_cast) {
    const auto& gran_mn = is_sfa ? std::get<0>(recipe) : std::get<1>(recipe);
    const auto& gran_k = std::get<2>(recipe);
    const auto& arch_major = device_runtime->get_arch_major();

    // Pre-transform checks
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 1, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(sf);
    }

    // (FP32, 128, 128) on SM90: no need to transform, check shape and contiguous
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, 128, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto& broadcasted = sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(128));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
    }

    // (INT, 1, 128) on SM100: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and gran_k == 128 and arch_major == 10)
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
                                                                 const std::vector<int>& ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
    const auto& arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (sf.scalar_type() == torch::kFloat and arch_major == 9)
        DG_HOST_UNREACHABLE("Unimplemented");

    // FP32 on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks);

    // INT on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        DG_HOST_UNREACHABLE("Unimplemented");

    DG_HOST_UNREACHABLE("Unknown cases");
}

static void register_apis(pybind11::module_& m) {
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
      py::arg("sf"), py::arg("mn"), py::arg("k"), py::arg("recipe"),
      py::arg("num_groups") = std::nullopt, py::arg("is_sfa") = false,
      py::arg("disable_ue8m0_cast") = false);

    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mk_alignment_for_contiguous_layout", &get_mk_alignment_for_contiguous_layout);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
}

} // namespace deep_gemm::layout
