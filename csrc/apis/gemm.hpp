#pragma once

#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"

#include "layout.hpp"

namespace deep_gemm::gemm {

static void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d,
                        const std::optional<torch::Tensor>& c,
                        std::optional<std::tuple<int, int, int>> recipe,
                        const std::string& compiled_dims,
                        const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto& [m , k ] = get_shape<2>(a.first);
    const auto& [n , k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Check C as well
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
    }

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), std::nullopt, false, disable_ue8m0_cast);

    // Dispatch into different implements
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void fp8_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d,
                        const std::optional<torch::Tensor>& c,
                        const std::optional<std::tuple<int, int, int>>& recipe,
                        const std::string& compiled_dims,
                        const bool& disable_ue8m0_cast) {
    fp8_gemm_nt(a, {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

static void fp8_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d,
                        const std::optional<torch::Tensor>& c,
                        const std::optional<std::tuple<int, int, int>>& recipe,
                        const std::string& compiled_dims,
                        const bool& disable_ue8m0_cast) {
    fp8_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)},
                {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

static void fp8_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                        const std::pair<torch::Tensor, torch::Tensor>& b,
                        const torch::Tensor& d,
                        const std::optional<torch::Tensor>& c,
                        const std::optional<std::tuple<int, int, int>>& recipe,
                        const std::string& compiled_dims,
                        const bool& disable_ue8m0_cast) {
    fp8_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)}, b,
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

static void m_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& m_indices,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(m_indices.is_contiguous());

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a.first);
    const auto& [num_groups, n, k_] = get_shape<3>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    const auto& m__ = static_cast<int>(m_indices.numel());
    DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(),   num_groups, false, disable_ue8m0_cast);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, m_indices,
                                                num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, m_indices,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, m_indices,
                                                 num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void m_grouped_fp8_gemm_nn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& m_indices,
                                             const std::optional<std::tuple<int, int, int>>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    m_grouped_fp8_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)},
                                     d, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
}

static void m_grouped_fp8_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                         const std::pair<torch::Tensor, torch::Tensor>& b,
                                         const torch::Tensor& d,
                                         const torch::Tensor& masked_m,
                                         const int& expected_m,
                                         std::optional<std::tuple<int, int, int>> recipe,
                                         const std::string& compiled_dims,
                                         const bool& disable_ue8m0_cast) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto& [num_groups, m, k] = get_shape<3>(a.first);
    const auto& [num_groups_, n, k_] = get_shape<3>(b.first);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    const auto& num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Transform scaling factors
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), num_groups,  true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), num_groups, false, disable_ue8m0_cast);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_m_grouped_fp8_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                             num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_m_grouped_fp8_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                             num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void k_grouped_fp8_gemm_tn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::vector<int>& ks,
                                             const torch::Tensor& ks_tensor,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    if (c.has_value()) {
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().is_contiguous());
    }

    // Do nothing if empty
    if (std::accumulate(ks.begin(), ks.end(), 0) == 0)
        return;

    // Transform SF with padding
    const auto& [_, m] = get_shape<2>(a.first);
    const auto& [__, n] = get_shape<2>(b.first);
    const auto& sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks, ks_tensor, recipe);
    const auto& sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks, ks_tensor, recipe);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        fp8_k_grouped_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks, ks_tensor,
                                cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bf16_gemm_nt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto& [m , k ] = get_shape<2>(a);
    const auto& [n , k_] = get_shape<2>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Check C as well
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
    }

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Dispatch into different implements
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bf16_gemm_nn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& m_indices,
                                              const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(m_indices.is_contiguous());

    // Type and shape checks
    const auto& [m, k] = get_shape<2>(a);
    const auto& [num_groups, n, k_] = get_shape<3>(b);
    const auto& [m_, n_] = get_shape<2>(d);
    const auto& m__ = static_cast<int>(m_indices.numel());
    DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_m_grouped_bf16_gemm_contiguous(a, b, d, m_indices,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a, const torch::Tensor& b,
                                          const torch::Tensor& d, const torch::Tensor& masked_m,
                                          const int& expected_m, const std::string& compiled_dims) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto& major_a = get_major_type_ab(a);
    const auto& major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto& [num_groups, m, k] = get_shape<3>(a);
    const auto& [num_groups_, n, k_] = get_shape<3>(b);
    const auto& [num_groups__, m_, n_] = get_shape<3>(d);
    const auto& num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_m_grouped_gemm_masked(a, b, d, masked_m,
                                        num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void register_apis(pybind11::module_& m) {
    // FP8 GEMMs
     m.def("fp8_gemm_nt", &fp8_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_gemm_nn", &fp8_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_gemm_tn", &fp8_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_gemm_tt", &fp8_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_gemm_nt_contiguous", &m_grouped_fp8_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_gemm_nn_contiguous", &m_grouped_fp8_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("recipe") = std::nullopt, py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_gemm_nt_masked", &m_grouped_fp8_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false);
    m.def("k_grouped_fp8_gemm_tn_contiguous", &k_grouped_fp8_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("ks"),
          py::arg("ks_tensor"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn");

    // BF16 GEMMs
    m.def("bf16_gemm_nt", &bf16_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("bf16_gemm_nn", &bf16_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("bf16_gemm_tn", &bf16_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn");
    m.def("bf16_gemm_tt", &bf16_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn");
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("m_indices"),
          py::arg("compiled_dims") = "nk");
    m.def("m_grouped_bf16_gemm_nt_masked", &m_grouped_bf16_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("compiled_dims") = "nk");
}

} // namespace deep_gemm::gemm
