#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit/compiler.hpp"
#include "jit/device_runtime.hpp"
#include "utils/layout.hpp"

#include "jit_kernels/impls/smxx_layout.hpp"
#include "jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "jit_kernels/impls/sm100_fp8_gemm_1d2d.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_gemm_cpp
#endif

namespace deep_gemm {
torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                const int& mn, const int& k,
                                                const std::optional<int>& num_groups,
                                                const std::tuple<int, int, int>& recipe,
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

torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
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

void fp8_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
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
    const auto& sfa = transform_sf_into_required_layout(a.second, m, k, std::nullopt, recipe.value(),  true, disable_ue8m0_cast);
    const auto& sfb = transform_sf_into_required_layout(b.second, n, k, std::nullopt, recipe.value(), false, disable_ue8m0_cast);

    // Dispatch into different implements
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unknown kernel or scaling factor types");
    }
}

void fp8_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::string& compiled_dims,
                 const bool& disable_ue8m0_cast) {
    fp8_gemm_nt(a, {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

void fp8_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
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

void fp8_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                 const std::pair<torch::Tensor, torch::Tensor>& b,
                 const torch::Tensor& d,
                 const std::optional<torch::Tensor>& c,
                 const std::optional<std::tuple<int, int, int>>& recipe,
                 const std::string& compiled_dims,
                 const bool& disable_ue8m0_cast) {
    fp8_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)}, b,
                d, c, recipe, compiled_dims, disable_ue8m0_cast);
}

void m_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
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
    const auto& sfa = transform_sf_into_required_layout(a.second, m, k, std::nullopt, recipe.value(),  true, disable_ue8m0_cast);
    const auto& sfb = transform_sf_into_required_layout(b.second, n, k,   num_groups, recipe.value(), false, disable_ue8m0_cast);

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
        DG_HOST_UNREACHABLE("Unknown kernel or scaling factor types");
    }
}

void m_grouped_fp8_gemm_nn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const torch::Tensor& m_indices,
                                      const std::optional<std::tuple<int, int, int>>& recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    m_grouped_fp8_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)},
                                     d, m_indices, recipe, compiled_dims, disable_ue8m0_cast);
}

void fp8_m_grouped_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
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
    const auto& sfa = transform_sf_into_required_layout(a.second, m, k, num_groups, recipe.value(),  true, disable_ue8m0_cast);
    const auto& sfb = transform_sf_into_required_layout(b.second, n, k, num_groups, recipe.value(), false, disable_ue8m0_cast);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        sm90_fp8_m_grouped_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_m_grouped_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                             num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_fp8_m_grouped_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                             num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported kernel or scaling factor types");
    }
}

void k_grouped_fp8_gemm_tn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
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
    const auto& sfa = transform_k_grouped_sf_into_required_layout(a.second, ks, ks_tensor, recipe);
    const auto& sfb = transform_k_grouped_sf_into_required_layout(b.second, ks, ks_tensor, recipe);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        fp8_k_grouped_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks, ks_tensor,
                                cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

} // namespace deep_gemm

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace deep_gemm;

    m.doc() = "DeepGEMM C++ library";

    // Runtime
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });

    // JIT
    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_torch) {
        DG_HOST_ASSERT(get_env("DG_JIT_USE_NVRTC", 0) == 0 and "Currently only support NVCC");
        compiler = std::make_shared<NVCCCompiler>(library_root_path, cuda_home_path_by_torch);
        KernelRuntime::set_cuda_home(cuda_home_path_by_torch);
    });

    // Stable kernel APIs with automatic arch/layout dispatch
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
    m.def("fp8_m_grouped_gemm_nt_masked", &fp8_m_grouped_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false);
    m.def("k_grouped_fp8_gemm_tn_contiguous", &k_grouped_fp8_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("ks"),
          py::arg("ks_tensor"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn");
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout);

    // Raw kernels or functions
    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mk_alignment_for_contiguous_layout", &get_mk_alignment_for_contiguous_layout);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
}
