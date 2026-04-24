#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp"
#endif

namespace deep_gemm::hyperconnection {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void sm12x_tf32_hc_prenorm_gemm_reference(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& sqr_sum,
                                                 const int& num_splits,
                                                 const bool& has_split_outputs) {
    auto a_float = a.to(torch::kFloat);
    auto d_value = at::matmul(a_float, b.transpose(0, 1));
    auto sqr_sum_value = (a_float * a_float).sum(-1);

    if (has_split_outputs) {
        d.zero_();
        sqr_sum.zero_();
        d.select(0, 0).copy_(d_value);
        sqr_sum.select(0, 0).copy_(sqr_sum_value);
        return;
    }

    DG_HOST_ASSERT(num_splits == 1);
    d.copy_(d_value);
    sqr_sum.copy_(sqr_sum_value);
}

static void tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& sqr_sum,
                                 const std::optional<int>& num_splits) {
    // A and B must be K-major, D must be N-major
    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == cute::UMMA::Major::K);
    check_major_type_cd(d);

    // S must be contiguous
    DG_HOST_ASSERT(sqr_sum.is_contiguous());

    // Type and shape checks
    const auto [m, k ] = get_shape<2>(a);
    const auto [n, k_] = get_shape<2>(b);
    if (num_splits.has_value()) {
        const auto [num_splits_, m_, n_] = get_shape<3>(d);
        const auto [num_splits__, m__] = get_shape<2>(sqr_sum);
        DG_HOST_ASSERT(num_splits.value() == num_splits_ and num_splits.value() == num_splits__ and num_splits.value() >= 1);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    } else {
        const auto [m_, n_] = get_shape<2>(d);
        const auto [m__] = get_shape<1>(sqr_sum);
        DG_HOST_ASSERT(m == m_ and m == m__ and n == n_ and k == k_);
    }
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sqr_sum.scalar_type() == torch::kFloat);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 10) {
        sm100_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 12) {
        sm12x_tf32_hc_prenorm_gemm_reference(a, b, d, sqr_sum,
                                             num_splits.has_value() ? num_splits.value() : 1,
                                             num_splits.has_value());
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

#endif

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("tf32_hc_prenorm_gemm", &tf32_hc_prenorm_gemm,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sqr_sum"),
          py::arg("num_splits") = std::nullopt);
#endif
}

} // namespace deep_gemm::hyperconnection
