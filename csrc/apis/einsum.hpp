#pragma once

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/layout.hpp"

#include "../jit_kernels/impls/sm90_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm100_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/smxx_cublaslt.hpp"

namespace deep_gemm::einsum {

static void bmk_bnk_mn(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                       const std::optional<torch::Tensor>& c) {
    // Currently FP32 only support the accumulated expression
    if (d.scalar_type() == torch::kFloat) {
        DG_HOST_ASSERT(c->data_ptr() == d.data_ptr() and c->sizes() == d.sizes() and c->strides() == d.strides());
    } else {
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(not c.has_value());

        const auto& workspace = torch::empty_like(d, d.options().dtype(torch::kFloat32));
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(workspace.data_ptr(), 0, workspace.nbytes(),
                              c10::cuda::getCurrentCUDAStream()));
        bmk_bnk_mn(a, b, workspace, workspace);

        // This line has an implicit FP32-to-BF16 casting
        d.copy_(workspace);
        return;
    }

    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());

    const auto& [s , m, k ] = get_shape<3>(a);
    const auto& [s_, n, k_] = get_shape<3>(b);
    DG_HOST_ASSERT(s == s_ and k == k_);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else if (arch_major == 10) {
        sm100_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhr_hdr_bhd(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D) {
    const auto& [b , h  , r ] = get_shape<3>(A);
    const auto& [h_, d  , r_] = get_shape<3>(B);
    const auto& [b_, h__, d_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kBFloat16 and D.stride(2) == 1);

    cublaslt_bhr_hdr_bhd(A, B, D, b, h, r, d);
}

static void bhd_hdr_bhr(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D) {
    const auto& [b , h  , d ] = get_shape<3>(A);
    const auto& [h_, d_ , r ] = get_shape<3>(B);
    const auto& [b_, h__, r_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kBFloat16 and D.stride(2) == 1);

    cublaslt_bhd_hdr_bhr(A, B, D, b, h, r, d);
}

static void einsum(const std::string& expr,
                   const torch::Tensor& a,
                   const torch::Tensor& b,
                   const torch::Tensor& d,
                   const std::optional<torch::Tensor>& c) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        DG_HOST_ASSERT(c->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(d.scalar_type() == torch::kFloat);
    }

    // Some hardcoded Einstein sum kernels
    // TODO: support any expression
    // TODO: canonicalize expression
    if (expr == "bmk,bnk->mn") {
        bmk_bnk_mn(a, b, d, c);
    } else if (expr == "bhr,hdr->bhd") {
        DG_HOST_ASSERT(not c.has_value());
        bhr_hdr_bhd(a, b, d);
    } else if (expr == "bhd,hdr->bhr") {
        DG_HOST_ASSERT(not c.has_value());
        bhd_hdr_bhr(a, b, d);
    } else {
        DG_HOST_UNREACHABLE(fmt::format("Unsupported einsum expression: {}", expr));
    }
}

static void register_apis(pybind11::module_& m) {
    m.def("einsum", &einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt);
}

} // namespace deep_gemm::einsum
