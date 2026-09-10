#pragma once

#include <format>
#include <variant>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"
#include "gemm.hpp"

#include "../jit_kernels/impls/sm90_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm100_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
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

        const auto workspace = torch::empty_like(d, d.options().dtype(torch::kFloat32));
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

    const auto [s , m, k ] = get_shape<3>(a);
    const auto [s_, n, k_] = get_shape<3>(b);
    DG_HOST_ASSERT(s == s_ and k == k_);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        sm90_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else if (arch_major == 10) {
        sm100_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhr_hdr_bhd(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D) {
    const auto [b , h  , r ] = get_shape<3>(A);
    const auto [h_, d  , r_] = get_shape<3>(B);
    const auto [b_, h__, d_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kBFloat16 and D.stride(2) == 1);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()) {
        cublaslt_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhd_hdr_bhr(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D) {
    const auto [b , h  , d ] = get_shape<3>(A);
    const auto [h_, d_ , r ] = get_shape<3>(B);
    const auto [b_, h__, r_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kBFloat16 and D.stride(2) == 1);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()) {
        cublaslt_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhd_bhr_hdr(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D,
                        const std::optional<torch::Tensor>& C) {
    const auto [b , h  , d ] = get_shape<3>(A);
    const auto [b_, h_ , r ] = get_shape<3>(B);
    const auto [h__, d_, r_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and h == h__ and d == d_ and r == r_);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kFloat and D.stride(2) == 1);
    if (C.has_value()) {
        DG_HOST_ASSERT(C->scalar_type() == D.scalar_type() and C->sizes() == D.sizes() and C->stride(2) == 1);
    }

    // Early return for trivial cases
    if (h == 0 or gemm::early_return(d, r, b, D, C))
        return;

    // TODO: investigate cuBLAS determinism.
    cublaslt_bhd_bhr_hdr(A, B, D, b, h, r, d, C.has_value());
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
        DG_HOST_ASSERT(d.scalar_type() == c->scalar_type());
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
    } else if (expr == "bhd,bhr->hdr") {
        bhd_bhr_hdr(a, b, d, c);
    } else {
        DG_HOST_UNREACHABLE(std::format("Unsupported einsum expression: {}", expr));
    }
}

// The D output is either a plain BF16/FP32 tensor, or an FP8 `(d, sfd)` pair
// quantized with dynamic per-32 packed UE8M0 SFs
static void fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                    const torch::Tensor& b, const torch::Tensor& sfb,
                    const std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>& d,
                    const std::optional<torch::Tensor>& c,
                    std::optional<std::tuple<int, int, int>> recipe,
                    const std::string& compiled_dims) {
    const auto d_fp8 = std::get_if<std::pair<torch::Tensor, torch::Tensor>>(&d);
    const auto d_tensor = d_fp8 != nullptr ? d_fp8->first : std::get<torch::Tensor>(d);
    const auto sfd = d_fp8 != nullptr ? std::make_optional(d_fp8->second) : std::nullopt;

    // Shape must be `[B, M, K] @ [B, N, K].T`
    const auto major_a = a.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    const auto major_b = b.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    DG_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    DG_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    DG_HOST_ASSERT(d_tensor.stride(-1) == 1);

    // Type and shape checks
    const auto [batch_size  , m , k ] = get_shape<3>(a);
    const auto [batch_size_ , n , k_] = get_shape<3>(b);
    const auto [batch_size__, m_, n_] = get_shape<3>(d_tensor);
    DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    if (sfd.has_value()) {
        // The SF layout matches a per-token cast of D viewed as a 2D `(m, batch_size * n)`
        // matrix, where each row concatenates the `n` columns of all batches. This requires:
        //  - `d.stride(0) == n`: the batches are contiguous along the columns of the view
        //  - `n % 128 == 0`: each batch is quantized independently, so neither an SF group
        //    nor a packed 4-SF word may cross a batch boundary
        DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and not c.has_value());
        DG_HOST_ASSERT(d_tensor.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(d_tensor.stride(0) == n and n % 128 == 0);
        check_sf_layout(sfd.value(), m, batch_size * n, 1, 32, std::nullopt, true, false, torch::kInt);
    } else {
        DG_HOST_ASSERT(d_tensor.scalar_type() == torch::kBFloat16 or d_tensor.scalar_type() == torch::kFloat);
    }

    // Early return for trivial cases
    if (batch_size == 0 or gemm::early_return(m, n, k, d_tensor, c, sfd))
        return;

    // Transform scaling factors
    const auto [transformed_sfa, transformed_sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        sfa, sfb, m, n, k, recipe, std::nullopt, std::nullopt, batch_size, batch_size, false);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 10) {
        sm100_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d_tensor, batch_size, m, n, k, gran_k_a, gran_k_b, major_a, major_b, compiled_dims, sfd);
    } else {
        const auto major_sfb = get_major_type_ab(sfb);
        DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);
        sm90_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d_tensor, batch_size, m, n, k, major_a, major_b, major_sfb, compiled_dims);
    }
}

static void fp8_einsum(const std::string& expr,
                       const std::pair<torch::Tensor, torch::Tensor>& a,
                       const std::pair<torch::Tensor, torch::Tensor>& b,
                       const std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>& d,
                       const std::optional<torch::Tensor>& c,
                       const std::tuple<int, int, int>& recipe) {
    // Some hardcoded Einstein sum kernels
    // NOTES: only `bhr,hdr->bhd` accepts an FP8 `(d, sfd)` output pair; the other
    //        expressions take a plain BF16/FP32 D
    const auto arch_major = jit->device.get_arch_major();
    if (expr == "bhr,hdr->bhd") {
        // Permute dims to satisfy the order of (batch_size, m, n, k)
        // (batch_size, m, n, k): (h, b, d, r)
        // NOTES: the FP8 output SF columns are flattened over D's last two dims (matching a
        //        per-token cast of `d.flatten(1, 2)`), so the SF layout is invariant under
        //        the `(b, h)` permute of the data dims
        const auto perm_a = a.first.permute({1, 0, 2});
        const auto perm_sfa = a.second.permute({1, 0, 2});
        auto perm_d = d;
        if (const auto d_fp8 = std::get_if<std::pair<torch::Tensor, torch::Tensor>>(&perm_d))
            d_fp8->first = d_fp8->first.permute({1, 0, 2});
        else
            std::get<torch::Tensor>(perm_d) = std::get<torch::Tensor>(perm_d).permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, b.first, b.second, perm_d, perm_c, recipe, "nk");
    } else if (expr == "bhd,hdr->bhr" and arch_major == 10) {
        // (batch_size, m, n, k): (h, b, r, d)
        DG_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        const auto perm_a = a.first.permute({1, 0, 2});
        const auto perm_sfa = a.second.permute({1, 0, 2});
        const auto perm_b = b.first.permute({0, 2, 1});
        const auto perm_sfb = b.second.permute({0, 2, 1});
        const auto perm_d = std::get<torch::Tensor>(d).permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, perm_d, perm_c, recipe, "nk");
    } else if (expr == "bhd,bhr->hdr" and arch_major == 10) {
        // (batch_size, m, n, k): (h, d, r, b)
        DG_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        const auto perm_a = a.first.permute({1, 2, 0});
        const auto perm_sfa = a.second.permute({1, 2, 0});
        const auto perm_b = b.first.permute({1, 2, 0});
        const auto perm_sfb = b.second.permute({1, 2, 0});
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, d, c, recipe, "mn");
    } else {
        DG_HOST_UNREACHABLE(std::format("Unsupported einsum expression: {}", expr));
    }
}
static void register_apis(pybind11::module_& m) {
    m.def("einsum", &einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt);
    m.def("fp8_einsum", &fp8_einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 128, 128));
}

} // namespace deep_gemm::einsum
