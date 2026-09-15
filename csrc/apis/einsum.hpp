#pragma once

#include <format>
#include <limits>
#include <variant>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"
#include "gemm.hpp"

#include "../jit_kernels/impls/sm90_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm100_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm120_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/smxx_cublaslt.hpp"

namespace deep_gemm::einsum {

static void bmk_bnk_mn(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                       const std::optional<torch::Tensor>& c) {
    DG_HOST_ASSERT(a.is_cuda() and b.device() == a.device() and d.device() == a.device());
    const c10::cuda::CUDAGuard device_guard(a.device());
    DG_HOST_ASSERT(jit->device.get_arch_major() == at::cuda::getDeviceProperties(a.get_device())->major);
    // Currently FP32 only support the accumulated expression
    if (d.scalar_type() == torch::kFloat) {
        DG_HOST_ASSERT(c.has_value() and c->device() == d.device() and c->scalar_type() == d.scalar_type()
                       and c->data_ptr() == d.data_ptr() and c->sizes() == d.sizes() and c->strides() == d.strides());
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
    } else if (arch_major == 12) {
        for (const auto& t: {a, b, d})
            for (const auto size: t.sizes())
                DG_HOST_ASSERT(size <= std::numeric_limits<int>::max() - 127);
        DG_HOST_ASSERT(k <= std::numeric_limits<int>::max() / 2);
        DG_HOST_ASSERT(d.dim() == 2 and d.size(0) == m and d.size(1) == n);
        if (s == 0 or m == 0 or n == 0 or k == 0)
            return;
        DG_HOST_ASSERT(static_cast<int64_t>(s) * k <= std::numeric_limits<int>::max());
        DG_HOST_ASSERT(static_cast<int64_t>(m) * n <= std::numeric_limits<int>::max());
        const auto native_a = reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 ? a : a.clone();
        const auto native_b = reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0 ? b : b.clone();
        const bool direct_d = reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 == 0;
        const auto native_d = direct_d ? d : d.clone();
        sm120_bmn_bnk_mn_gemm(native_a, native_b, native_d, s, m, n, k);
        if (not direct_d)
            d.copy_(native_d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void sm120_bf16_einsum(const torch::Tensor& A, const torch::Tensor& B,
                              const torch::Tensor& D, const bool b_k_major) {
    constexpr int64_t max_dim = std::numeric_limits<int>::max() - 7;
    constexpr int64_t max_stride = std::numeric_limits<int>::max() / 2;
    for (const auto& t: {A, B, D}) {
        DG_HOST_ASSERT(t.dim() == 3 and t.scalar_type() == torch::kBFloat16 and t.stride(2) == 1);
        for (const auto size: t.sizes())
            DG_HOST_ASSERT(size >= 0 and size <= max_dim);
    }
    const int m = static_cast<int>(A.size(0));
    const int h = static_cast<int>(A.size(1));
    const int k = static_cast<int>(A.size(2));
    const int n = static_cast<int>(D.size(2));
    if (m == 0 or h == 0 or n == 0)
        return;
    if (k == 0) {
        D.zero_();
        return;
    }

    const int native_n = (n + 7) / 8 * 8;
    const auto tma_aligned = [&](const torch::Tensor& t) {
        return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0
            and t.stride(0) > 0 and t.stride(0) <= max_stride and t.stride(0) % 8 == 0
            and t.stride(1) > 0 and t.stride(1) <= max_stride and t.stride(1) % 8 == 0;
    };
    const auto aligned_backing = [&](const torch::Tensor& t, const int64_t rows,
                                     const int64_t cols, const bool zero) {
        const int64_t stride = (cols + 7) / 8 * 8;
        DG_HOST_ASSERT(stride <= max_stride and rows <= max_stride / stride);
        auto backing = torch::empty({t.size(0), rows, stride}, t.options());
        if (zero)
            backing.zero_();
        return backing.narrow(2, 0, cols);
    };

    auto native_a = A;
    if (not tma_aligned(A)) {
        native_a = aligned_backing(A, h, k, false);
        native_a.copy_(A);
    }
    auto native_b = B;
    if (native_n != n or not tma_aligned(B)) {
        native_b = aligned_backing(B, b_k_major ? native_n : k,
                                    b_k_major ? k : native_n, native_n != n);
        native_b.narrow(b_k_major ? 1 : 2, 0, n).copy_(B);
    }
    const bool direct_output = native_n == n and tma_aligned(D)
        and D.stride(0) == static_cast<int64_t>(h) * n and D.stride(1) == n;
    auto native_d = D;
    if (not direct_output) {
        DG_HOST_ASSERT(static_cast<int64_t>(h) * native_n <= max_stride);
        native_d = torch::empty({m, h, native_n}, D.options());
    }
    if (b_k_major)
        sm120_bf16_bhr_hdr_bhd(native_a, native_b, native_d, m, h, k, native_n);
    else
        sm120_bf16_bhd_hdr_bhr(native_a, native_b, native_d, m, h, native_n, k);
    if (not direct_output)
        D.copy_(native_d.narrow(2, 0, n));
}

static void bhr_hdr_bhd(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D) {
    const auto [b , h  , r ] = get_shape<3>(A);
    const auto [h_, d  , r_] = get_shape<3>(B);
    const auto [b_, h__, d_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kBFloat16 and D.stride(2) == 1);
    DG_HOST_ASSERT(B.device() == A.device() and D.device() == A.device());
    const c10::cuda::CUDAGuard device_guard(A.device());
    const auto& device_prop = *at::cuda::getDeviceProperties(A.get_device());
    const auto& cached_prop = jit->device.get_prop();
    DG_HOST_ASSERT(cached_prop.major == device_prop.major and cached_prop.minor == device_prop.minor
                   and cached_prop.multiProcessorCount == device_prop.multiProcessorCount
                   and cached_prop.sharedMemPerBlockOptin == device_prop.sharedMemPerBlockOptin
                   and cached_prop.l2CacheSize == device_prop.l2CacheSize);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()) {
        cublaslt_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhr_hdr_bhd(A, B, D, b, h, r, d);
    } else if (arch_major == 12) {
        sm120_bf16_einsum(A, B, D, true);
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
    DG_HOST_ASSERT(B.device() == A.device() and D.device() == A.device());
    const c10::cuda::CUDAGuard device_guard(A.device());
    const auto& device_prop = *at::cuda::getDeviceProperties(A.get_device());
    const auto& cached_prop = jit->device.get_prop();
    DG_HOST_ASSERT(cached_prop.major == device_prop.major and cached_prop.minor == device_prop.minor
                   and cached_prop.multiProcessorCount == device_prop.multiProcessorCount
                   and cached_prop.sharedMemPerBlockOptin == device_prop.sharedMemPerBlockOptin
                   and cached_prop.l2CacheSize == device_prop.l2CacheSize);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()) {
        cublaslt_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 9) {
        sm90_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhd_hdr_bhr(A, B, D, b, h, r, d);
    } else if (arch_major == 12) {
        sm120_bf16_einsum(A, B, D, false);
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
    DG_HOST_ASSERT(a.is_cuda());
    for (const auto& t: {sfa, b, sfb, d_tensor})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == a.device());
    if (c.has_value())
        DG_HOST_ASSERT(c->is_cuda() and c->device() == a.device());
    const c10::cuda::CUDAGuard device_guard(a.device());
    const auto arch_major = jit->device.get_arch_major();
    DG_HOST_ASSERT(arch_major == at::cuda::getDeviceProperties(a.get_device())->major);
    DG_HOST_ASSERT(arch_major != 12 or not sfd.has_value());

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
    if (arch_major == 12) {
        if (c.has_value()) {
            DG_HOST_ASSERT(c->sizes() == d_tensor.sizes() and c->scalar_type() == d_tensor.scalar_type());
            DG_HOST_ASSERT(c->stride(-1) == 1);
            if (c->data_ptr() == d_tensor.data_ptr())
                DG_HOST_ASSERT(c->strides() == d_tensor.strides());
        }
        if (batch_size == 0 or m == 0 or n == 0)
            return;
        if (c.has_value() and c->data_ptr() != d_tensor.data_ptr())
            d_tensor.copy_(*c);
        if (k == 0) {
            if (not c.has_value())
                d_tensor.zero_();
            return;
        }
    } else if (batch_size == 0 or gemm::early_return(m, n, k, d_tensor, c, sfd)) {
        return;
    }

    if (arch_major == 12) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
        for (const auto& t: {a, b, d_tensor})
            for (const auto size: t.sizes())
                DG_HOST_ASSERT(size <= std::numeric_limits<int>::max() - 255);
        const auto tma_aligned = [](const torch::Tensor& t) {
            const auto bytes = t.element_size();
            return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0
                and t.stride(0) > 0 and t.stride(1) > 0
                and t.stride(0) <= std::numeric_limits<int>::max() / bytes
                and t.stride(1) <= std::numeric_limits<int>::max() / bytes
                and (t.stride(0) * bytes) % 16 == 0 and (t.stride(1) * bytes) % 16 == 0;
        };
        const auto aligned_copy = [&](const torch::Tensor& t, const bool copy_values) {
            const int64_t quantum = 16 / t.element_size();
            const int64_t width = (t.size(2) + quantum - 1) / quantum * quantum;
            DG_HOST_ASSERT(t.size(1) * width <= std::numeric_limits<int>::max() / t.element_size());
            auto buffer = torch::empty({t.size(0), t.size(1), width}, t.options()).narrow(2, 0, t.size(2));
            if (copy_values)
                buffer.copy_(t);
            return buffer;
        };
        const auto native_a = tma_aligned(a) ? a : aligned_copy(a, true);
        const auto native_b = tma_aligned(b) ? b : aligned_copy(b, true);
        const bool direct_d = tma_aligned(d_tensor);
        const auto native_d = direct_d ? d_tensor : aligned_copy(d_tensor, c.has_value());
        const auto native_c = c.has_value() ? std::make_optional(native_d) : std::nullopt;
        constexpr int kSwapAbMMax = 32;
        const bool swap_ab = m >= 1 and m <= kSwapAbMMax and not c.has_value();
        if (swap_ab) {
            const auto [ga, gb, gk] = recipe.value_or(get_default_recipe(sfa.scalar_type(), sfb.scalar_type()));
            std::optional<std::tuple<int, int, int>> swap_recipe = std::nullopt;
            const auto [sf_a, sf_b, gran_a, gran_b] = layout::transform_sf_pair_into_required_layout(
                sfb, sfa, n, m, k, swap_recipe, std::make_tuple(gb, gk), std::make_tuple(ga, gk),
                batch_size, batch_size, false);
            sm120_fp8_fp4_bmm(native_b, sf_a, native_a, sf_b, native_c, native_d,
                              batch_size, n, m, k, gran_a, gran_b, major_b, major_a, compiled_dims, true);
        } else {
            const auto [sf_a, sf_b, gran_a, gran_b] = layout::transform_sf_pair_into_required_layout(
                sfa, sfb, m, n, k, recipe, std::nullopt, std::nullopt, batch_size, batch_size, false);
            sm120_fp8_fp4_bmm(native_a, sf_a, native_b, sf_b, native_c, native_d,
                              batch_size, m, n, k, gran_a, gran_b, major_a, major_b, compiled_dims);
        }
        if (not direct_d)
            d_tensor.copy_(native_d);
        return;
    }

    // Transform scaling factors
    const auto [transformed_sfa, transformed_sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        sfa, sfb, m, n, k, recipe, std::nullopt, std::nullopt, batch_size, batch_size, false);

    // Dispatch implementation
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
    DG_HOST_ASSERT(a.first.is_cuda());
    const auto& output = std::holds_alternative<torch::Tensor>(d) ? std::get<torch::Tensor>(d)
        : std::get<std::pair<torch::Tensor, torch::Tensor>>(d).first;
    for (const auto& t: {a.second, b.first, b.second, output})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == a.first.device());
    if (c.has_value())
        DG_HOST_ASSERT(c->is_cuda() and c->device() == a.first.device());
    const c10::cuda::CUDAGuard device_guard(a.first.device());
    DG_HOST_ASSERT(jit->device.get_arch_major() == at::cuda::getDeviceProperties(a.first.get_device())->major);
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
    } else if (expr == "bhd,hdr->bhr" and (arch_major == 10 or arch_major == 12)) {
        // (batch_size, m, n, k): (h, b, r, d)
        DG_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        const auto perm_a = a.first.permute({1, 0, 2});
        const auto perm_sfa = a.second.permute({1, 0, 2});
        auto perm_b = b.first.permute({0, 2, 1});
        const auto perm_sfb = b.second.permute({0, 2, 1});
        if (arch_major == 12)
            perm_b = perm_b.contiguous();
        const auto perm_d = std::get<torch::Tensor>(d).permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, perm_d, perm_c, recipe, "nk");
    } else if (expr == "bhd,bhr->hdr" and (arch_major == 10 or arch_major == 12)) {
        // (batch_size, m, n, k): (h, d, r, b)
        DG_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        auto perm_a = a.first.permute({1, 2, 0});
        const auto perm_sfa = a.second.permute({1, 2, 0});
        auto perm_b = b.first.permute({1, 2, 0});
        const auto perm_sfb = b.second.permute({1, 2, 0});
        if (arch_major == 12) {
            perm_a = perm_a.contiguous();
            perm_b = perm_b.contiguous();
        }
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
