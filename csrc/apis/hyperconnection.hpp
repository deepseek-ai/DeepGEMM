#pragma once

#include "../utils/compatibility.hpp"

#include "../jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp"
#include "../jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <limits>

namespace deep_gemm::hyperconnection {

static void tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& sqr_sum,
                                 const std::optional<int>& num_splits) {
    DG_HOST_ASSERT(a.is_cuda());
    for (const auto& t: {b, d, sqr_sum})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == a.device());
    const c10::cuda::CUDAGuard device_guard(a.device());
    const auto& prop = *at::cuda::getDeviceProperties(a.get_device());
    const auto& cached = jit->device.get_prop();
    DG_HOST_ASSERT(cached.major == prop.major and cached.minor == prop.minor
                   and cached.multiProcessorCount == prop.multiProcessorCount
                   and cached.sharedMemPerBlockOptin == prop.sharedMemPerBlockOptin);
    // A and B must be K-major, D must be N-major
    DG_HOST_ASSERT(get_major_type_ab(a) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(b) == cute::UMMA::Major::K);
    // Empty split outputs have a nonzero batch stride in PyTorch; SM120 accepts them as a no-op.
    // Keep the existing SM90/SM100 layout validation unchanged.
    if (cached.major != 12 or a.size(0) != 0)
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
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        sm90_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 10) {
        sm100_tf32_hc_prenorm_gemm(a, b, d, sqr_sum, m, n, k, num_splits.has_value() ? num_splits.value() : 1);
    } else if (arch_major == 12) {
        const int splits = num_splits.value_or(1);
        for (const auto& t: {a, b, d, sqr_sum})
            for (const auto size: t.sizes())
                DG_HOST_ASSERT(size <= std::numeric_limits<int>::max());
        DG_HOST_ASSERT(m <= std::numeric_limits<int>::max() - 127 and k <= std::numeric_limits<int>::max() / 4);
        DG_HOST_ASSERT(n <= 128 and n % 8 == 0 and k % 64 == 0);
        DG_HOST_ASSERT(static_cast<int64_t>(m) * splits <= std::numeric_limits<int>::max());
        const auto aligned_input = [](const torch::Tensor& t) {
            if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 and
                t.stride(0) > 0 and t.stride(0) <= std::numeric_limits<int>::max() / t.element_size() and
                (t.stride(0) * t.element_size()) % 16 == 0)
                return t;
            auto copy = torch::empty(t.sizes(), t.options());
            copy.copy_(t);
            return copy;
        };
        const auto native_a = aligned_input(a);
        const auto native_b = aligned_input(b);
        const auto direct_d = [&]() {
            const auto row_stride = d.stride(-2);
            if (reinterpret_cast<uintptr_t>(d.data_ptr()) % 8 != 0 or row_stride < n
                or row_stride > std::numeric_limits<int>::max() / 4 or row_stride % 2 != 0)
                return false;
            const auto row_span = (m - 1LL) * row_stride + n;
            const auto split_stride = d.dim() == 3 ? d.stride(0) : 0;
            if (splits > 1 and (split_stride < row_span or split_stride % 2 != 0))
                return false;
            const auto span_bytes = ((splits - 1LL) * split_stride + row_span) * 4;
            return reinterpret_cast<uintptr_t>(d.data_ptr()) <= std::numeric_limits<uintptr_t>::max() - span_bytes;
        }();
        const auto native_d = direct_d ? d : torch::empty(d.sizes(), d.options());
        sm120_tf32_hc_prenorm_gemm(native_a, native_b, native_d, sqr_sum, m, n, k, splits);
        if (not direct_d)
            d.copy_(native_d);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}


static void register_apis(pybind11::module_& m) {
    m.def("tf32_hc_prenorm_gemm", &tf32_hc_prenorm_gemm,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("sqr_sum"),
          py::arg("num_splits") = std::nullopt);
}

} // namespace deep_gemm::hyperconnection
