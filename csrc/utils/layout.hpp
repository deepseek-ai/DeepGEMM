#pragma once

#include <c10/util/accumulate.h>
#include <c10/util/strides.h>
#include <cute/arch/mma_sm100_umma.hpp>
#include <torch/python.h>

#include <deep_gemm/common/types.cuh>

#include "math.hpp"
#include "exception.hpp"
#include "../runtime/jit.hpp"
#include "../runtime/runtime.hpp"

namespace deep_gemm {

// Major-ness stuffs
template <bool kRequireContiguousBatch = true>
static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if constexpr (kRequireContiguousBatch) {
        if (dim == 3)
            DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    }
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

template <bool kRequireContiguousBatch = true>
static cute::UMMA::Major get_major_type_ab(const torch::Tensor& t) {
    major_check<kRequireContiguousBatch>(t);
    return t.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
}

template <bool kRequireContiguousBatch = true>
static void check_major_type_cd(const torch::Tensor& t) {
    // NOTES: the library only supports row-major output layouts
    major_check<kRequireContiguousBatch>(t);
    DG_HOST_ASSERT(t.stride(-1) == 1);
}

static bool fp8_fp4_requires_k_major(const torch::Tensor& a, const torch::Tensor& b) {
    return jit->device.get_arch_major() == 9 or
           (a.scalar_type() == kPackedFP4 and b.scalar_type() == kPackedFP4);
}

// Tensor utils
template <int N>
static auto get_shape(const torch::Tensor& t) {
    DG_HOST_ASSERT(t.is_cuda());
    DG_HOST_ASSERT(t.dim() == N);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

// Returns logical shape for packed FP4 by expanding the last dimension.
template <int N>
static auto get_logical_shape(const torch::Tensor& t) {
    auto shape = get_shape<N>(t);
    if (t.scalar_type() == kPackedFP4)
        std::get<N - 1>(shape) *= 2;
    return shape;
}

static std::tuple<int, int> check_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& arch_major) {
    auto [mn, k] = get_shape<2>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and arch_major == 10);
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(mn, k);
}

static std::tuple<int, int, int> check_grouped_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& arch_major) {
    auto [num_groups, mn, k] = get_shape<3>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and arch_major == 10);
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(num_groups, mn, k);
}

// Recipe
static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(sfa_dtype == torch::kFloat and sfb_dtype == torch::kFloat);
        return {1, 128, 128};
    } else if (arch_major == 10) {
        DG_HOST_ASSERT(sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt);
        return sfb_dtype == torch::kFloat ?
            std::make_tuple(1, 128, 128):   // Legacy format
            std::make_tuple(1,   1, 128);   // 1D1D kernels
    }
    DG_HOST_UNREACHABLE("Unknown recipe");
}

// SF layouts
static torch::Tensor check_sf_layout(const torch::Tensor& sf,
                                     const int& mn, const int& k,
                                     const int& gran_mn, const int& gran_k,
                                     const std::optional<int>& num_groups,
                                     const bool& tma_stride_check = false,
                                     const bool& sm90_sfb_check = false,
                                     const std::optional<torch::ScalarType>& type_check = std::nullopt) {
    // Type check
    if (type_check.has_value())
        DG_HOST_ASSERT(sf.scalar_type() == type_check.value());

    // Always do shape checks
    const auto sf_dtype = sf.scalar_type();
    DG_HOST_ASSERT(sf_dtype == torch::kFloat or sf_dtype == torch::kInt);
    DG_HOST_ASSERT(sf.dim() == static_cast<int>(num_groups.has_value()) + 2);
    if (num_groups.has_value())
        DG_HOST_ASSERT(sf.size(-3) == num_groups.value());
    DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));
    DG_HOST_ASSERT(sf.size(-1) == ceil_div(k, gran_k * (sf_dtype == torch::kFloat ? 1 : 4)));

    // TMA stride checks: TMA aligned and MN-major
    if (tma_stride_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));
        // Check contiguity in the MN direction
        DG_HOST_ASSERT(sf.stride(-2) == 1 or mn == 1);
        const auto compact_stride = get_tma_aligned_size(mn, sf.element_size());
        DG_HOST_ASSERT(sf.stride(-1) >= compact_stride);
        DG_HOST_ASSERT(sf.stride(-1) % 4 == 0);
    }

    // SM90 SFB must be contiguous, or contiguous after transposing the last two dimensions
    if (sm90_sfb_check) {
        if (num_groups.has_value())
            DG_HOST_ASSERT(sf.stride(-3) == sf.size(-2) * sf.size(-1));
        DG_HOST_ASSERT((sf.stride(-1) == 1 and sf.stride(-2) == sf.size(-1)) or
                       (sf.stride(-1) == sf.size(-2) and sf.stride(-2) == 1));
    }
    return sf;
}

static bool is_localized(const torch::Tensor& t) {
    const auto& allocator = locality_domain_allocator;
    const auto num_domains = LocalityDomainAllocator::get_num_locality_domains();
    const auto slice_sizes = t.sizes().slice(1);
    const auto num_slice_bytes = c10::multiply_integers(slice_sizes) * t.element_size();
    bool localized = t.size(0) == num_domains and t.strides().slice(1) == c10::IntArrayRef(c10::contiguous_strides(slice_sizes));
    for (int d = 0; localized and d < num_domains; ++ d)
        localized = allocator.get_locality_domain(static_cast<const char*>(t.data_ptr()) + d * t.stride(0) * t.element_size(), num_slice_bytes) == d;
    DG_HOST_ASSERT(localized or t.is_contiguous());
    return localized;
}

// Accepts localized or unlocalized weights; returns the logical `(N, K)`
static std::tuple<int, int> check_weights_layout_2d(const torch::Tensor& t) {
    if (is_localized(t)) {
        const auto [num_domains, n_per_domain, k] = get_logical_shape<3>(t);
        return std::make_tuple(num_domains * n_per_domain, k);
    }
    return get_logical_shape<2>(t);
}

// Accepts localized or unlocalized weights; returns the logical `(B, N, K)`
static std::tuple<int, int, int> check_weights_layout_3d(const torch::Tensor& t) {
    if (is_localized(t)) {
        const auto [num_domains, num_batches, n_per_domain, k] = get_logical_shape<4>(t);
        return std::make_tuple(num_batches, num_domains * n_per_domain, k);
    }
    return get_logical_shape<3>(t);
}

} // namespace deep_gemm
