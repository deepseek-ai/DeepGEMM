#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <torch/python.h>

#include "math.hpp"
#include "exception.hpp"
#include "../jit/device_runtime.hpp"

namespace deep_gemm {

// Major-ness stuffs
static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if (dim == 3)
        DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

static cute::UMMA::Major get_major_type_ab(const torch::Tensor& t) {
    major_check(t);
    return t.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
}

static void check_major_type_cd(const torch::Tensor& t) {
    // NOTES: the library only supports row-major output layouts
    major_check(t);
    DG_HOST_ASSERT(t.stride(-1) == 1);
}

static bool fp8_requires_k_major() {
    return device_runtime->get_arch_major() == 9;
}

// Tensor utils
template <int N>
static auto get_shape(const torch::Tensor& t) {
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<N>());
}

// Recipe
static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        DG_HOST_ASSERT(sfa_dtype == torch::kFloat and sfb_dtype == torch::kFloat);
        return {1, 128, 128};
    } else if (arch_major == 10) {
        DG_HOST_ASSERT(sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt);
        return sfb_dtype == torch::kFloat ?
            std::make_tuple(1, 128, 128):   // Legacy format or 1D2D kernels
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
                                     const bool& contiguous_check = false,
                                     const std::optional<torch::ScalarType>& type_check = std::nullopt) {
    // Type check
    if (type_check.has_value())
        DG_HOST_ASSERT(sf.scalar_type() == type_check.value());

    // Always do shape checks
    const auto& sf_dtype = sf.scalar_type();
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
        DG_HOST_ASSERT(sf.stride(-2) == 1);
        DG_HOST_ASSERT(sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size()));
    }

    // Hopper SFB must be contiguous
    if (contiguous_check)
        DG_HOST_ASSERT(sf.is_contiguous());
    return sf;
}

// Value matrix layout
static int get_mk_alignment_for_contiguous_layout() {
    return 128;
}

} // namespace deep_gemm
