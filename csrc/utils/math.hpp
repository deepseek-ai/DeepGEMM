// TODO: merge this file with `math.cuh` (the device part)
#pragma once

#include <torch/python.h>

#include "exception.hpp"

namespace deep_gemm {

// TODO: use `torch::kFloat4_e2m1fn_x2`
constexpr auto kPackedFP4 = torch::kInt8;

template <typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

static int get_tma_aligned_size(const int& x, const int& element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}

static torch::Tensor fp4_repack_to_k_major(const torch::Tensor& a, int logical_mn) {
    DG_HOST_ASSERT(a.scalar_type() == kPackedFP4);
    const int ndim = a.dim();
    DG_HOST_ASSERT(ndim == 2 or ndim == 3);
    const int mn_packed = a.size(-2);
    const int k = a.size(-1);
    DG_HOST_ASSERT(mn_packed * 2 == logical_mn and k % 2 == 0);

    auto lo = a.bitwise_and(0x0F);
    auto hi = a.to(torch::kByte).bitwise_right_shift(4).to(torch::kInt8).bitwise_and(0x0F);

    auto shape_full = a.sizes().vec();
    shape_full[ndim - 2] = logical_mn;
    auto codes = torch::empty(shape_full, a.options());
    using S = torch::indexing::Slice;
    codes.index_put_({torch::indexing::Ellipsis, S(0, torch::indexing::None, 2), S()}, lo);
    codes.index_put_({torch::indexing::Ellipsis, S(1, torch::indexing::None, 2), S()}, hi);

    auto shape_view = shape_full;
    shape_view[ndim - 1] = k / 2;
    shape_view.push_back(2);
    auto codes2 = codes.view(shape_view);
    auto result = codes2.select(-1, 0).bitwise_and(0x0F)
                  .bitwise_or(codes2.select(-1, 1).bitwise_and(0x0F).to(torch::kByte)
                              .bitwise_left_shift(4).to(torch::kInt8));
    return result.contiguous();
}

} // namespace deep_gemm
