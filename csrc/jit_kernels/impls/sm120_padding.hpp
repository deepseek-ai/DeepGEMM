#pragma once

#include "runtime_utils.hpp"
#include "../../runtime/runtime.hpp"

namespace deep_gemm {

static void sm120_clear_padding(const torch::Tensor& d, const torch::Tensor& layout,
                                int m, int n, int groups, bool k_grouped,
                                bool psum, int alignment) {
    const auto kernel = jit->compile("sm120_clear_padding", std::format(R"(
#include <deep_gemm/impls/sm120_padding.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_clear_padding<{}, {}, {}>);
}}
)", to_string(d.scalar_type()), k_grouped, psum));
    jit->launch(kernel, {
        .grid_dim = dim3(k_grouped or psum ? groups : m, 1, 1),
        .block_dim = dim3(256, 1, 1),
    }, d.data_ptr(), layout.data_ptr(), static_cast<uint32_t>(m), static_cast<uint32_t>(n),
       static_cast<uint32_t>(groups), static_cast<uint32_t>(alignment), static_cast<uint64_t>(d.stride(-2)));
}

static void sm120_copy_grouped_output(const torch::Tensor& src, const torch::Tensor& d,
                                      const torch::Tensor& layout, int m, int n, int groups,
                                      bool masked, bool psum, int alignment) {
    const auto kernel = jit->compile("sm120_copy_grouped_output", std::format(R"(
#include <deep_gemm/impls/sm120_padding.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_copy_grouped_output<{}, {}>);
}}
)", masked, psum));
    jit->launch(kernel, {
        .grid_dim = dim3(masked or psum ? groups : m, 1, 1),
        .block_dim = dim3(256, 1, 1),
    }, src.data_ptr(), d.data_ptr(), layout.data_ptr(),
       static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(alignment),
       static_cast<uint64_t>(src.stride(-2)), static_cast<uint64_t>(src.stride(-1)),
       static_cast<uint64_t>(masked ? src.stride(0) : 0),
       static_cast<uint64_t>(d.stride(-2)), static_cast<uint64_t>(d.stride(-1)),
       static_cast<uint64_t>(masked ? d.stride(0) : 0));
}

} // namespace deep_gemm
