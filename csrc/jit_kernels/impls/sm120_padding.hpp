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

} // namespace deep_gemm
