#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm120_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& d,
                                       const torch::Tensor& sqr_sum,
                                       const int& m, const int& n, const int& k,
                                       const int& num_splits) {
    constexpr int block_m = 128;
    constexpr int block_k = 64;
    constexpr int num_math_threads = 256;
    constexpr int num_tma_threads = 128;
    constexpr int num_threads = num_math_threads + num_tma_threads;

    const int block_n = align(n, 16);
    DG_HOST_ASSERT(n <= 128 and n % 8 == 0);
    DG_HOST_ASSERT(k % block_k == 0);

    // A: BF16 [M, K], K-major (K contiguous). TMA inner=K, outer=M.
    const auto swizzle_a = get_swizzle_mode(block_k, a.element_size());
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                              block_m, block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              swizzle_a, 0, true);

    // B: FP32 [N, K], K-major (K contiguous). TMA inner=K, outer=N.
    // SMEM layout: [BLOCK_N rows, BLOCK_K cols] K-contiguous, swizzle based on K row bytes.
    const auto swizzle_b = get_swizzle_mode(block_k, b.element_size());
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                              block_n, block_k,
                                              static_cast<int>(b.stride(0)), 1,
                                              swizzle_b, 0, true);

    // Pipeline stages: no SMEM_D (direct FP32 store)
    const int smem_a_per_stage = block_m * block_k * static_cast<int>(a.element_size());
    const int smem_b_per_stage = block_n * block_k * static_cast<int>(b.element_size());
    int num_stages = 12;
    int smem_size = 0;
    while (num_stages > 0) {
        smem_size = (smem_a_per_stage + smem_b_per_stage) * num_stages + num_stages * 2 * 8;
        if (smem_size <= SM120ArchSpec::smem_capacity)
            break;
        --num_stages;
    }
    DG_HOST_ASSERT(num_stages >= 2);

    const int num_m_blocks = ceil_div(m, block_m);
    const int grid_size = num_m_blocks * num_splits;

    // Compile
    const auto kernel = jit->compile("sm120_tf32_hc_prenorm_gemm", std::format(R"(
#include <deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_tf32_hc_prenorm_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {},
        {}, {}
    >);
}};
)",
        n, k,
        block_m, block_n, block_k,
        num_splits,
        num_stages,
        num_math_threads, num_tma_threads));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(grid_size, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
        },
        m, tensor_map_a, tensor_map_b,
        reinterpret_cast<float*>(d.data_ptr()), reinterpret_cast<float*>(sqr_sum.data_ptr())
    );
}

} // namespace deep_gemm
