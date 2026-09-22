#pragma once

#include <algorithm>
#include <cstdio>
#include <format>
#include <limits>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm120_bmn_bnk_mn_gemm(const torch::Tensor &a,
                                  const torch::Tensor &b,
                                  const torch::Tensor &d,
                                  const int &s, const int &m, const int &n, const int &k) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    constexpr int block_k = 64;
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    DG_HOST_ASSERT(k % block_k == 0);
    DG_HOST_ASSERT(m % 64 == 0 and n % 64 == 0);
    DG_HOST_ASSERT(static_cast<int64_t>(s) * static_cast<int64_t>(std::max(m, n)) <= std::numeric_limits<int>::max());

    const int swizzle_ab_mode = get_swizzle_mode(block_k, static_cast<int>(a.element_size()));
    DG_HOST_ASSERT(swizzle_ab_mode == 128);

    const int num_sms = runtime->get_num_sms();
    const int num_mn_blocks = ceil_div(m, block_m) * ceil_div(n, block_n);
    const int num_sk_blocks = s * (k / block_k);
    const int split_factor = ceil_div(num_sk_blocks, std::max(num_sms / num_mn_blocks, 1));

    int num_stages = 3, smem_size = 0;
    while (true) {
        const int smem_a_per_stage = block_m * block_k * sizeof(cutlass::bfloat16_t);
        const int smem_b_per_stage = block_n * block_k * sizeof(cutlass::bfloat16_t);
        const int smem_barrier = num_stages * 8 * 2;

        smem_size = 0;
        smem_size += (smem_a_per_stage + smem_b_per_stage) * num_stages;
        smem_size += smem_barrier;

        if (smem_size <= SM120ArchSpec::smem_capacity)
            break;

        -- num_stages;
    }
    DG_HOST_ASSERT(num_stages > 0);

    // Print configs
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("SM120 bmk_bnk_mn: S: %d, M: %d, N: %d, K: %d -> "
               "split_factor: %d, stages: %d, shared memory: %d\n",
               s, m, n, k, split_factor, num_stages, smem_size);
    }

    const auto tensor_map_a = make_tma_2d_desc(a, k, s * m, block_k, block_m, k, swizzle_ab_mode);
    const auto tensor_map_b = make_tma_2d_desc(b, k, s * n, block_k, block_n, k, swizzle_ab_mode);

    // Compile
    const auto kernel = jit->compile("sm120_bmn_bnk_mn_gemm", std::format(R"(
#include <deep_gemm/impls/sm120_bmk_bnk_mn.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_bmn_bnk_mn_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {}
    >);
}};
)",
        m, n, k,
        block_m, block_n, block_k,
        split_factor,
        swizzle_ab_mode,
        num_stages,
        num_tma_threads, num_math_threads));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_mn_blocks * ceil_div(num_sk_blocks, split_factor), 1, 1),
            .block_dim = dim3(num_tma_threads + num_math_threads, 1, 1),
        },
        s, tensor_map_a, tensor_map_b, d.data_ptr<float>()
    );
}

} // namespace deep_gemm
