#pragma once

#include <format>
#include <torch/python.h>

#include "../jit_kernels/heuristics/sm100.hpp"
#include "../jit_kernels/impls/runtime_utils.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::experimental {

// A is [M, S, 256] with 384-byte segment pitch.
// Scales are prepacked MN-major UE8M0 with K granularity 32.
static void fp8_gemm_nt_segmented_k256(const std::pair<torch::Tensor, torch::Tensor>& a,
                                       const std::pair<torch::Tensor, torch::Tensor>& b,
                                       const torch::Tensor& d) {
    const auto [m, num_segments, segment_k] = get_shape<3>(a.first);
    const auto [n, k] = get_shape<2>(b.first);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_segments > 0 and segment_k == 256);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == num_segments * segment_k);

    DG_HOST_ASSERT(a.first.stride(2) == 1 and b.first.stride(1) == 1);
    const auto segment_stride = a.first.stride(1);
    DG_HOST_ASSERT(segment_stride == 384 and a.first.stride(0) == num_segments * segment_stride);
    check_major_type_cd(d);
    check_sf_layout(a.second, m, k, 1, 32, std::nullopt, true, false, torch::kInt);
    check_sf_layout(b.second, n, k, 1, 32, std::nullopt, true, false, torch::kInt);

    if (m == 0 or n == 0)
        return;

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.first.scalar_type(), .b_dtype = b.first.scalar_type(), .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = "nk"
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_3d_desc(a.first, segment_k, num_segments, m,
                                                config.layout.block_k, 1, config.storage_config.load_block_m,
                                                static_cast<int>(segment_stride), static_cast<int>(a.first.stride(0)),
                                                config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b.first, n, k,
                                              config.storage_config.load_block_n, config.layout.block_k,
                                              static_cast<int>(b.first.stride(0)), 1, config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m, config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(0)), 1, config.storage_config.swizzle_cd_mode);
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(
        config.layout.block_m, config.layout.block_n, desc.get_mma_kind());
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, a.second, m, k,
                                                 sf_block_m, 32, 1, 0, 0, false, 1, static_cast<int>(a.second.stride(1)));
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, b.second, n, k,
                                                 sf_block_n, 32, 1, 0, 0, false, 1, static_cast<int>(b.second.stride(1)));

    const auto kernel = jit->compile("sm100_fp8_gemm_nt_segmented_k256", std::format(R"(
#include <deep_gemm/experimental/sm100_fp8_gemm_nt_segmented_k256.cuh>

using namespace deep_gemm::experimental;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_nt_segmented_k256_impl<
        {}, {}, {}, {},
        {}, {}, {},
        {}, {}, {}, {}
    >);
}}
)", n, k, config.layout.block_m, config.layout.block_n,
        config.storage_config.swizzle_cd_mode,
        config.pipeline_config.num_stages, config.pipeline_config.num_tma_store_stages,
        config.layout.get_cluster_size(), config.layout.cluster_n > 1,
        config.launch_config.num_sms, config.layout.swap_ab));

    jit->launch(kernel, {
        .num_smem_bytes = config.pipeline_config.smem_size,
        .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
        .block_dim = dim3(config.launch_config.num_threads, 1, 1),
        .cluster_dim = dim3(config.layout.get_cluster_size(), 1, 1),
    }, static_cast<uint32_t>(m), tensor_map_a, tensor_map_b, tensor_map_sfa, tensor_map_sfb, tensor_map_cd);
}

} // namespace deep_gemm::experimental
