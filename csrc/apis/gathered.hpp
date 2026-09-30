#pragma once

#include <cstdint>

#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"

namespace deep_gemm::gemm {

struct SM90GatheredArchSpec : SM90ArchSpec {
    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        auto layouts = SM90ArchSpec::get_layout_candidates(desc);
        std::erase_if(layouts, [](const Layout& layout) { return layout.get_cluster_size() != 1; });
        return layouts;
    }
};

// Padding row indices must also reference valid source rows.
static void m_grouped_bf16_gemm_nt_contiguous_gathered(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const torch::Tensor& grouped_layout, const torch::Tensor& a_row_indices) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 9);
    DG_HOST_ASSERT(a.is_cuda() and a.scalar_type() == torch::kBFloat16 and a.dim() == 2);
    DG_HOST_ASSERT(b.device() == a.device() and b.scalar_type() == torch::kBFloat16 and b.dim() == 3);
    DG_HOST_ASSERT(d.device() == a.device() and d.scalar_type() == torch::kBFloat16 and d.dim() == 2);
    DG_HOST_ASSERT(grouped_layout.device() == a.device() and grouped_layout.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(a_row_indices.device() == a.device() and a_row_indices.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(grouped_layout.dim() == 1 and grouped_layout.is_contiguous());
    DG_HOST_ASSERT(a_row_indices.dim() == 1 and a_row_indices.is_contiguous());
    DG_HOST_ASSERT(a.stride(1) == 1 and a.stride(0) % 8 == 0 and b.is_contiguous());
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0);
    DG_HOST_ASSERT(heuristics_runtime->get_mk_alignment_for_contiguous_layout() == 128);
    check_major_type_cd(d);
    const auto [r, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b);
    const auto [m, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(n == n_ and k == k_ and k > 0 and k % 64 == 0 and n > 0 and num_groups > 0);
    DG_HOST_ASSERT(grouped_layout.numel() == m and a_row_indices.numel() == m);
    if (m == 0)
        return;
    DG_HOST_ASSERT(r > 0);
    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedContiguous,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(), .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = runtime->get_num_sms(), .tc_util = runtime->get_tc_util(),
        .compiled_dims = "nk", .expected_m = m, .expected_n = n, .expected_k = k,
        .expected_num_groups = 1
    };
    const auto config = get_best_config<SM90GatheredArchSpec>(desc);
    const auto tensor_map_b = make_tma_b_desc(desc.major_b, b, n, k,
        config.storage_config.load_block_n, config.layout.block_k,
        static_cast<int>(b.stride(-2)), num_groups, config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
        config.storage_config.store_block_m, config.storage_config.store_block_n,
        static_cast<int>(d.stride(-2)), 1, config.storage_config.swizzle_cd_mode);
    SM90BF16GemmRuntime::compile_and_launch("sm90_m_grouped_bf16_gemm_contiguous_gathered", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
        },
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = {}, .tensor_map_b = tensor_map_b, .tensor_map_cd = tensor_map_cd,
        .gathered_a = a.data_ptr(), .a_row_indices = a_row_indices.data_ptr(),
        .gathered_a_stride = static_cast<uint64_t>(a.stride(0)),
        .gathered_a_rows = static_cast<uint32_t>(r)
    });
}

} // namespace deep_gemm::gemm
