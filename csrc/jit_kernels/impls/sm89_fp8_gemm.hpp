#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/sm89.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM89FP8GemmRuntime final {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        deep_jit::cuda::LaunchOptions options;

        const void *gmem_a, *gmem_b, *gmem_c, *gmem_d, *gmem_sfa, *gmem_sfb;
        uint32_t shape_m, shape_n, shape_k;
        uint32_t sfa_k_stride, sfb_k_stride;
        uint32_t sfb_n_stride, sfb_n_block_stride;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/sm89_fp8_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm89_fp8_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {},
        {}
    >);
}};
)",
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.pipeline_config.num_stages, args.gemm_config.launch_config.num_threads,
        args.gemm_config.launch_config.num_sms,
        args.gemm_desc.with_accumulation ? "true" : "false",
        args.gemm_desc.kernel_type == KernelType::Kernel1D2D ? "true" : "false",
        to_string(args.gemm_desc.cd_dtype)));

        // Launch
        jit->launch(
            kernel, args.options,
            args.gmem_a, args.gmem_b, args.gmem_c, args.gmem_d,
            args.gmem_sfa, args.gmem_sfb,
            args.shape_m, args.shape_n, args.shape_k,
            args.sfa_k_stride, args.sfb_k_stride, args.sfb_n_stride, args.sfb_n_block_stride
        );
    }
};

// `a`/`b` are `[M, K]` and `[N, K]` (both K-major) with FP32 scales of granularity
// `(1, BLOCK_K)`. `d` is `[M, N]` N-major and optional accumulation happens in FP32.
static void sm89_fp8_gemm(const torch::Tensor& a, const torch::Tensor& sfa,
                          const torch::Tensor& b, const torch::Tensor& sfb,
                          const std::optional<torch::Tensor>& c,
                          const torch::Tensor& d,
                          const int& m, const int& n, const int& k,
                          const std::string& compiled_dims,
                          const KernelType& kernel_type) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 8 and jit->device.get_arch_minor() == 9);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn and b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat and sfb.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kFloat or d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(c.has_value() == false or c->scalar_type() == d.scalar_type());

    // `cp.async` stages 16 bytes at a time, so every K-major row of A/B (whose pitch is `k`) must
    // start 16-byte aligned. N and M stay unconstrained; ragged tiles are zero-filled.
    DG_HOST_ASSERT(k % 16 == 0 and m > 0 and n > 0 and k > 0);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(a.data_ptr()) % 16 == 0);
    DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(b.data_ptr()) % 16 == 0);

    const bool is_1d2d = kernel_type == KernelType::Kernel1D2D;

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = kernel_type,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = c.has_value(),
        .num_sms = runtime->get_num_sms(),
        .tc_util = runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM89ArchSpec>(desc);

    // `transform_sf_into_required_layout` returns MN-major, TMA-aligned FP32 SFA (the element index
    // of `(mn, k_block)` is `mn + k_block * tma_aligned_size(mn)`)
    const uint32_t sfa_k_stride = sfa.stride(-1);
    DG_HOST_ASSERT(sfa.stride(-2) == 1);
    DG_HOST_ASSERT(sfa_k_stride == static_cast<uint32_t>(get_tma_aligned_size(m, sizeof(float))));

    // 1D1D stages SFB the same way; 1D2D keeps it as `[ceil(N / 128), ceil(K / 128)]`, either
    // row- or column-major, and the kernel indexes it with the `(n_block, k_block)` strides
    uint32_t sfb_k_stride = 0, sfb_n_stride = 0, sfb_n_block_stride = 0;
    if (is_1d2d) {
        DG_HOST_ASSERT(sfb.dim() == 2 and sfb.size(0) == ceil_div(n, 128));
        DG_HOST_ASSERT(config.layout.block_n <= 128 and 128 % config.layout.block_n == 0);
        sfb_n_stride = static_cast<uint32_t>(sfb.stride(0));
        sfb_n_block_stride = static_cast<uint32_t>(sfb.stride(1));
    } else {
        sfb_k_stride = sfb.stride(-1);
        DG_HOST_ASSERT(sfb.stride(-2) == 1);
        DG_HOST_ASSERT(sfb_k_stride == static_cast<uint32_t>(get_tma_aligned_size(n, sizeof(float))));
    }

    // Compile and launch
    SM89FP8GemmRuntime::compile_and_launch("sm89_fp8_gemm", {
        .gemm_desc = desc,
        .gemm_config = config,
        .options = {
            .num_smem_bytes = config.pipeline_config.smem_size,
            .grid_dim = dim3(config.launch_config.num_sms, 1, 1),
            .block_dim = dim3(config.launch_config.num_threads, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
        },
        .gmem_a = a.data_ptr(),
        .gmem_b = b.data_ptr(),
        .gmem_c = c.has_value() ? c->data_ptr() : nullptr,
        .gmem_d = d.data_ptr(),
        .gmem_sfa = sfa.data_ptr(),
        .gmem_sfb = sfb.data_ptr(),
        .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n), .shape_k = static_cast<uint32_t>(k),
        .sfa_k_stride = sfa_k_stride, .sfb_k_stride = sfb_k_stride,
        .sfb_n_stride = sfb_n_stride, .sfb_n_block_stride = sfb_n_block_stride,
    });
}

} // namespace deep_gemm