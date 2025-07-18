#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8Gemm1D2DRuntime final: public LaunchRuntime<SM90FP8Gemm1D2DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void *sfb, *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        CUtensorMap tensor_map_sfa;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
        // TODO: add CD dtype
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.num_groups,
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages, args.gemm_config.num_last_stages,
        args.gemm_config.thread_config.num_tma_threads, args.gemm_config.thread_config.num_math_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        to_string(args.gemm_config.gemm_type));
    }

    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.sfb, args.grouped_layout,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_d, args.tensor_map_sfa));
    }
};

static void sm90_fp8_gemm_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const std::optional<torch::Tensor>& c,
                               const torch::Tensor& d,
                               const int& m, const int& n, const int& k,
                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                               const std::string& compiled_dims) {
    DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto& aligned_k = align(k, 128);
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::Normal, KernelType::Kernel1D2D,
        m, n, k, 1, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // Requires no TMA splits
    DG_HOST_ASSERT(config.smem_config.swizzle_a_mode == config.block_k);
    DG_HOST_ASSERT(config.smem_config.swizzle_b_mode == config.block_k);
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = 1,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .sfb = sfb.data_ptr(),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto& code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_gemm_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_m_grouped_fp8_gemm_contiguous_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                                    const torch::Tensor& d,
                                                    const torch::Tensor& m_indices,
                                                    const int& num_groups, const int& m, const int& n, const int& k,
                                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                    const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto& aligned_k = align(k, 128);
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::Kernel1D2D,
        m, n, k, num_groups, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Requires no TMA splits
    DG_HOST_ASSERT(config.smem_config.swizzle_a_mode == config.block_k);
    DG_HOST_ASSERT(config.smem_config.swizzle_b_mode == config.block_k);
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .sfb = sfb.data_ptr(),
        .grouped_layout = m_indices.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto& code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_m_grouped_fp8_gemm_contiguous_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_fp8_m_grouped_gemm_masked_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                const torch::Tensor& b, const torch::Tensor& sfb,
                                                const torch::Tensor& d,
                                                const torch::Tensor& masked_m,
                                                const int& num_groups, const int& m, const int& n, const int& k,
                                                const int& expected_m,
                                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 128);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::MGroupedMasked, KernelType::Kernel1D2D,
        expected_m, n, k, num_groups, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Requires no TMA splits
    DG_HOST_ASSERT(config.smem_config.swizzle_a_mode == config.block_k);
    DG_HOST_ASSERT(config.smem_config.swizzle_b_mode == config.block_k);
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, num_groups, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .sfb = sfb.data_ptr(),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto& code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_m_grouped_gemm_masked_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
