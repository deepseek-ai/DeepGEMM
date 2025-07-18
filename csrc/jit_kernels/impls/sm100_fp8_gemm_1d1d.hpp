#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100FP8Gemm1D1DRuntime final: public LaunchRuntime<SM100FP8Gemm1D1DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_c;
        CUtensorMap tensor_map_d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_1d1d_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {}
    >);
}};
)",
        to_string(args.gemm_config.major_a), to_string(args.gemm_config.major_b),
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.num_groups,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages, args.gemm_config.num_last_stages,
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        to_string(args.gemm_config.gemm_type),
        args.gemm_config.with_accumulation,
        to_string(args.gemm_config.cd_dtype));
    }

    static void launch_impl(const cudaKernel_t& kernel, const cudaLaunchConfig_t& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(&config, kernel,
            args.grouped_layout, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_c, args.tensor_map_d));
    }
};

static void sm100_fp8_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 128);
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::Normal, KernelType::Kernel1D1D,
        m, n, k, 1, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    const auto& cd = c.value_or(d);
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_c = make_tma_cd_desc(cd, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(cd.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, config.block_k, 1, 0);

    // Duplicate the accumulator if necessary
    if (c.has_value()) {
        if (c->data_ptr() == d.data_ptr()) {
            DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
        } else {
            // ReSharper disable once CppExpressionWithoutSideEffects
            d.copy_(c.value());
        }
    }

    // Launch
    const SM100FP8Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = 1,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_c,
        .tensor_map_d = tensor_map_d
    };
    const auto& code = SM100FP8Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_fp8_gemm_1d1d", code);
    SM100FP8Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp8_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& m_indices,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 128);
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedContiguous, KernelType::Kernel1D1D,
        m, n, k, num_groups, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, config.block_k, num_groups, 0);

    // Launch kernel
    const SM100FP8Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = m_indices.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_d,
        .tensor_map_d = tensor_map_d
    };
    const auto& code = SM100FP8Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp8_gemm_contiguous_1d1d", code);
    SM100FP8Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_fp8_m_grouped_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                 const torch::Tensor& b, const torch::Tensor& sfb,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& masked_m,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const int& expected_m,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims) {
    const auto& aligned_k = align(k, 128);
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedMasked, KernelType::Kernel1D1D,
        expected_m, n, k, num_groups, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), false,
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, num_groups, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, config.block_k, num_groups, 0);

    // Launch kernel
    const SM100FP8Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_d,
        .tensor_map_d = tensor_map_d
    };
    const auto& code = SM100FP8Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_fp8_m_grouped_gemm_masked_1d1d", code);
    SM100FP8Gemm1D1DRuntime::launch(runtime, args);
}

static void fp8_k_grouped_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                    const std::optional<torch::Tensor>& c,
                                    const torch::Tensor& d,
                                    const int& m, const int& n,
                                    const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);

    int sum_k = 0, sum_sf_k = 0;
    for (const auto& k: ks) {
        sum_k += k, sum_sf_k += ceil_div(k, 512);
        DG_HOST_ASSERT(k % 128 == 0);
    }
    const auto& num_groups = static_cast<int>(ks.size());

    // Get config using max K for better performance
    const auto& max_k = *std::ranges::max_element(ks);
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::KGroupedContiguous, KernelType::Kernel1D1D,
        m, n, max_k, num_groups, cute::UMMA::Major::MN, cute::UMMA::Major::MN,
        torch::kFloat8_e4m3fn, d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // Create tensor descriptors
    const auto& cd = c.value_or(d);
    const auto& tensor_map_a = make_tma_a_desc(cute::UMMA::Major::MN, a, m, sum_k,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(0)), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(cute::UMMA::Major::MN, b, n, sum_k,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(0)), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(1)), num_groups,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_c = make_tma_cd_desc(cd, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(cd.stride(1)), num_groups,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, sum_sf_k * 512,
                                                  config.block_m, config.block_k, num_groups, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, sum_sf_k * 512,
                                                  config.block_n, config.block_k, num_groups, 0);

    // Duplicate the accumulator if necessary
    if (c.has_value()) {
        DG_HOST_ASSERT(c->data_ptr() == d.data_ptr());
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    }

    // Launch kernel
    const SM100FP8Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = sum_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_c,
        .tensor_map_d = tensor_map_d
    };
    const auto& code = SM100FP8Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_fp8_k_grouped_gemm_1d1d", code);
    SM100FP8Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
