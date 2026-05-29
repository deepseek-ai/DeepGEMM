#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/common.hpp"
#include "../heuristics/sm100.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

// FP4xFP4 (MXFP4) GEMM via SM100_MMA_MXF4_SS instruction.
// Distinct from main's sm100_fp8_fp4_gemm_1d1d (MXF8F6F4 path) — this is the
// FP4-specialized hardware path, used when both A and B are kPackedFP4.
class SM100FP4Gemm1D1DRuntime final: public LaunchRuntime<SM100FP4Gemm1D1DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        int num_last_stages;

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
#include <deep_gemm/impls/sm100_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_gemm_1d1d_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}
    >);
}};
)",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        // FP4: my .cuh expects SHAPE_K in int32 count (= FP4_count / 8).
        // main's desc.k is FP4 logical count.
        get_compiled_dim(args.gemm_desc.k / 8, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n,
        // FP4: BLOCK_K in int32 count (main's heuristic gives bytes for int8 pack).
        args.gemm_config.layout.block_k / 4,
        args.gemm_desc.num_groups,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages, args.num_last_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        args.gemm_config.layout.swap_ab,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // FP4: my .cuh expects shape_k in int32 count (= FP4_count / 8) at runtime as well.
        // Note: must be non-const (launch_kernel takes &args, can't bind const* to void*).
        int shape_k_int32 = args.gemm_desc.k / 8;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout, args.gemm_desc.m, args.gemm_desc.n, shape_k_int32,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_c, args.tensor_map_d));
    }
};

// Helper: compute num_last_stages from k / block_k / num_stages.
// desc.k is FP4 logical count; main's heuristic block_k is in bytes (int8 elem of packed FP4 tensor).
// 1 byte = 2 FP4, so K_per_block_fp4 = block_k_bytes * 2.
static int compute_num_last_stages_fp4(int k, int block_k_bytes, int num_stages) {
    const int num_k_blocks = ceil_div(k, block_k_bytes * 2);
    const int rem = num_k_blocks % num_stages;
    return rem == 0 ? num_stages : rem;
}

// Build a GemmDesc forcing a_dtype=b_dtype=kPackedFP4 (since this is the FP4xFP4 wrapper).
static GemmDesc make_fp4_desc(GemmType gemm_type, int m, int n, int k, int num_groups,
                              const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                              const at::ScalarType& cd_dtype, bool with_accumulation,
                              const std::string& compiled_dims,
                              int expected_m = 0, int expected_num_groups = 1) {
    return GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = kPackedFP4, .b_dtype = kPackedFP4,
        .cd_dtype = cd_dtype,
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = with_accumulation,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m > 0 ? expected_m : m,
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_num_groups
    };
}

static void sm100_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                const std::string& compiled_dims) {
    constexpr int gran_k = 32;
    const auto desc = make_fp4_desc(GemmType::Normal, m, n, k, 1,
                                    major_a, major_b, d.scalar_type(),
                                    c.has_value(), compiled_dims);
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto cd = c.value_or(d);
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_c = make_tma_cd_desc(cd, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(cd.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k, 1, 0);

    if (c.has_value()) {
        if (c->data_ptr() == d.data_ptr()) {
            DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
        } else {
            d.copy_(c.value());
        }
    }

    const SM100FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_last_stages = compute_num_last_stages_fp4(k, config.layout.block_k, config.pipeline_config.num_stages),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_c,
        .tensor_map_d = tensor_map_d
    };
    const auto code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp4_gemm_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& grouped_layout,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    constexpr int gran_k = 32;
    const auto desc = make_fp4_desc(GemmType::MGroupedContiguous, m, n, k, num_groups,
                                    major_a, major_b, d.scalar_type(),
                                    false, compiled_dims);
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k, num_groups, 0);

    const SM100FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_last_stages = compute_num_last_stages_fp4(k, config.layout.block_k, config.pipeline_config.num_stages),
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_d,
        .tensor_map_d = tensor_map_d
    };
    const auto code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_m_grouped_fp4_gemm_contiguous_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp4_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                 const torch::Tensor& b, const torch::Tensor& sfb,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& masked_m,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const int& expected_m,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims) {
    constexpr int gran_k = 32;
    const auto desc = make_fp4_desc(GemmType::MGroupedMasked, m, n, k, num_groups,
                                    major_a, major_b, d.scalar_type(),
                                    false, compiled_dims,
                                    /*expected_m=*/expected_m, /*expected_num_groups=*/num_groups);
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), num_groups,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k, num_groups, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k, num_groups, 0);

    const SM100FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_last_stages = compute_num_last_stages_fp4(k, config.layout.block_k, config.pipeline_config.num_stages),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_d,
        .tensor_map_d = tensor_map_d
    };
    const auto code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_m_grouped_fp4_gemm_masked_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
