#pragma once

#include <algorithm>

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm90.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

template <bool kMasked>
class SM90MXFP8FP8Gemm1D2DRuntime final: public LaunchRuntime<SM90MXFP8FP8Gemm1D2DRuntime<kMasked>> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        void *sfa, *sfb, *grouped_layout;
        uint32_t sfa_stride_group, sfa_stride_m, sfa_stride_k;
        uint32_t sfa_gran_k;
        bool sfa_packed_int32;
        uint32_t sfb_stride_group, sfb_stride_n, sfb_stride_k;
        uint32_t sfb_gran_k;
        bool sfb_packed_int32;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_mxfp8_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;
static constexpr int kSm90MXFP8FP8ScaleRecipeJitVersion = 8;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_mxfp8_fp8_gemm_1d2d_impl<
        {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {},
        {}, {}
    >);
}}
)",
        kMasked ? "true" : "false",
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode,
        args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sfa, args.sfb, args.grouped_layout,
            args.sfa_stride_group, args.sfa_stride_m, args.sfa_stride_k,
            args.sfa_gran_k, args.sfa_packed_int32,
            args.sfb_stride_group, args.sfb_stride_n, args.sfb_stride_k,
            args.sfb_gran_k, args.sfb_packed_int32,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_d));
    }
};

static void tune_mxfp8_fp8_smem_config(GemmConfig& config, const GemmDesc& desc) {
    const int orig_num_stages = config.pipeline_config.num_stages;
    const int original_per_stage =
        config.storage_config.load_block_m * config.layout.block_k * c10::elementSize(desc.a_dtype) +
        config.storage_config.load_block_n * config.layout.block_k * c10::elementSize(desc.b_dtype) +
        align(config.layout.block_m * static_cast<int>(sizeof(float)), 128);
    const int sfa_per_stage = align(config.layout.block_m * (config.layout.block_k / 32) * static_cast<int>(sizeof(uint8_t)), 128);
    const int sfb_per_stage = align(config.layout.block_n * (config.layout.block_k / 32) * static_cast<int>(sizeof(uint8_t)), 128);
    const int smem_extra = config.pipeline_config.smem_size - orig_num_stages * original_per_stage;
    const int merged_per_stage =
        config.storage_config.load_block_m * config.layout.block_k * c10::elementSize(desc.a_dtype) +
        config.storage_config.load_block_n * config.layout.block_k * c10::elementSize(desc.b_dtype) +
        sfa_per_stage + sfb_per_stage;
    int chosen_stages = std::min(orig_num_stages, (SM90ArchSpec::smem_capacity - smem_extra) / merged_per_stage);
    DG_HOST_ASSERT(chosen_stages >= 1);
    config.pipeline_config.num_stages = chosen_stages;
    config.pipeline_config.smem_size = smem_extra + chosen_stages * merged_per_stage;
}

static void sm90_m_grouped_mxfp8_fp8_gemm_contiguous_1d2d(
        const torch::Tensor& a, const torch::Tensor& sfa,
        const torch::Tensor& b, const torch::Tensor& sfb,
        const torch::Tensor& d, const torch::Tensor& grouped_layout,
        const int& num_groups, const int& m, const int& n, const int& k,
        const std::string& compiled_dims,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kUInt8 or sfa.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kUInt8 or sfb.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt and grouped_layout.is_contiguous());
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());

    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedContiguous,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = k, .expected_num_groups = 1
    };
    auto config = get_best_config<SM90ArchSpec>(desc);
    tune_mxfp8_fp8_smem_config(config, desc);
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const auto sfa_gran_k = recipe_a.has_value()
        ? std::get<1>(recipe_a.value())
        : k / (static_cast<int>(sfa.size(1)) * (sfa.scalar_type() == torch::kInt ? 4 : 1));
    DG_HOST_ASSERT(sfa_gran_k == 32 or sfa_gran_k == 128);
    DG_HOST_ASSERT(sfa.size(1) == ceil_div(k, sfa_gran_k * (sfa.scalar_type() == torch::kInt ? 4 : 1)));
    const auto sfb_gran_k = recipe_b.has_value()
        ? std::get<1>(recipe_b.value())
        : k / (static_cast<int>(sfb.size(-1)) * (sfb.scalar_type() == torch::kInt ? 4 : 1));
    DG_HOST_ASSERT(sfb_gran_k == 32 or sfb_gran_k == 128);
    DG_HOST_ASSERT(sfb.size(-1) == ceil_div(k, sfb_gran_k * (sfb.scalar_type() == torch::kInt ? 4 : 1)));

    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(0)), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(1)), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const typename SM90MXFP8FP8Gemm1D2DRuntime<false>::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .sfa = sfa.data_ptr(),
        .sfb = sfb.data_ptr(),
        .grouped_layout = grouped_layout.data_ptr(),
        .sfa_stride_group = 0,
        .sfa_stride_m = static_cast<uint32_t>(sfa.stride(0)),
        .sfa_stride_k = static_cast<uint32_t>(sfa.stride(1)),
        .sfa_gran_k = static_cast<uint32_t>(sfa_gran_k),
        .sfa_packed_int32 = sfa.scalar_type() == torch::kInt,
        .sfb_stride_group = static_cast<uint32_t>(sfb.stride(0)),
        .sfb_stride_n = static_cast<uint32_t>(sfb.stride(1)),
        .sfb_stride_k = static_cast<uint32_t>(sfb.stride(2)),
        .sfb_gran_k = static_cast<uint32_t>(sfb_gran_k),
        .sfb_packed_int32 = sfb.scalar_type() == torch::kInt,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
    };
    const auto code = SM90MXFP8FP8Gemm1D2DRuntime<false>::generate(args);
    const auto runtime = compiler->build("sm90_m_grouped_mxfp8_fp8_gemm_contiguous_1d2d_scale_recipe_v8", code);
    SM90MXFP8FP8Gemm1D2DRuntime<false>::launch(runtime, args);
}

static void sm90_m_grouped_mxfp8_fp8_gemm_masked_1d2d(
        const torch::Tensor& a, const torch::Tensor& sfa,
        const torch::Tensor& b, const torch::Tensor& sfb,
        const torch::Tensor& d, const torch::Tensor& masked_m,
        const int& num_groups, const int& m, const int& n, const int& k,
        const std::string& compiled_dims,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kUInt8 or sfa.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kUInt8 or sfb.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt and masked_m.is_contiguous());
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());

    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    auto config = get_best_config<SM90ArchSpec>(desc);
    tune_mxfp8_fp8_smem_config(config, desc);
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    const auto sfa_gran_k = recipe_a.has_value()
        ? std::get<1>(recipe_a.value())
        : k / (static_cast<int>(sfa.size(-1)) * (sfa.scalar_type() == torch::kInt ? 4 : 1));
    DG_HOST_ASSERT(sfa_gran_k == 32 or sfa_gran_k == 128);
    DG_HOST_ASSERT(sfa.size(-1) == ceil_div(k, sfa_gran_k * (sfa.scalar_type() == torch::kInt ? 4 : 1)));
    const auto sfb_gran_k = recipe_b.has_value()
        ? std::get<1>(recipe_b.value())
        : k / (static_cast<int>(sfb.size(-1)) * (sfb.scalar_type() == torch::kInt ? 4 : 1));
    DG_HOST_ASSERT(sfb_gran_k == 32 or sfb_gran_k == 128);
    DG_HOST_ASSERT(sfb.size(-1) == ceil_div(k, sfb_gran_k * (sfb.scalar_type() == torch::kInt ? 4 : 1)));

    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(1)), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(1)), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), num_groups,
                                               config.storage_config.swizzle_cd_mode);
    const typename SM90MXFP8FP8Gemm1D2DRuntime<true>::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .sfa = sfa.data_ptr(),
        .sfb = sfb.data_ptr(),
        .grouped_layout = masked_m.data_ptr(),
        .sfa_stride_group = static_cast<uint32_t>(sfa.stride(0)),
        .sfa_stride_m = static_cast<uint32_t>(sfa.stride(1)),
        .sfa_stride_k = static_cast<uint32_t>(sfa.stride(2)),
        .sfa_gran_k = static_cast<uint32_t>(sfa_gran_k),
        .sfa_packed_int32 = sfa.scalar_type() == torch::kInt,
        .sfb_stride_group = static_cast<uint32_t>(sfb.stride(0)),
        .sfb_stride_n = static_cast<uint32_t>(sfb.stride(1)),
        .sfb_stride_k = static_cast<uint32_t>(sfb.stride(2)),
        .sfb_gran_k = static_cast<uint32_t>(sfb_gran_k),
        .sfb_packed_int32 = sfb.scalar_type() == torch::kInt,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
    };
    const auto code = SM90MXFP8FP8Gemm1D2DRuntime<true>::generate(args);
    const auto runtime = compiler->build("sm90_m_grouped_mxfp8_fp8_gemm_masked_1d2d_scale_recipe_v8", code);
    SM90MXFP8FP8Gemm1D2DRuntime<true>::launch(runtime, args);
}

} // namespace deep_gemm
