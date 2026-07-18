#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

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
static constexpr int kSm90MXFP8FP8ScaleRecipeJitVersion = 15;

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

struct SM90MXFP8FP8ContiguousLaunchKey {
    int device, num_groups, m, n, k;
    const void *a, *sfa, *b, *sfb, *d, *grouped_layout;
    int64_t a_stride_m, b_stride_group, b_stride_n, d_stride_m;
    int64_t sfa_stride_m, sfa_stride_k, sfb_stride_group, sfb_stride_n, sfb_stride_k;
    uint32_t sfa_gran_k, sfb_gran_k;
    bool sfa_packed_int32, sfb_packed_int32;
    std::string compiled_dims;

    bool operator==(const SM90MXFP8FP8ContiguousLaunchKey& other) const {
        return device == other.device and num_groups == other.num_groups and
               m == other.m and n == other.n and k == other.k and
               a == other.a and sfa == other.sfa and b == other.b and sfb == other.sfb and
               d == other.d and grouped_layout == other.grouped_layout and
               a_stride_m == other.a_stride_m and b_stride_group == other.b_stride_group and
               b_stride_n == other.b_stride_n and d_stride_m == other.d_stride_m and
               sfa_stride_m == other.sfa_stride_m and sfa_stride_k == other.sfa_stride_k and
               sfb_stride_group == other.sfb_stride_group and sfb_stride_n == other.sfb_stride_n and
               sfb_stride_k == other.sfb_stride_k and
               sfa_gran_k == other.sfa_gran_k and sfb_gran_k == other.sfb_gran_k and
               sfa_packed_int32 == other.sfa_packed_int32 and
               sfb_packed_int32 == other.sfb_packed_int32 and compiled_dims == other.compiled_dims;
    }
};

struct SM90MXFP8FP8ContiguousLaunchState {
    SM90MXFP8FP8ContiguousLaunchKey key;
    SM90MXFP8FP8Gemm1D2DRuntime<false>::Args args;
    std::shared_ptr<KernelRuntime> runtime;
};

// The normal serving path repeatedly launches the same GEMM over stable buffers. Cache the
// complete host launch state for that one hot call site: config selection, TMA map encoding,
// generated code and JIT runtime are all invariant while this key matches. The cache is
// thread-local so independent Python execution threads never race on tensor lifetimes.
static thread_local std::optional<SM90MXFP8FP8ContiguousLaunchState> sm90_mxfp8_fp8_contiguous_launch_state;

static void tune_mxfp8_fp8_smem_config(GemmConfig& config, const GemmDesc& desc) {
    const int orig_num_stages = config.pipeline_config.num_stages;
    const int original_per_stage =
        config.storage_config.load_block_m * config.layout.block_k * c10::elementSize(desc.a_dtype) +
        config.storage_config.load_block_n * config.layout.block_k * c10::elementSize(desc.b_dtype) +
        align(config.layout.block_m * static_cast<int>(sizeof(float)), 128);
    // The producer expands the four UE8M0 scales for every A/B row into FP32
    // before publishing a stage, so the math warpgroup only loads scales and
    // performs the exact scale product during promotion.
    const int sfa_per_stage = align(config.layout.block_m * (config.layout.block_k / 32) * static_cast<int>(sizeof(float)), 128);
    const int sfb_per_stage = align(config.layout.block_n * (config.layout.block_k / 32) * static_cast<int>(sizeof(float)), 128);
    const int smem_extra = config.pipeline_config.smem_size - orig_num_stages * original_per_stage;
    // One 64-bit scale-ready mbarrier per math warpgroup and stage. The existing
    // heuristic reserves the full/empty TMA barriers separately in smem_extra.
    const int scale_barriers_per_stage = (config.launch_config.num_math_threads / 128) * static_cast<int>(sizeof(uint64_t));
    const int merged_per_stage =
        config.storage_config.load_block_m * config.layout.block_k * c10::elementSize(desc.a_dtype) +
        config.storage_config.load_block_n * config.layout.block_k * c10::elementSize(desc.b_dtype) +
        sfa_per_stage + sfb_per_stage + scale_barriers_per_stage;
    int chosen_stages = std::min(orig_num_stages, (SM90ArchSpec::smem_capacity - smem_extra) / merged_per_stage);
    DG_HOST_ASSERT(chosen_stages >= 1);
    config.pipeline_config.num_stages = chosen_stages;
    config.pipeline_config.smem_size = smem_extra + chosen_stages * merged_per_stage;
}

// The MXFP8 kernel promotes every 32-wide K chunk with a distinct UE8M0 scale, so it cannot
// hardware-accumulate across the 4 chunks of a 128-K block the way FP8 does. To hide the
// per-chunk WGMMA drain it runs a software pipeline over a per-chunk FP32 accumulator whose
// depth is bounded by the register budget. Clamping to BLOCK_N == 96 keeps kNumAccum == 48,
// small enough for a 2-deep ping-pong pipeline under the 256-thread / 232-register budget.
// (BLOCK_N == 64 was measured slower at large M: the 64x64x32 WGMMA loses more tensor-core
// efficiency than the deeper pipeline recovers.) The shared SM90 heuristic maximizes MMA
// throughput and typically picks block_n up to 192, which would force the serial fallback;
// re-select the fastest candidate with block_n <= 96 so MXFP8 gets the pipeline. Other dtypes
// keep using the unconstrained heuristic.
static constexpr int kMXFP8PipelineMaxBlockN = 96;

static GemmConfig get_mxfp8_fp8_best_config(const GemmDesc& desc) {
    auto config = get_best_config<SM90ArchSpec>(desc);
    if (config.layout.block_n <= kMXFP8PipelineMaxBlockN)
        return config;

    // Filter the candidate layouts down to those that keep the ping-pong accumulator small
    // enough, then pick the fastest among them using the same cost model.
    const auto layout_candidates = SM90ArchSpec::get_layout_candidates(desc);
    bool found = false;
    Layout best_layout{};
    LayoutInfo best_info{};
    for (const auto& candidate: layout_candidates) {
        if (candidate.block_n > kMXFP8PipelineMaxBlockN)
            continue;
        const auto info = SM90ArchSpec::get_layout_info(desc, candidate);
        if (not found or SM90ArchSpec::compare(info, best_info)) {
            best_layout = candidate;
            best_info = info;
            found = true;
        }
    }

    // No narrow candidate survived the heuristic's stage/swizzle filters: keep the original
    // config so the kernel falls back to the (correct) serial path.
    if (not found)
        return config;

    config.layout = best_layout;
    config.storage_config = SM90ArchSpec::get_storage_config(desc, best_layout);
    config.pipeline_config = SM90ArchSpec::get_pipeline_config(desc, best_layout, config.storage_config);
    config.launch_config = SM90ArchSpec::get_launch_config(desc, best_layout);
    return config;
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

    const SM90MXFP8FP8ContiguousLaunchKey launch_key = {
        .device = a.get_device(), .num_groups = num_groups, .m = m, .n = n, .k = k,
        .a = a.data_ptr(), .sfa = sfa.data_ptr(), .b = b.data_ptr(), .sfb = sfb.data_ptr(),
        .d = d.data_ptr(), .grouped_layout = grouped_layout.data_ptr(),
        .a_stride_m = a.stride(0), .b_stride_group = b.stride(0), .b_stride_n = b.stride(1),
        .d_stride_m = d.stride(-2), .sfa_stride_m = sfa.stride(0), .sfa_stride_k = sfa.stride(1),
        .sfb_stride_group = sfb.stride(0), .sfb_stride_n = sfb.stride(1), .sfb_stride_k = sfb.stride(2),
        .sfa_gran_k = static_cast<uint32_t>(sfa_gran_k), .sfb_gran_k = static_cast<uint32_t>(sfb_gran_k),
        .sfa_packed_int32 = sfa.scalar_type() == torch::kInt, .sfb_packed_int32 = sfb.scalar_type() == torch::kInt,
        .compiled_dims = compiled_dims,
    };
    if (sm90_mxfp8_fp8_contiguous_launch_state.has_value() and
        sm90_mxfp8_fp8_contiguous_launch_state->key == launch_key) {
        const auto& cached = sm90_mxfp8_fp8_contiguous_launch_state.value();
        SM90MXFP8FP8Gemm1D2DRuntime<false>::launch(cached.runtime, cached.args);
        return;
    }

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
    auto config = get_mxfp8_fp8_best_config(desc);
    tune_mxfp8_fp8_smem_config(config, desc);
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);

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
    const auto runtime = compiler->build("sm90_m_grouped_mxfp8_fp8_gemm_contiguous_1d2d_scale_recipe_v15", code);
    sm90_mxfp8_fp8_contiguous_launch_state = SM90MXFP8FP8ContiguousLaunchState {
        .key = launch_key,
        .args = args,
        .runtime = runtime,
    };
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
    auto config = get_mxfp8_fp8_best_config(desc);
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
    const auto runtime = compiler->build("sm90_m_grouped_mxfp8_fp8_gemm_masked_1d2d_scale_recipe_v15", code);
    SM90MXFP8FP8Gemm1D2DRuntime<true>::launch(runtime, args);
}

} // namespace deep_gemm
