#pragma once

#include <cstdlib>

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/layout.hpp"
#include "epilogue.hpp"
#include "runtime_utils.hpp"
#include "sm90_fp8_fp4_gemm_1d2d_rs.hpp"

namespace deep_gemm {

class SM90FP8FP4Gemm1D1DRuntime final: public LaunchRuntime<SM90FP8FP4Gemm1D1DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void *gmem_b_ptr;
        void *gmem_d_ptr;
        void *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_gemm_1d1d_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {}
    >);
}};
)",
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        to_string(args.gemm_desc.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            nullptr, args.gmem_b_ptr,
            args.gmem_d_ptr, args.grouped_layout,
            nullptr,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

class SM90FP8FP4Gemm1D2DRuntime final: public LaunchRuntime<SM90FP8FP4Gemm1D2DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        cute::UMMA::Major major_sfb;
        bool decode_stub;
        void *gmem_b_ptr;
        void *gmem_d_ptr;
        void *sfb;
        void *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        CUtensorMap tensor_map_sfa;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_gemm_1d2d_impl<
        {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}
    >);
}};
)",
        to_string(args.major_sfb),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        get_default_epilogue_type(std::nullopt),
        args.decode_stub ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_b_ptr, args.sfb, args.grouped_layout, args.gmem_d_ptr,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_d, args.tensor_map_sfa));
    }
};

static void sm90_fp8_fp4_gemm_1d1d_fused(const std::pair<torch::Tensor, torch::Tensor>& a,
                                         const std::pair<torch::Tensor, torch::Tensor>& b,
                                         const torch::Tensor& d,
                                         const std::optional<torch::Tensor>& c,
                                         const int& gran_k,
                                         const std::string& compiled_dims) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(gran_k == 128);
    DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(b.first.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(b.second.scalar_type() == torch::kFloat or b.second.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous() and c->is_contiguous());

    const auto [m, k] = get_shape<2>(a.first);
    const auto [n, half_k] = get_shape<2>(b.first);
    const auto [m_d, n_d] = get_shape<2>(d);
    DG_HOST_ASSERT(k % 2 == 0 and half_k * 2 == k);
    DG_HOST_ASSERT(m == m_d and n == n_d);
    DG_HOST_ASSERT(a.second.size(0) == m and a.second.size(1) == ceil_div(k, gran_k));
    DG_HOST_ASSERT(b.second.size(0) == n and b.second.size(1) == ceil_div(k, gran_k));
    DG_HOST_ASSERT(c->sizes() == d.sizes());

    if (m == 0 or n == 0) {
        return;
    }
    if (c->data_ptr() != d.data_ptr()) {
        d.copy_(*c);
    }

    auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.first.scalar_type(),
        .b_dtype = torch::kFloat8_e4m3fn,
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K,
        .major_b = cute::UMMA::Major::K,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    auto config = get_best_config<SM90ArchSpec>(desc);
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);

    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a.first, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k, k, 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, a.second, m, k,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, b.second, n, k,
                                                 config.layout.block_n, config.layout.block_k, 1, 0);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    const SM90FP8FP4Gemm1D1DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .gmem_b_ptr = b.first.data_ptr(),
        .gmem_d_ptr = d.data_ptr(),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM90FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_fp4_gemm_1d1d_fused", code);
    SM90FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm90_m_grouped_fp8_fp4_gemm_contiguous_1d1d_fused(
        const std::pair<torch::Tensor, torch::Tensor>& a,
        const std::pair<torch::Tensor, torch::Tensor>& b,
        const torch::Tensor& d,
        const torch::Tensor& grouped_layout,
        const int& gran_k,
        const std::string& compiled_dims,
        const bool& use_psum_layout,
        const std::optional<int>& expected_m_for_psum_layout,
        const std::optional<int>& block_m_override,
        const std::optional<int>& block_n_override,
        const bool& decode_stub) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(gran_k == 128);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(b.first.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(b.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt and grouped_layout.is_contiguous());
    DG_HOST_ASSERT(a.first.is_contiguous() and b.first.is_contiguous() and d.is_contiguous());

    const auto [m, k] = get_shape<2>(a.first);
    const auto [num_groups, n, half_k] = get_shape<3>(b.first);
    const auto [m_d, n_d] = get_shape<2>(d);
    const auto [layout_size] = get_shape<1>(grouped_layout);
    DG_HOST_ASSERT(k % 2 == 0 and half_k * 2 == k);
    DG_HOST_ASSERT(m == m_d and n == n_d);
    DG_HOST_ASSERT(use_psum_layout ? (layout_size == num_groups) : (layout_size == m));
    if (expected_m_for_psum_layout) {
        DG_HOST_ASSERT(use_psum_layout);
    }
    DG_HOST_ASSERT(a.second.size(0) == m and a.second.size(1) == ceil_div(k, gran_k));
    DG_HOST_ASSERT(b.second.size(0) == num_groups and b.second.size(1) == n and b.second.size(2) == ceil_div(k, gran_k));

    if (m == 0 or n == 0) {
        return;
    }

    std::optional<std::tuple<int, int, int>> recipe = std::nullopt;
    std::optional<std::tuple<int, int>> recipe_a = std::make_tuple(1, gran_k);
    std::optional<std::tuple<int, int>> recipe_b = std::make_tuple(1, gran_k);
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b,
        std::nullopt, num_groups, false);
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);

    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;
    // NOTE: psum layout previously always took the 1d1d fallback path. With the
    // 1d2d psum scheduler / SFB indexing aligned, we let psum also flow into the
    // common 1d2d code below. This relies on:
    //   - All groups in `grouped_layout` having the same K (current_shape_k == shape_k);
    //     true for current call sites where K is shared across groups.
    //   - SFB physical layout `[num_groups, n, shape_k_scales]` already matches
    //     1d2d cooperative ld.global indexing.
    //   - 0-size groups handled by the scheduler's while-loop fallthrough.
    if (false /* use_psum_layout disabled: psum now goes through the 1d2d common path */) {
        auto desc = GemmDesc {
            .gemm_type = gemm_type,
            .kernel_type = KernelType::Kernel1D1D,
            .m = m, .n = n, .k = k, .num_groups = num_groups,
            .a_dtype = a.first.scalar_type(),
            .b_dtype = torch::kFloat8_e4m3fn,
            .cd_dtype = d.scalar_type(),
            .major_a = cute::UMMA::Major::K,
            .major_b = cute::UMMA::Major::K,
            .with_accumulation = false,
            .num_sms = device_runtime->get_num_sms(),
            .tc_util = device_runtime->get_tc_util(),
            .compiled_dims = compiled_dims,
            .expected_m = expected_m_for_psum_layout.value_or(m),
            .expected_n = n,
            .expected_k = k,
            .expected_num_groups = num_groups
        };
        auto config = get_best_config<SM90ArchSpec>(desc);
        DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
        DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);

        const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a.first, m, k,
                                                  config.storage_config.load_block_m,
                                                  config.layout.block_k, k, 1,
                                                  config.storage_config.swizzle_a_mode);
        const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                     config.layout.block_m, config.layout.block_k, 1, 0);
        const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                     config.layout.block_n, config.layout.block_k, num_groups, 0);
        const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                    config.storage_config.store_block_m,
                                                    config.storage_config.store_block_n,
                                                    static_cast<int>(d.stride(-2)), 1,
                                                    config.storage_config.swizzle_cd_mode);

        const SM90FP8FP4Gemm1D1DRuntime::Args& args = {
            .gemm_desc = desc,
            .gemm_config = config,
            .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                      config.pipeline_config.smem_size,
                                      config.layout.get_cluster_size()),
            .gmem_b_ptr = b.first.data_ptr(),
            .gmem_d_ptr = d.data_ptr(),
            .grouped_layout = grouped_layout.data_ptr(),
            .tensor_map_a = tensor_map_a,
            .tensor_map_sfa = tensor_map_sfa,
            .tensor_map_sfb = tensor_map_sfb,
            .tensor_map_cd = tensor_map_cd,
        };
        const auto code = SM90FP8FP4Gemm1D1DRuntime::generate(args);
        const auto runtime = compiler->build("sm90_m_grouped_fp8_fp4_gemm_contiguous_1d1d_fused", code);
        SM90FP8FP4Gemm1D1DRuntime::launch(runtime, args);
        return;
    }

    auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.first.scalar_type(),
        .b_dtype = torch::kFloat8_e4m3fn,
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K,
        .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n,
        .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    auto rebuild_config = [&](Layout layout) {
        const auto storage_config = SM90ArchSpec::get_storage_config(desc, layout);
        const auto pipeline_config = SM90ArchSpec::get_pipeline_config(desc, layout, storage_config);
        DG_HOST_ASSERT(pipeline_config.num_stages >= 3);
        const auto launch_config = SM90ArchSpec::get_launch_config(desc, layout);
        return GemmConfig{layout, storage_config, pipeline_config, launch_config};
    };

    const auto layout_candidates = SM90ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = SM90ArchSpec::get_layout_info(desc, layout);
    bool found_layout = false;
    // FP4 B is decoded by each CTA, so only A multicast (cluster_n) is enabled.
    for (const auto& candidate: layout_candidates) {
        if (candidate.cluster_m != 1) {
            continue;
        }
        const auto candidate_info = SM90ArchSpec::get_layout_info(desc, candidate);
        if (not found_layout or SM90ArchSpec::compare(candidate_info, layout_info)) {
            layout = candidate;
            layout_info = candidate_info;
            found_layout = true;
        }
    }
    DG_HOST_ASSERT(found_layout);
    // Psum grouped_layout is encoded with 128-row alignment. Keep BLOCK_M fixed
    // at 128 so scheduler psum boundaries match the physical layout.
    if (use_psum_layout) {
        layout.block_m = 128;
        layout.cluster_m = 1;
    }
    if (not use_psum_layout and n >= 1024 and num_groups > 0 and m % num_groups == 0) {
        const int expected_m_per_group = m / num_groups;
        if (expected_m_per_group >= 256) {
            layout.block_m = 256;
            layout.block_n = 64;
        } else if (expected_m_per_group == 128) {
            // Tuned on the SM90 FP8xFP4 contiguous fallback benchmark at
            // N=4096,K=7168. Earlier code carved out BN=128 for 16/24 groups,
            // but this consistently underperformed BN=64 (16g:0.74x, 24g:0.78x
            // vs the BN=64 baseline). Larger BN doubles the per-stage packed-B
            // and SFB cache footprint, which forces `num_stages` down without
            // recovering wave utilization, so always pick BN=64 here.
            layout.block_m = 128;
            layout.block_n = 64;
        }
    }
    layout.cluster_m = 1;
    auto config = rebuild_config(layout);

    // The contiguous fallback benchmark explicitly sweeps BLOCK_M/BLOCK_N via
    // these overrides; keep this path independent from masked RS heuristics so
    // regressions can be attributed to the selected block shape.
    if (block_m_override or block_n_override) {
        auto layout = config.layout;
        if (block_m_override) {
            DG_HOST_ASSERT(not use_psum_layout or *block_m_override == 128);
            layout.block_m = *block_m_override;
        }
        if (block_n_override) {
            layout.block_n = *block_n_override;
        }
        DG_HOST_ASSERT((layout.block_m == 64 or layout.block_m == 128 or layout.block_m == 256) and layout.block_n % 16 == 0);
        DG_HOST_ASSERT(layout.block_n <= 256);
        layout.cluster_m = 1;
        config = rebuild_config(layout);
    }
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);

    // Re-derive `num_stages` and `smem_size` to account for:
    //   (1) the extra packed-FP4 B staging buffer (BLOCK_N * BLOCK_K / 2 bytes per stage)
    //       — after the A/packed-B barrier merge it is sized at `num_stages` slots
    //       (same as A/SFA), so it folds into the per-stage cost.
    //   (2) the enlarged SFB cache: now stores (shape_k_scales, BLOCK_N) floats per block
    //       so the K loop reads SFB from smem instead of gmem. SFB cache aliases
    //       smem_d (no separate allocation), so we only subtract the original SFB bytes.
    {
        const int block_k = config.layout.block_k;
        const int block_n = config.layout.block_n;
        const int shape_k_scales = ceil_div(static_cast<int>(k), block_k);
        const bool uniform_scale_b = (block_k % block_n == 0);
        const int sfb_old_bytes = align(shape_k_scales * (uniform_scale_b ? 1 : 2) * static_cast<int>(sizeof(float)), 16);
        const int sfb_cache_bytes = align(shape_k_scales * block_n * static_cast<int>(sizeof(float)), 16);
        const int smem_d_bytes = align(config.layout.block_m * block_n * static_cast<int>(sizeof(nv_bfloat16)), 1024);
        const int sfb_extra = (sfb_cache_bytes > smem_d_bytes ? sfb_cache_bytes : 0) - sfb_old_bytes;

        const int packed_per_stage = block_n * (block_k / 2);
        const int smem_a_per_stage = config.storage_config.load_block_m * block_k *
                                     static_cast<int>(c10::elementSize(desc.a_dtype));
        const int smem_b_per_stage = config.storage_config.load_block_n * block_k *
                                     static_cast<int>(c10::elementSize(desc.b_dtype));
        const int smem_sfa_per_stage = align(config.layout.block_m * static_cast<int>(sizeof(float)), 128);
        const int original_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage;
        const int merged_per_stage = original_per_stage + packed_per_stage;
        const int orig_num_stages = config.pipeline_config.num_stages;
        const int smem_extra = config.pipeline_config.smem_size - orig_num_stages * original_per_stage + sfb_extra;

        auto fits = [&](int stages) {
            return smem_extra + stages * merged_per_stage <= SM90ArchSpec::smem_capacity;
        };

        // Packed-FP4 B halves the per-stage B footprint, so the smem freed by
        // the FP4 path can usually accommodate more pipeline stages than the
        // FP8 baseline. Mirror the masked path's logic: try to push `num_stages`
        // up to `kW4DefaultMaxStages` while it still fits, then fall back if
        // the chosen value does not fit (e.g. due to large SFB cache at BN=128).
        constexpr int kW4DefaultMaxStages = 8;
        int chosen_stages = orig_num_stages;
        while (chosen_stages + 1 <= kW4DefaultMaxStages and fits(chosen_stages + 1))
            ++ chosen_stages;
        while (chosen_stages >= 3 and not fits(chosen_stages))
            -- chosen_stages;
        DG_HOST_ASSERT(chosen_stages >= 3);
        config.pipeline_config.num_stages = chosen_stages;
        config.pipeline_config.smem_size = smem_extra + chosen_stages * merged_per_stage;
    }

    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a.first, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k, k, 1,
                                              config.storage_config.swizzle_a_mode);
    // View packed FP4 B as 1-byte FP8 so TMA loads raw packed bytes (no FP4 unpacking).
    const auto b_bytes = b.first.view(torch::kFloat8_e4m3fn);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b_bytes, n, half_k,
                                              config.layout.block_n, config.layout.block_k / 2,
                                              half_k, num_groups, 0);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    const SM90FP8FP4Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .major_sfb = get_major_type_ab(sfb),
        .decode_stub = decode_stub,
        .gmem_b_ptr = b.first.data_ptr(),
        .gmem_d_ptr = d.data_ptr(),
        .sfb = sfb.data_ptr(),
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8FP4Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_m_grouped_fp8_fp4_gemm_contiguous_1d2d_fused", code);
    SM90FP8FP4Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_m_grouped_fp8_fp4_gemm_masked_1d1d_fused(
        const std::pair<torch::Tensor, torch::Tensor>& a,
        const std::pair<torch::Tensor, torch::Tensor>& b,
        const torch::Tensor& d,
        const torch::Tensor& masked_m,
        const int& expected_m,
        const int& gran_k,
        const std::optional<int>& gran_k_a_override,
        const std::optional<int>& gran_k_b_override,
        const std::string& compiled_dims,
        const std::optional<int>& block_m_override,
        const std::optional<int>& block_n_override,
        const bool& decode_stub,
        // INT4-sym (signed [-8, 7]) variant for B. The wire format is shared
        // with packed-FP4 (2 nibbles/byte, kPackedFP4 dtype, fp32 SFB), so
        // the kernel reuses the same TMA descriptors and SFB layout. Only
        // the in-register decode primitive switches to int4_symx4_to_e4m3x4.
        const bool& b_is_int4_sym = false,
        // DSV4 MTP/speculative-verify hint: caller passes masked_m.max() so
        // the host can pick BM matching the hottest group instead of the
        // distribution-average expected_m. Fast-path gating (k32 quad_reduce
        // / direct_load / compact_sched) still keys on expected_m so existing
        // small-M optimizations are preserved. Defaults to expected_m when
        // unset.
        const std::optional<int>& masked_m_max_hint = std::nullopt,
        // active_groups_hint: caller passes count of groups with masked_m > 0
        // (i.e. (masked_m != 0).sum()). Combined with masked_m_max_hint this
        // is enough to estimate "工作量分布" and decide fast-path 是否合适：
        //   * 单热点 / 极少活跃 group → fast-path (BM=32 BN=128) 大胜
        //   * 大量活跃 group + 高 max_m → fan-out (BM 阶梯, BN=256) 取胜
        // 不传时退化为旧行为（仅看 max_hint）。
        const std::optional<int>& active_groups_hint = std::nullopt) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    const int gran_k_a_requested = gran_k_a_override.value_or(gran_k);
    const int gran_k_b_requested = gran_k_b_override.value_or(gran_k);
    DG_HOST_ASSERT(gran_k_a_requested == 128);
    DG_HOST_ASSERT(gran_k_b_requested == 32 or gran_k_b_requested == 128);
    // INT4-sym path-A is restricted to per-128 fp32 SFB on the device side.
    // Reject combinations that would otherwise silently bypass the
    // INT4-decode dispatch (e.g. per-32 K-block scales fall into the fused
    // decode path, which is not wired for INT4 yet).
    DG_HOST_ASSERT(not b_is_int4_sym or gran_k_b_requested == 128);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(b.first.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(b.second.scalar_type() == torch::kFloat or
                   b.second.scalar_type() == torch::kInt or
                   b.second.scalar_type() == torch::kBFloat16 or
                   b.second.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt and masked_m.is_contiguous());
    DG_HOST_ASSERT(a.first.is_contiguous() and b.first.is_contiguous() and d.is_contiguous());

    const auto [num_groups, m, k] = get_shape<3>(a.first);
    const auto [num_groups_b, n, half_k] = get_shape<3>(b.first);
    const auto [num_groups_d, m_d, n_d] = get_shape<3>(d);
    DG_HOST_ASSERT(k % 2 == 0 and half_k * 2 == k);
    DG_HOST_ASSERT(num_groups == num_groups_b and num_groups == num_groups_d);
    DG_HOST_ASSERT(masked_m.numel() == num_groups);
    DG_HOST_ASSERT(m == m_d and n == n_d);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.second.size(0) == num_groups and a.second.size(1) == m and
                   a.second.size(2) == ceil_div(k, gran_k_a_requested));
    const int gran_k_b_shape = b.second.scalar_type() == torch::kInt ?
        gran_k_b_requested * 4 : gran_k_b_requested;
    DG_HOST_ASSERT(b.second.size(0) == num_groups and b.second.size(1) == n and
                   b.second.size(2) == ceil_div(k, gran_k_b_shape));

    std::optional<std::tuple<int, int, int>> recipe = std::nullopt;
    std::optional<std::tuple<int, int>> recipe_a = std::make_tuple(1, gran_k_a_requested);
    std::optional<std::tuple<int, int>> recipe_b = std::make_tuple(1, gran_k_b_requested);
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b,
        num_groups, num_groups, false);
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == gran_k_b_requested);

    auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.first.scalar_type(),
        .b_dtype = torch::kFloat8_e4m3fn,
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K,
        .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m,
        .expected_n = n,
        .expected_k = k,
        .expected_num_groups = num_groups
    };
    auto rebuild_config = [&](Layout layout) {
        const auto storage_config = SM90ArchSpec::get_storage_config(desc, layout);
        const auto pipeline_config = SM90ArchSpec::get_pipeline_config(desc, layout, storage_config);
        DG_HOST_ASSERT(pipeline_config.num_stages >= 3);
        const auto launch_config = SM90ArchSpec::get_launch_config(desc, layout);
        return GemmConfig{layout, storage_config, pipeline_config, launch_config};
    };

    const auto layout_candidates = SM90ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = SM90ArchSpec::get_layout_info(desc, layout);
    bool found_layout = false;
    for (const auto& candidate: layout_candidates) {
        if (candidate.cluster_m != 1) {
            continue;
        }
        const auto candidate_info = SM90ArchSpec::get_layout_info(desc, candidate);
        if (not found_layout or SM90ArchSpec::compare(candidate_info, layout_info)) {
            layout = candidate;
            layout_info = candidate_info;
            found_layout = true;
        }
    }
    DG_HOST_ASSERT(found_layout);
    const auto env_enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr and value[0] != '\0' and value[0] != '0';
    };
    const auto env_disabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr and value[0] == '0';
    };
    const auto env_int = [](const char* name, int default_value) {
        const char* value = std::getenv(name);
        return (value != nullptr and value[0] != '\0') ? std::atoi(value) : default_value;
    };
    // BM=64 fast-path 是否启用（函数作用域统一判据，供三处共用：layout 选择 /
    // bm32_skew_layout(stages) / bm32_skew_fast_path(device 三件套总闸)）。
    //   两个触发源：
    //   (1) DG_W4_PATHB_BM64_FASTPATH=1：手动强制（对照实验用）。
    //   (2) DG_W4_PATHB_BM64_AUTO=1（默认开）+ 高置信子集 max_hint>=128 且
    //       active<=8：自动触发，抓“少数活跃组 + 大 max_m”（MTP verify spec-len
    //       命中形态）。详见 layout 选择处 bm64_fastpath 的长注释。
    //   (3) DG_W4_PATHB_BM64_FORCE_ALL=1：无条件全部走 BM=64 fast-path
    //       （用于在线业务峰值吞吐 A/B 对照，绕开 hint 判据；hint 在生产 cuda graph
    //       下恒为 None 时，是验证“假设拿到 hint 全命中”的速度上界）。
    //   注意：必须同时让 stages / fast-path 总闸认账，否则 layout.block_m=64 却
    //   走通用慢路径会退化 ~2x（见 DG_W4_PATHB_BM64 非 fastpath 的坑）。
    const bool bm64_fastpath_enabled =
        env_int("DG_W4_PATHB_BM64_FORCE_ALL", 0) != 0 or
        env_int("DG_W4_PATHB_BM64_FASTPATH", 0) != 0 or
        (env_int("DG_W4_PATHB_BM64_AUTO", 1) != 0 and
         masked_m_max_hint.value_or(0) >= 128 and
         active_groups_hint.value_or(static_cast<int>(desc.num_groups)) <= 8);
    // W4 masked 启发式：以 weight HBM 带宽为主要瓶颈，需要足够的 pipeline stages
    // 来隐藏 TMA B 的延迟。在 expected_m 较小时（典型 MoE 场景），优先选择能让
    // stages 数最深的 (BM, BN) 组合，并兼顾 wave 利用率。
    //
    // 参数空间：BM ∈ {8, 16, 32, 64, 128}（BM<64 用于 masked small-M），
    //          BN ∈ {64, 128, 256}。
    //
    // 选择目标：在 (waves <= ceil_div(total_tiles, num_sms)) 的前提下最大化 stages，
    //         其次最大化 last_wave 利用率，最后倾向更小的 per-stage（即更小 BN）。
    if (not block_m_override and not block_n_override) {
        const int num_sms = desc.num_sms;
        const int block_k = layout.block_k;
        const int shape_k_scales_b = ceil_div(static_cast<int>(k), gran_k_b);

        auto eval_layout = [&](int bm, int bn) -> std::tuple<int, int, int, int> {
            // 返回 (sat_stages, -waves, last_wave_util, -per_stage)，越大越好。
            // stages 在 ~6 之上对 TMA 隐藏几乎饱和，因此用饱和 stage 数比较，避免
            // 小 BN 因 stages=8 击败 wave 利用率更高的候选。
            const int tiles = ceil_div(expected_m, bm) * ceil_div(static_cast<int>(n), bn) * num_groups;
            const int waves = ceil_div(tiles, num_sms);
            const int last = tiles - (waves - 1) * num_sms;
            const int last_util = last <= 0 ? num_sms : last;
            const bool uniform_scale_b = (block_k % bn == 0);
            const int sfb_old_bytes = gran_k_b == 32 ? 0 :
                align(shape_k_scales_b * (uniform_scale_b ? 1 : 2) * static_cast<int>(sizeof(float)), 16);
            const int sfb_cache_bytes = gran_k_b == 32 ? 0 :
                align(shape_k_scales_b * bn * static_cast<int>(sizeof(float)), 16);
            const int rs_padded_bm = std::max(bm, 64);
            const int smem_d_bytes =
                align(rs_padded_bm * bn * static_cast<int>(sizeof(nv_bfloat16)), 1024);
            const int sfb_extra = (sfb_cache_bytes > smem_d_bytes ? sfb_cache_bytes : 0) - sfb_old_bytes;
            const int smem_a_per_stage = rs_padded_bm * block_k * static_cast<int>(c10::elementSize(desc.a_dtype));
            const int smem_sfa_per_stage =
                align(rs_padded_bm * static_cast<int>(sizeof(float)), 128);
            const int packed_per_stage = bn * (block_k / 2);
            const int merged_per_stage = smem_a_per_stage + smem_sfa_per_stage + packed_per_stage;
            constexpr int kMaxEvaluatedStages = 10;
            constexpr int kBarrierBytes = 16 * kMaxEvaluatedStages * 2;
            const int fixed = smem_d_bytes + kBarrierBytes + sfb_extra;
            const int max_stages = (SM90ArchSpec::smem_capacity - fixed) / merged_per_stage;
            constexpr int kStageSaturation = 6;
            const int sat_stages = std::min(std::min(max_stages, kMaxEvaluatedStages), kStageSaturation);
            return std::make_tuple(sat_stages, -waves, last_util, -merged_per_stage);
        };

        std::vector<std::pair<int, int>> w4_candidates = {
            {64, 64}, {64, 128}, {64, 256},
            {128, 64}, {128, 128},
        };
        if (expected_m <= 32) {
            w4_candidates.insert(w4_candidates.begin(), {{8, 64}, {16, 64}, {32, 64}});
        }

        std::pair<int, int> best{layout.block_m, layout.block_n};
        std::tuple<int, int, int, int> best_score{-1, 0, 0, 0};
        bool first = true;
        for (const auto& cand : w4_candidates) {
            const int bm = cand.first;
            const int bn = cand.second;
            // 1D2D 内核 unroll 要求
            if (bn > block_k and (bn % (bn - block_k) != 0 and block_k % (bn - block_k) != 0))
                continue;
            // masked 路径 multicast 合法性：当前固定 cluster_m=1, cluster_n=1，恒满足
            const auto score = eval_layout(bm, bn);
            if (std::get<0>(score) < 3)
                continue;
            if (first or score > best_score) {
                best_score = score;
                best = cand;
                first = false;
            }
        }
        layout.block_m = best.first;
        layout.block_n = best.second;

        // RS masked W4 empirical fallback:
        // pick BM close to expected_m to avoid over-computing promotion work,
        // and use BN=256 for fewer CTAs while staying within Hopper TMA limits.
        // bm_select_m 用 max(expected_m, masked_m_max_hint) —— 当调用方传入
        // hot group 的真实大小时，host 据此选 BM，避免 BM 与 hot group 严重
        // 失配；fast-path gating（k32 quad_reduce / direct_load 等）下面仍按
        // expected_m 判断，保留 small-m 路径的优化。
        const int bm_select_m = std::max(
            expected_m, masked_m_max_hint.value_or(expected_m));
        if (desc.gemm_type == GemmType::MGroupedMasked) {
            // 公共形状判定：DSV4 MTP/speculative verify 形状下
            // host scheduler 才会做 hard-code 干预，避开通用 shape。
            // A1+A2: 放开到 N∈{4096,7168}, K∈{2048,3072,4096,7168} 以覆盖 wide_n / wide_k 系列。
            // K=3072 引入动机：DSV4 gateup/down_proj 还有一组 n=7168+k=3072 的真实
            //   shape，K=3072 满足 BLOCK_K=128 整除 (3072/128=24) + gran_k_b=32 整除
            //   (3072/32=96) ⇒ device fast-path (quad_reduce/direct_load/compact_sched)
            //   物理可命中。原守护把 k=3072 拒在外面，强制走 path-B 通用 cache_sfb_k32
            //   路径，W4 GB/s 只有 ~600（vs fast-path 命中时 941+ GB/s）。
            // FAST_PATH_RELAX 扩展（DG_W4_PATHB_FAST_PATH_RELAX=1 **默认开启**）：
            //   把守护放宽到 num_groups>=8 + n∈{4096,6144,7168} + k∈{2048,3072,4096,7168}。
            //   动机：DSV4 down_proj 真实形状是 g24, n=6144, k=7168，原守护把它
            //   挡在 fast-path 外，强制走 path-B 通用 cache_sfb_k32 路径，W4 GB/s
            //   只有 627（vs fast-path 命中时 1500-3000+ GB/s），speedup 0.43-0.56x。
            //   device fast-path 三件套（quad_reduce / direct_load / compact_sched）
            //   实际 BN-agnostic，仅硬约束 BM==32 + gran_k_b==32，冗余专家场景
            //   kNumGroups<=36 也可按同一路径处理。
            //   实测：g24+n=6144+k=7168 整张表 0.43-0.56x → 0.71-0.81x；
            //         g24+n=7168+k=3072 整张表 0.30-0.54x → 0.45-0.79x；
            //         默认 dsv4 形状 (g32+n∈{4096,7168}+k∈{2048,4096,7168}) 不变。
            //   开启后所有依赖 dsv4_shape 的路径（BN256 门槛 / small_hot
            //   / large_bm / cluster_n / bm64_hint）都同步放开，已验证无退化。
            //   设 DG_W4_PATHB_FAST_PATH_RELAX=0 可显式回退到原守护做对照。
            const bool fast_path_relax =
                env_int("DG_W4_PATHB_FAST_PATH_RELAX", 1) != 0;
            const bool dsv4_shape =
                (static_cast<int64_t>(desc.num_groups) >= 8 and
                 (static_cast<int64_t>(desc.n) == 4096 or
                  static_cast<int64_t>(desc.n) == 7168) and
                 (static_cast<int64_t>(desc.k) == 2048 or
                  static_cast<int64_t>(desc.k) == 3072 or
                  static_cast<int64_t>(desc.k) == 4096 or
                  static_cast<int64_t>(desc.k) == 7168))
                or
                (fast_path_relax and
                 static_cast<int64_t>(desc.num_groups) >= 8 and
                 (static_cast<int64_t>(desc.n) == 4096 or
                  static_cast<int64_t>(desc.n) == 6144 or
                  static_cast<int64_t>(desc.n) == 7168) and
                 (static_cast<int64_t>(desc.k) == 2048 or
                  static_cast<int64_t>(desc.k) == 3072 or
                  static_cast<int64_t>(desc.k) == 4096 or
                  static_cast<int64_t>(desc.k) == 7168));

            if (gran_k_b == 32) {
                // path-B：device fast-path（BM=32 + quad_reduce + direct_load）
                // 是 gran_k_b=32 下的唯一高效形状。BN 选取根据 hint 大小：
                //   * hint <= 32（小 hot）：BM=32 BN=128（短 wgmma 流水线 + 高 stage 数）
                //   * hint >  32（大 hot）：BM=32 BN=256（grid 减半，scheduler 调度密度提升）
                //
                // 历史教训（skewed masked benchmark 实测）：
                //   * 试图基于 (active, max_m) 把"path-B 输给 FP8"的 case 切到
                //     BM 阶梯 + BN=256 fan-out → device kernel 落入 path-B 通用
                //     cache_sfb_k32 路径，比 fast-path 慢 1.5–3x（如 eight_hot_64
                //     190→298us、dense_tail 368→1011us）。
                //   * 结论：path-B 必须保持 BM=32 + fast-path 守护命中，否则崩盘。
                //     BN=128↔256 两挡都在 device fast-path 守护内（参见 device
                //     kernel kK32QuadReduce 路径，BN-agnostic：compute_n_0/1
                //     寻址含 m_offset = local_idx * WAVE_BLOCK_M，
                //     load_k32_quad_scale_b 按 compute_n 单独寻址）。
                //
                // hint<=16（纯 fan-out）保留原 BM 阶梯走 BN=256 的 path-B 通用
                // 路径——这部分 device 守护本来就不命中。
                // real_hot_present 判定：caller 是否真的传入 max_hint > 16？
                //   * has_value=true：按真实 max_hint 判断（>16 才认为 hot 存在）。
                //   * has_value=false（caller 没传 hint）：
                //       - expected_m > 16：保留旧行为（保守认为 hot 存在），
                //         走 BM=32 fast-path（m>=1024 时命中 fast-path 守护）。
                //       - expected_m <= 16：**新行为**，认为 hot 不存在，让 case
                //         落到 expected_m candidate 路径（BM∈{8,16,32} 自动选 +
                //         small_m_simple_sched 命中）。
                //   动机：DSV4 EP 业务真实 shape 大量出现 expected_m=1/2/3 +
                //     max_hint=None + m=256 的情形。旧逻辑强制 BM=32，但
                //     bm32_skew_fast_path 守护要 m>=1024，m=256 时 fast-path 不
                //     命中反而走通用 cache_sfb_k32 路径，padding 浪费
                //     (32-expected_m)/32 ≈ 94%。落到 expected_m candidate 后
                //     device 端 BLOCK_M<=16 命中 kUseSmallMSimpleSched，物理零浪费。
                const bool real_hot_present =
                    masked_m_max_hint.has_value()
                        ? (masked_m_max_hint.value() > 16)
                        : (expected_m > 16);
                // fuse_decode 路径与 fast-path 互斥：env 开启时强制走通用 BM
                // 阶梯（device 端 kFuseScaleBDecode 走 path-B 通用 cache_sfb_k32，
                // 而 fast-path layout 假设 quad_reduce + direct_load）。
                const bool fuse_decode_hint =
                    env_int("DG_W4_PATHB_FUSE_DECODE", 0) != 0;
                // BM=64 实验性路径（默认关，env DG_W4_PATHB_BM64=1）：
                //   想法：path-B swap_AB 下 wgmma 物理 N = BLOCK_M。BM=64 → wgmma
                //         64x64x32，单次 wgmma 内 b_decoded 服务 64 cols 而非 32，
                //         W4 解码 (prmt+lop3) 成本天然摊薄 2x，弥补 fast-path
                //         (quad_reduce + direct_load) 关闭带来的 sfb load + fmul
                //         overhead。同时 grid 减半。
                //   代价：device 端 kK32QuadReduce 假设 BM=32 (compute_n_0/1 只覆盖
                //         32 cols)，BM=64 必须走通用 cache_sfb_k32 + fp32 sfb 路径。
                //   实测结果（DG_W4_PATHB_BM64=1 vs 默认 fast-path）：
                //     gateup_dense_tail_hot64    356us → 748us  (2.1x 退化)
                //     gateup_mtp_dp4             158us → 275us  (1.74x 退化)
                //     gateup_eight_hot_64        188us → 226us  (1.20x 退化)
                //     gateup_one_hot_32           31us →  47us  (1.52x 退化)
                //     绝大多数 case 退化 1.2-2.1x，无一受益。
                //   退化根因：cache_sfb_k32 阶段 SMEM build LUT + register fmul + barrier
                //     比 fast-path 直读 gmem + quad_reduce 慢得多；BM=64 grid 砍半
                //     反而让单 SM 排队工作量翻倍，stages 不够掩 K-loop（GB/s 从
                //     1818 跌到 452）。decode 复用收益被 sfb pipeline 重建吃光。
                //   保留 env 入口仅做对照实验，生产环境永不开启。
                const bool bm64_hint =
                    dsv4_shape and env_int("DG_W4_PATHB_BM64", 0) != 0;
                // BM=64 fast-path 实验路径（默认关，env DG_W4_PATHB_BM64_FASTPATH=1）：
                //   与 DG_W4_PATHB_BM64 不同——后者刻意让 device 落回通用
                //   cache_sfb_k32 路径；本开关**保持** fast-path 三件套
                //   (kK32QuadReduce / kScaleBDirectLoad / kCompactMaskedSched) 开启，
                //   仅把 host BM 从 32 抬到 64，验证“weight 解码次数减半 + grid 减半”
                //   能否吃掉 max_m>=64 差场景。
                //   依赖：device 端 swap_AB 下 WGMMA N=BLOCK_M=64 指令已存在
                //   (FP8MMASelectorRS<64> = MMA_64x64x32_RS_TN)，scale 寻址 BM-agnostic。
                //   BM=64 强制 BN=128，避免 A-smem 叠加 BN=256 把 stages 压崩。
                //
                //   实测结论（DG_W4_PATHB_BM64_FASTPATH=1 vs 默认 BM=32，ncu 对照，
                //   g24 N=6144 K=7168 / N=7168 K=3072 skew benchmark）：**全面退化，已证伪**。
                //     uniform_17   784us/0.75x → 1391us/0.43x  (max_m=17)
                //     uniform_32   775us/0.77x → 1387us/0.42x  (max_m=32)
                //     dense_tail_hot17 767/0.77x → 1381/0.43x
                //     uniform_64  1541us/0.38x → 1390us/0.43x  (唯一微升)
                //     one_hot_512  542us/0.58x → 542us/0.58x   (完全无变化)
                //   ncu 四项硬指标 BM32 vs BM64 **逐字节相同**：
                //     Registers/Thread 168=168, Dyn SMEM 149984=149984,
                //     Grid (78,1,1)=(78,1,1), Achieved Occupancy 14.0%≈13.9%。
                //   证伪根因（结构性，非实现 bug；正确性 OK diff=0、无 spill）：
                //     1. persistent kernel grid 恒 = num_sms，BLOCK_M 不影响 grid。
                //     2. swap_AB 下 WGMMA 形状 / kNumAccum / WAVE_BLOCK_M 全由
                //        BLOCK_N 决定，BLOCK_M 加倍不动寄存器、不动 smem。
                //     3. M-tile 数 = ceil(max_m/BM)。DSV4 真实 max_m 多 <=32，
                //        BM 32→64 时 tile 数不减，但每 tile WGMMA 覆盖 64 行、
                //        ~47 行是 padding 空算 → 工作量翻倍、收益为零 → 耗时近翻倍。
                //     4. 仅 max_m=64 时 tile 2→1 抵消 padding 才微升；max_m 更大
                //        (one_hot_512 active=1) 瓶颈不在 tile 数，故毫无变化。
                //   结论：当前 DSV4 负载 (max_m 集中 17-64) 下 BM=64 是负优化。
                //   保留 env 入口仅做对照实验，生产环境永不开启。
                //   真正受益方向见下方思路 2（active=1 大 max_m 时跨 m_block 复用
                //   weight decode），不受 padding 问题影响。
                // BM=64 fast-path **自动触发规则**（默认开，env 可关）：
                //   前述 DG_W4_PATHB_BM64_FASTPATH 手动实验证明，BM 32→64 让
                //   M-tile 数 ceil(m/32)→ceil(m/64) 减半，仅当“每活跃组平均
                //   token 数”够大（每组 m>=64，padding 占比可忽略）才净赚；组小则
                //   tile 数不减、padding 翻倍纯亏。判别量本应是 sum_m/active，但
                //   host API 只有 max_hint / active（拿不到 sum_m，且不能同步读
                //   device 上的 masked_m）。
                //   折中：用高置信子集 `max_hint>=128 且 active<=8`（少数活跃组 +
                //   大 max_m，正是 MTP verify 的 spec-len 命中形态）触发，刻意避开
                //   max_m=64 / 多活跃组的模糊带（uniform_64 赢、dense_tail_hot64
                //   亏，二者 host 信号 (max_hint=64,active=24) 撞车无法区分）。
                //   实测（g24 N6144/7168 与 g32 N4096，BM64+pair vs 默认 quad）两族
                //   shape 一致：命中子集 one_hot_128/384、mtp_384/512_hot、
                //   one_hot_512、mtp_768_hot512 等普降 5~26%，被避开的
                //   dense_tail/uniform/eight_hot 无一受损。
                //   BM=64 下 quad 4 累加器会 spill 爆炸，故命中时**必须**配
                //   pair_reduce（见下方 k32_pair_reduce 联动）。
                const bool bm64_fastpath = dsv4_shape and bm64_fastpath_enabled;
                if (real_hot_present and not fuse_decode_hint and
                    not bm64_hint and
                    env_int("DG_W4_PATHB_FAST_PATH", 1) != 0) {
                    layout.block_m = bm64_fastpath ? 64 : 32;
                    // H1: 同时满足 max_m > 32 + active*max_m >= 1024 时走 BN=256，
                    // grid 减半。门槛公式来自 skewed masked benchmark 实测：
                    //   * eight_hot_64 (8×64=512)、four_hot_64 (4×64=256)、
                    //     one_hot_384 (1×384=384) 在 BN=256 下 grid 太稀，
                    //     单 tile WAVE_WGMMA=2 的两次 wgmma 反而拖慢 8-9%。
                    //   * mtp_dp4 (8×214=1712)、dense_tail_hot64 a32
                    //     (32×64=2048)、mtp_512_hot384 (8×384=3072) 等
                    //     总工作量充足的 case 才能从 BN=256 grid 减半中受益。
                    // 需 dsv4_shape 才能命中 device fast-path 守护
                    // (N=4096, K∈{4096,2048}, groups=32)，否则 BN=256 落到
                    // path-B 通用 cache_sfb_k32 路径反而崩盘。
                    const int hint_m = masked_m_max_hint.value_or(0);
                    // 缺省 active 估计为 num_groups 的一半（保守，避免误升 BN=256）
                    const int hint_active = active_groups_hint.value_or(
                        static_cast<int>(desc.num_groups) / 2);
                    // 总工作量估计（FLOPS 代理）：active × max_m × N × K。
                    // 用 int64 防溢出，N/K 单位 elements。
                    const int64_t hint_workload =
                        static_cast<int64_t>(hint_active) * hint_m *
                        static_cast<int64_t>(desc.n) *
                        static_cast<int64_t>(desc.k);
                    // 基础门槛：g32 + N=4096 + K=4096 实测 hint_active*hint_m >= 1024
                    // 且 hint_m > 32 时 BN=256 受益。
                    const bool bn256_baseline =
                        hint_m > 32 and
                        static_cast<int64_t>(hint_active) * hint_m >= 1024;
                    // 大 N×K 形态扩展（实验性，env DG_W4_PATHB_BN256_BIG=1 默认关）：
                    //   动机：DSV4 down_proj real shape g24, N=6144, K=7168 下，
                    //     uniform_17/24/32 + dense_tail_hot17 全被 hint_m > 32
                    //     卡住走 BN=128 (~941 GB/s)，理论上 grid 4-9 wave 已脱离
                    //     "grid 太稀"危险区，单 tile work 翻倍应可吞。
                    //   阈值原本设计：active>=8 + workload>=1.5e10。
                    //   实测结果（DG_W4_PATHB_BN256_BIG=1 vs 默认 BN=128）：
                    //     uniform_17        711us → 797us  (-12%, 941→838 GB/s)
                    //     uniform_24        716us → 799us  (-12%)
                    //     uniform_32        705us → 790us  (-12%)
                    //     dense_tail_hot17  701us → 786us  (-12%)
                    //   退化根因：K=7168 + BN=256 + BM=32 + stages=8 ⇒ SMEM B-tile
                    //     占用翻倍 (256 cols × 32 K × 8 stages = 64KB)，超出 228KB
                    //     上限被 JIT 自动降到 stages=4，K-loop 掩蔽崩盘 (GB/s
                    //     941→842 整齐下跌 11%)。BN=256 grid 减半的收益 << stages
                    //     减半的带宽损失。
                    //   保留 env 入口仅做对照实验，生产环境永不开启。
                    constexpr int64_t kBN256BigWorkloadThreshold = 15'000'000'000LL;
                    const bool bn256_big_shape =
                        env_int("DG_W4_PATHB_BN256_BIG", 0) != 0 and
                        hint_workload >= kBN256BigWorkloadThreshold and
                        static_cast<int64_t>(hint_active) >= 8;
                    const bool bn256_eligible =
                        dsv4_shape and
                        masked_m_max_hint.has_value() and
                        (bn256_baseline or bn256_big_shape) and
                        env_int("DG_W4_PATHB_BN256", 1) != 0;
                    // BM=64 fast-path 强制 BN=128：A-smem 已随 BM 翻倍，再叠加
                    // BN=256 会把 B-tile + A-tile 占用一起推高，stages 被压崩。
                    layout.block_n = (bm64_fastpath or not bn256_eligible) ? 128 : 256;
                    // H2: cluster_n=2（A 多播）实测**全军退化**，默认关闭。
                    // 退化数据（DG_W4_PATHB_CLUSTER_N=1 vs 默认）：
                    //   BN=256 路径 ~9-10% 退化（mtp_512: 201→220, dense_tail_hot64:
                    //   354→388, mtp_dp4: 158→172）
                    //   BN=128 路径 ~3-12% 退化（four_hot_64: 95→106,
                    //   longtail_128: 157→169）
                    // 退化根因：
                    //   1. compact_masked_sched 按 (group, n_block) 调度，cluster=2
                    //      把相邻 n_block 绑定到同 cluster，但 masked 下相邻
                    //      n_block 可能属于不同 group，A tile 复用率被打断。
                    //   2. cluster barrier 同步 + cluster launch stagger 对 small
                    //      grid（28-56 个 tile）开销大于多播收益。
                    //   3. W4 的 sfb 不参与 multicast，同步开销均摊到两个 CTA。
                    // 保留 env 入口仅做对照实验，生产环境永不开启。
                    if (dsv4_shape and
                        env_int("DG_W4_PATHB_CLUSTER_N", 0) != 0 and
                        static_cast<int64_t>(desc.n) %
                            (static_cast<int64_t>(layout.block_n) * 2) == 0 and
                        desc.num_sms % 2 == 0) {
                        layout.cluster_n = 2;
                    }
                } else if (bm64_hint and real_hot_present) {
                    // BM=64 实验路径：仅 dsv4_shape + DG_W4_PATHB_BM64=1 时进入。
                    // 走 path-B 通用 cache_sfb_k32 路径，依赖下面 fast-path 守护
                    // 自动关闭 (kK32QuadReduce / kScaleBDirectLoad / kCompactMaskedSched
                    // 通过 bm32_skew_fast_path 的 BM==32 检查失败而被关闭)。
                    layout.block_m = 64;
                    // BN=128 默认：BM=64 + BN=128 → wgmma 64x64x32 × 2 wgmma per
                    // K_block，b_decoded 在每次 wgmma 内部服务 64 cols。
                    // env DG_W4_PATHB_BM64_BN=256 时尝试更大 grid 减半。
                    const int bm64_bn_override = env_int("DG_W4_PATHB_BM64_BN", 0);
                    layout.block_n = (bm64_bn_override == 256) ? 256 : 128;
                } else {
                    // 纯 fan-out（hint<=16）：base 阶梯，BN=256
                    if (bm_select_m <= 8) layout.block_m = 8;
                    else if (bm_select_m <= 16) layout.block_m = 16;
                    else if (bm_select_m <= 32) layout.block_m = 32;
                    else if (bm_select_m <= 64) layout.block_m = 64;
                    layout.block_n = 256;
                }
            } else {
                // path-A：cooperative prefetch + sfb→smem，BM 阶梯有效。
                if (bm_select_m <= 8) layout.block_m = 8;
                else if (bm_select_m <= 16) layout.block_m = 16;
                else if (bm_select_m <= 32) layout.block_m = 32;
                else if (bm_select_m <= 64) layout.block_m = 64;
                layout.block_n = 256;

                // DSV4 + 大 BM：BN=256→128 减小单 tile promote/store 链长，
                // 让 grid 上 n-tile 翻倍提升 SM 间负载均衡。
                if (layout.block_m >= 64 and dsv4_shape and
                    env_int("DG_W4_LARGE_BM_BN128", 1) != 0) {
                    layout.block_n = 128;
                }
                // DSV4 + small hot (bm_select∈(32,64])：BM=64 padding 浪费太大，
                // 改 BM=32 BN=128 兼顾 hot/small group。
                if (bm_select_m > 32 and bm_select_m <= 64 and dsv4_shape and
                    env_int("DG_W4_SMALL_HOT_BM32", 1) != 0) {
                    layout.block_m = 32;
                    layout.block_n = 128;
                }
            }
            // 历史经验注记：
            //  * 曾尝试 path-B BM=32→64 升级（hot hint），device kernel 的
            //    kK32QuadReduce 在 BM>=64 + swap_AB 下 sfb 寻址 / 4-wgmma fence
            //    跟 BM=32 紧耦合，dp4 退化到 0.37x。
            //  * 已知天花板（DSV4 skew, groups=32, N=4096）：进一步提升须从
            //    device 侧扩展 BM=64 quad_reduce，工作量大且收益不确定。
        }
    }
    if (not block_m_override and not block_n_override) {
        // DG_W4_PATHB_BM64_FORCE_ALL=1 时，把 BLOCK_N 强制锁到 128（与 BM=64
        // fast-path 联动）。动机：BM=64 fast-path 下 final_accum 寄存器量
        // ∝ BLOCK_N/64，BN=256 → 4 份 tile 累加器逼出 168reg/thread、occupancy
        // 14%；BN=128 砍半 final_accum，腾出寄存器预算给更多 warp，对大 batch /
        // 高 active 场景 occupancy 上限更高。属于在线吞吐对照实验的联动开关，
        // 不影响 hint 判据自身路径。
        if (env_int("DG_W4_PATHB_BM64_FORCE_ALL", 0) != 0 and
            bm64_fastpath_enabled and layout.block_m == 64) {
            layout.block_n = 128;
        }
        DG_HOST_ASSERT(layout.block_m == 8 or layout.block_m == 16 or layout.block_m == 32 or
                       layout.block_m == 64 or layout.block_m == 128);
        DG_HOST_ASSERT(layout.block_n == 64 or layout.block_n == 128 or layout.block_n == 256);
    }
    layout.cluster_m = 1;
    auto config = rebuild_config(layout);

    if (block_m_override or block_n_override) {
        auto layout = config.layout;
        if (block_m_override) {
            layout.block_m = *block_m_override;
        }
        if (block_n_override) {
            layout.block_n = *block_n_override;
        }
        DG_HOST_ASSERT((layout.block_m == 8 or layout.block_m == 16 or layout.block_m == 32 or
                        layout.block_m == 64 or layout.block_m == 128 or layout.block_m == 256) and layout.block_n % 16 == 0);
        DG_HOST_ASSERT(layout.block_n <= 256);
        layout.cluster_m = 1;
        config = rebuild_config(layout);
    }
    // Packed FP4 B has half the K bytes of FP8 B. Match PR #287's W4 path:
    // TMA writes B with a 64B swizzle and the RS kernel reads it via ldmatrix.
    config.storage_config.swizzle_b_mode = config.layout.block_k / 2;
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k / 2);

    {
        const int block_k = config.layout.block_k;
        const int block_n = config.layout.block_n;
        const int shape_k_scales_b = ceil_div(static_cast<int>(k), gran_k_b);
        const bool uniform_scale_b = (block_k % block_n == 0);
        const int sfb_old_bytes = gran_k_b == 32 ? 0 :
            align(shape_k_scales_b * (uniform_scale_b ? 1 : 2) * static_cast<int>(sizeof(float)), 16);
        const int sfb_cache_bytes = gran_k_b == 32 ? 0 :
            align(shape_k_scales_b * block_n * static_cast<int>(sizeof(float)), 16);
        const int rs_padded_bm = std::max(config.layout.block_m, 64);
        const int base_smem_d_bytes =
            align(config.layout.block_m * block_n * static_cast<int>(sizeof(nv_bfloat16)), 1024);
        const int smem_d_bytes =
            align(rs_padded_bm * block_n * static_cast<int>(sizeof(nv_bfloat16)), 1024);
        const int smem_d_extra = smem_d_bytes - base_smem_d_bytes;
        const int sfb_extra = (sfb_cache_bytes > smem_d_bytes ? sfb_cache_bytes : 0) - sfb_old_bytes;

        const int packed_per_stage = block_n * (block_k / 2);
        const int base_smem_a_per_stage = config.storage_config.load_block_m * block_k *
                                          static_cast<int>(c10::elementSize(desc.a_dtype));
        const int base_smem_b_per_stage = config.storage_config.load_block_n * block_k *
                                          static_cast<int>(c10::elementSize(desc.b_dtype));
        const int base_smem_sfa_per_stage =
            align(config.layout.block_m * static_cast<int>(sizeof(float)), 128);
        const int original_per_stage =
            base_smem_a_per_stage + base_smem_b_per_stage + base_smem_sfa_per_stage;
        const int smem_a_per_stage = rs_padded_bm * block_k * static_cast<int>(c10::elementSize(desc.a_dtype));
        const int smem_sfa_per_stage =
            align(rs_padded_bm * static_cast<int>(sizeof(float)), 128);
        const int merged_per_stage = smem_a_per_stage + smem_sfa_per_stage + packed_per_stage;
        const int orig_num_stages = config.pipeline_config.num_stages;
        const int smem_extra =
            config.pipeline_config.smem_size - orig_num_stages * original_per_stage + smem_d_extra + sfb_extra;

        auto fits = [&](int stages) {
            return smem_extra + stages * merged_per_stage <= SM90ArchSpec::smem_capacity;
        };

        constexpr int kW4DefaultMaxStages = 8;
        int max_fitting = orig_num_stages;
        while (max_fitting + 1 <= kW4DefaultMaxStages and fits(max_fitting + 1))
            ++ max_fitting;
        int chosen_stages = max_fitting;
        while (chosen_stages >= 3 and not fits(chosen_stages))
            -- chosen_stages;
        if (gran_k_b == 32 and expected_m <= 16 and
            static_cast<int64_t>(desc.k) >= 4096 and static_cast<int64_t>(desc.n) <= 4096)
            chosen_stages = std::min(chosen_stages, 6);
        const bool bm32_skew_layout = gran_k_b == 32 and
            (config.layout.block_m == 32 or
             (config.layout.block_m == 64 and bm64_fastpath_enabled)) and
            (config.layout.block_n == 128 or config.layout.block_n == 256) and
            static_cast<int64_t>(desc.m) >= 1024 and
            ((static_cast<int64_t>(desc.num_groups) >= 8 and
              static_cast<int64_t>(desc.num_groups) <= 36 and
              (static_cast<int64_t>(desc.n) == 4096 or
               static_cast<int64_t>(desc.n) == 7168) and
              (static_cast<int64_t>(desc.k) == 2048 or
               static_cast<int64_t>(desc.k) == 3072 or
               static_cast<int64_t>(desc.k) == 4096 or
               static_cast<int64_t>(desc.k) == 7168)) or
             // FAST_PATH_RELAX 扩展（DG_W4_PATHB_FAST_PATH_RELAX=1 **默认开启**）：
             // 放开到 g>=8 + n∈{4096,6144,7168} + k∈{2048,3072,4096,7168}
             (env_int("DG_W4_PATHB_FAST_PATH_RELAX", 1) != 0 and
              static_cast<int64_t>(desc.num_groups) >= 8 and
              static_cast<int64_t>(desc.num_groups) <= 36 and
              (static_cast<int64_t>(desc.n) == 4096 or
               static_cast<int64_t>(desc.n) == 6144 or
               static_cast<int64_t>(desc.n) == 7168) and
              (static_cast<int64_t>(desc.k) == 2048 or
               static_cast<int64_t>(desc.k) == 3072 or
               static_cast<int64_t>(desc.k) == 4096 or
               static_cast<int64_t>(desc.k) == 7168)));
        const int bm32_skew_stage_override = env_int("DG_W4_BM32_SKEW_STAGES", 0);
        if (bm32_skew_layout and bm32_skew_stage_override > 0) {
            DG_HOST_ASSERT(bm32_skew_stage_override >= 3 and bm32_skew_stage_override <= kW4DefaultMaxStages);
            DG_HOST_ASSERT(fits(bm32_skew_stage_override));
            chosen_stages = bm32_skew_stage_override;
        } else if (bm32_skew_layout and fits(kW4DefaultMaxStages)) {
            chosen_stages = kW4DefaultMaxStages;
        }
        DG_HOST_ASSERT(chosen_stages >= 3);
        config.pipeline_config.num_stages = chosen_stages;
        config.pipeline_config.smem_size = smem_extra + chosen_stages * merged_per_stage;
    }

    // R2b-A swap_ab maps original N onto WGMMA M. Use enough math warpgroups
    // to cover the 64-row WGMMA-M strips selected by BLOCK_N.
    DG_HOST_ASSERT(config.layout.block_n == 64 or config.layout.block_n == 128 or config.layout.block_n == 256);
    int rs_num_math_threads = config.layout.block_n <= 64 ? 128 : 256;
    const int rs_num_math_threads_override = env_int("DG_W4_RS_MATH_THREADS", 0);
    if (rs_num_math_threads_override > 0) {
        DG_HOST_ASSERT(rs_num_math_threads_override == 128 or rs_num_math_threads_override == 256);
        rs_num_math_threads = rs_num_math_threads_override;
    }
    config.launch_config.num_math_threads = rs_num_math_threads;
    config.launch_config.num_threads = config.launch_config.num_tma_threads + rs_num_math_threads;

    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a.first, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k, k, num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto b_bytes = b.first.view(torch::kFloat8_e4m3fn);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b_bytes, n, half_k,
                                              config.layout.block_n, config.layout.block_k / 2,
                                              half_k, num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, num_groups, 0);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), num_groups,
                                               config.storage_config.swizzle_cd_mode);

    const bool bm32_skew_fast_path = gran_k_b == 32 and
        (config.layout.block_m == 32 or
         (config.layout.block_m == 64 and bm64_fastpath_enabled)) and
        (config.layout.block_n == 128 or config.layout.block_n == 256) and
        static_cast<int64_t>(desc.m) >= 1024 and
        ((static_cast<int64_t>(desc.num_groups) >= 8 and
          static_cast<int64_t>(desc.num_groups) <= 36 and
          (static_cast<int64_t>(desc.n) == 4096 or
           static_cast<int64_t>(desc.n) == 7168) and
          (static_cast<int64_t>(desc.k) == 2048 or
           static_cast<int64_t>(desc.k) == 3072 or
           static_cast<int64_t>(desc.k) == 4096 or
           static_cast<int64_t>(desc.k) == 7168)) or
         // FAST_PATH_RELAX 扩展（DG_W4_PATHB_FAST_PATH_RELAX=1 **默认开启**），
         // 与 bm32_skew_layout / dsv4_shape 同步生效。device fast-path 三件套
         // (quad_reduce / direct_load / compact_sched) 仅硬约束 BM==32，g<=36
         // 冗余专家场景按同一路径处理，
         // 与 N/K 数值无关，物理可命中。
         (env_int("DG_W4_PATHB_FAST_PATH_RELAX", 1) != 0 and
          static_cast<int64_t>(desc.num_groups) >= 8 and
          static_cast<int64_t>(desc.num_groups) <= 36 and
          (static_cast<int64_t>(desc.n) == 4096 or
           static_cast<int64_t>(desc.n) == 6144 or
           static_cast<int64_t>(desc.n) == 7168) and
          (static_cast<int64_t>(desc.k) == 2048 or
           static_cast<int64_t>(desc.k) == 3072 or
           static_cast<int64_t>(desc.k) == 4096 or
           static_cast<int64_t>(desc.k) == 7168)));
    // Fused-decode 路径（path-B 通用 cache_sfb_k32 + LUT decode）：
    //   * scale 在 cache 阶段编进 LUT，wgmma 后省一次 fmul（按 quad 算 = 4 次）
    //   * sfb 4x packed UE8M0：每 4 个 e8m0 打成 1 个 int32，体积 = fp32 的 1/4
    // 与 fast-path（kK32QuadReduce + kScaleBDirectLoad）**互斥**（cuh 强制
    // not kScaleBDirectLoad），开启后强制走通用路径。需要 sfb 是 packed int 布局。
    // 默认关，由 DG_W4_PATHB_FUSE_DECODE=1 显式开启。
    const bool fuse_scale_b_decode =
        gran_k_b == 32 and
        sfb.scalar_type() == torch::kInt and
        get_major_type_ab(sfb) == cute::UMMA::Major::MN and
        env_int("DG_W4_PATHB_FUSE_DECODE", 0) != 0;
    const bool scale_b_packed_ue8m0 = fuse_scale_b_decode;
    // 思路2 性能探针：把 scale_a 从 promote 内移除（device kScaleAStub 置 scale_a=1
    // 并跳过 SFA 的 TMA load），用于测 fuse_scale_b_decode 单累加器 + scale_a 后移到
    // silu_and_mul 的速度上界。开启后结果数值错误，仅做性能对照。默认关。
    const bool scale_a_stub = env_int("DG_W4_SCALE_A_STUB", 0) != 0;
    // 思路2 串行依赖实验：双缓冲预解码（staged_pair_a_regs，issue k 时预取 k+2），
    // 让 decode 与 WGMMA 多重叠一级，代价 +8 reg/线程。仅在 fuse_scale_b_decode 开启
    // 时有意义，用于测"解 issue 串行依赖"能否补回 fuse 路径在大 M 的退化。默认关。
    const bool fuse_predecode_pair =
        fuse_scale_b_decode and env_int("DG_W4_FUSE_PREDECODE_PAIR", 0) != 0;
    // fuse_scale_b_decode 一旦开启，所有 fast-path / direct-load / e8m0 / bf16
    // 互斥关闭——device 那边 static_assert 会拒绝同时开启。
    const bool scale_b_direct_load =
        gran_k_b == 32 and (expected_m <= 16 or bm32_skew_fast_path) and
        not fuse_scale_b_decode;
    const bool k32_quad_reduce =
        gran_k_b == 32 and (expected_m <= 16 or bm32_skew_fast_path) and
        not fuse_scale_b_decode;
    // 杠杆3 / BM=64 联动：把 4 累加器 quad-reduce（峰值 64 float，在
    // __launch_bounds__(384,1) 的 168reg 硬上限下触发 local spill）退化为已存在的
    // 2 累加器串行 pair-reduce（峰值 32 float）降低寄存器峰值活跃量。仅改单线程内
    // 累加器生命周期、不动线程模型/barrier，数值 bit 级等价。
    // 两种启用：
    //   (A) BM=32 下**单独**开 pair（DG_W4_K32_PAIR_REDUCE=1）：已证伪——ncu spill
    //       223680→134016(-40%) 但 wall-clock 反而 +10~25%（串行累加拉长依赖链，
    //       spill 不在 BM=32 关键路径上）。默认关。
    //   (B) BM=64 fast-path 命中时**必须**配 pair（自动，下面 block_m==64 分支）：
    //       BM=64 让 kNumAccum 32→quad 4 份 = 128 fp32，spill 爆炸退化 ~2x；pair
    //       2 份 = 64 fp32 受控。BM=64 靠 M-tile 数减半换吞吐，pair 是其必要前提。
    //       实测高置信子集（max_hint>=128 且 active<=8）净降 5~26%。
    const bool k32_pair_reduce =
        k32_quad_reduce and
        ((config.layout.block_m == 64 and bm64_fastpath_enabled) or
         env_enabled("DG_W4_K32_PAIR_REDUCE"));
    // 杠杆3 续（已证伪，默认关）：pair-reduce 之上把长驻 final_accum 从 fp32 改 bf16
    // 存储（nv_bfloat162）再减半长驻寄存器、继续压 spill（ncu 134016→6480，-97% vs
    // quad）。精度无损（W4 diff 0.0001/0.0000）。但 wall-clock 仍慢于 quad baseline
    // （随 pair-reduce 一起退化，叠加 bf16 promote 反而更长依赖链）。默认关，需
    // DG_W4_K32_BF16_FINAL_ACCUM=1 显式开启复现。
    const bool k32_bf16_final_accum =
        k32_pair_reduce and env_enabled("DG_W4_K32_BF16_FINAL_ACCUM");
    // 杠杆3 续2（已证伪，默认关）：把长驻 final_accum 整体搬进 smem_d（kDirectStore
    // + kFinalAccumScratch），final_accum_regs 缩成 1 元素。实测 spill 虽归零，但
    // (1) 寄存器仍顶 168（launch_bounds 硬顶，省下的 reg 没还给 occupancy，Block
    // Limit Registers 仍为 1）；(2) DirectStore 逐元素非合并写 gmem 严重拖累大
    // max_m：one_hot_256 0.77x→0.50x、one_hot_512 0.49x→0.29x。净负收益，仅保留
    // 开关供复现，默认关。需 DG_W4_K32_FINAL_ACCUM_SMEM=1 显式开启。
    const bool k32_final_accum_smem =
        k32_pair_reduce and env_enabled("DG_W4_K32_FINAL_ACCUM_SMEM");
    // RS baseline for direct E8M0 B scales: keep scale products in split form by default.
    // The exponent-adjust path is experimental and can regress some small-M shapes.
    const bool scale_b_pow2_promote =
        k32_quad_reduce and env_enabled("DG_W4_SCALE_B_POW2_PROMOTE");
    const bool k32_quad_split_promote =
        k32_quad_reduce and not env_disabled("DG_W4_K32_QUAD_SPLIT_PROMOTE");
    const bool k32_quad_scale_b_inline =
        k32_quad_reduce and env_enabled("DG_W4_K32_QUAD_SCALE_B_INLINE");
    const bool k32_quad_scale_b_prefetch =
        k32_quad_reduce and env_enabled("DG_W4_K32_QUAD_SCALE_B_PREFETCH");
    const bool k32_quad_scale_b_vec4 =
        k32_quad_reduce and env_enabled("DG_W4_K32_QUAD_SCALE_B_VEC4");
    const bool k32_quad_pair4x2_promote =
        k32_quad_reduce and not k32_quad_split_promote and
        env_enabled("DG_W4_K32_QUAD_PAIR4X2_PROMOTE");
    // small_m_simple_sched：device 端 kUseSmallMSimpleSched 仅编译期检查
    //   BLOCK_M<=16 + GroupedMasked + multicast=1，与 N/K 数值无关。
    // 默认守护 (k>=4096 + n<=4096) 来自历史 g32 + n=4096 dsv4 形状的保守覆盖。
    // 放开到 RELAX 形状集 (g>=8 + n∈{4096,6144,7168} + k∈{2048,3072,4096,7168})
    // 让 DSV4 EP 业务真实 shape (g24 + n∈{6144,7168} + k∈{3072,7168} + expected_m=1/2/3)
    // 也能命中 simple_sched，避开通用 masked scheduler 的额外开销。
    const bool small_m_simple_sched =
        gran_k_b == 32 and expected_m <= 16 and
        ((static_cast<int64_t>(desc.k) >= 4096 and static_cast<int64_t>(desc.n) <= 4096) or
         (static_cast<int64_t>(desc.num_groups) >= 8 and
          static_cast<int64_t>(desc.num_groups) <= 36 and
          (static_cast<int64_t>(desc.n) == 4096 or
           static_cast<int64_t>(desc.n) == 6144 or
           static_cast<int64_t>(desc.n) == 7168) and
          (static_cast<int64_t>(desc.k) == 2048 or
           static_cast<int64_t>(desc.k) == 3072 or
           static_cast<int64_t>(desc.k) == 4096 or
           static_cast<int64_t>(desc.k) == 7168)));
    const bool compact_masked_sched =
        bm32_skew_fast_path and not env_disabled("DG_W4_COMPACT_MASKED_SCHED");
    // compact_masked_sched 按 m_max 降序遍历 active group（实验性扩展，默认关）：
    //   想法：compact_masked_sched 默认按 group_idx 升序遍历 active group。
    //         当各 group masked_m 不均时，long-tail group 决定 last-wave 收敛节奏。
    //         开启后 inner loop 每步在剩余 active mask 内 O(active) 扫一次找 group_m
    //         最大者，让 wave 0 优先吃重 group，wave 末尾留小 group 收尾。
    //   实测结果（DG_W4_PATHB_REORDER_BY_MM=1 vs 默认 fast-path）：
    //     gateup_dense_tail_hot64    356us → 356us  (0%)
    //     gateup_dense_tail_hot128   399us → 396us  (+0.8%)
    //     gateup_dense_tail_hot214   401us → 400us  (-0.2%)
    //     gateup_mtp_dp4             157us → 157us  (0%)
    //     down_dense_tail_hot64      186us → 185us  (+0.5%)
    //     down_mtp_512_hot384_a8      98us → 100us  (-2%)
    //     down_dense_tail_hot214     206us → 206us  (0%)
    //     gateup_one_hot_32           31us →  31us  (+1%, 边缘 case)
    //     全军 ±2% 噪声振荡，无任何 case 出现 5-10% 实质收益。
    //   失败根因：compact_masked_sched 已经做了 (group, m_block, n_block) 三维紧凑
    //     展开，dense_tail_hot64 总 tile = 32 group × 2 m_blocks × 32 n_blocks =
    //     2048 ≈ 15.5 wave × 132 SM。当 total_tiles >> num_sms 时，wave 边界由
    //     ceil(total/132) 决定，与 group 顺序无关；重排只改变"哪些 SM 在 wave 0
    //     拿到 hot group"，last-wave 余数 = (total mod 132) 不变。
    //     边缘 case (one_hot_32 total=32 tile = 1 wave × 32 SM) 的 +1-2% 收益
    //     是 SM 提前进入 idle 的小幅噪声，不构成可推广收益。
    //   保留 env 入口仅做对照实验，生产环境永不开启。
    //   仅在 compact_masked_sched 已开启时生效（即 bm32_skew_fast_path），
    //   否则即便 env 开启也不会传到 device。
    const bool reorder_masked_by_max_m =
        compact_masked_sched and env_int("DG_W4_PATHB_REORDER_BY_MM", 0) != 0;
    // bf16 SFB：体积砍半，**默认开启**。两条 fast-path：
    //  - path-B (gran_k_b==32 + direct-load)：load_sfb / load_k32_quad_scale_b 直读 gmem。
    //  - path-A (gran_k_b==128)：cooperative prefetch 经 smem，smem 也存 bf16。
    // 仍要避开 packed-UE8M0 / fused-decode 等假设 fp32 位级布局的分支。
    // 显式 `DG_W4_SCALE_B_BF16=0` 时回退 fp32；用户传的 sfb 实际是 fp32 也自然回退。
    const bool scale_b_bf16 =
        ((scale_b_direct_load and gran_k_b == 32 and not k32_quad_scale_b_prefetch) or
         gran_k_b == 128) and
        not env_disabled("DG_W4_SCALE_B_BF16") and
        sfb.scalar_type() == torch::kBFloat16;
    // E8M0 SFB（仅 path-B fast-path）：每元素 1B = fp32 的 8 位指数，体积再砍 2x。
    // 解码 `__uint_as_float(uint32(e) << 23)` 零误差。**默认开启**：当用户传入
    // uint8 sfb 时自动启用；显式 `DG_W4_SCALE_B_E8M0=0` 时回退。
    // 与 bf16 互斥（因 sfb 实际 dtype 已不同），不会同时命中。
    const bool scale_b_e8m0 =
        scale_b_direct_load and gran_k_b == 32 and
        not k32_quad_scale_b_prefetch and
        (not env_disabled("DG_W4_SCALE_B_E8M0") or
         env_enabled("DG_W4_SCALE_B_E8M0_ONLY")) and
        sfb.scalar_type() == torch::kUInt8;
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kFloat or
                   (scale_b_bf16 and sfb.scalar_type() == torch::kBFloat16) or
                   (scale_b_e8m0 and sfb.scalar_type() == torch::kUInt8) or
                   (fuse_scale_b_decode and sfb.scalar_type() == torch::kInt));
    const SM90FP8FP4Gemm1D2DRSRuntime::Args& rs_args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .major_sfb = get_major_type_ab(sfb),
        .scale_b_direct_load = scale_b_direct_load,
        .scale_b_pow2_promote = scale_b_pow2_promote,
        .k32_quad_reduce = k32_quad_reduce,
        .k32_pair_reduce = k32_pair_reduce,
        .k32_bf16_final_accum = k32_bf16_final_accum,
        .k32_final_accum_smem = k32_final_accum_smem,
        .k32_quad_split_promote = k32_quad_split_promote,
        .k32_quad_scale_b_inline = k32_quad_scale_b_inline,
        .k32_quad_scale_b_prefetch = k32_quad_scale_b_prefetch,
        .k32_quad_scale_b_vec4 = k32_quad_scale_b_vec4,
        .k32_quad_pair4x2_promote = k32_quad_pair4x2_promote,
        .small_m_simple_sched = small_m_simple_sched,
        .compact_masked_sched = compact_masked_sched,
        .fuse_scale_b_decode = fuse_scale_b_decode,
        .scale_a_stub = scale_a_stub,
        .fuse_predecode_pair = fuse_predecode_pair,
        .scale_b_packed_ue8m0 = scale_b_packed_ue8m0,
        .scale_b_gran_k = static_cast<uint32_t>(gran_k_b),
        .b_is_int4_sym = b_is_int4_sym,
        .scale_b_bf16 = scale_b_bf16,
        .scale_b_e8m0 = scale_b_e8m0,
        .reorder_masked_by_max_m = reorder_masked_by_max_m,
        .gmem_b_ptr = b.first.data_ptr(),
        .gmem_d_ptr = d.data_ptr(),
        .sfb = sfb.data_ptr(),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8FP4Gemm1D2DRSRuntime::generate(rs_args);
    const auto runtime = compiler->build("sm90_m_grouped_fp8_fp4_gemm_masked_1d2d_rs_fused", code);
    SM90FP8FP4Gemm1D2DRSRuntime::launch(runtime, rs_args);
}

}  // namespace deep_gemm
