#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/layout.hpp"
#include "epilogue.hpp"
#include "runtime_utils.hpp"

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
        int num_packed_stages;
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
        {}, {},
        {}, {},
        {}, {},
        {}, {},
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
        args.gemm_config.pipeline_config.num_stages, args.num_packed_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        get_default_epilogue_type(std::nullopt));
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
    DG_HOST_ASSERT(b.second.scalar_type() == torch::kFloat);
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
        const std::optional<int>& block_n_override) {
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
    // FP4 B is decoded by each CTA, so only A multicast (cluster_n) is legal here.
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
    if (not use_psum_layout and m >= num_groups * 256 and n >= 1024) {
        layout.block_m = 256;
        layout.block_n = 64;
        layout.cluster_m = 1;
    }
    auto config = rebuild_config(layout);

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
    //   (1) the extra packed-FP4 B staging buffer (BLOCK_N * BLOCK_K / 2 bytes per packed stage)
    //   (2) the enlarged SFB cache: now stores (shape_k_scales, BLOCK_N) floats per block
    //       so the K loop reads SFB from smem instead of gmem.
    // The packed-B pipeline uses an independent (smaller) number of stages so we can keep
    // `num_stages` large for the WGMMA consumer pipeline.
    int num_packed_stages = 2;
    {
        const int block_k = config.layout.block_k;
        const int block_n = config.layout.block_n;
        const int shape_k_scales = ceil_div(static_cast<int>(k), block_k);
        const bool uniform_scale_b = (block_k % block_n == 0);
        const int sfb_old_bytes = align(shape_k_scales * (uniform_scale_b ? 1 : 2) * static_cast<int>(sizeof(float)), 16);
        // SFB cache now aliases smem_d (no separate allocation). We only need to
        // remove the original SFB bytes from the baseline.
        const int sfb_extra = -sfb_old_bytes;

        const int packed_per_stage = block_n * (block_k / 2);
        const int smem_a_per_stage = config.storage_config.load_block_m * block_k *
                                     static_cast<int>(c10::elementSize(desc.a_dtype));
        const int smem_b_per_stage = config.storage_config.load_block_n * block_k *
                                     static_cast<int>(c10::elementSize(desc.b_dtype));
        const int smem_sfa_per_stage = align(config.layout.block_m * static_cast<int>(sizeof(float)), 128);
        const int original_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage;
        const int orig_num_stages = config.pipeline_config.num_stages;
        const int smem_extra = config.pipeline_config.smem_size - orig_num_stages * original_per_stage + sfb_extra;

        auto fits = [&](int stages, int p_stages) {
            return smem_extra + stages * original_per_stage + p_stages * packed_per_stage
                   <= SM90ArchSpec::smem_capacity;
        };

        // Try to keep `num_stages = orig_num_stages` with packed_stages=2; otherwise drop num_stages,
        // and finally fall back to packed_stages=1 if needed.
        int chosen_stages = orig_num_stages;
        int chosen_packed = 2;
        while (chosen_stages >= 3 and not fits(chosen_stages, chosen_packed))
            -- chosen_stages;
        if (chosen_stages < 3) {
            chosen_packed = 1;
            chosen_stages = orig_num_stages;
            while (chosen_stages >= 3 and not fits(chosen_stages, chosen_packed))
                -- chosen_stages;
        }
        DG_HOST_ASSERT(chosen_stages >= 3);
        config.pipeline_config.num_stages = chosen_stages;
        num_packed_stages = chosen_packed;
        config.pipeline_config.smem_size = smem_extra + chosen_stages * original_per_stage +
                                           chosen_packed * packed_per_stage;
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
        .num_packed_stages = num_packed_stages,
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

}  // namespace deep_gemm
