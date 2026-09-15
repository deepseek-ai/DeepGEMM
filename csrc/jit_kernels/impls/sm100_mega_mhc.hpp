#pragma once

#include <cstdio>
#include <format>

#include <deep_gemm/layout/mega_mhc.cuh>

#include "../../runtime/jit.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

namespace mhc_layout = layout::mega_mhc;

static void sm100_mega_mhc(
    const torch::Tensor& x, const torch::Tensor& residual,
    const torch::Tensor& post_mix, const torch::Tensor& comb_res_mix,
    const std::optional<torch::Tensor>& shifted_prev_mix, const torch::Tensor& fn,
    const torch::Tensor& mix_scales, const torch::Tensor& mix_bases,
    const float& hc_norm_eps, const float& hc_pre_eps, const float& hc_post_scale,
    const float& sinkhorn_eps, const int& num_sinkhorn_iters,
    const torch::Tensor& rmsnorm_weight, const float& rmsnorm_eps,
    const torch::Tensor& new_residual,
    const std::optional<torch::Tensor>& new_prev_mix, const torch::Tensor& new_post_mix,
    const torch::Tensor& new_comb_res_mix, const torch::Tensor& y_bf16,
    const bool& store_bf16,
    const torch::Tensor& scratch,
    const torch::Tensor& split_barriers,
    const std::optional<torch::Tensor>& y_fp8,
    const std::optional<torch::Tensor>& y_primary_sf,
    const std::optional<torch::Tensor>& y_shared_sf,
    const int& shared_sf_block_m, const float& rmsnorm_scale,
    const int& num_tokens, const int& hidden, const int& num_splits, const int& num_sms) {
    const bool is_shifted = shifted_prev_mix.has_value();

    const auto make_hidden_tma_desc = [&](const torch::Tensor& tensor) {
        return make_tma_2d_desc(tensor, hidden, num_tokens, mhc_layout::BLOCK_K, mhc_layout::BLOCK_M,
                                static_cast<int>(tensor.stride(0)), mhc_layout::kSwizzleMode);
    };
    const auto make_residual_tma_desc = [&](const torch::Tensor& tensor) {
        return make_tma_3d_desc(
            tensor, hidden, num_tokens, mhc_layout::kNumRoutes,
            mhc_layout::BLOCK_K, mhc_layout::BLOCK_M, mhc_layout::kNumRoutes,
            static_cast<int>(tensor.stride(0)), static_cast<int>(tensor.stride(1)), mhc_layout::kSwizzleMode);
    };
    const auto make_coeff_tma_desc = [&](const torch::Tensor& tensor, const int channels, const int swizzle_mode) {
        return make_tma_2d_desc(
            tensor, channels, num_tokens, channels, mhc_layout::BLOCK_M,
            static_cast<int>(tensor.stride(0)), swizzle_mode);
    };

    const auto smem_size = static_cast<int>(sizeof(mhc_layout::SharedStorage));
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Mega %s mHC: T=%d, H=%d, splits=%d, IO/Fn/A stages=%u/%u/%u, "
               "threads=%u, shared memory=%d\n",
               is_shifted ? "shifted" : "normal", num_tokens, hidden, num_splits,
               mhc_layout::kNumIOStages, mhc_layout::kNumFnStages, mhc_layout::kNumAStages,
               mhc_layout::kNumThreads, smem_size);
    }

    // Tensor maps use fastest-moving dimensions first. Residual and Fn expose
    // the HC route as a third dimension without materializing transposed views.
    const auto tensor_map_residual = make_residual_tma_desc(residual);
    const auto tensor_map_x = make_hidden_tma_desc(x);
    const auto tensor_map_fn = make_tma_3d_desc(
        fn,
        hidden, mhc_layout::kNumHCOutputs, mhc_layout::kNumRoutes,
        mhc_layout::BLOCK_K, mhc_layout::kNumHCOutputs, mhc_layout::kNumRoutes,
        static_cast<int>(fn.stride(0)), hidden,
        mhc_layout::kSwizzleMode, 0, true);
    const auto tensor_map_post_mix = make_coeff_tma_desc(post_mix, mhc_layout::kNumRoutes, 0);
    const auto tensor_map_comb_res_mix = make_coeff_tma_desc(
        comb_res_mix, mhc_layout::kNumRoutes * mhc_layout::kNumRoutes, mhc_layout::kSwizzleMode / 2);
    const auto tensor_map_shifted_prev_mix = make_coeff_tma_desc(
        is_shifted ? shifted_prev_mix.value() : post_mix, mhc_layout::kNumRoutes, 0);
    const auto tensor_map_new_residual = make_residual_tma_desc(new_residual);
    const auto tensor_map_y_bf16 = make_hidden_tma_desc(is_shifted ? y_bf16 : x);

    const mhc_layout::MixArgs mix_args = {
        .scales = mix_scales.data_ptr<float>(),
        .bases = mix_bases.data_ptr<float>(),
        .new_prev_mix = is_shifted ? new_prev_mix->data_ptr<float>() : nullptr,
        .new_post_mix = new_post_mix.data_ptr<float>(),
        .new_comb_res_mix = new_comb_res_mix.data_ptr<float>(),
        .hc_norm_eps = hc_norm_eps,
        .hc_pre_eps = hc_pre_eps,
        .hc_post_scale = hc_post_scale,
        .sinkhorn_eps = sinkhorn_eps,
        .num_sinkhorn_iters = static_cast<uint32_t>(num_sinkhorn_iters),
    };
    const mhc_layout::NormArgs norm_args = {
        .num_tokens = static_cast<uint32_t>(num_tokens),
        .weight = reinterpret_cast<const nv_bfloat16*>(rmsnorm_weight.data_ptr()),
        .new_residual = reinterpret_cast<const nv_bfloat16*>(new_residual.data_ptr()),
        .eps = rmsnorm_eps,
        .scale = rmsnorm_scale,
        .y_bf16 = reinterpret_cast<nv_bfloat16*>(y_bf16.data_ptr()),
        .y_fp8 = y_fp8.has_value() ? reinterpret_cast<__nv_fp8_e4m3*>(y_fp8->data_ptr()) : nullptr,
        .y_primary_sf = y_primary_sf.has_value() ? reinterpret_cast<uint32_t*>(y_primary_sf->data_ptr<int32_t>()) : nullptr,
        .y_primary_sf_stride_token = y_primary_sf.has_value() ? y_primary_sf->stride(0) : 0,
        .y_primary_sf_stride_word = y_primary_sf.has_value() ? y_primary_sf->stride(1) : 0,
        .y_shared_sf = y_shared_sf.has_value() ? reinterpret_cast<uint32_t*>(y_shared_sf->data_ptr<int32_t>()) : nullptr,
        .y_shared_sf_stride_word = y_shared_sf.has_value() ? y_shared_sf->stride(1) : 0,
    };

    // Compile
    const auto kernel = jit->compile("sm100_mega_mhc", std::format(R"(
#include <deep_gemm/impls/sm100_mega_mhc.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mega_mhc_impl<
        {}, {}, {},
        {}, {}, {}, {}
    >);
}};
)",
        hidden, num_splits, num_sms,
        mix_args.new_prev_mix != nullptr,
        store_bf16,
        norm_args.y_fp8 != nullptr,
        shared_sf_block_m));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(mhc_layout::kNumThreads, 1, 1),
        },
        tensor_map_residual,
        tensor_map_x,
        tensor_map_fn,
        tensor_map_post_mix,
        tensor_map_comb_res_mix,
        tensor_map_shifted_prev_mix,
        tensor_map_new_residual,
        tensor_map_y_bf16,
        mix_args,
        norm_args,
        scratch.data_ptr(),
        reinterpret_cast<uint64_t*>(split_barriers.data_ptr()));
}

} // namespace deep_gemm
