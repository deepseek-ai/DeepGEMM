#pragma once

#include <cstdio>
#include <format>
#include <unordered_map>

#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <deep_gemm/layout/mega_mhc.cuh>

#include "../../runtime/jit.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

namespace mhc_layout = layout::mega_mhc;

// Partition count of the norm phase, resolved per launch and passed to the kernel as a
// compile-time constant.
//
// The ranges below override the geometry heuristic for contiguous token-count spans;
// token counts outside them keep the heuristic.
struct NormPartitionRange {
    int hidden;
    int first_tokens;
    int last_tokens;
    uint32_t num_partitions;
};

static constexpr NormPartitionRange kNormPartitionRanges[] = {
    {5120,  196,  304, 12},
    {5120,  368,  520,  7},
    {5120,  524,  576,  6},
    {5120,  676,  848, 12},
    {5120,  944, 1196,  5},
    {7168,  116,  176,  7},
    {7168,  244,  304, 12},
    {7168,  308,  364, 10},
    {7168,  368,  520,  7},
    {7168,  524,  576,  6},
    {7168,  800,  876,  7},
    {7168,  880, 1024,  6},
};

static uint32_t get_num_splits(const uint32_t num_m_blocks, const uint32_t hidden, const uint32_t num_sms) {
    if (heuristics_runtime->get_deterministic_algorithms())
        return mhc_layout::kDefaultNumSplits;

    const uint32_t num_k_blocks = hidden / mhc_layout::BLOCK_K;
    // Estimate grid waves at four K blocks per SM, then distribute the split budget.
    const uint32_t num_waves = cute::max(1u, num_m_blocks * num_k_blocks / (4 * num_sms));
    const uint32_t max_num_splits = cute::min(
        mhc_layout::kNumMaxSplits, cute::max(mhc_layout::kDefaultNumSplits,
                                             math::ceil_div(num_waves * num_sms, num_m_blocks)));
    return math::ceil_div(num_k_blocks, math::ceil_div(num_k_blocks, max_num_splits));
}

// Look up the ranges, then fall back to the geometry heuristic.
static uint32_t get_norm_partitions(const int hidden, const int num_tokens,
                                    const int num_splits, const int num_sms,
                                    const bool store_fp8) {
    for (const auto& range: kNormPartitionRanges)
        if (store_fp8 and range.hidden == hidden and
            num_tokens >= range.first_tokens and num_tokens <= range.last_tokens)
            return range.num_partitions;

    // Keep at most two hidden packs per norm task.
    const auto num_hidden_packs = hidden / mhc_layout::kHiddenPerNormPack;
    const auto num_token_pairs = math::ceil_div<uint32_t>(num_tokens, mhc_layout::kNumTokensPerNormTask);
    const auto min_num_partitions = math::ceil_div(
        num_splits * mhc_layout::kNumWarpsPerWG, mhc_layout::BLOCK_M / mhc_layout::kNumTokensPerNormTask);
    const auto num_target_tasks = num_sms * mhc_layout::kNumWarpsPerWG * (store_fp8 ? 8u : 1u);
    return cute::max(min_num_partitions, math::ceil_div(
        num_hidden_packs, cute::max(store_fp8 ? 2u : 1u,
            num_token_pairs * num_hidden_packs / num_target_tasks)));
}

static const torch::Tensor& get_split_barriers(const torch::TensorOptions& options) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    DG_HOST_ASSERT(options.device() == stream.device());
    static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> split_barriers_by_stream;
    auto& split_barriers = split_barriers_by_stream[stream];
    if (not split_barriers.defined()) {
        // Warm up each stream before capture so one-time zeroing is not replayed with the graph.
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        split_barriers = torch::zeros(
            {mhc_layout::kNumSplitBarriers, mhc_layout::kNumMaxMBlocks, mhc_layout::kSplitBarrierLineBytes},
            options.dtype(torch::kByte));
    }
    return split_barriers;
}

static void sm100_mega_mhc(
    const torch::Tensor& x, const torch::Tensor& residual,
    const torch::Tensor& post_mix, const torch::Tensor& comb_res_mix,
    const torch::Tensor& shifted_prev_mix, const torch::Tensor& fn,
    const torch::Tensor& mix_scales, const torch::Tensor& mix_bases,
    const float& hc_norm_eps, const float& hc_pre_eps, const float& hc_post_scale,
    const float& sinkhorn_eps, const int& num_sinkhorn_iters,
    const torch::Tensor& rmsnorm_weight, const float& rmsnorm_eps,
    const torch::Tensor& new_residual,
    const torch::Tensor& new_prev_mix, const torch::Tensor& new_post_mix,
    const torch::Tensor& new_comb_res_mix, const torch::Tensor& y_bf16,
    const bool& store_bf16,
    const torch::Tensor& scratch,
    const torch::Tensor& split_barriers,
    const std::optional<torch::Tensor>& y_fp8,
    const std::optional<torch::Tensor>& y_primary_sf,
    const std::optional<torch::Tensor>& y_shared_sf,
    const int& shared_sf_block_m, const float& rmsnorm_scale,
    const int& num_tokens, const int& hidden, const int& num_splits, const int& num_sms) {
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

    const auto norm_partitions = get_norm_partitions(hidden, num_tokens, num_splits, num_sms, y_fp8.has_value());

    const auto smem_size = static_cast<int>(sizeof(mhc_layout::SharedStorage));
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("Mega shifted mHC: T=%d, H=%d, splits=%d, norm partitions=%u, IO/Fn/A stages=%u/%u/%u, "
               "threads=%u, shared memory=%d\n",
               num_tokens, hidden, num_splits, norm_partitions,
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
    const auto tensor_map_prev_mix = make_coeff_tma_desc(
        shifted_prev_mix, mhc_layout::kNumRoutes, 0);
    const auto tensor_map_new_residual = make_residual_tma_desc(new_residual);
    const auto tensor_map_y_bf16 = make_hidden_tma_desc(y_bf16);

    const mhc_layout::MixArgs mix_args = {
        .scales = mix_scales.data_ptr<float>(),
        .bases = mix_bases.data_ptr<float>(),
        .new_prev_mix = new_prev_mix.data_ptr<float>(),
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
        .eps = rmsnorm_eps,
        .scale = rmsnorm_scale,
        .x1 = y_bf16.data_ptr(),
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
#include <deep_gemm/impls/sm100_mega_mhc_inference.cuh>

using namespace deep_gemm::mega_mhc_inference;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mega_mhc_impl<
        {}, {}, {},
        {}, {}, {}, {}
    >);
}};
)",
        hidden, num_splits, num_sms,
        store_bf16,
        norm_args.y_fp8 != nullptr,
        shared_sf_block_m, norm_partitions));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(mhc_layout::kNumThreads, 1, 1),
            .cooperative = true,
        },
        tensor_map_residual,
        tensor_map_x,
        tensor_map_fn,
        tensor_map_post_mix,
        tensor_map_comb_res_mix,
        tensor_map_prev_mix,
        tensor_map_new_residual,
        tensor_map_y_bf16,
        mix_args,
        norm_args,
        scratch.data_ptr(),
        split_barriers.data_ptr());
}

} // namespace deep_gemm
