#pragma once

#include <unordered_map>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include "../utils/compatibility.hpp"

#include "../jit_kernels/impls/sm100_mega_mhc.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::mega_mhc {

namespace mhc_layout = layout::mega_mhc;

static uint32_t get_num_splits(const uint32_t num_m_blocks, const uint32_t hidden, const uint32_t num_sms) {
    if (heuristics_runtime->get_deterministic_algorithms())
        return mhc_layout::kDefaultNumSplits;

    // Fewest splits that still reach the shortest possible longest task
    const uint32_t num_k_blocks = hidden / mhc_layout::BLOCK_K;
    const uint32_t max_num_splits = cute::min(
        mhc_layout::kNumMaxSplits, cute::max(mhc_layout::kDefaultNumSplits, num_sms / num_m_blocks));
    return math::ceil_div(num_k_blocks, math::ceil_div(num_k_blocks, max_num_splits));
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

/**
 * Execute shifted or normal mHC, generate the next mixes, and apply RMSNorm
 * with optional FP8 E4M3 casting.
 *
 * Args:
 *     x                              [T,H]                             BF16
 *     residual                       [T,hc_mult,H]                     BF16
 *     shifted_prev_mix               [T,hc_mult,1] FP32 | None
 *     post_mix                       [T,hc_mult,1]                     FP32
 *     comb_res_mix                   [T,hc_mult,hc_mult]               FP32
 *     fn                             [hc_mult*(hc_mult+2),hc_mult*H]   FP32
 *     mix_scales                     [3]                               FP32
 *     mix_bases                      [hc_mult*(hc_mult+2)]             FP32
 *     hc_mult                        int
 *     hc_norm_eps                    float
 *     hc_pre_eps                     float
 *     hc_post_scale                  float
 *     sinkhorn_eps                   float
 *     num_sinkhorn_iters             int
 *     rmsnorm_weight                 [H]                               BF16
 *     rmsnorm_eps                    float
 *     rmsnorm_scale                  float
 *     new_residual                   [T,hc_mult,H]                     BF16
 *     new_prev_mix                   [T,hc_mult,1] FP32 | None
 *     new_post_mix                   [T,hc_mult,1]                     FP32
 *     new_comb_res_mix               [T,hc_mult,hc_mult]               FP32
 *     y_bf16                         [T,H] BF16 | None
 *     y_fp8                          [T,H] E4M3 | None
 *     y_gemm_sf                      [T,H/128] int32 | None
 *     y_routed_sf                    [T,H/128] int32 | None
 *     y_shared_sf                    [T,H/128] int32 | None
 *     shared_sf_block_m              int
 *
 * Returns:
 *     None
 *
 * Notes:
 *     shifted_prev_mix and new_prev_mix are both tensors for shifted mHC and
 *     both None for normal mHC. Split-K is selected per invocation: 16 with
 *     deterministic algorithms; otherwise the fewest unequal whole-block splits
 *     that reach the shortest longest producer task. FP8 gran_k is 32, with four
 *     UE8M0 scaling factors packed into each int32.
 *     y_bf16 and y_fp8 are independent optional outputs; at least one is required.
 *     With y_fp8, supply exactly either y_gemm_sf alone, or y_routed_sf and
 *     y_shared_sf together. y_gemm_sf is TMA-aligned column-major,
 *     y_routed_sf is contiguous row-major, and y_shared_sf is the Mega MoE
 *     shared-expert layout and requires shared_sf_block_m > 0.
 */
static void mega_mhc(const torch::Tensor& x,
                     const torch::Tensor& residual,
                     const std::optional<torch::Tensor>& shifted_prev_mix,
                     const torch::Tensor& post_mix,
                     const torch::Tensor& comb_res_mix,
                     const torch::Tensor& fn,
                     const torch::Tensor& mix_scales,
                     const torch::Tensor& mix_bases,
                     const int& hc_mult,
                     const float& hc_norm_eps,
                     const float& hc_pre_eps,
                     const float& hc_post_scale,
                     const float& sinkhorn_eps,
                     const int& num_sinkhorn_iters,
                     const torch::Tensor& rmsnorm_weight,
                     const float& rmsnorm_eps,
                     const float& rmsnorm_scale,
                     const torch::Tensor& new_residual,
                     const std::optional<torch::Tensor>& new_prev_mix,
                     const torch::Tensor& new_post_mix,
                     const torch::Tensor& new_comb_res_mix,
                     const std::optional<torch::Tensor>& y_bf16,
                     const std::optional<torch::Tensor>& y_fp8,
                     const std::optional<torch::Tensor>& y_gemm_sf,
                     const std::optional<torch::Tensor>& y_routed_sf,
                     const std::optional<torch::Tensor>& y_shared_sf,
                     const int& shared_sf_block_m) {
    // Shifted state is all-or-nothing; FP8 uses either GEMM SF or routed and shared SF together
    const bool is_shifted = shifted_prev_mix.has_value();
    DG_HOST_ASSERT(is_shifted == new_prev_mix.has_value());
    DG_HOST_ASSERT(y_routed_sf.has_value() == y_shared_sf.has_value());
    DG_HOST_ASSERT(not (y_gemm_sf.has_value() and y_routed_sf.has_value()));
    DG_HOST_ASSERT(y_bf16.has_value() or y_fp8.has_value());
    DG_HOST_ASSERT(y_fp8.has_value() == (y_gemm_sf.has_value() or y_routed_sf.has_value()));
    DG_HOST_ASSERT(y_shared_sf.has_value() ? shared_sf_block_m > 0 : shared_sf_block_m == 0);

    const auto [num_tokens, hidden] = get_shape<2>(x);
    const auto [residual_tokens, residual_num_routes, residual_hidden] = get_shape<3>(residual);
    const auto [post_mix_tokens, post_mix_num_routes, post_mix_width] = get_shape<3>(post_mix);
    const auto [comb_mix_tokens, comb_mix_rows, comb_mix_cols] = get_shape<3>(comb_res_mix);
    const auto [fn_num_outputs, fn_hidden] = get_shape<2>(fn);
    const auto [num_mix_scales] = get_shape<1>(mix_scales);
    const auto [num_mix_bases] = get_shape<1>(mix_bases);
    const auto [rmsnorm_hidden] = get_shape<1>(rmsnorm_weight);

    if (num_tokens == 0)
        return;

    const auto num_routes = hc_mult;
    DG_HOST_ASSERT(num_routes == mhc_layout::kNumRoutes);
    const auto num_hc_outputs = mhc_layout::kNumHCOutputs;
    DG_HOST_ASSERT(num_tokens <= static_cast<int>(mhc_layout::kNumMaxTokens));
    DG_HOST_ASSERT(hidden > 0 and hidden % (mhc_layout::kDefaultNumSplits * mhc_layout::BLOCK_K) == 0);
    DG_HOST_ASSERT(num_sinkhorn_iters >= 1);
    DG_HOST_ASSERT(num_tokens == residual_tokens and num_tokens == post_mix_tokens and num_tokens == comb_mix_tokens);
    DG_HOST_ASSERT(num_routes == residual_num_routes and num_routes == post_mix_num_routes and
                   num_routes == comb_mix_rows and num_routes == comb_mix_cols);
    DG_HOST_ASSERT(hidden == residual_hidden and post_mix_width == 1);
    DG_HOST_ASSERT(num_hc_outputs == fn_num_outputs and fn_hidden == num_routes * hidden);
    DG_HOST_ASSERT(num_mix_scales == 3 and num_mix_bases == num_hc_outputs);
    DG_HOST_ASSERT(rmsnorm_hidden == hidden);
    DG_HOST_ASSERT(new_residual.sizes() == residual.sizes());
    DG_HOST_ASSERT(new_post_mix.sizes() == post_mix.sizes());
    DG_HOST_ASSERT(new_comb_res_mix.sizes() == comb_res_mix.sizes());

    if (is_shifted) {
        const auto [prev_tokens, prev_num_routes, prev_width] = get_shape<3>(shifted_prev_mix.value());
        DG_HOST_ASSERT(prev_tokens == num_tokens and prev_num_routes == num_routes and prev_width == 1);
        DG_HOST_ASSERT(new_prev_mix->sizes() == shifted_prev_mix->sizes());
    }

    const auto device = x.device();
    const auto check_dense = [&](const torch::Tensor& tensor, const torch::ScalarType& dtype) {
        DG_HOST_ASSERT(tensor.scalar_type() == dtype);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(tensor.device() == device);
    };

    check_dense(x, torch::kBFloat16);
    check_dense(residual, torch::kBFloat16);
    check_dense(post_mix, torch::kFloat);
    check_dense(comb_res_mix, torch::kFloat);
    check_dense(fn, torch::kFloat);
    check_dense(mix_scales, torch::kFloat);
    check_dense(mix_bases, torch::kFloat);
    check_dense(rmsnorm_weight, torch::kBFloat16);
    check_dense(new_residual, torch::kBFloat16);
    check_dense(new_post_mix, torch::kFloat);
    check_dense(new_comb_res_mix, torch::kFloat);
    if (is_shifted) {
        check_dense(shifted_prev_mix.value(), torch::kFloat);
        check_dense(new_prev_mix.value(), torch::kFloat);
    }

    const auto check_norm_output = [&](const torch::Tensor& tensor, const torch::ScalarType& dtype) {
        const auto [output_tokens, output_hidden] = get_shape<2>(tensor);
        DG_HOST_ASSERT(output_tokens == num_tokens and output_hidden == hidden);
        check_dense(tensor, dtype);
    };

    if (y_bf16.has_value())
        check_norm_output(y_bf16.value(), torch::kBFloat16);
    if (y_fp8.has_value())
        check_norm_output(y_fp8.value(), torch::kFloat8_e4m3fn);

    const auto check_sf = [&](const torch::Tensor& tensor) {
        const auto [sf_tokens, num_sf_words] = get_shape<2>(tensor);
        DG_HOST_ASSERT(sf_tokens == num_tokens and
                       num_sf_words == hidden / static_cast<int>(mhc_layout::kHiddenPerSFWord));
        DG_HOST_ASSERT(tensor.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(tensor.device() == device);
    };

    if (y_gemm_sf.has_value()) {
        check_sf(y_gemm_sf.value());
        DG_HOST_ASSERT(y_gemm_sf->stride(0) == 1);
        DG_HOST_ASSERT(y_gemm_sf->stride(1) >= num_tokens);
        DG_HOST_ASSERT(y_gemm_sf->stride(1) % get_tma_aligned_size(1, sizeof(uint32_t)) == 0);
    }
    if (y_routed_sf.has_value()) {
        check_sf(y_routed_sf.value());
        DG_HOST_ASSERT(y_routed_sf->is_contiguous());
    }
    if (y_shared_sf.has_value()) {
        check_sf(y_shared_sf.value());
        const auto num_required_rows = ceil_div<int64_t>(num_tokens, shared_sf_block_m) * align<int64_t>(shared_sf_block_m, 128);
        DG_HOST_ASSERT(y_shared_sf->stride(0) == 1);
        DG_HOST_ASSERT(y_shared_sf->stride(1) >= num_required_rows);
        const auto num_required_elements = y_shared_sf->storage_offset() + num_required_rows +
                                           (hidden / mhc_layout::kHiddenPerSFWord - 1) * y_shared_sf->stride(1);
        const auto num_required_bytes = static_cast<uint64_t>(num_required_elements) * y_shared_sf->element_size();
        DG_HOST_ASSERT(num_required_bytes <= y_shared_sf->storage().nbytes());
    }

    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    const auto num_sms = runtime->get_num_sms();
    const auto num_m_blocks = mhc_layout::Workspace<>::get_num_m_blocks(static_cast<uint32_t>(num_tokens));
    const auto num_splits = get_num_splits(num_m_blocks, static_cast<uint32_t>(hidden), static_cast<uint32_t>(num_sms));
    const auto num_mhc_scratch_bytes = mhc_layout::Workspace<>::get_num_scratch_bytes(num_m_blocks, num_splits);

    const bool needs_bf16_scratch = not y_bf16.has_value() and is_shifted;
    const auto num_bf16_scratch_bytes = needs_bf16_scratch ? static_cast<int64_t>(num_tokens) * hidden * x.element_size() : 0;
    const auto scratch = torch::empty({static_cast<int64_t>(num_mhc_scratch_bytes) + num_bf16_scratch_bytes},
                                      x.options().dtype(torch::kByte));
    const auto& split_barriers = get_split_barriers(x.options());

    // Shifted FP8 is derived from rounded BF16, so FP8-only calls borrow the scratch tail for that numerical boundary.
    const auto y_bf16_storage = needs_bf16_scratch ? scratch.narrow(0, num_mhc_scratch_bytes, num_bf16_scratch_bytes).
                                                     view(torch::kBFloat16).view({num_tokens, hidden})
                                                   : y_bf16.value_or(x);

    // GEMM and routed SF differ only in strides and share one primary writer.
    const auto& y_primary_sf = y_routed_sf.has_value() ? y_routed_sf : y_gemm_sf;

    sm100_mega_mhc(
        x, residual, post_mix, comb_res_mix, shifted_prev_mix,
        fn, mix_scales, mix_bases,
        hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters,
        rmsnorm_weight, rmsnorm_eps,
        new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
        y_bf16_storage, y_bf16.has_value(), scratch, split_barriers,
        y_fp8, y_primary_sf, y_shared_sf,
        shared_sf_block_m, rmsnorm_scale, num_tokens, hidden, num_splits, num_sms);
}

static void register_apis(pybind11::module_& m) {
    m.def(
        "mega_mhc",
        &mega_mhc,
        py::arg("x"),
        py::arg("residual"),
        py::arg("shifted_prev_mix"),
        py::arg("post_mix"),
        py::arg("comb_res_mix"),
        py::arg("fn"),
        py::arg("mix_scales"),
        py::arg("mix_bases"),
        py::arg("hc_mult"),
        py::arg("hc_norm_eps"),
        py::arg("hc_pre_eps"),
        py::arg("hc_post_scale"),
        py::arg("sinkhorn_eps"),
        py::arg("num_sinkhorn_iters"),
        py::arg("rmsnorm_weight"),
        py::arg("rmsnorm_eps"),
        py::arg("rmsnorm_scale"),
        py::arg("new_residual"),
        py::arg("new_prev_mix"),
        py::arg("new_post_mix"),
        py::arg("new_comb_res_mix"),
        py::arg("y_bf16") = std::nullopt,
        py::arg("y_fp8") = std::nullopt,
        py::arg("y_gemm_sf") = std::nullopt,
        py::arg("y_routed_sf") = std::nullopt,
        py::arg("y_shared_sf") = std::nullopt,
        py::arg("shared_sf_block_m") = 0);
}

} // namespace deep_gemm::mega_mhc
