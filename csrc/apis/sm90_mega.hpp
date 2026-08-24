#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/functional.h>

#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"
#include "../utils/layout.hpp"
#include "../utils/system.hpp"

namespace deep_gemm::mega {

static constexpr int kSM90MegaMoEBlockM = MegaMoESM90Config::block_m;
static constexpr int kSM90MegaMoETokenAlignment = 128;
static constexpr int kSM90MegaMoEBlockN = MegaMoESM90Config::block_n;
static constexpr int kSM90MegaMoEWorkerCTAsPerSM = 2;
static constexpr int kSM90MegaMoECTAsPerTask = 1;
// The dispatch-side NVLink barrier assigns one signaling thread per rank and
// the compact MXFP4 frontend has 64 dispatch threads.
static constexpr int kSM90MegaMoEMaxRanks = 64;

using SM90MegaMoEBufferViews = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using SM90MegaMoEBufferSlicer = std::function<SM90MegaMoEBufferViews(const torch::Tensor&)>;

static bool is_valid_hidden_for_sm90_mega_moe(const int hidden) {
    // The combine epilogue uses four chunks above 8192 elements, with each
    // chunk vectorized in groups of 256 BF16 values.
    return hidden % 512 == 0 and (hidden <= 8192 or hidden % 1024 == 0);
}

static int get_token_alignment_for_sm90_mega_moe() {
    return kSM90MegaMoETokenAlignment;
}

static int get_num_ring_tokens_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden) {
    const auto num_experts_per_rank = num_experts / num_ranks;
    // Only physical activation/SF/top-k-weight slots are ring-sized; Workspace
    // retains the full logical-pool source metadata.
    const auto num_worker_ctas =
        kSM90MegaMoEWorkerCTAsPerSM * device_runtime->get_num_sms();
    const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
    const auto num_max_routed_tokens =
        num_max_tokens_per_rank * num_ranks * num_active_topk;
    const auto num_pool_blocks =
        math::ceil_div(num_max_routed_tokens, kSM90MegaMoEBlockM) +
        num_experts_per_rank;
    const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
        num_pool_blocks, num_worker_ctas, hidden, intermediate_hidden,
        kSM90MegaMoEBlockN, kSM90MegaMoECTAsPerTask);
    const auto num_ring_tokens =
        num_live_pool_blocks * kSM90MegaMoEBlockM;
    return math::align(num_ring_tokens, kSM90MegaMoETokenAlignment);
}

static std::tuple<int64_t, SM90MegaMoEBufferSlicer>
get_symm_buffer_size_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation,
    const int& num_shared_experts = 0) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(num_ranks > 0 and num_ranks <= kSM90MegaMoEMaxRanks);
    DG_HOST_ASSERT(num_experts > 0);
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank % kSM90MegaMoETokenAlignment == 0);
    DG_HOST_ASSERT(num_topk > 0 and num_topk <= std::min(num_experts, 32));
    DG_HOST_ASSERT(num_shared_experts >= 0);
    DG_HOST_ASSERT(num_topk + (num_shared_experts > 0 ? 1 : 0) <= 32);
    DG_HOST_ASSERT(hidden > 0 and intermediate_hidden > 0);
    // Input/L1 K128 FP32 SF rows occupy H/32 bytes and must retain 16-byte
    // TMA alignment. L2 K64 FP32 SF rows occupy I/16 bytes.
    DG_HOST_ASSERT(is_valid_hidden_for_sm90_mega_moe(hidden) and
                   intermediate_hidden % 256 == 0);
    DG_HOST_ASSERT(use_fp8_dispatch);
    DG_HOST_ASSERT(activation == "swiglu");
    const auto num_ring_tokens = get_num_ring_tokens_for_sm90_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden);
    const auto num_sf_ring_tokens = layout::get_num_sf_ring_tokens(
        num_ring_tokens, kSM90MegaMoEBlockM);
    const auto scale_layout_spec =
        layout::ScaleLayoutSpec::sm90_fp32_k128_k64(hidden, intermediate_hidden);
    const auto mega_buffer = layout::MegaMoEBuffer(
        nullptr, hidden, intermediate_hidden,
        num_ranks, num_experts, num_max_tokens_per_rank,
        num_topk, num_ring_tokens, num_sf_ring_tokens,
        /*with_sf=*/ true, num_shared_experts, scale_layout_spec);
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
    const auto num_max_shared_sf_tokens =
        layout::get_num_max_shared_sf_tokens(num_max_tokens_per_rank);

    // Slice function follows main's twelve-view ABI.  Shared weights remain
    // FP8+FP32 on SM90; only routed weights use packed MXFP4+UE8M0.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));

        auto shared_l1_acts = x;
        auto shared_l1_acts_sf = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l1_sf_buffer.base)),
            {num_max_shared_sf_tokens, hidden / 128},
            {1, num_max_shared_sf_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_token_buffer.base)),
            {num_max_tokens_per_rank, shared_intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device())) : torch::Tensor();
        auto shared_l2_acts_sf = num_shared_experts > 0 ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.shared_l2_sf_buffer.base)),
            {num_max_shared_sf_tokens, shared_intermediate_hidden / 64},
            {1, num_max_shared_sf_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device())) : torch::Tensor();

        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_token_buffer.base)),
            {num_ring_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(mega_buffer.l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden / 64},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(
            x, x_sf, topk_idx, topk_weights,
            shared_l1_acts, shared_l1_acts_sf,
            shared_l2_acts, shared_l2_acts_sf,
            l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {mega_buffer.get_num_bytes(), slice_input_buffers};
}

// SM90 (Hopper) FP8-activation MegaMoE entry point.
//
// Shared validation and dispatch for preprocessed packed MXFP4 weights with
// K32 relative UE8M0 scales plus one FP32 secondary scale per expert.
// Top-level routing is the caller's responsibility (see
// `deep_gemm/mega/__init__.py`).
static void sm90_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto [l1_weights, l1_mxfp4_weights_sf, l1_mxfp4_secondary] = l1_weights_tuple;
    const auto [l2_weights, l2_mxfp4_weights_sf, l2_mxfp4_secondary] = l2_weights_tuple;
    torch::Tensor shared_l1_weights, shared_l1_weights_sf;
    torch::Tensor shared_l2_weights, shared_l2_weights_sf;
    DG_HOST_ASSERT(
        shared_l1_weights_tuple_opt.has_value() ==
        shared_l2_weights_tuple_opt.has_value());

    // Architecture check
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    // MXFP4 weights use K32 relative UE8M0 SF plus an FP32 per-expert
    // secondary. Activations use per-token per-128-K float SF.
    DG_HOST_ASSERT(y.is_cuda());
    DG_HOST_ASSERT(y.dim() == 2);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.is_contiguous());
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks: preprocessed Humming MXFP4 uses int8 packed E2M1
    // [E, N, K/2], an opaque contiguous uint8 relative-UE8M0 payload with
    // public shape [E, N, K/32], and the FP32 per-expert secondary below.
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == kPackedFP4);
    const auto [num_experts_per_rank, intermediate_hidden_2, l1_stored_k] =
        get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, l2_stored_k] =
        get_shape<3>(l2_weights);
    const int hidden = static_cast<int>(l1_stored_k) * 2;
    const int intermediate_hidden = static_cast<int>(l2_stored_k) * 2;
    DG_HOST_ASSERT(y.size(1) == hidden);
    DG_HOST_ASSERT(num_tokens >= 0);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(l1_mxfp4_weights_sf.is_cuda() and l2_mxfp4_weights_sf.is_cuda());
    DG_HOST_ASSERT(l1_weights.device() == y.device() and
                   l2_weights.device() == y.device() and
                   l1_mxfp4_weights_sf.device() == y.device() and
                   l2_mxfp4_weights_sf.device() == y.device());

    // Keep host validation aligned with the default TMA-aligned Data layouts
    // reconstructed by the generated SM90 kernel.
    DG_HOST_ASSERT(is_valid_hidden_for_sm90_mega_moe(hidden) and
                   intermediate_hidden % 256 == 0);

    // Check the public weight-SF shape. SF is producer-warp loaded rather than
    // TMA-loaded; preprocessing owns its hidden-dependent physical ordering.
    DG_HOST_ASSERT(l1_mxfp4_weights_sf.scalar_type() == torch::kUInt8 and
                   l2_mxfp4_weights_sf.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(l1_mxfp4_weights_sf.is_contiguous() and
                   l2_mxfp4_weights_sf.is_contiguous());
    DG_HOST_ASSERT(l1_mxfp4_weights_sf.dim() == 3 and
                   l1_mxfp4_weights_sf.size(0) == num_experts_per_rank and
                   l1_mxfp4_weights_sf.size(1) == intermediate_hidden * 2 and
                   l1_mxfp4_weights_sf.size(2) == hidden / 32);
    DG_HOST_ASSERT(l2_mxfp4_weights_sf.dim() == 3 and
                   l2_mxfp4_weights_sf.size(0) == num_experts_per_rank and
                   l2_mxfp4_weights_sf.size(1) == hidden and
                   l2_mxfp4_weights_sf.size(2) == intermediate_hidden / 32);
    DG_HOST_ASSERT(l1_mxfp4_secondary.is_cuda() and
                   l2_mxfp4_secondary.is_cuda());
    DG_HOST_ASSERT(l1_mxfp4_secondary.device() == y.device() and
                   l2_mxfp4_secondary.device() == y.device());
    DG_HOST_ASSERT(l1_mxfp4_secondary.scalar_type() == torch::kFloat and
                   l2_mxfp4_secondary.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(l1_mxfp4_secondary.is_contiguous() and
                   l2_mxfp4_secondary.is_contiguous());
    DG_HOST_ASSERT(l1_mxfp4_secondary.dim() == 1 and
                   l1_mxfp4_secondary.size(0) == num_experts_per_rank);
    DG_HOST_ASSERT(l2_mxfp4_secondary.dim() == 1 and
                   l2_mxfp4_secondary.size(0) == num_experts_per_rank);

    // Match main's mixed-precision shared-expert contract: routed weights are
    // MXFP4, while one or more shared experts are concatenated into a single
    // wide FP8 FFN.  Its output occupies one extra combine slot and has no
    // router weight. SharedLinear1/SharedLinear2 are generated by the same
    // persistent runtime that schedules routed live-ring tasks.
    int num_shared_experts = 0;
    if (shared_l1_weights_tuple_opt.has_value()) {
        std::tie(shared_l1_weights, shared_l1_weights_sf) =
            shared_l1_weights_tuple_opt.value();
        std::tie(shared_l2_weights, shared_l2_weights_sf) =
            shared_l2_weights_tuple_opt.value();
        DG_HOST_ASSERT(shared_l1_weights.dim() == 2 and shared_l2_weights.dim() == 2);
        DG_HOST_ASSERT(shared_l1_weights.scalar_type() == torch::kFloat8_e4m3fn and
                       shared_l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(shared_l1_weights.is_contiguous() and
                       shared_l2_weights.is_contiguous());
        DG_HOST_ASSERT(get_major_type_ab(shared_l1_weights) == cute::UMMA::Major::K and
                       get_major_type_ab(shared_l2_weights) == cute::UMMA::Major::K);
        DG_HOST_ASSERT(shared_l1_weights.device() == y.device() and
                       shared_l2_weights.device() == y.device());

        const auto shared_intermediate_hidden =
            static_cast<int>(shared_l2_weights.size(1));
        DG_HOST_ASSERT(shared_intermediate_hidden > 0 and
                       shared_intermediate_hidden % intermediate_hidden == 0);
        num_shared_experts = shared_intermediate_hidden / intermediate_hidden;
        DG_HOST_ASSERT(shared_l1_weights.size(0) == shared_intermediate_hidden * 2 and
                       shared_l1_weights.size(1) == hidden);
        DG_HOST_ASSERT(shared_l2_weights.size(0) == hidden);

        // FP32 block scales remain in natural C-contiguous layout.  Unlike
        // main's SM100 UE8M0 scales, these are neither packed nor UTCCP-swizzled.
        DG_HOST_ASSERT(shared_l1_weights_sf.is_cuda() and
                       shared_l2_weights_sf.is_cuda());
        DG_HOST_ASSERT(shared_l1_weights_sf.device() == y.device() and
                       shared_l2_weights_sf.device() == y.device());
        DG_HOST_ASSERT(shared_l1_weights_sf.scalar_type() == torch::kFloat and
                       shared_l2_weights_sf.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(shared_l1_weights_sf.is_contiguous() and
                       shared_l2_weights_sf.is_contiguous());
        DG_HOST_ASSERT(shared_l1_weights_sf.dim() == 2 and
                       shared_l1_weights_sf.size(0) == shared_intermediate_hidden * 2 / 128 and
                       shared_l1_weights_sf.size(1) == hidden / 128);
        DG_HOST_ASSERT(shared_l2_weights_sf.dim() == 2 and
                       shared_l2_weights_sf.size(0) == hidden / 128 and
                       shared_l2_weights_sf.size(1) == shared_intermediate_hidden / 128);
        DG_HOST_ASSERT(num_topk + 1 <= 32);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda());
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == y.device());
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() ==
                       num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_ranks > 0 and num_ranks <= kSM90MegaMoEMaxRanks);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    DG_HOST_ASSERT(sym_buffer.device() == y.device());
    DG_HOST_ASSERT(sym_buffer.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(sym_buffer.is_contiguous());
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank % kSM90MegaMoETokenAlignment == 0);
    DG_HOST_ASSERT(num_experts > 0 and num_topk > 0 and
                   num_topk <= std::min(num_experts, 32));
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    DG_HOST_ASSERT(num_experts == num_experts_);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_sm90_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation, num_shared_experts);
    // The live-ring offsets depend on H/I/S and the active worker-SM count.
    // Require the exact allocation contract so a buffer created for another
    // shape, shared-expert count, or `set_num_sms` value cannot be re-sliced
    // with silently shifted routed regions.
    DG_HOST_ASSERT(sym_buffer.nbytes() == static_cast<size_t>(num_required_bytes));

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights,
                shared_l1_acts, shared_l1_acts_sf,
                shared_l2_acts, shared_l2_acts_sf,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    sm90_fp8_mxfp4_mega_moe(y,
                            l1_acts, l1_acts_sf,
                            l2_acts, l2_acts_sf,
                            shared_l1_acts, shared_l1_acts_sf,
                            shared_l2_acts, shared_l2_acts_sf,
                            l1_weights, l2_weights,
                            l1_mxfp4_weights_sf, l2_mxfp4_weights_sf,
                            shared_l1_weights, shared_l2_weights,
                            shared_l1_weights_sf, shared_l2_weights_sf,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_shared_experts,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math,
                            l1_mxfp4_secondary, l2_mxfp4_secondary);

    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void fp8_mxfp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math) {
    sm90_mega_moe(
        y, l1_weights_tuple, l2_weights_tuple,
        shared_l1_weights_tuple_opt, shared_l2_weights_tuple_opt,
        cumulative_local_expert_recv_stats,
        sym_buffer, sym_buffer_ptrs, rank_idx,
        num_max_tokens_per_rank, num_experts, num_topk,
        recipe, activation, activation_clamp_opt, fast_math);
}

static void register_sm90_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_sm90_mega_moe", &get_token_alignment_for_sm90_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_mega_moe", &get_symm_buffer_size_for_sm90_mega_moe);
    m.def("fp8_mxfp4_mega_moe", &fp8_mxfp4_mega_moe);
#endif
}

} // namespace deep_gemm::mega
