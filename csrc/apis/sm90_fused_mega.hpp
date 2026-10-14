#pragma once
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/functional.h>

#include <deep_jit/utils/env.hpp>

#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm90_fp8_fused_mega_moe.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::sm90_fused_mega {

static int get_token_alignment_for_sm90_fused_mega_moe() {
    return layout::kSM90FusedLCMBlockM;
}

// Byte layout of a fused symm buffer whose data pools hold `num_ring_tokens` tokens (0 = the full pool)
static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_layout_for_sm90_fused_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation,
    const int& num_ring_tokens, const int& l2_act_sf_gran_k) {
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    if (not use_fp8_dispatch)
        DG_HOST_UNREACHABLE("SM90 fused FP8 MegaMoE supports FP8 dispatch only");
    if (activation != "swiglu")
        DG_HOST_UNREACHABLE("SM90 fused FP8 MegaMoE supports the swiglu activation only");
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and
                   num_max_tokens_per_rank % layout::kSM90FusedLCMBlockM == 0);
    if (hidden <= 0 or hidden % 128 != 0 or intermediate_hidden <= 0 or intermediate_hidden % 128 != 0)
        DG_HOST_UNREACHABLE("SM90 fused FP8 MegaMoE requires hidden and intermediate_hidden to be positive multiples of 128");
    // The kernel L2 arrival mask covers at most 64 per-64-K groups. Checked here as well as at launch so a shape this path
    // cannot serve is reported before a symmetric buffer is allocated for it.
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);
    DG_HOST_ASSERT(l2_act_sf_gran_k == 64 or l2_act_sf_gran_k == 128);
    DG_HOST_ASSERT(num_ring_tokens >= 0);

    // Workspace bytes
    const int num_experts_per_rank = num_experts / num_ranks;
    const int num_max_pool_tokens = get_num_max_pool_tokens_sm90_fused(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const auto workspace_end_ptr = layout::SM90FusedWorkspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_ring_tokens).get_end_ptr();

    // Layouts
    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    // Slot-major L2 act SF pool (`[k_sf_idx][pool token]`): the per-token byte count only sizes it and need not be TMA-aligned
    const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden * 4 / l2_act_sf_gran_k, false);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_tokens_per_rank,
        workspace_end_ptr);
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Data pools hold the ring (the full pool when no ring); SF pools are padded for the worst-case BLOCK_M
    const int num_data_pool_tokens = num_ring_tokens == 0 ? num_max_pool_tokens : num_ring_tokens;
    const int num_max_padded_sf_pool_tokens = get_num_padded_sf_pool_tokens_sm90_fused(num_data_pool_tokens);

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_data_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_data_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_data_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    // `x_sf` is K-major; pool scale factors are M-major.
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_data_pool_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_data_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, intermediate_hidden / l2_act_sf_gran_k},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    // + the split-K region (ready flags + fp32 partials, layout::kSM90FusedSplitKBytes; zero at allocation, reset by every launch)
    return {reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()) + static_cast<int64_t>(layout::kSM90FusedSplitKBytes),
            slice_input_buffers};
}

// Returns (num_bytes, slicer, num_ring_tokens, l2_lag_encoded). The ring capacity and the encoded L2-lag schedule are
// derived here from `num_experts_per_wave` (0 = full pool, -1 = auto wave size, N > 0 = fixed wave size) and must be
// passed back unchanged at every launch.
static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>, int, int>
get_symm_buffer_size_for_sm90_fused_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation,
    const int& num_experts_per_wave, const int& l2_act_sf_gran_k) {
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    const auto [num_ring_tokens, l2_lag_encoded] = get_num_ring_tokens_for_sm90_fused_mega_moe(
        num_ranks, num_experts, num_experts / num_ranks,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_experts_per_wave, l2_act_sf_gran_k);
    const auto [num_bytes, slice_input_buffers] = get_symm_buffer_layout_for_sm90_fused_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation,
        num_ring_tokens, l2_act_sf_gran_k);
    return {num_bytes, slice_input_buffers, num_ring_tokens, l2_lag_encoded};
}

// SM90 (Hopper) fused FP8 MegaMoE entry point: the same contract as `fp8_mega_moe` (FP8 e4m3 weights with block
// (128, 128) float scale factors) on a buffer sized by `get_symm_buffer_size_for_sm90_fused_mega_moe`.
static void sm90_fused_fp8_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    // `num_ring_tokens`, `l2_lag_encoded` and `l2_act_sf_gran_k` are the values `get_symm_buffer_size_for_sm90_fused_mega_moe`
    // sized `sym_buffer` with; `num_tokens_bound` is the caller's bound on this call's per-rank token count on every rank (0 = capacity)
    const int& num_ring_tokens,
    const int& num_tokens_bound,
    const int& l2_lag_encoded,
    const int& l2_act_sf_gran_k
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Architecture check
    if (jit->device.get_arch_major() != 9)
        DG_HOST_UNREACHABLE("SM90 fused FP8 MegaMoE requires a compute capability 9.x GPU");

    // Config checks: block (128, 128) float SF for weights, per-token per-128-K float SF for activations
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    if (rm != 128 or rn != 128 or rk != 128)
        DG_HOST_UNREACHABLE("SM90 fused FP8 MegaMoE requires recipe=(128, 128, 128)");
    if (activation != "swiglu")
        DG_HOST_UNREACHABLE("SM90 fused FP8 MegaMoE supports the swiglu activation only");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks: weights must be FP8 e4m3, K-major
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
    // The kernel's L2 arrival mask covers at most 64 per-64-K groups
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    // Weight SFs are raw global-memory loads in natural MN-major order.
    constexpr int kGranMN = 128, kGranK = 128;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    if (not l1_weights_sf.is_contiguous() or not l2_weights_sf.is_contiguous())
        DG_HOST_UNREACHABLE(
            "SM90 fused FP8 MegaMoE weight scale factors must use contiguous natural layouts");

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() ==
                       num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes against the geometry it was sized with
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_experts == num_experts_per_rank * num_ranks);
    // 0 means the buffer capacity. A bound is the per-rank token count every rank's schedule is sized for, so it must
    // be the same on every rank: this checks only the local count, and a bound below a peer's count sizes this rank's
    // single-wave pool for fewer rows than arrive, which reuses a ring slot inside one wave and hangs.
    DG_HOST_ASSERT(num_tokens_bound >= 0);
    DG_HOST_ASSERT(num_tokens_bound == 0 or num_tokens_bound >= num_tokens);
    DG_HOST_ASSERT(num_tokens_bound <= num_max_tokens_per_rank);
    const auto [num_required_bytes, slice] = get_symm_buffer_layout_for_sm90_fused_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation,
        num_ring_tokens, l2_act_sf_gran_k);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    sm90_fp8_fused_mega_moe(y,
                            l1_acts, l1_acts_sf,
                            l2_acts, l2_acts_sf,
                            l1_weights, l2_weights,
                            l1_weights_sf, l2_weights_sf,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            num_ring_tokens, l2_act_sf_gran_k,
                            activation_clamp, fast_math,
                            num_tokens_bound, l2_lag_encoded);

    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

// SM90 fused MegaMoE with packed MXFP4 routed weights (opt-in; PR #411 weight format): each weight is the processed triple
// `(packed E2M1 int8 [E, N, K/2], relative UE8M0 uint8 [E, N, K/32] with physical [E, K/128, N, 4] payload, FP32 secondary [E])`
// returned by `transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90`. Activations, the symm buffer and the schedule are those of
// `sm90_fused_fp8_mega_moe`; phase 1 always runs the routed split-N BLOCK_M 64 x BLOCK_N 256 tile.
static void sm90_fused_fp8_mxfp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens,
    const int& num_tokens_bound,
    const int& l2_lag_encoded,
    const int& l2_act_sf_gran_k,
    // the caller's hint that this call is a decode step of one global token (selects the split-K instance; any routing stays correct)
    const bool& split_k_hint
) {
    const auto [l1_weights, l1_sfb, l1_secondary] = l1_weights_tuple;
    const auto [l2_weights, l2_sfb, l2_secondary] = l2_weights_tuple;

    if (jit->device.get_arch_major() != 9)
        DG_HOST_UNREACHABLE("SM90 fused MXFP4 MegaMoE requires a compute capability 9.x GPU");

    // Config checks: MXFP4 weights (K32 block scale), per-token per-128-K float SF for activations
    DG_HOST_ASSERT(y.is_cuda() and y.dim() == 2 and y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    if (rm != 1 or rn != 1 or rk != 32)
        DG_HOST_UNREACHABLE("SM90 fused MXFP4 MegaMoE requires recipe=(1, 1, 32)");
    if (activation != "swiglu")
        DG_HOST_UNREACHABLE("SM90 fused MXFP4 MegaMoE supports the swiglu activation only");
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Packed weights: int8 [E, N, K/2], K-major, contiguous; or the tiled layout [E, N / 256, K / 128, 16384] (both or neither)
    const bool tiled = l1_weights.dim() == 4;
    for (const auto* w: {&l1_weights, &l2_weights}) {
        DG_HOST_ASSERT(w->is_cuda() and w->device() == y.device());
        DG_HOST_ASSERT(w->scalar_type() == kPackedFP4 and w->dim() == (tiled ? 4 : 3) and w->is_contiguous());
        DG_HOST_ASSERT(not tiled or (w->size(3) == 256 * 64 and reinterpret_cast<uintptr_t>(w->data_ptr()) % 16 == 0));
    }
    const auto logical_shape = [&](const torch::Tensor& w) {
        return tiled ? std::make_tuple(w.size(0), w.size(1) * 256, w.size(2) * 64)
                     : std::make_tuple(w.size(0), w.size(1), w.size(2));
    };
    const auto [num_experts_per_rank, intermediate_hidden_2, l1_packed_k] = logical_shape(l1_weights);
    const auto [num_experts_per_rank_, hidden_, l2_packed_k] = logical_shape(l2_weights);
    const int hidden = static_cast<int>(l1_packed_k) * 2;
    const int intermediate_hidden = static_cast<int>(l2_packed_k) * 2;
    DG_HOST_ASSERT(y.size(1) == hidden);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    // 256-wide split-N tile (even N block counts), the coalesced SFB payload (hidden <= 8192), the L2 arrival mask
    if (hidden % 512 != 0 or (2 * intermediate_hidden) % 512 != 0 or hidden > 8192)
        DG_HOST_UNREACHABLE("SM90 fused MXFP4 MegaMoE requires hidden and 2 x intermediate_hidden to be multiples of 512 and hidden <= 8192");
    DG_HOST_ASSERT(intermediate_hidden / 64 <= 64);

    // Relative UE8M0 SFB: uint8, public shape [E, N, K/32], contiguous (physical [E, K/128, N, 4])
    const auto check_sfb = [&](const torch::Tensor& sfb, const int& n, const int& k) {
        DG_HOST_ASSERT(sfb.is_cuda() and sfb.device() == y.device());
        DG_HOST_ASSERT(sfb.scalar_type() == torch::kUInt8 and sfb.is_contiguous());
        DG_HOST_ASSERT(sfb.dim() == 3 and sfb.size(0) == num_experts_per_rank and sfb.size(1) == n and sfb.size(2) == k / 32);
        // 16-byte aligned bulk copies of [BLOCK_N] u32 words
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(sfb.data_ptr()) % 16 == 0);
    };
    check_sfb(l1_sfb, intermediate_hidden * 2, hidden);
    check_sfb(l2_sfb, hidden, intermediate_hidden);
    for (const auto* sec: {&l1_secondary, &l2_secondary}) {
        DG_HOST_ASSERT(sec->is_cuda() and sec->device() == y.device());
        DG_HOST_ASSERT(sec->scalar_type() == torch::kFloat and sec->is_contiguous());
        DG_HOST_ASSERT(sec->dim() == 1 and sec->size(0) == num_experts_per_rank);
    }

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Buffer checks: identical to the FP8 entry point (the symm buffer holds activations only)
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_experts == num_experts_per_rank * num_ranks);
    DG_HOST_ASSERT(num_tokens_bound >= 0);
    DG_HOST_ASSERT(num_tokens_bound == 0 or num_tokens_bound >= num_tokens);
    DG_HOST_ASSERT(num_tokens_bound <= num_max_tokens_per_rank);
    const auto [num_required_bytes, slice] = get_symm_buffer_layout_for_sm90_fused_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation,
        num_ring_tokens, l2_act_sf_gran_k);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // the FP32 weight-SF arguments of the FP8 launcher are unused on this path
    const auto unused_sf = torch::empty({0}, y.options().dtype(torch::kFloat));
    sm90_fp8_fused_mega_moe(y,
                            l1_acts, l1_acts_sf,
                            l2_acts, l2_acts_sf,
                            l1_weights, l2_weights,
                            unused_sf, unused_sf,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            num_ring_tokens, l2_act_sf_gran_k,
                            activation_clamp, fast_math,
                            num_tokens_bound, l2_lag_encoded,
                            std::make_tuple(l1_sfb, l1_secondary, l2_sfb, l2_secondary), split_k_hint);

    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
    m.def("get_token_alignment_for_sm90_fused_mega_moe", &get_token_alignment_for_sm90_fused_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_fused_mega_moe", &get_symm_buffer_size_for_sm90_fused_mega_moe);
    m.def("sm90_fused_fp8_mega_moe", &sm90_fused_fp8_mega_moe);
    m.def("sm90_fused_fp8_mxfp4_mega_moe", &sm90_fused_fp8_mxfp4_mega_moe);
}

} // namespace deep_gemm::sm90_fused_mega
