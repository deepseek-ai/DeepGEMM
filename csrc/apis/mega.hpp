#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <pybind11/functional.h>

#include <deep_gemm/common/types.cuh>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm100_bf16_mega_moe.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"

namespace deep_gemm::mega {

static int get_token_alignment_for_mega_moe() {
    return layout::kLCMCandidateBlockM;
}

static std::pair<int, int> get_ring_limit_for_mega_moe(
    const int& num_max_tokens_per_rank, const int& num_experts_per_rank, const int& num_topk, const int& num_ranks) {
    return {
        get_num_wave_pool_tokens(num_ranks, num_topk, num_max_tokens_per_rank, 1, layout::kLCMCandidateBlockM),
        get_num_wave_pool_tokens(num_ranks, num_topk, num_max_tokens_per_rank, num_experts_per_rank, layout::kLCMCandidateBlockM)
    };
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& mma_type, const std::string& activation,
    const int& num_ring_tokens) {
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(activation == "swiglu");

    // Pool capacity must fit at least one full wave (one expert per wave) and aligned to block size
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto [num_min_ring_tokens, num_max_ring_tokens] =
        get_ring_limit_for_mega_moe(num_max_tokens_per_rank, num_experts_per_rank, num_topk, num_ranks);
    DG_HOST_ASSERT(num_ring_tokens % layout::kLCMCandidateBlockM == 0);
    DG_HOST_ASSERT(num_min_ring_tokens <= num_ring_tokens and num_ring_tokens <= num_max_ring_tokens);

    // Parse MMA type
    const auto mma_kind = parse_mma_kind(mma_type);
    const auto num_mma_elem_bytes = get_num_mma_elem_bytes(mma_kind);
    const auto with_sf = is_mma_with_sf(mma_kind);

    // Workspace
    const auto workspace = layout::Workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_ring_tokens);

    // Layouts
    const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
    const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0);
    const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        input_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        input_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Padded SF pool tokens
    int num_sf_ring_tokens = 0;
    for (int block_m: layout::kCandidateBlockM) {
        num_sf_ring_tokens = std::max(
            num_sf_ring_tokens,
            layout::get_num_sf_ring_tokens(num_ring_tokens, block_m)
        );
    }

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        input_token_layout, 1, num_ring_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        input_sf_layout, 1, num_sf_ring_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_ring_tokens,
        with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        intermediate_token_layout, 1, num_ring_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        intermediate_sf_layout, 1, num_sf_ring_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());

    // Check SF buffer requirements
    if (with_sf) {
        DG_HOST_ASSERT(hidden % 128 == 0 and intermediate_hidden % 128 == 0);
        DG_HOST_ASSERT(num_sf_ring_tokens % 4 == 0);
    }

    // Slice function: creates `(x, x_sf, topk_weights, topk_idx, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf)` tensor views from the raw buffer
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto x_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
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
            {num_ring_tokens, hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l1_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_sf_ring_tokens, hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_ring_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(with_sf ? torch::kFloat8_e4m3fn : torch::kBFloat16).device(buffer.device()));
        auto l2_acts_sf = with_sf ? torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_sf_ring_tokens, intermediate_hidden / 128},
            {1, num_sf_ring_tokens},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device())) : torch::Tensor();
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()), slice_input_buffers};
}

static void fp8_fp4_mega_moe(
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
    const int& num_ring_tokens
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(l1_weights, cute::UMMA::Major::K, arch_major);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] =
        check_grouped_ab_fp8_fp4(l2_weights, cute::UMMA::Major::K, arch_major);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check weight SF layout for UE8M0 packing, MN-major, and TMA alignment
    constexpr int kGranMN = 1, kGranK = 32;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, true, false, torch::kInt);

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "fp8xfp4", activation, num_ring_tokens);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // Adaptive heuristics consume a per-iteration receive-count distribution.
    // The public tensor is cumulative, so using its absolute values makes the
    // selected JIT config drift as the counter grows. Snapshot the cumulative
    // values and pass only a valid delta from the previous snapshot. Sampling is
    // opt-in, restricted to the calibrated tokens/expert band, and amortized
    // over 256 launches; the cached distribution also avoids host synchronization
    // and JIT-config churn on the steady-state path. Each logical counter tensor
    // has an independent bounded cache entry, so alternating MegaMoE layers do
    // not invalidate one another. The first call, counter resets, and zero deltas
    // fall back to the default heuristic until a new valid delta is observed.
    const int* host_recv_stats_ptr = nullptr;
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens * num_topk) / num_experts_per_rank;
    const bool is_calibrated_shape =
        num_ranks == 8 and num_experts == 256 and num_experts_per_rank == 32 and num_topk == 8 and
        hidden == 7168 and intermediate_hidden == 2048;
    const bool use_adaptive_stats =
        get_env<int>("DG_MEGA_MOE_ADAPTIVE_WAVE", 0) != 0 and
        is_calibrated_shape and
        expected_tokens_per_expert > 127.5f and expected_tokens_per_expert <= 128.5f;
    if (use_adaptive_stats and cumulative_local_expert_recv_stats.has_value()) {
        const auto* device_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        struct AdaptiveRecvStatsKey {
            const void* tensor_impl;
            const int* device_ptr;

            bool operator==(const AdaptiveRecvStatsKey& other) const {
                return tensor_impl == other.tensor_impl and device_ptr == other.device_ptr;
            }
        };
        struct AdaptiveRecvStatsKeyHash {
            size_t operator()(const AdaptiveRecvStatsKey& key) const {
                const auto tensor_hash = std::hash<const void*>{}(key.tensor_impl);
                const auto data_hash = std::hash<const int*>{}(key.device_ptr);
                return tensor_hash ^ (data_hash + (tensor_hash << 6) + (tensor_hash >> 2));
            }
        };
        struct AdaptiveRecvStatsCache {
            // Retain the tensor identity while this entry exists. Besides making
            // the logical identity explicit, this prevents its TensorImpl and
            // CUDA allocation from being recycled into a false cache hit.
            torch::Tensor counter_identity;
            std::vector<int> previous_cumulative;
            std::vector<int> cached_delta;
            int calls_since_refresh = 0;
            bool has_previous_snapshot = false;
            bool sample_next_call = false;
            bool has_cached_delta = false;
        };
        static thread_local std::unordered_map<
            AdaptiveRecvStatsKey, AdaptiveRecvStatsCache,
            AdaptiveRecvStatsKeyHash> caches;

        // The B200 A/B calibration used a stationary routing distribution. A
        // 256-launch interval amortizes the synchronous D2H copy while still
        // periodically detecting workload changes. Keep the cache bounded for
        // models that create transient counter tensors on a long-lived thread.
        constexpr int kRefreshInterval = 256;
        constexpr size_t kMaxCachedCounters = 64;
        const AdaptiveRecvStatsKey cache_key = {
            cumulative_local_expert_recv_stats->unsafeGetTensorImpl(), device_ptr};
        auto cache_it = caches.find(cache_key);
        if (cache_it == caches.end()) {
            if (caches.size() >= kMaxCachedCounters)
                caches.clear();
            cache_it = caches.try_emplace(cache_key).first;
            cache_it->second.counter_identity = *cumulative_local_expert_recv_stats;
        }
        auto& cache = cache_it->second;

        // Count this launch before testing the periodic deadline so refreshes
        // are exactly kRefreshInterval launches apart after initialization.
        ++ cache.calls_since_refresh;
        const bool has_matching_snapshot = cache.has_previous_snapshot and
            cache.previous_cumulative.size() ==
                static_cast<size_t>(cumulative_local_expert_recv_stats->numel());
        const bool should_refresh = not has_matching_snapshot or
            cache.sample_next_call or cache.calls_since_refresh >= kRefreshInterval;

        if (should_refresh) {
            const auto cpu = cumulative_local_expert_recv_stats->to(torch::kCPU, torch::kInt);
            const auto* current_ptr = cpu.data_ptr<int>();
            bool valid_delta = has_matching_snapshot;
            int64_t delta_sum = 0;
            cache.cached_delta.resize(cpu.numel());
            if (valid_delta) {
                for (int i = 0; i < cpu.numel(); ++ i) {
                    const int delta = current_ptr[i] - cache.previous_cumulative[i];
                    if (delta < 0) {
                        valid_delta = false;
                        break;
                    }
                    cache.cached_delta[i] = delta;
                    delta_sum += delta;
                }
            }

            cache.previous_cumulative.assign(current_ptr, current_ptr + cpu.numel());
            cache.has_previous_snapshot = true;
            // Only the initial snapshot samples again immediately so adaptivity
            // can start on the next launch. Zero deltas and counter resets back
            // off for the regular interval instead of synchronizing every call.
            cache.sample_next_call = not has_matching_snapshot;
            cache.has_cached_delta = valid_delta and delta_sum > 0;
            cache.calls_since_refresh = 0;
        }

        if (cache.has_cached_delta)
            host_recv_stats_ptr = cache.cached_delta.data();
    }

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y,
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
                               activation_clamp, fast_math,
                               host_recv_stats_ptr);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const int& num_ring_tokens
) {
    // Config checks
    const auto num_tokens = static_cast<int>(y.size(0));
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        "bf16xbf16", activation, num_ring_tokens);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, _x_sf, topk_idx, topk_weights, l1_acts, _l1_acts_sf, l2_acts, _l2_acts_sf] = slice(sym_buffer);

    // Dispatch into different architectures
    if (arch_major == 10) {
        sm100_bf16_mega_moe(y,
                            l1_acts, l2_acts, 
                            l1_weights, l2_weights,
                            cumulative_local_expert_recv_stats,
                            sym_buffer_ptrs,
                            rank_idx, num_max_tokens_per_rank,
                            num_experts_per_rank,
                            num_tokens, num_topk,
                            hidden, intermediate_hidden,
                            activation_clamp, fast_math);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Zero the entire symmetric buffer for debug mode
    // NOTES: caller must re-copy inputs into the buffer before each kernel call
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_ring_limit_for_mega_moe", &get_ring_limit_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe);
    m.def("bf16_mega_moe", &bf16_mega_moe);
#endif
}

} // namespace deep_gemm::mega
