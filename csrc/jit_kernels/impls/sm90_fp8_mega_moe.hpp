#pragma once

#include <torch/python.h>
#include <cstdlib>
#include <string>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime
// ----------------------------------------------------------------------------
// This is the SM90 counterpart of `SM100FP8FP4MegaMoERuntime`. The kernel
// itself lives in `deep_gemm/impls/sm90_fp8_mega_moe.cuh` and is currently a
// skeleton: dispatch/combine paths are intended to be portable from the SM100
// version, while the GEMM (TMA load + WGMMA + epilogue) is being implemented
// in a follow-up step.
//
// Differences from SM100 path:
//   * Activations and weights are both FP8 (e4m3); no FP4.
//   * Activation/weight scale factors (SF) are per-128-channel float (not UE8M0
//     int + per-32 UTCCP layout).
//   * No tensor memory: WGMMA accumulators are register-resident.
//   * Cluster size is at most 2 (TMA multicast on A); no 2-CTA UMMA.
// ============================================================================

class SM90FP8MegaMoERuntime final : public LaunchRuntime<SM90FP8MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps (same set as SM100, but SF descriptors use float per-128 layout)
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const char* dbg_env = std::getenv("DG_DEBUG_SCHED_TRACE");
        const bool dbg_on = dbg_env != nullptr && std::string(dbg_env) != "0";
        return fmt::format(R"(
// dbg_trace_v4_idle_warp_dealloc (bump to invalidate JIT cache when sm90_fp8_mega_moe.cuh changes)
{}#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", dbg_on ? "#define DG_DEBUG_SCHED_TRACE\n" : "",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf
        ));
    }
};

static void sm90_fp8_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // SF: per-128 channel float, MN-major, no swizzle (gran_k = 128 for SM90).
    // Exception: L2 *activation* SF uses gran_k = 64 so that each L1 epilogue
    // block (which produces 64 post-SwiGLU columns and quantises them with its
    // own amax) can write its SF independently without cross-CTA reduction.
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);
    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The TMA store is issued *per warpgroup*, each writing a `WG_BLOCK_M`
    // (= block_m / num_epilogue_warpgroups) row tile from its own SMEM offset.
    // The descriptor outer-box dim therefore must be `WG_BLOCK_M`, not block_m.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const int wg_block_m = config.block_m / num_epilogue_warpgroups_h;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, wg_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_mega_moe", code);
    SM90FP8MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
