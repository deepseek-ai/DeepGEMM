#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime
// ----------------------------------------------------------------------------
// This is the SM90 counterpart of `SM100FP8FP4MegaMoERuntime`. The kernel
// itself lives in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`.
//
// Differences from SM100 path:
//   * Activations and weights are both FP8 (e4m3); no FP4.
//   * Activation/weight scale factors (SF) are float, not UE8M0 int + per-32
//     UTCCP layout. L1 activation SF and weight SF are per-128 K; the fused L1
//     epilogue writes L2 activation SF at per-64 K granularity.
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
        int epilogue_registers;
        bool reuse_accum_as_final;
        bool l2_arrival_counter;
        bool l2_epilogue_requires_full_sync;
        bool split_phase_hot_path;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations and weights. Weight scale factors use
        // block (128, 128) quantization and are loaded by the math warpgroup
        // directly from global memory (no TMA descriptor required).
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const float* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

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
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
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
    args.fast_math ? "true" : "false",
    args.epilogue_registers,
    args.reuse_accum_as_final ? "true" : "false",
    args.l2_arrival_counter ? "true" : "false",
    args.l2_epilogue_requires_full_sync ? "true" : "false",
    args.split_phase_hot_path ? "true" : "false");
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
            args.l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l2_weights_sf
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
    const int default_epilogue_registers =
        config.num_epilogue_threads == 512 ? 112 : 0;
    const int epilogue_registers = default_epilogue_registers;
    if (epilogue_registers > 0) {
        const int dispatch_registers =
            config.num_epilogue_threads == 512 ? 32 : 48;
        const int non_epilogue_registers =
            config.num_epilogue_threads == 512 ? 24 : 40;
        DG_HOST_ASSERT(dispatch_registers * config.num_dispatch_threads +
                       non_epilogue_registers * config.num_non_epilogue_threads +
                       epilogue_registers * config.num_epilogue_threads <= 64512);
    }
    const bool reuse_accum_as_final = config.block_m == 128;
    const bool default_split_mn_barrier_opt =
        config.block_m == 128 and config.block_n == 256 and
        config.num_epilogue_threads == 512;
    const bool split_phase_hot_path =
        config.block_m == 128 and config.block_n == 256 and hidden >= 7168;
    const bool decode_split_n_path =
        config.block_m == 64 and config.num_epilogue_threads == 256;
    const bool decode_split_n_bn256 =
        decode_split_n_path and config.block_n == 256;
    const bool decode_l2_counter =
        decode_split_n_bn256 and num_tokens >= 4 and num_tokens <= 128;
    const bool l2_arrival_counter =
        default_split_mn_barrier_opt or decode_l2_counter;
    const bool l2_epilogue_requires_full_sync =
        not l2_arrival_counter;

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
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
    const int weight_tma_block_n = config.block_n > 256 ? 256 : config.block_n;
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, weight_tma_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    // L1 output (post-SwiGLU FP8): N is halved. The correctness path stages
    // this tile in plain row-major SMEM before the TMA store. Later L2 TMA
    // loads may still swizzle from this row-major global buffer into their own
    // SMEM tile.
    // The usual TMA store is issued per warpgroup, each writing a `WG_BLOCK_M`
    // row tile from its own SMEM offset. The m64n128 2-WG split-N decode path is
    // different: both warpgroups stage one joint 64-column L1-output tile and a
    // single warpgroup issues the combined store, so the descriptor must cover
    // the full block_m x (block_n / 2) tile.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups =
        config.block_m == 64 and num_epilogue_warpgroups_h > 1 and
        config.block_n % num_epilogue_warpgroups_h == 0 and
        (config.block_n / num_epilogue_warpgroups_h == 64 or
         config.block_n / num_epilogue_warpgroups_h == 128);
    const bool split_mn_warpgroups =
        config.block_m == 128 and config.block_n == 256 and num_epilogue_warpgroups_h == 4;
    const int wg_split_m = split_n_warpgroups ? 1 :
        (split_mn_warpgroups ? 2 : num_epilogue_warpgroups_h);
    const int wg_split_n = split_n_warpgroups ? num_epilogue_warpgroups_h :
        (split_mn_warpgroups ? 2 : 1);
    DG_HOST_ASSERT(wg_split_m * wg_split_n == num_epilogue_warpgroups_h);
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;
    const bool split_n_shares_sf =
        split_n_warpgroups and wg_l1_out_block_n < kL2ActsSFGranK;
    const int l1_output_swizzle_mode = 0;
    const int l1_output_box_n =
        split_n_shares_sf ? config.block_n / 2 : wg_l1_out_block_n;
    const int l1_output_box_m =
        split_n_shares_sf ? config.block_m : wg_block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       l1_output_box_n, l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       l1_output_swizzle_mode);
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
                                                        config.block_k, weight_tma_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);

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
        .epilogue_registers = epilogue_registers,
        .reuse_accum_as_final = reuse_accum_as_final,
        .l2_arrival_counter = l2_arrival_counter,
        .l2_epilogue_requires_full_sync = l2_epilogue_requires_full_sync,
        .split_phase_hot_path = split_phase_hot_path,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = l1_weights_sf.data_ptr<float>(),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = l2_weights_sf.data_ptr<float>(),
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_mega_moe", code);
    SM90FP8MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
