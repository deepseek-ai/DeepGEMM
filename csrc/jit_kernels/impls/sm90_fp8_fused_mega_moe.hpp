#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_fused_mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_fused_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime
// ----------------------------------------------------------------------------
// This is the SM90 counterpart of `SM100FP8FP4MegaMoERuntime`. The kernel
// itself lives in `deep_gemm/impls/sm90_fp8_fused_mega_moe.cuh`.
//
// Differences from SM100 path:
//   * Activations and weights are both FP8 (e4m3); no FP4.
//   * Activation/weight scale factors (SF) are float, not UE8M0 int + per-32
//     UTCCP layout. L1 activation SF and weight SF are per-128 K; the fused L1
//     epilogue writes the L2 activation SF at per-128 or per-64 K granularity (kL2ActSFK).
//   * No tensor memory: WGMMA accumulators are register-resident.
//   * Cluster size is at most 2 (TMA multicast on A); no 2-CTA UMMA.
// ============================================================================

class SM90FP8FusedMegaMoERuntime final : public LaunchRuntime<SM90FP8FusedMegaMoERuntime> {
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
        bool use_swap_ab;
        bool half_l2_cd;
        int num_ring_tokens;
        MegaMoESM90FusedConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        // in-kernel event trace buffer; nullptr = tracing compiled out (kTrace = false)
        void* trace_ptr;
        bool trace_epi;
        // L2 activation SF K granularity, 128 or 64
        int l2_act_sf_k;
        bool pdl;
        bool pull_eager_publish;

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
#include <deep_gemm/impls/sm90_fp8_fused_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fused_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
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
        {},
        {},
        {},
        {},
        {},
        {},
        {},
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
    // Must be the *effective* capacity (full pool when the caller passed 0):
    // the kernel derives `kRingCoversFullPool` and `kNumRingBlocks` from it.
    args.config.num_ring_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.epilogue_registers,
    args.reuse_accum_as_final ? "true" : "false",
    args.l2_arrival_counter ? "true" : "false",
    args.l2_epilogue_requires_full_sync ? "true" : "false",
    args.use_swap_ab ? "true" : "false",
    args.half_l2_cd ? "true" : "false",
    args.config.cluster_size,
    args.config.multicast_on_b ? "true" : "false",
    args.config.l2_lag_units,
    args.config.l2_cd_passes,
    args.trace_ptr != nullptr ? "true" : "false",
    args.trace_epi ? "true" : "false",
    args.l2_act_sf_k,
    args.config.l2_stage_mode,
    args.pdl ? "true" : "false",
    args.config.early_combine,
    args.config.pull_publish_batch,
    args.pull_eager_publish ? "true" : "false");
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
            args.l2_weights_sf,
            args.trace_ptr
        ));
    }
};

static void sm90_fp8_fused_mega_moe(
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
    const int& num_ring_tokens,
    const int& l2_act_sf_gran_k,
    const float& activation_clamp,
    const bool& fast_math,
    const int& num_tokens_bound = 0,
    const int& l2_lag_encoded = -1
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_max_pool_tokens_h = get_num_max_pool_tokens_sm90_fused(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    // SF pools are sized by the data pool (ring capacity when ringing), so the
    // padded token count must be derived from the same base as the allocation
    // side in `get_symm_buffer_size_for_sm90_fused_mega_moe`.
    const int num_data_pool_tokens =
        num_ring_tokens == 0 ? num_max_pool_tokens_h : num_ring_tokens;
    const int num_padded_sf_pool_tokens = get_num_padded_sf_pool_tokens_sm90_fused(num_data_pool_tokens);
    DG_HOST_ASSERT(static_cast<int>(l1_acts_sf.size(0)) == num_padded_sf_pool_tokens);

    // Heuristics
    const auto config = get_mega_moe_config_sm90_fused(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens, l2_act_sf_gran_k,
        num_ring_tokens, num_tokens_bound, l2_lag_encoded);
    // a BLOCK_M outside the SM90 candidate set would index SF rows past the pool sized above
    DG_HOST_ASSERT(is_candidate_block_m_sm90_fused(config.block_m) and
                   config.num_padded_sf_pool_tokens >= (num_data_pool_tokens / config.block_m) * align(config.block_m, 128));
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
        (config.num_epilogue_threads == 512 or config.num_epilogue_threads == 256);
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
    // the swapAB epilogues assume per-64 L2 activation scales (kernel static_assert); per-128 takes the non-swap path
    const bool use_swap_ab = config.block_n == 128 and l2_act_sf_gran_k == 64 and
        should_use_swap_ab_sm90_fused(
            num_experts_per_rank, num_tokens, num_topk,
            config.block_m, config.num_epilogue_threads, l2_act_sf_gran_k);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-128 or per-64 K for L2 (MN-major, no swizzle). Weight SF: raw float pointer.
    constexpr int kGranK = 128;
    DG_HOST_ASSERT(l2_act_sf_gran_k == 64 or l2_act_sf_gran_k == 128);
    const int kL2ActsSFGranK = l2_act_sf_gran_k;
    DG_HOST_ASSERT(static_cast<int>(l2_acts_sf.size(1)) == intermediate_hidden / kL2ActsSFGranK);
    DG_HOST_ASSERT(static_cast<int>(l1_acts.size(0)) == num_data_pool_tokens);
    DG_HOST_ASSERT(num_data_pool_tokens == config.num_ring_tokens);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, num_data_pool_tokens,
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
    if (not layout::is_sm90_fused_moe_combine_vectorization_legal(
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(config.num_epilogue_threads / 32),
            get_sm90_fused_pre_barrier_smem_size_for_combine(hidden, config, use_swap_ab),
            split_mn_warpgroups))
        DG_HOST_UNREACHABLE(
            "SM90 fused FP8 MegaMoE hidden size is incompatible with the selected combine vectorization");
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;
    const bool split_n_shares_sf =
        split_n_warpgroups and wg_l1_out_block_n < kL2ActsSFGranK;
    // The L1 fp8 output tile is staged in the TMA SWIZZLE_128B layout when a warpgroup's staging row is exactly 128 bytes;
    // the descriptor must use the layout the kernel stages (the kernel derives the same condition, kL1OutSwizzled).
    const int l1_output_swizzle_mode =
        (wg_l1_out_block_n == 128 and not split_n_shares_sf and not use_swap_ab) ? 128 : 0;
    const int l1_output_box_n = split_n_shares_sf ? config.block_n / 2 : wg_l1_out_block_n;
    const int l1_output_box_m = split_n_shares_sf ? config.block_m : wg_block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, num_data_pool_tokens,
                                                       l1_output_box_n, l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       l1_output_swizzle_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, num_data_pool_tokens,
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

    // opt-in in-kernel event trace: device address of the caller's [num_sms][4][kTraceSlots][2] uint64 buffer; 0 = off (kTrace false)
    void* trace_ptr = nullptr;
    if (const auto trace_env = get_env<std::string>("DG_SM90_TRACE_PTR"); not trace_env.empty()) {
        const auto trace_addr = std::stoull(trace_env);
        trace_ptr = trace_addr != 0 ? reinterpret_cast<void*>(static_cast<uintptr_t>(trace_addr)) : nullptr;
    }
    // epilogue sub-events in the trace; a role's slots overflow silently above kTraceSlots events
    const bool trace_epi = trace_ptr != nullptr and get_env<int>("DG_SM90_TRACE_EPI", 0) != 0;

    // Programmatic dependent launch: a launch that carries the attribute must run the instantiation with the matching griddepcontrol
    // wait + trigger (the wait-less cubin could read the pre-dispatch outputs before they are flushed); calls of at most
    // kSm90FusedPdlMaxTokens tokens per rank take the attribute on their own.
    const int rule_tokens_per_rank = sm90_fused_rule_tokens_per_rank(num_tokens, num_tokens_bound);
    const bool pdl = device_runtime->get_pdl() or rule_tokens_per_rank <= kSm90FusedPdlMaxTokens;

    const bool pull_eager_publish = config.pull_publish_batch == 1 and rule_tokens_per_rank <= kSm90FusedPullEagerMaxTokens;

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8FusedMegaMoERuntime::Args args = {
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
        .use_swap_ab = use_swap_ab,
        .half_l2_cd = config.half_l2_cd,
        .num_ring_tokens = num_ring_tokens,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .trace_ptr = trace_ptr,
        .trace_epi = trace_epi,
        .l2_act_sf_k = kL2ActsSFGranK,
        .pdl = pdl,
        .pull_eager_publish = pull_eager_publish,
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
                                  config.smem_size, config.cluster_size, /*enable_pdl=*/true, /*force_pdl=*/pdl)
    };
    const auto code = SM90FP8FusedMegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_fused_mega_moe", code);
    SM90FP8FusedMegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
