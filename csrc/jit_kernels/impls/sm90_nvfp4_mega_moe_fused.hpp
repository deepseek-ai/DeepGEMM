#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_nvfp4_mega_moe.hpp"

namespace deep_gemm {

class SM90NVFP4FusedRuntime final
    : public LaunchRuntime<SM90NVFP4FusedRuntime> {
public:
    struct Args {
        int num_ranks;
        int num_experts;
        int num_max_tokens_per_rank;
        int hidden;
        int intermediate_hidden;
        int num_topk;
        float activation_clamp;
        bool fast_math;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
        bool use_interleaved_scheduler;
        SM90NVFP4FusedConfig config;

        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l1_global_scales;
        const float* l2_global_scales;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const auto barrier_timeout_mode =
            get_env<int>("DG_SM90_NVFP4_MOE_BARRIER_DEBUG")
            ? fmt::format(
                  "#define DG_SM90_NVFP4_MOE_COUNTER_DEBUG 1\n"
                  "#define DG_SM90_NVFP4_MOE_PROGRESS_MASK {}\n"
                  "#define DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS {}\n",
                  get_env<int>("DG_SM90_NVFP4_MOE_PROGRESS_MASK", 0xffff),
                  get_env<int>("DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS", 60))
            : "#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1\n";
        const std::string kernel_header = fmt::format(
            "{}#include <deep_gemm/impls/"
            "sm90_nvfp4_mega_moe_fused.cuh>",
            barrier_timeout_mode);
        const std::string policy_template_args = fmt::format(
            "/* kSwapABRequested */ {},\n"
            "        /* kSingleActiveDispatchWarp */ {},\n"
            "        /* kUseMode2RowDecoder */ {},\n"
            "        /* kUseInterleavedScheduler */ {}",
            args.swap_ab ? "true" : "false",
            args.single_active_dispatch_warp ? "true" : "false",
            args.use_mode2_row_decoder ? "true" : "false",
            args.use_interleaved_scheduler ? "true" : "false");
        return fmt::format(R"(
{}

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        /* kNumMaxTokensPerRank */ {},
        /* BLOCK_M */ {},
        /* BLOCK_N */ {},
        /* kNumRingTokens */ {},
        /* kNumPaddedSFPoolTokens */ {},
        /* kNumStages */ {},
        /* kActivationClamp */ {},
        /* kFastMath */ {},
        {},
        /* kNumSMs */ {},
        /* kNumRanks */ {},
        /* kNumExperts */ {},
        /* kHidden */ {},
        /* kIntermediateHidden */ {},
        /* kNumTopk */ {}
    >);
}};
)",
            kernel_header,
            "sm90_nvfp4_mega_moe_fused_impl",
            args.num_max_tokens_per_rank,
            args.config.block_m,
            args.config.block_n,
            args.config.num_ring_tokens,
            args.config.num_padded_sf_pool_tokens,
            args.config.num_stages,
            to_string(args.activation_clamp),
            args.fast_math ? "true" : "false",
            policy_template_args,
            args.launch_args.grid_dim.first,
            args.num_ranks,
            args.num_experts,
            args.hidden,
            args.intermediate_hidden,
            args.num_topk);
    }

    static void launch_impl(
            const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l1_global_scales,
            args.l2_global_scales));
    }
};

static void sm90_nvfp4_fused_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> l1_global_scales,
    const std::optional<torch::Tensor> l2_global_scales,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int num_experts = num_experts_per_rank * num_ranks;
    const int num_sms = device_runtime->get_num_sms();
    const int num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));
    const SM90NVFP4FusedInput heuristic_input {
        num_sms,
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
    };
    const auto plan = select_sm90_nvfp4_fused(heuristic_input);
    const auto& config = plan.config;
    using KernelConfig = SM90NVFP4FusedConfig;
    DG_HOST_ASSERT((config.block_m == 8 || config.block_m == 16 ||
                    config.block_m == 24 || config.block_m == 64 ||
                    config.block_m == 128));
    DG_HOST_ASSERT(config.block_n == 128 || config.block_n == 256);
    DG_HOST_ASSERT(plan.swap_ab == (num_tokens <= 64));

    constexpr int kL1ScaleGranK = 128;
    const int l2_scale_gran_k = config.block_n / 2;
    const auto tensor_map_l1_acts = make_tma_2d_desc(
        l1_acts, hidden, config.num_ring_tokens,
        KernelConfig::kBlockK, config.block_m,
        static_cast<int>(l1_acts.stride(-2)), KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_acts_sf,
        config.num_padded_sf_pool_tokens, hidden,
        config.block_m, kL1ScaleGranK, 1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, static_cast<int>(l1_weights.size(2)),
        num_experts_per_rank * intermediate_hidden * 2,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l1_weights.stride(-2)), 0);

    const int l1_output_store_block_n = config.block_n / 2;
    const auto tensor_map_l1_output = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_ring_tokens,
        l1_output_store_block_n, config.block_m,
        static_cast<int>(l2_acts.stride(-2)), 0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_ring_tokens,
        KernelConfig::kBlockK, config.block_m,
        static_cast<int>(l2_acts.stride(-2)), KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l2_acts_sf,
        config.num_padded_sf_pool_tokens, intermediate_hidden,
        config.block_m, l2_scale_gran_k, 1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights, static_cast<int>(l2_weights.size(2)),
        num_experts_per_rank * hidden,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l2_weights.stride(-2)), 0);
    int* cumulative_stats_ptr = cumulative_local_expert_recv_stats.has_value() ?
        cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr;
    const float* l1_global_scales_ptr = l1_global_scales.has_value() ?
        l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr = l2_global_scales.has_value() ?
        l2_global_scales->data_ptr<float>() : nullptr;

    const SM90NVFP4FusedRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_experts = num_experts,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_topk = num_topk,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .swap_ab = plan.swap_ab,
        .use_mode2_row_decoder = plan.use_mode2_row_decoder,
        .single_active_dispatch_warp = plan.single_active_dispatch_warp,
        .use_interleaved_scheduler = plan.use_interleaved_scheduler,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l1_global_scales = l1_global_scales_ptr,
        .l2_global_scales = l2_global_scales_ptr,
        .launch_args = LaunchArgs(
            num_sms,
            KernelConfig::kNumThreads,
            config.smem_size, 1)
    };

    const auto code = SM90NVFP4FusedRuntime::generate(args);
    const auto runtime = compiler->build(
        plan.use_interleaved_scheduler ?
            "sm90_nvfp4_fused_interleaved" :
            (plan.use_mode2_row_decoder ?
                "sm90_nvfp4_fused_mode2_row" :
                "sm90_nvfp4_fused_lut_window"),
        code);
    SM90NVFP4FusedRuntime::launch(runtime, args);
}

}  // namespace deep_gemm
