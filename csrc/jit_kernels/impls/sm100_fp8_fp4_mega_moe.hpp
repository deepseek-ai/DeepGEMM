#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_shared_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool use_situ;
        float situ_beta, situ_linear_beta;
        int shared_hidden, shared_intermediate_hidden;
        bool use_bf16_shared, apply_rms_norm;
        bool publish_shared_rs, publish_shared_sp_rs;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        void* shared_y;
        const uint32_t* shared_rs_flags;
        const int64_t* shared_rs_peer_ptrs;
        const void* rms_weight;
        float rms_epsilon;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        int num_shared_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        CUtensorMap tensor_map_shared_l1_acts;
        CUtensorMap tensor_map_shared_l1_acts_sf;
        CUtensorMap tensor_map_shared_l1_weights;
        CUtensorMap tensor_map_shared_l1_weights_sf;
        CUtensorMap tensor_map_shared_l1_output;
        CUtensorMap tensor_map_shared_l2_acts;
        CUtensorMap tensor_map_shared_l2_acts_sf;
        CUtensorMap tensor_map_shared_l2_weights;
        CUtensorMap tensor_map_shared_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
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
        {}, {},
        {}, {}, {}, {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_shared_experts,
    args.num_topk,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_ring_tokens,
    args.config.num_sf_ring_tokens,
    args.config.num_stages,
    args.config.num_bytes_per_pull,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.use_situ ? "true" : "false",
    to_string(args.situ_beta), to_string(args.situ_linear_beta),
    args.shared_hidden, args.shared_intermediate_hidden,
    args.use_bf16_shared ? "true" : "false",
    args.apply_rms_norm ? "true" : "false",
    args.publish_shared_rs ? "true" : "false",
    args.publish_shared_sp_rs ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.shared_y,
            args.shared_rs_flags,
            args.shared_rs_peer_ptrs,
            args.rms_weight,
            args.rms_epsilon,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.num_shared_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            args.tensor_map_shared_l1_acts,
            args.tensor_map_shared_l1_acts_sf,
            args.tensor_map_shared_l1_weights,
            args.tensor_map_shared_l1_weights_sf,
            args.tensor_map_shared_l1_output,
            args.tensor_map_shared_l2_acts,
            args.tensor_map_shared_l2_acts_sf,
            args.tensor_map_shared_l2_weights,
            args.tensor_map_shared_l2_weights_sf
        ));
    }
};

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& use_situ,
    const float& situ_beta,
    const float& situ_linear_beta,
    const torch::Tensor& bf16_shared_l1_acts = torch::Tensor(),
    const torch::Tensor& bf16_shared_l2_acts = torch::Tensor(),
    const torch::Tensor& bf16_shared_l1_weights = torch::Tensor(),
    const torch::Tensor& bf16_shared_l2_weights = torch::Tensor(),
    const torch::Tensor& bf16_shared_y = torch::Tensor(),
    const torch::Tensor& rms_weight = torch::Tensor(),
    const float& rms_epsilon = 0.0f,
    const bool& use_bf16_shared = false,
    const bool& apply_rms_norm = false,
    const torch::Tensor& shared_rs_flags = torch::Tensor(),
    const torch::Tensor& shared_rs_peer_ptrs = torch::Tensor(),
    const bool& publish_shared_rs = false,
    const bool& publish_shared_sp_rs = false
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_hidden = use_bf16_shared ?
        static_cast<int>(bf16_shared_l1_acts.size(1)) : hidden;
    const auto shared_intermediate_hidden = use_bf16_shared ?
        static_cast<int>(bf16_shared_l2_acts.size(1)) :
        intermediate_hidden * num_shared_experts;
    const auto num_shared_tokens = use_bf16_shared ?
        static_cast<int>(bf16_shared_l1_acts.size(0)) : num_tokens;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4);

    // Make tensormap
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);

    // BF16 shared tiles use half of routed K so that their two-byte elements
    // fit the existing FP8/expanded-FP4 shared-memory tiles.
    const auto shared_block_k = use_bf16_shared ? config.block_k / 2 : config.block_k;
    const auto has_shared_work = num_shared_experts > 0 and num_shared_tokens > 0;
    const auto& actual_shared_l1_acts = use_bf16_shared ? bf16_shared_l1_acts : shared_l1_acts;
    const auto& actual_shared_l2_acts = use_bf16_shared ? bf16_shared_l2_acts : shared_l2_acts;
    const auto& actual_shared_l1_weights = use_bf16_shared ? bf16_shared_l1_weights : shared_l1_weights;
    const auto& actual_shared_l2_weights = use_bf16_shared ? bf16_shared_l2_weights : shared_l2_weights;
    const auto tensor_map_shared_l1_acts = has_shared_work ? make_tma_2d_desc(
        actual_shared_l1_acts,
        shared_hidden, use_bf16_shared ? num_shared_tokens : num_max_tokens_per_rank,
        shared_block_k, config.load_block_m,
        static_cast<int>(actual_shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = has_shared_work and not use_bf16_shared ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = has_shared_work ? make_tma_2d_desc(
        actual_shared_l1_weights,
        shared_hidden, shared_intermediate_hidden * 2,
        shared_block_k, config.load_block_n,
        static_cast<int>(actual_shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = has_shared_work and not use_bf16_shared ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, shared_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = has_shared_work ? make_tma_2d_desc(
        actual_shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m,
        static_cast<int>(actual_shared_l2_acts.stride(-2)),
        use_bf16_shared ? config.swizzle_acts_mode : config.swizzle_acts_mode / 2) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = has_shared_work ? make_tma_2d_desc(
        actual_shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        shared_block_k, config.load_block_m,
        static_cast<int>(actual_shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = has_shared_work and not use_bf16_shared ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = has_shared_work ? make_tma_2d_desc(
        actual_shared_l2_weights,
        shared_intermediate_hidden, shared_hidden,
        shared_block_k, config.load_block_n,
        static_cast<int>(actual_shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = has_shared_work and not use_bf16_shared ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        shared_hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_shared_experts = num_shared_experts,
        .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .use_situ = use_situ,
        .situ_beta = situ_beta,
        .situ_linear_beta = situ_linear_beta,
        .shared_hidden = shared_hidden,
        .shared_intermediate_hidden = shared_intermediate_hidden,
        .use_bf16_shared = use_bf16_shared,
        .apply_rms_norm = apply_rms_norm,
        .publish_shared_rs = publish_shared_rs,
        .publish_shared_sp_rs = publish_shared_sp_rs,
        .config = config,
        .y = y.data_ptr(),
        .shared_y = use_bf16_shared and not publish_shared_rs ? bf16_shared_y.data_ptr() : nullptr,
        .shared_rs_flags = publish_shared_rs ?
            reinterpret_cast<const uint32_t*>(shared_rs_flags.data_ptr<int32_t>()) : nullptr,
        .shared_rs_peer_ptrs = publish_shared_rs ?
            shared_rs_peer_ptrs.data_ptr<int64_t>() : nullptr,
        .rms_weight = apply_rms_norm ? rms_weight.data_ptr() : nullptr,
        .rms_epsilon = rms_epsilon,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .num_shared_tokens = num_shared_tokens,
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
        .tensor_map_shared_l1_acts = tensor_map_shared_l1_acts,
        .tensor_map_shared_l1_acts_sf = tensor_map_shared_l1_acts_sf,
        .tensor_map_shared_l1_weights = tensor_map_shared_l1_weights,
        .tensor_map_shared_l1_weights_sf = tensor_map_shared_l1_weights_sf,
        .tensor_map_shared_l1_output = tensor_map_shared_l1_output,
        .tensor_map_shared_l2_acts = tensor_map_shared_l2_acts,
        .tensor_map_shared_l2_acts_sf = tensor_map_shared_l2_acts_sf,
        .tensor_map_shared_l2_weights = tensor_map_shared_l2_weights,
        .tensor_map_shared_l2_weights_sf = tensor_map_shared_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
