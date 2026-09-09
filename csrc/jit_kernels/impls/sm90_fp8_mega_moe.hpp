#pragma once

#include <torch/all.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_mega_moe.hpp"

namespace deep_gemm {

// SM90 uses FP8 operands, float scale factors, register-resident WGMMA
// accumulators, and one CTA per scheduled work item.

class SM90FP8MegaMoERuntime final : public LaunchRuntime<SM90FP8MegaMoERuntime> {
public:
    enum class KernelPhase {
        Linear1,
        Linear2
    };

    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool bf16_scaled_accum;
        KernelPhase kernel_phase;
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
        const char* kernel_symbol = args.kernel_phase == KernelPhase::Linear1 ? "sm90_fp8_mega_moe_l1_impl" :
            "sm90_fp8_mega_moe_l2_impl";
        const auto phase_template_args = args.kernel_phase == KernelPhase::Linear1 ?
            fmt::format(",\n        {}", args.config.nmajor_schedule ? "true" : "false") :
            fmt::format(",\n        {}, {}, {}",
                        args.config.direct_l2_scatter ? "true" : "false",
                        args.config.nmajor_schedule ? "true" : "false",
                        args.config.one_warp_cleanup ? "true" : "false");
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {}{}
    >);
}};
)",
    kernel_symbol,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.sf_pool_stride_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.num_sms, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.config.swap_ab ? "true" : "false",
    args.bf16_scaled_accum ? "true" : "false",
    phase_template_args);
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

    // Resolve hardware, generic fallback, phase schedules, and numerical modes
    // once. The runtime only consumes the resulting complete launch config.
    const int num_sms = device_runtime->get_num_sms();
    const Sm90MoeHeuristicInput heuristic_input {
        num_sms,
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden,
        num_padded_sf_pool_tokens
    };
    const auto launch_config = select_mega_moe_sm90(heuristic_input);
    const auto& l1_config = launch_config.l1;
    const auto& l2_config = launch_config.l2;
    const auto combine_smem_before_barriers =
        get_sm90_moe_pre_barrier_smem_size_for_combine(
            heuristic_input, l2_config);
    if (not layout::is_sm90_moe_combine_vectorization_legal(
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(l2_config.num_epilogue_threads / 32),
            combine_smem_before_barriers,
            static_cast<uint32_t>(sizeof(nv_bfloat16)),
            static_cast<uint32_t>(sizeof(nv_bfloat16))))
        DG_HOST_UNREACHABLE(
            "SM90 FP8 MegaMoE hidden size is incompatible with the selected combine vectorization");

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    constexpr int kTMATileK = 128;
    constexpr int kMaxTMABoxN = 256;
    // A BK256 pipeline stage is represented in shared memory as two adjacent
    // independently-swizzled BK128 TMA tiles. Keep the tensor-map box at 128
    // and issue two copies; the kernel config/scheduler still advances by 256.
    const int l1_tma_block_k = std::min(l1_config.block_k, kTMATileK);
    const int l2_tma_block_k = std::min(l2_config.block_k, kTMATileK);
    const int l1_tma_block_n = std::min(l1_config.block_n, kMaxTMABoxN);
    const int l2_tma_block_n = std::min(l2_config.block_n, kMaxTMABoxN);
    DG_HOST_ASSERT(l1_config.block_k % l1_tma_block_k == 0 and
                   l2_config.block_k % l2_tma_block_k == 0);
    DG_HOST_ASSERT(l1_config.block_n <= 2 * kMaxTMABoxN and
                   l2_config.block_n <= 2 * kMaxTMABoxN and
                   l1_config.block_n % l1_tma_block_n == 0 and
                   l2_config.block_n % l2_tma_block_n == 0);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, l1_config.num_max_pool_tokens,
                                                     l1_tma_block_k, l1_config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     128);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        l1_config.sf_pool_stride_tokens, hidden,
                                                        l1_config.block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        l1_tma_block_k, l1_tma_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        128);
    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The default TMA store is issued per warpgroup, each writing a WG_BLOCK_M
    // row tile. In split-N mode, two WGs produce different N halves of the same
    // M rows, then one TMA store writes the full 64x128 post-SwiGLU tile.
    const int num_epilogue_warpgroups_h = l1_config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        l1_config.block_m, l1_config.block_n, num_epilogue_warpgroups_h);
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    const int wg_l1_out_block_n = wg_block_n / 2;
    const int l1_output_box_n = wg_layout.split_n ? l1_config.block_n / 2 : wg_l1_out_block_n;
    const int l1_output_box_m = wg_layout.split_n ? l1_config.block_m : wg_block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, l1_config.num_max_pool_tokens,
                                                       l1_output_box_n, l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, l2_config.num_max_pool_tokens,
                                                     l2_tma_block_k, l2_config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     128);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        l2_config.sf_pool_stride_tokens, intermediate_hidden,
                                                        l2_config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        l2_tma_block_k, l2_tma_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        128);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const bool bf16_scaled_accum = launch_config.numerical.bf16_scaled_accum;
    const SM90FP8MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .bf16_scaled_accum = bf16_scaled_accum,
        .kernel_phase = SM90FP8MegaMoERuntime::KernelPhase::Linear1,
        .config = l1_config,
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
        .launch_args = LaunchArgs(l1_config.num_sms,
                                  l1_config.num_dispatch_threads + l1_config.num_non_epilogue_threads +
                                      l1_config.num_epilogue_threads,
                                  l1_config.smem_size, 1)
    };
    const auto launch_with_phase = [&](const SM90FP8MegaMoERuntime::KernelPhase kernel_phase,
                                       const char* kernel_name) {
        auto split_args = args;
        split_args.kernel_phase = kernel_phase;
        const bool is_linear2 =
            kernel_phase == SM90FP8MegaMoERuntime::KernelPhase::Linear2;
        split_args.config = is_linear2 ? l2_config : l1_config;
        split_args.launch_args = LaunchArgs(
            split_args.config.num_sms,
            split_args.config.num_dispatch_threads + split_args.config.num_non_epilogue_threads +
                split_args.config.num_epilogue_threads,
            split_args.config.smem_size, 1);
        const auto code = SM90FP8MegaMoERuntime::generate(split_args);
        const auto runtime = compiler->build(kernel_name, code);
        SM90FP8MegaMoERuntime::launch(runtime, split_args);
    };

    launch_with_phase(SM90FP8MegaMoERuntime::KernelPhase::Linear1, "sm90_fp8_mega_moe_l1_impl");
    launch_with_phase(SM90FP8MegaMoERuntime::KernelPhase::Linear2, "sm90_fp8_mega_moe_l2_impl");
}

} // namespace deep_gemm
