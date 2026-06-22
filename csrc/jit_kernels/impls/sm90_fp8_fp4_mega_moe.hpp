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
// SM90 (Hopper) FP8 x FP4 MegaMoE host runtime
// ----------------------------------------------------------------------------
// Counterpart of `SM90FP8MegaMoERuntime` with these differences:
//   * L1/L2 weights are packed E2M1 (FP4): each storage byte holds 2 nibbles
//     (low nibble = even K, high nibble = odd K). Host code builds the TMA
//     descriptors from a byte view of the packed tensors, so TensorMap sees
//     dense bytes while the kernel interprets each byte as two FP4 elements
//     in the `b_packed_dtype_t = int8_t` SMEM tile.
//   * Weight scale factors switch from per-128 K float to per-32 K UE8M0,
//     packed as int32 along K (4 bytes = 4 K-groups = BLOCK_K=128 K-cols
//     for one N-row). They are passed as `const uint32_t*` instead of
//     `const float*`. The kernel reads them via `__ldg` from global, so
//     no TMA descriptor is required.
//   * Activation side is identical to the FP8 path: FP8 e4m3 K-major with
//     128B swizzle, per-128 K float SFA for L1 and per-64 K float SFA for
//     L2 (filled by the L1 epilogue's per-token SwiGLU+quant).
//   * The kernel applies the per-32 SFB on the fly during dequant through a
//     constant-memory UE8M0->E4M3 LUT, so the only SF that the promote loop
//     still applies is SFA. There is no `weight_sf` ldg in the math warpgroup
//     beyond the SFB UE8M0 word.
// ============================================================================

class SM90FP8FP4MegaMoERuntime final : public LaunchRuntime<SM90FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        // Read the four packed FP4 words for one K/32 group with a
        // single wide shared load while keeping the default work partition.
        bool use_wide_load_decode;
        // Overlap FP4 decode with WGMMA. When false, the math
        // warpgroup only waits on the decode barrier; non-epilogue warps do the
        // decode work and can run ahead through pipeline stages.
        bool math_wg_participates_in_fp4_decode;
        // Limit how many warps inside the math warpgroup help decode.
        // This keeps CTA size fixed while reducing math-side non-tensor-core
        // work that can interfere with WGMMA issue.
        int num_math_wg_decode_warps;
        // Skip early non-epilogue warps as FP4 decode helpers.
        // 0 keeps the existing 4 assist warps; 2 skips the two TMA loader
        // warps; 4 leaves all decode work to the math warpgroup.
        int first_fp4_decode_assist_warp;
        // Split packed-B readiness from the A+B full barrier so the
        // assist warps can start FP4 decode while A/SFA TMA is still in flight.
        bool use_early_b_decode;
        // Replace the FP4 decode rendezvous sync with a per-stage
        // mbarrier so assist warps can run ahead after publishing a decoded tile.
        bool use_decode_done_mbarrier;
        // Mirror the FP8 split-MN arrival-counter path for FP4 L1->L2
        // readiness, avoiding the bitmask update's CTA-wide epilogue sync.
        bool use_l2_arrival_counter;
        // Split each SS N=128 WGMMA into two N=64 WGMMAs so the
        // per-K-block accumulator is 32 floats instead of 64. Large-token SS
        // shapes enable this to reduce accumulator pressure while keeping SS
        // scheduling.
        bool use_ss_nsplit;
        // swapAB path: use decoded weight as WGMMA-M and tokens as WGMMA-N.
        bool use_swap_ab;
        bool use_swap_ab_fast_amax;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations (FP8) and packed FP4 weights.
        // Weight UE8M0 SFB are passed as raw uint32* (no TMA descriptor).
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const uint32_t* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const uint32_t* l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_mega_moe_impl<
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
    args.use_wide_load_decode ? "true" : "false",
    args.math_wg_participates_in_fp4_decode ? "true" : "false",
    args.num_math_wg_decode_warps,
    args.first_fp4_decode_assist_warp,
    args.use_early_b_decode ? "true" : "false",
    args.use_decode_done_mbarrier ? "true" : "false",
    args.use_l2_arrival_counter ? "true" : "false",
    args.use_ss_nsplit ? "true" : "false",
    args.use_swap_ab ? "true" : "false",
    args.use_swap_ab_fast_amax ? "true" : "false");
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

static void sm90_fp8_fp4_mega_moe(
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
    const bool& fast_math,
    const bool& math_wg_participates_in_fp4_decode = true,
    const int& num_math_wg_decode_warps = 4,
    const int& first_fp4_decode_assist_warp = 0,
    const bool& use_wide_load_decode = false,
    const bool& use_early_b_decode = false,
    const bool& use_decode_done_mbarrier = false,
    const bool& use_l2_arrival_counter = false,
    const bool& use_ss_nsplit = false,
    const bool& use_swap_ab = false,
    const bool& use_swap_ab_fast_amax = false
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Sanity: SFB tensors must be uint32 (UE8M0 packed) and weight tensors
    // must use byte-addressable packed FP4 storage (1 byte = 2 nibbles).
    DG_HOST_ASSERT(l1_weights_sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(l2_weights_sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(num_math_wg_decode_warps >= 0 and num_math_wg_decode_warps <= 4);
    DG_HOST_ASSERT(math_wg_participates_in_fp4_decode or num_math_wg_decode_warps == 0);
    DG_HOST_ASSERT(first_fp4_decode_assist_warp >= 0 and first_fp4_decode_assist_warp <= 4);

    // Heuristics
    const auto config = get_mega_moe_config_sm90_fp4(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        use_early_b_decode, use_decode_done_mbarrier,
        use_swap_ab, use_swap_ab_fast_amax);

    // Tensormap construction
    constexpr int kGranK         = 128;  // L1 acts SF granularity (per-128 K)
    const int kL2ActsSFGranK = config.block_n == 64 ? 32 : 64;

    // Acts: FP8 e4m3, identical to FP8 path
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);

    // Packed FP4 weight tile: each byte = 2 nibbles. SM90 loads these as raw
    // bytes and software-decodes them before WGMMA, so the TensorMap must be a
    // UINT8 view with a packed K axis rather than a native FP4 TensorMap.
    const auto l1_weights_bytes = l1_weights.scalar_type() == torch::kByte
        ? l1_weights : l1_weights.view(torch::kByte);
    const auto l2_weights_bytes = l2_weights.scalar_type() == torch::kByte
        ? l2_weights : l2_weights.view(torch::kByte);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights_bytes,
                                                        hidden / 2, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k / 2, config.block_n,
                                                        static_cast<int>(l1_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false);

    // L1 output (post-SwiGLU FP8): N is halved.
    // Mirror the FP8 split-N infrastructure: when BLOCK_M=64 and the host
    // heuristics asked for multiple math warpgroups (`num_epilogue_warpgroups
    // > 1`), each warpgroup shares the same BLOCK_M rows but only owns
    // `WG_BLOCK_N = BLOCK_N / num_wg` columns. Each warpgroup issues its own
    // TMA store with that column tile, so the descriptor outer-box must be
    // `(WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2, l1_output_box_m)`. The split-N
    // gating must match the kernel-side `kSplitNWarpgroups` predicate, which
    // requires WG_BLOCK_N >= 64 (so the FP8MMASelector remains valid).
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups =
        config.block_m == 64 and num_epilogue_warpgroups_h > 1 and
        config.block_n % num_epilogue_warpgroups_h == 0 and
        (config.block_n / num_epilogue_warpgroups_h) >= 64;
    const int wg_split_m = split_n_warpgroups ? 1 : num_epilogue_warpgroups_h;
    const int wg_split_n = split_n_warpgroups ? num_epilogue_warpgroups_h : 1;
    DG_HOST_ASSERT(wg_split_m * wg_split_n == num_epilogue_warpgroups_h);
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;
    const int l1_output_box_m = wg_block_m;
    // Split-N with 32 post-SwiGLU cols per WG uses one combined 64-col TMA
    // store from WG0, matching the 64-col L2 activation-scale group.
    const bool split_n_combines_l1_store = split_n_warpgroups and wg_l1_out_block_n < 64;
    const int tma_l1_out_box_n = split_n_combines_l1_store ? (config.block_n / 2) : wg_l1_out_block_n;
    const int tma_l1_out_box_m = split_n_combines_l1_store ? config.block_m : l1_output_box_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       tma_l1_out_box_n, tma_l1_out_box_m,
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
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights_bytes,
                                                        intermediate_hidden / 2, num_experts_per_rank * hidden,
                                                        config.block_k / 2, config.block_n,
                                                        static_cast<int>(l2_weights_bytes.stride(-2)),
                                                        config.swizzle_weights_mode, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .use_wide_load_decode = use_wide_load_decode,
        .math_wg_participates_in_fp4_decode = math_wg_participates_in_fp4_decode,
        .num_math_wg_decode_warps = num_math_wg_decode_warps,
        .first_fp4_decode_assist_warp = first_fp4_decode_assist_warp,
        .use_early_b_decode = use_early_b_decode,
        .use_decode_done_mbarrier = use_decode_done_mbarrier,
        .use_l2_arrival_counter = use_l2_arrival_counter,
        .use_ss_nsplit = use_ss_nsplit,
        .use_swap_ab = use_swap_ab,
        .use_swap_ab_fast_amax = use_swap_ab_fast_amax,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_sf = reinterpret_cast<const uint32_t*>(l1_weights_sf.data_ptr()),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = reinterpret_cast<const uint32_t*>(l2_weights_sf.data_ptr()),
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8FP4MegaMoERuntime::generate(args);
    const auto runtime_name = "sm90_fp8_fp4_mega_moe";
    const auto runtime = compiler->build(runtime_name, code);
    SM90FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
