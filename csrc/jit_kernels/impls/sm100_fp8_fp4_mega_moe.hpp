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
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        // Stream A0.1: enable FP4 (E2M1) activations from L1 epilogue.
        // Default false — keeps the FP8-acts baseline byte-identical.
        bool use_fp4_acts;
        // Stream A0.5: when set, run kind::mxf4 (K=64 dense) instead of
        // kind::mxf8f6f4 (K=32 with-padding) for both L1 and L2 mainloops.
        // Only honored when `use_fp4_acts` is also set.
        bool use_mxf4_kind;
        // Stream B (combine path): when set, the L2 epilogue writes FP8 E4M3
        // + per-(token,N=128) UE8M0 SF over NVLink instead of BF16. The
        // combine reduce dequantizes on the fly. Halves NVLink bytes/token
        // on the second a2a (kHidden + kHidden/128 vs kHidden*2).
        bool use_fp8_combine;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
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
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.use_fp4_acts ? "true" : "false",
    args.use_mxf4_kind ? "true" : "false",
    args.use_fp8_combine ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
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

static void sm100_fp8_fp4_mega_moe(
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
    const bool& use_fp4_acts = false,
    const bool& use_mxf4_kind = false,
    const bool& use_fp8_combine = false
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));
    // Stream A0.5 sanity: kind::mxf4 only accepts FP4 inputs.
    DG_HOST_ASSERT(not use_mxf4_kind or use_fp4_acts);

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        use_mxf4_kind);

    // Make tensormap
    constexpr int kGranK = 32;
    // Stream A0.5: when `use_mxf4_kind` is on, BOTH L1 and L2 acts AND
    // weights TMA descriptors switch from `_ALIGN16B` (FP4 with-padding,
    // 8 data + 8 pad bytes per 16-byte atom) to `_ALIGN8B` (dense FP4,
    // 2 nibbles/byte). The smem byte stride per K-row halves accordingly,
    // and swizzle mode halves to match (128B → 64B). The gmem layout is
    // unchanged — the underlying `l1_acts` / `l1_weights` storage is still
    // packed FP4 nibbles; only how TMA expands them into smem changes.
    const bool fp4_unpacked = not use_mxf4_kind;
    const int swizzle_acts = use_mxf4_kind ? config.swizzle_acts_mode / 2
                                           : config.swizzle_acts_mode;
    const int swizzle_weights = use_mxf4_kind ? config.swizzle_weights_mode / 2
                                              : config.swizzle_weights_mode;
    // Stream A0.0b: when `use_fp4_acts` is on, the L1 token pool buffer
    // (`l1_acts`) is already viewed as `kPackedFP4` (int8) by the symm-buffer
    // slice (see `csrc/apis/mega.hpp`), with shape `[num_pool_tokens, hidden/2]`
    // of packed E2M1 (low nibble = even col, high nibble = odd col).
    // `make_tma_2d_desc` then auto-selects `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`
    // via `aten_dtype_to_tensor_map_dtype` (runtime_utils.hpp:84-87) — or
    // `_ALIGN8B` under `use_mxf4_kind` (Stream A0.5).
    //
    // TMA descriptor: `gmem_inner_dim = hidden` U4 elements (the descriptor
    // reads `hidden/2` storage bytes per row); smem inner box `BLOCK_K = 128`
    // elements expands to 128 smem bytes after `_ALIGN16B`. 128 B swizzle
    // matches the production swizzle_acts_mode (same as B weights, which
    // have used `_ALIGN16B` from day one).
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     swizzle_acts, /*swizzle_base=*/0,
                                                     /*allow_tf32=*/false,
                                                     /*fp4_unpacked_smem=*/fp4_unpacked);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        swizzle_weights, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false,
                                                        /*fp4_unpacked_smem=*/fp4_unpacked);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    //
    // Stream A0.2: when `use_fp4_acts` is on, the L1 epilogue emits packed
    // E2M1 (FP4) where each byte holds 2 elements. The kernel writes a
    // **dense canonical** smem layout (no swizzle XOR) — see the FP4 store
    // branch in `sm100_fp8_fp4_mega_moe.cuh`. To match, we build the L1
    // output TMA descriptor with `swizzle = 0`. The gmem result is the
    // canonical `[M, intermediate_hidden / 2]` packed FP4 layout, byte-
    // identical to what `kernels/fused_gemm_swiglu_fp4_quant_1cta` produces
    // (Stream A2). The L2 reader (built below) consumes this same canonical
    // layout via `_ALIGN16B`. The per-row gmem byte footprint halves
    // (`intermediate_hidden / 2` bytes vs `intermediate_hidden` for FP8);
    // outer stride in the underlying buffer is unchanged.
    const auto tensor_map_l1_output = use_fp4_acts
        ? make_tma_2d_desc(l2_acts,
                           intermediate_hidden / 2, config.num_max_pool_tokens,
                           config.block_n / 4, config.store_block_m,
                           static_cast<int>(l2_acts.stride(-2)),
                           /*swizzle_mode=*/0)
        : make_tma_2d_desc(l2_acts,
                           intermediate_hidden, config.num_max_pool_tokens,
                           config.block_n / 2, config.store_block_m,
                           static_cast<int>(l2_acts.stride(-2)),
                           config.swizzle_acts_mode / 2);
    // Stream A0.2: when FP4 acts on, L2 reads packed E2M1 via `_ALIGN16B`.
    // `make_tma_2d_desc` selects the descriptor dtype from the source
    // tensor's `scalar_type`; `l2_acts` is allocated as FP8 (1 byte/elem).
    // For the FP4 path we re-view the same byte buffer as `kPackedFP4` so
    // the descriptor dtype is `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`.
    //
    // gmem layout (FP4 path, set up by L1 epilogue):
    //   - per row: first `intermediate_hidden / 2` bytes are packed E2M1
    //     (low nibble = even col, high nibble = odd col — canonical MXFP4),
    //     remaining bytes in the row are stale FP8 from prior runs.
    //   - row stride: `l2_acts.stride(-2)` source bytes (= same as FP8
    //     because the buffer view's underlying allocation hasn't changed).
    //
    // TMA descriptor tells the hardware:
    //   - `gmem_inner_dim = intermediate_hidden` U4 elements (=
    //     `intermediate_hidden / 2` source bytes are read per row).
    //   - `gmem_outer_stride = stride(-2)` source bytes (the actual storage
    //     row pitch — leaves the unused tail of each FP8-sized row alone).
    //   - smem inner box = `BLOCK_K = 128` elements (= 64 source bytes per
    //     row, expands to 128 smem bytes after `_ALIGN16B` doubling); 128B
    //     swizzle aligns with the per-stage atom (same as B-side, which has
    //     used this layout for FP4 weights from day one).
    const auto tensor_map_l2_acts = use_fp4_acts
        ? make_tma_2d_desc(l2_acts.view(kPackedFP4),
                           intermediate_hidden, config.num_max_pool_tokens,
                           config.block_k, config.load_block_m,
                           static_cast<int>(l2_acts.stride(-2)),
                           swizzle_acts, /*swizzle_base=*/0,
                           /*allow_tf32=*/false,
                           /*fp4_unpacked_smem=*/fp4_unpacked)
        : make_tma_2d_desc(l2_acts,
                           intermediate_hidden, config.num_max_pool_tokens,
                           config.block_k, config.load_block_m,
                           static_cast<int>(l2_acts.stride(-2)),
                           config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        swizzle_weights, /*swizzle_base=*/0,
                                                        /*allow_tf32=*/false,
                                                        /*fp4_unpacked_smem=*/fp4_unpacked);
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
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .use_fp4_acts = use_fp4_acts,
        .use_mxf4_kind = use_mxf4_kind,
        .use_fp8_combine = use_fp8_combine,
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
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
