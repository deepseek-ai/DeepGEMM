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

#include "../heuristics/sm90_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime — cooperative kernel
// ----------------------------------------------------------------------------
// SM90 counterpart of `SM100FP8FP4MegaMoERuntime`, driving the N-split
// cooperative kernel in `deep_gemm/impls/sm90_fp8_mega_moe_cooperative.cuh`
// (BLOCK_M=64, BLOCK_N=256): the two math warpgroups split the same tile's
// 256-wide N into two 128-wide halves (each an m64n128 WGMMA), share one A-tile
// load (halving activation HBM traffic), and cross-reduce the per-row amax so
// the L2-input activations are quantized at per-128 K. The single SM90 mega-MoE
// path for all token counts.
//
// Differences from the SM100 path:
//   * Activations and weights are both FP8 (e4m3); no FP4.
//   * Activation/weight scale factors (SF) are per-128-channel float (not UE8M0
//     int + per-32 UTCCP layout).
//   * No tensor memory: WGMMA accumulators are register-resident.
//   * Cluster size is at most 2 (TMA multicast on A); no 2-CTA UMMA.
// ============================================================================

class SM90FP8MegaMoECooperativeRuntime final : public LaunchRuntime<SM90FP8MegaMoECooperativeRuntime> {
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
// JIT cache key: bump this comment to invalidate the cache when
// sm90_fp8_mega_moe_cooperative.cuh changes in a way the template args don't capture.
// cooperative_v29_epgather_fma: intra-rank accumulate uses __fmaf_rn (1 rounding) to
//   match ep_gather's Triton `acc += tmp.to(fp32)*w` (compiler-contracted to fma.rn.f32).
//   v28 used 2-round fadd∘fmul → diverged 1 ULP only on ranks with ≥2 experts (residual
//   ~1e-10, abs-max 0.03-0.125 on a few cols). FMA matches single-expert ranks identically
//   and the multi-expert ranks bit-for-bit.
// cooperative_v28_twolayer_combine: combine epilogue rewritten to sglang's TWO-LAYER
//   model (ep_gather intra-rank + deep_ep cross-rank). Was single-layer: each (token,
//   topk_slot) bf16 contribution rounded independently then ALL fp32-summed at once —
//   which double-rounds same-rank contributions vs real ep_gather. Now: group slots by
//   sender rank (expert_id/kNumExpertsPerRank), fp32-accumulate `L2_bf16*weight` per
//   group (2 roundings, __fadd_rn∘__fmul_rn, NOT FMA), ONE bf16 round per rank, then
//   fp32-accumulate the per-rank bf16 (final round once). Verified bit-exact vs deep_ep's
//   hadd(≤2-rank)/fp32(>2-rank) combine in /tmp/combine_model5.py (0/262144 — a unified
//   fp32 fold reproduces both branches since 2 bf16 sum exactly in fp32). L2 epilogue no
//   longer applies the topk weight (scatters raw unweighted bf16 L2).
// cooperative_v27_truediv_quant: the down-input quant matches sglang's REAL path
//   `sglang_per_token_group_quant_fp8` (enable_v2=None) → JIT kernel
//   per_token_group_quant_8bit.cuh (NOT v2, NOT fast-math). BOTH divides are PRECISE
//   div.rn: stored y_s = amax/448 (was amax*(1/448f) = 1 ULP high, two roundings) and
//   code = val / y_s (was val*(448/amax) reciprocal-mul). BIT-VERIFIED: amax=3328 →
//   amax/448 = 0x40edb6db (== real sglang) vs amax*(1/448f) = 0x40edb6dc. This fixes the
//   NT=32 down-input 16/65536 diffs (all sfULP=1 from the 1-ULP-high stored SF).
// cooperative_v26_precise_qdiv: the quant `y_scale = 448/amax` must be PRECISE div.rn,
//   NOT div.approx.ftz. Verified vs REAL sglang quant on a 378/2016 group: real gives
//   378*(448/2016)=84.0 exact → e4m3 80; div.approx gave 84.0000063 → 88. silu still
//   uses fast-math approx (sgl silu IS fast-math); only this divide is precise.
// cooperative_v25_contig_path: runtime probe CONFIRMED deepep normal mode goes through
//   `_run_contiguous_gemm` (NOT masked). Reverted v24's masked silu (silu→bf16→*up).
//   Now: silu = fast-math __expf+approx-div (kFastMath) fp32 silu*up → store bf16;
//   quant = sgl v2 (sf=amax*(1/448) stored, code=clamp(val*(448/amax),±448) e4m3).
//   v24 (masked, silu→bf16-before-*up) made calc_diff WORSE (4e-6 → 1.3e-5).
// cooperative_v23_fastmath_switch: the `fast_math` arg (kFastMath template param) now
//   controls the SwiGLU/quant transcendental lowering: true => reproduce sglang's
//   `-use_fast_math` PTX (ex2.approx.ftz + div.approx.ftz for silu, div.approx.ftz for
//   448/amax) which matches sgl_kernel/flashinfer bit-for-bit incl. outlier bf16
//   midpoints (e.g. silu(-15)*10.9375 -> 0xb853); false => precise expf + div.rn.
// cooperative_v22_legacy_path: re-aligned SwiGLU+quant to sglang's LEGACY DeepEP
//   down-proj (silu_and_mul → sglang_per_token_group_quant_fp8, the default
//   SGLANG_OPT_FIX_MEGA_MOE_MEMORY=False path): silu uses PRECISE expf + true div,
//   silu*up in fp32 then ROUNDED to bf16 before quant; quant multiplier = 448/amax
//   (true div), stored sf = amax*(1/448) (recip-mult), code = clamp(val*ys,±448) e4m3.
//   (Replaces v21 which matched the FUSED path: fast __expf, fp32-direct quant.)
// cooperative_v21_bitexact: BIT-EXACT with sglang DeepEP fp8 path (8-rank mimo,
//   real weights, outlier acts: abs-max=0, calc_diff=0). Root cause was DeepGEMM's
//   JIT not using -use_fast_math while sglang production does, so three ops diverged
//   by 1 fp32 ULP and got amplified at bf16 grid midpoints:
//   (1) GEMM promote `final += scale*accum` → forced __fmaf_rn (FFMA, 1 rounding;
//       was separate FMUL+FADD); (2) quant scale sf=amax/448 → div.approx.ftz
//       (was div.rn); (3) quant code=val*(1/sf) → rcp.approx.ftz + mul.ftz (was
//       val/sf div.rn). SiLU already used inline ex2.approx.ftz+div.approx.ftz.
//   All debug probes removed.
// cooperative_v15_silumul_ftz: also fold the final `silu*up` into the inline-PTX
//   helper as mul.ftz.f32 (production's val=silu*u is .ftz under -use_fast_math).
//   A plain `*` emitted mul.f32 (non-ftz) and left a 1-ULP gap on the exact element
//   that becomes the per-128 amax, so the float32 SF still differed by 1 ULP.
// cooperative_v14_silu_fastmath_ptx: reproduce production's -use_fast_math PTX for
//   SiLU via inline asm (ex2.approx.ftz.f32 + div.approx.ftz.f32, all-.ftz mul/add),
//   since DeepGEMM's JIT does NOT pass -use_fast_math and the same __expf source
//   otherwise emits ex2.approx.f32 + div.rn.f32 — a 1-ULP gap that perturbed the
//   float32 per-128 SF vs sglang production. Matches production silu bit-for-bit
//   without enabling fast-math globally (which would change every GEMM's rounding).
// cooperative_v13_silu_singlediv: SwiGLU uses fast-approx __expf and quant uses
//   clamp floor 1e-10 + sf=amax/448 (matching sglang production), but quantizes
//   with a single direct division `val/sf` (ONE rounding) instead of production's
//   `val*(1/sf)` reciprocal (two roundings) — the L1 GEMM accumulation-order 1-ULP
//   dominates so the reciprocal rounding is immaterial; simpler form kept.
// cooperative_v12_prod_silu_quant: align SwiGLU+quant to sglang's PRODUCTION
//   fused kernel `silu_and_mul_masked_post_quant.cuh` bit-for-bit: (1) silu uses
//   the FAST-approx __expf (not true expf); (2) quant clamp floor is 1e-10 (not
//   1e-4); (3) quant multiplies by the true reciprocal `val*(1/sf)` (not `val/sf`).
//   These revert earlier changes that had matched a hand-written torch replica in
//   the mimo test rather than the real operator; the test now calls the real
//   silu_and_mul_contig_post_quant so both sides share identical numerics.
// cooperative_v11_l2_bf16_round: in the L2 epilogue, round each WGMMA fp32
//   accumulator to BF16 BEFORE applying the topk weight (sglang's L2 fp8 GEMM
//   emits bf16, then DeepEP combine does (l2_out.float()*weight).to(bf16) — two
//   bf16 rounds). Previously we multiplied the fp32 accumulator by the weight
//   directly, skipping the inner round, giving a 1-ULP per-contribution gap that
//   blows up at cancellation columns (8 contributions summing to ~0).
// cooperative_v10_nsplit_fused_quant: bit-exact alignment with sglang's Hopper
//   fused down-proj path (silu_and_mul_contig_post_quant): (1) L1-output bf16
//   round before SwiGLU (matches grouped_gemm_nt_f8f8bf16 epilogue cast); (2)
//   SwiGLU result stays FP32 — NO bf16 round before quant (fused path quantizes
//   the fp32 silu*up directly); (3) quant uses TRUE division x/sf and sf=amax/448
//   true division (not *rcp), so e4m3 rounding matches bit-for-bit. Verified
//   0/2048 differing codes vs sglang on the single-token L1 isolation test.
// cooperative_v9_nsplit_bf16round: L1-output bf16 round before SwiGLU to match
//   sglang's grouped_gemm_nt_f8f8bf16 epilogue cast (precision alignment).
// cooperative_v8_nsplit: N-split tiling (BLOCK_M=64, BLOCK_N=256). The two math
//   warpgroups split the 256-wide N tile into two 128-wide halves (each m64n128),
//   share one A-tile load, and cross-reduce the per-row amax over their 64-col
//   halves so the L2-input activations are quantized at per-128 K (matches the
//   standard DeepEP runner). Replaces the per-64 M-split cooperative kernel.
// cooperative_v7: double-buffered CD staging (kNumCDStages=2) to overlap the L1
//   TMA store with the next tile's MMA+epilogue; arrival-mask publish deferred one
//   L1 tile and flushed at the L1->L2 transition. Host pipeline config passes
//   cd_stages=2 (one fewer GEMM stage in exchange).
// cooperative_v3: L2-acts SF uses true-float scale (sf = amax/448) instead of
//   UE8M0 power-of-2 alignment (no storage saving on Hopper; matches reference).
//   v3: sf_inv = rcp(sf) (one mul fewer per row than kE4M3Max*rcp(amax)).
// Eliminate all vprintf calls that cause ptxas C7510 (WGMMA pipeline
// serialization due to function call boundary):
// 1. DG_DEVICE_ASSERT → trap-only (no printf)
// 2. DG_NO_DEVICE_PRINTF → suppresses direct printf in barrier.cuh timeout path
#define DG_DEVICE_ASSERT(cond) do {{ if (not (cond)) asm("trap;"); }} while (0)
#define DG_NO_DEVICE_PRINTF
#include <deep_gemm/impls/sm90_fp8_mega_moe_cooperative.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mega_moe_cooperative_impl<
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
        {}, {}
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
    args.config.l2_nmajor_schedule ? "true" : "false");
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

static void sm90_fp8_mega_moe_cooperative(
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
    const auto config = get_mega_moe_cooperative_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: block (128, 128) raw float pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 128;
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
    // L1 output (post-SwiGLU FP8): one L1 block emits `block_n / 2` = 128 columns
    // (gate*up SwiGLU halves the 256-wide weight tile) for all `block_m` = 64
    // rows. The SM90 epilogue writes this staging tile to SMEM as plain row-major
    // bytes, so the TMA store descriptor must use no shared-memory swizzle. Later
    // L2 TMA loads may still swizzle from this row-major global buffer into their
    // own SMEM tile.
    // N-split cooperative: the two math warpgroups write disjoint column halves
    // (cols 0..63 and 64..127) of the SAME staging tile, then a SINGLE full-tile
    // TMA store covers all `block_m` rows — so the descriptor outer-box dim is
    // `block_m` (the full tile), not `block_m / 2`.
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, config.block_m,
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

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM90FP8MegaMoECooperativeRuntime::Args args = {
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
        .l1_weights_sf = l1_weights_sf.data_ptr<float>(),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_sf = l2_weights_sf.data_ptr<float>(),
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const auto code = SM90FP8MegaMoECooperativeRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_mega_moe_cooperative", code);
    SM90FP8MegaMoECooperativeRuntime::launch(runtime, args);
}

} // namespace deep_gemm
