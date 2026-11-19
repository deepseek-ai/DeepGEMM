#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/swizzle.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm90_mxfp4_decode.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_fused_mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/sm90_fused_mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// kFastMath SiLU with `ex2.approx.ftz.f32` instead of `__expf`'s non-ftz form: the two only differ when exp(-x) < 2^-126,
// where `1.0f + e` is 1.0f either way, so the result is bit-identical. Same operation order: e = ex2(x * -log2e), rcp.approx.ftz(1 + e), x * sig.
// kTrace mainloop profiling: an ordered SM clock read (`clock()` is a plain intrinsic the compiler may move or merge)
__forceinline__ __device__ uint32_t sm90_fp8_fused_mega_moe_prof_clock() {
    uint32_t t;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t) :: "memory");
    return t;
}

__forceinline__ __device__ float sm90_fp8_fused_mega_moe_silu_ftz_exp(float x) {
    float e;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(__fmul_rn(-x, 1.4426950408889634f)));
    return x * math::fast_rcp(1.0f + e);
}

// Continuous FP32 activation scale. SM90 WGMMA has no hardware block-scale operand (the SF
// is a plain FFMA in the epilogue), so the previous UE8M0 (power-of-two) scale bought nothing
// on SM90 and only cost precision; the SF pool is already fp32, so this is byte/layout neutral.
// clamp amax before the reciprocal: padded rows have amax==0, and 448/0=inf -> 0*inf=NaN.
template <bool MXFP4 = false>
__forceinline__ __device__ void sm90_fp8_fused_mega_moe_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    constexpr float floor = 1e-10f;
    const auto ax = fmaxf(amax.x, floor);
    const auto ay = fmaxf(amax.y, floor);
    sf.x = __fmul_rn(ax, kScale), sf_inv.x = 1.0f / sf.x;
    sf.y = __fmul_rn(ay, kScale), sf_inv.y = 1.0f / sf.y;
}

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE: dispatch warps pull FP8 tokens + per-128 channel float SF from remote ranks over NVLink
// into the local pool; TMA loader warps (A+SFA, B+SFB) feed the stages; math warpgroups run WGMMA and then either the
// L1 epilogue (SwiGLU on the gate/up gran-8 interleaved layout, per-row amax per output-SF group, FP8 e4m3 quantize,
// TMA store; the row SF is written as a float at per-kL2ActSFK granularity, no cross-CTA amax) or the L2 epilogue
// (BF16 cast, NVLink scatter to the remote combine buffers); after all tiles the math warps run the top-k COMBINE.
// ============================================================================

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumRingTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t kEpilogueRegisterBudget,
    bool kReuseAccumAsFinal,
    bool kL2ArrivalCounter,
    bool kL2EpilogueRequiresFullSync,
    bool kFP8SwapAB = false,
    bool kHalfL2CD = false,
    // 2-CTA cluster with TMA multicast on A
    uint32_t kClusterSize = 1,
    bool kMulticastOnB = false,
    // L2-lag interleaved schedule (scheduler/sm90_fused_mega_moe.cuh); 0 = wave schedule
    uint32_t kL2LagUnits = 0,
    // L2 BF16 staging passes per tile (0 = 2 with kHalfL2CD, else 1); 4 = quarter-width buffer
    uint32_t kL2CDPasses = 0,
    // opt-in in-kernel event trace (host: DG_SM90_TRACE_PTR); false compiles every trace call out
    bool kTrace = false,
    // epilogue sub-events in the trace (events 50-54 L1, 60-62 L2; host: DG_SM90_TRACE_EPI); needs kTrace
    bool kTraceEpi = false,
    // K granularity of the L2 activation SF the L1 epilogue writes (the symm buffer's `l2_act_sf_gran_k`): 64 = two wgmma groups
    // + two rescales per 128-K block in L2; 128 = the L1-style single-group k-block and half the L2 act-SF pool
    uint32_t kL2ActSFK = 64,
    // L2 staging passes: 0 = column passes; 2 = 4 passes over ROWS (4 rows x 256 columns) through plain shared stores; 4 = the
    // same via stmatrix.m8n8.x4 (junk rows parked in the weight-SF slot). Needs
    // kL2CDPasses == 4 and the 64x256 tile
    uint32_t kL2StageMode = 0,
    // programmatic dependent launch (host: forced for calls of at most kSm90FusedPdlMaxTokens tokens per rank, else `deep_gemm.set_pdl`):
    // every thread executes griddepcontrol.wait after the CTA-local prologue and before its first global memory access; the
    // trigger is issued per CTA at TILES_DONE
    bool kPDL = false,
    // early combine: 0 off (host: the smem gate hidden < 512 x topk, or either GEMM having no more k-blocks than num_stages -- so it
    // follows this rank's token count, not the shapes alone); 1 = while the CTA waits in the pre-combine all-rank barrier, math warp
    // 0 runs the barrier protocol and warps 1.. combine their tokens whose top-k experts are all flagged (fixed slot-order fp32 sum,
    // output bit-identical). Contract of the done flag of expert e (set on every rank after the expert's last L2 tile,
    // fence.acq_rel.sys + relaxed sys REDs): every combine row written by e is final
    uint32_t kEarlyCombineMode = 0,
    // pull publish batch N: the arrivals of up to N landed rows are published behind ONE gpu-scope release fence + N relaxed reds;
    // 1 = one red.release.gpu per row. A warp's pending rows are in different pool blocks and all are flushed after the
    // loop, so the per-block counts are unchanged
    uint32_t kPullPublishBatch = 1,
    // eager pull publish (per-row publish path only): a warp's first-iteration rows are published right after the local store
    bool kPullEagerPublish = false,
    // opt-in packed MXFP4 routed weights (PR #411 format): TMA packed E2M1 B + K32 UE8M0 SFB per stage, software decode into a
    // per-warpgroup expanded E4M3 tile, FP8 WGMMA; the per-expert FP32 secondary scale replaces the block-(128, 128) weight SF.
    // Only the routed split-N decode topology (BLOCK_M 64, BLOCK_N 256, 2 math warpgroups, cluster 1, no swapAB) supports it
    bool kMXFP4Weights = false,
    // MXFP4 expanded-B tiles (1 or 2): with 2, the decode of k-block k + 1 runs while the WGMMA of k-block k is in flight
    uint32_t kMXFP4ExpandedBuffers = 1,
    // MXFP4 decode lane mapping: 0 = one packed word per lane (LDS.32 / STS.64, PR #411's generic path); 1 = two adjacent words of
    // one K32 group per lane (LDS.64 / x16 decode / STS.128); 2 = one whole row per lane with its own scale word (PR #411's
    // small-M full-row pair decode: per K32 one lookup, 2 x (LDS.64 + STS.128), word pairs visited in a bank-permuted order);
    // 3 = the preprocessed "tiled" weight layout (`transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(..., layout='tiled')`):
    // every (256-row n-block, k-block) stage is one contiguous 16 KiB [4 K32 groups][256 rows][16 B] chunk moved by one bulk copy,
    // decoded one row per lane (LDS.128 per K32 group) into a no-swizzle K-major expanded tile (8 x 16 B core matrices, LBO 128,
    // SBO 1024): both the loads and the stores of a quarter warp cover 128 contiguous bytes, no swizzle address terms;
    // 4 = the tiled layout with register-A (RS) swapAB for every token bucket (N8 / N16 / N32): every lane decodes the m64k32 A
    // fragments of its own 4 weight rows straight into registers (the unit's word c is the fragment register pair of lane c), so swap
    // tiles write no expanded tile and need no async-proxy fence or warpgroup barrier per k-block; regular tiles as mode 3;
    // 5 = mode 4 plus an N24 bucket (17..24 valid rows). A separate mode because the N24 code alone, never executed, costs the
    // M8..M32 tiles ~1.5 % (code size / layout, notes §22), so the host compiles it only when tiles of 17..32 rows are expected;
    // 6 = mode 5 plus an N48 bucket (33..48 valid rows; notes §36.3 / §38), with every register-A bucket double-buffered (§39);
    // 7 = mode 6 plus an N40 bucket (33..40 valid rows; notes §41), compiled separately for the same code-size reason as N24
    uint32_t kMXFP4DecodeMode = 0,
    // MXFP4 swapAB mainloop: the decoded weights are the WGMMA A operand (two M64 halves per warpgroup) and the tile's tokens the
    // N operand, in a bucket of 8 / 16 / 32 / 64 chosen per tile from valid_m; the accumulator is transposed back to the regular
    // layout through the (then idle) expanded-B tiles, so the regular epilogue runs unchanged
    bool kMXFP4SwapAB = false,
    // MXFP4 small-B split-K (notes §52): > 0 = when every local expert has at most this many tokens and the tiles' halves fit the
    // grid, each tile runs as two K halves on two CTAs (scheduler publish_split_k); the K half 1 CTA leaves its fp32 swap accumulators
    // in the split-K region behind the combine buffer, the K half 0 CTA adds them and runs the epilogue. 0 = off
    uint32_t kSplitKMaxTokens = 0,
    // split-K K-range of half 1 in sixteenths (scheduler kSplitKHalf1Sixteenths; 8 = even split)
    uint32_t kSplitKHalf1Sixteenths = 8,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_fused_mega_moe_impl(void* y,
                       int* cumulative_local_expert_recv_stats,
                       const uint32_t num_tokens,
                       const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                       const float* __restrict__ l1_weights_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                       const float* __restrict__ l2_weights_sf,
                       uint64_t* trace,
                       // kMXFP4Weights only (nullptr otherwise): processed relative-UE8M0 SFB, physical [E, K/128, N, 4] bytes, and
                       // the FP32 per-expert secondary scale [E] of L1 / L2
                       const uint8_t* __restrict__ l1_mxfp4_sfb,
                       const float* __restrict__ l1_mxfp4_secondary,
                       const uint8_t* __restrict__ l2_mxfp4_sfb,
                       const float* __restrict__ l2_mxfp4_secondary,
                       // kMXFP4DecodeMode 3..7 only: the tiled packed weights [E, N / 256, K / 128, 16 KiB]
                       const uint8_t* __restrict__ l1_mxfp4_tiled_w,
                       const uint8_t* __restrict__ l2_mxfp4_tiled_w) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads >= 64 and kNumDispatchThreads % 64 == 0,
                     "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64 or kNumNonEpilogueThreads == 128,
                     "Invalid number of GEMM TMA warps");
    DG_STATIC_ASSERT((kNumDispatchThreads + kNumNonEpilogueThreads) % 128 == 0,
                     "Math warpgroup start must be 128-thread aligned");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0, "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N == 128 or BLOCK_N == 256 or BLOCK_N == 512,
                     "SM90 MegaMoE supports CTA BLOCK_N=128/256/512");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");
    DG_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2,
                     "Only 1- or 2-CTA clusters are supported");
    DG_STATIC_ASSERT(not kMulticastOnB or kClusterSize == 2,
                     "B multicast requires a 2-CTA cluster");
    DG_STATIC_ASSERT(kL2ActSFK == 64 or kL2ActSFK == 128, "L2 activation SF granularity must be 64 or 128 K");
    DG_STATIC_ASSERT(kL2StageMode == 0 or kL2StageMode == 2 or kL2StageMode == 4,
                     "L2 staging mode: 0 column passes, 2 row passes (STS), 4 row passes (stmatrix)");
    DG_STATIC_ASSERT(kEarlyCombineMode <= 1, "early combine: 0 off, 1 on");
    DG_STATIC_ASSERT(kPullPublishBatch == 1 or not kPullEagerPublish, "the eager publish is defined for the per-row publish path (the batch rule wins)");
    DG_STATIC_ASSERT(kNumRanks <= 32 and kNumRanks <= kNumDispatchThreads, "low-latency head: one count flag per lane / dispatch thread");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Event trace (kTrace only): `trace` is [kNumSMs][4 roles][kTraceSlots] pairs of
    // (event_id << 56 | aux, %globaltimer ns), written with plain stores by ONE thread per role:
    // role 0 = dispatch warp 0 lane 0, 1 = A/SFA loader lane 0, 2 + g = math warpgroup g warp 0 lane 0.
    // Slots past kTraceSlots are dropped, never written.
    constexpr uint32_t kTraceSlots = 2048;
    uint32_t trace_slot = 0;
    const auto trace_event_at = [&](const uint32_t& role, const uint32_t& event_id, const uint64_t& aux, const uint64_t& t) {
        if constexpr (kTrace) {
            if (trace_slot < kTraceSlots) {
                auto ptr = trace + ((static_cast<uint64_t>(sm_idx) * 4 + role) * kTraceSlots + trace_slot) * 2;
                ptr[0] = (static_cast<uint64_t>(event_id) << 56) | (aux & 0x00ffffffffffffffull);
                ptr[1] = t;
            }
            ++ trace_slot;
        }
    };
    const auto trace_event = [&](const uint32_t& role, const uint32_t& event_id, const uint64_t& aux) {
        if constexpr (kTrace) {
            uint64_t t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
            trace_event_at(role, event_id, aux, t);
        }
    };
    const auto trace_dispatch = [&](const uint32_t& event_id, const uint64_t& aux) {
        if constexpr (kTrace) {
            if (warp_idx == 0 and lane_idx == 0)
                trace_event(0, event_id, aux);
        }
    };
    // with PDL nothing may be stored to global memory before griddepcontrol.wait (the trace buffer included: the
    // harness zeroes it with a memset right before the traced launch), so KERNEL_START only takes its timestamp here and
    // is stored after the wait
    uint64_t trace_t_kernel_start = 0;
    if constexpr (kTrace and kPDL) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(trace_t_kernel_start) :: "memory");
    } else {
        trace_dispatch(1, 0);   // KERNEL_START
    }

    // Prefetch all TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing
    // =====================================================================
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, 128u);
    DG_STATIC_ASSERT(kNumMaxPoolTokens % BLOCK_M == 0, "Invalid SM90 MegaMoE pool size");
    // the host sizes the pool with the same helper the workspace uses, so the template value and the workspace agree
    DG_STATIC_ASSERT(kNumMaxPoolTokens == layout::get_num_max_pool_tokens_sm90(kNumRanks, kNumMaxTokensPerRank, kNumTopk, kNumExpertsPerRank),
                     "SM90 MegaMoE pool size does not match the workspace layout");

    // Fixed-capacity ring pool: data pools are sized by
    // `kNumRingTokens` while all metadata (token source table, scheduler
    // offsets) keeps full-pool absolute addressing.
    constexpr bool kRingCoversFullPool = kNumRingTokens >= kNumMaxPoolTokens;
    // early combine (see the template parameter)
    constexpr bool kEarlyCombine = kEarlyCombineMode != 0;
    // mode 1: the source side publishes the flags and the math warps combine their flagged tokens while they wait in the tag-2
    // all-rank barrier (warp 0 drives the barrier alone; the dispatch warps' pre-cleanup rendezvous is the stop flag), see COMBINE
    constexpr bool kEarlyCombinePublish = kEarlyCombineMode == 1;
    constexpr bool kEarlyCombineMathWait = kEarlyCombineMode == 1;
    constexpr uint32_t kNumDataPoolTokens = kRingCoversFullPool ? kNumMaxPoolTokens : kNumRingTokens;
    DG_STATIC_ASSERT(kRingCoversFullPool or kNumRingTokens % BLOCK_M == 0,
                     "Ring capacity must be BLOCK_M aligned");
    // SF pool is addressed by (ring) pool block only, and the host sizes it by
    // the data pool (`get_num_sf_ring_tokens` maximized over candidate BLOCK_M).
    constexpr uint32_t kNumDataPoolBlocks = kNumDataPoolTokens / BLOCK_M;
    DG_STATIC_ASSERT(kNumPaddedSFPoolTokens >= kNumDataPoolBlocks * SF_BLOCK_M,
                     "Invalid SM90 MegaMoE SF pool capacity");

    // The words remote ranks write into this workspace (recv counts, done flags, count flags) are double-buffered by launch
    // parity (SM90FusedWorkspace::t2_bank, non-const: selected after the PDL wait; every role's scheduler holds a reference to
    // this thread's copy), so the kernel needs no exit all-rank barrier before its cleanup, and the head's count exchange
    // completes on per-source release flags instead of an all-rank barrier.
    auto workspace = layout::SM90FusedWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk,
        kRingCoversFullPool ? 0 : kNumRingTokens);

    // Ring translation layer: the scheduler produces absolute pool indices; ring mode maps them onto a fixed-capacity set of
    // physical slots. `kRingCoversFullPool` degrades to the identity mapping.
    constexpr uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M;
    const auto get_ring_block_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kRingCoversFullPool)
            return pool_block_idx;
        else
            return pool_block_idx % kNumRingBlocks;
    };
    const auto get_ring_token_idx = [](const uint32_t& pool_token_idx) {
        if constexpr (kRingCoversFullPool)
            return pool_token_idx;
        else
            return pool_token_idx % kNumRingTokens;
    };
    const auto get_ring_wave_idx = [](const uint32_t& pool_block_idx) {
        if constexpr (kRingCoversFullPool)
            return 0u;
        else
            return pool_block_idx / kNumRingBlocks;
    };

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // L2 acts float SF: 4 bytes per per-kL2ActSFK group. The pool is slot-major (`[k_sf_idx][pool token]`, row pitch = padded
    // pool token count), so the per-token byte count only sizes it and need not be a multiple of 16
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden * 4 / kL2ActSFK, false);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area (ring-sized data pools in ring mode)
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumDataPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumDataPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumDataPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // Split-K region (layout::kSM90FusedSplitK*): ready flags, then the fp32 partials of the K half 1 CTAs
    constexpr bool kSplitK = kSplitKMaxTokens > 0;
    // pool blocks of a split-K call: one m-block per expert (kSplitKMaxTokens <= BLOCK_M), and the tile slots of both GEMMs fit
    // (both halves of every L1 tile and every L2 tile in one pass of the grid: m-blocks <= kNumSMs / (2 x L1 n-blocks), Flash 2, Pro 1)
    constexpr uint32_t kSplitKMaxPoolBlocks = kSplitK ?
        math::constexpr_min(math::constexpr_min(kNumExpertsPerRank, kNumSMs / (2 * (L1_SHAPE_N / BLOCK_N))),
                            math::constexpr_min(kNumSMs / (L2_SHAPE_N / BLOCK_N),
                                                layout::kSM90FusedSplitKTileSlots / ((L1_SHAPE_N + L2_SHAPE_N) / BLOCK_N))) : 0u;
    const auto split_k_flags = static_cast<uint32_t*>(combine_token_buffer.get_end_ptr());
    const auto split_k_partials = reinterpret_cast<float*>(math::advance_ptr(combine_token_buffer.get_end_ptr(), layout::kSM90FusedSplitKFlagBytes));

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    // under ping-pong a warpgroup owns whole tiles, so the N split (and everything it implied: shared SF group, joint staging
    // tile, cross-warpgroup amax) is off
    constexpr bool kSplitNWarpgroups =
        BLOCK_M == 64 and kNumEpilogueWarpgroups > 1 and
        BLOCK_N % kNumEpilogueWarpgroups == 0 and
        ((BLOCK_N / kNumEpilogueWarpgroups == 64) or (BLOCK_N / kNumEpilogueWarpgroups == 128));
    constexpr bool kSplitMNWarpgroups =
        BLOCK_M == 128 and BLOCK_N == 256 and kNumEpilogueWarpgroups == 4;
    // the decode topology: BLOCK_M-64 tiles split along N over two math warpgroups (the host picks it below ~56 expected
    // tokens per expert); the levers below that only pay there key on it
    constexpr bool kDecodeTopology = BLOCK_M == 64 and kNumEpilogueWarpgroups == 2;
    // deferred ring fill (decode topology under the early combine): the B loader issues stage 0 of its first tile at once and the
    // rest of the ring fill only after both dispatch warps published their first pulled row (or found none). Scheduling hint only
    constexpr bool kBFillDefer = kDecodeTopology and kEarlyCombineMode == 1;
    constexpr uint32_t kWarpgroupSplitM = kSplitNWarpgroups ? 1 :
        (kSplitMNWarpgroups ? 2 : kNumEpilogueWarpgroups);
    constexpr uint32_t kWarpgroupSplitN = kSplitNWarpgroups ? kNumEpilogueWarpgroups :
        (kSplitMNWarpgroups ? 2 : 1);
    constexpr uint32_t WG_BLOCK_M = BLOCK_M / kWarpgroupSplitM;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kWarpgroupSplitN;
    constexpr uint32_t kNumCombineWarps = kNumEpilogueWarps;
    using L1WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;  // M=64, N=WG_BLOCK_N, K=32
    using L2WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    constexpr uint32_t kL1OutputArrivalParts = 1;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups,
                     "Invalid warpgroup split");
    DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M,
                     "Each warpgroup must run exactly one WGMMA-M tile");
    DG_STATIC_ASSERT(kNumCombineWarps <= kNumEpilogueWarps,
                     "Combine warp count must fit in epilogue warps");

    // Cluster=1 -> no multicast, A/B are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N  = BLOCK_N / 2;  // post-SwiGLU
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2;
    // 128-column weight-SF blocks covered by one warpgroup tile (L2: lo/hi weight SF per k-block; with per-64 L2 act SF
    // this is also the number of per-64 post-SwiGLU SF groups of the L1 epilogue)
    constexpr uint32_t kNumSFGroupsPerWG = WG_BLOCK_N >= 128 ? WG_BLOCK_N / 128 : 1;
    DG_STATIC_ASSERT(kNumSFGroupsPerWG <= 2, "At most two weight-SF blocks per warpgroup tile");
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = kL2ActSFK;    // L2 acts SF (per-64 or per-128 K)
    // one L2 act SF per 128 K -> the L2 mainloop consumes one act SF per k-block like L1 (single wgmma group)
    constexpr bool kL2ActSFPerBlockK   = kL2ActsSFGranK == BLOCK_K;
    // When WG_L1_OUT_BLOCK_N < kL2ActsSFGranK the two N-split warpgroups jointly own one L2-acts SF group: they publish ONE
    // shared SF slot (k_sf_idx == n_block_idx) and the amax feeding it is reduced across both warpgroups.
    constexpr bool kSplitNSharesSF = kSplitNWarpgroups and (WG_L1_OUT_BLOCK_N < kL2ActsSFGranK);
    // Both N-split warpgroups sit inside one 128-column weight-SF block (they stage the same weight-SF row)
    constexpr bool kSplitNSharesWeightSF = kSplitNWarpgroups and (WG_BLOCK_N < 128);
    DG_STATIC_ASSERT(kL2ActsSFGranK != 64 or kSplitNSharesWeightSF == kSplitNSharesSF,
                     "per-64 L2 act SF: the shared act-SF split is exactly the shared weight-SF split");
    // L2 act-SF groups (post-SwiGLU columns / kL2ActsSFGranK) produced by one warpgroup's L1 epilogue: 2 per-64 groups
    // for the 128-column production tile, 1 per-128 group; the shared-SF split publishes one joint group
    constexpr uint32_t kNumL1OutSFGroups = kSplitNSharesSF ? 1u :
        (WG_L1_OUT_BLOCK_N >= kL2ActsSFGranK ? WG_L1_OUT_BLOCK_N / kL2ActsSFGranK : 1u);
    DG_STATIC_ASSERT(kL2ActsSFGranK != 64 or kNumL1OutSFGroups == kNumSFGroupsPerWG,
                     "per-64 L2 act SF: one post-SwiGLU SF group per 128-column weight-SF block");
    DG_STATIC_ASSERT(kSplitNSharesSF or (WG_L1_OUT_BLOCK_N % kL2ActsSFGranK == 0),
                     "A warpgroup's L1 output must cover whole L2 act-SF groups");
    // the shared-SF split publishes ONE SF slot per tile (slot n_block): the joint post-SwiGLU output of the two warpgroups must
    // be exactly one L2 act-SF group (the host picks the tile width so)
    DG_STATIC_ASSERT(not kSplitNSharesSF or L1_OUT_BLOCK_N == kL2ActsSFGranK,
                     "shared-SF split: the tile's post-SwiGLU output must be exactly one L2 act-SF group");
    constexpr bool kSwapABEligible =
        kFP8SwapAB and kSplitNWarpgroups and (BLOCK_M == 64) and (BLOCK_N == 128) and
        (kWarpgroupSplitN == 2);
    constexpr bool kSwapABActive = kSwapABEligible;
    // The L1 fp8 output tile is staged in the TMA SWIZZLE_128B layout when a warpgroup's staging row is exactly 128 bytes
    // (the 2-warpgroup split-M 128x256 tile; neither shared-SF split nor swapAB); the host builds the descriptor with the same rule
    constexpr bool kL1OutSwizzled = WG_L1_OUT_BLOCK_N == 128 and not kSplitNSharesSF and not kSwapABActive;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    // Ring mode reuses the mask-mode arrival flow (one CTA-wide barrier per
    // (m, n) L1 block, then a single elected publish), because per-WG counter
    // arrivals are `valid_m`-dependent and cannot carry ring generations.
    constexpr bool kL2ArrivalNeedsFullSync = (not kL2ArrivalCounter) or (not kRingCoversFullPool);
    DG_STATIC_ASSERT(not kSwapABEligible or (BLOCK_M % 8 == 0),
                     "swapAB epilogue token chunks assume BLOCK_M is a multiple of 8");
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kSwizzleBMode   = BLOCK_K * sizeof(b_dtype_t);   // 128
    constexpr uint32_t kSwizzleCDMode  = 128;
    DG_STATIC_ASSERT(not kSwapABActive or kL2ActsSFGranK == 64, "the swapAB epilogues assume per-64 L2 activation scales");
    // The swapAB L1 epilogue lets the first N-split warpgroup quantize and store the other warpgroup's half too; that
    // cross-warpgroup hand-off is not bitwise stable run to run, so a warpgroup must own a whole activation-SF group of the
    // tile. The non-swap epilogue combines the warpgroups only through a commutative amax reduction (host: should_use_swap_ab_sm90_fused).
    DG_STATIC_ASSERT(not kSwapABActive or not kSplitNSharesSF,
                     "swapAB needs a tile whose post-SwiGLU columns one warpgroup owns end to end");
    // MXFP4 split-M: the 128 x 256 two-warpgroup tile of the FP8 prefill path (cluster 1). Warpgroup g owns rows [64g, 64g + 64) and
    // all 256 B columns and decodes B rows [128g, 128g + 128) of the shared expanded tile, so a B row is decoded once per 128 tokens
    // (per 64 on the split-N tile); the accumulator is the kReuseAccumAsFinal rescale chain
    constexpr bool kMXFP4SplitM = kMXFP4Weights and BLOCK_M == 128 and BLOCK_N == 256 and kWarpgroupSplitM == 2 and kWarpgroupSplitN == 1;
    // MXFP4 split-M L1 publish. kMXFP4SplitMWGPublish (ring smaller than the pool): each math warpgroup publishes its own half of the
    // (m, n) L1 block (kRingL1Arrivals arrivals and slot releases per block, the inactive half of a partial block included, so the
    // count stays valid_m independent) instead of a CTA-wide barrier and one publish: the leading warpgroup no longer waits for the
    // other one's TMA store and release. kMXFP4SplitMDeferPublish: the TMA store drain and the release of the L1 arrival leave the
    // epilogue; the warpgroup carries them (mxfp4_pending_pub) into the next tile and issues them under that tile's first WGMMA
    // (see mxfp4_flush_pub for why that point cannot deadlock)
    // MXFP4 split-M swap tail: tiles of at most this many valid rows run the swapped WGMMA on both warpgroups (see the split-M mainloop)
    constexpr uint32_t kMXFP4SplitMSwapN = 32;
    DG_STATIC_ASSERT(not kMXFP4SplitM or kMXFP4SplitMSwapN * 2 <= BLOCK_M, "MXFP4 split-M swap tail: the rows of warpgroup 0 only");
    // MXFP4 split-M dual swap tail (kMode 4): WG_BLOCK_M < valid_m <= kMXFP4SplitMSwapDualN
    constexpr uint32_t kMXFP4SplitMSwapDualN = 96;
    DG_STATIC_ASSERT(not kMXFP4SplitM or (kMXFP4SplitMSwapDualN > WG_BLOCK_M and kMXFP4SplitMSwapDualN < BLOCK_M and kMXFP4SplitMSwapDualN % 16 == 0),
                     "MXFP4 split-M dual swap tail: past warpgroup 0's rows, whole warps of warpgroup 1");
    constexpr bool kMXFP4SplitMWGPublish = kMXFP4SplitM;
    constexpr bool kMXFP4SplitMDeferPublish = kMXFP4SplitM;
    constexpr uint32_t kRingL1Arrivals = kMXFP4SplitMWGPublish ? kNumEpilogueWarpgroups : 1u;
    // MXFP4 split-N: each warpgroup decodes exactly the WG_BLOCK_N = 128 B rows it multiplies (one row per thread for the SFB word), so
    // no B row is decoded twice and the expanded tile needs no cross-warpgroup sync
    DG_STATIC_ASSERT(not kMXFP4Weights or (((kDecodeTopology and kSplitNWarpgroups and BLOCK_N == 256 and WG_BLOCK_N == 128) or kMXFP4SplitM) and
                                           kClusterSize == 1 and not kMulticastOnB and not kFP8SwapAB and kReuseAccumAsFinal == kMXFP4SplitM),
                     "MXFP4 weights: routed split-N BLOCK_M 64 x BLOCK_N 256 or split-M 128 x 256, 2 math warpgroups, cluster 1, no swapAB");
    DG_STATIC_ASSERT(kMXFP4ExpandedBuffers == 1 or kMXFP4ExpandedBuffers == 2, "MXFP4 expanded-B tiles: 1 or 2");
    // largest swapAB token bucket; tiles with more valid rows take the regular MXFP4 mainloop
    constexpr uint32_t kMXFP4SwapMaxBucket = 32;
    // decode modes 6 / 7: tiles of up to 48 valid rows swap (N48, mode 7 also N40; transposed back to the regular layout after the loop)
    constexpr uint32_t kMXFP4SwapMaxValidM = kMXFP4DecodeMode >= 6 ? 48u : kMXFP4SwapMaxBucket;
    // MXFP4 swap tiles: the epilogue runs on the swap accumulator layout (weight row x token) instead of transposing the tile back to
    // the regular 64-row layout: SwiGLU / amax / quantize / bf16 cast only for the tile's tokens, no transpose (notes §28)
    constexpr bool kMXFP4DirectSwapEpi = kMXFP4Weights and kMXFP4SwapAB;
    constexpr bool kMXFP4Tiled = kMXFP4Weights and kMXFP4DecodeMode >= 3;
    DG_STATIC_ASSERT(kMXFP4DecodeMode <= 7, "MXFP4 decode mode 0..7");
    DG_STATIC_ASSERT(not kMXFP4SplitM or (kMXFP4DecodeMode == 3 and kMXFP4ExpandedBuffers == 2 and not kMXFP4SwapAB and kL2ActSFPerBlockK),
                     "MXFP4 split-M: tiled weights with the shared-memory decode (mode 3), two expanded tiles, per-128 L2 act SF");
    // register-A swap: the decode of k-block k + 1 lands in the other of two fragment register sets, so the k-loop is unrolled by 2
    // (K is a multiple of 256 for both GEMMs: hidden % 512 == 0, 2 x intermediate_hidden % 512 == 0)
    DG_STATIC_ASSERT(kMXFP4DecodeMode < 4 or (kMXFP4SwapAB and (L1_SHAPE_K / BLOCK_K) % 2 == 0 and (L2_SHAPE_K / BLOCK_K) % 2 == 0),
                     "MXFP4 decode modes 4..7: register-A swapAB with an even k-block count");
    // byte step of the expanded tile per 32 B of K: SWIZZLE_128B rows are K-contiguous, the no-swizzle tile steps 2 core matrices
    constexpr uint32_t kMXFP4ExpandedKStep = kMXFP4Tiled ? 8u : 1u;
    DG_STATIC_ASSERT(not kMXFP4SwapAB or (kMXFP4Weights and kMXFP4ExpandedBuffers == 2 and BLOCK_M == 64),
                     "MXFP4 swapAB: needs the double-buffered expanded tiles (transpose staging) and BLOCK_M 64 (bucket <= 64)");
    DG_STATIC_ASSERT(not kSplitK or (kMXFP4Weights and kMXFP4SwapAB and kMXFP4DecodeMode >= 4 and kSplitKMaxTokens <= kMXFP4SwapMaxBucket and
                                     kL2LagUnits == 0 and kClusterSize == 1 and kSplitKMaxPoolBlocks > 0 and
                                     kSplitKMaxPoolBlocks <= kNumDataPoolBlocks and kNumEpilogueThreads == 256),
                     "split-K: register-A swap tiles of the decode topology, wave schedule, the split pool blocks in one ring generation");
    // the halves keep the even k-block counts of the register-A double buffer (k1 is even by construction, K is even) and, with
    // early combine, more than kNumStages k-blocks: the B loader signals the previous L2 tile at k-block kNumStages of the next tile
    constexpr uint32_t kSplitKL1Half1 = 2 * ((L1_SHAPE_K / BLOCK_K) * kSplitKHalf1Sixteenths / 32);
    constexpr uint32_t kSplitKL2Half1 = 2 * ((L2_SHAPE_K / BLOCK_K) * kSplitKHalf1Sixteenths / 32);
    DG_STATIC_ASSERT(not kSplitK or not kEarlyCombine or
                     (math::constexpr_min(kSplitKL1Half1, L1_SHAPE_K / BLOCK_K - kSplitKL1Half1) > kNumStages and
                      math::constexpr_min(kSplitKL2Half1, L2_SHAPE_K / BLOCK_K - kSplitKL2Half1) > kNumStages),
                     "split-K: more than kNumStages k-blocks in every K half with early combine");
    DG_STATIC_ASSERT(not kMXFP4Weights or kHidden <= 8192,
                     "MXFP4 weights: the SFB bulk copy needs the coalesced [E, K/128, N, 4] payload (hidden <= 8192)");

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // The per-expert routing counts (kNumExperts x u32) are dead after the routing grid sync; they live at the head of the
    // tile-table region at the END of the layout (see SMEM_TILE_TABLE_OFFSET / smem_expert_count)
    constexpr uint32_t SMEM_EXPERT_COUNT_BYTES = kNumExperts * sizeof(uint32_t);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    // MXFP4: a B stage holds the packed E2M1 bytes (K/2 per row, TMA SWIZZLE_64B); the FP8 path keeps the full E4M3 tile
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = kMXFP4Weights ? LOAD_BLOCK_N * BLOCK_K / 2 : LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // MXFP4 expanded E4M3 B tile (SWIZZLE_128B, the layout the FP8 TMA writes): one per CTA, each split-N warpgroup decodes and
    // reads only its own WG_BLOCK_N rows, so the tile is warpgroup-private storage, not a pipeline stage
    constexpr uint32_t SMEM_EXPANDED_B_TILE_SIZE = BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_EXPANDED_B_SIZE = kMXFP4Weights ? kMXFP4ExpandedBuffers * SMEM_EXPANDED_B_TILE_SIZE : 0u;
    // SFA per stage: BLOCK_M floats for L1 and for the per-128 L2 act SF (one (BLOCK_M, 1) TMA per k-block), 2*BLOCK_M
    // floats (lo/hi halves) with the per-64 L2 act SF
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>((kL2ActSFPerBlockK ? 1u : 2u) * BLOCK_M * sizeof(float), 128u);
    // MXFP4 SFB per stage: one u32 (four K32 UE8M0 bytes) per B row; a bulk copy of the contiguous [N] slice of [E, K/128, N, 4]
    constexpr uint32_t kMXFP4GranK = 32;
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = kMXFP4Weights ? BLOCK_N * (BLOCK_K / kMXFP4GranK) : 0u;
    constexpr uint32_t kNumL1WeightSFFloatsPerWG = 2 * (kHidden / 128);
    constexpr uint32_t kNumL2WeightSFFloatsPerWG = kNumSFGroupsPerWG * (kIntermediateHidden / 128);
    constexpr uint32_t kNumWeightSFFloatsPerWG =
        kNumL1WeightSFFloatsPerWG > kNumL2WeightSFFloatsPerWG ?
            kNumL1WeightSFFloatsPerWG : kNumL2WeightSFFloatsPerWG;
    constexpr uint32_t SMEM_WEIGHT_SF_SIZE =
        math::constexpr_align<uint32_t>(
            kNumEpilogueWarpgroups * kNumWeightSFFloatsPerWG * sizeof(float), 128u);

    // CD output: max of L1 FP8 (BLOCK_M * BLOCK_N/2), L2 BF16 (BLOCK_M * BLOCK_N * 2) and the swapAB L1 FP32+FP8 staging;
    // split-M warpgroups own disjoint row slices, shared-SF split-N warpgroups disjoint column slices of one CTA tile.
    constexpr uint32_t SMEM_CD_L1_SIZE = BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    // kHalfL2CD stages the L2 BF16 output one N-half at a time (two-pass scatter), halving this buffer
    constexpr uint32_t kNumL2CDPasses = kL2CDPasses != 0 ? kL2CDPasses : (kHalfL2CD ? 2u : 1u);
    constexpr uint32_t L2_CD_STAGE_N = BLOCK_N / kNumL2CDPasses;
    constexpr uint32_t SMEM_CD_L2_SIZE = BLOCK_M * L2_CD_STAGE_N * sizeof(nv_bfloat16);
    // row-pass staging: 2 KiB per math warp == the 4-column-pass buffer (BLOCK_M x BLOCK_N/4 bf16)
    DG_STATIC_ASSERT(kL2StageMode == 0 or (kNumL2CDPasses == 4 and not kFP8SwapAB),
                     "L2 row-pass staging needs the quarter-width (4-pass) buffer");
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(float) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        kSwapABActive ? (SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE) : 0;
    constexpr uint32_t SMEM_CD_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_SIZE    = math::constexpr_align(
        SMEM_CD_BASE_SIZE > SMEM_CD_SWAP_L1_SIZE ? SMEM_CD_BASE_SIZE : SMEM_CD_SWAP_L1_SIZE,
        kSharedMemoryAlignment);

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + SMEM_EXPANDED_B_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

    // SMEM pointers (the layout starts with the dispatch send buffers; the routing counts are at the end, see below)
    const auto smem_send_buffers = layout::Buffer(fp8_token_layout, kNumDispatchWarps, 1, smem_buffer);

    auto smem_gemm_base = math::advance_ptr(smem_buffer, SMEM_SEND_BUFFER_SIZE);

    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_gemm_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SWAP_L1_FP32_SIZE));

    // [CD][expanded B (MXFP4)][A stages][B stages][SFA stages][SFB stages (MXFP4)][weight SF][barriers]: the FP8 layout has
    // SMEM_EXPANDED_B_SIZE == SMEM_SFB_SIZE_PER_STAGE == 0, i.e. it is unchanged
    constexpr uint32_t SMEM_A_OFFSET = SMEM_CD_SIZE + SMEM_EXPANDED_B_SIZE;
    DG_STATIC_ASSERT(SMEM_A_OFFSET % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "A / B stages must keep the 1024 B alignment of the 128 B (64 B) swizzle atoms");
    auto smem_expanded_b = math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE);
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_A_OFFSET + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_A_OFFSET + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_A_OFFSET + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });
    DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 16 == 0 and SMEM_SFB_SIZE_PER_STAGE % 16 == 0, "SFB bulk copies need 16 B alignment");

    // Per-warpgroup weight-SF staging slots (see SMEM_WEIGHT_SF_SIZE above)
    auto smem_weight_sf = reinterpret_cast<float*>(
        sf_start_ptr + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));

    // Barriers live after the weight-SF staging area
    // A (+ per-group SFA) is multicast only when the pairing is on m
    // (n-inner order); with B multicast the pair's A tiles differ.
    constexpr uint32_t kNumMulticastA = kMulticastOnB ? 1u : kClusterSize;

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE) + SMEM_WEIGHT_SF_SIZE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });

    // Tile table right after the barriers (host: `smem_tile_table` mirrors this). Its head holds the routing counts
    // (`smem_expert_count`, kNumExperts x u32), dead after the dispatch warps' second `read_topk_idx`; the table is initialised by
    // the dispatch warps after that (past a bar.sync, no routing atomic pending) and handed to the loaders with
    // kDispatchWithLoadersBarrierIdx, to the math warpgroups with kDispatchWithEpilogueBarrierIdx (both after the routing grid
    // sync); the cluster partner's DSMEM tail entries are written after the leader's `fetch_expert_recv_count`, i.e. past that sync.
    constexpr uint32_t kNumBarriers = kNumDispatchWarps + kNumStages * 2 + kNumCombineWarps * 2;
    // Early-combine region AFTER the tile table (kEarlyCombine only; host `smem_early_combine`): one more transaction barrier per
    // dispatch warp, then a 64 B control block: word 0 = L2 tiles whose scatter stores are issued (the B loader adds 1 for tile T-1
    // once tile T's k-block kNumStages passed the empty-barrier wait), word 1 = stop flag, word 2 = done flag (every tile finished),
    // words 4/5 = trace counters of the dispatch warps (0), words 8.. = the bitmap of tokens the math warps combined while
    // waiting in the barrier, bit (stripe * kNumCombineWarps + math warp)
    constexpr uint32_t SMEM_EC_BYTES = kEarlyCombine ? (kNumDispatchWarps * static_cast<uint32_t>(sizeof(Barrier)) + 64u) : 0u;
    constexpr uint32_t kNumECTokenStripes = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumSMs * kNumCombineWarps);
    constexpr uint32_t kNumECClaimWords = math::constexpr_ceil_div(kNumECTokenStripes * kNumCombineWarps, 32u);
    DG_STATIC_ASSERT(not kEarlyCombine or 32u + kNumECClaimWords * 4u <= 64u, "early combine: too many tokens per CTA for the combined-token bitmap");
    DG_STATIC_ASSERT(not kEarlyCombine or kNumECTokenStripes <= 32, "early combine: the math warps' wait-combine keeps one bit per token stripe");
    // (8-byte entries, one more spare entry for the bounded dynamic tail, see the scheduler)
    constexpr uint32_t kNumTaskPoolTokens = kNumMaxPoolTokens;
    DG_STATIC_ASSERT(kNumTaskPoolTokens <= kNumMaxPoolTokens and kNumTaskPoolTokens % BLOCK_M == 0,
                     "invalid bounded task pool");
    constexpr uint32_t kNumTileTableEntries = layout::get_sm90_tile_table_entries_compact<uint32_t>(
        kNumTaskPoolTokens, BLOCK_M, L1_SHAPE_N / BLOCK_N, L2_SHAPE_N / BLOCK_N, kNumSMs);
    constexpr uint32_t SMEM_TILE_TABLE_OFFSET = math::constexpr_align<uint32_t>(
        SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + SMEM_EXPANDED_B_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE) +
        SMEM_WEIGHT_SF_SIZE + kNumBarriers * static_cast<uint32_t>(sizeof(Barrier)), 16u);
    using TileEntry = uint2;
    constexpr uint32_t SMEM_TILE_TABLE_BYTES = kNumTileTableEntries * static_cast<uint32_t>(sizeof(TileEntry));
    constexpr uint32_t SMEM_TILE_TABLE_REGION_SIZE =
        SMEM_TILE_TABLE_BYTES > SMEM_EXPERT_COUNT_BYTES ? SMEM_TILE_TABLE_BYTES : SMEM_EXPERT_COUNT_BYTES;
    // Early-combine region (see above): 8-byte aligned right after the tile-table region (whose size is a multiple of 8)
    constexpr uint32_t SMEM_EC_OFFSET = math::constexpr_align<uint32_t>(SMEM_TILE_TABLE_OFFSET + SMEM_TILE_TABLE_REGION_SIZE, 8u);
    // SM90 dynamic shared memory capacity (227 KiB); the host derives kNumStages from the same accounting
    DG_STATIC_ASSERT(SMEM_EC_OFFSET + SMEM_EC_BYTES <= 232448u, "SM90 MegaMoE smem layout overflows");
    auto tile_table = reinterpret_cast<TileEntry*>(smem_buffer + SMEM_TILE_TABLE_OFFSET);
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_TILE_TABLE_OFFSET);
    auto ec_barriers = utils::PatternVisitor([=](const uint32_t& i) { return reinterpret_cast<Barrier*>(smem_buffer + SMEM_EC_OFFSET) + i; });
    auto smem_ec = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_EC_OFFSET + kNumDispatchWarps * static_cast<uint32_t>(sizeof(Barrier)));
    auto smem_ec_tiles = smem_ec, smem_ec_stop = smem_ec + 1, smem_ec_done = smem_ec + 2,
         smem_ec_count = smem_ec + 4, smem_ec_gs_old = smem_ec + 6;
    // CTA-wide token bitmap, bit (stripe * kNumCombineWarps + warp), words 8.. of the early-combine control block; the static
    // sliced combine records the tokens the wait-combine already finished in it
    auto smem_ec_claim = smem_ec + 8;

    // Pull pending list (kPullPublishBatch > 1): per dispatch warp, the landed-but-unpublished pull rows (u32: pool token |
    // is_last_of_expert << 31), 16-byte aligned after the early-combine region; 0 bytes with per-row publish (host: `smem_pull_pending`)
    DG_STATIC_ASSERT(kPullPublishBatch >= 1 and kPullPublishBatch <= 256, "Invalid pull publish batch");
    constexpr uint32_t SMEM_PULL_PENDING_OFFSET = math::constexpr_align<uint32_t>(SMEM_EC_OFFSET + SMEM_EC_BYTES, 16u);
    constexpr uint32_t SMEM_PULL_PENDING_SIZE = kPullPublishBatch > 1 ?
        math::constexpr_align<uint32_t>(kNumDispatchWarps * kPullPublishBatch * sizeof(uint32_t), 16u) : 0u;
    DG_STATIC_ASSERT(SMEM_PULL_PENDING_OFFSET + SMEM_PULL_PENDING_SIZE <= 232448u, "SM90 MegaMoE smem layout overflows (pull pending list)");
    auto smem_pull_pending = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_PULL_PENDING_OFFSET);

    // Release a pipeline stage toward every CTA whose producer writes into it.
    auto release_empty_stage = [&](const uint32_t& stage_idx) {
        if constexpr (kMXFP4Weights) {
            // the decode reads the stage's packed B and SFB through the generic proxy; the producer's next TMA write is in the
            // async proxy, which the mbarrier does not order against those reads (notes/pr411-sm90-stage-release-proxy-race.md)
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
        }
        if constexpr (kClusterSize == 1) {
            if (lane_idx == 0)
                empty_barriers[stage_idx]->arrive();
        } else {
            if (lane_idx < kClusterSize)
                empty_barriers[stage_idx]->arrive(lane_idx);
        }
    };

    // =====================================================================
    // Initialization
    // =====================================================================
    if (warp_idx == 0) {
        // Clean expert-count shared memory (it is the head of the tile-table region; the table itself is
        // initialised by the dispatch warps once routing is done, see kDispatchWithLoadersBarrierIdx)
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
        if constexpr (kEarlyCombine) {
            if (lane_idx < 16)
                ptx::st_shared(smem_ec + lane_idx, 0u);
        }
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        if constexpr (kEarlyCombine) {
            #pragma unroll
            for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
                ec_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Two producer warps (A+SFA loader, B+SFB loader) each call
                // `arrive_and_expect_tx` per stage, so init count must be 2.
                full_barriers[i]->init(2);
                // Each math warp arrives once per stage release
                empty_barriers[i]->init(kNumEpilogueWarps * kClusterSize);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumCombineWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (kClusterSize > 1)
        comm::cluster_sync_with_relaxed_arrive();
    else
        __syncthreads();

    // PDL: everything above is CTA-local; everything below reads the previous grid's outputs (topk buffers, the workspace
    // counters and recv counts zeroed by its cleanup), so every thread waits here for the prerequisite grids.
    if constexpr (kPDL) {
        cudaGridDependencySynchronize();
        if constexpr (kTrace) {
            if (warp_idx == 0 and lane_idx == 0)
                trace_event_at(0, 1, 0, trace_t_kernel_start);   // KERNEL_START (timestamp taken before the prologue)
        }
    }
    // launch parity -> bank of the remotely written words (see SM90FusedWorkspace::t2_bank). Every thread reads the word here,
    // before any role touches the workspace; SM0's flip in the tail is ordered behind every read of this launch (loaders and
    // math warps sync with their dispatch warps before those pass the routing / tag-1 grid syncs that precede SM0's tail)
    workspace.t2_bank = ptx::ld_relaxed_sys(workspace.get_t2_parity_ptr()) & 1u;
    // trace: 23 PROLOGUE_DONE = the CTA-local prologue is over (with PDL: the wait has returned, so KERNEL_START -> 23 also
    // holds the time this CTA sat waiting for the previous grid)
    trace_dispatch(23, 0);

    // =====================================================================
    // Scheduler (cluster=1 / 2)
    // =====================================================================
    constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
    constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K;
    auto scheduler = sched::SM90FusedMegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks,
        kMulticastOnB,
        kNumExpertsPerLane, kNumL1BlockNs, kNumL2BlockNs,
        kNumL1BlockKs, kNumL2BlockKs,
        layout::SM90FusedWorkspace, kL2LagUnits,
        kNumMaxPoolTokens / BLOCK_M, kNumRanks * kNumMaxTokensPerRank, kClusterSize,
        kSplitKMaxPoolBlocks, kSplitKMaxTokens, kSplitKHalf1Sixteenths>(workspace);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
    // dispatch warps + the A and B loader warps, once per launch, after the dispatch warps initialised the tile
    // table in the region the routing counts occupied
    constexpr uint32_t kDispatchWithLoadersBarrierIdx   = kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups;
    constexpr uint32_t kNumDispatchWithLoadersThreads   = kNumDispatchThreads + 2 * 32;
    DG_STATIC_ASSERT(kDispatchWithLoadersBarrierIdx < 16, "Out of named barriers");
    // MXFP4 split-M: the hand-off of the shared expanded tiles between the two math warpgroups, one barrier per (direction, warpgroup g,
    // tile b): ready (g, b) = g decoded its half of tile b (g arrives, the peer syncs before its WGMMA on b); free (g, b) = g's WGMMA
    // on b completed (g arrives, the peer syncs before decoding into b). bar.arrive does not wait, so the warpgroups may drift by up
    // to a k-block instead of meeting on every k-block (a CTA-wide bar.sync there measured ~4350 cyc per k-block, the tensor bound ~4130)
    constexpr uint32_t kMXFP4SplitMReadyBarrierIdx = kDispatchWithLoadersBarrierIdx + 1;
    constexpr uint32_t kMXFP4SplitMFreeBarrierIdx  = kMXFP4SplitMReadyBarrierIdx + 4;
    // swap tail (kMXFP4SplitMSwapN): staged = warpgroup 1 staged its half of the result (1 arrives, 0 syncs); read = warpgroup 0 read it
    // (0 arrives, 1 syncs before it may decode into that half again)
    constexpr uint32_t kMXFP4SplitMSwapStagedBarrierIdx = kMXFP4SplitMFreeBarrierIdx + 4;
    constexpr uint32_t kMXFP4SplitMSwapReadBarrierIdx   = kMXFP4SplitMSwapStagedBarrierIdx + 1;
    DG_STATIC_ASSERT(not kMXFP4SplitM or kMXFP4SplitMSwapReadBarrierIdx < 16, "Out of named barriers");
    // MXFP4 shared-SF split-N (the decode topology): the joint L1 staging tile is released split-phase instead of by a CTA-wide
    // bar.sync after the publish: the storing warp (warpgroup 0, warp 0) arrives once its TMA store drained and published, the
    // other math warps sync only before their next write of the CD region (after the next tile's mainloop) or at TILES_DONE
    constexpr bool kMXFP4SplitCDFree = kMXFP4Weights and kSplitNSharesSF;
    constexpr uint32_t kMXFP4CDFreeBarrierIdx = kDispatchWithLoadersBarrierIdx + 1;
    DG_STATIC_ASSERT(not kMXFP4SplitCDFree or (not kMXFP4SplitM and kMXFP4CDFreeBarrierIdx < 16), "Out of named barriers");
    const auto mxfp4_splitm_bar_arrive = [](const uint32_t& barrier_idx) {
        asm volatile("bar.arrive %0, %1;" :: "r"(barrier_idx), "r"(kNumEpilogueThreads) : "memory");
    };

    // Cross-rank NVLink barrier tag (the head and the tail need no all-rank barrier: the head completes on per-source
    // release flags, the tail on the launch-parity banks)
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;

    // Register reconfiguration counts (64512 budget). 256-epilogue-thread split-N decode: 64*48 + 64*40 + 256*168 = 48640;
    // kNumThreads <= 256 decode: 64*48 + 64*40 + 128*256 = 38400 (launch-bounds ceiling 256, the accumulator double-buffer fits);
    // 2 math warpgroups on the 128x256 split-M tile: 64*104 + 64*104 + 256*200 = 64512 (setmaxnreg is warpgroup-collective,
    // so the dispatch warps and the TMA loader warps share one count); the 512-epilogue-thread split-MN path trims both roles.
    constexpr bool kSplitMWide = kNumEpilogueThreads == 256 and BLOCK_M == 128 and BLOCK_N == 256 and
                                 kNumDispatchThreads == 64 and kNumNonEpilogueThreads == 64;
    constexpr uint32_t kNumEpilogueRegisters    =
        kEpilogueRegisterBudget == 0 ?
            (kNumEpilogueThreads == 512 ? 112 :
                (kSplitMWide ? 200 :
                (kNumEpilogueThreads == 256 ? (kMXFP4Weights ? 224u : 168u) :
                    (kNumThreads <= 256u ? 256 : 208)))) :
            kEpilogueRegisterBudget;
    constexpr uint32_t kNumDispatchRegisters =
        kNumEpilogueThreads == 512 ? 32 : (kSplitMWide ? 104 : 48);
    constexpr uint32_t kNumNonEpilogueRegisters =
        kNumEpilogueThreads == 512 ? 24 : (kSplitMWide ? 104 : 40);
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");
    DG_STATIC_ASSERT(kNumEpilogueRegisters % 8 == 0 and kNumEpilogueRegisters >= 24 and kNumEpilogueRegisters <= 256 and
                     kNumDispatchRegisters % 8 == 0 and kNumDispatchRegisters >= 24 and kNumDispatchRegisters <= 256 and
                     kNumNonEpilogueRegisters % 8 == 0 and kNumNonEpilogueRegisters >= 24 and kNumNonEpilogueRegisters <= 256,
                     "setmaxnreg counts must be multiples of 8 in [24, 256]");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;
    // dynamic tail: pair ticket counter (a spare grid-sync slot; zeroed by the dispatch cleanup every launch)
    constexpr uint32_t kTailCounterGridSyncIndex = 2;

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   SF is per-128 channel float, stored straight into the local L1 SF buffer in MN-major layout
    //   `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`; token_idx_in_expert -> SF token index is the per-block linear mapping.
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Stake out per-expert SM offsets via global atomic
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
#if defined(DG_SM90_ROUTED_HEADONLY)
            static_assert(kMXFP4Weights and layout::SM90FusedWorkspace::kHeadLL);
            // Every local source row belongs to CTA0. Offsets start at zero;
            // system release of counts still follows source-index stores.
            if (num_tokens <= kNumDispatchWarps * kNumTokensPerWarp) {
                if (sm_idx == 0) {
                    *workspace.get_expert_send_count_ptr(i) =
                        static_cast<uint64_t>(smem_expert_count[i]);
                    smem_expert_count[i] = 0;
                }
            } else
#endif
            if constexpr (kMXFP4Weights) {
                // only the low word (token count) is read (SM0's publish below); an expert this CTA sends nothing to needs no slot
                // offset, so its contended atomic is skipped (decode: ~all of the 78 x kNumExperts atomics)
                const uint32_t count = smem_expert_count[i];
                if (count != 0)
                    smem_expert_count[i] = static_cast<uint32_t>(
                        ptx::atomic_add(workspace.get_expert_send_count_ptr(i), (1ull << 32) | static_cast<uint64_t>(count)));
            } else {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        // The routing counts are dead from here on (last readers: the slot allocations above, complete in both dispatch warps past
        // this barrier). Their region is the head of the tile table: mark every entry "not published" and release the A/B loader
        // warps, which wait on this barrier before their first table access; the math warpgroups are ordered behind it by
        // kDispatchWithEpilogueBarrierIdx.
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        for (uint32_t i = thread_idx; i < kNumTileTableEntries; i += kNumDispatchThreads)
            ptx::st_shared(tile_table + i, 0xffffffffu, 0xffffffffu);
        ptx::sync_aligned(kNumDispatchWithLoadersThreads, kDispatchWithLoadersBarrierIdx);

#if defined(DG_SM90_ROUTED_HEADONLY)
        // CTA0's named barriers order all local source-index writes before
        // count release. Keep remote flags and every subsequent grid barrier.
        if (num_tokens > kNumDispatchWarps * kNumTokensPerWarp)
#endif
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex, comm::BarrierTimeoutPolicy::TrapOnly>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
        trace_dispatch(2, 0);   // ROUTING_DONE (local grid sync: every CTA of this rank has staked out its slots)

        if (sm_idx == 0) {
            {
                // loads first, then the
                // recv-count stores; no sum atomics -- the release flag below publishes the counts
                constexpr uint32_t kNumPublishPerThread = math::constexpr_ceil_div(kNumExperts, kNumDispatchThreads);
                uint64_t expert_status[kNumPublishPerThread];
                #pragma unroll
                for (uint32_t k = 0; k < kNumPublishPerThread; ++ k) {
                    const uint32_t i = thread_idx + k * kNumDispatchThreads;
                    expert_status[k] = i < kNumExperts ? *workspace.get_expert_send_count_ptr(i) : 0ull;
                }
                #pragma unroll
                for (uint32_t k = 0; k < kNumPublishPerThread; ++ k) {
                    const uint32_t i = thread_idx + k * kNumDispatchThreads;
                    if (i < kNumExperts) {
                        *sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, i % kNumExpertsPerRank),
                                        i / kNumExpertsPerRank) = expert_status[k] & 0xffffffff;
                        *workspace.get_expert_send_count_ptr(i) = 0;   // (its only reader was the load above)
                    }
                }
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        if (sm_idx == 0)
            trace_dispatch(10, 0);   // PUBLISH_DONE (SM0: every thread's recv-count stores issued)

        {
            // SM0 publishes "my counts for you are stored" to every rank (release: cumulative over this CTA's stores
            // through the barrier above and over every CTA's pass-2 slot stores through the routing grid sync); then every CTA
            // waits for the num_ranks flags of this generation itself (one lane per source) -- no all-rank barrier, no grid sync
            if (sm_idx == 0) {
                if (thread_idx < kNumRanks)
                    ptx::st_release_sys_u32(sym_buffer.map(workspace.get_hll_count_flag_ptr(sym_buffer.rank_idx), thread_idx), 1u);
                trace_dispatch(11, 0);   // SIGNAL_SENT (the count flags issued = the count stores drained)
            }
            if (thread_idx < kNumRanks)
                while (ptx::ld_acq_sys(workspace.get_hll_count_flag_ptr(thread_idx)) == 0u);
            trace_dispatch(12, 0);   // SIGNALS_SEEN (thread 0 saw... its own source's flag; 8 below = all of them)
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        }
        trace_dispatch(8, 0);   // NVLINK1_DONE (all ranks' expert counts published)

        {
            // Zero the previous generation's bank (SM90FusedWorkspace::t2_bank): every remote generation-(g - 1) write into it precedes that
            // rank's launch-g count flag, acquired above; the remote generation-(g + 1) writes follow our tag-2 signal, ordered behind
            // these stores by the DISPATCH_SYNC_DONE barrier (math warps) -> tag-2 grid sync 1 (release). Bank layout (u32 words):
            // [recv_count u64 x kNumExperts | recv_count_sum u64 x kNumExpertsPerRank] then [flags | tile counts | count flags]
            constexpr uint32_t kNumT2CountWords = 2 * (kNumExperts + kNumExpertsPerRank);
            constexpr uint32_t kNumT2Words = kNumT2CountWords + kNumExperts + kNumExpertsPerRank + kNumRanks;
            constexpr uint32_t kNumT2WordsPerCTA = math::constexpr_ceil_div(kNumT2Words, kNumSMs);
            DG_STATIC_ASSERT(kNumT2WordsPerCTA <= kNumDispatchThreads, "one bank word per dispatch thread");
            const uint32_t prev_bank = workspace.t2_bank ^ 1u;
            const uint32_t w = sm_idx * kNumT2WordsPerCTA + thread_idx;
            if (thread_idx < kNumT2WordsPerCTA and w < kNumT2Words) {
                if (w < kNumT2CountWords)
                    reinterpret_cast<uint32_t*>(workspace.get_t2_recv_count_bank_ptr(prev_bank))[w] = 0u;
                else
                    workspace.get_t2_flag_bank_ptr(prev_bank)[w - kNumT2CountWords] = 0u;
            }
        }

        // Sync with epilogue warps before pulling tokens.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        trace_dispatch(4, 0);   // DISPATCH_SYNC_DONE (math warps released; pull loop starts)

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];
        // Pull pipeline: the local TMA store of row t is not waited for in iteration t; its arrival
        // count is published in iteration t+1 (or after the loop) once the store has completed, so the
        // store drain overlaps the next row's remote loads. The single pull buffer is reused only after
        // the store has finished reading it (`.read` wait right before the next remote TMA load).
        uint32_t* pending_arrival_ptr = nullptr;
        uint32_t  pending_arrival_add = 0;
        bool bfd_signalled = false;   // deferred ring fill (lane 0): this warp's first-row publish has been signalled to the B loader

        // Batched publish (kPullPublishBatch > 1): lane 0 publishes the warp's landed-but-unpublished rows together: one
        // fence.acq_rel.gpu, then one relaxed red per row (the release pattern the loaders' ld.acquire pairs with; every pending
        // row's store is complete and its SF / weight / metadata stores are ordered before by the __syncwarp that ended their
        // iteration). Pending rows are flushed before the warp blocks on a ring slot (see the ring wait), so no slot release this
        // warp waits for can depend on them.
        constexpr bool kPullBatched = kPullPublishBatch > 1;
        uint32_t num_pending = 0;   // lane 0 (<= kPullPublishBatch)
        const auto pending_list = smem_pull_pending + warp_idx * kPullPublishBatch;
        const auto flush_pending = [&](const uint32_t& trace_aux) {
            if constexpr (kPullBatched) {
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                for (uint32_t i = 0; i < num_pending; ++ i) {
                    const uint32_t word = ptx::ld_shared(pending_list + i);
                    const uint32_t p_token_idx = word & 0x7fffffffu;
                    const uint32_t p_block_idx = p_token_idx / BLOCK_M;
                    if constexpr (kRingCoversFullPool) {
                        asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;"
                                     :: "l"(workspace.get_l1_arrival_count_ptr(p_block_idx)), "r"(1u) : "memory");
                    } else {
                        // the expert's last token pads its tail block to BLOCK_M arrivals (see the per-row path)
                        const uint32_t add = (word >> 31) ? BLOCK_M - (p_token_idx % BLOCK_M) : 1u;
                        asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;"
                                     :: "l"(workspace.get_l1_full_count_ptr(get_ring_block_idx(p_block_idx))), "r"(add) : "memory");
                    }
                }
                trace_dispatch(9, static_cast<uint64_t>(num_pending) | (static_cast<uint64_t>(trace_aux) << 8));   // PUBLISH (aux = rows | token_idx << 8)
                num_pending = 0;
            }
        };

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }
                if (current_expert_idx >= kNumExpertsPerRank)
                    break;

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                    }
                }

                // Round-robin rank selection
                uint32_t current_rank_in_expert_idx;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t slot_idx = token_idx_in_expert;
                uint32_t token_idx_in_rank;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }

                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

                const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                const uint32_t pool_block_idx = expert_pool_block_offset + token_idx_in_expert / BLOCK_M;

                // Ring mode: wait until the previous generation of consumers
                // (all L1 N-blocks of the pool block mapped to this slot) has
                // released the physical slot before overwriting it.
                if constexpr (not kRingCoversFullPool) {
                    constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                    const auto l1_empty_target = get_ring_wave_idx(pool_block_idx) * kNumL1BlockNs * kRingL1Arrivals;
                    if (l1_empty_target > 0) {
                        const auto empty_ptr = workspace.get_l1_empty_count_ptr(get_ring_block_idx(pool_block_idx));
                        if (ptx::ld_acq(empty_ptr) < l1_empty_target) {
                            // the slot is still held by the L1 tiles of block p - R, which complete only once every row of that block
                            // is published: publish this warp's pending rows before blocking, so the wait can never depend on a row
                            // this warp holds back (their stores have had the remote round trip to land)
                            if (lane_idx == 0) {
                                ptx::tma_store_wait<0>();
                                if constexpr (kPullBatched) {
                                    if (num_pending > 0)
                                        flush_pending(token_idx);
                                } else if (pending_arrival_ptr != nullptr) {
                                    ptx::red_add_rel(pending_arrival_ptr, pending_arrival_add);
                                    pending_arrival_ptr = nullptr;
                                }
                            }
                            trace_dispatch(6, pool_block_idx);   // RING_SLOT_WAIT_START (aux = pool block)
                            while (ptx::ld_acq(empty_ptr) < l1_empty_target);
                            trace_dispatch(7, pool_block_idx);   // RING_SLOT_WAIT_END
                        }
                    }
                }

                // Pull token data. Overlap a remote TMA load with SF copy and
                // then use TMA store to materialize the local L1 input.
                if (lane_idx == 0) {
                    // previous row's store must have finished *reading* the pull buffer (not necessarily
                    // landed in global) before it is overwritten by the next remote load
                    cute::tma_store_wait<0>();
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                const uint32_t token_idx_in_block = token_idx_in_expert % BLOCK_M;
                const uint32_t sf_pool_token_idx = get_ring_block_idx(pool_block_idx) * SF_BLOCK_M + token_idx_in_block;
                // weight (lane 0) and SF (all lanes) remote loads are issued together
                float weight_val = 0.f;
                if (lane_idx == 0) {
                    weight_val = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                }
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
                }
                if (lane_idx == 0) {
                    *l1_topk_weights_buffer.get_data_buffer(get_ring_token_idx(pool_token_idx)).template get_base_ptr<float>() = weight_val;
                    if constexpr (not kPullBatched) {
                        // previous row: its local store has had the whole remote round trip to land
                        if (pending_arrival_ptr != nullptr) {
                            ptx::tma_store_wait<0>();
                            ptx::red_add_rel(pending_arrival_ptr, pending_arrival_add);
                            pending_arrival_ptr = nullptr;
                        }
                    } else {
                        // batched publish: the pending rows (the latest stored one iteration ago, with the whole remote
                        // round trip to land) are published once the list reaches the batch limit; the ramp limit is
                        // min(N, 1 << (rows pulled so far by this warp / 8))
                        const uint32_t limit = cute::min(kPullPublishBatch, 1u << cute::min(token_idx / (kNumGlobalWarps * 8u), 8u));
                        if (num_pending >= limit) {
                            ptx::tma_store_wait<0>();
                            flush_pending(token_idx);
                        }
                    }
                }
                __syncwarp();

                if (lane_idx == 0) {
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(get_ring_token_idx(pool_token_idx)).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    if constexpr (kPullBatched) {
                        // batched publish: queue this row (the list has room: it was flushed above once it reached the limit)
                        const bool is_last_token = (token_idx == expert_end_idx - 1);
                        ptx::st_shared(pending_list + num_pending, pool_token_idx | (is_last_token ? 0x80000000u : 0u));
                        ++ num_pending;
                    } else if constexpr (kRingCoversFullPool) {
                        pending_arrival_ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                        pending_arrival_add = 1;
                    } else {
                        // Pad the tail of the expert's last m-block so that
                        // every pass of a ring slot contributes exactly
                        // BLOCK_M arrivals regardless of `valid_m`.
                        const bool is_last_token = (token_idx == expert_end_idx - 1);
                        pending_arrival_ptr = workspace.get_l1_full_count_ptr(get_ring_block_idx(pool_block_idx));
                        pending_arrival_add = is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u;
                    }
                    if constexpr (kPullEagerPublish) {
                        // eager publish: publish this row now (the SF / weight / metadata stores above are ordered before the
                        // release by the __syncwarp + program order, exactly as in the deferred form)
                        if (token_idx < kNumGlobalWarps) {
                            ptx::tma_store_wait<0>();
                            ptx::red_add_rel(pending_arrival_ptr, pending_arrival_add);
                            pending_arrival_ptr = nullptr;
                            trace_dispatch(9, 1ull | (static_cast<uint64_t>(token_idx) << 8));   // PUBLISH (aux = rows | token_idx << 8)
                            if constexpr (kBFillDefer) {
                                if (token_idx < kNumGlobalWarps and not bfd_signalled) {
                                    ptx::red_release_cta_shared_add(smem_ec_gs_old, 1u);   // deferred ring fill: this warp's first row is published
                                    bfd_signalled = true;
                                }
                            }
                        }
                    } else if constexpr (kBFillDefer) {
                        // deferred ring fill without the eager publish: the first row is published at the top of the next iteration
                        // (or in the drain) -- signal when that publish is issued (pending_arrival_ptr == nullptr again after a first row)
                        if (token_idx >= kNumGlobalWarps and not bfd_signalled) {
                            ptx::red_release_cta_shared_add(smem_ec_gs_old, 1u);
                            bfd_signalled = true;
                        }
                    }
                }
                __syncwarp();
            }
        // drain: the last row's store and arrival / the pending list (batched publish)
        if (lane_idx == 0) {
            if constexpr (kPullBatched) {
                if (num_pending > 0) {
                    ptx::tma_store_wait<0>();
                    flush_pending(0xffffffu);
                }
            } else if (pending_arrival_ptr != nullptr) {
                ptx::tma_store_wait<0>();
                ptx::red_add_rel(pending_arrival_ptr, pending_arrival_add);
                pending_arrival_ptr = nullptr;
            }
        }
        if constexpr (kBFillDefer) {
            // deferred ring fill: a warp with no row (or whose first row was published by the drain) signals here, so the B loader never waits forever
            if (lane_idx == 0 and not bfd_signalled)
                ptx::red_release_cta_shared_add(smem_ec_gs_old, 1u);
        }
        __syncwarp();
        trace_dispatch(3, 0);   // PULL_DONE (this warp's last pull stored and its arrival published)

        // ================= Early combine on the dispatch warps (see the template parameter) =================
        // Source (warp 0, kEarlyCombinePublish): the math warps bump smem_ec_tiles (release.cta) per warp per L2 tile after their
        // scatter stores; warp 0 walks the tile table in step: one fence.acq_rel.gpu per batch of finished tiles, one relaxed
        // gpu-scope count add per tile, and for an expert's last tile fence.acq_rel.sys + relaxed sys-scope adds to its flag on
        // every rank. Handover: the math warps set smem_ec_stop after the tag-2 barrier; both warps poll it and join the sync below.
        uint32_t ec_tiles_published = 0;
        if constexpr (kEarlyCombine) {
            // source-side publish (warp 0)
            uint32_t ec_table_pos = 0;
            bool ec_table_done = false;
            const auto ec_publish = [&]() {
                if constexpr (kEarlyCombinePublish) {
                    const uint32_t tiles_done = ptx::ld_acquire_cta_shared(smem_ec_tiles);
                    const bool all_done = ec_table_done or ptx::ld_acquire_cta_shared(smem_ec_done) != 0;
                    if (ec_table_done or (not all_done and tiles_done == ec_tiles_published))
                        return;
                    // one gpu-scope fence per batch: the scatter stores of these tiles (ordered before the counter / done flag by
                    // the math warps' release arrives and the loader's release add) become visible at gpu scope before the
                    // relaxed count adds
                    ptx::fence_acq_rel_gpu();
                    while (all_done or ec_tiles_published < tiles_done) {
                        const uint32_t tag = scheduler.load_tile_entry(tile_table + ec_table_pos);
                        if (tag == 0u) {   // end marker: every tile is accounted
                            ec_table_done = true;
                            break;
                        }
                        ++ ec_table_pos;
                        if (tag != 2u)
                            continue;
                        ++ ec_tiles_published;
                        // split-K: a K half 1 tile scatters nothing (its half 0 counts the tile)
                        if constexpr (kSplitK) {
                            if (scheduler.current_split and scheduler.current_k_half == 1)
                                continue;
                        }
                        const uint32_t e = scheduler.current_local_expert_idx;
                        // the expert's L2 tile count: from the entry's token count, or from the per-expert recv counts this
                        // warp fetched for the pull when the entry carries the tile's own row count (scheduler kTilePayloadByRows)
                        uint32_t expert_num_tokens = scheduler.current_num_tokens;
                        if constexpr (std::remove_reference_t<decltype(scheduler)>::kTilePayloadByRows)
                            expert_num_tokens = scheduler.get_num_tokens(e);
                        const uint32_t total = math::ceil_div(expert_num_tokens, BLOCK_M) * kNumL2BlockNs;
                        if (lane_idx == 0) {
                            const uint32_t old = ptx::atom_add_relaxed_gpu(workspace.get_l2_tile_done_count_ptr(e), 1u);
                            if (old + 1u == total) {
                                // this CTA scattered the expert's last tile: the sys fence is the acquire side of the count (the
                                // other CTAs' releases) and the release side of the flags every rank's dispatch warps poll
                                ptx::fence_acq_rel_sys();
                                const auto flag_ptr = workspace.get_expert_done_flag_ptr(sym_buffer.rank_idx * kNumExpertsPerRank + e);
                                #pragma unroll
                                for (uint32_t r = 0; r < kNumRanks; ++ r)
                                    ptx::red_add_relaxed_sys(sym_buffer.map(flag_ptr, r), 1u);
                                trace_dispatch(71, e);   // EC_EXPERT_PUBLISHED (aux = local expert)
                            }
                        }
                    }
                    __syncwarp();
                }
            };

            while (true) {
                if (warp_idx == 0)
                    ec_publish();
                if (ptx::ld_volatile_shared(smem_ec_stop) != 0)
                    break;
                __nanosleep(1000);
            }
            // handover: every store of this warp is performed before the math warps' combine (nothing of this SM is outstanding at
            // this point, so the fence is cheap)
            ptx::fence_acq_rel_gpu();
            if (lane_idx == 0)
                ptx::st_shared(smem_ec_count + warp_idx, 0u);   // trace counter (tokens combined by this warp: none)
        }

        // Cleanup workspace, overlapping with combine.
        if constexpr (kEarlyCombine) {
            // early combine: the math warps do not rendezvous here; their thread 0 sets the stop flag once the all-rank barrier completed
            while (ptx::ld_acquire_cta_shared(smem_ec_stop) == 0)
                __nanosleep(500);
        } else {
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        }
        trace_dispatch(5, 0);   // CLEAN_START (math warps passed the pre-combine all-rank barrier)
        if constexpr (kEarlyCombine and kTrace) {
            // EC_STOP: aux = tokens combined by dispatch warp 0 | warp 1 << 16 | L2 tiles this CTA published << 32
            trace_dispatch(72, static_cast<uint64_t>(ptx::ld_shared(smem_ec_count)) | (static_cast<uint64_t>(ptx::ld_shared(smem_ec_count + 1)) << 16) |
                               (static_cast<uint64_t>(ec_tiles_published) << 32));
        }

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // (the send counts were zeroed in the head, right after the publish loop)
            // every B loader fetched its end-of-tail ticket before its math warps reached the pre-combine barrier
            if (thread_idx == 0)
                *workspace.template get_grid_sync_count_ptr<kTailCounterGridSyncIndex>() = 0;
            // flip the launch parity for the next launch (every read of this launch precedes this store: the loaders / math
            // warps of each CTA sync with their dispatch warps before those reach the routing grid sync, and the math warps'
            // tag-2 grid sync 1 precedes this CTA's CLEAN_START); the next launch reads it after the kernel boundary
            if (thread_idx == 0)
                *workspace.get_t2_parity_ptr() = workspace.t2_bank ^ 1u;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // the expert's count is the sum of the per-rank recv counts (all landed before the head's flags)
                uint32_t num_recv_tokens = 0;
                #pragma unroll
                for (uint32_t r = 0; r < kNumRanks; ++ r)
                    num_recv_tokens += static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(r, i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    if constexpr (kRingCoversFullPool) {
                        *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                        *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                    } else {
                        const auto ring_block_idx = get_ring_block_idx(expert_pool_block_offset + j);
                        *workspace.get_l1_full_count_ptr(ring_block_idx) = 0;
                        *workspace.get_l2_full_count_ptr(ring_block_idx) = 0;
                        *workspace.get_l1_empty_count_ptr(ring_block_idx) = 0;
                        *workspace.get_l2_empty_count_ptr(ring_block_idx) = 0;
                    }
                }
                __syncwarp();
            }
        }

        // No exit barrier. The arrival / ring counts zeroed above have no user left on this rank (every CTA is past
        // TILES_DONE = tag-2 grid sync 1) and their next users are in the next launch, behind the kernel boundary; the
        // remotely written words are bank words, zeroed by the next launch after its head. Nothing to wait for: the
        // dispatch warps exit while the math warps combine.
        trace_dispatch(22, 0);  // KERNEL_END (dispatch warp's last statement)

    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (warp 0 of `kNumNonEpilogueThreads` loads A + SFA, warp 1 loads B + SFB)
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        // the tile table is initialised by the dispatch warps after routing (its head is the routing-count region)
        ptx::sync_aligned(kNumDispatchWithLoadersThreads, kDispatchWithLoadersBarrierIdx);

        // trace role 1: tile ordinal (same schedule as the math warps) and per-tile arrival wait / issue done
        uint32_t trace_tile = 0;
        const auto trace_loader = [&](const uint32_t& event_id, const uint64_t& aux) {
            if constexpr (kTrace) {
                if (lane_idx == 0)
                    trace_event(1, event_id, aux);
            }
        };

        auto process_a_sfa_block = [&](const auto& block_phase,
                                       const uint32_t& local_expert_idx,
                                       const uint32_t& num_k_blocks,
                                       const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t ring_block_idx = get_ring_block_idx(pool_block_idx);

            const uint64_t trace_aux = static_cast<uint64_t>(trace_tile) |
                (static_cast<uint64_t>(block_phase == sched::BlockPhase::Linear1 ? 1u : 2u) << 32);
            if constexpr (kTrace)
                ++ trace_tile;
            trace_loader(30, trace_aux);   // WAIT_ARRIVAL_START

            // Wait for the pool to be ready
            if (block_phase == sched::BlockPhase::Linear1) {
                if constexpr (kRingCoversFullPool) {
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const auto expected = scheduler.template get_valid_m<false>();
                    while (ptx::ld_acq(ptr) != expected);
                } else {
                    // Ring mode: every pass of this slot contributed exactly
                    // BLOCK_M arrivals (dispatch pads the tail block).
                    const auto ptr = workspace.get_l1_full_count_ptr(ring_block_idx);
                    const uint32_t expected = BLOCK_M * (get_ring_wave_idx(pool_block_idx) + 1);
                    while (ptx::ld_acq(ptr) != expected);
                }
            } else {
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                if constexpr (not kRingCoversFullPool) {
                    // Ring mode: one arrival per (m, n) L1 block after its
                    // output TMA store drained (pass-constant accounting).
                    const auto ptr = workspace.get_l2_full_count_ptr(ring_block_idx);
                    const uint32_t expected = kNumL1BlockNs * kRingL1Arrivals * (get_ring_wave_idx(pool_block_idx) + 1);
                    while (ptx::ld_acq(ptr) != expected);
                } else if constexpr (kL2ArrivalCounter) {
                    const auto ptr = reinterpret_cast<const uint32_t*>(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                    const uint32_t active_m_wgs = math::ceil_div(
                        scheduler.template get_valid_m<false>(), WG_BLOCK_M);
                    const uint32_t expected =
                        kNumL1BlockNs * active_m_wgs * kWarpgroupSplitN * kL1OutputArrivalParts;
                    while (ptx::ld_acq(ptr) != expected);
                } else {
                    const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                    const uint64_t expected = (kNumL1BlockNs >= 64)
                        ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
                    while (ptx::ld_acq_gpu(ptr) != expected);
                }
            }
            trace_loader(31, trace_aux);   // WAIT_ARRIVAL_END
            // split-K: the first k-block of the tile's K half (the replay passed the half's count)
            const uint32_t k_block_base = scheduler.get_split_k_base(block_phase);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx = ring_block_idx * BLOCK_M;
                    const uint32_t sfa_m_idx = ring_block_idx * SF_BLOCK_M;
                    const uint32_t k_idx = (k_block_base + k_block_idx) * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, kNumMulticastA);

                    // TMA load SFA
                    if (kL2ActSFPerBlockK or block_phase == sched::BlockPhase::Linear1) {
                        // L1 SFA per-128 (and the per-128 L2 SFA): load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            sfa_m_idx, k_block_base + k_block_idx, kNumMulticastA);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                    } else {
                        // L2 SFA per-64: descriptor box is (block_mn, 1) (see make_tma_sf_desc),
                        // so we must issue two single-group TMAs and place them at smem offsets
                        // 0 and BLOCK_M to match math's load offsets (`+ 0 * BLOCK_M` / `+ 1 * BLOCK_M`).
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            sfa_m_idx, (k_block_base + k_block_idx) * 2, kNumMulticastA);
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx] + BLOCK_M,
                            sfa_m_idx, (k_block_base + k_block_idx) * 2 + 1, kNumMulticastA);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                    }
                }
                __syncwarp();
            }
            trace_loader(32, trace_aux);   // TILE_LOADS_DONE (last k-block's TMA issued)
        };

        scheduler.for_each_block_replay(tile_table,
            [&](const uint32_t& local_expert_idx,
                const uint32_t& num_k_blocks,
                const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_a_sfa_block(
                    std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                    local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            },
            [&](const uint32_t& local_expert_idx,
                const uint32_t& num_k_blocks,
                const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_a_sfa_block(
                    std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                    local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        // publish only into the initialised table (see the A loader / the dispatch warps)
        ptx::sync_aligned(kNumDispatchWithLoadersThreads, kDispatchWithLoadersBarrierIdx);
        bool bfd_pending = kBFillDefer;   // deferred ring fill: the ring fill waits for the CTA's dispatch warps' first-row publishes

        // early combine: the previous tile was an L2 tile whose "scatter issued" signal is still owed (see smem_ec_tiles)
        bool ec_prev_l2 = false;
        auto process_b_block = [&](const sched::BlockPhase& block_phase,
                                   const uint32_t& local_expert_idx,
                                   const uint32_t& num_k_blocks,
                                   const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

            // B multicast validity.
            uint32_t num_multicast_b = 1;
            if constexpr (kMulticastOnB and kClusterSize > 1) {
                const bool pair_valid = scheduler.is_pair_valid(cute::block_rank_in_cluster() == 0);
                num_multicast_b = pair_valid ? kClusterSize : 1;
            }

            const uint32_t k_block_base = scheduler.get_split_k_base(block_phase);
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if constexpr (kBFillDefer) {
                    // deferred ring fill: the first stage goes out at once, the rest of the ring fill after the first pulled rows
                    if (bfd_pending and k_block_idx == 1) {
                        if (lane_idx == 0)
                            while (ptx::ld_acquire_cta_shared(smem_ec_gs_old) < kNumDispatchWarps);
                        __syncwarp();
                        bfd_pending = false;
                    }
                }
                empty_barriers[stage_idx]->wait(phase ^ 1);
                // early combine: this wait returned once every consumer warp released this tile's k-block 0 (the stage kNumStages
                // k-blocks back), so all of them have finished the previous tile's epilogue: signal that L2 tile's scatter as
                // issued (release.cta: the math warps' arrives are release.cta, this wait is acquire.cta -> the stores are ordered)
                if constexpr (kEarlyCombine) {
                    if (k_block_idx == kNumStages and ec_prev_l2) {
                        if (lane_idx == 0)
                            ptx::red_release_cta_shared_add(smem_ec_tiles, 1u);
                        ec_prev_l2 = false;
                    }
                }

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    const uint32_t k_idx = (k_block_base + k_block_idx) * BLOCK_K;

                    // TMA load B (weight SF is now loaded directly by math warps from global)
                    if constexpr (kMXFP4Weights) {
                        // packed E2M1 rows as raw bytes (K/2 = 64 B per row, SWIZZLE_64B) + the SFB slice: [N] u32 words of
                        // k-block k of the [E, K/128, N, 4] payload, contiguous because the payload is N-inner
                        if constexpr (kMXFP4Tiled) {
                            // tiled layout: the stage is the contiguous 16 KiB chunk (n_idx / 256, k-block) of [E * N / 256, K / 128]
                            const uint8_t* w_base = block_phase == sched::BlockPhase::Linear2 ? l2_mxfp4_tiled_w : l1_mxfp4_tiled_w;
                            const uint32_t num_k_chunks = (block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K) / BLOCK_K;
                            ptx::tma_load_1d(smem_b[stage_idx],
                                             w_base + (static_cast<uint64_t>(n_idx / BLOCK_N) * num_k_chunks + k_block_base + k_block_idx) * SMEM_B_SIZE_PER_STAGE,
                                             full_barriers[stage_idx], SMEM_B_SIZE_PER_STAGE, cute::TMA::CacheHintSm100::EVICT_NORMAL);
                        } else {
                            tma::copy<BLOCK_K / 2, LOAD_BLOCK_N, BLOCK_K / 2, uint8_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx], reinterpret_cast<uint8_t*>(smem_b[stage_idx]),
                                k_idx / 2, n_idx, 1);
                        }
                        const uint8_t* sfb_base = block_phase == sched::BlockPhase::Linear2 ? l2_mxfp4_sfb : l1_mxfp4_sfb;
                        const uint32_t sfb_k_stride = (block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K) / kMXFP4GranK;
                        // (expert * shape_n + n) rows of sfb_k_stride bytes per expert, re-laid out as [K/128][N][4] per expert
                        const uint8_t* sfb_src = sfb_base +
                            static_cast<uint64_t>(local_expert_idx) * shape_n * sfb_k_stride +
                            static_cast<uint64_t>(k_block_base + k_block_idx) * shape_n * (BLOCK_K / kMXFP4GranK) +
                            static_cast<uint64_t>(n_block_idx) * BLOCK_N * (BLOCK_K / kMXFP4GranK);
                        ptx::tma_load_1d(smem_sfb[stage_idx], sfb_src, full_barriers[stage_idx], SMEM_SFB_SIZE_PER_STAGE,
                                         cute::TMA::CacheHintSm100::EVICT_NORMAL);
                    } else if constexpr (LOAD_BLOCK_N <= 256) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                            k_idx, n_idx, num_multicast_b);
                    } else {
                        DG_STATIC_ASSERT(LOAD_BLOCK_N % 256 == 0,
                                         "Large B tiles are loaded as 256-column TMA slices");
                        #pragma unroll
                        for (uint32_t b_slice_idx = 0; b_slice_idx < LOAD_BLOCK_N / 256; ++ b_slice_idx) {
                            tma::copy<BLOCK_K, 256, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx],
                                smem_b[stage_idx] + b_slice_idx * 256 * BLOCK_K,
                                k_idx, n_idx + b_slice_idx * 256, num_multicast_b);
                        }
                    }

                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
            if constexpr (kEarlyCombine) {
                DG_STATIC_ASSERT(kNumL2BlockKs > kNumStages and kNumL1BlockKs > kNumStages, "the early-combine signal needs k-block kNumStages of the next tile");
                ec_prev_l2 = block_phase == sched::BlockPhase::Linear2;
            }
        };
        // MXFP4 split-M: the swap / half tail tiles (last m-blocks of at most 96 rows) end the dynamic tail (see the scheduler)
        scheduler.template for_each_block_publish_dynamic_tail<kClusterSize, kMXFP4SplitM>(
            tile_table, workspace.template get_grid_sync_count_ptr<kTailCounterGridSyncIndex>(),
            kClusterSize > 1 ? cute::block_rank_in_cluster() : 0u, kNumTileTableEntries, process_b_block);

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (kNumDispatchWarps+2, +3). They must still
        // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
        // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
    // =====================================================================
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
    // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        // When the two N-split warpgroups share a single per-64 SF group they
        // also stage into ONE shared row-major L1-output tile (stride
        // L1_OUT_BLOCK_N), each writing its own WG_L1_OUT_BLOCK_N-column half,
        // so a single combined TMA store matches the host descriptor box.
        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N =
            kSplitNSharesSF ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;
        constexpr uint32_t WG_SMEM_CD_L2_STRIDE_N = WG_BLOCK_N;

        // Sync with dispatch in the full communication path.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // trace roles 2/3: warp 0 lane 0 of math warpgroups 0/1 (further warpgroups are not traced)
        uint32_t trace_tile = 0;
        const auto trace_math = [&](const uint32_t& event_id, const uint64_t& aux) {
            if constexpr (kTrace) {
                if (warp_idx_in_wg == 0 and lane_idx == 0 and epilogue_wg_idx < 2)
                    trace_event(2 + epilogue_wg_idx, event_id, aux);
            }
        };
        // epilogue sub-events (kTraceEpi): L1 50 EPI_MATH_DONE 51 EPI_STAGED 52 EPI_STORE_ISSUED 53 EPI_STORE_WAITED
        // 54 EPI_PUBLISHED; L2 60 EPI_CVT_DONE / 61 EPI_SCATTER_DONE (aux = pass) 62 EPI_FULL_SYNC_DONE
        const auto trace_epi = [&](const uint32_t& event_id, const uint64_t& aux) {
            if constexpr (kTrace and kTraceEpi)
                trace_math(event_id, aux);
        };
        DG_STATIC_ASSERT(kTrace or not kTraceEpi, "epilogue sub-events need the trace");

        // Weight-SF prologue prefetch (decode topology; hidden <= 4096 and intermediate <= 2048 so one
        // warp's lanes stage the whole weight-SF row): math warp 0 peeks the NEXT tile-table entry right after the mainloop
        // (non-blocking; an unpublished entry falls back to the plain prologue) and loads that tile's weight SF row into
        // registers, which the next prologue stores: the same values, the same smem layout.
        constexpr bool kSFProloguePrefetch = kDecodeTopology and kHidden / 128 <= 32 and 2 * (kIntermediateHidden / 128) <= 32 and
                                             not kMXFP4Weights;
        // the next tile's weight SF, loaded by math warp 0 (lanes j < 32: L1 gate / up SF of
        // k-block j, L2 lo / hi SF) under the drain of the current tile's k-block num_k - 2 and stored in the next prologue.
        // sf_pf_key identifies the tile the values belong to (1 | is_l1 << 1 | sf_n_block << 2 | expert << 10; 0 = none).
        uint32_t sf_pf_key = 0;
        float sf_pf_v0 = 0.0f, sf_pf_v1 = 0.0f;

        // CTA-wide barrier of the math warps between tile phases
        const auto sync_tile_math = [&]() {
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        };
        const bool tile_lead_warp = epilogue_warp_idx == 0;
        // MXFP4 expanded-B buffer of the next decode (kMXFP4ExpandedBuffers == 2), carried across tiles
        uint32_t mxfp4_buf = 0;
        // MXFP4 split-M: the first k-block of the next tile was already decoded (into mxfp4_buf) under the last WGMMA of this tile
        bool mxfp4_pre_decoded = false;
        // kMXFP4SplitMDeferPublish: the L1 arrival counter this warpgroup still owes one red.release (its TMA store not yet drained);
        // warpgroup-uniform. Flushed before the warpgroup's first wait that could depend on it: at the start of the next tile's mainloop
        // (before the full-barrier wait) unless that tile's first k-block was pre-decoded, else right after the first WGMMA issue. A
        // pre-decoded first k-block means its full barrier had completed, i.e. the loader had passed the tile's arrival wait: none of the
        // tile's loads, and so none of the peer's waits in it, can depend on the pending release. Every warpgroup flushes before it
        // blocks, so no wait cycle can form through pending releases (the end of the schedule flushes after the replay)
        uint32_t* mxfp4_pending_pub = nullptr;
        const auto mxfp4_flush_pub = [&]() {
            if constexpr (kMXFP4SplitMDeferPublish) {
                if (mxfp4_pending_pub != nullptr) {
                    // one lane of warp 0 issued the TMA store: the whole warp drains (a no-op without a pending bulk group), then one
                    // lane releases, as the epilogue's wait / sync / release did
                    if (warp_idx_in_wg == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                        if (cute::elect_one_sync())
                            ptx::red_add_rel(mxfp4_pending_pub, 1u);
                        __syncwarp();
                    }
                    mxfp4_pending_pub = nullptr;
                }
            }
        };
        // kMXFP4SplitCDFree: this warp owes the sync of the last L1 tile's CD-free barrier (never set in the storing warp)
        bool mxfp4_cd_free_pending = false;
        const auto mxfp4_cd_free_sync = [&]() {
            if constexpr (kMXFP4SplitCDFree) {
                if (mxfp4_cd_free_pending) {
                    ptx::sync_aligned(kNumEpilogueThreads, kMXFP4CDFreeBarrierIdx);
                    mxfp4_cd_free_pending = false;
                }
            }
        };
        // kTrace only: mainloop cycle totals of this warpgroup over all its tiles, [L1, L2] x [full wait, MXFP4 decode, rest]; written
        // once at TILES_DONE to the last three trace slots of the role (plain stores, independent of the event cursor)
        uint64_t prof_total[2][6] = {};
        // swap-loop sub-split of `rest` (kTrace only): WGMMA wait<0> / promote + stage release / trailing warpgroup barrier
        uint32_t prof_sub[3] = {};

        // forced inline (SASS-neutral on the untraced FP8 / MXFP4 instances); see also kForceInlineMathTiles at the tile replay
        auto process_math_block = [&](const auto& block_phase,
                                      const uint32_t& local_expert_idx,
                                      const uint32_t& num_k_blocks,
                                      const uint32_t& m_block_idx, const uint32_t& n_block_idx) CUTLASS_LAMBDA_FUNC_INLINE {
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t ring_block_idx = get_ring_block_idx(pool_block_idx);
            const uint32_t m_idx = pool_block_idx * BLOCK_M;            // Full-pool offset for metadata
            const uint32_t ring_m_idx = ring_block_idx * BLOCK_M;       // Ring offset for data buffers
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            // TILE_START: aux = phase | expert << 8 | m_block << 24 | n_block << 32 | tile ordinal << 40
            trace_math(40, static_cast<uint64_t>(block_phase == sched::BlockPhase::Linear1 ? 1u : 2u) |
                           (static_cast<uint64_t>(local_expert_idx) << 8) |
                           (static_cast<uint64_t>(m_block_idx) << 24) |
                           (static_cast<uint64_t>(n_block_idx) << 32) |
                           (static_cast<uint64_t>(trace_tile) << 40));
            if constexpr (kTrace)
                ++ trace_tile;
            const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
            const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
            const uint32_t wg_n_offset = epilogue_wg_n_idx * WG_BLOCK_N;
            // first B row this warpgroup decodes (split-M: its half of the shared tile; split-N: the rows it multiplies)
            const uint32_t mxfp4_decode_n_offset = kMXFP4SplitM ? epilogue_wg_idx * (BLOCK_N / kNumEpilogueWarpgroups) : wg_n_offset;
            const uint32_t wg_l1_out_n_offset = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t row_base = epilogue_wg_m_idx * WG_BLOCK_M;
            const uint32_t row_offset_r0 = row_base + r_0;
            const uint32_t row_offset_r1 = row_base + r_1;
            // 128-column weight-SF block of the warpgroup's first column (L1: gate/up block pair; L2: lo block)
            const uint32_t sf_n_block_idx = kSplitNSharesWeightSF ? n_block_idx
                : ((n_block_idx * kWarpgroupSplitN + epilogue_wg_n_idx) * kNumSFGroupsPerWG);
            // L2 act-SF group (post-SwiGLU columns / kL2ActsSFGranK) of the warpgroup's first L1 output column
            const uint32_t l2_sf_group_base = kSplitNSharesSF ? n_block_idx
                : ((n_block_idx * kWarpgroupSplitN + epilogue_wg_n_idx) * kNumL1OutSFGroups);
            const uint32_t smem_a_wg_offset = epilogue_wg_m_idx * WG_BLOCK_M * BLOCK_K;
            const uint32_t smem_b_wg_offset = epilogue_wg_n_idx * WG_BLOCK_N * BLOCK_K;
            // In the shared-tile case the WG stages into the joint L1-output tile
            // at its own column offset (row stride L1_OUT_BLOCK_N); otherwise each
            // WG owns a disjoint contiguous WG_BLOCK_M x WG_L1_OUT_BLOCK_N slice.
            const uint32_t smem_cd_l1_wg_offset =
                kSplitNSharesSF ? wg_l1_out_n_offset : (epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N);
            // With kHalfL2CD the L2 BF16 tile is staged one N-half at a time, so
            // each WG's slot is only WG_BLOCK_N/2 wide (see the L2 epilogue).
            const uint32_t smem_cd_l2_wg_offset =
                epilogue_wg_idx * WG_BLOCK_M * (WG_BLOCK_N / kNumL2CDPasses);
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
            constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
            constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
            constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
            constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
            float* smem_weight_sf_wg = smem_weight_sf + epilogue_wg_idx * kNumWeightSFFloatsPerWG;
            const uint32_t thread_idx_in_wg = warp_idx_in_wg * 32 + lane_idx;

            // the two warpgroup barriers order the weight-SF staging below against the previous tile's readers. MXFP4 split-M stages
            // nothing (its only writes to the slot are the L2 stmatrix junk rows, never read) and every later cross-warp hand-off of
            // the tile has its own barrier, so they only made the warpgroup wait for its slowest warp before the next mainloop
            if constexpr (not kMXFP4SplitM)
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
            // Without them, the WAR of the previous tile's epilogue smem reads by one warp (the L2 staging LDS; the L1 TMA store, drained
            // by the deferred flush at k-block 0) against this tile's epilogue smem writes by a sibling warp is ordered by the mainloop:
            // own free_arrive (k 0) -> peer free_wait (its k 1 decode) -> peer ready_arrive -> own ready_wait (k 2). The flush itself
            // always came after these barriers, so the deferred publish relied on that chain already; it needs three k-blocks
            DG_STATIC_ASSERT(not kMXFP4SplitM or (kHidden / BLOCK_K >= 3 and kIntermediateHidden / BLOCK_K >= 3),
                             "MXFP4 split-M: the cross-tile smem WAR relies on at least three k-blocks per tile");

            // prologue prefetch: warp 0 holds this tile's SF in registers when its peek at the previous tile's k-block num_k - 2 saw this
            // entry published (warp-uniform: the key is lane-uniform in warp 0 and 0 in the other warps, whose staging loop
            // is empty anyway). The stores below are the loads of the else branch evaluated earlier: same values.
            bool sf_pf_hit = false;
            if constexpr (kSFProloguePrefetch) {
                const uint32_t my_key = 1u | ((block_phase == sched::BlockPhase::Linear1 ? 1u : 0u) << 1) | (sf_n_block_idx << 2) |
                                        (local_expert_idx << 10);
                sf_pf_hit = sf_pf_key == my_key;
                sf_pf_key = 0;
            }
            if constexpr (kMXFP4Weights) {
                // no block weight SF: the K32 scale is folded into the decoded E4M3 bytes, the per-expert secondary is read below
            } else if (kSFProloguePrefetch and sf_pf_hit) {
                if (block_phase == sched::BlockPhase::Linear1) {
                    if (thread_idx_in_wg < kL1SFKBlocks) {
                        smem_weight_sf_wg[thread_idx_in_wg] = sf_pf_v0;
                        smem_weight_sf_wg[kL1SFKBlocks + thread_idx_in_wg] = sf_pf_v1;
                    }
                } else {
                    if (thread_idx_in_wg < kNumSFGroupsPerWG * kL2SFKBlocks)
                        smem_weight_sf_wg[thread_idx_in_wg] = sf_pf_v0;
                }
            } else if (block_phase == sched::BlockPhase::Linear1) {
                const uint32_t gate_n = sf_n_block_idx / 2u;
                const uint32_t up_n   = kL1SFGateBlks + gate_n;
                const float* expert_base = l1_weights_sf + local_expert_idx * kL1SFPerExpert;
                #pragma unroll
                for (uint32_t j = thread_idx_in_wg; j < kL1SFKBlocks; j += 128) {
                    smem_weight_sf_wg[j] = __ldg(expert_base + gate_n * kL1SFKBlocks + j);
                    smem_weight_sf_wg[kL1SFKBlocks + j] = __ldg(expert_base + up_n * kL1SFKBlocks + j);
                }
            } else {
                const float* sf_row = l2_weights_sf + local_expert_idx * kL2SFPerExpert + sf_n_block_idx * kL2SFKBlocks;
                #pragma unroll
                for (uint32_t j = thread_idx_in_wg; j < kNumSFGroupsPerWG * kL2SFKBlocks; j += 128)
                    smem_weight_sf_wg[j] = __ldg(sf_row + j);
            }
            if constexpr (not kMXFP4SplitM)
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
            trace_math(41, kSFProloguePrefetch ? (sf_pf_hit ? 1u : 0u) : 0u);   // SF_READY (weight SF staged; mainloop about to start; aux 1 = prefetched)

            // prologue prefetch: warp 0 peeks the tile table's next entry (a not-ready tag (3) or the end marker (0) leaves
            // sf_pf_key at 0 and the next prologue loads as before) and issues the next tile's weight-SF loads into registers:
            // lane j takes k-block j's (gate, up) SF for an L1 tile, the lo / hi SF for an L2 tile (event 48 SF_PEEK in the trace).
            const auto sf_peek = [&]() {
                if constexpr (kSFProloguePrefetch) {
                    using Sched = std::remove_reference_t<decltype(scheduler)>;
                    const TileEntry* next_entry = tile_table + scheduler.replay_next_idx;
                    const uint32_t tag_word = Sched::ld_acquire_tile_tag(next_entry);
                    const uint32_t tag = tag_word >> 30;
                    trace_math(48, tag | (2u << 4));   // SF_PEEK (aux = tag | 2 << 4: the peek after the mainloop)
                    if (tag == 1u or tag == 2u) {
                        uint32_t nx_expert, nx_n_block;
                        nx_expert = tag_word & ((1u << Sched::kEBits) - 1u);
                        nx_n_block = (tag_word >> (Sched::kEBits + Sched::kMBits)) & ((1u << Sched::kNBits) - 1u);
                        const uint32_t nx_sf_n = kSplitNSharesWeightSF ? nx_n_block
                            : ((nx_n_block * kWarpgroupSplitN + epilogue_wg_n_idx) * kNumSFGroupsPerWG);
                        if (tag == 1u) {
                            const uint32_t gate_n = nx_sf_n / 2u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* expert_base = l1_weights_sf + nx_expert * kL1SFPerExpert;
                            if (lane_idx < kL1SFKBlocks) {
                                sf_pf_v0 = __ldg(expert_base + gate_n * kL1SFKBlocks + lane_idx);
                                sf_pf_v1 = __ldg(expert_base + up_n * kL1SFKBlocks + lane_idx);
                            }
                        } else {
                            const float* sf_row = l2_weights_sf + nx_expert * kL2SFPerExpert + nx_sf_n * kL2SFKBlocks;
                            if (lane_idx < kNumSFGroupsPerWG * kL2SFKBlocks)
                                sf_pf_v0 = __ldg(sf_row + lane_idx);
                        }
                        sf_pf_key = 1u | ((tag == 1u ? 1u : 0u) << 1) | (nx_sf_n << 2) | (nx_expert << 10);
                    }
                }
            };

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;

            DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 16 == 0 and SMEM_B_SIZE_PER_STAGE % 16 == 0,
                             "Descriptor folding needs 16B-aligned stage strides");
            const auto desc_a_stage0 = mma::sm90::make_smem_desc(smem_a[0], 1);
            const auto desc_b_stage0 = mma::sm90::make_smem_desc(kMXFP4Weights ? smem_expanded_b : smem_b[0], 1);
            auto smem_desc_a = [&](const uint32_t& stage, const uint32_t& byte_offset) {
                return mma::sm90::advance_smem_desc(desc_a_stage0, stage * SMEM_A_SIZE_PER_STAGE + byte_offset);
            };
            auto smem_desc_b = [&](const uint32_t& stage, const uint32_t& byte_offset) {
                if constexpr (kMXFP4Weights)
                    return mma::sm90::advance_smem_desc(desc_b_stage0, byte_offset);
                else
                    return mma::sm90::advance_smem_desc(desc_b_stage0, stage * SMEM_B_SIZE_PER_STAGE + byte_offset);
            };

            // MXFP4 weight scale of the tile's expert: decoded E4M3 = e2m1 * 2^offset / 64 and the logical scale is
            // 2^offset * secondary, so the promotion multiplies by secondary * 64 in place of the block weight SF. As in PR #411,
            // the x64 goes on the secondary unless that would overflow (secondary > 2^121), then on the activation scale; the
            // branch is uniform per expert
            float mxfp4_weight_scale = 1.0f, mxfp4_act_scale_mul = 1.0f;
            constexpr float kMaxSecondaryBeforeX64 = 0x1p121f;
            if constexpr (kMXFP4Weights) {
                const float secondary = __ldg((block_phase == sched::BlockPhase::Linear1 ? l1_mxfp4_secondary : l2_mxfp4_secondary) +
                                              local_expert_idx);
                if constexpr (kMXFP4SplitM) {
                    // split-M: the raw secondary, compensated after the mainloop (mxfp4_splitm_mainloop). Compensated here, its compare
                    // waited for the load in front of the first WGMMA of a pre-decoded tile (no full-barrier wait in between)
                    mxfp4_weight_scale = secondary;
                } else {
                    const bool compensate_secondary = secondary <= kMaxSecondaryBeforeX64;
                    mxfp4_weight_scale = compensate_secondary ? secondary * 64.0f : secondary;
                    mxfp4_act_scale_mul = compensate_secondary ? 1.0f : 64.0f;
                }
            }

            // MXFP4 decode of one stage into this warpgroup's WG_BLOCK_N rows of expanded tile `buf` (the PR #411 two-lane-per-word
            // mapping: a half warp covers 8 rows x 2 of the 4 packed words of each K32 group, the other half warp the other 2).
            // The packed stage is TMA SWIZZLE_64B (Swizzle<2, 4, 3>), the expanded tile the SWIZZLE_128B layout an FP8 TMA writes
            // (Swizzle<3, 4, 3>). Barriers and the async-proxy fence are the caller's (see mxfp4_mainloop)
            const auto decode_mxfp4_stage = [&](const uint32_t& stage, const uint32_t& buf) {
                if constexpr (kMXFP4Weights) {
                    constexpr uint32_t kPackedRowBytes = BLOCK_K / 2;
                    constexpr uint32_t kPackedWordsPerK32 = kMXFP4GranK / 8;
                    constexpr uint32_t kNumK32PerBlock = BLOCK_K / kMXFP4GranK;
                    constexpr uint32_t kRowsPerDecodeGroup = 8;
                    DG_STATIC_ASSERT(kPackedWordsPerK32 == 4 and kNumK32PerBlock == 4 and kPackedRowBytes == 64,
                                     "MXFP4 decode mapping assumes BLOCK_K 128: 64 packed bytes = 4 K32 groups x 4 words per row");
                    const auto packed = reinterpret_cast<const uint8_t*>(smem_b[stage]);
                    auto expanded = reinterpret_cast<uint8_t*>(smem_expanded_b) + buf * SMEM_EXPANDED_B_TILE_SIZE;
                    const uint32_t thread_idx_in_wg = warp_idx_in_wg * 32 + lane_idx;
                    // row (mxfp4_decode_n_offset + thread) owns the SFB word of its 4 K32 groups
                    const uint32_t scale_word = ptx::ld_shared(smem_sfb[stage] + mxfp4_decode_n_offset + thread_idx_in_wg);
                    if constexpr (kMXFP4Tiled) {
                        // tiled stage [4 K32 groups][256 rows][16 B] -> no-swizzle K-major expanded tile: byte (n, k) at
                        // (n / 8) * 1024 + (k / 16) * 128 + (n % 8) * 16 + k % 16. Lane owns row n; K32 group g is one LDS.128
                        // (4 packed words = 32 K) and two STS.128 into 16 B chunks 2g, 2g + 1 of the row. Word c of a unit holds
                        // K 4c..4c+3 (decoded low half) and K 16+4c..16+4c+3 (high half), see `_tile_mxfp4_weights_for_sm90`
                        constexpr uint32_t kGroupBytes = BLOCK_N * 16;
                        const uint32_t n = mxfp4_decode_n_offset + thread_idx_in_wg;
                        const uint8_t* src = packed + n * 16u;
                        uint8_t* dst = expanded + (n / 8u) * 1024u + (n % 8u) * 16u;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumK32PerBlock; ++ g) {
                            const uint4 w = ptx::ld_shared(reinterpret_cast<const uint4*>(src + g * kGroupBytes));
                            const uint2 lut = sm90_mxfp4_e4m3_lookup((scale_word >> (g * 8u)) & 0xffu);
                            const uint2 d0 = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(w.x, lut);
                            const uint2 d1 = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(w.y, lut);
                            const uint2 d2 = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(w.z, lut);
                            const uint2 d3 = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(w.w, lut);
                            ptx::st_shared(dst + (2u * g) * 128u, d0.x, d1.x, d2.x, d3.x);
                            ptx::st_shared(dst + (2u * g + 1u) * 128u, d0.y, d1.y, d2.y, d3.y);
                        }
                        return;
                    }
                    if constexpr (kMXFP4DecodeMode == 2) {
                        // lane owns row n = mxfp4_decode_n_offset + thread: the swizzle terms of the row are loop invariant. Lanes 8..23 visit
                        // the K32 group's word pair (2, 3) first, so the 8 rows of one 128 B bank window read both 8 B halves of
                        // their 16 B packed chunk (PR #411 `__byte_perm(0x00020200, 0, lane >> 3)`); stores: 8 consecutive rows hit 8
                        // distinct 16 B chunks of the SWIZZLE_128B row
                        const uint32_t n = mxfp4_decode_n_offset + thread_idx_in_wg;
                        const uint32_t packed_row_base = n * kPackedRowBytes;
                        const uint32_t packed_row_xor = cute::Swizzle<2, 4, 3>::apply(packed_row_base) ^ packed_row_base;
                        const uint32_t first_word = __byte_perm(0x00020200u, 0u, lane_idx >> 3);
                        #pragma unroll
                        for (uint32_t k32_idx = 0; k32_idx < kNumK32PerBlock; ++ k32_idx) {
                            const uint2 lut = sm90_mxfp4_e4m3_lookup((scale_word >> (k32_idx * 8u)) & 0xffu);
                            #pragma unroll
                            for (uint32_t p = 0; p < 2; ++ p) {
                                const uint32_t word = k32_idx * kPackedWordsPerK32 + (first_word ^ (p * 2u));
                                const uint2 w = ptx::ld_shared(reinterpret_cast<const uint2*>(
                                    packed + packed_row_base + ((word * sizeof(uint32_t)) ^ packed_row_xor)));
                                const uint2 d0 = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(w.x, lut);
                                const uint2 d1 = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(w.y, lut);
                                ptx::st_shared(expanded + cute::Swizzle<3, 4, 3>::apply(n * BLOCK_K + word * 8u), d0.x, d0.y, d1.x, d1.y);
                            }
                        }
                        return;
                    }
                    if constexpr (kMXFP4DecodeMode == 1) {
                        // lane = (row in a group of 4) x (word pair 0..7); a pair is words 2p, 2p + 1 of the 16-word row, i.e. half of
                        // K32 group p / 2. An 8-lane quarter warp covers one row: 8 LDS.64 on one 64 B packed row, 8 STS.128 on the
                        // 8 distinct 16 B chunks of one 128 B expanded row (the Swizzle<3,4,3> permutation of a row is a bijection)
                        const uint32_t pair = lane_idx % 8;
                        const uint32_t k32_idx = pair / 2;
                        #pragma unroll
                        for (uint32_t row_group = 0; row_group < 32 / 4; ++ row_group) {
                            const uint32_t row_in_warp = row_group * 4 + lane_idx / 8;
                            const uint32_t n = mxfp4_decode_n_offset + warp_idx_in_wg * 32 + row_in_warp;
                            const uint32_t row_scale_word = __shfl_sync(0xffffffffu, scale_word, row_in_warp);
                            const uint32_t exponent_offset = (row_scale_word >> (k32_idx * 8u)) & 0xffu;
                            const uint32_t packed_row_base = n * kPackedRowBytes;
                            const uint32_t packed_row_xor = cute::Swizzle<2, 4, 3>::apply(packed_row_base) ^ packed_row_base;
                            const uint2 words = ptx::ld_shared(reinterpret_cast<const uint2*>(
                                packed + packed_row_base + ((pair * sizeof(uint2)) ^ packed_row_xor)));
                            const uint4 decoded = sm90_mxfp4_reordered_signs_e2m1x16_to_e4m3x16_bits(words.x, words.y, exponent_offset);
                            const uint32_t swizzled = cute::Swizzle<3, 4, 3>::apply(n * BLOCK_K + pair * 16u);
                            ptx::st_shared(expanded + swizzled, decoded.x, decoded.y, decoded.z, decoded.w);
                        }
                        return;
                    }
                    const uint32_t lane_in_half_warp = lane_idx % 16;
                    const uint32_t row_in_decode_group = lane_in_half_warp / 2;
                    const uint32_t packed_k_in_k32 = (lane_idx / 16) * 2 + lane_in_half_warp % 2;
                    #pragma unroll
                    for (uint32_t row_group = 0; row_group < 32 / kRowsPerDecodeGroup; ++ row_group) {
                        const uint32_t row_in_warp = row_group * kRowsPerDecodeGroup + row_in_decode_group;
                        const uint32_t n = mxfp4_decode_n_offset + warp_idx_in_wg * 32 + row_in_warp;
                        const uint32_t row_scale_word = __shfl_sync(0xffffffffu, scale_word, row_in_warp);
                        const uint32_t packed_row_base = n * kPackedRowBytes;
                        const uint32_t packed_row_xor = cute::Swizzle<2, 4, 3>::apply(packed_row_base) ^ packed_row_base;
                        #pragma unroll
                        for (uint32_t k32_idx = 0; k32_idx < kNumK32PerBlock; ++ k32_idx) {
                            const uint32_t packed_k = k32_idx * kPackedWordsPerK32 + packed_k_in_k32;
                            const uint32_t exponent_offset = (row_scale_word >> (k32_idx * 8u)) & 0xffu;
                            const uint32_t word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(
                                packed + packed_row_base + ((packed_k * sizeof(uint32_t)) ^ packed_row_xor)));
                            const uint2 decoded = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(word, exponent_offset);
                            const uint32_t swizzled = cute::Swizzle<3, 4, 3>::apply(n * BLOCK_K + packed_k * 8u);
                            ptx::st_shared(expanded + swizzled, decoded.x, decoded.y);
                        }
                    }
                }
            };

            // Register-A decode of one tiled stage (kMXFP4DecodeMode 4..7): lane (warp w, lane l, c = l % 4) builds the m64k32 E4M3 A
            // fragments of the 4 weight rows wg_n_offset + 64h + 16w + 8s + l / 4 (h, s = 0, 1) for the 4 K32 groups g; word c of
            // unit (g, row) decodes to (K 4c..4c+3, K 16+4c..16+4c+3) = registers (s, 2 + s) of fragment (h, g). Each quarter warp
            // reads 8 consecutive 16 B units: one wavefront per LDS.32. The decode is ALU-bound (the swap loop keeps the ALU pipe ~60 %
            // busy), so the lane's word / scale addresses are two 32-bit shared addresses and every (h, s, g) term is an LDS immediate,
            // and the scale bytes are PRMT / mul.hi (microbench: N8 / N16 / N24 -11 / -7 / -4 % per k-block, ALU 136 -> 108 per lane)
            const uint32_t smem_b_stage0_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_b[0]));
            const uint32_t smem_sfb_stage0_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_sfb[0]));
            const auto decode_mxfp4_stage_regs = [&](const uint32_t& stage, uint32_t (&frag)[2][4][4]) {
                if constexpr (kMXFP4Weights) {
                    DG_STATIC_ASSERT(BLOCK_K / kMXFP4GranK == 4, "register-A decode: 4 K32 groups per k-block");
                    const uint32_t lane_row = wg_n_offset + warp_idx_in_wg * 16 + lane_idx / 4;
                    const uint32_t word_addr = smem_b_stage0_addr + stage * SMEM_B_SIZE_PER_STAGE + lane_row * 16u + (lane_idx % 4) * 4u;
                    const uint32_t scale_addr = smem_sfb_stage0_addr + stage * SMEM_SFB_SIZE_PER_STAGE + lane_row * 4u;
                    // weight rows lane_row + 64h + 8s
                    sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 0, 0>(word_addr, scale_addr, frag[0]);
                    sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 8, 1>(word_addr, scale_addr, frag[0]);
                    sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 64, 0>(word_addr, scale_addr, frag[1]);
                    sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 72, 1>(word_addr, scale_addr, frag[1]);
                }
            };

            // MXFP4 mainloop of the promotion path (split-N, non-swap). Per k-block of one warpgroup, with the expanded tile(s)
            // private to the warpgroup's 128 rows (`wg_bar` is the warpgroup's named barrier):
            //   1 buffer : full wait -> wg_bar (WAR: every warp's previous WGMMA waited) -> decode -> fence -> wg_bar (RAW) ->
            //              WGMMA -> wait -> release -> promote
            //   2 buffers: (k-block 0 of the tile: full wait -> decode into buf -> fence -> wg_bar) then per k-block: WGMMA on buf ->
            //              full wait of k + 1 -> decode k + 1 into buf ^ 1 -> fence -> wait -> release -> promote -> wg_bar (RAW for
            //              buf ^ 1 and WAR for buf) -> buf ^= 1
            // The stage of k-block k is released only after its own WGMMA, so waiting for k + 1 while holding k needs >= 2 stages.
            // `mxfp4_buf` persists across tiles: a tile's first decode targets the buffer its predecessor's last WGMMA did not
            // read (both are free anyway: the predecessor ends on wg_bar after its last wait<0>).
            const auto mxfp4_mainloop = [&](float* accum, float* final_accum,
                                            uint32_t& prof_wait, uint32_t& prof_decode, uint32_t& prof_rest) {
                if constexpr (kMXFP4Weights) {
                    DG_STATIC_ASSERT(kNumStages >= 2, "MXFP4 decode lookahead holds one stage while waiting for the next");
                    const auto wg_bar = [&]() { ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx); };
                    const auto desc_b_buf0 = kMXFP4Tiled ? mma::sm90::make_smem_desc(smem_expanded_b + smem_b_wg_offset, 0, 128, 1024)
                                                         : mma::sm90::make_smem_desc(smem_expanded_b + smem_b_wg_offset, 1);
                    const auto issue_wgmma = [&](const uint32_t& stage, const uint32_t& buf, const uint32_t& k_begin, const uint32_t& num_k) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < num_k; ++ k) {
                            const uint32_t k_off = (k_begin + k) * WGMMA::K;
                            auto desc_a = smem_desc_a(stage, smem_a_wg_offset + k_off);
                            auto desc_b = mma::sm90::advance_smem_desc(desc_b_buf0, buf * SMEM_EXPANDED_B_TILE_SIZE + k_off * kMXFP4ExpandedKStep);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    };
                    const auto promote = [&](const float& s0, const float& s1) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += s0 * accum[i*4+0];
                            final_accum[i*4+1] += s0 * accum[i*4+1];
                            final_accum[i*4+2] += s1 * accum[i*4+2];
                            final_accum[i*4+3] += s1 * accum[i*4+3];
                        }
                    };
                    // one weight scale per expert (L1 gate and up share the L1 secondary); the activation scale carries the x64
                    // when the secondary could not
                    const float w_scale = mxfp4_weight_scale * mxfp4_act_scale_mul;
                    const bool per64 = block_phase == sched::BlockPhase::Linear2 and not kL2ActSFPerBlockK;
                    uint32_t prof_t = 0;
                    const auto prof_mark = [&](uint32_t& acc) {
                        if constexpr (kTrace) {
                            const uint32_t now = sm90_fp8_fused_mega_moe_prof_clock();
                            acc += now - prof_t;
                            prof_t = now;
                        }
                    };
                    if constexpr (kTrace)
                        prof_t = sm90_fp8_fused_mega_moe_prof_clock();

                    if constexpr (kMXFP4ExpandedBuffers == 2) {
                        if (num_k_blocks != 0) {
                            full_barriers[stage_idx]->wait(phase);
                            prof_mark(prof_wait);
                            decode_mxfp4_stage(stage_idx, mxfp4_buf);
                            cutlass::arch::fence_view_async_shared();
                            wg_bar();
                            prof_mark(prof_decode);
                        }
                    }
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        if constexpr (kMXFP4ExpandedBuffers == 1) {
                            full_barriers[stage_idx]->wait(phase);
                            prof_mark(prof_wait);
                            wg_bar();
                            decode_mxfp4_stage(stage_idx, 0);
                            cutlass::arch::fence_view_async_shared();
                            wg_bar();
                            prof_mark(prof_decode);
                        }
                        if constexpr (kTrace) {
                            if (k_block_idx == 0)
                                trace_math(42, 0);   // FIRST_STAGE_READY
                        }
                        const float* sfa = smem_sfa[stage_idx];
                        const float s0_lo = ptx::ld_shared(sfa + row_offset_r0) * w_scale;
                        const float s1_lo = ptx::ld_shared(sfa + row_offset_r1) * w_scale;
                        const uint32_t buf = kMXFP4ExpandedBuffers == 2 ? mxfp4_buf : 0u;
                        issue_wgmma(stage_idx, buf, 0, per64 ? (BLOCK_K / 2) / WGMMA::K : BLOCK_K / WGMMA::K);
                        if constexpr (kMXFP4ExpandedBuffers == 2) {
                            // decode the next k-block under this WGMMA
                            if (k_block_idx + 1 < num_k_blocks) {
                                const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                                const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                                prof_mark(prof_rest);
                                full_barriers[next_stage]->wait(next_phase);
                                prof_mark(prof_wait);
                                decode_mxfp4_stage(next_stage, buf ^ 1u);
                                cutlass::arch::fence_view_async_shared();
                                prof_mark(prof_decode);
                            }
                        }
                        ptx::warpgroup_wait<0>();
                        if (per64) {
                            promote(s0_lo, s1_lo);
                            const float s0_hi = ptx::ld_shared(sfa + BLOCK_M + row_offset_r0) * w_scale;
                            const float s1_hi = ptx::ld_shared(sfa + BLOCK_M + row_offset_r1) * w_scale;
                            issue_wgmma(stage_idx, buf, (BLOCK_K / 2) / WGMMA::K, (BLOCK_K / 2) / WGMMA::K);
                            ptx::warpgroup_wait<0>();
                            release_empty_stage(stage_idx);
                            promote(s0_hi, s1_hi);
                        } else {
                            release_empty_stage(stage_idx);
                            promote(s0_lo, s1_lo);
                        }
                        if constexpr (kMXFP4ExpandedBuffers == 2) {
                            wg_bar();
                            mxfp4_buf ^= 1u;
                        }
                        prof_mark(prof_rest);
                    }
                }
            };

            // MXFP4 swapAB mainloop (kMXFP4SwapAB): same stage / decode / barrier protocol as mxfp4_mainloop with 2 buffers, but per
            // k-block the warpgroup issues WGMMA 64 x N with A = expanded weight rows [h * 64, h * 64 + 64) of its 128 (h = 0, 1)
            // and B = the first N token rows of the A stage, N = the valid_m bucket. The activation scale is per accumulator column
            // (token), padded tokens get scale 0. After the loop the swapped sum is staged as [token][64] fp32 in this warpgroup's
            // 16 KiB region of expanded tile h (both tiles are idle: every warp passed the k-loop's trailing wg_bar after its
            // last wait<0>) and read back in the regular (token-row) layout into final_accum; the next tile's prologue barriers
            // order these generic accesses before its first decode overwrites the tiles
            // kMXFP4DirectSwapEpi: the swap mainloop leaves its accumulators in final_accum[h * kMXFP4SwapMaxBucket / 2 + i] (weight
            // half h, swap accumulator i) for the direct epilogue instead of transposing them into the regular layout
            // forced inline: with the N40 bucket (mode 7) NVVM compiled this lambda as a real function (STACK 2128, ~1000 LDL / 880 STL);
            // SASS-neutral on the mode 4 / 5 instances (notes §41)
            const auto mxfp4_swap_mainloop = [&](float* final_accum, uint32_t& prof_wait, uint32_t& prof_decode, uint32_t& prof_rest) CUTLASS_LAMBDA_FUNC_INLINE {
                if constexpr (kMXFP4Weights and kMXFP4SwapAB) {
                    const auto wg_bar = [&]() { ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx); };
                    const float w_scale = mxfp4_weight_scale * mxfp4_act_scale_mul;
                    const bool per64 = block_phase == sched::BlockPhase::Linear2 and not kL2ActSFPerBlockK;
                    const auto desc_w0 = kMXFP4Tiled ? mma::sm90::make_smem_desc(smem_expanded_b + smem_b_wg_offset, 0, 128, 1024)
                                                     : mma::sm90::make_smem_desc(smem_expanded_b + smem_b_wg_offset, 1);
                    uint32_t prof_t = 0;
                    const auto prof_mark = [&](uint32_t& acc) {
                        if constexpr (kTrace) {
                            const uint32_t now = sm90_fp8_fused_mega_moe_prof_clock();
                            acc += now - prof_t;
                            prof_t = now;
                        }
                    };
                    const auto run = [&]<uint32_t N>() {
                        using SwapWGMMA = typename mma::sm90::FP8MMASelector<N>::type;
                        constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                        DG_STATIC_ASSERT(kSwapAccum * 2 == N and N <= BLOCK_M, "swap bucket");
                        float acc[2][kSwapAccum];
                        float fin[2][kSwapAccum];
                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapAccum; ++ i)
                            fin[0][i] = 0.0f, fin[1][i] = 0.0f;
                        const auto issue = [&](const uint32_t& stage, const uint32_t& buf, const uint32_t& k_begin, const uint32_t& num_k) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(acc[0][i]), ptx::warpgroup_fence_operand(acc[1][i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++ h) {
                                #pragma unroll
                                for (uint32_t k = 0; k < num_k; ++ k) {
                                    const uint32_t k_off = (k_begin + k) * SwapWGMMA::K;
                                    const auto desc_a = mma::sm90::advance_smem_desc(
                                        desc_w0, buf * SMEM_EXPANDED_B_TILE_SIZE + h * 64 * BLOCK_K + k_off * kMXFP4ExpandedKStep);
                                    const auto desc_b = smem_desc_a(stage, k_off);
                                    SwapWGMMA::wgmma(desc_a, desc_b, acc[h], k);
                                }
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(acc[0][i]), ptx::warpgroup_fence_operand(acc[1][i]);
                        };
                        const auto promote = [&](const float* sfa) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t t0 = i * 8 + col_idx * 2, t1 = t0 + 1;
                                const float s0 = t0 < valid_m ? ptx::ld_shared(sfa + t0) * w_scale : 0.0f;
                                const float s1 = t1 < valid_m ? ptx::ld_shared(sfa + t1) * w_scale : 0.0f;
                                #pragma unroll
                                for (uint32_t h = 0; h < 2; ++ h) {
                                    fin[h][i * 4 + 0] += s0 * acc[h][i * 4 + 0];
                                    fin[h][i * 4 + 1] += s1 * acc[h][i * 4 + 1];
                                    fin[h][i * 4 + 2] += s0 * acc[h][i * 4 + 2];
                                    fin[h][i * 4 + 3] += s1 * acc[h][i * 4 + 3];
                                }
                            }
                        };
                        if constexpr (kTrace)
                            prof_t = sm90_fp8_fused_mega_moe_prof_clock();
                        // every bucket up to N48 (modes 6 / 7) runs the register-A loop; N > kMXFP4SwapMaxBucket transposes through the expanded
                        // tiles after the loop (the direct epilogue is written for 32 tokens)
                        if constexpr (kMXFP4DecodeMode >= 4) {
                            // register-A swap: A = the fragments decoded into registers, B = the A stage's token rows; no smem tile is
                            // written, so the k-loop has neither an async-proxy fence nor a warpgroup barrier. Up to N24 the fragments
                            // of k-block k + 1 are decoded into a second register set under the WGMMA of k-block k (the N24 second set
                            // spilled the regular mainloop before the phase-16 decode diet, notes §22 / §23). Modes 6 / 7 double-buffer N32 / N40 / N48
                            // too: one set serialises the decode before the WGMMA (trace N32: 1333 -> 1162 cyc per k-block, notes §39), but the
                            // second set costs the small calls ~1 % (register allocation of the whole function), so modes 4 / 5 keep one set
                            using RSWGMMA = typename mma::sm90::FP8MMARSSelector<N>::type;
                            DG_STATIC_ASSERT(RSWGMMA::kNumAccum == kSwapAccum, "RS / SS swap accumulators differ");
                            const auto fence_frag = [](uint32_t (&frag)[2][4][4]) {
                                #pragma unroll
                                for (uint32_t h = 0; h < 2; ++ h)
                                    #pragma unroll
                                    for (uint32_t g = 0; g < 4; ++ g)
                                        #pragma unroll
                                        for (uint32_t i = 0; i < 4; ++ i)
                                            asm volatile("" : "+r"(frag[h][g][i]) :: "memory");
                            };
                            constexpr bool kRSDoubleBuffer = N <= 24 or kMXFP4DecodeMode >= 6;
                            const auto issue_rs = [&]<uint32_t kBegin, uint32_t kNum, uint32_t kStage = kNumStages>(uint32_t (&frag)[2][4][4], const uint32_t& stage) {
                                // double-buffered buckets: the stage index is warp-uniform but loop-carried in a vector register;
                                // broadcasting it puts the token descriptors in the uniform datapath (no R2UR / vector address math
                                // per WGMMA; Flash / Pro M8..M32 -1.4..-2.8 %). Broadcasting in the single-set loops as well measured
                                // +1.3..+3.1 % at Flash M96 / M128 and Pro M48..M80 (there the WGMMA issue directly follows the decode),
                                // so the single-set N32 loop keeps the vector descriptors
                                // (kStage < kNumStages: the compile-time stage of the unrolled loop below, no broadcast needed)
                                const uint32_t desc_stage_idx = kStage < kNumStages ? kStage : (kRSDoubleBuffer ? __shfl_sync(0xffffffffu, stage, 0) : stage);
                                const auto desc_stage = smem_desc_a(desc_stage_idx, 0);
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[0][i]), ptx::warpgroup_fence_operand(acc[1][i]);
                                fence_frag(frag);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t h = 0; h < 2; ++ h) {
                                    #pragma unroll
                                    for (uint32_t k = 0; k < kNum; ++ k)
                                        RSWGMMA::wgmma(frag[h][kBegin + k], mma::sm90::advance_smem_desc(desc_stage, (kBegin + k) * RSWGMMA::K).desc_,
                                                       acc[h], k != 0);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[0][i]), ptx::warpgroup_fence_operand(acc[1][i]);
                            };
                            uint32_t frag0[2][4][4], frag1[2][4][kRSDoubleBuffer ? 4 : 1];
                            const auto step = [&](uint32_t (&cur)[2][4][4], uint32_t (&nxt)[2][4][4], uint32_t& k_block_idx) {
                                if (per64)
                                    issue_rs.template operator()<0, 2>(cur, stage_idx);
                                else
                                    issue_rs.template operator()<0, 4>(cur, stage_idx);
                                if (k_block_idx + 1 < num_k_blocks) {
                                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                                    const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                                    prof_mark(prof_rest);
                                    full_barriers[next_stage]->wait(next_phase);
                                    prof_mark(prof_wait);
                                    decode_mxfp4_stage_regs(next_stage, nxt);
                                    prof_mark(prof_decode);
                                }
                                ptx::warpgroup_wait<0>();
                                prof_mark(prof_sub[0]);
                                const float* sfa = smem_sfa[stage_idx];
                                promote(sfa);
                                if (per64) {
                                    issue_rs.template operator()<2, 2>(cur, stage_idx);
                                    ptx::warpgroup_wait<0>();
                                    promote(sfa + BLOCK_M);
                                }
                                // the fragments stay allocated until the WGMMA reading them completed
                                fence_frag(cur);
                                release_empty_stage(stage_idx);
                                prof_mark(prof_sub[1]);
                                advance_pipeline(k_block_idx);
                            };
                            // compile-time stages (notes §47): every tile consumes num_k_blocks stages of the ring and both GEMMs' k-block counts are
                            // multiples of kNumStages, so every tile starts at stage 0 and the k-loop unrolled by kNumStages knows each k-block's
                            // stage, fragment set and barrier at compile time (the runtime stage / phase / address arithmetic and the descriptor
                            // broadcast leave the issue-bound loop)
                            // only the N8 / N16 buckets of decode mode 4: unrolling every bucket grew the code 10..35 %, outlined the mode-5 tile
                            // callback (STACK 912: Flash M96 / Pro M48 2.0..2.3x) and slowed modes 6 / 7 by 1..5 % (notes §47). And only with
                            // >= 48 L1 k-blocks: Pro (56) M8..M32 -0.5..-1.3 %, Flash (32) M8..M32 +0.2..+0.9 % (cause not analysed)
                            constexpr bool kStaticStages = kRSDoubleBuffer and N <= 16 and kMXFP4DecodeMode == 4 and kNumL1BlockKs >= 48 and not kSplitK and
                                                           kNumStages % 2 == 0 and kNumL1BlockKs % kNumStages == 0 and kNumL2BlockKs % kNumStages == 0;
                            if constexpr (kStaticStages) {
                                constexpr uint32_t kNumKBlocks = std::remove_cvref_t<decltype(block_phase)>::value == sched::BlockPhase::Linear1 ?
                                                                 kNumL1BlockKs : kNumL2BlockKs;
                                constexpr uint32_t kNumRounds = kNumKBlocks / kNumStages;
                                DG_STATIC_ASSERT(kNumRounds > 0, "compile-time stages: at least one round of the ring per tile");
#ifdef DG_P29_CHECK_STAGE
                                if (stage_idx != 0) {
                                    printf("P29 stage check failed: stage_idx %u at the swap mainloop entry (block %u)\n", stage_idx, blockIdx.x);
                                    __trap();
                                }
#endif
                                // k-block (round r, stage kS) uses fragment set kS % 2; the stages of a round share `round_phase`
                                const auto step_s = [&]<uint32_t kS>(uint32_t (&cur)[2][4][4], uint32_t (&nxt)[2][4][4], const uint32_t& round_phase,
                                                                     const bool& has_next) CUTLASS_LAMBDA_FUNC_INLINE {
                                    constexpr uint32_t kNext = kS + 1 == kNumStages ? 0u : kS + 1;
                                    if (per64)
                                        issue_rs.template operator()<0, 2, kS>(cur, kS);
                                    else
                                        issue_rs.template operator()<0, 4, kS>(cur, kS);
                                    if (has_next) {
                                        prof_mark(prof_rest);
                                        full_barriers[kNext]->wait(kNext == 0 ? round_phase ^ 1u : round_phase);
                                        prof_mark(prof_wait);
                                        decode_mxfp4_stage_regs(kNext, nxt);
                                        prof_mark(prof_decode);
                                    }
                                    ptx::warpgroup_wait<0>();
                                    prof_mark(prof_sub[0]);
                                    const float* sfa = smem_sfa[kS];
                                    promote(sfa);
                                    if (per64) {
                                        issue_rs.template operator()<2, 2, kS>(cur, kS);
                                        ptx::warpgroup_wait<0>();
                                        promote(sfa + BLOCK_M);
                                    }
                                    // the fragments stay allocated until the WGMMA reading them completed
                                    fence_frag(cur);
                                    release_empty_stage(kS);
                                    prof_mark(prof_sub[1]);
                                };
                                full_barriers[0]->wait(phase);
                                prof_mark(prof_wait);
                                decode_mxfp4_stage_regs(0, frag0);
                                prof_mark(prof_decode);
                                if constexpr (kTrace)
                                    trace_math(42, 0);   // FIRST_STAGE_READY
                                uint32_t round_phase = phase;
                                #pragma unroll 1
                                for (uint32_t r = 0; r < kNumRounds; ++ r) {
                                    const bool more = r + 1 < kNumRounds;
                                    [&]<uint32_t... kS>(std::integer_sequence<uint32_t, kS...>) CUTLASS_LAMBDA_FUNC_INLINE {
                                        ((kS % 2 == 0 ? step_s.template operator()<kS>(frag0, frag1, round_phase, kS + 1 < kNumStages or more)
                                                      : step_s.template operator()<kS>(frag1, frag0, round_phase, kS + 1 < kNumStages or more)), ...);
                                    }(std::make_integer_sequence<uint32_t, kNumStages>{});
                                    round_phase ^= 1u;
                                }
                                // the ring is back at stage 0 (stage_idx unchanged); kNumRounds phase flips
                                phase = round_phase;
                            } else {
                            if (num_k_blocks != 0) {
                                full_barriers[stage_idx]->wait(phase);
                                prof_mark(prof_wait);
                                decode_mxfp4_stage_regs(stage_idx, frag0);
                                prof_mark(prof_decode);
                                if constexpr (kTrace)
                                    trace_math(42, 0);   // FIRST_STAGE_READY
                            }
                            if constexpr (kRSDoubleBuffer) {
                                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ) {
                                    step(frag0, frag1, k_block_idx);
                                    step(frag1, frag0, k_block_idx);
                                }
                            } else {
                                // single set: decode k-block k + 1 after the WGMMA of k completed (`step` with nxt = cur would overwrite
                                // the fragments the in-flight WGMMA reads)
                                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                                    if (k_block_idx != 0) {
                                        prof_mark(prof_rest);
                                        full_barriers[stage_idx]->wait(phase);
                                        prof_mark(prof_wait);
                                        decode_mxfp4_stage_regs(stage_idx, frag0);
                                        prof_mark(prof_decode);
                                    }
                                    if (per64)
                                        issue_rs.template operator()<0, 2>(frag0, stage_idx);
                                    else
                                        issue_rs.template operator()<0, 4>(frag0, stage_idx);
                                    ptx::warpgroup_wait<0>();
                                    prof_mark(prof_sub[0]);
                                    const float* sfa = smem_sfa[stage_idx];
                                    promote(sfa);
                                    if (per64) {
                                        issue_rs.template operator()<2, 2>(frag0, stage_idx);
                                        ptx::warpgroup_wait<0>();
                                        promote(sfa + BLOCK_M);
                                    }
                                    fence_frag(frag0);
                                    release_empty_stage(stage_idx);
                                    prof_mark(prof_sub[1]);
                                }
                            }
                            }   // kStaticStages
                        } else {
                        if (num_k_blocks != 0) {
                            full_barriers[stage_idx]->wait(phase);
                            prof_mark(prof_wait);
                            decode_mxfp4_stage(stage_idx, mxfp4_buf);
                            cutlass::arch::fence_view_async_shared();
                            wg_bar();
                            prof_mark(prof_decode);
                        }
                        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                            if constexpr (kTrace) {
                                if (k_block_idx == 0)
                                    trace_math(42, 0);   // FIRST_STAGE_READY
                            }
                            const uint32_t buf = mxfp4_buf;
                            issue(stage_idx, buf, 0, per64 ? (BLOCK_K / 2) / SwapWGMMA::K : BLOCK_K / SwapWGMMA::K);
                            if (k_block_idx + 1 < num_k_blocks) {
                                const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                                const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                                prof_mark(prof_rest);
                                full_barriers[next_stage]->wait(next_phase);
                                prof_mark(prof_wait);
                                decode_mxfp4_stage(next_stage, buf ^ 1u);
                                cutlass::arch::fence_view_async_shared();
                                prof_mark(prof_decode);
                            }
                            ptx::warpgroup_wait<0>();
                            prof_mark(prof_sub[0]);
                            const float* sfa = smem_sfa[stage_idx];
                            if (per64) {
                                promote(sfa);
                                issue(stage_idx, buf, (BLOCK_K / 2) / SwapWGMMA::K, (BLOCK_K / 2) / SwapWGMMA::K);
                                ptx::warpgroup_wait<0>();
                                promote(sfa + BLOCK_M);
                                release_empty_stage(stage_idx);
                            } else {
                                promote(sfa);
                                release_empty_stage(stage_idx);
                            }
                            prof_mark(prof_sub[1]);
                            wg_bar();
                            mxfp4_buf ^= 1u;
                            prof_mark(prof_sub[2]);
                        }
                        }
                        if constexpr (kMXFP4DirectSwapEpi and N <= kMXFP4SwapMaxBucket) {
                            #pragma unroll
                            for (uint32_t h = 0; h < 2; ++ h) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    final_accum[h * (kMXFP4SwapMaxBucket / 2) + i] = fin[h][i];
                            }
                        } else {
                        // transpose: weight row n = h * 64 + (warp * 16 + lane / 4) (+ 8), token t = i * 8 + col * 2 (+ 1). The RS loop
                        // never touched the expanded tiles; after the SS loop every warp passed its trailing wg_bar after its last wait<0>
                        const uint32_t n0 = warp_idx_in_wg * 16 + row_idx;
                        #pragma unroll
                        for (uint32_t h = 0; h < 2; ++ h) {
                            auto stg = reinterpret_cast<float*>(smem_expanded_b + h * SMEM_EXPANDED_B_TILE_SIZE + smem_b_wg_offset);
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t t0 = i * 8 + col_idx * 2;
                                ptx::st_shared(stg + t0 * 64 + n0, fin[h][i * 4 + 0]);
                                ptx::st_shared(stg + (t0 + 1) * 64 + n0, fin[h][i * 4 + 1]);
                                ptx::st_shared(stg + t0 * 64 + n0 + 8, fin[h][i * 4 + 2]);
                                ptx::st_shared(stg + (t0 + 1) * 64 + n0 + 8, fin[h][i * 4 + 3]);
                            }
                        }
                        wg_bar();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const uint32_t c = i * 8 + col_idx * 2;
                            const auto stg = reinterpret_cast<const float*>(
                                smem_expanded_b + (c / 64) * SMEM_EXPANDED_B_TILE_SIZE + smem_b_wg_offset);
                            const uint32_t cc = c % 64;
                            const float2 v0 = r_0 < N ? ptx::ld_shared(reinterpret_cast<const float2*>(stg + r_0 * 64 + cc)) : make_float2(0.0f, 0.0f);
                            const float2 v1 = r_1 < N ? ptx::ld_shared(reinterpret_cast<const float2*>(stg + r_1 * 64 + cc)) : make_float2(0.0f, 0.0f);
                            final_accum[i * 4 + 0] = v0.x, final_accum[i * 4 + 1] = v0.y;
                            final_accum[i * 4 + 2] = v1.x, final_accum[i * 4 + 3] = v1.y;
                        }
                        // N48: the staging spans both expanded tiles of this warpgroup; every warp has read it back before the next
                        // tile's first decode overwrites them (that decode precedes the next tile's first warpgroup barrier)
                        if constexpr (N > kMXFP4SwapMaxBucket)
                            wg_bar();
                        }
                    };
                    // valid_m > kMXFP4SwapMaxBucket: the caller runs the regular mainloop for this tile (an N64 swap issues the
                    // same WGMMA work as the 64 x 128 regular tile, plus the transpose, and its 2 x 2 x 32 fp32 live set spills)
                    if (valid_m <= 8) {
                        run.template operator()<8>();
                    } else if (valid_m <= 16) {
                        run.template operator()<16>();
                    } else {
                        // mode 5: an N24 bucket (register-A, tensor-bound region: 782 vs 1032 cyc/k-block for N32 in the microbench)
                        if constexpr (kMXFP4DecodeMode == 5) {
                            if (valid_m <= 24)
                                run.template operator()<24>();
                            else
                                run.template operator()<kMXFP4SwapMaxBucket>();
                        } else if constexpr (kMXFP4DecodeMode >= 6) {
                            // mode 6: N24 as mode 5, plus the N48 bucket for 33..48 valid rows (0.75 x the regular tile's tensor work);
                            // mode 7 adds N40 for 33..40 rows (0.63 x, microbench notes §36.3)
                            if (valid_m <= 24)
                                run.template operator()<24>();
                            else if (valid_m <= kMXFP4SwapMaxBucket)
                                run.template operator()<kMXFP4SwapMaxBucket>();
                            else if constexpr (kMXFP4DecodeMode == 7) {
                                if (valid_m <= 40)
                                    run.template operator()<40>();
                                else
                                    run.template operator()<48>();
                            } else {
                                run.template operator()<48>();
                            }
                        } else {
                            run.template operator()<kMXFP4SwapMaxBucket>();
                        }
                    }
                }
            };

            // MXFP4 split-M mainloop (kMXFP4SplitM): the 128 x 256 tile, warpgroup g multiplies its 64 A rows with all 256 rows of the
            // shared expanded tile and decodes rows [128g, 128g + 128) of it. Per k-block, both warpgroups:
            //   rescale final_accum by prev / this k-block's activation SF -> WGMMA on buf -> full wait of k + 1 -> decode k + 1 into
            //   buf ^ 1 -> fence -> wait -> release -> CTA math barrier (RAW of buf ^ 1, WAR of buf, across both warpgroups) -> buf ^= 1
            // The weight scale is one value per expert, so the chain carries only the activation SF (the FP8 split-M chain also
            // carries the per-k-block weight SF) and the last activation SF and the weight scale are applied after the loop
            // forced inline: with the register-A kMode 2 loop NVVM outlined it (a .func per phase, final_accum in local memory: STACK 1168)
            const auto mxfp4_splitm_mainloop = [&](float* final_accum) CUTLASS_LAMBDA_FUNC_INLINE {
                if constexpr (kMXFP4SplitM) {
                    DG_STATIC_ASSERT(kNumStages >= 2, "MXFP4 decode lookahead holds one stage while waiting for the next");
                    const uint32_t peer_wg_idx = epilogue_wg_idx ^ 1u;
                    // see kMXFP4SplitMReadyBarrierIdx: arrive on this warpgroup's barrier, sync on the peer's (256 threads each)
                    const auto ready_arrive = [&](const uint32_t& b) { mxfp4_splitm_bar_arrive(kMXFP4SplitMReadyBarrierIdx + epilogue_wg_idx * 2 + b); };
                    const auto ready_wait   = [&](const uint32_t& b) { ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMReadyBarrierIdx + peer_wg_idx * 2 + b); };
                    const auto free_arrive  = [&](const uint32_t& b) { mxfp4_splitm_bar_arrive(kMXFP4SplitMFreeBarrierIdx + epilogue_wg_idx * 2 + b); };
                    const auto free_wait    = [&](const uint32_t& b) { ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMFreeBarrierIdx + peer_wg_idx * 2 + b); };
                    const auto desc_b_buf0 = mma::sm90::make_smem_desc(smem_expanded_b, 0, 128, 1024);
                    // warpgroup-wide OR of one predicate per thread, returned to every thread (the warpgroup's named barrier)
                    const auto mxfp4_wg_vote_or = [&](const bool& pred) {
                        uint32_t r;
                        asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, %1, 0;\n\tbarrier.cta.red.or.pred q, %2, 128, p;\n\tselp.u32 %0, 1, 0, q;\n\t}"
                                     : "=r"(r) : "r"(pred ? 1u : 0u), "r"(kEpilogueWGBarrierStartIdx + epilogue_wg_idx) : "memory");
                        return r != 0u;
                    };
                    uint32_t prof_t = 0, prof_wait = 0, prof_decode = 0, prof_rest = 0;
                    const auto prof_mark = [&](uint32_t& acc) {
                        if constexpr (kTrace) {
                            const uint32_t now = sm90_fp8_fused_mega_moe_prof_clock();
                            acc += now - prof_t;
                            prof_t = now;
                        }
                    };
                    if constexpr (kTrace)
                        prof_t = sm90_fp8_fused_mega_moe_prof_clock();
                    // decode k-block `stage` into tile b once the peer's WGMMA on b (two k-blocks back) completed; every own warp passed
                    // its wait<0> of that WGMMA before this sync, so the sync also orders the own warpgroup's WAR
                    const auto decode_into = [&](const uint32_t& stage, const uint32_t& b) {
                        free_wait(b);
                        decode_mxfp4_stage(stage, b);
                        cutlass::arch::fence_view_async_shared();
                        ready_arrive(b);
                    };
                    // kTrace only: FIRST_STAGE_READY aux 1 = the first k-block was decoded under the previous tile's last WGMMA
                    const bool trace_pre_decoded = mxfp4_pre_decoded;
                    if (num_k_blocks != 0) {
                        if (not mxfp4_pre_decoded) {
                            mxfp4_flush_pub();
                            full_barriers[stage_idx]->wait(phase);
                            prof_mark(prof_wait);
                            decode_into(stage_idx, mxfp4_buf);
                            prof_mark(prof_decode);
                        }
                        mxfp4_pre_decoded = false;
                    }
                    // kMode 1: the regular 64 x 256 WGMMA of this warpgroup's rows. kMode 0: this warpgroup has no valid row (the partial
                    // last m-block of an expert, warpgroup 1 only); it decodes its half and keeps the barrier protocol, but issues no WGMMA,
                    // so the peer's WGMMA has the tensor cores alone. kMode 2 (swap tail, both warpgroups, valid_m <= kMXFP4SplitMSwapN):
                    // each warpgroup multiplies the 128 weight rows it decodes (A, 2 x M64) with the tile's first 16 / 24 / 32 / 40 token
                    // rows (B, from the A stage; 40 for 33 .. 40 valid rows) and reads nothing of the peer's half (k-block 0 from the expanded tile, the rest with
                    // register-A fragments, see the kMode 2 loop); afterwards warpgroup 0 gathers both halves into its
                    // regular accumulator layout and warpgroup 1 skips the epilogue. kMode 4 (dual swap tail, both warpgroups, WG_BLOCK_M <
                    // valid_m <= kMXFP4SplitMSwapDualN): the same against the first kMXFP4SplitMSwapDualN token rows, and both warpgroups
                    // gather their rows, so both run the regular epilogue. The swap accumulators run the regular chain element by element
                    // (rescale by prev / this activation SF, WGMMA accumulate, the same final scale), so a token's result does not depend on
                    // whether it landed in the tail block: the pool order of an expert's tokens differs between launches, and a per-k-block
                    // fp32 promote (the split-N swap tile's) broke the bitwise repeat. Per k-block (Flash / Pro trace): the full tile 2.26 us,
                    // warpgroup 0 alone on 64 rows 1.26 us; kMode 2 issues a quarter of the full tile's work. For 64 < valid_m <= 96 the
                    // former kMode 3 (warpgroup 0 regular, warpgroup 1 N32 over all 256 weight rows) took 0.85 of the full tile for 0.75 of
                    // its work: N32 costs ~1.45x per FLOP of N256; kMode 4 issues the same work as N96 (notes pr440-mxfp4-port.md, Phase 24b).
                    // Separate compile-time loops: guarding the WGMMA with the runtime condition inside one loop made ptxas serialize every WGMMA of it (C7511,
                    // Flash M2048 / M8192 +8.0 / +9.6 % vs the unconditional loop)
                    const auto run = [&]<uint32_t kMode, uint32_t kSwapN>() CUTLASS_LAMBDA_FUNC_INLINE {
                        constexpr bool kActive = kMode == 1;
                        constexpr bool kSwap = kMode == 2;
                        constexpr bool kSwapDual = kMode == 4;
                        constexpr bool kSwapAny = kSwap or kSwapDual;
                        constexpr uint32_t kSwapHalves = 2;
                        // kMode 2: the token bucket kSwapN (16 / 24 / kMXFP4SplitMSwapN); the other modes pass kMXFP4SplitMSwapN
                        using SwapWGMMA = typename mma::sm90::FP8MMASelector<kSwapDual ? kMXFP4SplitMSwapDualN : kSwapN>::type;
                        constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                        float prev_a0 = 1.0f, prev_a1 = 1.0f;
                        // swap tails: [M64 half h][accumulator] (own weight row, token), and the previous activation SF of this thread's
                        // token columns t = 8 i + 2 col + j
                        float swap_fin[kSwapHalves][kSwapAny ? kSwapAccum : 1], swap_prev[kSwapAny ? kSwapAccum / 2 : 1];
                        if constexpr (kSwapAny) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i) {
                                #pragma unroll
                                for (uint32_t h = 0; h < kSwapHalves; ++ h)
                                    swap_fin[h][i] = 0.0f;
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 2; ++ i)
                                swap_prev[i] = 1.0f;
                        }
                        if constexpr (kSwap) {
                            // kMode 2: k-block 0 runs the SS swap on tile mxfp4_buf (decoded at the tile start or pre-decoded under the
                            // previous tile), so a tile does exactly one decode / read pair on the expanded tiles and the ready / free
                            // protocol is that of a one-k-block tile (mxfp4_buf toggles once). k-blocks 1.. decode the own 128 weight rows
                            // into A fragments (the split-N register-A decode at row mxfp4_decode_n_offset) and run the register-A swap on
                            // the first kSwapN token rows: no expanded-tile write, no async-proxy fence and no barrier with the peer per
                            // k-block. The fragments of k-block k + 1 are decoded under the WGMMA of k (two register sets); the last
                            // k-block tests the next tile's first stage as the SS loop does and pre-decodes it into tile mxfp4_buf
                            using RSWGMMA = typename mma::sm90::FP8MMARSSelector<kSwapN>::type;
                            DG_STATIC_ASSERT(RSWGMMA::kNumAccum == kSwapAccum and BLOCK_K / RSWGMMA::K == 4 and BLOCK_K / kMXFP4GranK == 4,
                                             "split-M register-A swap: the SS swap's accumulators, 4 K32 groups per k-block");
                            constexpr uint32_t kNumKBlocks = std::remove_cvref_t<decltype(block_phase)>::value == sched::BlockPhase::Linear1 ?
                                kNumL1BlockKs : kNumL2BlockKs;
                            DG_STATIC_ASSERT(kNumKBlocks >= 3, "split-M register-A swap: k-block 0, a decode-ahead k-block, the last");
                            const uint32_t lane_row = mxfp4_decode_n_offset + warp_idx_in_wg * 16 + lane_idx / 4;
                            const uint32_t word_lane = smem_b_stage0_addr + lane_row * 16u + (lane_idx % 4) * 4u;
                            const uint32_t scale_lane = smem_sfb_stage0_addr + lane_row * 4u;
                            const auto decode_regs = [&](const uint32_t& stage, uint32_t (&frag)[2][4][4]) CUTLASS_LAMBDA_FUNC_INLINE {
                                const uint32_t word_addr = word_lane + stage * SMEM_B_SIZE_PER_STAGE;
                                const uint32_t scale_addr = scale_lane + stage * SMEM_SFB_SIZE_PER_STAGE;
                                // weight rows lane_row + 64 h + 8 s
                                sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 0, 0>(word_addr, scale_addr, frag[0]);
                                sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 8, 1>(word_addr, scale_addr, frag[0]);
                                sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 64, 0>(word_addr, scale_addr, frag[1]);
                                sm90_mxfp4_decode_rs_row<BLOCK_N * 16, 72, 1>(word_addr, scale_addr, frag[1]);
                            };
                            const auto fence_frag = [](uint32_t (&frag)[2][4][4]) CUTLASS_LAMBDA_FUNC_INLINE {
                                #pragma unroll
                                for (uint32_t h = 0; h < 2; ++ h)
                                    #pragma unroll
                                    for (uint32_t g = 0; g < 4; ++ g)
                                        #pragma unroll
                                        for (uint32_t i = 0; i < 4; ++ i)
                                            asm volatile("" : "+r"(frag[h][g][i]) :: "memory");
                            };
                            const auto fence_fin = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
                                #pragma unroll
                                for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i) ptx::warpgroup_fence_operand(swap_fin[h][i]);
                                }
                            };
                            // this k-block's activation SF of the thread's token columns t = 8 i + 2 col + j, and the regular rescale
                            // (the regular path's r0 / r1 expression) by prev / this SF
                            float swap_a[kSwapAccum / 2];
                            const auto load_sf = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    #pragma unroll
                                    for (uint32_t j = 0; j < 2; ++ j)
                                        swap_a[i * 2 + j] = ptx::ld_shared(smem_sfa[stage_idx] + i * 8 + col_idx * 2 + j);
                                }
                            };
                            const auto rescale = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const float r0 = swap_prev[i * 2 + 0] * (kFastMath ? math::fast_rcp(swap_a[i * 2 + 0]) : 1.0f / swap_a[i * 2 + 0]);
                                    const float r1 = swap_prev[i * 2 + 1] * (kFastMath ? math::fast_rcp(swap_a[i * 2 + 1]) : 1.0f / swap_a[i * 2 + 1]);
                                    #pragma unroll
                                    for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                        swap_fin[h][i * 4 + 0] *= r0;
                                        swap_fin[h][i * 4 + 1] *= r1;
                                        swap_fin[h][i * 4 + 2] *= r0;
                                        swap_fin[h][i * 4 + 3] *= r1;
                                    }
                                }
                            };
                            const auto issue_rs = [&](uint32_t (&frag)[2][4][4]) CUTLASS_LAMBDA_FUNC_INLINE {
                                // the stage index is warp-uniform but loop-carried in a vector register: broadcast, so the token descriptors
                                // sit in the uniform datapath (the split-N register-A loop's finding)
                                const auto desc_stage = smem_desc_a(__shfl_sync(0xffffffffu, stage_idx, 0), 0);
                                fence_fin();
                                fence_frag(frag);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t k = 0; k < BLOCK_K / RSWGMMA::K; ++ k)
                                        RSWGMMA::wgmma(frag[h][k], mma::sm90::advance_smem_desc(desc_stage, k * RSWGMMA::K).desc_, swap_fin[h], true);
                                }
                                ptx::warpgroup_commit_batch();
                                fence_fin();
                            };
                            uint32_t k_block_idx = 0;
                            // the WGMMA of the k-block completed: its SF becomes prev, the stage goes back to the producer
                            // buckets past kMXFP4SplitMSwapN (N40) read the SF again here instead of holding swap_a across the decode-ahead
                            // (the kMode 4 choice; holding it spilled the wider buckets)
                            const auto retire = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    #pragma unroll
                                    for (uint32_t j = 0; j < 2; ++ j)
                                        swap_prev[i * 2 + j] = kSwapN > kMXFP4SplitMSwapN ?
                                            ptx::ld_shared(smem_sfa[stage_idx] + i * 8 + col_idx * 2 + j) : swap_a[i * 2 + j];
                                }
                                release_empty_stage(stage_idx);
                            };
                            uint32_t frag0[2][4][4], frag1[2][4][4];
                            {
                                // k-block 0: SS on tile mxfp4_buf, the fragments of k-block 1 into frag0 under it
                                if constexpr (kTrace)
                                    trace_math(42, trace_pre_decoded ? 1u : 0u);   // FIRST_STAGE_READY (aux 1: pre-decoded)
                                const uint32_t buf = mxfp4_buf;
                                load_sf();
                                ready_wait(buf);
                                fence_fin();
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                        const uint32_t k_off = k * SwapWGMMA::K;
                                        const auto desc_w = mma::sm90::advance_smem_desc(desc_b_buf0, buf * SMEM_EXPANDED_B_TILE_SIZE +
                                            (mxfp4_decode_n_offset + h * 64) * BLOCK_K + k_off * kMXFP4ExpandedKStep);
                                        SwapWGMMA::wgmma(desc_w, smem_desc_a(stage_idx, k_off), swap_fin[h], true);
                                    }
                                }
                                ptx::warpgroup_commit_batch();
                                fence_fin();
                                if constexpr (kMXFP4SplitMDeferPublish)
                                    mxfp4_flush_pub();
                                const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                                const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                                prof_mark(prof_rest);
                                full_barriers[next_stage]->wait(next_phase);
                                prof_mark(prof_wait);
                                decode_regs(next_stage, frag0);
                                prof_mark(prof_decode);
                                ptx::warpgroup_wait<0>();
                                retire();
                                free_arrive(buf);
                                mxfp4_buf ^= 1u;
                                advance_pipeline(k_block_idx);
                                prof_mark(prof_rest);
                            }
                            // a register-A k-block that decodes the next one into `nxt` under its WGMMA
                            const auto rs_step = [&](uint32_t (&cur)[2][4][4], uint32_t (&nxt)[2][4][4]) CUTLASS_LAMBDA_FUNC_INLINE {
                                load_sf();
                                rescale();
                                issue_rs(cur);
                                const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                                const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                                prof_mark(prof_rest);
                                full_barriers[next_stage]->wait(next_phase);
                                prof_mark(prof_wait);
                                decode_regs(next_stage, nxt);
                                prof_mark(prof_decode);
                                ptx::warpgroup_wait<0>();
                                // the fragments stay allocated until the WGMMA reading them completed
                                fence_frag(cur);
                                retire();
                                advance_pipeline(k_block_idx);
                                prof_mark(prof_rest);
                            };
                            // the last k-block: the next stage of the ring is the next tile's first k-block, decoded into tile mxfp4_buf
                            // under this WGMMA only when it has landed (see the SS loop below for why this must not block)
                            const auto rs_last = [&](uint32_t (&cur)[2][4][4]) CUTLASS_LAMBDA_FUNC_INLINE {
                                load_sf();
                                rescale();
                                issue_rs(cur);
                                const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                                const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                                const bool next_ready = mxfp4_wg_vote_or(warp_idx_in_wg == 0 and lane_idx == 0 and
                                                                         full_barriers[next_stage]->test_wait(next_phase));
                                if (next_ready) {
                                    prof_mark(prof_rest);
                                    full_barriers[next_stage]->wait(next_phase);
                                    decode_into(next_stage, mxfp4_buf);
                                    prof_mark(prof_decode);
                                    mxfp4_pre_decoded = true;
                                }
                                ptx::warpgroup_wait<0>();
                                fence_frag(cur);
                                retire();
                                advance_pipeline(k_block_idx);
                                prof_mark(prof_rest);
                            };
                            // k-blocks 1 .. K - 2 decode ahead (frag0 holds k-block 1), K - 1 is the last
                            constexpr uint32_t kNumAheadSteps = kNumKBlocks - 2;
                            #pragma unroll 1
                            for (uint32_t j = 0; j < kNumAheadSteps / 2; ++ j) {
                                rs_step(frag0, frag1);
                                rs_step(frag1, frag0);
                            }
                            if constexpr (kNumAheadSteps % 2 == 1) {
                                rs_step(frag0, frag1);
                                rs_last(frag1);
                            } else {
                                rs_last(frag0);
                            }
                        } else {
                        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                            if constexpr (kTrace) {
                                if (k_block_idx == 0)
                                    trace_math(42, trace_pre_decoded ? 1u : 0u);   // FIRST_STAGE_READY (aux 1: pre-decoded)
                            }
                            const uint32_t buf = mxfp4_buf;
                            float a0 = 1.0f, a1 = 1.0f;
                            // dual swap tail: the regular rescale, per token column (the regular path's r0 / r1 expression). The SF is read
                            // here for the ratio and again there (swap_prev): its 24 held across the decode spilled (STACK 8)
                            const auto swap_sf = [&](const uint32_t& i, const uint32_t& j) {
                                return ptx::ld_shared(smem_sfa[stage_idx] + i * 8 + col_idx * 2 + j);
                            };
                            if constexpr (kSwapAny) {
                                if (k_block_idx != 0) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        const float a0s = swap_sf(i, 0);
                                        const float a1s = swap_sf(i, 1);
                                        const float r0 = swap_prev[i * 2 + 0] * (kFastMath ? math::fast_rcp(a0s) : 1.0f / a0s);
                                        const float r1 = swap_prev[i * 2 + 1] * (kFastMath ? math::fast_rcp(a1s) : 1.0f / a1s);
                                        #pragma unroll
                                        for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                            swap_fin[h][i * 4 + 0] *= r0;
                                            swap_fin[h][i * 4 + 1] *= r1;
                                            swap_fin[h][i * 4 + 2] *= r0;
                                            swap_fin[h][i * 4 + 3] *= r1;
                                        }
                                    }
                                }
                            }
                            if constexpr (kActive) {
                                a0 = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                                a1 = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                                if (k_block_idx != 0) {
                                    const float r0 = prev_a0 * (kFastMath ? math::fast_rcp(a0) : 1.0f / a0);
                                    const float r1 = prev_a1 * (kFastMath ? math::fast_rcp(a1) : 1.0f / a1);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                        final_accum[i*4+0] *= r0;
                                        final_accum[i*4+1] *= r0;
                                        final_accum[i*4+2] *= r1;
                                        final_accum[i*4+3] *= r1;
                                    }
                                }
                            }
                            // the peer's half of tile buf is decoded (and, through the own threads' arrival, the own half)
                            ready_wait(buf);
                            if constexpr (kActive) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                    const uint32_t k_off = k * WGMMA::K;
                                    auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k_off);
                                    auto desc_b = mma::sm90::advance_smem_desc(desc_b_buf0, buf * SMEM_EXPANDED_B_TILE_SIZE + k_off * kMXFP4ExpandedKStep);
                                    WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            } else if constexpr (kSwapAny) {
                                #pragma unroll
                                for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i) ptx::warpgroup_fence_operand(swap_fin[h][i]);
                                }
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                        const uint32_t k_off = k * SwapWGMMA::K;
                                        // the own weight rows mxfp4_decode_n_offset + 64 h .. + 64 of tile buf (8-row groups of BLOCK_K bytes),
                                        // the first N token rows of the stage
                                        const auto desc_w = mma::sm90::advance_smem_desc(desc_b_buf0, buf * SMEM_EXPANDED_B_TILE_SIZE +
                                            (mxfp4_decode_n_offset + h * 64) * BLOCK_K + k_off * kMXFP4ExpandedKStep);
                                        SwapWGMMA::wgmma(desc_w, smem_desc_a(stage_idx, k_off), swap_fin[h], true);
                                    }
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i) ptx::warpgroup_fence_operand(swap_fin[h][i]);
                                }
                            }
                            // deferred L1 release of the previous tile, under this WGMMA (no-op when flushed before the first wait)
                            if constexpr (kMXFP4SplitMDeferPublish) {
                                if (k_block_idx == 0)
                                    mxfp4_flush_pub();
                            }
                            const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                            const uint32_t next_phase = phase ^ (next_stage == 0 ? 1u : 0u);
                            if (k_block_idx + 1 < num_k_blocks) {
                                prof_mark(prof_rest);
                                full_barriers[next_stage]->wait(next_phase);
                                prof_mark(prof_wait);
                                decode_into(next_stage, buf ^ 1u);
                                prof_mark(prof_decode);
                            } else {
                                // the next stage of the ring is the next tile's first k-block: when it has landed, decode it under this
                                // WGMMA. Only a test: that tile may be an L2 tile waiting on this tile's L1 output, so blocking here could
                                // deadlock. A stage nobody loads (end of the schedule) never completes the phase. The decision must be
                                // warpgroup-uniform (a warp that pre-decodes blocks in free_wait until its sibling warps decode the same
                                // tile, which a per-warp decision defers past the epilogue's warpgroup barrier: deadlock), so warp 0 lane 0
                                // tests and the warpgroup barrier ORs its vote (barrier.red, no shared memory); every thread then takes
                                // its own acquire of the completed phase before reading the stage
                                const bool next_ready = mxfp4_wg_vote_or(warp_idx_in_wg == 0 and lane_idx == 0 and
                                                                         full_barriers[next_stage]->test_wait(next_phase));
                                if (next_ready) {
                                    prof_mark(prof_rest);
                                    full_barriers[next_stage]->wait(next_phase);
                                    decode_into(next_stage, buf ^ 1u);
                                    prof_mark(prof_decode);
                                    mxfp4_pre_decoded = true;
                                }
                            }
                            if constexpr (kActive or kSwapAny)
                                ptx::warpgroup_wait<0>();
                            if constexpr (kSwapAny) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i)
                                    swap_prev[i * 2 + 0] = swap_sf(i, 0), swap_prev[i * 2 + 1] = swap_sf(i, 1);
                            }
                            release_empty_stage(stage_idx);
                            free_arrive(buf);
                            mxfp4_buf ^= 1u;
                            prev_a0 = a0, prev_a1 = a1;
                            prof_mark(prof_rest);
                        }
                        }
                        if constexpr (kActive) {
                            if (num_k_blocks != 0) {
                                // the activation side carries the x64 when the secondary could not (mxfp4_weight_scale holds the raw
                                // secondary on this path, see its load)
                                const bool compensate_secondary = mxfp4_weight_scale <= kMaxSecondaryBeforeX64;
                                const float weight_scale = compensate_secondary ? mxfp4_weight_scale * 64.0f : mxfp4_weight_scale;
                                const float act_scale_mul = compensate_secondary ? 1.0f : 64.0f;
                                const float s0 = (prev_a0 * act_scale_mul) * weight_scale;
                                const float s1 = (prev_a1 * act_scale_mul) * weight_scale;
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                    final_accum[i*4+0] *= s0;
                                    final_accum[i*4+1] *= s0;
                                    final_accum[i*4+2] *= s1;
                                    final_accum[i*4+3] *= s1;
                                }
                            }
                        }
                        if constexpr (kSwapAny) {
                            // the regular final scale (s0 / s1 of the regular path), per token column; accumulator (weight row r, token t):
                            // [4 i] (r, t0) [4 i + 1] (r, t0 + 1) [4 i + 2] (r + 8, t0) [4 i + 3] (r + 8, t0 + 1)
                            if (num_k_blocks != 0) {
                                const bool compensate_secondary = mxfp4_weight_scale <= kMaxSecondaryBeforeX64;
                                const float weight_scale = compensate_secondary ? mxfp4_weight_scale * 64.0f : mxfp4_weight_scale;
                                const float act_scale_mul = compensate_secondary ? 1.0f : 64.0f;
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const float s0 = (swap_prev[i * 2 + 0] * act_scale_mul) * weight_scale;
                                    const float s1 = (swap_prev[i * 2 + 1] * act_scale_mul) * weight_scale;
                                    #pragma unroll
                                    for (uint32_t h = 0; h < kSwapHalves; ++ h) {
                                        swap_fin[h][i * 4 + 0] *= s0;
                                        swap_fin[h][i * 4 + 1] *= s1;
                                        swap_fin[h][i * 4 + 2] *= s0;
                                        swap_fin[h][i * 4 + 3] *= s1;
                                    }
                                }
                            }
                            // the swap results -> the regular accumulator layout (token r_0 / r_1, weight row c) through shared memory: fp32
                            // [token][64 weight rows] per M64 half, the 64-row index XOR-swizzled with the token (bank-conflict-free LDS.64).
                            // Each warpgroup stages into its own half of the tile of the last k-block (mxfp4_buf ^ 1), which only its own WGMMA
                            // read in this tile (completed: wait<0>); the other tile may already hold the next tile's pre-decoded first k-block
                            const auto tile_half = [&](const uint32_t& wg) {
                                return reinterpret_cast<float*>(smem_expanded_b + (mxfp4_buf ^ 1u) * SMEM_EXPANDED_B_TILE_SIZE +
                                                                wg * (BLOCK_N / kNumEpilogueWarpgroups) * BLOCK_K);
                            };
                            const auto swz = [](const uint32_t& t) { return (t & 7u) << 3; };
                            const uint32_t n0 = warp_idx_in_wg * 16 + row_idx;
                            if constexpr (kSwap and kSwapN <= kMXFP4SplitMSwapN) {
                                // tokens 0 .. kSwapN - 1 of both M64 halves fill the own tile half; warpgroup 0 gathers all 256 columns
                                DG_STATIC_ASSERT(kMXFP4SplitMSwapN * 64 * sizeof(float) == 64 * BLOCK_K, "swap staging: one M64 half per 64 weight rows");
                                #pragma unroll
                                for (uint32_t h = 0; h < 2; ++ h) {
                                    float* stg = tile_half(epilogue_wg_idx) + h * (kMXFP4SplitMSwapN * 64);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        const uint32_t t0 = i * 8 + col_idx * 2, t1 = t0 + 1;
                                        ptx::st_shared(stg + t0 * 64 + (n0 ^ swz(t0)), swap_fin[h][i * 4 + 0]);
                                        ptx::st_shared(stg + t1 * 64 + (n0 ^ swz(t1)), swap_fin[h][i * 4 + 1]);
                                        ptx::st_shared(stg + t0 * 64 + ((n0 + 8) ^ swz(t0)), swap_fin[h][i * 4 + 2]);
                                        ptx::st_shared(stg + t1 * 64 + ((n0 + 8) ^ swz(t1)), swap_fin[h][i * 4 + 3]);
                                    }
                                }
                                if (epilogue_wg_idx == 0) {
                                    ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMSwapStagedBarrierIdx);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                        // column c = 8 i + 2 col: warpgroup half c / 128, M64 half (c / 64) % 2, row c % 64
                                        const uint32_t c = i * 8 + col_idx * 2;
                                        const float* stg = tile_half(c / (BLOCK_N / kNumEpilogueWarpgroups)) + ((c / 64) % 2) * (kMXFP4SplitMSwapN * 64);
                                        const uint32_t n = c % 64;
                                        float2 v0 = make_float2(0.0f, 0.0f), v1 = make_float2(0.0f, 0.0f);
                                        // the bucket's token rows (warp-uniform: rows 16 w .. 16 w + 7 and 16 w + 8 .. 16 w + 15); the
                                        // other rows of the tile are padding
                                        if (r_0 < kSwapN)
                                            v0 = ptx::ld_shared(reinterpret_cast<const float2*>(stg + r_0 * 64 + (n ^ swz(r_0))));
                                        if (r_1 < kSwapN)
                                            v1 = ptx::ld_shared(reinterpret_cast<const float2*>(stg + r_1 * 64 + (n ^ swz(r_1))));
                                        final_accum[i * 4 + 0] = v0.x, final_accum[i * 4 + 1] = v0.y;
                                        final_accum[i * 4 + 2] = v1.x, final_accum[i * 4 + 3] = v1.y;
                                    }
                                    mxfp4_splitm_bar_arrive(kMXFP4SplitMSwapReadBarrierIdx);
                                } else {
                                    mxfp4_splitm_bar_arrive(kMXFP4SplitMSwapStagedBarrierIdx);
                                    ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMSwapReadBarrierIdx);
                                }
                            } else {
                                // kSwapDual: one M64 half per pass, so a half's accumulators die while the receiving columns fill (live
                                // registers stay near the regular accumulator). Tokens 0 .. 63 of a half (16 KiB) fill the own tile half,
                                // tokens 64 .. 95 (8 KiB) the own half of the CD tile: the previous tile's TMA store from it drained at this
                                // tile's first k-block (deferred publish), before the peer's first ready arrive, and this tile's epilogue
                                // writes it only past the last read barrier below
                                // kMode 2 with a bucket wider than kMXFP4SplitMSwapN (N40): the same per-half passes, all tokens in the own
                                // tile half (no CD tile); warpgroup 0 gathers its rows, warpgroup 1 has none
                                constexpr uint32_t kSwapTokens = kSwapDual ? kMXFP4SplitMSwapDualN : kSwapN;
                                constexpr uint32_t kHiTokens = kSwapTokens > WG_BLOCK_M ? kSwapTokens - WG_BLOCK_M : 0u;
                                DG_STATIC_ASSERT(WG_BLOCK_M * 64 * sizeof(float) == (BLOCK_N / kNumEpilogueWarpgroups) * BLOCK_K and
                                                 kNumEpilogueWarpgroups * kHiTokens * 64 * sizeof(float) <= SMEM_CD_SIZE and
                                                 kSwapAccum / 4 * 8 == kSwapTokens and (kSwapDual or kSwapTokens <= WG_BLOCK_M),
                                                 "dual swap staging: 64 tokens per own tile half, the rest in the CD tile");
                                const auto cd_half = [&](const uint32_t& wg) {
                                    return reinterpret_cast<float*>(smem_cd_l1) + wg * (kHiTokens * 64);
                                };
                                // this warpgroup's rows row_base + r_0 / r_1: tokens 0 .. 63 (warpgroup 0, staged in the tile halves) or
                                // 64 .. 127 (warpgroup 1, staged in the CD halves up to kMXFP4SplitMSwapDualN)
                                const bool rows_staged = row_base + r_1 < kMXFP4SplitMSwapDualN;   // warp-uniform (kSwapDual)
                                #pragma unroll
                                for (uint32_t p = 0; p < 2; ++ p) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        const uint32_t t0 = i * 8 + col_idx * 2, t1 = t0 + 1;
                                        // compile-time per unrolled i: t0, t1 < WG_BLOCK_M iff i < WG_BLOCK_M / 8
                                        constexpr uint32_t kLoGroups = WG_BLOCK_M / 8;
                                        float* stg = i < kLoGroups ? tile_half(epilogue_wg_idx) : cd_half(epilogue_wg_idx);
                                        const uint32_t l0 = i < kLoGroups ? t0 : t0 - WG_BLOCK_M, l1 = l0 + 1;
                                        ptx::st_shared(stg + l0 * 64 + (n0 ^ swz(t0)), swap_fin[p][i * 4 + 0]);
                                        ptx::st_shared(stg + l1 * 64 + (n0 ^ swz(t1)), swap_fin[p][i * 4 + 1]);
                                        ptx::st_shared(stg + l0 * 64 + ((n0 + 8) ^ swz(t0)), swap_fin[p][i * 4 + 2]);
                                        ptx::st_shared(stg + l1 * 64 + ((n0 + 8) ^ swz(t1)), swap_fin[p][i * 4 + 3]);
                                    }
                                    ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMSwapStagedBarrierIdx);
                                    #pragma unroll
                                    for (uint32_t w = 0; w < kNumEpilogueWarpgroups; ++ w) {
                                        // M64 half p of warpgroup w's weight rows: columns c = 8 i + 2 col in [128 w + 64 p, + 64)
                                        const float* src = row_base < WG_BLOCK_M ? tile_half(w) : cd_half(w);
                                        #pragma unroll
                                        for (uint32_t j = 0; j < 8; ++ j) {
                                            const uint32_t i = w * 16 + p * 8 + j;
                                            const uint32_t n = j * 8 + col_idx * 2;
                                            float2 v0 = make_float2(0.0f, 0.0f), v1 = make_float2(0.0f, 0.0f);
                                            if constexpr (kSwapDual) {
                                                if (rows_staged) {
                                                    v0 = ptx::ld_shared(reinterpret_cast<const float2*>(src + r_0 * 64 + (n ^ swz(r_0))));
                                                    v1 = ptx::ld_shared(reinterpret_cast<const float2*>(src + r_1 * 64 + (n ^ swz(r_1))));
                                                }
                                            } else {
                                                // the bucket's token rows of warpgroup 0 (warp-uniform per 8-row half: the buckets are multiples of 8)
                                                if (row_base + r_0 < kSwapTokens)
                                                    v0 = ptx::ld_shared(reinterpret_cast<const float2*>(src + r_0 * 64 + (n ^ swz(r_0))));
                                                if (row_base + r_1 < kSwapTokens)
                                                    v1 = ptx::ld_shared(reinterpret_cast<const float2*>(src + r_1 * 64 + (n ^ swz(r_1))));
                                            }
                                            final_accum[i * 4 + 0] = v0.x, final_accum[i * 4 + 1] = v0.y;
                                            final_accum[i * 4 + 2] = v1.x, final_accum[i * 4 + 3] = v1.y;
                                        }
                                    }
                                    // the next pass rewrites the staging; after the last one the epilogue writes the CD tile and the next
                                    // tile's decode the tile halves
                                    ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMSwapReadBarrierIdx);
                                }
                            }
                        }
                    };
                    // the swap tails run on both warpgroups (after kMode 2 warpgroup 1 takes the no-valid-row epilogue path)
                    // kMode 2 buckets N16 / N24 / N32 (register-A, the split-N swap's measured per-k-block cost 0.55 / 0.60 / 0.79 us vs
                    // 0.98 us of the SS N32 swap here, Flash / Pro trace)
                    if (valid_m <= 16)
                        run.template operator()<2, 16>();
                    else if (valid_m <= 24)
                        run.template operator()<2, 24>();
                    else if (valid_m <= kMXFP4SplitMSwapN)
                        run.template operator()<2, kMXFP4SplitMSwapN>();
                    // 33 .. 40 rows: both warpgroups N40 instead of warpgroup 0's regular 64 x 256 with warpgroup 1 idle (1.275 us per
                    // k-block). 41 .. 64 stay there: an N48 bucket (double or single fragment set) spilled the cap-8192 / M4096 instances
                    // inside the mainloop (STACK 32..40, Flash / Pro M8192 +0.4..+0.7 %, Flash M320 +0.7..+1.4 %), N56 every instance
                    // (STACK 64..160); notes pr440-mxfp4-port.md, Phase 24e
                    else if (valid_m <= 40)
                        run.template operator()<2, 40>();
                    else if (valid_m > WG_BLOCK_M and valid_m <= kMXFP4SplitMSwapDualN)
                        run.template operator()<4, kMXFP4SplitMSwapN>();
                    else if (row_base < valid_m)
                        run.template operator()<1, kMXFP4SplitMSwapN>();
                    else
                        run.template operator()<0, kMXFP4SplitMSwapN>();
                    if constexpr (kTrace) {
                        const uint32_t ph = block_phase == sched::BlockPhase::Linear1 ? 0u : 1u;
                        prof_total[ph][0] += prof_wait, prof_total[ph][1] += prof_decode, prof_total[ph][2] += prof_rest;
                    }
                }
            };

            float final_accum[kAccumPerThread] = {};

            if constexpr (kReuseAccumAsFinal) {
                if constexpr (kMXFP4SplitM) {
                    mxfp4_splitm_mainloop(final_accum);
                } else {
                auto prescale_l1_final = [&](const float& scale_a_0, const float& scale_a_1,
                                             const float& gate_sf, const float& up_sf) {
                    const float inv_s0_gate = kFastMath ? math::fast_rcp(scale_a_0 * gate_sf) : 1.0f / (scale_a_0 * gate_sf);
                    const float inv_s1_gate = kFastMath ? math::fast_rcp(scale_a_1 * gate_sf) : 1.0f / (scale_a_1 * gate_sf);
                    const float inv_s0_up = kFastMath ? math::fast_rcp(scale_a_0 * up_sf) : 1.0f / (scale_a_0 * up_sf);
                    const float inv_s1_up = kFastMath ? math::fast_rcp(scale_a_1 * up_sf) : 1.0f / (scale_a_1 * up_sf);
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float inv_s0 = (i & 1u) ? inv_s0_up : inv_s0_gate;
                        const float inv_s1 = (i & 1u) ? inv_s1_up : inv_s1_gate;
                        final_accum[i*4+0] *= inv_s0;
                        final_accum[i*4+1] *= inv_s0;
                        final_accum[i*4+2] *= inv_s1;
                        final_accum[i*4+3] *= inv_s1;
                    }
                };
                auto postscale_l1_final = [&](const float& scale_a_0, const float& scale_a_1,
                                              const float& gate_sf, const float& up_sf) {
                    const float s0_gate = scale_a_0 * gate_sf;
                    const float s1_gate = scale_a_1 * gate_sf;
                    const float s0_up = scale_a_0 * up_sf;
                    const float s1_up = scale_a_1 * up_sf;
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float s0 = (i & 1u) ? s0_up : s0_gate;
                        const float s1 = (i & 1u) ? s1_up : s1_gate;
                        final_accum[i*4+0] *= s0;
                        final_accum[i*4+1] *= s0;
                        final_accum[i*4+2] *= s1;
                        final_accum[i*4+3] *= s1;
                    }
                };
                // Chunk i (8 columns) of a >=256-wide warpgroup tile falls in weight-SF block i / 16.
                constexpr uint32_t kChunksPerSFGroup = kAccumPerThread / 4 / kNumSFGroupsPerWG;
                auto prescale_l2_final = [&](const float& scale_a_0, const float& scale_a_1,
                                             const float& l2_sf, const float& l2_sf_hi) {
                    const float inv_s0 = kFastMath ? math::fast_rcp(scale_a_0 * l2_sf) : 1.0f / (scale_a_0 * l2_sf);
                    const float inv_s1 = kFastMath ? math::fast_rcp(scale_a_1 * l2_sf) : 1.0f / (scale_a_1 * l2_sf);
                    const float inv_s0_hi = kFastMath ? math::fast_rcp(scale_a_0 * l2_sf_hi) : 1.0f / (scale_a_0 * l2_sf_hi);
                    const float inv_s1_hi = kFastMath ? math::fast_rcp(scale_a_1 * l2_sf_hi) : 1.0f / (scale_a_1 * l2_sf_hi);
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const bool hi = kNumSFGroupsPerWG > 1 and i >= kChunksPerSFGroup;
                        final_accum[i*4+0] *= hi ? inv_s0_hi : inv_s0;
                        final_accum[i*4+1] *= hi ? inv_s0_hi : inv_s0;
                        final_accum[i*4+2] *= hi ? inv_s1_hi : inv_s1;
                        final_accum[i*4+3] *= hi ? inv_s1_hi : inv_s1;
                    }
                };
                auto postscale_l2_final = [&](const float& scale_a_0, const float& scale_a_1,
                                              const float& l2_sf, const float& l2_sf_hi) {
                    const float s0 = scale_a_0 * l2_sf;
                    const float s1 = scale_a_1 * l2_sf;
                    const float s0_hi = scale_a_0 * l2_sf_hi;
                    const float s1_hi = scale_a_1 * l2_sf_hi;
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const bool hi = kNumSFGroupsPerWG > 1 and i >= kChunksPerSFGroup;
                        final_accum[i*4+0] *= hi ? s0_hi : s0;
                        final_accum[i*4+1] *= hi ? s0_hi : s0;
                        final_accum[i*4+2] *= hi ? s1_hi : s1;
                        final_accum[i*4+3] *= hi ? s1_hi : s1;
                    }
                };
                auto rescale_l1_final = [&](const float& prev_scale_a_0, const float& prev_scale_a_1,
                                            const float& prev_gate_sf, const float& prev_up_sf,
                                            const float& scale_a_0, const float& scale_a_1,
                                            const float& gate_sf, const float& up_sf) {
                    const float r0_gate = (prev_scale_a_0 * prev_gate_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * gate_sf) : 1.0f / (scale_a_0 * gate_sf));
                    const float r1_gate = (prev_scale_a_1 * prev_gate_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * gate_sf) : 1.0f / (scale_a_1 * gate_sf));
                    const float r0_up = (prev_scale_a_0 * prev_up_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * up_sf) : 1.0f / (scale_a_0 * up_sf));
                    const float r1_up = (prev_scale_a_1 * prev_up_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * up_sf) : 1.0f / (scale_a_1 * up_sf));
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float r0 = (i & 1u) ? r0_up : r0_gate;
                        const float r1 = (i & 1u) ? r1_up : r1_gate;
                        final_accum[i*4+0] *= r0;
                        final_accum[i*4+1] *= r0;
                        final_accum[i*4+2] *= r1;
                        final_accum[i*4+3] *= r1;
                    }
                };
                auto rescale_l2_final = [&](const float& prev_scale_a_0, const float& prev_scale_a_1,
                                            const float& prev_l2_sf, const float& prev_l2_sf_hi,
                                            const float& scale_a_0, const float& scale_a_1,
                                            const float& l2_sf, const float& l2_sf_hi) {
                    const float r0 = (prev_scale_a_0 * prev_l2_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * l2_sf) : 1.0f / (scale_a_0 * l2_sf));
                    const float r1 = (prev_scale_a_1 * prev_l2_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * l2_sf) : 1.0f / (scale_a_1 * l2_sf));
                    const float r0_hi = (prev_scale_a_0 * prev_l2_sf_hi) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * l2_sf_hi) : 1.0f / (scale_a_0 * l2_sf_hi));
                    const float r1_hi = (prev_scale_a_1 * prev_l2_sf_hi) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * l2_sf_hi) : 1.0f / (scale_a_1 * l2_sf_hi));
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const bool hi = kNumSFGroupsPerWG > 1 and i >= kChunksPerSFGroup;
                        final_accum[i*4+0] *= hi ? r0_hi : r0;
                        final_accum[i*4+1] *= hi ? r0_hi : r0;
                        final_accum[i*4+2] *= hi ? r1_hi : r1;
                        final_accum[i*4+3] *= hi ? r1_hi : r1;
                    }
                };
                auto rescale_l2_act_final = [&](const float& prev_scale_a_0, const float& prev_scale_a_1,
                                                const float& scale_a_0, const float& scale_a_1) {
                    const float r0 = prev_scale_a_0 * (kFastMath ? math::fast_rcp(scale_a_0) : 1.0f / scale_a_0);
                    const float r1 = prev_scale_a_1 * (kFastMath ? math::fast_rcp(scale_a_1) : 1.0f / scale_a_1);
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= r0;
                        final_accum[i*4+1] *= r0;
                        final_accum[i*4+2] *= r1;
                        final_accum[i*4+3] *= r1;
                    }
                };

                auto load_weight_sf = [&](const uint32_t& k, float& g, float& u, float& l, float& l_hi) {
                    if (block_phase == sched::BlockPhase::Linear1) {
                        g = ptx::ld_shared(smem_weight_sf_wg + k);
                        u = ptx::ld_shared(smem_weight_sf_wg + kL1SFKBlocks + k);
                    } else {
                        l = ptx::ld_shared(smem_weight_sf_wg + k);
                        if constexpr (kNumSFGroupsPerWG > 1)
                            l_hi = ptx::ld_shared(smem_weight_sf_wg + kL2SFKBlocks + k);
                    }
                };
                float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f, l2_sf_hi = 0.0f;
                if (num_k_blocks != 0)
                    load_weight_sf(0, gate_sf, up_sf, l2_sf, l2_sf_hi);

                if constexpr (kHidden >= 4096) {
                    float prev_scale_a_0 = 1.0f, prev_scale_a_1 = 1.0f;
                    float prev_gate_sf = 1.0f, prev_up_sf = 1.0f, prev_l2_sf = 1.0f, prev_l2_sf_hi = 1.0f;
                    // Hoisted k-loop (single-group k-block: L1, and L2 under the per-128 act SF): stage-static wgmma descriptors, an
                    // mbarrier.test_wait on the NEXT k-block's full barrier right after the commit and, when it passed, its activation-SF
                    // loads issued under the drain. The pipeline handshake (full wait per k-block by every thread, empty arrive after the
                    // drain) is unchanged; the per-64 L2 k-block (two groups, act SF hi/lo) keeps the original loop.
                    constexpr bool kHoistedMainloop = kReuseAccumAsFinal and kHidden >= 4096;
                    const bool hoisted = kHoistedMainloop and (block_phase == sched::BlockPhase::Linear1 or kL2ActSFPerBlockK);
                    if constexpr (kHoistedMainloop) {
                    if (hoisted) {
                        const bool is_l1 = block_phase == sched::BlockPhase::Linear1;
                        // per-warpgroup base descriptors (stage 0 + the warpgroup's A / B offset), opaque so that ptxas keeps them; the
                        // stage / k advance below is the same low-word add as advance_smem_desc (no field can overflow: smem addresses < 2^18 B)
                        uint64_t desc_a_wg = mma::sm90::make_smem_desc(smem_a[0] + smem_a_wg_offset, 1).desc_;
                        uint64_t desc_b_wg = mma::sm90::make_smem_desc(smem_b[0] + smem_b_wg_offset, 1).desc_;
                        asm volatile("" : "+l"(desc_a_wg));
                        asm volatile("" : "+l"(desc_b_wg));
                        const auto stage_desc = [](const uint64_t& base, const uint32_t& lo_advance) {
                            cute::GmmaDescriptor d;
                            d.desc_ = base;
                            d.reg32_[0] += lo_advance;
                            return d.desc_;
                        };
                        // the four ratios prev -> cur of the rescale chain, exactly rescale_l1_final / rescale_l2_final's expressions
                        const auto ratio = [](const float& prev_a, const float& prev_w, const float& cur_a, const float& cur_w) {
                            return (prev_a * prev_w) * (kFastMath ? math::fast_rcp(cur_a * cur_w) : 1.0f / (cur_a * cur_w));
                        };
                        const auto ratios = [&](const float& pa0, const float& pa1, const float& pw0, const float& pw1,
                                                const float& a0, const float& a1, const float& w0, const float& w1,
                                                float& r0, float& r1, float& r2, float& r3) {
                            r0 = ratio(pa0, pw0, a0, w0);   // L1: r0_gate, L2: r0
                            r1 = ratio(pa1, pw0, a1, w0);   // L1: r1_gate, L2: r1
                            r2 = ratio(pa0, pw1, a0, w1);   // L1: r0_up,   L2: r0_hi
                            r3 = ratio(pa1, pw1, a1, w1);   // L1: r1_up,   L2: r1_hi
                        };
                        const auto apply_ratios = [&](const float& r0, const float& r1, const float& r2, const float& r3) {
                            if (is_l1) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                    const float ra = (i & 1u) ? r2 : r0;
                                    const float rb = (i & 1u) ? r3 : r1;
                                    final_accum[i*4+0] *= ra;
                                    final_accum[i*4+1] *= ra;
                                    final_accum[i*4+2] *= rb;
                                    final_accum[i*4+3] *= rb;
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                    const bool hi = kNumSFGroupsPerWG > 1 and i >= kChunksPerSFGroup;
                                    final_accum[i*4+0] *= hi ? r2 : r0;
                                    final_accum[i*4+1] *= hi ? r2 : r0;
                                    final_accum[i*4+2] *= hi ? r3 : r1;
                                    final_accum[i*4+3] *= hi ? r3 : r1;
                                }
                            }
                        };
                        const auto ld_act_sf = [&](const uint32_t& stage, float& s0, float& s1) {
                            s0 = ptx::ld_shared(smem_sfa[stage] + row_offset_r0);   // L2 per-128: the lo half at offset 0
                            s1 = ptx::ld_shared(smem_sfa[stage] + row_offset_r1);
                        };
                        // weight SF of this k-block: (w0, w1) = (gate, up) for L1, (l2, l2_hi) for L2
                        float w0 = is_l1 ? gate_sf : l2_sf, w1 = is_l1 ? up_sf : l2_sf_hi;
                        float pw0 = 1.0f, pw1 = 1.0f;                        // previous k-block's weight SF
                        float cur_a0 = 1.0f, cur_a1 = 1.0f;                  // this k-block's activation SF
                        float r0 = 1.0f, r1 = 1.0f, r2 = 1.0f, r3 = 1.0f;    // ratios previous -> this k-block
                        bool ready = false;                                  // this k-block's full barrier known complete
                        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                            if (__builtin_expect(not ready, 0)) {
                                // k == 0, or the early test of this stage did not see the phase complete
                                full_barriers[stage_idx]->wait(phase);
                                ld_act_sf(stage_idx, cur_a0, cur_a1);
                            }
                            if constexpr (kTrace) {
                                if (k_block_idx == 0)
                                    trace_math(42, 0);   // FIRST_STAGE_READY
                            }
                            if (k_block_idx != 0) {
                                ratios(prev_scale_a_0, prev_scale_a_1, pw0, pw1, cur_a0, cur_a1, w0, w1, r0, r1, r2, r3);
                                apply_ratios(r0, r1, r2, r3);
                            }

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                const uint64_t desc_a = stage_desc(desc_a_wg, stage_idx * (SMEM_A_SIZE_PER_STAGE >> 4) + k * (WGMMA::K >> 4));
                                const uint64_t desc_b = stage_desc(desc_b_wg, stage_idx * (SMEM_B_SIZE_PER_STAGE >> 4) + k * (WGMMA::K >> 4));
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();

                            // under the drain: k+1's weight SF, the test of k+1's full barrier and, when it passed, its
                            // activation SF
                            const bool has_next_k = k_block_idx + 1 < num_k_blocks;
                            float next_gate_sf = gate_sf, next_up_sf = up_sf, next_l2_sf = l2_sf, next_l2_sf_hi = l2_sf_hi;
                            if (has_next_k)
                                load_weight_sf(k_block_idx + 1, next_gate_sf, next_up_sf, next_l2_sf, next_l2_sf_hi);
                            const float next_w0 = is_l1 ? next_gate_sf : next_l2_sf, next_w1 = is_l1 ? next_up_sf : next_l2_sf_hi;
                            float next_a0 = cur_a0, next_a1 = cur_a1;
                            bool next_ready = false;
                            const uint32_t next_stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                            const uint32_t next_phase = phase ^ (next_stage_idx == 0 ? 1u : 0u);
                            if (has_next_k) {
                                next_ready = full_barriers[next_stage_idx]->test_wait(next_phase);
                                if (next_ready)
                                    ld_act_sf(next_stage_idx, next_a0, next_a1);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            // roll the chain state (the postscale after the loop reads prev_* of the last k-block)
                            prev_scale_a_0 = cur_a0;
                            prev_scale_a_1 = cur_a1;
                            pw0 = w0, pw1 = w1;
                            if (is_l1) {
                                prev_gate_sf = gate_sf, prev_up_sf = up_sf;
                            } else {
                                prev_l2_sf = l2_sf, prev_l2_sf_hi = l2_sf_hi;
                            }
                            cur_a0 = next_a0, cur_a1 = next_a1;
                            w0 = next_w0, w1 = next_w1;
                            gate_sf = next_gate_sf;
                            up_sf   = next_up_sf;
                            l2_sf   = next_l2_sf;
                            l2_sf_hi = next_l2_sf_hi;
                            ready = next_ready;
                            stage_idx = next_stage_idx;
                            phase = next_phase;
                        }
                    }
                    }
                    if (not hoisted)
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (kTrace) {
                            if (k_block_idx == 0)
                                trace_math(42, 0);   // FIRST_STAGE_READY
                        }

                        float scale_a_0_lo, scale_a_1_lo;
                        float scale_a_0_hi, scale_a_1_hi;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        } else {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                            if constexpr (kL2ActSFPerBlockK) {
                                // one act SF per k-block, the hi half does not exist
                                scale_a_0_hi = scale_a_0_lo, scale_a_1_hi = scale_a_1_lo;
                            } else {
                                scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                                scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                            }
                        }

                        float next_gate_sf = gate_sf, next_up_sf = up_sf, next_l2_sf = l2_sf, next_l2_sf_hi = l2_sf_hi;
                        const bool has_next_k = k_block_idx + 1 < num_k_blocks;

                        if (block_phase == sched::BlockPhase::Linear1) {
                            if (k_block_idx != 0)
                                rescale_l1_final(prev_scale_a_0, prev_scale_a_1,
                                                 prev_gate_sf, prev_up_sf,
                                                 scale_a_0_lo, scale_a_1_lo,
                                                 gate_sf, up_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            // Prefetch k+1's weight SF while the WGMMAs drain.
                            if (has_next_k)
                                load_weight_sf(k_block_idx + 1, next_gate_sf, next_up_sf, next_l2_sf, next_l2_sf_hi);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            prev_scale_a_0 = scale_a_0_lo;
                            prev_scale_a_1 = scale_a_1_lo;
                            prev_gate_sf = gate_sf;
                            prev_up_sf = up_sf;
                        } else {
                            if (k_block_idx != 0)
                                rescale_l2_final(prev_scale_a_0, prev_scale_a_1, prev_l2_sf, prev_l2_sf_hi,
                                                 scale_a_0_lo, scale_a_1_lo, l2_sf, l2_sf_hi);

                            if constexpr (kL2ActSFPerBlockK) {
                                // The per-128 act SF: no per-64 split — one full-BLOCK_K
                                // WGMMA group, one drain, half the fences (the L1 shape).
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                    auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                    auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                    WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                                }
                                ptx::warpgroup_commit_batch();
                                // Prefetch k+1's weight SF while the WGMMAs drain.
                                if (has_next_k)
                                    load_weight_sf(k_block_idx + 1, next_gate_sf, next_up_sf, next_l2_sf, next_l2_sf_hi);
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                                ptx::warpgroup_wait<0>();
                            } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            // Prefetch k+1's weight SF while the WGMMAs drain.
                            if (has_next_k)
                                load_weight_sf(k_block_idx + 1, next_gate_sf, next_up_sf, next_l2_sf, next_l2_sf_hi);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            rescale_l2_act_final(scale_a_0_lo, scale_a_1_lo,
                                                 scale_a_0_hi, scale_a_1_hi);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k_off);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k_off);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();
                            }

                            release_empty_stage(stage_idx);

                            prev_scale_a_0 = scale_a_0_hi;
                            prev_scale_a_1 = scale_a_1_hi;
                            prev_l2_sf = l2_sf;
                            prev_l2_sf_hi = l2_sf_hi;
                        }

                        gate_sf = next_gate_sf;
                        up_sf   = next_up_sf;
                        l2_sf   = next_l2_sf;
                        l2_sf_hi = next_l2_sf_hi;
                    }

                    if (num_k_blocks != 0) {
                        if (block_phase == sched::BlockPhase::Linear1) {
                            postscale_l1_final(prev_scale_a_0, prev_scale_a_1,
                                               prev_gate_sf, prev_up_sf);
                        } else {
                            postscale_l2_final(prev_scale_a_0, prev_scale_a_1, prev_l2_sf, prev_l2_sf_hi);
                        }
                    }
                } else {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);
                        if constexpr (kTrace) {
                            if (k_block_idx == 0)
                                trace_math(42, 0);   // FIRST_STAGE_READY
                        }

                        float scale_a_0_lo, scale_a_1_lo;
                        float scale_a_0_hi, scale_a_1_hi;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        } else {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                            if constexpr (kL2ActSFPerBlockK) {
                                // one act SF per k-block, the hi half does not exist
                                scale_a_0_hi = scale_a_0_lo, scale_a_1_hi = scale_a_1_lo;
                            } else {
                                scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                                scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                            }
                        }

                        // Weight SF for this k was prefetched into registers
                        float next_gate_sf = gate_sf, next_up_sf = up_sf, next_l2_sf = l2_sf, next_l2_sf_hi = l2_sf_hi;
                        const bool has_next_k = k_block_idx + 1 < num_k_blocks;

                        if (block_phase == sched::BlockPhase::Linear1) {
                            if (k_block_idx != 0)
                                prescale_l1_final(scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            // Prefetch k+1's weight SF while the WGMMAs drain.
                            if (has_next_k)
                                load_weight_sf(k_block_idx + 1, next_gate_sf, next_up_sf, next_l2_sf, next_l2_sf_hi);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            postscale_l1_final(scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);
                        } else {
                            if (k_block_idx != 0)
                                prescale_l2_final(scale_a_0_lo, scale_a_1_lo, l2_sf, l2_sf_hi);

                            if constexpr (kL2ActSFPerBlockK) {
                            // one act SF per k-block -> one full-BLOCK_K WGMMA group (the L1 shape)
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            // Prefetch k+1's weight SF while the WGMMAs drain.
                            if (has_next_k)
                                load_weight_sf(k_block_idx + 1, next_gate_sf, next_up_sf, next_l2_sf, next_l2_sf_hi);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            postscale_l2_final(scale_a_0_lo, scale_a_1_lo, l2_sf, l2_sf_hi);
                            } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            // Prefetch k+1's weight SF while the WGMMAs drain.
                            if (has_next_k)
                                load_weight_sf(k_block_idx + 1, next_gate_sf, next_up_sf, next_l2_sf, next_l2_sf_hi);
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            postscale_l2_final(scale_a_0_lo, scale_a_1_lo, l2_sf, l2_sf_hi);
                            prescale_l2_final(scale_a_0_hi, scale_a_1_hi, l2_sf, l2_sf_hi);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k_off);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k_off);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            postscale_l2_final(scale_a_0_hi, scale_a_1_hi, l2_sf, l2_sf_hi);
                            }
                        }

                        gate_sf = next_gate_sf;
                        up_sf   = next_up_sf;
                        l2_sf   = next_l2_sf;
                        l2_sf_hi = next_l2_sf_hi;
                    }
                }
                }
            } else {
                DG_STATIC_ASSERT(kNumSFGroupsPerWG == 1, "The promotion path assumes one weight-SF block per warpgroup tile");
                float accum[kAccumPerThread];
                // kTrace only: per-tile cycle split of the mainloop (clock(), SM-local): full-barrier wait / MXFP4 decode / the rest
                // (WGMMA issue + wait + stage release + promotion), emitted as events 46 / 47 after the loop
                uint32_t prof_wait = 0, prof_decode = 0, prof_rest = 0;

                if constexpr (kMXFP4Weights and kMXFP4SwapAB) {
                    // per-tile hybrid: swapAB while the tile's tokens fit the largest swap bucket, else the regular mainloop (both
                    // keep the same stage / expanded-buffer / barrier protocol, so tiles of either kind may follow each other)
                    if (valid_m <= kMXFP4SwapMaxValidM)
                        mxfp4_swap_mainloop(final_accum, prof_wait, prof_decode, prof_rest);
                    else
                        mxfp4_mainloop(accum, final_accum, prof_wait, prof_decode, prof_rest);
                } else if constexpr (kMXFP4Weights) {
                    mxfp4_mainloop(accum, final_accum, prof_wait, prof_decode, prof_rest);
                } else {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    uint32_t prof_t0 = 0;
                    if constexpr (kTrace)
                        prof_t0 = sm90_fp8_fused_mega_moe_prof_clock();
                    full_barriers[stage_idx]->wait(phase);
                    if constexpr (kTrace) {
                        const uint32_t prof_t1 = sm90_fp8_fused_mega_moe_prof_clock();
                        prof_wait += prof_t1 - prof_t0;
                        prof_t0 = prof_t1;
                    }
                    const auto prof_end_of_k_block = [&]() {
                        if constexpr (kTrace)
                            prof_rest += sm90_fp8_fused_mega_moe_prof_clock() - prof_t0;
                    };
                    if constexpr (kTrace) {
                        if (k_block_idx == 0)
                            trace_math(42, 0);   // FIRST_STAGE_READY
                    }

                    // Read SF (must precede warpgroup_arrive)
                    float scale_a_0_lo, scale_a_1_lo;
                    float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                    if (block_phase == sched::BlockPhase::Linear1) {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    } else {
                        // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                        if constexpr (kL2ActSFPerBlockK) {
                            // one act SF per k-block, the hi half does not exist
                            scale_a_0_hi = scale_a_0_lo, scale_a_1_hi = scale_a_1_lo;
                        } else {
                            scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                            scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                        }
                    }

                    // ----- Block (128, 128) weight SF (staged in SMEM per block) -----
                    // L1 weight SF: (E, 2*IH/128, H/128) MN-major, N axis = [gate(IH/128), up(IH/128)]; with the gate/up gran-8 interleave a
                    // logical 128-wide N tile covers 64 gate + 64 up rows of the same original 128-row block, so gate_sf_n = sf_n_block_idx / 2,
                    // up_sf_n = IH/128 + sf_n_block_idx / 2. L2 weight SF: (E, H/128, IH/128) MN-major, one scalar per logical 128x128 tile.
                    float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                    if (block_phase == sched::BlockPhase::Linear1) {
                        gate_sf = ptx::ld_shared(smem_weight_sf_wg + k_block_idx);
                        up_sf   = ptx::ld_shared(smem_weight_sf_wg + kL1SFKBlocks + k_block_idx);
                    } else {
                        l2_sf = ptx::ld_shared(smem_weight_sf_wg + k_block_idx);
                    }

                    if (block_phase == sched::BlockPhase::Linear1) {
                        if constexpr (kSwapABActive) {
                            auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                                using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                                constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                                float swap_accum[kSwapAccum];

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = smem_desc_b(stage_idx, smem_b_wg_offset + k * SwapWGMMA::K);
                                    auto desc_b = smem_desc_a(stage_idx, k * SwapWGMMA::K);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    const float scale_0 = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + token_1) : 0.0f;
                                    final_accum[i * 4 + 0] += scale_0 * gate_sf * swap_accum[i * 4 + 0];
                                    final_accum[i * 4 + 2] += scale_0 * up_sf * swap_accum[i * 4 + 2];
                                    final_accum[i * 4 + 1] += scale_1 * gate_sf * swap_accum[i * 4 + 1];
                                    final_accum[i * 4 + 3] += scale_1 * up_sf * swap_accum[i * 4 + 3];
                                }

                                release_empty_stage(stage_idx);
                            };

                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if constexpr (kIntermediateHidden <= 2048) {
                                if (n_swap <= 8) {
                                    run_swap_ab_l1.template operator()<8>();
                                } else if (n_swap <= 16) {
                                    run_swap_ab_l1.template operator()<16>();
                                } else if (n_swap <= 32) {
                                    run_swap_ab_l1.template operator()<32>();
                                } else {
                                    run_swap_ab_l1.template operator()<64>();
                                }
                            } else {
                                switch (n_swap) {
                                    case 8:  run_swap_ab_l1.template operator()<8>();  break;
                                    case 16: run_swap_ab_l1.template operator()<16>(); break;
                                    case 24: run_swap_ab_l1.template operator()<24>(); break;
                                    case 32: run_swap_ab_l1.template operator()<32>(); break;
                                    case 40: run_swap_ab_l1.template operator()<40>(); break;
                                    case 48: run_swap_ab_l1.template operator()<48>(); break;
                                    case 56: run_swap_ab_l1.template operator()<56>(); break;
                                    default: run_swap_ab_l1.template operator()<64>(); break;
                                }
                            }
                        } else {
                            // Single per-128 K-block WGMMA group
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            // L1: gate/up alternate at gran=8 along N; each `i` block of 8
                            // cols belongs entirely to one of {gate, up}, so .x and .y
                            // share the same scalar.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float sb = (i & 1u) ? up_sf : gate_sf;
                                final_accum[i*4+0] += scale_a_0_lo * sb * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * sb * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * sb * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * sb * accum[i*4+3];
                            }
                        }
                    } else {
                        if constexpr (kSwapABActive) {
                            DG_STATIC_ASSERT(kL2ActsSFGranK == 64,
                                             "L2 swapAB assumes per-64 activation scales");
                            auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                                using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                                constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                                float swap_accum[kSwapAccum];

                                auto promote_swap_accum = [&](const uint32_t& sf_group) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        const uint32_t token_0 = i * 8 + col_idx * 2;
                                        const uint32_t token_1 = token_0 + 1;
                                        const float scale_0 = token_0 < valid_m ?
                                            ptx::ld_shared(smem_sfa[stage_idx] + sf_group * BLOCK_M + token_0) : 0.0f;
                                        const float scale_1 = token_1 < valid_m ?
                                            ptx::ld_shared(smem_sfa[stage_idx] + sf_group * BLOCK_M + token_1) : 0.0f;
                                        final_accum[i * 4 + 0] += scale_0 * l2_sf * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * l2_sf * swap_accum[i * 4 + 2];
                                        final_accum[i * 4 + 1] += scale_1 * l2_sf * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * l2_sf * swap_accum[i * 4 + 3];
                                    }
                                };

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                    auto desc_a = smem_desc_b(stage_idx, smem_b_wg_offset + k * SwapWGMMA::K);
                                    auto desc_b = smem_desc_a(stage_idx, k * SwapWGMMA::K);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                promote_swap_accum(0);

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                    const uint32_t k_off = (BLOCK_K / 2) + k * SwapWGMMA::K;
                                    auto desc_a = smem_desc_b(stage_idx, smem_b_wg_offset + k_off);
                                    auto desc_b = smem_desc_a(stage_idx, k_off);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                promote_swap_accum(1);

                                release_empty_stage(stage_idx);
                            };

                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if constexpr (kIntermediateHidden <= 2048) {
                                if (n_swap <= 8) {
                                    run_swap_ab_l2.template operator()<8>();
                                } else if (n_swap <= 16) {
                                    run_swap_ab_l2.template operator()<16>();
                                } else if (n_swap <= 32) {
                                    run_swap_ab_l2.template operator()<32>();
                                } else {
                                    run_swap_ab_l2.template operator()<64>();
                                }
                            } else {
                                switch (n_swap) {
                                    case 8:  run_swap_ab_l2.template operator()<8>();  break;
                                    case 16: run_swap_ab_l2.template operator()<16>(); break;
                                    case 24: run_swap_ab_l2.template operator()<24>(); break;
                                    case 32: run_swap_ab_l2.template operator()<32>(); break;
                                    case 40: run_swap_ab_l2.template operator()<40>(); break;
                                    case 48: run_swap_ab_l2.template operator()<48>(); break;
                                    case 56: run_swap_ab_l2.template operator()<56>(); break;
                                    default: run_swap_ab_l2.template operator()<64>(); break;
                                }
                            }
                        } else if constexpr (kL2ActSFPerBlockK) {
                            // The per-128 act SF: no per-64 split — one full-BLOCK_K
                            // WGMMA group, single promotion with the k-block's per-row scale.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                            }
                        } else {
                            // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                            // First half: K=0..63, SFA = scale_a_*_lo
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k * WGMMA::K);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k * WGMMA::K);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // L2 first half: single scalar `l2_sf` broadcast across N.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                            }

                            // Second half: K=64..127, SFA = scale_a_*_hi
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = smem_desc_a(stage_idx, smem_a_wg_offset + k_off);
                                auto desc_b = smem_desc_b(stage_idx, smem_b_wg_offset + k_off);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            release_empty_stage(stage_idx);

                            // L2 second half: same broadcast scalar `l2_sf`.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_hi * l2_sf * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_hi * l2_sf * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_hi * l2_sf * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_hi * l2_sf * accum[i*4+3];
                            }
                        }
                    }
                    prof_end_of_k_block();
                }
                }
                if constexpr (kTrace) {
                    const uint32_t p = block_phase == sched::BlockPhase::Linear1 ? 0u : 1u;
                    prof_total[p][0] += prof_wait, prof_total[p][1] += prof_decode, prof_total[p][2] += prof_rest;
                    prof_total[p][3] += prof_sub[0], prof_total[p][4] += prof_sub[1], prof_total[p][5] += prof_sub[2];
                    prof_sub[0] = prof_sub[1] = prof_sub[2] = 0;
                }

            }
            trace_math(43, 0);   // MAINLOOP_END (last warpgroup_wait<0> of the tile done)
            if constexpr (kSplitK) {
                if (scheduler.current_split) {
                    // tile slot: L1 tiles by (pool block, n-block), L2 tiles behind all L1 slots
                    const uint32_t slot = block_phase == sched::BlockPhase::Linear1 ?
                        pool_block_idx * kNumL1BlockNs + n_block_idx :
                        kSplitKMaxPoolBlocks * kNumL1BlockNs + pool_block_idx * kNumL2BlockNs + n_block_idx;
                    // this thread's 32 swap accumulators (2 weight halves x 16), float4-strided so a warp's lanes are contiguous
                    float4* part = reinterpret_cast<float4*>(split_k_partials + static_cast<uint64_t>(slot) * layout::kSM90FusedSplitKTileFloats) +
                                   epilogue_thread_idx;
                    if (scheduler.current_k_half == 1) {
                        #pragma unroll
                        for (uint32_t i = 0; i < 8; ++ i)
                            part[i * kNumEpilogueThreads] = make_float4(final_accum[i * 4 + 0], final_accum[i * 4 + 1],
                                                                        final_accum[i * 4 + 2], final_accum[i * 4 + 3]);
                        // every math thread's partial before the flag: CTA barrier, then one gpu-scope release by thread 0
                        sync_tile_math();
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                            ptx::red_add_rel(split_k_flags + slot, 1u);
                        __syncwarp();
                        trace_math(44, 2);   // EPILOGUE_END (aux 2: split-K half 1, partial handed off)
                        return;
                    }
                    // every lane polls (warp-converged, no single-thread spin in the tile code)
                    while (ptx::ld_acq(split_k_flags + slot) == 0u);
                    sync_tile_math();
                    if (epilogue_thread_idx == 0)
                        split_k_flags[slot] = 0u;   // for the next launch (every reader passed the barrier above)
                    #pragma unroll
                    for (uint32_t i = 0; i < 8; ++ i) {
                        const float4 v = __ldcg(part + i * kNumEpilogueThreads);
                        final_accum[i * 4 + 0] += v.x, final_accum[i * 4 + 1] += v.y;
                        final_accum[i * 4 + 2] += v.z, final_accum[i * 4 + 3] += v.w;
                    }
                }
            }
            mxfp4_cd_free_sync();
            // MXFP4 swap tile with the direct epilogue (final_accum holds the swap accumulators, see the swap mainloop)
            bool mxfp4_direct_swap = false;
            if constexpr (kMXFP4DirectSwapEpi)
                mxfp4_direct_swap = valid_m <= kMXFP4SwapMaxBucket;
            // prologue prefetch: peek after the mainloop; the epilogue hides the load
            if constexpr (kSFProloguePrefetch) {
                if (warp_idx_in_wg == 0 and sf_pf_key == 0)
                    sf_peek();
            }

            // Skip epilogue when block is past valid M (still must release via empty)
            if (row_base >= valid_m) {
                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (kMXFP4SplitMWGPublish and not kRingCoversFullPool) {
                        // this half's arrival and slot release; it wrote nothing, so a relaxed red (the release sequence of the
                        // active half's red.release carries its data to the L2 loader's acquire). The arrival must still wait for the
                        // previous generation's L2 tiles of the slot (as the active half's epilogue does): the L2 loader of that
                        // generation tests the count for equality, so an early arrival of this generation would hang it
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            const uint32_t l2_empty_target = kNumL2BlockNs * get_ring_wave_idx(pool_block_idx);
                            if (l2_empty_target > 0)
                                while (ptx::ld_acq(workspace.get_l2_empty_count_ptr(ring_block_idx)) != l2_empty_target);
                            ptx::red_add(workspace.get_l2_full_count_ptr(ring_block_idx), 1u);
                            ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                        }
                        __syncwarp();
                    } else if constexpr (kL2ArrivalNeedsFullSync) {
                        sync_tile_math();
                    }
                } else {
                    if constexpr (kL2EpilogueRequiresFullSync)
                        sync_tile_math();
                }
                trace_math(44, 1);   // EPILOGUE_END (aux 1: no valid rows, epilogue skipped)
                return;
            }

            if (block_phase == sched::BlockPhase::Linear1) {
                // Ring mode: wait until the previous generation of L2 blocks
                // (all N blocks of the earlier pool block mapped to this slot)
                // has consumed the L2-acts slot before overwriting it.
                if constexpr (not kRingCoversFullPool) {
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const uint32_t l2_empty_target = kNumL2BlockNs * get_ring_wave_idx(pool_block_idx);
                    // generation 0 has no previous consumers (the counter is zeroed by the last launch's cleanup and no L2 tile of this
                    // slot can complete before its L1 tiles publish), so the acquire round trip is skipped; later generations wait
                    if (l2_empty_target > 0)
                        while (ptx::ld_acq(l2_empty_ptr) != l2_empty_target);
                    trace_math(45, 0);   // RING_L2_SLOT_FREE (L1 epilogue may overwrite the L2-acts slot)
                }

                if constexpr (kSwapABActive) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    const uint32_t out_col_base =
                        wg_l1_out_n_offset + warp_idx_in_wg * 8 + row_idx;
                    // XOR-swizzle of the FP32 staging tile: the 16 B column chunk is permuted by the token index ((c ^ t) & 7), a bijection
                    // within each row that keeps 4-float vector accesses contiguous; the FP8 tile feeding the TMA store stays linear.
                    DG_STATIC_ASSERT((L1_OUT_BLOCK_N & (L1_OUT_BLOCK_N - 1)) == 0 and
                                     L1_OUT_BLOCK_N >= 4,
                                     "swapAB FP32 staging swizzle needs a power-of-2 row");
                    constexpr uint32_t kSwapL1FP32SwizzleMask = L1_OUT_BLOCK_N / 4 - 1;
                    auto swap_l1_fp32_idx = [](const uint32_t& token, const uint32_t& col) {
                        return token * L1_OUT_BLOCK_N +
                               (col ^ ((token & kSwapL1FP32SwizzleMask) << 2));
                    };
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            float g0 = final_accum[i * 4 + 0];
                            float u0 = final_accum[i * 4 + 2];
                            clamp_gate(g0);
                            clamp_up(u0);
                            smem_cd_swap_l1_fp32[swap_l1_fp32_idx(token_0, out_col_base)] =
                                silu(g0) * u0;
                        }
                        if (token_1 < valid_m) {
                            float g1 = final_accum[i * 4 + 1];
                            float u1 = final_accum[i * 4 + 3];
                            clamp_gate(g1);
                            clamp_up(u1);
                            smem_cd_swap_l1_fp32[swap_l1_fp32_idx(token_1, out_col_base)] =
                                silu(g1) * u1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    for (uint32_t token = epilogue_thread_idx; token < valid_m; token += kNumEpilogueThreads) {
                        const float weight = *l1_topk_weights_buffer
                            .get_data_buffer(ring_m_idx + token)
                            .get_base_ptr<float>();
                        float weight_sf_inv;
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; ++ col) {
                            const float v = smem_cd_swap_l1_fp32[swap_l1_fp32_idx(token, col)];
                            amax = cute::max(amax, cute::abs(v));
                        }
                        amax *= cute::abs(weight);
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        sm90_fp8_fused_mega_moe_get_e4m3_sf_and_sf_inv<kMXFP4Weights>(amax_pair, sf_pair, sf_inv_pair);
                        const float sf = sf_pair.x;
                        weight_sf_inv = weight * sf_inv_pair.x;

                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        // The L2-activation SF pool is strided by SF_BLOCK_M (= align(BLOCK_M, 128)): the L2 producer reads it as
                        // sfa_m_idx = ring_block_idx * SF_BLOCK_M and the non-swap L1 writes it so; this path must use the same stride.
                        const uint32_t token_idx = ring_block_idx * SF_BLOCK_M + token;
                        sf_base_ptr[n_block_idx * kNumPaddedSFPoolTokens + token_idx] = sf;

                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; col += 2) {
                            // col is even, so col and col+1 share one 16B chunk and
                            // stay adjacent under the swizzle.
                            const uint32_t fp32_idx = swap_l1_fp32_idx(token, col);
                            const float v0 = smem_cd_swap_l1_fp32[fp32_idx + 0] * weight_sf_inv;
                            const float v1 = smem_cd_swap_l1_fp32[fp32_idx + 1] * weight_sf_inv;
                            const __nv_fp8x2_e4m3 pair(make_float2(v0, v1));
                            auto* ptr = reinterpret_cast<uint16_t*>(
                                smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col);
                            *ptr = pair.__x;
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8,
                            n_block_idx * L1_OUT_BLOCK_N,
                            ring_m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                    if constexpr (kL2ArrivalCounter and kRingCoversFullPool) {
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
                        }
                    } else if constexpr (kRingCoversFullPool) {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            ptx::red_or_rel_gpu(
                                workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                                1ull << n_block_idx);
                        }
                    } else {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                workspace.get_l2_full_count_ptr(ring_block_idx), 1u);
                            ptx::red_add(
                                workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                        }
                    }
                    __syncwarp();
                    if constexpr (kL2ArrivalCounter and kRingCoversFullPool)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {

                if (mxfp4_direct_swap) {
                    // MXFP4 swap tile, direct epilogue on the swap accumulator layout. Thread (warp w, lane l) holds, per weight half h, the
                    // gate (weight row 16 (4 h + w) + l / 4) and up (+ 8) rows of post-SwiGLU column oc_h = 8 (4 h + w) + l / 4 (the
                    // gran-8 gate / up interleave), for the tokens 8 i + 2 (l % 4) + {0, 1}. The per-element operations and their order are
                    // the regular epilogue's (amax over a token's columns is a max: order-free), so the output is bitwise unchanged
                    DG_STATIC_ASSERT(not kMXFP4DirectSwapEpi or kNumL1OutSFGroups == 1 and not kL1OutSwizzled and WG_L1_OUT_BLOCK_N == 64 and kMXFP4SwapMaxBucket == 32,
                                     "MXFP4 direct swap epilogue: one act-SF group per warpgroup tile, linear staging, 32-token buckets");
                    DG_STATIC_ASSERT(not kMXFP4DirectSwapEpi or SMEM_CD_SIZE / 2 >= SMEM_CD_L1_SIZE and SMEM_CD_SIZE / 2 + 2 * kMXFP4SwapMaxBucket * sizeof(float) <= SMEM_CD_SIZE,
                                     "MXFP4 direct swap epilogue: the amax scratch lives in the upper half of the CD region");
                    constexpr uint32_t kChunks = kMXFP4SwapMaxBucket / 8;
                    const uint32_t num_chunks = (valid_m + 7u) / 8u;
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };
                    auto silu = [](float x) -> float {
                        if constexpr (kFastMath)
                            return sm90_fp8_fused_mega_moe_silu_ftz_exp(x);
                        return x * (1.0f / (1.0f + expf(-x)));
                    };
                    // across the warps (and, sharing one act-SF group, both warpgroups): one scratch float per token in the upper half of
                    // the CD region (the staging tile occupies the lower half; the other warpgroup is in this tile's mainloop or epilogue)
                    float* amax_scratch = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_cd_l1) + SMEM_CD_SIZE / 2) +
                                          (kSplitNSharesSF ? 0u : epilogue_wg_idx * kMXFP4SwapMaxBucket);
                    const auto amax_bar = [&]() {
                        if constexpr (kSplitNSharesSF)
                            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        else
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    };
                    // the tokens' topk weights, all loaded ahead of the scratch reset and its barrier: inside the SwiGLU loop each load
                    // waited behind the previous token's shared atomicMax (generic pointers), one global round trip per token pair
                    // (Flash M8 L1 tile: 0.25 of MAINLOOP_END -> EPI_MATH 1.15 us; a probe without the loads, notes §44)
                    float topk_w[kChunks][2];
                    #pragma unroll
                    for (uint32_t i = 0; i < kChunks; ++ i) {
                        #pragma unroll
                        for (uint32_t j = 0; j < 2; ++ j) {
                            const uint32_t t = i * 8 + col_idx * 2 + j;
                            topk_w[i][j] = t < valid_m ? *l1_topk_weights_buffer
                                .get_data_buffer(ring_m_idx + row_base + t)
                                .template get_base_ptr<float>() : 0.0f;
                        }
                    }
                    if ((kSplitNSharesSF ? epilogue_warp_idx : warp_idx_in_wg) == 0)
                        amax_scratch[lane_idx] = 0.0f;
                    amax_bar();
                    // SwiGLU in place: the weighted value replaces the gate slot final_accum[h * 16 + i * 4 + j] (fewer live registers)
                    #pragma unroll
                    for (uint32_t i = 0; i < kChunks; ++ i) {
                        if (i < num_chunks) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++ j) {
                                const uint32_t t = i * 8 + col_idx * 2 + j;
                                const float w = topk_w[i][j];
                                float amax = 0.0f;
                                #pragma unroll
                                for (uint32_t h = 0; h < 2; ++ h) {
                                    float g = final_accum[h * (kMXFP4SwapMaxBucket / 2) + i * 4 + j];
                                    float u = final_accum[h * (kMXFP4SwapMaxBucket / 2) + i * 4 + 2 + j];
                                    clamp_gate(g);
                                    clamp_up(u);
                                    const float v = silu(g) * u;
                                    amax = fmaxf(amax, fabsf(v));
                                    final_accum[h * (kMXFP4SwapMaxBucket / 2) + i * 4 + j] = v * w;
                                }
                                amax *= cute::abs(w);
                                // the 8 lanes of a column quad (same l % 4) hold the same tokens
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));
                                if (row_idx == 0 and t < valid_m)
                                    atomicMax(reinterpret_cast<unsigned int*>(amax_scratch + t), __float_as_uint(amax));
                            }
                        }
                    }
                    amax_bar();
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    const bool sf_writer = warp_idx_in_wg == 0 and row_idx == 0 and (not kSplitNSharesSF or epilogue_wg_n_idx == 0);
                    #pragma unroll
                    for (uint32_t i = 0; i < kChunks; ++ i) {
                        if (i < num_chunks) {
                            const uint32_t t0 = i * 8 + col_idx * 2;
                            float2 sf_pair, sf_inv_pair;
                            sm90_fp8_fused_mega_moe_get_e4m3_sf_and_sf_inv<kMXFP4Weights>(make_float2(amax_scratch[t0], amax_scratch[t0 + 1]), sf_pair, sf_inv_pair);
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++ j) {
                                const uint32_t t = t0 + j;
                                if (t < valid_m) {
                                    const float sf_inv = j == 0 ? sf_inv_pair.x : sf_inv_pair.y;
                                    #pragma unroll
                                    for (uint32_t h = 0; h < 2; ++ h) {
                                        const uint32_t oc = (h * 4 + warp_idx_in_wg) * 8 + row_idx;
                                        const __nv_fp8_e4m3 q(final_accum[h * (kMXFP4SwapMaxBucket / 2) + i * 4 + j] * sf_inv);
                                        *reinterpret_cast<uint8_t*>(smem_cd_l1 + smem_cd_l1_wg_offset + (row_base + t) * WG_SMEM_CD_L1_STRIDE_N + oc) = q.__x;
                                    }
                                    if (sf_writer)
                                        sf_base_ptr[l2_sf_group_base * kNumPaddedSFPoolTokens + ring_block_idx * SF_BLOCK_M + row_base + t] =
                                            j == 0 ? sf_pair.x : sf_pair.y;
                                }
                            }
                        }
                    }
                    trace_epi(50, 0);   // EPI_MATH_DONE
                } else {
                // ---------------- L1 EPILOGUE: activation + FP8 quantize + TMA store ----------------
                // `final_accum`: kAccumPerThread/4 chunks of 4 floats per thread = (r0c0, r0c1, r1c0, r1c1); gate and up chunks
                // alternate, pair `p` uses chunks 2p and 2p+1 and yields 4 post-SwiGLU floats at output cols p*8 + col_idx*2 + {0,1}.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                // Output columns [kL2ActsSFGranK*g, kL2ActsSFGranK*(g+1)) of this warpgroup's tile form L2 act-SF group g
                // (per-64: two groups on the 128-column production tile; per-128: one group over the whole tile).
                constexpr uint32_t kNumSFGroups = kNumL1OutSFGroups;
                constexpr uint32_t kPairsPerSFGroup = kNumPairs / kNumSFGroups;
                DG_STATIC_ASSERT(kNumPairs % kNumSFGroups == 0, "L2 act-SF groups must cover whole 8-column pairs");
                DG_STATIC_ASSERT(not kSplitNSharesSF or kNumSFGroups == 1, "Shared-SF split implies one SF group per warpgroup");
                float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];

                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];
                float amax_r0[kNumSFGroups], amax_r1[kNumSFGroups];
                #pragma unroll
                for (uint32_t g = 0; g < kNumSFGroups; ++ g)
                    amax_r0[g] = amax_r1[g] = 0.0f;

                auto clamp_gate = [](float& x) {
                    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                        x = cute::min(x, kActivationClamp);
                };
                auto clamp_up = [](float& x) {
                    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                        x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                };
                auto silu = [](float x) -> float {
                    if constexpr (kFastMath)
                        return sm90_fp8_fused_mega_moe_silu_ftz_exp(x);
                    return x * (1.0f / (1.0f + expf(-x)));
                };

                // Rows at or beyond `valid_m` are computed like the others; nothing of theirs is observable (the staging and SF stores
                // stay predicated on the row's validity). fmaxf/fabsf agree bit-for-bit with cute::max/abs for non-NaN values (amax
                // starts at +0 and never adopts a NaN in either form).
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t gate = 2 * p, up = 2 * p + 1;

                    float g_r0_c0 = final_accum[gate*4 + 0];
                    float g_r0_c1 = final_accum[gate*4 + 1];
                    float g_r1_c0 = final_accum[gate*4 + 2];
                    float g_r1_c1 = final_accum[gate*4 + 3];
                    float u_r0_c0 = final_accum[up*4   + 0];
                    float u_r0_c1 = final_accum[up*4   + 1];
                    float u_r1_c0 = final_accum[up*4   + 2];
                    float u_r1_c1 = final_accum[up*4   + 3];
                    clamp_gate(g_r0_c0);
                    clamp_gate(g_r0_c1);
                    clamp_gate(g_r1_c0);
                    clamp_gate(g_r1_c1);
                    clamp_up(u_r0_c0);
                    clamp_up(u_r0_c1);
                    clamp_up(u_r1_c0);
                    clamp_up(u_r1_c1);

                    const uint32_t g = p / kPairsPerSFGroup;
                    swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                    swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                    amax_r0[g] = fmaxf(amax_r0[g], fmaxf(fabsf(swiglu_r0[p][0]), fabsf(swiglu_r0[p][1])));
                    swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                    swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                    amax_r1[g] = fmaxf(amax_r1[g], fmaxf(fabsf(swiglu_r1[p][0]), fabsf(swiglu_r1[p][1])));
                }

                // Apply token weight: SwiGLU * topk_weight (single load per row)
                const float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                    .get_data_buffer(ring_m_idx + row_offset_r0)
                    .template get_base_ptr<float>() : 0.0f;
                const float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                    .get_data_buffer(ring_m_idx + row_offset_r1)
                    .template get_base_ptr<float>() : 0.0f;
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    swiglu_r0[p][0] *= weight_r0;
                    swiglu_r0[p][1] *= weight_r0;
                    swiglu_r1[p][0] *= weight_r1;
                    swiglu_r1[p][1] *= weight_r1;
                }

                #pragma unroll
                for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                amax_r0[g] *= cute::abs(weight_r0);
                amax_r1[g] *= cute::abs(weight_r1);

                // Reduce amax across the 4 col-lanes that share a row (same `lane_idx >> 2`, different `lane_idx & 3` partition the
                // WG-owned columns of the same r_0/r_1): an INTRA-group reduction (xor 2, xor 1); an inter-group one would merge 8 rows.
                amax_r0[g] = fmaxf(amax_r0[g], __shfl_xor_sync(0xffffffffu, amax_r0[g], 2));
                amax_r1[g] = fmaxf(amax_r1[g], __shfl_xor_sync(0xffffffffu, amax_r1[g], 2));
                amax_r0[g] = fmaxf(amax_r0[g], __shfl_xor_sync(0xffffffffu, amax_r0[g], 1));
                amax_r1[g] = fmaxf(amax_r1[g], __shfl_xor_sync(0xffffffffu, amax_r1[g], 1));

                // Phase 2: cross-WG amax. When two N-split warpgroups share one per-64 SF group each has only seen its own
                // WG_L1_OUT_BLOCK_N columns; reduce across both through a small smem scratch (upper, currently unused half of the
                // CD staging region) so BOTH WGs quantize with the SAME SF.
                if constexpr (kSplitNSharesSF) {
                    float* amax_scratch = reinterpret_cast<float*>(
                        reinterpret_cast<uint8_t*>(smem_cd_l1) + SMEM_CD_SIZE / 2);
                    #pragma unroll
                    for (uint32_t i = epilogue_thread_idx; i < BLOCK_M; i += kNumEpilogueThreads)
                        amax_scratch[i] = 0.0f;
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (col_idx == 0) {
                        atomicMax(reinterpret_cast<unsigned int*>(&amax_scratch[r_0]), __float_as_uint(amax_r0[g]));
                        atomicMax(reinterpret_cast<unsigned int*>(&amax_scratch[r_1]), __float_as_uint(amax_r1[g]));
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    amax_r0[g] = amax_scratch[r_0];
                    amax_r1[g] = amax_scratch[r_1];
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }

                // Compute SF and inverse SF for each row
                float2 amax_pair = {amax_r0[g], amax_r1[g]};
                float2 sf_pair, sf_inv_pair;
                sm90_fp8_fused_mega_moe_get_e4m3_sf_and_sf_inv<kMXFP4Weights>(amax_pair, sf_pair, sf_inv_pair);
                sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x;
                sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y;
                }
                trace_epi(50, 0);   // EPI_MATH_DONE (SwiGLU, weights, amax, SF; fp8 conversion is fused into the staging loop)

                // Quantize into the staging tile through predicated 16-bit stores. With kL1OutSwizzled the tile is staged in the TMA
                // SWIZZLE_128B layout: the 16 B chunk index of a 128 B row is XORed with (row & 7); r_0 and r_1 = r_0 + 8 share
                // row & 7 == row_idx, so the XOR term is a per-thread constant.
                auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
                auto l1_stage_offset = [&](const uint32_t& row, const uint32_t& col) {
                    if constexpr (kL1OutSwizzled)
                        return row * WG_SMEM_CD_L1_STRIDE_N + (((col >> 4) ^ (row & 7u)) << 4) + (col & 15u);
                    else
                        return row * WG_SMEM_CD_L1_STRIDE_N + col;
                };
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t g = p / kPairsPerSFGroup;
                    const float v00 = swiglu_r0[p][0] * sf_inv_r0[g];
                    const float v01 = swiglu_r0[p][1] * sf_inv_r0[g];
                    const float v10 = swiglu_r1[p][0] * sf_inv_r1[g];
                    const float v11 = swiglu_r1[p][1] * sf_inv_r1[g];
                    const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                    const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                    const uint32_t col = p * 8 + col_idx * 2;
                    auto* p0 = reinterpret_cast<uint16_t*>(smem_cd_l1_wg + l1_stage_offset(r_0, col));
                    auto* p1 = reinterpret_cast<uint16_t*>(smem_cd_l1_wg + l1_stage_offset(r_1, col));
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
                }

                // Write SF as float at `[token, group]` in the L2 acts SF buffer (per-kL2ActsSFGranK layout): only col_idx == 0 writes,
                // and in the shared-SF split only the first N-split warpgroup publishes (both own the same slot and rows).
                if (col_idx == 0 and (not kSplitNSharesSF or epilogue_wg_n_idx == 0)) {
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    // SF buffer is (kNumPaddedSFPoolTokens x kIntermediateHidden/kL2ActsSFGranK), MN-major:
                    //   addr[k_idx * num_padded_sf_pool_tokens + token_idx]
                    const uint32_t token_r0 = ring_block_idx * SF_BLOCK_M + row_offset_r0;
                    const uint32_t token_r1 = ring_block_idx * SF_BLOCK_M + row_offset_r1;
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        const uint32_t k_sf_idx = l2_sf_group_base + g;  // post-SwiGLU SF group
                        if (valid_r0)
                            sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0[g];
                        if (valid_r1)
                            sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1[g];
                    }
                }

                }
                // Sync the warpgroup before TMA store. In the shared-tile split
                // both N-split warpgroups must finish writing their halves of the
                // joint L1-output tile, so sync across all epilogue threads.
                if constexpr (kSplitNSharesSF)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                else
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                trace_epi(51, 0);   // EPI_STAGED (fp8 tile in smem, warpgroup barrier passed)

                // TMA store of the entire tile. Padding rows beyond `valid_m` hold stale FP8 / SF but are never consumed: the L2 tile
                // loads them, but its NVLink-scatter epilogue is gated by `m_idx_in_block >= valid_m` and NaN accumulators stay in
                // registers (only valid rows are converted to BF16 and STSM'd into smem).
                if constexpr (kSplitNSharesSF) {
                    // One combined store of the joint L1_OUT_BLOCK_N tile, issued
                    // by the first N-split warpgroup once both halves are staged.
                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            ring_m_idx + row_base);
                        cute::tma_store_arrive();
                    }
                } else {
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1 + smem_cd_l1_wg_offset,
                            out_n_idx,
                            ring_m_idx + row_base);
                        cute::tma_store_arrive();
                    }
                }
                __syncwarp();
                trace_epi(52, 0);   // EPI_STORE_ISSUED
                if constexpr (kMXFP4SplitMDeferPublish) {
                    DG_STATIC_ASSERT(not kMXFP4SplitMDeferPublish or (not kSplitNSharesSF and (kL2ArrivalCounter or not kRingCoversFullPool)),
                                     "deferred MXFP4 split-M publish: per-warpgroup counter arrivals");
                    if constexpr (not kRingCoversFullPool) {
                        // the A slot's loads all landed before the mainloop ended: release it now (relaxed, as before)
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync())
                            ptx::red_add(workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                        __syncwarp();
                        mxfp4_pending_pub = workspace.get_l2_full_count_ptr(ring_block_idx);
                    } else {
                        mxfp4_pending_pub = reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                    }
                    trace_epi(54, 1);   // EPI_PUBLISHED (aux 1: the drain and release deferred to the next tile)
                } else {
                ptx::tma_store_wait<0>();
                trace_epi(53, 0);   // EPI_STORE_WAITED

                // Notify L2 that this L1 output (and SF) is ready. Counter mode: independent WG tiles publish arrivals without a
                // CTA-wide barrier. Ring mode: the mask-mode barrier flow with counting publishes, so every (m, n) L1 block
                // contributes exactly one arrival and one L1-slot release, independent of `valid_m`.
                if constexpr (kL2ArrivalCounter and kRingCoversFullPool) {
                    if constexpr (kSplitNSharesSF) {
                        // The combined tile counts for both N-split warpgroups; the
                        // storing warpgroup publishes all kWarpgroupSplitN arrivals
                        // after its TMA store has drained.
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
                        }
                    } else if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        ptx::red_add_rel(
                            reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)), 1);
                    }
                } else if constexpr (kRingCoversFullPool) {
                    sync_tile_math();
                    if (tile_lead_warp and cute::elect_one_sync()) {
                        ptx::red_or_rel_gpu(
                            workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                            1ull << n_block_idx);
                    }
                } else if constexpr (kMXFP4SplitMWGPublish) {
                    // per-warpgroup ring publish: the warpgroup barrier before the TMA store ordered this half's SF stores
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        ptx::red_add_rel(
                            workspace.get_l2_full_count_ptr(ring_block_idx), 1u);
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                } else {
                    sync_tile_math();
                    if (tile_lead_warp and cute::elect_one_sync()) {
                        ptx::red_add_rel(
                            workspace.get_l2_full_count_ptr(ring_block_idx), 1u);
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx), 1u);
                    }
                }
                __syncwarp();
                trace_epi(54, 0);   // EPI_PUBLISHED
                }
                // In the shared-tile split only the first warpgroup issues and drains the combined TMA store; gate the other
                // warpgroup so it cannot overwrite the joint smem tile in the next block until that store has drained.
                if constexpr (kMXFP4SplitCDFree) {
                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0)
                        asm volatile("bar.arrive %0, %1;" :: "r"(kMXFP4CDFreeBarrierIdx), "r"(kNumEpilogueThreads) : "memory");
                    else
                        mxfp4_cd_free_pending = true;
                } else if constexpr (kSplitNSharesSF) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr bool kFullRowL2Stage = kL2StageMode == 2 or kL2StageMode == 4;
                constexpr bool kL2MetaPrefetch = kFullRowL2Stage, kL2StageSwizzle = kFullRowL2Stage;
                constexpr bool kL2EpiSparse = BLOCK_M == 64 and not kFP8SwapAB and kL2StageMode == 0;
                // row-metadata prefetch (kL2MetaPrefetch): lane l < 16 loads the TokenSrcMetadata of the warp's row l (row-pass staging
                // emits the warp's 16 rows as h * 8 + g * 4 + rr, see below); each row's store takes it from that lane with shfl
                uint32_t pf_meta_rank = 0, pf_meta_token = 0, pf_meta_topk = 0;
                if constexpr (kL2MetaPrefetch) {
                    const uint32_t pf_row_in_wg = warp_idx_in_wg * 16u + (lane_idx & 15u);
                    if (lane_idx < 16 and row_base + pf_row_in_wg < valid_m) {
                        const auto* pf_meta = workspace.get_token_src_metadata_ptr(m_idx + row_base + pf_row_in_wg);
                        pf_meta_rank = pf_meta->rank_idx;
                        pf_meta_token = pf_meta->token_idx;
                        pf_meta_topk = pf_meta->topk_idx;
                    }
                }
                // Ring mode: the L2-acts input of this slot was fully consumed
                // into SMEM by the time any warpgroup reaches the epilogue, so
                // release the slot for the L1 epilogue of the next generation
                // (one increment per N block).
                if constexpr (not kRingCoversFullPool) {
                    if (tile_lead_warp and cute::elect_one_sync()) {
                        ptx::red_add(
                            workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                    }
                }

                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                const uint32_t lane_in_row = lane_idx % 16;
                const uint32_t cols_per_lane = WG_BLOCK_N / 16;

                DG_STATIC_ASSERT(not kSwapABActive or WG_BLOCK_N == 64,
                                 "swapAB BF16 staging swizzle assumes WG_BLOCK_N == 64");
                auto smem_cd_l2_token_idx = [](const uint32_t& token, const uint32_t& col) {
                    if constexpr (kSwapABActive) {
                        constexpr uint32_t kSwizzleMask = WG_BLOCK_N / 4 - 1;
                        return token * WG_BLOCK_N + (col ^ ((token & kSwizzleMask) << 2));
                    } else {
                        return token * WG_BLOCK_N + col;
                    }
                };

                if (mxfp4_direct_swap) {
                    // MXFP4 swap tile: the tile's tokens are staged as bf16 rows [token][WG_BLOCK_N] straight from the swap accumulators
                    // (weight row n = 64 h + 16 w + l / 4 (+ 8), token 8 i + 2 (l % 4) + {0, 1}), then every half-warp scatters whole rows
                    DG_STATIC_ASSERT(not kMXFP4DirectSwapEpi or WG_BLOCK_N == 128 and kNumL2CDPasses == 1 and kL2StageMode == 0 and kMXFP4SwapMaxBucket == 32,
                                     "MXFP4 direct swap L2 epilogue: one full-width staging pass");
                    constexpr uint32_t kChunks = kMXFP4SwapMaxBucket / 8;
                    constexpr uint32_t kRowsPerHalfWarp = kMXFP4SwapMaxBucket / 8;   // 8 half-warps per warpgroup
                    const uint32_t num_chunks = (valid_m + 7u) / 8u;
                    const uint32_t first_row = warp_idx_in_wg * 2 + row_in_warp_block;
                    // source metadata first: its load latency overlaps the conversion
                    layout::TokenSrcMetadata swap_meta[kRowsPerHalfWarp];
                    #pragma unroll
                    for (uint32_t k = 0; k < kRowsPerHalfWarp; ++ k) {
                        const uint32_t t = first_row + k * 8;
                        swap_meta[k] = {};
                        if (t < valid_m)
                            swap_meta[k] = *workspace.get_token_src_metadata_ptr(m_idx + row_base + t);
                    }
                    nv_bfloat16* swap_stage = smem_cd_l2 + smem_cd_l2_wg_offset;
                    #pragma unroll
                    for (uint32_t i = 0; i < kChunks; ++ i) {
                        if (i < num_chunks) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 2; ++ j) {
                                const uint32_t t = i * 8 + col_idx * 2 + j;
                                if (t < valid_m) {
                                    #pragma unroll
                                    for (uint32_t h = 0; h < 2; ++ h) {
                                        #pragma unroll
                                        for (uint32_t e = 0; e < 2; ++ e)
                                            swap_stage[t * WG_BLOCK_N + h * 64 + warp_idx_in_wg * 16 + row_idx + e * 8] =
                                                __float2bfloat16_rn(final_accum[h * (kMXFP4SwapMaxBucket / 2) + i * 4 + e * 2 + j]);
                                    }
                                }
                            }
                        }
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    trace_epi(60, 0);   // EPI_CVT_DONE
                    #pragma unroll
                    for (uint32_t k = 0; k < kRowsPerHalfWarp; ++ k) {
                        const uint32_t t = first_row + k * 8;
                        if (t < valid_m) {
                            const auto packed = *reinterpret_cast<const uint4*>(swap_stage + t * WG_BLOCK_N + lane_in_row * 8);
                            const auto dst_token = combine_token_buffer.get_rank_buffer(swap_meta[k].topk_idx)
                                                   .get_data_buffer(swap_meta[k].token_idx);
                            auto dst_ptr = math::advance_ptr<uint4>(dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, swap_meta[k].rank_idx) = packed;
                        }
                    }
                    trace_epi(61, 0);   // EPI_SCATTER_DONE
                } else {
                if constexpr (kSwapABActive) {
                    auto store_bf16 = [&](const uint32_t& token, const uint32_t& col, float value) {
                        smem_cd_l2[smem_cd_l2_wg_offset + smem_cd_l2_token_idx(token, col)] =
                            __float2bfloat16_rn(value);
                    };

                    auto store_l2_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            store_bf16(token_0, r_0, final_accum[i * 4 + 0]);
                            store_bf16(token_0, r_1, final_accum[i * 4 + 2]);
                        }
                        if (token_1 < valid_m) {
                            store_bf16(token_1, r_0, final_accum[i * 4 + 1]);
                            store_bf16(token_1, r_1, final_accum[i * 4 + 3]);
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l2_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l2_swap_chunk(i);
                        }
                    }
                    // swapAB: the whole warpgroup produced the tile; sync then
                    // scatter the full WG_BLOCK_N-wide staged rows.
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    using ScatterVec = std::conditional_t<(WG_BLOCK_N <= 64), uint2, uint4>;
                    DG_STATIC_ASSERT(cols_per_lane * sizeof(nv_bfloat16) == sizeof(ScatterVec),
                                     "Scatter vector width must match cols_per_lane");
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_base + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;
                        auto smem_ptr = smem_cd_l2 + smem_cd_l2_wg_offset
                            + smem_cd_l2_token_idx(row_in_wg, lane_in_row * cols_per_lane);
                        const auto packed = *reinterpret_cast<ScatterVec*>(smem_ptr);
                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const auto dst_token = combine_token_buffer.get_rank_buffer(src_metadata.topk_idx)
                                               .get_data_buffer(src_metadata.token_idx);
                        auto dst_ptr = math::advance_ptr<ScatterVec>(dst_token.get_base_ptr(),
                            (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(ScatterVec));
                        *sym_buffer.map(dst_ptr, src_metadata.rank_idx) = packed;
                    }
                } else if constexpr (kL2StageMode != 0) {
                    // Row-pass staging: every math warp owns a 2 KiB slot and emits its 16 rows in 4 passes over ROWS (mode 1: 8 rows x
                    // 128 columns per pass, two rows per store; mode 2: 4 rows x 256 columns, one whole row per store). Passes are
                    // warp-local like the column passes: STS, __syncwarp, LDS + remote store, __syncwarp.
                    DG_STATIC_ASSERT(WG_BLOCK_M == 64 and WG_BLOCK_N == 256 and kAccumPerThread == 128,
                                     "L2 row-pass staging assumes the 64x256 warpgroup tile");
                    constexpr uint32_t kWarpStageElems = 4u * WG_BLOCK_N;   // 2 KiB of bf16 per warp
                    DG_STATIC_ASSERT(kWarpStageElems * 4u == WG_BLOCK_M * (WG_BLOCK_N / kNumL2CDPasses),
                                     "row-pass staging must fit the warpgroup's quarter-width slot");
                    nv_bfloat16* warp_stage = smem_cd_l2 + smem_cd_l2_wg_offset + warp_idx_in_wg * kWarpStageElems;
                    constexpr bool kStageSTSM = kL2StageMode == 4;
                    // stmatrix.m8n8.x4: matrix i (0..3) of an instruction is chunk 4q+i of the accumulator row group h
                    // (rows 8h..8h+7 of the warp); lane t provides the address of row t%8 of matrix t/8, and its
                    // register i holds the pair (row t/4, columns 8(4q+i) + 2(t%4), +1) = exactly the wgmma fragment
                    const auto stsm_x4 = [](const uint32_t& addr, const uint32_t& r0, const uint32_t& r1,
                                            const uint32_t& r2, const uint32_t& r3) {
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                                     :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
                    };
                    const uint32_t warp_stage_addr = static_cast<uint32_t>(__cvta_generic_to_shared(warp_stage));
                    const uint32_t stsm_mat = lane_idx >> 3;        // matrix index inside an x4 instruction
                    const uint32_t stsm_row = lane_idx & 7u;        // row of that matrix this lane addresses
                    constexpr uint32_t kRowChunks = WG_BLOCK_N / 8;   // 32 chunks of 16 B per row
                    // chunk XOR f(row): without kL2StageSwizzle f = row & 3 (the 4 rows of a pass on distinct banks for the STS); with it
                    // f = (row & 1) << 2 | row >> 1, so that in a stmatrix.x4 matrix the 4 wanted rows and the 4 junk rows land on 8 distinct
                    // bank groups (512-byte rows start on bank 0, group of a 16 B chunk = (chunk ^ f) & 7). The LDS.128 of a row reads
                    // chunk lane_idx: any XOR below 8 keeps a quarter-warp on 8 distinct groups.
                    const auto swz_f = [](const uint32_t& row) {
                        if constexpr (kL2StageSwizzle)
                            return ((row & 1u) << 2) | (row >> 1);
                        else
                            return row & 3u;
                    };
                    auto stage_elem = [&](const uint32_t& row, const uint32_t& chunk) {
                        return row * WG_BLOCK_N + ((chunk ^ swz_f(row)) << 3);
                    };
                    // stmatrix: only matrix rows (row & 4) == 4g belong to pass g; the other four rows of every matrix go to this
                    // warpgroup's weight-SF slot (256 B, dead between the mainloop's last SF read and the next tile's prologue). Wanted
                    // address = row * 512 + 16 * ((4q + mat) ^ f(row)) = base + 16 * (mat ^ (row >> 1)) + 64 * (q ^ (row & 1)), quad q is
                    // base ^ (q << 6); junk slot s = (mat & 1) << 3 | (mat ^ 2 ^ f(row)) & 7: 16 distinct 16 B slots per instruction.
                    const uint32_t stsm_r = stsm_row & 3u;
                    const uint32_t stsm_base = kL2StageSwizzle
                        ? warp_stage_addr + stsm_r * (WG_BLOCK_N * 2u) + ((stsm_mat ^ (stsm_r >> 1)) << 4) + ((stsm_r & 1u) << 6)
                        : warp_stage_addr + stsm_r * (WG_BLOCK_N * 2u) + ((stsm_mat ^ stsm_r) << 4);
                    const uint32_t stsm_junk = static_cast<uint32_t>(__cvta_generic_to_shared(smem_weight_sf_wg)) +
                        (kL2StageSwizzle ? ((((stsm_mat & 1u) << 3) | ((stsm_mat ^ 2u ^ swz_f(stsm_r)) & 7u)) << 4)
                                         : ((stsm_mat * 4u + stsm_r) << 4));
                    // the slot must exist in the layout, not just in the float count
                    DG_STATIC_ASSERT(not kStageSTSM or (SMEM_WEIGHT_SF_SIZE >= kNumEpilogueWarpgroups * 256u and
                                     kNumWeightSFFloatsPerWG * sizeof(float) >= 256u),
                                     "stmatrix junk rows need a 256 B weight-SF scratch slot per warpgroup (use L2 stage mode 2)");
                    DG_STATIC_ASSERT(not kL2StageSwizzle or WG_BLOCK_N * 2u == 512u, "the staging swizzle assumes 512-byte staging rows");
                    #pragma unroll
                    for (uint32_t pass = 0; pass < 4; ++ pass) {
                        const uint32_t h = pass >> 1;   // accumulator row r_0 / r_1
                        const uint32_t g = pass & 1;    // which 4 of the 8 accumulator rows (lanes 0-15 / 16-31)
                        const uint32_t row_in_pass = row_idx & 3u;
                        if constexpr (kStageSTSM) {
                            const bool wanted = (stsm_row >> 2) == g;
                            #pragma unroll
                            for (uint32_t q = 0; q < kRowChunks / 4; ++ q) {
                                const uint32_t c0 = 4 * q;
                                // junk rows always land on the same 16 B slot of the scratch (no per-q offset)
                                const uint32_t stsm_addr = kL2StageSwizzle ? (stsm_base ^ (q << 6)) : (stsm_base + (q << 6));
                                stsm_x4(wanted ? stsm_addr : stsm_junk,
                                        math::cast_into_bf16_and_pack(final_accum[(c0 + 0) * 4 + 2 * h], final_accum[(c0 + 0) * 4 + 2 * h + 1]),
                                        math::cast_into_bf16_and_pack(final_accum[(c0 + 1) * 4 + 2 * h], final_accum[(c0 + 1) * 4 + 2 * h + 1]),
                                        math::cast_into_bf16_and_pack(final_accum[(c0 + 2) * 4 + 2 * h], final_accum[(c0 + 2) * 4 + 2 * h + 1]),
                                        math::cast_into_bf16_and_pack(final_accum[(c0 + 3) * 4 + 2 * h], final_accum[(c0 + 3) * 4 + 2 * h + 1]));
                            }
                        } else if ((row_idx >> 2) == g and (h == 0 ? valid_r0 : valid_r1)) {
                            #pragma unroll
                            for (uint32_t c = 0; c < kRowChunks; ++ c) {
                                const uint32_t packed = math::cast_into_bf16_and_pack(
                                    final_accum[c * 4 + 2 * h], final_accum[c * 4 + 2 * h + 1]);
                                *reinterpret_cast<uint32_t*>(warp_stage + stage_elem(row_in_pass, c) + col_idx * 2) = packed;
                            }
                        }
                        __syncwarp();
                        trace_epi(60, pass);   // EPI_CVT_DONE
                        if constexpr (kMXFP4SplitM and kL2MetaPrefetch) {
                            // the pass's 4 rows: every LDS and metadata shuffle first, then the stores, each predicated on its
                            // (warp-uniform) row bound. With the loop's early `break` the compiler sank each row's LDS behind the
                            // previous row's STG: a serial LDS -> SHFL -> STG chain of ~0.3 us per pass (Flash M8192 trace). Rows past
                            // valid_m read stale staging bytes and the zero metadata of the prefetch; neither is stored
                            uint4 row_data[4];
                            uint4* row_dst[4];
                            #pragma unroll
                            for (uint32_t rr = 0; rr < 4; ++ rr) {
                                row_data[rr] = *reinterpret_cast<const uint4*>(warp_stage + stage_elem(rr, lane_idx));
                                const uint32_t src_lane = h * 8 + g * 4 + rr;
                                const uint32_t dst_rank = __shfl_sync(0xffffffffu, pf_meta_rank, src_lane);
                                const uint32_t dst_token_idx = __shfl_sync(0xffffffffu, pf_meta_token, src_lane);
                                const uint32_t dst_topk_idx = __shfl_sync(0xffffffffu, pf_meta_topk, src_lane);
                                const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx);
                                row_dst[rr] = sym_buffer.map(math::advance_ptr<uint4>(dst_token.get_base_ptr(),
                                    (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_idx * sizeof(uint4)), dst_rank);
                            }
                            #pragma unroll
                            for (uint32_t rr = 0; rr < 4; ++ rr) {
                                if (row_base + warp_idx_in_wg * 16 + h * 8 + g * 4 + rr < valid_m)
                                    *row_dst[rr] = row_data[rr];
                            }
                        } else {
                        #pragma unroll
                        for (uint32_t rr = 0; rr < 4; ++ rr) {
                            const uint32_t row_in_wg = warp_idx_in_wg * 16 + h * 8 + g * 4 + rr;
                            const uint32_t m_idx_in_block = row_base + row_in_wg;
                            if (m_idx_in_block >= valid_m) break;
                            const auto packed = *reinterpret_cast<const uint4*>(warp_stage + stage_elem(rr, lane_idx));
                            uint32_t dst_rank, dst_token_idx, dst_topk_idx;
                            if constexpr (kL2MetaPrefetch) {
                                // row-metadata prefetch: the row's record sits in lane h * 8 + g * 4 + rr (loaded at the epilogue entry)
                                dst_rank = __shfl_sync(0xffffffffu, pf_meta_rank, h * 8 + g * 4 + rr);
                                dst_token_idx = __shfl_sync(0xffffffffu, pf_meta_token, h * 8 + g * 4 + rr);
                                dst_topk_idx = __shfl_sync(0xffffffffu, pf_meta_topk, h * 8 + g * 4 + rr);
                            } else {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                                dst_rank = src_metadata.rank_idx;
                                dst_token_idx = src_metadata.token_idx;
                                dst_topk_idx = src_metadata.topk_idx;
                            }
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_ptr = math::advance_ptr<uint4>(dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_idx * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank) = packed;
                        }
                        }
                        trace_epi(61, pass);   // EPI_SCATTER_DONE
                        __syncwarp();
                    }
                } else {
                    // Non-swap L2: STSM the BF16 tile into SMEM then NVLink-scatter; with kHalfL2CD the tile is emitted in two N-halves
                    // through a half-width SMEM buffer, otherwise in one full-width pass.
                    constexpr uint32_t kL2Passes     = kNumL2CDPasses;
                    constexpr uint32_t WG_L2_STAGE_N = WG_BLOCK_N / kL2Passes;
                    constexpr uint32_t kIterPerPass  = (kAccumPerThread / 8) / kL2Passes;
                    using ScatterVec = std::conditional_t<(WG_L2_STAGE_N <= 64), uint2, uint4>;
                    const uint32_t l2_cols_per_lane = WG_L2_STAGE_N / 16;
                    DG_STATIC_ASSERT(l2_cols_per_lane * sizeof(nv_bfloat16) == sizeof(ScatterVec),
                                     "Scatter vector width must match cols_per_lane");
                    DG_STATIC_ASSERT(WG_L2_STAGE_N >= 64,
                                     "L2 staging swizzle needs >= 8 16B chunks per row");
                    auto l2_stage_idx = [](const uint32_t& row, const uint32_t& col) {
                        return row * WG_L2_STAGE_N + (col ^ ((row & 7u) << 3));
                    };
                    // sparse-tile epilogue (kL2EpiSparse): the half-warp's source metadata for all of its rows is loaded once per tile
                    // BEFORE the conversion / staging stores (predicated on the row bound), so the scatter loop is a register-fed chain
                    layout::TokenSrcMetadata sparse_meta[kNumRowsPerWarp];
                    if constexpr (kL2EpiSparse) {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                            const uint32_t m_idx_in_block = row_base + row_in_wg;
                            sparse_meta[j] = {};
                            if (m_idx_in_block < valid_m)
                                sparse_meta[j] = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        }
                    }
                    #pragma unroll
                    for (uint32_t pass = 0; pass < kL2Passes; ++ pass) {
                        const uint32_t col_base = pass * WG_L2_STAGE_N;
                        // STSM this pass's columns [col_base, col_base+WG_L2_STAGE_N)
                        // into the (half-)width buffer at (global col - col_base).
                        #pragma unroll
                        for (uint32_t t = 0; t < kIterPerPass; ++ t) {
                            const uint32_t i = pass * kIterPerPass + t;
                            const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                            auto write_pair = [&](uint32_t row, uint32_t gcol, uint32_t packed) {
                                *reinterpret_cast<uint32_t*>(
                                    smem_cd_l2 + smem_cd_l2_wg_offset
                                    + l2_stage_idx(row, gcol - col_base)) = packed;
                            };
                            if (valid_r0) {
                                const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_lo*4 + 0], final_accum[chunk_lo*4 + 1]);
                                const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_hi*4 + 0], final_accum[chunk_hi*4 + 1]);
                                write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                                write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                            }
                            if (valid_r1) {
                                const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_lo*4 + 2], final_accum[chunk_lo*4 + 3]);
                                const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_hi*4 + 2], final_accum[chunk_hi*4 + 3]);
                                write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                                write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                            }
                        }
                        __syncwarp();
                        trace_epi(60, pass);   // EPI_CVT_DONE (this pass's bf16 columns staged in smem)
                        // Scatter this pass's columns to remote ranks via NVLink.
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                            const uint32_t m_idx_in_block = row_base + row_in_wg;
                            if (m_idx_in_block >= valid_m) break;
                            const auto packed = *reinterpret_cast<ScatterVec*>(
                                smem_cd_l2 + smem_cd_l2_wg_offset
                                + l2_stage_idx(row_in_wg, lane_in_row * l2_cols_per_lane));
                            layout::TokenSrcMetadata src_metadata;
                            if constexpr (kL2EpiSparse)
                                src_metadata = sparse_meta[j];
                            else
                                src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                            const auto dst_token = combine_token_buffer.get_rank_buffer(src_metadata.topk_idx)
                                                   .get_data_buffer(src_metadata.token_idx);
                            auto dst_ptr = math::advance_ptr<ScatterVec>(dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset + col_base) * sizeof(nv_bfloat16)
                                + lane_in_row * sizeof(ScatterVec));
                            *sym_buffer.map(dst_ptr, src_metadata.rank_idx) = packed;
                        }
                        trace_epi(61, pass);   // EPI_SCATTER_DONE (this pass's remote stores issued)
                        // Pass 0's scatter must finish reading before pass 1's STSM
                        // overwrites the shared half-width buffer.
                        if constexpr (kL2Passes > 1)
                            __syncwarp();
                    }
                }
                }

                if constexpr (kL2EpilogueRequiresFullSync) {
                    sync_tile_math();
                    trace_epi(62, 0);   // EPI_FULL_SYNC_DONE
                }
            }
            trace_math(44, 0);   // EPILOGUE_END
        };

        // MXFP4 on a ring smaller than the pool, and every traced MXFP4 instance: the tile callbacks are forced inline. Left to NVVM,
        // the mode-5 (N24) instances compiled the L2 callback as a real call (CALL.ABS, STACK 1896: Flash M4096 15.9 vs 4.9 ms, M256 on
        // an 8192-token buffer 1448 vs 455 us); the trace code tips the full-pool mode-5 instances over too (STACK 2144, a 3.5x longer
        // span; there the outlined function is process_math_block itself, hence its attribute above). The untraced full-pool
        // instances keep the plain lambdas (forcing them re-schedules those, Flash M8..M128 +1..+1.6 %).
        // The outlining is a code-size heuristic: check new instances for CALL.ABS (cuobjdump -sass)
        // split-M: the register-A kMode 2 loop tips the full-pool instances over as well (STACK 992, final_accum in local memory)
        // decode modes 6 / 7 (the N48 bucket): NVVM outlined the tile callback even with the ring covering the pool (CALL.ABS x 37, STACK 1728)
        constexpr bool kForceInlineMathTiles = kMXFP4Weights and (not kRingCoversFullPool or kTrace or kMXFP4SplitM or kMXFP4DecodeMode >= 6);
        // trace: 39 MATH_CLOCK_START / 38 MATH_CLOCK_END (aux = %clock64): effective SM clock of the tile loop (cycles per globaltimer ns)
        if constexpr (kTrace) {
            uint64_t clk;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(clk) :: "memory");
            trace_math(39, clk);
        }
        // MXFP4 split-M: both expanded tiles start free (one free arrival per tile ahead of the first decode into it)
        if constexpr (kMXFP4SplitM) {
            mxfp4_splitm_bar_arrive(kMXFP4SplitMFreeBarrierIdx + epilogue_wg_idx * 2 + 0);
            mxfp4_splitm_bar_arrive(kMXFP4SplitMFreeBarrierIdx + epilogue_wg_idx * 2 + 1);
        }
        if constexpr (kForceInlineMathTiles) {
            scheduler.for_each_block_replay(tile_table,
                [&](const uint32_t& local_expert_idx,
                    const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) CUTLASS_LAMBDA_FUNC_INLINE {
                    process_math_block(
                        std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                        local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                },
                [&](const uint32_t& local_expert_idx,
                    const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) CUTLASS_LAMBDA_FUNC_INLINE {
                    process_math_block(
                        std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                        local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                });
        } else {
        scheduler.for_each_block_replay(tile_table,
            [&](const uint32_t& local_expert_idx,
                const uint32_t& num_k_blocks,
                const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_math_block(
                    std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                    local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            },
            [&](const uint32_t& local_expert_idx,
                const uint32_t& num_k_blocks,
                const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_math_block(
                    std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                    local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            });
        }
        // the last tile's deferred L1 release (nothing waits on this warpgroup any more)
        mxfp4_flush_pub();
        // MXFP4 split-M: consume the peer's one free arrival per tile that no decode took (the initial ones, shifted to the end)
        if constexpr (kMXFP4SplitM) {
            ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMFreeBarrierIdx + (epilogue_wg_idx ^ 1u) * 2 + 0);
            ptx::sync_aligned(kNumEpilogueThreads, kMXFP4SplitMFreeBarrierIdx + (epilogue_wg_idx ^ 1u) * 2 + 1);
        }
        if constexpr (kTrace) {
            uint64_t clk;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(clk) :: "memory");
            trace_math(38, clk);
        }
        mxfp4_cd_free_sync();   // the combine reuses the CD region
        trace_math(19, trace_tile);   // TILES_DONE (aux = number of tiles this CTA ran)
        if constexpr (kTrace) {
            if (warp_idx_in_wg == 0 and lane_idx == 0 and epilogue_wg_idx < 2) {
                auto ptr = trace + ((static_cast<uint64_t>(sm_idx) * 4 + 2 + epilogue_wg_idx) * kTraceSlots + kTraceSlots - 6) * 2;
                #pragma unroll
                for (uint32_t i = 0; i < 12; ++ i)
                    ptr[i] = prof_total[i / 6][i % 6];
            }
        }
        // Combine staging: whole rows (kNumChunks == 1) when three whole-row slots per warp fit the smem in front of the
        // barriers, else the row is combined in chunks
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // the ladder itself lives in `layout/sm90_fused_mega_moe.cuh`, where the host reads it too
        constexpr uint32_t kNumChunkSlots = layout::kSM90FusedCombineChunkSlots;
        constexpr uint32_t kNumChunks = layout::get_sm90_fused_moe_combine_num_chunks(
            kHidden, kNumCombineWarps, SMEM_BEFORE_BARRIER_SIZE, kSplitMNWarpgroups);
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;

        // Sliced post-barrier combine of the decode topology (whole-row slots, one 512 B slice per warp; each element still sums
        // the slots in ascending order in fp32 -> bitwise): 2 = dynamic work-list form, 1 = static form (one CTA barrier, every
        // warp walks the same remaining set); 0 = the plain per-warp loop of the prefill topology and of the chunked combine
        constexpr uint32_t kCombineSlicedForm = 2u;
        // staging capacity (the static_asserts of the two forms below): the dynamic form stages one item's kNumTopk slices in the
        // warp's whole-row stage-0 buffer, the static form kNumTopk + 1 slices per warp in the two whole-row stages
        constexpr bool kCombineSlicedFits = kCombineSlicedForm == 2 ? (kNumTopk <= kNumCombineWarps)
                                                                    : (kNumTopk + 1 <= 2 * kNumCombineWarps);
        // MXFP4 decode topology: the dynamic sliced form for every hidden (Pro 7168 too, whose rows the per-warp loop combined in chunks:
        // one warp per token, Pro M8 9 us on SM 0 after the barrier while the other SMs idled) with 1 KiB slices (512 B when the
        // warp's stage-0 buffer cannot stage kNumTopk of those), and SM-major token ownership (see combine_token_of)
        constexpr bool kCombineMXFP4 = kMXFP4Weights and kDecodeTopology;
        constexpr uint32_t kMXFP4CombineSliceBytes =
            (kNumHiddenBytes % 1024u == 0 and kNumTopk * 1024u <= kNumChunkBytes and kNumHiddenBytes / 1024u >= kNumCombineWarps) ? 1024u : 512u;
        constexpr uint32_t kCombineSliced = kCombineMXFP4 ?
            ((kNumHiddenBytes % kMXFP4CombineSliceBytes == 0 and kNumTopk * kMXFP4CombineSliceBytes <= kNumChunkBytes) ? 2u : 0u) :
            ((kDecodeTopology and kNumChunks == 1 and kHidden % (kNumCombineWarps * 256) == 0 and kCombineSlicedFits) ? kCombineSlicedForm : 0u);
        DG_STATIC_ASSERT(not kCombineMXFP4 or kCombineSliced == 2, "MXFP4 SM-major token ownership needs the dynamic sliced combine");
        // stripe j of (this SM, math warp w): FP8 SM-minor (a CTA owns kNumCombineWarps consecutive tokens per stripe); MXFP4 SM-major
        // (token = (j * kNumCombineWarps + w) * kNumSMs + sm), so a small batch lands one token per SM instead of all on SM 0. Every
        // token still has exactly one owner, and the per-token sum order is unchanged (bitwise)
        const auto combine_token_of = [&](const uint32_t& j, const uint32_t& w) {
            return kCombineMXFP4 ? (j * kNumCombineWarps + w) * kNumSMs + sm_idx
                                 : j * kNumSMs * kNumCombineWarps + sm_idx * kNumCombineWarps + w;
        };
        // dynamic sliced combine: the work list lives in the SFA stage area once that is dead: word 0 = item counter, words 1.. = one
        // entry per (warp, stripe): 0xffffffff unknown (owner not yet past its wait-combine), 0xfffffffe nothing left, else the token
        // index. Initialised below, behind a CTA-wide sync of the math warps
        constexpr uint32_t kNumWLEntries = kNumCombineWarps * kNumECTokenStripes;
        auto wl_words = reinterpret_cast<uint32_t*>(sf_start_ptr);
        // All math warps rendezvous before initialising this worklist below.
        // SFA/SFB stages are contiguous and retired there; weight SF/barriers
        // remain outside this alias. cap=16384 requires 868 B, vs 768 B of SFA.
        constexpr uint32_t kCombineWorkListBytes =
            kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
        DG_STATIC_ASSERT(kCombineSliced != 2 or (SMEM_SFA_SIZE_PER_STAGE > 0 and
                         (1u + kNumWLEntries) * 4u <= kCombineWorkListBytes),
                         "the sliced-combine work list needs the retired SFA/SFB stage area");
        // PDL: this warpgroup's last tile is published (L2 scatter issued, arrivals released); let the runtime schedule the
        // next grid. Its CTAs only get an SM once this CTA exits and may not touch memory before their own wait, so the
        // combine and the workspace cleanup below are never observed early.
        if constexpr (kPDL)
            cudaTriggerProgrammaticLaunchCompletion();

        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumCombineWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumCombineWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumCombineWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;

        // Combine of one token (also used by the early-combine wait): the valid slots' rows are TMA-loaded chunk by chunk into
        // the two staging stages, summed in fp32 in ascending slot order (ptx::accumulate) and stored once through smem + TMA.
        // `abort_check()` (warp-uniform) is consulted before every slot load; when it fires the loads in flight are drained,
        // nothing is stored and false is returned (the token is combined again later: bit-identical, same fixed-order sum).
        const auto combine_one_token = [&](const uint32_t& token_idx, const uint32_t& total_mask, const auto& abort_check) -> bool {
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };
                bool do_reduce = move_mask_and_load(load_stage_idx);
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    if (abort_check()) {
                        // drain the load in flight (stage load_stage_idx), keep the stage / parity bookkeeping consistent
                        combine_load_barriers[load_stage_idx]->wait(combine_phase);
                        combine_phase ^= load_stage_idx;
                        load_stage_idx ^= 1;
                        return false;
                    }
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
            return true;
        };

        uint32_t ec_wait_done = 0;   // early-combine wait: bit j = this warp's token of stripe j is already combined
        // ---------------- COMBINE ----------------
        // early combine: every math warp of the CTA is past its last epilogue -> the remaining L2 tiles of the table (the last one has
        // no successor tile for the B loader's signal) are complete
        if constexpr (kEarlyCombine) {
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            if (epilogue_thread_idx == 0)
                ptx::st_release_cta_shared(smem_ec_done, 1u);
        }
        if constexpr (kCombineSliced == 2) {
            // the SFA stage area is dead only once EVERY math warp is past its last mainloop (a warpgroup may trail the other by up
            // to kNumStages k-blocks, still reading stage SFA): the work list is initialised behind a CTA-wide sync of the math warps;
            // the barriers before the combine (arrive sync of the wait-combine, or the tag-2 barrier's own syncs) order these stores
            // before any warp's post / grab
            if constexpr (not kEarlyCombine)
                sync_tile_math();
            if (epilogue_warp_idx < kNumCombineWarps and lane_idx < kNumECTokenStripes)
                ptx::st_shared(wl_words + 1 + epilogue_warp_idx * kNumECTokenStripes + lane_idx, 0xffffffffu);
            if (epilogue_thread_idx == 0)
                ptx::st_shared(wl_words, 0u);
        }
        if constexpr (kEarlyCombineMathWait) {
            // early-combine wait: the same all-rank barrier as comm::nvlink_barrier (grid sync, SM0's sys signal, grid sync), but
            // while a CTA waits for a grid sync its math warps combine their own tokens whose top-k experts have all published
            // their done flag. A token in flight when the sync completes is dropped (nothing stored) and redone by the ordinary
            // combine below; tokens combined here are remembered per warp (one bit per stripe) and skipped below.
            static constexpr uint32_t kFinishSumTag = 0x80000000u;
            const auto count_ptr = workspace.template get_grid_sync_count_ptr<kEpilogueGridSyncIndex>();
            const auto grid_flipped = [&](const uint32_t& old_value) {
                uint32_t v = 0;
                if (lane_idx == 0)
                    v = ptx::ld_acq(count_ptr);
                v = __shfl_sync(0xffffffffu, v, 0);
                return ((v ^ old_value) & kFinishSumTag) != 0;
            };
            // this warp's tokens: stripe j <-> token combine_token_of(j, warp)
            const auto my_token = [&](const uint32_t& j) { return combine_token_of(j, epilogue_warp_idx); };
            uint32_t wc_pending = 0;   // bit j: stripe j still to do (and not combined by a dispatch receiver)
            #pragma unroll
            for (uint32_t j = 0; j < kNumECTokenStripes; ++ j) {
                bool p = my_token(j) < num_tokens;
                wc_pending |= p ? (1u << j) : 0u;
            }
            // token j: relaxed check of its experts' flags, then the acquire loads (8 lanes) once they are all set
            const auto token_flags_ready = [&](const uint32_t& token_idx, uint32_t& total_mask) {
                const int slot = lane_idx < kNumTopk ?
                    static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
                total_mask = __ballot_sync(0xffffffffu, slot >= 0);
                const auto flag_ptr = workspace.get_expert_done_flag_ptr(slot >= 0 ? static_cast<uint32_t>(slot) : 0u);
                const bool set = slot < 0 or ptx::ld_relaxed_sys(flag_ptr) != 0;
                if (not __all_sync(0xffffffffu, set))
                    return false;
                if (slot >= 0) {
                    const uint32_t v = ptx::ld_acq_sys(flag_ptr);
                    asm volatile("" :: "r"(v));
                }
                __syncwarp();
                return true;
            };
            uint32_t num_wait_combined = 0;
            {
                // arrive (release covers every warp's scatter stores: bar.sync first), then warp 0 runs the whole barrier
                // protocol while warps 1.. combine their ready tokens; thread 0 publishes the completion in smem
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0) {
                    uint32_t old_value = 0;
                    if (lane_idx == 0)
                        old_value = ptx::atomic_add_rel(count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
                    old_value = __shfl_sync(0xffffffffu, old_value, 0);
                    while (not grid_flipped(old_value));
                    if (sm_idx == 0) {
                        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
                        const auto status = (*counter_ptr) & 3;
                        const auto signal_phase = status & 1, signal_sign = status >> 1;
                        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);
                        if (lane_idx < kNumRanks)
                            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, lane_idx), signal_sign ? -1 : 1);
                        __syncwarp();
                        if (lane_idx == 0) {
                            ptx::red_add(counter_ptr, 1);
                            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
                            const auto start_clock = clock64();
                            // (the timeout trap sits after the loop: a trap inside a loop body of the math region makes
                            // ptxas cap the region at the 168-register launch bound)
                            bool timed_out = false;
                            while (ptx::ld_acq_sys(signal_ptr) != target) {
                                if (clock64() - start_clock >= comm::kNumTimeoutCycles) {
                                    timed_out = true;
                                    break;
                                }
                            }
                            DG_TRAP_ONLY_DEVICE_ASSERT(not timed_out);
                        }
                        __syncwarp();
                    }
                    if (lane_idx == 0)
                        old_value = ptx::atomic_add_rel(count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
                    old_value = __shfl_sync(0xffffffffu, old_value, 0);
                    while (not grid_flipped(old_value));
                    // every rank's scatter is complete and visible: release the other warps and the dispatch warps
                    if (lane_idx == 0)
                        ptx::st_release_cta_shared(smem_ec_stop, 1u);
                } else {
                    const auto never = [&]() { return false; };
                    while (ptx::ld_acquire_cta_shared(smem_ec_stop) == 0) {
                        bool progressed = false;
                        uint32_t bits = wc_pending;
                        while (bits != 0) {
                            const uint32_t j = __ffs(bits) - 1;
                            bits &= bits - 1;
                            const uint32_t token_idx = my_token(j);
                            uint32_t total_mask;
                            if (not token_flags_ready(token_idx, total_mask))
                                continue;
                            combine_one_token(token_idx, total_mask, never);
                            wc_pending &= ~(1u << j);
                            ++ num_wait_combined;
                            progressed = true;
                        }
                        if (not progressed)
                            __nanosleep(500);
                    }
                }
                trace_math(25, num_wait_combined | (static_cast<uint64_t>(__popc(wc_pending)) << 16));   // WAIT_COMBINE_DONE
                ec_wait_done = ~wc_pending;
            }
        } else {
            // NVLink barrier first: signals remote ranks that this rank's GEMM
            // outputs (NVLink scatter targets) are fully written.
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag,
                                 comm::BarrierTimeoutPolicy::TrapOnly>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
            );
        }

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state. (early combine: the dispatch warps wait on the stop flag instead)
        if constexpr (not kEarlyCombine)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        trace_math(20, 0);   // COMBINE_START (all-rank barrier passed; combine runs on the math warps)

        if (epilogue_warp_idx >= kNumCombineWarps)
            return;

        DG_TRAP_ONLY_DEVICE_ASSERT(kNumChunkSlots * kNumCombineWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        uint32_t num_combined = 0, num_ec_skipped = 0;   // trace (COMBINE_END aux)
        if constexpr (kCombineSliced == 2) {
            // dynamic sliced combine (see the template parameter). This warp is free: post its leftover tokens, then take items.
            constexpr uint32_t kSliceBytes = kCombineMXFP4 ? kMXFP4CombineSliceBytes : kNumHiddenBytes / kNumCombineWarps;
            constexpr uint32_t kNumSlices = kNumHiddenBytes / kSliceBytes;
            constexpr uint32_t kSliceUint4PerLane = kSliceBytes / sizeof(uint4) / 32;
            DG_STATIC_ASSERT((kCombineMXFP4 or kNumChunks == 1) and kNumHiddenBytes % kSliceBytes == 0 and kSliceBytes % (32 * sizeof(uint4)) == 0,
                             "dynamic sliced combine: hidden slices of a multiple of 512 bytes");
            DG_STATIC_ASSERT(kNumTopk * kSliceBytes <= kNumChunkBytes, "the slot slices of one item fit the warp's stage-0 buffer");
            const auto wl_entries = wl_words + 1;
            if (lane_idx < kNumECTokenStripes) {
                const uint32_t token_idx = combine_token_of(lane_idx, epilogue_warp_idx);
                bool pending = token_idx < num_tokens;
                if constexpr (kEarlyCombine) {
                    pending = pending and not ((ec_wait_done >> lane_idx) & 1u);
                    if (not pending and token_idx < num_tokens)
                        ++ num_ec_skipped;
                }
                ptx::st_shared(wl_entries + epilogue_warp_idx * kNumECTokenStripes + lane_idx, pending ? token_idx : 0xfffffffeu);
            }
            __syncwarp();
            const auto sl_load_buffer = [&](const uint32_t& s) { return math::advance_ptr<uint4>(combine_load_buffer[0], s * kSliceBytes); };
            const auto sl_store_buffer = combine_store_buffer;
            constexpr uint32_t kNumWLItems = kNumWLEntries * kNumSlices;   // (warp, stripe) x slice
            // MXFP4: walk only the stripes this call's tokens reach, stripe-major (item = (stripe * warps + warp) * warps + slice), instead
            // of the capacity's kNumECTokenStripes: on a cap-8192 buffer the capacity walk is 14 x 64 contended shared-memory atomics
            // per CTA for a handful of tokens (Flash M8: combine 14.5 vs 1.5 us, notes §29). Items are independent (token, slice)
            // reductions, so the order does not change the output. FP8 keeps the capacity walk (default path unchanged)
            // SM-major ownership: this CTA's tokens are sm + kNumSMs * k for k = stripe * warps + warp, so its valid entries are exactly the
            // first ceil((num_tokens - sm) / kNumSMs) in item order (a CTA without tokens takes no item)
            constexpr bool kWLBounded = kMXFP4Weights;
            const uint32_t num_wl_items = kCombineMXFP4 ?
                (sm_idx < num_tokens ? math::ceil_div(num_tokens - sm_idx, kNumSMs) : 0u) * kNumSlices :
                (kWLBounded ? math::ceil_div(num_tokens, kNumSMs * kNumCombineWarps) * (kNumCombineWarps * kNumCombineWarps) : kNumWLItems);
            #pragma unroll 1
            while (true) {
                uint32_t idx = 0;
                if (lane_idx == 0)
                    idx = atomicAdd(wl_words, 1u);
                idx = __shfl_sync(0xffffffffu, idx, 0);
                if (idx >= num_wl_items)
                    break;
                const uint32_t slice = idx % kNumSlices;
                const uint32_t entry = kWLBounded ?
                    ((idx / kNumSlices) % kNumCombineWarps) * kNumECTokenStripes + (idx / kNumSlices) / kNumCombineWarps :
                    idx / kNumSlices;
                uint32_t token_idx;
                while ((token_idx = ptx::ld_volatile_shared(wl_entries + entry)) == 0xffffffffu)
                    __nanosleep(64);
                if (token_idx == 0xfffffffeu)
                    continue;
                ++ num_combined;
                const int slot_of_lane = lane_idx < kNumTopk ?
                    static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
                const uint32_t mask = __ballot_sync(0xffffffffu, slot_of_lane >= 0);
                float2 reduced[kSliceUint4PerLane * kNumElemsPerUint4] = {};
                if (mask) {
                    const uint32_t num_slots = __popc(mask);
                    if (cute::elect_one_sync()) {
                        uint32_t issue_mask = mask;
                        uint32_t s = 0;
                        while (issue_mask) {
                            const uint32_t slot_idx = __ffs(issue_mask) - 1;
                            issue_mask ^= 1u << slot_idx;
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx).get_data_buffer(token_idx).get_base_ptr(), slice * kSliceBytes);
                            ptx::tma_load_1d(sl_load_buffer(s), src_ptr, combine_load_barriers[load_stage_idx], kSliceBytes);
                            ++ s;
                        }
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[load_stage_idx], num_slots * kSliceBytes);
                    }
                    __syncwarp();
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll 1
                    for (uint32_t s = 0; s < num_slots; ++ s) {
                        #pragma unroll
                        for (uint32_t j = 0; j < kSliceUint4PerLane; ++ j) {
                            const auto uint4_values = sl_load_buffer(s)[j * 32 + lane_idx];
                            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                                ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                        }
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }
                #pragma unroll
                for (uint32_t j = 0; j < kSliceUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(sl_store_buffer + j * 32 + lane_idx, casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + slice * kSliceBytes),
                                      sl_store_buffer, kSliceBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        } else if constexpr (kCombineSliced == 1) {
            // static sliced combine: the CTA's remaining tokens, one at a time by all warps; warp w reduces and stores hidden slice w. Layout in the
            // two whole-row load stages' region (free once every warp is past the barrier below; the in-flight stores read the
            // store buffers behind it): warp w = kNumTopk staging slices + its store slice at w x (kNumTopk + 1) slices.
            constexpr uint32_t kSliceBytes = kNumHiddenBytes / kNumCombineWarps;
            constexpr uint32_t kSliceUint4PerLane = kSliceBytes / sizeof(uint4) / 32;
            DG_STATIC_ASSERT(kNumChunks == 1 and kNumHiddenBytes % kNumCombineWarps == 0 and kSliceBytes % (32 * sizeof(uint4)) == 0,
                             "static sliced combine: whole-row slots, hidden slices of a multiple of 512 bytes per warp");
            DG_STATIC_ASSERT(kNumCombineWarps * (kNumTopk + 1) * kSliceBytes <= 2 * kNumCombineWarps * kNumChunkBytes,
                             "the sliced staging must fit the whole-row load stages' region");
            const auto sl_load_buffer = [&](const uint32_t& s) {
                return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx * (kNumTopk + 1) + s) * kSliceBytes);
            };
            const auto sl_store_buffer = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx * (kNumTopk + 1) + kNumTopk) * kSliceBytes);
            // tokens already combined during the wait (per-warp ec_wait_done) -> bit (stripe x warps + warp) of the CTA-wide bitmap
            // (the claim words of the early-combine control block), so every warp walks the same remaining set
            if constexpr (kEarlyCombine) {
                DG_STATIC_ASSERT(kNumECTokenStripes * kNumCombineWarps <= 8u * 32u,
                                 "the done bitmap needs stripes x warps <= 256 bits (words 8..15 of the early-combine control block)");
                if (lane_idx < kNumECTokenStripes and ((ec_wait_done >> lane_idx) & 1u)) {
                    const uint32_t bit = lane_idx * kNumCombineWarps + epilogue_warp_idx;
                    atomicOr(smem_ec_claim + (bit >> 5), 1u << (bit & 31u));
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
            #pragma unroll 1
            for (uint32_t token_base = sm_idx * kNumCombineWarps, stripe = 0; token_base < num_tokens;
                 token_base += kNumSMs * kNumCombineWarps, ++ stripe) {
                #pragma unroll 1
                for (uint32_t w = 0; w < kNumCombineWarps; ++ w) {
                    const uint32_t token_idx = token_base + w;
                    if (token_idx >= num_tokens)
                        break;
                    if constexpr (kEarlyCombine) {
                        const uint32_t bit = stripe * kNumCombineWarps + w;
                        if ((ptx::ld_shared(smem_ec_claim + (bit >> 5)) >> (bit & 31u)) & 1u) {
                            ++ num_ec_skipped;
                            continue;
                        }
                    }
                    if constexpr (kTrace)
                        ++ num_combined;
                    const int slot_of_lane = lane_idx < kNumTopk ?
                        static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
                    uint32_t mask = __ballot_sync(0xffffffffu, slot_of_lane >= 0);
                    float2 reduced[kSliceUint4PerLane * kNumElemsPerUint4] = {};
                    if (mask) {
                        const uint32_t num_slots = __popc(mask);
                        // all valid slots' slices in flight together, one arrive with the total byte count
                        if (cute::elect_one_sync()) {
                            uint32_t issue_mask = mask;
                            uint32_t s = 0;
                            while (issue_mask) {
                                const uint32_t slot_idx = __ffs(issue_mask) - 1;
                                issue_mask ^= 1u << slot_idx;
                                const auto src_ptr = math::advance_ptr<uint8_t>(
                                    combine_token_buffer.get_rank_buffer(slot_idx).get_data_buffer(token_idx).get_base_ptr(),
                                    epilogue_warp_idx * kSliceBytes);
                                ptx::tma_load_1d(sl_load_buffer(s), src_ptr, combine_load_barriers[load_stage_idx], kSliceBytes);
                                ++ s;
                            }
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[load_stage_idx], num_slots * kSliceBytes);
                        }
                        __syncwarp();
                        combine_load_barriers[load_stage_idx]->wait(combine_phase);
                        // ascending slot order per element, as the whole-row combine
                        #pragma unroll 1
                        for (uint32_t s = 0; s < num_slots; ++ s) {
                            #pragma unroll
                            for (uint32_t j = 0; j < kSliceUint4PerLane; ++ j) {
                                const auto uint4_values = sl_load_buffer(s)[j * 32 + lane_idx];
                                const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                                #pragma unroll
                                for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                                    ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                            }
                        }
                        // one wait on the current stage's barrier: same bookkeeping as the whole-row loop
                        combine_phase ^= load_stage_idx;
                        load_stage_idx ^= 1;
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < kSliceUint4PerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        ptx::st_shared(sl_store_buffer + j * 32 + lane_idx, casted.x, casted.y, casted.z, casted.w);
                    }
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        ptx::tma_store_1d(
                            math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + epilogue_warp_idx * kSliceBytes),
                            sl_store_buffer, kSliceBytes);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }
            }
        } else
        for (uint32_t token_idx = sm_idx * kNumCombineWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumCombineWarps) {
            // early combine: this warp combined the token while waiting in the barrier
            if constexpr (kEarlyCombine) {
                const uint32_t stripe = token_idx / (kNumSMs * kNumCombineWarps);
                if ((ec_wait_done >> stripe) & 1u) {
                    ++ num_ec_skipped;
                    continue;
                }
            }
            if constexpr (kTrace)
                ++ num_combined;
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        trace_math(21, num_combined | (static_cast<uint64_t>(num_ec_skipped) << 16));   // COMBINE_END (aux: tokens combined | skipped (early combine) << 16)
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
