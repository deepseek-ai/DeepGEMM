#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
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
__forceinline__ __device__ float sm90_fp8_fused_mega_moe_silu_ftz_exp(float x) {
    float e;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(__fmul_rn(-x, 1.4426950408889634f)));
    return x * math::fast_rcp(1.0f + e);
}

// Continuous FP32 activation scale. SM90 WGMMA has no hardware block-scale operand (the SF
// is a plain FFMA in the epilogue), so the previous UE8M0 (power-of-two) scale bought nothing
// on SM90 and only cost precision; the SF pool is already fp32, so this is byte/layout neutral.
// clamp amax before the reciprocal: padded rows have amax==0, and 448/0=inf -> 0*inf=NaN.
__forceinline__ __device__ void sm90_fp8_fused_mega_moe_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    const auto ax = fmaxf(amax.x, 1e-10f);
    const auto ay = fmaxf(amax.y, 1e-10f);
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
                       uint64_t* trace) {
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
        kNumMaxPoolTokens, kRingCoversFullPool ? 0 : kNumRingTokens);

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
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // SFA per stage: BLOCK_M floats for L1 and for the per-128 L2 act SF (one (BLOCK_M, 1) TMA per k-block), 2*BLOCK_M
    // floats (lo/hi halves) with the per-64 L2 act SF
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>((kL2ActSFPerBlockK ? 1u : 2u) * BLOCK_M * sizeof(float), 128u);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = 0;
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
        SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
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

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });

    // Per-warpgroup weight-SF staging slots (see SMEM_WEIGHT_SF_SIZE above)
    auto smem_weight_sf = reinterpret_cast<float*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);

    // Barriers live after the weight-SF staging area
    // A (+ per-group SFA) is multicast only when the pairing is on m
    // (n-inner order); with B multicast the pair's A tiles differ.
    constexpr uint32_t kNumMulticastA = kMulticastOnB ? 1u : kClusterSize;

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + SMEM_WEIGHT_SF_SIZE);
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
    constexpr uint32_t kNumTileTableEntries = layout::get_sm90_tile_table_entries_compact<uint32_t>(
        kNumMaxPoolTokens, BLOCK_M, L1_SHAPE_N / BLOCK_N, L2_SHAPE_N / BLOCK_N, kNumSMs);
    constexpr uint32_t SMEM_TILE_TABLE_OFFSET = math::constexpr_align<uint32_t>(
        SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE) +
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
        kNumMaxPoolTokens / BLOCK_M, kNumRanks * kNumMaxTokensPerRank, kClusterSize>(workspace);

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
                (kNumEpilogueThreads == 256 ? 168 :
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
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
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
                    const auto l1_empty_target = get_ring_wave_idx(pool_block_idx) * kNumL1BlockNs;
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
                    const uint32_t expected = kNumL1BlockNs * (get_ring_wave_idx(pool_block_idx) + 1);
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
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx = ring_block_idx * BLOCK_M;
                    const uint32_t sfa_m_idx = ring_block_idx * SF_BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, kNumMulticastA);

                    // TMA load SFA
                    if (kL2ActSFPerBlockK or block_phase == sched::BlockPhase::Linear1) {
                        // L1 SFA per-128 (and the per-128 L2 SFA): load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            sfa_m_idx, k_block_idx, kNumMulticastA);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                    } else {
                        // L2 SFA per-64: descriptor box is (block_mn, 1) (see make_tma_sf_desc),
                        // so we must issue two single-group TMAs and place them at smem offsets
                        // 0 and BLOCK_M to match math's load offsets (`+ 0 * BLOCK_M` / `+ 1 * BLOCK_M`).
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            sfa_m_idx, k_block_idx * 2, kNumMulticastA);
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx] + BLOCK_M,
                            sfa_m_idx, k_block_idx * 2 + 1, kNumMulticastA);
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
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load B (weight SF is now loaded directly by math warps from global)
                    if constexpr (LOAD_BLOCK_N <= 256) {
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

                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
            if constexpr (kEarlyCombine) {
                DG_STATIC_ASSERT(kNumL2BlockKs > kNumStages and kNumL1BlockKs > kNumStages, "the early-combine signal needs k-block kNumStages of the next tile");
                ec_prev_l2 = block_phase == sched::BlockPhase::Linear2;
            }
        };
        scheduler.template for_each_block_publish_dynamic_tail<kClusterSize>(
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
        constexpr bool kSFProloguePrefetch = kDecodeTopology and kHidden / 128 <= 32 and 2 * (kIntermediateHidden / 128) <= 32;
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

        auto process_math_block = [&](const auto& block_phase,
                                      const uint32_t& local_expert_idx,
                                      const uint32_t& num_k_blocks,
                                      const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
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

            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

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
            if (kSFProloguePrefetch and sf_pf_hit) {
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
            const auto desc_b_stage0 = mma::sm90::make_smem_desc(smem_b[0], 1);
            auto smem_desc_a = [&](const uint32_t& stage, const uint32_t& byte_offset) {
                return mma::sm90::advance_smem_desc(desc_a_stage0, stage * SMEM_A_SIZE_PER_STAGE + byte_offset);
            };
            auto smem_desc_b = [&](const uint32_t& stage, const uint32_t& byte_offset) {
                return mma::sm90::advance_smem_desc(desc_b_stage0, stage * SMEM_B_SIZE_PER_STAGE + byte_offset);
            };

            float final_accum[kAccumPerThread] = {};

            if constexpr (kReuseAccumAsFinal) {
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
            } else {
                DG_STATIC_ASSERT(kNumSFGroupsPerWG == 1, "The promotion path assumes one weight-SF block per warpgroup tile");
                float accum[kAccumPerThread];

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
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
                }

            }
            trace_math(43, 0);   // MAINLOOP_END (last warpgroup_wait<0> of the tile done)
            // prologue prefetch: peek after the mainloop; the epilogue hides the load
            if constexpr (kSFProloguePrefetch) {
                if (warp_idx_in_wg == 0 and sf_pf_key == 0)
                    sf_peek();
            }

            // Skip epilogue when block is past valid M (still must release via empty)
            if (row_base >= valid_m) {
                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (kL2ArrivalNeedsFullSync)
                        sync_tile_math();
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
                        sm90_fp8_fused_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
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
                sm90_fp8_fused_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
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
                // In the shared-tile split only the first warpgroup issues and drains the combined TMA store; gate the other
                // warpgroup so it cannot overwrite the joint smem tile in the next block until that store has drained.
                if constexpr (kSplitNSharesSF)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
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

                if constexpr (kL2EpilogueRequiresFullSync) {
                    sync_tile_math();
                    trace_epi(62, 0);   // EPI_FULL_SYNC_DONE
                }
            }
            trace_math(44, 0);   // EPILOGUE_END
        };

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
        trace_math(19, trace_tile);   // TILES_DONE (aux = number of tiles this CTA ran)
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
        constexpr uint32_t kCombineSliced =
            (kDecodeTopology and kNumChunks == 1 and kHidden % (kNumCombineWarps * 256) == 0 and kCombineSlicedFits) ? kCombineSlicedForm : 0u;
        // dynamic sliced combine: the work list lives in the SFA stage area once that is dead: word 0 = item counter, words 1.. = one
        // entry per (warp, stripe): 0xffffffff unknown (owner not yet past its wait-combine), 0xfffffffe nothing left, else the token
        // index. Initialised below, behind a CTA-wide sync of the math warps
        constexpr uint32_t kNumWLEntries = kNumCombineWarps * kNumECTokenStripes;
        auto wl_words = reinterpret_cast<uint32_t*>(sf_start_ptr);
        DG_STATIC_ASSERT(kCombineSliced != 2 or (SMEM_SFA_SIZE_PER_STAGE > 0 and (1u + kNumWLEntries) * 4u <= kNumStages * SMEM_SFA_SIZE_PER_STAGE),
                         "the sliced-combine work list needs the block-scaled SFA stage area");
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
            // this warp's tokens: stripe j <-> token j * kNumSMs * kNumCombineWarps + sm_idx * kNumCombineWarps + warp
            const auto my_token = [&](const uint32_t& j) { return j * kNumSMs * kNumCombineWarps + sm_idx * kNumCombineWarps + epilogue_warp_idx; };
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
            constexpr uint32_t kSliceBytes = kNumHiddenBytes / kNumCombineWarps;
            constexpr uint32_t kSliceUint4PerLane = kSliceBytes / sizeof(uint4) / 32;
            DG_STATIC_ASSERT(kNumChunks == 1 and kNumHiddenBytes % kNumCombineWarps == 0 and kSliceBytes % (32 * sizeof(uint4)) == 0,
                             "dynamic sliced combine: whole-row slots, hidden slices of a multiple of 512 bytes per warp");
            DG_STATIC_ASSERT(kNumTopk * kSliceBytes <= kNumChunkBytes, "the slot slices of one item fit the warp's stage-0 buffer");
            const auto wl_entries = wl_words + 1;
            if (lane_idx < kNumECTokenStripes) {
                const uint32_t token_idx = lane_idx * kNumSMs * kNumCombineWarps + sm_idx * kNumCombineWarps + epilogue_warp_idx;
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
            constexpr uint32_t kNumWLItems = kNumWLEntries * kNumCombineWarps;   // (warp, stripe) x slice
            #pragma unroll 1
            while (true) {
                uint32_t idx = 0;
                if (lane_idx == 0)
                    idx = atomicAdd(wl_words, 1u);
                idx = __shfl_sync(0xffffffffu, idx, 0);
                if (idx >= kNumWLItems)
                    break;
                const uint32_t slice = idx % kNumCombineWarps, entry = idx / kNumCombineWarps;
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
