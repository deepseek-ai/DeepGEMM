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
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#define __CLION_IDE__

namespace deep_gemm {

template <float kActivationClamp>
__forceinline__ __device__ float sm90_fp8_mega_moe_clamp_gate(float x) {
    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
        x = cute::min(x, kActivationClamp);
    return x;
}

template <float kActivationClamp>
__forceinline__ __device__ float sm90_fp8_mega_moe_clamp_up(float x) {
    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
        x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
    return x;
}

template <bool kFastMath>
__forceinline__ __device__ float sm90_fp8_mega_moe_silu(float x) {
    const float e = kFastMath ? __expf(-x) : expf(-x);
    const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
    return x * sig;
}

template <bool kFastMath, float kActivationClamp>
__forceinline__ __device__ float sm90_fp8_mega_moe_swiglu(float g, float u) {
    g = sm90_fp8_mega_moe_clamp_gate<kActivationClamp>(g);
    u = sm90_fp8_mega_moe_clamp_up<kActivationClamp>(u);
    return sm90_fp8_mega_moe_silu<kFastMath>(g) * u;
}

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE — full implementation
// ----------------------------------------------------------------------------
// Pipeline (cluster=1, no TMA multicast):
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * GEMM TMA-load warps (1 for A+SFA, 1 for B+SFB) feed the pipeline stages.
//   * Math warpgroups (1 or 2, totalling kNumEpilogueThreads) consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout,
//         per-row amax over the 64 post-SwiGLU columns of this block, FP8 e4m3
//         quantize, STSM into SMEM, TMA store to local L1 output buffer.
//         The per-row SF is written as a *float* into the L2-acts SF buffer at
//         per-64 K granularity (one SF per L1 N block), so each block is fully
//         self-contained and no cross-CTA amax synchronisation is needed.
//       - L2 (Linear2): BF16 cast of the GEMM output, STSM into SMEM, then
//         NVLink scatter to remote combine buffers.
//   * After all GEMM blocks, the math warps run the COMBINE step (top-k
//     reduction in BF16) — ported verbatim from the SM100 kernel.
// ============================================================================

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    uint32_t kEpilogueRegisterBudget,
    bool kReuseAccumAsFinal,
    bool kL2ArrivalCounter,
    bool kSkipL2EpilogueSync,
    bool kSplitPhaseHotPath,
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
sm90_fp8_mega_moe_impl(void* y,
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
                       const float* __restrict__ l2_weights_sf) {
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

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    constexpr bool kGemmOnly = false;
    constexpr bool kGemmOnlySkipL2OutputStore = false;
    constexpr bool kGemmOnlyFixedL1Scale = false;
    constexpr bool kGemmOnlySkipL1OutputStore = false;
    constexpr bool kGemmOnlySkipL1Epilogue = false;
    constexpr bool kGemmOnlySkipL2ActLoad = false;
    constexpr bool kGemmOnlySkipBLoad = false;
    constexpr bool kGemmOnlySkipL2ArrivalWait = false;
    constexpr bool kGemmOnlySkipL1TopkWeight = false;
    constexpr bool kGemmOnlySkipL1FP8Pack = false;
    constexpr bool kGemmOnlySkipL1SmemStore = false;
    constexpr bool kGemmOnlyDirectL1GlobalStore = false;
    constexpr bool kGemmOnlyDirectL1GlobalNoFence = false;
    constexpr bool kSkipCombine = false;
    constexpr bool kSkipL2Scatter = false;
    constexpr bool kDirectL2RegisterScatter = false;
    constexpr bool kDispatchOnly = false;
    constexpr bool kDispatchOnlySkipPull = false;
    constexpr uint32_t kMBlockInterleaveGroup = 0;
    constexpr bool kStreamSwigluQuant = false;
    constexpr bool kInplaceSwiglu = false;
    constexpr bool kOverlapKScale = false;
    constexpr bool kDispatchDirectTokenCopy = false;
    constexpr bool kDispatchWarpTokenStore = false;
    constexpr bool kDispatchTokenCopyFence = false;
    constexpr bool kDispatchSkipTokenCopy = false;
    constexpr bool kDispatchSkipSFCopy = false;
    constexpr bool kDispatchSkipTopkWeight = false;
    constexpr bool kSkipWorkspaceCleanup = false;
    constexpr bool kL1StsmX4 = false;
    constexpr bool kL1StsmX2 = false;
    constexpr bool kPackedL1SmemStore = false;
    constexpr bool kL1U64SmemStore = false;
    constexpr bool kL1WarpTMAStore = false;
    constexpr bool kL1TMAStoreWaitIssuerOnly = false;
    constexpr bool kSplitSFALoaderWarp = false;
    constexpr bool kNMajorSchedule = false;

    // Prefetch all TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // Per-64 K float SF (SM90 only): 4 bytes per per-64 group => `kIntermediateHidden / 16` bytes/token
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    constexpr bool kSplitNWarpgroups =
        BLOCK_M == 64 and BLOCK_N % 128 == 0 and
        kNumEpilogueWarpgroups == BLOCK_N / 128 and
        kNumEpilogueWarpgroups > 1;
    constexpr bool kSplitMNWarpgroups =
        BLOCK_M == 128 and BLOCK_N == 256 and kNumEpilogueWarpgroups == 4;
    constexpr uint32_t kWarpgroupSplitM = kSplitNWarpgroups ? 1 :
        (kSplitMNWarpgroups ? 2 : kNumEpilogueWarpgroups);
    constexpr uint32_t kWarpgroupSplitN = kSplitNWarpgroups ? kNumEpilogueWarpgroups :
        (kSplitMNWarpgroups ? 2 : 1);
    constexpr uint32_t WG_BLOCK_M = BLOCK_M / kWarpgroupSplitM;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kWarpgroupSplitN;
    constexpr uint32_t kNumCombineWarps = kNumEpilogueWarps;
    using L1WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;  // M=64, N=128, K=32
    using L2WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    constexpr uint32_t kL1WarpTMAStoreRows = 16;
    constexpr uint32_t kL1OutputArrivalParts =
        kL1WarpTMAStore ? (WG_BLOCK_M / kL1WarpTMAStoreRows) : 1;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups,
                     "Invalid warpgroup split");
    DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M,
                     "Each warpgroup must run exactly one WGMMA-M tile");
    DG_STATIC_ASSERT(not kL1WarpTMAStore or (kL2ArrivalCounter and WG_BLOCK_M % kL1WarpTMAStoreRows == 0),
                     "Warp-sliced L1 TMA store requires counter mode and 16-row warp slices");
    DG_STATIC_ASSERT(kNumCombineWarps <= kNumEpilogueWarps,
                     "Combine warp count must fit in epilogue warps");

    // Cluster=1 -> no multicast, A/B are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N  = BLOCK_N / 2;  // post-SwiGLU
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kSwizzleBMode   = BLOCK_K * sizeof(b_dtype_t);   // 128
    constexpr uint32_t kSwizzleCDMode  = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = 64;           // L2 acts SF (per-64 K, SM90 only)

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and L2 (2*BLOCK_M floats per-64).
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(2 * BLOCK_M * sizeof(float), 128u);
    // Block (128, 128) weight SF: 1 float per (BLOCK_N, BLOCK_K) tile for L2,
    // 2 floats (gate/up) for L1. Loaded by math warpgroup directly from global,
    // so no SMEM is needed.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = 0;

    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes). Each math WG writes a disjoint
    // WG_BLOCK_M slice, so the total rows are BLOCK_M, not BLOCK_M * num_wg.
    constexpr uint32_t SMEM_CD_L1_SIZE = BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE    = math::constexpr_align(
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE, kSharedMemoryAlignment);

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_gemm_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_gemm_base);

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

    // Barriers live after SF (SFB is loaded directly from global, no SMEM)
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });

    // =====================================================================
    // Initialization
    // =====================================================================
    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Two producer warps (A+SFA loader, B+SFB loader) each call
                // `arrive_and_expect_tx` per stage, so init count must be 2.
                full_barriers[i]->init(kSplitSFALoaderWarp ? 3 : 2);
                // Each math warp arrives once per stage release.
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumCombineWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if constexpr (kGemmOnly) {
        // Synthetic local-only setup for GEMM utilization experiments.  Each
        // local expert receives an equal share of this rank's routed tokens:
        // num_tokens * topk / experts_per_rank.  The Python harness pre-fills
        // L1 activation buffer; this block publishes scheduler metadata and
        // L1 readiness only. L2 readiness is released after the matching L1
        // output has been materialized, preserving the L1-then-L2 ordering.
        const uint32_t tokens_per_expert = num_tokens * kNumTopk / kNumExpertsPerRank;
        const uint32_t num_m_blocks_per_expert = math::ceil_div(tokens_per_expert, BLOCK_M);
        if (thread_idx < kNumExpertsPerRank) {
            const uint64_t finalized =
                (static_cast<uint64_t>(kNumSMs) * kNumRanks << 32) |
                static_cast<uint64_t>(tokens_per_expert);
            *workspace.get_expert_recv_count_sum_ptr(thread_idx) = finalized;
            if (cumulative_local_expert_recv_stats != nullptr)
                cumulative_local_expert_recv_stats[thread_idx] = static_cast<int>(tokens_per_expert);
        }
        const uint32_t num_pool_blocks = kNumExpertsPerRank * num_m_blocks_per_expert;
        for (uint32_t i = thread_idx; i < num_pool_blocks; i += kNumThreads) {
            const uint32_t block_in_expert = i % num_m_blocks_per_expert;
            const uint32_t valid_m = cute::min(tokens_per_expert - block_in_expert * BLOCK_M, BLOCK_M);
            *workspace.get_l1_arrival_count_ptr(i) = valid_m;
            *workspace.get_l2_arrival_mask_ptr(i) = 0;
        }
        const uint32_t num_pool_tokens = kNumExpertsPerRank * tokens_per_expert;
        for (uint32_t i = thread_idx; i < num_pool_tokens; i += kNumThreads)
            *l1_topk_weights_buffer.get_data_buffer(i).get_base_ptr<float>() = 1.0f;
        __syncthreads();

        if (warp_idx < kNumDispatchWarps)
            return;
    }

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, /*kClusterSize=*/1u, kNMajorSchedule, kMBlockInterleaveGroup>(workspace);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // For the 256-epilogue-thread case (block_m=128, 2 math WGs):
    //   128*48 + 128*40 + 256*208 = 64512 exactly.
    // For the experimental 512-epilogue-thread 2D split path, trim dispatch
    // and loader roles so launch bounds still leave enough WGMMA registers.
    constexpr uint32_t kNumEpilogueRegisters    =
        kEpilogueRegisterBudget == 0 ?
            (kNumEpilogueThreads == 512 ? 112 : 208) :
            kEpilogueRegisterBudget;
    // The 512-epilogue-thread path has only 3584 registers of headroom at
    // epilogue=112.  Raising epilogue to 120 is not viable without changing
    // the role topology: dispatch=24 stalls the split-MN path, while
    // non-epilogue=16 is below ptxas' setmaxnreg.dec legal minimum.
    constexpr uint32_t kNumDispatchRegisters =
        kNumEpilogueThreads == 512 ? 32 : 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        kNumEpilogueThreads == 512 ? 24 : 40;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        if constexpr (not kGemmOnly) {
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

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);

        // Sync with epilogue warps before pulling tokens. Dispatch-only
        // diagnostics keep epilogue warps out of the path to avoid measuring a
        // synthetic CTA barrier pattern that the real fused kernel never uses.
        if constexpr (not kDispatchOnly)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        if constexpr (not (kDispatchOnly and kDispatchOnlySkipPull)) {
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

                // Round-robin rank selection (identical to SM100)
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

                // Pull token data. The default path overlaps a remote TMA load
                // with SF copy and then uses TMA store to materialize the local
                // L1 input. Experimental paths below keep the same metadata and
                // arrival ordering but replace the token payload copy.
                if constexpr (not kDispatchDirectTokenCopy and not kDispatchSkipTokenCopy) {
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                }
                __syncwarp();

                if constexpr (not kDispatchSkipSFCopy) {
                    // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
                    constexpr uint32_t kNumSFFloats = kHidden / 128;
                    DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                    const auto remote_sf_ptr = sym_buffer.map(
                        input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                        current_rank_in_expert_idx);
                    const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                    const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                    #pragma unroll
                    for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        if (j < kNumSFFloats)
                            local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
                    }
                }
                __syncwarp();

                if constexpr (not kDispatchSkipTopkWeight) {
                if (cute::elect_one_sync()) {
                    const auto weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
                }
                }
                __syncwarp();

                if constexpr (kDispatchSkipTokenCopy) {
                    if (cute::elect_one_sync()) {
                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                        ptx::red_add_rel(
                            workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                    }
                } else if constexpr (kDispatchDirectTokenCopy) {
                    DG_STATIC_ASSERT(kHidden % sizeof(uint4) == 0, "Direct token copy requires uint4 alignment");
                    const auto src_token_ptr = reinterpret_cast<const uint4*>(
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx));
                    const auto dst_token_ptr = reinterpret_cast<uint4*>(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr());
                    #pragma unroll
                    for (uint32_t i = lane_idx; i < kHidden / sizeof(uint4); i += 32)
                        dst_token_ptr[i] = src_token_ptr[i];
                    if constexpr (kDispatchTokenCopyFence)
                        __threadfence();
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                        ptx::red_add_rel(
                            workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                    }
                } else if constexpr (kDispatchWarpTokenStore) {
                    DG_STATIC_ASSERT(kHidden % sizeof(uint4) == 0, "Warp token store requires uint4 alignment");
                    if (cute::elect_one_sync())
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    __syncwarp();
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                    const auto src_token_ptr = reinterpret_cast<const uint4*>(pull_buffer.get_base_ptr());
                    const auto dst_token_ptr = reinterpret_cast<uint4*>(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr());
                    #pragma unroll
                    for (uint32_t i = lane_idx; i < kHidden / sizeof(uint4); i += 32)
                        dst_token_ptr[i] = src_token_ptr[i];
                    if constexpr (kDispatchTokenCopyFence)
                        __threadfence();
                    __syncwarp();
                    if (cute::elect_one_sync()) {
                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
                        ptx::red_add_rel(
                            workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                    }
                } else {
                    if (cute::elect_one_sync()) {
                        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                        ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                        ptx::tma_store_1d(
                            l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                            pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                        *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                            {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();
                        ptx::red_add_rel(
                            workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                    }
                }
                __syncwarp();
            }
        }

        // Cleanup workspace, overlapping with combine. A default-off
        // diagnostic can skip the zero stores, but it must still keep the
        // synchronization/barrier protocol intact across ranks and launches.
        if constexpr (not kDispatchOnly)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if constexpr (not kSkipWorkspaceCleanup) {
            if (sm_idx == 0) {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                    *workspace.get_expert_send_count_ptr(i) = 0;
            } else {
                for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                    const auto num_recv_tokens = static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_sum_ptr(i));
                    const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                    expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                    DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }

                    for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                        *workspace.get_expert_recv_count_ptr(j, i) = 0;
                    __syncwarp();

                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                        *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                        *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                    }
                    __syncwarp();
                }
            }
        }

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);
        }

    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
    //   Warps inside `kNumNonEpilogueThreads`: warp 0 loads A + SFA,
    //   warp 1 loads B, optional warp 2 loads SFA in split-SFA mode.
    //   Experimental split-SFA mode moves the SFA TMAs to warp 2, keeping
    //   the same full-barrier dependency while issuing A and SFA independently.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        auto process_a_sfa_block = [&](const auto& block_phase,
                                       const uint32_t& local_expert_idx,
                                       const uint32_t& num_k_blocks,
                                       const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait for the pool to be ready
            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                while (ptx::ld_acq(ptr) != expected);
            } else if constexpr (not (kGemmOnly and kGemmOnlySkipL2ArrivalWait)) {
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                if constexpr (kL2ArrivalCounter) {
                    const auto ptr = reinterpret_cast<const uint32_t*>(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                    const uint32_t active_m_wgs = math::ceil_div(
                        scheduler.template get_valid_m<false>(), WG_BLOCK_M);
                    const uint32_t expected =
                        kNumL1BlockNs * active_m_wgs * kWarpgroupSplitN * kL1OutputArrivalParts;
                    while (ptx::ld_acq(ptr) != expected);
                } else {
                    const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                    // Each L1 N block sets one bit; total bits = L1_SHAPE_N / BLOCK_N.
                    const uint64_t expected = (kNumL1BlockNs >= 64)
                        ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
                    while (ptx::ld_acq_gpu(ptr) != expected);
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    if constexpr (kGemmOnly and kGemmOnlySkipL2ActLoad) {
                        if (block_phase == sched::BlockPhase::Linear2) {
                            full_barriers[stage_idx]->arrive_and_expect_tx(0);
                        } else {
                            // TMA load A
                            tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                                k_idx, m_idx, 1);

                            if constexpr (kSplitSFALoaderWarp) {
                                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                            } else {
                                // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                                tma::copy<BLOCK_M, 1, 0, float>(
                                    tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                    m_idx, k_block_idx, 1);
                                full_barriers[stage_idx]->arrive_and_expect_tx(
                                    SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                            }
                        }
                    } else {
                        // TMA load A
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                            k_idx, m_idx, 1);

                        if constexpr (kSplitSFALoaderWarp) {
                            full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                        } else {
                            // TMA load SFA
                            if (block_phase == sched::BlockPhase::Linear1) {
                                // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                                tma::copy<BLOCK_M, 1, 0, float>(
                                    tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                    m_idx, k_block_idx, 1);
                                full_barriers[stage_idx]->arrive_and_expect_tx(
                                    SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                            } else {
                                // L2 SFA per-64: descriptor box is (block_mn, 1) (see make_tma_sf_desc),
                                // so we must issue two single-group TMAs and place them at smem offsets
                                // 0 and BLOCK_M to match math's load offsets (`+ 0 * BLOCK_M` / `+ 1 * BLOCK_M`).
                                tma::copy<BLOCK_M, 1, 0, float>(
                                    tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                    m_idx, k_block_idx * 2, 1);
                                tma::copy<BLOCK_M, 1, 0, float>(
                                    tensor_map_sfa_ptr, full_barriers[stage_idx],
                                    smem_sfa[stage_idx] + BLOCK_M,
                                    m_idx, k_block_idx * 2 + 1, 1);
                                full_barriers[stage_idx]->arrive_and_expect_tx(
                                    SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                            }
                        }
                    }
                }
                __syncwarp();
            }
        };

        if constexpr (kSplitPhaseHotPath) {
            scheduler.for_each_block_split(
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
        } else {
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_a_sfa_block(block_phase, local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            });
        }

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    if constexpr (kGemmOnly and kGemmOnlySkipBLoad) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(0);
                    } else {
                        // TMA load B (weight SF is now loaded directly by math warps from global)
                        if constexpr (LOAD_BLOCK_N <= 256) {
                            tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                                k_idx, n_idx, 1);
                        } else {
                            DG_STATIC_ASSERT(LOAD_BLOCK_N % 256 == 0,
                                             "Large B tiles are loaded as 256-column TMA slices");
                            #pragma unroll
                            for (uint32_t b_slice_idx = 0; b_slice_idx < LOAD_BLOCK_N / 256; ++ b_slice_idx) {
                                tma::copy<BLOCK_K, 256, kSwizzleBMode, b_dtype_t>(
                                    tensor_map_b_ptr, full_barriers[stage_idx],
                                    smem_b[stage_idx] + b_slice_idx * 256 * BLOCK_K,
                                    k_idx, n_idx + b_slice_idx * 256, 1);
                            }
                        }

                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                    }
                }
                __syncwarp();
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 2 and kSplitSFALoaderWarp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                while (ptx::ld_acq(ptr) != expected);
            } else if constexpr (not (kGemmOnly and kGemmOnlySkipL2ArrivalWait)) {
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                if constexpr (kL2ArrivalCounter) {
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

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    if (block_phase == sched::BlockPhase::Linear1) {
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
                    } else if constexpr (kGemmOnly and kGemmOnlySkipL2ActLoad) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(0);
                    } else {
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx * 2, 1);
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx] + BLOCK_M,
                            m_idx, k_block_idx * 2 + 1, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(2 * BLOCK_M * sizeof(float));
                    }
                }
                __syncwarp();
            }
        });

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

        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N = WG_L1_OUT_BLOCK_N;
        constexpr uint32_t WG_SMEM_CD_L2_STRIDE_N = WG_BLOCK_N;

        // Sync with dispatch in the full communication path. GEMM-only mode
        // has no dispatch participants after the synthetic setup above.
        if constexpr (not kGemmOnly and not kDispatchOnly)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        if constexpr (kDispatchOnly) {
            return;
        }

        auto process_math_block = [&](const auto& block_phase,
                                      const uint32_t& local_expert_idx,
                                      const uint32_t& num_k_blocks,
                                      const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
            const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
            const uint32_t wg_n_offset = epilogue_wg_n_idx * WG_BLOCK_N;
            const uint32_t wg_l1_out_n_offset = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t row_base = epilogue_wg_m_idx * WG_BLOCK_M;
            const uint32_t row_offset_r0 = row_base + r_0;
            const uint32_t row_offset_r1 = row_base + r_1;
            const uint32_t sf_n_block_idx = n_block_idx * kWarpgroupSplitN + epilogue_wg_n_idx;
            const uint32_t smem_a_wg_offset = epilogue_wg_m_idx * WG_BLOCK_M * BLOCK_K;
            const uint32_t smem_b_wg_offset = epilogue_wg_n_idx * WG_BLOCK_N * BLOCK_K;
            const uint32_t smem_cd_l1_wg_offset = epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N;
            const uint32_t smem_cd_l2_wg_offset = epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
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
                auto prescale_l2_final = [&](const float& scale_a_0, const float& scale_a_1,
                                             const float& l2_sf) {
                    const float inv_s0 = kFastMath ? math::fast_rcp(scale_a_0 * l2_sf) : 1.0f / (scale_a_0 * l2_sf);
                    const float inv_s1 = kFastMath ? math::fast_rcp(scale_a_1 * l2_sf) : 1.0f / (scale_a_1 * l2_sf);
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= inv_s0;
                        final_accum[i*4+1] *= inv_s0;
                        final_accum[i*4+2] *= inv_s1;
                        final_accum[i*4+3] *= inv_s1;
                    }
                };
                auto postscale_l2_final = [&](const float& scale_a_0, const float& scale_a_1,
                                              const float& l2_sf) {
                    const float s0 = scale_a_0 * l2_sf;
                    const float s1 = scale_a_1 * l2_sf;
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= s0;
                        final_accum[i*4+1] *= s0;
                        final_accum[i*4+2] *= s1;
                        final_accum[i*4+3] *= s1;
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
                                            const float& prev_l2_sf,
                                            const float& scale_a_0, const float& scale_a_1,
                                            const float& l2_sf) {
                    const float r0 = (prev_scale_a_0 * prev_l2_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_0 * l2_sf) : 1.0f / (scale_a_0 * l2_sf));
                    const float r1 = (prev_scale_a_1 * prev_l2_sf) *
                        (kFastMath ? math::fast_rcp(scale_a_1 * l2_sf) : 1.0f / (scale_a_1 * l2_sf));
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] *= r0;
                        final_accum[i*4+1] *= r0;
                        final_accum[i*4+2] *= r1;
                        final_accum[i*4+3] *= r1;
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

                if constexpr (kHidden >= 7168) {
                    float prev_scale_a_0 = 1.0f, prev_scale_a_1 = 1.0f;
                    float prev_gate_sf = 1.0f, prev_up_sf = 1.0f, prev_l2_sf = 1.0f;
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);

                        float scale_a_0_lo, scale_a_1_lo;
                        float scale_a_0_hi, scale_a_1_hi;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        } else {
                            if constexpr (kGemmOnly and kGemmOnlySkipL2ActLoad) {
                                scale_a_0_lo = 1.0f;
                                scale_a_1_lo = 1.0f;
                                scale_a_0_hi = 1.0f;
                                scale_a_1_hi = 1.0f;
                            } else {
                                scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                                scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                                scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                                scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                            }
                        }

                        constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                        constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                        constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                        float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            const uint32_t gate_n = sf_n_block_idx / 2u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert + k_block_idx;
                            gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                            up_sf   = __ldg(base + up_n   * kL1SFKBlocks);
                        } else {
                            l2_sf = __ldg(l2_weights_sf + local_expert_idx * kL2SFPerExpert
                                                        + sf_n_block_idx * kL2SFKBlocks + k_block_idx);
                        }

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
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            prev_scale_a_0 = scale_a_0_lo;
                            prev_scale_a_1 = scale_a_1_lo;
                            prev_gate_sf = gate_sf;
                            prev_up_sf = up_sf;
                        } else {
                            if (k_block_idx != 0)
                                rescale_l2_final(prev_scale_a_0, prev_scale_a_1, prev_l2_sf,
                                                 scale_a_0_lo, scale_a_1_lo, l2_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
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
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            prev_scale_a_0 = scale_a_0_hi;
                            prev_scale_a_1 = scale_a_1_hi;
                            prev_l2_sf = l2_sf;
                        }
                    }

                    if (num_k_blocks != 0) {
                        if (block_phase == sched::BlockPhase::Linear1) {
                            postscale_l1_final(prev_scale_a_0, prev_scale_a_1,
                                               prev_gate_sf, prev_up_sf);
                        } else {
                            postscale_l2_final(prev_scale_a_0, prev_scale_a_1, prev_l2_sf);
                        }
                    }
                } else {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);

                        float scale_a_0_lo, scale_a_1_lo;
                        float scale_a_0_hi, scale_a_1_hi;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                            scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        } else {
                            if constexpr (kGemmOnly and kGemmOnlySkipL2ActLoad) {
                                scale_a_0_lo = 1.0f;
                                scale_a_1_lo = 1.0f;
                                scale_a_0_hi = 1.0f;
                                scale_a_1_hi = 1.0f;
                            } else {
                                scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                                scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                                scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                                scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                            }
                        }

                        constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                        constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                        constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                        constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                        float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            const uint32_t gate_n = sf_n_block_idx / 2u;
                            const uint32_t up_n   = kL1SFGateBlks + gate_n;
                            const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert + k_block_idx;
                            gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                            up_sf   = __ldg(base + up_n   * kL1SFKBlocks);
                        } else {
                            l2_sf = __ldg(l2_weights_sf + local_expert_idx * kL2SFPerExpert
                                                        + sf_n_block_idx * kL2SFKBlocks + k_block_idx);
                        }

                        if (block_phase == sched::BlockPhase::Linear1) {
                            if (k_block_idx != 0)
                                prescale_l1_final(scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            postscale_l1_final(scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);
                        } else {
                            if (k_block_idx != 0)
                                prescale_l2_final(scale_a_0_lo, scale_a_1_lo, l2_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            postscale_l2_final(scale_a_0_lo, scale_a_1_lo, l2_sf);
                            prescale_l2_final(scale_a_0_hi, scale_a_1_hi, l2_sf);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();

                            postscale_l2_final(scale_a_0_hi, scale_a_1_hi, l2_sf);
                        }
                    }
                }
            } else if constexpr (kOverlapKScale) {
            float accum0[kAccumPerThread];
            float accum1[kAccumPerThread];

            auto get_accum = [&](uint32_t buf_idx) -> float* {
                return buf_idx == 0 ? accum0 : accum1;
            };
            auto fence_accum = [&](float* accum_buf) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                    ptx::warpgroup_fence_operand(accum_buf[i]);
            };
            auto issue_l1_group = [&](float* accum_buf, uint32_t issue_stage_idx) {
                fence_accum(accum_buf);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_a[issue_stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_b[issue_stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum_buf, k);
                }
                ptx::warpgroup_commit_batch();
                fence_accum(accum_buf);
            };
            auto issue_l2_group = [&](float* accum_buf, uint32_t issue_stage_idx, uint32_t k_offset) {
                fence_accum(accum_buf);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                    const uint32_t k_off = k_offset + k * WGMMA::K;
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_a[issue_stage_idx] + smem_a_wg_offset + k_off, 1);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_b[issue_stage_idx] + smem_b_wg_offset + k_off, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum_buf, k);
                }
                ptx::warpgroup_commit_batch();
                fence_accum(accum_buf);
            };
            auto accumulate_l1_group = [&](float* accum_buf,
                                           const float scale_a_0, const float scale_a_1,
                                           const float gate_sf, const float up_sf) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                    const float sb = (i & 1u) ? up_sf : gate_sf;
                    final_accum[i*4+0] += scale_a_0 * sb * accum_buf[i*4+0];
                    final_accum[i*4+1] += scale_a_0 * sb * accum_buf[i*4+1];
                    final_accum[i*4+2] += scale_a_1 * sb * accum_buf[i*4+2];
                    final_accum[i*4+3] += scale_a_1 * sb * accum_buf[i*4+3];
                }
            };
            auto accumulate_l2_group = [&](float* accum_buf,
                                           const float scale_a_0, const float scale_a_1,
                                           const float l2_sf) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                    final_accum[i*4+0] += scale_a_0 * l2_sf * accum_buf[i*4+0];
                    final_accum[i*4+1] += scale_a_0 * l2_sf * accum_buf[i*4+1];
                    final_accum[i*4+2] += scale_a_1 * l2_sf * accum_buf[i*4+2];
                    final_accum[i*4+3] += scale_a_1 * l2_sf * accum_buf[i*4+3];
                }
            };

            constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
            constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
            constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
            constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
            constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;

            if (block_phase == sched::BlockPhase::Linear1) {
                bool have_prev = false;
                uint32_t cur_buf = 0, prev_buf = 0, prev_stage = 0;
                float prev_scale_a_0 = 1.0f, prev_scale_a_1 = 1.0f;
                float prev_gate_sf = 1.0f, prev_up_sf = 1.0f;

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);

                    const float scale_a_0 = ptx::ld_shared(
                        smem_sfa[stage_idx] + row_offset_r0);
                    const float scale_a_1 = ptx::ld_shared(
                        smem_sfa[stage_idx] + row_offset_r1);
                    const uint32_t gate_n = sf_n_block_idx / 2u;
                    const uint32_t up_n   = kL1SFGateBlks + gate_n;
                    const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert + k_block_idx;
                    const float gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                    const float up_sf   = __ldg(base + up_n   * kL1SFKBlocks);

                    issue_l1_group(get_accum(cur_buf), stage_idx);

                    if (have_prev) {
                        ptx::warpgroup_wait<1>();
                        if (lane_idx == 0)
                            empty_barriers[prev_stage]->arrive();
                        accumulate_l1_group(get_accum(prev_buf),
                                            prev_scale_a_0, prev_scale_a_1,
                                            prev_gate_sf, prev_up_sf);
                    }

                    have_prev = true;
                    prev_buf = cur_buf;
                    prev_stage = stage_idx;
                    prev_scale_a_0 = scale_a_0;
                    prev_scale_a_1 = scale_a_1;
                    prev_gate_sf = gate_sf;
                    prev_up_sf = up_sf;
                    cur_buf ^= 1;
                }

                if (have_prev) {
                    ptx::warpgroup_wait<0>();
                    if (lane_idx == 0)
                        empty_barriers[prev_stage]->arrive();
                    accumulate_l1_group(get_accum(prev_buf),
                                        prev_scale_a_0, prev_scale_a_1,
                                        prev_gate_sf, prev_up_sf);
                }
            } else {
                bool have_prev = false;
                uint32_t cur_buf = 0, prev_buf = 0, prev_stage = 0;
                bool prev_release_stage = false;
                float prev_scale_a_0 = 1.0f, prev_scale_a_1 = 1.0f, prev_l2_sf = 1.0f;

                auto consume_prev_l2 = [&]() {
                    if (have_prev) {
                        ptx::warpgroup_wait<1>();
                        if (prev_release_stage and lane_idx == 0)
                            empty_barriers[prev_stage]->arrive();
                        accumulate_l2_group(get_accum(prev_buf),
                                            prev_scale_a_0, prev_scale_a_1,
                                            prev_l2_sf);
                    }
                };

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);

                    float scale_a_0_lo, scale_a_1_lo;
                    float scale_a_0_hi, scale_a_1_hi;
                    if constexpr (kGemmOnly and kGemmOnlySkipL2ActLoad) {
                        scale_a_0_lo = 1.0f;
                        scale_a_1_lo = 1.0f;
                        scale_a_0_hi = 1.0f;
                        scale_a_1_hi = 1.0f;
                    } else {
                        scale_a_0_lo = ptx::ld_shared(
                            smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(
                            smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                        scale_a_0_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                        scale_a_1_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                    }
                    const float l2_sf = __ldg(l2_weights_sf + local_expert_idx * kL2SFPerExpert
                                                          + sf_n_block_idx * kL2SFKBlocks + k_block_idx);

                    issue_l2_group(get_accum(cur_buf), stage_idx, 0);
                    consume_prev_l2();
                    have_prev = true;
                    prev_buf = cur_buf;
                    prev_stage = stage_idx;
                    prev_release_stage = false;
                    prev_scale_a_0 = scale_a_0_lo;
                    prev_scale_a_1 = scale_a_1_lo;
                    prev_l2_sf = l2_sf;
                    cur_buf ^= 1;

                    issue_l2_group(get_accum(cur_buf), stage_idx, BLOCK_K / 2);
                    consume_prev_l2();
                    have_prev = true;
                    prev_buf = cur_buf;
                    prev_stage = stage_idx;
                    prev_release_stage = true;
                    prev_scale_a_0 = scale_a_0_hi;
                    prev_scale_a_1 = scale_a_1_hi;
                    prev_l2_sf = l2_sf;
                    cur_buf ^= 1;
                }

                if (have_prev) {
                    ptx::warpgroup_wait<0>();
                    if (prev_release_stage and lane_idx == 0)
                        empty_barriers[prev_stage]->arrive();
                    accumulate_l2_group(get_accum(prev_buf),
                                        prev_scale_a_0, prev_scale_a_1,
                                        prev_l2_sf);
                }
            }
            } else {
            float accum[kAccumPerThread];

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                if (block_phase == sched::BlockPhase::Linear1) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                } else {
                    if constexpr (kGemmOnly and kGemmOnlySkipL2ActLoad) {
                        scale_a_0_lo = 1.0f;
                        scale_a_1_lo = 1.0f;
                        scale_a_0_hi = 1.0f;
                        scale_a_1_hi = 1.0f;
                    } else {
                        // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + row_offset_r1);
                        scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r0);
                        scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + row_offset_r1);
                    }
                }

                // ----- Block (128, 128) weight SF (loaded directly from global) -----
                // L1 weight SF shape: (E, 2*IH/128, H/128) MN-major. The N axis is
                // [gate(IH/128), up(IH/128)]; with the gate/up gran-8 interleave on
                // the FP8 weight, each BLOCK_N=128 tile covers 64 rows of gate plus
                // 64 rows of up taken from the same original 128-row block, so:
                //     gate_sf_n = n_block_idx / 2
                //     up_sf_n   = (IH/128) + n_block_idx / 2
                //
                // L2 weight SF shape: (E, H/128, IH/128) MN-major. One scalar per
                // (BLOCK_N, BLOCK_K) tile, broadcast across all WGMMA accumulators.
                //
                // NOTE: we tried hoisting these LDGs above the barrier wait and/or
                // having only lane 0 load + shfl-broadcast. Both regressed on H20
                // by 7-11% across all batch sizes, presumably because (a) Hopper's
                // L1 read-only cache already coalesces same-address LDGs from all
                // 128 WG threads and (b) hoisting contended with the dispatch
                // warps' NVLink LDGs on the MIO unit. Keep the simple parallel
                // post-wait load.
                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;
                float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                if (block_phase == sched::BlockPhase::Linear1) {
                    const uint32_t gate_n = sf_n_block_idx / 2u;
                    const uint32_t up_n   = kL1SFGateBlks + gate_n;
                    const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert + k_block_idx;
                    gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                    up_sf   = __ldg(base + up_n   * kL1SFKBlocks);
                } else {
                    l2_sf = __ldg(l2_weights_sf + local_expert_idx * kL2SFPerExpert
                                                + sf_n_block_idx * kL2SFKBlocks + k_block_idx);
                }

                if (block_phase == sched::BlockPhase::Linear1) {
                    // Single per-128 K-block WGMMA group
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();

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
                } else {
                    // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                    // First half: K=0..63, SFA = scale_a_*_lo
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + smem_a_wg_offset + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + smem_b_wg_offset + k * WGMMA::K, 1);
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
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();

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

            // Skip epilogue when block is past valid M (still must release via empty)
            if (row_base >= valid_m) {
                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (not kL2ArrivalCounter)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    if constexpr (not kSkipL2EpilogueSync)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                return;
            }

            if (block_phase == sched::BlockPhase::Linear1) {
                if constexpr (kGemmOnly and kGemmOnlySkipL1Epilogue) {
                    if (epilogue_thread_idx == 0 and cumulative_local_expert_recv_stats != nullptr)
                        cumulative_local_expert_recv_stats[0] = __float_as_int(final_accum[0]);
                    if constexpr (kL2ArrivalCounter) {
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)), 1);
                        }
                    } else {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            ptx::red_or_rel_gpu(
                                workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                                1ull << n_block_idx);
                        }
                    }
                    __syncwarp();
                    return;
                }

                // ---------------- L1 EPILOGUE: activation + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;  // 8 for BLOCK_N=128
                float sf_r0, sf_inv_r0;
                float sf_r1, sf_inv_r1;

                if constexpr (kInplaceSwiglu) {
                    DG_STATIC_ASSERT(not (kL1StsmX4 or kL1StsmX2),
                                     "inplace SwiGLU path only supports the row-major L1 store path");

                    float weight_r0 = 1.0f, weight_r1 = 1.0f;
                    if constexpr (not (kGemmOnly and kGemmOnlySkipL1TopkWeight)) {
                        weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .get_base_ptr<float>() : 0.0f;
                        weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .get_base_ptr<float>() : 0.0f;
                    }

                    float amax_r0 = 0.0f, amax_r1 = 0.0f;
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t gate = 2 * p, up = 2 * p + 1;
                        if (valid_r0) {
                            float v0 = sm90_fp8_mega_moe_swiglu<kFastMath, kActivationClamp>(
                                final_accum[gate*4 + 0], final_accum[up*4 + 0]) * weight_r0;
                            float v1 = sm90_fp8_mega_moe_swiglu<kFastMath, kActivationClamp>(
                                final_accum[gate*4 + 1], final_accum[up*4 + 1]) * weight_r0;
                            final_accum[gate*4 + 0] = v0;
                            final_accum[gate*4 + 1] = v1;
                            if constexpr (not (kGemmOnly and kGemmOnlyFixedL1Scale))
                                amax_r0 = cute::max(amax_r0, cute::max(cute::abs(v0), cute::abs(v1)));
                        } else {
                            final_accum[gate*4 + 0] = 0.0f;
                            final_accum[gate*4 + 1] = 0.0f;
                        }
                        if (valid_r1) {
                            float v0 = sm90_fp8_mega_moe_swiglu<kFastMath, kActivationClamp>(
                                final_accum[gate*4 + 2], final_accum[up*4 + 2]) * weight_r1;
                            float v1 = sm90_fp8_mega_moe_swiglu<kFastMath, kActivationClamp>(
                                final_accum[gate*4 + 3], final_accum[up*4 + 3]) * weight_r1;
                            final_accum[gate*4 + 2] = v0;
                            final_accum[gate*4 + 3] = v1;
                            if constexpr (not (kGemmOnly and kGemmOnlyFixedL1Scale))
                                amax_r1 = cute::max(amax_r1, cute::max(cute::abs(v0), cute::abs(v1)));
                        } else {
                            final_accum[gate*4 + 2] = 0.0f;
                            final_accum[gate*4 + 3] = 0.0f;
                        }
                    }

                    if constexpr (kGemmOnly and kGemmOnlyFixedL1Scale) {
                        sf_r0 = 1.0f; sf_inv_r0 = 1.0f;
                        sf_r1 = 1.0f; sf_inv_r1 = 1.0f;
                    } else {
                        amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                        amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());
                        float2 amax_pair = {amax_r0, amax_r1};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                        sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                    }

                    auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
                    DG_STATIC_ASSERT(kNumPairs % 2 == 0, "L1 stores two 8-byte chunks at once");
                    #pragma unroll
                    for (uint32_t p_base = 0; p_base < kNumPairs; p_base += 2) {
                        uint16_t r0_bits[2], r1_bits[2];
                        #pragma unroll
                        for (uint32_t q = 0; q < 2; ++ q) {
                            const uint32_t p = p_base + q;
                            const uint32_t gate = 2 * p;
                            const float v00 = final_accum[gate*4 + 0] * sf_inv_r0;
                            const float v01 = final_accum[gate*4 + 1] * sf_inv_r0;
                            const float v10 = final_accum[gate*4 + 2] * sf_inv_r1;
                            const float v11 = final_accum[gate*4 + 3] * sf_inv_r1;

                            if constexpr (kGemmOnly and kGemmOnlySkipL1FP8Pack) {
                                const uint32_t r0_mix = (__float_as_uint(v00) & 0xffffu) ^ (__float_as_uint(v01) >> 16);
                                const uint32_t r1_mix = (__float_as_uint(v10) & 0xffffu) ^ (__float_as_uint(v11) >> 16);
                                r0_bits[q] = valid_r0 ? static_cast<uint16_t>(r0_mix) : 0u;
                                r1_bits[q] = valid_r1 ? static_cast<uint16_t>(r1_mix) : 0u;
                            } else {
                                const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                                const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                                r0_bits[q] = valid_r0 ? r0_pair.__x : 0u;
                                r1_bits[q] = valid_r1 ? r1_pair.__x : 0u;
                            }
                        }

                        if constexpr (kGemmOnly and kGemmOnlyDirectL1GlobalStore) {
                            #pragma unroll
                            for (uint32_t q = 0; q < 2; ++ q) {
                                const uint32_t r0_buddy = __shfl_xor_sync(
                                    0xffffffffu, static_cast<uint32_t>(r0_bits[q]), 1);
                                const uint32_t r1_buddy = __shfl_xor_sync(
                                    0xffffffffu, static_cast<uint32_t>(r1_bits[q]), 1);
                                const uint32_t r0_packed =
                                    (static_cast<uint32_t>(r0_bits[q]) & 0xffffu) | ((r0_buddy & 0xffffu) << 16);
                                const uint32_t r1_packed =
                                    (static_cast<uint32_t>(r1_bits[q]) & 0xffffu) | ((r1_buddy & 0xffffu) << 16);

                                if ((col_idx & 1u) == 0) {
                                    const uint32_t col = (p_base + q) * 8 + col_idx * 2;
                                    const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset + col;
                                    if (valid_r0) {
                                        auto* g0 = reinterpret_cast<uint32_t*>(
                                            l2_token_buffer.get_data_buffer(m_idx + row_offset_r0)
                                                .get_base_ptr<uint8_t>() + out_n_idx);
                                        *g0 = r0_packed;
                                    }
                                    if (valid_r1) {
                                        auto* g1 = reinterpret_cast<uint32_t*>(
                                            l2_token_buffer.get_data_buffer(m_idx + row_offset_r1)
                                                .get_base_ptr<uint8_t>() + out_n_idx);
                                        *g1 = r1_packed;
                                    }
                                }
                            }
                        } else if constexpr (kGemmOnly and kGemmOnlySkipL1SmemStore) {
                            const uint32_t r0_sink =
                                (static_cast<uint32_t>(r0_bits[0]) & 0xffffu) |
                                ((static_cast<uint32_t>(r0_bits[1]) & 0xffffu) << 16);
                            const uint32_t r1_sink =
                                (static_cast<uint32_t>(r1_bits[0]) & 0xffffu) |
                                ((static_cast<uint32_t>(r1_bits[1]) & 0xffffu) << 16);
                            asm volatile("" :: "r"(r0_sink), "r"(r1_sink));
                        } else if constexpr (not kGemmOnly) {
                            #pragma unroll
                            for (uint32_t q = 0; q < 2; ++ q) {
                                const uint32_t p = p_base + q;
                                const uint32_t col = p * 8 + col_idx * 2;
                                auto* p0 = reinterpret_cast<uint16_t*>(
                                    smem_cd_l1_wg + r_0 * WG_L1_OUT_BLOCK_N + col);
                                auto* p1 = reinterpret_cast<uint16_t*>(
                                    smem_cd_l1_wg + r_1 * WG_L1_OUT_BLOCK_N + col);
                                if (valid_r0)
                                    *p0 = r0_bits[q];
                                if (valid_r1)
                                    *p1 = r1_bits[q];
                            }
                        } else {
                            const uint32_t row_lane_base = lane_idx & ~3u;
                            const uint32_t src_pair = col_idx & 1u;
                            const uint32_t src_lane_0 = row_lane_base + src_pair * 2u;
                            const uint32_t src_lane_1 = src_lane_0 + 1u;
                            const bool use_hi_chunk = col_idx >= 2u;
                            const uint32_t r0_lo = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r0_bits[use_hi_chunk]), src_lane_0);
                            const uint32_t r0_hi = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r0_bits[use_hi_chunk]), src_lane_1);
                            const uint32_t r1_lo = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r1_bits[use_hi_chunk]), src_lane_0);
                            const uint32_t r1_hi = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r1_bits[use_hi_chunk]), src_lane_1);
                            const uint32_t r0_stsm = (r0_lo & 0xffffu) | ((r0_hi & 0xffffu) << 16);
                            const uint32_t r1_stsm = (r1_lo & 0xffffu) | ((r1_hi & 0xffffu) << 16);
                            constexpr uint32_t kL1OutSwizzleBytes = WG_L1_OUT_BLOCK_N;
                            constexpr uint32_t kNumBankGroupBytes = 16;
                            const uint32_t in_atom_offset = p_base / 2;
                            const uint32_t bank_group_index =
                                in_atom_offset + lane_idx * (kL1OutSwizzleBytes / kNumBankGroupBytes);
                            uint32_t row = bank_group_index / 8;
                            uint32_t col = bank_group_index % 8;
                            col ^= row % (kL1OutSwizzleBytes / kNumBankGroupBytes);
                            auto* smem_ptr = reinterpret_cast<uint8_t*>(smem_cd_l1_wg)
                                + warp_idx_in_wg * 16 * kL1OutSwizzleBytes
                                + row * (kNumBankGroupBytes * 8)
                                + col * kNumBankGroupBytes;
                            ptx::SM90_U32x2_STSM_N<uint32_t>::copy(r0_stsm, r1_stsm, smem_ptr);
                        }
                    }
                } else if constexpr (kStreamSwigluQuant) {
                    auto clamp_gate_value = [](float x) -> float {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                        return x;
                    };
                    auto clamp_up_value = [](float x) -> float {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                        return x;
                    };
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto swiglu = [&](float g, float u) -> float {
                        if constexpr (not kGemmOnly) {
                            g = clamp_gate_value(g);
                            u = clamp_up_value(u);
                            return silu(g) * u;
                        } else {
                            return g * u;
                        }
                    };

                    float amax_r0 = 0.0f, amax_r1 = 0.0f;
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t gate = 2 * p, up = 2 * p + 1;
                        if (valid_r0) {
                            const float v0 = swiglu(final_accum[gate*4 + 0], final_accum[up*4 + 0]);
                            const float v1 = swiglu(final_accum[gate*4 + 1], final_accum[up*4 + 1]);
                            if constexpr (not (kGemmOnly and kGemmOnlyFixedL1Scale))
                                amax_r0 = cute::max(amax_r0, cute::max(cute::abs(v0), cute::abs(v1)));
                        }
                        if (valid_r1) {
                            const float v0 = swiglu(final_accum[gate*4 + 2], final_accum[up*4 + 2]);
                            const float v1 = swiglu(final_accum[gate*4 + 3], final_accum[up*4 + 3]);
                            if constexpr (not (kGemmOnly and kGemmOnlyFixedL1Scale))
                                amax_r1 = cute::max(amax_r1, cute::max(cute::abs(v0), cute::abs(v1)));
                        }
                    }

                    float weight_r0 = 1.0f, weight_r1 = 1.0f;
                    if constexpr (not (kGemmOnly and kGemmOnlySkipL1TopkWeight)) {
                        weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .get_base_ptr<float>() : 0.0f;
                        weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .get_base_ptr<float>() : 0.0f;
                    }

                    if constexpr (kGemmOnly and kGemmOnlyFixedL1Scale) {
                        sf_r0 = 1.0f; sf_inv_r0 = 1.0f;
                        sf_r1 = 1.0f; sf_inv_r1 = 1.0f;
                    } else {
                        amax_r0 *= cute::abs(weight_r0);
                        amax_r1 *= cute::abs(weight_r1);
                        amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                        amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());
                        float2 amax_pair = {amax_r0, amax_r1};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                        sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                    }

                    auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
                    DG_STATIC_ASSERT(kNumPairs % 2 == 0, "L1 STSM staging stores two 8-byte chunks at once");
                    #pragma unroll
                    for (uint32_t p_base = 0; p_base < kNumPairs; p_base += 2) {
                        uint16_t r0_bits[2], r1_bits[2];
                        #pragma unroll
                        for (uint32_t q = 0; q < 2; ++ q) {
                            const uint32_t p = p_base + q;
                            const uint32_t gate = 2 * p, up = 2 * p + 1;
                            const float v00 = swiglu(final_accum[gate*4 + 0], final_accum[up*4 + 0]) * weight_r0 * sf_inv_r0;
                            const float v01 = swiglu(final_accum[gate*4 + 1], final_accum[up*4 + 1]) * weight_r0 * sf_inv_r0;
                            const float v10 = swiglu(final_accum[gate*4 + 2], final_accum[up*4 + 2]) * weight_r1 * sf_inv_r1;
                            const float v11 = swiglu(final_accum[gate*4 + 3], final_accum[up*4 + 3]) * weight_r1 * sf_inv_r1;

                            if constexpr (kGemmOnly and kGemmOnlySkipL1FP8Pack) {
                                const uint32_t r0_mix = (__float_as_uint(v00) & 0xffffu) ^ (__float_as_uint(v01) >> 16);
                                const uint32_t r1_mix = (__float_as_uint(v10) & 0xffffu) ^ (__float_as_uint(v11) >> 16);
                                r0_bits[q] = valid_r0 ? static_cast<uint16_t>(r0_mix) : 0u;
                                r1_bits[q] = valid_r1 ? static_cast<uint16_t>(r1_mix) : 0u;
                            } else {
                                const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                                const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                                r0_bits[q] = valid_r0 ? r0_pair.__x : 0u;
                                r1_bits[q] = valid_r1 ? r1_pair.__x : 0u;
                            }
                        }

                        if constexpr (kGemmOnly and kGemmOnlyDirectL1GlobalStore) {
                            #pragma unroll
                            for (uint32_t q = 0; q < 2; ++ q) {
                                const uint32_t r0_buddy = __shfl_xor_sync(
                                    0xffffffffu, static_cast<uint32_t>(r0_bits[q]), 1);
                                const uint32_t r1_buddy = __shfl_xor_sync(
                                    0xffffffffu, static_cast<uint32_t>(r1_bits[q]), 1);
                                const uint32_t r0_packed =
                                    (static_cast<uint32_t>(r0_bits[q]) & 0xffffu) | ((r0_buddy & 0xffffu) << 16);
                                const uint32_t r1_packed =
                                    (static_cast<uint32_t>(r1_bits[q]) & 0xffffu) | ((r1_buddy & 0xffffu) << 16);

                                if ((col_idx & 1u) == 0) {
                                    const uint32_t col = (p_base + q) * 8 + col_idx * 2;
                                    const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset + col;
                                    if (valid_r0) {
                                        auto* g0 = reinterpret_cast<uint32_t*>(
                                            l2_token_buffer.get_data_buffer(m_idx + row_offset_r0)
                                                .get_base_ptr<uint8_t>() + out_n_idx);
                                        *g0 = r0_packed;
                                    }
                                    if (valid_r1) {
                                        auto* g1 = reinterpret_cast<uint32_t*>(
                                            l2_token_buffer.get_data_buffer(m_idx + row_offset_r1)
                                                .get_base_ptr<uint8_t>() + out_n_idx);
                                        *g1 = r1_packed;
                                    }
                                }
                            }
                        } else if constexpr (kGemmOnly and kGemmOnlySkipL1SmemStore) {
                            const uint32_t r0_sink =
                                (static_cast<uint32_t>(r0_bits[0]) & 0xffffu) |
                                ((static_cast<uint32_t>(r0_bits[1]) & 0xffffu) << 16);
                            const uint32_t r1_sink =
                                (static_cast<uint32_t>(r1_bits[0]) & 0xffffu) |
                                ((static_cast<uint32_t>(r1_bits[1]) & 0xffffu) << 16);
                            asm volatile("" :: "r"(r0_sink), "r"(r1_sink));
                        } else if constexpr (not kGemmOnly) {
                            #pragma unroll
                            for (uint32_t q = 0; q < 2; ++ q) {
                                const uint32_t p = p_base + q;
                                const uint32_t col = p * 8 + col_idx * 2;
                                auto* p0 = reinterpret_cast<uint16_t*>(
                                    smem_cd_l1_wg + r_0 * WG_L1_OUT_BLOCK_N + col);
                                auto* p1 = reinterpret_cast<uint16_t*>(
                                    smem_cd_l1_wg + r_1 * WG_L1_OUT_BLOCK_N + col);
                                if (valid_r0)
                                    *p0 = r0_bits[q];
                                if (valid_r1)
                                    *p1 = r1_bits[q];
                            }
                        } else {
                            const uint32_t row_lane_base = lane_idx & ~3u;
                            const uint32_t src_pair = col_idx & 1u;
                            const uint32_t src_lane_0 = row_lane_base + src_pair * 2u;
                            const uint32_t src_lane_1 = src_lane_0 + 1u;
                            const bool use_hi_chunk = col_idx >= 2u;
                            const uint32_t r0_lo = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r0_bits[use_hi_chunk]), src_lane_0);
                            const uint32_t r0_hi = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r0_bits[use_hi_chunk]), src_lane_1);
                            const uint32_t r1_lo = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r1_bits[use_hi_chunk]), src_lane_0);
                            const uint32_t r1_hi = __shfl_sync(
                                0xffffffffu, static_cast<uint32_t>(r1_bits[use_hi_chunk]), src_lane_1);
                            const uint32_t r0_stsm = (r0_lo & 0xffffu) | ((r0_hi & 0xffffu) << 16);
                            const uint32_t r1_stsm = (r1_lo & 0xffffu) | ((r1_hi & 0xffffu) << 16);
                            constexpr uint32_t kL1OutSwizzleBytes = WG_L1_OUT_BLOCK_N;
                            constexpr uint32_t kNumBankGroupBytes = 16;
                            const uint32_t in_atom_offset = p_base / 2;
                            const uint32_t bank_group_index =
                                in_atom_offset + lane_idx * (kL1OutSwizzleBytes / kNumBankGroupBytes);
                            uint32_t row = bank_group_index / 8;
                            uint32_t col = bank_group_index % 8;
                            col ^= row % (kL1OutSwizzleBytes / kNumBankGroupBytes);
                            auto* smem_ptr = reinterpret_cast<uint8_t*>(smem_cd_l1_wg)
                                + warp_idx_in_wg * 16 * kL1OutSwizzleBytes
                                + row * (kNumBankGroupBytes * 8)
                                + col * kNumBankGroupBytes;
                            ptx::SM90_U32x2_STSM_N<uint32_t>::copy(r0_stsm, r1_stsm, smem_ptr);
                        }
                    }
                } else {
                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];

	                // Per-row amax across all 8 pairs
	                float amax_r0 = 0.0f, amax_r1 = 0.0f;

	                // Compute SwiGLU + per-pair amax
	                #pragma unroll
	                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t gate = 2 * p, up = 2 * p + 1;

                    // Apply optional clamp on gate / up before SwiGLU
                    // Match SM100 reference: gate is clamped only on the upper
                    // side (very-negative gate is fine because SiLU(-inf) -> 0),
                    // while up is clamped both sides.
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };
                    float g_r0_c0 = final_accum[gate*4 + 0];
                    float g_r0_c1 = final_accum[gate*4 + 1];
                    float g_r1_c0 = final_accum[gate*4 + 2];
                    float g_r1_c1 = final_accum[gate*4 + 3];
                    float u_r0_c0 = final_accum[up*4   + 0];
                    float u_r0_c1 = final_accum[up*4   + 1];
                    float u_r1_c0 = final_accum[up*4   + 2];
                    float u_r1_c1 = final_accum[up*4   + 3];
                    if constexpr (not kGemmOnly) {
                        clamp_gate(g_r0_c0);
                        clamp_gate(g_r0_c1);
                        clamp_gate(g_r1_c0);
                        clamp_gate(g_r1_c1);
                        clamp_up(u_r0_c0);
                        clamp_up(u_r0_c1);
                        clamp_up(u_r1_c0);
                        clamp_up(u_r1_c1);
                    }

                    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };

	                    if (valid_r0) {
	                        if constexpr (kGemmOnly) {
	                            swiglu_r0[p][0] = g_r0_c0 * u_r0_c0;
	                            swiglu_r0[p][1] = g_r0_c1 * u_r0_c1;
	                        } else {
	                            swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
	                            swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
	                        }
	                        if constexpr (not (kGemmOnly and kGemmOnlyFixedL1Scale))
	                            amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
	                    } else {
                        swiglu_r0[p][0] = 0.0f;
                        swiglu_r0[p][1] = 0.0f;
                    }
	                    if (valid_r1) {
	                        if constexpr (kGemmOnly) {
	                            swiglu_r1[p][0] = g_r1_c0 * u_r1_c0;
	                            swiglu_r1[p][1] = g_r1_c1 * u_r1_c1;
	                        } else {
	                            swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
	                            swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
	                        }
	                        if constexpr (not (kGemmOnly and kGemmOnlyFixedL1Scale))
	                            amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
	                    } else {
                        swiglu_r1[p][0] = 0.0f;
                        swiglu_r1[p][1] = 0.0f;
	                    }
	                }

	                // Apply token weight: SwiGLU * topk_weight (single load per row)
	                float weight_r0 = 1.0f, weight_r1 = 1.0f;
	                if constexpr (not (kGemmOnly and kGemmOnlySkipL1TopkWeight)) {
	                    weight_r0 = valid_r0 ? *l1_topk_weights_buffer
	                        .get_data_buffer(m_idx + row_offset_r0)
	                        .get_base_ptr<float>() : 0.0f;
	                    weight_r1 = valid_r1 ? *l1_topk_weights_buffer
	                        .get_data_buffer(m_idx + row_offset_r1)
	                        .get_base_ptr<float>() : 0.0f;
	                    #pragma unroll
	                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
	                        swiglu_r0[p][0] *= weight_r0;
	                        swiglu_r0[p][1] *= weight_r0;
	                        swiglu_r1[p][0] *= weight_r1;
	                        swiglu_r1[p][1] *= weight_r1;
	                    }
	                }
	                if constexpr (kGemmOnly and kGemmOnlyFixedL1Scale) {
	                    sf_r0 = 1.0f; sf_inv_r0 = 1.0f;
	                    sf_r1 = 1.0f; sf_inv_r1 = 1.0f;
	                } else {
	                    amax_r0 *= cute::abs(weight_r0);
	                    amax_r1 *= cute::abs(weight_r1);

	                    // Reduce amax across the 4 col-lanes that share the same row.
                    // In WGMMA m64n128k32 output, the 4 lanes (`lane_idx & 3` differs,
                    // `lane_idx >> 2` same) hold all N positions for the same r_0/r_1,
                    // so we need an INTRA-group reduction (`xor 1, xor 2`), which is
                    // `warp_reduce<4, false>`. Using `<4, true>` would instead merge
                    // amax across 8 different rows -- giving wrong per-row SF.
                    amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                    amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

                    // Compute SF and inverse SF for each row
                    float2 amax_pair = {amax_r0, amax_r1};
                    float2 sf_pair, sf_inv_pair;
                    math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                    sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                    sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                }

                // Quantize and write to a 64B-swizzled smem_cd_l1 tile.
                // SM100 uses byte-wide `.m16n8.b8` STSM, which is not
                // available on SM90. Since each FP8 pair is one 16-bit unit,
                // two adjacent 8-byte chunks form an 8x8 `.b16` fragment for
                // `.m8n8.x2.b16` STSM. The matching TMA descriptor unswizzles
                // this staging tile into the row-major global L2 activation
                // buffer.
                auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
                DG_STATIC_ASSERT(kNumPairs % 2 == 0, "L1 STSM staging stores two 8-byte chunks at once");
                if constexpr (kL1StsmX4) {
                    DG_STATIC_ASSERT(kNumPairs % 4 == 0, "L1 x4 STSM staging stores four 8-byte chunks at once");
                    #pragma unroll
                    for (uint32_t p_base = 0; p_base < kNumPairs; p_base += 4) {
                        auto make_stsm_pair = [&](const uint32_t pair_base, uint32_t& r0_stsm, uint32_t& r1_stsm) {
                            uint16_t r0_bits[2], r1_bits[2];
                            #pragma unroll
                            for (uint32_t q = 0; q < 2; ++ q) {
                                const uint32_t p = pair_base + q;
                                const float v00 = swiglu_r0[p][0] * sf_inv_r0;
                                const float v01 = swiglu_r0[p][1] * sf_inv_r0;
                                const float v10 = swiglu_r1[p][0] * sf_inv_r1;
                                const float v11 = swiglu_r1[p][1] * sf_inv_r1;

                                if constexpr (kGemmOnly and kGemmOnlySkipL1FP8Pack) {
                                    const uint32_t r0_mix = (__float_as_uint(v00) & 0xffffu) ^ (__float_as_uint(v01) >> 16);
                                    const uint32_t r1_mix = (__float_as_uint(v10) & 0xffffu) ^ (__float_as_uint(v11) >> 16);
                                    r0_bits[q] = valid_r0 ? static_cast<uint16_t>(r0_mix) : 0u;
                                    r1_bits[q] = valid_r1 ? static_cast<uint16_t>(r1_mix) : 0u;
                                } else {
                                    const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                                    const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                                    r0_bits[q] = valid_r0 ? r0_pair.__x : 0u;
                                    r1_bits[q] = valid_r1 ? r1_pair.__x : 0u;
                                }
                            }

                            const uint32_t row_lane_base = lane_idx & ~3u;
                            const uint32_t src_pair = col_idx & 1u;
                            const uint32_t src_lane_0 = row_lane_base + src_pair * 2u;
                            const uint32_t src_lane_1 = src_lane_0 + 1u;
                            const bool use_hi_chunk = col_idx >= 2u;
                            const uint32_t r0_q0 = static_cast<uint32_t>(r0_bits[0]);
                            const uint32_t r0_q1 = static_cast<uint32_t>(r0_bits[1]);
                            const uint32_t r1_q0 = static_cast<uint32_t>(r1_bits[0]);
                            const uint32_t r1_q1 = static_cast<uint32_t>(r1_bits[1]);
                            const uint32_t r0_lo_q0 = __shfl_sync(0xffffffffu, r0_q0, src_lane_0);
                            const uint32_t r0_hi_q0 = __shfl_sync(0xffffffffu, r0_q0, src_lane_1);
                            const uint32_t r1_lo_q0 = __shfl_sync(0xffffffffu, r1_q0, src_lane_0);
                            const uint32_t r1_hi_q0 = __shfl_sync(0xffffffffu, r1_q0, src_lane_1);
                            const uint32_t r0_lo_q1 = __shfl_sync(0xffffffffu, r0_q1, src_lane_0);
                            const uint32_t r0_hi_q1 = __shfl_sync(0xffffffffu, r0_q1, src_lane_1);
                            const uint32_t r1_lo_q1 = __shfl_sync(0xffffffffu, r1_q1, src_lane_0);
                            const uint32_t r1_hi_q1 = __shfl_sync(0xffffffffu, r1_q1, src_lane_1);
                            const uint32_t r0_lo = use_hi_chunk ? r0_lo_q1 : r0_lo_q0;
                            const uint32_t r0_hi = use_hi_chunk ? r0_hi_q1 : r0_hi_q0;
                            const uint32_t r1_lo = use_hi_chunk ? r1_lo_q1 : r1_lo_q0;
                            const uint32_t r1_hi = use_hi_chunk ? r1_hi_q1 : r1_hi_q0;
                            r0_stsm = (r0_lo & 0xffffu) | ((r0_hi & 0xffffu) << 16);
                            r1_stsm = (r1_lo & 0xffffu) | ((r1_hi & 0xffffu) << 16);
                        };

                        uint32_t r0_stsm_0, r1_stsm_0, r0_stsm_1, r1_stsm_1;
                        make_stsm_pair(p_base, r0_stsm_0, r1_stsm_0);
                        make_stsm_pair(p_base + 2, r0_stsm_1, r1_stsm_1);

                        constexpr uint32_t kL1OutSwizzleBytes = WG_L1_OUT_BLOCK_N;
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto get_stsm_ptr = [&](const uint32_t in_atom_offset) {
                            const uint32_t bank_group_index =
                                in_atom_offset + lane_idx * (kL1OutSwizzleBytes / kNumBankGroupBytes);
                            uint32_t row = bank_group_index / 8;
                            uint32_t col = bank_group_index % 8;
                            col ^= row % (kL1OutSwizzleBytes / kNumBankGroupBytes);
                            return reinterpret_cast<uint8_t*>(smem_cd_l1_wg)
                                + warp_idx_in_wg * 16 * kL1OutSwizzleBytes
                                + row * (kNumBankGroupBytes * 8)
                                + col * kNumBankGroupBytes;
                        };
                        ptx::SM90_U32x2_STSM_N<uint32_t>::copy(
                            r0_stsm_0, r1_stsm_0, get_stsm_ptr(p_base / 2));
                        ptx::SM90_U32x2_STSM_N<uint32_t>::copy(
                            r0_stsm_1, r1_stsm_1, get_stsm_ptr(p_base / 2 + 1));
                    }
                } else {
                #pragma unroll
                for (uint32_t p_base = 0; p_base < kNumPairs; p_base += 2) {
                    uint16_t r0_bits[2], r1_bits[2];
                    #pragma unroll
                    for (uint32_t q = 0; q < 2; ++ q) {
                        const uint32_t p = p_base + q;
                        const float v00 = swiglu_r0[p][0] * sf_inv_r0;
                        const float v01 = swiglu_r0[p][1] * sf_inv_r0;
                        const float v10 = swiglu_r1[p][0] * sf_inv_r1;
                        const float v11 = swiglu_r1[p][1] * sf_inv_r1;

                        if constexpr (kGemmOnly and kGemmOnlySkipL1FP8Pack) {
                            const uint32_t r0_mix = (__float_as_uint(v00) & 0xffffu) ^ (__float_as_uint(v01) >> 16);
                            const uint32_t r1_mix = (__float_as_uint(v10) & 0xffffu) ^ (__float_as_uint(v11) >> 16);
                            r0_bits[q] = valid_r0 ? static_cast<uint16_t>(r0_mix) : 0u;
                            r1_bits[q] = valid_r1 ? static_cast<uint16_t>(r1_mix) : 0u;
                        } else {
                            const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                            const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                            r0_bits[q] = valid_r0 ? r0_pair.__x : 0u;
                            r1_bits[q] = valid_r1 ? r1_pair.__x : 0u;
                        }
                    }

                    if constexpr (kGemmOnly and kGemmOnlyDirectL1GlobalStore) {
                        #pragma unroll
                        for (uint32_t q = 0; q < 2; ++ q) {
                            const uint32_t r0_buddy = __shfl_xor_sync(
                                0xffffffffu, static_cast<uint32_t>(r0_bits[q]), 1);
                            const uint32_t r1_buddy = __shfl_xor_sync(
                                0xffffffffu, static_cast<uint32_t>(r1_bits[q]), 1);
                            const uint32_t r0_packed =
                                (static_cast<uint32_t>(r0_bits[q]) & 0xffffu) | ((r0_buddy & 0xffffu) << 16);
                            const uint32_t r1_packed =
                                (static_cast<uint32_t>(r1_bits[q]) & 0xffffu) | ((r1_buddy & 0xffffu) << 16);

                            if ((col_idx & 1u) == 0) {
                                const uint32_t col = (p_base + q) * 8 + col_idx * 2;
                                const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset + col;
                                if (valid_r0) {
                                    auto* g0 = reinterpret_cast<uint32_t*>(
                                        l2_token_buffer.get_data_buffer(m_idx + row_offset_r0)
                                            .get_base_ptr<uint8_t>() + out_n_idx);
                                    *g0 = r0_packed;
                                }
                                if (valid_r1) {
                                    auto* g1 = reinterpret_cast<uint32_t*>(
                                        l2_token_buffer.get_data_buffer(m_idx + row_offset_r1)
                                            .get_base_ptr<uint8_t>() + out_n_idx);
                                    *g1 = r1_packed;
                                }
                            }
                        }
                    } else if constexpr (kGemmOnly and kGemmOnlySkipL1SmemStore) {
                        const uint32_t r0_sink =
                            (static_cast<uint32_t>(r0_bits[0]) & 0xffffu) |
                            ((static_cast<uint32_t>(r0_bits[1]) & 0xffffu) << 16);
                        const uint32_t r1_sink =
                            (static_cast<uint32_t>(r1_bits[0]) & 0xffffu) |
                            ((static_cast<uint32_t>(r1_bits[1]) & 0xffffu) << 16);
                        asm volatile("" :: "r"(r0_sink), "r"(r1_sink));
                    } else if constexpr (not kGemmOnly and kL1U64SmemStore) {
                        #pragma unroll
                        for (uint32_t q = 0; q < 2; ++ q) {
                            const uint32_t row_lane_base = lane_idx & ~3u;
                            const uint32_t p = p_base + q;
                            const uint32_t col = p * 8;
                            const uint32_t r0_self = static_cast<uint32_t>(r0_bits[q]) & 0xffffu;
                            const uint32_t r0_c1 = __shfl_sync(0xffffffffu, r0_self, row_lane_base + 1u);
                            const uint32_t r0_c2 = __shfl_sync(0xffffffffu, r0_self, row_lane_base + 2u);
                            const uint32_t r0_c3 = __shfl_sync(0xffffffffu, r0_self, row_lane_base + 3u);
                            if (col_idx == 0) {
                                auto* p0 = smem_cd_l1_wg + r_0 * WG_L1_OUT_BLOCK_N + col;
                                if (valid_r0)
                                    ptx::st_shared(p0, r0_self | (r0_c1 << 16), r0_c2 | (r0_c3 << 16));
                            }

                            const uint32_t r1_self = static_cast<uint32_t>(r1_bits[q]) & 0xffffu;
                            const uint32_t r1_c1 = __shfl_sync(0xffffffffu, r1_self, row_lane_base + 1u);
                            const uint32_t r1_c2 = __shfl_sync(0xffffffffu, r1_self, row_lane_base + 2u);
                            const uint32_t r1_c3 = __shfl_sync(0xffffffffu, r1_self, row_lane_base + 3u);
                            if (col_idx == 0) {
                                auto* p1 = smem_cd_l1_wg + r_1 * WG_L1_OUT_BLOCK_N + col;
                                if (valid_r1)
                                    ptx::st_shared(p1, r1_self | (r1_c1 << 16), r1_c2 | (r1_c3 << 16));
                            }
                        }
                    } else if constexpr (not kGemmOnly and kPackedL1SmemStore) {
                        #pragma unroll
                        for (uint32_t q = 0; q < 2; ++ q) {
                            const uint32_t r0_buddy = __shfl_xor_sync(
                                0xffffffffu, static_cast<uint32_t>(r0_bits[q]), 1);
                            const uint32_t r1_buddy = __shfl_xor_sync(
                                0xffffffffu, static_cast<uint32_t>(r1_bits[q]), 1);
                            const uint32_t r0_packed =
                                (static_cast<uint32_t>(r0_bits[q]) & 0xffffu) | ((r0_buddy & 0xffffu) << 16);
                            const uint32_t r1_packed =
                                (static_cast<uint32_t>(r1_bits[q]) & 0xffffu) | ((r1_buddy & 0xffffu) << 16);

                            if ((col_idx & 1u) == 0) {
                                const uint32_t p = p_base + q;
                                const uint32_t col = p * 8 + col_idx * 2;
                                auto* p0 = reinterpret_cast<uint32_t*>(
                                    smem_cd_l1_wg + r_0 * WG_L1_OUT_BLOCK_N + col);
                                auto* p1 = reinterpret_cast<uint32_t*>(
                                    smem_cd_l1_wg + r_1 * WG_L1_OUT_BLOCK_N + col);
                                if (valid_r0)
                                    ptx::st_shared(p0, r0_packed);
                                if (valid_r1)
                                    ptx::st_shared(p1, r1_packed);
                            }
                        }
                    } else if constexpr (not kGemmOnly and not kL1StsmX2) {
                        #pragma unroll
                        for (uint32_t q = 0; q < 2; ++ q) {
                            const uint32_t p = p_base + q;
                            const uint32_t col = p * 8 + col_idx * 2;
                            auto* p0 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1_wg + r_0 * WG_L1_OUT_BLOCK_N + col);
                            auto* p1 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1_wg + r_1 * WG_L1_OUT_BLOCK_N + col);
                            if (valid_r0)
                                *p0 = r0_bits[q];
                            if (valid_r1)
                                *p1 = r1_bits[q];
                        }
                    } else {
                        const uint32_t row_lane_base = lane_idx & ~3u;
                        const uint32_t src_pair = col_idx & 1u;
                        const uint32_t src_lane_0 = row_lane_base + src_pair * 2u;
                        const uint32_t src_lane_1 = src_lane_0 + 1u;
                        const bool use_hi_chunk = col_idx >= 2u;
                        const uint32_t r0_lo = __shfl_sync(
                            0xffffffffu, static_cast<uint32_t>(r0_bits[use_hi_chunk]), src_lane_0);
                        const uint32_t r0_hi = __shfl_sync(
                            0xffffffffu, static_cast<uint32_t>(r0_bits[use_hi_chunk]), src_lane_1);
                        const uint32_t r1_lo = __shfl_sync(
                            0xffffffffu, static_cast<uint32_t>(r1_bits[use_hi_chunk]), src_lane_0);
                        const uint32_t r1_hi = __shfl_sync(
                            0xffffffffu, static_cast<uint32_t>(r1_bits[use_hi_chunk]), src_lane_1);
                        const uint32_t r0_stsm = (r0_lo & 0xffffu) | ((r0_hi & 0xffffu) << 16);
                        const uint32_t r1_stsm = (r1_lo & 0xffffu) | ((r1_hi & 0xffffu) << 16);
                        constexpr uint32_t kL1OutSwizzleBytes = WG_L1_OUT_BLOCK_N;
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        const uint32_t in_atom_offset = p_base / 2;
                        const uint32_t bank_group_index =
                            in_atom_offset + lane_idx * (kL1OutSwizzleBytes / kNumBankGroupBytes);
                        uint32_t row = bank_group_index / 8;
                        uint32_t col = bank_group_index % 8;
                        col ^= row % (kL1OutSwizzleBytes / kNumBankGroupBytes);
                        auto* smem_ptr = reinterpret_cast<uint8_t*>(smem_cd_l1_wg)
                            + warp_idx_in_wg * 16 * kL1OutSwizzleBytes
                            + row * (kNumBankGroupBytes * 8)
                            + col * kNumBankGroupBytes;
                        ptx::SM90_U32x2_STSM_N<uint32_t>::copy(r0_stsm, r1_stsm, smem_ptr);
                    }
                }
                }
                }

                // Write SF as float at `[token, n_block_idx]` in L2 acts SF buffer (per-64 layout).
                // Each row is contributed by lanes col_idx ∈ {0..3}; only col_idx == 0 writes.
                if constexpr (not (kGemmOnly and kGemmOnlyFixedL1Scale)) {
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        // SF buffer is (kNumPaddedSFPoolTokens × kIntermediateHidden/64), MN-major:
                        //   addr[k_idx * num_padded_sf_pool_tokens + token_idx]
                        const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                        const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                        const uint32_t k_sf_idx = sf_n_block_idx;  // one per-64 SF per L1 WG tile
                        if (valid_r0)
                            sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
                        if (valid_r1)
                            sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
                    }
                }

                if constexpr (kGemmOnly and kGemmOnlyDirectL1GlobalStore and
                              not kGemmOnlyDirectL1GlobalNoFence) {
                    __threadfence();
                }

                if constexpr (not (kGemmOnly and (kGemmOnlySkipL1OutputStore or kGemmOnlySkipL1SmemStore or kGemmOnlyDirectL1GlobalStore))) {
                    if constexpr (kL1WarpTMAStore) {
                        __syncwarp();
                        if (cute::elect_one_sync()) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset;
                            const uint32_t warp_row_offset = warp_idx_in_wg * kL1WarpTMAStoreRows;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1 + smem_cd_l1_wg_offset + warp_row_offset * WG_L1_OUT_BLOCK_N,
                                out_n_idx,
                                m_idx + row_base + warp_row_offset);
                            cute::tma_store_arrive();
                            ptx::tma_store_wait<0>();
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)), 1);
                        }
                    } else {
                        // Sync the warpgroup before TMA store
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Issue TMA store of the entire tile. Padding rows beyond
                        // `valid_m` are written with stale/garbage FP8 to the L1-output
                        // pool buffer, but they are never consumed downstream: the L2
                        // GEMM tile loads them, but its NVLink-scatter epilogue is
                        // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                        // padding rows can produce NaN accumulators that simply stay
                        // in registers (only valid rows are converted to BF16 and
                        // STSM'd into smem). Using TMA for partial tiles is a large
                        // win for low-batch / decode where every tile is partial.
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1 + smem_cd_l1_wg_offset,
                                out_n_idx,
                                m_idx + row_base);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                        if constexpr (kL1TMAStoreWaitIssuerOnly) {
                            if (warp_idx_in_wg == 0 and cute::elect_one_sync())
                                ptx::tma_store_wait<0>();
                            if constexpr (kL2ArrivalCounter)
                                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        } else {
                            ptx::tma_store_wait<0>();
                        }
                    }
                }

                // Notify L2 that this L1 output (and SF) is ready. The counter
                // mode lets each WG publish its own 64x64 slice and avoids the
                // 512-thread CTA barrier that is otherwise needed before the
                // single bit-mask update.
                if constexpr (kL2ArrivalCounter) {
                    if constexpr (not kL1WarpTMAStore) {
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)), 1);
                        }
                    }
                } else {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_or_rel_gpu(
                            workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                            1ull << n_block_idx);
                    }
                }
                __syncwarp();
            } else {
                if constexpr (kGemmOnly and kGemmOnlySkipL2OutputStore) {
                    if (epilogue_thread_idx == 0 and cumulative_local_expert_recv_stats != nullptr)
                        cumulative_local_expert_recv_stats[0] = __float_as_int(final_accum[0]);
                    if constexpr (not kSkipL2EpilogueSync)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    return;
                }

                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                const uint32_t lane_in_row = lane_idx % 16;
                const uint32_t cols_per_lane = WG_BLOCK_N / 16;

	                uint32_t scatter_sink = 0;

		                if constexpr (kDirectL2RegisterScatter) {
		                    DG_STATIC_ASSERT(not kDirectL2RegisterScatter,
		                                     "Direct L2 register scatter is disabled: current prototypes either spill badly or hang.");
		                    DG_STATIC_ASSERT(WG_BLOCK_N == 128,
		                                     "Direct L2 register scatter assumes 16 lanes x 8 BF16 columns");
	                    const uint32_t source_lane_base = lane_idx & ~3u;
	                    const uint32_t row_in_wg_r0 = warp_idx_in_wg * 16 + row_idx;
	                    const uint32_t row_in_wg_r1 = row_in_wg_r0 + 8;
	                    #pragma unroll
	                    for (uint32_t chunk = 0; chunk < WG_BLOCK_N / 8; ++ chunk) {
	                        const uint32_t r0_pair = valid_r0 ? math::cast_into_bf16_and_pack(
	                            final_accum[chunk*4 + 0], final_accum[chunk*4 + 1]) : 0u;
	                        const uint32_t r1_pair = valid_r1 ? math::cast_into_bf16_and_pack(
	                            final_accum[chunk*4 + 2], final_accum[chunk*4 + 3]) : 0u;

	                        if (col_idx == 0) {
	                            const uint4 packed_r0 = {
	                                __shfl_sync(0xffffffffu, r0_pair, source_lane_base + 0),
	                                __shfl_sync(0xffffffffu, r0_pair, source_lane_base + 1),
	                                __shfl_sync(0xffffffffu, r0_pair, source_lane_base + 2),
	                                __shfl_sync(0xffffffffu, r0_pair, source_lane_base + 3)
	                            };
	                            const uint32_t m_idx_r0 = row_base + row_in_wg_r0;
	                            if (m_idx_r0 < valid_m) {
	                                if constexpr (kSkipL2Scatter) {
	                                    scatter_sink ^= packed_r0.x ^ packed_r0.y ^ packed_r0.z ^ packed_r0.w;
	                                } else if constexpr (kGemmOnly) {
	                                    const uint32_t pool_token_idx = m_idx + m_idx_r0;
	                                    const uint32_t dst_token_idx = pool_token_idx / kNumTopk;
	                                    const uint32_t dst_topk_idx = pool_token_idx - dst_token_idx * kNumTopk;
	                                    const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
	                                                           .get_data_buffer(dst_token_idx);
	                                    auto dst_ptr = math::advance_ptr<uint4>(
	                                        dst_token.get_base_ptr(),
	                                        (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + chunk * sizeof(uint4));
	                                    *dst_ptr = packed_r0;
	                                } else {
	                                    const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_r0);
	                                    const uint32_t dst_rank_idx = src_metadata.rank_idx;
	                                    const uint32_t dst_token_idx = src_metadata.token_idx;
	                                    const uint32_t dst_topk_idx = src_metadata.topk_idx;
	                                    const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
	                                                           .get_data_buffer(dst_token_idx);
	                                    auto dst_ptr = math::advance_ptr<uint4>(
	                                        dst_token.get_base_ptr(),
	                                        (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + chunk * sizeof(uint4));
	                                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed_r0;
	                                }
	                            }

	                            const uint4 packed_r1 = {
	                                __shfl_sync(0xffffffffu, r1_pair, source_lane_base + 0),
	                                __shfl_sync(0xffffffffu, r1_pair, source_lane_base + 1),
	                                __shfl_sync(0xffffffffu, r1_pair, source_lane_base + 2),
	                                __shfl_sync(0xffffffffu, r1_pair, source_lane_base + 3)
	                            };
	                            const uint32_t m_idx_r1 = row_base + row_in_wg_r1;
	                            if (m_idx_r1 < valid_m) {
	                                if constexpr (kSkipL2Scatter) {
	                                    scatter_sink ^= packed_r1.x ^ packed_r1.y ^ packed_r1.z ^ packed_r1.w;
	                                } else if constexpr (kGemmOnly) {
	                                    const uint32_t pool_token_idx = m_idx + m_idx_r1;
	                                    const uint32_t dst_token_idx = pool_token_idx / kNumTopk;
	                                    const uint32_t dst_topk_idx = pool_token_idx - dst_token_idx * kNumTopk;
	                                    const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
	                                                           .get_data_buffer(dst_token_idx);
	                                    auto dst_ptr = math::advance_ptr<uint4>(
	                                        dst_token.get_base_ptr(),
	                                        (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + chunk * sizeof(uint4));
	                                    *dst_ptr = packed_r1;
	                                } else {
	                                    const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_r1);
	                                    const uint32_t dst_rank_idx = src_metadata.rank_idx;
	                                    const uint32_t dst_token_idx = src_metadata.token_idx;
	                                    const uint32_t dst_topk_idx = src_metadata.topk_idx;
	                                    const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
	                                                           .get_data_buffer(dst_token_idx);
	                                    auto dst_ptr = math::advance_ptr<uint4>(
	                                        dst_token.get_base_ptr(),
	                                        (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + chunk * sizeof(uint4));
	                                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed_r1;
	                                }
	                            }
	                        }
	                    }
	                } else {
                    // STSM into smem_cd_l2 (BF16). Reuse SM100 column-swizzle layout.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                        // Each i consumes 8 floats (one 16x256b chunk in SM100 terms).
                        // For SM90 WGMMA layout, 8 floats per i correspond to 2 chunks of 4 floats:
                        //   final_accum[i*8 + (0..3)] = chunk 2i: (r0c0, r0c1, r1c0, r1c1)
                        //   final_accum[i*8 + (4..7)] = chunk 2i+1: same shape
                        const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;

                        // Write to SMEM at appropriate position
                        // Row r_0 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r0_lo
                        // Row r_0 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r0_hi
                        // Row r_1 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r1_lo
                        // Row r_1 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r1_hi
                        auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                            auto smem_ptr = smem_cd_l2
                                + smem_cd_l2_wg_offset
                                + row * WG_BLOCK_N
                                + col;
                            // BF16 STS: 2 bf16 elements
                            *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
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

                    // Each warp writes and then scatters only its own 16-row
                    // slice, so a warp-level fence is enough before reading
                    // back from shared memory.
                    __syncwarp();

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows × 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp → 1 row.
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_base + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        // Read 8 BF16s (= 16 bytes = 1 uint4) from smem
                        auto smem_ptr = smem_cd_l2
                            + smem_cd_l2_wg_offset
                            + row_in_wg * WG_BLOCK_N
                            + lane_in_row * cols_per_lane;
                        const auto packed = *reinterpret_cast<uint4*>(smem_ptr);

                        if constexpr (kSkipL2Scatter) {
                            scatter_sink ^= packed.x ^ packed.y ^ packed.z ^ packed.w;
                        } else if constexpr (kGemmOnly) {
                            const uint32_t pool_token_idx = m_idx + m_idx_in_block;
                            const uint32_t dst_token_idx = pool_token_idx / kNumTopk;
                            const uint32_t dst_topk_idx = pool_token_idx - dst_token_idx * kNumTopk;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *dst_ptr = packed;
                        } else {
                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            const uint32_t dst_token_idx = src_metadata.token_idx;
                            const uint32_t dst_topk_idx = src_metadata.topk_idx;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                }
                if constexpr (kSkipL2Scatter)
                    asm volatile("" :: "r"(scatter_sink));

                if constexpr (not kSkipL2EpilogueSync)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        };

        if constexpr (kSplitPhaseHotPath) {
            scheduler.for_each_block_split(
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
        } else {
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                process_math_block(block_phase, local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
            });
        }

        if constexpr (kGemmOnly)
            return;

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state, or skip only the zero
        // stores while preserving the cross-rank launch protocol.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        if constexpr (kSkipCombine)
            return;

        if (epilogue_warp_idx >= kNumCombineWarps)
            return;

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kDefaultNumChunks =
            (kNumChunkSlots * kNumCombineWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
             and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
        // Flash-style hidden=7168 is 7 * 1024. Splitting combine into 7 chunks
        // keeps each lane's BF16 reduce accumulator much smaller without
        // violating the 32-lane uint4 mapping.
        constexpr uint32_t kSplitMNNumChunks = (kHidden % 7 == 0) ? 7 : (kHidden >= 1024 ? 4 : 1);
        constexpr uint32_t kNumChunks = kSplitMNWarpgroups ? kSplitMNNumChunks : kDefaultNumChunks;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumCombineWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_TRAP_ONLY_DEVICE_ASSERT(kNumChunkSlots * kNumCombineWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

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
        for (uint32_t token_idx = sm_idx * kNumCombineWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumCombineWarps) {
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
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
