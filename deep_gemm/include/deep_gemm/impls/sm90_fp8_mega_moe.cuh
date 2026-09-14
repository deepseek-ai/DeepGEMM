#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/sm90_mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
namespace deep_gemm {

// Pipeline (cluster=1, no TMA multicast):
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * One A+SFA producer warp and one B producer warp feed the pipeline.
//   * Math warpgroups consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout,
//         per-row amax over the 64 post-SwiGLU columns of this block, FP8 e4m3
//         quantize, STSM into SMEM, TMA store to local L1 output buffer.
//         The per-row SF is written as a *float* into the L2-acts SF buffer at
//         per-64 K granularity (one SF per L1 N block), so each block is fully
//         self-contained and no cross-CTA amax synchronisation is needed.
//       - L2 (Linear2): BF16 cast of the GEMM output, STSM into SMEM, then
//         NVLink scatter to remote combine buffers.
//   * After all GEMM blocks, the math warps run the COMBINE step using the
//     same BF16 top-k reduction scheme as the SM100 path.

enum class MegaMoEPhaseKind {
    Linear1,
    Linear2
};

__forceinline__ __device__ void sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    const auto scaled = make_float2(
        __fmul_rn(amax.x, kScale), __fmul_rn(amax.y, kScale));
    const auto exp_x = math::fast_log2_ceil(scaled.x);
    const auto exp_y = math::fast_log2_ceil(scaled.y);
    sf.x = math::fast_pow2(exp_x), sf_inv.x = math::fast_pow2(-exp_x);
    sf.y = math::fast_pow2(exp_y), sf_inv.y = math::fast_pow2(-exp_y);
}

template <MegaMoEPhaseKind kKind,
          bool kNMajorScheduleRequested,
          bool kDirectL2ScatterRequested = false,
          bool kOneWarpCleanupRequested = false>
struct MegaMoEPhasePolicy {
    static constexpr bool runs_linear1 = kKind == MegaMoEPhaseKind::Linear1;
    static constexpr bool runs_linear2 = kKind == MegaMoEPhaseKind::Linear2;
    static constexpr bool nmajor_schedule = kNMajorScheduleRequested;
    static constexpr bool direct_l2_scatter =
        runs_linear2 and kDirectL2ScatterRequested;
    static constexpr bool one_warp_cleanup =
        runs_linear2 and kOneWarpCleanupRequested;

    template <typename Scheduler, typename Func>
    CUTLASS_DEVICE static void for_each_selected_block(Scheduler& scheduler, Func&& func) {
        if constexpr (runs_linear1) {
            scheduler.template for_each_phase_block<sched::SM90BlockPhase::Linear1>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                });
        } else {
            scheduler.template for_each_phase_block<sched::SM90BlockPhase::Linear2>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
                });
        }
    }
};

#define DG_SM90_FP8_MOE_TEMPLATE_PARAMS \
    uint32_t kNumMaxTokensPerRank, \
    uint32_t kHidden, uint32_t kIntermediateHidden, \
    uint32_t kNumExperts, uint32_t kNumTopk, \
    uint32_t kNumExpertsPerWave, \
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, \
    uint32_t kNumMaxPoolTokens, \
    uint32_t kNumPaddedSFPoolTokens, \
    uint32_t kSFPoolStrideTokens, \
    uint32_t kNumStages, \
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads, \
    uint32_t kNumEpilogueThreads, \
    uint32_t kNumSMs, uint32_t kNumRanks, \
    float kActivationClamp, \
    float kActivationAlpha, \
    float kActivationUpBias, \
    bool kFastMath, \
    bool kFP8SwapAB = false, \
    bool kBF16ScaledAccumRequested = false

#define DG_SM90_FP8_MOE_KERNEL_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights, \
    const float* __restrict__ l1_weights_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights, \
    const float* __restrict__ l2_weights_sf

#define DG_SM90_FP8_MOE_CORE_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const layout::SymBuffer<kNumRanks>& sym_buffer, \
    const cute::TmaDescriptor& tensor_map_l1_acts, \
    const cute::TmaDescriptor& tensor_map_l1_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l1_weights, \
    const float* __restrict__ l1_weights_sf, \
    const cute::TmaDescriptor& tensor_map_l1_output, \
    const cute::TmaDescriptor& tensor_map_l2_acts, \
    const cute::TmaDescriptor& tensor_map_l2_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l2_weights, \
    const float* __restrict__ l2_weights_sf

#define DG_SM90_FP8_MOE_KERNEL_ARGS \
    y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer, \
    tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, \
    l1_weights_sf, tensor_map_l1_output, tensor_map_l2_acts, \
    tensor_map_l2_acts_sf, tensor_map_l2_weights, l2_weights_sf

#define DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS \
    kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk, \
    kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, kNumMaxPoolTokens, \
    kNumPaddedSFPoolTokens, kSFPoolStrideTokens, kNumStages, kNumDispatchThreads, \
    kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, \
    kActivationClamp, kActivationAlpha, kActivationUpBias, kFastMath, \
    kFP8SwapAB, kBF16ScaledAccumRequested

template <typename MegaMoEPhase, DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_DEVICE void
sm90_fp8_mega_moe_core(DG_SM90_FP8_MOE_CORE_ARGS_DECL) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
    constexpr uint32_t L1_SHAPE_K = kHidden;
    constexpr uint32_t L2_SHAPE_N = kHidden;
    constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
    constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads == 64 or kNumDispatchThreads % 128 == 0,
                     "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64 or kNumNonEpilogueThreads == 128,
                     "Invalid number of GEMM TMA warps (2 or 4 warps expected)");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0,
                     "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N == 64 or BLOCK_N == 128 or BLOCK_N == 256 or
                     BLOCK_N == 512,
                     "BLOCK_N must be 64/128/256/512 for this SM90 path");
    DG_STATIC_ASSERT(BLOCK_K == 128 or BLOCK_K == 256,
                     "BLOCK_K must be 128 or 256");
    DG_STATIC_ASSERT(kHidden % BLOCK_K == 0 and kIntermediateHidden % BLOCK_K == 0,
                     "GEMM K dimensions must be divisible by BLOCK_K");
    DG_STATIC_ASSERT(BLOCK_K % 128 == 0,
                     "BLOCK_K must comprise whole BK128 TMA tiles");
    DG_STATIC_ASSERT(BLOCK_N <= 512,
                     "BLOCK_N exceeds the two-box TMA staging contract");

    // Thread / warp identification
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Prefetch the TMA descriptors used by this split phase.
    if (warp_idx == 0 and cute::elect_one_sync()) {
        if constexpr (MegaMoEPhase::runs_linear1) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
            cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        }
        if constexpr (MegaMoEPhase::runs_linear2) {
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        }
    }

    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank,
        kNumTopk, kNumMaxPoolTokens);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // MN-major pool SFs keep a TMA-aligned 128-token outer stride even when
    // one logical token's SF byte count is not 16-byte aligned.
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32, false);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 16, false);
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
    DG_STATIC_ASSERT(kSFPoolStrideTokens <= kNumPaddedSFPoolTokens,
                     "Logical SF pool stride must fit in the allocated SF pool capacity");

    // Combine input area stores BF16 L2 contributions for the final reduction.
    constexpr uint32_t kCombineElementBytes = sizeof(nv_bfloat16);
    constexpr auto combine_token_layout = layout::Data(kHidden * kCombineElementBytes);
    const auto combine_token_buffer = layout::Buffer(
        combine_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // GEMM data types and shape constants
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    constexpr auto kWarpgroupLayout = layout::get_sm90_moe_warpgroup_layout(
        BLOCK_M, BLOCK_N, kNumEpilogueWarpgroups);
    constexpr bool kSplitNWarpgroups = kWarpgroupLayout.split_n;
    constexpr uint32_t kWarpgroupSplitM = kWarpgroupLayout.split_m;
    constexpr uint32_t kWarpgroupSplitN = kWarpgroupLayout.split_n_count;
    constexpr uint32_t WG_BLOCK_M = kWarpgroupLayout.block_m;
    constexpr uint32_t WG_BLOCK_N = kWarpgroupLayout.block_n;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr bool kSwapABActive =
        kFP8SwapAB and kSplitNWarpgroups and BLOCK_M == 64 and BLOCK_N == 128 and
        kNumEpilogueWarpgroups == 2;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    DG_STATIC_ASSERT(not kSwapABActive or (BLOCK_M % 8 == 0),
                     "swapAB epilogue token chunks assume BLOCK_M is a multiple of 8");
    constexpr bool kDirectL2Scatter =
        (!kSwapABActive) && MegaMoEPhase::direct_l2_scatter && WG_BLOCK_N == 128;
    constexpr bool kBF16ScaledAccum = kBF16ScaledAccumRequested;
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT((!kSplitNWarpgroups) or
                     (BLOCK_M == 64 and (WG_BLOCK_N == 64 or WG_BLOCK_N == 128)),
                     "Split-N path expects M64N64 or M64N128 WGMMA consumers");

    // SM90 MegaMoE uses one CTA per work item; A and B are CTA-local.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = 128;
    constexpr uint32_t kSwizzleBMode   = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = 64;           // L2 acts SF (per-64 K, SM90 only)
    constexpr uint32_t kTMATileK       = 128;
    constexpr uint32_t kNumTMATilesPerStage = BLOCK_K / kTMATileK;
    constexpr uint32_t kTMATileN = BLOCK_N > 256 ? 256 : BLOCK_N;
    constexpr uint32_t kNumTMANTilesPerStage = BLOCK_N / kTMATileN;
    DG_STATIC_ASSERT(BLOCK_N % kTMATileN == 0,
                     "BLOCK_N must comprise whole TMA N boxes");

    // Shared memory layout
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Combine reuses the pre-barrier SMEM region, so split L2 keeps this
    // dispatch scratch capacity even though it does not run dispatch pull.
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // SFA holds one aligned BLOCK_M-float vector per 64 channels. L1 uses every
    // other vector (per-128); L2 uses all of them (per-64).
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t kNumL2SFAKGroups = BLOCK_K / kL2ActsSFGranK;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        kNumL2SFAKGroups * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 contribution.
    constexpr uint32_t SMEM_CD_L1_SIZE = MegaMoEPhase::runs_linear1 ?
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_L2_SIZE = (!MegaMoEPhase::runs_linear2 || kDirectL2Scatter) ? 0u :
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_BLOCK_N * kCombineElementBytes;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(float) : 0u;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        kSwapABActive ? (SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_WITH_SWAP_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE > SMEM_CD_SWAP_L1_SIZE ?
            SMEM_CD_OUTPUT_BASE_SIZE : SMEM_CD_SWAP_L1_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_WITH_SWAP_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_OUTPUT_SIZE;

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

    constexpr uint32_t kCombineInputHiddenBytes = kHidden * kCombineElementBytes;
    constexpr uint32_t kCombineOutputHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kCombineNumChunks = layout::get_sm90_moe_combine_num_chunks(
        kHidden, kNumEpilogueWarps, SMEM_BEFORE_BARRIER_SIZE,
        kCombineElementBytes, sizeof(nv_bfloat16));
    constexpr uint32_t kCombineChunkElems = kHidden / kCombineNumChunks;
    constexpr uint32_t kCombineInputChunkBytes =
        kCombineInputHiddenBytes / kCombineNumChunks;
    constexpr uint32_t kCombineOutputChunkBytes =
        kCombineOutputHiddenBytes / kCombineNumChunks;
    constexpr uint32_t SMEM_COMBINE_ALIAS_SIZE = MegaMoEPhase::runs_linear2
        ? kNumEpilogueWarps *
            (2 * kCombineInputChunkBytes + kCombineOutputChunkBytes) : 0u;
    DG_STATIC_ASSERT(kHidden % kCombineNumChunks == 0, "Hidden must be divisible by number of combine chunks");
    DG_STATIC_ASSERT(layout::is_sm90_moe_combine_vectorization_legal(
                         kHidden, kNumEpilogueWarps, SMEM_BEFORE_BARRIER_SIZE,
                         kCombineElementBytes, sizeof(nv_bfloat16)),
                     "Hidden is incompatible with the combine vectorization");
    DG_STATIC_ASSERT(SMEM_COMBINE_ALIAS_SIZE <= SMEM_BEFORE_BARRIER_SIZE,
                     "Combine SMEM alias exceeds the pre-barrier scratch region");

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    auto smem_cd_base = smem_gemm_base;
    // CD output is shared by L1 and L2; reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_cd_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_cd_base, SMEM_CD_SWAP_L1_FP32_SIZE));

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
    // Barriers live after the activation-SF stages.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });

    // Initialization
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
                // Producer arrivals: A(+SFA) + B.
                full_barriers[i]->init(2);
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            if constexpr (MegaMoEPhase::runs_linear2) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                    combine_barriers[i]->init(1);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Scheduler (cluster=1)
    auto scheduler = sched::SM90MegaMoESchedulerAdapter<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave, kNumSMs, kNumRanks,
        MegaMoEPhase::runs_linear2 and MegaMoEPhase::nmajor_schedule,
        MegaMoEPhase::runs_linear1 and MegaMoEPhase::nmajor_schedule>(workspace);

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

    // Register allocations keep each launch within the 64K CTA budget. The
    // four-warpgroup split-N path uses smaller front-end allocations.
    constexpr uint32_t kNumDispatchRegisters    = kNumEpilogueThreads == 512 ? 32 : 48;
    constexpr bool kCompactFrontendWarpgroup = (kNumDispatchWarps == 2 and kNumMMANonEpilogueWarps == 2);
    constexpr uint32_t kNumNonEpilogueRegisters = kNumEpilogueThreads == 512 ? 24 :
        (kCompactFrontendWarpgroup ? kNumDispatchRegisters : 40);
    constexpr uint32_t kNumEpilogueRegisters    =
        kNumEpilogueThreads == 512 ? 112 : 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    const auto for_each_selected_block = [&](auto&& func) {
        MegaMoEPhase::for_each_selected_block(scheduler, func);
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                if constexpr (!MegaMoEPhase::one_warp_cleanup)
                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if constexpr (MegaMoEPhase::one_warp_cleanup) {
                    if (warp_idx == 0) {
                        if (lane_idx == 0) {
                            *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                            if (cumulative_local_expert_recv_stats != nullptr)
                                ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        }
                    }
                } else {
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }
                }

                if constexpr (MegaMoEPhase::one_warp_cleanup) {
                    if (warp_idx == 0) {
                        for (uint32_t j = lane_idx; j < num_recv_m_blocks; j += 32) {
                            *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                        }
                        __syncwarp();
                    }
                } else {
                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                        *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                    }
                    __syncwarp();
                }
            }
        }
    };

    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        if constexpr (MegaMoEPhase::runs_linear2) {
            scheduler.fetch_expert_recv_count();
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            cleanup_workspace();
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag,
                comm::BarrierTimeoutPolicy::TrapOnly>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                true, false);
            return;
        }

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    const int expert_idx = static_cast<int>(
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

        comm::grid_sync<
            kNumSMs, kDispatchGridSyncIndex,
            comm::BarrierTimeoutPolicy::TrapOnly>(
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

        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumDispatchThreads,
            kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag,
            comm::BarrierTimeoutPolicy::TrapOnly>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);

        // Sync with epilogue warps before pulling tokens
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

            // TMA pull token data into SMEM
            if (cute::elect_one_sync()) {
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
            const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFFloats)
                    local_sf_ptr[j * kSFPoolStrideTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

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
                    workspace.get_l1_full_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        return;

    // ROLE 2: GEMM TMA load warps (A+SFA and B)
    // The first two non-epilogue warps issue the TMA loads. Four-warp
    // configurations keep the remaining two warps idle for register handoff.
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const auto tensor_map_a_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
            // the tail M unit when an expert has an odd number of M blocks.
            if (has_valid_m) {
                if (is_linear1_phase) {
                    const auto ptr = workspace.get_l1_full_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A. BK256 is two independently-swizzled BK128
                    // planes so each tensor-map box stays within one 128B atom.
                    #pragma unroll
                    for (uint32_t k_tile = 0; k_tile < kNumTMATilesPerStage; ++ k_tile) {
                        tma::copy<kTMATileK, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx],
                            smem_a[stage_idx] + k_tile * LOAD_BLOCK_M * kTMATileK,
                            k_idx + k_tile * kTMATileK, m_idx, 1);
                    }

                    // TMA load SFA with A on the same producer warp.
                    if (is_linear1_phase) {
                        // L1 SFA per-128: one vector per BK128 plane.
                        #pragma unroll
                        for (uint32_t sf_group = 0;
                             sf_group < BLOCK_K / kGranK; ++ sf_group) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] +
                                    sf_group * kL2SFAHalfStride,
                                m_idx,
                                k_block_idx * (BLOCK_K / kGranK) + sf_group,
                                1);
                        }
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE +
                                (BLOCK_K / kGranK) * BLOCK_M * sizeof(float));
                    } else {
                        // L2 SFA per-64: one TMA per scale group.
                        #pragma unroll
                        for (uint32_t sf_group = 0;
                             sf_group < kNumL2SFAKGroups; ++ sf_group) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] +
                                    sf_group * kL2SFAHalfStride,
                                m_idx,
                                k_block_idx * kNumL2SFAKGroups + sf_group,
                                1);
                        }
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE +
                                kNumL2SFAKGroups * BLOCK_M * sizeof(float));
                    }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const auto tensor_map_b_ptr =
                !is_linear1_phase ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = !is_linear1_phase ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load B in independent BK128 x BN<=256 planes. The
                    // tensor-map box limit requires two N planes for BN512.
                    #pragma unroll
                    for (uint32_t k_tile = 0; k_tile < kNumTMATilesPerStage; ++ k_tile) {
                        #pragma unroll
                        for (uint32_t n_tile = 0;
                             n_tile < kNumTMANTilesPerStage; ++ n_tile) {
                            tma::copy<kTMATileK, kTMATileN, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx],
                                smem_b[stage_idx] +
                                    k_tile * LOAD_BLOCK_N * kTMATileK +
                                    n_tile * kTMATileN * kTMATileK,
                                k_idx + k_tile * kTMATileK,
                                n_idx + n_tile * kTMATileN,
                                1);
                        }
                    }

                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
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
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if (lane_idx == 0)
                empty_barriers[s]->arrive();
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;
        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N = kSplitNWarpgroups ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;

        DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups, "Invalid warpgroup split");
        if constexpr (kSplitNWarpgroups) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Split WGs must each run one WGMMA tile per K-block");
        } else {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");
        }

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        for_each_selected_block([&](const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
            const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
            const uint32_t wg_n_idx = epilogue_wg_n_idx * WG_BLOCK_N;
            const uint32_t wg_l1_out_n_idx = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
            const uint32_t row_block_offset = epilogue_wg_m_idx * WG_BLOCK_M;
            const uint32_t smem_cd_l1_wg_offset = kSplitNWarpgroups ? 0 :
                epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N;
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;
            // GEMM
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;
            float final_accum[kAccumPerThread];
            if constexpr (not kBF16ScaledAccum) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                    final_accum[i] = 0.0f;
            }
            float accum[kAccumPerThread];

            const auto run_default_gemm_loop = [&]() {
                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert =
                    (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert =
                    (kHidden / 128) * kL2SFKBlocks;
                nv_bfloat162 swap_final_bf16[kAccumPerThread / 2];
                if constexpr (kBF16ScaledAccum and kSwapABActive) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i)
                        swap_final_bf16[i] = __float2bfloat162_rn(0.0f);
                }

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                     advance_pipeline(k_block_idx)) {
                float gate_sf = 0.0f, up_sf = 0.0f;
                float l2_sf_lo = 0.0f, l2_sf_hi = 0.0f;
                full_barriers[stage_idx]->wait(phase);

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                if (is_linear1_phase) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                } else {
                    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
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
                if (is_linear1_phase) {
                    const uint32_t gate_n = (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                    const uint32_t up_n   = kL1SFGateBlks + gate_n;
                    const float* base = l1_weights_sf +
                        local_expert_idx * kL1SFPerExpert + k_block_idx;
                    gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                    up_sf   = __ldg(base + up_n * kL1SFKBlocks);
                } else {
                    const uint32_t sf_n = (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                    const float* base = l2_weights_sf +
                        local_expert_idx * kL2SFPerExpert + k_block_idx;
                    l2_sf_lo = __ldg(base + sf_n * kL2SFKBlocks);
                    if constexpr (WG_BLOCK_N > 128)
                        l2_sf_hi = __ldg(base + (sf_n + 1u) * kL2SFKBlocks);
                    else
                        l2_sf_hi = l2_sf_lo;
                }

                if (is_linear1_phase) {
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
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * SwapWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + k * SwapWGMMA::K, 1);
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
                                if constexpr (kBF16ScaledAccum) {
                                    swap_final_bf16[2 * i] = __hfma2(
                                        __floats2bfloat162_rn(
                                            swap_accum[4 * i], swap_accum[4 * i + 1]),
                                        __floats2bfloat162_rn(
                                            scale_0 * gate_sf, scale_1 * gate_sf),
                                        swap_final_bf16[2 * i]);
                                    swap_final_bf16[2 * i + 1] = __hfma2(
                                        __floats2bfloat162_rn(
                                            swap_accum[4 * i + 2], swap_accum[4 * i + 3]),
                                        __floats2bfloat162_rn(
                                            scale_0 * up_sf, scale_1 * up_sf),
                                        swap_final_bf16[2 * i + 1]);
                                } else {
                                    final_accum[i * 4 + 0] +=
                                        scale_0 * gate_sf * swap_accum[i * 4 + 0];
                                    final_accum[i * 4 + 2] +=
                                        scale_0 * up_sf * swap_accum[i * 4 + 2];
                                    final_accum[i * 4 + 1] +=
                                        scale_1 * gate_sf * swap_accum[i * 4 + 1];
                                    final_accum[i * 4 + 3] +=
                                        scale_1 * up_sf * swap_accum[i * 4 + 3];
                                }
                            }

                            arrive_empty_barrier(stage_idx);
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
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

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
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_0) : 0.0f;
                                    const float scale_1 = token_1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_1) : 0.0f;
                                    if constexpr (kBF16ScaledAccum) {
                                        swap_final_bf16[2 * i] = __hfma2(
                                            __floats2bfloat162_rn(
                                                swap_accum[4 * i],
                                                swap_accum[4 * i + 1]),
                                            __floats2bfloat162_rn(
                                                scale_0 * l2_sf_lo,
                                                scale_1 * l2_sf_lo),
                                            swap_final_bf16[2 * i]);
                                        swap_final_bf16[2 * i + 1] = __hfma2(
                                            __floats2bfloat162_rn(
                                                swap_accum[4 * i + 2],
                                                swap_accum[4 * i + 3]),
                                            __floats2bfloat162_rn(
                                                scale_0 * l2_sf_lo,
                                                scale_1 * l2_sf_lo),
                                            swap_final_bf16[2 * i + 1]);
                                    } else {
                                        final_accum[i * 4 + 0] +=
                                            scale_0 * l2_sf_lo * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] +=
                                            scale_0 * l2_sf_lo * swap_accum[i * 4 + 2];
                                        final_accum[i * 4 + 1] +=
                                            scale_1 * l2_sf_lo * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] +=
                                            scale_1 * l2_sf_lo * swap_accum[i * 4 + 3];
                                    }
                                }
                            };

                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * SwapWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + k * SwapWGMMA::K, 1);
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
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + k_off, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_wait<0>();
                            promote_swap_accum(1);

                            arrive_empty_barrier(stage_idx);
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
                    } else {
                    // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                    // First half: K=0..63, SFA = scale_a_*_lo
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    // L2 weight SF is per 128 output columns; M64N256 spans two SF groups.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
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
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    arrive_empty_barrier(stage_idx);

                    // L2 second half: same SFA half, still choose weight SF by N chunk.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                        final_accum[i*4+0] += scale_a_0_hi * l2_sf * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_hi * l2_sf * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_hi * l2_sf * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_hi * l2_sf * accum[i*4+3];
                    }
                    }
                }
                }

                if constexpr (kBF16ScaledAccum and kSwapABActive) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                        const float2 pair =
                            __bfloat1622float2(swap_final_bf16[i]);
                        final_accum[2 * i] = pair.x;
                        final_accum[2 * i + 1] = pair.y;
                    }
                }
            };

            const auto run_bk256_gemm_loop = [&]() {
                if constexpr (BLOCK_K == 256) {
                    DG_STATIC_ASSERT(not kSwapABActive,
                                     "BK256 supports FP32 or packed-BF16 scaled accumulation");
                    constexpr uint32_t kL1SFKBlocks = kHidden / kGranK;
                    constexpr uint32_t kL2SFKBlocks =
                        kIntermediateHidden / kGranK;
                    constexpr uint32_t kL1SFGateBlks =
                        kIntermediateHidden / kGranK;
                    constexpr uint32_t kL1SFPerExpert =
                        (kIntermediateHidden * 2 / kGranK) * kL1SFKBlocks;
                    constexpr uint32_t kL2SFPerExpert =
                        (kHidden / kGranK) * kL2SFKBlocks;
                    nv_bfloat162 final_bf16[kAccumPerThread / 2];
                    if constexpr (kBF16ScaledAccum) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i)
                            final_bf16[i] = __float2bfloat162_rn(0.0f);
                    }

                    const auto issue_wgmma = [&]<uint32_t kNumWGMMAs>(
                                                     const uint32_t& k_plane,
                                                     const uint32_t& k_offset) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < kNumWGMMAs; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] +
                                    k_plane * LOAD_BLOCK_M * kTMATileK +
                                    row_block_offset * kTMATileK + k_offset +
                                    k * WGMMA::K,
                                1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] +
                                    k_plane * LOAD_BLOCK_N * kTMATileK +
                                    wg_n_idx * kTMATileK + k_offset +
                                    k * WGMMA::K,
                                1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k != 0);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();
                    };

                    const auto promote_l1 = [&](const float& scale_a_0,
                                                 const float& scale_a_1,
                                                 const float& gate_sf,
                                                 const float& up_sf) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float weight_sf = (i & 1u) ? up_sf : gate_sf;
                            if constexpr (kBF16ScaledAccum) {
                                const nv_bfloat162 scale0 =
                                    __float2bfloat162_rn(scale_a_0 * weight_sf);
                                const nv_bfloat162 scale1 =
                                    __float2bfloat162_rn(scale_a_1 * weight_sf);
                                final_bf16[2 * i] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i], accum[4 * i + 1]),
                                    scale0, final_bf16[2 * i]);
                                final_bf16[2 * i + 1] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i + 2], accum[4 * i + 3]),
                                    scale1, final_bf16[2 * i + 1]);
                            } else {
                                final_accum[i*4+0] +=
                                    scale_a_0 * weight_sf * accum[i*4+0];
                                final_accum[i*4+1] +=
                                    scale_a_0 * weight_sf * accum[i*4+1];
                                final_accum[i*4+2] +=
                                    scale_a_1 * weight_sf * accum[i*4+2];
                                final_accum[i*4+3] +=
                                    scale_a_1 * weight_sf * accum[i*4+3];
                            }
                        }
                    };
                    const auto promote_l2 = [&](const float& scale_a_0,
                                                 const float& scale_a_1,
                                                 const float& l2_sf_lo,
                                                 const float& l2_sf_hi) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float weight_sf =
                                (WG_BLOCK_N > 128 and i >= 16u) ?
                                    l2_sf_hi : l2_sf_lo;
                            if constexpr (kBF16ScaledAccum) {
                                const nv_bfloat162 scale0 =
                                    __float2bfloat162_rn(scale_a_0 * weight_sf);
                                const nv_bfloat162 scale1 =
                                    __float2bfloat162_rn(scale_a_1 * weight_sf);
                                final_bf16[2 * i] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i], accum[4 * i + 1]),
                                    scale0, final_bf16[2 * i]);
                                final_bf16[2 * i + 1] = __hfma2(
                                    __floats2bfloat162_rn(
                                        accum[4 * i + 2], accum[4 * i + 3]),
                                    scale1, final_bf16[2 * i + 1]);
                            } else {
                                final_accum[i*4+0] +=
                                    scale_a_0 * weight_sf * accum[i*4+0];
                                final_accum[i*4+1] +=
                                    scale_a_0 * weight_sf * accum[i*4+1];
                                final_accum[i*4+2] +=
                                    scale_a_1 * weight_sf * accum[i*4+2];
                                final_accum[i*4+3] +=
                                    scale_a_1 * weight_sf * accum[i*4+3];
                            }
                        }
                    };

                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                         advance_pipeline(k_block_idx)) {
                        full_barriers[stage_idx]->wait(phase);

                        if (is_linear1_phase) {
                            const uint32_t gate_n =
                                (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                            const uint32_t up_n = kL1SFGateBlks + gate_n;
                            const float* base = l1_weights_sf +
                                local_expert_idx * kL1SFPerExpert;
                            #pragma unroll
                            for (uint32_t k_plane = 0;
                                 k_plane < kNumTMATilesPerStage; ++ k_plane) {
                                const float scale_a_0 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        k_plane * kL2SFAHalfStride + row_offset_r0);
                                const float scale_a_1 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        k_plane * kL2SFAHalfStride + row_offset_r1);
                                const uint32_t sf_k =
                                    k_block_idx * kNumTMATilesPerStage + k_plane;
                                const float gate_sf =
                                    __ldg(base + gate_n * kL1SFKBlocks + sf_k);
                                const float up_sf =
                                    __ldg(base + up_n * kL1SFKBlocks + sf_k);
                                issue_wgmma.template operator()<
                                    kTMATileK / WGMMA::K>(k_plane, 0);
                                promote_l1(
                                    scale_a_0, scale_a_1, gate_sf, up_sf);
                            }
                        } else {
                            const uint32_t sf_n =
                                (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                            const float* base = l2_weights_sf +
                                local_expert_idx * kL2SFPerExpert;
                            #pragma unroll
                            for (uint32_t sf_group = 0;
                                 sf_group < kNumL2SFAKGroups; ++ sf_group) {
                                const uint32_t k_plane = sf_group / 2u;
                                const uint32_t k_offset =
                                    (sf_group & 1u) * kL2ActsSFGranK;
                                const uint32_t sf_k =
                                    k_block_idx * kNumTMATilesPerStage + k_plane;
                                const float scale_a_0 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        sf_group * kL2SFAHalfStride + row_offset_r0);
                                const float scale_a_1 = ptx::ld_shared(
                                    smem_sfa[stage_idx] +
                                        sf_group * kL2SFAHalfStride + row_offset_r1);
                                const float l2_sf_lo =
                                    __ldg(base + sf_n * kL2SFKBlocks + sf_k);
                                const float l2_sf_hi = WG_BLOCK_N > 128 ?
                                    __ldg(base +
                                          (sf_n + 1u) * kL2SFKBlocks + sf_k) :
                                    l2_sf_lo;
                                issue_wgmma.template operator()<
                                    kL2ActsSFGranK / WGMMA::K>(
                                        k_plane, k_offset);
                                promote_l2(scale_a_0, scale_a_1,
                                           l2_sf_lo, l2_sf_hi);
                            }
                        }
                        arrive_empty_barrier(stage_idx);
                    }

                    if constexpr (kBF16ScaledAccum) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                            const float2 pair = __bfloat1622float2(final_bf16[i]);
                            final_accum[2 * i] = pair.x;
                            final_accum[2 * i + 1] = pair.y;
                        }
                    }
                }
            };

            const auto run_bf16_scaled_accum_gemm_loop = [&]() {
                DG_STATIC_ASSERT(not kBF16ScaledAccum or
                                 kSwapABActive or
                                 (WG_BLOCK_M == 64 and
                                  (WG_BLOCK_N == 128 or WG_BLOCK_N == 256) and
                                  not kSwapABActive),
                                 "BF16 scaled accumulation expects swap-AB or non-swap M64N128/N256");
                nv_bfloat162 final_bf16[kAccumPerThread / 2];
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i)
                    final_bf16[i] = __float2bfloat162_rn(0.0f);

                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert =
                    (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert =
                    (kHidden / 128) * kL2SFKBlocks;

                const auto issue_wgmma = [&]<uint32_t kOffset,
                                              uint32_t kNumWGMMAs>() {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < kNumWGMMAs; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                kOffset + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K +
                                kOffset + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();
                };

                const auto promote_l1 = [&](const float& scale_a_0,
                                             const float& scale_a_1,
                                             const float& gate_sf,
                                             const float& up_sf) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float weight_sf = (i & 1u) ? up_sf : gate_sf;
                        const nv_bfloat162 scale0 =
                            __float2bfloat162_rn(scale_a_0 * weight_sf);
                        const nv_bfloat162 scale1 =
                            __float2bfloat162_rn(scale_a_1 * weight_sf);
                        final_bf16[2 * i] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i], accum[4 * i + 1]),
                            scale0, final_bf16[2 * i]);
                        final_bf16[2 * i + 1] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i + 2], accum[4 * i + 3]),
                            scale1, final_bf16[2 * i + 1]);
                    }
                };
                const auto promote_l2 = [&](const float& scale_a_0,
                                             const float& scale_a_1,
                                             const float& weight_sf_lo,
                                             const float& weight_sf_hi) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const bool high_n = WG_BLOCK_N > 128 and i >= 16u;
                        const float weight_sf = high_n ? weight_sf_hi : weight_sf_lo;
                        const nv_bfloat162 scale0 =
                            __float2bfloat162_rn(scale_a_0 * weight_sf);
                        const nv_bfloat162 scale1 =
                            __float2bfloat162_rn(scale_a_1 * weight_sf);
                        final_bf16[2 * i] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i], accum[4 * i + 1]),
                            scale0, final_bf16[2 * i]);
                        final_bf16[2 * i + 1] = __hfma2(
                            __floats2bfloat162_rn(
                                accum[4 * i + 2], accum[4 * i + 3]),
                            scale1, final_bf16[2 * i + 1]);
                    }
                };

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                     advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    const float scale_a_0_lo =
                        ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    const float scale_a_1_lo =
                        ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);

                    if (is_linear1_phase) {
                        const uint32_t gate_n =
                            (n_block_idx * BLOCK_N + wg_n_idx) / 256u;
                        const uint32_t up_n = kL1SFGateBlks + gate_n;
                        const float* base = l1_weights_sf +
                            local_expert_idx * kL1SFPerExpert + k_block_idx;
                        const float gate_sf =
                            __ldg(base + gate_n * kL1SFKBlocks);
                        const float up_sf =
                            __ldg(base + up_n * kL1SFKBlocks);
                        issue_wgmma.template operator()<
                            0, BLOCK_K / WGMMA::K>();
                        arrive_empty_barrier(stage_idx);
                        promote_l1(
                            scale_a_0_lo, scale_a_1_lo, gate_sf, up_sf);
                    } else {
                        const float scale_a_0_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + kL2SFAHalfStride +
                                row_offset_r0);
                        const float scale_a_1_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + kL2SFAHalfStride +
                                row_offset_r1);
                        const uint32_t sf_n =
                            (n_block_idx * BLOCK_N + wg_n_idx) / 128u;
                        const float* base = l2_weights_sf +
                            local_expert_idx * kL2SFPerExpert + k_block_idx;
                        const float l2_sf_lo =
                            __ldg(base + sf_n * kL2SFKBlocks);
                        const float l2_sf_hi = WG_BLOCK_N > 128 ?
                            __ldg(base + (sf_n + 1u) * kL2SFKBlocks) :
                            l2_sf_lo;
                        issue_wgmma.template operator()<
                            0, (BLOCK_K / 2) / WGMMA::K>();
                        promote_l2(
                            scale_a_0_lo, scale_a_1_lo, l2_sf_lo, l2_sf_hi);
                        issue_wgmma.template operator()<
                            BLOCK_K / 2, (BLOCK_K / 2) / WGMMA::K>();
                        arrive_empty_barrier(stage_idx);
                        promote_l2(
                            scale_a_0_hi, scale_a_1_hi, l2_sf_lo, l2_sf_hi);
                    }
                }

                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                    const float2 pair = __bfloat1622float2(final_bf16[i]);
                    final_accum[2 * i] = pair.x;
                    final_accum[2 * i + 1] = pair.y;
                }
            };

            if constexpr (BLOCK_K == 256) {
                run_bk256_gemm_loop();
            } else if constexpr (kBF16ScaledAccum and not kSwapABActive) {
                run_bf16_scaled_accum_gemm_loop();
            } else {
                run_default_gemm_loop();
            }

            // Skip epilogue when block is past valid M (still must release via empty).
            if (row_block_offset >= valid_m) {
                if constexpr (MegaMoEPhase::runs_linear1 and kWarpgroupSplitM > 1 and not kSplitNWarpgroups) {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                } else {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                return;
            }

            if (is_linear1_phase) {
                if constexpr (kSwapABActive) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-kActivationAlpha * x) :
                                                   expf(-kActivationAlpha * x);
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

                    const uint32_t out_col_base = wg_l1_out_n_idx + warp_idx_in_wg * 8 + row_idx;
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            float g0 = final_accum[i * 4 + 0];
                            float u0 = final_accum[i * 4 + 2];
                            clamp_gate(g0);
                            clamp_up(u0);
                            const float weight_0 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_0)
                                .template get_base_ptr<float>();
                            smem_cd_swap_l1_fp32[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                silu(g0) * (u0 + kActivationUpBias) * weight_0;
                        }
                        if (token_1 < valid_m) {
                            float g1 = final_accum[i * 4 + 1];
                            float u1 = final_accum[i * 4 + 3];
                            clamp_gate(g1);
                            clamp_up(u1);
                            const float weight_1 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_1)
                                .template get_base_ptr<float>();
                            smem_cd_swap_l1_fp32[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                silu(g1) * (u1 + kActivationUpBias) * weight_1;
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
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; ++ col) {
                            const float v = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col];
                            amax = cute::max(amax, cute::abs(v));
                        }
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
                            amax_pair, sf_pair, sf_inv_pair);
                        const float sf = sf_pair.x;
                        const float sf_inv = sf_inv_pair.x;

                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_idx = pool_block_idx * BLOCK_M + token;
                        sf_base_ptr[n_block_idx * kSFPoolStrideTokens + token_idx] = sf;

                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; col += 2) {
                            const float v0 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 0] * sf_inv;
                            const float v1 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 1] * sf_inv;
                            const __nv_fp8x2_e4m3 pair(make_float2(v0, v1));
                            auto* ptr = reinterpret_cast<uint16_t*>(
                                smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col);
                            *ptr = pair.__x;
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                } else {
                    // L1 epilogue: SwiGLU, FP8 quantization, and TMA store
                    // Layout in `final_accum`:
                    //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                    //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                    //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                    //
                    // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                    // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                    constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    DG_STATIC_ASSERT(WG_L1_OUT_BLOCK_N % 64 == 0,
                                     "Each L1 consumer must cover complete 64-column SF groups");
                    constexpr uint32_t kNumSFGroups = WG_L1_OUT_BLOCK_N / 64;
                    float swiglu_r0[kNumPairs][2];
                    float swiglu_r1[kNumPairs][2];

                    // Per-row amax, one scale for each 64-col L1 output group.
                    float amax_r0[kNumSFGroups] = {};
                    float amax_r1[kNumSFGroups] = {};

                    // Compute SwiGLU + per-group amax.
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t gate = 2 * p, up = 2 * p + 1;
                        const uint32_t sf_group = p / 8;

                        auto clamp_gate = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(x, kActivationClamp);
                        };
                        auto clamp_up = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                        };
                        float g_r0_c0 = final_accum[gate*4 + 0]; clamp_gate(g_r0_c0);
                        float g_r0_c1 = final_accum[gate*4 + 1]; clamp_gate(g_r0_c1);
                        float g_r1_c0 = final_accum[gate*4 + 2]; clamp_gate(g_r1_c0);
                        float g_r1_c1 = final_accum[gate*4 + 3]; clamp_gate(g_r1_c1);
                        float u_r0_c0 = final_accum[up*4   + 0]; clamp_up(u_r0_c0);
                        float u_r0_c1 = final_accum[up*4   + 1]; clamp_up(u_r0_c1);
                        float u_r1_c0 = final_accum[up*4   + 2]; clamp_up(u_r1_c0);
                        float u_r1_c1 = final_accum[up*4   + 3]; clamp_up(u_r1_c1);

                        auto silu = [](float x) {
                            const float e = kFastMath ? __expf(-kActivationAlpha * x) :
                                                       expf(-kActivationAlpha * x);
                            const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                            return x * sig;
                        };

                        if (valid_r0) {
                            swiglu_r0[p][0] = silu(g_r0_c0) * (u_r0_c0 + kActivationUpBias);
                            swiglu_r0[p][1] = silu(g_r0_c1) * (u_r0_c1 + kActivationUpBias);
                            amax_r0[sf_group] = cute::max(
                                amax_r0[sf_group],
                                cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                        } else {
                            swiglu_r0[p][0] = 0.0f;
                            swiglu_r0[p][1] = 0.0f;
                        }
                        if (valid_r1) {
                            swiglu_r1[p][0] = silu(g_r1_c0) * (u_r1_c0 + kActivationUpBias);
                            swiglu_r1[p][1] = silu(g_r1_c1) * (u_r1_c1 + kActivationUpBias);
                            amax_r1[sf_group] = cute::max(
                                amax_r1[sf_group],
                                cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                        } else {
                            swiglu_r1[p][0] = 0.0f;
                            swiglu_r1[p][1] = 0.0f;
                        }
                    }

                    float weight_r0 = 0.0f, weight_r1 = 0.0f;
                    // For large-capacity specializations, cached same-address
                    // loads avoid a shuffle dependency in this register-heavy path.
                    if constexpr (kNumMaxTokensPerRank <= 1024) {
                        const int topk_weight_src_lane = static_cast<int>(lane_idx - col_idx);
                        if (col_idx == 0) {
                            weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + row_offset_r0)
                                .template get_base_ptr<float>() : 0.0f;
                            weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + row_offset_r1)
                                .template get_base_ptr<float>() : 0.0f;
                        }
                        weight_r0 = __shfl_sync(0xffffffff, weight_r0, topk_weight_src_lane);
                        weight_r1 = __shfl_sync(0xffffffff, weight_r1, topk_weight_src_lane);
                    } else {
                        weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .template get_base_ptr<float>() : 0.0f;
                        weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .template get_base_ptr<float>() : 0.0f;
                    }
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
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
                        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
                    }

                    float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                    float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        float2 amax_pair = {amax_r0[g], amax_r1[g]};
                        float2 sf_pair, sf_inv_pair;
                        sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
                            amax_pair, sf_pair, sf_inv_pair);
                        sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x;
                        sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y;
                    }

                    // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                    auto* smem_cd_l1_wg = smem_cd_l1
                        + smem_cd_l1_wg_offset;
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t sf_group = p / 8;
                        const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
                        const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
                        const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
                        const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

                        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                        const uint32_t col = p * 8 + col_idx * 2;
                        auto* p0 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1_wg + r_0 * WG_SMEM_CD_L1_STRIDE_N +
                            (kSplitNWarpgroups ? wg_l1_out_n_idx : 0) + col);
                        auto* p1 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1_wg + r_1 * WG_SMEM_CD_L1_STRIDE_N +
                            (kSplitNWarpgroups ? wg_l1_out_n_idx : 0) + col);
                        if (valid_r0)
                            *p0 = r0_pair.__x;
                        if (valid_r1)
                            *p1 = r1_pair.__x;
                    }

                    // Write L2-activation SF as float, one value per 64 output columns.
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                        const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                        const uint32_t base_k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / 64u;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                            if (valid_r0)
                                sf_base_ptr[(base_k_sf_idx + g) * kSFPoolStrideTokens + token_r0] = sf_r0[g];
                            if (valid_r1)
                                sf_base_ptr[(base_k_sf_idx + g) * kSFPoolStrideTokens + token_r1] = sf_r1[g];
                        }
                    }

                    // Issue TMA store of the entire tile. Padding rows beyond
                    // `valid_m` are written with stale/garbage FP8 to the L1-output
                    // pool buffer, but they are never consumed downstream: the L2
                    // GEMM tile loads them, but its NVLink-scatter epilogue is
                    // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                    // padding rows can produce NaN accumulators that simply stay
                    // in registers (only valid rows are converted to BF16 and
                    // STSM'd into smem). Using TMA for partial tiles is a large
                    // win for low-batch / decode where every tile is partial.
                    if constexpr (kSplitNWarpgroups) {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1,
                                out_n_idx,
                                m_idx);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                        ptx::tma_store_wait<0>();
                    } else {
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1_wg,
                                out_n_idx,
                                m_idx + row_block_offset);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                        ptx::tma_store_wait<0>();
                    }
                }
            } else {
                // L2 epilogue: BF16 cast and NVLink scatter
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kDirectL2Scatter) {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 128, "Direct L2 scatter only supports N128");

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row,
                                                  const uint32_t& row_accum_offset) {
                        if (valid_row) {
                            uint32_t dst_rank_idx = 0, dst_token_idx = 0, dst_topk_idx = 0;
                            const uint32_t row_group_base = lane_idx - col_idx;
                            if (col_idx == 0) {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + row_offset);
                                dst_rank_idx = src_metadata.rank_idx;
                                dst_token_idx = src_metadata.token_idx;
                                dst_topk_idx = src_metadata.topk_idx;
                            }
                            const uint32_t row_group_mask = 0xfu << row_group_base;
                            const int src_lane = static_cast<int>(row_group_base);
                            dst_rank_idx = __shfl_sync(row_group_mask, dst_rank_idx, src_lane);
                            dst_token_idx = __shfl_sync(row_group_mask, dst_token_idx, src_lane);
                            dst_topk_idx = __shfl_sync(row_group_mask, dst_topk_idx, src_lane);
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_base = math::advance_ptr<uint8_t>(
                                dst_token.get_base_ptr(), n_idx * kCombineElementBytes);
                            auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                const uint32_t packed_lo = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                    final_accum[chunk_lo * 4 + row_accum_offset + 1]);
                                const uint32_t packed_hi = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                    final_accum[chunk_hi * 4 + row_accum_offset + 1]);
                                *reinterpret_cast<uint32_t*>(
                                    mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                *reinterpret_cast<uint32_t*>(
                                    mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                            }
                        }
                    };

                    scatter_direct_row(row_offset_r0, valid_r0, 0);
                    scatter_direct_row(row_offset_r1, valid_r1, 2);
                } else {
                    auto store_l2_scalar = [&](const uint32_t& elem_idx, const float& value) {
                        smem_cd_l2[elem_idx] = __float2bfloat16_rn(value);
                    };
                    auto store_l2_pair = [&](const uint32_t& elem_idx,
                                             float value0, float value1) {
                        *reinterpret_cast<uint32_t*>(smem_cd_l2 + elem_idx) =
                            math::cast_into_bf16_and_pack(value0, value1);
                    };
                    if constexpr (kSwapABActive) {
                        auto store_value = [&](const uint32_t& token,
                                               const uint32_t& col,
                                               const float value) {
                            store_l2_scalar(
                                epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N +
                                    token * WG_BLOCK_N + col,
                                value);
                        };

                        auto store_l2_swap_chunk = [&](const uint32_t& i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if (token_0 < valid_m) {
                                store_value(token_0, r_0, final_accum[i * 4 + 0]);
                                store_value(token_0, r_1, final_accum[i * 4 + 2]);
                            }
                            if (token_1 < valid_m) {
                                store_value(token_1, r_0, final_accum[i * 4 + 1]);
                                store_value(token_1, r_1, final_accum[i * 4 + 3]);
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
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                            const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                            auto write_pair = [&](const uint32_t row,
                                                  const uint32_t col,
                                                  const float value0,
                                                  const float value1) {
                                const uint32_t elem_idx =
                                    epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N +
                                    row * WG_BLOCK_N + col;
                                store_l2_pair(elem_idx, value0, value1);
                            };
                            if (valid_r0) {
                                write_pair(r_0, chunk_lo * 8 + col_idx * 2,
                                    final_accum[chunk_lo * 4 + 0],
                                    final_accum[chunk_lo * 4 + 1]);
                                write_pair(r_0, chunk_hi * 8 + col_idx * 2,
                                    final_accum[chunk_hi * 4 + 0],
                                    final_accum[chunk_hi * 4 + 1]);
                            }
                            if (valid_r1) {
                                write_pair(r_1, chunk_lo * 8 + col_idx * 2,
                                    final_accum[chunk_lo * 4 + 2],
                                    final_accum[chunk_lo * 4 + 3]);
                                write_pair(r_1, chunk_hi * 8 + col_idx * 2,
                                    final_accum[chunk_hi * 4 + 2],
                                    final_accum[chunk_hi * 4 + 3]);
                            }
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows × 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp → 1 row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    const uint32_t cols_per_lane = WG_BLOCK_N / 16;
                    static_assert(WG_BLOCK_N == 64 or WG_BLOCK_N == 128 or WG_BLOCK_N == 256,
                                  "L2 scatter supports per-WG N64/N128/N256");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        const uint32_t smem_elem_idx =
                            epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N +
                            row_in_wg * WG_BLOCK_N + lane_in_row * cols_per_lane;
                        auto smem_ptr = smem_cd_base + smem_elem_idx * kCombineElementBytes;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        constexpr uint32_t kScatterBytesPerLane =
                            (WG_BLOCK_N / 16) * kCombineElementBytes;
                        auto dst_ptr = math::advance_ptr<uint8_t>(
                            dst_token.get_base_ptr(),
                            n_idx * kCombineElementBytes + lane_in_row * kScatterBytesPerLane);
                        auto mapped_dst_ptr = sym_buffer.map(dst_ptr, dst_rank_idx);

                        if constexpr (kScatterBytesPerLane == 32) {
                            const auto packed0 = *reinterpret_cast<uint4*>(smem_ptr);
                            const auto packed1 = *(reinterpret_cast<uint4*>(smem_ptr) + 1);
                            reinterpret_cast<uint4*>(mapped_dst_ptr)[0] = packed0;
                            reinterpret_cast<uint4*>(mapped_dst_ptr)[1] = packed1;
                        } else if constexpr (kScatterBytesPerLane == 16) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            *reinterpret_cast<uint4*>(mapped_dst_ptr) = packed;
                        } else if constexpr (kScatterBytesPerLane == 8) {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            *reinterpret_cast<uint2*>(mapped_dst_ptr) = packed;
                        } else {
                            DG_STATIC_ASSERT(kScatterBytesPerLane == 4,
                                             "Unexpected L2 scatter width");
                            *reinterpret_cast<uint32_t*>(mapped_dst_ptr) =
                                *reinterpret_cast<uint32_t*>(smem_ptr);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            }
        });

        if constexpr (!MegaMoEPhase::runs_linear2) {
            return;
        }

        // Combine
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumEpilogueThreads,
            kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag,
            comm::BarrierTimeoutPolicy::TrapOnly>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );
        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        constexpr uint32_t kNumChunks = kCombineNumChunks;
        constexpr uint32_t kInputChunkBytes = kCombineInputChunkBytes;
        constexpr uint32_t kOutputChunkBytes = kCombineOutputChunkBytes;
        constexpr uint32_t kElemsPerVector = 8;
        constexpr uint32_t kInputVectorBytes = kElemsPerVector * kCombineElementBytes;
        constexpr uint32_t kNumVectorsPerLane =
            kCombineChunkElems / (32 * kElemsPerVector);
        constexpr uint32_t kNumBF16PairsPerVector = kElemsPerVector / 2;
        DG_STATIC_ASSERT(kInputChunkBytes % 16 == 0,
                         "Combine input chunk must be TMA-aligned");
        DG_STATIC_ASSERT(kOutputChunkBytes % 16 == 0,
                         "Combine output chunk must be TMA-aligned");
        DG_STATIC_ASSERT(kCombineChunkElems % (32 * kElemsPerVector) == 0,
                         "Combine chunk must distribute evenly across a warp");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint8_t>(
                smem_buffer,
                (epilogue_warp_idx + i * kNumEpilogueWarps) * kInputChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(smem_buffer,
            2 * kNumEpilogueWarps * kInputChunkBytes +
                epilogue_warp_idx * kOutputChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t input_chunk_byte_offset = chunk * kInputChunkBytes;
                const uint32_t output_chunk_byte_offset = chunk * kOutputChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                input_chunk_byte_offset);
                            ptx::tma_load_1d(
                                combine_load_buffer[i], src_ptr,
                                combine_load_barriers[i], kInputChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(
                                combine_load_barriers[i], kInputChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumVectorsPerLane * kNumBF16PairsPerVector] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumVectorsPerLane; ++ j) {
                        const uint32_t vector_idx = j * 32 + lane_idx;
                        const auto packed = *reinterpret_cast<const uint4*>(
                            combine_load_buffer[load_stage_idx] +
                            vector_idx * kInputVectorBytes);
                        const auto bf16_values =
                            reinterpret_cast<const nv_bfloat162*>(&packed);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                            ptx::accumulate(
                                reduced[j * kNumBF16PairsPerVector + l],
                                bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumVectorsPerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(
                            reduced[j * kNumBF16PairsPerVector + l]);

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
                        math::advance_ptr(
                            y,
                            static_cast<uint64_t>(token_idx) * kCombineOutputHiddenBytes +
                                output_chunk_byte_offset),
                        combine_store_buffer, kOutputChunkBytes);
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

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS,
          bool kNMajorScheduleRequested = false>
CUTLASS_GLOBAL __launch_bounds__(
    kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1) void
sm90_fp8_mega_moe_l1_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    using Phase = MegaMoEPhasePolicy<
        MegaMoEPhaseKind::Linear1, kNMajorScheduleRequested>;
    sm90_fp8_mega_moe_core<Phase, DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
}

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS,
          bool kDirectL2ScatterRequested = false,
          bool kNMajorScheduleRequested = false,
          bool kOneWarpCleanupRequested = false>
CUTLASS_GLOBAL __launch_bounds__(
    kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1) void
sm90_fp8_mega_moe_l2_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    using Phase = MegaMoEPhasePolicy<
        MegaMoEPhaseKind::Linear2, kNMajorScheduleRequested,
        kDirectL2ScatterRequested, kOneWarpCleanupRequested>;
    sm90_fp8_mega_moe_core<Phase, DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS>(
        DG_SM90_FP8_MOE_KERNEL_ARGS);
}

#undef DG_SM90_FP8_MOE_TEMPLATE_PARAMS
#undef DG_SM90_FP8_MOE_KERNEL_ARGS_DECL
#undef DG_SM90_FP8_MOE_CORE_ARGS_DECL
#undef DG_SM90_FP8_MOE_KERNEL_ARGS
#undef DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS

} // namespace deep_gemm

#pragma clang diagnostic pop
