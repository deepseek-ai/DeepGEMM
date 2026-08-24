#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/swizzle.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 x MXFP4 MegaMoE — persistent implementation
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

// The model-load preprocessing keeps every magnitude nibble in place but moves
// the eight signs in each packed word to [s0,s4,s1,s5,s2,s6,s3,s7].  The low
// nibble signs of the four bytes therefore belong to outputs 0..3, and the
// high nibble signs belong to outputs 4..7.  This removes the two sign-gather
// PRMTs from every decode without changing its byte size.
CUTLASS_DEVICE uint2 sm90_mxfp4_e4m3_lookup(
    const uint32_t exponent_offset) {
    uint2 result;
    asm volatile(
        "{\n\t"
        "mad.lo.u32 %0, %2, 0x08080800, 0x0c080000;\n\t"
        "mad.lo.u32 %1, %2, 0x08080808, 0x1c181410;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y)
        : "r"(exponent_offset));
    return result;
}

CUTLASS_DEVICE uint2 sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
    const uint32_t packed, const uint2 lookup) {
    uint2 result;
    asm volatile(
        "{\n\t"
        ".reg .b32 temp0, temp1, out_lo;\n\t"
        "shl.b32 out_lo, %2, 4;\n\t"
        "and.b32 temp0, %2, 0x77777777;\n\t"
        "prmt.b32 temp1, %3, %4, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, %3, %4, temp0;\n\t"
        "lop3.b32 %1, %2, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %0, out_lo;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y)
        : "r"(packed), "r"(lookup.x), "r"(lookup.y));
    return result;
}

CUTLASS_DEVICE uint2 sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
    const uint32_t packed, const uint32_t exponent_offset) {
    uint2 result;
    asm volatile(
        "{\n\t"
        ".reg .b32 lookup_lo, lookup_hi, temp0, temp1, out_lo;\n\t"
        "mad.lo.u32 lookup_lo, %3, 0x08080800, 0x0c080000;\n\t"
        "mad.lo.u32 lookup_hi, %3, 0x08080808, 0x1c181410;\n\t"
        "shl.b32 out_lo, %2, 4;\n\t"
        "and.b32 temp0, %2, 0x77777777;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 %1, %2, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %0, out_lo;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y)
        : "r"(packed), "r"(exponent_offset));
    return result;
}

CUTLASS_DEVICE uint4 sm90_mxfp4_reordered_signs_e2m1x16_to_e4m3x16_bits(
    const uint32_t packed0,
    const uint32_t packed1,
    const uint32_t exponent_offset) {
    uint4 result;
    asm volatile(
        "{\n\t"
        ".reg .b32 lookup_lo, lookup_hi, temp0, temp1, out_lo;\n\t"
        "mad.lo.u32 lookup_lo, %6, 0x08080800, 0x0c080000;\n\t"
        "mad.lo.u32 lookup_hi, %6, 0x08080808, 0x1c181410;\n\t"
        "shl.b32 out_lo, %4, 4;\n\t"
        "and.b32 temp0, %4, 0x77777777;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 %1, %4, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %0, out_lo;\n\t"
        "shl.b32 out_lo, %5, 4;\n\t"
        "and.b32 temp0, %5, 0x77777777;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 %3, %5, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %2, out_lo;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
        : "r"(packed0), "r"(packed1), "r"(exponent_offset));
    return result;
}

__forceinline__ __device__ uint32_t sm90_extract_u8_prmt(
    const uint32_t packed, const uint32_t byte_idx) {
    uint32_t result = 0;
    switch (byte_idx) {
        case 0:
            asm volatile("prmt.b32 %0, %1, 0, 0x4440;"
                         : "=r"(result) : "r"(packed));
            break;
        case 1:
            asm volatile("prmt.b32 %0, %1, 0, 0x4441;"
                         : "=r"(result) : "r"(packed));
            break;
        case 2:
            asm volatile("prmt.b32 %0, %1, 0, 0x4442;"
                         : "=r"(result) : "r"(packed));
            break;
        default:
            asm volatile("prmt.b32 %0, %1, 0, 0x4443;"
                         : "=r"(result) : "r"(packed));
            break;
    }
    return result;
}

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

// Keep SM90 MegaMoE grid barriers on a trap-only timeout path. The shared
// SM100 barrier intentionally retains nv_dev's diagnostic printf, but compiling
// that printf into this register-heavy Hopper kernel creates a per-thread stack
// frame and can invalidate the two-CTA occupancy contract.
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
CUTLASS_DEVICE void sm90_grid_sync(
    const layout::Workspace& workspace,
    const uint32_t& sm_idx, const uint32_t& thread_idx,
    const sync_scope_t& sync_scope) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        const auto start_clock = clock64();
        uint32_t new_value;
        do {
            new_value = ptx::ld_acq(count_ptr);
            if (clock64() - start_clock >= comm::kNumTimeoutCycles)
                DG_TRAP_ONLY_DEVICE_ASSERT(false and "SM90 grid sync timeout");
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    sync_scope();
}

// SM90-local NVLink barrier using the trap-only grid barrier above. The NVLink
// rendezvous keeps the long 300-second timeout because all participating
// ranks may compile or enter the first launch at different times.
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void sm90_nvlink_barrier(
    const layout::Workspace& workspace,
    const layout::SymBuffer<kNumRanks>& sym_buffer,
    const uint32_t& sm_idx, const uint32_t& thread_idx,
    const sync_scope_t& sync_scope,
    const bool& sync_prologue = true,
    const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    if (sync_prologue)
        sm90_grid_sync<kNumSMs, kGridSyncIndex>(
            workspace, sm_idx, thread_idx, sync_scope);

    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(
                sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        constexpr int64_t kNumTimeoutCycles = 300ll * 2000000000ll;
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            const auto start_clock = clock64();
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles)
                    DG_TRAP_ONLY_DEVICE_ASSERT(false and "NVLink barrier timeout");
            }
        }
    }

    if (sync_epilogue)
        sm90_grid_sync<kNumSMs, kGridSyncIndex>(
            workspace, sm_idx, thread_idx, sync_scope);
}

#define DG_SM90_FP8_MOE_TEMPLATE_PARAMS \
    uint32_t kNumMaxTokensPerRank, \
    uint32_t kHidden, uint32_t kIntermediateHidden, \
    uint32_t kNumExperts, uint32_t kNumTopk, \
    uint32_t kNumSMs, uint32_t kNumRanks, \
    float kActivationClamp, \
    bool kFastMath, \
    bool kSmallMSwapAB, \
    uint32_t kMaxSwapABTokens, \
    uint32_t kLocalSwapABTokens, \
    bool kPackedBF16SwapEpilogue, \
    bool kSwizzleL2CD, \
    bool kOverlapMXFP4ScalePath, \
    bool kUsePRMTMXFP4Exponent, \
    bool kUseIncrementalMXFP4Descriptor, \
    bool kBankPermuteMXFP4PairLoads, \
    uint32_t kNumRingTokens, \
    uint32_t kNumSFRingTokens, \
    uint32_t kNumSharedExperts

#define DG_SM90_FP8_MOE_KERNEL_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights, \
    const float* __restrict__ l1_mxfp4_secondary, \
    const uint8_t* __restrict__ l1_mxfp4_weights_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights, \
    const float* __restrict__ l2_mxfp4_secondary, \
    const uint8_t* __restrict__ l2_mxfp4_weights_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_weights, \
    const float* __restrict__ shared_l1_weights_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l1_output, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_shared_l2_weights, \
    const float* __restrict__ shared_l2_weights_sf

#define DG_SM90_FP8_MOE_CORE_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const layout::SymBuffer<kNumRanks>& sym_buffer, \
    const cute::TmaDescriptor& tensor_map_l1_acts, \
    const cute::TmaDescriptor& tensor_map_l1_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l1_weights, \
    const float* __restrict__ l1_mxfp4_secondary, \
    const uint8_t* __restrict__ l1_mxfp4_weights_sf, \
    const cute::TmaDescriptor& tensor_map_l1_output, \
    const cute::TmaDescriptor& tensor_map_l2_acts, \
    const cute::TmaDescriptor& tensor_map_l2_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l2_weights, \
    const float* __restrict__ l2_mxfp4_secondary, \
    const uint8_t* __restrict__ l2_mxfp4_weights_sf, \
    const cute::TmaDescriptor& tensor_map_shared_l1_acts, \
    const cute::TmaDescriptor& tensor_map_shared_l1_acts_sf, \
    const cute::TmaDescriptor& tensor_map_shared_l1_weights, \
    const float* __restrict__ shared_l1_weights_sf, \
    const cute::TmaDescriptor& tensor_map_shared_l1_output, \
    const cute::TmaDescriptor& tensor_map_shared_l2_acts, \
    const cute::TmaDescriptor& tensor_map_shared_l2_acts_sf, \
    const cute::TmaDescriptor& tensor_map_shared_l2_weights, \
    const float* __restrict__ shared_l2_weights_sf

#define DG_SM90_FP8_MOE_KERNEL_ARGS \
    y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer, \
    tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, \
    l1_mxfp4_secondary, l1_mxfp4_weights_sf, tensor_map_l1_output, tensor_map_l2_acts, \
    tensor_map_l2_acts_sf, tensor_map_l2_weights, l2_mxfp4_secondary, \
    l2_mxfp4_weights_sf, tensor_map_shared_l1_acts, \
    tensor_map_shared_l1_acts_sf, tensor_map_shared_l1_weights, \
    shared_l1_weights_sf, tensor_map_shared_l1_output, \
    tensor_map_shared_l2_acts, tensor_map_shared_l2_acts_sf, \
    tensor_map_shared_l2_weights, shared_l2_weights_sf

#define DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS \
    kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk, \
    kNumSMs, kNumRanks, \
    kActivationClamp, kFastMath, kSmallMSwapAB, \
    kMaxSwapABTokens, kLocalSwapABTokens, \
    kPackedBF16SwapEpilogue, kSwizzleL2CD, \
    kOverlapMXFP4ScalePath, \
    kUsePRMTMXFP4Exponent, \
    kUseIncrementalMXFP4Descriptor, \
    kBankPermuteMXFP4PairLoads, \
    kNumRingTokens, kNumSFRingTokens, kNumSharedExperts

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_DEVICE void
sm90_fp8_mega_moe_core(DG_SM90_FP8_MOE_CORE_ARGS_DECL) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Schedule = layout::Sm90HummingMoeSchedule;
    constexpr uint32_t BLOCK_M = Schedule::block_m;
    constexpr uint32_t BLOCK_N = Schedule::block_n;
    constexpr uint32_t BLOCK_K = Schedule::block_k;
    constexpr uint32_t kNumStages = Schedule::num_stages;
    constexpr uint32_t kNumDispatchThreads = Schedule::num_dispatch_threads;
    constexpr uint32_t kNumNonEpilogueThreads =
        Schedule::num_non_epilogue_threads;
    constexpr uint32_t kNumEpilogueThreads = Schedule::num_epilogue_threads;
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
    constexpr uint32_t kNumRingBlocks = kNumRingTokens / BLOCK_M;
    constexpr bool kHasSharedExperts = kNumSharedExperts > 0;
    DG_STATIC_ASSERT(not kSmallMSwapAB or
                         ((kHidden == 4096 or kHidden == 7168) and
                          not kHasSharedExperts),
                     "Small-M swap-AB is routed-only and model-specific");
    DG_STATIC_ASSERT(not kPackedBF16SwapEpilogue or
                         (kSmallMSwapAB and
                          (kHidden == 4096 or kHidden == 7168) and
                          not kHasSharedExperts),
                     "Packed-BF16 swap epilogue is routed small-M only");
    DG_STATIC_ASSERT(
        kMaxSwapABTokens == 8 or kMaxSwapABTokens == 16 or
            kMaxSwapABTokens == 32 or kMaxSwapABTokens == 64,
        "Swap-AB token bound must select a supported WGMMA bucket");
    DG_STATIC_ASSERT(
        kLocalSwapABTokens == 8 or kLocalSwapABTokens == 16 or
            kLocalSwapABTokens == 32 or kLocalSwapABTokens == 64,
        "Local swap-AB token bucket must be supported");
    DG_STATIC_ASSERT(kMaxSwapABTokens >= kLocalSwapABTokens,
                     "Swap-AB storage must cover the local policy bucket");

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(kNumRingTokens > 0 and kNumRingTokens % BLOCK_M == 0,
                     "Ring tokens must contain complete M blocks");
    DG_STATIC_ASSERT(kNumSFRingTokens >= kNumRingTokens,
                     "SM90 SF ring must cover every physical ring token");
    DG_STATIC_ASSERT(kNumTopk + (kHasSharedExperts ? 1u : 0u) <= 32,
                     "Top-k plus shared contribution must fit one warp");
    DG_STATIC_ASSERT(kHidden % BLOCK_K == 0 and kIntermediateHidden % BLOCK_K == 0,
                     "GEMM K dimensions must be divisible by BLOCK_K");
    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // The persistent kernel executes both logical phases.
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        if constexpr (kHasSharedExperts) {
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_weights);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l1_output);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_shared_l2_weights);
        }
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    const auto buffer = layout::MegaMoEBuffer(
        sym_buffer.get_base_ptr(), kHidden, kIntermediateHidden,
        kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens, kNumSFRingTokens, /*with_sf=*/ true,
        kNumSharedExperts,
        layout::ScaleLayoutSpec::sm90_fp32_k128_k64(
            kHidden, kIntermediateHidden));
    const auto workspace = buffer.workspace;

    constexpr auto fp8_token_layout = layout::Data(kHidden);
    const auto input_token_buffer = buffer.input_token_buffer;
    const auto input_sf_buffer = buffer.input_sf_buffer;
    const auto input_topk_idx_buffer = buffer.input_topk_idx_buffer;
    const auto input_topk_weights_buffer = buffer.input_topk_weights_buffer;
    const auto shared_l1_token_buffer = buffer.shared_l1_token_buffer;
    const auto shared_l1_sf_buffer = buffer.shared_l1_sf_buffer;
    const auto shared_l2_token_buffer = buffer.shared_l2_token_buffer;
    const auto shared_l2_sf_buffer = buffer.shared_l2_sf_buffer;
    const auto l1_token_buffer = buffer.l1_token_buffer;
    const auto l1_sf_buffer = buffer.l1_sf_buffer;
    const auto l1_topk_weights_buffer = buffer.l1_topk_weights_buffer;
    const auto l2_token_buffer = buffer.l2_token_buffer;
    const auto l2_sf_buffer = buffer.l2_sf_buffer;

    // Combine input area stores BF16 L2 contributions for the final reduction.
    constexpr uint32_t kCombineElementBytes = sizeof(nv_bfloat16);
    const auto combine_token_buffer = buffer.combine_token_buffer;

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    using task_info_t = sched::TaskInfo<kHasSharedExperts>;
    DG_STATIC_ASSERT(sizeof(Barrier) == sizeof(uint64_t),
                     "Host SMEM accounting assumes eight-byte barriers");
    DG_STATIC_ASSERT(sizeof(task_info_t) == 32,
                     "Host SMEM accounting assumes 32-byte TaskInfo payloads");
    constexpr uint32_t WG_BLOCK_M = BLOCK_M;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    // The JIT knows the global token count. No routed expert can own more
    // tokens than that, so small-M epilogues only need to materialize chunks
    // up to this compile-time bound. The math mainloop still selects N8/N16/
    // N32/N64 from each expert's actual valid_m.
    constexpr uint32_t kSwapABTokenChunks = kMaxSwapABTokens / 8;
    constexpr uint32_t kSwapABWeightHalves = BLOCK_N / 64;
    constexpr uint32_t kSwapABHalfAccumPerThread = 64 * 64 / 128;
    DG_STATIC_ASSERT(BLOCK_M == 64 and BLOCK_N == 128 and
                         kSwapABWeightHalves == 2,
                     "Small-M swap-AB requires the fixed Humming tile");
    // Two-CTA MXFP4 is compiled at 128 registers/thread. Keeping both the
    // 64-value WGMMA fragment and a 64-value FP32 persistent sum live creates
    // a short local-memory frame. Retain the cross-promotion sum as 32 packed
    // BF16 pairs and expand it only after the mainloop has released the WGMMA
    // fragment. Strict mode multiplies and accumulates in FP32 and rounds only
    // the persistent storage. Fast math also rounds each promotion's scale and
    // WGMMA fragment to BF16 so the packed pairs can be updated with HFMA2.
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");

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

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Combine reuses the pre-barrier SMEM region, including dispatch scratch.
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
        LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // Flash reserves a second expanded tile so the math WG can decode the
    // next packed-B stage while the current phase's final WGMMA group is in
    // flight. Pro cannot reserve another 16 KiB without breaking the two-CTA
    // occupancy contract, so it aliases the second mainloop-only tile with the
    // C/D region, whose lifetime starts after the routed GEMM loop. Shared FP8
    // tasks keep their original expanded/packed slots because their independent
    // B producer may prefetch across the math warpgroup's epilogue.
    constexpr bool kDoubleBufferedMXFP4ExpandedBStorage =
        kHidden == 4096 and
        kOverlapMXFP4ScalePath;
    constexpr bool kAliasMXFP4ExpandedBWithCD =
        kHidden > 4096 and
        kOverlapMXFP4ScalePath;
    constexpr bool kPipelineMXFP4ExpandedB =
        kDoubleBufferedMXFP4ExpandedBStorage or
        kAliasMXFP4ExpandedBWithCD;
    constexpr uint32_t SMEM_B_STORAGE_SIZE =
        (kDoubleBufferedMXFP4ExpandedBStorage ? 2u : 1u) *
            SMEM_B_SIZE_PER_STAGE;
    // Packed MXFP4 is TMA-loaded with B64 swizzle into a temporary half-sized
    // region. The packed row is exactly 64 bytes at BK128, so B64 spreads the
    // one-word-per-row decoder loads across banks. The math warpgroup expands
    // it into the normal B128-swizzled FP8 B tile before issuing WGMMA.
    constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE =
        LOAD_BLOCK_N * BLOCK_K / 2;
    // SFA holds one aligned BLOCK_M-float vector per 64 channels. L1 uses every
    // other vector (per-128); L2 uses all of them (per-64).
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t kNumL2SFAKGroups = BLOCK_K / kL2ActsSFGranK;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        kNumL2SFAKGroups * kL2SFAHalfStride * sizeof(float);
    // MXFP4 relative scales contain one UE8M0 byte per logical (N, K32)
    // group. Preprocessing stores K128 words contiguously across N for the
    // supported DSV4 hidden sizes; larger shapes retain natural row-major
    // storage. The B
    // producer stages one word per output row alongside each packed-B tile.
    constexpr uint32_t kMXFP4WeightGranK = 32;
    constexpr uint32_t kMXFP4CoalescedScaleMaxHidden = 8192;
    constexpr uint32_t kNumMXFP4SFBKGroups = BLOCK_K / kMXFP4WeightGranK;
    constexpr uint32_t kL1MXFP4WeightSFStrideK = kHidden / kMXFP4WeightGranK;
    constexpr uint32_t kL2MXFP4WeightSFStrideK =
        kIntermediateHidden / kMXFP4WeightGranK;
    constexpr uint32_t kL1MXFP4WeightSFPerExpert =
        (kIntermediateHidden * 2) * kL1MXFP4WeightSFStrideK;
    constexpr uint32_t kL2MXFP4WeightSFPerExpert =
        kHidden * kL2MXFP4WeightSFStrideK;
    // The unified persistent specialization stages routed weight SF for both
    // logical phases. One shared-memory layout must cover whichever task the
    // dynamic scheduler publishes next.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE =
        BLOCK_N * kNumMXFP4SFBKGroups * sizeof(uint8_t);
    constexpr uint32_t SMEM_SFB_STORAGE_SIZE =
        kNumStages * SMEM_SFB_SIZE_PER_STAGE;
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 contribution.
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N *
        sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_BLOCK_N *
        kCombineElementBytes;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_BASE_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_OUTPUT_SIZE;
    DG_STATIC_ASSERT(not kAliasMXFP4ExpandedBWithCD or
                         SMEM_CD_SIZE >= SMEM_B_SIZE_PER_STAGE,
                     "C/D alias must hold one expanded MXFP4 B tile");

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * SMEM_A_SIZE_PER_STAGE + SMEM_B_STORAGE_SIZE +
        kNumStages * SMEM_B_PACKED_SIZE_PER_STAGE;

    constexpr uint32_t kCombineInputHiddenBytes = kHidden * kCombineElementBytes;
    constexpr uint32_t kCombineOutputHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kCombineMaxRegistersForBuffer = 128;
    constexpr bool kCombineOneChunkFits =
        kNumEpilogueWarps * (2 * kCombineInputHiddenBytes + kCombineOutputHiddenBytes) <=
            SMEM_BEFORE_BARRIER_SIZE and
        kHidden <= 32 * kCombineMaxRegistersForBuffer;
    constexpr bool kCombineTwoChunksFits =
        kHidden % 2 == 0 and
        kNumEpilogueWarps *
                (2 * (kCombineInputHiddenBytes / 2) + kCombineOutputHiddenBytes / 2) <=
            SMEM_BEFORE_BARRIER_SIZE and
        kHidden <= 2 * 32 * kCombineMaxRegistersForBuffer;
    constexpr uint32_t kCombineNumChunks = kCombineOneChunkFits ? 1 :
        (kCombineTwoChunksFits ? 2 : 4);
    constexpr uint32_t kCombineChunkElems = kHidden / kCombineNumChunks;
    constexpr uint32_t kCombineInputChunkBytes =
        kCombineInputHiddenBytes / kCombineNumChunks;
    constexpr uint32_t kCombineOutputChunkBytes =
        kCombineOutputHiddenBytes / kCombineNumChunks;
    constexpr uint32_t SMEM_COMBINE_ALIAS_SIZE = kNumEpilogueWarps *
        (2 * kCombineInputChunkBytes + kCombineOutputChunkBytes);
    DG_STATIC_ASSERT(kHidden % kCombineNumChunks == 0, "Hidden must be divisible by number of combine chunks");
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

    constexpr uint32_t SMEM_A_OFFSET = SMEM_CD_SIZE;
    constexpr uint32_t SMEM_B_OFFSET =
        SMEM_A_OFFSET + kNumStages * SMEM_A_SIZE_PER_STAGE;
    constexpr uint32_t SMEM_B_PACKED_OFFSET =
        SMEM_B_OFFSET + SMEM_B_STORAGE_SIZE;
    constexpr uint32_t SMEM_SFA_OFFSET =
        SMEM_B_PACKED_OFFSET + kNumStages * SMEM_B_PACKED_SIZE_PER_STAGE;
    constexpr uint32_t SMEM_BARRIER_OFFSET =
        SMEM_SFA_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
        SMEM_SFB_STORAGE_SIZE;
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE == 8192 and
                     SMEM_B_STORAGE_SIZE ==
                         (kDoubleBufferedMXFP4ExpandedBStorage ?
                              32768u : 16384u) and
                     SMEM_B_PACKED_SIZE_PER_STAGE == 8192 and
                     SMEM_SFA_SIZE_PER_STAGE == 512 and
                     SMEM_SFB_SIZE_PER_STAGE == 512 and
                     SMEM_SFB_STORAGE_SIZE == 1536,
                     "Unexpected compact MXFP4 shared-memory tile sizes");
    DG_STATIC_ASSERT(SMEM_A_OFFSET % 128 == 0 and
                     SMEM_B_OFFSET % 128 == 0 and
                     SMEM_B_PACKED_OFFSET % 128 == 0 and
                     SMEM_SFA_OFFSET % 128 == 0 and
                     SMEM_BARRIER_OFFSET % alignof(Barrier) == 0,
                     "SM90 MegaMoE shared-memory regions must be 128-byte aligned");

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(
            smem_gemm_base, SMEM_A_OFFSET + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b_expanded_base = math::advance_ptr<b_dtype_t>(
        smem_gemm_base, SMEM_B_OFFSET);
    auto smem_b_expanded = utils::PatternVisitor([=](const uint32_t& i) {
        DG_DEVICE_ASSERT(i < (kPipelineMXFP4ExpandedB ? 2u : 1u));
        if (i == 0)
            return smem_b_expanded_base;
        if constexpr (kDoubleBufferedMXFP4ExpandedBStorage)
            return smem_b_expanded_base + SMEM_B_SIZE_PER_STAGE;
        return reinterpret_cast<b_dtype_t*>(smem_cd_base);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t&) {
        return smem_b_expanded_base;
    });
    auto smem_b_packed = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(
            smem_gemm_base, SMEM_B_PACKED_OFFSET +
                i * SMEM_B_PACKED_SIZE_PER_STAGE);
    });
    // Shared experts keep FP8 weights. Reuse two disjoint physical B slots
    // from the routed MXFP4 expanded/packed regions and protect the alias with
    // its own two-stage consumer barrier. This preserves two CTAs/SM without
    // reserving three additional 16-KiB FP8 tiles.
    constexpr uint32_t kNumSharedBStages = 2;
    auto smem_shared_b = utils::PatternVisitor([=](const uint32_t& i) {
        DG_DEVICE_ASSERT(i < kNumSharedBStages);
        if (i == 0)
            return smem_b_expanded_base;
        if constexpr (kDoubleBufferedMXFP4ExpandedBStorage)
            return smem_b_expanded[1];
        return reinterpret_cast<b_dtype_t*>(smem_b_packed[0]);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_SFA_OFFSET);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto sfb_start_ptr = sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return sfb_start_ptr + i * SMEM_SFB_SIZE_PER_STAGE;
    });
    // Barriers live after the activation- and weight-SF stages.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        math::advance_ptr(smem_gemm_base, SMEM_BARRIER_OFFSET));
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    constexpr uint32_t kNumBaseBarriers =
        kNumDispatchWarps + kNumStages * 2 + kNumEpilogueWarps * 2;
    constexpr uint32_t kNumScheduleStages = 2;
    constexpr uint32_t kTaskInfoSmemOffset = SMEM_BARRIER_OFFSET +
        (kNumBaseBarriers +
         (kHasSharedExperts ? kNumSharedBStages : 0u) +
         kNumScheduleStages * 2) * sizeof(Barrier);
    DG_STATIC_ASSERT(kTaskInfoSmemOffset % alignof(task_info_t) == 0,
                     "TaskInfo mailboxes must remain 16-byte aligned");
    auto shared_b_empty_barriers = barrier_start_ptr + kNumBaseBarriers;
    auto task_info_full_barriers =
        shared_b_empty_barriers + (kHasSharedExperts ? kNumSharedBStages : 0u);
    auto task_info_empty_barriers =
        task_info_full_barriers + kNumScheduleStages;
    auto task_infos = reinterpret_cast<task_info_t*>(
        task_info_empty_barriers + kNumScheduleStages);

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
                // Producer arrivals: A(+SFA) + B.
                full_barriers[i]->init(2);
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
            if constexpr (kHasSharedExperts) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumSharedBStages; ++ i)
                    shared_b_empty_barriers[i].init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumScheduleStages; ++ i) {
                task_info_full_barriers[i].init(1);
                // A-loader and math threads both consume the CTA-local
                // mailbox. The B-loader may reuse the slot only after all
                // of them have copied the payload into registers.
                task_info_empty_barriers[i].init(
                    kNumEpilogueThreads + 32);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    using scheduler_t = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumSMs, kNumRanks,
        kNumRingBlocks, kNumSharedExperts, 1,
        kSmallMSwapAB and
            (kLocalSwapABTokens == 8 or kLocalSwapABTokens == 64)>;
    auto scheduler = scheduler_t(
        workspace, task_info_full_barriers,
        task_info_empty_barriers, task_infos);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    uint32_t shared_b_stage_idx = 0, shared_b_phase = 0;
    auto advance_shared_b_pipeline = [&]() {
        shared_b_stage_idx ^= 1;
        shared_b_phase ^= shared_b_stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
    constexpr uint32_t kGemmPhaseBoundaryBarrierIdx =
        kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups;
    constexpr uint32_t kExpertCountCacheBarrierIdx =
        kGemmPhaseBoundaryBarrierIdx + 1;
    constexpr bool kCacheProExpertCounts =
        not kHasSharedExperts and kHidden == 7168 and
        ((kSmallMSwapAB and kLocalSwapABTokens == 8) or
         (not kSmallMSwapAB and kBankPermuteMXFP4PairLoads and
          not kSwizzleL2CD));

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration for the fixed 64+64+128-thread CTA consumes
    // half of the SM register file, preserving the two-CTA occupancy contract.
    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 48;
    constexpr uint32_t kNumEpilogueRegisters = 208;
    constexpr uint32_t kCTARegisterBudget =
        kNumDispatchRegisters * kNumDispatchThreads +
        kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
        kNumEpilogueRegisters * kNumEpilogueThreads;
    DG_STATIC_ASSERT(kCTARegisterBudget == 32768,
                     "Two-CTA MXFP4 must use half the SM register file");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    int previous_task_kind = -1;
    const auto sync_task_storage_alias = [&](const bool is_shared_task) {
        if constexpr (kHasSharedExperts) {
            const int task_kind = is_shared_task ? 1 : 0;
            if (previous_task_kind >= 0 and previous_task_kind != task_kind) {
                ptx::sync_unaligned(
                    kNumNonEpilogueThreads + kNumEpilogueThreads,
                    kGemmPhaseBoundaryBarrierIdx);
            }
            previous_task_kind = task_kind;
        }
    };

    const auto invoke_persistent_task = [&](const task_info_t& task_info,
                                             auto&& func) {
        const uint32_t num_k_blocks =
            math::ceil_div(task_info.shape_k, BLOCK_K);
        const uint32_t n_block_idx =
            scheduler_t::get_n_block_idx(task_info);
        if (task_info.block_phase == sched::BlockPhase::Linear1) {
            func(std::integral_constant<
                     sched::BlockPhase, sched::BlockPhase::Linear1>{},
                 task_info.local_expert_idx, num_k_blocks,
                 task_info.m_block_idx, n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m);
        } else if (task_info.block_phase == sched::BlockPhase::Linear2) {
            func(std::integral_constant<
                     sched::BlockPhase, sched::BlockPhase::Linear2>{},
                 task_info.local_expert_idx, num_k_blocks,
                 task_info.m_block_idx, n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m);
        } else if constexpr (kHasSharedExperts) {
            if (task_info.block_phase == sched::BlockPhase::SharedLinear1) {
                func(std::integral_constant<
                         sched::BlockPhase, sched::BlockPhase::SharedLinear1>{},
                     task_info.local_expert_idx, num_k_blocks,
                     task_info.m_block_idx, n_block_idx,
                     task_info.pool_block_idx, task_info.valid_m);
            } else {
                func(std::integral_constant<
                         sched::BlockPhase, sched::BlockPhase::SharedLinear2>{},
                     task_info.local_expert_idx, num_k_blocks,
                     task_info.m_block_idx, n_block_idx,
                     task_info.pool_block_idx, task_info.valid_m);
            }
        }
    };

    const auto for_each_selected_block = [&](auto&& func) {
        task_info_t task_info;
        while (scheduler.get_next_task(task_info))
            invoke_persistent_task(task_info, func);
    };

    const auto produce_selected_blocks = [&](auto&& func) {
        const auto process_task = [&](const task_info_t& task_info) {
            invoke_persistent_task(task_info, func);
        };
        if constexpr (kCacheProExpertCounts)
            scheduler.template mainloop_with_task<false>(
                num_tokens, process_task);
        else
            scheduler.mainloop_with_task(num_tokens, process_task);
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            for (uint32_t i = thread_idx; i < kNumExperts;
                 i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            for (uint32_t i = thread_idx; i < workspace.num_ring_blocks;
                 i += kNumDispatchThreads) {
                *workspace.get_l1_full_count_ptr(i) = 0;
                *workspace.get_l1_empty_count_ptr(i) = 0;
                *workspace.get_l2_full_count_ptr(i) = 0;
                *workspace.get_l2_empty_count_ptr(i) = 0;
            }
            for (uint32_t i = thread_idx;
                 i < workspace.num_shared_l2_pool_blocks;
                 i += kNumDispatchThreads)
                *workspace.get_shared_l2_full_count_ptr(i) = 0;
            if (thread_idx == 0) {
                *workspace.get_l1_task_count_ptr() = 0;
                *workspace.get_l2_task_count_ptr() = 0;
                *workspace.get_shared_l1_task_count_ptr() = 0;
                *workspace.get_shared_l2_task_count_ptr() = 0;
            }
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank;
                 i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                ptx::sync_aligned(
                    kNumDispatchThreads, kDispatchBarrierIdx);
                if (warp_idx == 0) {
                    if (lane_idx == 0)
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    for (uint32_t rank = lane_idx; rank < kNumRanks;
                         rank += 32)
                        *workspace.get_expert_recv_count_ptr(rank, i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and
                        cumulative_local_expert_recv_stats != nullptr) {
                        ptx::red_add(
                            cumulative_local_expert_recv_stats + i,
                            static_cast<int>(num_recv_tokens));
                    }
                }
                __syncwarp();
            }
        }
    };

    // The compact frontend's dispatch and TMA warps share one warpgroup.
    // `setmaxnreg` is warpgroup-collective, so all four warps must execute this
    // single dynamic instruction rather than equivalent role-local call sites.
    if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps)
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
    else
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

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
            const uint32_t local_count = smem_expert_count[i];
#if defined(DG_SM90_SPARSE_DISPATCH_COMPLETION)
            if (local_count != 0) {
#endif
                const uint64_t send_value =
                    (1ull << 32) | static_cast<uint64_t>(local_count);
                smem_expert_count[i] = static_cast<uint32_t>(
                    ptx::atomic_add(
                        workspace.get_expert_send_count_ptr(i), send_value));
#if defined(DG_SM90_SPARSE_DISPATCH_COMPLETION)
            }
#endif
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

        sm90_grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto raw_expert_status =
                    *workspace.get_expert_send_count_ptr(i);
#if defined(DG_SM90_SPARSE_DISPATCH_COMPLETION)
                // The grid barrier proves every CTA has completed its token
                // writes, so publish the equivalent all-CTA completion count.
                const uint64_t expert_status =
                    (static_cast<uint64_t>(kNumSMs) << 32) |
                    static_cast<uint32_t>(raw_expert_status);
#else
                const uint64_t expert_status = raw_expert_status;
#endif
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                // Keep the cross-rank completion protocol identical even when
                // ranks select different local sparse-count specializations.
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        sm90_nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                            kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);

        // Selected Pro buckets make both dispatch warps and the B-loader
        // scheduler independently poll and load the same completed expert
        // totals. Publish one warp's snapshot through the dispatch scratch
        // after the NVLink rendezvous, then let all three consumers initialize
        // their private scheduler state from shared memory.
        if constexpr (kCacheProExpertCounts) {
            if (warp_idx == 0)
                scheduler.cache_expert_recv_count(smem_expert_count);
            ptx::sync_unaligned(
                kNumDispatchThreads + 32, kExpertCountCacheBarrierIdx);
            scheduler.fetch_cached_expert_recv_count(smem_expert_count);
        }

        // Shared L1 does not depend on routed dispatch. Let dispatch pull
        // routed tokens while the math warpgroup computes shared L1 instead
        // of serializing both paths at the frontend barrier.
        if constexpr (not kHasSharedExperts)
            ptx::sync_unaligned(
                kNumDispatchThreads + kNumEpilogueThreads,
                kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        if constexpr (not kCacheProExpertCounts)
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

            uint32_t current_rank_in_expert_idx = 0;
            uint32_t token_idx_in_rank = 0;
            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;

            // At small routed Flash M, most experts receive at most one route
            // from each source rank. Select the source directly from the
            // nonempty mask in that common case; preserve the general
            // round-robin path for duplicate routes from any rank.
            bool used_single_slot_rank_selection = false;
            if constexpr (kSmallMSwapAB and kHidden == 4096 and
                          (kLocalSwapABTokens == 8 or
                           kLocalSwapABTokens == 16 or
                           kLocalSwapABTokens == 32) and
                          kNumRanks <= 32) {
                const uint32_t multi_slot_rank_mask = __ballot_sync(
                    0xffffffff, stored_rank_count[0] > 1);
                if (multi_slot_rank_mask == 0) {
                    const uint32_t nonempty_rank_mask = __ballot_sync(
                        0xffffffff, stored_rank_count[0] != 0);
                    current_rank_in_expert_idx = __fns(
                        nonempty_rank_mask, 0, token_idx_in_expert + 1);
                    used_single_slot_rank_selection = true;
                }
            }

            // General round-robin rank selection (identical to SM100).
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t slot_idx = token_idx_in_expert;
            while (not used_single_slot_rank_selection) {
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
            const uint32_t pool_token_idx =
                expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const uint32_t pool_block_idx = pool_token_idx / BLOCK_M;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t physical_token_idx =
                ring_block_idx * BLOCK_M + token_idx_in_expert % BLOCK_M;

            // Do not overwrite a live L1 slot from the previous generation.
            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            const uint32_t empty_target =
                (pool_block_idx / kNumRingBlocks) * kNumL1BlockNs;
            if (empty_target > 0) {
                const auto empty_ptr =
                    workspace.get_l1_empty_count_ptr(ring_block_idx);
                if constexpr (kHidden == 4096 and
                              kUseIncrementalMXFP4Descriptor) {
                    // Poll the monotonic counter without invalidating L1 on
                    // every miss, then acquire once before reusing the slot.
                    while (ptx::ld_relaxed_gpu(empty_ptr) < empty_target) {}
                    while (ptx::ld_acq(empty_ptr) < empty_target) {}
                } else {
                    while (ptx::ld_acq(empty_ptr) < empty_target) {
                        // For one-block Pro dispatch, avoid hammering the
                        // counter while the wider GEMM tile retires the
                        // previous slot.
                        if constexpr (kNumRanks > 1 and kHidden > 4096) {
                            if (num_tokens <= BLOCK_M)
                                __nanosleep(64);
                        }
                        // Flash M128 wraps padded expert blocks onto a second
                        // ring generation. Briefly back off contending
                        // dispatch lanes.
                        if constexpr (kNumRanks > 1 and kHidden == 4096) {
                            if (num_tokens == 2 * BLOCK_M)
                                __nanosleep(16);
                        }
                    }
                }
            }

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
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFFloats)
                    local_sf_ptr[j * kNumSFRingTokens + physical_token_idx] =
                        remote_sf_ptr[j];
            }
            __syncwarp();

            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(physical_token_idx)
                    .get_base_ptr<float>() = weight;

                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                ptx::tma_store_1d(
                    l1_token_buffer.get_data_buffer(physical_token_idx).get_base_ptr(),
                    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
                const bool is_last_token = token_idx == expert_end_idx - 1;
                const uint32_t token_idx_in_block =
                    token_idx_in_expert % BLOCK_M;
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(ring_block_idx),
                    is_last_token ? BLOCK_M - token_idx_in_block : 1u);
            }
            __syncwarp();
        }

        // Pair with the epilogue after all L2 writes and combine loads are
        // globally visible, then clean every generation/task counter.
        ptx::sync_unaligned(
            kNumDispatchThreads + kNumEpilogueThreads,
            kDispatchWithEpilogueBarrierIdx);
        cleanup_workspace();
        sm90_nvlink_barrier<
            kNumRanks, kNumSMs, kNumDispatchThreads,
            kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                },
                true, false);
        return;

    // =====================================================================
    // ROLE 2: two GEMM TMA producer warps, one for A+SFA and one for B+SFB.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        for_each_selected_block([&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            using BlockPhaseTag = std::remove_cv_t<
                std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool is_linear1_phase =
                BlockPhaseTag::value == sched::BlockPhase::Linear1 or
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear1;
            constexpr bool is_shared_phase =
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear1 or
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear2;
            sync_task_storage_alias(is_shared_phase);
            scheduler.release_task_info();
            const auto tensor_map_a_ptr = is_shared_phase ?
                (is_linear1_phase ? &tensor_map_shared_l1_acts :
                                    &tensor_map_shared_l2_acts) :
                (is_linear1_phase ? &tensor_map_l1_acts :
                                    &tensor_map_l2_acts);
            const auto tensor_map_sfa_ptr = is_shared_phase ?
                (is_linear1_phase ? &tensor_map_shared_l1_acts_sf :
                                    &tensor_map_shared_l2_acts_sf) :
                (is_linear1_phase ? &tensor_map_l1_acts_sf :
                                    &tensor_map_l2_acts_sf);

            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = is_shared_phase ? pool_block_idx :
                ring_block_idx;
            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
            // the tail M unit when an expert has an odd number of M blocks.
            if (has_valid_m) {
                if constexpr (BlockPhaseTag::value == sched::BlockPhase::Linear1) {
                    const auto ptr =
                        workspace.get_l1_full_count_ptr(ring_block_idx);
                    const auto expected = BLOCK_M *
                        (pool_block_idx / kNumRingBlocks + 1);
                    while (ptx::ld_acq(ptr) != expected) {}
                } else if constexpr (BlockPhaseTag::value == sched::BlockPhase::Linear2) {
                    const auto ptr =
                        workspace.get_l2_full_count_ptr(ring_block_idx);
                    const auto expected = (L1_SHAPE_N / BLOCK_N) *
                        (pool_block_idx / kNumRingBlocks + 1);
                    while (ptx::ld_acq(ptr) != expected) {}
                } else if constexpr (
                    BlockPhaseTag::value == sched::BlockPhase::SharedLinear2) {
                    const auto ptr = workspace.get_shared_l2_full_count_ptr(
                        pool_block_idx);
                    constexpr uint32_t kNumSharedL1BlockNs =
                        (kIntermediateHidden * kNumSharedExperts * 2) /
                        BLOCK_N;
                    while (ptx::ld_acq(ptr) != kNumSharedL1BlockNs) {}
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    #pragma unroll
                    for (uint32_t k_tile = 0;
                         k_tile < kNumTMATilesPerStage; ++ k_tile) {
                        tma::copy<kTMATileK, LOAD_BLOCK_M,
                                  kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx],
                            smem_a[stage_idx] +
                                k_tile * LOAD_BLOCK_M * kTMATileK,
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
        if constexpr (kCacheProExpertCounts) {
            ptx::sync_unaligned(
                kNumDispatchThreads + 32, kExpertCountCacheBarrierIdx);
            scheduler.fetch_cached_expert_recv_count(smem_expert_count);
        }
        const auto load_b_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            using BlockPhaseTag = std::remove_cv_t<
                std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool is_linear1_phase =
                BlockPhaseTag::value == sched::BlockPhase::Linear1 or
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear1;
            constexpr bool is_shared_phase =
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear1 or
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear2;
            sync_task_storage_alias(is_shared_phase);
            constexpr bool use_mxfp4_task = not is_shared_phase;
            const auto tensor_map_b_ptr = is_shared_phase ?
                (is_linear1_phase ? &tensor_map_shared_l1_weights :
                                    &tensor_map_shared_l2_weights) :
                (is_linear1_phase ? &tensor_map_l1_weights :
                                    &tensor_map_l2_weights);

            constexpr uint32_t kSharedL1ShapeN =
                kIntermediateHidden * kNumSharedExperts * 2;
            constexpr uint32_t kSharedL2ShapeN = kHidden;
            const uint32_t shape_n = is_shared_phase ?
                (is_linear1_phase ? kSharedL1ShapeN : kSharedL2ShapeN) :
                (is_linear1_phase ? L1_SHAPE_N : L2_SHAPE_N);

            constexpr bool kCoalescedMXFP4WeightSF =
                kHidden <= kMXFP4CoalescedScaleMaxHidden;
            const uint32_t weight_sf_stride_k = is_linear1_phase ?
                kL1MXFP4WeightSFStrideK : kL2MXFP4WeightSFStrideK;
            const uint32_t weight_sf_per_expert = is_linear1_phase ?
                kL1MXFP4WeightSFPerExpert : kL2MXFP4WeightSFPerExpert;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                if constexpr (is_shared_phase)
                    shared_b_empty_barriers[shared_b_stage_idx].wait(
                        shared_b_phase ^ 1);

                const bool elected = cute::elect_one_sync();
                const uint32_t local_n_idx = n_block_idx * BLOCK_N;
                if (elected) {
                    const uint32_t n_idx = is_shared_phase ? local_n_idx :
                        local_expert_idx * shape_n + local_n_idx;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    if constexpr (use_mxfp4_task) {
                        // Load the packed storage as ordinary bytes.  Using a
                        // UINT8 descriptor over K/2 bytes preserves the
                        // existing SM90 CUDA baseline and avoids relying on
                        // newer packed-FP4 TMA layout semantics.
                        tma::copy<BLOCK_K / 2, BLOCK_N, 64, uint8_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx],
                            smem_b_packed[stage_idx], k_idx / 2, n_idx, 1);
                    } else {
                        #pragma unroll
                        for (uint32_t k_tile = 0;
                             k_tile < kNumTMATilesPerStage; ++ k_tile) {
                            #pragma unroll
                            for (uint32_t n_tile = 0;
                                 n_tile < kNumTMANTilesPerStage; ++ n_tile) {
                                tma::copy<kTMATileK, kTMATileN,
                                          kSwizzleBMode, b_dtype_t>(
                                    tensor_map_b_ptr,
                                    full_barriers[stage_idx],
                                    (is_shared_phase ?
                                         smem_shared_b[shared_b_stage_idx] :
                                         smem_b[stage_idx]) +
                                        k_tile * LOAD_BLOCK_N * kTMATileK +
                                        n_tile * kTMATileN * kTMATileK,
                                    k_idx + k_tile * kTMATileK,
                                    n_idx + n_tile * kTMATileN,
                                    1);
                            }
                        }
                    }
                }

                if constexpr (use_mxfp4_task) {
                    // A natural-layout TMA box would have only four contiguous
                    // bytes per row, below Hopper's 16-byte requirement. Use
                    // the B producer warp to issue four strided row groups and
                    // publish them before its transaction-barrier arrival.
                    const auto* weight_sf_base = (is_linear1_phase ?
                        l1_mxfp4_weights_sf : l2_mxfp4_weights_sf) +
                        local_expert_idx * weight_sf_per_expert;
                    if constexpr (kCoalescedMXFP4WeightSF) {
                        // The preprocessed layout makes all 128 K128 scale
                        // words contiguous in N. Coalesce the producer's four
                        // scalar transfers into one 16-byte transaction per
                        // lane for every routed bucket.
                        const auto* weight_sf_words =
                            reinterpret_cast<const uint32_t*>(weight_sf_base) +
                            k_block_idx * shape_n + local_n_idx;
                        const uint4 scale_words = __ldg(
                            reinterpret_cast<const uint4*>(weight_sf_words) +
                            lane_idx);
                        ptx::st_shared(
                            reinterpret_cast<uint4*>(smem_sfb[stage_idx]) +
                                lane_idx,
                            scale_words.x, scale_words.y,
                            scale_words.z, scale_words.w);
                    } else {
                        #pragma unroll
                        for (uint32_t local_n = lane_idx; local_n < BLOCK_N;
                             local_n += 32) {
                            uint32_t scale_word;
                            if constexpr (kCoalescedMXFP4WeightSF) {
                                const auto* weight_sf_words =
                                    reinterpret_cast<const uint32_t*>(
                                        weight_sf_base) +
                                    k_block_idx * shape_n;
                                scale_word = __ldg(
                                    weight_sf_words + local_n_idx + local_n);
                            } else {
                                scale_word = __ldg(
                                    reinterpret_cast<const uint32_t*>(
                                        weight_sf_base +
                                        (local_n_idx + local_n) *
                                            weight_sf_stride_k +
                                        k_block_idx *
                                            kNumMXFP4SFBKGroups));
                            }
                            ptx::st_shared(
                                reinterpret_cast<uint32_t*>(
                                    smem_sfb[stage_idx]) + local_n,
                                scale_word);
                        }
                    }
                    __syncwarp();
                }

                if (elected) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        use_mxfp4_task ? SMEM_B_PACKED_SIZE_PER_STAGE :
                                          SMEM_B_SIZE_PER_STAGE);
                }
                __syncwarp();
                if constexpr (is_shared_phase)
                    advance_shared_b_pipeline();
            }
        };
        produce_selected_blocks(load_b_task);

    } else {
    // =====================================================================
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
    // =====================================================================
        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg = epilogue_warp_idx;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if (lane_idx == 0)
                empty_barriers[s]->arrive();
        };

        // WGMMA-output register layout helpers
        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N = WG_L1_OUT_BLOCK_N;
        DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and
                         WG_BLOCK_N == L1WGMMA::N,
                         "The Humming warpgroup owns one complete WGMMA tile");

        // With shared experts, dispatch and shared L1 are independent and can
        // start concurrently. Their end-of-kernel rendezvous remains paired.
        if constexpr (not kHasSharedExperts)
            ptx::sync_unaligned(
                kNumDispatchThreads + kNumEpilogueThreads,
                kDispatchWithEpilogueBarrierIdx);

        for_each_selected_block([&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            using BlockPhaseTag = std::remove_cv_t<
                std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool is_linear1_phase =
                BlockPhaseTag::value == sched::BlockPhase::Linear1 or
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear1;
            constexpr bool is_shared_phase =
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear1 or
                BlockPhaseTag::value == sched::BlockPhase::SharedLinear2;
            sync_task_storage_alias(is_shared_phase);
            scheduler.release_task_info();
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t block_idx = is_shared_phase ? pool_block_idx :
                ring_block_idx;
            const uint32_t m_idx = block_idx * BLOCK_M;
            const uint32_t pool_m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const auto arrive_task_empty_barrier = [&](const uint32_t& s) {
                arrive_empty_barrier(s);
                if constexpr (is_shared_phase) {
                    if (lane_idx == 0)
                        shared_b_empty_barriers[shared_b_stage_idx].arrive();
                    advance_shared_b_pipeline();
                }
            };
            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            // Regular routed tasks can keep both the WGMMA fragment and the
            // fast-math cross-K sum packed as FP16x2.  Packed accumulation
            // saves conversion/accumulator instructions but increases WGMMA
            // barrier latency, so enable it only when the routed GEMMs expose
            // enough aggregate work to amortize that latency.  Count the
            // M64xN128xK128 mainloop macro-tiles for the two L1 projections
            // and L2.  This depends on the routed workload and topology, not
            // a model name or an exact hidden-size tuple.
            constexpr uint64_t kRoutedMBlockEstimate =
                (static_cast<uint64_t>(kNumMaxTokensPerRank) * kNumTopk +
                 BLOCK_M - 1) / BLOCK_M;
            constexpr uint64_t kWGMMAMacroTilesPerRoutedMBlock =
                (2ull * kIntermediateHidden / BLOCK_N) *
                    (kHidden / BLOCK_K) +
                (kHidden / BLOCK_N) *
                    (kIntermediateHidden / BLOCK_K);
            constexpr uint64_t kRoutedWGMMAMacroTileEstimate =
                kRoutedMBlockEstimate * kWGMMAMacroTilesPerRoutedMBlock;
            constexpr uint64_t kMinPackedF16WGMMAMacroTiles = 1ull << 17;
            constexpr bool kUsePackedF16Accum =
                kFastMath and not kSmallMSwapAB and not kHasSharedExperts and
                kRoutedWGMMAMacroTileEstimate >=
                    kMinPackedF16WGMMAMacroTiles;
            using PackedF16WGMMA = typename
                mma::sm90::FP8MMAF16Selector<WG_BLOCK_N>::type;
            DG_STATIC_ASSERT(
                PackedF16WGMMA::kNumAccum == kAccumPerThread and
                    PackedF16WGMMA::kNumPackedAccum * 2 == kAccumPerThread,
                "Packed FP16 and FP32 WGMMA fragments must cover the same tile");
            float final_accum[
                kPackedBF16SwapEpilogue ? 1 : kAccumPerThread];
            if constexpr (is_shared_phase) {
                #pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                    final_accum[i] = 0.0f;
            }
            // Swap-AB consumes one N64 weight half at a time. Reuse the same
            // fragment for the second half after promotion instead of keeping
            // both halves live across WGMMA. The regular orientation still
            // needs the complete M64xN128 fragment.
            constexpr bool kReuseSwapABFragment = kSmallMSwapAB;
            constexpr uint32_t kAccumStorage = kReuseSwapABFragment ?
                kSwapABHalfAccumPerThread : kAccumPerThread;
            float accum[kAccumStorage];
            uint32_t packed_f16_accum[PackedF16WGMMA::kNumPackedAccum];
            using PersistentAccum2 = std::conditional_t<
                kUsePackedF16Accum, __half2, nv_bfloat162>;
            PersistentAccum2 mxfp4_final_bf16[kAccumPerThread / 2];

            const auto run_mxfp4_gemm_loop = [&]() {
                {
                    constexpr uint32_t kWeightGranK = kMXFP4WeightGranK;
                    constexpr uint32_t kWGThreads = 128;
                    const uint32_t wg_thread_idx = warp_idx_in_wg * 32 + lane_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                        if constexpr (kUsePackedF16Accum)
                            mxfp4_final_bf16[i] =
                                __float2half2_rn(0.0f);
                        else
                            mxfp4_final_bf16[i] =
                                __float2bfloat162_rn(0.0f);
                    }

                    const auto prepare_stage_weights = [&](
                            const uint32_t pipeline_stage,
                            const uint32_t expanded_slot) {
                        const auto* packed = smem_b_packed[pipeline_stage];
                        auto* expanded = reinterpret_cast<uint8_t*>(
                            smem_b_expanded[expanded_slot]);

                        const uint32_t local_n = wg_thread_idx;
                        const uint32_t scale_word = ptx::ld_shared(
                            reinterpret_cast<const uint32_t*>(
                                smem_sfb[pipeline_stage]) + local_n);
                        {
                            DG_STATIC_ASSERT(WG_BLOCK_N == kWGThreads,
                                             "MXFP4 assigns one N row per WG thread");
                            constexpr uint32_t kPackedWordsPerK32 = 32 / 8;
                            constexpr uint32_t kRowsPerDecodeGroup = 8;
                            DG_STATIC_ASSERT(
                                kPackedWordsPerK32 == 4 and
                                32 % kRowsPerDecodeGroup == 0,
                                "MXFP4 warp mapping requires 8 rows x 4 words");

                            constexpr bool kPairPackedWords =
                                kOverlapMXFP4ScalePath;
                            if constexpr (kPairPackedWords) {
                                // Routed small-M gives each lane one complete
                                // packed row. Both adjacent word pairs share
                                // the row's exponent lookup, and the lane owns
                                // the scale directly instead of shuffling it
                                // twice from the two-lane-per-row mapping.
                                constexpr bool kFullRowPairDecode =
                                    kSmallMSwapAB;
                                if constexpr (kFullRowPairDecode) {
                                    const uint32_t decoded_local_n = local_n;
                                    const uint32_t packed_row_base =
                                        decoded_local_n * (BLOCK_K / 2);
                                    const uint32_t packed_row_xor =
                                        cute::Swizzle<2, 4, 3>::apply(
                                            packed_row_base) ^ packed_row_base;
                                    const uint32_t first_pair_in_k32 =
                                        __byte_perm(
                                            0x00020200u, 0u, lane_idx >> 3);
                                    uint2 packed_current[2];
                                    #pragma unroll
                                    for (uint32_t pair_idx = 0;
                                         pair_idx < 2; ++ pair_idx) {
                                        const uint32_t pair_in_k32 =
                                            first_pair_in_k32 ^ (pair_idx * 2u);
                                        const uint32_t packed_byte_offset =
                                            packed_row_base +
                                            ((pair_in_k32 * sizeof(uint32_t)) ^
                                             packed_row_xor);
                                        packed_current[pair_idx] =
                                            ptx::ld_shared(
                                                reinterpret_cast<const uint2*>(
                                                    packed +
                                                    packed_byte_offset));
                                    }
                                    #pragma unroll
                                    for (uint32_t k32_idx = 0;
                                         k32_idx < kNumMXFP4SFBKGroups;
                                         ++ k32_idx) {
                                        uint2 packed_next[2] = {
                                            make_uint2(0u, 0u),
                                            make_uint2(0u, 0u)};
                                        if (k32_idx + 1 <
                                            kNumMXFP4SFBKGroups) {
                                            #pragma unroll
                                            for (uint32_t pair_idx = 0;
                                                 pair_idx < 2; ++ pair_idx) {
                                                const uint32_t pair_in_k32 =
                                                    first_pair_in_k32 ^
                                                    (pair_idx * 2u);
                                                const uint32_t next_packed_pair =
                                                    (k32_idx + 1) *
                                                        kPackedWordsPerK32 +
                                                    pair_in_k32;
                                                const uint32_t
                                                    next_packed_byte_offset =
                                                        packed_row_base +
                                                        ((next_packed_pair *
                                                          sizeof(uint32_t)) ^
                                                         packed_row_xor);
                                                packed_next[pair_idx] =
                                                    ptx::ld_shared(
                                                        reinterpret_cast<
                                                            const uint2*>(
                                                            packed +
                                                            next_packed_byte_offset));
                                            }
                                        }
                                        const uint32_t exponent_offset =
                                            sm90_extract_u8_prmt(
                                                scale_word, k32_idx);
                                        const uint2 lookup =
                                            sm90_mxfp4_e4m3_lookup(
                                                exponent_offset);
                                        #pragma unroll
                                        for (uint32_t pair_idx = 0;
                                             pair_idx < 2; ++ pair_idx) {
                                            const uint2 decoded0 =
                                                sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
                                                    packed_current[pair_idx].x,
                                                    lookup);
                                            const uint2 decoded1 =
                                                sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
                                                    packed_current[pair_idx].y,
                                                    lookup);
                                            const uint32_t pair_in_k32 =
                                                first_pair_in_k32 ^
                                                (pair_idx * 2u);
                                            const uint32_t packed_k_pair =
                                                k32_idx *
                                                    kPackedWordsPerK32 +
                                                pair_in_k32;
                                            const uint32_t flat =
                                                decoded_local_n * BLOCK_K +
                                                packed_k_pair * 8u;
                                            const uint32_t swizzled =
                                                cute::Swizzle<3, 4, 3>::apply(
                                                    flat);
                                            ptx::st_shared(
                                                expanded + swizzled,
                                                decoded0.x, decoded0.y,
                                                decoded1.x, decoded1.y);
                                            packed_current[pair_idx] =
                                                packed_next[pair_idx];
                                        }
                                    }
                                } else {
                                constexpr uint32_t kPairRowsPerDecodeGroup = 16;
                                const uint32_t pair_row_in_decode_group =
                                    lane_idx % 16;
                                // For the conflict-heavy Flash and selected
                                // Pro swap-AB/regular buckets,
                                // alternate the two adjacent word pairs across
                                // each half warp's lower and upper eight rows.
                                // Under the packed B64 swizzle this covers every
                                // bank once per LDS.64 wave instead of aliasing
                                // rows r and r+8 onto the same bank pair. The
                                // full warp still owns the identical 16-row x
                                // 4-word address set.
                                constexpr bool kBankPermutedPairLoads =
                                    kBankPermuteMXFP4PairLoads;
                                constexpr bool kPrmtPairLoads =
                                    kSmallMSwapAB and
                                    ((kHidden == 4096 and
                                      (kLocalSwapABTokens == 8 or
                                       kLocalSwapABTokens == 16 or
                                       kLocalSwapABTokens == 32 or
                                       kLocalSwapABTokens == 64)) or
                                     (kHidden == 7168 and
                                      (kLocalSwapABTokens == 8 or
                                       kLocalSwapABTokens == 32 or
                                       kLocalSwapABTokens == 64)));
                                const uint32_t packed_k_pair_in_k32 =
                                    kBankPermutedPairLoads ?
                                        (kPrmtPairLoads ?
                                             __byte_perm(
                                                 0x00020200u, 0u,
                                                 lane_idx >> 3) :
                                             (((lane_idx >> 3) ^
                                               (lane_idx >> 4)) & 1u) * 2u) :
                                        (lane_idx / 16) * 2;
                                #pragma unroll
                                for (uint32_t row_group = 0;
                                     row_group <
                                         32 / kPairRowsPerDecodeGroup;
                                     ++ row_group) {
                                    const uint32_t row_in_warp =
                                        row_group * kPairRowsPerDecodeGroup +
                                        pair_row_in_decode_group;
                                    const uint32_t decoded_local_n =
                                        warp_idx_in_wg * 32 + row_in_warp;
                                    const uint32_t decoded_scale_word =
                                        __shfl_sync(
                                            0xffffffffu, scale_word, row_in_warp);
                                    const uint32_t packed_row_base =
                                        decoded_local_n * (BLOCK_K / 2);
                                    const uint32_t packed_row_xor =
                                        cute::Swizzle<2, 4, 3>::apply(
                                            packed_row_base) ^ packed_row_base;
                                    const uint32_t first_packed_byte_offset =
                                        packed_row_base +
                                        ((packed_k_pair_in_k32 *
                                          sizeof(uint32_t)) ^ packed_row_xor);
                                    uint2 packed_current = ptx::ld_shared(
                                        reinterpret_cast<const uint2*>(
                                            packed + first_packed_byte_offset));
                                    #pragma unroll
                                    for (uint32_t k32_idx = 0;
                                         k32_idx < kNumMXFP4SFBKGroups;
                                         ++ k32_idx) {
                                        const uint32_t packed_k_pair =
                                            k32_idx * kPackedWordsPerK32 +
                                            packed_k_pair_in_k32;
                                        uint2 packed_next{0u, 0u};
                                        if (k32_idx + 1 <
                                            kNumMXFP4SFBKGroups) {
                                            const uint32_t next_packed_k_pair =
                                                packed_k_pair +
                                                kPackedWordsPerK32;
                                            const uint32_t next_packed_byte_offset =
                                                packed_row_base +
                                                ((next_packed_k_pair *
                                                  sizeof(uint32_t)) ^
                                                 packed_row_xor);
                                            packed_next = ptx::ld_shared(
                                                reinterpret_cast<const uint2*>(
                                                    packed +
                                                    next_packed_byte_offset));
                                        }
                                        const uint32_t exponent_offset = [&]() {
                                            if constexpr (kUsePRMTMXFP4Exponent) {
                                                return sm90_extract_u8_prmt(
                                                    decoded_scale_word, k32_idx);
                                            } else {
                                                return (decoded_scale_word >>
                                                        (k32_idx * 8u)) & 0xffu;
                                            }
                                        }();
                                        uint4 decoded;
                                        if constexpr (kHidden == 4096) {
                                            decoded =
                                                sm90_mxfp4_reordered_signs_e2m1x16_to_e4m3x16_bits(
                                                    packed_current.x,
                                                    packed_current.y,
                                                    exponent_offset);
                                        } else {
                                            const uint2 lookup =
                                                sm90_mxfp4_e4m3_lookup(
                                                    exponent_offset);
                                            const uint2 decoded0 =
                                                sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
                                                    packed_current.x, lookup);
                                            const uint2 decoded1 =
                                                sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
                                                    packed_current.y, lookup);
                                            decoded = make_uint4(
                                                decoded0.x, decoded0.y,
                                                decoded1.x, decoded1.y);
                                        }
                                        const uint32_t logical_k0 =
                                            packed_k_pair * 8;
                                        const uint32_t flat0 =
                                            decoded_local_n * BLOCK_K +
                                            logical_k0;
                                        const uint32_t swizzled0 =
                                            cute::Swizzle<3, 4, 3>::apply(flat0);
                                        ptx::st_shared(
                                            expanded + swizzled0,
                                            decoded.x, decoded.y,
                                            decoded.z, decoded.w);
                                        packed_current = packed_next;
                                    }
                                }
                                }
                            } else {
                                // Each half warp covers the same eight rows and
                                // two disjoint words. This distributes B64 loads
                                // over all 32 banks and also gives each half-warp
                                // B128 STS.64 transaction one bank-word per bank.
                                const uint32_t lane_in_half_warp = lane_idx % 16;
                                const uint32_t row_in_decode_group =
                                    lane_in_half_warp / 2;
                                const uint32_t packed_k_in_k32 =
                                    (lane_idx / 16) * 2 + lane_in_half_warp % 2;
                                #pragma unroll
                                for (uint32_t row_group = 0;
                                     row_group < 32 / kRowsPerDecodeGroup; ++ row_group) {
                                    const uint32_t row_in_warp =
                                        row_group * kRowsPerDecodeGroup +
                                        row_in_decode_group;
                                    const uint32_t decoded_local_n =
                                        warp_idx_in_wg * 32 + row_in_warp;
                                    const uint32_t decoded_scale_word = __shfl_sync(
                                        0xffffffffu, scale_word, row_in_warp);
                                    const uint32_t packed_row_base =
                                        decoded_local_n * (BLOCK_K / 2);
                                    const uint32_t packed_row_xor =
                                        cute::Swizzle<2, 4, 3>::apply(packed_row_base) ^
                                        packed_row_base;

                                    // Reuse the validated two-word LDS lookahead
                                    // in every overlap-enabled routed phase.
                                    // Decoder temporaries die before the first
                                    // QGMMA, so they do not cross accumulator
                                    // lifetime like rejected next-stage overlap.
                                    constexpr bool kPipelinePackedLDS =
                                        kOverlapMXFP4ScalePath;
                                    if constexpr (kPipelinePackedLDS) {
                                        const uint32_t first_packed_byte_offset =
                                            packed_row_base +
                                            ((packed_k_in_k32 * sizeof(uint32_t)) ^
                                             packed_row_xor);
                                        uint32_t packed_current = ptx::ld_shared(
                                            reinterpret_cast<const uint32_t*>(
                                                packed + first_packed_byte_offset));
                                        #pragma unroll
                                        for (uint32_t k32_idx = 0;
                                             k32_idx < kNumMXFP4SFBKGroups;
                                             ++ k32_idx) {
                                            const uint32_t packed_k =
                                                k32_idx * kPackedWordsPerK32 +
                                                packed_k_in_k32;
                                            uint32_t packed_next = 0;
                                            if (k32_idx + 1 <
                                                kNumMXFP4SFBKGroups) {
                                                const uint32_t next_packed_k =
                                                    packed_k + kPackedWordsPerK32;
                                                const uint32_t next_packed_byte_offset =
                                                    packed_row_base +
                                                    ((next_packed_k * sizeof(uint32_t)) ^
                                                     packed_row_xor);
                                                packed_next = ptx::ld_shared(
                                                    reinterpret_cast<const uint32_t*>(
                                                        packed +
                                                        next_packed_byte_offset));
                                            }
                                            const uint32_t exponent_offset = [&]() {
                                                if constexpr (kUsePRMTMXFP4Exponent) {
                                                    return sm90_extract_u8_prmt(
                                                        decoded_scale_word, k32_idx);
                                                } else {
                                                    return (decoded_scale_word >>
                                                            (k32_idx * 8u)) & 0xffu;
                                                }
                                            }();
                                            const uint2 decoded =
                                                sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
                                                    packed_current,
                                                    exponent_offset);
                                            const uint32_t logical_n =
                                                decoded_local_n;
                                            const uint32_t logical_k0 =
                                                packed_k * 8;
                                            const uint32_t flat0 =
                                                logical_n * BLOCK_K + logical_k0;
                                            const uint32_t swizzled0 =
                                                cute::Swizzle<3, 4, 3>::apply(flat0);
                                            ptx::st_shared(
                                                expanded + swizzled0,
                                                decoded.x, decoded.y);
                                            packed_current = packed_next;
                                        }
                                    } else {
                                        #pragma unroll
                                        for (uint32_t k32_idx = 0;
                                             k32_idx < kNumMXFP4SFBKGroups;
                                             ++ k32_idx) {
                                            const uint32_t packed_k =
                                                k32_idx * kPackedWordsPerK32 +
                                                packed_k_in_k32;
                                            const uint32_t exponent_offset = [&]() {
                                                if constexpr (kUsePRMTMXFP4Exponent) {
                                                    return sm90_extract_u8_prmt(
                                                        decoded_scale_word, k32_idx);
                                                } else {
                                                    return (decoded_scale_word >>
                                                            (k32_idx * 8u)) & 0xffu;
                                                }
                                            }();
                                            const uint32_t packed_byte_offset =
                                                packed_row_base +
                                                ((packed_k * sizeof(uint32_t)) ^
                                                 packed_row_xor);
                                            const uint2 decoded =
                                                sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
                                                    ptx::ld_shared(
                                                        reinterpret_cast<const uint32_t*>(
                                                            packed +
                                                            packed_byte_offset)),
                                                    exponent_offset);
                                            const uint32_t logical_n =
                                                decoded_local_n;
                                            const uint32_t logical_k0 = packed_k * 8;
                                            const uint32_t flat0 =
                                                logical_n * BLOCK_K + logical_k0;
                                            const uint32_t swizzled0 =
                                                cute::Swizzle<3, 4, 3>::apply(flat0);
                                            ptx::st_shared(
                                                expanded + swizzled0,
                                                decoded.x, decoded.y);
                                        }
                                    }
                                }
                            }
                        }
                        // Generic shared stores are not ordered with WGMMA's
                        // async proxy by a named barrier alone. Publish every
                        // decoded E4M3 byte before the warpgroup consumes it.
                        cutlass::arch::fence_view_async_shared();
                        ptx::sync_aligned(
                            kWGThreads, kEpilogueWGBarrierStartIdx);
                    };

                    // Producer-staged SF uses source-order release for short
                    // phase-K loops.
                    // For longer loops ptxas may still advance the loop-bottom
                    // arrival once the final stage read and WGMMA wait finish;
                    // no machine-level order relative to register-only HFMA2
                    // is required for correctness.
                    constexpr uint32_t kPhaseKBlocks = is_linear1_phase ?
                        kHidden / BLOCK_K : kIntermediateHidden / BLOCK_K;
                    constexpr bool kEarlyReleaseMXFP4Stage =
                        kPhaseKBlocks <= 16;
                    const auto issue_mxfp4_wgmma = [&]<uint32_t kStartK32,
                                                       uint32_t kNumWGMMAs>(
                            const uint32_t pipeline_stage,
                            const uint32_t expanded_slot) {
                        if constexpr (kUsePackedF16Accum) {
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < PackedF16WGMMA::kNumPackedAccum; ++ i)
                                ptx::warpgroup_fence_operand(
                                    packed_f16_accum[i]);
                        } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                        }
                        ptx::warpgroup_arrive();
                        if constexpr (kUseIncrementalMXFP4Descriptor) {
                            const auto desc_a_base = mma::sm90::make_smem_desc(
                                smem_a[pipeline_stage] +
                                    kStartK32 * kWeightGranK, 1);
                            const auto desc_b_base = mma::sm90::make_smem_desc(
                                smem_b_expanded[expanded_slot] +
                                    kStartK32 * kWeightGranK, 1);
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumWGMMAs; ++ k) {
                                // K32 advances the descriptor's 16-byte start
                                // address field by two without changing layout.
                                const cute::GmmaDescriptor desc_a(
                                    desc_a_base.desc_ + k * 2u);
                                const cute::GmmaDescriptor desc_b(
                                    desc_b_base.desc_ + k * 2u);
                                if constexpr (kUsePackedF16Accum)
                                    PackedF16WGMMA::wgmma(
                                        desc_a, desc_b, packed_f16_accum,
                                        k != 0);
                                else
                                    WGMMA::wgmma(
                                        desc_a, desc_b, accum,
                                        k != 0);
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t k = 0; k < kNumWGMMAs; ++ k) {
                                const uint32_t k32_idx = kStartK32 + k;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[pipeline_stage] +
                                        k32_idx * kWeightGranK, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b_expanded[expanded_slot] +
                                        k32_idx * kWeightGranK, 1);
                                if constexpr (kUsePackedF16Accum)
                                    PackedF16WGMMA::wgmma(
                                        desc_a, desc_b, packed_f16_accum,
                                        k != 0);
                                else
                                    WGMMA::wgmma(
                                        desc_a, desc_b, accum,
                                        k != 0);
                            }
                        }
                        ptx::warpgroup_commit_batch();
                        if constexpr (kUsePackedF16Accum) {
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < PackedF16WGMMA::kNumPackedAccum; ++ i)
                                ptx::warpgroup_fence_operand(
                                    packed_f16_accum[i]);
                        } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                        }
                        if constexpr (not kOverlapMXFP4ScalePath)
                            ptx::warpgroup_wait<0>();
                    };

                    const auto promote_mxfp4 = [&]<bool kReleaseStage>(
                            const uint32_t pipeline_stage,
                            const uint32_t activation_sf_group,
                            const float secondary) {
                        // Rematerialize the cheap row offset at the SFA load.
                        // Keeping it live across decode/WGMMA spills it to the
                        // local stack once per stage on the 128-register build.
                        const uint32_t stage_row_offset_r0 =
                            warp_idx_in_wg * 16 + lane_idx / 4;
                        const float scale_a_0 = ptx::ld_shared(
                            smem_sfa[pipeline_stage] +
                                activation_sf_group * kL2SFAHalfStride +
                                stage_row_offset_r0);
                        const float scale_a_1 = ptx::ld_shared(
                            smem_sfa[pipeline_stage] +
                                activation_sf_group * kL2SFAHalfStride +
                                stage_row_offset_r0 + 8);
                        // Applying the E2M1-to-E4M3 bias correction after
                        // scale_a * secondary can underflow for valid UE8M0
                        // codes 0/1. Apply x64 to the secondary first except
                        // at the high endpoint where that product would
                        // overflow; the branch is uniform for the expert.
                        constexpr float kMaxSecondaryBeforeX64 = 0x1p121f;
                        const bool compensate_secondary =
                            secondary <= kMaxSecondaryBeforeX64;
                        const float compensated_secondary =
                            compensate_secondary ? secondary * 64.0f : secondary;
                        const float compensated_scale_a_0 =
                            compensate_secondary ? scale_a_0 : scale_a_0 * 64.0f;
                        const float compensated_scale_a_1 =
                            compensate_secondary ? scale_a_1 : scale_a_1 * 64.0f;
                        const float combined_scale_0 =
                            compensated_scale_a_0 * compensated_secondary;
                        const float combined_scale_1 =
                            compensated_scale_a_1 * compensated_secondary;
                        // The scale path is independent of the WGMMA result.
                        // Keep the group in flight while loading SFA and
                        // preparing the two promotion multipliers, then wait
                        // immediately before the first accumulator access.
                        if constexpr (kFastMath) {
                            const PersistentAccum2 combined_scale_packed_0 =
                                [&]() {
                                    if constexpr (kUsePackedF16Accum)
                                        return __float2half2_rn(
                                            combined_scale_0);
                                    else
                                        return __float2bfloat162_rn(
                                            combined_scale_0);
                                }();
                            const PersistentAccum2 combined_scale_packed_1 =
                                [&]() {
                                    if constexpr (kUsePackedF16Accum)
                                        return __float2half2_rn(
                                            combined_scale_1);
                                    else
                                        return __float2bfloat162_rn(
                                            combined_scale_1);
                                }();
                            if constexpr (kOverlapMXFP4ScalePath)
                                ptx::warpgroup_wait<0>();
                            // The final wait ends every shared-memory access
                            // to this stage. Release it before register-only
                            // accumulator promotion so the producer can start
                            // refilling the stage in parallel.
                            if constexpr (kReleaseStage) {
                                if constexpr (kOverlapMXFP4ScalePath)
                                    arrive_task_empty_barrier(pipeline_stage);
                            }
                            if constexpr (kUsePackedF16Accum) {
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < PackedF16WGMMA::kNumPackedAccum;
                                     ++ i) {
                                    const auto fragment =
                                        reinterpret_cast<const __half2&>(
                                            packed_f16_accum[i]);
                                    mxfp4_final_bf16[i] = __hfma2(
                                        (i & 1u) ?
                                            combined_scale_packed_1 :
                                            combined_scale_packed_0,
                                        fragment,
                                        mxfp4_final_bf16[i]);
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < kAccumPerThread / 4; ++ i) {
                                    mxfp4_final_bf16[i * 2] = __hfma2(
                                        combined_scale_packed_0,
                                        __floats2bfloat162_rn(
                                            accum[i * 4], accum[i * 4 + 1]),
                                        mxfp4_final_bf16[i * 2]);
                                    mxfp4_final_bf16[i * 2 + 1] = __hfma2(
                                        combined_scale_packed_1,
                                        __floats2bfloat162_rn(
                                            accum[i * 4 + 2],
                                            accum[i * 4 + 3]),
                                        mxfp4_final_bf16[i * 2 + 1]);
                                }
                            }
                        } else {
                            if constexpr (kOverlapMXFP4ScalePath)
                                ptx::warpgroup_wait<0>();
                            if constexpr (kReleaseStage) {
                                if constexpr (kOverlapMXFP4ScalePath)
                                    arrive_task_empty_barrier(pipeline_stage);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float2 persistent_0 =
                                    __bfloat1622float2(mxfp4_final_bf16[i * 2]);
                                const float2 persistent_1 =
                                    __bfloat1622float2(mxfp4_final_bf16[i * 2 + 1]);
                                mxfp4_final_bf16[i * 2] =
                                    __floats2bfloat162_rn(
                                        fmaf(combined_scale_0,
                                             accum[i * 4], persistent_0.x),
                                        fmaf(combined_scale_0,
                                             accum[i * 4 + 1], persistent_0.y));
                                mxfp4_final_bf16[i * 2 + 1] =
                                    __floats2bfloat162_rn(
                                        fmaf(combined_scale_1,
                                             accum[i * 4 + 2], persistent_1.x),
                                        fmaf(combined_scale_1,
                                             accum[i * 4 + 3], persistent_1.y));
                            }
                        }
                    };

                    const float mxfp4_secondary = __ldg(
                        (is_linear1_phase ? l1_mxfp4_secondary : l2_mxfp4_secondary) +
                        local_expert_idx);
                    if constexpr (kSmallMSwapAB) {
                        // The regular M64xN128 orientation wastes nearly all
                        // tensor-core work when each routed expert owns only a
                        // handful of tokens.  Use each N64 weight half as the
                        // WGMMA M operand and bucket valid tokens along N.
                        // Unlike the retired per-tensor path, every K128/K64
                        // group is promoted with the token's staged activation
                        // scale before the temporary accumulator is reused.
                        auto run_swap_ab = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename
                                mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum =
                                SwapWGMMA::kNumAccum;
                            DG_STATIC_ASSERT(
                                kSwapAccum <= kSwapABHalfAccumPerThread,
                                "Invalid swap-AB accumulator bucket");
                            // Selected compact Pro buckets can keep two
                            // bucket-sized fragments inside the existing
                            // 32-float allocation. Commit them as separate
                            // WGMMA groups so promoting half 0 can overlap
                            // execution of half 1. Packed-BF16 buckets and N64
                            // keep reusing one fragment sequentially.
                            constexpr bool kPipelineWeightHalves =
                                kReuseSwapABFragment and
                                not kPackedBF16SwapEpilogue and
                                2 * kSwapAccum <= kAccumStorage;
                            constexpr uint32_t kWeightHalfAccumStride =
                                kPipelineWeightHalves ? kSwapAccum :
                                    kSwapABHalfAccumPerThread;
                            constexpr bool kVectorProM8ActivationScales =
                                kHidden == 7168 and
                                kLocalSwapABTokens == 8;
                            constexpr bool kPrecomputeFlashM16ScaleBeforeWait =
                                kHidden == 4096 and
                                kLocalSwapABTokens == 16;
                            constexpr bool kPrecomputeN8ScaleBeforeWait =
                                kFastMath and
                                (kVectorProM8ActivationScales or
                                 kPrecomputeFlashM16ScaleBeforeWait) and
                                N_SWAP == 8;
                            DG_STATIC_ASSERT(
                                not kPipelineWeightHalves or
                                    2 * kWeightHalfAccumStride <= kAccumStorage,
                                "Pipelined swap-AB fragments exceed storage");
                            const uint32_t swap_col_idx = lane_idx % 4;

                            const auto issue_swap_wgmma = [&]<
                                    uint32_t kFirstWeightHalf,
                                    uint32_t kNumWeightHalves,
                                    uint32_t kStartK32,
                                    uint32_t kNumWGMMAs>(
                                    const uint32_t pipeline_stage,
                                    const uint32_t expanded_slot) {
                                DG_STATIC_ASSERT(
                                    kNumWeightHalves > 0 and
                                        kFirstWeightHalf + kNumWeightHalves <=
                                            kSwapABWeightHalves,
                                    "Invalid swap-AB weight-half range");
                                #pragma unroll
                                for (uint32_t half_idx = 0;
                                     half_idx < kNumWeightHalves; ++ half_idx) {
                                    const uint32_t weight_half =
                                        kFirstWeightHalf + half_idx;
                                    auto* wgmma_accum = accum +
                                        (kNumWeightHalves == 1 and
                                                 not kPipelineWeightHalves ?
                                             0u :
                                            weight_half *
                                                kWeightHalfAccumStride);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(
                                            wgmma_accum[i]);
                                }
                                ptx::warpgroup_arrive();
                                const auto desc_b_base =
                                    mma::sm90::make_smem_desc(
                                        smem_a[pipeline_stage], 1);
                                #pragma unroll
                                for (uint32_t half_idx = 0;
                                     half_idx < kNumWeightHalves; ++ half_idx) {
                                    const uint32_t weight_half =
                                        kFirstWeightHalf + half_idx;
                                    auto* wgmma_accum = accum +
                                        (kNumWeightHalves == 1 and
                                                 not kPipelineWeightHalves ?
                                             0u :
                                            weight_half *
                                                kWeightHalfAccumStride);
                                    const auto desc_a_base =
                                        mma::sm90::make_smem_desc(
                                            smem_b_expanded[expanded_slot] +
                                                weight_half * 64u * BLOCK_K,
                                            1);
                                    #pragma unroll
                                    for (uint32_t k = 0;
                                         k < kNumWGMMAs; ++ k) {
                                        const uint32_t k32_idx = kStartK32 + k;
                                        const cute::GmmaDescriptor desc_a(
                                            desc_a_base.desc_ + k32_idx * 2u);
                                        const cute::GmmaDescriptor desc_b(
                                            desc_b_base.desc_ + k32_idx * 2u);
                                        SwapWGMMA::wgmma(
                                            desc_a, desc_b, wgmma_accum,
                                            k != 0);
                                    }
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t half_idx = 0;
                                     half_idx < kNumWeightHalves; ++ half_idx) {
                                    const uint32_t weight_half =
                                        kFirstWeightHalf + half_idx;
                                    auto* wgmma_accum = accum +
                                        (kNumWeightHalves == 1 and
                                                 not kPipelineWeightHalves ?
                                             0u :
                                            weight_half *
                                                kWeightHalfAccumStride);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(
                                            wgmma_accum[i]);
                                }
                            };

                            const auto promote_swap_ab = [&]<
                                    uint32_t kFirstWeightHalf,
                                    uint32_t kNumWeightHalves,
                                    uint32_t kWaitGroups,
                                    bool kReleaseStage>(
                                    const uint32_t pipeline_stage,
                                    const uint32_t activation_sf_group) {
                                DG_STATIC_ASSERT(
                                    kNumWeightHalves > 0 and
                                        kFirstWeightHalf + kNumWeightHalves <=
                                            kSwapABWeightHalves,
                                    "Invalid swap-AB weight-half range");
                                constexpr float kMaxSecondaryBeforeX64 =
                                    0x1p121f;
                                const bool compensate_secondary =
                                    mxfp4_secondary <=
                                        kMaxSecondaryBeforeX64;
                                const float compensated_secondary =
                                    compensate_secondary ?
                                        mxfp4_secondary * 64.0f :
                                        mxfp4_secondary;
                                const nv_bfloat162
                                    precomputed_scale_pair = [&]() {
                                        if constexpr (
                                                kPrecomputeN8ScaleBeforeWait) {
                                            DG_STATIC_ASSERT(
                                                kSwapAccum / 4 == 1,
                                                "N8 precomputes one scale pair");
                                            const uint32_t token_0 =
                                                swap_col_idx * 2;
                                            const uint32_t token_1 = token_0 + 1;
                                            // Every producer TMA fills the
                                            // complete BLOCK_M SFA vector.  In
                                            // the exact Flash M16/N8 bucket,
                                            // values for padded tokens are
                                            // irrelevant because their
                                            // accumulator columns are never
                                            // written.  Loading them directly
                                            // removes two per-promotion bounds
                                            // selects from the critical path.
                                            const float2 scale_pair = [&]() {
                                                if constexpr (
                                                        kPrecomputeFlashM16ScaleBeforeWait) {
                                                    return ptx::ld_shared(
                                                        reinterpret_cast<
                                                            const float2*>(
                                                            smem_sfa[
                                                                pipeline_stage] +
                                                            activation_sf_group *
                                                                kL2SFAHalfStride +
                                                            token_0));
                                                } else {
                                                    return token_0 < valid_m ?
                                                        ptx::ld_shared(
                                                            reinterpret_cast<
                                                                const float2*>(
                                                                smem_sfa[
                                                                    pipeline_stage] +
                                                                activation_sf_group *
                                                                    kL2SFAHalfStride +
                                                                token_0)) :
                                                        make_float2(0.0f, 0.0f);
                                                }
                                            }();
                                            const float scale_a_0 = scale_pair.x;
                                            const float scale_a_1 = [&]() {
                                                if constexpr (
                                                        kPrecomputeFlashM16ScaleBeforeWait) {
                                                    return scale_pair.y;
                                                } else {
                                                    return token_1 < valid_m ?
                                                        scale_pair.y : 0.0f;
                                                }
                                            }();
                                            const float combined_scale_0 =
                                                (compensate_secondary ?
                                                     scale_a_0 :
                                                     scale_a_0 * 64.0f) *
                                                compensated_secondary;
                                            const float combined_scale_1 =
                                                (compensate_secondary ?
                                                     scale_a_1 :
                                                     scale_a_1 * 64.0f) *
                                                compensated_secondary;
                                            return __floats2bfloat162_rn(
                                                combined_scale_0,
                                                combined_scale_1);
                                        } else {
                                            return __float2bfloat162_rn(0.0f);
                                        }
                                    }();
                                ptx::warpgroup_wait<kWaitGroups>();

                                #pragma unroll
                                for (uint32_t chunk = 0;
                                     chunk < kSwapAccum / 4; ++ chunk) {
                                    const uint32_t token_0 =
                                        chunk * 8 + swap_col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    float combined_scale_0, combined_scale_1;
                                    if constexpr (
                                            not kPrecomputeN8ScaleBeforeWait) {
                                        float scale_a_0, scale_a_1;
                                        if constexpr (
                                                kVectorProM8ActivationScales) {
                                            DG_STATIC_ASSERT(
                                                kL2SFAHalfStride % 2 == 0,
                                                "Vector SFA rows must stay aligned");
                                            const float2 scale_pair =
                                                token_0 < valid_m ?
                                                    ptx::ld_shared(
                                                        reinterpret_cast<
                                                            const float2*>(
                                                            smem_sfa[
                                                                pipeline_stage] +
                                                            activation_sf_group *
                                                                kL2SFAHalfStride +
                                                            token_0)) :
                                                    make_float2(0.0f, 0.0f);
                                            scale_a_0 = scale_pair.x;
                                            scale_a_1 = token_1 < valid_m ?
                                                scale_pair.y : 0.0f;
                                        } else {
                                            scale_a_0 = token_0 < valid_m ?
                                                ptx::ld_shared(
                                                    smem_sfa[pipeline_stage] +
                                                        activation_sf_group *
                                                            kL2SFAHalfStride +
                                                        token_0) :
                                                0.0f;
                                            scale_a_1 = token_1 < valid_m ?
                                                ptx::ld_shared(
                                                    smem_sfa[pipeline_stage] +
                                                        activation_sf_group *
                                                            kL2SFAHalfStride +
                                                        token_1) :
                                                0.0f;
                                        }
                                        combined_scale_0 =
                                            (compensate_secondary ?
                                                 scale_a_0 :
                                                 scale_a_0 * 64.0f) *
                                            compensated_secondary;
                                        combined_scale_1 =
                                            (compensate_secondary ?
                                                 scale_a_1 :
                                                 scale_a_1 * 64.0f) *
                                            compensated_secondary;
                                    }
                                    #pragma unroll
                                    for (uint32_t half_idx = 0;
                                         half_idx < kNumWeightHalves;
                                         ++ half_idx) {
                                        const uint32_t weight_half =
                                            kFirstWeightHalf + half_idx;
                                        const uint32_t accum_offset =
                                            (kNumWeightHalves == 1 and
                                                     not kPipelineWeightHalves ?
                                                 0u :
                                                weight_half *
                                                    kWeightHalfAccumStride) +
                                            chunk * 4;
                                        const uint32_t pair_offset =
                                            weight_half *
                                                (kSwapABHalfAccumPerThread / 2) +
                                            chunk * 2;
                                        // Matched cold-L2 A/B selects the
                                        // endpoint buckets for both routed
                                        // DSV4 models and Flash M16/M32's joint
                                        // PRMT/HFMA2 plus packed-epilogue path.
                                        // Pro M16/M32 retain scalar FP32
                                        // promotion.
                                        if constexpr (
                                                kFastMath and
                                                (kHidden == 4096 or
                                                 kHidden == 7168) and
                                                kSmallMSwapAB and
                                                (kLocalSwapABTokens == 8 or
                                                 kLocalSwapABTokens == 16 or
                                                 (kHidden == 4096 and
                                                  kLocalSwapABTokens == 32 and
                                                  kPackedBF16SwapEpilogue) or
                                                 kLocalSwapABTokens == 64)) {
                                            const nv_bfloat162 scale_pair = [&]() {
                                                if constexpr (
                                                        kPrecomputeN8ScaleBeforeWait) {
                                                    return precomputed_scale_pair;
                                                } else {
                                                    return __floats2bfloat162_rn(
                                                        combined_scale_0,
                                                        combined_scale_1);
                                                }
                                            }();
                                            mxfp4_final_bf16[pair_offset] =
                                                __hfma2(
                                                    scale_pair,
                                                    __floats2bfloat162_rn(
                                                        accum[accum_offset],
                                                        accum[accum_offset + 1]),
                                                    mxfp4_final_bf16[
                                                        pair_offset]);
                                            mxfp4_final_bf16[pair_offset + 1] =
                                                __hfma2(
                                                    scale_pair,
                                                    __floats2bfloat162_rn(
                                                        accum[accum_offset + 2],
                                                        accum[accum_offset + 3]),
                                                    mxfp4_final_bf16[
                                                        pair_offset + 1]);
                                        } else {
                                            const float2 persistent_0 =
                                                __bfloat1622float2(
                                                    mxfp4_final_bf16[
                                                        pair_offset]);
                                            const float2 persistent_1 =
                                                __bfloat1622float2(
                                                    mxfp4_final_bf16[
                                                        pair_offset + 1]);
                                            mxfp4_final_bf16[pair_offset] =
                                                __floats2bfloat162_rn(
                                                    fmaf(combined_scale_0,
                                                         accum[accum_offset],
                                                         persistent_0.x),
                                                    fmaf(combined_scale_1,
                                                         accum[accum_offset + 1],
                                                         persistent_0.y));
                                            mxfp4_final_bf16[pair_offset + 1] =
                                                __floats2bfloat162_rn(
                                                    fmaf(combined_scale_0,
                                                         accum[accum_offset + 2],
                                                         persistent_1.x),
                                                    fmaf(combined_scale_1,
                                                         accum[accum_offset + 3],
                                                         persistent_1.y));
                                        }
                                    }
                                }
                                // SFA belongs to the same producer stage as A.
                                // Release it only after every token scale has
                                // been consumed; otherwise the loader can
                                // overwrite a later chunk during promotion.
                                if constexpr (kReleaseStage)
                                    arrive_task_empty_barrier(pipeline_stage);
                            };

                            for (uint32_t k_block_idx = 0;
                                 k_block_idx < num_k_blocks;
                                 advance_pipeline(k_block_idx)) {
                                const uint32_t expanded_slot =
                                    kPipelineMXFP4ExpandedB ?
                                        (k_block_idx & 1u) : 0u;
                                if constexpr (kPipelineMXFP4ExpandedB) {
                                    if (k_block_idx == 0) {
                                        full_barriers[stage_idx]->wait(phase);
                                        prepare_stage_weights(
                                            stage_idx, expanded_slot);
                                    }
                                } else {
                                    full_barriers[stage_idx]->wait(phase);
                                    prepare_stage_weights(
                                        stage_idx, expanded_slot);
                                }

                                if constexpr (kPipelineWeightHalves) {
                                    if constexpr (is_linear1_phase) {
                                        issue_swap_wgmma.template operator()<
                                            0, 1, 0, 4>(
                                            stage_idx, expanded_slot);
                                        issue_swap_wgmma.template operator()<
                                            1, 1, 0, 4>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            0, 1, 1, false>(stage_idx, 0);
                                    } else {
                                        issue_swap_wgmma.template operator()<
                                            0, 1, 0, 2>(
                                            stage_idx, expanded_slot);
                                        issue_swap_wgmma.template operator()<
                                            1, 1, 0, 2>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            0, 1, 1, false>(stage_idx, 0);
                                        promote_swap_ab.template operator()<
                                            1, 1, 0, false>(stage_idx, 0);
                                        issue_swap_wgmma.template operator()<
                                            0, 1, 2, 2>(
                                            stage_idx, expanded_slot);
                                        issue_swap_wgmma.template operator()<
                                            1, 1, 2, 2>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            0, 1, 1, false>(stage_idx, 1);
                                    }
                                } else if constexpr (kReuseSwapABFragment) {
                                    if constexpr (is_linear1_phase) {
                                        issue_swap_wgmma.template operator()<
                                            0, 1, 0, 4>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            0, 1, 0, false>(stage_idx, 0);
                                        issue_swap_wgmma.template operator()<
                                            1, 1, 0, 4>(
                                            stage_idx, expanded_slot);
                                    } else {
                                        issue_swap_wgmma.template operator()<
                                            0, 1, 0, 2>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            0, 1, 0, false>(stage_idx, 0);
                                        issue_swap_wgmma.template operator()<
                                            1, 1, 0, 2>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            1, 1, 0, false>(stage_idx, 0);
                                        issue_swap_wgmma.template operator()<
                                            0, 1, 2, 2>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            0, 1, 0, false>(stage_idx, 1);
                                        issue_swap_wgmma.template operator()<
                                            1, 1, 2, 2>(
                                            stage_idx, expanded_slot);
                                    }
                                } else {
                                    if constexpr (is_linear1_phase) {
                                        issue_swap_wgmma.template operator()<
                                            0, 2, 0, 4>(
                                            stage_idx, expanded_slot);
                                    } else {
                                        issue_swap_wgmma.template operator()<
                                            0, 2, 0, 2>(
                                            stage_idx, expanded_slot);
                                        promote_swap_ab.template operator()<
                                            0, 2, 0, false>(stage_idx, 0);
                                        issue_swap_wgmma.template operator()<
                                            0, 2, 2, 2>(
                                            stage_idx, expanded_slot);
                                    }
                                }

                                if constexpr (kPipelineMXFP4ExpandedB) {
                                    if (k_block_idx + 1 < num_k_blocks) {
                                        const uint32_t next_stage =
                                            stage_idx == kNumStages - 1 ?
                                                0u : stage_idx + 1u;
                                        const uint32_t next_phase =
                                            phase ^ (next_stage == 0u);
                                        full_barriers[next_stage]->wait(
                                            next_phase);
                                        prepare_stage_weights(
                                            next_stage,
                                            expanded_slot ^ 1u);
                                    }
                                }
                                if constexpr (kPipelineWeightHalves) {
                                    promote_swap_ab.template operator()<
                                        1, 1, 0, true>(
                                        stage_idx,
                                        is_linear1_phase ? 0u : 1u);
                                } else if constexpr (kReuseSwapABFragment) {
                                    promote_swap_ab.template operator()<
                                        1, 1, 0, true>(
                                        stage_idx,
                                        is_linear1_phase ? 0u : 1u);
                                } else {
                                    promote_swap_ab.template operator()<
                                        0, 2, 0, true>(
                                        stage_idx,
                                        is_linear1_phase ? 0u : 1u);
                                }
                            }
                        };

                        const uint32_t n_swap =
                            math::ceil_div(valid_m, 8u) * 8u;
                        if (n_swap <= 8)
                            run_swap_ab.template operator()<8>();
                        else if (n_swap <= 16)
                            run_swap_ab.template operator()<16>();
                        else if (n_swap <= 32)
                            run_swap_ab.template operator()<32>();
                        else
                            run_swap_ab.template operator()<64>();

                        if constexpr (not kPackedBF16SwapEpilogue) {
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < kAccumPerThread / 2; ++ i) {
                                const float2 pair =
                                    __bfloat1622float2(
                                        mxfp4_final_bf16[i]);
                                final_accum[i * 2] = pair.x;
                                final_accum[i * 2 + 1] = pair.y;
                            }
                        }
                    } else {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                         advance_pipeline(k_block_idx)) {
                        const uint32_t expanded_slot =
                            kPipelineMXFP4ExpandedB ?
                                (k_block_idx & 1u) : 0u;
                        if constexpr (kPipelineMXFP4ExpandedB) {
                            // Stage zero is the prologue. Later expanded tiles
                            // were decoded during the previous WGMMA flight.
                            if (k_block_idx == 0) {
                                full_barriers[stage_idx]->wait(phase);
                                prepare_stage_weights(stage_idx, expanded_slot);
                            }
                        } else {
                            full_barriers[stage_idx]->wait(phase);
                            prepare_stage_weights(stage_idx, expanded_slot);
                        }

                        if constexpr (is_linear1_phase) {
                            issue_mxfp4_wgmma.template operator()<0, 4>(
                                stage_idx, expanded_slot);
                            if constexpr (kPipelineMXFP4ExpandedB) {
                                // Decode the next packed stage into the
                                // other expanded slot while this WGMMA
                                // group is still in flight.
                                if (k_block_idx + 1 < num_k_blocks) {
                                    const uint32_t next_stage =
                                        stage_idx == kNumStages - 1 ?
                                            0u : stage_idx + 1u;
                                    const uint32_t next_phase =
                                        phase ^ (next_stage == 0u);
                                    full_barriers[next_stage]->wait(
                                        next_phase);
                                    prepare_stage_weights(
                                        next_stage, expanded_slot ^ 1u);
                                }
                            }
                            promote_mxfp4.template operator()<
                                kEarlyReleaseMXFP4Stage>(
                                stage_idx, 0, mxfp4_secondary);
                        } else {
                            issue_mxfp4_wgmma.template operator()<0, 2>(
                                stage_idx, expanded_slot);
                            promote_mxfp4.template operator()<false>(
                                stage_idx, 0, mxfp4_secondary);
                            issue_mxfp4_wgmma.template operator()<2, 2>(
                                stage_idx, expanded_slot);
                            if constexpr (kPipelineMXFP4ExpandedB) {
                                // L2's first K64 group must be promoted
                                // before the second group can reuse the
                                // fragment. Hide the next packed-stage
                                // decode under the final K64 WGMMA flight.
                                if (k_block_idx + 1 < num_k_blocks) {
                                    const uint32_t next_stage =
                                        stage_idx == kNumStages - 1 ?
                                            0u : stage_idx + 1u;
                                    const uint32_t next_phase =
                                        phase ^ (next_stage == 0u);
                                    full_barriers[next_stage]->wait(
                                        next_phase);
                                    prepare_stage_weights(
                                        next_stage, expanded_slot ^ 1u);
                                }
                            }
                            promote_mxfp4.template operator()<
                                kEarlyReleaseMXFP4Stage>(
                                stage_idx, 1, mxfp4_secondary);
                        }
                        if constexpr (kEarlyReleaseMXFP4Stage) {
                            if constexpr (not kOverlapMXFP4ScalePath)
                                arrive_task_empty_barrier(stage_idx);
                        } else {
                            arrive_task_empty_barrier(stage_idx);
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 2; ++ i) {
                        const float2 pair = [&]() {
                            if constexpr (kUsePackedF16Accum)
                                return __half22float2(
                                    mxfp4_final_bf16[i]);
                            else
                                return __bfloat1622float2(
                                    mxfp4_final_bf16[i]);
                        }();
                        final_accum[i * 2] = pair.x;
                        final_accum[i * 2 + 1] = pair.y;
                    }
                    }
                }
            };

            const auto run_shared_fp8_gemm_loop = [&]() {
                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kTaskIntermediateHidden = is_shared_phase ?
                    kIntermediateHidden * kNumSharedExperts :
                    kIntermediateHidden;
                constexpr uint32_t kL2SFKBlocks =
                    kTaskIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks =
                    kTaskIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert =
                    (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert =
                    (kHidden / 128) * kL2SFKBlocks;
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                     advance_pipeline(k_block_idx)) {
                float gate_sf = 0.0f, up_sf = 0.0f;
                float l2_sf_lo = 0.0f, l2_sf_hi = 0.0f;
                full_barriers[stage_idx]->wait(phase);
                const auto task_smem_b = is_shared_phase ?
                    smem_shared_b[shared_b_stage_idx] : smem_b[stage_idx];

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                const uint32_t stage_row_offset_r0 =
                    warp_idx_in_wg * 16 + lane_idx / 4;
                if (is_linear1_phase) {
                    scale_a_0_lo = ptx::ld_shared(
                        smem_sfa[stage_idx] + stage_row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(
                        smem_sfa[stage_idx] + stage_row_offset_r0 + 8);
                } else {
                    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                    scale_a_0_lo = ptx::ld_shared(
                        smem_sfa[stage_idx] + stage_row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(
                        smem_sfa[stage_idx] + stage_row_offset_r0 + 8);
                    scale_a_0_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride +
                            stage_row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride +
                            stage_row_offset_r0 + 8);
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
                    const uint32_t gate_n = n_block_idx * BLOCK_N / 256u;
                    const uint32_t up_n   = kL1SFGateBlks + gate_n;
                    const float* base = (is_shared_phase ?
                        shared_l1_weights_sf : l1_mxfp4_secondary) +
                        (is_shared_phase ? 0u :
                            local_expert_idx * kL1SFPerExpert) +
                        k_block_idx;
                    gate_sf = __ldg(base + gate_n * kL1SFKBlocks);
                    up_sf   = __ldg(base + up_n * kL1SFKBlocks);
                } else {
                    const uint32_t sf_n = n_block_idx * BLOCK_N / 128u;
                    const float* base = (is_shared_phase ?
                        shared_l2_weights_sf : l2_mxfp4_secondary) +
                        (is_shared_phase ? 0u :
                            local_expert_idx * kL2SFPerExpert) +
                        k_block_idx;
                    l2_sf_lo = __ldg(base + sf_n * kL2SFKBlocks);
                    l2_sf_hi = l2_sf_lo;
                }

                if (is_linear1_phase) {
                    // Single per-128 K-block WGMMA group
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                task_smem_b + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        arrive_task_empty_barrier(stage_idx);

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
                            smem_a[stage_idx] + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            task_smem_b + k * WGMMA::K, 1);
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
                            smem_a[stage_idx] + k_off, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            task_smem_b + k_off, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    arrive_task_empty_barrier(stage_idx);

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
            };

            if constexpr (not is_shared_phase)
                run_mxfp4_gemm_loop();
            else
                run_shared_fp8_gemm_loop();

            if constexpr (BlockPhaseTag::value == sched::BlockPhase::Linear1) {
                // L1 may only overwrite an intermediate slot after every L2 N
                // task from the previous generation has consumed it.
                const auto empty_ptr =
                    workspace.get_l2_empty_count_ptr(ring_block_idx);
                const uint32_t empty_target = (L2_SHAPE_N / BLOCK_N) *
                    (pool_block_idx / kNumRingBlocks);
                while (ptx::ld_acq(empty_ptr) != empty_target) {}
            }
            if constexpr (BlockPhaseTag::value == sched::BlockPhase::Linear2) {
                // Every math thread has completed all A loads before this
                // barrier; one thread releases the physical L2 slot.
                ptx::sync_aligned(
                    kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                    ptx::red_add(
                        workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                __syncwarp();
            }

            // Skip epilogue when block is past valid M (still must release via empty).
            if (valid_m == 0) {
                ptx::sync_aligned(
                    kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            const uint32_t row_idx = lane_idx / 4;
            const uint32_t col_idx = lane_idx % 4;
            const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
            const uint32_t r_1 = r_0 + 8;
            const uint32_t row_offset_r0 = r_0;
            const uint32_t row_offset_r1 = r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            if (is_linear1_phase) {
                auto* smem_cd_l1_wg = smem_cd_l1;
                if constexpr (kSmallMSwapAB and not is_shared_phase) {
                    // Swap-AB maps tokens across WGMMA N and output channels
                    // across lanes/warps.  Derive the dynamic per-token K64
                    // output scale with a two-level warpgroup reduction, then
                    // write the same row-major FP8/SF contract consumed by L2.
                    auto silu = [](float x) {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ?
                            math::fast_rcp(1.0f + e) :
                            1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp !=
                                      cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp !=
                                      cute::numeric_limits<float>::infinity())
                            x = cute::min(
                                cute::max(x, -kActivationClamp),
                                kActivationClamp);
                    };
                    auto run_swap_l1_epilogue = [&]<
                            uint32_t kEpilogueTokenChunks>() {
                        DG_STATIC_ASSERT(
                            kEpilogueTokenChunks ==
                                kLocalSwapABTokens / 8 or
                            kEpilogueTokenChunks ==
                                kSwapABTokenChunks,
                            "Unexpected swap-AB epilogue bucket");
                        float swap_swiglu[
                            kSwapABWeightHalves][kEpilogueTokenChunks][2];
                        const uint32_t num_swap_token_chunks =
                            math::ceil_div(valid_m, 8u);
                        // Pipeline SFA is reusable by the producer immediately
                        // after the math loop releases a stage.  Use C/D
                        // storage, which is epilogue-exclusive, for the
                        // cross-warp amax.
                        auto* swap_scale_scratch =
                            reinterpret_cast<float*>(smem_cd_base);

                        #pragma unroll
                        for (uint32_t chunk = 0;
                             chunk < kEpilogueTokenChunks; ++ chunk) {
                            const uint32_t token_0 =
                                chunk * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            const bool active_chunk =
                                chunk < num_swap_token_chunks;
                            const float weight_0 =
                                active_chunk and token_0 < valid_m ?
                                    *l1_topk_weights_buffer
                                        .get_data_buffer(m_idx + token_0)
                                        .template get_base_ptr<float>() :
                                    0.0f;
                            const float weight_1 =
                                active_chunk and token_1 < valid_m ?
                                    *l1_topk_weights_buffer
                                        .get_data_buffer(m_idx + token_1)
                                        .template get_base_ptr<float>() :
                                    0.0f;
                            #pragma unroll
                            for (uint32_t half = 0;
                                 half < kSwapABWeightHalves; ++ half) {
                                const uint32_t accum_offset =
                                    half * kSwapABHalfAccumPerThread +
                                    chunk * 4;
                                float gate_0, gate_1, up_0, up_1;
                                if constexpr (kPackedBF16SwapEpilogue) {
                                    const float2 gate_pair =
                                        __bfloat1622float2(
                                            mxfp4_final_bf16[
                                                accum_offset / 2]);
                                    const float2 up_pair =
                                        __bfloat1622float2(
                                            mxfp4_final_bf16[
                                                accum_offset / 2 + 1]);
                                    gate_0 = gate_pair.x;
                                    gate_1 = gate_pair.y;
                                    up_0 = up_pair.x;
                                    up_1 = up_pair.y;
                                } else {
                                    gate_0 = final_accum[accum_offset];
                                    gate_1 = final_accum[accum_offset + 1];
                                    up_0 = final_accum[accum_offset + 2];
                                    up_1 = final_accum[accum_offset + 3];
                                }
                                // Keep compile-time chunk indices so the M64
                                // safety bound remains spill-free, but skip
                                // inactive M16 chunks' special-function work.
                                if constexpr (
                                        kHidden == 4096 and
                                        kLocalSwapABTokens == 16) {
                                    if (active_chunk) {
                                        clamp_gate(gate_0);
                                        clamp_gate(gate_1);
                                        clamp_up(up_0);
                                        clamp_up(up_1);
                                        swap_swiglu[half][chunk][0] =
                                            silu(gate_0) * up_0 * weight_0;
                                        swap_swiglu[half][chunk][1] =
                                            silu(gate_1) * up_1 * weight_1;
                                    } else {
                                        swap_swiglu[half][chunk][0] = 0.0f;
                                        swap_swiglu[half][chunk][1] = 0.0f;
                                    }
                                } else {
                                    clamp_gate(gate_0);
                                    clamp_gate(gate_1);
                                    clamp_up(up_0);
                                    clamp_up(up_1);
                                    swap_swiglu[half][chunk][0] =
                                        silu(gate_0) * up_0 * weight_0;
                                    swap_swiglu[half][chunk][1] =
                                        silu(gate_1) * up_1 * weight_1;
                                }
                            }

                            float partial_0 = cute::max(
                                cute::abs(swap_swiglu[0][chunk][0]),
                                cute::abs(swap_swiglu[1][chunk][0]));
                            float partial_1 = cute::max(
                                cute::abs(swap_swiglu[0][chunk][1]),
                                cute::abs(swap_swiglu[1][chunk][1]));
                            #pragma unroll
                            for (uint32_t delta = 4;
                                 delta <= 16; delta *= 2) {
                                partial_0 = cute::max(
                                    partial_0,
                                    __shfl_xor_sync(
                                        0xffffffffu, partial_0, delta));
                                partial_1 = cute::max(
                                    partial_1,
                                    __shfl_xor_sync(
                                        0xffffffffu, partial_1, delta));
                            }
                            if (row_idx == 0 and active_chunk) {
                                swap_scale_scratch[
                                    warp_idx_in_wg * BLOCK_M + token_0] =
                                        partial_0;
                                swap_scale_scratch[
                                    warp_idx_in_wg * BLOCK_M + token_1] =
                                        partial_1;
                            }
                        }
                        ptx::sync_aligned(
                            kNumEpilogueThreads,
                            kEpilogueWGBarrierStartIdx);

                        if (warp_idx_in_wg == 0) {
                            #pragma unroll
                            for (uint32_t half = 0; half < 2; ++ half) {
                                const uint32_t token = lane_idx + half * 32;
                                if (token < valid_m) {
                                    float amax = 0.0f;
                                    #pragma unroll
                                    for (uint32_t warp = 0;
                                         warp < kNumEpilogueWarps; ++ warp)
                                        amax = cute::max(
                                            amax,
                                            swap_scale_scratch[
                                                warp * BLOCK_M + token]);
                                    float2 amax_pair = {amax, amax};
                                    float2 sf_pair, sf_inv_pair;
                                    sm90_fp8_mega_moe_get_e4m3_sf_and_sf_inv(
                                        amax_pair, sf_pair, sf_inv_pair);
                                    swap_scale_scratch[token] = sf_pair.x;
                                    swap_scale_scratch[BLOCK_M + token] =
                                        sf_inv_pair.x;
                                }
                            }
                        }
                        ptx::sync_aligned(
                            kNumEpilogueThreads,
                            kEpilogueWGBarrierStartIdx);

                        float swap_sf_inv[kEpilogueTokenChunks][2];
                        #pragma unroll
                        for (uint32_t chunk = 0;
                             chunk < kEpilogueTokenChunks; ++ chunk) {
                            const uint32_t token_0 =
                                chunk * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            swap_sf_inv[chunk][0] = token_0 < valid_m ?
                                swap_scale_scratch[BLOCK_M + token_0] : 0.0f;
                            swap_sf_inv[chunk][1] = token_1 < valid_m ?
                                swap_scale_scratch[BLOCK_M + token_1] : 0.0f;
                        }
                        if (warp_idx_in_wg == 0) {
                            auto sf_base_ptr =
                                l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t base_k_sf_idx = n_block_idx;
                            #pragma unroll
                            for (uint32_t half = 0; half < 2; ++ half) {
                                const uint32_t token = lane_idx + half * 32;
                                if (token < valid_m)
                                    sf_base_ptr[
                                        base_k_sf_idx * kNumSFRingTokens +
                                        m_idx + token] =
                                        swap_scale_scratch[token];
                            }
                        }
                        // Every thread must retain its inverse scales before
                        // the row-major FP8 stores overwrite C/D scratch.
                        ptx::sync_aligned(
                            kNumEpilogueThreads,
                            kEpilogueWGBarrierStartIdx);

                        #pragma unroll
                        for (uint32_t chunk = 0;
                             chunk < kEpilogueTokenChunks; ++ chunk) {
                            if (chunk < num_swap_token_chunks) {
                                const uint32_t token_0 =
                                    chunk * 8 + col_idx * 2;
                                const uint32_t token_1 = token_0 + 1;
                                #pragma unroll
                                for (uint32_t half = 0;
                                     half < kSwapABWeightHalves; ++ half) {
                                    const uint32_t out_col =
                                        half * 32u +
                                        warp_idx_in_wg * 8 + row_idx;
                                    if (token_0 < valid_m) {
                                        const __nv_fp8_e4m3 q(
                                            swap_swiglu[half][chunk][0] *
                                            swap_sf_inv[chunk][0]);
                                        reinterpret_cast<uint8_t*>(
                                            smem_cd_l1_wg)[
                                                token_0 *
                                                    WG_SMEM_CD_L1_STRIDE_N +
                                                out_col] =
                                            *reinterpret_cast<const uint8_t*>(
                                                &q);
                                    }
                                    if (token_1 < valid_m) {
                                        const __nv_fp8_e4m3 q(
                                            swap_swiglu[half][chunk][1] *
                                            swap_sf_inv[chunk][1]);
                                        reinterpret_cast<uint8_t*>(
                                            smem_cd_l1_wg)[
                                                token_1 *
                                                    WG_SMEM_CD_L1_STRIDE_N +
                                                out_col] =
                                            *reinterpret_cast<const uint8_t*>(
                                                &q);
                                    }
                                }
                            }
                        }
                    };

                    if constexpr (
                            kLocalSwapABTokens == kMaxSwapABTokens or
                            kHidden != 4096 or
                            kLocalSwapABTokens != 8) {
                        run_swap_l1_epilogue.template operator()<
                            kSwapABTokenChunks>();
                    } else if (valid_m <= kLocalSwapABTokens) {
                        run_swap_l1_epilogue.template operator()<
                            kLocalSwapABTokens / 8>();
                    } else {
                        run_swap_l1_epilogue.template operator()<
                            kSwapABTokenChunks>();
                    }

                } else {
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
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
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };

                    if (valid_r0) {
                        swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                        swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                        amax_r0[sf_group] = cute::max(
                            amax_r0[sf_group],
                            cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                    } else {
                        swiglu_r0[p][0] = 0.0f;
                        swiglu_r0[p][1] = 0.0f;
                    }
                    if (valid_r1) {
                        swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                        swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                        amax_r1[sf_group] = cute::max(
                            amax_r1[sf_group],
                            cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                    } else {
                        swiglu_r1[p][0] = 0.0f;
                        swiglu_r1[p][1] = 0.0f;
                    }
                }


                float weight_r0 = 0.0f, weight_r1 = 0.0f;
                if constexpr (kNumMaxTokensPerRank <= 1024) {
                    const int topk_weight_src_lane = static_cast<int>(lane_idx - col_idx);
                    if (col_idx == 0) {
                        weight_r0 = is_shared_phase ? 1.0f :
                            (valid_r0 ? *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + row_offset_r0)
                                .template get_base_ptr<float>() : 0.0f);
                        weight_r1 = is_shared_phase ? 1.0f :
                            (valid_r1 ? *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + row_offset_r1)
                                .template get_base_ptr<float>() : 0.0f);
                    }
                    weight_r0 = __shfl_sync(0xffffffff, weight_r0, topk_weight_src_lane);
                    weight_r1 = __shfl_sync(0xffffffff, weight_r1, topk_weight_src_lane);
                } else {
                    weight_r0 = is_shared_phase ? 1.0f :
                        (valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .template get_base_ptr<float>() : 0.0f);
                    weight_r1 = is_shared_phase ? 1.0f :
                        (valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .template get_base_ptr<float>() : 0.0f);
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
                        col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * WG_SMEM_CD_L1_STRIDE_N +
                        col);
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
                }

                // Write L2-activation SF as float, one value per 64 output columns.
                if (col_idx == 0) {
                    auto sf_base_ptr = is_shared_phase ?
                        shared_l2_sf_buffer.get_base_ptr<float>() :
                        l2_sf_buffer.get_base_ptr<float>();
                    constexpr uint32_t kSharedSFStride =
                        layout::get_num_max_shared_sf_tokens(
                            kNumMaxTokensPerRank);
                    const uint32_t sf_stride = is_shared_phase ?
                        kSharedSFStride : kNumSFRingTokens;
                    const uint32_t token_r0 = m_idx + row_offset_r0;
                    const uint32_t token_r1 = m_idx + row_offset_r1;
                    const uint32_t base_k_sf_idx =
                        n_block_idx * L1_OUT_BLOCK_N / 64u;
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        if (valid_r0)
                            sf_base_ptr[(base_k_sf_idx + g) * sf_stride + token_r0] = sf_r0[g];
                        if (valid_r1)
                            sf_base_ptr[(base_k_sf_idx + g) * sf_stride + token_r1] = sf_r1[g];
                    }
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
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            is_shared_phase ? &tensor_map_shared_l1_output :
                                              &tensor_map_l1_output,
                            smem_cd_l1_wg,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();

                // Publish L1 only after every ordinary SF store and every
                // asynchronous TMA output store is globally visible. One
                // completion is contributed by each L1 N task.
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(
                    kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    if constexpr (is_shared_phase) {
                        ptx::red_add_rel(
                            workspace.get_shared_l2_full_count_ptr(
                                pool_block_idx),
                            1u);
                    } else {
                        ptx::red_add_rel(
                            workspace.get_l2_full_count_ptr(ring_block_idx),
                            1u);
                        ptx::red_add(
                            workspace.get_l1_empty_count_ptr(ring_block_idx),
                            1u);
                    }
                }
                __syncwarp();
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                const auto get_l2_cd_byte_idx = [](const uint32_t byte_idx) {
                    if constexpr (kSwizzleL2CD)
                        return cute::Swizzle<3, 4, 3>::apply(byte_idx);
                    return byte_idx;
                };
                auto store_l2_pair = [&](const uint32_t& elem_idx,
                                         float value0, float value1) {
                    const uint32_t storage_byte_idx = get_l2_cd_byte_idx(
                        elem_idx * sizeof(nv_bfloat16));
                    *reinterpret_cast<uint32_t*>(
                        static_cast<uint8_t*>(smem_cd_base) +
                            storage_byte_idx) =
                            math::cast_into_bf16_and_pack(value0, value1);
                };
                auto store_l2_scalar = [&](const uint32_t& elem_idx,
                                           float value) {
                    const uint32_t storage_byte_idx = get_l2_cd_byte_idx(
                        elem_idx * sizeof(nv_bfloat16));
                    *reinterpret_cast<nv_bfloat16*>(
                        static_cast<uint8_t*>(smem_cd_base) +
                            storage_byte_idx) = __float2bfloat16_rn(value);
                };
                if constexpr (kSmallMSwapAB and not is_shared_phase) {
                        const uint32_t num_swap_token_chunks =
                            math::ceil_div(valid_m, 8u);
                        #pragma unroll
                        for (uint32_t chunk = 0;
                             chunk < kSwapABTokenChunks; ++ chunk) {
                            if (chunk < num_swap_token_chunks) {
                                const uint32_t token_0 =
                                    chunk * 8 + col_idx * 2;
                                const uint32_t token_1 = token_0 + 1;
                                #pragma unroll
                                for (uint32_t half = 0;
                                     half < kSwapABWeightHalves; ++ half) {
                                    const uint32_t accum_offset =
                                        half * kSwapABHalfAccumPerThread +
                                        chunk * 4;
                                    const uint32_t col_offset = half * 64u;
                                    if constexpr (
                                            kPackedBF16SwapEpilogue) {
                                        const float2 gate_pair =
                                            __bfloat1622float2(
                                                mxfp4_final_bf16[
                                                    accum_offset / 2]);
                                        const float2 up_pair =
                                            __bfloat1622float2(
                                                mxfp4_final_bf16[
                                                    accum_offset / 2 + 1]);
                                        if (token_0 < valid_m) {
                                            store_l2_scalar(
                                                token_0 * WG_BLOCK_N +
                                                    col_offset + r_0,
                                                gate_pair.x);
                                            store_l2_scalar(
                                                token_0 * WG_BLOCK_N +
                                                    col_offset + r_1,
                                                up_pair.x);
                                        }
                                        if (token_1 < valid_m) {
                                            store_l2_scalar(
                                                token_1 * WG_BLOCK_N +
                                                    col_offset + r_0,
                                                gate_pair.y);
                                            store_l2_scalar(
                                                token_1 * WG_BLOCK_N +
                                                    col_offset + r_1,
                                                up_pair.y);
                                        }
                                    } else {
                                        if (token_0 < valid_m) {
                                            store_l2_scalar(
                                                token_0 * WG_BLOCK_N +
                                                    col_offset + r_0,
                                                final_accum[
                                                    accum_offset]);
                                            store_l2_scalar(
                                                token_0 * WG_BLOCK_N +
                                                    col_offset + r_1,
                                                final_accum[
                                                    accum_offset + 2]);
                                        }
                                        if (token_1 < valid_m) {
                                            store_l2_scalar(
                                                token_1 * WG_BLOCK_N +
                                                    col_offset + r_0,
                                                final_accum[
                                                    accum_offset + 1]);
                                            store_l2_scalar(
                                                token_1 * WG_BLOCK_N +
                                                    col_offset + r_1,
                                                final_accum[
                                                    accum_offset + 3]);
                                        }
                                    }
                                }
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

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows × 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp → 1 row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    const uint32_t cols_per_lane = WG_BLOCK_N / 16;
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg =
                            warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
                        if constexpr (is_shared_phase) {
                            dst_rank_idx = sym_buffer.rank_idx;
                            dst_token_idx = pool_m_idx + m_idx_in_block;
                            dst_topk_idx = kNumTopk;
                        } else {
                            const auto src_metadata =
                                *workspace.get_token_src_metadata_ptr(
                                    pool_m_idx + m_idx_in_block);
                            dst_rank_idx = src_metadata.rank_idx;
                            dst_token_idx = src_metadata.token_idx;
                            dst_topk_idx = src_metadata.topk_idx;
                        }

                        const uint32_t smem_elem_idx =
                            row_in_wg * WG_BLOCK_N +
                            lane_in_row * cols_per_lane;
                        constexpr uint32_t kScatterBytesPerLane =
                            (WG_BLOCK_N / 16) * kCombineElementBytes;
                        DG_STATIC_ASSERT(
                            kScatterBytesPerLane == 4 or
                            kScatterBytesPerLane == 8 or
                            kScatterBytesPerLane == 16 or
                            kScatterBytesPerLane == 32,
                            "Unexpected L2 scatter width");
                        // B128 swizzle preserves each aligned 16-byte segment,
                        // so the vectorized scatter load stays contiguous while
                        // epilogue stores spread row groups across SMEM banks.
                        const uint32_t storage_smem_byte_idx = get_l2_cd_byte_idx(
                            smem_elem_idx * kCombineElementBytes);
                        auto smem_ptr = math::advance_ptr<uint8_t>(
                            smem_cd_base, storage_smem_byte_idx);
                        const auto dst_token =
                            combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                .get_data_buffer(dst_token_idx);
                        auto dst_ptr = math::advance_ptr<uint8_t>(
                            dst_token.get_base_ptr(),
                            n_idx * kCombineElementBytes +
                                lane_in_row * kScatterBytesPerLane);
                        auto mapped_dst_ptr = sym_buffer.map(dst_ptr, dst_rank_idx);

                        if constexpr (kScatterBytesPerLane == 32) {
                            const auto packed0 =
                                *reinterpret_cast<uint4*>(smem_ptr);
                            const auto packed1 =
                                *(reinterpret_cast<uint4*>(smem_ptr) + 1);
                            reinterpret_cast<uint4*>(mapped_dst_ptr)[0] = packed0;
                            reinterpret_cast<uint4*>(mapped_dst_ptr)[1] = packed1;
                        } else if constexpr (kScatterBytesPerLane == 16) {
                            const auto packed =
                                *reinterpret_cast<uint4*>(smem_ptr);
                            *reinterpret_cast<uint4*>(mapped_dst_ptr) = packed;
                        } else if constexpr (kScatterBytesPerLane == 8) {
                            const auto packed =
                                *reinterpret_cast<uint2*>(smem_ptr);
                            *reinterpret_cast<uint2*>(mapped_dst_ptr) = packed;
                        } else {
                            // The width assertion above leaves only the 4-byte case.
                            *reinterpret_cast<uint32_t*>(mapped_dst_ptr) =
                                *reinterpret_cast<uint32_t*>(smem_ptr);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        });

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        sm90_nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                            kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
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
        DG_STATIC_ASSERT(
            kNumTopk + (kHasSharedExperts ? 1u : 0u) <= 32,
            "Top-k plus shared expert must fit in a single warp");

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
                static_cast<int>(__ldg(
                    input_topk_idx_buffer.get_base_ptr<int64_t>() +
                    token_idx * kNumTopk + lane_idx)) :
                (kHasSharedExperts and lane_idx == kNumTopk ?
                     static_cast<int>(kNumTopk) : -1);
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

                // Fast mode rounds after each routed contribution so the combine
                // state stays in packed BF16 registers; strict mode retains FP32.
                using combine_accum_t = std::conditional_t<
                    kFastMath, nv_bfloat162, float2>;
                combine_accum_t reduced[
                    kNumVectorsPerLane * kNumBF16PairsPerVector] = {};
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
                        for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l) {
                            const uint32_t accum_idx =
                                j * kNumBF16PairsPerVector + l;
                            if constexpr (kFastMath) {
                                reduced[accum_idx] = __hadd2(
                                    reduced[accum_idx], bf16_values[l]);
                            } else {
                                ptx::accumulate(
                                    reduced[accum_idx], bf16_values[l]);
                            }
                        }
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumVectorsPerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumBF16PairsPerVector; ++ l) {
                        const uint32_t accum_idx =
                            j * kNumBF16PairsPerVector + l;
                        if constexpr (kFastMath) {
                            casted_bf16[l] = reduced[accum_idx];
                        } else {
                            casted_bf16[l] = __float22bfloat162_rn(
                                reduced[accum_idx]);
                        }
                    }

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

template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS>
CUTLASS_GLOBAL __launch_bounds__(
    layout::Sm90HummingMoeSchedule::num_dispatch_threads +
        layout::Sm90HummingMoeSchedule::num_non_epilogue_threads +
        layout::Sm90HummingMoeSchedule::num_epilogue_threads,
    2) void
sm90_fp8_mxfp4_mega_moe_persistent_impl(
        DG_SM90_FP8_MOE_KERNEL_ARGS_DECL) {
    sm90_fp8_mega_moe_core<
        DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS>(
            DG_SM90_FP8_MOE_KERNEL_ARGS);
}

#undef DG_SM90_FP8_MOE_TEMPLATE_PARAMS
#undef DG_SM90_FP8_MOE_KERNEL_ARGS_DECL
#undef DG_SM90_FP8_MOE_CORE_ARGS_DECL
#undef DG_SM90_FP8_MOE_KERNEL_ARGS
#undef DG_SM90_FP8_MOE_CORE_TEMPLATE_ARGS

} // namespace deep_gemm

#pragma clang diagnostic pop
