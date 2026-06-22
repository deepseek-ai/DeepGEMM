#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/fp4_decode_detail.cuh>
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

namespace deep_gemm {

__forceinline__ __device__ void sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(
    const float2& amax, float2& sf, float2& sf_inv) {
    constexpr float kScale = 1.0f / 448.0f;
    const auto scaled = make_float2(__fmul_rn(amax.x, kScale), __fmul_rn(amax.y, kScale));
    const auto exp_x = math::fast_log2_ceil(scaled.x);
    const auto exp_y = math::fast_log2_ceil(scaled.y);
    sf.x = math::fast_pow2(exp_x), sf_inv.x = math::fast_pow2(-exp_x);
    sf.y = math::fast_pow2(exp_y), sf_inv.y = math::fast_pow2(-exp_y);
}

struct SM90FP8FP4MegaMoEData {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit SM90FP8FP4MegaMoEData(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
#if defined(__CUDA_ARCH__)
        DG_TRAP_ONLY_DEVICE_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
#else
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
#endif
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct SM90FP8FP4MegaMoEBuffer {
    SM90FP8FP4MegaMoEData data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    void* base;

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoEBuffer(const SM90FP8FP4MegaMoEData& data_layout,
                            const uint32_t& num_ranks,
                            const uint32_t& max_num_tokens_per_rank,
                            void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoEBuffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    SM90FP8FP4MegaMoEData get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
#if defined(__CUDA_ARCH__)
        DG_TRAP_ONLY_DEVICE_ASSERT(num_ranks == 1 or global);
#else
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
#endif
        return SM90FP8FP4MegaMoEData(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

template <uint32_t kNumExpertsPerRank, uint32_t kNumExpertsPerLane, typename Scheduler>
CUTLASS_DEVICE void sm90_fp8_fp4_mega_moe_fetch_cached_expert_recv_count(
    Scheduler& scheduler,
    const uint32_t* cached_recv_counts) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
        const auto expert_idx = i * 32 + ptx::get_lane_idx();
        uint32_t value = 0;
        if (expert_idx < kNumExpertsPerRank)
            value = cached_recv_counts[expert_idx];
        scheduler.stored_num_tokens_per_expert[i] = value;
    }
    __syncwarp();
}

template <
    uint32_t kNumExpertsPerRank,
    uint32_t kNumExpertsPerLane,
    uint32_t kNumL1BlockKs,
    uint32_t kNumL2BlockKs,
    typename Scheduler,
    typename Func>
CUTLASS_DEVICE void sm90_fp8_fp4_mega_moe_for_each_cached_block(
    Scheduler& scheduler,
    Func&& func,
    const uint32_t* cached_recv_counts) {
    sm90_fp8_fp4_mega_moe_fetch_cached_expert_recv_count<
        kNumExpertsPerRank, kNumExpertsPerLane>(scheduler, cached_recv_counts);
    scheduler.set_expert_idx(0);

    while (true) {
        CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
        if (block_phase == sched::BlockPhase::None)
            break;

        func(block_phase, current_local_expert_idx,
             block_phase == sched::BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
             m_block_idx, n_block_idx);
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_wide_load(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    constexpr uint32_t kPackedWordsPerKG = kScaleBGranK / 8;  // 4
    constexpr uint32_t kGroupsPerTile = LOAD_BLOCK_N * kNumSFBPerBlockK;
    DG_STATIC_ASSERT(kPackedWordsPerKG == 4, "Wide-load decode assumes per-32K groups");

    for (uint32_t group = decode_thread_idx; group < kGroupsPerTile; group += num_decode_threads) {
        const uint32_t n_row = group / kNumSFBPerBlockK;
        const uint32_t kg = group - n_row * kNumSFBPerBlockK;
        const uint32_t sfb_word = smem_sfb_stage[n_row];
        const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

        const auto* packed_row = reinterpret_cast<const uint32_t*>(
            smem_b_packed_stage + n_row * (BLOCK_K / 2));
        auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
            smem_b_stage + n_row * BLOCK_K);
        const uint32_t row_swizzle = n_row & 7u;

        const uint32_t seg_base = kg * 2u;
        const uint32_t swz_seg_0 = seg_base ^ row_swizzle;
        const uint32_t swz_seg_1 = (seg_base + 1u) ^ row_swizzle;
        const uint64_t scaled_lut =
            fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
        const uint32_t scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
        const uint32_t scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);

        const uint4 packed = reinterpret_cast<const uint4*>(packed_row)[kg];
        const uint32_t lo_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.x & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.x >> 16, scaled_lut_lo, scaled_lut_hi);
        const uint32_t lo_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.y & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.y >> 16, scaled_lut_lo, scaled_lut_hi);
        const uint32_t lo_2 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.z & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_2 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.z >> 16, scaled_lut_lo, scaled_lut_hi);
        const uint32_t lo_3 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.w & 0xffffu, scaled_lut_lo, scaled_lut_hi);
        const uint32_t hi_3 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
            packed.w >> 16, scaled_lut_lo, scaled_lut_hi);
        ptx::st_shared(
            decoded_row_u64 + swz_seg_0 * 2u,
            lo_0, hi_0, lo_1, hi_1);
        ptx::st_shared(
            decoded_row_u64 + swz_seg_1 * 2u,
            lo_2, hi_2, lo_3, hi_3);
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_vec_store(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    constexpr uint32_t kPackedWordsPerKG = kScaleBGranK / 8;  // 4
    constexpr uint32_t kPackedWordPairsPerKG = kPackedWordsPerKG / 2;
    constexpr uint32_t kGroupsPerTile = LOAD_BLOCK_N * kNumSFBPerBlockK;
    DG_STATIC_ASSERT(kPackedWordsPerKG == 4, "Vector-store decode assumes per-32K groups");

    for (uint32_t group = decode_thread_idx; group < kGroupsPerTile; group += num_decode_threads) {
        const uint32_t n_row = group / kNumSFBPerBlockK;
        const uint32_t kg = group - n_row * kNumSFBPerBlockK;
        const uint32_t sfb_word = smem_sfb_stage[n_row];
        const uint32_t e8m0 = (sfb_word >> (kg * 8)) & 0xffu;

        const auto* packed_row = reinterpret_cast<const uint32_t*>(
            smem_b_packed_stage + n_row * (BLOCK_K / 2));
        auto* decoded_row_u64 = reinterpret_cast<uint64_t*>(
            smem_b_stage + n_row * BLOCK_K);
        const uint32_t row_swizzle = n_row & 7u;
        const uint64_t scaled_lut =
            fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(e8m0);
        const uint32_t scaled_lut_lo = static_cast<uint32_t>(scaled_lut);
        const uint32_t scaled_lut_hi = static_cast<uint32_t>(scaled_lut >> 32);

        #pragma unroll
        for (uint32_t pair = 0; pair < kPackedWordPairsPerKG; ++ pair) {
            const uint32_t pw_global_0 = kg * kPackedWordsPerKG + pair * 2u;
            const uint32_t packed_0 = packed_row[pw_global_0];
            const uint32_t packed_1 = packed_row[pw_global_0 + 1u];
            const uint32_t lo_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_0 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
            const uint32_t hi_0 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_0 >> 16, scaled_lut_lo, scaled_lut_hi);
            const uint32_t lo_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_1 & 0xffffu, scaled_lut_lo, scaled_lut_hi);
            const uint32_t hi_1 = fp4_decode_detail::fp4x4_to_scaled_e4m3x4_lut(
                packed_1 >> 16, scaled_lut_lo, scaled_lut_hi);
            const uint32_t seg_id = pw_global_0 >> 1;
            const uint32_t swz_seg = seg_id ^ row_swizzle;
            ptx::st_shared(
                decoded_row_u64 + swz_seg * 2u,
                lo_0, hi_0, lo_1, hi_1);
        }
    }
}

template <
    uint32_t LOAD_BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t kScaleBGranK,
    uint32_t kNumSFBPerBlockK,
    bool kUseWideLoadDecode,
    typename PackedT,
    typename DecodedT>
__device__ __forceinline__ void dequant_fp4_b_tile_to_e4m3_smem_dispatch(
    const uint32_t decode_thread_idx,
    const uint32_t num_decode_threads,
    const PackedT* __restrict__ smem_b_packed_stage,
    DecodedT* __restrict__ smem_b_stage,
    const uint32_t* __restrict__ smem_sfb_stage) {
    if constexpr (kUseWideLoadDecode) {
        dequant_fp4_b_tile_to_e4m3_smem_wide_load<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK>(
            decode_thread_idx, num_decode_threads,
            smem_b_packed_stage, smem_b_stage, smem_sfb_stage);
    } else {
        dequant_fp4_b_tile_to_e4m3_smem_vec_store<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK>(
            decode_thread_idx, num_decode_threads,
            smem_b_packed_stage, smem_b_stage, smem_sfb_stage);
    }
}

// ============================================================================
// SM90 (Hopper) FP8 x FP4 MegaMoE - software-dequant path.
// ----------------------------------------------------------------------------
// Variant of `sm90_fp8_mega_moe_impl` for DSV4-style packed FP4 expert weights.
// The dispatch / scheduler / SwiGLU / combine machinery is identical to the
// FP8 implementation; the only differences are confined to:
//
//   1. Weight TMA load:  shape changes from (LOAD_BLOCK_N, BLOCK_K) of e4m3
//      to (LOAD_BLOCK_N, BLOCK_K/2) of packed int8 (each byte = 2 nibbles).
//   2. SFB:              loaded as UE8M0 packed int32 (per-32 K granularity)
//      via `cp.async`, since TMA does not natively stride FP4 layouts.
//   3. Mainloop decode:  the host path uses the UE8M0 LUT decoder to dequant
//      the packed FP4 weight tile into an E4M3 shared-memory tile. SS-mode WGMMA
//      then consumes that tile exactly like the FP8 path, preserving the existing
//      per-token SwiGLU amax / quantize epilogue.
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
    bool kUseWideLoadDecode        = false,  // Read one K-group's packed FP4 words as uint4
    bool kMathWGParticipatesInFP4Decode = true,
    uint32_t kNumMathWGDecodeWarps = kMathWGParticipatesInFP4Decode ? (kNumEpilogueThreads / 32) : 0,
    uint32_t kFirstFP4DecodeAssistWarp = 0,  // Skip early non-epilogue warps as decode helpers
    bool kEarlyBDecode            = false,  // Overlap assist decode with A/SFA TMA
    bool kDecodeDoneMBarrier      = false,  // One-way decode-done mbarrier instead of rendezvous sync
    bool kL2ArrivalCounter        = false,  // Count ready L1 output slices instead of bitmask + CTA sync
    bool kFP4SSNSplit             = false,  // Split SS N=128 WGMMA into 2x N=64 to reduce accum pressure
    bool kFP4SwapAB               = false,  // weight@M, token@N for small-batch padding relief
    bool kFP4SwapABFastAmax       = false,  // Reuse L1 swapAB store lanes to publish per-token partial amax
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
sm90_fp8_fp4_mega_moe_impl(void* y,
                           int* cumulative_local_expert_recv_stats,
                           const uint32_t num_tokens,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                           const uint32_t* __restrict__ l1_weights_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                           const uint32_t* __restrict__ l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads == 64 or kNumDispatchThreads == 128,
                     "Dispatch supports 2 or 4 warps");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads >= 128 and kNumNonEpilogueThreads % 64 == 0,
                     "Invalid number of GEMM TMA/decode-assist warps");
    DG_STATIC_ASSERT((kNumDispatchThreads + kNumNonEpilogueThreads) % 128 == 0,
                     "Math warps must start on a warpgroup boundary");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0, "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N % 8 == 0, "BLOCK_N must be compatible with SM90 FP8 WGMMA shapes");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");
    DG_STATIC_ASSERT(kNumMathWGDecodeWarps <= kNumEpilogueWarps,
                     "Math decode warps cannot exceed epilogue warps");
    DG_STATIC_ASSERT(kMathWGParticipatesInFP4Decode or kNumMathWGDecodeWarps == 0,
                     "Math decode warp count requires math WG decode participation");
    DG_STATIC_ASSERT(kFirstFP4DecodeAssistWarp <= kNumMMANonEpilogueWarps,
                     "First FP4 decode assist warp is out of range");
    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

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

    constexpr auto fp8_token_layout              = SM90FP8FP4MegaMoEData(kHidden);
    constexpr auto bf16_token_layout             = SM90FP8FP4MegaMoEData(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = SM90FP8FP4MegaMoEData(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = SM90FP8FP4MegaMoEData(kHidden / 32);
    // L2 activation SF is per-64 for BLOCK_N=128 and per-32 for BLOCK_N=64.
    constexpr uint32_t kL2ActsSFGranK = BLOCK_N == 64 ? 32 : 64;
    constexpr auto fp8_intermediate_sf_layout =
        SM90FP8FP4MegaMoEData(kIntermediateHidden * sizeof(float) / kL2ActsSFGranK);
    constexpr auto input_topk_idx_layout         = SM90FP8FP4MegaMoEData(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = SM90FP8FP4MegaMoEData(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = SM90FP8FP4MegaMoEData(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = SM90FP8FP4MegaMoEBuffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = SM90FP8FP4MegaMoEBuffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = SM90FP8FP4MegaMoEBuffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = SM90FP8FP4MegaMoEBuffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = SM90FP8FP4MegaMoEBuffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = SM90FP8FP4MegaMoEBuffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = SM90FP8FP4MegaMoEBuffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = SM90FP8FP4MegaMoEBuffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = SM90FP8FP4MegaMoEBuffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = SM90FP8FP4MegaMoEBuffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    // The WGMMA still consumes E4M3 on both operands. We reuse the SS-mode
    // selector and dequant the packed FP4 weight into a second SMEM tile of
    // E4M3 right before issuing each k-block's WGMMA group; see decode_b_tile.
    using b_dtype_t        = cutlass::float_e4m3_t;
    // Storage type for the packed FP4 weight tile in SMEM/global. Each byte
    // packs 2 nibbles (low nibble = lower-K element, high nibble = upper-K),
    // matching DSV4's TMA-friendly layout.
    using b_packed_dtype_t = int8_t;
    // FP4 SS split-N infrastructure: when BLOCK_M=64 and BLOCK_N is a
    // multiple of 128, the host heuristics may request
    // `kNumEpilogueWarpgroups == BLOCK_N / 128 > 1` math warpgroups. In that
    // mode every WG shares the same BLOCK_M rows and partitions the N
    // columns, so each WG owns WG_BLOCK_N = BLOCK_N / num_wg columns. The
    // packed-B / SFB / decoded-B SMEM tiles still cover the full LOAD_BLOCK_N
    // because FP4 decode is shared across WGs (see comment on smem_b below);
    // split-N only manifests in WGMMA descriptors and the L1/L2 epilogue.
    constexpr bool kSplitNWarpgroups =
        BLOCK_M == 64 and
        kNumEpilogueWarpgroups > 1 and
        BLOCK_N % kNumEpilogueWarpgroups == 0 and
        (BLOCK_N / kNumEpilogueWarpgroups) >= 64;
    constexpr uint32_t kWarpgroupSplitM = kSplitNWarpgroups ? 1u : kNumEpilogueWarpgroups;
    constexpr uint32_t kWarpgroupSplitN = kSplitNWarpgroups ? kNumEpilogueWarpgroups : 1u;
    constexpr uint32_t WG_BLOCK_M = BLOCK_M / kWarpgroupSplitM;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kWarpgroupSplitN;
    constexpr bool kSwapABEligible =
        kFP4SwapAB and kSplitNWarpgroups and (BLOCK_M == 64) and (BLOCK_N == 128)
        and (kWarpgroupSplitN == 2) and (not kFP4SSNSplit);
    constexpr bool kSwapABL1Active = kSwapABEligible;
    constexpr bool kSwapABL2Active = kSwapABEligible;
    constexpr bool kSwapABFastAmaxActive =
        kSwapABL1Active and kFP4SwapABFastAmax;
    // Flash b4 is assigned a distinct epw16 kernel. Keep one intermediate
    // swap bucket there so this band avoids unnecessary padding without
    // making the ultra-small epw32 kernel heavier.
    constexpr bool kSwapABFlashN24Dispatch =
        kSwapABEligible and kIntermediateHidden <= 2048 and kNumExpertsPerWave == 16;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    DG_STATIC_ASSERT(not kSwapABEligible or (BLOCK_M % 8 == 0),
                     "swapAB epilogue token chunks assume BLOCK_M is a multiple of 8");
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;  // M=64, N=WG_BLOCK_N, K=32
    using L2WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups,
                     "Invalid warpgroup split");
    DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M,
                     "Each warpgroup must run exactly one WGMMA-M tile");
    DG_STATIC_ASSERT(BLOCK_M % kWarpgroupSplitM == 0 and BLOCK_N % kWarpgroupSplitN == 0,
                     "Invalid warpgroup tile shape");

    // Cluster=1 -> no multicast, A/B are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N  = BLOCK_N / 2;  // post-SwiGLU
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2;
    // In the split-N=2, BLOCK_N=128 path each WG produces only
    // WG_L1_OUT_BLOCK_N post-SwiGLU columns. WG0 issues one combined 64-column
    // TMA store after both WGs reduce amax, matching the 64-column SF block.
    constexpr bool kSplitNCombinesL1Store = kSplitNWarpgroups and (WG_L1_OUT_BLOCK_N < 64);
    constexpr bool kSplitNSharesSF = kSplitNWarpgroups and (WG_L1_OUT_BLOCK_N < kL2ActsSFGranK);
    DG_STATIC_ASSERT(not kSplitNSharesSF or kSplitNWarpgroups,
                     "share-SF only meaningful under split-N");
    DG_STATIC_ASSERT(not kSplitNSharesSF or (kWarpgroupSplitN == 2),
                     "share-SF currently only supports split-N=2");
    DG_STATIC_ASSERT(not kSwapABFastAmaxActive or kSplitNSharesSF,
                     "swapAB fast-amax currently assumes the split-N shared-SF shape");
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    // The decoded E4M3 B tile uses 128B swizzle to match the SS WGMMA
    // descriptor. The packed FP4 source tile is linear in the default
    // decode-to-SMEM path because only the dequant code reads it by (row, col).
    constexpr uint32_t kSwizzleBMode        = BLOCK_K * sizeof(b_dtype_t);  // 128
    constexpr uint32_t kSwizzleBPackedMode  = 0;
    constexpr uint32_t kSwizzleCDMode  = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF base granularity
    constexpr uint32_t kNumL2SFAPerBlockK = BLOCK_K / kL2ActsSFGranK;
    // SFB granularity for FP4 weights: per-32 K (DSV4 standard, UE8M0).
    // BLOCK_K=128 has 4 SFB groups along K, exactly one per WGMMA::K tile.
    constexpr uint32_t kScaleBGranK     = 32;
    constexpr uint32_t kNumSFBPerBlockK = BLOCK_K / kScaleBGranK;  // 4
    static_assert(L1WGMMA::K == kScaleBGranK,
                  "WGMMA::K must equal kScaleBGranK so that 1 wgmma == 1 SFB block");

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
    // Decoded e4m3 B tile (consumed by WGMMA via SS descriptor)
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
        LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // Packed FP4 source tile (TMA-loaded raw nibbles)
    constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE = LOAD_BLOCK_N * (BLOCK_K / 2) * sizeof(b_packed_dtype_t);
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and
    // L2 (2*BLOCK_M floats per-64, or 4*BLOCK_M floats per-32 with BLOCK_N=64).
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(kNumL2SFAPerBlockK * BLOCK_M * sizeof(float), 128u);
    // SFB UE8M0 per-32: the decode-to-SMEM path stages one packed uint32 per
    // N row per BLOCK_K in SMEM. Each word contains the 4 K/32 scale bytes, so
    // dequant avoids reloading the same word once per K group.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(LOAD_BLOCK_N * sizeof(uint32_t), 128u);

    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes). With split-M each math WG
    // writes a disjoint WG_BLOCK_M slice (rows are partitioned), and with
    // split-N each WG writes a disjoint column slice of the same row range
    // (rows are shared); in both cases the total rows x cols footprint is
    // exactly BLOCK_M x BLOCK_N (resp. BLOCK_M x L1_OUT_BLOCK_N for the L1
    // FP8 staging tile), so the total size does NOT scale with
    // kNumEpilogueWarpgroups.
    constexpr uint32_t SMEM_CD_L1_SIZE = BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        (kSwapABL1Active and not kSwapABFastAmaxActive)
            ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(float)
            : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        kSwapABL1Active ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_AMAX_SIZE =
        kSwapABL1Active ? BLOCK_M * kNumEpilogueWarps * sizeof(float) : 0;
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        kSwapABL1Active ? (SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE) : 0;
    constexpr uint32_t SMEM_CD_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_SIZE    = math::constexpr_align(
        SMEM_CD_BASE_SIZE > SMEM_CD_SWAP_L1_SIZE ? SMEM_CD_BASE_SIZE : SMEM_CD_SWAP_L1_SIZE,
        kSharedMemoryAlignment);
    DG_STATIC_ASSERT(not kSwapABL1Active or SMEM_CD_SWAP_L1_AMAX_SIZE <= SMEM_CD_SWAP_L1_FP8_SIZE,
                     "swapAB fast-amax partials must fit in the FP8 staging tile");

    // When SF is shared by two split-N WGs, reduce the per-row amax in SMEM.
    // Only col_idx==0 lanes write SF, so those lanes publish each WG's amax,
    // synchronize once, and read both halves back to avoid atomicMax and a
    // second epilogue-wide barrier.
    //   row_slot = warp_idx_in_wg * 8 + row_idx  (row_idx = lane_idx / 4)
    //   scratch[row_slot][wg_n_idx][r0/r1]
    // 32 rows x 2 WGs x 2 row values (r0, r1) = 128 float slots.
    constexpr uint32_t kAmaxScratchSlots = 32 * 2 * 2;
    constexpr uint32_t SMEM_AMAX_SCRATCH_SIZE = kSplitNSharesSF ?
        math::constexpr_align<uint32_t>(kAmaxScratchSlots * sizeof(uint32_t),
                                        kSharedMemoryAlignment) : 0;

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        SMEM_AMAX_SCRATCH_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                      SMEM_B_PACKED_SIZE_PER_STAGE +
                      SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = SM90FP8FP4MegaMoEBuffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_gemm_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_gemm_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SWAP_L1_FP32_SIZE));
    auto smem_cd_swap_l1_amax = reinterpret_cast<float*>(smem_cd_swap_l1_fp8);
    // share-SF amax scratch lives in its own region after SMEM_CD.
    auto smem_amax_scratch = reinterpret_cast<uint32_t*>(
        math::advance_ptr(smem_gemm_base, SMEM_CD_SIZE));

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    // Decoded e4m3 B tile (the operand actually consumed by WGMMA).
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base,
            SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE +
            kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    // Packed FP4 source tile (TMA-loaded; consumed only by the math warpgroup
    // during the FP4-to-E4M3 dequant pass).
    auto smem_b_packed = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_packed_dtype_t>(smem_gemm_base,
            SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE
            + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
            + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + SMEM_AMAX_SCRATCH_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE
                      + SMEM_B_SIZE_PER_STAGE
                      + SMEM_B_PACKED_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });

    auto sfb_start_ptr = sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sfb_start_ptr + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers live after SFA and staged SFB.
    constexpr bool kUseEarlyBDecode = kEarlyBDecode;
    constexpr uint32_t kNumDecodeFullBarriers = kUseEarlyBDecode ? kNumStages : 0;
    constexpr bool kUseDecodeDoneMBarrier = kDecodeDoneMBarrier;
    constexpr uint32_t kNumDecodeDoneBarriers = kUseDecodeDoneMBarrier ? kNumStages : 0;
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sfb_start_ptr + kNumStages * SMEM_SFB_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto decode_full_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + i;
    });
    auto decode_done_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + kNumDecodeFullBarriers + i;
    });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + kNumDecodeFullBarriers + kNumDecodeDoneBarriers + i;
    });
    auto combine_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + kNumDecodeFullBarriers + kNumDecodeDoneBarriers + kNumStages + i;
    });

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
                // Default path uses one A+B full barrier. The early-B path
                // splits packed-B readiness so assist warps can decode while
                // A/SFA TMA is still in flight; the main full barrier then only
                // tracks the A/SFA producer.
                full_barriers[i]->init(kUseEarlyBDecode ? 1 : 2);
                if constexpr (kUseEarlyBDecode)
                    decode_full_barriers[i]->init(1);
                if constexpr (kUseDecodeDoneMBarrier) {
                    // decode_done is a one-way producer->consumer mbarrier:
                    // only the warps that actually run `decode_fp4_b_stage`
                    // arrive on it (via `arrive_or_sync_fp4_decode_done`).
                    // Those are the decode-assist warps -- i.e.
                    // `kNumMMANonEpilogueWarps` minus the leading loader warps
                    // that skip decode-assist when `kFirstFP4DecodeAssistWarp`
                    // > 0 -- plus the optional math-WG decode warps. Counting
                    // all `kNumMMANonEpilogueWarps` here over-counts arrivals
                    // by `kFirstFP4DecodeAssistWarp`, so the consumer `wait()`
                    // would never complete.
                    constexpr uint32_t kDecodeDoneArrivers =
                        (kNumMMANonEpilogueWarps - kFirstFP4DecodeAssistWarp) +
                        kNumMathWGDecodeWarps;
                    decode_done_barriers[i]->init(kDecodeDoneArrivers);
                }
                // Each math warp arrives once per stage release.
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // =====================================================================
    // Scheduler
    // =====================================================================
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks>(workspace);
    constexpr uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
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
    constexpr uint32_t kSchedulerCountCacheBarrierIdx   = 14;
    constexpr uint32_t kFP4DecodeBarrierIdx             = 15;
    DG_STATIC_ASSERT(kEpilogueWGBarrierStartIdx + kNumEpilogueWarpgroups <= kSchedulerCountCacheBarrierIdx,
                     "Epilogue WG barriers overlap scheduler-count cache barrier");
    const uint32_t* cached_recv_counts = smem_expert_count;
    auto cache_expert_recv_counts = [&]() {
        if (thread_idx < kNumExpertsPerRank) {
            uint64_t value = 0;
            do {
                value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(thread_idx));
            } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            smem_expert_count[thread_idx] = static_cast<uint32_t>(value);
        }
        ptx::sync_unaligned(kNumThreads, kSchedulerCountCacheBarrierIdx);
    };

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // Split-N halves the live accumulator footprint per math warpgroup, so it
    // does not need the full 208-register epilogue allocation used by the
    // regular N=128 path.
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters    = kSplitNWarpgroups ? 160 : 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // SFB UE8M0 layouts (one uint32 per (n_row, 4 K-groups = BLOCK_K=128)):
    //   L1: shape [E, 2*IH, H/128] uint32, gran_mn=1 along N.
    //   L2: shape [E, H, IH/128] uint32.
    constexpr uint32_t kL1SFBKWords     = kHidden / 128;
    constexpr uint32_t kL2SFBKWords     = kIntermediateHidden / 128;
    constexpr uint32_t kL1SFBPerExpert  = (kIntermediateHidden * 2) * kL1SFBKWords;
    constexpr uint32_t kL2SFBPerExpert  = kHidden * kL2SFBKWords;
    constexpr uint32_t kNumFP4DecodeAssistWarps =
        kNumMMANonEpilogueWarps - kFirstFP4DecodeAssistWarp;
    constexpr uint32_t kNumFP4DecodeAssistThreads = kNumFP4DecodeAssistWarps * 32;
    constexpr uint32_t kNumFP4DecodeWorkerThreads = kNumFP4DecodeAssistThreads +
        kNumMathWGDecodeWarps * 32;
    constexpr uint32_t kNumFP4DecodeBarrierThreads =
        kNumFP4DecodeAssistThreads + kNumEpilogueThreads;
    auto arrive_or_sync_fp4_decode_done = [&](const uint32_t& cur_stage_idx) {
        if constexpr (kUseDecodeDoneMBarrier) {
            __syncwarp();
            if (lane_idx == 0)
                decode_done_barriers[cur_stage_idx]->arrive();
        } else {
            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
        }
    };
    auto wait_fp4_decode_done = [&](const uint32_t& cur_stage_idx,
                                    const uint32_t& cur_phase) {
        if constexpr (kUseDecodeDoneMBarrier) {
            decode_done_barriers[cur_stage_idx]->wait(cur_phase);
        } else {
            ptx::sync_aligned(kNumFP4DecodeBarrierThreads, kFP4DecodeBarrierIdx);
        }
    };
    auto wait_fp4_decode_input_ready = [&](const uint32_t& cur_stage_idx,
                                           const uint32_t& cur_phase) {
        if constexpr (kUseEarlyBDecode) {
            decode_full_barriers[cur_stage_idx]->wait(cur_phase);
        } else {
            full_barriers[cur_stage_idx]->wait(cur_phase);
        }
    };
    auto decode_fp4_b_stage = [&](const uint32_t& cur_stage_idx,
                                  const uint32_t& decode_thread_idx) {
        dequant_fp4_b_tile_to_e4m3_smem_dispatch<
            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
            kUseWideLoadDecode>(
            decode_thread_idx, kNumFP4DecodeWorkerThreads,
            smem_b_packed[cur_stage_idx], smem_b[cur_stage_idx], smem_sfb[cur_stage_idx]);
        arrive_or_sync_fp4_decode_done(cur_stage_idx);
    };

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" to SF token index is now the simple
    //       per-block linear mapping (no 4x32 transpose).
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

        // Cache finalized expert counts before the dispatch/epilogue rendezvous
        // so loader warps can leave the all-CTA count barrier and start waiting
        // on L1 arrivals while dispatch and epilogue complete their handshake.
        cache_expert_recv_counts();

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        sm90_fp8_fp4_mega_moe_fetch_cached_expert_recv_count<
            kNumExpertsPerRank, kNumExpertsPerLane>(scheduler, cached_recv_counts);

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
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
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
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        // Cleanup workspace, overlapping with combine.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

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

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);

    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
    //   Warps inside `kNumNonEpilogueThreads`: warp 0 loads A + SFA,
    //   warp 1 loads B + SFB, remaining warps are decode-assist only.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        cache_expert_recv_counts();

        sm90_fp8_fp4_mega_moe_for_each_cached_block<
            kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
            scheduler, [&](const sched::BlockPhase& block_phase,
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
            } else {
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                if constexpr (kL2ArrivalCounter) {
                    const auto ptr = reinterpret_cast<const uint32_t*>(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                    const uint32_t expected = kNumL1BlockNs * kNumEpilogueWarpgroups;
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

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    // TMA load SFA
                    if (block_phase == sched::BlockPhase::Linear1) {
                        // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                    } else {
                        // L2 SFA descriptor box is (block_mn, 1).  Default
                        // BLOCK_N=128 loads two per-64 groups; BLOCK_N=64
                        // loads four per-32 groups so each 32-column L1
                        // output block keeps its own quant scale.
                        #pragma unroll
                        for (uint32_t sf_group = 0; sf_group < kNumL2SFAPerBlockK; ++ sf_group) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + sf_group * BLOCK_M,
                                m_idx, k_block_idx * kNumL2SFAPerBlockK + sf_group, 1);
                        }
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + kNumL2SFAPerBlockK * BLOCK_M * sizeof(float));
                    }
                }
                __syncwarp();

                if constexpr (kFirstFP4DecodeAssistWarp == 0) {
                    const uint32_t decode_thread_idx =
                        (warp_idx - kNumDispatchWarps) * 32 + lane_idx;
                    wait_fp4_decode_input_ready(stage_idx, phase);
                    decode_fp4_b_stage(stage_idx, decode_thread_idx);
                }
            }
        }, cached_recv_counts);

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        cache_expert_recv_counts();

        sm90_fp8_fp4_mega_moe_for_each_cached_block<
            kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
            scheduler, [&](const sched::BlockPhase& block_phase,
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
                    // Packed FP4: each byte covers two K elements, so the
                    // logical K index advances by BLOCK_K (matching the FP8
                    // path), but the TMA box is BLOCK_K/2 bytes along the K
                    // dimension and the underlying tensor descriptor was built
                    // with `(N, K_packed = K/2)` shape on the host side. We
                    // pass `k_idx_packed = k_block_idx * (BLOCK_K / 2)` so the
                    // descriptor coordinate matches the packed K axis.
                    const uint32_t k_idx_packed = k_block_idx * (BLOCK_K / 2);

                    // TMA load packed FP4 weight tile into smem_b_packed.
                    auto b_full_barrier = kUseEarlyBDecode
                        ? decode_full_barriers[stage_idx]
                        : full_barriers[stage_idx];
                    tma::copy<BLOCK_K / 2, LOAD_BLOCK_N, kSwizzleBPackedMode, b_packed_dtype_t>(
                        tensor_map_b_ptr, b_full_barrier, smem_b_packed[stage_idx],
                        k_idx_packed, n_idx, 1);
                }
                __syncwarp();

                {
                    const bool is_l1 = block_phase == sched::BlockPhase::Linear1;
                    const uint32_t* sfb_base = is_l1 ? l1_weights_sf : l2_weights_sf;
                    const uint32_t sfb_per_expert = is_l1 ? kL1SFBPerExpert : kL2SFBPerExpert;
                    const uint32_t sfb_k_words = is_l1 ? kL1SFBKWords : kL2SFBKWords;
                    #pragma unroll
                    for (uint32_t row = lane_idx; row < LOAD_BLOCK_N; row += 32) {
                        const uint32_t n_global = n_block_idx * BLOCK_N + row;
                        smem_sfb[stage_idx][row] = __ldg(sfb_base
                            + local_expert_idx * sfb_per_expert
                            + n_global * sfb_k_words
                            + k_block_idx);
                    }
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    if constexpr (kUseEarlyBDecode) {
                        decode_full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                    } else {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                    }
                }
                __syncwarp();

                if constexpr (kFirstFP4DecodeAssistWarp <= 1) {
                    const uint32_t decode_thread_idx =
                        (1u - kFirstFP4DecodeAssistWarp) * 32 + lane_idx;
                    wait_fp4_decode_input_ready(stage_idx, phase);
                    decode_fp4_b_stage(stage_idx, decode_thread_idx);
                }
            }
        }, cached_recv_counts);

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Remaining non-epilogue warps keep the non-epilogue register allocation
        // and, when selected by kFirstFP4DecodeAssistWarp, assist FP4 decode.
        // They still participate in the warpgroup-collective
        // `setmaxnreg.dec.sync.aligned` so the math warpgroup's
        // `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        cache_expert_recv_counts();

        {
            const uint32_t non_epilogue_warp_idx = warp_idx - kNumDispatchWarps;
            if (non_epilogue_warp_idx >= kFirstFP4DecodeAssistWarp) {
                const uint32_t decode_thread_idx =
                    (non_epilogue_warp_idx - kFirstFP4DecodeAssistWarp) * 32 + lane_idx;

                sm90_fp8_fp4_mega_moe_for_each_cached_block<
                    kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
                    scheduler, [&](const sched::BlockPhase& block_phase,
                                   const uint32_t& local_expert_idx,
                                   const uint32_t& num_k_blocks,
                                   const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        wait_fp4_decode_input_ready(stage_idx, phase);
                        decode_fp4_b_stage(stage_idx, decode_thread_idx);
                    }
                }, cached_recv_counts);
            }
        }

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

        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N =
            kSplitNCombinesL1Store ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;
        constexpr uint32_t WG_SMEM_CD_L2_STRIDE_N = BLOCK_N;

        // WG_BLOCK_M / WG_BLOCK_N are now defined at the outer template scope
        // (see the GEMM data-types block). They are aware of split-N mode, so
        // we must NOT redefine them locally as `BLOCK_M / kNumEpilogueWarpgroups`
        // -- in split-N that would be 32 instead of 64.
        DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");

        // Decompose `epilogue_wg_idx` into (m,n) coordinates over the
        // (kWarpgroupSplitM, kWarpgroupSplitN) grid:
        //   - split-M path: kWarpgroupSplitN == 1, n_idx == 0
        //   - split-N path: kWarpgroupSplitM == 1, m_idx == 0
        // Both factors collapse cleanly so the same expressions cover both.
        const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
        const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
        const uint32_t wg_m_offset       = epilogue_wg_m_idx * WG_BLOCK_M;
        const uint32_t wg_n_offset       = epilogue_wg_n_idx * WG_BLOCK_N;
        const uint32_t wg_l1_out_n_offset = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
        const uint32_t smem_a_wg_offset   = wg_m_offset * BLOCK_K;
        // smem_b in FP4 SS path is the *decoded* E4M3 tile and stays full
        // LOAD_BLOCK_N rows because FP4 decode is shared across WGs; split-N
        // only shifts the WGMMA-B descriptor base by `wg_n_offset * BLOCK_K`
        // bytes, picking up the WG's own column slice.
        const uint32_t smem_b_wg_offset   = wg_n_offset * BLOCK_K;
        // When two split-N WGs share one SF block (32 output cols/WG), they stage
        // into one joint L1 tile so WG0 can issue a combined TMA store. Otherwise
        // each WG owns a compact contiguous staging tile, matching the TMA box.
        const uint32_t smem_cd_l1_wg_offset =
            kSplitNCombinesL1Store ? wg_l1_out_n_offset
                                   : epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N;
        // L2 BF16 staging keeps the full BLOCK_N row stride. In split-N the
        // WG offset selects the column half while preserving the original tile
        // layout used by the scatter path.
        const uint32_t smem_cd_l2_wg_offset = wg_m_offset * BLOCK_N + wg_n_offset;

        cache_expert_recv_counts();

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        sm90_fp8_fp4_mega_moe_for_each_cached_block<
            kNumExpertsPerRank, kNumExpertsPerLane, L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
            scheduler, [&](const sched::BlockPhase& block_phase,
                           const uint32_t& local_expert_idx,
                           const uint32_t& num_k_blocks,
                           const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            constexpr bool kSSNSplitActive =
                kFP4SSNSplit and (WG_BLOCK_N == 128)
                and (kL2ActsSFGranK == 64)
                and (kNumEpilogueWarpgroups > 1);
            using SSHalfWGMMA =
                typename mma::sm90::FP8MMASelector<(WG_BLOCK_N >= 64 ? WG_BLOCK_N / 2 : WG_BLOCK_N)>::type;
            constexpr uint32_t kSSHalfAccum = SSHalfWGMMA::kNumAccum;
            constexpr uint32_t kSSAccum = kSSNSplitActive ? kSSHalfAccum : kAccumPerThread;
            float final_accum[kAccumPerThread] = {};
            {
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                if (block_phase == sched::BlockPhase::Linear1) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + wg_m_offset + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + wg_m_offset + r_1);
                } else if constexpr (kL2ActsSFGranK == 64) {
                    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + wg_m_offset + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + 0 * BLOCK_M + wg_m_offset + r_1);
                    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + wg_m_offset + r_0);
                    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + 1 * BLOCK_M + wg_m_offset + r_1);
                }

                // ----- FP4-to-E4M3 dequant of the packed weight tile -----
                // The packed FP4 tile in `smem_b_packed[stage_idx]` is decoded
                // into the E4M3 tile in `smem_b[stage_idx]` with the per-32
                // UE8M0 SFB baked in via the constant FP4-to-E4M3 LUT.
                // After this call, `smem_b[stage]` is byte-equivalent to a
                // pre-scaled FP8 weight tile: the subsequent SS-mode WGMMA
                // accumulator already includes SFB, and only SFA needs to be
                // applied in the promote loop below.
                //
                // Non-epilogue warps assist the math warpgroup. Decode work
                // is partitioned over the assist threads plus the
                // epilogue/math threads, then all participants rendezvous
                // before WGMMA reads the decoded shared tile.
                if constexpr (kUseEarlyBDecode)
                    wait_fp4_decode_input_ready(stage_idx, phase);
                const bool math_warp_decodes =
                    epilogue_warp_idx < kNumMathWGDecodeWarps;
                if constexpr (kNumMathWGDecodeWarps > 0) {
                    if (math_warp_decodes) {
                        const uint32_t decode_thread_idx =
                            kNumFP4DecodeAssistThreads + epilogue_thread_idx;
                        dequant_fp4_b_tile_to_e4m3_smem_dispatch<
                            LOAD_BLOCK_N, BLOCK_K, kScaleBGranK, kNumSFBPerBlockK,
                            kUseWideLoadDecode>(
                            decode_thread_idx, kNumFP4DecodeWorkerThreads,
                            smem_b_packed[stage_idx], smem_b[stage_idx], smem_sfb[stage_idx]);
                    }
                }
                if constexpr (kNumMathWGDecodeWarps > 0) {
                    if (math_warp_decodes)
                        arrive_or_sync_fp4_decode_done(stage_idx);
                    if constexpr (kUseDecodeDoneMBarrier) {
                        wait_fp4_decode_done(stage_idx, phase);
                    } else {
                        if (!math_warp_decodes)
                            wait_fp4_decode_done(stage_idx, phase);
                    }
                } else {
                    wait_fp4_decode_done(stage_idx, phase);
                }

                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (kSwapABL1Active) {
                        // L1 swapAB: WGMMA-M is the 64-row weight slice owned by
                        // this split-N WG; WGMMA-N is the valid token count,
                        // bucketed to Hopper's 8-column granularity.
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
                                    smem_b[stage_idx] + smem_b_wg_offset + k * SwapWGMMA::K, 1);
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
                                if constexpr (kSwapABFastAmaxActive) {
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                                        final_accum[i * 4 + 0] += scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                                        final_accum[i * 4 + 1] += scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * swap_accum[i * 4 + 3];
                                    }
                                } else {
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                                        final_accum[i * 4 + 0] += scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[i * 4 + 2] += scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                                        final_accum[i * 4 + 1] += scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[i * 4 + 3] += scale_1 * swap_accum[i * 4 + 3];
                                    }
                                }
                            }

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (kSwapABFlashN24Dispatch) {
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else if (n_swap <= 24) {
                                run_swap_ab_l1.template operator()<24>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l1.template operator()<32>();
                            } else {
                                run_swap_ab_l1.template operator()<64>();
                            }
                        } else {
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l1.template operator()<32>();
                            } else {
                                run_swap_ab_l1.template operator()<64>();
                            }
                        }
                    } else {
                    // Single per-128 K-block WGMMA group
                    if constexpr (kSSNSplitActive) {
                        float accum[kSSAccum];
                        #pragma unroll
                        for (uint32_t nh = 0; nh < 2; ++ nh) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSSHalfAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / SSHalfWGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + smem_a_wg_offset + k * SSHalfWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + smem_b_wg_offset
                                        + nh * (WG_BLOCK_N / 2) * BLOCK_K
                                        + k * SSHalfWGMMA::K, 1);
                                SSHalfWGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSSHalfAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kSSHalfAccum / 4; ++ i) {
                                const uint32_t f = nh * kSSHalfAccum + i * 4;
                                final_accum[f+0] += scale_a_0_lo * accum[i*4+0];
                                final_accum[f+1] += scale_a_0_lo * accum[i*4+1];
                                final_accum[f+2] += scale_a_1_lo * accum[i*4+2];
                                final_accum[f+3] += scale_a_1_lo * accum[i*4+3];
                            }
                        }
                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();
                    } else {
                        float accum[kAccumPerThread];
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

                        // L1: SFB is already baked into the decoded E4M3 tile,
                        // so only SFA remains.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                        }
                    }
                    }
                } else {
                    if constexpr (kSwapABL2Active) {
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
                                    if constexpr (kSwapABFastAmaxActive) {
                                        if (token_0 < valid_m) {
                                            const float scale_0 = ptx::ld_shared(
                                                smem_sfa[stage_idx] + sf_group * BLOCK_M + token_0);
                                            final_accum[i * 4 + 0] += scale_0 * swap_accum[i * 4 + 0];
                                            final_accum[i * 4 + 2] += scale_0 * swap_accum[i * 4 + 2];
                                        }
                                        if (token_1 < valid_m) {
                                            const float scale_1 = ptx::ld_shared(
                                                smem_sfa[stage_idx] + sf_group * BLOCK_M + token_1);
                                            final_accum[i * 4 + 1] += scale_1 * swap_accum[i * 4 + 1];
                                            final_accum[i * 4 + 3] += scale_1 * swap_accum[i * 4 + 3];
                                        }
                                    } else {
                                        if (token_0 < valid_m) {
                                            const float scale_0 = ptx::ld_shared(
                                                smem_sfa[stage_idx] + sf_group * BLOCK_M + token_0);
                                            final_accum[i * 4 + 0] += scale_0 * swap_accum[i * 4 + 0];
                                            final_accum[i * 4 + 2] += scale_0 * swap_accum[i * 4 + 2];
                                        }
                                        if (token_1 < valid_m) {
                                            const float scale_1 = ptx::ld_shared(
                                                smem_sfa[stage_idx] + sf_group * BLOCK_M + token_1);
                                            final_accum[i * 4 + 1] += scale_1 * swap_accum[i * 4 + 1];
                                            final_accum[i * 4 + 3] += scale_1 * swap_accum[i * 4 + 3];
                                        }
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
                                    smem_b[stage_idx] + smem_b_wg_offset + k * SwapWGMMA::K, 1);
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
                                    smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
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

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (kSwapABFlashN24Dispatch) {
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else if (n_swap <= 24) {
                                run_swap_ab_l2.template operator()<24>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l2.template operator()<32>();
                            } else {
                                run_swap_ab_l2.template operator()<64>();
                            }
                        } else {
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l2.template operator()<32>();
                            } else {
                                run_swap_ab_l2.template operator()<64>();
                            }
                        }
                    } else if constexpr (kL2ActsSFGranK == 32) {
                        // L2 BLOCK_N=64: L1 produced 32-column FP8 chunks with
                        // independent SF, so promote each WGMMA::K=32 slice with
                        // its own activation scale.
                        float accum[kAccumPerThread];
                        #pragma unroll
                        for (uint32_t sf_group = 0; sf_group < kNumL2SFAPerBlockK; ++ sf_group) {
                            const float scale_a_0 = ptx::ld_shared(
                                smem_sfa[stage_idx] + sf_group * BLOCK_M + wg_m_offset + r_0);
                            const float scale_a_1 = ptx::ld_shared(
                                smem_sfa[stage_idx] + sf_group * BLOCK_M + wg_m_offset + r_1);
                            const uint32_t k_off = sf_group * WGMMA::K;
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + smem_b_wg_offset + k_off, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, false);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0 * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0 * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1 * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1 * accum[i*4+3];
                            }
                        }

                        if (lane_idx == 0)
                            empty_barriers[stage_idx]->arrive();
                    } else {
                        if constexpr (kSSNSplitActive) {
                            // L2 per-64 SFA with split-N WGMMA: each N half owns a
                            // 32-float accumulator, then promotes into its slice of
                            // final_accum before the next half reuses the accumulator.
                            float accum[kSSAccum];
                            #pragma unroll
                            for (uint32_t nh = 0; nh < 2; ++ nh) {
                                const uint32_t n_off = nh * (WG_BLOCK_N / 2) * BLOCK_K;
                                const uint32_t fbase = nh * kSSHalfAccum;

                                // First K half: K=0..63, SFA = scale_a_*_lo
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SSHalfWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + smem_a_wg_offset + k * SSHalfWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset + n_off + k * SSHalfWGMMA::K, 1);
                                    SSHalfWGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();

                                // L2 first half: SFB baked into decoded E4M3 tile.
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSHalfAccum / 4; ++ i) {
                                    final_accum[fbase+i*4+0] += scale_a_0_lo * accum[i*4+0];
                                    final_accum[fbase+i*4+1] += scale_a_0_lo * accum[i*4+1];
                                    final_accum[fbase+i*4+2] += scale_a_1_lo * accum[i*4+2];
                                    final_accum[fbase+i*4+3] += scale_a_1_lo * accum[i*4+3];
                                }

                                // Second K half: K=64..127, SFA = scale_a_*_hi
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < (BLOCK_K / 2) / SSHalfWGMMA::K; ++ k) {
                                    const uint32_t k_off = (BLOCK_K / 2) + k * SSHalfWGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + smem_a_wg_offset + k_off, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + smem_b_wg_offset + n_off + k_off, 1);
                                    SSHalfWGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();

                                // L2 second half: SFB baked into decoded E4M3 tile.
                                #pragma unroll
                                for (uint32_t i = 0; i < kSSHalfAccum / 4; ++ i) {
                                    final_accum[fbase+i*4+0] += scale_a_0_hi * accum[i*4+0];
                                    final_accum[fbase+i*4+1] += scale_a_0_hi * accum[i*4+1];
                                    final_accum[fbase+i*4+2] += scale_a_1_hi * accum[i*4+2];
                                    final_accum[fbase+i*4+3] += scale_a_1_hi * accum[i*4+3];
                                }
                            }

                            if (lane_idx == 0)
                                empty_barriers[stage_idx]->arrive();
                        } else {
                            float accum[kAccumPerThread];
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

                            // L2 first half: SFB baked into decoded E4M3 tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
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

                            // L2 second half: SFB baked into decoded E4M3 tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_a_0_hi * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_hi * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_hi * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_hi * accum[i*4+3];
                            }
                        }
                    }
                }
            }
            }

            // Skip epilogue when block is past valid M (still must release via empty).
            // In split-N mode, `wg_m_offset` is 0 for all WGs (they share the same M
            // rows), so this skip is effectively per-block, not per-WG.
            if (wg_m_offset >= valid_m) {
                // Trigger any combine/sync logic minimally
                if (block_phase == sched::BlockPhase::Linear1)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                else
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            const uint32_t row_offset_r0 = wg_m_offset + r_0;
            const uint32_t row_offset_r1 = wg_m_offset + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;

            if (block_phase == sched::BlockPhase::Linear1) {
                if constexpr (kSwapABL1Active) {
                    // Each split-N WG writes its disjoint 32-column range into
                    // a shared token-major tile. The default path stages FP32,
                    // then scans complete token rows for amax before quantizing.
                    // The fast-amax path keeps each thread's small token slice in
                    // registers, publishes per-token partial amax, and writes FP8
                    // directly after the cross-warp reduction.
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
                    float swap_v0[kSwapABTokenChunks] = {};
                    float swap_v1[kSwapABTokenChunks] = {};
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        const bool valid_token_0 = token_0 < valid_m;
                        const bool valid_token_1 = token_1 < valid_m;

                        float v0 = 0.0f;
                        if (valid_token_0) {
                            float g0 = final_accum[i * 4 + 0];
                            float u0 = final_accum[i * 4 + 2];
                            clamp_gate(g0);
                            clamp_up(u0);
                            const float weight_0 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_0)
                                .get_base_ptr<float>();
                            v0 = silu(g0) * u0 * weight_0;
                            if constexpr (kSwapABFastAmaxActive) {
                                swap_v0[i] = v0;
                            } else {
                                smem_cd_swap_l1_fp32[token_0 * L1_OUT_BLOCK_N + out_col_base] = v0;
                            }
                        }

                        float v1 = 0.0f;
                        if (valid_token_1) {
                            float g1 = final_accum[i * 4 + 1];
                            float u1 = final_accum[i * 4 + 3];
                            clamp_gate(g1);
                            clamp_up(u1);
                            const float weight_1 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_1)
                                .get_base_ptr<float>();
                            v1 = silu(g1) * u1 * weight_1;
                            if constexpr (kSwapABFastAmaxActive) {
                                swap_v1[i] = v1;
                            } else {
                                smem_cd_swap_l1_fp32[token_1 * L1_OUT_BLOCK_N + out_col_base] = v1;
                            }
                        }

                        if constexpr (kSwapABFastAmaxActive) {
                            const float amax0 = math::warp_reduce<4, true>(
                                cute::abs(v0), math::ReduceMax<float>());
                            const float amax1 = math::warp_reduce<4, true>(
                                cute::abs(v1), math::ReduceMax<float>());
                            if (row_idx == 0) {
                                if (valid_token_0)
                                    smem_cd_swap_l1_amax[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax0;
                                if (valid_token_1)
                                    smem_cd_swap_l1_amax[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax1;
                            }
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

                    if constexpr (kSwapABFastAmaxActive) {
                        for (uint32_t token = epilogue_thread_idx;
                             token < valid_m;
                             token += kNumEpilogueThreads) {
                            float amax = 0.0f;
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w)
                                amax = cute::max(
                                    amax, smem_cd_swap_l1_amax[token * kNumEpilogueWarps + w]);
                            float2 amax_pair = {amax, amax};
                            float2 sf_pair, sf_inv_pair;
                            sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(
                                amax_pair, sf_pair, sf_inv_pair);

                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_idx = pool_block_idx * BLOCK_M + token;
                            sf_base_ptr[n_block_idx * kNumPaddedSFPoolTokens + token_idx] = sf_pair.x;
                            ptx::st_shared(smem_amax_scratch + token, __float_as_uint(sf_inv_pair.x));
                        }

                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                        #pragma unroll
                        for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                            const uint32_t token_0 = i * 8 + col_idx * 2;
                            const uint32_t token_1 = token_0 + 1;
                            if (token_0 < valid_m) {
                                const float sf_inv = __uint_as_float(
                                    ptx::ld_shared(smem_amax_scratch + token_0));
                                const __nv_fp8_e4m3 q(swap_v0[i] * sf_inv);
                                reinterpret_cast<uint8_t*>(
                                    smem_cd_swap_l1_fp8)[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                        *reinterpret_cast<const uint8_t*>(&q);
                            }
                            if (token_1 < valid_m) {
                                const float sf_inv = __uint_as_float(
                                    ptx::ld_shared(smem_amax_scratch + token_1));
                                const __nv_fp8_e4m3 q(swap_v1[i] * sf_inv);
                                reinterpret_cast<uint8_t*>(
                                    smem_cd_swap_l1_fp8)[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                        *reinterpret_cast<const uint8_t*>(&q);
                            }
                        }
                    } else {
                        for (uint32_t token = epilogue_thread_idx; token < valid_m; token += kNumEpilogueThreads) {
                            float amax = 0.0f;
                            #pragma unroll
                            for (uint32_t col = 0; col < L1_OUT_BLOCK_N; ++ col) {
                                const float v = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col];
                                amax = cute::max(amax, cute::abs(v));
                            }
                            float2 amax_pair = {amax, amax};
                            float2 sf_pair, sf_inv_pair;
                            sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                            const float sf = sf_pair.x;
                            const float sf_inv = sf_inv_pair.x;

                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_idx = pool_block_idx * BLOCK_M + token;
                            sf_base_ptr[n_block_idx * kNumPaddedSFPoolTokens + token_idx] = sf;

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
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8,
                            n_block_idx * L1_OUT_BLOCK_N,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                    if constexpr (kL2ArrivalCounter) {
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
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
                    if constexpr (kL2ArrivalCounter)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` in [0, 8): gate chunk = 2p, up chunk = 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;  // 8 for BLOCK_N=128
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
                    float g_r0_c0 = final_accum[gate*4 + 0]; clamp_gate(g_r0_c0);
                    float g_r0_c1 = final_accum[gate*4 + 1]; clamp_gate(g_r0_c1);
                    float g_r1_c0 = final_accum[gate*4 + 2]; clamp_gate(g_r1_c0);
                    float g_r1_c1 = final_accum[gate*4 + 3]; clamp_gate(g_r1_c1);
                    float u_r0_c0 = final_accum[up*4   + 0]; clamp_up(u_r0_c0);
                    float u_r0_c1 = final_accum[up*4   + 1]; clamp_up(u_r0_c1);
                    float u_r1_c0 = final_accum[up*4   + 2]; clamp_up(u_r1_c0);
                    float u_r1_c1 = final_accum[up*4   + 3]; clamp_up(u_r1_c1);

                    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };

                    if (valid_r0) {
                        swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                        swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                        amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                    } else {
                        swiglu_r0[p][0] = 0.0f;
                        swiglu_r0[p][1] = 0.0f;
                    }
                    if (valid_r1) {
                        swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                        swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                        amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                    } else {
                        swiglu_r1[p][0] = 0.0f;
                        swiglu_r1[p][1] = 0.0f;
                    }
                }

                // Fold topk weight into the row amax and final quantization scale.
                float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                    .get_data_buffer(m_idx + row_offset_r0)
                    .get_base_ptr<float>() : 0.0f;
                float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                    .get_data_buffer(m_idx + row_offset_r1)
                    .get_base_ptr<float>() : 0.0f;
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

                // Shared-SF split-N reduction: both N-half WGs publish their
                // per-row amax, synchronize once, then read both values and
                // take the max in registers.
                if constexpr (kSplitNSharesSF) {
                    if (col_idx == 0) {
                        const uint32_t row_slot = warp_idx_in_wg * 8 + row_idx;
                        const uint32_t slot = row_slot * 4 + epilogue_wg_n_idx * 2;
                        ptx::st_shared(smem_amax_scratch + slot + 0,
                                       __float_as_uint(amax_r0));
                        ptx::st_shared(smem_amax_scratch + slot + 1,
                                       __float_as_uint(amax_r1));
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    // Both WGs read the merged amax.
                    if (col_idx == 0) {
                        const uint32_t row_slot = warp_idx_in_wg * 8 + row_idx;
                        const uint32_t slot = row_slot * 4;
                        const float wg0_r0 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 0));
                        const float wg0_r1 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 1));
                        const float wg1_r0 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 2));
                        const float wg1_r1 = __uint_as_float(
                            ptx::ld_shared(smem_amax_scratch + slot + 3));
                        amax_r0 = cute::max(wg0_r0, wg1_r0);
                        amax_r1 = cute::max(wg0_r1, wg1_r1);
                    }
                    // Broadcast the merged col_idx==0 amax across the row's
                    // four column lanes; this is no longer a reduction after
                    // both split-N WGs have published their row max.
                    const int row_group_leader = static_cast<int>(lane_idx - col_idx);
                    amax_r0 = __shfl_sync(0xffffffff, amax_r0, row_group_leader);
                    amax_r1 = __shfl_sync(0xffffffff, amax_r1, row_group_leader);
                }

                // Compute SF and inverse SF for each row
                float sf_r0, sf_inv_r0;
                float sf_r1, sf_inv_r1;
                {
                    float2 amax_pair = {amax_r0, amax_r1};
                    float2 sf_pair, sf_inv_pair;
                    sm90_fp8_fp4_mega_moe_get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                    sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                    sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                }
                const float weighted_sf_inv_r0 = weight_r0 * sf_inv_r0;
                const float weighted_sf_inv_r1 = weight_r1 * sf_inv_r1;

                // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                // The L1-output TMA store descriptor is built with swizzle_mode = 0
                // to match this plain row-major SMEM staging tile.
                //
                // Per pair `p`, each thread holds 4 FP8 values to write at:
                //   (row r_0, cols p*8 + col_idx*2 + {0,1})  -> packed as fp8x2 (2 bytes)
                //   (row r_1, cols p*8 + col_idx*2 + {0,1})  -> packed as fp8x2 (2 bytes)
                // The shared tile is either the joint SF-sharing tile or a compact
                // per-WG tile; WG_SMEM_CD_L1_STRIDE_N selects the matching row stride.
                auto* smem_cd_l1_wg = smem_cd_l1 + smem_cd_l1_wg_offset;
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const float v00 = swiglu_r0[p][0] * weighted_sf_inv_r0;
                    const float v01 = swiglu_r0[p][1] * weighted_sf_inv_r0;
                    const float v10 = swiglu_r1[p][0] * weighted_sf_inv_r1;
                    const float v11 = swiglu_r1[p][1] * weighted_sf_inv_r1;

                    const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                    const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                    const uint32_t col = p * 8 + col_idx * 2;
                    auto* p0 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_0 * WG_SMEM_CD_L1_STRIDE_N + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * WG_SMEM_CD_L1_STRIDE_N + col);
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
                }

                // Write SF as float at `[token, k_sf_idx]` in the L2 acts SF buffer.
                // Each row is contributed by lanes col_idx in [0, 3]; only col_idx == 0 writes.
                // Shared-SF writes once from the first N-half WG.
                const bool sf_writer = (col_idx == 0) and
                    (not kSplitNSharesSF or epilogue_wg_n_idx == 0);
                if (sf_writer) {
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    // SF buffer is (kNumPaddedSFPoolTokens x kIntermediateHidden/64), MN-major:
                    //   addr[k_idx * num_padded_sf_pool_tokens + token_idx]
                    const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                    const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                    // Shared-SF spans both split-N WGs; otherwise each WG owns
                    // a distinct local SF N block.
                    const uint32_t sf_n_block_idx_local = n_block_idx * kWarpgroupSplitN + epilogue_wg_n_idx;
                    const uint32_t k_sf_idx = kSplitNSharesSF ? n_block_idx : sf_n_block_idx_local;
                    if (valid_r0)
                        sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
                    if (valid_r1)
                        sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
                }

                // Combined split-N store needs an epilogue-wide sync so WG0 can
                // store the full L1 staging tile after WG1 writes its slice.
                if constexpr (kSplitNCombinesL1Store) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
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
                if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                    // In the 32-col/WG split-N path, WG0 stores the combined
                    // BLOCK_M x L1_OUT_BLOCK_N tile. Other paths store per-WG
                    // slices independently.
                    if constexpr (kSplitNCombinesL1Store) {
                        if (epilogue_wg_n_idx == 0) {
                            const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                            cute::tma_store_fence();
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_l1_output,
                                smem_cd_l1,
                                out_n_idx,
                                m_idx);
                            cute::tma_store_arrive();
                        }
                        // WG1 is covered by WG0's combined store.
                    } else {
                        // The TMA descriptor was built with box
                        // (WG_L1_OUT_BLOCK_N, l1_output_box_m=WG_BLOCK_M), so each
                        // WG advances column by `wg_l1_out_n_offset` (split-N) and
                        // row by `wg_m_offset` (split-M). In default single-WG
                        // mode both offsets are zero and this reduces to the
                        // historical `(n_block_idx * L1_OUT_BLOCK_N, m_idx)`.
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_offset;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1 + smem_cd_l1_wg_offset,
                            out_n_idx,
                            m_idx + wg_m_offset);
                        cute::tma_store_arrive();
                    }
                }
                __syncwarp();
                ptx::tma_store_wait<0>();

                // Notify L2 that this N block's L1 output (and SF) is ready
                if constexpr (kL2ArrivalCounter) {
                    if constexpr (kSplitNCombinesL1Store) {
                        // Only WG0 issues the combined TMA store. Once it
                        // drains, both N slices are visible, so a single
                        // +kWarpgroupSplitN keeps the counter expectation valid.
                        if (epilogue_wg_n_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                            ptx::red_add_rel(
                                reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(pool_block_idx)),
                                kWarpgroupSplitN);
                        }
                    } else if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
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
                // Counter mode skips the bitmask path's epilogue-wide sync.
                // Add a tail sync so WG1 cannot overwrite the shared L1 staging
                // tile for the next N block while WG0's combined store drains.
                if constexpr (kSplitNCombinesL1Store and kL2ArrivalCounter)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kSwapABL2Active) {
                    auto store_bf16 = [&](const uint32_t& token, const uint32_t& col, float value) {
                        smem_cd_l2[smem_cd_l2_wg_offset + token * BLOCK_N + col] =
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
                } else if constexpr (not kSwapABL2Active) {
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
                                + row * WG_SMEM_CD_L2_STRIDE_N
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
                }

                {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows x 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp maps to one row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    // In split-N each WG owns WG_BLOCK_N (= BLOCK_N/num_wg)
                    // columns of every row; in split-M each WG owns the full
                    // BLOCK_N columns of its WG_BLOCK_M rows. Either way the
                    // per-WG column footprint is WG_BLOCK_N.
                    constexpr uint32_t kColsPerScatterLane = WG_BLOCK_N / 16;
                    static_assert(WG_BLOCK_N % 16 == 0, "Scatter layout expects an even lane partition");
                    static_assert(kColsPerScatterLane == 4 or kColsPerScatterLane == 8,
                                  "L2 scatter currently supports WG_BLOCK_N=64 or 128");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = wg_m_offset + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        // WG_BLOCK_N=128 scatters 8 BF16s/lane (=16B, uint4).
                        // For WG_BLOCK_N=64 each lane owns 4 BF16s (=8B), so
                        // use uint2; a uint4 load would be misaligned for
                        // odd lanes.
                        auto smem_ptr = smem_cd_l2
                            + smem_cd_l2_wg_offset
                            + row_in_wg * WG_SMEM_CD_L2_STRIDE_N
                            + lane_in_row * kColsPerScatterLane;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        if constexpr (kColsPerScatterLane == 8) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                (n_idx + wg_n_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                }

                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        }, cached_recv_counts);

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr bool kOneCombineChunkFits =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE;
        constexpr bool kTwoCombineChunksFit =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / 2 <= SMEM_BEFORE_BARRIER_SIZE;
        constexpr uint32_t kNumChunks =
            (kOneCombineChunkFits and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 :
            (kTwoCombineChunksFit ? 2 : 4);
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_TRAP_ONLY_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

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
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
