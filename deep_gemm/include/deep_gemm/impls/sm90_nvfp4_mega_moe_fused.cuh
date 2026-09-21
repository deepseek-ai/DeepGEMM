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
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace deep_gemm {
namespace nvfp4 {

__device__ __forceinline__ uint2 dequant_mode2_nibble_word(
        const uint32_t packed, const uint2& lut) {
    const uint32_t magnitude_selectors = packed & 0x77777777u;
    uint32_t out_hi =
        byte_perm_unchecked(lut.x, lut.y, magnitude_selectors);
    uint32_t out_lo =
        byte_perm_unchecked(lut.x, lut.y, magnitude_selectors >> 16);
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;"
        : "+r"(out_hi) : "r"(packed));
    const uint32_t shifted = packed << 4;
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;"
        : "+r"(out_lo) : "r"(shifted));
    return make_uint2(out_hi, out_lo);
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_mode2_nibble_row_regs(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint2& scale_words,
        const uint32_t row_swizzle,
        const uint2* __restrict__ lut_smem) {
#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint32_t scale_word =
            quad_i < 2 ? scale_words.x : scale_words.y;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 =
            (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;
        const uint32_t scale1 =
            (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
        const uint2 lut0 = lut_smem[scale0];
        const uint2 lut1 = lut_smem[scale1];

        const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
        const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
        if constexpr (!kQuadILP) {
            *reinterpret_cast<uint4*>(
                fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
        }

        const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
        const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
        if constexpr (kQuadILP) {
            *reinterpret_cast<uint4*>(
                fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
        }
        *reinterpret_cast<uint4*>(
            fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src =
        reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    const uint2 scale_words =
        *reinterpret_cast<const uint2*>(row_ptr + 64);
    dequant_mode2_nibble_row_regs<kQuadILP>(
        smem_b + row * 128, fp4_quads, scale_words,
        (row & 7u) << 4, lut_smem);
}

// Threads 0-127 and 128-255 each decode one K64 half of the same N128 tile,
// allowing the two M64 warpgroups to reuse the decoded weights.
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble_split_m(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t thread_idx,
        const uint2* __restrict__ lut_smem) {
    const uint32_t row = thread_idx & 127u;
    const uint32_t k_half_idx = thread_idx >> 7;
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80u;
    const uint4* __restrict__ fp4_src =
        reinterpret_cast<const uint4*>(row_ptr + k_half_idx * 32u);
    const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(
        row_ptr + 64u + k_half_idx * sizeof(uint32_t));
    uint8_t* __restrict__ fp8_dst = smem_b + row * 128u;
    const uint32_t row_swizzle = (row & 7u) << 4;

    #pragma unroll
    for (uint32_t quad_i = 0; quad_i < 2; ++ quad_i) {
        const uint4 q = fp4_src[quad_i];
        const uint32_t scale_i0 = quad_i * 2u;
        const uint32_t scale_i1 = scale_i0 + 1u;
        const uint32_t scale0 = (scale_word >> (scale_i0 * 8u)) & 0x7fu;
        const uint32_t scale1 = (scale_word >> (scale_i1 * 8u)) & 0x7fu;
        const uint2 lut0 = lut_smem[scale0];
        const uint2 lut1 = lut_smem[scale1];
        const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
        const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
        const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
        const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
        const uint32_t k_offset0 =
            k_half_idx * 64u + scale_i0 * 16u;
        const uint32_t k_offset1 =
            k_half_idx * 64u + scale_i1 * 16u;
        *reinterpret_cast<uint4*>(fp8_dst + (k_offset0 ^ row_swizzle)) =
            make_uint4(q0.x, q0.y, q1.x, q1.y);
        *reinterpret_cast<uint4*>(fp8_dst + (k_offset1 ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
}

__device__ __forceinline__ uint2 dequant_braided_selector_word(
        const uint32_t braided, const uint2& lut) {
    const uint32_t sel0 = braided & 0x00007777u;
    const uint32_t sel1 = (braided >> 16) & 0x00007777u;
    uint32_t out0 = byte_perm_unchecked(lut.x, lut.y, sel0);
    uint32_t out1 = byte_perm_unchecked(lut.x, lut.y, sel1);
    out0 |= braided & 0x80808080u;
    out1 |= (braided << 4) & 0x80808080u;
    return make_uint2(out0, out1);
}

__device__ __forceinline__ void dequant_braided_quad(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const uint2& lut0,
        const uint2& lut1,
        const int scale_i0,
        const uint32_t row_swizzle) {
    const uint2 q0 = dequant_braided_selector_word(q.x, lut0);
    const uint2 q1 = dequant_braided_selector_word(q.y, lut0);
    *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
        make_uint4(q0.x, q0.y, q1.x, q1.y);

    const uint2 q2 = dequant_braided_selector_word(q.z, lut1);
    const uint2 q3 = dequant_braided_selector_word(q.w, lut1);
    *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
        make_uint4(q2.x, q2.y, q3.x, q3.y);
}

__device__ __forceinline__ void dequant_braided_quad_ilp(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const uint2& lut0,
        const uint2& lut1,
        const int scale_i0,
        const uint32_t row_swizzle) {
    // Expose all four independent PRMT chains together so ptxas can overlap
    // their selector/sign work before either 128-bit shared-memory store.
    const uint32_t q0_sel0 = q.x & 0x00007777u;
    const uint32_t q0_sel1 = (q.x >> 16) & 0x00007777u;
    const uint32_t q1_sel0 = q.y & 0x00007777u;
    const uint32_t q1_sel1 = (q.y >> 16) & 0x00007777u;
    const uint32_t q2_sel0 = q.z & 0x00007777u;
    const uint32_t q2_sel1 = (q.z >> 16) & 0x00007777u;
    const uint32_t q3_sel0 = q.w & 0x00007777u;
    const uint32_t q3_sel1 = (q.w >> 16) & 0x00007777u;

    uint32_t q0_out0 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel0);
    uint32_t q0_out1 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel1);
    uint32_t q1_out0 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel0);
    uint32_t q1_out1 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel1);
    uint32_t q2_out0 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel0);
    uint32_t q2_out1 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel1);
    uint32_t q3_out0 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel0);
    uint32_t q3_out1 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel1);

    q0_out0 |= q.x & 0x80808080u;
    q0_out1 |= (q.x << 4) & 0x80808080u;
    q1_out0 |= q.y & 0x80808080u;
    q1_out1 |= (q.y << 4) & 0x80808080u;
    q2_out0 |= q.z & 0x80808080u;
    q2_out1 |= (q.z << 4) & 0x80808080u;
    q3_out0 |= q.w & 0x80808080u;
    q3_out1 |= (q.w << 4) & 0x80808080u;

    *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
        make_uint4(q0_out0, q0_out1, q1_out0, q1_out1);
    *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
        make_uint4(q2_out0, q2_out1, q3_out0, q3_out1);
}

template <int kQuad, bool kQuadIlp>
__device__ __forceinline__ void dequant_braided_quad_lut_window(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint32_t scale_word_lo,
        const uint32_t scale_word_hi,
        const uint2* __restrict__ lut_smem,
        const uint2 lut0,
        const uint2 lut1,
        const uint32_t row_swizzle) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t next_scale_word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
        const uint32_t next_scale0 =
            (next_scale_word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t next_scale1 =
            (next_scale_word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        next_lut0 = lut_smem[next_scale0];
        next_lut1 = lut_smem[next_scale1];
    }

    if constexpr (kQuadIlp) {
        dequant_braided_quad_ilp(
            fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
    } else {
        dequant_braided_quad(
            fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
    }

    if constexpr (kQuad + 1 < 4) {
        dequant_braided_quad_lut_window<kQuad + 1, kQuadIlp>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut_smem,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <bool kQuadIlp = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_braided_lut_window(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint2 lut0 = lut_smem[scale_words.x & 0x7fu];
    const uint2 lut1 = lut_smem[(scale_words.x >> 8) & 0x7fu];
    dequant_braided_quad_lut_window<0, kQuadIlp>(
        smem_b + row * 128, fp4_quads, scale_words.x, scale_words.y,
        lut_smem, lut0, lut1, (row & 7u) << 4);
}

}  // namespace nvfp4

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t kNumRingTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    float kActivationClamp,
    bool kFastMath,
    bool kSwapABRequested,
    bool kSingleActiveDispatchWarp,
    bool kUseMode2RowDecoder,
    bool kUseInterleavedScheduler,
    uint32_t kNumSMs,
    uint32_t kNumRanks,
    uint32_t kNumExperts,
    uint32_t kHidden,
    uint32_t kIntermediateHidden,
    uint32_t kNumTopk
>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
sm90_nvfp4_mega_moe_fused_impl(
        void* y,
        int* cumulative_local_expert_recv_stats,
        const uint32_t num_tokens,
        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
        const float* __restrict__ l1_global_scales,
        const float* __restrict__ l2_global_scales) {
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumDispatchThreads = 64;
    constexpr uint32_t kNumNonEpilogueThreads = 64;
    constexpr uint32_t kNumEpilogueThreads = 256;
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

#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumSMs > 1, "SM90 MegaMoE requires multiple SMs");
    DG_STATIC_ASSERT(kNumRanks > 0 && kNumRanks <= 72,
                     "Invalid number of EP ranks");
    DG_STATIC_ASSERT(kNumExperts >= kNumTopk && kNumExperts <= 512,
                     "Invalid number of routed experts");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0,
                     "Routed experts must be divisible by EP ranks");
    DG_STATIC_ASSERT(BLOCK_M == 8 || BLOCK_M == 16 ||
                     BLOCK_M == 24 || BLOCK_M == 64 || BLOCK_M == 128,
                     "SM90 NVFP4 fused kernel requires BM8/BM16/BM24/BM64/BM128");
    DG_STATIC_ASSERT((BLOCK_M == 8 && kNumStages == 4) ||
                     ((BLOCK_M == 16 || BLOCK_M == 24) && kNumStages == 3) ||
                     (BLOCK_M == 64 && kNumStages == 3) ||
                     (BLOCK_M == 128 && kNumStages == 6),
                     "Unexpected SM90 NVFP4 pipeline depth");
    DG_STATIC_ASSERT((BLOCK_M == 128) == (BLOCK_N == 128),
                     "BM128 is paired with the BN128 split-M topology");
    DG_STATIC_ASSERT(!kSwapABRequested || BLOCK_M <= 24,
                     "swap-AB is only selected through the M64 bucket");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

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
    // Workspaces and symmetric buffer slicing. The framework reserves
    // per-64 SF capacity; this per-128 path uses its first half
    // as a dense per-128 layout so no framework allocation change is needed.
    // =====================================================================
    // NOTES: this kernel reuses the same live ring buffer other MMA kinds
    // use for L1/L2 activations -- `kNumRingTokens` must match what
    // `get_symm_buffer_size_for_mega_moe` (host side) used for the same
    // `mma_type="fp4xfp4"` SM90 buffer (see
    // `SM90NVFP4FusedShape::get_num_ring_tokens`) so that every
    // downstream offset (ring slot counters, dispatch pulling) lines up
    // with what was actually allocated. Combine metadata still spans the
    // full (non-ring) pool -- see `workspace.get_token_src_metadata_ptr`.
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32, false);
    // Physical per-64 capacity: logical per-128 scales occupy the first half.
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
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumRingTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumRingTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumRingTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    constexpr float kNvfp4ToFp8ScaleCompensation = 8.0f;
    using task_info_t = sched::SM90TaskInfo;
    using interleaved_scheduler_t = sched::InterleavedMegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumSMs, kNumRanks>;
    constexpr uint32_t kNumRoutedL1BlockNs = L1_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumRoutedL2BlockNs = L2_SHAPE_N / BLOCK_N;
    constexpr bool kSplitMDecodedWeightReuse =
        BLOCK_M == 128 && BLOCK_N == 128 && kNumEpilogueWarpgroups == 2;
    constexpr uint32_t WG_BLOCK_M =
        kSplitMDecodedWeightReuse ? BLOCK_M / 2 : BLOCK_M;
    constexpr uint32_t WG_BLOCK_N =
        kSplitMDecodedWeightReuse ? BLOCK_N : BLOCK_N / 2;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    constexpr uint32_t kSwapABWeightHalves = WG_BLOCK_N / 64;
    constexpr uint32_t kSwapABHalfAccumPerThread = 64 * 64 / 128;
    DG_STATIC_ASSERT(!kSwapABRequested || WG_L1_OUT_BLOCK_N == 64,
                     "swapAB expects BN256 split-N with 64 L1 output columns per WG");
    // Both dispatch warps participate in CTA-wide barriers. Selected plans may
    // use one warp for routing and token pulls, leaving the other warp's send
    // buffer available for an additional GEMM stage.
    constexpr uint32_t kNumActiveDispatchWarps =
        kSingleActiveDispatchWarp ? 1u : kNumDispatchWarps;
    constexpr uint32_t kNumActiveDispatchThreads = kNumActiveDispatchWarps * 32;
    constexpr bool kQuadDequantIlp =
        BLOCK_M == 8 && kNumStages == 4;
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    // A and B are CTA-local in the fixed cluster-size-one plan.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kL2ActsSFGranK =
        kSplitMDecodedWeightReuse ? 64u : 128u;
    DG_STATIC_ASSERT(kSplitMDecodedWeightReuse ||
                     WG_L1_OUT_BLOCK_N < kL2ActsSFGranK,
                     "split-N warpgroups must share one L2 activation scale");

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumActiveDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_NVFP4_LUT_SIZE =
        math::constexpr_align<uint32_t>(128u * sizeof(uint2), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // BM128 split-M alternates two decoded-B slots. This lets one WG begin
    // decoding K+1 after its K WGMMA completes without overwriting the slot
    // that the paired WG may still be consuming.
    constexpr uint32_t kNumDecodedBStages =
        kSplitMDecodedWeightReuse ? 2u : kNumStages;
    constexpr uint32_t B_LOAD_BYTES_PER_ROW = 80u;
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_STAGE =
        LOAD_BLOCK_N * B_LOAD_BYTES_PER_ROW * sizeof(b_dtype_t);
    // L1 and L2 each consume one per-128 activation scale per row and K tile.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t kNumL2SFAGroups =
        kSplitMDecodedWeightReuse ? 2u : 1u;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        kNumL2SFAGroups * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = kSwapABRequested ?
        BLOCK_M * BLOCK_N * sizeof(nv_bfloat16) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SLOTS =
        kNumEpilogueWarpgroups * BLOCK_M;
    constexpr uint32_t SMEM_CD_L1_SWAP_AMAX_SLOTS = kSwapABRequested ?
        BLOCK_M * kNumEpilogueWarps : 0u;
    constexpr uint32_t SMEM_CD_L1_EXTRA_FLOAT_SLOTS =
        SMEM_CD_L1_SHARED_SF_SLOTS > SMEM_CD_L1_SWAP_AMAX_SLOTS ?
        SMEM_CD_L1_SHARED_SF_SLOTS : SMEM_CD_L1_SWAP_AMAX_SLOTS;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SIZE =
        SMEM_CD_L1_EXTRA_FLOAT_SLOTS * sizeof(float);
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE + SMEM_CD_L1_SHARED_SF_SIZE;
    constexpr uint32_t SMEM_CD_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE;

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumActiveDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
    auto smem_nvfp4_lut = reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE);

    auto smem_cd_base = smem_gemm_base;
    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l1_shared_sf =
        math::advance_ptr<float>(smem_cd_base, SMEM_CD_OUTPUT_BASE_SIZE);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        const uint32_t decoded_stage =
            kSplitMDecodedWeightReuse ? (i & 1u) : i;
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base,
            SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            decoded_stage * SMEM_B_SIZE_PER_STAGE);
    });
    auto smem_packed_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE +
            i * SMEM_PACKED_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    // Barriers live after SF.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    constexpr uint32_t kNumBaseBarriers =
        kNumDispatchWarps + kNumStages * 2 + kNumEpilogueWarps * 2;
    auto task_info_full_barriers = barrier_start_ptr + kNumBaseBarriers;
    auto task_info_empty_barriers = task_info_full_barriers +
        interleaved_scheduler_t::kNumScheduleStages;
    auto task_infos = reinterpret_cast<task_info_t*>(
        task_info_empty_barriers +
        interleaved_scheduler_t::kNumScheduleStages);
    constexpr uint32_t kInterleavedSchedulerSMEMBytes =
        2 * interleaved_scheduler_t::kNumScheduleStages * sizeof(Barrier) +
        interleaved_scheduler_t::kNumScheduleStages * sizeof(task_info_t);
    DG_STATIC_ASSERT(
        kInterleavedSchedulerSMEMBytes ==
            layout::kSM90InterleavedSchedulerSMEMBytes,
        "Host and device scheduler shared-memory layouts disagree");
    constexpr uint32_t kInterleavedSMEMEnd =
        SMEM_BEFORE_BARRIER_SIZE + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
        kNumBaseBarriers * sizeof(Barrier) +
        kInterleavedSchedulerSMEMBytes;
    DG_STATIC_ASSERT(!kUseInterleavedScheduler || kInterleavedSMEMEnd <= 232448,
                     "Interleaved scheduler exceeds the SM90 shared-memory capacity");

    // =====================================================================
    // Initialization
    // =====================================================================
    if (thread_idx < 64) {
        reinterpret_cast<uint4*>(smem_nvfp4_lut)[thread_idx] =
            reinterpret_cast<const uint4*>(
                deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut)[thread_idx];
    }

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
                // Producer arrivals: A(+SFA) + B(TMA+SFB). SFB is copied with
                // cp.async.bulk and counted as B-loader transaction bytes, so
                // it does not need a separate producer arrival.
                full_barriers[i]->init(2);
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
            if constexpr (kUseInterleavedScheduler) {
                #pragma unroll
                for (uint32_t i = 0;
                     i < interleaved_scheduler_t::kNumScheduleStages;
                     ++ i) {
                    task_info_full_barriers[i].init(1);
                    task_info_empty_barriers[i].init(kNumEpilogueWarps);
                }
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    // NOTES: this port only wires up the interleaved (dynamic mailbox) task
    // scheduler -- `select_sm90_nvfp4_fused` always selects
    // `use_interleaved_scheduler = true`, so the static/non-interleaved path
    // below is unreachable and intentionally left unimplemented.
    auto interleaved_scheduler = interleaved_scheduler_t(
        workspace,
        task_info_full_barriers,
        task_info_empty_barriers,
        task_infos);

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
    constexpr uint32_t kSplitMDecodeBarrierIdx          = 8;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        kUseInterleavedScheduler ? 64 : 40;
    constexpr uint32_t kNumEpilogueRegisters    = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    const auto wait_live_ring_counter = [&] (
            uint32_t* ptr, const uint32_t target, const bool at_least,
            const uint32_t wait_tag, const uint32_t pool_block_idx,
            const uint32_t ring_block_idx) {
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
        const auto start_clock = clock64();
#endif
        while (true) {
            const auto current = ptx::ld_acq(ptr);
            if (at_least ? current >= target : current == target)
                return;
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
            if (clock64() - start_clock >=
                (DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS + 2ll) * 2000000000ll) {
                printf("DeepGEMM live-ring counter timeout: "
                       "rank=%d, sm=%u, warp=%u, wait_tag=%u, current=%u, "
                       "target=%u, pool_block=%u, ring_block=%u\n",
                       sym_buffer.rank_idx, sm_idx, warp_idx, wait_tag,
                       current, target, pool_block_idx, ring_block_idx);
                DG_DEVICE_ASSERT(false and "Live-ring counter timeout");
            }
#endif
        }
    };

    const auto wait_gemm_barrier = [&] (
            const Barrier* barrier, const uint32_t expected_phase,
            const uint32_t wait_kind, const uint32_t wait_stage,
            const uint32_t k_block_idx) {
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
        const auto start_clock = clock64();
        while (!barrier->try_wait(expected_phase)) {
            if (clock64() - start_clock >=
                (DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS + 2ll) * 2000000000ll) {
                if (lane_idx == 0) {
                    printf("DeepGEMM GEMM mbarrier timeout: "
                           "rank=%d, sm=%u, warp=%u, wait_kind=%u, "
                           "stage=%u, phase=%u, k_block=%u\n",
                           sym_buffer.rank_idx, sm_idx, warp_idx, wait_kind,
                           wait_stage, expected_phase, k_block_idx);
                }
                DG_DEVICE_ASSERT(false and "GEMM mbarrier timeout");
            }
        }
#else
        barrier->wait(expected_phase);
#endif
    };

    const auto record_math_progress = [&] (
            const uint32_t checkpoint, const uint32_t detail) {
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
        if ((DG_SM90_NVFP4_MOE_PROGRESS_MASK & (1u << checkpoint)) == 0)
            return;
        if (cumulative_local_expert_recv_stats != nullptr &&
            sm_idx < kNumExpertsPerRank &&
            warp_idx == kNumDispatchWarps + kNumMMANonEpilogueWarps &&
            lane_idx == 0) {
            cumulative_local_expert_recv_stats[sm_idx] =
                static_cast<int>((checkpoint << 28) | (detail & 0x0fffffffu));
            auto progress_word = workspace.get_debug_progress_word_ptr(sm_idx / 8);
            const uint32_t shift = (sm_idx % 8) * 4;
            const uint32_t mask = 0xfu << shift;
            uint32_t old_word = *progress_word;
            while (true) {
                const uint32_t new_word =
                    (old_word & ~mask) | ((checkpoint & 0xfu) << shift);
                const uint32_t observed = atomicCAS(
                    progress_word, old_word, new_word);
                if (observed == old_word)
                    break;
                old_word = observed;
            }
            __threadfence();
        }
#endif
    };

    // NOTES: the static (non-interleaved) scheduling path is unreachable (see
    // note above) and is not ported; `if constexpr (kUseInterleavedScheduler)`
    // guards below fail to compile with a clear message if ever instantiated
    // with `kUseInterleavedScheduler = false`.

    const auto invoke_interleaved_task = [&](const task_info_t& task_info,
                                              auto&& func) {
        if (task_info.block_phase == sched::BlockPhase::Linear1) {
            func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                 task_info.local_expert_idx, L1_SHAPE_K / BLOCK_K,
                 task_info.m_block_idx, task_info.n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m);
        } else {
            func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                 task_info.local_expert_idx, L2_SHAPE_K / BLOCK_K,
                 task_info.m_block_idx, task_info.n_block_idx,
                 task_info.pool_block_idx, task_info.valid_m);
        }
    };

    const auto for_each_published_block = [&](auto&& func) {
        task_info_t task_info;
        while (interleaved_scheduler.get_published_task(task_info))
            invoke_interleaved_task(task_info, func);
    };

    const auto produce_interleaved_blocks = [&](auto&& func) {
        interleaved_scheduler.fetch_expert_recv_count();
        while (true) {
            interleaved_scheduler.wait_task_slot_empty();
            const auto task_info = interleaved_scheduler.claim_next_task();
            interleaved_scheduler.publish_task(task_info);
            if (!task_info.is_valid())
                break;
            invoke_interleaved_task(task_info, func);
        }
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if constexpr (kUseInterleavedScheduler) {
                if (thread_idx == 0) {
                    *workspace.get_l1_task_count_ptr() = 0;
                    *workspace.get_l2_task_count_ptr() = 0;
                }
            }
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = interleaved_scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and
                        cumulative_local_expert_recv_stats != nullptr
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
                        and false
#endif
                    )
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_full_count_ptr((cleanup_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l1_empty_count_ptr((cleanup_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_full_count_ptr((cleanup_pool_block_offset + j) % kNumRingBlocks) = 0;
                    *workspace.get_l2_empty_count_ptr((cleanup_pool_block_offset + j) % kNumRingBlocks) = 0;
                }
                __syncwarp();
            }
        }
    };

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
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            if (warp_idx < kNumActiveDispatchWarps) {
                #pragma unroll
                for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                        expert_idx = static_cast<int>(
                            __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                        if (expert_idx >= 0)
                            process(i * kNumTopk + lane_idx, expert_idx);
                    }
                    __syncwarp();
                }
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

        if (sm_idx == 0 and thread_idx < kNumActiveDispatchThreads) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumActiveDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status;
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

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        // NOTES: every dispatch warp must populate its own per-lane
        // `stored_num_tokens_per_expert`/`num_total_m_blocks` state here, even
        // when `kSingleActiveDispatchWarp` leaves this warp idle for the rest
        // of the pull loop below -- `cleanup_workspace()` later calls
        // `get_pool_block_offset()` from every dispatch thread (both warps) to
        // find where each expert's data lives in the pool before zeroing its
        // ring-buffer full/empty counters, and that read is warp-local register
        // state, not shared across warps. A warp that skipped this fetch would
        // derive an always-zero offset there and leave the *other* warp's
        // counters un-zeroed, which the next launch can then wait on forever.
        // This must happen here (before any cleanup zeroing) rather than
        // inside `cleanup_workspace()` itself, since that function zeros
        // `get_expert_recv_count_sum_ptr` per-expert as it goes and a fetch
        // racing against another SM's cleanup could read an already-zeroed
        // slot and hang waiting for a completion tag that will never reappear
        // this launch.
        interleaved_scheduler.fetch_expert_recv_count();

        if (warp_idx < kNumActiveDispatchWarps) {
            uint32_t pull_mbarrier_phase = 0;
            const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
            const auto pull_mbarrier = dispatch_barriers[warp_idx];

            constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
            int      current_expert_idx = -1;
            uint32_t stored_rank_count[kNumRanksPerLane] = {};
            uint32_t expert_start_idx = 0, expert_end_idx = 0;
            uint32_t expert_pool_block_offset = 0;

            constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumActiveDispatchWarps;
            for (uint32_t token_idx = sm_idx * kNumActiveDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += interleaved_scheduler.get_num_tokens(current_expert_idx);
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
                // `pool_token_idx` stays absolute for the full-pool combine
                // metadata below; `ring_token_idx` (its flat reduction mod
                // `kNumRingTokens`) addresses the ring-buffered L1 token/SF/
                // weight storage. Since `kNumRingTokens` is `BLOCK_M`-divisible
                // (host side aligns it to the LCM of every candidate BLOCK_M),
                // `ring_token_idx == ring_block_idx * BLOCK_M + token_in_block`
                // exactly, so no separate block/offset split is needed here.
                const uint32_t pool_block_idx =
                    expert_pool_block_offset + token_idx_in_expert / BLOCK_M;
                const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
                const uint32_t pool_token_idx =
                    expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                const uint32_t ring_token_idx = pool_token_idx % kNumRingTokens;

                // Wait for the ring slot to be free: the previous lap's L1
                // GEMM-epilogue consumers must have fully vacated it before
                // this lap's tokens may overwrite it.
                const auto l1_empty_count_target = (pool_block_idx / kNumRingBlocks) * kNumRoutedL1BlockNs;
                if (l1_empty_count_target > 0) {
                    const auto empty_ptr = workspace.get_l1_empty_count_ptr(ring_block_idx);
                    wait_live_ring_counter(
                        empty_ptr, l1_empty_count_target, true, 1,
                        pool_block_idx, ring_block_idx);
                }

                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + ring_token_idx] = remote_sf_ptr[j];
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    const auto weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    *l1_topk_weights_buffer.get_data_buffer(ring_token_idx).get_base_ptr<float>() = weight;

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(ring_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    // Pad the last token in a block to a constant BLOCK_M per
                    // block, so the lap-scaled full-count target the readers
                    // wait on (`BLOCK_M * (lap + 1)`) is independent of any
                    // block's actual `valid_m` -- required because different
                    // laps through the same ring slot can have different
                    // `valid_m`, unlike a static full pool where each slot is
                    // only ever visited once.
                    const bool is_last_token = (token_idx == expert_end_idx - 1);
                    ptx::red_add_rel(
                        workspace.get_l1_full_count_ptr(ring_block_idx),
                        is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u);
                }
                __syncwarp();
            }
        }

        // Cleanup workspace, overlapping with combine
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        cleanup_workspace();
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);
    } else if (warp_idx == kNumDispatchWarps) {
        // =====================================================================
        // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
        //   The two warps inside `kNumNonEpilogueThreads` load A + SFA and
        //   B + SFB, respectively.
        // =====================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_a_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            const auto tensor_map_a_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const bool has_valid_m = valid_m > 0;
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;

            // Wait for the ring slot to be ready: a lap-scaled target on a
            // monotonic count (never reset mid-kernel), matching every other
            // MMA kind's ring scheme. L1's writer pads its per-token count to
            // a constant BLOCK_M per block (see the dispatch loop) so this
            // target is lap-invariant regardless of any block's actual
            // `valid_m`; L2's writer increments once per (pool_block, n_block)
            // in `notify_l1_ready`, so `kNumRoutedL1BlockNs` per lap is exact.
            if (has_valid_m) {
                if constexpr (!kBlockIsL2) {
                    const auto ptr = workspace.get_l1_full_count_ptr(ring_block_idx);
                    const auto num_expected_tokens = BLOCK_M * (pool_block_idx / kNumRingBlocks + 1);
                    wait_live_ring_counter(
                        ptr, num_expected_tokens, false, 2,
                        pool_block_idx, ring_block_idx);
                } else {
                    const auto ptr = workspace.get_l2_full_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = kNumRoutedL1BlockNs * (pool_block_idx / kNumRingBlocks + 1);
                    wait_live_ring_counter(
                        ptr, num_expected_blocks, false, 3,
                        pool_block_idx, ring_block_idx);
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                wait_gemm_barrier(
                    empty_barriers[stage_idx], phase ^ 1, 3u,
                    stage_idx, k_block_idx);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                        const uint32_t m_idx = ring_block_idx * BLOCK_M;
                        const uint32_t k_idx = k_block_idx * BLOCK_K;

                        // TMA load A
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                            k_idx, m_idx, 1);

                        if constexpr (!kBlockIsL2 || !kSplitMDecodedWeightReuse) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, k_block_idx, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                        } else {
                            // BN128 L1 produces per-64 activation scales. L2
                            // consumes both scale groups for each BK128 tile.
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, k_block_idx * 2, 1);
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + kL2SFAHalfStride,
                                m_idx, k_block_idx * 2 + 1, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                        }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
        };
        if constexpr (kUseInterleavedScheduler)
            for_each_published_block(load_a_task);
        else
            static_assert(kUseInterleavedScheduler, "Static (non-interleaved) scheduling is not supported");

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_b_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            const auto tensor_map_b_ptr = kBlockIsL2 ?
                &tensor_map_l2_weights : &tensor_map_l1_weights;
            constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                wait_gemm_barrier(
                    empty_barriers[stage_idx], phase ^ 1, 4u,
                    stage_idx, k_block_idx);

                const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                // NVFP4 fused B+scale layout stores 64B packed FP4 + 8B
                // UE4M3 scale + 8B zero padding per BK128 row.
                const uint32_t k_idx = k_block_idx * B_LOAD_BYTES_PER_ROW;
                if (cute::elect_one_sync()) {
                    tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx],
                        smem_packed_b[stage_idx],
                        k_idx, n_idx, 1);
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        SMEM_PACKED_B_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        };
        if constexpr (kUseInterleavedScheduler)
            produce_interleaved_blocks(load_b_task);
        else
            static_assert(kUseInterleavedScheduler, "Static (non-interleaved) scheduling is not supported");

    } else {
        // =====================================================================
        // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
        // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if (lane_idx == 0)
                empty_barriers[s]->arrive();
        };

        // Signals that this ring slot's L1 output (this task's N block) is
        // ready for L2 to consume (`l2_full_count`), and that the L1 input
        // buffers this task read from are no longer needed, so a future lap
        // may overwrite this ring slot (`l1_empty_count`).
        const auto notify_l1_ready = [&](const uint32_t& ready_ring_block_idx) {
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                ptx::red_add_rel(
                    workspace.get_l2_full_count_ptr(ready_ring_block_idx), 1u);
                ptx::red_add(
                    workspace.get_l1_empty_count_ptr(ready_ring_block_idx), 1u);
            }
            __syncwarp();
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        DG_STATIC_ASSERT(kSwapABRequested ||
                         (WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N),
                         "Split-N WGs must each run one M64N128 WGMMA per K-block");

        // Sync with dispatch
        record_math_progress(6u, 0u);
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        record_math_progress(7u, 0u);

        const auto run_math_task = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            // NOTES: `m_idx` (ring-slot-relative) addresses this task's L1
            // input (`l1_topk_weights_buffer`) and L1-output/L2-input
            // (`l2_token_buffer`, `l2_sf_buffer`) -- all ring buffers. The L2
            // branch's own combine-metadata addressing uses the raw absolute
            // `pool_block_idx * BLOCK_M` directly (full, non-ring pool), not
            // this variable -- see below.
            const uint32_t ring_block_idx = pool_block_idx % kNumRingBlocks;
            const uint32_t m_idx = ring_block_idx * BLOCK_M;
            const uint32_t wg_n_idx =
                kSplitMDecodedWeightReuse ? 0u : epilogue_wg_idx * WG_BLOCK_N;
            const uint32_t wg_l1_out_n_idx =
                kSplitMDecodedWeightReuse ? 0u : epilogue_wg_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
            const uint32_t row_block_offset =
                kSplitMDecodedWeightReuse ? epilogue_wg_idx * WG_BLOCK_M : 0u;
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            record_math_progress(
                1u, (static_cast<uint32_t>(BlockPhaseTag::value) << 24) |
                    ((pool_block_idx & 0xfffu) << 12) |
                    (n_block_idx & 0xfffu));
            const float l2_global_scale = kNvfp4ToFp8ScaleCompensation *
                (l2_global_scales == nullptr ? 1.0f :
                                                __ldg(l2_global_scales + local_expert_idx));
            const auto cast_l2_scaled_bf16_pair = [&](float x, float y) -> uint32_t {
                x *= l2_global_scale;
                y *= l2_global_scale;
                return math::cast_into_bf16_and_pack(x, y);
            };

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};
            const auto decode_b_stage = [&](const uint32_t& decoded_stage) {
                if constexpr (kSplitMDecodedWeightReuse) {
                    // Both M64 consumers share the same decoded N128 tile.
                    // The two physical decoded slots remove the overwrite
                    // hazard; the trailing pair barrier publishes K+1 before
                    // either consumer issues its WGMMA.
                    DG_STATIC_ASSERT(kUseMode2RowDecoder,
                                     "BM128 split-M uses the cooperative Mode2 decoder");
                    deep_gemm::nvfp4::
                        dequant_smem_b_from_packed_mode2_nibble_split_m(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    cutlass::arch::fence_view_async_shared();
                    asm volatile("bar.sync %0, %1;" : :
                                 "n"(kSplitMDecodeBarrierIdx), "n"(256) : "memory");
                } else {
                    if constexpr (kUseMode2RowDecoder) {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_mode2_nibble<
                            kQuadDequantIlp>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    } else {
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_braided_lut_window<
                            kQuadDequantIlp>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_nvfp4_lut);
                    }
                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(
                        128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                }
            };
            for (uint32_t k_block_idx = 0;
                 k_block_idx < num_k_blocks;
                 advance_pipeline(k_block_idx)) {
                wait_gemm_barrier(
                    full_barriers[stage_idx], phase, 5u,
                    stage_idx, k_block_idx);
                record_math_progress(
                    2u, (static_cast<uint32_t>(BlockPhaseTag::value) << 24) |
                        ((stage_idx & 0xffu) << 16) | k_block_idx);
                if constexpr (kUseInterleavedScheduler) {
                    if (k_block_idx == 0)
                        interleaved_scheduler.release_task_info(lane_idx);
                }
                decode_b_stage(stage_idx);
                record_math_progress(
                    3u, (static_cast<uint32_t>(BlockPhaseTag::value) << 24) |
                        ((stage_idx & 0xffu) << 16) | k_block_idx);

                // Read SF (must precede warpgroup_arrive)
                const float scale_a_0_lo =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                const float scale_a_1_lo =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                float scale_a_0_hi = 0.0f;
                float scale_a_1_hi = 0.0f;
                if constexpr (kBlockIsL2 && kSplitMDecodedWeightReuse) {
                    scale_a_0_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                }

                // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8 smem
                // expansion, so the WGMMA accumulator only needs activation SF.

                if constexpr (!kBlockIsL2) {
                    if constexpr (kSwapABRequested) {
                        auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum] = {};

                            #pragma unroll
                            for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
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
                                    const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] += scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[accum_offset + 2] += scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] += scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[accum_offset + 3] += scale_1 * swap_accum[i * 4 + 3];
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l1.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else {
                                run_swap_ab_l1.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else {
                                run_swap_ab_l1.template operator()<24>();
                            }
                        }
                    } else {
                        float accum[kAccumPerThread] = {};
                        // Single per-128 K-block WGMMA group
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

                        // L1: gate/up alternate at gran=8 along N; each `i` block
                        // of 8 cols belongs entirely to one of {gate, up}, so .x
                        // and .y share the same scalar.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                        }
                    }
                } else {
                    if constexpr (kSwapABRequested) {
                        DG_STATIC_ASSERT(kL2ActsSFGranK == 128,
                                         "L2 swap-AB requires per-128 activation scales");
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum] = {};

                            #pragma unroll
                            for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
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
                                    const uint32_t accum_offset =
                                        half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] +=
                                            scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[accum_offset + 2] +=
                                            scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] +=
                                            scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[accum_offset + 3] +=
                                            scale_1 * swap_accum[i * 4 + 3];
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l2.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else {
                                run_swap_ab_l2.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else {
                                run_swap_ab_l2.template operator()<24>();
                            }
                        }
                    } else {
                        float accum[kAccumPerThread] = {};
                        const auto promote_l2_accum = [&](const float& scale_r0,
                                                          const float& scale_r1) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_r0 * accum[i*4+0];
                                final_accum[i*4+1] += scale_r0 * accum[i*4+1];
                                final_accum[i*4+2] += scale_r1 * accum[i*4+2];
                                final_accum[i*4+3] += scale_r1 * accum[i*4+3];
                            }
                        };
                        if constexpr (kSplitMDecodedWeightReuse) {
                            #pragma unroll
                            for (uint32_t sf_group = 0; sf_group < 2; ++ sf_group) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                    const uint32_t k_off =
                                        sf_group * (BLOCK_K / 2) + k * WGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();
                                if (sf_group == 0)
                                    promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                                else
                                    promote_l2_accum(scale_a_0_hi, scale_a_1_hi);
                            }
                            arrive_empty_barrier(stage_idx);
                        } else {
                            // One per-128 scale permits a single four-instruction
                            // WGMMA group and one accumulator promotion per K tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                    k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();
                            arrive_empty_barrier(stage_idx);
                            promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                        }
                    }
                }
            }

            record_math_progress(
                4u, (static_cast<uint32_t>(BlockPhaseTag::value) << 24) |
                    ((pool_block_idx & 0xfffu) << 12) |
                    (n_block_idx & 0xfffu));

            // Skip epilogue when block is past valid M (the GEMM loop already
            // released its pipeline stages). Drain any prior L1 async store.
            if (valid_m == 0) {
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            if constexpr (!kBlockIsL2) {
                // Wait for the ring slot's previous-lap L2 consumers to have
                // fully vacated `l2_token_buffer`/`l2_sf_buffer` before this
                // lap's L1 epilogue overwrites them (see the matching
                // `l2_empty_count` increment in the L2 branch below).
                {
                    const auto l2_empty_ptr = workspace.get_l2_empty_count_ptr(ring_block_idx);
                    const auto num_expected_blocks = kNumRoutedL2BlockNs * (pool_block_idx / kNumRingBlocks);
                    wait_live_ring_counter(
                        l2_empty_ptr, num_expected_blocks, false, 4,
                        pool_block_idx, ring_block_idx);
                }
                const float l1_global_scale = kNvfp4ToFp8ScaleCompensation *
                    (l1_global_scales == nullptr ? 1.0f :
                                                  __ldg(l1_global_scales + local_expert_idx));
                if constexpr (kSwapABRequested) {
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

                    constexpr uint32_t reduce_warp_start = 0;
                    constexpr uint32_t reduce_warp_count = kNumEpilogueWarps;
                    const uint32_t scale_token_thread = epilogue_thread_idx;
                    constexpr uint32_t scale_token_stride = kNumEpilogueThreads;
                    const uint32_t sf_base_k_idx =
                        n_block_idx * L1_OUT_BLOCK_N / kL2ActsSFGranK;
                    float swap_v0[kSwapABWeightHalves][kSwapABTokenChunks] = {};
                    float swap_v1[kSwapABWeightHalves][kSwapABTokenChunks] = {};

                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;

                        float v0_amax = 0.0f;
                        float v1_amax = 0.0f;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            float v0 = 0.0f;
                            if (token_0 < valid_m) {
                                float g0 = final_accum[accum_offset + 0] * l1_global_scale;
                                float u0 = final_accum[accum_offset + 2] * l1_global_scale;
                                clamp_gate(g0);
                                clamp_up(u0);
                                const float weight_0 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_0)
                                    .template get_base_ptr<float>();
                                v0 = silu(g0) * u0 * weight_0;
                                swap_v0[half][i] = v0;
                                v0_amax = cute::max(v0_amax, cute::abs(v0));
                            }

                            float v1 = 0.0f;
                            if (token_1 < valid_m) {
                                float g1 = final_accum[accum_offset + 1] * l1_global_scale;
                                float u1 = final_accum[accum_offset + 3] * l1_global_scale;
                                clamp_gate(g1);
                                clamp_up(u1);
                                const float weight_1 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_1)
                                    .template get_base_ptr<float>();
                                v1 = silu(g1) * u1 * weight_1;
                                swap_v1[half][i] = v1;
                                v1_amax = cute::max(v1_amax, cute::abs(v1));
                            }
                        }

                        const float amax0 = math::warp_reduce<4, true>(
                            v0_amax, math::ReduceMax<float>());
                        const float amax1 = math::warp_reduce<4, true>(
                            v1_amax, math::ReduceMax<float>());
                        if (row_idx == 0) {
                            if (token_0 < valid_m)
                                smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax0;
                            if (token_1 < valid_m)
                                smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax1;
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

                    for (uint32_t token = scale_token_thread;
                         token < valid_m;
                         token += scale_token_stride) {
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t w = 0; w < reduce_warp_count; ++ w)
                            amax = cute::max(
                                amax, smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start + w]);
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);

                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_idx = m_idx + token;
                        sf_base_ptr[sf_base_k_idx * kNumPaddedSFPoolTokens + token_idx] =
                            sf_pair.x;
                        smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start] = sf_inv_pair.x;
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    #pragma unroll
                    for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t out_col_base =
                                wg_l1_out_n_idx + half * 32u + warp_idx_in_wg * 8 + row_idx;
                            if (token_0 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + reduce_warp_start];
                                const __nv_fp8_e4m3 q(swap_v0[half][i] * sf_inv);
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                    *reinterpret_cast<const uint8_t*>(&q);
                            }
                            if (token_1 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + reduce_warp_start];
                                const __nv_fp8_e4m3 q(swap_v1[half][i] * sf_inv);
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                    *reinterpret_cast<const uint8_t*>(&q);
                            }
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
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
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(ring_block_idx);
                } else {
                    // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                    const bool valid_r0 = row_offset_r0 < valid_m;
                    const bool valid_r1 = row_offset_r1 < valid_m;
                    // Layout in `final_accum`:
                    //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                    //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                    //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                    //
                    // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                    // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                    constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    constexpr uint32_t kNumSFGroups = 1;
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
                        float g_r0_c0 = final_accum[gate*4 + 0] * l1_global_scale; clamp_gate(g_r0_c0);
                        float g_r0_c1 = final_accum[gate*4 + 1] * l1_global_scale; clamp_gate(g_r0_c1);
                        float g_r1_c0 = final_accum[gate*4 + 2] * l1_global_scale; clamp_gate(g_r1_c0);
                        float g_r1_c1 = final_accum[gate*4 + 3] * l1_global_scale; clamp_gate(g_r1_c1);
                        float u_r0_c0 = final_accum[up*4   + 0] * l1_global_scale; clamp_up(u_r0_c0);
                        float u_r0_c1 = final_accum[up*4   + 1] * l1_global_scale; clamp_up(u_r0_c1);
                        float u_r1_c0 = final_accum[up*4   + 2] * l1_global_scale; clamp_up(u_r1_c0);
                        float u_r1_c1 = final_accum[up*4   + 3] * l1_global_scale; clamp_up(u_r1_c1);

                        auto silu = [](float x) -> float {
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


                    const float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r0)
                        .template get_base_ptr<float>() : 0.0f;
                    const float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r1)
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
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
                        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
                    }

                    if (col_idx == 0) {
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r0] = amax_r0[0];
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r1] = amax_r1[0];
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if constexpr (kSplitMDecodedWeightReuse) {
                        amax_r0[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r0];
                        amax_r1[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r1];
                    } else {
                        amax_r0[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r0],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r0]);
                        amax_r1[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r1],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r1]);
                    }

                    float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                    float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        float2 amax_pair = {amax_r0[g], amax_r1[g]};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
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
                            smem_cd_l1 + row_offset_r0 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        auto* p1 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1 + row_offset_r1 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        if (valid_r0)
                            *p0 = r0_pair.__x;
                        if (valid_r1)
                            *p1 = r1_pair.__x;
                    }

                    // Write one physical L2-activation scale per 128 output columns.
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_r0 = m_idx + row_offset_r0;
                        const uint32_t token_r1 = m_idx + row_offset_r1;
                        const uint32_t base_k_sf_idx =
                            (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / kL2ActsSFGranK;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                            const uint32_t sf_k_idx = base_k_sf_idx + g;
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r0)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0[g];
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r1)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1[g];
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
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(ring_block_idx);
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                // This task's GEMM load (`load_a_task`/`load_b_task`, a
                // different warp) has already finished reading `l2_token_buffer`
                // /`l2_sf_buffer` for this ring slot by construction (the GEMM
                // accumulate above depends on that load). Signal the slot free
                // for a future lap's L1 epilogue to overwrite.
                if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                    ptx::red_add(workspace.get_l2_empty_count_ptr(ring_block_idx), 1u);
                __syncwarp();

                if constexpr (kSwapABRequested) {
                    // Each active warp scatters a contiguous group of up to 16 rows.
                    constexpr uint32_t kNumRowsPerWarp =
                        BLOCK_M == 8 ? 4u : 8u;
                    auto store_swap_bf16 = [&](const uint32_t& token, const uint32_t& col, const float& value) {
                        if (token < valid_m)
                            smem_cd_l2[token * BLOCK_N + wg_n_idx + col] =
                                __float2bfloat16_rn(value * l2_global_scale);
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    auto store_l2_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            const uint32_t col_offset = half * 64u;
                            store_swap_bf16(token_0, col_offset + r_0, final_accum[accum_offset + 0]);
                            store_swap_bf16(token_0, col_offset + r_1, final_accum[accum_offset + 2]);
                            store_swap_bf16(token_1, col_offset + r_0, final_accum[accum_offset + 1]);
                            store_swap_bf16(token_1, col_offset + r_1, final_accum[accum_offset + 3]);
                        }
                    };

                    store_l2_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l2_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    const uint32_t row_in_warp_block = lane_idx / 16;
                    const uint32_t lane_in_row = lane_idx % 16;
                    constexpr uint32_t kColsPerScatterLane = WG_BLOCK_N / 16;
                    DG_STATIC_ASSERT(WG_BLOCK_N % 16 == 0,
                                     "SwapAB L2 scatter expects an even lane partition");
                    DG_STATIC_ASSERT(kColsPerScatterLane == 4 || kColsPerScatterLane == 8,
                                     "SwapAB L2 scatter supports WG_BLOCK_N=64 or 128");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t token = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        if (token >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                            pool_block_idx * BLOCK_M + token);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        auto smem_ptr = smem_cd_l2
                            + token * BLOCK_N
                            + wg_n_idx
                            + lane_in_row * kColsPerScatterLane;
                        if constexpr (kColsPerScatterLane == 8) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 64 || WG_BLOCK_N == 128,
                                     "Direct L2 scatter requires N64/N128");
                    const bool valid_r0 = row_offset_r0 < valid_m;
                    const bool valid_r1 = row_offset_r1 < valid_m;

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row, const uint32_t& row_accum_offset) {
                        if (valid_row) {
                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(
                                pool_block_idx * BLOCK_M + row_offset);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            const uint32_t dst_token_idx = src_metadata.token_idx;
                            const uint32_t dst_topk_idx = src_metadata.topk_idx;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_base = math::advance_ptr<uint8_t>(
                                dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16));
                            auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                const uint32_t packed_lo = cast_l2_scaled_bf16_pair(
                                    final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                    final_accum[chunk_lo * 4 + row_accum_offset + 1]);
                                const uint32_t packed_hi = cast_l2_scaled_bf16_pair(
                                    final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                    final_accum[chunk_hi * 4 + row_accum_offset + 1]);
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                            }
                        }
                    };

                    scatter_direct_row(row_offset_r0, valid_r0, 0);
                    scatter_direct_row(row_offset_r1, valid_r1, 2);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            }
            record_math_progress(
                5u, (static_cast<uint32_t>(BlockPhaseTag::value) << 24) |
                    ((pool_block_idx & 0xfffu) << 12) |
                    (n_block_idx & 0xfffu));
        };
        if constexpr (kUseInterleavedScheduler)
            for_each_published_block(run_math_task);
        else
            static_assert(kUseInterleavedScheduler, "Static (non-interleaved) scheduling is not supported");

#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
        record_math_progress(15u, 0u);
#endif

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
        constexpr uint32_t kNumChunks =
            (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
             and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

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
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

}  // namespace deep_gemm

#pragma clang diagnostic pop
