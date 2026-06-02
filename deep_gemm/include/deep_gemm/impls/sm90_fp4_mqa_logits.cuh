#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// Apply B128 swizzle transform to a byte address within a smem row.
// WGMMA with layout_type=B128 (K-major, FP8) expects data in this swizzle format.
// Within a 128B-wide row (kHeadDim=128 FP8 elements = 128 bytes):
//   The swizzle XORs 16-byte column index with (row_in_tile % 8).
//   swizzled_byte_col = byte_col ^ ((row_in_tile % 8) * 16)
// where byte_col is the byte offset within the row (0..127)
//   and row_in_tile is the row index within the swizzle tile (0..7 repeating).
//
// This allows software-written FP8 smem to match TMA-loaded swizzled layout.
CUTLASS_DEVICE __nv_fp8_e4m3* swizzled_fp8_ptr(
        __nv_fp8_e4m3* base,           // base of the 2D FP8 tile
        const uint32_t row,            // row index (0-based within tile)
        const uint32_t col,            // col index (0-based, in elements)
        const uint32_t head_dim) {     // stride (= kHeadDim = 128)
    // For B128 swizzle: interleave 16-byte blocks across rows
    // Each "group of 16 bytes" at col c is swizzled with row % 8
    const uint32_t col_group = col / 16;         // which 16-element group (0..7 for head_dim=128)
    const uint32_t col_in_group = col % 16;      // position within the 16-element group
    const uint32_t swizzled_col_group = col_group ^ (row % 8);
    const uint32_t swizzled_col = swizzled_col_group * 16 + col_in_group;
    return base + row * head_dim + swizzled_col;
}

// Dequantize a packed FP4 (E2M1) value with a MX UE8M0 block scale factor to FP8 E4M3.
// packed contains two FP4 values:  low nibble = element i, high nibble = element i+1.
// sf_src is the packed UE8M0 scale factor (one int32 per row in KV, encoding 4 UE8M0 bytes).
// The MX block granularity is 32 elements sharing one scale.
// We dequantize by: fp32 = fp4_as_float * 2^(sf_exp - 127), then saturate-cast to FP8.
// Output is written to smem with B128 swizzle so WGMMA can consume it directly.
CUTLASS_DEVICE void dequant_fp4_block_to_fp8(
        const uint8_t* __restrict__ fp4_src,     // packed FP4, head_dim/2 bytes per row
        const uint32_t* __restrict__ sf_src,     // packed UE8M0 SFs, one int32 per row
        __nv_fp8_e4m3* __restrict__ fp8_tile,    // output FP8 tile base ptr (NOT row ptr)
        const uint32_t row,                      // row index within the tile
        const uint32_t lane_idx,
        const uint32_t head_dim) {
    const uint32_t base = lane_idx * 4;
    const uint8_t sf_byte = (sf_src[0] >> ((base >> 5) * 8)) & 0xFF;
    const float sf_float = __int_as_float(static_cast<uint32_t>(sf_byte) << 23);

    const uint8_t p0 = fp4_src[base >> 1];
    const uint8_t p1 = fp4_src[(base >> 1) + 1];

    auto fp4_nibble_to_float = [](uint8_t nibble) -> float {
        static const float lut[16] = {
            0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        return lut[nibble & 0xF];
    };

    const float v0 = fp4_nibble_to_float(p0 & 0xF) * sf_float;
    const float v1 = fp4_nibble_to_float(p0 >> 4)  * sf_float;
    const float v2 = fp4_nibble_to_float(p1 & 0xF) * sf_float;
    const float v3 = fp4_nibble_to_float(p1 >> 4)  * sf_float;

    // lane_idx*4 never crosses a 16-byte swizzle group, so calculate the
    // B128-swizzled base address once and store four contiguous FP8 values.
    const uint32_t swizzled_col = (((base >> 4) ^ (row & 7)) << 4) + (base & 15);
    __nv_fp8_e4m3* out = fp8_tile + row * head_dim + swizzled_col;
    const __nv_fp8x4_e4m3 packed(make_float4(v0, v1, v2, v3));
    ptx::st_shared(reinterpret_cast<const uint32_t*>(out), packed.__x);
}

// SM90 FP4 MQA Logits kernel.
//
// Strategy: FP4 Q and KV are loaded into SMEM via TMA (packed format),
// then dequantized to FP8 in SMEM by math warpgroups before issuing WGMMA.
// This allows running on SM90 which has no native FP4 WGMMA support.
//
// Thread organization (identical to sm90_fp8_mqa_logits):
//   kNumTMAThreads  = 128  : TMA warp-group (one warp does TMA, rest exit)
//   kNumMathThreads = 512  : Math warpgroups (4 x 128) for dequant + WGMMA + reduce
//
// Synchronization design:
//   - TMA full/empty barriers (ClusterTransactionBarrier) coordinate TMA <-> Math threads
//   - NamedBarrier::sync(kNumMathThreads, N) coordinates dequant completion within Math threads
//     (NamedBarrier does NOT require TMA threads to participate, unlike __syncthreads())
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp4_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                         const uint32_t max_seqlen_k, const uint32_t stride_logits,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         logits_dtype_t* logits,
                         // Q: packed FP4 [seq_len * num_heads, head_dim/2] viewed as UINT8
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         // SF Q: packed UE8M0 [seq_len, num_heads] (one int32 per token-head)
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                         // KV: packed FP4 [seq_len_kv, head_dim/2] viewed as UINT8
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         // SF KV: packed UE8M0 [seq_len_kv, 1] (one int32 per KV token)
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                         // Weights: float [seq_len, num_heads]
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(kHeadDim == 128, "Only head_dim=128 supported");
    DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size: BLOCK_KV must equal kNumMathThreads/2");

    // Prefetch TMA descriptors
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // -------------------------------------------------------------------------
    // Shared memory layout
    // -------------------------------------------------------------------------
    // FP4 raw Q:   BLOCK_Q * kNumHeads * head_dim/2 bytes per stage  (packed FP4)
    // FP4 raw KV:  BLOCK_KV * head_dim/2 bytes per stage
    // SF Q:        BLOCK_Q * kNumHeads * sizeof(int32) bytes per stage
    // SF KV:       BLOCK_KV * sizeof(int32) per stage
    // Weights:     BLOCK_Q * kNumHeads * sizeof(float) per stage
    // FP8 dequant Q:   BLOCK_Q * kNumHeads * kHeadDim per stage (FP8 after dequant)
    // FP8 dequant KV:  BLOCK_KV * kHeadDim per stage
    // Barriers:    (kNumQStages * 2 + kNumKVStages * 2) barriers

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8; // 1024B for FP8 with kHeadDim=128

    // Packed FP4 stage sizes
    static constexpr uint32_t SMEM_FP4_Q_SIZE_PER_STAGE  = BLOCK_Q * kNumHeads * (kHeadDim / 2); // bytes
    static constexpr uint32_t SMEM_FP4_KV_SIZE_PER_STAGE = BLOCK_KV * (kHeadDim / 2);             // bytes
    // SF sizes: 1 int32 per (token, head) for Q, 1 int32 per token for KV
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE   = BLOCK_Q * kNumHeads * sizeof(uint32_t);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE  = BLOCK_KV * sizeof(uint32_t);
    // Weights
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    // Dequantized FP8 (written by math warpgroups, consumed by WGMMA)
    static constexpr uint32_t SMEM_FP8_Q_SIZE_PER_STAGE  = BLOCK_Q * kNumHeads * kHeadDim; // FP8
    static constexpr uint32_t SMEM_FP8_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim;             // FP8

    // Verify swizzle alignment for FP8 Q and KV
    DG_STATIC_ASSERT(SMEM_FP8_Q_SIZE_PER_STAGE  % kSwizzleAlignment == 0, "Unaligned TMA swizzling for FP8 Q");
    DG_STATIC_ASSERT(SMEM_FP8_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling for FP8 KV");

    // Total unaligned region (FP4 + SF + Weights)
    static constexpr uint32_t SMEM_RAW_OFFSET =
        SMEM_FP4_Q_SIZE_PER_STAGE  * kNumQStages  +
        SMEM_FP4_KV_SIZE_PER_STAGE * kNumKVStages +
        SMEM_SF_Q_SIZE_PER_STAGE   * kNumQStages  +
        SMEM_SF_KV_SIZE_PER_STAGE  * kNumKVStages +
        SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages;
    // FP8 region needs alignment
    static constexpr uint32_t SMEM_FP8_OFFSET = math::constexpr_align(SMEM_RAW_OFFSET, kSwizzleAlignment);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    // FP4 packed Q [kNumQStages]
    auto smem_fp4_q = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_FP4_Q_SIZE_PER_STAGE * i;
    });
    // FP4 packed KV [kNumKVStages]
    auto smem_fp4_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_FP4_Q_SIZE_PER_STAGE * kNumQStages
                           + SMEM_FP4_KV_SIZE_PER_STAGE * i;
    });
    // SF Q [kNumQStages]: int32 per (token, head)
    auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer
            + SMEM_FP4_Q_SIZE_PER_STAGE  * kNumQStages
            + SMEM_FP4_KV_SIZE_PER_STAGE * kNumKVStages
            + SMEM_SF_Q_SIZE_PER_STAGE * i);
    });
    // SF KV [kNumKVStages]: int32 per token
    auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer
            + SMEM_FP4_Q_SIZE_PER_STAGE  * kNumQStages
            + SMEM_FP4_KV_SIZE_PER_STAGE * kNumKVStages
            + SMEM_SF_Q_SIZE_PER_STAGE   * kNumQStages
            + SMEM_SF_KV_SIZE_PER_STAGE * i);
    });
    // Weights [kNumQStages]
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer
            + SMEM_FP4_Q_SIZE_PER_STAGE  * kNumQStages
            + SMEM_FP4_KV_SIZE_PER_STAGE * kNumKVStages
            + SMEM_SF_Q_SIZE_PER_STAGE   * kNumQStages
            + SMEM_SF_KV_SIZE_PER_STAGE  * kNumKVStages
            + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    // FP8 dequantized Q [kNumQStages] -- WGMMA operand B (N-dim)
    auto smem_fp8_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer
            + SMEM_FP8_OFFSET
            + SMEM_FP8_Q_SIZE_PER_STAGE * i);
    });
    // FP8 dequantized KV [kNumKVStages] -- WGMMA operand A (M-dim)
    auto smem_fp8_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer
            + SMEM_FP8_OFFSET
            + SMEM_FP8_Q_SIZE_PER_STAGE * kNumQStages
            + SMEM_FP8_KV_SIZE_PER_STAGE * i);
    });

    // Barriers layout (at end of smem)
    // Only TMA barriers: full/empty for Q and KV pipeline (4 groups)
    static constexpr uint32_t SMEM_BARRIER_OFFSET =
        SMEM_FP8_OFFSET
        + SMEM_FP8_Q_SIZE_PER_STAGE  * kNumQStages
        + SMEM_FP8_KV_SIZE_PER_STAGE * kNumKVStages;

    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_BARRIER_OFFSET);
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });

    // Initialize barriers (done by TMA warp only -- one thread)
    const bool is_tma_load_warp = kNumMathThreads <= threadIdx.x and threadIdx.x < kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Register reconfigurations
    // -------------------------------------------------------------------------
    constexpr uint32_t kNumTMARegisters  = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // -------------------------------------------------------------------------
    // Block scheduler
    // -------------------------------------------------------------------------
    const auto sm_idx = blockIdx.x;
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };

    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end   = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i]   = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end   = max(end,   min(seq_k_end[i],   seq_len_kv));
        }
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1,
                start, math::ceil_div(end - start, BLOCK_KV)};
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1
        };
    };

    // -------------------------------------------------------------------------
    // Wait for primary kernel completion
    // -------------------------------------------------------------------------
    cudaGridDependencySynchronize();

    if (threadIdx.x >= kNumMathThreads) {
        // =====================================================================
        // TMA warp-group: load FP4 Q, SF Q, FP4 KV, SF KV, Weights into SMEM
        // =====================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Only the first warp in the TMA warpgroup does the actual work
        if (not is_tma_load_warp)
            return;

        // Helper: issue TMA for Q (packed FP4 + SF Q + Weights)
        const auto issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            // Load packed FP4 Q (viewed as UINT8): shape [BLOCK_Q * kNumHeads, kHeadDim/2], no swizzle
            tma::copy<kHeadDim / 2, BLOCK_Q * kNumHeads, 0>(
                &tensor_map_q, full_q_barriers[stage_idx],
                smem_fp4_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            // Load SF Q: shape [kNumHeads, BLOCK_Q] (int32 per token-head)
            tma::copy<kNumHeads, BLOCK_Q, 0>(
                &tensor_map_sf_q, full_q_barriers[stage_idx],
                reinterpret_cast<uint8_t*>(smem_sf_q[stage_idx]), 0, block_idx * BLOCK_Q);
            // Load weights: shape [kNumHeads, BLOCK_Q]
            tma::copy<kNumHeads, BLOCK_Q, 0>(
                &tensor_map_weights, full_q_barriers[stage_idx],
                reinterpret_cast<uint8_t*>(smem_weights[stage_idx]), 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(
                SMEM_FP4_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };

        // Prefetch first Q block
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer (math threads) release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q for next block
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx());
                    next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Load packed FP4 KV (viewed as UINT8): shape [BLOCK_KV, kHeadDim/2], no swizzle
                    tma::copy<kHeadDim / 2, BLOCK_KV, 0>(
                        &tensor_map_kv, full_kv_barriers[kv_stage_idx],
                        smem_fp4_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    // Load SF KV: shape [BLOCK_KV, 1] (int32 per KV token)
                    tma::copy<BLOCK_KV, 1, 0>(
                        &tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
                        reinterpret_cast<uint8_t*>(smem_sf_kv[kv_stage_idx]),
                        kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
                        SMEM_FP4_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else {
        // =====================================================================
        // Math warp-groups: dequantize FP4->FP8, run WGMMA, reduce and store
        // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto& thread_idx    = threadIdx.x % kNumMathThreads;
        const auto& warp_idx      = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx      = ptx::get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset  = lane_idx / 4 + 0;
        const auto& v_1_offset  = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // ------------------------------------------------------------------
            // Step 1: Wait for TMA Q arrival, then dequantize FP4 Q -> FP8 Q
            // ------------------------------------------------------------------
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Each thread in the warpgroup dequantizes a portion of Q.
            // Q is [BLOCK_Q * kNumHeads, kHeadDim/2] in packed FP4.
            // SF Q is [BLOCK_Q, kNumHeads] in int32 (packed UE8M0).
            // We dequantize to FP8: [BLOCK_Q * kNumHeads, kHeadDim].
            {
                const uint32_t total_rows = BLOCK_Q * kNumHeads;
                for (uint32_t row = thread_idx / 32; row < total_rows; row += kNumMathThreads / 32) {
                    const uint8_t* fp4_row_ptr = smem_fp4_q[q_stage_idx] + row * (kHeadDim / 2);
                    // SF: row = q_token_in_block * kNumHeads + head_idx
                    const uint32_t q_token_in_block = row / kNumHeads;
                    const uint32_t head_idx = row % kNumHeads;
                    // smem_sf_q[stage][token][head] = one int32 per (token, head)
                    const uint32_t* sf_row_ptr = smem_sf_q[q_stage_idx] + q_token_in_block * kNumHeads + head_idx;
                    // Pass tile base + row index so the function can apply B128 swizzle
                    dequant_fp4_block_to_fp8(fp4_row_ptr, sf_row_ptr, smem_fp8_q[q_stage_idx], row, lane_idx, kHeadDim);
                }
            }
            // Barrier id 0: synchronize all kNumMathThreads after Q dequant
            // This ensures FP8 Q smem writes are visible to all threads before WGMMA
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            // Read weights (after TMA Q is done, weights are in smem)
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            // ------------------------------------------------------------------
            // Step 2: For each KV block, dequantize FP4 KV -> FP8 KV, then WGMMA
            // ------------------------------------------------------------------
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Dequantize only the KV rows consumed by this warpgroup.
                // Each WGMMA warpgroup owns a contiguous WGMMA::M-row slice of
                // the KV tile, so it only needs to synchronize with itself
                // before issuing WGMMA on that slice.
                {
                    const uint32_t wg_thread_idx = thread_idx % 128;
                    const uint32_t wg_warp_idx = wg_thread_idx / 32;
                    const uint32_t row_end = (warpgroup_idx + 1) * WGMMA::M;
                    for (uint32_t row = warpgroup_idx * WGMMA::M + wg_warp_idx; row < row_end; row += 4) {
                        const uint8_t* fp4_row_ptr = smem_fp4_kv[kv_stage_idx] + row * (kHeadDim / 2);
                        const uint32_t* sf_row_ptr = smem_sf_kv[kv_stage_idx] + row;
                        // Pass tile base + row index so the function can apply B128 swizzle
                        dequant_fp4_block_to_fp8(fp4_row_ptr, sf_row_ptr, smem_fp8_kv[kv_stage_idx], row, lane_idx, kHeadDim);
                    }
                }
                // Barrier ids 1..kNumMathThreads/128: one local barrier per
                // WGMMA warpgroup after its KV slice is ready.
                cutlass::arch::NamedBarrier::sync(128, warpgroup_idx + 1);

                // Per-KV scales: the fp8 representation already has the scale embedded
                // (FP4 * UE8M0_scale -> FP8). For the reduce step we need a per-KV
                // scale to match the sm90_fp8 interface. Since we baked the scale into FP8,
                // the effective scale is 1.0.
                float scale_kv_0 = 1.0f;
                float scale_kv_1 = 1.0f;

                // ------------------------------------------------------------------
                // WGMMA: [BLOCK_KV, kHeadDim] @ [BLOCK_Q * kNumHeads, kHeadDim].T
                //        -> [BLOCK_KV, BLOCK_Q * kNumHeads] (per warpgroup)
                // A = smem_fp8_kv (KV, M-dim), B = smem_fp8_q (Q, N-dim)
                // ------------------------------------------------------------------
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_fp8_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_fp8_q[q_stage_idx] + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                // Release KV TMA buffer (all math threads arrive)
                empty_kv_barriers[kv_stage_idx]->arrive();

                // ------------------------------------------------------------------
                // Reduce over the head dim and store
                // ------------------------------------------------------------------
                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto transform = [&](const uint32_t& j) {
                        return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                    };

                    // Intra-thread reduction
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k++)
                            sum[k] += transform(j * 4 + k);
                    }
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    // Store into global memory
                    const auto q_offset = (block_q_idx * BLOCK_Q + i) * static_cast<uint64_t>(stride_logits);
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[i] <= kv_offset + v_0_offset and kv_offset + v_0_offset < seq_k_end[i])
                            logits[q_offset + kv_offset + v_0_offset - seq_k_start[i]] = static_cast<logits_dtype_t>(v_0);
                        if (seq_k_start[i] <= kv_offset + v_1_offset and kv_offset + v_1_offset < seq_k_end[i])
                            logits[q_offset + kv_offset + v_1_offset - seq_k_start[i]] = static_cast<logits_dtype_t>(v_1);
                    } else {
                        logits[q_offset + kv_offset + v_0_offset] = static_cast<logits_dtype_t>(v_0);
                        logits[q_offset + kv_offset + v_1_offset] = static_cast<logits_dtype_t>(v_1);
                    }
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q TMA buffer (all math threads arrive)
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
}

} // namespace deep_gemm
