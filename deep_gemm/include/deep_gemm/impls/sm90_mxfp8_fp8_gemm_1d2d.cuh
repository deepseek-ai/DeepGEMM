#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

namespace mxfp8_fp8_detail {

CUTLASS_DEVICE float e8m0_to_float(uint8_t scale) {
    return __uint_as_float(static_cast<uint32_t>(scale) << 23);
}

CUTLASS_DEVICE uint8_t load_e8m0_scale(const void* ptr, uint32_t base_offset,
                                       uint32_t k_scale_idx, uint32_t stride_k,
                                       bool packed_int32) {
    if (packed_int32) {
        const auto packed = reinterpret_cast<const uint32_t*>(ptr)[base_offset + (k_scale_idx / 4) * stride_k];
        return static_cast<uint8_t>((packed >> ((k_scale_idx % 4) * 8)) & 0xff);
    }
    return reinterpret_cast<const uint8_t*>(ptr)[base_offset + k_scale_idx * stride_k];
}

// Load the four UE8M0 scales that cover one 128-wide K block (chunks kk = 0..3)
// and pack them into a single 32-bit word (byte kk = scale for chunk kk). This
// replaces four dependent per-byte global loads with a single 32-bit load on the
// common `gran_k == 32`, unit-stride path. Bounds/granularity are handled here so
// callers can drop their per-element K guards.
CUTLASS_DEVICE uint32_t load_e8m0_scale_quad(const void* ptr, uint32_t base_offset,
                                             uint32_t k_block_idx, uint32_t stride_k,
                                             uint32_t gran_k, bool packed_int32,
                                             uint32_t shape_k) {
    constexpr uint32_t kIdentityScalePack = 0x7f7f7f7f;
    const uint32_t first_k_scale_idx = (k_block_idx * 128) / gran_k;
    const uint32_t first_k = k_block_idx * 128;

    if (first_k >= shape_k)
        return kIdentityScalePack;

    if (gran_k == 32 and first_k + 128 <= shape_k) {
        if (packed_int32 and stride_k == 1)
            return reinterpret_cast<const uint32_t*>(ptr)[base_offset + first_k_scale_idx / 4];
        if (not packed_int32 and stride_k == 1 and base_offset % 4 == 0)
            return *reinterpret_cast<const uint32_t*>(
                reinterpret_cast<const uint8_t*>(ptr) + base_offset + first_k_scale_idx);
    }

    uint32_t packed = 0;
    #pragma unroll
    for (uint32_t kk = 0; kk < 4; ++ kk) {
        const uint32_t k = first_k + kk * 32;
        const uint32_t k_scale_idx = k / gran_k;
        const uint8_t scale = k < shape_k ?
            load_e8m0_scale(ptr, base_offset, k_scale_idx, stride_k, packed_int32) :
            static_cast<uint8_t>(127);
        packed |= static_cast<uint32_t>(scale) << (kk * 8);
    }
    return packed;
}

} // namespace mxfp8_fp8_detail

template <bool kMasked,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_mxfp8_fp8_gemm_1d2d_impl(void* sfa, void* sfb, int* grouped_layout,
                              uint32_t sfa_stride_group, uint32_t sfa_stride_m, uint32_t sfa_stride_k,
                              uint32_t sfa_gran_k, bool sfa_packed_int32,
                              uint32_t sfb_stride_group, uint32_t sfb_stride_n, uint32_t sfb_stride_k,
                              uint32_t sfb_gran_k, bool sfb_packed_int32,
                              uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(BLOCK_K == 128, "MXFP8 scale stage assumes 4 K/32 scale groups");
    DG_STATIC_ASSERT(kNumStages >= 1, "Invalid pipeline stages");

    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static constexpr uint32_t kNumMathWarpgroups = kNumMathThreads / 128;

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SHAPE_K_SFA_PER_STAGE = BLOCK_K / 32;
    // Materialize UE8M0 as FP32 in the producer warp. This shifts exponent unpacking out
    // of the math warpgroups' promotion loop, where instruction issue is the bottleneck.
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * SHAPE_K_SFA_PER_STAGE * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    static constexpr uint32_t SHAPE_K_SFB_PER_STAGE = BLOCK_K / 32;
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * SHAPE_K_SFB_PER_STAGE * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u);

    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE +
                                        i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_SF_OFFSET + kNumStages * (ALIGNED_SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto scale_full_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return barrier_start_ptr + kNumStages * 2 + i;
    });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
            #pragma unroll
            for (uint32_t wg = 0; wg < kNumMathWarpgroups; ++ wg)
                scale_full_barriers[i * kNumMathWarpgroups + wg]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    cudaGridDependencySynchronize();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (warp_idx == kNumMathThreads / 32 + 2) {
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    const bool is_producer_leader = cute::elect_one_sync();
                    if (is_producer_leader)
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                    __syncwarp();

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    if (is_producer_leader) {
                        tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, false>(&tensor_map_a, &full_barrier,
                                 smem_a[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                                 num_tma_multicast_a);
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, false>(&tensor_map_b, &full_barrier,
                                 smem_b[stage_idx], k_idx, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                                 num_tma_multicast_b);
                        // Publish A/B readiness immediately. Scale staging is guarded by
                        // scale_full_barriers below, allowing chunk-0 WGMMA to overlap it.
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                    }

                    const uint32_t sfa_base_m = scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx);
                    for (uint32_t m_offset = lane_idx; m_offset < BLOCK_M; m_offset += 32) {
                        const uint32_t m_idx = sfa_base_m + m_offset;
                        const bool is_valid = m_idx < shape_m * (kMasked ? kNumGroups : 1);
                        const uint32_t sfa_local_m = kMasked ? (m_idx - scheduler.current_group_idx * shape_m) : m_idx;
                        const uint32_t sfa_base_offset = (kMasked ? scheduler.current_group_idx * sfa_stride_group : 0) +
                                                         sfa_local_m * sfa_stride_m;
                        const uint32_t scales = is_valid ?
                            mxfp8_fp8_detail::load_e8m0_scale_quad(
                                sfa, sfa_base_offset, k_block_idx, sfa_stride_k,
                                sfa_gran_k, sfa_packed_int32, shape_k) :
                            0x7f7f7f7f;
                        ptx::st_shared(reinterpret_cast<float4*>(
                            smem_sfa[stage_idx] + m_offset * SHAPE_K_SFA_PER_STAGE), make_float4(
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales)),
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales >> 8)),
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales >> 16)),
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales >> 24))));
                    }

                    const uint32_t sfb_group_idx = kMasked ?
                        scheduler.current_group_idx :
                        static_cast<uint32_t>(cute::max(0, grouped_layout[m_block_idx * BLOCK_M]));
                    for (uint32_t n_offset = lane_idx; n_offset < BLOCK_N; n_offset += 32) {
                        const uint32_t n_idx = n_block_idx * BLOCK_N + n_offset;
                        const bool is_valid = n_idx < shape_n;
                        const uint32_t sfb_base_offset = sfb_group_idx * sfb_stride_group +
                                                         n_idx * sfb_stride_n;
                        const uint32_t scales = is_valid ?
                            mxfp8_fp8_detail::load_e8m0_scale_quad(
                                sfb, sfb_base_offset, k_block_idx, sfb_stride_k,
                                sfb_gran_k, sfb_packed_int32, shape_k) :
                            0x7f7f7f7f;
                        ptx::st_shared(reinterpret_cast<float4*>(
                            smem_sfb[stage_idx] + n_offset * SHAPE_K_SFB_PER_STAGE), make_float4(
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales)),
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales >> 8)),
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales >> 16)),
                            mxfp8_fp8_detail::e8m0_to_float(static_cast<uint8_t>(scales >> 24))));
                    }
                    __threadfence_block();
                    __syncwarp();

                    if (is_producer_leader) {
                        #pragma unroll
                        for (uint32_t wg = 0; wg < kNumMathWarpgroups; ++ wg)
                            scale_full_barriers[stage_idx * kNumMathWarpgroups + wg]->arrive();
                    }
                }
            }

            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            // MXFP8 promotes each 32-wide K chunk with its own UE8M0 scale, so unlike FP8 it
            // cannot hardware-accumulate the 4 chunks of a 128-K block into one accumulator:
            // every WGMMA must be drained before its FP32 promotion. To hide that drain we run a
            // software pipeline whose depth is bounded by the FP32 register budget. We reserve
            // ~168 of the 232 math registers for accumulators (the rest cover scales, descriptors
            // and indices) and pick the deepest form that fits at compile time:
            //   * kFullPipeline    - one independent accumulator per chunk; all WGMMAs are issued
            //                        back-to-back and drained once (mirrors FP8's full pipeline);
            //   * kTwoDeepPipeline - 2-deep ping-pong overlapping chunk kk's WGMMA with kk-1's
            //                        promotion (used when only two accumulators fit);
            //   * otherwise        - serial single-buffer fallback (no spill).
            // All pipelined paths require BLOCK_M >= WGMMA::M so `do_wgmma_store` is warpgroup-
            // uniform and the collective WGMMA is always issued.
            constexpr uint32_t kNumKChunks = BLOCK_K / WGMMA::K;
            constexpr uint32_t kPromoteWaves = BLOCK_M / WAVE_BLOCK_M;
            constexpr uint32_t kAccumRegBudget = 168;
            constexpr bool kFullPipeline = BLOCK_M >= WGMMA::M and
                (kNumKChunks + kPromoteWaves) * WGMMA::kNumAccum <= kAccumRegBudget;
            constexpr bool kTwoDeepPipeline = BLOCK_M >= WGMMA::M and not kFullPipeline and
                (2 + kPromoteWaves) * WGMMA::kNumAccum <= kAccumRegBudget;
            float final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for BLOCK_M < 64");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                    const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);
                    full_barriers[stage_idx]->wait(phase);

                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                        auto m_offset = local_idx * WAVE_BLOCK_M;

                        // The producer materializes the four per-32-K scales as FP32. Hoist the
                        // two A rows with LDS.128, but leave B scales in SMEM until promotion:
                        // preloading every B scale would consume ~96 FP32 registers at BLOCK_N=96
                        // and spill the 2-deep accumulator pipeline.
                        float4 sfa_pack_0 = {}, sfa_pack_1 = {};
                        auto load_scales = [&]() {
                            if (do_wgmma_store) {
                                sfa_pack_0 = ptx::ld_shared(reinterpret_cast<const float4*>(
                                    smem_sfa[stage_idx] + (r_0 + m_offset) * SHAPE_K_SFA_PER_STAGE));
                                sfa_pack_1 = ptx::ld_shared(reinterpret_cast<const float4*>(
                                    smem_sfa[stage_idx] + (r_1 + m_offset) * SHAPE_K_SFA_PER_STAGE));
                            }
                        };

                        auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;

                        // FP32 promotion of one drained 32-K chunk `kk` from accumulator `src`.
                        auto promote_chunk = [&](const float* src, const uint32_t& kk) {
                            const float* scale_a_0 = reinterpret_cast<const float*>(&sfa_pack_0);
                            const float* scale_a_1 = reinterpret_cast<const float*>(&sfa_pack_1);
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                const uint32_t n_scale_offset = i * 8 + (lane_idx % 4) * 2;
                                const float scale_b_0 = ptx::ld_shared(
                                    smem_sfb[stage_idx] + n_scale_offset * SHAPE_K_SFB_PER_STAGE + kk);
                                const float scale_b_1 = ptx::ld_shared(
                                    smem_sfb[stage_idx] + (n_scale_offset + 1) * SHAPE_K_SFB_PER_STAGE + kk);
                                const float scale_00 = scale_a_0[kk] * scale_b_0;
                                const float scale_01 = scale_a_0[kk] * scale_b_1;
                                const float scale_10 = scale_a_1[kk] * scale_b_0;
                                const float scale_11 = scale_a_1[kk] * scale_b_1;
                                shifted_accum[i * 4 + 0] += scale_00 * src[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += scale_01 * src[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += scale_10 * src[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += scale_11 * src[i * 4 + 3];
                            }
                        };

                        // Issue one 32-K chunk's collective WGMMA into accumulator `dst`.
                        auto issue_chunk = [&](float* dst, const uint32_t& kk) {
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(dst[i]);
                            ptx::warpgroup_arrive();
                            a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + kk * WGMMA::K) / 16;
                            b_desc.reg32_[0] = b_desc_base_lo + kk * WGMMA::K / 16;
                            WGMMA::wgmma(a_desc, b_desc, dst, false);
                            ptx::warpgroup_commit_batch();
                        };

                        // TMA data is ready after full_barrier, but FP32 scale staging is
                        // independent. Defer this wait until after chunk-0 has been issued so
                        // WGMMA can cover producer-side scale expansion.
                        bool scales_loaded = false;
                        auto ensure_scales_loaded = [&]() {
                            if (do_wgmma_store and not scales_loaded) {
                                scale_full_barriers[stage_idx * kNumMathWarpgroups + math_wg_idx]->wait(phase);
                                load_scales();
                                scales_loaded = true;
                            }
                        };

                        if constexpr (kFullPipeline) {
                            // Full pipeline: one dedicated accumulator per chunk. Issue every
                            // WGMMA back-to-back as a single warpgroup batch and drain once
                            // (`warpgroup_wait<0>`), matching FP8's fully-loaded tensor pipeline,
                            // then promote each chunk with its own scale. Fits only when
                            // kNumKChunks accumulators (+ the final_accum wave) stay under the
                            // register budget (BLOCK_N <= 64 => kNumAccum 32).
                            float accum_buf[kNumKChunks][WGMMA::kNumAccum];
                            #pragma unroll
                            for (uint32_t kk = 0; kk < kNumKChunks; ++ kk)
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum_buf[kk][i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t kk = 0; kk < kNumKChunks; ++ kk) {
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + kk * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + kk * WGMMA::K / 16;
                                WGMMA::wgmma(a_desc, b_desc, accum_buf[kk], false);
                            }
                            ptx::warpgroup_commit_batch();
                            ensure_scales_loaded();
                            #pragma unroll
                            for (uint32_t kk = 0; kk < kNumKChunks; ++ kk)
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum_buf[kk][i]);
                            ptx::warpgroup_wait<0>();
                            if (do_wgmma_store) {
                                #pragma unroll
                                for (uint32_t kk = 0; kk < kNumKChunks; ++ kk)
                                    promote_chunk(accum_buf[kk], kk);
                            }
                        } else if constexpr (kTwoDeepPipeline) {
                            // 2-deep software pipeline: overlap chunk (kk+1)'s WGMMA with chunk kk's
                            // FP32 promotion using a ping-pong accumulator. Used when only two
                            // accumulators fit under the register budget (BLOCK_N == 96).
                            float accum_buf[2][WGMMA::kNumAccum];

                            // Prologue: launch the first chunk, then in each step launch the next
                            // chunk before draining the current one. Defer scale loading until
                            // chunk-0 is in flight; issuing chunks 0+1 before the scale wait was
                            // measured to increase scoreboard pressure.
                            issue_chunk(accum_buf[0], 0);
                            ensure_scales_loaded();
                            #pragma unroll
                            for (uint32_t kk = 0; kk < kNumKChunks; ++ kk) {
                                const uint32_t cur = kk & 1;
                                if (kk + 1 < kNumKChunks)
                                    issue_chunk(accum_buf[(kk + 1) & 1], kk + 1);
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum_buf[cur][i]);
                                // Non-last chunks keep the freshly issued (kk+1) group in flight and
                                // only drain kk; the last chunk drains everything.
                                if (kk + 1 < kNumKChunks)
                                    ptx::warpgroup_wait<1>();
                                else
                                    ptx::warpgroup_wait<0>();
                                if (do_wgmma_store)
                                    promote_chunk(accum_buf[cur], kk);
                            }
                        } else {
                            float accum[WGMMA::kNumAccum];
                            #pragma unroll
                            for (uint32_t kk = 0; kk < kNumKChunks; ++ kk) {
                                issue_chunk(accum, kk);
                                ensure_scales_loaded();
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();

                                if (not do_wgmma_store)
                                    continue;

                                promote_chunk(accum, kk);
                            }
                        }

                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                            empty_barrier_arrive();
                    }
                }
            } else {
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            if (not do_wgmma_store)
                continue;

            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +
                            m_offset * kSwizzleDMode +
                            atom_offset * BLOCK_M * kSwizzleDMode +
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    } else {
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = n_block_idx * BLOCK_N + in_block_n_offset;
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
