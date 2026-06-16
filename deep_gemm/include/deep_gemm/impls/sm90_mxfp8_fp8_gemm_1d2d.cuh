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
sm90_mxfp8_fp8_gemm_1d2d_impl(uint8_t* sfa, uint8_t* sfb, int* grouped_layout,
                              uint32_t sfa_stride_m, uint32_t sfa_stride_k,
                              uint32_t sfb_stride_group, uint32_t sfb_stride_n, uint32_t sfb_stride_k,
                              uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(BLOCK_K == 128, "MXFP8 scale stage assumes 4 K/32 scale groups");
    DG_STATIC_ASSERT(kNumStages >= 1, "Invalid pipeline stages");

    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SHAPE_K_SFA_PER_STAGE = BLOCK_K / 32;
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * SHAPE_K_SFA_PER_STAGE * sizeof(uint8_t);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    static constexpr uint32_t SHAPE_K_SFB_PER_STAGE = BLOCK_K / 32;
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * SHAPE_K_SFB_PER_STAGE * sizeof(uint8_t);
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
        return reinterpret_cast<uint8_t*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE +
                                          i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_SF_OFFSET + kNumStages * (ALIGNED_SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
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
                    }

                    const uint32_t sfa_base_m = scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx);
                    for (uint32_t i = lane_idx; i < BLOCK_M * SHAPE_K_SFA_PER_STAGE; i += 32) {
                        const uint32_t m_offset = i / SHAPE_K_SFA_PER_STAGE;
                        const uint32_t k_scale_offset = i % SHAPE_K_SFA_PER_STAGE;
                        const uint32_t m_idx = sfa_base_m + m_offset;
                        const uint32_t k_scale_idx = k_block_idx * SHAPE_K_SFA_PER_STAGE + k_scale_offset;
                        const bool is_valid = m_idx < shape_m * (kMasked ? kNumGroups : 1) and
                                              k_scale_idx < math::ceil_div(shape_k, 32u);
                        smem_sfa[stage_idx][i] = is_valid ? sfa[m_idx * sfa_stride_m + k_scale_idx * sfa_stride_k] :
                                                            static_cast<uint8_t>(127);
                    }

                    for (uint32_t i = lane_idx; i < BLOCK_N * SHAPE_K_SFB_PER_STAGE; i += 32) {
                        const uint32_t n_offset = i / SHAPE_K_SFB_PER_STAGE;
                        const uint32_t k_scale_offset = i % SHAPE_K_SFB_PER_STAGE;
                        const uint32_t n_idx = n_block_idx * BLOCK_N + n_offset;
                        const uint32_t k_scale_idx = k_block_idx * SHAPE_K_SFB_PER_STAGE + k_scale_offset;
                        const bool is_valid = n_idx < shape_n and k_scale_idx < math::ceil_div(shape_k, 32u);
                        const uint32_t gmem_offset = scheduler.current_group_idx * sfb_stride_group +
                                                     n_idx * sfb_stride_n + k_scale_idx * sfb_stride_k;
                        smem_sfb[stage_idx][i] = is_valid ? sfb[gmem_offset] : static_cast<uint8_t>(127);
                    }
                    __threadfence_block();
                    __syncwarp();

                    if (is_producer_leader)
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
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
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

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
                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    const uint32_t sfa_base_m = scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx);
                    auto load_sfa = [&](uint32_t m_offset, uint32_t k_scale_offset) {
                        const uint32_t m_idx = sfa_base_m + m_offset;
                        const uint32_t k_scale_idx = k_block_idx * SHAPE_K_SFA_PER_STAGE + k_scale_offset;
                        const bool is_valid = m_idx < shape_m * (kMasked ? kNumGroups : 1) and
                                              k_scale_idx < math::ceil_div(shape_k, 32u);
                        return mxfp8_fp8_detail::e8m0_to_float(
                            is_valid ? sfa[m_idx * sfa_stride_m + k_scale_idx * sfa_stride_k] :
                                       static_cast<uint8_t>(127));
                    };
                    auto load_sfb = [&](uint32_t n_offset, uint32_t k_scale_offset) {
                        const uint32_t n_idx = n_block_idx * BLOCK_N + n_offset;
                        const uint32_t k_scale_idx = k_block_idx * SHAPE_K_SFB_PER_STAGE + k_scale_offset;
                        const bool is_valid = n_idx < shape_n and k_scale_idx < math::ceil_div(shape_k, 32u);
                        const uint32_t gmem_offset = scheduler.current_group_idx * sfb_stride_group +
                                                     n_idx * sfb_stride_n + k_scale_idx * sfb_stride_k;
                        return mxfp8_fp8_detail::e8m0_to_float(
                            is_valid ? sfb[gmem_offset] : static_cast<uint8_t>(127));
                    };

                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                        auto m_offset = local_idx * WAVE_BLOCK_M;

                        #pragma unroll
                        for (uint32_t kk = 0; kk < BLOCK_K / WGMMA::K; ++ kk) {
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + kk * WGMMA::K) / 16;
                            b_desc.reg32_[0] = b_desc_base_lo + kk * WGMMA::K / 16;
                            WGMMA::wgmma(a_desc, b_desc, accum, false);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            if (not do_wgmma_store)
                                continue;

                            const float scale_a_0 = load_sfa(r_0 + m_offset, kk);
                            const float scale_a_1 = load_sfa(r_1 + m_offset, kk);
                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                const uint32_t n_scale_offset = i * 8 + (lane_idx % 4) * 2;
                                const float scale_b_0 = load_sfb(n_scale_offset, kk);
                                const float scale_b_1 = load_sfb(n_scale_offset + 1, kk);
                                shifted_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
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
