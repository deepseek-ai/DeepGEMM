#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>

#include <cute/int_tuple.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm120_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t = epilogue::transform::EpilogueIdentity,
          bool kIsFP4 = false>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm120_fp8_fp4_gemm_1d1d_impl(cd_dtype_t* gmem_d, const cd_dtype_t* gmem_c,
                             __nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                             int* grouped_layout,
                             cute::TmaDescriptor* tensor_map_buffer,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             uint32_t stride_d,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    namespace sm120_mma = mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = 16;
    static constexpr uint32_t MMA_N = 8;
    static constexpr uint32_t MMA_K = kIsFP4 ? sm120_mma::FP4_MMA_K : sm120_mma::FP8_MMA_K;
    static constexpr uint32_t MMA_ACCUM = 4;

    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Only float or bfloat16 output supported");
    DG_STATIC_ASSERT(kNumTMAThreads > 0, "SM120a always uses warp-specialized pipeline");
    DG_STATIC_ASSERT(kNumMathThreads % 32 == 0, "Invalid math threads");
    DG_STATIC_ASSERT(BLOCK_M % MMA_M == 0 and BLOCK_N % MMA_N == 0 and BLOCK_K % MMA_K == 0, "Invalid block dims");

    static constexpr uint32_t kNumSFAStagesPerLoad = (4 * kGranKA) / BLOCK_K;
    static constexpr uint32_t kNumSFBStagesPerLoad = (4 * kGranKB) / BLOCK_K;

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kMTilesPerWarp = BLOCK_M / kNumMathWarps / MMA_M;
    static constexpr uint32_t kNTiles = BLOCK_N / MMA_N;
    static constexpr uint32_t kKSteps = BLOCK_K / MMA_K;
    static constexpr uint32_t kAccumPerWarp = kMTilesPerWarp * kNTiles * MMA_ACCUM;

    DG_STATIC_ASSERT(BLOCK_M == kNumMathWarps * kMTilesPerWarp * MMA_M, "M tiles must divide evenly");
    DG_STATIC_ASSERT(kNTiles % 2 == 0, "kNTiles must be even for ldmatrix.x2 B loading");

    static constexpr uint32_t kTMARegisters = 40;
    static constexpr uint32_t kMMARegisters = 232;

    // SMEM D buffer for TMA store epilogue
    static constexpr bool kUseTMAStoreEpilogue = sizeof(cd_dtype_t) <= 2 and kSwizzleCDMode > 0
        and BLOCK_N * sizeof(cd_dtype_t) >= kSwizzleCDMode
        and (BLOCK_N * sizeof(cd_dtype_t)) % kSwizzleCDMode == 0;
    static constexpr uint32_t SMEM_D = kUseTMAStoreEpilogue
        ? static_cast<uint32_t>((BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode) * kSwizzleCDMode * BLOCK_M)
        : 0u;
    static constexpr uint32_t kSwizzleCDShift = kSwizzleCDMode > 0 ? (7 - __builtin_ctz(kSwizzleCDMode)) : 0;
    static constexpr uint32_t kSwizzleCDMask = kSwizzleCDMode > 0 ? (kSwizzleCDMode / 16 - 1) : 0;
    static constexpr uint32_t kTMAStoreInnerDim = kSwizzleCDMode / sizeof(cd_dtype_t);
    static constexpr uint32_t kNumTMAStores = kUseTMAStoreEpilogue
        ? BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode : 0;

    static constexpr uint32_t SMEM_TM = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    // FP4 uses packed SMEM (4-bit per element = 0.5 bytes), FP8 uses 1 byte per element.
    static constexpr uint32_t kSMEMKBytes = kIsFP4 ? (BLOCK_K / 2) : BLOCK_K;
    static constexpr uint32_t SMEM_A  = BLOCK_M * kSMEMKBytes;
    static constexpr uint32_t SMEM_B  = BLOCK_N * kSMEMKBytes;
    static constexpr uint32_t SMEM_SFA = math::constexpr_align(static_cast<uint32_t>(BLOCK_M * sizeof(int32_t)), 128u);
    static constexpr uint32_t SMEM_SFB = math::constexpr_align(static_cast<uint32_t>(BLOCK_N * sizeof(int32_t)), 128u);
    static constexpr uint32_t TMA_SFA_BYTES = BLOCK_M * sizeof(int32_t);
    static constexpr uint32_t TMA_SFB_BYTES = BLOCK_N * sizeof(int32_t);
    static constexpr uint32_t SMEM_TMA_BYTES = SMEM_A + SMEM_B + TMA_SFA_BYTES + TMA_SFB_BYTES;
    // ldmatrix K stride in bytes: FP4 packed = MMA_K/2, FP8 = MMA_K. Both = 32 bytes.
    static constexpr uint32_t kLdmK = kIsFP4 ? (MMA_K / 2) : MMA_K;
    // tma::copy swizzle for split computation: FP4 packed with B64 has 64 byte rows = full BLOCK_K,
    // so one TMA copy covers the entire tile. Use 0 to get single-copy path.
    static constexpr uint32_t kTMACopySwizzleA = kIsFP4 ? 0u : kSwizzleAMode;
    static constexpr uint32_t kTMACopySwizzleB = kIsFP4 ? 0u : kSwizzleBMode;

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // SMEM layout: pipeline data first (1024-aligned for B128 swizzle),
    // tensor map descriptors at the end (K-grouped only)
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_d_base = reinterpret_cast<cd_dtype_t*>(smem_buffer);

    constexpr uint32_t PIPE_BASE = SMEM_D;
    auto smem_a = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + s * SMEM_A);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + kNumStages * SMEM_A + s * SMEM_B);
    });
    constexpr uint32_t SF_BASE = PIPE_BASE + kNumStages * (SMEM_A + SMEM_B);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SF_BASE + s * SMEM_SFA);
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SF_BASE + kNumStages * SMEM_SFA + s * SMEM_SFB);
    });
    constexpr uint32_t BAR_BASE = SF_BASE + kNumStages * (SMEM_SFA + SMEM_SFB);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + s * sizeof(Barrier));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + (kNumStages + s) * sizeof(Barrier));
    });

    // Tensor map descriptors at the end of SMEM (K-grouped only)
    constexpr uint32_t TM_BASE = BAR_BASE + 2 * kNumStages * sizeof(Barrier);
    auto smem_tm_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + TM_BASE);
    auto smem_tm_b = smem_tm_a + 1;
    auto gmem_tm_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tm_b = gmem_tm_a + 1;

    // Prefetch TMA descriptors
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Barrier init (done by warp 1 before producer/consumer split)
    if (warp_idx == 1 and cute::elect_one_sync()) {
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tm_a = tensor_map_a_base;
            *smem_tm_b = tensor_map_b_base;
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    cudaGridDependencySynchronize();

    // Persistent scheduler
    uint32_t m_block_idx, n_block_idx;
    static constexpr uint32_t kSFKAlignment = kGranKA * 4;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs, kSFKAlignment>(
        shape_m, shape_n, shape_k, grouped_layout);
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1};
    };

    // =====================================================================
    // PRODUCER WARP GROUP (TMA warps, 40 regs)
    // =====================================================================
    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kTMARegisters>();

        const bool is_tma_leader = (warp_idx == kNumMathWarps and lane_idx == 0);
        uint32_t tma_iter_idx = 0;

        if (is_tma_leader) {
            uint32_t last_group_idx = kNumGroups;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                if constexpr (kGemmType == GemmType::KGroupedContiguous) {
                    if (last_group_idx != scheduler.current_group_idx) {
                        last_group_idx = scheduler.current_group_idx;

                        const auto a_base = reinterpret_cast<const char*>(gmem_a_ptr);
                        const auto b_base = reinterpret_cast<const char*>(gmem_b_ptr);
                        const uint64_t a_offset = kIsFP4
                            ? (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_m / 2)
                            : (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_m);
                        const uint64_t b_offset = kIsFP4
                            ? (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_n / 2)
                            : (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_n);

                        ptx::tensor_map_replace_global_addr_in_smem(smem_tm_a, a_base + a_offset);
                        ptx::tensor_map_replace_global_addr_in_smem(smem_tm_b, b_base + b_offset);

                        const uint64_t new_stride = kIsFP4
                            ? static_cast<uint64_t>(scheduler.current_shape_k / 2)
                            : static_cast<uint64_t>(scheduler.current_shape_k);
                        ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                            smem_tm_a, scheduler.current_shape_k, new_stride);
                        ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                            smem_tm_b, scheduler.current_shape_k, new_stride);

                        *gmem_tm_a = *smem_tm_a;
                        *gmem_tm_b = *smem_tm_b;
                        ptx::tensor_map_release_gpu();
                        ptx::tensor_map_acquire_gpu(gmem_tm_a);
                        ptx::tensor_map_acquire_gpu(gmem_tm_b);
                    }
                }

                const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
                const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
                constexpr bool kAGroupOffset = (kGemmType == GemmType::MGroupedMasked);
                const uint32_t m_idx = scheduler.template get_global_idx<kAGroupOffset>(shape_m, BLOCK_M, m_block_idx);
                constexpr bool kBGroupOffset = not (kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous);
                const uint32_t n_idx = scheduler.template get_global_idx<kBGroupOffset>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                const auto tma_a_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_a : &tensor_map_a_base);
                const auto tma_b_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_b : &tensor_map_b_base);

                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = kIsBatchedMM ? scheduler.current_group_idx : 0;

                for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                    CUTE_TIE_DECL(get_pipeline(tma_iter_idx++), s, p);
                    empty_barriers[s]->wait(p ^ 1);

                    const uint32_t k_idx = kb * BLOCK_K;
                    uint32_t sfa_k, sfb_k;
                    if constexpr (kGemmType == GemmType::KGroupedContiguous) {
                        sfa_k = scheduler.current_sf_k_cumsum + kb / kNumSFAStagesPerLoad;
                        sfb_k = scheduler.current_sf_k_cumsum + kb / kNumSFBStagesPerLoad;
                    } else {
                        const uint32_t shape_sfa_k = math::ceil_div(shape_k, BLOCK_K * kNumSFAStagesPerLoad);
                        const uint32_t shape_sfb_k = math::ceil_div(shape_k, BLOCK_K * kNumSFBStagesPerLoad);
                        constexpr bool kSFAGroupOffset = not is_m_grouped_contiguous(kGemmType);
                        sfa_k = scheduler.template get_global_idx<kSFAGroupOffset, sched::IndexType::SF_K>(
                            shape_sfa_k, 1, kb / kNumSFAStagesPerLoad, m_block_idx);
                        constexpr bool kSFBGroupOffset = not (kGemmType == GemmType::Normal);
                        sfb_k = scheduler.template get_global_idx<kSFBGroupOffset, sched::IndexType::SF_K>(
                            shape_sfb_k, 1, kb / kNumSFBStagesPerLoad, m_block_idx);
                    }
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M, sfa_k, 1);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N, sfb_k, 1);
                    tma::copy<BLOCK_K, BLOCK_M, kTMACopySwizzleA, char, kIsBatchedMM>(tma_a_desc, full_barriers[s], smem_a[s], k_idx, m_idx, 1, batch_idx);
                    tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    full_barriers[s]->arrive_and_expect_tx(SMEM_TMA_BYTES);
                }
            }
        }
    }
    // =====================================================================
    // CONSUMER WARP GROUPS (math warps, 232 regs)
    // =====================================================================
    else {
        cutlass::arch::warpgroup_reg_alloc<kMMARegisters>();

        const uint32_t math_warp_idx = warp_idx;
        const uint32_t group_id = lane_idx / 4;
        const uint32_t thread_id = lane_idx % 4;
        const uint32_t m_tile_base = math_warp_idx * kMTilesPerWarp;

        float accum[kAccumPerWarp];
        uint32_t iter_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerWarp; ++i) accum[i] = 0.f;

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                // Wait for TMA data
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);

                full_barriers[stage]->wait(phase);

                // Pre-compute swizzle contexts (row stride = kSMEMKBytes for packed FP4)
                sm120::SwizzleContext<kSwizzleAMode> a_ctx[kMTilesPerWarp];
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    int a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + (m_tile_base + mt) * 16;
                    a_ctx[mt].init(a_row, kSMEMKBytes);
                }
                sm120::SwizzleContext<kSwizzleBMode> b_ctx[kNTiles];
                #pragma unroll
                for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                    int b_row = (lane_idx & 7) + nt * 8;
                    b_ctx[nt].init(b_row, kSMEMKBytes);
                }

                const uint32_t sf_byte_a_base = (kb * BLOCK_K / kGranKA) % 4;
                const uint32_t sf_byte_b_base = (kb * BLOCK_K / kGranKB) % 4;

                // Ping-pong fragment buffers for K-step double-buffer
                uint32_t a_frag[2][kMTilesPerWarp][4];
                uint32_t b_tile[2][kNTiles][2];

                // SF type: FP4 scale_vec::2X uses uint16_t (2 SF bytes per MMA step),
                // FP8 uses uint8_t (1 SF byte per MMA step).
                using sf_t = cute::conditional_t<kIsFP4, uint16_t, uint8_t>;
                sf_t sfa_bytes[2][kMTilesPerWarp];
                sf_t sfb_bytes[2][kNTiles];

                // SF hoist: when gran_k >= BLOCK_K, SF is constant across K-steps
                sf_t sfa_hoisted[kMTilesPerWarp];
                sf_t sfb_hoisted[kNTiles];
                if constexpr (kGranKB >= BLOCK_K) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                        auto packed = sm120::load_sf(smem_sfb[stage], nt * MMA_N + group_id);
                        if constexpr (kIsFP4) {
                            uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                        } else {
                            sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                        }
                    }
                }
                if constexpr (kGranKA >= BLOCK_K) {
                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        auto packed = sm120::load_sf(smem_sfa[stage],
                            (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                        if constexpr (kIsFP4) {
                            uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                        } else {
                            sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                        }
                    }
                }

                auto load_kstep = [&](int buf, uint32_t ks) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTiles; ++nt)
                        sm120::load_b_fragment_x2(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                        sm120::load_a_fragment(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);

                    if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                        const uint32_t sf_step = (kb * kKSteps + ks);
                        if constexpr (kIsFP4) {
                            // FP4 scale_vec::2X: MMA_K=64 covers 2 groups of 32 elements
                            // kGranK<=32 → each group gets its own SF → extract_sf_pair
                            // kGranK>32  → both groups share one SF → duplicate byte
                            const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                            const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                                auto packed = sm120::load_sf(smem_sfb[stage], nt * MMA_N + group_id);
                                if constexpr (kGranKB <= 32) {
                                    sfb_bytes[buf][nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                } else {
                                    uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                    sfb_bytes[buf][nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                }
                            }
                            #pragma unroll
                            for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                auto packed = sm120::load_sf(smem_sfa[stage],
                                    (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                if constexpr (kGranKA <= 32) {
                                    sfa_bytes[buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                } else {
                                    uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                    sfa_bytes[buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                }
                            }
                        } else {
                            const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                            const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTiles; ++nt)
                                sfb_bytes[buf][nt] = sm120_mma::extract_sf_byte(
                                    sm120::load_sf(smem_sfb[stage], nt * MMA_N + group_id), sf_byte_b);
                            #pragma unroll
                            for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                sfa_bytes[buf][mt] = sm120_mma::extract_sf_byte(
                                    sm120::load_sf(smem_sfa[stage],
                                        (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8),
                                    sf_byte_a);
                        }
                    }
                };

                auto compute_kstep = [&](int buf) {
                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        const sf_t sfa = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_bytes[buf][mt];
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                            float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTiles + nt) * MMA_ACCUM]);
                            const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_bytes[buf][nt];
                            if constexpr (kIsFP4)
                                sm120_mma::fp4_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                            else
                                sm120_mma::fp8_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                        }
                    }
                };

                // Pre-load first K-step
                load_kstep(0, 0);

                // K-step loop with double-buffer
                #pragma unroll
                for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                    int cur = ks & 1;
                    int nxt = (ks + 1) & 1;
                    if (ks < kKSteps - 1) {
                        load_kstep(nxt, ks + 1);
                    }
                    compute_kstep(cur);
                }

                // Release stage
                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }

            // ======== EPILOGUE ========
            constexpr bool kEpilogueGroupOffset = not is_m_grouped_contiguous(kGemmType);
            const uint32_t m_base = scheduler.template get_global_idx<kEpilogueGroupOffset>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t n_base = n_block_idx * BLOCK_N;
            const uint32_t total_shape_m = (kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::MGroupedMasked)
                ? shape_m * kNumGroups : shape_m;

            auto read_cd = [&](const cd_dtype_t& x) -> float {
                if constexpr (cute::is_same_v<cd_dtype_t, float>) return x;
                else return static_cast<float>(x);
            };

            constexpr bool kIsBatchedEpilogue = (kGemmType == GemmType::Batched);
            // Batched D is [M, batch, N] physical layout: stride_m = kNumGroups * stride_d
            const int64_t cd_m_stride = kIsBatchedEpilogue
                ? static_cast<int64_t>(kNumGroups) * stride_d : static_cast<int64_t>(stride_d);
            const int64_t cd_batch_offset = kIsBatchedEpilogue
                ? static_cast<int64_t>(scheduler.current_group_idx) * stride_d : 0;

            if constexpr (kUseTMAStoreEpilogue) {
                if (math_warp_idx == 0 and lane_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                        const uint32_t ai = (mt * kNTiles + nt) * MMA_ACCUM;
                        const uint32_t local_row0 = (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t local_row1 = local_row0 + 8;
                        const uint32_t local_col  = nt * MMA_N + thread_id * 2;
                        float v0 = accum[ai + 0], v1 = accum[ai + 1];
                        float v2 = accum[ai + 2], v3 = accum[ai + 3];

                        if constexpr (kWithAccumulation) {
                            const uint32_t gr0 = m_base + local_row0, gr1 = m_base + local_row1;
                            const uint32_t gc = epilogue_type_t::template apply_index_n<MMA_N>(
                                n_base + nt * MMA_N) + thread_id * 2;
                            if (gr0 < total_shape_m and gc + 1 < stride_d) {
                                const auto ci = cd_batch_offset + static_cast<int64_t>(gr0) * cd_m_stride + gc;
                                v0 += read_cd(gmem_c[ci]); v1 += read_cd(gmem_c[ci + 1]);
                            }
                            if (gr1 < total_shape_m and gc + 1 < stride_d) {
                                const auto ci = cd_batch_offset + static_cast<int64_t>(gr1) * cd_m_stride + gc;
                                v2 += read_cd(gmem_c[ci]); v3 += read_cd(gmem_c[ci + 1]);
                            }
                        }

                        const uint32_t sub_tile = local_col / kTMAStoreInnerDim;
                        const uint32_t col_in_sub = local_col % kTMAStoreInnerDim;
                        const uint32_t col_byte_in_sub = col_in_sub * sizeof(cd_dtype_t);
                        const uint32_t sw0 = col_byte_in_sub ^ (((local_row0 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                        const uint32_t sw1 = col_byte_in_sub ^ (((local_row1 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                        cd_dtype_t p0[2] = {cd_dtype_t(v0), cd_dtype_t(v1)};
                        cd_dtype_t p1[2] = {cd_dtype_t(v2), cd_dtype_t(v3)};
                        auto* smem_d_bytes = reinterpret_cast<char*>(smem_d_base);
                        const uint32_t sub_base = sub_tile * kSwizzleCDMode * BLOCK_M;
                        *reinterpret_cast<uint32_t*>(smem_d_bytes + sub_base + local_row0 * kSwizzleCDMode + sw0) =
                            *reinterpret_cast<const uint32_t*>(p0);
                        *reinterpret_cast<uint32_t*>(smem_d_bytes + sub_base + local_row1 * kSwizzleCDMode + sw1) =
                            *reinterpret_cast<const uint32_t*>(p1);
                    }
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                if (math_warp_idx == 0 and lane_idx == 0) {
                    const uint32_t batch_store_idx = kIsBatchedEpilogue ? scheduler.current_group_idx : 0;
                    #pragma unroll
                    for (uint32_t ts = 0; ts < kNumTMAStores; ++ts) {
                        auto* smem_src = reinterpret_cast<char*>(smem_d_base) + ts * kSwizzleCDMode * BLOCK_M;
                        const uint32_t n_store = epilogue_type_t::template apply_index_n<kTMAStoreInnerDim>(
                            n_base + ts * kTMAStoreInnerDim);
                        if constexpr (kIsBatchedEpilogue) {
                            if constexpr (kWithAccumulation)
                                cute::SM90_TMA_REDUCE_ADD_3D::copy(
                                    &tensor_map_cd, smem_src,
                                    n_store, m_base, batch_store_idx);
                            else
                                cute::SM90_TMA_STORE_3D::copy(
                                    &tensor_map_cd, smem_src,
                                    n_store, m_base, batch_store_idx);
                        } else {
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_cd, smem_src,
                                n_store, m_base);
                        }
                    }
                    cute::tma_store_arrive();
                }
            } else {
                auto store_pair = [&](cd_dtype_t* ptr, float a, float b) {
                    if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                        *reinterpret_cast<float2*>(ptr) = make_float2(a, b);
                    } else {
                        ptr[0] = cd_dtype_t(a);
                        ptr[1] = cd_dtype_t(b);
                    }
                };

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                        const uint32_t ai = (mt * kNTiles + nt) * MMA_ACCUM;
                        const uint32_t n_tile_base = n_base + nt * MMA_N;
                        const uint32_t col = epilogue_type_t::template apply_index_n<MMA_N>(n_tile_base) + thread_id * 2;
                        const uint32_t row0 = m_base + (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t row1 = row0 + 8;

                        if (row0 < total_shape_m and col + 1 < stride_d) {
                            auto idx = cd_batch_offset + static_cast<int64_t>(row0) * cd_m_stride + col;
                            float v0 = accum[ai + 0], v1 = accum[ai + 1];
                            if constexpr (kWithAccumulation) { v0 += read_cd(gmem_c[idx]); v1 += read_cd(gmem_c[idx + 1]); }
                            store_pair(&gmem_d[idx], v0, v1);
                        }
                        if (row1 < total_shape_m and col + 1 < stride_d) {
                            auto idx = cd_batch_offset + static_cast<int64_t>(row1) * cd_m_stride + col;
                            float v2 = accum[ai + 2], v3 = accum[ai + 3];
                            if constexpr (kWithAccumulation) { v2 += read_cd(gmem_c[idx]); v3 += read_cd(gmem_c[idx + 1]); }
                            store_pair(&gmem_d[idx], v2, v3);
                        }
                    }
                }
            }
        } // persistent loop

        // Final TMA store drain
        if constexpr (kUseTMAStoreEpilogue) {
            if (math_warp_idx == 0 and lane_idx == 0)
                cute::tma_store_wait<0>();
        }
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
