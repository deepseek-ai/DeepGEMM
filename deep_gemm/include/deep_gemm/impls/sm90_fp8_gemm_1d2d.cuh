#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t,
          bool kIsW4 = false>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(
        math::constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or
        (math::constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    constexpr int WEIGHT_RATIO = kIsW4 ? 2 : 1;
    // W4 RS-mode swaps layout↔compute dimensions: WGMMA M=64 tiles along BLOCK_N, MMA_N=BLOCK_M
    constexpr uint32_t COMPUTE_M = kIsW4 ? BLOCK_N : BLOCK_M;
    constexpr uint32_t COMPUTE_N = kIsW4 ? BLOCK_M : BLOCK_N;
    constexpr int MMA_N = kIsW4 ? (SHAPE_M != 0 ? SHAPE_M : COMPUTE_N) : COMPUTE_N;
    using WGMMA = typename mma::sm90::FP8MMASelector<MMA_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    if constexpr (kIsW4) {
        DG_STATIC_ASSERT(BLOCK_M <= 256, "keep BLOCK_M <= 256, if use w4afp8.");
        DG_STATIC_ASSERT(BLOCK_N % WGMMA::M == 0, "keep BLOCK_N % WGMMA::M == 0, if use w4afp8.");
    } else {
        DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = kIsW4 ? true : (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3) / WEIGHT_RATIO;
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = COMPUTE_M * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = math::ceil_div(shape_n, BLOCK_K);
    const uint32_t smem_sfb_size = kIsW4 ? 0 : math::align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier));

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
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
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    auto full_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA A
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                             num_tma_multicast_a, batch_idx);
                    // W4: SFA slot carries weight scales (block_n dim, indexed by n_block_idx)
                    tma::copy<COMPUTE_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                             smem_sfa[stage_idx], kIsW4 ? (n_block_idx * BLOCK_N) : (m_block_idx * BLOCK_M),
                             scheduler.template get_global_idx<kIsW4 or kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales,
                                                                                  1,
                                                                                  k_block_idx,
                                                                                  (kGemmType == GemmType::MGroupedContiguous) ? m_block_idx : 0),
                             kIsW4 ? num_tma_multicast_b : num_tma_multicast_a);

                    // Issue TMA B
                    tma::copy<BLOCK_K / WEIGHT_RATIO, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                             smem_b[stage_idx], k_idx / WEIGHT_RATIO, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b, batch_idx);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);

        const auto r_0 = warp_idx * 16 + lane_idx / 4;
        const auto r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + (kIsW4 ? 0 : math_wg_idx * WGMMA::M * BLOCK_K), 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0] + (kIsW4 ? math_wg_idx * WGMMA::M * BLOCK_K : 0), 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // W4: Precompute thread-invariant values for ldmatrix (hoisted out of k-loop)
        constexpr uint32_t NV = 16 / sizeof(__nv_fp8_e4m3);
        const uint32_t tidG = threadIdx.x % 128;
        const uint32_t tRow = (tidG & 15) | ((tidG >> 5) << 4);
        const uint32_t tCol = ((tidG >> 4) & 1) * NV;
        const uint32_t sRow = math_wg_idx * WGMMA::M + tRow;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

            // W4: sfb is mounted on sfa (per-tensor scale), skip global SFB loading
            if constexpr (not kIsW4) {
                // Load B scales with math warp-groups
                // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
                if (threadIdx.x >= 32) {
                    auto previous_group_offset = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                    const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                    const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
                    auto local_sfb = sfb + previous_group_offset + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;

                    #pragma unroll
                    for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                        ptx::st_shared(smem_sfb + i, i < shape_k_scales ? local_sfb[i * stride_k_sfb] : local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
            }

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = COMPUTE_M <= WGMMA::M
                                              ? COMPUTE_M
                                              : WGMMA::M * 2;
            DG_STATIC_ASSERT(COMPUTE_M % WAVE_BLOCK_M == 0, "Invalid block sizes");

            constexpr int WAVE_WGMMA = COMPUTE_M / WAVE_BLOCK_M;
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * WAVE_WGMMA] = {0};

            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(COMPUTE_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // Skip useless computations
            if (scheduler.is_computation_valid(m_block_idx, kIsW4 ? 0 : math_wg_idx * WGMMA::M)) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / math::constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = math::constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // Dispatch `num_former_iters` and launch MMAs
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        float scale_b_0;
                        float scale_b_1;

                        if constexpr (not kIsW4) {
                            // Read B scales
                            scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx);
                            // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                            if constexpr (not kMustUseUniformedScaleB)
                                scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);
                        }

                        // Wait TMA arrivals
                        full_barriers[stage_idx]->wait(phase);

                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < WAVE_WGMMA; ++ local_idx) {
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // Read A scales (or weight scales for W4)
                            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                            float scale_a_0;
                            float scale_a_1;

                            if constexpr (kIsW4) {
                                scale_b_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset);
                                scale_b_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1 + m_offset);
                            } else {
                                scale_a_0 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
                                scale_a_1 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_1 + m_offset) : 0;
                            }

                            // Commit WGMMA instructions
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();

                            if constexpr (kIsW4) {
                                using WGMMARS = typename mma::sm90::FP8MMASelectorRS<MMA_N>::type;

                                uint32_t fragB[std::min(BLOCK_K / WGMMA::K, 2U)][4];
                                uint32_t unpackB[std::min(BLOCK_K / WGMMA::K, 2U)][8];

                                const __nv_fp8_e4m3* smem_b_w4_ptr = smem_b[stage_idx] + (sRow + m_offset) * (BLOCK_K / 2);

                                // Prologue: ldmatrix + dequant for k=0
                                const uint32_t pCol = ptx::permute_col<BLOCK_K / 2 * sizeof(__nv_fp8_e4m3), NV>(tRow, 0 * WGMMA::K / 2 + tCol);
                                ptx::SM90_U32x4_LDSM_N::copy(fragB[0][0], fragB[0][1], fragB[0][2], fragB[0][3], (void*)(smem_b_w4_ptr + pCol));

                                ptx::fast_int4_to_fp8_convert(unpackB[0] + 0, fragB[0][0]);
                                ptx::fast_int4_to_fp8_convert(unpackB[0] + 2, fragB[0][1]);
                                ptx::fast_int4_to_fp8_convert(unpackB[0] + 4, fragB[0][2]);
                                ptx::fast_int4_to_fp8_convert(unpackB[0] + 6, fragB[0][3]);

                                DG_STATIC_ASSERT((BLOCK_K / WGMMA::K) % 2 == 0, "Invalid unroll for w4 dequant.");

                                #pragma unroll
                                for (uint32_t k = 2; k < BLOCK_K / WGMMA::K; k += 2) {
                                    const uint32_t pCol = ptx::permute_col<BLOCK_K / 2 * sizeof(__nv_fp8_e4m3), NV>(tRow, k * WGMMA::K / 2 + tCol);
                                    ptx::SM90_U32x4_LDSM_N::copy(fragB[(k / 2) % 2][0], fragB[(k / 2) % 2][1], fragB[(k / 2) % 2][2], fragB[(k / 2) % 2][3], (void*)(smem_b_w4_ptr + pCol));

                                    ptx::fast_int4_to_fp8_convert(unpackB[(k / 2) % 2] + 0, fragB[(k / 2) % 2][0]);
                                    ptx::fast_int4_to_fp8_convert(unpackB[(k / 2) % 2] + 2, fragB[(k / 2) % 2][1]);
                                    ptx::fast_int4_to_fp8_convert(unpackB[(k / 2) % 2] + 4, fragB[(k / 2) % 2][2]);
                                    ptx::fast_int4_to_fp8_convert(unpackB[(k / 2) % 2] + 6, fragB[(k / 2) % 2][3]);

                                    #pragma unroll
                                    for (int j = 0; j < 2; ++j) {
                                        a_desc.reg32_[0] = a_desc_base_lo + (k - 2 + j) * WGMMA::K / 16;
                                        WGMMARS::wgmma(unpackB[((k - 2) / 2) % 2] + j * 4, a_desc, accum, k - 2 + j);
                                    }
                                }

                                // Epilogue: last batch of wgmma
                                #pragma unroll
                                for (int j = 0; j < 2; ++j) {
                                    a_desc.reg32_[0] = a_desc_base_lo + (BLOCK_K / WGMMA::K - 2 + j) * WGMMA::K / 16;
                                    WGMMARS::wgmma(unpackB[std::max(BLOCK_K / WGMMA::K - 1, 1U) % 2] + j * 4, a_desc, accum, BLOCK_K / WGMMA::K - 2 + j);
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                    a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                    b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                    WGMMA::wgmma(a_desc, b_desc, accum, k);
                                }
                            }

                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // Notify barrier arrival at the last warpgroup wave
                            if (local_idx == WAVE_WGMMA - 1)
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store)
                                continue;

                            // Promote with scales
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            if constexpr (kIsW4) {
                                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;

                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
                                    shifted_accum[i * 4 + 0] += scale_b_0 * accum[i * 4 + 0];
                                    shifted_accum[i * 4 + 1] += scale_b_0 * accum[i * 4 + 1];
                                    shifted_accum[i * 4 + 2] += scale_b_1 * accum[i * 4 + 2];
                                    shifted_accum[i * 4 + 3] += scale_b_1 * accum[i * 4 + 3];
                                }
                            } else {
                                float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                                float scale_0_1, scale_1_1;
                                if constexpr (not kMustUseUniformedScaleB)
                                    scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                                #pragma unroll
                                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                    // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                                    const bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                                    shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                    shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                    shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                    shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                                }
                            }
                        }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < WAVE_WGMMA; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        constexpr uint32_t kNumBankGroupBytes = 16;

                        if constexpr (kIsW4) {
                            auto row = i * 8 + lane_idx % 8;
                            auto col = (warp_idx % 4) * 2 + lane_idx / 8;
                            col ^= row % (kSwizzleDMode / 16);

                            auto n_atom_idx = m_offset / WGMMA::M + math_wg_idx;
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                                    n_atom_idx * BLOCK_M * kSwizzleDMode +
                                    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                        } else {
                            // Calculate the swizzling atom offset and in-atom offset
                            auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                            // Calculate the index of the bank group to be written in the atom
                            auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                            // Reshape the atom in another view and swizzle
                            //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                            //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                            constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                            col ^= row % (kSwizzleDMode / 16);

                            // Add back into the base pointer
                            // NOTES: think twice before modifying this, as changes may affect the number of instructions
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                                warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                                m_offset * kSwizzleDMode +                                 // Wave offset
                                atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                        }
                    } else {
                        if constexpr (kIsW4) {
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d +
                                                                  m_offset +
                                                                  warp_idx * WGMMA_M_PER_WARP +
                                                                  lane_idx / 8 * 8 + (lane_idx % 8) * BLOCK_N +
                                                                  BLOCK_N * i * 8);
                        } else {
                            // No swizzling, just padding
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                        }
                    }

                    if constexpr (kIsW4) {
                        shifted_accum[i * 4 + 0] *= sfb[0];
                        shifted_accum[i * 4 + 1] *= sfb[0];
                        shifted_accum[i * 4 + 2] *= sfb[0];
                        shifted_accum[i * 4 + 3] *= sfb[0];
                    }

                    // NOTES: only 16 lanes' addresses are used
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::template copy<kIsW4>(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                  n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
