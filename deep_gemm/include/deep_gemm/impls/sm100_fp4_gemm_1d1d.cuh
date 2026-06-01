#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/float8.h>

#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm {

using namespace deep_gemm::mma::sm100;
using namespace deep_gemm::math;
using namespace deep_gemm::ptx;
using namespace deep_gemm::utils;

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp4_gemm_1d1d_impl(int* grouped_layout,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // MMA configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t kNumMWaves = BLOCK_M / LAYOUT_AD_M;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // MXFP4 / SF configs
    constexpr uint32_t kNumSFAStagesPerLoad = 1;
    constexpr uint32_t kNumSFBStagesPerLoad = 1;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t FP4_ELEMS_PER_INT32 = 8;
    constexpr uint32_t MXF4_VS = 32;
    constexpr uint32_t BLOCK_K_FP4 = BLOCK_K * FP4_ELEMS_PER_INT32;
    constexpr uint32_t UMMA_K_FP4 = 64;
    constexpr uint32_t SF_K_PER_STAGE = BLOCK_K_FP4 / MXF4_VS;
    constexpr uint32_t SF_PACKED_K_PER_STAGE = SF_K_PER_STAGE / 4;

    DG_STATIC_ASSERT(BLOCK_M % LAYOUT_AD_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_K == 16 or BLOCK_K == 32, "FP4 BLOCK_K must be 16 or 32 int32");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const uint32_t total_scales_k = ceil_div(shape_k * FP4_ELEMS_PER_INT32, MXF4_VS);
    const uint32_t total_packed_k = ceil_div(total_scales_k, uint32_t(4));
    const uint32_t shape_sfa_k = total_packed_k;
    const uint32_t shape_sfb_k = total_packed_k;

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Load/store block sizes
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    // Swap-AB: STORE_BLOCK_M=16 (fine-grained M slices to skip padding rows), STORE_BLOCK_N=BLOCK_N.
    // Non-swap: STORE_BLOCK_M=BLOCK_M, STORE_BLOCK_N derived from swizzle.
    constexpr uint32_t STORE_BLOCK_M = kSwapAB ? 16u : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);

    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "FP4 only supports B-multicast (2CTA along M)");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M and BLOCK_M % LAYOUT_AD_M == 0, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    // Swap-AB requires BLOCK_N = LAYOUT_AD_M (= 128) so UMMA_M after swap stays = 128.
    DG_STATIC_ASSERT(not kSwapAB or BLOCK_N == LAYOUT_AD_M, "kSwapAB requires BLOCK_N = LAYOUT_AD_M");
    DG_STATIC_ASSERT(not kSwapAB or kNumMulticast == 1, "kSwapAB initial impl: no multicast");

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = kSwapAB
        ? STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t)
        : STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_PACKED_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(uint32_t);
    constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(uint32_t);
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);

    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // Tensor memory size and offsets
    constexpr uint32_t kNumSFATmemCols = (SF_BLOCK_M / 32) * SF_PACKED_K_PER_STAGE;
    constexpr uint32_t kNumSFBTmemCols = (SF_BLOCK_N / 32) * SF_PACKED_K_PER_STAGE;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    // Prefetch TMA descriptors at the very beginning
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // D/A/B shared memory
    cd_dtype_t* smem_cd[kNumTMAStoreStages];
    uint32_t* smem_sfa[kNumStages];
    uint32_t* smem_sfb[kNumStages];
    uint32_t* smem_a_packed[kNumStages];
    uint32_t* smem_b_packed[kNumStages];

    #pragma unroll
    for (uint32_t i = 0; i < kNumTMAStoreStages; ++ i)
        smem_cd[i] = reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);

    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_a_packed[i] = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_PACKED_SIZE_PER_STAGE);
        smem_b_packed[i] = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_PACKED_SIZE_PER_STAGE + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    }

    // SFA/SFB shared memory
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_PACKED_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_sfa[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
        smem_sfb[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    }

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_PACKED_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE) +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers    = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers   = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });

    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");


    if (threadIdx.x == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumEpilogueThreads);
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    } else if (threadIdx.x >= 32 and threadIdx.x < 64) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // K-loop driver
    struct DivisibleK {};
    struct NotDivisibleK {};
    uint32_t phase = 0;

    auto launch_k_iterations = [&](const auto& func) {
        const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
        const uint32_t num_iterations = ceil_div(current_shape_k, kNumStages * BLOCK_K);
        const uint32_t num_last_stages = ceil_div(current_shape_k, BLOCK_K) % kNumStages;

        if (num_last_stages == 0) {
            for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter, phase ^= 1)
                func(k_iter, DivisibleK{}, k_iter == num_iterations - 1, num_last_stages);
        } else {
            for (uint32_t k_iter = 0; k_iter < num_iterations - 1; ++ k_iter, phase ^= 1)
                func(k_iter, DivisibleK{}, false, num_last_stages);
            func(num_iterations - 1, NotDivisibleK{}, true, num_last_stages), phase ^= 1;
        }
    };

    auto dispatch_accum_stage_idx = [&](uint32_t accum_stage_idx, const auto& func) {
        DG_STATIC_ASSERT(1 <= kNumEpilogueStages and kNumEpilogueStages <= 2, "Too many epilogue stages");
        accum_stage_idx == 0 ? func(0) : func(1);
    };

    // Dispatch warps into different roles:
    //   warp 0   : TMA load producer
    //   warp 1   : MMA consumer + UTCCP SF copy to TMEM
    //   warp 2   : SF SMEM warp transpose for UTCCP
    //   warp 3+  : Epilogue
    if (warp_idx == 0) {
        // TMA load warp
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    empty_barriers[s]->wait(phase ^ 1);

                    uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                    uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);

                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_block_idx = k_iter * kNumStages + s;
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    if constexpr (kNumMulticast > 1) {
                        m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                        n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                    }

                    if (cute::elect_one_sync()) {
                        if constexpr (kMajorA == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[s], smem_a_packed[s], k_a_idx, m_idx);
                        if constexpr (kMajorA == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(&tensor_map_a, full_barriers[s], smem_a_packed[s], m_idx, k_a_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[s], smem_b_packed[s], k_b_idx, n_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(&tensor_map_b, full_barriers[s], smem_b_packed[s], n_idx, k_b_idx);
                    }
                    auto num_arrival_bytes = SMEM_A_PACKED_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE;

                    const uint32_t sfa_tma_stage = (k_iter * kNumStages + s) % kNumSFAStagesPerLoad;
                    if (sfa_tma_stage == 0 and cute::elect_one_sync()) {
                        uint32_t sf_k_base = k_block_idx / kNumSFAStagesPerLoad * SF_PACKED_K_PER_STAGE;
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            tma_copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[s], smem_sfa[s] + pk * SF_BLOCK_M, m_block_idx * BLOCK_M,
                                                       scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), sched::IndexType::SF_K>(shape_sfa_k, 1, sf_k_base + pk));
                        }
                        num_arrival_bytes += BLOCK_M * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
                    }
                    const uint32_t sfb_tma_stage = (k_iter * kNumStages + s) % kNumSFBStagesPerLoad;
                    if (sfb_tma_stage == 0 and cute::elect_one_sync()) {
                        uint32_t sf_k_base = k_block_idx / kNumSFBStagesPerLoad * SF_PACKED_K_PER_STAGE;
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            tma_copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[s], smem_sfb[s] + pk * SF_BLOCK_N, n_block_idx * BLOCK_N,
                                                       scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_sfb_k, 1, sf_k_base + pk, m_block_idx));
                        }
                        num_arrival_bytes += BLOCK_N * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
                    }

                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive_and_expect_tx(num_arrival_bytes);
                }

                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    empty_barriers[s]->wait(phase ^ 1);
                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive();
                }
            });
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA + UTCCP SF copy warp
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        // Swap-AB: UMMA_N becomes BLOCK_M (the original M dim now plays N role in MMA).
        constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K_INT32 = UMMA_K_FP4 / FP4_ELEMS_PER_INT32;
        constexpr uint32_t NUM_K_ITERS_PER_STAGE = BLOCK_K / UMMA_K_INT32;
        // After swap, the "N iters" walk over BLOCK_M (now the MMA-N axis); without swap, over BLOCK_N.
        constexpr uint32_t NUM_N_ITERS = (kSwapAB ? BLOCK_M : BLOCK_N) / UMMA_N;

        // Swap-AB: pass (b_dtype, a_dtype, ..., kMajorB, kMajorA) to MMA so B occupies MMA-A slot
        //            and A occupies MMA-B slot. Output in TMEM is D^T (N rows, M cols).
        auto instr_desc_mxf4 = kSwapAB
            ? cute::UMMA::make_instr_desc_block_scaled<
                cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N, kMajorB, kMajorA>()
            : cute::UMMA::make_instr_desc_block_scaled<
                cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N, kMajorA, kMajorB>();

        using cute_mma_mxf4_noswap_t = cute::conditional_t<kNumMulticast == 1,
            cute::SM100_MMA_MXF4_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
                                    cutlass::float_ue8m0_t, UMMA_M, UMMA_N, MXF4_VS,
                                    kMajorA, kMajorB>,
            cute::SM100_MMA_MXF4_2x1SM_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
                                          cutlass::float_ue8m0_t, UMMA_M, UMMA_N, MXF4_VS,
                                          kMajorA, kMajorB>>;
        using cute_mma_mxf4_swap_t = cute::conditional_t<kNumMulticast == 1,
            cute::SM100_MMA_MXF4_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
                                    cutlass::float_ue8m0_t, UMMA_M, UMMA_N, MXF4_VS,
                                    kMajorB, kMajorA>,
            cute::SM100_MMA_MXF4_2x1SM_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
                                          cutlass::float_ue8m0_t, UMMA_M, UMMA_N, MXF4_VS,
                                          kMajorB, kMajorA>>;
        using cute_mma_mxf4_t = cute::conditional_t<kSwapAB, cute_mma_mxf4_swap_t, cute_mma_mxf4_noswap_t>;

        DG_STATIC_ASSERT(UMMA_M == 128 or UMMA_M == 256, "MXF4 supports M=128 (1CTA) or M=256 (2CTA)");
        DG_STATIC_ASSERT((UMMA_N % 8 == 0) and (8 <= UMMA_N) and (UMMA_N <= 256), "Invalid MXF4 N-mode size");

        using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
            cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
        auto sf_desc = make_sf_desc(nullptr);

        constexpr uint32_t SMEM_A_SIZE_PER_STAGE_PACKED = LOAD_BLOCK_M * BLOCK_K * sizeof(uint32_t);
        constexpr uint32_t SMEM_B_SIZE_PER_STAGE_PACKED = LOAD_BLOCK_N * BLOCK_K * sizeof(uint32_t);
        auto a_desc_base = make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a_packed[0], 0, 0);
        auto b_desc_base = make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b_packed[0], 0, 0);

        // MXF4 SF addressing constants
        constexpr uint32_t kGroupsPerUmmaStep = UMMA_K_FP4 / MXF4_VS;            // 2
        constexpr uint32_t kGroupsPerPacked = 4;
        constexpr uint32_t kUmmaStepsPerPacked = kGroupsPerPacked / kGroupsPerUmmaStep; // 2
        constexpr uint32_t kSfaColsPerPackedGroup = SF_BLOCK_M / 32;
        constexpr uint32_t kSfbColsPerPackedGroup = SF_BLOCK_N / 32;

        // Pre-compute runtime descriptors (sf_id alternates 0, 2, 0, 2)
        const auto runtime_desc_sf0 = make_runtime_instr_desc_with_sf_id(instr_desc_mxf4, 0, 0);
        const auto runtime_desc_sf2 = make_runtime_instr_desc_with_sf_id(instr_desc_mxf4, kGroupsPerUmmaStep, kGroupsPerUmmaStep);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                tcgen05_after_thread_sync();

                auto empty_barrier_arrive = [&](uint32_t s, bool do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        if constexpr (kNumMulticast == 1) {
                            cutlass::arch::umma_arrive(barrier);
                        } else {
                            constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                            cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                        }
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[s]));
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                };

                launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                    constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                    const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        with_sf_full_barriers[s]->wait(phase);
                        tcgen05_after_thread_sync();

                        // UTCCP: copy SF from SMEM → TMEM
                        // Must stay on warp 1: SF TMEM cols are reused across stages,
                        // so UTCCP must be serialized with MMA on the same warp.
                        const uint32_t sfa_copy_stage = (k_iter * kNumStages + s) % kNumSFAStagesPerLoad;
                        if (sfa_copy_stage == 0 and cute::elect_one_sync()) {
                            #pragma unroll
                            for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                                #pragma unroll
                                for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                    replace_smem_desc_addr(sf_desc, smem_sfa[s] + pk * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                                    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + pk * (SF_BLOCK_M / 32) + i * 4);
                                }
                            }
                            #pragma unroll
                            for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                                #pragma unroll
                                for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                    replace_smem_desc_addr(sf_desc, smem_sfb[s] + pk * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                                    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + pk * (SF_BLOCK_N / 32) + i * 4);
                                }
                            }
                        }
                        __syncwarp();

                        // MMA loop
                        uint32_t a_desc_stage_lo = a_desc_base.lo + s * (SMEM_A_SIZE_PER_STAGE_PACKED / 16);
                        uint32_t b_desc_stage_lo = b_desc_base.lo + s * (SMEM_B_SIZE_PER_STAGE_PACKED / 16);

                        #pragma unroll
                        for (uint32_t k = 0; k < NUM_K_ITERS_PER_STAGE; ++k) {
                            uint32_t packed_group = k / kUmmaStepsPerPacked;
                            uint32_t tmem_sfa_base = kTmemStartColOfSFA + packed_group * kSfaColsPerPackedGroup;
                            uint32_t tmem_sfb_k    = kTmemStartColOfSFB + packed_group * kSfbColsPerPackedGroup;
                            const auto& runtime_desc_k = (k % kUmmaStepsPerPacked == 0) ? runtime_desc_sf0 : runtime_desc_sf2;

                            #pragma unroll
                            for (uint32_t n = 0; n < NUM_N_ITERS; ++n) {
                                auto b_desc = b_desc_base;
                                b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, uint32_t>(
                                    b_desc_stage_lo, n * UMMA_N * BLOCK_K, k * UMMA_K_INT32);

                                #pragma unroll
                                for (uint32_t w = 0; w < kNumMWaves; ++w) {
                                    auto a_desc = a_desc_base;
                                    a_desc.lo = advance_umma_desc_lo<kMajorA, BLOCK_M, kSwizzleAMode, uint32_t>(
                                        a_desc_stage_lo, w * LAYOUT_AD_M * BLOCK_K, k * UMMA_K_INT32);

                                    uint32_t tmem_col = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + n * UMMA_N;
                                    if constexpr (kSwapAB) {
                                        // Swap-AB: B goes into MMA-A slot, A into MMA-B slot.
                                        // SF column args also swap (SFB first, SFA second).
                                        cute_mma_mxf4_t::fma(b_desc, a_desc, tmem_col,
                                                             k_iter > 0 or s > 0 or k > 0,
                                                             runtime_desc_k,
                                                             tmem_sfb_k,
                                                             tmem_sfa_base + w * (kNumUTCCPAlignedElems / 32));
                                    } else {
                                        cute_mma_mxf4_t::fma(a_desc, b_desc, tmem_col,
                                                             k_iter > 0 or s > 0 or k > 0,
                                                             runtime_desc_k,
                                                             tmem_sfa_base + w * (kNumUTCCPAlignedElems / 32),
                                                             tmem_sfb_k);
                                    }
                                }
                            }
                        }

                        empty_barrier_arrive(s, is_last_iter and s == kNumInnerStages - 1);
                    }

                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        with_sf_full_barriers[s]->wait(phase);
                        empty_barrier_arrive(s, false);
                    }
                });
            });
        }
    } else if (warp_idx == 2) {
        // SF transpose warp
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };
        auto fill_sfb_missing_k_groups = [&](uint32_t* smem_ptr) {
            if constexpr (BLOCK_N < kNumUTCCPAlignedElems) {
                // Zero-fill [BLOCK_N, SF_BLOCK_N) before warp-transpose: XOR pattern
                // reads all 128 elements, so uninitialized positions corrupt valid data.
                #pragma unroll
                for (uint32_t pos = lane_idx; pos < kNumUTCCPAlignedElems; pos += 32) {
                    if (pos >= BLOCK_N)
                        st_shared(smem_ptr + pos, 0u);
                }
                __syncwarp();
            }
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    full_barriers[s]->wait(phase);

                    const uint32_t sfa_ut_stage = (k_iter * kNumStages + s) % kNumSFAStagesPerLoad;
                    if (sfa_ut_stage == 0) {
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                                utccp_required_smem_warp_transpose(smem_sfa[s] + pk * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                        }
                        cutlass::arch::fence_view_async_shared();
                    }

                    const uint32_t sfb_ut_stage = (k_iter * kNumStages + s) % kNumSFBStagesPerLoad;
                    if (sfb_ut_stage == 0) {
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                fill_sfb_missing_k_groups(smem_sfb[s] + pk * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                                utccp_required_smem_warp_transpose(smem_sfb[s] + pk * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                            }
                        }
                        cutlass::arch::fence_view_async_shared();
                    }

                    with_sf_full_barriers[s]->arrive(0u);
                }

                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait(phase);
                    with_sf_full_barriers[s]->arrive(0u);
                }
            });
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32) {
        // Epilogue warps
        const auto epilogue_thread_idx = threadIdx.x - kNumNonEpilogueThreads;
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);

      if constexpr (kSwapAB) {
        // Swap-AB epilogue: TMEM holds D^T (BLOCK_N rows x BLOCK_M cols); each
        // accum stage covers BLOCK_M TMEM cols. STORE_BLOCK_M=16 lets the loop
        // skip 16-row M slices that are entirely padding (effective_m / 16),
        // while STORE_BLOCK_N=BLOCK_N stores the full N tile per iteration.
        constexpr uint32_t kNumSwizzleAtomRows = 8;
        constexpr uint32_t STORE_BLOCK_N_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
        constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
        DG_STATIC_ASSERT(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "Invalid swap-AB store_block_m");
        DG_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swap-AB store_block_n");
        DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Swap-AB requires full warpgroup");

        uint32_t tma_stage_idx = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();

                // Effective M (aligned up to STORE_BLOCK_M=16): how many M-cols of D^T are valid.
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                const uint32_t num_stores = effective_m / STORE_BLOCK_M;

                // TMEM col where this accum stage's tile starts.
                const auto tmem_base_addr = accum_stage_idx * BLOCK_M;
                const auto base_m_idx = scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                const auto base_n_idx = n_block_idx * BLOCK_N;

                #pragma unroll 1
                for (uint32_t s = 0; s < num_stores; ++ s) {
                    // Wait if TMA store pipeline full
                    if (s >= kNumTMAStoreStages) {
                        if (epilogue_thread_idx == 0)
                            cute::tma_store_wait<kNumTMAStoreStages - 1>();
                        cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                    }

                    // SMEM store: 4 warps cooperatively write STORE_BLOCK_M × STORE_BLOCK_N tile.
                    // Each warp owns STORE_BLOCK_M rows × STORE_BLOCK_N_ATOM cols.
                    // Within each warp, loop covers STORE_BLOCK_M / kNumSwizzleAtomRows = 2 sub-blocks of 8 rows.
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / kNumSwizzleAtomRows; ++ i) {
                        uint32_t tmem_addr = tmem_base_addr +
                                             s * STORE_BLOCK_M +              // M-slice (cols of TMEM)
                                             i * kNumSwizzleAtomRows;          // Sub-block within slice

                        uint32_t outer_atom_offset = (epilogue_warp_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
                        uint32_t inner_atom_offset = i * kNumSwizzleAtomRows * kSwizzleCDMode;
                        auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                             outer_atom_offset + inner_atom_offset;

                        if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                            uint32_t values[kNumSwizzleAtomRows];
                            // Load 32dp × 8 cols of TMEM per warp (= 8 M-cols × 32 N-rows of D).
                            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                values[0], values[1], values[2], values[3],
                                values[4], values[5], values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();
                            uint32_t col = lane_idx / 4;
                            #pragma unroll
                            for (uint32_t row = 0; row < kNumSwizzleAtomRows; ++ row) {
                                auto smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8)
                                                              + (col ^ row) * kNumBankGroupBytes
                                                              + (lane_idx % 4) * sizeof(float);
                                st_shared(reinterpret_cast<uint32_t*>(smem_ptr), values[row]);
                            }
                        }
                    }

                    // Notify TMEM empty on last store
                    if (s == num_stores - 1) {
                        tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }
                    __syncwarp();

                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();

                    if (epilogue_thread_idx == 0) {
                        uint32_t m_idx = base_m_idx + s * STORE_BLOCK_M;
                        #pragma unroll
                        for (uint32_t ai = 0; ai < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++ ai) {
                            auto smem_ptr = smem_cd[tma_stage_idx] + ai * STORE_BLOCK_M * STORE_BLOCK_N_ATOM;
                            uint32_t n_idx = base_n_idx + ai * STORE_BLOCK_N_ATOM;
                            using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                            cute_tma_t::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                        }
                        cute::tma_store_arrive();
                    }

                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }

                // If entire tile is padding (effective_m=0, hence num_stores=0): still arrive at empty barrier
                // so MMA pipeline can advance.
                if (num_stores == 0) {
                    tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                }
            });
        }
      } else {

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

                if (epilogue_thread_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();

                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();

                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++ s) {
                        const uint32_t iter_idx = w * kNumStores + s;
                        if (iter_idx >= kNumTMAStoreStages) {
                            if (epilogue_thread_idx == 0)
                                cute::tma_store_wait<kNumTMAStoreStages - 1>();
                            cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                        }

                        const auto tma_stage_idx = iter_idx % kNumTMAStoreStages;
                        const auto m_idx = scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx) + w * LAYOUT_AD_M;
                        const auto n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                            col ^= row % (kSwizzleCDMode / 16);

                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N
                                                 + i * kNumElemsPerBankGroup;
                            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                            epilogue_warp_idx * 32 * kSwizzleCDMode +
                                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                            uint32_t values[kNumElemsPerBankGroup];
                            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                            }
                        }

                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }
                        __syncwarp();

                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                        if (epilogue_thread_idx == 0) {
                            using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                            cute_tma_t::copy(&tensor_map_d, smem_cd[tma_stage_idx], n_idx, m_idx);
                            cute::tma_store_arrive();
                        }
                    }
                }
            });
        }
      }  // end of if constexpr (kSwapAB) else (non-swap)

        if (epilogue_thread_idx == 0)
            cute::tma_store_wait<0>();

        if (epilogue_warp_idx == 1)
            Allocator().free(0, kNumTmemCols);
    }

    if constexpr (kNumMulticast > 1)
        cute::cluster_sync();
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
