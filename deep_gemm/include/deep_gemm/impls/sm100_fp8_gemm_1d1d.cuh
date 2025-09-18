#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumLastStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_gemm_1d1d_impl(int* grouped_layout,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_c,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t kNumMWaves = BLOCK_M / LAYOUT_AD_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumSFStagesPerLoad = sizeof(uint32_t) / sizeof(cutlass::float_ue8m0_t);
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M % LAYOUT_AD_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const uint32_t shape_sf_k = ceil_div(shape_k, BLOCK_K * kNumSFStagesPerLoad);

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // 2-CTA MMA
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M and BLOCK_M % LAYOUT_AD_M == 0, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // Automatically deduce the number of epilogue stages (1 or 2), according to the tensor memory size
    // TODO: test cases of `kNumMWaves == 2 and kNumEpilogueStages == 2`
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    // Prefetch TMA descriptors at the very beginning
    if (threadIdx.x == 0) {
        // NOTES: `reinterpret_cast` must be here, or NVRTC will fail
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_d);
        if constexpr (kWithAccumulation)
            cute::prefetch_tma_descriptor(&tensor_map_c);
    }

    // Data on shared memory (layout as ordered below)
    cd_dtype_t* smem_cd[kNumTMAStoreStages];
    cutlass::float_e4m3_t* smem_a[kNumStages];
    cutlass::float_e4m3_t* smem_b[kNumStages];
    uint32_t* smem_sfa[kNumStages];
    uint32_t* smem_sfb[kNumStages];

    // Fill D/A/B pointers
    #pragma unroll
    for (uint32_t i = 0; i < kNumTMAStoreStages; ++ i)
        smem_cd[i] = reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    }

    // Fill SFA/SFB
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_sfa[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
        smem_sfb[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    }

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers      = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (threadIdx.x == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumEpilogueThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    } else if (threadIdx.x >= 32 and threadIdx.x < 64) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, grouped_layout);

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    uint32_t phase = 0;
    auto launch_k_iterations = [&](const auto& func) {
        const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
        const uint32_t num_iterations = ceil_div(current_shape_k, kNumStages * BLOCK_K);
        const uint32_t num_last_stages = ceil_div(current_shape_k, BLOCK_K) % kNumStages;

        // TODO: refactor here
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
        DG_STATIC_ASSERT(1 <= kNumEpilogueStages and kNumEpilogueStages <= 2,
                         "Too many epilogue stages, please modify the Python heuristic as well");
        accum_stage_idx == 0 ? func(0) : func(1);
    };

    // Dispatch warps into different roles
    if (warp_idx == 0) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    // Wait consumer release
                    empty_barriers[s]->wait(phase ^ 1);

                    // Compute offsets
                    // NOTES: the group is always concatenated with the outer dimension
                    uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), KGroupedIndexType::MN> (
                        shape_m, BLOCK_M, m_block_idx);
                    uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), KGroupedIndexType::MN> (
                        shape_n, BLOCK_N, n_block_idx, m_block_idx);

                    // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                    // And for all m-grouped GEMMs, A must be K-majored
                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_block_idx = k_iter * kNumStages + s;
                    uint32_t k_idx = k_block_idx * BLOCK_K;
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), KGroupedIndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), KGroupedIndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    // Add 2 CTA offsets
                    if constexpr (kNumMulticast > 1) {
                        m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                        n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                    }

                    // Issue TMAs
                    if (cute::elect_one_sync()) {
                        if constexpr (kMajorA == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, 1>(&tensor_map_a, full_barriers[s], smem_a[s], k_a_idx, m_idx);
                        if constexpr (kMajorA == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, 1>(&tensor_map_a, full_barriers[s], smem_a[s], m_idx, k_a_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, 1>(&tensor_map_b, full_barriers[s], smem_b[s], k_b_idx, n_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, 1>(&tensor_map_b, full_barriers[s], smem_b[s], n_idx, k_b_idx);
                    }
                    auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;

                    // Issue SFA and SFB TMAs at certain stages
                    // No swizzling, so one TMA for one SF is enough
                    const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                    if (sf_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                        tma_copy<BLOCK_M, 1, 0, 1>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M,
                                                   scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), KGroupedIndexType::SF_K>(shape_sf_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFStagesPerLoad)));
                        tma_copy<BLOCK_N, 1, 0, 1>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N,
                                                   scheduler.template get_global_idx<true, KGroupedIndexType::SF_K>(shape_sf_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFStagesPerLoad), m_block_idx));
                        num_arrival_bytes += (BLOCK_M + BLOCK_N) * sizeof(uint32_t);
                    }

                    // Arrive at full barriers
                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive_and_expect_tx(num_arrival_bytes);
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    empty_barriers[s]->wait(phase ^ 1);
                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive();
                }
            });
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        // TODO: refactor `UMMA_M` calculation
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                                   float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                // Wait tensor memory empty barrier arrival
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                tcgen05_after_thread_sync();

                // Empty barrier arrival
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

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                };

                // Launch MMAs
                launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                    constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                    const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait TMA and SF-transpose arrival
                        with_sf_full_barriers[s]->wait(phase);
                        tcgen05_after_thread_sync();

                        // Do SF copy at certain stages
                        // NOTES: CUTLASS UTCCP's interface does not have `elect_one_sync`, we must do it by ourselves
                        const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                        if (sf_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                            using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                                cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;

                            // SFA and SFB copy
                            // TODO: process shared memory descriptor by addition
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = smem_sfa[s] + i * kNumUTCCPAlignedElems;
                                replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = smem_sfb[s] + i * kNumUTCCPAlignedElems;
                                replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }
                        }
                        __syncwarp();

                        // Issue UMMA in the leader CTA
                        using cute_mma_t = cute::conditional_t<kNumMulticast == 1,
                            cute::SM100_MMA_MXF8F6F4_SS      <cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                              cutlass::float_ue8m0_t, UMMA_M, UMMA_N, kMajorA, kMajorB>,
                            cute::SM100_MMA_MXF8F6F4_2x1SM_SS<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                              cutlass::float_ue8m0_t, UMMA_M, UMMA_N, kMajorA, kMajorB>>;
                        const auto& runtime_instr_desc = make_runtime_instr_desc_with_sf_id(instr_desc, sf_stage_in_group_idx);
                        const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, s);
                        const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, s);
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, cutlass::float_e4m3_t>(b_desc_base_lo, 0, k * UMMA_K);
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                                a_desc.lo = advance_umma_desc_lo<kMajorA, BLOCK_M, kSwizzleAMode, cutlass::float_e4m3_t>(a_desc_base_lo, w * LAYOUT_AD_M * BLOCK_K, k * UMMA_K);
                                cute_mma_t::fma(a_desc, b_desc,
                                                accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                                k_iter > 0 or s > 0 or k > 0,
                                                runtime_instr_desc,
                                                kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
                                                kTmemStartColOfSFB);
                            }
                        }

                        // Commit to the mbarrier object
                        // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                        empty_barrier_arrive(s, is_last_iter and s == kNumInnerStages - 1);
                    }

                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        with_sf_full_barriers[s]->wait(phase);
                        empty_barrier_arrive(s, false);
                    }
                });
            });
        }
    } else if (warp_idx == 2) {
        // UTCCP transposer
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

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    // Wait TMA arrival
                    full_barriers[s]->wait(phase);

                    // Transpose for UTCCP at certain stages
                    const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                    if (sf_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                            utccp_required_smem_warp_transpose(smem_sfa[s] + i * kNumUTCCPAlignedElems);
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                            utccp_required_smem_warp_transpose(smem_sfb[s] + i * kNumUTCCPAlignedElems);
                        // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                        cutlass::arch::fence_view_async_shared();
                    }

                    // Arrive
                    with_sf_full_barriers[s]->arrive(0u);
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait(phase);
                    with_sf_full_barriers[s]->arrive(0u);
                }
            });
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32) {
        // Epilogue warp groups
        const auto epilogue_thread_idx = threadIdx.x - kNumNonEpilogueThreads;
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // TMA checks
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

                // Flush TMA stores
                // NOTES: for the first store, we have to flush all previous TMA,
                // as we don't share pipeline stages between two blocks
                if (epilogue_thread_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();

                // Wait UMMA arrival
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();

                // Load from tensor memory into registers, and write shared memory with STSM
                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                // Iterate over M waves
                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                    // Issue every swizzled atom and pipeline STSM and TMA store
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++ s) {
                        // Wait shared memory to be released
                        const uint32_t iter_idx = w * kNumStores + s;
                        if (iter_idx >= kNumTMAStoreStages) {
                            if (epilogue_thread_idx == 0)
                                cute::tma_store_wait<kNumTMAStoreStages - 1>();
                            cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                        }

                        // The pipeline stage
                        const auto tma_stage_idx = iter_idx % kNumTMAStoreStages;
                        const auto m_idx = scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), KGroupedIndexType::MN>(shape_m, BLOCK_M, m_block_idx) + w * LAYOUT_AD_M;
                        const auto n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                        // Store into shared memory
                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                            // Calculate the index of the bank group to be written in the atom
                            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                            // Reshape the atom in another view and swizzle
                            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
                            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
                            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
                            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                            col ^= row % (kSwizzleCDMode / 16);

                            // Source and destination memory address
                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +               // Accumulator offset
                                                 w * BLOCK_N +                                          // Wave offset
                                                 s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
                            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // Base pointer
                                            epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp offset
                                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

                            // Load from tensor memory, store into shared memory
                            uint32_t values[kNumElemsPerBankGroup];
                            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                                // For FP32 output, read and store
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                            } else {
                                // For BF16 output, read, cast and store
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3],
                                    values[4], values[5], values[6], values[7]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr,
                                          cast_into_bf16_and_pack(values[0], values[1]),
                                          cast_into_bf16_and_pack(values[2], values[3]),
                                          cast_into_bf16_and_pack(values[4], values[5]),
                                          cast_into_bf16_and_pack(values[6], values[7]));
                            }
                        }

                        // Notify tensor memory empty (only at the leader CTA) arrival ASAP
                        // NOTES: only the last stage needs to do this
                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }
                        __syncwarp();

                        // Synchronize all threads and issue TMA
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

        // Flush all stages in the pipeline to make TMA stores visible to the next kernel
        if (epilogue_thread_idx == 0)
            cute::tma_store_wait<0>();

        // Deallocate tensor memory by warp 1
        // NOTES: warp 0 is waiting TMA store
        if (epilogue_warp_idx == 1)
            Allocator().free(0, kNumTmemCols);
    }

    // To safely deconstruct all barriers, we need a cluster sync
    // TODO: optimize it by another round of barrier waits
    if constexpr (kNumMulticast > 1)
        cute::cluster_sync();
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100a/sm_101a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
