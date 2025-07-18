#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

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
          GemmType kGemmType, typename cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ CUtensorMap tensor_map_a,
                         const __grid_constant__ CUtensorMap tensor_map_b,
                         const __grid_constant__ CUtensorMap tensor_map_d,
                         const __grid_constant__ CUtensorMap tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or (constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t kNumMWaves = BLOCK_M / LAYOUT_AD_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M % LAYOUT_AD_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_M == kNumEpilogueThreads, "Invalid block M");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_k_scales = ceil_div(shape_k, BLOCK_K);

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // 2-CTA MMA
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = std::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M and BLOCK_M % LAYOUT_AD_M == 0, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");

    // Share memory sizes
    // NOTES: do not use `LOAD_BLOCK_M` for SFA, as we need full SFA for promotion
    constexpr bool kMustUseUniformedSFB = (BLOCK_K % BLOCK_N == 0);
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // Must have 2 epilogue stages
    constexpr uint32_t kNumEpilogueStages = 2;

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // Prefetch TMA descriptors at the very beginning
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
    }

    // Data on shared memory (layout as ordered below)
    cd_dtype_t* smem_cd[kNumTMAStoreStages];
    cutlass::float_e4m3_t* smem_a[kNumStages];
    cutlass::float_e4m3_t* smem_b[kNumStages];
    float* smem_sfa[kNumStages];

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
    for (uint32_t i = 0; i < kNumStages; ++ i)
        smem_sfa[i] = reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
        kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (threadIdx.x == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMulticast * kNumEpilogueThreads / 32);
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
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    const uint32_t num_iterations = ceil_div(shape_k, kNumStages * BLOCK_K);
    auto launch_k_iterations = [=](const auto& func) {
        if constexpr (kNumLastStages == 0) {
            for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter)
                func(k_iter, DivisibleK{});
        } else {
            for (uint32_t k_iter = 0; k_iter < num_iterations - 1; ++ k_iter)
                func(k_iter, DivisibleK{});
            func(num_iterations - 1, NotDivisibleK{});
        }
    };

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA>(shape_m, shape_n, grouped_layout);

    // Register configurations
    constexpr uint32_t kNumNonEpilogueRegisters = 64;
    constexpr uint32_t kNumEpilogueRegisters = 216;
    DG_STATIC_ASSERT(kNumNonEpilogueRegisters * kNumNonEpilogueThreads + kNumEpilogueRegisters * kNumEpilogueThreads <= 65535, "Too many registers");

    // Dispatch warps into different roles
    if (warp_idx == 0) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : kNumLastStages;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    // Wait consumer release
                    empty_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter + 1) & 1);

                    // Compute offsets
                    // NOTES: the group is always concatenated with the outer dimension
                    uint32_t m_idx = scheduler.get_global_idx<(kGemmType != GemmType::MGroupedContiguous)>(
                        shape_m, BLOCK_M, m_block_idx);
                    uint32_t n_idx = scheduler.get_global_idx<(kMajorB == cute::UMMA::Major::K)>(
                        shape_n, BLOCK_N, n_block_idx, m_block_idx);

                    // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                    // And for all grouped GEMMs, A must be K-majored
                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_block_idx = k_iter * kNumStages + s;
                    uint32_t k_idx = k_block_idx * BLOCK_K;
                    uint32_t k_b_idx = scheduler.get_global_idx<(kMajorB == cute::UMMA::Major::MN)>(
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    // Add 2 CTA offsets
                    if constexpr (kNumMulticast > 1) {
                        m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                        n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                    }

                    // Issue TMAs
                    if (cute::elect_one_sync()) {
                        if constexpr (kMajorA == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, kNumMulticast>(&tensor_map_a, full_barriers[s], smem_a[s], k_idx, m_idx);
                        if constexpr (kMajorA == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, kNumMulticast>(&tensor_map_a, full_barriers[s], smem_a[s], m_idx, k_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, kNumMulticast>(&tensor_map_b, full_barriers[s], smem_b[s], k_b_idx, n_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, kNumMulticast>(&tensor_map_b, full_barriers[s], smem_b[s], n_idx, k_b_idx);

                        // Issue SFA TMA
                        tma_copy<BLOCK_M, 1, 0, kNumMulticast>(
                            &tensor_map_sfa, full_barriers[s],
                            smem_sfa[s], m_block_idx * BLOCK_M,
                            scheduler.get_global_idx<(kGemmType != GemmType::MGroupedContiguous)>(shape_k_scales, 1, k_block_idx));
                    }

                    // Arrive at full barriers
                    constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE;
                    if (is_leader_cta and cute::elect_one_sync())
                        full_barriers[s]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    empty_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter + 1) & 1);
                    if (is_leader_cta and cute::elect_one_sync())
                        full_barriers[s]->arrive();
                }
            });
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        // TODO: refactor `UMMA_M` calculation
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Launch MMAs
            launch_k_iterations([&](uint32_t k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : kNumLastStages;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    // Wait TMA full
                    auto iter_idx = scheduler.current_iter * num_iterations + k_iter;
                    full_barriers[s]->wait(iter_idx & 1);

                    // Wait tensor memory empty
                    auto accum_stage_idx = (iter_idx * kNumStages + s) % kNumEpilogueStages;
                    auto accum_stage_phase = ((iter_idx * kNumStages + s) / kNumEpilogueStages) & 1;
                    tmem_empty_barriers[accum_stage_idx]->wait(accum_stage_phase ^ 1);

                    // Issue UMMA in the leader CTA
                    if (s < kNumInnerStages) {
                        using cute_mma_t = std::conditional_t<kNumMulticast == 1,
                            cute::SM100_MMA_F8F6F4_SS, cute::SM100_MMA_F8F6F4_2x1SM_SS>;
                        tcgen05_after_thread_sync();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            auto b_desc = make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[s], 0, k * UMMA_K);
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                                auto a_desc = make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[s], w * LAYOUT_AD_M, k * UMMA_K);
                                cute_mma_t::fma(a_desc, b_desc,
                                                accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                                k > 0,
                                                runtime_instr_desc);
                            }
                        }
                        tcgen05_before_thread_sync();
                    }

                    // Commit to the TMA empty and tensor memory full barrier
                    auto umma_arrive = [](const uint64_t* barrier) {
                        if constexpr (kNumMulticast == 1) {
                            cutlass::arch::umma_arrive(barrier);
                        } else {
                            constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                            cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                        }
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                }
            });
        }
    } else if (warp_idx < kNumNonEpilogueThreads / 32) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
    } else if (warp_idx >= kNumNonEpilogueThreads / 32) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // Epilogue warp groups
        const auto epilogue_thread_idx = threadIdx.x - kNumNonEpilogueThreads;
        const auto epilogue_thread_idx_in_warpgroup = epilogue_thread_idx % 128;
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);
        const auto epilogue_warpgroup_idx = epilogue_thread_idx / 128;

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
            constexpr uint32_t kNumElemsPerLDTM = 16;
            DG_STATIC_ASSERT(kNumElemsPerLDTM == 16 and BLOCK_N % kNumElemsPerLDTM == 0 and BLOCK_K % kNumElemsPerLDTM == 0, "Invalid LDTM width");

            // SFB stuffs
            uint32_t num_former_iters = BLOCK_N, num_full_iters = BLOCK_N;
            if constexpr (not kMustUseUniformedSFB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - ((n_block_idx * BLOCK_N) % BLOCK_K));
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N);
            }
            num_former_iters /= kNumElemsPerLDTM, num_full_iters /= kNumElemsPerLDTM;
            const auto sfb_offset = scheduler.get_global_idx<true>(ceil_div(shape_n, BLOCK_K), 0, 0, m_block_idx);
            const auto sfb_ptr = sfb + (sfb_offset + ((n_block_idx * BLOCK_N) / BLOCK_K)) * shape_k_scales;

            // Launch promotion
            float accum[BLOCK_N] = {0};
            launch_k_iterations([&](uint32_t k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : kNumLastStages;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    // Load SFB
                    float sf_0 = 0, sf_1 = 0;
                    if (s < kNumInnerStages) {
                        const auto k_block_idx = k_iter * kNumStages + s;
                        sf_0 = __ldg(sfb_ptr + k_block_idx);
                        sf_1 = num_former_iters < num_full_iters ? __ldg(sfb_ptr + k_block_idx + shape_k_scales) : 0;
                    }

                    // Wait UMMA arrival
                    auto iter_idx = scheduler.current_iter * num_iterations + k_iter;
                    auto accum_stage_idx = (iter_idx * kNumStages + s) % kNumEpilogueStages;
                    auto accum_stage_phase = ((iter_idx * kNumStages + s) / kNumEpilogueStages) & 1;
                    tmem_full_barriers[accum_stage_idx]->wait(accum_stage_phase);
                    tcgen05_after_thread_sync();

                    // Commit to the TMA empty barrier for all CTAs after loading SFA
                    float sfa = s < kNumInnerStages ? ld_shared(smem_sfa[s] + epilogue_thread_idx) : 0;
                    sf_0 *= sfa, sf_1 *= sfa;
                    __syncwarp();
                    if (lane_idx < kNumMulticast)
                        empty_barriers[s]->arrive(lane_idx);
                    __syncwarp();

                    // Do promotion like the SM90 kernel
                    if (s < kNumInnerStages) {
                        uint32_t values[kNumElemsPerLDTM];
                        #pragma unroll
                        for (uint32_t i = 0; i < BLOCK_N / kNumElemsPerLDTM; ++ i) {
                            // Load from tensor memory
                            cute::SM100_TMEM_LOAD_32dp32b16x::copy(
                                accum_stage_idx * kNumMWaves * BLOCK_N + epilogue_warpgroup_idx * BLOCK_N + i * kNumElemsPerLDTM,
                                values[ 0], values[ 1], values[ 2], values[ 3],
                                values[ 4], values[ 5], values[ 6], values[ 7],
                                values[ 8], values[ 9], values[10], values[11],
                                values[12], values[13], values[14], values[15]);
                            cutlass::arch::fence_view_async_tmem_load();

                            // Promote
                            const auto sf = (kMustUseUniformedSFB or i < num_former_iters) ? sf_0 : sf_1;
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumElemsPerLDTM; ++ j)
                                accum[i * kNumElemsPerLDTM + j] += *reinterpret_cast<float*>(&values[j]) * sf;
                        }
                    }

                    // Commit to the tensor memory empty barrier (only at the leader CTA)
                    tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                }
            });

            // Flush TMA stores
            // NOTES: for the first store, we have to flush all previous TMA,
            // as we don't share pipeline stages between two blocks
            if (epilogue_thread_idx_in_warpgroup == 0)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier(STORE_BLOCK_M, epilogue_warpgroup_idx).sync();

            // Write shared memory
            DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

            // Epilogue store and addition
            // Issue every swizzled atom and pipeline: store shared, add C, and TMA store
            constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
            #pragma unroll
            for (uint32_t s = 0; s < kNumStores; ++ s) {
                // Wait shared memory to be released
                if (s >= kNumTMAStoreStages) {
                    if (epilogue_thread_idx_in_warpgroup == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier(STORE_BLOCK_M, epilogue_warpgroup_idx).sync();
                }

                // The pipeline stage
                const auto tma_stage_idx = s % kNumTMAStoreStages;
                const auto m_idx = scheduler.get_global_idx<(kGemmType != GemmType::MGroupedContiguous)>(shape_m, BLOCK_M, m_block_idx);
                const auto n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;
                const auto local_smem_cd = smem_cd[tma_stage_idx] + epilogue_warpgroup_idx * STORE_BLOCK_M * STORE_BLOCK_N;

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
                    auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // Base pointer
                                    epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp offset
                                    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

                    // Load from tensor memory, store into shared memory
                    // NOTES: if you want to do accumulation, please notice that you need two accumulation barriers
                    const auto offset = s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                    if constexpr (std::is_same_v<cd_dtype_t, float>) {
                        // For FP32 output, read and store
                        DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                        st_shared(smem_ptr,
                                  *reinterpret_cast<uint32_t*>(&accum[offset + 0]),
                                  *reinterpret_cast<uint32_t*>(&accum[offset + 1]),
                                  *reinterpret_cast<uint32_t*>(&accum[offset + 2]),
                                  *reinterpret_cast<uint32_t*>(&accum[offset + 3]));
                    } else {
                        // For BF16 output, read, cast and store
                        DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and std::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                        st_shared(smem_ptr,
                                  cast_into_bf16_and_pack(accum[offset + 0], accum[offset + 1]),
                                  cast_into_bf16_and_pack(accum[offset + 2], accum[offset + 3]),
                                  cast_into_bf16_and_pack(accum[offset + 4], accum[offset + 5]),
                                  cast_into_bf16_and_pack(accum[offset + 6], accum[offset + 7]));
                    }
                }

                // Synchronize all threads and issue TMA
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(STORE_BLOCK_M, epilogue_warpgroup_idx).sync();
                if (epilogue_thread_idx_in_warpgroup == 0) {
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_d, local_smem_cd,
                        n_idx, m_idx + epilogue_warpgroup_idx * STORE_BLOCK_M);
                    cute::tma_store_arrive();
                }
            }
        }

        // Flush all stages in the pipeline to make TMA stores visible to the next kernel
        // TODO: do we actually need this?
        if (epilogue_thread_idx_in_warpgroup == 0)
            cute::tma_store_wait<0>();

        // Deallocate tensor memory by warp 1
        // NOTES: warp 0 is waiting TMA store
        // TODO: do we need 2 SM allocation?
        if (epilogue_warp_idx == 1)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
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
