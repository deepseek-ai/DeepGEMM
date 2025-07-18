#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd>
__device__ __host__ void outer_launch_k_iterations(const auto& inner_launch_k_iterations, const auto& func, uint32_t num_former_iters) {
    if (num_former_iters == kNumFormerIters) {
        inner_launch_k_iterations(func, cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        outer_launch_k_iterations<kNumFormerIters + kGap, kGap, kEnd>(inner_launch_k_iterations, func, num_former_iters);
}

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleDMode,
          uint32_t kNumStages, uint32_t kNumLastStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          GemmType kGemmType>
__global__ void __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
sm90_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ CUtensorMap tensor_map_a,
                        const __grid_constant__ CUtensorMap tensor_map_b,
                        const __grid_constant__ CUtensorMap tensor_map_d,
                        const __grid_constant__ CUtensorMap tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or (constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    const uint32_t& shape_k_scales = ceil_div(shape_k, BLOCK_K);
    const uint32_t& smem_sfb_size = align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier));

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    const uint32_t num_iterations = ceil_div(shape_k, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (threadIdx.x == kNumMathThreads) {
        // NOTES: `reinterpret_cast` must be here, or NVRTC will fail
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_sfa));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_sfa[kNumStages];
    float* smem_sfb;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_sfa[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SFA_SIZE_PER_STAGE);
    }
    smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    struct SkipComputation {};
    struct NotSkipComputation {};
    auto launch_k_iterations = [=](const auto& func, bool skip_computation, uint32_t num_former_iters) {
        constexpr bool kShouldOptimize = BLOCK_K / constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
        constexpr uint32_t kGap = constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
        constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

        // NOTES: for too-many branches (> 5), we disable this optimization
        // Otherwise, the compiler must know the dynamic variable `num_former_iters`'s real value
        outer_launch_k_iterations<0, kGap, kEnd>([=](const auto& func, auto num_former_iters_type) {
            if (skip_computation) {
                for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter)
                    func(k_iter, DivisibleK{}, SkipComputation{}, num_former_iters_type);
            } else if (shape_k % kFullKOfAllStages == 0) {
                for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter)
                    func(k_iter, DivisibleK{}, NotSkipComputation{}, num_former_iters_type);
            } else {
                for (uint32_t k_iter = 0; k_iter < num_iterations - 1; ++ k_iter)
                    func(k_iter, DivisibleK{}, NotSkipComputation{}, num_former_iters_type);
                func(num_iterations - 1, NotDivisibleK{}, NotSkipComputation{}, num_former_iters_type);
            }
        }, func, kShouldOptimize ? num_former_iters : 0);
    };

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA>(shape_m, shape_n, grouped_layout);

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                launch_k_iterations([&](uint32_t k_iter, auto divisible_type, auto _, auto __) {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(divisible_type), DivisibleK>;
                    constexpr uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : kNumLastStages;

                    // Assign TMA multicast number into A and B
                    // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                    const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                    const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                    const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                    DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                    // NOTES: unrolling and `kNumInnerStages` are vital for performance, NVCC will try to eliminate all
                    // shared memory pointers, e.g. `full_barriers` registers, if all the access indices are constant
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter + 1) & 1);

                        // Issue TMA A
                        constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                        auto& full_barrier = *full_barriers[s];
                        uint32_t k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                        tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_a[s], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                                 num_tma_multicast_a);
                        tma_copy(&tensor_map_sfa, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_sfa[s], m_block_idx * BLOCK_M,
                                 scheduler.get_global_idx<kWithGroupOffsetA>(shape_k_scales, 1, k_idx / BLOCK_K),
                                 num_tma_multicast_a);

                        // Issue TMA B
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_b[s], k_idx, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                                 num_tma_multicast_b);
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                    }

                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                }, false, 0);
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * num_iterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

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

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32) {
                auto num_previous_lines = scheduler.get_global_idx<true>(ceil_div(shape_n, BLOCK_K), 0, 0, m_block_idx);
                auto local_sfb = sfb + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * shape_k_scales;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                    st_shared(smem_sfb + i, __ldg(local_sfb + i));
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = WGMMA::M * (BLOCK_M <= 64 ? 1 : 2);
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations([&](uint32_t k_iter, auto divisible_type, auto skip_type, auto _) {
                constexpr bool kSkipComputation = std::is_same_v<decltype(skip_type), SkipComputation>;
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(divisible_type), DivisibleK>;
                constexpr uint32_t kNumInnerStages = kSkipComputation ? 0 : (kHasDivisibleStages ? kNumStages : kNumLastStages);

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    // Read B scales
                    float scale_b_0 = ld_shared(smem_sfb + k_iter * kNumStages + s), scale_b_1;
                    // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_b_1 = ld_shared(smem_sfb + k_iter * kNumStages + s + shape_k_scales);

                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter) & 1);

                    // TODO: remove some useless computation for unaligned Ms
                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                      	auto m_offset = local_idx * WAVE_BLOCK_M;

                    	// Read A scales
                    	// NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    	auto scale_a_0 = ld_shared(smem_sfa[s] + r_0 + m_offset);
                        auto scale_a_1 = ld_shared(smem_sfa[s] + r_1 + m_offset);

                    	// Commit WGMMA instructions
                    	#pragma unroll
                    	for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                    	warpgroup_arrive();
                    	#pragma unroll
                    	for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = make_smem_desc(smem_a[s] + (math_wg_idx * WGMMA::M + m_offset) * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                    	}
                    	warpgroup_commit_batch();
                    	#pragma unroll
                    	for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                    	warpgroup_wait<0>();

                    	// Notify barrier arrival at the last warpgroup wave
                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                    	    empty_barrier_arrive(s);

                    	// Promote with scales
                    	// NOTES: making it as predicates is very important for performance, comparing to two loops
                    	float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                    	float scale_0_1, scale_1_1;
                    	if constexpr (not kMustUseUniformedScaleB)
                            scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                        auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                    	#pragma unroll
                    	for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                            bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                            shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                            shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                            shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                            shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                    	}
                    }
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            }, not scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M), num_former_iters);

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
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
                    } else {
                        // No swizzling, just padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumMathThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr,
                                              n_block_idx * BLOCK_N + in_block_n_offset,
                                              scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx));
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
