#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"


namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M),
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumBlocksPerProblem  = kNumMBlocks * kNumNBlocks,
          uint32_t kNumNBlocksPerGroup = 16>
struct SchedulerDW {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks;

    int num_blocks;
    // For grouped GEMM
    int* grouped_layout;
    // Only used for masked layout
    uint32_t curr_group_idx{}, curr_cumsum{};

    __device__ __forceinline__ explicit SchedulerDW(int* grouped_layout = nullptr) {
        static_assert(kGemmType == GemmType::GroupedContiguous);

        if constexpr (kGemmType == GemmType::Normal) {
            // static_assert(kGemmType != kGemmType, "Only support contiguous grouped GEMM");
        } else if (kGemmType == GemmType::GroupedContiguous) {
            this->num_blocks = kNumGroups * kNumBlocksPerProblem;
            this->grouped_layout = grouped_layout;
        } else if (kGemmType == GemmType::GroupedMasked) {
            // static_assert(kGemmType != kGemmType, "Only support contiguous grouped GEMM");
        }
    }

    __device__ __forceinline__ void get_swizzled_block_idx(
        const int block_idx,
        uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t &k_dim_size
    ){
        DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");
        // Swizzle for better L2 usages
        auto _curr_group_idx = block_idx / kNumBlocksPerProblem;
        if(curr_group_idx != _curr_group_idx) {
            for(auto i = curr_group_idx; i < _curr_group_idx; i++) {
                curr_cumsum += __ldg(grouped_layout + i);
            }
        }
        curr_group_idx = _curr_group_idx;
        k_dim_size  = __ldg(grouped_layout + curr_group_idx);
        
        // Swizzle for better L2 usages
        auto in_problem_idx  = block_idx % (kNumBlocksPerProblem);
        auto num_blocks_per_group = kNumMBlocks * kNumNBlocksPerGroup;
        auto group_idx = in_problem_idx / num_blocks_per_group;
        auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
        auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
        auto in_group_idx = in_problem_idx % num_blocks_per_group;
        m_block_idx = in_group_idx / num_n_blocks_in_group;
        n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            // static_assert(kGemmType != kGemmType, "Only support contiguous grouped GEMM");
        } else if (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : __ldg(grouped_layout + m_block_idx * BLOCK_M);
            return offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedMasked) {
            // static_assert(kGemmType != kGemmType, "Only support contiguous grouped GEMM");
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& k_dim_size) {
        const auto next_block_idx = (++ current_iter) * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            static_assert(kGemmType != kGemmType, "Only support contiguous grouped GEMM");
        } else {
            if (next_block_idx >= num_blocks) return false;
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx, k_dim_size);
        }
        return true;
    }
};

enum class Layout {
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

/// X.T @ dy: [SHAPE_M, k1]@[k1, SHAPE_N], [SHAPE_M, k2]@[k2, SHAPE_N], 
/// @tparams: `SHAPE_M` => original (before transpose) `SHAPE_K`
/// @tparams: `SHAPE_N` => original `SHAPE_N`
/// @params:  `shape_k` => original `shape_m`
template <uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel_dw(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                uint32_t shape_k,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    // DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1, "Too much B scales in a single block");
    DG_STATIC_ASSERT(SHAPE_M % 128 == 0, "");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);

    const uint32_t SHAPE_K_SCALES = ceil_div(shape_k, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE = 0;

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* smem_scales_b;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    auto launch_k_iterations = [](const uint32_t dim_k, const auto& func) {
        auto num_k_iters = ceil_div(dim_k, kFullKOfAllStages);
        if (dim_k % kFullKOfAllStages == 0) {
            for (int k_iter = 0; k_iter < num_k_iters; ++ k_iter)
                func(k_iter, DivisibleK{});
        } else {
            for (int k_iter = 0; k_iter < num_k_iters - 1; ++ k_iter)
                func(k_iter, DivisibleK{});
            func(num_k_iters - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx, curr_k_dim_size, num_iterations_cumsum{};
    auto scheduler = SchedulerDW<kGemmType, SHAPE_M, SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast>(grouped_layout);

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>(); // 设置当前线程的最大寄存器数量

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx, curr_k_dim_size)) {
                auto num_iterations = ceil_div(curr_k_dim_size, kFullKOfAllStages);
                launch_k_iterations(curr_k_dim_size, [&](int k_iter, auto type) {
                    if constexpr (std::is_same_v<decltype(type), DivisibleK>){
                        #pragma unroll
                        for (uint32_t s = 0; s < kNumStages; ++ s) {
                            // Wait consumer release
                            empty_barriers[s]->wait((num_iterations_cumsum + k_iter + 1) & 1);

                            // Issue TMA A with broadcasting
                            auto& full_barrier = *full_barriers[s];
                            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K + scheduler.curr_cumsum;
                            tma_copy<kNumTMAMulticast>(
                                &tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_a[s], k_idx, m_block_idx * BLOCK_M
                            );
                            tma_copy<kNumTMAMulticast>(
                                &tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_scales_a[s], m_block_idx * BLOCK_M, k_idx / BLOCK_K
                            );
                            // Issue TMA B without broadcasting
                            tma_copy(
                                &tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_b[s], k_idx, n_block_idx * BLOCK_N 
                            );

                            // if (blockIdx.x == 0){
                            //     printf("load data k_idx %d, s %d, expect %d\n", k_idx, s, SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                            // }
                            full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                        }
                    } else {
                        auto residual_stages = ceil_div((curr_k_dim_size % kFullKOfAllStages), BLOCK_K);
                        for (uint32_t s = 0; s < residual_stages; ++ s) {
                            // Wait consumer release
                            empty_barriers[s]->wait((num_iterations_cumsum + k_iter + 1) & 1);

                            // Issue TMA A with broadcasting
                            auto& full_barrier = *full_barriers[s];
                            int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K + scheduler.curr_cumsum;
                            tma_copy<kNumTMAMulticast>(
                                &tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_a[s], k_idx, m_block_idx * BLOCK_M
                            );
                            tma_copy<kNumTMAMulticast>(
                                &tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_scales_a[s], m_block_idx * BLOCK_M, k_idx / BLOCK_K
                            );
                            // Issue TMA B without broadcasting
                            tma_copy(
                                &tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                smem_b[s], k_idx, n_block_idx * BLOCK_N 
                            );
                            full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                        }

                        // Wait unaligned cases
                        for (uint32_t s = residual_stages; s < kNumStages; ++ s) {
                            empty_barriers[s]->wait((num_iterations_cumsum + k_iter + 1) & 1);
                            full_barriers[s]->arrive();
                        }
                    }
                });
                num_iterations_cumsum += num_iterations;
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((num_iterations_cumsum + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx, curr_k_dim_size)) {
            // Decide the number of scales B to load
            DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(SHAPE_N - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            const auto* local_scale_b = scales_b + (n_block_idx * BLOCK_N / BLOCK_K) * SHAPE_K_SCALES + scheduler.curr_cumsum / BLOCK_K;

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // Launch MMAs
            auto num_iterations = ceil_div(curr_k_dim_size, kFullKOfAllStages);
            launch_k_iterations(curr_k_dim_size, [&](int k_iter, auto type) {
                if constexpr (std::is_same_v<decltype(type), DivisibleK>) {
                    #pragma unroll
                    for (int s = 0; s < kNumStages; ++ s) {
                        // Read B scales
                        float scale_b_0 = __ldg(local_scale_b + k_iter * kNumStages + s), scale_b_1;
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ld_shared(local_scale_b + k_iter * kNumStages + s + SHAPE_K_SCALES);

                        // Wait TMA arrivals
                        full_barriers[s]->wait((num_iterations_cumsum + k_iter) & 1);
                        // if (blockIdx.x == 0 && threadIdx.x == 0){
                        //         printf("calculate data, s %d, num_iterations_cumsum %d, k_iter %d, wait %d, full_barriers[s] %d\n", s, num_iterations_cumsum, k_iter, int32_t((num_iterations_cumsum + k_iter) & 1), full_barriers[s]);
                        //     }

                        // Read A scales
                        // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                        auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0), scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

                        // Commit WGMMA instructions
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_arrive();
                        #pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        warpgroup_commit_batch();
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_wait<0>();

                        // Notify barrier arrival
                        empty_barrier_arrive(s);

                        // Promote with scales
                        float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                        float scale_0_1, scale_1_1;
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                            final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                        }
                    }
                } else {
                    auto residual_stages = ceil_div((curr_k_dim_size % kFullKOfAllStages), BLOCK_K);
                    for (int s = 0; s < residual_stages; ++ s) {
                        // // Read B scales
                        // float scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s), scale_b_1;
                        // // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        // if constexpr (not kMustUseUniformedScaleB)
                        //     scale_b_1 = ld_shared(smem_scales_b + k_iter * kNumStages + s + SHAPE_K_SCALES);

                        // Read B scales
                        float scale_b_0 = __ldg(local_scale_b + k_iter * kNumStages + s), scale_b_1;
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ld_shared(local_scale_b + k_iter * kNumStages + s + SHAPE_K_SCALES);


                        // Wait TMA arrivals
                        full_barriers[s]->wait((num_iterations_cumsum + k_iter) & 1);

                        // Read A scales
                        // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                        auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0), scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

                        // Commit WGMMA instructions
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_arrive();
                        #pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        warpgroup_commit_batch();
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_wait<0>();

                        // Notify barrier arrival
                        empty_barrier_arrive(s);

                        // Promote with scales
                        float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                        float scale_0_1, scale_1_1;
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                            final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                        }
                    }

                    // Wait unaligned cases
                    for (uint32_t s = residual_stages; s < kNumStages; ++ s) {
                        full_barriers[s]->wait((num_iterations_cumsum + k_iter) & 1);
                        empty_barrier_arrive(s);
                    }
                }
            });
            num_iterations_cumsum += num_iterations;

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++ i) {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16)
                );
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0) {
                SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0], final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + WGMMA::kNumAccum / 8 * 16
                );
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Use TMA store to write back to global memory
            if (threadIdx.x == 0) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, n_block_idx * BLOCK_N, scheduler.curr_group_idx * SHAPE_M + m_block_idx * BLOCK_M);
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
class GemmDW {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    GemmDW() = default;

    static void run(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                    uint32_t shape_k,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel_dw<SHAPE_M, SHAPE_N, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, kGemmType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel,
                                         gmem_d, scales_b, grouped_layout,
                                         shape_k,
                                         tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_k) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                SHAPE_M, shape_k, BLOCK_M, BLOCK_K);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address, uint32_t shape_k) {
        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                shape_k, SHAPE_N, BLOCK_K, BLOCK_N);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_d_desc(T* global_address) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                kNumGroups * SHAPE_M, SHAPE_N,
                                BLOCK_M, BLOCK_N,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_k) {
        // Make TMA aligned to 16 bytes
        constexpr uint32_t kAlignment = 16 / sizeof(T);
        auto shape_m = ceil_div(SHAPE_M, kAlignment) * kAlignment;

        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                shape_m, shape_k / BLOCK_K,
                                BLOCK_M, 1,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
            T* global_address, Layout layout,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_cols * sizeof(T), smem_dim, swizzle_type);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
        }
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
