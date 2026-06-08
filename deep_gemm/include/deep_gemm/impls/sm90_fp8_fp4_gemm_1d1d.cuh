#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <stdint.h>
#include <cuda_fp8.h>

#include <cute/int_tuple.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {


__device__ __forceinline__ uint8_t sm90_fp8_fp4_fused_e2m1_to_e4m3_bits(uint8_t code) {
    // E2M1 values {0, 0.5, 1, 1.5, 2, 3, 4, 6} map exactly to E4M3.
    constexpr uint8_t kE2M1ToE4M3[8] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
    const uint8_t value_idx = code & 0x07u;
    const uint8_t sign = (value_idx != 0u && (code & 0x08u) != 0u) ? 0x80u : 0x00u;
    return kE2M1ToE4M3[value_idx] | sign;
}

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_fp4_gemm_1d1d_impl(__nv_fp8_e4m3* gmem_a_ptr, int8_t* gmem_b_ptr,
                        cd_dtype_t* gmem_d_ptr, int* grouped_layout,
                        cute::TmaDescriptor* tensor_map_buffer,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedContiguous or
                     kGemmType == GemmType::MGroupedContiguousWithPsumLayout,
                     "SM90 FP8xFP4 fused only supports normal and m-grouped contiguous GEMM");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(cd_dtype_t);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u);
    DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment");

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Tensor maps on shared and global memory
    (void)gmem_a_ptr;
    (void)tensor_map_buffer;
    (void)gmem_d_ptr;

    // Data on shared memory
    auto smem_d = reinterpret_cast<cd_dtype_t*>(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE)); 
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });
    constexpr auto SMEM_SF_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + i * SMEM_SFA_SIZE_PER_STAGE));
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_SF_OFFSET + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * ALIGNED_SMEM_SFB_SIZE_PER_STAGE));
    });

    // Barriers on shared memory
    constexpr auto SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + ALIGNED_SMEM_SFB_SIZE_PER_STAGE);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + i * static_cast<uint32_t>(sizeof(Barrier))));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + (kNumStages + i) * static_cast<uint32_t>(sizeof(Barrier))));
    });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // Initialize barriers
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

    // Pipeline unroll control
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);

    // Register reconfigurations (more math registers are needed with unrolling)
    constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
    constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();
    
    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, 128u>(shape_m, shape_n, shape_k, grouped_layout);
    auto get_current_group_idx = [&]() {
        if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return static_cast<uint32_t>(cute::max(0, grouped_layout[m_block_idx * BLOCK_M]));
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return scheduler.current_group_idx;
        } else {
            return 0u;
        }
    };

    // TMA and MMA pipeline
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1}; // Pipeline stage and phase
    };
    uint32_t iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
                
                const uint32_t num_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                #pragma unroll kNumPipelineUnrolls
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t current_group_idx = get_current_group_idx();
                    const uint32_t sf_k_idx_a = k_block_idx;
                    const uint32_t sf_k_idx_b = current_group_idx * math::ceil_div(shape_k, BLOCK_K) + k_block_idx;
                    const auto tensor_map_a_ptr = &tensor_map_a_base;
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], m_idx, sf_k_idx_a, num_tma_multicast_a);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], n_idx, sf_k_idx_b, num_tma_multicast_b);
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a_ptr, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
                    full_barrier.arrive_and_expect_tx(
                        SMEM_A_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                }
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Accumulation for WGMMA or CUDA promotion
            DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2), "Invalid block sizes");
            const uint32_t current_shape_k = shape_k;
            const uint32_t current_group_idx = get_current_group_idx();
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
            float2 scales_b[WGMMA::kNumAccum / 4];

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            #pragma unroll kNumPipelineUnrolls
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                // Wait TMA arrivals
                CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);

                // Decode one packed byte into two FP8 values. SM90 cannot use
                // Blackwell's FP4 TMA/UMMA path, so keep B loads coalesced here.
                for (uint32_t idx = threadIdx.x; idx < BLOCK_N * (BLOCK_K / 2); idx += kNumMathThreads) {
                    const uint32_t tile_n = idx / (BLOCK_K / 2);
                    const uint32_t packed_k = idx % (BLOCK_K / 2);
                    const uint32_t tile_k = packed_k * 2;
                    const uint32_t global_n = n_block_idx * BLOCK_N + tile_n;
                    const uint32_t global_k = k_block_idx * BLOCK_K + tile_k;
                    constexpr bool kFullStaticNK = SHAPE_N != 0 and SHAPE_K != 0 and
                                                   SHAPE_N % BLOCK_N == 0 and SHAPE_K % BLOCK_K == 0;
                    uint8_t packed;
                    const uint32_t b_group_offset = current_group_idx * shape_n * (shape_k / 2);
                    if constexpr (kFullStaticNK) {
                        packed = static_cast<uint8_t>(gmem_b_ptr[b_group_offset + global_n * (shape_k / 2) + global_k / 2]);
                    } else {
                        packed = 0;
                        if (global_n < shape_n and global_k < current_shape_k)
                            packed = static_cast<uint8_t>(gmem_b_ptr[b_group_offset + global_n * (shape_k / 2) + global_k / 2]);
                    }

                    const uint32_t n_group = tile_n / 8;
                    const uint32_t n_in_group = tile_n % 8;
                    const uint32_t swizzled_k = tile_k ^ (n_in_group * 16);
                    const uint32_t smem_idx = n_group * 8 * BLOCK_K + n_in_group * BLOCK_K + swizzled_k;
                    auto smem_b_bytes = reinterpret_cast<uint8_t*>(smem_b[stage_idx]);
                    const uint8_t lo = sm90_fp8_fp4_fused_e2m1_to_e4m3_bits(packed & 0x0fu);
                    const uint8_t hi = sm90_fp8_fp4_fused_e2m1_to_e4m3_bits((packed >> 4) & 0x0fu);
                    if constexpr (kFullStaticNK) {
                        *reinterpret_cast<uint16_t*>(smem_b_bytes + smem_idx) = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
                    } else {
                        if (global_k + 1 < current_shape_k) {
                            *reinterpret_cast<uint16_t*>(smem_b_bytes + smem_idx) = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
                        } else {
                            smem_b_bytes[smem_idx] = lo;
                        }
                    }
                }
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 7);

                // Read A scales
                // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                auto scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);
                auto scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);

                // Read B scales
                #pragma unroll
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                    scales_b[i] = ptx::ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + i * 8 + col_idx * 2));

                // Commit WGMMA instructions
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                    auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                // Notify barrier arrival
                empty_barrier_arrive(stage_idx);

                // Promote with scales
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const float &scale_b_0 = scales_b[i].x;
                    const float &scale_b_1 = scales_b[i].y;
                    final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
                    final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
                    final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
                    final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
                }
            }

            if constexpr (kGemmType == GemmType::Normal) {
                // Flush previous stores
                if (warp_idx % 4 == 0 and cute::elect_one_sync())
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

                // Store to D shared memory
                const auto smem_d_0 = reinterpret_cast<float2*>(smem_d + r_0 * BLOCK_N + col_idx * 2);
                const auto smem_d_1 = reinterpret_cast<float2*>(smem_d + r_1 * BLOCK_N + col_idx * 2);
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    ptx::st_shared(smem_d_0 + i * 4, {final_accum[i * 4 + 0], final_accum[i * 4 + 1]});
                    ptx::st_shared(smem_d_1 + i * 4, {final_accum[i * 4 + 2], final_accum[i * 4 + 3]});
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

                // Use TMA reduce-add to accumulate C into D for the normal FP32 path.
                if (warp_idx % 4 == 0 and cute::elect_one_sync()) {
                    cute::SM90_TMA_REDUCE_ADD_2D::copy(
                        &tensor_map_cd, smem_d_0, n_block_idx * BLOCK_N,
                        m_block_idx * BLOCK_M + r_0);
                    cute::tma_store_arrive();
                }
            } else {
                const uint32_t row_0 = m_block_idx * BLOCK_M + r_0;
                const uint32_t row_1 = m_block_idx * BLOCK_M + r_1;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    const uint32_t col = n_block_idx * BLOCK_N + i * 8 + col_idx * 2;
                    if (row_0 < shape_m and col < shape_n)
                        gmem_d_ptr[row_0 * shape_n + col] = cd_dtype_t(final_accum[i * 4 + 0]);
                    if (row_0 < shape_m and col + 1 < shape_n)
                        gmem_d_ptr[row_0 * shape_n + col + 1] = cd_dtype_t(final_accum[i * 4 + 1]);
                    if (row_1 < shape_m and col < shape_n)
                        gmem_d_ptr[row_1 * shape_n + col] = cd_dtype_t(final_accum[i * 4 + 2]);
                    if (row_1 < shape_m and col + 1 < shape_n)
                        gmem_d_ptr[row_1 * shape_n + col + 1] = cd_dtype_t(final_accum[i * 4 + 3]);
                }
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
