#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>

#include <deep_gemm/mma/sm120.cuh> 
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
CUTLASS_DEVICE
uint32_t get_sm120_swizzled_bank_group_idx(const uint32_t& offset, const uint32_t& lane_idx) {
    constexpr uint32_t kGroupsInSwizzleRange = kSwizzleMode / kSwizzleBase;

    const auto bank_group_idx = offset + lane_idx * kGroupsInSwizzleRange;

    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = kGroupsInSwizzleRange == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % kGroupsInSwizzleRange;

    return (row * kNumBankGroups + col) % kGroupsInSwizzleRange;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumMathThreads, uint32_t kNumTMAThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
sm120_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                               float* sqr_sum) {

#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(nv_bfloat16), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K");
    DG_STATIC_ASSERT(kSwizzleAMode == 128, "Invalid swizzle A mode");
    DG_STATIC_ASSERT(kSwizzleBMode == 128, "Invalid swizzle B mode");

    DG_STATIC_ASSERT(kSwizzleCDMode / sizeof(float) == BLOCK_N, "Invalid block N");
    DG_STATIC_ASSERT(kNumMathThreads == 128, "Invalid MMA threads");

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    auto smem_cd = reinterpret_cast<float*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + (SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers           = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);
    
    constexpr uint32_t kNumTMARegisters = 40; 

    cudaGridDependencySynchronize();

    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            const auto stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

            uint32_t m_idx = m_block_idx * BLOCK_M;
            uint32_t k_idx = k_offset + s * BLOCK_K;

            tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx);
            tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, 0);

            constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
            full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
        }

        for (uint32_t s = num_total_stages; s < num_total_stages + kNumStages; ++ s) {
            const auto stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);
        }
    } else if (warp_idx < kNumMathThreads / 32) {

        DG_STATIC_ASSERT(BLOCK_M == 64, "Invalid block M");
        DG_STATIC_ASSERT(BLOCK_K * sizeof(nv_bfloat16) == kSwizzleAMode, "Invalid block K");
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        constexpr uint32_t WGMMA_N = BLOCK_N;

        using MMASelector = mma::sm120::TF32MMASelector<WGMMA_N>;
        float accum[MMASelector::N_atoms * MMASelector::kNumAccumPerAtom] = {0};

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
        constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
        float sqr_sum_acc_0 = 0;
        float sqr_sum_acc_1 = 0;

        #pragma unroll kNumStages < 8 ? kNumStages : kNumStages / 2
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            // Using MMA atom dimensions mapped to registers
            constexpr uint32_t kNumRegPerMMA = MMASelector::M * MMASelector::K / 32; // 4 floats
            constexpr uint32_t kNumMMAPerBlockK = BLOCK_K / MMASelector::K;

            float a[kNumRegPerMMA * kNumMMAPerBlockK];
            DG_STATIC_ASSERT(kSwizzleAMode == 128, "Invalid swizzle A mode");

            uint32_t row = warp_idx * 16 + lane_idx / 4;

        #pragma unroll
        for (uint32_t i = 0; i < kNumLoads; ++ i) {
            uint32_t bank_group_idx = (row ^ i) % 8;
            nv_bfloat16* a_bf16_smem_ptr_upper = smem_a[stage_idx] + row * BLOCK_K + bank_group_idx * kNumElemsPerBankGroup;
            nv_bfloat16* a_bf16_smem_ptr_lower = smem_a[stage_idx] + (row + 8) * BLOCK_K + bank_group_idx * kNumElemsPerBankGroup;

            uint32_t elem_offset = lane_idx % 4;

            nv_bfloat16 a_bf16[kNumRegPerMMA];
            a_bf16[0] = a_bf16_smem_ptr_upper[elem_offset];
            a_bf16[1] = a_bf16_smem_ptr_lower[elem_offset];
            a_bf16[2] = a_bf16_smem_ptr_upper[elem_offset + 4];
            a_bf16[3] = a_bf16_smem_ptr_lower[elem_offset + 4];

            auto a_bf16x2_ptr = reinterpret_cast<nv_bfloat162*>(a_bf16);
            auto a_float2_ptr = reinterpret_cast<float2*>(a);

            float2 a_float2_0 = __bfloat1622float2(a_bf16x2_ptr[0]);
            float2 a_float2_1 = __bfloat1622float2(a_bf16x2_ptr[1]);

            a_float2_ptr[i * 2 + 0] = a_float2_0;
            a_float2_ptr[i * 2 + 1] = a_float2_1;

            sqr_sum_acc_0 += a_float2_0.x * a_float2_0.x + a_float2_1.x * a_float2_1.x;
            sqr_sum_acc_1 += a_float2_0.y * a_float2_0.y + a_float2_1.y * a_float2_1.y;
        }

            __syncwarp();
            if (s > 0)
                empty_barriers[(s - 1) % kNumStages]->arrive();

            constexpr int kNumElemsInSwizzleRange = 128 / sizeof(float);
            constexpr uint32_t kNumAtomsInSwizzleRange = kNumElemsInSwizzleRange / MMASelector::K;
            DG_STATIC_ASSERT(BLOCK_K % kNumElemsInSwizzleRange == 0, "Invalid block K");

            #pragma unroll
            for (int i = 0; i < BLOCK_K / kNumElemsInSwizzleRange; i++) {
                #pragma unroll
                for (int k = 0; k < kNumAtomsInSwizzleRange; k++) {

                    float* a_step = a + (i * kNumAtomsInSwizzleRange + k) * kNumRegPerMMA;

                    #pragma unroll
                    for (int n = 0; n < MMASelector::N_atoms; n++) {

                        uint32_t atom_n = lane_idx / 4;
                        uint32_t atom_k = lane_idx % 4;

                        uint32_t global_n = n * 8 + atom_n;
                        uint32_t global_k = (i * kNumAtomsInSwizzleRange + k) * 8 + atom_k;

                        uint32_t b_atom_idx = global_k / kNumElemsInSwizzleRange;
                        uint32_t b_atom_k = global_k % kNumElemsInSwizzleRange;
                        uint32_t atom_linear_idx_0 = global_n * kNumElemsInSwizzleRange + b_atom_k;
                        uint32_t atom_linear_idx_1 = atom_linear_idx_0 + 4;

                        uint32_t swizzle_xor = ((atom_linear_idx_0 >> 5) & 7) << 2;
                        uint32_t atom_swizzled_idx_0 = atom_linear_idx_0 ^ swizzle_xor;
                        uint32_t atom_swizzled_idx_1 = atom_linear_idx_1 ^ swizzle_xor;
                        uint32_t atom_base_idx = b_atom_idx * BLOCK_N * kNumElemsInSwizzleRange;

                        float b0 = smem_b[stage_idx][atom_base_idx + atom_swizzled_idx_0];
                        float b1 = smem_b[stage_idx][atom_base_idx + atom_swizzled_idx_1];

                        MMASelector::type::fma(
                            accum[n * 4 + 0], accum[n * 4 + 1], accum[n * 4 + 2], accum[n * 4 + 3],
                            a_step[0], a_step[1], a_step[2], a_step[3],
                            b0, b1,
                            accum[n * 4 + 0], accum[n * 4 + 1], accum[n * 4 + 2], accum[n * 4 + 3]
                        );
                    }
                }
            }
        }

        const auto& reduced_sum_0 = math::warp_reduce_sum<4>(sqr_sum_acc_0);
        const auto& reduced_sum_1 = math::warp_reduce_sum<4>(sqr_sum_acc_1);

        const auto& m_idx = m_block_idx * BLOCK_M + (warp_idx * BLOCK_M_PER_WARP + lane_idx / 4);
        if (lane_idx % 4 == 0) {
            if (m_idx < shape_m)
                sqr_sum[m_offset + m_idx] = reduced_sum_0;
            if (m_idx + 8 < shape_m)
                sqr_sum[m_offset + m_idx + 8] = reduced_sum_1;
        }
        
        __syncwarp();
        empty_barriers[(num_total_stages-1) % kNumStages]->arrive();

        uint32_t is_odd_pair = lane_idx / 2 % 2;
        uint32_t row_idx = lane_idx / 4;
        uint32_t reordered_pair_idx = is_odd_pair * 8 + row_idx;

        auto shifted_smem_ptr = reinterpret_cast<uint8_t*>(smem_cd) +
                                (warp_idx * BLOCK_M_PER_WARP + row_idx) * kSwizzleCDMode +  
                                lane_idx % 2 * 8;                                           

        #pragma unroll
        for (uint32_t i = 0; i < (kSwizzleCDMode / sizeof(float)) / 4; i += 2) {
            uint32_t bank_group_idx = get_sm120_swizzled_bank_group_idx<kSwizzleCDMode>(i + is_odd_pair, reordered_pair_idx);
            auto smem_ptr = shifted_smem_ptr + bank_group_idx * kNumBankGroupBytes; 

            auto values = reinterpret_cast<uint32_t*>(accum + i * 2);
            ptx::st_shared(smem_ptr, values[0], values[1]);
            ptx::st_shared(smem_ptr + 8 * kSwizzleCDMode, values[2], values[3]);
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(128, 1);

        if (warp_idx == 0 and cute::elect_one_sync()) {
            if constexpr (kNumSplits == 1) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M);
            } else {
                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M, k_split_idx);
            }
            cute::tma_store_arrive();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_120");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
