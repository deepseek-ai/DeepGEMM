#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/math.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
CUTLASS_DEVICE
uint32_t get_swizzled_smem_offset(const uint32_t& offset, const uint32_t& lane_idx) {
    // Calculate the index of the bank group to be written in the atom
    const auto bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);

    // Reshape the atom in another view and swizzle
    //  - original: `(BLOCK_N, kSwizzleMode / kSwizzleBase)`
    //  - new: `(BLOCK_N * kSwizzleMode / kSwizzleBase / kNumBankGroups, kNumBankGroups)`
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % (kSwizzleMode / kSwizzleBase);

    return row * 128 + col * kSwizzleBase;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumMMAThreads, uint32_t kNumCastAndReduceThreads,
          typename a_dtype_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumMMAThreads + kNumCastAndReduceThreads, 1)
sm100_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                                float* sqr_sum) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Configs
    constexpr uint32_t kNumCastStages = 2;
    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(a_dtype_t), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    constexpr auto kMajorA = cute::UMMA::Major::K;
    constexpr auto kMajorB = cute::UMMA::Major::K;
    DG_STATIC_ASSERT(kNumCastStages <= kNumStages, "Invalid cast stages");
    DG_STATIC_ASSERT(kSwizzleCDMode / sizeof(float) == BLOCK_N, "Invalid block N");
    DG_STATIC_ASSERT(kNumMMAThreads == 128, "Invalid MMA threads");
    DG_STATIC_ASSERT(cute::is_same_v<a_dtype_t, float> or cute::is_same_v<a_dtype_t, nv_bfloat16>, "Invalid data type");

    // Utils
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Real tensor memory size and offsets
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // Data on shared memory (layout as ordered below)
    // Fill D/A/B pointers
    auto smem_cd = reinterpret_cast<float*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + (SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers           = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_cast_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto empty_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto empty_cast_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_full_barrier       = barrier_start_ptr + kNumStages * 4;

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 4 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            full_cast_barriers[i]->init(kNumCastAndReduceThreads);
            empty_barriers[i]->init(1);
            empty_cast_barriers[i]->init(1);
        }
        tmem_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    DG_STATIC_ASSERT(kNumSplits <= kNumKBlocks, "Too many K splits");

    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Dispatch warps into different roles
    if (warp_idx < kNumMMAThreads / 32) {
        // TMA load warp
        if (warp_idx == 0 and cute::elect_one_sync()) {
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                // Wait consumer release
                const auto& stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

                // Compute offsets
                uint32_t m_idx = m_block_idx * BLOCK_M;
                uint32_t k_idx = k_offset + s * BLOCK_K;

                // Issue TMAs
                tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx);
                tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, 0);

                // Arrive at full barriers
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
            }
        }

        // MMA issue warp
        if (warp_idx == 1) {
            // Make instruction descriptor
            constexpr uint32_t UMMA_M = BLOCK_M;
            constexpr uint32_t UMMA_N = BLOCK_N;
            constexpr uint32_t UMMA_K = 32 / sizeof(float);
            constexpr uint32_t BLOCK_SWIZZLED_BK = kSwizzleBMode / sizeof(float);
            using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                                                   BLOCK_M, BLOCK_N, kMajorA, kMajorB>;
            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                                                          UMMA_M, UMMA_N, kMajorA, kMajorB>();
            const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto b_desc = mma::sm100::make_umma_desc<kMajorB, BLOCK_N, BLOCK_SWIZZLED_BK, kSwizzleBMode>(smem_b[0], 0, 0);
            const uint32_t& b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            // Checks for MMA instructions
            // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Launch MMAs
            // We can not unroll this part
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                // Wait TMA arrival
                const auto& stage_idx = s % kNumStages;
                const auto& cast_stage_idx = s % kNumCastStages;
                full_cast_barriers[cast_stage_idx]->wait((s / kNumCastStages) & 1);
                ptx::tcgen05_after_thread_sync();

                // Issue UMMA
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                    const uint32_t& atom_idx = (k * UMMA_K) / BLOCK_SWIZZLED_BK;
                    const uint32_t& in_atom_idx = (k * UMMA_K) % BLOCK_SWIZZLED_BK;
                    const uint32_t& offset = atom_idx * BLOCK_N * BLOCK_SWIZZLED_BK;
                    b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, float>(b_desc_base_lo, offset, in_atom_idx);
                    umma_t::fma(BLOCK_K * cast_stage_idx + k * UMMA_K, b_desc, BLOCK_K * kNumCastStages, s > 0 or k > 0, runtime_instr_desc);
                }

                // Commit
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_cast_barriers[cast_stage_idx]));
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
            }

            // Commit to epilogue threads
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barrier));
        }

        // TMA checks
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(float);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // Only support layout F (M = 64) and D (M = 128)
        DG_STATIC_ASSERT(BLOCK_M == 64 or BLOCK_M == 128, "Invalid block M");

        // Wait UMMA arrival
        tmem_full_barrier->wait(0);
        ptx::tcgen05_after_thread_sync();

        // Load from tensor memory into registers, and write shared memory with STSM
        DG_STATIC_ASSERT(kNumMMAThreads == 128, "Epilogue threads not enough");

        // Store into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_N / kNumElemsPerBankGroup; ++ i) {
            // Source and destination memory address
            uint32_t tmem_addr = BLOCK_K * kNumCastStages + i * kNumElemsPerBankGroup;
            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd) +                   // Base pointer
                            warp_idx * BLOCK_M / 4 * kSwizzleCDMode +               // Warp offset
                            get_swizzled_smem_offset<kSwizzleCDMode>(i, lane_idx);  // In-atom offset

            // Load from tensor memory, store into shared memory
            uint32_t values[kNumElemsPerBankGroup];
            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                values[0], values[1], values[2], values[3]);
            cutlass::arch::fence_view_async_tmem_load();
            if (BLOCK_M == 128 or (BLOCK_M == 64 and lane_idx < 16))
                ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
            if constexpr (BLOCK_M == 64)
                __syncwarp();
        }

        // Synchronize all threads and issue TMA
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumMMAThreads, 0);
        if (warp_idx == 0 and cute::elect_one_sync()) {
            if constexpr (kNumSplits == 1) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M);
            } else {
                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M, k_split_idx);
            }
            cute::tma_store_arrive();
        }

        // Deallocate tensor memory by warp 1
        // NOTES: warp 0 is waiting TMA store
        if (warp_idx == 1)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    } else {
        DG_STATIC_ASSERT(BLOCK_M == 64, "Invalid block M");
        DG_STATIC_ASSERT(kNumCastAndReduceThreads == 128, "Invalid cast-and-reduce threads");
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        const uint32_t sub_warp_idx = warp_idx - kNumMMAThreads / 32;

        DG_STATIC_ASSERT(BLOCK_K * sizeof(a_dtype_t) % kSwizzleAMode == 0, "Invalid block K");

        // Launch reductions
        float2 sum[2] = {float2{0, 0}, float2{0, 0}};
        #pragma unroll kNumStages
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrival
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            // Four lanes hold eight K elements for each of two rows in the `.16x256b` TMEM layout
            constexpr uint32_t kNumLoads = BLOCK_K / 8;
            using load_t = cute::conditional_t<cute::is_same_v<a_dtype_t, float>, float2, uint32_t>;
            load_t input_values[2][kNumLoads];
            if constexpr (cute::is_same_v<a_dtype_t, float>) {
                constexpr uint32_t kNumElemsPerSwizzle = kSwizzleAMode / sizeof(a_dtype_t);
                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < kNumLoads; ++ load_idx) {
                    #pragma unroll
                    for (uint32_t row_idx = 0; row_idx < 2; ++ row_idx) {
                        const auto m_idx = sub_warp_idx * BLOCK_M_PER_WARP + lane_idx / 4 + row_idx * 8;
                        const auto k_idx = load_idx * 8 + lane_idx % 4 * 2;
                        const auto smem_offset = k_idx / kNumElemsPerSwizzle * BLOCK_M * kSwizzleAMode +
                            get_swizzled_smem_offset<kSwizzleAMode>(k_idx % kNumElemsPerSwizzle / 4, m_idx) +
                            k_idx % 4 * sizeof(a_dtype_t);
                        const auto smem_ptr = reinterpret_cast<uint8_t*>(smem_a[stage_idx]) + smem_offset;
                        input_values[row_idx][load_idx] = ptx::ld_shared(reinterpret_cast<float2*>(smem_ptr));
                    }
                }
            } else {
                const auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_a[stage_idx]) +
                                           sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleAMode;
                DG_STATIC_ASSERT(kNumLoads % 2 == 0, "Invalid number of loads");
                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < kNumLoads; load_idx += 2) {
                    const auto smem_ptr = smem_base_ptr + get_swizzled_smem_offset<kSwizzleAMode>(load_idx + lane_idx / 16, lane_idx % 16);
                    ptx::SM90_U32x4_LDSM_N::copy(input_values[0][load_idx], input_values[1][load_idx],
                                                 input_values[0][load_idx + 1], input_values[1][load_idx + 1], smem_ptr);
                }
            }

            // Wait tensor memory empty
            const auto& cast_stage_idx = s % kNumCastStages;
            empty_cast_barriers[cast_stage_idx]->wait(((s / kNumCastStages) & 1) ^ 1);

            // Cast, reduce and store into tensor memory
            float2 fp32x2_values[2][kNumLoads];
            const auto& upper_view = reinterpret_cast<uint32_t*>(&fp32x2_values[0]);
            const auto& lower_view = reinterpret_cast<uint32_t*>(&fp32x2_values[1]);
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                #pragma unroll
                for (uint32_t u = 0; u < 2; ++ u) {
                    if constexpr (cute::is_same_v<a_dtype_t, float>) {
                        fp32x2_values[u][i] = input_values[u][i];
                    } else {
                        fp32x2_values[u][i] = ptx::cvt_bf16x2_to_fp32x2(*reinterpret_cast<nv_bfloat162*>(&input_values[u][i]));
                    }
                    sum[u] = __ffma2_rn(fp32x2_values[u][i], fp32x2_values[u][i], sum[u]);
                    if constexpr (cute::is_same_v<a_dtype_t, float>) {
                        fp32x2_values[u][i].x = ptx::cvt_rn_tf32_f32(fp32x2_values[u][i].x);
                        fp32x2_values[u][i].y = ptx::cvt_rn_tf32_f32(fp32x2_values[u][i].y);
                    }
                }

                // Store upper and lower part at the same time
                const auto idx_0 = i * 2, idx_1 = i * 2 + 1;
                cute::SM100_TMEM_STORE_16dp256b1x::copy(
                    upper_view[idx_0], upper_view[idx_1],
                    lower_view[idx_0], lower_view[idx_1],
                    cast_stage_idx * BLOCK_K + i * 8);
            }
            cutlass::arch::fence_view_async_tmem_store();
            cutlass::arch::fence_view_async_shared();

            // Arrive for issuing MMAs
            ptx::tcgen05_before_thread_sync();
            full_cast_barriers[cast_stage_idx]->arrive();
        }

        // Intra-warp reduction and write back
        #pragma unroll
        for (uint32_t u = 0; u < 2; ++ u) {
            const auto reduced_sum = math::warp_reduce_sum<4>(sum[u].x + sum[u].y);
            const auto m_idx = m_block_idx * BLOCK_M + sub_warp_idx * BLOCK_M_PER_WARP + lane_idx / 4 + u * 8;
            if (lane_idx % 4 == 0 and m_idx < shape_m)
                sqr_sum[m_offset + m_idx] = reduced_sum;
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
