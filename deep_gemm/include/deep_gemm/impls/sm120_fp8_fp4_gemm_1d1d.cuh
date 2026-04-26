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
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t = epilogue::transform::EpilogueIdentity>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm120_fp8_fp4_gemm_1d1d_impl(cd_dtype_t* gmem_d, const cd_dtype_t* gmem_c,
                             __nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                             int* grouped_layout,
                             cute::TmaDescriptor* tensor_map_buffer,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    namespace sm120_mma = mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = sm120_mma::FP8_MMA_M;   // 16
    static constexpr uint32_t MMA_N = sm120_mma::FP8_MMA_N;   // 8
    static constexpr uint32_t MMA_K = sm120_mma::FP8_MMA_K;   // 32

    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Only float or bfloat16 output supported");
    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Invalid TMA threads");
    DG_STATIC_ASSERT(kNumMathThreads % 32 == 0, "Invalid math threads");
    DG_STATIC_ASSERT(BLOCK_M % MMA_M == 0 and BLOCK_N % MMA_N == 0 and BLOCK_K % MMA_K == 0, "Invalid block dims");

    // Packed UE8M0: 4 K-blocks per int32 → load SF every N K-blocks
    static constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    static constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kMTilesPerWarp = BLOCK_M / kNumMathWarps / MMA_M;
    static constexpr uint32_t kNTiles = BLOCK_N / MMA_N;
    static constexpr uint32_t kKSteps = BLOCK_K / MMA_K;
    static constexpr uint32_t kAccumPerWarp = kMTilesPerWarp * kNTiles * sm120_mma::FP8_MMA_ACCUM;
    DG_STATIC_ASSERT(BLOCK_M == kNumMathWarps * kMTilesPerWarp * MMA_M, "M tiles must divide evenly");

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // SMEM sizes — no D buffer (register epilogue)
    static constexpr uint32_t SMEM_TM = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    static constexpr uint32_t SMEM_A  = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B  = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    // SMEM allocation (aligned for TMA destination)
    static constexpr uint32_t SMEM_SFA = math::constexpr_align(static_cast<uint32_t>(BLOCK_M * sizeof(int32_t)), 128u);
    static constexpr uint32_t SMEM_SFB = math::constexpr_align(static_cast<uint32_t>(BLOCK_N * sizeof(int32_t)), 128u);
    // Actual TMA transfer bytes (unaligned — this is what TMA reports to the mbarrier)
    static constexpr uint32_t TMA_SFA_BYTES = BLOCK_M * sizeof(int32_t);
    static constexpr uint32_t TMA_SFB_BYTES = BLOCK_N * sizeof(int32_t);
    static constexpr uint32_t SMEM_TMA_BYTES = SMEM_A + SMEM_B + TMA_SFA_BYTES + TMA_SFB_BYTES;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t group_id = lane_idx / 4;
    const uint32_t thread_id = lane_idx % 4;

    // Prefetch TMA descriptors
    if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
    }
    __syncwarp();

    // SMEM layout
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_tm_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer);
    auto smem_tm_b = smem_tm_a + 1;
    auto gmem_tm_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tm_b = gmem_tm_a + 1;

    auto smem_a = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SMEM_TM + s * SMEM_A);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SMEM_TM + kNumStages * SMEM_A + s * SMEM_B);
    });
    constexpr uint32_t SF_BASE = SMEM_TM + kNumStages * (SMEM_A + SMEM_B);
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

    // Barrier init
    if (warp_idx == kNumMathWarps + 1 and cute::elect_one_sync()) {
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

    // Wait for primary kernel completion (PDL)
    cudaGridDependencySynchronize();

    // Scheduler + pipeline
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs, 128u>(
        shape_m, shape_n, shape_k, grouped_layout);
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1};
    };
    uint32_t iter_idx = 0;

    if (warp_idx >= kNumMathWarps) {
        // ======== TMA PATH ========
        cutlass::arch::warpgroup_reg_dealloc<40>();

        if (warp_idx == kNumMathWarps and cute::elect_one_sync()) {
            uint32_t last_group_idx = kNumGroups;

            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                const uint32_t num_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t m_idx = scheduler.template get_global_idx<false>(shape_m, BLOCK_M, m_block_idx);
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                if (kGemmType == GemmType::KGroupedContiguous && last_group_idx != scheduler.current_group_idx) {
                    last_group_idx = scheduler.current_group_idx;
                    const uint64_t k_off = scheduler.current_k_cumsum;
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tm_a, gmem_a_ptr + k_off * shape_m);
                    ptx::tensor_map_replace_global_addr_in_smem(smem_tm_b, gmem_b_ptr + k_off * shape_n);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tm_a, scheduler.current_shape_k, scheduler.current_shape_k);
                    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tm_b, scheduler.current_shape_k, scheduler.current_shape_k);
                    *gmem_tm_a = *smem_tm_a;
                    *gmem_tm_b = *smem_tm_b;
                    ptx::tensor_map_release_gpu();
                    ptx::tensor_map_acquire_gpu(gmem_tm_a);
                    ptx::tensor_map_acquire_gpu(gmem_tm_b);
                }

                for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);
                    empty_barriers[stage]->wait(phase ^ 1);

                    auto& fb = *full_barriers[stage];
                    const uint32_t k_idx = kb * BLOCK_K;
                    const auto tma_a = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_a : &tensor_map_a_base);
                    const auto tma_b = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_b : &tensor_map_b_base);

                    // SF is packed UE8M0: K-coordinate divided by packing factor.
                    // Every K-block loads its own SF copy (may redundantly load the same packed int32).
                    const uint32_t sfa_k_idx = scheduler.current_sf_k_cumsum + kb / kNumSFAStagesPerLoad;
                    const uint32_t sfb_k_idx = scheduler.current_sf_k_cumsum + kb / kNumSFBStagesPerLoad;
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &fb, smem_sfa[stage], m_idx, sfa_k_idx, 1);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &fb, smem_sfb[stage], n_idx, sfb_k_idx, 1);
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tma_a, &fb, smem_a[stage], k_idx, m_idx, 1);
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(tma_b, &fb, smem_b[stage], k_idx, n_idx, 1);
                    fb.arrive_and_expect_tx(SMEM_TMA_BYTES);
                }
            }
        }
    } else {
        // ======== MATH PATH ========
        cutlass::arch::warpgroup_reg_alloc<232>();

        const uint32_t m_tile_base = warp_idx * kMTilesPerWarp;
        float accum[kAccumPerWarp];

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerWarp; ++i) accum[i] = 0.f;

            // K-block loop
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);
                full_barriers[stage]->wait(phase);

                // K-step loop (fully unrolled for compile-time byte_id)
                #pragma unroll
                for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                    const uint32_t sf_byte_a = kGranKA == 32 ? ks : (kb % kNumSFAStagesPerLoad);
                    const uint32_t sf_byte_b = kGranKB == 32 ? ks : (kb % kNumSFBStagesPerLoad);

                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        const uint32_t m_tile = m_tile_base + mt;

                        uint32_t a_frag[4];
                        sm120::load_a_fragment(a_frag, smem_a[stage], lane_idx, m_tile, ks, BLOCK_K, MMA_K);

                        // SFA: thread 0 in each quad → row group_id; thread 1 → row group_id+8
                        uint32_t sfa_data;
                        if (thread_id == 0)
                            sfa_data = sm120::load_sf(smem_sfa[stage], m_tile * MMA_M + group_id);
                        else if (thread_id == 1)
                            sfa_data = sm120::load_sf(smem_sfa[stage], m_tile * MMA_M + group_id + 8);
                        else
                            sfa_data = 0;

                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                            uint32_t b_frag[2];
                            sm120::load_b_fragment(b_frag, smem_b[stage], lane_idx, nt, ks, BLOCK_K, MMA_K);

                            // SFB: thread 0 in each quad → column group_id
                            uint32_t sfb_data = (thread_id == 0)
                                ? sm120::load_sf(smem_sfb[stage], nt * MMA_N + group_id)
                                : 0u;

                            const uint32_t ai = (mt * kNTiles + nt) * sm120_mma::FP8_MMA_ACCUM;
                            float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[ai]);
                            sm120_mma::fp8_mma_block_scaled(d, a_frag, b_frag, sfa_data, sfb_data, sf_byte_a, sf_byte_b);
                        }
                    }
                }

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }

            // ======== EPILOGUE: register → global ========
            const uint32_t m_base = scheduler.template get_global_idx<true>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t n_base = n_block_idx * BLOCK_N;

            #pragma unroll
            for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                #pragma unroll
                for (uint32_t nt = 0; nt < kNTiles; ++nt) {
                    const uint32_t ai = (mt * kNTiles + nt) * sm120_mma::FP8_MMA_ACCUM;

                    const uint32_t n_tile_base = n_base + nt * MMA_N;
                    const uint32_t n_transformed = epilogue_type_t::template apply_index_n<MMA_N>(n_tile_base);
                    const uint32_t row0 = m_base + (m_tile_base + mt) * MMA_M + group_id;
                    const uint32_t row1 = row0 + 8;
                    const uint32_t col  = n_transformed + thread_id * 2;

                    // Helper: read cd_dtype_t as float
                    auto read_cd = [&](const cd_dtype_t& x) -> float {
                        if constexpr (cute::is_same_v<cd_dtype_t, float>) return x;
                        else return static_cast<float>(x);
                    };
                    // Helper: write float as cd_dtype_t
                    auto write_cd = [&](cd_dtype_t& dst, float val) {
                        if constexpr (cute::is_same_v<cd_dtype_t, float>) dst = val;
                        else dst = cd_dtype_t(val);
                    };

                    // Write row0 (group_id)
                    if (row0 < shape_m and col < shape_n) {
                        const auto idx0 = static_cast<int64_t>(row0) * shape_n + col;
                        float v0 = accum[ai + 0], v1 = accum[ai + 1];
                        if constexpr (kWithAccumulation) {
                            v0 += read_cd(gmem_c[idx0]);
                            if (col + 1 < shape_n) v1 += read_cd(gmem_c[idx0 + 1]);
                        }
                        write_cd(gmem_d[idx0], v0);
                        if (col + 1 < shape_n) write_cd(gmem_d[idx0 + 1], v1);
                    }
                    // Write row1 (group_id + 8)
                    if (row1 < shape_m and col < shape_n) {
                        const auto idx1 = static_cast<int64_t>(row1) * shape_n + col;
                        float v2 = accum[ai + 2], v3 = accum[ai + 3];
                        if constexpr (kWithAccumulation) {
                            v2 += read_cd(gmem_c[idx1]);
                            if (col + 1 < shape_n) v3 += read_cd(gmem_c[idx1 + 1]);
                        }
                        write_cd(gmem_d[idx1], v2);
                        if (col + 1 < shape_n) write_cd(gmem_d[idx1 + 1], v3);
                    }
                }
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
