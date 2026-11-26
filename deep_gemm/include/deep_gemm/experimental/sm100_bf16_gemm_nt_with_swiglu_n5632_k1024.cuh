#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cuda_runtime.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::experimental {

// `d = silu(a @ b_gate.T) * (a @ b_up.T)`, where `b = [b_gate; b_up]` is a `2 * d.size(1)`-row matrix.
template <uint32_t kNumMulticast, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumAStagesOverride = 0, uint32_t kNumBStagesOverride = 0>
struct alignas(1024) SM100BF16GemmWithSwiGLUSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t BLOCK_K = 64;
    static constexpr bool kAsymAB = kNumAStagesOverride > 0 and kNumBStagesOverride > 0;
    static constexpr uint32_t kNumBStageRows = kNumMulticast == 1 ? 2 * BLOCK_N : BLOCK_N;
    static constexpr uint32_t kNumCD0Cols = (kNumMulticast == 2 and BLOCK_M == 64)
        ? (BLOCK_N == 40 ? 64 : 128) : (BLOCK_N == 24 ? 16 : (BLOCK_N == 40 ? 32 : 64));
    static constexpr uint32_t kNumCD1Cols = (kNumMulticast == 2 and BLOCK_M == 64)
        ? (2 * BLOCK_N - kNumCD0Cols) : (BLOCK_N <= 40 ? 8 : (BLOCK_N == 64 ? 0 : (BLOCK_N == 128 ? 64 : BLOCK_N - 64)));
    static constexpr uint32_t kNumCDBytes = BLOCK_M * (kNumCD0Cols + kNumCD1Cols) * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t kNumStageBytes = (BLOCK_M + kNumBStageRows) * BLOCK_K * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t kNumRawStages = (232448 - kNumCDBytes - 1024) / kNumStageBytes;
    static constexpr uint32_t kMax1SmStages = 9;
    static constexpr uint32_t kNumStages = (kNumMulticast == 1 and kNumRawStages > kMax1SmStages) ? kMax1SmStages : kNumRawStages;
    static constexpr uint32_t kNumAStages = kAsymAB ? kNumAStagesOverride : kNumStages;
    static constexpr uint32_t kNumBStages = kAsymAB ? kNumBStagesOverride : kNumStages;

    using a_stage_t = cutlass::bfloat16_t[BLOCK_M * BLOCK_K];
    using b_stage_t = cutlass::bfloat16_t[kNumBStageRows * BLOCK_K];

    DG_STATIC_ASSERT(kNumCDBytes % 1024 == 0 and sizeof(a_stage_t) % 1024 == 0 and
                     sizeof(b_stage_t) % 1024 == 0, "Shared memory stages must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumCDBytes + kNumAStages * sizeof(a_stage_t) + kNumBStages * sizeof(b_stage_t) + 1024 <= 232448,
                     "Shared memory capacity exceeded");

    alignas(1024) cutlass::bfloat16_t cd_storage[BLOCK_M * (kNumCD0Cols + kNumCD1Cols)];
    CUTLASS_DEVICE cutlass::bfloat16_t* cd(uint32_t stage_idx) {
        return cd_storage + stage_idx * (BLOCK_M * kNumCD0Cols);
    }
    a_stage_t a[kNumAStages];
    b_stage_t b[kNumBStages];
    Barrier full_barriers[kNumAStages];
    Barrier empty_barriers[kNumAStages];
    Barrier tmem_full_barriers[2];
    Barrier tmem_empty_barriers[2];
    alignas(16) uint4 clc_response;
    Barrier clc_full_barrier;
    Barrier clc_empty_barrier;
    uint32_t tmem_ptr;
    Barrier b_full_barriers[kAsymAB ? kNumBStages : 1];
    Barrier b_empty_barriers[kAsymAB ? kNumBStages : 1];
};

CUTLASS_DEVICE void swiglu_clc_try_cancel_multicast(void* response_ptr, void* mbarrier_ptr) {
    const uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_ptr);
    const uint32_t mbarrier_addr = cute::cast_smem_ptr_to_uint(mbarrier_ptr);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
        : : "r"(response_addr), "r"(mbarrier_addr) : "memory");
}

CUTLASS_DEVICE bool swiglu_clc_query_cancel(const void* response_ptr, uint32_t& first_cta_idx) {
    const uint32_t response_addr = cute::cast_smem_ptr_to_uint(response_ptr);
    uint32_t is_valid = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b128 clc_handle;\n\t"
        "ld.shared.b128 clc_handle, [%2];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, clc_handle;\n\t"
        "selp.u32 %1, 1, 0, p;\n\t"
        "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_handle;\n\t"
        "}\n"
        : "=r"(first_cta_idx), "=r"(is_valid) : "r"(response_addr) : "memory");
    cutlass::arch::fence_view_async_shared();
    return is_valid != 0;
}

template <uint32_t kNumSwizzleBytes>
CUTLASS_DEVICE uint8_t* get_swizzled_cd_ptr(void* smem_base, uint32_t epilogue_warp_idx,
                                            uint32_t lane_idx, uint32_t bank_group_idx) {
    constexpr uint32_t kNumGroupsPerRow = kNumSwizzleBytes / 16;
    const uint32_t shifted_idx = bank_group_idx + lane_idx * kNumGroupsPerRow;
    const uint32_t row_idx = shifted_idx >> 3;
    const uint32_t col_idx = (shifted_idx & 7) ^ (row_idx & (kNumGroupsPerRow - 1));
    return reinterpret_cast<uint8_t*>(smem_base) +
           epilogue_warp_idx * 32 * kNumSwizzleBytes + row_idx * 128 + col_idx * 16;
}

template <int kNumSMs, int kNumThreads, uint32_t kNumMulticast = 2,
          uint32_t BLOCK_M = 128, uint32_t BLOCK_N = 128,
          uint32_t kTailSplitLog2 = 0, bool kEvictLastA = true,
          uint32_t kNumAStagesOverride = 0, uint32_t kNumBStagesOverride = 0>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
sm100_bf16_gemm_nt_with_swiglu_n5632_k1024_impl(uint32_t shape_m,
                                                const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                                const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                                const __grid_constant__ cute::TmaDescriptor tensor_map_b_tail,
                                                const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
                                                const __grid_constant__ cute::TmaDescriptor tensor_map_cd_tail) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    constexpr bool kUse1SmM64 = kNumMulticast == 1 and BLOCK_M == 64;
    constexpr bool kUse2SmM64 = kNumMulticast == 2 and BLOCK_M == 64;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    using MMAOp = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
    using SharedStorage = SM100BF16GemmWithSwiGLUSharedStorage<
        kNumMulticast, BLOCK_M, BLOCK_N, kNumAStagesOverride, kNumBStagesOverride>;
    DG_STATIC_ASSERT(sizeof(SharedStorage) <= 232448, "SharedStorage exceeds SM100 per-CTA SMEM capacity");

    constexpr uint32_t BLOCK_K = 64, kShapeN = 2816, kShapeK = 1024, kNumSwizzleBytes = 128, UMMA_K = 16;
    constexpr bool kAsymAB = SharedStorage::kAsymAB;
    constexpr bool kUseCLC = kNumMulticast == 2 and BLOCK_N == 128;
    constexpr bool kMultiWaveStatic = not kUseCLC and kNumMulticast == 2 and (kAsymAB or BLOCK_N > 80) and not (BLOCK_M == 64 and BLOCK_N == 120);
    constexpr uint32_t kNumBStageRows = SharedStorage::kNumBStageRows;
    constexpr uint32_t kNumAStages = SharedStorage::kNumAStages, kNumBStages = SharedStorage::kNumBStages;
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumNonEpilogueThreads = 128, kNumEpilogueThreads = kNumThreads - kNumNonEpilogueThreads;
    constexpr uint32_t kNumEpilogueWGThreads = kUse1SmM64 ? 128 : BLOCK_M;
    constexpr bool kDualEpilogueWG = kNumEpilogueThreads == 2 * kNumEpilogueWGThreads;
    constexpr uint32_t kNumNBlocks = math::constexpr_ceil_div(kShapeN, BLOCK_N), kNumKBlocks = kShapeK / BLOCK_K;
    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
    constexpr uint32_t UMMA_M = kNumMulticast * BLOCK_M, UMMA_N = 2 * BLOCK_N;
    constexpr uint32_t kNumBBytes = kNumBStageRows * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kNumTmemStageCols = kUse2SmM64 ? 256 : UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumEpilogueStages * kNumTmemStageCols>();
    constexpr uint32_t kNumTailSubN = BLOCK_N >> kTailSplitLog2;
    constexpr uint32_t kNumClcEmptyArrivals = 1 + kNumMulticast + 1 + kNumMulticast * (kNumEpilogueThreads / 32);

    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Invalid multicast factor");
    DG_STATIC_ASSERT(not kAsymAB or (kNumMulticast == 2 and kMultiWaveStatic),
                     "Decoupled asymmetric AB pipeline specializes on 2-CTA multi-wave static modes");
    DG_STATIC_ASSERT(BLOCK_M == 64 or BLOCK_M == 128, "Unsupported BLOCK_M");
    DG_STATIC_ASSERT(BLOCK_N == 24 or BLOCK_N == 40 or BLOCK_N == 64 or BLOCK_N == 80 or
                     BLOCK_N == 88 or BLOCK_N == 96 or BLOCK_N == 104 or BLOCK_N == 120 or BLOCK_N == 128,
                     "Unsupported BLOCK_N");
    DG_STATIC_ASSERT(not kUse1SmM64 or (BLOCK_N == 24 and not kDualEpilogueWG),
                     "1-CTA M=64 mode specializes on BLOCK_N == 24 with 128 epilogue threads");
    DG_STATIC_ASSERT(not kUse2SmM64 or ((BLOCK_N == 40 or BLOCK_N == 104 or BLOCK_N == 120) and kTailSplitLog2 == 0 and kDualEpilogueWG),
                     "2x1SM M=64 mode requires BLOCK_N == 40, 104, or 120, kTailSplitLog2 == 0, and 128 epilogue threads");
    DG_STATIC_ASSERT(kTailSplitLog2 == 0 or (kUseCLC and kTailSplitLog2 <= 3), "Invalid tail split");
    DG_STATIC_ASSERT(not kUseCLC or not kDualEpilogueWG, "CLC mode uses a single epilogue warpgroup");
    DG_STATIC_ASSERT(kNumEpilogueThreads == kNumEpilogueWGThreads or kNumEpilogueThreads == 2 * kNumEpilogueWGThreads,
                     "Invalid thread count");
    DG_STATIC_ASSERT(kNumAStages <= 32 and kNumBStages <= 32 and kNumTmemCols <= 512, "Resource limit exceeded");
    DG_STATIC_ASSERT(kNumSMs > 0 and kNumSMs % kNumMulticast == 0, "Invalid SM count");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    const auto warp_idx = static_cast<uint32_t>(cutlass::canonical_warp_idx_sync());
    const auto lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = kNumMulticast > 1 ? static_cast<uint32_t>(cute::block_rank_in_cluster()) : 0u;
    const bool is_leader_cta = rank_idx == 0;

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
        if constexpr (kTailSplitLog2 > 0)
            cute::prefetch_tma_descriptor(&tensor_map_b_tail);
        if constexpr (kTailSplitLog2 > 1 or (BLOCK_N != 64 and BLOCK_N != 128))
            cute::prefetch_tma_descriptor(&tensor_map_cd_tail);
    }
    __syncwarp();

    if (warp_idx == 1) {
        if (lane_idx < kNumAStages) {
            smem.full_barriers[lane_idx].init(kNumMulticast);
            smem.empty_barriers[lane_idx].init(1);
        }
        cutlass::arch::fence_barrier_init();
        __syncwarp();
    } else if (warp_idx == 2) {
        if (lane_idx < kNumEpilogueStages) {
            smem.tmem_full_barriers[lane_idx].init(1);
            if constexpr (kUseCLC or kMultiWaveStatic)
                smem.tmem_empty_barriers[lane_idx].init(kNumMulticast * kNumEpilogueThreads);
        }
        if (kUseCLC and lane_idx == 0) {
            smem.clc_response = make_uint4(0u, 0u, 0u, 0u);
            smem.clc_full_barrier.init(1);
            smem.clc_empty_barrier.init(kNumClcEmptyArrivals);
        }
        cutlass::arch::fence_barrier_init();
        __syncwarp();
    } else if (kAsymAB and warp_idx == 3) {
        if constexpr (kAsymAB) {
            if (lane_idx < kNumBStages) {
                smem.b_full_barriers[lane_idx].init(kNumMulticast);
                smem.b_empty_barriers[lane_idx].init(1);
            }
            cutlass::arch::fence_barrier_init();
            __syncwarp();
        }
    }
    if constexpr (kNumMulticast > 1)
        comm::cluster_sync_with_relaxed_arrive();
    else
        __syncthreads();
    if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, &smem.tmem_ptr);
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);
    }
    if (warp_idx == 1 or warp_idx == 2)
        cutlass::arch::NamedBarrier::sync(64, 3);

    const uint32_t num_m_groups = math::ceil_div(shape_m, UMMA_M);
    const uint32_t num_tiles = num_m_groups * kNumNBlocks;
    const uint32_t cluster_idx = blockIdx.x / kNumMulticast;
    constexpr uint32_t kNumClusters = kNumSMs / kNumMulticast;
    constexpr uint32_t kSplitGain = (1u << kTailSplitLog2) - 1u;
    const uint32_t num_split_tiles = kTailSplitLog2 == 0 ? 0u : ((gridDim.x / kNumMulticast - num_tiles) / kSplitGain);
    const uint32_t num_full_tiles = num_tiles - num_split_tiles;

    // Order tiles N-fastest so each wave reuses a narrow M-strip of A in L2 while spreading B reads across L2 slices
    const auto decode_tile_coords = [&](uint32_t linear_tile_idx, uint32_t& m_idx, uint32_t& n_idx) {
        const uint32_t m_group_idx = linear_tile_idx / kNumNBlocks;
        const uint32_t n_block_idx = linear_tile_idx % kNumNBlocks;
        m_idx = (m_group_idx * kNumMulticast + rank_idx) * BLOCK_M;
        n_idx = n_block_idx * BLOCK_N;
    };
    const auto decode_tail_coords = [&](uint32_t tail_subtile_idx, uint32_t& m_idx, uint32_t& n_idx) {
        decode_tile_coords(num_full_tiles + (tail_subtile_idx >> kTailSplitLog2), m_idx, n_idx);
        n_idx += (tail_subtile_idx & ((1u << kTailSplitLog2) - 1u)) * kNumTailSubN;
    };
    const auto advance_clc_single_lane = [&](uint32_t& cur_tile_idx, uint32_t& clc_phase_idx) {
        smem.clc_full_barrier.wait(clc_phase_idx);
        uint32_t next_cta_idx;
        if (not swiglu_clc_query_cancel(&smem.clc_response, next_cta_idx))
            return false;
        smem.clc_empty_barrier.arrive(0u);
        cur_tile_idx = next_cta_idx / kNumMulticast;
        clc_phase_idx ^= 1;
        return true;
    };
    const auto advance_clc_consumer = [&](uint32_t& cur_tile_idx, uint32_t& clc_phase_idx) {
        const auto [is_leader, leader_lane_idx] = cute::elect_one_leader_sync();
        uint32_t next_tile_idx = 0xffffffffu;
        if (is_leader) {
            smem.clc_full_barrier.wait(clc_phase_idx);
            uint32_t next_cta_idx;
            if (swiglu_clc_query_cancel(&smem.clc_response, next_cta_idx)) {
                smem.clc_empty_barrier.arrive(0u);
                next_tile_idx = next_cta_idx / kNumMulticast;
            }
        }
        next_tile_idx = __shfl_sync(0xffffffff, next_tile_idx, leader_lane_idx);
        if (next_tile_idx == 0xffffffffu)
            return false;
        cur_tile_idx = next_tile_idx;
        clc_phase_idx ^= 1;
        return true;
    };

    if (warp_idx == 0) {
        if (cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            constexpr uint64_t kCacheHintA = kEvictLastA
                ? static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST)
                : static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
            constexpr uint64_t kCacheHintB = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST);
            const uint32_t n_offset_idx = rank_idx * kShapeN;
            uint32_t stage_idx = 0, phase_idx = 0;

            const uint32_t num_a_bytes = cute::min(shape_m, BLOCK_M) * BLOCK_K * sizeof(cutlass::bfloat16_t);
            const auto load_tile_k_loop = [&](const cute::TmaDescriptor& desc_b,
                                              uint32_t m_idx, uint32_t n_idx, uint32_t num_b_bytes) {
                if constexpr (kNumMulticast == 2) {
                    if (not is_leader_cta and m_idx >= shape_m) {
                        for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                            smem.empty_barriers[stage_idx].wait(phase_idx ^ 1);
                            if constexpr (not kAsymAB) {
                                const auto barrier_ptr = reinterpret_cast<uint64_t*>(&smem.full_barriers[stage_idx]);
                                cute::SM100_TMA_2SM_LOAD_2D::copy(&desc_b, barrier_ptr, kCacheHintB,
                                                                  smem.b[stage_idx], k_block_idx * BLOCK_K, n_idx + n_offset_idx);
                            }
                            smem.full_barriers[stage_idx].arrive(0u);
                            stage_idx = (stage_idx + 1 == kNumAStages) ? 0u : (stage_idx + 1);
                            phase_idx ^= (stage_idx == 0);
                        }
                        return;
                    }
                }
                const uint32_t num_expected_tx_bytes =
                    (m_idx + BLOCK_M < shape_m ? 2 * num_a_bytes : num_a_bytes) + (kAsymAB ? 0u : 2 * num_b_bytes);
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                    smem.empty_barriers[stage_idx].wait(phase_idx ^ 1);
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const auto barrier_ptr = reinterpret_cast<uint64_t*>(&smem.full_barriers[stage_idx]);
                    if constexpr (kNumMulticast == 1) {
                        cute::SM90_TMA_LOAD_2D::copy(&tensor_map_a, barrier_ptr, kCacheHintA, smem.a[stage_idx], k_idx, m_idx);
                        cute::SM90_TMA_LOAD_2D::copy(&desc_b, barrier_ptr, kCacheHintB, smem.b[stage_idx], k_idx, n_idx);
                        cute::SM90_TMA_LOAD_2D::copy(&desc_b, barrier_ptr, kCacheHintB,
                                                     smem.b[stage_idx] + BLOCK_N * BLOCK_K, k_idx, n_idx + kShapeN);
                        smem.full_barriers[stage_idx].arrive_and_expect_tx(num_a_bytes + kNumBBytes);
                    } else {
                        cute::SM100_TMA_2SM_LOAD_2D::copy(&tensor_map_a, barrier_ptr, kCacheHintA, smem.a[stage_idx], k_idx, m_idx);
                        if constexpr (not kAsymAB) {
                            cute::SM100_TMA_2SM_LOAD_2D::copy(&desc_b, barrier_ptr, kCacheHintB,
                                                              smem.b[stage_idx], k_idx, n_idx + n_offset_idx);
                        }
                        if (is_leader_cta) {
                            smem.full_barriers[stage_idx].arrive_and_expect_tx(num_expected_tx_bytes);
                        } else {
                            smem.full_barriers[stage_idx].arrive(0u);
                        }
                    }
                    stage_idx = (stage_idx + 1 == kNumAStages) ? 0u : (stage_idx + 1);
                    phase_idx ^= (stage_idx == 0);
                }
            };

            if constexpr (kUseCLC) {
                uint32_t cur_tile_idx = cluster_idx, clc_phase_idx = 0;
                while (true) {
                    uint32_t m_idx, n_idx;
                    if (kTailSplitLog2 == 0 or cur_tile_idx < num_full_tiles) {
                        decode_tile_coords(cur_tile_idx, m_idx, n_idx);
                        load_tile_k_loop(tensor_map_b, m_idx, n_idx, kNumBBytes);
                    } else {
                        constexpr uint32_t kNumTailBBytes = kNumTailSubN * BLOCK_K * sizeof(cutlass::bfloat16_t);
                        decode_tail_coords(cur_tile_idx - num_full_tiles, m_idx, n_idx);
                        load_tile_k_loop(tensor_map_b_tail, m_idx, n_idx, kNumTailBBytes);
                    }
                    if (not advance_clc_single_lane(cur_tile_idx, clc_phase_idx))
                        break;
                }
            } else if constexpr (kMultiWaveStatic) {
                for (uint32_t cur_tile_idx = cluster_idx; cur_tile_idx < num_tiles; cur_tile_idx += kNumClusters) {
                    uint32_t m_idx, n_idx;
                    decode_tile_coords(cur_tile_idx, m_idx, n_idx);
                    load_tile_k_loop(tensor_map_b, m_idx, n_idx, kNumBBytes);
                }
            } else if (cluster_idx < num_tiles) {
                DG_TRAP_ONLY_DEVICE_ASSERT(num_tiles <= kNumClusters);
                uint32_t m_idx, n_idx;
                decode_tile_coords(cluster_idx, m_idx, n_idx);
                load_tile_k_loop(tensor_map_b, m_idx, n_idx, kNumBBytes);
            }
        }
        __syncwarp();
    } else if (warp_idx == 1 and is_leader_cta) {
        if (cute::elect_one_sync()) {
            const auto instr_desc = cute::UMMA::make_instr_desc<
                cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
            const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
            const auto tail_instr_desc = cute::UMMA::make_instr_desc<
                cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                UMMA_M, (UMMA_N >> kTailSplitLog2), cute::UMMA::Major::K, cute::UMMA::Major::K>();
            const auto runtime_tail_instr_desc = cute::UMMA::make_runtime_instr_desc(tail_instr_desc);

            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kNumSwizzleBytes>(smem.a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, kNumBStageRows, BLOCK_K, kNumSwizzleBytes>(smem.b[0], 0, 0);
            const uint32_t a_init_lo = a_desc.lo, b_init_lo = b_desc.lo;

            const auto umma_arrive = [](cutlass::arch::ClusterTransactionBarrier& barrier) {
                if constexpr (kNumMulticast == 1) {
                    ptx::umma_arrive_no_elect(barrier);
                } else {
                    cutlass::arch::umma_arrive_multicast_2x1SM_no_elect(reinterpret_cast<uint64_t*>(&barrier), kCTAMask);
                }
            };

            uint32_t stage_idx = 0, phase_idx = 0, b_stage_idx = 0, b_phase_idx = 0, iter_idx = 0;
            const auto issue_mma_tile = [&](const auto cur_instr_desc) {
                const uint32_t accum_stage_idx = iter_idx % kNumEpilogueStages;
                const uint32_t accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
                const uint32_t tmem_base_idx = accum_stage_idx * kNumTmemStageCols;
                if constexpr (kUseCLC or kMultiWaveStatic) {
                    smem.tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
                    ptx::tcgen05_after_thread_sync();
                }

                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                    smem.full_barriers[stage_idx].wait(phase_idx);
                    if constexpr (kAsymAB)
                        smem.b_full_barriers[b_stage_idx].wait(b_phase_idx);
                    ptx::tcgen05_after_thread_sync();

                    const uint32_t a_stage_offset = stage_idx * ((BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t)) >> 4);
                    const uint32_t b_stage_offset = (kAsymAB ? b_stage_idx : stage_idx) * (kNumBBytes >> 4);
                    #pragma unroll
                    for (uint32_t mma_k_idx = 0; mma_k_idx < BLOCK_K / UMMA_K; ++ mma_k_idx) {
                        const uint32_t k_offset = (mma_k_idx * UMMA_K * sizeof(cutlass::bfloat16_t)) >> 4;
                        a_desc.lo = a_init_lo + a_stage_offset + k_offset;
                        b_desc.lo = b_init_lo + b_stage_offset + k_offset;
                        MMAOp::fma(a_desc, b_desc, tmem_base_idx, k_block_idx > 0 or mma_k_idx > 0, cur_instr_desc);
                    }
                    umma_arrive(smem.empty_barriers[stage_idx]);
                    if constexpr (kAsymAB)
                        umma_arrive(smem.b_empty_barriers[b_stage_idx]);
                    if (k_block_idx == kNumKBlocks - 1)
                        umma_arrive(smem.tmem_full_barriers[accum_stage_idx]);

                    stage_idx = (stage_idx + 1 == kNumAStages) ? 0u : (stage_idx + 1);
                    phase_idx ^= (stage_idx == 0);
                    if constexpr (kAsymAB) {
                        b_stage_idx = (b_stage_idx + 1 == kNumBStages) ? 0u : (b_stage_idx + 1);
                        b_phase_idx ^= (b_stage_idx == 0);
                    }
                }
                ++ iter_idx;
            };

            if constexpr (kUseCLC) {
                uint32_t cur_tile_idx = cluster_idx, clc_phase_idx = 0;
                while (true) {
                    const auto cur_instr_desc = (kTailSplitLog2 == 0 or cur_tile_idx < num_full_tiles)
                        ? runtime_instr_desc : runtime_tail_instr_desc;
                    bool has_next_tile = false;
                    if constexpr (not kEvictLastA or kTailSplitLog2 >= 2)
                        has_next_tile = advance_clc_single_lane(cur_tile_idx, clc_phase_idx);
                    issue_mma_tile(cur_instr_desc);
                    if constexpr (kEvictLastA and kTailSplitLog2 < 2)
                        has_next_tile = advance_clc_single_lane(cur_tile_idx, clc_phase_idx);
                    if (not has_next_tile)
                        break;
                }
            } else if constexpr (kMultiWaveStatic) {
                for (uint32_t cur_tile_idx = cluster_idx; cur_tile_idx < num_tiles; cur_tile_idx += kNumClusters)
                    issue_mma_tile(runtime_instr_desc);
            } else if (cluster_idx < num_tiles) {
                issue_mma_tile(runtime_instr_desc);
            }
        }
        __syncwarp();
    } else if (kUseCLC and warp_idx == 2 and is_leader_cta) {
        if (cute::elect_one_sync()) {
            uint32_t clc_phase_idx = 0;
            while (true) {
                smem.clc_empty_barrier.wait(clc_phase_idx ^ 1);
                smem.clc_full_barrier.arrive_and_expect_tx(16, 0u);
                smem.clc_full_barrier.arrive_and_expect_tx(16, 1u);
                swiglu_clc_try_cancel_multicast(&smem.clc_response, &smem.clc_full_barrier);
                smem.clc_full_barrier.wait(clc_phase_idx);

                uint32_t next_cta_idx;
                if (not swiglu_clc_query_cancel(&smem.clc_response, next_cta_idx))
                    break;
                smem.clc_empty_barrier.arrive(0u);
                clc_phase_idx ^= 1;
            }
        }
        __syncwarp();
    } else if (kAsymAB and warp_idx == 3) {
        if constexpr (kAsymAB) {
            if (cute::elect_one_sync()) {
                cudaGridDependencySynchronize();
                constexpr uint64_t kCacheHintB = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST);
                const uint32_t n_offset_idx = rank_idx * kShapeN;
                uint32_t b_stage_idx = 0, b_phase_idx = 0;

                for (uint32_t cur_tile_idx = cluster_idx; cur_tile_idx < num_tiles; cur_tile_idx += kNumClusters) {
                    uint32_t m_idx, n_idx;
                    decode_tile_coords(cur_tile_idx, m_idx, n_idx);
                    for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                        smem.b_empty_barriers[b_stage_idx].wait(b_phase_idx ^ 1);
                        const auto barrier_ptr = reinterpret_cast<uint64_t*>(&smem.b_full_barriers[b_stage_idx]);
                        cute::SM100_TMA_2SM_LOAD_2D::copy(&tensor_map_b, barrier_ptr, kCacheHintB,
                                                          smem.b[b_stage_idx], k_block_idx * BLOCK_K, n_idx + n_offset_idx);
                        if (is_leader_cta) {
                            smem.b_full_barriers[b_stage_idx].arrive_and_expect_tx(2 * kNumBBytes);
                        } else {
                            smem.b_full_barriers[b_stage_idx].arrive(0u);
                        }
                        b_stage_idx = (b_stage_idx + 1 == kNumBStages) ? 0u : (b_stage_idx + 1);
                        b_phase_idx ^= (b_stage_idx == 0);
                    }
                }
            }
            __syncwarp();
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32) {
        constexpr uint32_t kNumWarpsPerWG = BLOCK_M / 32, kNumTmemChunk = 32;
        const uint32_t epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;
        const uint32_t epilogue_wg_idx = kDualEpilogueWG ? (epilogue_warp_idx / kNumWarpsPerWG) : 0u;
        const uint32_t local_warp_idx = kDualEpilogueWG ? (epilogue_warp_idx % kNumWarpsPerWG) : epilogue_warp_idx;
        const uint32_t row_idx = local_warp_idx * 32 + lane_idx;
        uint32_t iter_idx = 0;

        const auto eval_swiglu_pair = [](uint32_t g0, uint32_t g1, uint32_t u0, uint32_t u1) {
            const auto gate = make_float2(__uint_as_float(g0), __uint_as_float(g1));
            const auto up = make_float2(__uint_as_float(u0), __uint_as_float(u1));
            // `d = silu(gate) * up` in fp32, rounded once at the end
            const auto denom = __fadd2_rn(make_float2(1.0f, 1.0f), math::fast_exp_ftz(make_float2(-gate.x, -gate.y)));
            const auto sigmoid = make_float2(math::fast_rcp(denom.x), math::fast_rcp(denom.y));
            const auto value = __fmul2_rn(__fmul2_rn(gate, sigmoid), up);
            return static_cast<uint32_t>(math::cast_into_bf16_and_pack(value.x, value.y));
        };
        const auto write_cd_chunk = [&]<uint32_t kNumChunks, uint32_t kChunkCols, uint32_t kNumSubtileSwizzleBytes>(
            void* smem_base, uint32_t chunk_idx, const uint32_t* packed) {
            constexpr uint32_t kNumGroupsPerChunk = kChunkCols / 8;
            if constexpr (kNumSubtileSwizzleBytes == 0) {
                constexpr uint32_t kNumRowBytes = kNumChunks * kChunkCols * sizeof(cutlass::bfloat16_t);
                const auto base_ptr = reinterpret_cast<uint8_t*>(smem_base) + row_idx * kNumRowBytes + chunk_idx * (kChunkCols * 2);
                #pragma unroll
                for (uint32_t group_idx = 0; group_idx < kNumGroupsPerChunk; ++ group_idx) {
                    const uint32_t packed_idx = group_idx * 4;
                    ptx::st_shared(base_ptr + group_idx * 16,
                                   packed[packed_idx], packed[packed_idx + 1], packed[packed_idx + 2], packed[packed_idx + 3]);
                }
            } else {
                #pragma unroll
                for (uint32_t group_idx = 0; group_idx < kNumGroupsPerChunk; ++ group_idx) {
                    const uint32_t packed_idx = group_idx * 4;
                    ptx::st_shared(get_swizzled_cd_ptr<kNumSubtileSwizzleBytes>(
                                       smem_base, local_warp_idx, lane_idx, chunk_idx * kNumGroupsPerChunk + group_idx),
                                   packed[packed_idx], packed[packed_idx + 1], packed[packed_idx + 2], packed[packed_idx + 3]);
                }
            }
        };
        const auto issue_tma_store = [&](const cute::TmaDescriptor& desc_cd, void* smem_base,
                                         uint32_t n_idx, uint32_t m_idx, uint32_t barrier_idx) {
            cutlass::arch::NamedBarrier::sync(kNumEpilogueWGThreads, barrier_idx);
            if (local_warp_idx == 0) {
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_2d(&desc_cd, smem_base, n_idx, m_idx, cute::TMA::CacheHintSm100::EVICT_FIRST);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        };
        const auto release_tmem_stage = [&](uint32_t accum_stage_idx) {
            if constexpr (kUseCLC or kMultiWaveStatic)
                smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
        };
        const auto store_subtile = [&]<uint32_t kNumChunks, uint32_t kNumSubtileSwizzleBytes,
                                       uint32_t kChunkCols = 16, uint32_t kNumStores = 1>(
            const cute::TmaDescriptor& desc_cd, uint32_t tmem_gate_idx, uint32_t tmem_up_idx,
            uint32_t m_idx, uint32_t n_idx, uint32_t accum_stage_idx) {
            constexpr uint32_t kNumTotalChunks = kNumChunks * kNumStores;
            DG_STATIC_ASSERT(kNumTotalChunks >= 1 and (kChunkCols == 8 or kChunkCols == 16 or kChunkCols == 32), "Invalid TMEM subtile config");
            constexpr uint32_t kNumBufs = kNumTotalChunks > 1 ? 2 : 1, kNumPairs = kChunkCols / 2;
            uint32_t gate_values[kNumBufs][kChunkCols], up_values[kNumBufs][kChunkCols];
            ptx::tmem_load_32dp32b<kChunkCols>(tmem_gate_idx, gate_values[0]);
            ptx::tmem_load_32dp32b<kChunkCols>(tmem_up_idx, up_values[0]);
            if ((kUseCLC or kMultiWaveStatic) and iter_idx > 0) {
                if (local_warp_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumEpilogueWGThreads, epilogue_wg_idx);
            }

            #pragma unroll
            for (uint32_t chunk_idx = 0; chunk_idx < kNumTotalChunks; ++ chunk_idx) {
                const uint32_t buf_idx = kNumTotalChunks > 1 ? (chunk_idx & 1) : 0u;
                cutlass::arch::fence_view_async_tmem_load();
                if constexpr (kNumTotalChunks > 1) {
                    if (chunk_idx + 1 < kNumTotalChunks) {
                        const uint32_t next_col_idx = (chunk_idx + 1) * kChunkCols;
                        ptx::tmem_load_32dp32b<kChunkCols>(tmem_gate_idx + next_col_idx, gate_values[buf_idx ^ 1]);
                        ptx::tmem_load_32dp32b<kChunkCols>(tmem_up_idx + next_col_idx, up_values[buf_idx ^ 1]);
                    }
                }
                if (chunk_idx + 1 == kNumTotalChunks) {
                    ptx::tcgen05_before_thread_sync();
                    release_tmem_stage(accum_stage_idx);
                }

                uint32_t packed[kNumPairs];
                #pragma unroll
                for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
                    packed[pair_idx] = eval_swiglu_pair(
                        gate_values[buf_idx][pair_idx * 2], gate_values[buf_idx][pair_idx * 2 + 1],
                        up_values[buf_idx][pair_idx * 2], up_values[buf_idx][pair_idx * 2 + 1]);
                }
                const uint32_t cd_stage_idx = kDualEpilogueWG ? epilogue_wg_idx : (chunk_idx / kNumChunks);
                asm volatile("" ::: "memory");
                write_cd_chunk.template operator()<kNumChunks, kChunkCols, kNumSubtileSwizzleBytes>(
                    smem.cd(cd_stage_idx), chunk_idx % kNumChunks, packed);
                if (chunk_idx % kNumChunks == kNumChunks - 1) {
                    issue_tma_store(desc_cd, smem.cd(cd_stage_idx),
                                    n_idx + (chunk_idx / kNumChunks) * (kNumChunks * kChunkCols),
                                    m_idx, epilogue_wg_idx);
                }
            }
        };

        // Unified 2-CTA BLOCK_M=64 (`UMMA_M=128`, 2x2 TMEM partition) epilogue
        const auto store_2sm_m64_tile = [&](uint32_t m_idx, uint32_t n_idx, uint32_t accum_stage_idx) {
            if (m_idx >= shape_m) {
                release_tmem_stage(accum_stage_idx);
                return;
            }
            if (kMultiWaveStatic and iter_idx > 0) {
                if (local_warp_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 2);
            }

            const uint32_t tmem_gate_idx = accum_stage_idx * kNumTmemStageCols;
            const auto get_exch_ptr = [&](void* smem_base, uint32_t u4_idx) {
                return reinterpret_cast<uint4*>(smem_base) + (u4_idx * 2 + local_warp_idx) * 32 + (lane_idx ^ (u4_idx & 7));
            };
            const auto exch_tmem_to_smem = [&]<uint32_t kNumChunks, uint32_t kChunkCols>(
                void* smem_base, uint32_t tmem_col_idx, bool release_tmem) {
                constexpr uint32_t kNumBufs = kNumChunks > 1 ? 2 : 1, kNumU4s = kChunkCols / 4;
                uint32_t tmp[kNumBufs][kChunkCols];
                ptx::tmem_load_32dp32b<kChunkCols>(tmem_col_idx, tmp[0]);
                #pragma unroll
                for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++ chunk_idx) {
                    const uint32_t buf_idx = kNumChunks > 1 ? (chunk_idx & 1) : 0u;
                    cutlass::arch::fence_view_async_tmem_load();
                    if constexpr (kNumChunks > 1) {
                        if (chunk_idx + 1 < kNumChunks)
                            ptx::tmem_load_32dp32b<kChunkCols>(tmem_col_idx + (chunk_idx + 1) * kChunkCols, tmp[buf_idx ^ 1]);
                    }
                    if (chunk_idx + 1 == kNumChunks and release_tmem) {
                        ptx::tcgen05_before_thread_sync();
                        release_tmem_stage(accum_stage_idx);
                    }
                    #pragma unroll
                    for (uint32_t u4_idx = 0; u4_idx < kNumU4s; ++ u4_idx) {
                        *get_exch_ptr(smem_base, chunk_idx * kNumU4s + u4_idx) = make_uint4(
                            tmp[buf_idx][u4_idx * 4], tmp[buf_idx][u4_idx * 4 + 1],
                            tmp[buf_idx][u4_idx * 4 + 2], tmp[buf_idx][u4_idx * 4 + 3]);
                    }
                }
            };
            const auto eval_exchanged_chunks = [&]<uint32_t kNumChunks, uint32_t kChunkCols, bool kGateInTmem>(
                void* smem_base, uint32_t tmem_col_idx, uint32_t (&packed)[kNumChunks][kChunkCols / 2]) {
                constexpr uint32_t kNumBufs = kNumChunks > 1 ? 2 : 1;
                constexpr uint32_t kNumU4s = kChunkCols / 4, kNumPairs = kChunkCols / 2;
                uint32_t tmem_vals[kNumBufs][kChunkCols], peer_vals[kChunkCols];
                ptx::tmem_load_32dp32b<kChunkCols>(tmem_col_idx, tmem_vals[0]);
                #pragma unroll
                for (uint32_t chunk_idx = 0; chunk_idx < kNumChunks; ++ chunk_idx) {
                    const uint32_t buf_idx = kNumChunks > 1 ? (chunk_idx & 1) : 0u;
                    cutlass::arch::fence_view_async_tmem_load();
                    if constexpr (kNumChunks > 1) {
                        if (chunk_idx + 1 < kNumChunks)
                            ptx::tmem_load_32dp32b<kChunkCols>(tmem_col_idx + (chunk_idx + 1) * kChunkCols, tmem_vals[buf_idx ^ 1]);
                    }
                    if (chunk_idx + 1 == kNumChunks) {
                        ptx::tcgen05_before_thread_sync();
                        release_tmem_stage(accum_stage_idx);
                    }
                    #pragma unroll
                    for (uint32_t u4_idx = 0; u4_idx < kNumU4s; ++ u4_idx) {
                        const uint4 v = *get_exch_ptr(smem_base, chunk_idx * kNumU4s + u4_idx);
                        peer_vals[u4_idx * 4] = v.x, peer_vals[u4_idx * 4 + 1] = v.y;
                        peer_vals[u4_idx * 4 + 2] = v.z, peer_vals[u4_idx * 4 + 3] = v.w;
                    }
                    if (chunk_idx + 1 == kNumChunks)
                        cutlass::arch::NamedBarrier::sync(kNumEpilogueWGThreads, epilogue_wg_idx);
                    #pragma unroll
                    for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
                        if constexpr (kGateInTmem) {
                            packed[chunk_idx][pair_idx] = eval_swiglu_pair(
                                tmem_vals[buf_idx][pair_idx * 2], tmem_vals[buf_idx][pair_idx * 2 + 1],
                                peer_vals[pair_idx * 2], peer_vals[pair_idx * 2 + 1]);
                        } else {
                            packed[chunk_idx][pair_idx] = eval_swiglu_pair(
                                peer_vals[pair_idx * 2], peer_vals[pair_idx * 2 + 1],
                                tmem_vals[buf_idx][pair_idx * 2], tmem_vals[buf_idx][pair_idx * 2 + 1]);
                        }
                    }
                    asm volatile("" ::: "memory");
                }
            };

            constexpr uint32_t kNumMainCols = BLOCK_N == 40 ? 32 : 64;
            constexpr uint32_t kNumMainChunks = kNumMainCols / kNumTmemChunk, kNumMainSwizzleBytes = kNumMainCols * 2;
            constexpr uint32_t kNumRemCols = BLOCK_N > kNumMainCols ? (BLOCK_N - kNumMainCols) : 8;
            constexpr uint32_t kRemChunkCols = 8, kNumRemChunks = kNumRemCols / kRemChunkCols;
            constexpr uint32_t kNumRemSwizzleBytes = BLOCK_N == 40 ? 16 : 0;

            const bool has_rem_cols = n_idx + kNumMainCols < kShapeN;
            if (epilogue_wg_idx == 1)
                exch_tmem_to_smem.template operator()<kNumMainChunks, kNumTmemChunk>(smem.cd(0), tmem_gate_idx, not has_rem_cols);
            else if (has_rem_cols)
                exch_tmem_to_smem.template operator()<kNumRemChunks, kRemChunkCols>(smem.cd(1), tmem_gate_idx + kNumMainCols, false);
            cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 2);

            if (epilogue_wg_idx == 0) {
                uint32_t packed_main[kNumMainChunks][kNumTmemChunk / 2];
                eval_exchanged_chunks.template operator()<kNumMainChunks, kNumTmemChunk, true>(smem.cd(0), tmem_gate_idx, packed_main);

                if (BLOCK_N == 104 or n_idx + kNumMainCols <= kShapeN) {
                    #pragma unroll
                    for (uint32_t chunk_idx = 0; chunk_idx < kNumMainChunks; ++ chunk_idx)
                        write_cd_chunk.template operator()<kNumMainChunks, kNumTmemChunk, kNumMainSwizzleBytes>(smem.cd(0), chunk_idx, packed_main[chunk_idx]);
                    issue_tma_store(tensor_map_cd, smem.cd(0), n_idx, m_idx, 0);
                } else if constexpr (BLOCK_N == 40) {
                    write_cd_chunk.template operator()<1, 8, 16>(smem.cd(0), 0, packed_main[0]);
                    write_cd_chunk.template operator()<1, 8, 16>(smem.cd(1), 0, packed_main[0] + 4);
                    issue_tma_store(tensor_map_cd_tail, smem.cd(0), n_idx, m_idx, 0);
                    issue_tma_store(tensor_map_cd_tail, smem.cd(1), n_idx + 8, m_idx, 1);
                } else {
                    const auto row_ptr = reinterpret_cast<uint8_t*>(smem.cd(0)) + row_idx * 112;
                    #pragma unroll
                    for (uint32_t group_idx = 0; group_idx < kNumRemChunks; ++ group_idx) {
                        const uint32_t chunk_idx = group_idx >> 2, half_idx = (group_idx & 3) * 4;
                        ptx::st_shared(row_ptr + group_idx * 16,
                                       packed_main[chunk_idx][half_idx], packed_main[chunk_idx][half_idx + 1],
                                       packed_main[chunk_idx][half_idx + 2], packed_main[chunk_idx][half_idx + 3]);
                    }
                    issue_tma_store(tensor_map_cd_tail, smem.cd(0), n_idx, m_idx, 0);
                }
            } else if (has_rem_cols) {
                uint32_t packed_rem[kNumRemChunks][kRemChunkCols / 2];
                eval_exchanged_chunks.template operator()<kNumRemChunks, kRemChunkCols, false>(
                    smem.cd(1), tmem_gate_idx + kNumMainCols, packed_rem);

                #pragma unroll
                for (uint32_t chunk_idx = 0; chunk_idx < kNumRemChunks; ++ chunk_idx) {
                    write_cd_chunk.template operator()<kNumRemChunks, kRemChunkCols, kNumRemSwizzleBytes>(
                        smem.cd(1), chunk_idx, packed_rem[chunk_idx]);
                }
                issue_tma_store(tensor_map_cd_tail, smem.cd(1), n_idx + kNumMainCols, m_idx, 1);
            }
        };
        const auto store_full_tile = [&](uint32_t m_idx, uint32_t n_idx, uint32_t accum_stage_idx) {
            if constexpr (kUse2SmM64) {
                store_2sm_m64_tile(m_idx, n_idx, accum_stage_idx);
                return;
            }
            if (m_idx >= shape_m) {
                release_tmem_stage(accum_stage_idx);
                return;
            }
            const uint32_t tmem_gate_idx = accum_stage_idx * kNumTmemStageCols;
            const uint32_t tmem_up_idx = tmem_gate_idx + BLOCK_N;
            if constexpr (kUse1SmM64) {
                const uint32_t row_base_idx = local_warp_idx * 16 + (lane_idx >> 2);
                const uint32_t col_pair_idx = lane_idx & 3;
                const auto write_sw32_pair = [&](void* base_ptr, uint32_t row_offset_idx, uint32_t half_idx, uint32_t packed_val) {
                    const uint32_t blk_idx = (row_offset_idx >> 2) * 8 + (((row_offset_idx & 3) * 2 + half_idx) ^ ((row_offset_idx >> 2) & 1));
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(base_ptr) + blk_idx * 16 + col_pair_idx * 4) = packed_val;
                };
                const auto write_tail_pair = [&](void* base_ptr, uint32_t row_offset_idx, uint32_t packed_val) {
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(base_ptr) + row_offset_idx * 16 + col_pair_idx * 4) = packed_val;
                };
                if (n_idx + 16 <= kShapeN) {
                    uint32_t gate_vals[3][4], up_vals[3][4];
                    #pragma unroll
                    for (uint32_t chunk_idx = 0; chunk_idx < 3; ++ chunk_idx) {
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                            tmem_gate_idx + chunk_idx * 8, gate_vals[chunk_idx][0], gate_vals[chunk_idx][1], gate_vals[chunk_idx][2], gate_vals[chunk_idx][3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                            tmem_up_idx + chunk_idx * 8, up_vals[chunk_idx][0], up_vals[chunk_idx][1], up_vals[chunk_idx][2], up_vals[chunk_idx][3]);
                    }
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::tcgen05_before_thread_sync();
                    release_tmem_stage(accum_stage_idx);
                    #pragma unroll
                    for (uint32_t chunk_idx = 0; chunk_idx < 2; ++ chunk_idx) {
                        write_sw32_pair(smem.cd(0), row_base_idx, chunk_idx,
                                        eval_swiglu_pair(gate_vals[chunk_idx][0], gate_vals[chunk_idx][1], up_vals[chunk_idx][0], up_vals[chunk_idx][1]));
                        write_sw32_pair(smem.cd(0), row_base_idx + 8, chunk_idx,
                                        eval_swiglu_pair(gate_vals[chunk_idx][2], gate_vals[chunk_idx][3], up_vals[chunk_idx][2], up_vals[chunk_idx][3]));
                    }
                    write_tail_pair(smem.cd(1), row_base_idx, eval_swiglu_pair(gate_vals[2][0], gate_vals[2][1], up_vals[2][0], up_vals[2][1]));
                    write_tail_pair(smem.cd(1), row_base_idx + 8, eval_swiglu_pair(gate_vals[2][2], gate_vals[2][3], up_vals[2][2], up_vals[2][3]));
                    cutlass::arch::NamedBarrier::sync(kNumEpilogueWGThreads, 0);
                    if (local_warp_idx == 0) {
                        if (cute::elect_one_sync()) {
                            cute::tma_store_fence();
                            ptx::tma_store_2d(&tensor_map_cd, smem.cd(0), n_idx, m_idx, cute::TMA::CacheHintSm100::EVICT_FIRST);
                            ptx::tma_store_2d(&tensor_map_cd_tail, smem.cd(1), n_idx + 16, m_idx, cute::TMA::CacheHintSm100::EVICT_FIRST);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                    }
                } else {
                    uint32_t gate_vals[4], up_vals[4];
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_gate_idx, gate_vals[0], gate_vals[1], gate_vals[2], gate_vals[3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_up_idx, up_vals[0], up_vals[1], up_vals[2], up_vals[3]);
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::tcgen05_before_thread_sync();
                    release_tmem_stage(accum_stage_idx);
                    write_tail_pair(smem.cd(0), row_base_idx, eval_swiglu_pair(gate_vals[0], gate_vals[1], up_vals[0], up_vals[1]));
                    write_tail_pair(smem.cd(0), row_base_idx + 8, eval_swiglu_pair(gate_vals[2], gate_vals[3], up_vals[2], up_vals[3]));
                    issue_tma_store(tensor_map_cd_tail, smem.cd(0), n_idx, m_idx, 0);
                }
            } else if constexpr (BLOCK_N == 64 or BLOCK_N == 128) {
                store_subtile.template operator()<64 / kNumTmemChunk, 128, kNumTmemChunk, BLOCK_N / 64>(
                    tensor_map_cd, tmem_gate_idx, tmem_up_idx, m_idx, n_idx, accum_stage_idx);
            } else {
                constexpr uint32_t kNumMainCols = BLOCK_N == 40 ? 32 : 64, kNumMainSwizzleBytes = kNumMainCols * 2;
                constexpr uint32_t kNumMainChunk = BLOCK_N <= 80 ? kNumTmemChunk : 16;
                constexpr uint32_t kNumRemCols = BLOCK_N - kNumMainCols;
                constexpr uint32_t kNumRemChunk = kNumRemCols >= kNumTmemChunk ? kNumTmemChunk : (kNumRemCols == 16 ? 16 : 8);
                constexpr uint32_t kNumRemSwizzleBytes = BLOCK_N == 88 ? 0 : kNumRemCols * 2;
                const bool has_main = n_idx + kNumMainCols <= kShapeN;
                if (epilogue_wg_idx == 0) {
                    if (has_main) {
                        store_subtile.template operator()<kNumMainCols / kNumMainChunk, kNumMainSwizzleBytes, kNumMainChunk>(
                            tensor_map_cd, tmem_gate_idx, tmem_up_idx, m_idx, n_idx, accum_stage_idx);
                    } else if constexpr (BLOCK_N == 40) {
                        store_subtile.template operator()<1, 16, 8>(
                            tensor_map_cd_tail, tmem_gate_idx, tmem_up_idx, m_idx, n_idx, accum_stage_idx);
                    } else {
                        release_tmem_stage(accum_stage_idx);
                    }
                } else {
                    const uint32_t rem_col_idx = has_main ? kNumMainCols : (BLOCK_N == 40 ? 8u : 0u);
                    store_subtile.template operator()<kNumRemCols / kNumRemChunk, kNumRemSwizzleBytes, kNumRemChunk>(
                        tensor_map_cd_tail, tmem_gate_idx + rem_col_idx, tmem_up_idx + rem_col_idx,
                        m_idx, n_idx + rem_col_idx, accum_stage_idx);
                }
            }
        };

        if constexpr (kUseCLC) {
            uint32_t cur_tile_idx = cluster_idx, clc_phase_idx = 0;
            while (true) {
                const uint32_t accum_stage_idx = iter_idx % kNumEpilogueStages;
                const uint32_t accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
                uint32_t m_idx, n_idx;
                const bool is_full_tile = kTailSplitLog2 == 0 or cur_tile_idx < num_full_tiles;
                if (is_full_tile)
                    decode_tile_coords(cur_tile_idx, m_idx, n_idx);
                else
                    decode_tail_coords(cur_tile_idx - num_full_tiles, m_idx, n_idx);
                const bool has_next_tile = advance_clc_consumer(cur_tile_idx, clc_phase_idx);
                smem.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
                ptx::tcgen05_after_thread_sync();
                if (is_full_tile) {
                    store_full_tile(m_idx, n_idx, accum_stage_idx);
                } else if (m_idx >= shape_m) {
                    smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                } else {
                    constexpr uint32_t kNumTailChunk = kNumTailSubN >= kNumTmemChunk ? kNumTmemChunk : 16;
                    constexpr uint32_t kNumTailChunks = kNumTailSubN / kNumTailChunk;
                    constexpr uint32_t kNumTailSwizzleBytes = kTailSplitLog2 <= 1 ? kNumSwizzleBytes : kNumTailSubN * 2;
                    const uint32_t tmem_gate_idx = accum_stage_idx * kNumTmemStageCols;
                    const uint32_t tmem_up_idx = tmem_gate_idx + kNumTailSubN;
                    store_subtile.template operator()<kNumTailChunks, kNumTailSwizzleBytes, kNumTailChunk>(
                        kTailSplitLog2 <= 1 ? tensor_map_cd : tensor_map_cd_tail,
                        tmem_gate_idx, tmem_up_idx, m_idx, n_idx, accum_stage_idx);
                }
                ++ iter_idx;
                if (not has_next_tile)
                    break;
            }
        } else if constexpr (kMultiWaveStatic) {
            for (uint32_t cur_tile_idx = cluster_idx; cur_tile_idx < num_tiles; cur_tile_idx += kNumClusters) {
                const uint32_t accum_stage_idx = iter_idx % kNumEpilogueStages;
                const uint32_t accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
                uint32_t m_idx, n_idx;
                decode_tile_coords(cur_tile_idx, m_idx, n_idx);
                smem.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
                ptx::tcgen05_after_thread_sync();
                store_full_tile(m_idx, n_idx, accum_stage_idx);
                ++ iter_idx;
            }
        } else if (cluster_idx < num_tiles) {
            uint32_t m_idx, n_idx;
            decode_tile_coords(cluster_idx, m_idx, n_idx);
            smem.tmem_full_barriers[0].wait(0);
            ptx::tcgen05_after_thread_sync();
            store_full_tile(m_idx, n_idx, 0);
        }
    }
    if constexpr (kNumMulticast > 1)
        comm::cluster_sync_with_relaxed_arrive();
    else
        __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100");
#endif
}

} // namespace deep_gemm::experimental

#pragma clang diagnostic pop
