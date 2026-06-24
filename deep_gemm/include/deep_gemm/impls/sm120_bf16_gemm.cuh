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
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t = epilogue::transform::EpilogueIdentity,
          bool kBKMajor = true,
          uint32_t kNWarps_ = 1>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm120_bf16_gemm_impl(cd_dtype_t* gmem_d, const cd_dtype_t* gmem_c,
                     cutlass::bfloat16_t* gmem_a_ptr, cutlass::bfloat16_t* gmem_b_ptr,
                     int* grouped_layout,
                     cute::TmaDescriptor* tensor_map_buffer,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    namespace sm120_mma = mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = 16;
    static constexpr uint32_t MMA_N = 8;
    static constexpr uint32_t MMA_K = sm120_mma::BF16_MMA_K;
    static constexpr uint32_t MMA_ACCUM = 4;

    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Only float or bfloat16 output supported");
    DG_STATIC_ASSERT(kNumTMAThreads > 0, "SM120a always uses warp-specialized pipeline");
    DG_STATIC_ASSERT(kNumMathThreads % 32 == 0, "Invalid math threads");
    DG_STATIC_ASSERT(BLOCK_M % MMA_M == 0 and BLOCK_N % MMA_N == 0 and BLOCK_K % MMA_K == 0, "Invalid block dims");

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kNTiles = BLOCK_N / MMA_N;
    static constexpr uint32_t kKSteps = BLOCK_K / MMA_K;

    static constexpr uint32_t kNWarps = kNWarps_;
    static constexpr uint32_t kMWarps = kNumMathWarps / kNWarps;
    static constexpr uint32_t kMTilesPerWarp = BLOCK_M / kMWarps / MMA_M;
    static constexpr uint32_t kNTilesPerWarp = kNTiles / kNWarps;
    static constexpr uint32_t kAccumPerWarp = kMTilesPerWarp * kNTilesPerWarp * MMA_ACCUM;

    DG_STATIC_ASSERT(kNumMathWarps % kNWarps == 0, "kNWarps must divide kNumMathWarps");
    DG_STATIC_ASSERT(BLOCK_M == kMWarps * kMTilesPerWarp * MMA_M, "M tiles must divide evenly");
    DG_STATIC_ASSERT(kNTiles % kNWarps == 0, "N tiles must divide evenly among N warps");
    DG_STATIC_ASSERT(kMTilesPerWarp >= 1, "Need at least 1 M-tile per warp");
    DG_STATIC_ASSERT(kBKMajor or kNTilesPerWarp > 0, "Need at least one N tile per warp");
    DG_STATIC_ASSERT(not kBKMajor or kNTilesPerWarp % 2 == 0, "kNTilesPerWarp must be even for ldmatrix.x2 B loading");

    static constexpr uint32_t kTMARegisters = 40;
    static constexpr uint32_t kMMARegisters = 232;

    // SMEM D buffer for TMA store epilogue
    // Require BLOCK_N to be a multiple of the TMA store atom (kSwizzleCDMode / sizeof(cd_dtype_t))
    // to avoid partial sub-tile writes corrupting adjacent N-tiles in global memory
    static constexpr bool kUseTMAStoreEpilogue = sizeof(cd_dtype_t) <= 2 and kSwizzleCDMode > 0
        and BLOCK_N * sizeof(cd_dtype_t) >= kSwizzleCDMode
        and (BLOCK_N * sizeof(cd_dtype_t)) % kSwizzleCDMode == 0;
    static constexpr uint32_t SMEM_D = kUseTMAStoreEpilogue
        ? static_cast<uint32_t>((BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode) * kSwizzleCDMode * BLOCK_M)
        : 0u;
    static constexpr uint32_t kSwizzleCDShift = kSwizzleCDMode > 0 ? (7 - __builtin_ctz(kSwizzleCDMode)) : 0;
    static constexpr uint32_t kSwizzleCDMask = kSwizzleCDMode > 0 ? (kSwizzleCDMode / 16 - 1) : 0;
    static constexpr uint32_t kTMAStoreInnerDim = kSwizzleCDMode / sizeof(cd_dtype_t);
    static constexpr uint32_t kNumTMAStores = kUseTMAStoreEpilogue
        ? BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode : 0;

    static constexpr uint32_t SMEM_TM = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 2 : 0);
    static constexpr uint32_t kSMEMKBytes = BLOCK_K * 2;
    static constexpr uint32_t kSMEMNBytes = BLOCK_N * 2;
    static constexpr uint32_t SMEM_A  = BLOCK_M * kSMEMKBytes;
    static constexpr uint32_t SMEM_B  = BLOCK_N * BLOCK_K * 2;
    static constexpr uint32_t SMEM_TMA_BYTES = SMEM_A + SMEM_B;
    static constexpr uint32_t kLdmK = MMA_K * 2;

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_d_base = reinterpret_cast<cd_dtype_t*>(smem_buffer);

    constexpr uint32_t PIPE_BASE = SMEM_D;
    auto smem_a = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + s * SMEM_A);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + kNumStages * SMEM_A + s * SMEM_B);
    });
    constexpr uint32_t BAR_BASE = PIPE_BASE + kNumStages * (SMEM_A + SMEM_B);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + s * sizeof(Barrier));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + (kNumStages + s) * sizeof(Barrier));
    });

    constexpr uint32_t TM_BASE = BAR_BASE + 2 * kNumStages * sizeof(Barrier);
    auto smem_tm_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + TM_BASE);
    auto smem_tm_b = smem_tm_a + 1;
    auto gmem_tm_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tm_b = gmem_tm_a + 1;

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    if (warp_idx == 1 and cute::elect_one_sync()) {
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

    cudaGridDependencySynchronize();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs>(
        shape_m, shape_n, shape_k, grouped_layout);
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1};
    };

    // PRODUCER WARP GROUP (TMA warps, 40 regs)
    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kTMARegisters>();

        const bool is_tma_leader = (warp_idx == kNumMathWarps and lane_idx == 0);
        uint32_t tma_iter_idx = 0;

        if (is_tma_leader) {
            uint32_t last_group_idx = kNumGroups;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                if constexpr (kGemmType == GemmType::KGroupedContiguous) {
                    if (last_group_idx != scheduler.current_group_idx) {
                        last_group_idx = scheduler.current_group_idx;

                        const auto a_base = reinterpret_cast<const char*>(gmem_a_ptr);
                        const auto b_base = reinterpret_cast<const char*>(gmem_b_ptr);
                        const uint64_t a_offset = static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_m * 2;
                        const uint64_t b_offset = static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_n * 2;

                        ptx::tensor_map_replace_global_addr_in_smem(smem_tm_a, a_base + a_offset);
                        ptx::tensor_map_replace_global_addr_in_smem(smem_tm_b, b_base + b_offset);

                        const uint64_t new_stride = static_cast<uint64_t>(scheduler.current_shape_k * 2);
                        ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                            smem_tm_a, scheduler.current_shape_k, new_stride);
                        ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                            smem_tm_b, scheduler.current_shape_k, new_stride);

                        *gmem_tm_a = *smem_tm_a;
                        *gmem_tm_b = *smem_tm_b;
                        ptx::tensor_map_release_gpu();
                        ptx::tensor_map_acquire_gpu(gmem_tm_a);
                        ptx::tensor_map_acquire_gpu(gmem_tm_b);
                    }
                }

                const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
                const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
                constexpr bool kAGroupOffset = (kGemmType == GemmType::MGroupedMasked);
                const uint32_t m_idx = scheduler.template get_global_idx<kAGroupOffset>(shape_m, BLOCK_M, m_block_idx);
                constexpr bool kBGroupOffset = not (kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous);
                // For K-major B: group offset on outer=N. For MN-major B: group offset on outer=K.
                const uint32_t n_idx = scheduler.template get_global_idx<kBGroupOffset and kBKMajor>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                const uint32_t k_group_offset = (kBGroupOffset and not kBKMajor) ?
                    scheduler.template get_global_idx<true>(shape_k, 0, 0, m_block_idx) : 0;
                const auto tma_a_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_a : &tensor_map_a_base);
                const auto tma_b_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_b : &tensor_map_b_base);

                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = kIsBatchedMM ? scheduler.current_group_idx : 0;

                for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                    CUTE_TIE_DECL(get_pipeline(tma_iter_idx++), s, p);
                    empty_barriers[s]->wait(p ^ 1);

                    const uint32_t k_idx = kb * BLOCK_K;
                    tma::copy<kSMEMKBytes, BLOCK_M, kSwizzleAMode, char, kIsBatchedMM>(tma_a_desc, full_barriers[s], smem_a[s], k_idx, m_idx, 1, batch_idx);
                    if constexpr (kBKMajor) {
                        tma::copy<kSMEMKBytes, BLOCK_N, kSwizzleBMode, char, kIsBatchedMM>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    } else {
                        tma::copy<BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                            tma_b_desc, full_barriers[s], reinterpret_cast<cutlass::bfloat16_t*>(smem_b[s]),
                            n_idx, k_group_offset + k_idx, 1, batch_idx);
                    }
                    full_barriers[s]->arrive_and_expect_tx(SMEM_TMA_BYTES);
                }
            }
        }
    }
    // CONSUMER WARP GROUPS (math warps, 232 regs)
    else {
        cutlass::arch::warpgroup_reg_alloc<kMMARegisters>();

        const uint32_t math_warp_idx = warp_idx;
        const uint32_t group_id = lane_idx / 4;
        const uint32_t thread_id = lane_idx % 4;
        const uint32_t warp_m = math_warp_idx / kNWarps;
        const uint32_t warp_n = math_warp_idx % kNWarps;
        const uint32_t m_tile_base = warp_m * kMTilesPerWarp;
        const uint32_t n_tile_base = warp_n * kNTilesPerWarp;

        float accum[kAccumPerWarp];
        uint32_t iter_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerWarp; ++i) accum[i] = 0.f;

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);

                full_barriers[stage]->wait(phase);

                sm120::SwizzleContext<kSwizzleAMode> a_ctx[kMTilesPerWarp];
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    int a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + (m_tile_base + mt) * 16;
                    a_ctx[mt].init(a_row, kSMEMKBytes);
                }

                // B context: K-major uses SwizzleContext + ldmatrix, MN-major uses scalar loads
                [[maybe_unused]] sm120::SwizzleContext<kSwizzleBMode> b_ctx[kBKMajor ? kNTilesPerWarp : 1];
                if constexpr (kBKMajor) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                        b_ctx[nt].init(b_row, kSMEMKBytes);
                    }
                }

                uint32_t a_frag[2][kMTilesPerWarp][4];
                uint32_t b_tile[2][kNTilesPerWarp][2];

                auto load_kstep = [&](int buf, uint32_t ks) {
                    if constexpr (kBKMajor) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sm120::load_b_fragment_x2(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                    } else {
                        static constexpr uint32_t kBAtomElems = kSwizzleBMode > 0 ? kSwizzleBMode / 2 : BLOCK_N;
                        static constexpr uint32_t kBNumAtoms = (BLOCK_N + kBAtomElems - 1) / kBAtomElems;
                        static constexpr uint32_t kNTilesPerAtom = kBAtomElems / MMA_N;
                        const int b_k_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + ks * MMA_K;
                        #pragma unroll
                        for (uint32_t atom = 0; atom < kBNumAtoms; ++atom) {
                            sm120::SwizzleContext<kSwizzleBMode> b_mn_ctx;
                            b_mn_ctx.init(b_k_row, kSwizzleBMode > 0 ? kSwizzleBMode : kSMEMNBytes);
                            char* atom_base = smem_b[stage] + atom * BLOCK_K * kSwizzleBMode;
                            #pragma unroll
                            for (uint32_t nt_in = 0; nt_in < kNTilesPerAtom; ++nt_in) {
                                const uint32_t nt_global = atom * kNTilesPerAtom + nt_in;
                                if (nt_global >= n_tile_base and nt_global < n_tile_base + kNTilesPerWarp) {
                                    const uint32_t nt = nt_global - n_tile_base;
                                    void* addr = b_mn_ctx.addr(atom_base, nt_in * MMA_N * 2);
                                    sm120::ldmatrix_x2_trans(b_tile[buf][nt][0], b_tile[buf][nt][1], addr);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                        sm120::load_a_fragment(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                };

                auto compute_kstep = [&](int buf) {
                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                            sm120_mma::bf16_mma(d, a_frag[buf][mt], b_tile[buf][nt]);
                        }
                    }
                };

                load_kstep(0, 0);

                #pragma unroll
                for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                    int cur = ks & 1;
                    int nxt = (ks + 1) & 1;
                    if (ks < kKSteps - 1) {
                        load_kstep(nxt, ks + 1);
                    }
                    compute_kstep(cur);
                }

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }

            // Epilogue
            constexpr bool kEpilogueGroupOffset = not is_m_grouped_contiguous(kGemmType);
            const uint32_t m_base = scheduler.template get_global_idx<kEpilogueGroupOffset>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t n_base = n_block_idx * BLOCK_N;
            const uint32_t total_shape_m = (kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::MGroupedMasked)
                ? shape_m * kNumGroups : shape_m;

            auto read_cd = [&](const cd_dtype_t& x) -> float {
                if constexpr (cute::is_same_v<cd_dtype_t, float>) return x;
                else return static_cast<float>(x);
            };

            constexpr bool kIsBatchedEpilogue = (kGemmType == GemmType::Batched);
            // Batched D is [M, batch, N] physical layout: stride_m = kNumGroups * shape_n
            const int64_t cd_m_stride = kIsBatchedEpilogue
                ? static_cast<int64_t>(kNumGroups) * shape_n : static_cast<int64_t>(shape_n);
            const int64_t cd_batch_offset = kIsBatchedEpilogue
                ? static_cast<int64_t>(scheduler.current_group_idx) * shape_n : 0;

            if constexpr (kUseTMAStoreEpilogue) {
                if (math_warp_idx == 0 and lane_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t local_row0 = (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t local_row1 = local_row0 + 8;
                        const uint32_t local_col  = (n_tile_base + nt) * MMA_N + thread_id * 2;
                        float v0 = accum[ai + 0], v1 = accum[ai + 1];
                        float v2 = accum[ai + 2], v3 = accum[ai + 3];

                        if constexpr (kWithAccumulation) {
                            const uint32_t gr0 = m_base + local_row0, gr1 = m_base + local_row1;
                            const uint32_t gc = epilogue_type_t::template apply_index_n<MMA_N>(
                                n_base + (n_tile_base + nt) * MMA_N) + thread_id * 2;
                            if (gr0 < total_shape_m and gc + 1 < shape_n) {
                                const auto ci = cd_batch_offset + static_cast<int64_t>(gr0) * cd_m_stride + gc;
                                v0 += read_cd(gmem_c[ci]); v1 += read_cd(gmem_c[ci + 1]);
                            }
                            if (gr1 < total_shape_m and gc + 1 < shape_n) {
                                const auto ci = cd_batch_offset + static_cast<int64_t>(gr1) * cd_m_stride + gc;
                                v2 += read_cd(gmem_c[ci]); v3 += read_cd(gmem_c[ci + 1]);
                            }
                        }

                        const uint32_t sub_tile = local_col / kTMAStoreInnerDim;
                        const uint32_t col_in_sub = local_col % kTMAStoreInnerDim;
                        const uint32_t col_byte_in_sub = col_in_sub * sizeof(cd_dtype_t);
                        const uint32_t sw0 = col_byte_in_sub ^ (((local_row0 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                        const uint32_t sw1 = col_byte_in_sub ^ (((local_row1 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                        cd_dtype_t p0[2] = {cd_dtype_t(v0), cd_dtype_t(v1)};
                        cd_dtype_t p1[2] = {cd_dtype_t(v2), cd_dtype_t(v3)};
                        auto* smem_d_bytes = reinterpret_cast<char*>(smem_d_base);
                        const uint32_t sub_base = sub_tile * kSwizzleCDMode * BLOCK_M;
                        *reinterpret_cast<uint32_t*>(smem_d_bytes + sub_base + local_row0 * kSwizzleCDMode + sw0) =
                            *reinterpret_cast<const uint32_t*>(p0);
                        *reinterpret_cast<uint32_t*>(smem_d_bytes + sub_base + local_row1 * kSwizzleCDMode + sw1) =
                            *reinterpret_cast<const uint32_t*>(p1);
                    }
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                if (math_warp_idx == 0 and lane_idx == 0) {
                    const uint32_t batch_store_idx = kIsBatchedEpilogue ? scheduler.current_group_idx : 0;
                    #pragma unroll
                    for (uint32_t ts = 0; ts < kNumTMAStores; ++ts) {
                        auto* smem_src = reinterpret_cast<char*>(smem_d_base) + ts * kSwizzleCDMode * BLOCK_M;
                        const uint32_t n_store = epilogue_type_t::template apply_index_n<kTMAStoreInnerDim>(
                            n_base + ts * kTMAStoreInnerDim);
                        if constexpr (kIsBatchedEpilogue) {
                            if constexpr (kWithAccumulation)
                                cute::SM90_TMA_REDUCE_ADD_3D::copy(
                                    &tensor_map_cd, smem_src,
                                    n_store, m_base, batch_store_idx);
                            else
                                cute::SM90_TMA_STORE_3D::copy(
                                    &tensor_map_cd, smem_src,
                                    n_store, m_base, batch_store_idx);
                        } else {
                            cute::SM90_TMA_STORE_2D::copy(
                                &tensor_map_cd, smem_src,
                                n_store, m_base);
                        }
                    }
                    cute::tma_store_arrive();
                }
            } else {
                auto store_pair = [&](cd_dtype_t* ptr, float a, float b) {
                    if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                        *reinterpret_cast<float2*>(ptr) = make_float2(a, b);
                    } else {
                        ptr[0] = cd_dtype_t(a);
                        ptr[1] = cd_dtype_t(b);
                    }
                };

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t nt_global_base = n_base + (n_tile_base + nt) * MMA_N;
                        const uint32_t col = epilogue_type_t::template apply_index_n<MMA_N>(nt_global_base) + thread_id * 2;
                        const uint32_t row0 = m_base + (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t row1 = row0 + 8;

                        if (row0 < total_shape_m and col + 1 < shape_n) {
                            auto idx = cd_batch_offset + static_cast<int64_t>(row0) * cd_m_stride + col;
                            float v0 = accum[ai + 0], v1 = accum[ai + 1];
                            if constexpr (kWithAccumulation) { v0 += read_cd(gmem_c[idx]); v1 += read_cd(gmem_c[idx + 1]); }
                            store_pair(&gmem_d[idx], v0, v1);
                        }
                        if (row1 < total_shape_m and col + 1 < shape_n) {
                            auto idx = cd_batch_offset + static_cast<int64_t>(row1) * cd_m_stride + col;
                            float v2 = accum[ai + 2], v3 = accum[ai + 3];
                            if constexpr (kWithAccumulation) { v2 += read_cd(gmem_c[idx]); v3 += read_cd(gmem_c[idx + 1]); }
                            store_pair(&gmem_d[idx], v2, v3);
                        }
                    }
                }
            }
        } // persistent loop

        if constexpr (kUseTMAStoreEpilogue) {
            if (math_warp_idx == 0 and lane_idx == 0)
                cute::tma_store_wait<0>();
        }
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
