#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/operators.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/layout/gemm.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::experimental {

namespace v41f_qkv_impl {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr uint32_t kHidden = 5120, kQRank = 1280, kKVRank = 512, kNumHeads = 64;
constexpr uint32_t kQDim = kNumHeads * kKVRank, kIndexQDim = 4096;
constexpr uint32_t kNumMulticast = 2, kRopeDim = 64;

enum class ReadyKind : uint32_t { Q, KV, QNorm };

struct SignalWorkspace {
    uint32_t* signals;
    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE uint32_t* get_grid_sync_count_ptr() const { return signals + kIndex; }
};

template <uint32_t kNumStages>
struct GemmBarriers {
    Barrier full_barriers[kNumStages];
    Barrier sf_full_barriers[kNumStages];
    Barrier empty_barriers[kNumStages];
    Barrier tmem_full_barriers[2];
    Barrier tmem_empty_barriers[2];
    Barrier tmem_overlap_barriers[2];

    CUTLASS_DEVICE void init() {
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumStages; ++ stage_idx) {
            full_barriers[stage_idx].init((1 + 32) * kNumMulticast);
            sf_full_barriers[stage_idx].init(1);
            empty_barriers[stage_idx].init(1);
        }
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < 2; ++ stage_idx) {
            tmem_full_barriers[stage_idx].init(1);
            tmem_empty_barriers[stage_idx].init(128 * kNumMulticast);
            tmem_overlap_barriers[stage_idx].init(128 * kNumMulticast);
        }
    }
};

// A separate cache line per counter avoids sharing active publication
// counters with the launch's initial grid synchronization counter.
CUTLASS_DEVICE uint32_t* ready_count(void* signals, const uint32_t& tile_idx, const ReadyKind& kind) {
    return static_cast<uint32_t*>(signals) + 32 + (tile_idx * 3 + static_cast<uint32_t>(kind)) * 32;
}

CUTLASS_DEVICE void wait_ready(void* signals, const uint32_t& tile_idx,
                               const ReadyKind& kind, const uint32_t& expected) {
    const auto count_ptr = ready_count(signals, tile_idx, kind);
    uint32_t current = 0;
    comm::wait_until([&]() { return (current = ptx::ld_acq(count_ptr)) == expected; }, [&]() {
        printf("QKV readiness timeout: block=%u, tile=%u, kind=%u, count=%u/%u\n",
               blockIdx.x, tile_idx, static_cast<uint32_t>(kind), current, expected);
    });
}

template <uint32_t kQOutDim>
struct alignas(1024) SharedStorage {
    static constexpr uint32_t kQBlockN = kQOutDim == kQDim ? 224 : 256;

    template <bool kIsQKVProjection>
    using GemmStorage = layout::SM100FP8FP4GemmSharedStorage<
        kIsQKVProjection ? 8 : 6, 2, 2,
        kIsQKVProjection ? 128 / kNumMulticast : 128,
        kIsQKVProjection ? 128 : kQBlockN / kNumMulticast,
        128, 128, 32, 128, kIsQKVProjection ? 128 : 256, 1,
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t>;

    union {
        GemmStorage<true> qkv;
        GemmStorage<false> q;
    };
    float norm_sums[2][4 * 5];
    GemmBarriers<8> qkv_barriers;
    GemmBarriers<6> q_barriers;
    Barrier reset_ready;
    uint32_t tmem_ptr;
};

template <uint32_t kNumMBlocks, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumSMs, bool kIsQKVProjection,
          uint32_t kSplitK = 1, typename storage_t, uint32_t kNumStages>
CUTLASS_DEVICE void gemm(
    storage_t& smem, GemmBarriers<kNumStages>& barriers, Barrier& reset_ready,
    const cute::TmaDescriptor& tensor_map_a, const cute::TmaDescriptor& tensor_map_b,
    const cute::TmaDescriptor& tensor_map_sfa, const cute::TmaDescriptor& tensor_map_sfb,
    const cute::TmaDescriptor* tensor_map_cd,
    const uint32_t& shape_m, float* partials, void* signals) {
    constexpr auto kMajor = cute::UMMA::Major::K;
    // QKV projection uses swapped MMA; Q expansion uses the regular layout.
    constexpr bool kSwapAB = kIsQKVProjection;
    constexpr uint32_t BLOCK_M = 128, BLOCK_N = kIsQKVProjection ? 128 : (SHAPE_N == kQDim ? 224 : 256), BLOCK_K = 128;
    constexpr uint32_t STORE_BLOCK_N = 32, kSwizzleCDMode = 64;
    constexpr uint32_t kNumNTiles = math::constexpr_ceil_div(SHAPE_N, BLOCK_N);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, 128u);
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t LOAD_BLOCK_M = kSwapAB ? BLOCK_M / kNumMulticast : BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = kSwapAB ? BLOCK_N : BLOCK_N / kNumMulticast;
    constexpr uint32_t kNumMTiles = kSwapAB ? kNumMBlocks : math::constexpr_ceil_div(kNumMBlocks, kNumMulticast);
    constexpr uint32_t kNumBaseTiles = kNumMTiles * kNumNTiles / (kSwapAB ? kNumMulticast : 1);
    constexpr uint32_t kNumTiles = kNumBaseTiles * kSplitK;
    constexpr bool kReuseSFA = not kIsQKVProjection and BLOCK_N <= 224 and (kNumSMs / kNumMulticast) % kNumMTiles == 0;
    constexpr uint32_t kNumSFATmemCols = kReuseSFA ? SHAPE_K / 32 : 4;
    constexpr uint32_t kNumSFTmemCols = kNumSFATmemCols + SF_BLOCK_N / 32;
    constexpr uint32_t kNumOverlappedTmemCols = cute::max(2 * UMMA_N + kNumSFTmemCols, 512u) - 512u;
    constexpr uint32_t kTmemStartColOfSFA = 2 * UMMA_N - kNumOverlappedTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kTmemStartColOfSFA + kNumSFATmemCols;
    // TMA zero-fills the final split's padded K blocks.
    constexpr uint32_t kSliceK = math::constexpr_ceil_div(SHAPE_K / BLOCK_K, kSplitK) * BLOCK_K;
    constexpr uint32_t kPaddedM = kNumMBlocks * BLOCK_M;

    DG_STATIC_ASSERT(kIsQKVProjection or kSplitK == 1, "Q-B uses unsplit K");
    DG_STATIC_ASSERT(kNumOverlappedTmemCols == 0 or (not kSwapAB and kNumOverlappedTmemCols <= STORE_BLOCK_N), "Invalid TMEM overlap");
    DG_STATIC_ASSERT(kTmemStartColOfSFB + SF_BLOCK_N / 32 <= 512, "TMEM overflow");
    DG_STATIC_ASSERT(kSliceK % BLOCK_K == 0, "Split K must contain whole scaling groups");

    const uint32_t rank_idx = cute::block_rank_in_cluster();
    const auto get_tile_mn = [&](uint32_t tile_idx) {
        tile_idx %= kNumBaseTiles;
        if constexpr (kSwapAB) {
            return cute::make_tuple(tile_idx / (kNumNTiles / kNumMulticast) * BLOCK_M,
                                    (tile_idx % (kNumNTiles / kNumMulticast) * kNumMulticast + rank_idx) * BLOCK_N);
        } else {
            return cute::make_tuple((tile_idx % kNumMTiles * kNumMulticast + rank_idx) * BLOCK_M,
                                    tile_idx / kNumMTiles * BLOCK_N);
        }
    };

    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    uint32_t pipeline_idx = 0;

    if (warp_idx == 0 and cute::elect_one_sync()) {
        for (uint32_t tile_idx = blockIdx.x / kNumMulticast; tile_idx < kNumTiles; tile_idx += kNumSMs / kNumMulticast) {
            const auto [m_idx, n_idx] = get_tile_mn(tile_idx);
            const bool load_sfa = not kReuseSFA or tile_idx == blockIdx.x / kNumMulticast;
            if constexpr (not kIsQKVProjection) {
                if (m_idx < shape_m and load_sfa) {
                    wait_ready(signals, m_idx / 128, ReadyKind::QNorm, cute::min(128u, shape_m - m_idx));
                    ptx::fence_async_global();
                }
            }
            uint32_t load_m_idx = m_idx;
            if constexpr (kSwapAB)
                load_m_idx += rank_idx * (math::align(cute::min(BLOCK_M, shape_m - m_idx), 16u) / 2);
            for (uint32_t k_idx = 0; k_idx < kSliceK; k_idx += BLOCK_K, ++ pipeline_idx) {
                const auto stage_idx = pipeline_idx % kNumStages;
                const auto phase = (pipeline_idx / kNumStages) & 1;
                barriers.empty_barriers[stage_idx].wait(phase ^ 1);

                const auto global_k_idx = k_idx + tile_idx / kNumBaseTiles * kSliceK;
                if (load_sfa) {
                    tma::copy<128, 1, 0>(&tensor_map_sfa, &barriers.sf_full_barriers[stage_idx],
                                         smem.sfa[stage_idx], m_idx, global_k_idx / BLOCK_K);
                }
                if (kIsQKVProjection or pipeline_idx >= 2) {
                    tma::copy<SF_BLOCK_N, 1, 0>(&tensor_map_sfb, &barriers.sf_full_barriers[stage_idx],
                                                smem.sfb[stage_idx], n_idx, global_k_idx / BLOCK_K);
                }
                barriers.sf_full_barriers[stage_idx].arrive_and_expect_tx(((load_sfa ? 128 : 0) + SF_BLOCK_N) * 4);

                tma::copy<BLOCK_K, LOAD_BLOCK_M, 128>(&tensor_map_a, &barriers.full_barriers[stage_idx],
                                                      smem.a[stage_idx], global_k_idx, load_m_idx, 2);
                if (kIsQKVProjection or pipeline_idx >= 2) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, 128>(&tensor_map_b, &barriers.full_barriers[stage_idx],
                                                          smem.b[stage_idx], global_k_idx,
                                                          n_idx + (kSwapAB ? 0 : rank_idx * LOAD_BLOCK_N), 2);
                }
                if (rank_idx == 0)
                    barriers.full_barriers[stage_idx].arrive_and_expect_tx((LOAD_BLOCK_M + LOAD_BLOCK_N) * BLOCK_K * kNumMulticast);
                else
                    barriers.full_barriers[stage_idx].arrive(0u);
            }
        }
    } else if (warp_idx == 1 and rank_idx == 0) {
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t,
            128 * kNumMulticast, UMMA_N, kMajor, kMajor>();
        DG_STATIC_ASSERT(kNumStages <= 32, "Too many descriptor stages");
        auto a_desc = mma::sm100::make_umma_desc<kMajor, LOAD_BLOCK_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajor, LOAD_BLOCK_N, BLOCK_K, 128>(smem.b[0], 0, 0);
        const uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(smem.a[0]) / 16 : 0;
        const uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(smem.b[0]) / 16 : 0;
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        uint32_t iter_idx = 0;
        for (uint32_t tile_idx = blockIdx.x / kNumMulticast; tile_idx < kNumTiles; tile_idx += kNumSMs / kNumMulticast, ++ iter_idx) {
            const auto accum_stage_idx = iter_idx % 2;
            const auto accum_phase_idx = (iter_idx / 2) & 1;
            if constexpr (kSwapAB) {
                const auto m_idx = cute::get<0>(get_tile_mn(tile_idx));
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, math::align(cute::min(BLOCK_M, shape_m - m_idx), 16u));
            }
            barriers.tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            for (uint32_t k_idx = 0; k_idx < kSliceK; k_idx += BLOCK_K, ++ pipeline_idx) {
                const auto stage_idx = pipeline_idx % kNumStages;
                const auto phase = (pipeline_idx / kNumStages) & 1;
                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                const auto sfa_col = kTmemStartColOfSFA + (kReuseSFA ? k_idx / 32 : 0);
                barriers.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();

                if (cute::elect_one_sync()) {
                    if (not kReuseSFA or iter_idx == 0) {
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem.sfa[stage_idx]);
                        cute::SM100_UTCCP_4x32dp128bit_2cta::copy(sf_desc, sfa_col);
                    }
                    #pragma unroll
                    for (uint32_t sf_idx = 0; sf_idx < SF_BLOCK_N / 128; ++ sf_idx) {
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem.sfb[stage_idx] + sf_idx * 128);
                        cute::SM100_UTCCP_4x32dp128bit_2cta::copy(sf_desc, kTmemStartColOfSFB + sf_idx * 4);
                    }
                }
                if constexpr (kNumOverlappedTmemCols > 0) {
                    if (k_idx == 0 and iter_idx > 0) {
                        __syncwarp();
                        ptx::tcgen05_before_thread_sync();
                        barriers.tmem_overlap_barriers[(iter_idx - 1) % 2].wait(((iter_idx - 1) / 2) & 1);
                        ptx::tcgen05_after_thread_sync();
                    }
                }
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < 4; ++ umma_k_idx) {
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajor, LOAD_BLOCK_M, 128, cutlass::float_e4m3_t>(
                            a_desc_base_lo, 0, umma_k_idx * 32);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajor, LOAD_BLOCK_N, 128, cutlass::float_e4m3_t>(
                            b_desc_base_lo, 0, umma_k_idx * 32);
                        const auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(
                            instr_desc, umma_k_idx, umma_k_idx);
                        ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                            kSwapAB ? b_desc : a_desc, kSwapAB ? a_desc : b_desc,
                            accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols), k_idx > 0 or umma_k_idx > 0, runtime_instr_desc,
                            kSwapAB ? kTmemStartColOfSFB : sfa_col,
                            kSwapAB ? sfa_col : kTmemStartColOfSFB);
                    }
                }
                cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&barriers.empty_barriers[stage_idx]), 3);
                if (k_idx + BLOCK_K == kSliceK)
                    cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&barriers.tmem_full_barriers[accum_stage_idx]), 3);
                __syncwarp();
            }
        }
    } else if (warp_idx == 2) {
        const auto utccp_required_smem_warp_transpose = [&](uint32_t* smem_ptr) {
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };
        for (uint32_t tile_idx = blockIdx.x / kNumMulticast; tile_idx < kNumTiles; tile_idx += kNumSMs / kNumMulticast) {
            for (uint32_t k_idx = 0; k_idx < kSliceK; k_idx += BLOCK_K, ++ pipeline_idx) {
                const auto stage_idx = pipeline_idx % kNumStages;
                const auto phase = (pipeline_idx / kNumStages) & 1;
                barriers.sf_full_barriers[stage_idx].wait(phase);
                if (not kReuseSFA or tile_idx == blockIdx.x / kNumMulticast)
                    utccp_required_smem_warp_transpose(smem.sfa[stage_idx]);
                #pragma unroll
                for (uint32_t sf_idx = 0; sf_idx < SF_BLOCK_N / 128; ++ sf_idx)
                    utccp_required_smem_warp_transpose(smem.sfb[stage_idx] + sf_idx * 128);
                cutlass::arch::fence_view_async_shared();
                barriers.full_barriers[stage_idx].arrive(0u);
            }
        }
    } else if (warp_idx == 3 and kIsQKVProjection) {
        for (uint32_t flag_idx = blockIdx.x * 32 + lane_idx; flag_idx < 3 * kNumMBlocks; flag_idx += kNumSMs * 32)
            static_cast<uint32_t*>(signals)[32 + flag_idx * 32] = 0;
        comm::grid_sync<kNumSMs>(SignalWorkspace(static_cast<uint32_t*>(signals)),
                                 blockIdx.x, lane_idx, []() { __syncwarp(); });
        if (cute::elect_one_sync())
            reset_ready.arrive();
    } else if (warp_idx >= 4 and warp_idx < 8) {
        if constexpr (kIsQKVProjection)
            reset_ready.wait(0);
        const auto bf16_args = EpilogueOperatorArgs();
        auto bf16_op = epilogue::operators::Identity(bf16_args);
        uint32_t iter_idx = 0, tma_stage_idx = 0;
        for (uint32_t tile_idx = blockIdx.x / kNumMulticast; tile_idx < kNumTiles; tile_idx += kNumSMs / kNumMulticast, ++ iter_idx) {
            const auto accum_stage_idx = iter_idx % 2;
            const auto accum_phase_idx = (iter_idx / 2) & 1;
            const auto [m_idx, n_idx] = get_tile_mn(tile_idx);
            barriers.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            if constexpr (kIsQKVProjection) {
                const auto effective_m = math::align(cute::min(BLOCK_M, shape_m - m_idx), 16u);
                const auto row_base_idx = static_cast<uint64_t>(tile_idx / kNumBaseTiles) * kPaddedM + m_idx;
                const auto col_idx = n_idx + (warp_idx - 4) * 32 + lane_idx;
                for (uint32_t row_idx = 0; row_idx < effective_m; row_idx += 16) {
                    uint32_t values[16];
                    ptx::tmem_load_32dp32b<16>(accum_stage_idx * BLOCK_M + row_idx, values);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (row_idx + 16 == effective_m) {
                        ptx::tcgen05_before_thread_sync();
                        barriers.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                    }
                    #pragma unroll
                    for (uint32_t value_idx = 0; value_idx < 16; ++ value_idx) {
                        partials[(row_base_idx + row_idx + value_idx) * SHAPE_N + col_idx] =
                            __uint_as_float(values[value_idx]);
                    }
                }
                cutlass::arch::NamedBarrier::sync(128, 1);
                if (threadIdx.x == 128)
                    ptx::atomic_add_rel(ready_count(signals, m_idx / 128, n_idx < kQRank ? ReadyKind::Q : ReadyKind::KV), 1u);
            } else {
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, BLOCK_M, STORE_BLOCK_N, kSwizzleCDMode, 2, 128, kNumOverlappedTmemCols, GemmType::Normal, false>(
                    smem, tma_stage_idx, accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols), m_idx, n_idx, 0,
                    false, warp_idx - 4, lane_idx, bf16_op,
                    kNumOverlappedTmemCols > 0 and accum_stage_idx == 0,
                    &barriers.tmem_overlap_barriers[accum_stage_idx],
                    &barriers.tmem_empty_barriers[accum_stage_idx], *tensor_map_cd);
            }
        }
    }
    if constexpr (kIsQKVProjection)
        reset_ready.wait(0);
    comm::cluster_sync_with_relaxed_arrive();
}

// Five Q warps and one KV warp cooperate per row.
template <uint32_t kNumMBlocks, uint32_t kNumSMs, uint32_t kSplitK, uint32_t kNumRowWarps>
CUTLASS_DEVICE void normalize(
    const float* qkv_partials, const nv_bfloat16* q_weight,
    const nv_bfloat16* kv_weight, const int* positions,
    const float* cos_sin, const int* slots,
    uint8_t* cache, const uint32_t& shape_m,
    const uint32_t& num_cache_slots, const uint32_t& cache_block_size, const uint32_t& cache_stride,
    uint8_t* q_a, uint32_t* q_a_sf, void* signals, float norm_sums[2][4 * 5], const float& eps) {
    constexpr uint32_t kPaddedM = kNumMBlocks * 128;
    constexpr uint32_t kReadyM = 128;
    constexpr uint32_t NORM_BLOCK_N = 256;
    constexpr uint32_t kNumQParts = kQRank / NORM_BLOCK_N;
    constexpr uint32_t kNumQWarps = kNumQParts * kNumRowWarps;
    constexpr uint32_t kNumQTiles = kQRank / 128, kNumKVTiles = kKVRank / 128;

    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    if (warp_idx >= kNumQWarps + kNumRowWarps)
        return;

    const uint32_t sf_stride = math::align(shape_m, 4u);
    const auto load_intermediate = [&]<size_t kNumValues>(const uint32_t& row_idx, const uint32_t& col_idx,
                                                          float (&values)[kNumValues]) {
        #pragma unroll
        for (uint32_t chunk_idx = 0; chunk_idx < kNumValues / 8; ++ chunk_idx) {
            float sum[8] = {};
            #pragma unroll
            for (uint32_t split_idx = 0; split_idx < kSplitK; ++ split_idx) {
                const auto packed = *reinterpret_cast<const ptx::longlong4_t*>(
                    qkv_partials + (static_cast<uint64_t>(split_idx) * kPaddedM + row_idx) * (kQRank + kKVRank) + col_idx + chunk_idx * 8);
                const auto values_f32 = reinterpret_cast<const float*>(&packed);
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < 8; ++ value_idx)
                    sum[value_idx] += values_f32[value_idx];
            }
            #pragma unroll
            for (uint32_t value_idx = 0; value_idx < 8; ++ value_idx)
                values[chunk_idx * 8 + value_idx] = __bfloat162float(__float2bfloat16_rn(sum[value_idx]));
        }
    };

    const bool is_q_warp = warp_idx < kNumQWarps;
    const uint32_t q_part_idx = warp_idx / kNumRowWarps;
    const uint32_t row_warp_idx = warp_idx % kNumRowWarps;
    constexpr uint32_t kNumAWorkers = cute::min(kNumSMs, kNumMBlocks * (kNumQTiles + kNumKVTiles) * kSplitK);
    const auto norm_block_idx = (blockIdx.x + kNumSMs - kNumAWorkers) % kNumSMs;

    const uint4 packed_weight[2] = {
        *reinterpret_cast<const uint4*>(is_q_warp ? q_weight + q_part_idx * NORM_BLOCK_N + lane_idx * 8 : kv_weight + lane_idx * 16),
        is_q_warp ? uint4() : *reinterpret_cast<const uint4*>(kv_weight + lane_idx * 16 + 8),
    };
    const auto weights = reinterpret_cast<const nv_bfloat16*>(packed_weight);

    constexpr bool kSingleNormIter = kNumRowWarps < 4 or kNumMBlocks * 128 <= kNumSMs * kNumRowWarps;
    DG_TRAP_ONLY_DEVICE_ASSERT(not kSingleNormIter or shape_m <= kNumSMs * kNumRowWarps);
    for (uint32_t row_base_idx = norm_block_idx * kNumRowWarps, iter_idx = 0;
         row_base_idx < shape_m and (not kSingleNormIter or iter_idx == 0);
         row_base_idx += kNumSMs * kNumRowWarps, ++ iter_idx) {
        const auto row_idx = row_base_idx + row_warp_idx;
        if (is_q_warp) {
            float values[8] = {};
            float total = 0.0f;
            if (row_idx < shape_m) {
                wait_ready(signals, row_idx / 128, ReadyKind::Q, kNumQTiles * kSplitK);
                load_intermediate(row_idx, q_part_idx * NORM_BLOCK_N + lane_idx * 8, values);
                #pragma unroll
                for (uint32_t value_idx = 0; value_idx < 8; ++ value_idx)
                    total = __fadd_rn(total, __fmul_rn(values[value_idx], values[value_idx]));
            }
            total = math::warp_reduce_sum(total);
            const auto sum_row_ptr = &norm_sums[iter_idx & 1][row_warp_idx * kNumQParts];
            if (lane_idx == 0)
                ptx::st_shared(sum_row_ptr + q_part_idx, total);
            cutlass::arch::NamedBarrier::sync(kNumQWarps * 32, 2);
            if (row_idx >= shape_m)
                continue;
            total = ptx::ld_shared(sum_row_ptr);
            #pragma unroll
            for (uint32_t part_idx = 1; part_idx < kNumQParts; ++ part_idx)
                total += ptx::ld_shared(sum_row_ptr + part_idx);
            const auto rstd = rsqrtf(total * (1.0f / kQRank) + eps);

            // V41F Q normalization includes a factor of two in its weight.
            nv_bfloat162 rounded[4];
            #pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < 4; ++ pair_idx) {
                rounded[pair_idx] = __floats2bfloat162_rn(
                    values[pair_idx * 2] * rstd * (__bfloat162float(weights[pair_idx * 2]) * 2.0f),
                    values[pair_idx * 2 + 1] * rstd * (__bfloat162float(weights[pair_idx * 2 + 1]) * 2.0f));
            }
            const auto amax = math::warp_reduce_max<4>(__bfloat162float(math::get_packed_bf16_amax(rounded)));
            const auto exponent = math::get_ue8m0_sf_exp(amax);
            const auto multiplier = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(exponent));
            const auto out = reinterpret_cast<uint32_t*>(
                q_a + static_cast<uint64_t>(row_idx) * kQRank + q_part_idx * NORM_BLOCK_N + lane_idx * 8);
            out[0] = math::scale_bf16x2_into_fp8x4(rounded[0], rounded[1], multiplier, multiplier);
            out[1] = math::scale_bf16x2_into_fp8x4(rounded[2], rounded[3], multiplier, multiplier);
            if (lane_idx % 4 == 0) {
                const auto group_idx = q_part_idx * 8 + lane_idx / 4;
                reinterpret_cast<uint8_t*>(q_a_sf + static_cast<uint64_t>(group_idx / 4) * sf_stride + row_idx)[group_idx % 4] = exponent;
            }
        } else {
            if (row_idx >= shape_m)
                continue;
            const auto slot_idx = static_cast<uint32_t>(slots[row_idx]);
            if (slot_idx >= num_cache_slots)
                continue;
            wait_ready(signals, row_idx / 128, ReadyKind::KV, kNumKVTiles * kSplitK);
            float kv[16], even = 0.0f, odd = 0.0f;
            load_intermediate(row_idx, kQRank + lane_idx * 16, kv);
            #pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < 8; ++ pair_idx) {
                even = __fadd_rn(even, __fmul_rn(kv[pair_idx * 2], kv[pair_idx * 2]));
                odd = __fadd_rn(odd, __fmul_rn(kv[pair_idx * 2 + 1], kv[pair_idx * 2 + 1]));
            }
            const auto total = math::warp_reduce_sum(even + odd);
            const auto kv_rstd = rsqrtf(total * (1.0f / kKVRank) + eps);
            nv_bfloat162 rounded_kv[8];
            #pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < 8; ++ pair_idx) {
                const auto col_idx = lane_idx * 16 + pair_idx * 2;
                auto x = __bfloat162float(__float2bfloat16_rn(kv[pair_idx * 2] * kv_rstd * __bfloat162float(weights[pair_idx * 2])));
                auto y = __bfloat162float(__float2bfloat16_rn(kv[pair_idx * 2 + 1] * kv_rstd * __bfloat162float(weights[pair_idx * 2 + 1])));
                if (lane_idx >= (kKVRank - kRopeDim) / 16) {
                    const auto rope_idx = static_cast<uint64_t>(static_cast<uint32_t>(positions[row_idx])) * kRopeDim +
                                          (col_idx - (kKVRank - kRopeDim)) / 2;
                    const auto c = cos_sin[rope_idx], s = cos_sin[rope_idx + kRopeDim / 2];
                    const auto x_rope = __fmaf_rn(x, c, -__fmul_rn(y, s));
                    y = __fmaf_rn(x, s, __fmul_rn(y, c));
                    x = x_rope;
                }
                rounded_kv[pair_idx] = __floats2bfloat162_rn(x, y);
            }
            const auto amax = math::warp_reduce_max<2>(__bfloat162float(math::get_packed_bf16_amax(rounded_kv)));
            const auto exponent = math::get_ue8m0_sf_exp(amax);
            const auto multiplier = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(exponent));
            const auto token_ptr = cache + static_cast<uint64_t>(slot_idx / cache_block_size) * cache_stride +
                                   (slot_idx % cache_block_size) * (kKVRank + kKVRank / 32);
            const auto payload = reinterpret_cast<uint32_t*>(token_ptr + lane_idx * 16);
            #pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < 4; ++ pair_idx)
                payload[pair_idx] = math::scale_bf16x2_into_fp8x4(rounded_kv[pair_idx * 2], rounded_kv[pair_idx * 2 + 1], multiplier, multiplier);
            if (lane_idx % 2 == 0)
                token_ptr[kKVRank + lane_idx / 2] = exponent;
        }
    }
    if (is_q_warp) {
        cutlass::arch::NamedBarrier::sync(kNumQWarps * 32, 2);
        if (threadIdx.x == 0) {
            for (uint32_t row_base_idx = norm_block_idx * kNumRowWarps, iter_idx = 0;
                 row_base_idx < shape_m and (not kSingleNormIter or iter_idx == 0);
                 row_base_idx += kNumSMs * kNumRowWarps, ++ iter_idx) {
                const auto num_rows = cute::min(kNumRowWarps, shape_m - row_base_idx);
                const auto first_rows = cute::min(num_rows, kReadyM - row_base_idx % kReadyM);
                ptx::atomic_add_rel(ready_count(signals, row_base_idx / kReadyM, ReadyKind::QNorm), first_rows);
                if (first_rows < num_rows)
                    ptx::atomic_add_rel(ready_count(signals, row_base_idx / kReadyM + 1, ReadyKind::QNorm), num_rows - first_rows);
            }
        }
    }
}

} // namespace v41f_qkv_impl

template <uint32_t kNumMBlocks, uint32_t kQOutDim, uint32_t kNumSMs, uint32_t kSplitK,
          uint32_t kNumRowWarps, uint32_t kNumWarps = cute::max(8u, kNumRowWarps * 6)>
CUTLASS_GLOBAL void __launch_bounds__(kNumWarps * 32, 1) __cluster_dims__(2, 1, 1)
v41f_qkv_lora_kernel(const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_qkv_weight,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_x_sf,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_qkv_weight_sf,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_q_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_q_b_weight,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_q_a_sf,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_q_b_weight_sf,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                     float* qkv_partials, const nv_bfloat16* q_weight, const nv_bfloat16* kv_weight,
                     const int* positions, const float* cos_sin, const int* slots,
                     uint8_t* cache, uint32_t shape_m,
                     uint32_t num_cache_slots, uint32_t cache_block_size, uint32_t cache_stride,
                     uint8_t* q_a, uint32_t* q_a_sf, void* signals, float eps) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using namespace v41f_qkv_impl;
    using Allocator = cute::TMEM::Allocator2Sm;
    using Storage = SharedStorage<kQOutDim>;
    constexpr uint32_t kQBlockN = Storage::kQBlockN;

    DG_STATIC_ASSERT(kNumMBlocks >= 1 and kSplitK > 0 and kSplitK <= 5, "Invalid QKV config");
    DG_STATIC_ASSERT(kNumRowWarps >= 1 and kNumRowWarps <= 4, "Invalid normalization warp count");
    DG_STATIC_ASSERT(kQOutDim == kQDim or kQOutDim == kQDim + kIndexQDim, "Invalid Q-B output dimension");
    DG_STATIC_ASSERT(kNumSMs % kNumMulticast == 0, "Incomplete CTA cluster");
    DG_STATIC_ASSERT(sizeof(Storage) <= 227840, "Shared storage overflow");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<Storage*>(smem_buffer);
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 1 and cute::elect_one_sync()) {
        smem.qkv_barriers.init();
        smem.q_barriers.init();
        smem.reset_ready.init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(512, &smem.tmem_ptr);
    }
    comm::cluster_sync_with_relaxed_arrive();
    DG_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_ptr == 0);
    cudaGridDependencySynchronize();

    gemm<kNumMBlocks, kQRank + kKVRank, kHidden, kNumSMs, /*kIsQKVProjection=*/ true, kSplitK>(
        smem.qkv, smem.qkv_barriers, smem.reset_ready, tensor_map_x, tensor_map_qkv_weight,
        tensor_map_x_sf, tensor_map_qkv_weight_sf, nullptr, shape_m, qkv_partials, signals);
    if (warp_idx == 0 and cute::elect_one_sync()) {
        const uint32_t n_idx = (blockIdx.x / 2) / math::constexpr_ceil_div(kNumMBlocks, 2u) * kQBlockN;
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++ i) {
            tma::copy<256, 1, 0>(&tensor_map_q_b_weight_sf, &smem.q_barriers.sf_full_barriers[i], smem.q.sfb[i], n_idx, i);
            tma::copy<128, kQBlockN / 2, 128>(&tensor_map_q_b_weight, &smem.q_barriers.full_barriers[i],
                                              smem.q.b[i], i * 128, n_idx + cute::block_rank_in_cluster() * (kQBlockN / 2), 2);
        }
    }
    normalize<kNumMBlocks, kNumSMs, kSplitK, kNumRowWarps>(
        qkv_partials, q_weight, kv_weight, positions, cos_sin, slots, cache, shape_m,
        num_cache_slots, cache_block_size, cache_stride,
        q_a, q_a_sf, signals, smem.norm_sums, eps);
    gemm<kNumMBlocks, kQOutDim, kQRank, kNumSMs, /*kIsQKVProjection=*/ false>(
        smem.q, smem.q_barriers, smem.reset_ready, tensor_map_q_a, tensor_map_q_b_weight,
        tensor_map_q_a_sf, tensor_map_q_b_weight_sf, &tensor_map_q, shape_m, nullptr, signals);

    if (warp_idx == 2)
        Allocator().free(0, 512);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100");
#endif
}

} // namespace deep_gemm::experimental

#pragma clang diagnostic pop
