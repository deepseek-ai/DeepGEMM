#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

namespace sched {

// Decode a linear index `t` in a `T x T` lower triangle (0 <= t < T*(T+1)/2)
// into `(row, col)` with `0 <= col <= row < T`, ordered in horizontal bands
// of height `kBandHeight` (traversed column-by-column inside each band).
template <uint32_t kBandHeight>
CUTLASS_DEVICE void decode_banded_lower_tri(const uint32_t& t, const uint32_t& T,
                                            uint32_t& row_idx, uint32_t& col_idx) {
    // Exact inverse triangular number: largest r such that r * (r + 1) / 2 <= t
    uint32_t r = static_cast<uint32_t>((sqrt(static_cast<double>(8ull * t + 1ull)) - 1.0) * 0.5);
    while (static_cast<uint64_t>(r + 1) * (r + 2) / 2 <= t)
        ++ r;
    while (static_cast<uint64_t>(r) * (r + 1) / 2 > t)
        -- r;

    const uint32_t r0 = (r / kBandHeight) * kBandHeight;
    const uint32_t h = cute::min(kBandHeight, T - r0);
    const uint32_t local = t - r0 * (r0 + 1) / 2;
    const uint32_t rect_tiles = r0 * h;

    if (local < rect_tiles) {
        col_idx = local / h;
        row_idx = r0 + (local - col_idx * h);
    } else {
        uint32_t tri = local - rect_tiles;
        uint32_t c = 0;
        while (tri >= h - c) {
            tri -= (h - c);
            ++ c;
        }
        col_idx = r0 + c;
        row_idx = col_idx + tri;
    }
}

// Persistent lower-triangular tile scheduler for batched SYRK:
// Supports both square tiles (`BLOCK_M == BLOCK_N`, `kNumSubTiles == 1`) and
// 1x2 horizontal super-tiles (`BLOCK_N == 2 * BLOCK_M`, `kNumSubTiles == 2`).
// For `kNumSubTiles == 2`, rows `2*q` and `2*q + 1` share the exact same
// column-pair range `p = 0..q`, yielding 100% 2x1 cluster multicast pairing
// on operand B across all full super-rows.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumTMAMulticast, uint32_t kNumSMs, uint32_t kGroupM = 8>
struct TriangularSyrkScheduler {
    static constexpr uint32_t kNumSubTiles = BLOCK_N / BLOCK_M;
    static_assert(kNumSubTiles == 1 or kNumSubTiles == 2, "Only BLOCK_N == BLOCK_M or 2 * BLOCK_M is supported");
    static_assert(kNumTMAMulticast == 1 or kNumTMAMulticast == 2, "Only cluster size 1 or 2 is supported");
    static_assert(kGroupM >= 2 and kGroupM % 2 == 0, "kGroupM must be an even positive integer");

    uint32_t num_m_blocks;
    uint32_t num_super_rows;
    uint32_t num_pair_tiles;
    uint32_t num_tiles_per_batch;
    uint32_t num_total_tiles;
    int current_iter = -1;

    bool is_peer_cta_alive = true;
    bool is_multicast_valid = false;

    CUTLASS_DEVICE explicit TriangularSyrkScheduler(const uint32_t& num_batches, const uint32_t& shape_n) {
        num_m_blocks = math::ceil_div(shape_n, BLOCK_M);
        if constexpr (kNumSubTiles == 1) {
            num_super_rows = num_m_blocks;
            num_pair_tiles = num_m_blocks * (num_m_blocks + 1) / 2;
            num_tiles_per_batch = num_pair_tiles;
        } else {
            num_super_rows = num_m_blocks / 2;
            num_pair_tiles = num_super_rows * (num_super_rows + 1);
            const uint32_t tail_tiles = (num_m_blocks & 1u) ? (num_super_rows + 1) : 0u;
            num_tiles_per_batch = num_pair_tiles + tail_tiles;
        }
        num_total_tiles = num_batches * num_tiles_per_batch;
    }

    CUTLASS_DEVICE void decode_tile(const uint32_t& linear_idx,
                                    uint32_t& batch_idx, uint32_t& row_idx, uint32_t& col_tile_idx) const {
        batch_idx = linear_idx / num_tiles_per_batch;
        const uint32_t t = linear_idx - batch_idx * num_tiles_per_batch;

        if constexpr (kNumSubTiles == 1) {
            decode_banded_lower_tri<kGroupM>(t, num_m_blocks, row_idx, col_tile_idx);
        } else {
            if (t < num_pair_tiles) {
                const uint32_t pair_idx = t >> 1;
                const uint32_t sub_row = t & 1u;
                uint32_t q_idx, p_idx;
                decode_banded_lower_tri<kGroupM / 2>(pair_idx, num_super_rows, q_idx, p_idx);
                row_idx = (q_idx << 1) + sub_row;
                col_tile_idx = p_idx;
            } else {
                row_idx = num_m_blocks - 1;
                col_tile_idx = t - num_pair_tiles;
            }
        }
    }

    CUTLASS_DEVICE bool get_next_block(uint32_t& batch_idx, uint32_t& row_idx, uint32_t& col_tile_idx) {
        const uint32_t next_block_idx = static_cast<uint32_t>(++ current_iter) * kNumSMs + blockIdx.x;
        if (next_block_idx >= num_total_tiles)
            return false;

        decode_tile(next_block_idx, batch_idx, row_idx, col_tile_idx);

        if constexpr (kNumTMAMulticast > 1) {
            const uint32_t idx_odd = (next_block_idx & ~1u) + 1u;
            is_peer_cta_alive = idx_odd < num_total_tiles;
            if (is_peer_cta_alive) {
                uint32_t peer_batch, peer_row, peer_col_tile;
                decode_tile(next_block_idx ^ 1u, peer_batch, peer_row, peer_col_tile);
                is_multicast_valid = (batch_idx == peer_batch) and (col_tile_idx == peer_col_tile);
            } else {
                is_multicast_valid = false;
            }
        }
        return true;
    }
};

} // namespace sched

namespace syrk_detail {

template <typename c_dtype_t>
CUTLASS_DEVICE float load_c_scalar(const c_dtype_t* base,
                                   const uint32_t& row, const uint32_t& col,
                                   const uint32_t& shape_n, const int64_t& row_stride) {
    if (row < shape_n and col < shape_n) {
        const auto offset = static_cast<int64_t>(row) * row_stride + static_cast<int64_t>(col);
        if constexpr (cute::is_same_v<c_dtype_t, cutlass::bfloat16_t>) {
            return __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(base + offset));
        } else {
            return base[offset];
        }
    }
    return 0.0f;
}

// `can_vec` must guarantee that `base` is aligned to two elements and `row_stride` is even
template <typename c_dtype_t>
CUTLASS_DEVICE float2 load_c_pair(const c_dtype_t* base,
                                  const uint32_t& row, const uint32_t& col0,
                                  const uint32_t& shape_n, const int64_t& row_stride,
                                  const bool& can_vec) {
    if (row < shape_n) {
        const auto offset = static_cast<int64_t>(row) * row_stride + static_cast<int64_t>(col0);
        if (can_vec and col0 + 1 < shape_n) {
            if constexpr (cute::is_same_v<c_dtype_t, cutlass::bfloat16_t>) {
                const auto v = *reinterpret_cast<const nv_bfloat162*>(base + offset);
                return __bfloat1622float2(v);
            } else {
                return *reinterpret_cast<const float2*>(base + offset);
            }
        }
        return make_float2(
            load_c_scalar(base, row, col0, shape_n, row_stride),
            load_c_scalar(base, row, col0 + 1, shape_n, row_stride)
        );
    }
    return make_float2(0.0f, 0.0f);
}

// Coalesced load of a `BLOCK_M x BLOCK_M` BF16 tile of `C` starting at
// `(g_row, g_col)` into a 16B-XOR-swizzled SMEM staging buffer.
template <uint32_t BLOCK_M, uint32_t kNumMathThreads>
CUTLASS_DEVICE void load_c_tile_bf16_to_smem(const cutlass::bfloat16_t* c_batch_ptr,
                                             cutlass::bfloat16_t* smem_dst,
                                             const uint32_t& g_row, const uint32_t& g_col,
                                             const uint32_t& shape_n, const int64_t& c_row_stride) {
    constexpr uint32_t kGroupsPerRow = BLOCK_M / 8;
    constexpr uint32_t kTotalGroups = BLOCK_M * kGroupsPerRow;
    static_assert(kTotalGroups % kNumMathThreads == 0, "Invalid C staging tile size");
    constexpr uint32_t kGroupsPerThread = kTotalGroups / kNumMathThreads;

    const bool can_vec =
        (reinterpret_cast<uintptr_t>(c_batch_ptr) % 16u == 0u) and
        ((c_row_stride & 7) == 0) and
        (g_row + BLOCK_M <= shape_n) and
        (g_col + BLOCK_M <= shape_n);

    if (can_vec) {
        #pragma unroll
        for (uint32_t v = 0; v < kGroupsPerThread; ++ v) {
            const uint32_t idx = threadIdx.x + v * kNumMathThreads;
            const uint32_t r = idx / kGroupsPerRow;
            const uint32_t g = idx - r * kGroupsPerRow;
            const uint32_t sw_g = g ^ (r & 7u);
            const auto val = *reinterpret_cast<const uint4*>(
                c_batch_ptr + static_cast<int64_t>(g_row + r) * c_row_stride + static_cast<int64_t>(g_col + g * 8));
            *reinterpret_cast<uint4*>(smem_dst + r * BLOCK_M + sw_g * 8) = val;
        }
    } else {
        #pragma unroll
        for (uint32_t v = 0; v < kGroupsPerThread; ++ v) {
            const uint32_t idx = threadIdx.x + v * kNumMathThreads;
            const uint32_t r = idx / kGroupsPerRow;
            const uint32_t g = idx - r * kGroupsPerRow;
            const uint32_t sw_g = g ^ (r & 7u);
            const uint32_t gr = g_row + r;
            const uint32_t gc = g_col + g * 8;
            auto* dst_row = smem_dst + r * BLOCK_M + sw_g * 8;
            #pragma unroll
            for (uint32_t e = 0; e < 8; ++ e) {
                if (gr < shape_n and gc + e < shape_n) {
                    dst_row[e] = c_batch_ptr[static_cast<int64_t>(gr) * c_row_stride + static_cast<int64_t>(gc + e)];
                } else {
                    dst_row[e] = cutlass::bfloat16_t(0.0f);
                }
            }
        }
    }
}

template <uint32_t BLOCK_M>
CUTLASS_DEVICE float read_swizzled_smem_bf16(const cutlass::bfloat16_t* smem_src,
                                             const uint32_t& r, const uint32_t& c) {
    const uint32_t sw_c = c ^ ((r & 7u) << 3);
    const auto* ptr = reinterpret_cast<const nv_bfloat16*>(smem_src + r * BLOCK_M + sw_c);
    return __bfloat162float(*ptr);
}

} // namespace syrk_detail

template <cute::UMMA::Major kMajorA,
          uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kSwizzleAMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages_, uint32_t kNumDStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast,
          uint32_t kNumSMs,
          typename cd_dtype_t,
          typename c_dtype_t,
          bool kHasC,
          uint32_t kGroupM = 8>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bf16_batched_syrk_gemm_impl(uint32_t num_batches, uint32_t shape_n, uint32_t shape_k,
                                 const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                 const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                 const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                                 const c_dtype_t* c_ptr,
                                 int64_t c_row_stride,
                                 int64_t c_batch_stride,
                                 float alpha,
                                 float beta) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Merge K stages for single-math-warpgroup K-major kernels to amortize warpgroup_wait overhead
    constexpr bool kDoMergeStages =
        kNumStages_ >= 10 and
        kMajorA == cute::UMMA::Major::K and
        kNumMathThreads == 128;
    constexpr uint32_t kNumMinStages = 5;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;
    constexpr uint32_t kNumSubTiles = BLOCK_N / BLOCK_M;

    // Types and static assertions
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, kMajorA, kMajorA>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M == 64 or BLOCK_M == 128, "BLOCK_M must be 64 or 128");
    DG_STATIC_ASSERT(BLOCK_N == BLOCK_M or (BLOCK_M == 128 and BLOCK_N == 256), "Unsupported (BLOCK_M, BLOCK_N)");
    DG_STATIC_ASSERT(kNumMathThreads == (BLOCK_M == 64 ? 128 : 256), "Invalid math thread count");
    DG_STATIC_ASSERT(kNumDStages == 1 or kNumDStages == 2, "Invalid D stage count");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid D dtype");
    DG_STATIC_ASSERT(cute::is_same_v<c_dtype_t, float> or cute::is_same_v<c_dtype_t, cutlass::bfloat16_t>, "Invalid C dtype");

    // Compile-time shape specialization
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory sizes (D staging holds `BLOCK_M x BLOCK_M` sub-tiles)
    static constexpr uint32_t SMEM_D_SIZE_PER_STAGE = math::constexpr_align(BLOCK_M * BLOCK_M * static_cast<uint32_t>(sizeof(cd_dtype_t)), 1024u);
    static constexpr uint32_t SMEM_D_TOTAL_SIZE = kNumDStages * SMEM_D_SIZE_PER_STAGE;
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
    DG_STATIC_ASSERT(SMEM_D_TOTAL_SIZE >= BLOCK_M * BLOCK_M * sizeof(float),
                     "SMEM D region must fit a full FP32 sub-tile for diagonal symmetrization");
    DG_STATIC_ASSERT(SMEM_D_SIZE_PER_STAGE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B/D must be aligned to 1024 bytes");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        if constexpr (kNumSubTiles > 1)
            cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_d0 = reinterpret_cast<cd_dtype_t*>(smem_buffer);
    auto smem_d1 = reinterpret_cast<cd_dtype_t*>(smem_buffer + (kNumDStages == 2 ? SMEM_D_SIZE_PER_STAGE : 0));
    auto smem_d_f32 = reinterpret_cast<float*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_TOTAL_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_TOTAL_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_D_TOTAL_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }

    (kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Register budgets
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    cudaGridDependencySynchronize();

    // Triangular scheduler
    uint32_t batch_idx, row_block_idx, col_tile_idx;
    auto scheduler = sched::TriangularSyrkScheduler<BLOCK_M, BLOCK_N, kNumTMAMulticast, kNumSMs, kGroupM>(num_batches, shape_n);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA producer warp-group
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            DG_STATIC_ASSERT(kNumTMAThreads >= 128, "Need at least 128 threads for TMA warp-group");

            while (scheduler.get_next_block(batch_idx, row_block_idx, col_tile_idx)) {
                const uint32_t num_tma_multicast_b = (kNumTMAMulticast > 1 and scheduler.is_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t m_idx = row_block_idx * BLOCK_M;
                const uint32_t n_idx = col_tile_idx * BLOCK_N;

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    if constexpr (kMajorA == cute::UMMA::Major::K) {
                        tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, true>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx, m_idx, 1, batch_idx);
                        tma::copy<BLOCK_K, BLOCK_N, kSwizzleAMode, cutlass::bfloat16_t, true>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b, batch_idx);
                    } else {
                        tma::copy<BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, true>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], m_idx, k_idx, 1, batch_idx);
                        tma::copy<BLOCK_N, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, true>(
                            &tensor_map_b, &full_barrier, smem_b[stage_idx], n_idx, k_idx, num_tma_multicast_b, batch_idx);
                    }
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
            }

            // Safely drain remote empty barrier arrivals before CTA exit
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        // Math consumer warp-group(s)
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const uint32_t math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);

        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm90::make_gmma_desc<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = mma::sm90::make_gmma_desc<kMajorA, BLOCK_N, BLOCK_ATOM_K, kSwizzleAMode>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        constexpr uint32_t kNumElemBytes = sizeof(cd_dtype_t);
        constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_M : (kSwizzleDMode / kNumElemBytes);
        constexpr uint32_t kNumTMADBlocks = BLOCK_M / TMA_D_BLOCK_N;
        constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
        constexpr uint32_t kSubAccum = WGMMA::kNumAccum / kNumSubTiles;
        DG_STATIC_ASSERT(BLOCK_M % TMA_D_BLOCK_N == 0 and kNumTMADBlocks <= 32, "Invalid TMA D blocks");
        DG_STATIC_ASSERT(kSubAccum % 4 == 0, "Invalid WGMMA sub-tile accumulator count");
        DG_STATIC_ASSERT(kNumMathThreads >= kNumTMADBlocks, "Too many TMA blocks");

        const uint32_t row_base = warp_idx * WGMMA_M_PER_WARP;
        const uint32_t t_row = lane_idx >> 2;
        const uint32_t t_col = (lane_idx & 3u) << 1;

        constexpr bool kOverlapWgmma = (kNumStages >= 3);

        while (scheduler.get_next_block(batch_idx, row_block_idx, col_tile_idx)) {
            float accum[WGMMA::kNumAccum] = {0};

            auto empty_barrier_arrive = [&](uint32_t s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    const auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // Main K-loop
            uint32_t prev_stage_idx = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                full_barriers[stage_idx]->wait(phase);

                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    const uint32_t atom_k_idx = k * WGMMA::K / BLOCK_ATOM_K;
                    a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorA, BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode, nv_bfloat16>(
                        a_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_M * BLOCK_ATOM_K);
                    b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<kMajorA, BLOCK_N, BLOCK_ATOM_K, kSwizzleAMode, nv_bfloat16>(
                        b_desc_base_lo, 0, (k * WGMMA::K) % BLOCK_ATOM_K, atom_k_idx * BLOCK_N * BLOCK_ATOM_K);
                    WGMMA::wgmma(a_desc, b_desc, accum, 1);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);

                if constexpr (kOverlapWgmma) {
                    if (k_block_idx > 0) {
                        ptx::warpgroup_wait<1>();
                        empty_barrier_arrive(prev_stage_idx);
                    }
                    prev_stage_idx = stage_idx;
                } else {
                    ptx::warpgroup_wait<0>();
                    empty_barrier_arrive(stage_idx);
                }
            }
            if constexpr (kOverlapWgmma) {
                ptx::warpgroup_wait<0>();
                empty_barrier_arrive(prev_stage_idx);
            }

            const c_dtype_t* c_batch_ptr = kHasC ? (c_ptr + static_cast<int64_t>(batch_idx) * c_batch_stride) : nullptr;
            // Paired `C` loads need 2-element alignment, which offset views or odd batch strides may break
            const bool can_vec_c = kHasC and
                (reinterpret_cast<uintptr_t>(c_batch_ptr) % (2 * sizeof(c_dtype_t)) == 0) and
                ((c_row_stride & 1) == 0);
            const uint32_t tile_row = row_block_idx * BLOCK_M;

            // Epilogue: process each `BLOCK_M x BLOCK_M` lower-triangle sub-tile
            #pragma unroll
            for (uint32_t sub_idx = 0; sub_idx < kNumSubTiles; ++ sub_idx) {
                const uint32_t col_block_idx = col_tile_idx * kNumSubTiles + sub_idx;
                if constexpr (kNumSubTiles > 1) {
                    if (col_block_idx > row_block_idx)
                        continue;
                }

                float* sub_accum = accum + sub_idx * kSubAccum;
                const bool is_diag = (row_block_idx == col_block_idx);
                const uint32_t tile_col = col_block_idx * BLOCK_M;

                // Wait for previous TMA store(s) to release smem_d
                if (threadIdx.x < kNumTMADBlocks)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                // Symmetrize diagonal sub-tiles in FP32 via SMEM before scaling/epilogue so
                // D[r, c] and D[c, r] receive bit-identical (A @ A.T) values.
                if (is_diag) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kSubAccum / 4; ++ i) {
                        const uint32_t c_loc = i * 8 + t_col;
                        ptx::st_shared(reinterpret_cast<float2*>(smem_d_f32 + (row_base + t_row + 0) * BLOCK_M + c_loc),
                                       make_float2(sub_accum[i * 4 + 0], sub_accum[i * 4 + 1]));
                        ptx::st_shared(reinterpret_cast<float2*>(smem_d_f32 + (row_base + t_row + 8) * BLOCK_M + c_loc),
                                       make_float2(sub_accum[i * 4 + 2], sub_accum[i * 4 + 3]));
                    }
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    #pragma unroll
                    for (uint32_t i = 0; i < kSubAccum / 4; ++ i) {
                        const uint32_t r0 = row_base + t_row;
                        const uint32_t r1 = r0 + 8;
                        const uint32_t c0 = i * 8 + t_col;
                        const uint32_t c1 = c0 + 1;

                        sub_accum[i * 4 + 0] = ptx::ld_shared(smem_d_f32 + (r0 >= c0 ? r0 * BLOCK_M + c0 : c0 * BLOCK_M + r0));
                        sub_accum[i * 4 + 1] = ptx::ld_shared(smem_d_f32 + (r0 >= c1 ? r0 * BLOCK_M + c1 : c1 * BLOCK_M + r0));
                        sub_accum[i * 4 + 2] = ptx::ld_shared(smem_d_f32 + (r1 >= c0 ? r1 * BLOCK_M + c0 : c0 * BLOCK_M + r1));
                        sub_accum[i * 4 + 3] = ptx::ld_shared(smem_d_f32 + (r1 >= c1 ? r1 * BLOCK_M + c1 : c1 * BLOCK_M + r1));
                    }
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
                }

                auto eval_lower_quad = [&](const uint32_t& i, float& v0, float& v1, float& v2, float& v3) {
                    v0 = alpha * sub_accum[i * 4 + 0];
                    v1 = alpha * sub_accum[i * 4 + 1];
                    v2 = alpha * sub_accum[i * 4 + 2];
                    v3 = alpha * sub_accum[i * 4 + 3];
                    if constexpr (kHasC) {
                        const uint32_t gr0 = tile_row + row_base + t_row;
                        const uint32_t gr1 = gr0 + 8;
                        const uint32_t gc0 = tile_col + i * 8 + t_col;
                        const float2 c_r0 = syrk_detail::load_c_pair(c_batch_ptr, gr0, gc0, shape_n, c_row_stride, can_vec_c);
                        const float2 c_r1 = syrk_detail::load_c_pair(c_batch_ptr, gr1, gc0, shape_n, c_row_stride, can_vec_c);
                        v0 = fmaf(beta, c_r0.x, v0);
                        v1 = fmaf(beta, c_r0.y, v1);
                        v2 = fmaf(beta, c_r1.x, v2);
                        v3 = fmaf(beta, c_r1.y, v3);
                    }
                };

                auto eval_upper_quad_global = [&](const uint32_t& i, float& v0, float& v1, float& v2, float& v3) {
                    v0 = alpha * sub_accum[i * 4 + 0];
                    v1 = alpha * sub_accum[i * 4 + 1];
                    v2 = alpha * sub_accum[i * 4 + 2];
                    v3 = alpha * sub_accum[i * 4 + 3];
                    if constexpr (kHasC) {
                        const uint32_t gr0 = tile_row + row_base + t_row;
                        const uint32_t gr1 = gr0 + 8;
                        const uint32_t gc0 = tile_col + i * 8 + t_col;
                        const uint32_t gc1 = gc0 + 1;
                        v0 = fmaf(beta, syrk_detail::load_c_scalar(c_batch_ptr, gc0, gr0, shape_n, c_row_stride), v0);
                        v1 = fmaf(beta, syrk_detail::load_c_scalar(c_batch_ptr, gc1, gr0, shape_n, c_row_stride), v1);
                        v2 = fmaf(beta, syrk_detail::load_c_scalar(c_batch_ptr, gc0, gr1, shape_n, c_row_stride), v2);
                        v3 = fmaf(beta, syrk_detail::load_c_scalar(c_batch_ptr, gc1, gr1, shape_n, c_row_stride), v3);
                    }
                };

                if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                    DG_STATIC_ASSERT(kSwizzleDMode == 128, "BF16 output requires 128B swizzle");
                    DG_STATIC_ASSERT(kNumDStages == 2, "BF16 output uses dual SMEM D stages");
                    constexpr uint32_t kNumBankGroupBytes = 16;

                    // Precompute invariant transposed STSM_T column offsets for lanes 0..15
                    const uint32_t c_t = row_base + ((lane_idx >> 3) & 1u) * 8u;
                    const uint32_t atom_t = c_t / 64u;
                    const uint32_t bg_t = (c_t & 63u) >> 3;
                    const uint32_t lane_mod8 = lane_idx & 7u;
                    const uint32_t swizzled_bg_t = bg_t ^ lane_mod8;
                    uint8_t* smem_d1_lane_base = reinterpret_cast<uint8_t*>(smem_d1) +
                        atom_t * (BLOCK_M * kSwizzleDMode) +
                        lane_mod8 * kSwizzleDMode +
                        swizzled_bg_t * kNumBankGroupBytes;

                    // Fast coalesced staging path when kHasC with BF16 C:
                    // Stage C_upper coalesced into smem_d0 -> transpose-store D_upper into smem_d1 ->
                    // launch D_upper TMA store early -> store D_lower into smem_d0 with hoisted bounds.
                    bool upper_tma_issued = false;
                    if constexpr (kHasC and cute::is_same_v<c_dtype_t, cutlass::bfloat16_t>) {
                        if (not is_diag) {
                            syrk_detail::load_c_tile_bf16_to_smem<BLOCK_M, kNumMathThreads>(
                                c_batch_ptr, smem_d0, tile_col, tile_row, shape_n, c_row_stride);
                            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                            #pragma unroll
                            for (uint32_t i = 0; i < kSubAccum / 4; ++ i) {
                                const uint32_t ur0 = i * 8 + t_col;
                                const uint32_t ur1 = ur0 + 1;
                                const uint32_t uc0 = row_base + t_row;
                                const uint32_t uc1 = uc0 + 8;
                                const float u0 = fmaf(beta, syrk_detail::read_swizzled_smem_bf16<BLOCK_M>(smem_d0, ur0, uc0), alpha * sub_accum[i * 4 + 0]);
                                const float u1 = fmaf(beta, syrk_detail::read_swizzled_smem_bf16<BLOCK_M>(smem_d0, ur1, uc0), alpha * sub_accum[i * 4 + 1]);
                                const float u2 = fmaf(beta, syrk_detail::read_swizzled_smem_bf16<BLOCK_M>(smem_d0, ur0, uc1), alpha * sub_accum[i * 4 + 2]);
                                const float u3 = fmaf(beta, syrk_detail::read_swizzled_smem_bf16<BLOCK_M>(smem_d0, ur1, uc1), alpha * sub_accum[i * 4 + 3]);
                                const auto uv0 = __float22bfloat162_rn({u0, u1});
                                const auto uv1 = __float22bfloat162_rn({u2, u3});
                                uint8_t* smem_ptr1 = smem_d1_lane_base + (i * 8u) * kSwizzleDMode;
                                ptx::SM90_U32x2_STSM_T<nv_bfloat162>::copy(uv0, uv1, smem_ptr1);
                            }
                            cute::tma_store_fence();
                            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                            if (threadIdx.x < kNumTMADBlocks) {
                                const uint32_t in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                                const uint32_t upper_col = tile_row + in_block_n_offset;
                                if (upper_col < shape_n) {
                                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d,
                                                                  smem_d1 + in_block_n_offset * BLOCK_M,
                                                                  upper_col, tile_col, batch_idx);
                                }
                                cute::tma_store_arrive();
                            }
                            upper_tma_issued = true;
                        }

                        const bool fast_lower_c =
                            (tile_row + BLOCK_M <= shape_n) and
                            (tile_col + BLOCK_M <= shape_n) and
                            can_vec_c;
                        const auto* c_row0_base = fast_lower_c
                            ? (c_batch_ptr + static_cast<int64_t>(tile_row + row_base + t_row) * c_row_stride + static_cast<int64_t>(tile_col + t_col))
                            : nullptr;
                        const auto* c_row1_base = fast_lower_c ? (c_row0_base + static_cast<int64_t>(8) * c_row_stride) : nullptr;

                        #pragma unroll
                        for (uint32_t i = 0; i < kSubAccum / 4; ++ i) {
                            float l0, l1, l2, l3;
                            if (fast_lower_c) {
                                const float2 c_r0 = __bfloat1622float2(__ldg(reinterpret_cast<const nv_bfloat162*>(c_row0_base + i * 8)));
                                const float2 c_r1 = __bfloat1622float2(__ldg(reinterpret_cast<const nv_bfloat162*>(c_row1_base + i * 8)));
                                l0 = fmaf(beta, c_r0.x, alpha * sub_accum[i * 4 + 0]);
                                l1 = fmaf(beta, c_r0.y, alpha * sub_accum[i * 4 + 1]);
                                l2 = fmaf(beta, c_r1.x, alpha * sub_accum[i * 4 + 2]);
                                l3 = fmaf(beta, c_r1.y, alpha * sub_accum[i * 4 + 3]);
                            } else {
                                eval_lower_quad(i, l0, l1, l2, l3);
                            }
                            const auto lv0 = __float22bfloat162_rn({l0, l1});
                            const auto lv1 = __float22bfloat162_rn({l2, l3});

                            const uint32_t atom_offset = i / (TMA_D_BLOCK_N / 8);
                            const uint32_t in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                            const uint32_t swizzled_col = in_atom_offset ^ (lane_idx & 7u);
                            uint8_t* smem_ptr0 = reinterpret_cast<uint8_t*>(smem_d0) +
                                row_base * kSwizzleDMode +
                                atom_offset * (BLOCK_M * kSwizzleDMode) +
                                lane_idx * kSwizzleDMode +
                                swizzled_col * kNumBankGroupBytes;
                            ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(lv0, lv1, smem_ptr0);
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSubAccum / 4; ++ i) {
                            float l0, l1, l2, l3;
                            eval_lower_quad(i, l0, l1, l2, l3);
                            const auto lv0 = __float22bfloat162_rn({l0, l1});
                            const auto lv1 = __float22bfloat162_rn({l2, l3});

                            const uint32_t atom_offset = i / (TMA_D_BLOCK_N / 8);
                            const uint32_t in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                            const uint32_t swizzled_col = in_atom_offset ^ (lane_idx & 7u);
                            uint8_t* smem_ptr0 = reinterpret_cast<uint8_t*>(smem_d0) +
                                row_base * kSwizzleDMode +
                                atom_offset * (BLOCK_M * kSwizzleDMode) +
                                lane_idx * kSwizzleDMode +
                                swizzled_col * kNumBankGroupBytes;
                            ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(lv0, lv1, smem_ptr0);

                            if (not is_diag) {
                                nv_bfloat162 uv0 = lv0;
                                nv_bfloat162 uv1 = lv1;
                                if constexpr (kHasC) {
                                    float u0, u1, u2, u3;
                                    eval_upper_quad_global(i, u0, u1, u2, u3);
                                    uv0 = __float22bfloat162_rn({u0, u1});
                                    uv1 = __float22bfloat162_rn({u2, u3});
                                }
                                uint8_t* smem_ptr1 = smem_d1_lane_base + (i * 8u) * kSwizzleDMode;
                                ptx::SM90_U32x2_STSM_T<nv_bfloat162>::copy(uv0, uv1, smem_ptr1);
                            }
                        }
                    }

                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    if (threadIdx.x < kNumTMADBlocks) {
                        const uint32_t in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                        const uint32_t lower_col = tile_col + in_block_n_offset;
                        if (lower_col < shape_n) {
                            cute::SM90_TMA_STORE_3D::copy(&tensor_map_d,
                                                          smem_d0 + in_block_n_offset * BLOCK_M,
                                                          lower_col, tile_row, batch_idx);
                        }
                        if (not is_diag and not upper_tma_issued) {
                            const uint32_t upper_col = tile_row + in_block_n_offset;
                            if (upper_col < shape_n) {
                                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d,
                                                              smem_d1 + in_block_n_offset * BLOCK_M,
                                                              upper_col, tile_col, batch_idx);
                            }
                        }
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                } else {
                    // FP32 output: kSwizzleDMode == 0
                    DG_STATIC_ASSERT(kSwizzleDMode == 0, "FP32 output expects unswizzled SMEM");

                    auto store_normal_f32 = [&](float* dst) {
                        auto dst_0 = reinterpret_cast<float2*>(dst + (row_base + t_row + 0) * BLOCK_M + t_col);
                        auto dst_1 = reinterpret_cast<float2*>(dst + (row_base + t_row + 8) * BLOCK_M + t_col);
                        #pragma unroll
                        for (uint32_t i = 0; i < kSubAccum / 4; ++ i) {
                            float l0, l1, l2, l3;
                            eval_lower_quad(i, l0, l1, l2, l3);
                            ptx::st_shared(dst_0 + i * 4, make_float2(l0, l1));
                            ptx::st_shared(dst_1 + i * 4, make_float2(l2, l3));
                        }
                    };

                    auto store_transposed_f32 = [&](float* dst) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kSubAccum / 4; ++ i) {
                            float u0, u1, u2, u3;
                            eval_upper_quad_global(i, u0, u1, u2, u3);
                            const uint32_t tr0 = i * 8 + t_col;
                            const uint32_t tr1 = tr0 + 1;
                            const uint32_t tc0 = row_base + t_row;
                            const uint32_t tc1 = tc0 + 8;
                            ptx::st_shared(dst + tr0 * BLOCK_M + tc0, u0);
                            ptx::st_shared(dst + tr1 * BLOCK_M + tc0, u1);
                            ptx::st_shared(dst + tr0 * BLOCK_M + tc1, u2);
                            ptx::st_shared(dst + tr1 * BLOCK_M + tc1, u3);
                        }
                    };

                    if constexpr (kNumDStages == 2) {
                        store_normal_f32(smem_d0);
                        if (not is_diag)
                            store_transposed_f32(smem_d1);
                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                        if (threadIdx.x == 0) {
                            cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_d0, tile_col, tile_row, batch_idx);
                            if (not is_diag)
                                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_d1, tile_row, tile_col, batch_idx);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();
                    } else {
                        // Single-stage FP32 SMEM: issue normal TMA store, wait, then issue transposed mirror store
                        store_normal_f32(smem_d0);
                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                        if (threadIdx.x == 0) {
                            cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_d0, tile_col, tile_row, batch_idx);
                            cute::tma_store_arrive();
                        }
                        __syncwarp();

                        if (not is_diag) {
                            if (threadIdx.x == 0)
                                cute::tma_store_wait<0>();
                            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                            store_transposed_f32(smem_d0);
                            cute::tma_store_fence();
                            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                            if (threadIdx.x == 0) {
                                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_d0, tile_row, tile_col, batch_idx);
                                cute::tma_store_arrive();
                            }
                            __syncwarp();
                        }
                    }
                }
            }
        }

        // Ensure all TMA stores complete before CTA exit
        if (threadIdx.x < kNumTMADBlocks)
            cute::tma_store_wait<0>();
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
