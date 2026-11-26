#pragma once

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/layout/gemm.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::experimental::sched {

template <uint32_t kNumBatchTiles, uint32_t kBatches>
struct NewtonSchulzDependency {
    static constexpr uint32_t kNumMaxCachedWords = 32 * 4;
    static constexpr uint32_t kNumWordsPerLane = kNumBatchTiles <= 32 * 32 - 31 ? 1 :
                                                kNumBatchTiles <= 32 * 32 * 2 - 31 ? 2 : 4;
    static constexpr uint32_t kNumCachedWords = 32 * kNumWordsPerLane;
    static constexpr uint32_t kNumReadyWords = math::constexpr_ceil_div(
        math::constexpr_ceil_div(kNumBatchTiles * kBatches, 32u), kNumMaxCachedWords) * kNumMaxCachedWords;
    static constexpr bool kSingleWindow = kNumBatchTiles <= 32 * kNumCachedWords - 31;
    uint32_t base_word_idx;
    uint64_t ready_words[kNumWordsPerLane] = {};

    CUTLASS_DEVICE explicit NewtonSchulzDependency(const uint32_t batch_tile_idx):
        base_word_idx(kSingleWindow ? cute::min(batch_tile_idx / 32, kNumReadyWords - kNumCachedWords) : 0) {}

    // Each task reads one batch; the extra 31 tiles account for its unaligned starting word.
    // Completion bits are monotonic, so only a missing dependency requires a refresh.
    CUTLASS_DEVICE void wait_until_ready(const uint64_t* ready, const uint32_t tile_idx) {
        const auto word_idx = tile_idx / 32;
        if constexpr (not kSingleWindow) {
            const auto next_base_word_idx = word_idx / kNumCachedWords * kNumCachedWords;
            if (base_word_idx != next_base_word_idx) {
                base_word_idx = next_base_word_idx;
                #pragma unroll
                for (uint32_t component_idx = 0; component_idx < kNumWordsPerLane; ++component_idx)
                    ready_words[component_idx] = 0;
            }
        }
        const uint64_t required_mask = 3ull << (tile_idx % 32 * 2);
        const auto source_lane_idx = (word_idx - base_word_idx) / kNumWordsPerLane;
        const auto component_idx = (word_idx - base_word_idx) % kNumWordsPerLane;
        while (true) {
            uint64_t local_word = 0;
            #pragma unroll
            for (uint32_t cached_idx = 0; cached_idx < kNumWordsPerLane; ++cached_idx)
                local_word |= component_idx == cached_idx ? ready_words[cached_idx] : 0;
            if (__all_sync(0xffffffffu, ptx::get_lane_idx() != source_lane_idx or
                           (local_word & required_mask) == required_mask))
                break;
            const auto ptr = ready + base_word_idx + ptx::get_lane_idx() * kNumWordsPerLane;
            #pragma unroll
            for (uint32_t cached_idx = 0; cached_idx < kNumWordsPerLane; ++cached_idx)
                ready_words[cached_idx] = ptx::ld_acq_gpu(ptr + cached_idx);
            __syncwarp();
        }
    }

    CUTLASS_DEVICE static void publish_ready(uint64_t* ready, const uint32_t tile_idx, const uint32_t cta_rank) {
        const auto ptr = ready + tile_idx / 32;
        const uint64_t mask = 1ull << (tile_idx % 32 * 2 + cta_rank);
        asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(ptr), "l"(mask) : "memory");
    }
};

template <uint32_t kRows>
struct NewtonSchulzTriangle {
    static constexpr uint32_t kNumTiles = kRows * (kRows + 1) / 2;

    // Column-major 8x8 tile groups retain row panels while advancing the reduction frontier.
    CUTLASS_HOST_DEVICE static uint32_t get_tile_idx(uint32_t row_idx, uint32_t col_idx) {
        if (row_idx < col_idx) {
            const auto saved = row_idx;
            row_idx = col_idx;
            col_idx = saved;
        }
        const auto row_start = row_idx / 8 * 8;
        const auto col_start = col_idx / 8 * 8;
        const auto height = cute::min(8u, kRows - row_start);
        const auto width = cute::min(8u, kRows - col_start);
        const auto local_row = row_idx - row_start;
        const auto local_col = col_idx - col_start;
        const auto prefix = col_start * kRows - col_start * (col_start - 1) / 2;
        if (row_start == col_start)
            return prefix + local_col * width - local_col * (local_col - 1) / 2 + local_row - local_col;
        return prefix + width * (width + 1) / 2 + (row_start - col_start - width) * width +
               local_col * height + local_row;
    }

    CUTLASS_HOST_DEVICE static void get_tile_coords(uint32_t tile_idx, uint32_t& row_idx, uint32_t& col_idx) {
        for (uint32_t col_start = 0; col_start < kRows; col_start += 8) {
            const auto width = cute::min(8u, kRows - col_start);
            const auto count = width * (width + 1) / 2 + width * (kRows - col_start - width);
            if (tile_idx >= count) {
                tile_idx -= count;
                continue;
            }
            const auto diagonal_count = width * (width + 1) / 2;
            if (tile_idx < diagonal_count) {
                uint32_t local_col = 0;
                while (tile_idx >= width - local_col) {
                    tile_idx -= width - local_col;
                    ++local_col;
                }
                col_idx = col_start + local_col;
                row_idx = col_idx + tile_idx;
                return;
            }
            tile_idx -= diagonal_count;
            const auto row_start = col_start + width + uint32_t(tile_idx / (8 * width)) * 8;
            const auto height = cute::min(8u, kRows - row_start);
            tile_idx %= 8 * width;
            row_idx = row_start + tile_idx % height;
            col_idx = col_start + tile_idx / height;
            return;
        }
    }
};

struct NewtonSchulzTaskQueue {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static constexpr uint32_t kNumStages = 2;
    static constexpr uint32_t kNumCTAs = 2;
    static constexpr uint32_t kNumTmaThreads = 32;
    static constexpr uint32_t kNumMmaThreads = 32;
    static constexpr uint32_t kNumEpilogueThreads = 128;
    static constexpr uint32_t kNumConsumers = kNumCTAs * (kNumTmaThreads + kNumEpilogueThreads) + kNumMmaThreads;
    alignas(16) uint4 tasks[kNumStages];
    Barrier full[kNumStages];
    Barrier empty[kNumStages];

    CUTLASS_DEVICE void initialize() {
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumStages; ++stage_idx) {
            full[stage_idx].init(1);
            empty[stage_idx].init(kNumConsumers);
        }
    }

    // Both CTAs receive the same task; the slot remains live until every consumer has read it.
    template <typename Scheduler>
    CUTLASS_DEVICE void publish_tasks(uint32_t* task_count) {
        const auto lane_idx = ptx::get_lane_idx();
        uint32_t sequence_idx = 0;
        while (true) {
            const auto stage_idx = sequence_idx % kNumStages;
            const auto phase = sequence_idx / kNumStages & 1;
            empty[stage_idx].wait(phase ^ 1);
            uint32_t task_idx = 0;
            if (lane_idx == 0)
                task_idx = ptx::atomic_add(task_count, 1u);
            task_idx = ptx::exchange(task_idx, 0);
            const auto task = Scheduler::get_task_info(task_idx);
            if (lane_idx < kNumCTAs) {
                full[stage_idx].arrive_and_expect_tx(sizeof(task), lane_idx);
                ptx::st_async_cluster(tasks + stage_idx, task, lane_idx, full[stage_idx]);
            }
            __syncwarp();
            if (task_idx >= Scheduler::kNumTasks)
                break;
            ++sequence_idx;
        }
    }

    CUTLASS_DEVICE uint4 get_task(const uint32_t sequence_idx) {
        const auto stage_idx = sequence_idx % kNumStages;
        full[stage_idx].wait(sequence_idx / kNumStages & 1);
        const auto task = tasks[stage_idx];
        cutlass::arch::fence_view_async_shared();
        empty[stage_idx].arrive(0u);
        return task;
    }
};

template <uint32_t kRows, uint32_t kColumns, uint32_t kBatches>
struct NewtonSchulzScheduler {
    using Triangle = NewtonSchulzTriangle<kRows>;
    static constexpr uint32_t kNumTriangleTasks = kBatches * Triangle::kNumTiles;
    static constexpr uint32_t kNumTasks = 2 * kNumTriangleTasks + kBatches * kRows * kColumns;
    NewtonSchulzTaskQueue& queue;
    uint32_t batch_idx = 0;
    uint32_t tile_idx = 0;
    uint32_t row_idx = 0;
    uint32_t col_idx = 0;
    uint32_t task_sequence_idx = 0xffffffffu;
    uint32_t task_type = 0;
    bool polynomial = false;
    bool symm = false;

    CUTLASS_DEVICE explicit NewtonSchulzScheduler(NewtonSchulzTaskQueue& queue): queue(queue) {}

    // Tasks are allocated in topological order: all X2A, then A2B, then B2alt.
    CUTLASS_DEVICE static uint4 get_task_info(const uint32_t task_idx) {
        auto task = make_uint4(task_idx, 0, 0, 0);
        if (task_idx >= kNumTasks)
            return task;
        if (task_idx >= 2 * kNumTriangleTasks) {
            const auto tile_idx = task_idx - 2 * kNumTriangleTasks;
            task.y = tile_idx / (kRows * kColumns);
            const auto local_idx = tile_idx % (kRows * kColumns);
            const auto group_idx = local_idx / (32 * kColumns);
            const auto group_rows = cute::min(32u, kRows - group_idx * 32);
            const auto in_group = local_idx - group_idx * 32 * kColumns;
            task.z = group_idx * 32 + in_group % group_rows;
            task.w = in_group / group_rows;
        } else {
            const auto tile_idx = task_idx % kNumTriangleTasks;
            task.y = tile_idx / Triangle::kNumTiles;
            Triangle::get_tile_coords(tile_idx % Triangle::kNumTiles, task.z, task.w);
        }
        return task;
    }

    CUTLASS_DEVICE bool get_next_task() {
        const auto task = queue.get_task(task_sequence_idx + 1);
        if (task.x >= kNumTasks)
            return false;
        task_type = task.x >= 2 * kNumTriangleTasks ? 2 : task.x / kNumTriangleTasks;
        polynomial = task_type == 1;
        symm = task_type == 2;
        batch_idx = task.y;
        row_idx = task.z;
        col_idx = task.w;
        tile_idx = task.x - task_type * kNumTriangleTasks;
        ++task_sequence_idx;
        return true;
    }
};

}

namespace deep_gemm::experimental {

template <uint32_t kRows, uint32_t kNumThreads, uint32_t kNumElementsPerBlock>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads) void bf16_newton_schulz_polynomial_impl(
    const nv_bfloat16* gram, nv_bfloat16* polynomial, const uint64_t num_elements,
    const float identity, const float linear, const float quadratic) {
    const auto block_offset = uint64_t(blockIdx.x) * kNumElementsPerBlock;
    constexpr uint64_t kNumMatrixElements = uint64_t(kRows) * kRows;

    const auto update = [&]<uint32_t kNumElementsPerVector>() {
        using Vector = std::conditional_t<kNumElementsPerVector == 8, uint4, nv_bfloat16>;

        // Each vector lies within one matrix and contains at most one diagonal element.
        #pragma unroll
        for (uint32_t local_idx = threadIdx.x * kNumElementsPerVector; local_idx < kNumElementsPerBlock;
             local_idx += kNumThreads * kNumElementsPerVector) {
            const auto element_idx = block_offset + local_idx;
            if (element_idx < num_elements) {
                const auto matrix_idx = element_idx % kNumMatrixElements;
                const auto gram_vector = *reinterpret_cast<const Vector*>(gram + element_idx);
                auto polynomial_vector = *reinterpret_cast<const Vector*>(polynomial + element_idx);
                const auto diagonal_offset = math::ceil_div(matrix_idx, uint64_t(kRows) + 1) * (uint64_t(kRows) + 1) - matrix_idx;

                #pragma unroll
                for (uint32_t vector_idx = 0; vector_idx < kNumElementsPerVector; ++vector_idx) {
                    const auto base = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&gram_vector)[vector_idx]);
                    const auto squared = __bfloat162float(reinterpret_cast<const nv_bfloat16*>(&polynomial_vector)[vector_idx]);
                    auto value = fmaf(quadratic, squared, __fmul_rn(linear, base));
                    if (vector_idx == diagonal_offset)
                        value = __fadd_rn(value, identity);
                    reinterpret_cast<nv_bfloat16*>(&polynomial_vector)[vector_idx] = __float2bfloat16_rn(value);
                }
                *reinterpret_cast<Vector*>(polynomial + element_idx) = polynomial_vector;
            }
        }
    };

    // A contiguous view can start at an unaligned storage offset; only aligned views use vector accesses.
    if constexpr (kRows % 8 == 0) {
        if ((reinterpret_cast<uintptr_t>(gram) | reinterpret_cast<uintptr_t>(polynomial)) % alignof(uint4) == 0) {
            update.template operator()<8>();
            return;
        }
    }
    update.template operator()<1>();
}

template <uint32_t kBatches, uint32_t kRows, uint32_t kColumns, uint32_t kNumStages>
CUTLASS_GLOBAL __launch_bounds__(256, 1) void
sm100_bf16_newton_schulz_impl(
        uint64_t* ready,
        float identity_coeff, float linear_coeff, float quadratic_coeff,
        const __grid_constant__ cute::TmaDescriptor map_x,
        const __grid_constant__ cute::TmaDescriptor map_a,
        const __grid_constant__ cute::TmaDescriptor map_at,
        const __grid_constant__ cute::TmaDescriptor map_b,
        const __grid_constant__ cute::TmaDescriptor map_bt,
        const __grid_constant__ cute::TmaDescriptor map_xt,
        const __grid_constant__ cute::TmaDescriptor map_out) {
#if defined(__CUDA_ARCH__) and __CUDA_ARCH__ >= 1000
    // Thread configs: one mainloop warpgroup and one register-rich epilogue warpgroup.
    constexpr uint32_t kNumMainloopThreads = 128;
    constexpr uint32_t kNumEpilogueThreads = 128;
    constexpr uint32_t kNumMainloopWarps = kNumMainloopThreads / 32;
    constexpr uint32_t kTmaWarpIdx = 0;
    constexpr uint32_t kMmaWarpIdx = 1;
    constexpr uint32_t kTmemWarpIdx = 2;
    constexpr uint32_t kScheduleWarpIdx = 3;
    constexpr uint32_t kNumMainloopRegisters = 64;
    constexpr uint32_t kNumEpilogueRegisters = 256;

    // MMA configs: each CTA loads half of each operand's non-reduction dimension.
    constexpr uint32_t BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2, LOAD_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t STORE_BLOCK_M = 64, STORE_BLOCK_N = 128;
    constexpr uint32_t kNumKBlocksPerTile = BLOCK_M / BLOCK_K;
    constexpr uint32_t kTileRows = math::constexpr_ceil_div(kRows, BLOCK_M);
    constexpr uint32_t kTileColumns = math::constexpr_ceil_div(kColumns, BLOCK_N);
    constexpr uint32_t kNumColumnKBlocks = math::constexpr_ceil_div(kColumns, BLOCK_K);
    constexpr uint32_t kNumRowKBlocks = kTileRows * kNumKBlocksPerTile;
    constexpr uint32_t kSwizzleAMode = 128, kSwizzleBMode = 128, kSwizzleCDMode = 128;

    // Two accumulator tiles and two independent output slots overlap MMA with writeback.
    constexpr uint32_t kNumEpilogueStages = 2, kNumTMAStoreStages = 2;
    constexpr uint32_t kAccumColumns = BLOCK_N;
    constexpr uint32_t kNumTmemColumns = kAccumColumns * kNumEpilogueStages;
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    using Allocator = cute::TMEM::Allocator2Sm;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Scheduler = sched::NewtonSchulzScheduler<kTileRows, kTileColumns, kBatches>;
    using Triangle = sched::NewtonSchulzTriangle<kTileRows>;
    using Dependency = sched::NewtonSchulzDependency<Triangle::kNumTiles, kBatches>;
    constexpr uint32_t kReadyWords = math::constexpr_ceil_div(
        math::constexpr_ceil_div(kBatches * Triangle::kNumTiles, 32u),
        Dependency::kNumMaxCachedWords) * Dependency::kNumMaxCachedWords;

    struct SharedStorage {
        using cd_dtype = cutlass::bfloat16_t;
        alignas(kSharedMemoryAlignment) cd_dtype cd[kNumTMAStoreStages][LOAD_BLOCK_M * BLOCK_K];
        cd_dtype smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
        cd_dtype smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];
        Barrier full_barriers[kNumStages];
        Barrier empty_barriers[kNumStages];
        Barrier tmem_full_barriers[kNumEpilogueStages];
        Barrier tmem_empty_barriers[kNumEpilogueStages];
        Barrier capture_ready;
        Barrier capture_done;
        uint32_t tmem_ptr;
        sched::NewtonSchulzTaskQueue task_queue;
    };
    constexpr uint32_t kNumOperandBytes = kNumStages * (LOAD_BLOCK_M + LOAD_BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t kNumStoreBytes = kNumTMAStoreStages * LOAD_BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t kNumSMemBytes = kNumOperandBytes + kNumStoreBytes + kSharedMemoryAlignment;
    DG_STATIC_ASSERT(sizeof(SharedStorage) <= kNumSMemBytes, "Shared storage exceeds the launch allocation");
    DG_STATIC_ASSERT(kNumSMemBytes <= 232448, "Shared memory exceeds capacity");
    DG_STATIC_ASSERT(kNumMainloopThreads * kNumMainloopRegisters + kNumEpilogueThreads * kNumEpilogueRegisters <= 64512,
                     "Warpgroup register budgets exceed the CTA register pool");

    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];
    auto& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    const auto cta_rank = cute::block_rank_in_cluster();
    const bool leader = cta_rank == 0;

    // Initialization
    if (warp_idx == kTmaWarpIdx) {
        cute::prefetch_tma_descriptor(&map_x);
        cute::prefetch_tma_descriptor(&map_a);
        cute::prefetch_tma_descriptor(&map_at);
        cute::prefetch_tma_descriptor(&map_b);
        cute::prefetch_tma_descriptor(&map_bt);
        cute::prefetch_tma_descriptor(&map_xt);
        cute::prefetch_tma_descriptor(&map_out);
    }
    if (warp_idx == kMmaWarpIdx and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumStages; ++stage_idx) {
            shared_storage.full_barriers[stage_idx].init(2);
            shared_storage.empty_barriers[stage_idx].init(1);
        }
        #pragma unroll
        for (uint32_t stage_idx = 0; stage_idx < kNumEpilogueStages; ++stage_idx) {
            shared_storage.tmem_full_barriers[stage_idx].init(1);
            shared_storage.tmem_empty_barriers[stage_idx].init(2 * kNumEpilogueThreads);
        }
        shared_storage.capture_ready.init(1);
        shared_storage.capture_done.init(1);
        shared_storage.task_queue.initialize();
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == kTmemWarpIdx) {
        Allocator().allocate(kNumTmemColumns, &shared_storage.tmem_ptr);
    }
    comm::cluster_sync_with_relaxed_arrive();

    auto commit_barrier = [](const auto barrier) {
        constexpr uint32_t kNumCTAs = 2;
        constexpr uint16_t kCTAMask = (1u << kNumCTAs) - 1;
        cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<const uint64_t*>(barrier), kCTAMask);
    };
    Scheduler scheduler(shared_storage.task_queue);

    // Symmetry selects the stored triangle; the polynomial reduction captures its A tile last.
    // Each output tile uses a fixed K order, independent of producer completion order.
    // TMA zero-fills the final tile, keeping every capture window four panels wide.
    auto for_each_k_block = [&]<bool kPolynomial, bool kSymm = false, typename Process>(Process&& process) {
        if constexpr (kSymm) {
            const auto boundary = (scheduler.row_idx + 1) * kNumKBlocksPerTile;
            for (uint32_t block_idx = 0; block_idx < boundary; ++block_idx)
                process.template operator()<cute::UMMA::Major::K, cute::UMMA::Major::MN>(block_idx, kNumKBlocksPerTile);
            for (uint32_t block_idx = boundary; block_idx < kNumRowKBlocks; ++block_idx)
                process.template operator()<cute::UMMA::Major::MN, cute::UMMA::Major::MN>(block_idx, kNumKBlocksPerTile);
        } else if constexpr (not kPolynomial) {
            for (uint32_t block_idx = 0; block_idx < kNumColumnKBlocks; ++block_idx)
                process.template operator()<cute::UMMA::Major::K, cute::UMMA::Major::K>(block_idx, kNumKBlocksPerTile);
        } else {
            const auto begin = scheduler.col_idx * kNumKBlocksPerTile;
            const auto end = begin + kNumKBlocksPerTile;
            const auto mixed_end = (scheduler.row_idx + 1) * kNumKBlocksPerTile;
            for (uint32_t block_idx = 0; block_idx < begin; ++block_idx)
                process.template operator()<cute::UMMA::Major::K, cute::UMMA::Major::K>(block_idx, kNumKBlocksPerTile);
            for (uint32_t block_idx = end; block_idx < mixed_end; ++block_idx)
                process.template operator()<cute::UMMA::Major::K, cute::UMMA::Major::MN>(block_idx, kNumKBlocksPerTile);
            for (uint32_t block_idx = mixed_end; block_idx < kNumRowKBlocks; ++block_idx)
                process.template operator()<cute::UMMA::Major::MN, cute::UMMA::Major::MN>(block_idx, kNumKBlocksPerTile);
            for (uint32_t block_idx = begin; block_idx < end; ++block_idx)
                process.template operator()<cute::UMMA::Major::K, cute::UMMA::Major::K>(block_idx, block_idx - begin);
        }
    };

    if (warp_idx < kNumMainloopWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumMainloopRegisters>();
        uint32_t stage_idx = 0, phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
            phase ^= stage_idx == 0;
        };

        if (warp_idx == kScheduleWarpIdx and leader) {
            shared_storage.task_queue.template publish_tasks<Scheduler>(reinterpret_cast<uint32_t*>(ready + 2 * kReadyWords));
        } else if (warp_idx == kTmaWarpIdx) {
            // TMA producer: input storage is reusable only after both MMA and capture release it.
            uint32_t pending = 0xffffffffu;
            uint32_t capture_phase = 0;
            auto release_capture = [&](const uint32_t current_stage) {
                if (pending == current_stage) {
                    shared_storage.capture_done.wait(capture_phase);
                    capture_phase ^= 1;
                    pending = 0xffffffffu;
                }
            };
            while (scheduler.get_next_task()) {
                Dependency dependency(scheduler.batch_idx * Triangle::kNumTiles);
                auto run = [&]<bool kPolynomial, bool kSymm = false>() {
                    auto process = [&]<cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB>(
                            const uint32_t block_idx, const uint32_t capture_idx) {
                        if constexpr (kPolynomial or kSymm) {
                            if (block_idx % kNumKBlocksPerTile == 0) {
                                const auto panel_idx = block_idx / kNumKBlocksPerTile;
                                const auto offset = scheduler.batch_idx * Triangle::kNumTiles;
                                const auto ready_ptr = ready + (kSymm ? kReadyWords : 0);
                                dependency.wait_until_ready(ready_ptr, offset + Triangle::get_tile_idx(scheduler.row_idx, panel_idx));
                                if (kPolynomial and scheduler.row_idx != scheduler.col_idx)
                                    dependency.wait_until_ready(ready_ptr, offset + Triangle::get_tile_idx(scheduler.col_idx, panel_idx));
                            }
                        }
                        if constexpr (kPolynomial or kSymm) {
                            if (capture_idx == 0 and pending != 0xffffffffu)
                                release_capture(pending);
                            release_capture(stage_idx);
                        }
                        shared_storage.empty_barriers[stage_idx].wait(phase ^ 1);
                        if (cute::elect_one_sync()) {
                            const auto row_a = scheduler.row_idx * BLOCK_M + cta_rank * LOAD_BLOCK_M;
                            const auto row_b = scheduler.col_idx * BLOCK_N + cta_rank * LOAD_BLOCK_N;
                            const auto& normal = kSymm ? map_b : (kPolynomial ? map_a : map_x);
                            const auto& transposed = kSymm ? map_bt : map_at;
                            if constexpr (kMajorA == cute::UMMA::Major::K)
                                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, true>(
                                    &normal, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx],
                                    block_idx * BLOCK_K, row_a, 2, scheduler.batch_idx);
                            else
                                tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, true>(
                                    &transposed, &shared_storage.full_barriers[stage_idx], shared_storage.smem_a[stage_idx],
                                    row_a, block_idx * BLOCK_K, 2, scheduler.batch_idx);
                            if constexpr (kSymm) {
                                tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, true>(
                                    &map_xt, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx],
                                    row_b, block_idx * BLOCK_K, 2, scheduler.batch_idx);
                            } else if (scheduler.row_idx != scheduler.col_idx) {
                                if constexpr (kMajorB == cute::UMMA::Major::K)
                                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, true>(
                                        &normal, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx],
                                        block_idx * BLOCK_K, row_b, 2, scheduler.batch_idx);
                                else
                                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, true>(
                                        &transposed, &shared_storage.full_barriers[stage_idx], shared_storage.smem_b[stage_idx],
                                        row_b, block_idx * BLOCK_K, 2, scheduler.batch_idx);
                            }
                            const auto bytes = sizeof(shared_storage.smem_a[0]) * (not kSymm and scheduler.row_idx == scheduler.col_idx ? 2 : 4);
                            if (leader)
                                shared_storage.full_barriers[stage_idx].arrive_and_expect_tx(bytes);
                            else
                                shared_storage.full_barriers[stage_idx].arrive(0u);
                        }
                        if constexpr (kPolynomial)
                            if (capture_idx == 0)
                                pending = stage_idx;
                        advance_pipeline();
                    };
                    for_each_k_block.template operator()<kPolynomial, kSymm>(process);
                };
                if (scheduler.symm)
                    run.template operator()<false, true>();
                else if (scheduler.polynomial)
                    run.template operator()<true>();
                else
                    run.template operator()<false>();
            }
            #pragma unroll
            for (uint32_t stage_idx = 0; stage_idx < kNumStages; ++stage_idx)
                release_capture(stage_idx);
        } else if (warp_idx == kMmaWarpIdx and leader) {
            // MMA consumer: a stage's empty barrier tracks completion of all reads by the tensor core.
            while (scheduler.get_next_task()) {
                const auto accum_idx = scheduler.task_sequence_idx % 2;
                shared_storage.tmem_empty_barriers[accum_idx].wait((scheduler.task_sequence_idx / 2 & 1) ^ 1);
                ptx::tcgen05_after_thread_sync();
                uint32_t issued = 0;
                const auto blocks = scheduler.task_type != 0 ? kNumRowKBlocks : kNumColumnKBlocks;
                auto process = [&]<cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB>(
                        const uint32_t, const uint32_t capture_idx) {
                    shared_storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();
                    const auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(shared_storage.smem_a[stage_idx], 0, 0);
                    const auto b_desc = not scheduler.symm and scheduler.row_idx == scheduler.col_idx ? a_desc :
                        mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(shared_storage.smem_b[stage_idx], 0, 0);
                    const auto instruction = scheduler.symm ?
                        cute::UMMA::make_runtime_instr_desc(cute::UMMA::make_instr_desc<
                            cutlass::bfloat16_t, cutlass::bfloat16_t, float, BLOCK_M, BLOCK_N, kMajorB, kMajorA>()) :
                        cute::UMMA::make_runtime_instr_desc(cute::UMMA::make_instr_desc<
                            cutlass::bfloat16_t, cutlass::bfloat16_t, float, BLOCK_M, BLOCK_N, kMajorA, kMajorB>());
                    if (cute::elect_one_sync()) {
                        if (capture_idx == kNumKBlocksPerTile - 1) {
                            asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
                            shared_storage.capture_ready.arrive(0u, 1u);
                            shared_storage.capture_ready.arrive(1u, 1u);
                        }
                        #pragma unroll
                        for (uint32_t atom_idx = 0; atom_idx < 4; ++atom_idx) {
                            auto a_iter = a_desc;
                            auto b_iter = b_desc;
                            a_iter.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(a_desc.lo, 0, atom_idx * 16);
                            b_iter.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(b_desc.lo, 0, atom_idx * 16);
                            if (scheduler.symm)
                                ptx::SM100_MMA_F16BF16_2x1SM_SS::fma(b_iter, a_iter, accum_idx * kAccumColumns,
                                    issued != 0 or atom_idx != 0, instruction);
                            else
                                ptx::SM100_MMA_F16BF16_2x1SM_SS::fma(a_iter, b_iter, accum_idx * kAccumColumns,
                                    issued != 0 or atom_idx != 0, instruction);
                        }
                    }
                    __syncwarp();
                    commit_barrier(&shared_storage.empty_barriers[stage_idx]);
                    if (++issued == blocks)
                        commit_barrier(&shared_storage.tmem_full_barriers[accum_idx]);
                    __syncwarp();
                    advance_pipeline();
                };
                if (scheduler.symm)
                    for_each_k_block.template operator()<false, true>(process);
                else if (scheduler.polynomial)
                    for_each_k_block.template operator()<true>(process);
                else
                    for_each_k_block.template operator()<false>(process);
            }
            if (scheduler.task_sequence_idx != 0xffffffffu)
                shared_storage.tmem_empty_barriers[scheduler.task_sequence_idx % 2].wait(scheduler.task_sequence_idx / 2 & 1);
        }
    } else {
        // The linear term remains in registers while the quadratic term accumulates in TMEM.
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        const auto epilogue_warp_idx = warp_idx - kNumMainloopWarps;
        const auto local_row_idx = epilogue_warp_idx * 32 + lane_idx;
        uint4 captured[kNumKBlocksPerTile][8];
        uint32_t capture_phase = 0;
        uint32_t store_stage_idx = 0;
        uint32_t input_stage_idx = 0;
        auto smem_ptr = [&](const auto base, const uint32_t bank_idx) {
            return reinterpret_cast<uint8_t*>(base) + local_row_idx * 128 + (bank_idx ^ (local_row_idx % 8)) * 16;
        };
        while (scheduler.get_next_task()) {
            if (scheduler.polynomial) {
                input_stage_idx = (input_stage_idx + kNumRowKBlocks - kNumKBlocksPerTile) % kNumStages;
                shared_storage.capture_ready.wait(capture_phase);
                cutlass::arch::fence_view_async_shared();
                #pragma unroll
                for (uint32_t panel_idx = 0; panel_idx < kNumKBlocksPerTile; ++panel_idx) {
                    #pragma unroll
                    for (uint32_t bank_idx = 0; bank_idx < 8; ++bank_idx)
                        captured[panel_idx][bank_idx] = ptx::ld_shared(
                            reinterpret_cast<const uint4*>(smem_ptr(shared_storage.smem_a[input_stage_idx], bank_idx)));
                    input_stage_idx = (input_stage_idx + 1) % kNumStages;
                }
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                    shared_storage.capture_done.arrive();
                capture_phase ^= 1;
            } else {
                input_stage_idx = (input_stage_idx + (scheduler.symm ? kNumRowKBlocks : kNumColumnKBlocks)) % kNumStages;
            }
            const auto accum_idx = scheduler.task_sequence_idx % 2;
            shared_storage.tmem_full_barriers[accum_idx].wait(scheduler.task_sequence_idx / 2 & 1);
            ptx::tcgen05_after_thread_sync();
            if (scheduler.symm) {
                const auto epilogue_args = EpilogueOperatorArgs();
                auto epilogue_operator = epilogue::operators::Identity(epilogue_args);
                epilogue::sm100_store_cd_swap_ab<
                    BLOCK_M, LOAD_BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N, kSwizzleCDMode,
                    kNumTMAStoreStages, kNumEpilogueThreads, 0, GemmType::Batched, false>(
                        shared_storage, store_stage_idx, accum_idx * kAccumColumns, scheduler.row_idx * BLOCK_M,
                        scheduler.col_idx * BLOCK_N + cta_rank * LOAD_BLOCK_N, scheduler.batch_idx, false,
                        BLOCK_M, epilogue_warp_idx, lane_idx, epilogue_operator, false,
                        &shared_storage.tmem_empty_barriers[accum_idx], &shared_storage.tmem_empty_barriers[accum_idx], map_out);
                continue;
            }
            const auto row_idx = scheduler.row_idx * BLOCK_M + cta_rank * LOAD_BLOCK_M + local_row_idx;
            #pragma unroll
            for (uint32_t panel_idx = 0; panel_idx < kNumKBlocksPerTile; ++panel_idx) {
                const auto col_offset = panel_idx * BLOCK_K;
                const auto store_idx = store_stage_idx;
                if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                    cute::tma_store_wait<1>();
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                auto store_values = [&]<bool kDiagonal, bool kPolynomial>() {
                    constexpr uint32_t kLoadWidth = 32;
                    constexpr uint32_t kBanksPerLoad = kLoadWidth / 8;
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < BLOCK_K / kLoadWidth; ++load_idx) {
                        uint32_t values[kLoadWidth];
                        using Load = cute::SM100_TMEM_LOAD_32dp32b32x;
                        [&]<size_t... Indices>(cute::index_sequence<Indices...>) {
                            Load::copy(accum_idx * kAccumColumns + panel_idx * BLOCK_K + load_idx * kLoadWidth,
                                       values[Indices]...);
                        }(cute::make_index_sequence<kLoadWidth>());
                        cutlass::arch::fence_view_async_tmem_load();
                        #pragma unroll
                        for (uint32_t bank_part = 0; bank_part < kBanksPerLoad; ++bank_part) {
                            const auto bank_idx = load_idx * kBanksPerLoad + bank_part;
                            uint32_t packed[4];
                            #pragma unroll
                            for (uint32_t pair_idx = 0; pair_idx < 4; ++pair_idx) {
                                auto first = __uint_as_float(values[bank_part * 8 + pair_idx * 2]);
                                auto second = __uint_as_float(values[bank_part * 8 + pair_idx * 2 + 1]);
                                const auto col_idx = scheduler.col_idx * BLOCK_N + col_offset + bank_idx * 8 + pair_idx * 2;
                                if constexpr (kPolynomial) {
                                    const auto base_bits = reinterpret_cast<const uint32_t*>(&captured[panel_idx][bank_idx])[pair_idx];
                                    const auto base = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&base_bits));
                                    first = fmaf(quadratic_coeff, first, __fmul_rn(linear_coeff, base.x));
                                    second = fmaf(quadratic_coeff, second, __fmul_rn(linear_coeff, base.y));
                                    if constexpr (kDiagonal) {
                                        if (row_idx == col_idx)
                                            first = __fadd_rn(first, identity_coeff);
                                        if (row_idx == col_idx + 1)
                                            second = __fadd_rn(second, identity_coeff);
                                    }
                                }
                                auto first_bits = __float_as_uint(first);
                                auto second_bits = __float_as_uint(second);
                                packed[pair_idx] = math::cast_into_bf16_and_pack(first_bits, second_bits);
                            }
                            ptx::st_shared(smem_ptr(shared_storage.cd[store_idx], bank_idx), packed[0], packed[1], packed[2], packed[3]);
                        }
                    }
                };
                if (scheduler.polynomial) {
                    if (scheduler.row_idx == scheduler.col_idx)
                        store_values.template operator()<true, true>();
                    else
                        store_values.template operator()<false, true>();
                } else {
                    store_values.template operator()<false, false>();
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    cute::SM90_TMA_STORE_3D::copy(scheduler.polynomial ? &map_b : &map_a, shared_storage.cd[store_idx],
                        scheduler.col_idx * BLOCK_N + col_offset,
                        scheduler.row_idx * BLOCK_M + cta_rank * LOAD_BLOCK_M, scheduler.batch_idx);
                    cute::tma_store_arrive();
                }
                store_stage_idx ^= 1;
            }
            ptx::tcgen05_before_thread_sync();
            shared_storage.tmem_empty_barriers[accum_idx].arrive(0u);
            // Completion publication orders the global writes before downstream TMA loads.
            if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                ptx::tma_store_wait<>();
            cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                Dependency::publish_ready(ready + (scheduler.polynomial ? kReadyWords : 0), scheduler.tile_idx, cta_rank);
        }
    }
    // Both CTAs must finish reading the accumulators before releasing their shared TMEM allocation.
    comm::cluster_sync_with_relaxed_arrive();
    if (cutlass::canonical_warp_idx_sync() == kTmaWarpIdx)
        Allocator().free(0, kNumTmemColumns);
#endif
}

}
