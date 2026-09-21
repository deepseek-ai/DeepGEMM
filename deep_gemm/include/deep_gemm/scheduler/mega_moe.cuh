#pragma once

#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// TODO: add MegaMoE namespace
namespace deep_gemm::sched {

// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock
// constexpr func, no runtime overhead in kernel
CUTLASS_HOST_DEVICE constexpr
int get_num_l1_warmup_waves(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters) {
    // The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_clusters, num_l2_n_clusters);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

    // When each M block has more L1 tasks than L2 tasks, the interleaved schedule
    // leaves `(num_l1_pair_n_blocks - num_l2_pair_n_blocks)` extra L1 tasks behind
    // per M block. To avoid deadlock, no L2 task for an M block should be scheduled
    // before that block's L1 tasks have been issued. The last M block is the
    // bottleneck: it needs its own `num_l1_pair_n_blocks` L1 tasks, and the preceding
    // `num_total_m_blocks - 1` M blocks accumulate
    // `(num_total_m_blocks - 1) * (num_l1_pair_n_blocks - num_l2_pair_n_blocks)`
    // pending L1 tasks, so we issue them during the warmup phase. Add one extra
    // CTA-pair wave to cover partial-wave rounding.
    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    // TODO: may delay combine NVLink
    return cute::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// Host-side ring capacity helper for the MegaMoE schedule.
CUTLASS_HOST_DEVICE constexpr int get_num_max_live_pool_blocks(
    const int& num_total_m_blocks,
    const int& num_sms,
    const int& hidden,
    const int& intermediate_hidden) {
    constexpr int kMegaMoEBlockN = 128;
    constexpr int kNumCTAsPerCluster = 2;

    DG_UNIFIED_ASSERT((intermediate_hidden * 2) % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    DG_UNIFIED_ASSERT(hidden % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    const int num_clusters = num_sms / kNumCTAsPerCluster;
    const int num_l1_n_clusters = intermediate_hidden * 2 / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l2_n_clusters = hidden / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const int num_l1_waves = math::constexpr_ceil_div(num_l1_clusters, num_clusters);
    const int num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters);
    const int num_l1_warmup_waves = cute::min(num_min_l1_warmup_waves, num_l1_waves);
    const int num_l1_warmup_clusters = cute::min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const int num_live_blocks_after_warmup = math::constexpr_ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);

    // Conservative closed-form bound for the L2/L1 alternating tail. If L1 advances faster in
    // M-block space (B1 < B2), live blocks can grow by about total_m_blocks * (B2 - B1) / B2.
    // Add one global-wave margin to cover partial waves / pipeline lag.
    // This is slightly larger than the tight bound, but cheaper and simpler.
    // TODO: refactor here
    const int frontier_growth = num_l2_n_clusters > num_l1_n_clusters ?
        math::constexpr_ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    const int wave_margin = math::constexpr_ceil_div(
        num_clusters, cute::min(num_l1_n_clusters, num_l2_n_clusters));
    return cute::min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    TaskInfo(): TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& pair_n_block_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_cluster_idx(pair_n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m), shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_DEVICE
    uint32_t is_valid() const {
        return (block_phase != BlockPhase::None);
    }

    CUTLASS_DEVICE
    uint32_t get_umma_aligned_valid_m() const {
        return math::align(valid_m, 16u);
    }

    CUTLASS_DEVICE uint32_t is_shared() const {
        return kHasSharedExperts ? (block_phase > BlockPhase::Linear2) : false;
    }
};

DG_STATIC_ASSERT(sizeof(sched::TaskInfo<true>) == sizeof(sched::TaskInfo<false>), "Invalid layout");

#if defined(__CUDACC__) or defined(__CLION_IDE__)

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks,
          uint32_t kNumSharedExperts = 0,
          uint32_t SHARED_BLOCK_K = BLOCK_K,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
          uint32_t kNumL2Clusters = kNumL2BlockNs / 2>
struct MegaMoEScheduler {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % SHARED_BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % SHARED_BLOCK_K == 0, "Invalid shared shape");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");

    // Workspace
    const layout::Workspace& workspace;

    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = 2;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts.
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

    // Per-scheduler warmup waves; all CTA-pair schedulers together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace):
        workspace(workspace) {}

    CUTLASS_DEVICE MegaMoEScheduler(const layout::Workspace& workspace,
                                    Barrier* task_info_full_barriers,
                                    Barrier* task_info_empty_barriers,
                                    task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    CUTLASS_DEVICE void release_task_info() const {
        task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array.
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSMs / 2);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs / 2, kNumL1Clusters, kNumL2Clusters);
        num_sched_l1_waves = cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

    CUTLASS_DEVICE task_info_t create_task(const BlockPhase& block_phase,
                                        const uint32_t& task_idx,
                                        const uint32_t& num_clusters,
                                        const uint32_t& shape_n,
                                        const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t m_block_idx = task_idx / num_clusters;
        const uint32_t n_cluster_idx = task_idx % num_clusters;

        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task fall in the expert
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange the expert info
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_task_idx(const uint32_t* global_task_count_ptr) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(global_task_count_ptr, 1u);
        return ptx::exchange(result, 0);
    }

    CUTLASS_DEVICE task_info_t get_next_task() {
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally this is one CTA-pair wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                const uint32_t l1_task_idx = get_next_task_idx(workspace.get_l1_task_count_ptr());
                if (l1_task_idx >= num_total_m_blocks * kNumL1Clusters) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task(BlockPhase::Linear1, l1_task_idx, kNumL1Clusters, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                const uint32_t l2_task_idx = get_next_task_idx(workspace.get_l2_task_count_ptr());
                if (l2_task_idx >= num_total_m_blocks * kNumL2Clusters)
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                auto task_info = create_task(BlockPhase::Linear2, l2_task_idx, kNumL2Clusters, L2_SHAPE_N, L2_SHAPE_K);

                // Wait until all required L1 tasks are fetched
                const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * kNumL1Clusters;
                while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx < 2) {
            task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
            ptx::st_async_cluster(
                task_infos + sched_stage_idx, task_info,
                lane_idx, task_info_full_barriers[sched_stage_idx]
            );
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        const uint32_t num_tasks = num_m_blocks * kNumNClusters;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            const uint32_t task_idx = get_next_task_idx(task_count_ptr);
            if (task_idx >= num_tasks)
                break;
            const uint32_t m_block_idx = task_idx / kNumNClusters;
            const uint32_t n_cluster_idx = task_idx % kNumNClusters;
            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

    CUTLASS_DEVICE void mainloop(const uint32_t& num_tokens) {
        const auto lane_idx = ptx::get_lane_idx();

        if constexpr (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            // Shared expert L2 tasks depend on SharedLinear1 completion.
            shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};

// SM90 one-CTA-per-task dynamic scheduler (no 2-SM clustering), used by the
// fused SM90 NVFP4 MegaMoE kernel. Unlike `MegaMoEScheduler` above (which
// pairs adjacent CTAs into a 2-SM cluster per task), each SM independently
// claims and executes whole L1/L2 tasks.
//
// The weight-loader warp is the producer: it claims globally indexed L1/L2
// tasks and publishes them through a two-stage shared-memory mailbox.  The A
// loader and math warps consume the exact same payload, so task assignment is
// dynamic across SMs without requiring every role to race on a global atomic.
// After a minimal L1-only warm-up, each producer alternates L2 and L1 claims.
struct alignas(16) SM90TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_block_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    SM90TaskInfo(): SM90TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    SM90TaskInfo(const BlockPhase& block_phase,
                 const uint32_t& local_expert_idx,
                 const uint32_t& m_block_idx,
                 const uint32_t& n_block_idx,
                 const uint32_t& pool_block_idx,
                 const uint32_t& valid_m,
                 const uint32_t& shape_n,
                 const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx), n_block_idx(n_block_idx),
        pool_block_idx(pool_block_idx), valid_m(valid_m),
        shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_HOST_DEVICE bool is_valid() const {
        return block_phase != BlockPhase::None;
    }
};

DG_STATIC_ASSERT(sizeof(SM90TaskInfo) == 32, "Invalid task payload layout");

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N>
struct InterleavedMegaMoEScheduler {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using task_info_t = SM90TaskInfo;

    static constexpr uint32_t kNumScheduleStages = 2;
    static constexpr uint32_t kNumL1WavesDone = 0xffffffffu;

    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid L1 shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid L2 shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid L1 K shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid L2 K shape");
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "L1 readiness mask is too small");
    DG_STATIC_ASSERT(kNumL2BlockNs >= kNumL1BlockNs,
                     "Alternating scheduler requires at least as many L2 tasks as L1 tasks");

    const layout::Workspace& workspace;
    Barrier* task_info_full_barriers;
    Barrier* task_info_empty_barriers;
    task_info_t* task_infos;

    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;
    uint32_t num_l1_warmup_waves = 0;

    CUTLASS_DEVICE
    InterleavedMegaMoEScheduler(
            const layout::Workspace& workspace,
            Barrier* task_info_full_barriers,
            Barrier* task_info_empty_barriers,
            task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_schedule_pipeline() {
        sched_stage_idx ^= 1u;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE void wait_schedule_barrier(
            const Barrier& barrier,
            const uint32_t& expected_phase,
            const uint32_t& wait_kind) const {
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
        const auto start_clock = clock64();
        while (!barrier.try_wait(expected_phase)) {
            if (clock64() - start_clock >=
                (DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS + 2ll) * 2000000000ll) {
                if (ptx::get_lane_idx() == 0) {
                    printf("DeepGEMM scheduler mbarrier timeout: "
                           "sm=%u, warp=%u, wait_kind=%u, stage=%u, "
                           "phase=%u\n",
                           static_cast<uint32_t>(blockIdx.x),
                           static_cast<uint32_t>(threadIdx.x / 32),
                           wait_kind, sched_stage_idx, expected_phase);
                }
                DG_DEVICE_ASSERT(false and "Scheduler mbarrier timeout");
            }
        }
#else
        barrier.wait(expected_phase);
#endif
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
                const auto start_clock = clock64();
                bool timeout_reported = false;
#endif
                do {
                    value = ptx::ld_volatile(
                        workspace.get_expert_recv_count_sum_ptr(expert_idx));
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
                    const auto elapsed_clocks = clock64() - start_clock;
                    if (!timeout_reported &&
                        elapsed_clocks >=
                            DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS * 2000000000ll) {
                        if (blockIdx.x == 0) {
                            printf("DeepGEMM scheduler recv-count timeout: "
                                   "expert=%u, lane=%u, tokens=%u, arrivals=%u, "
                                   "target_arrivals=%u",
                                   expert_idx, ptx::get_lane_idx(),
                                   static_cast<uint32_t>(value),
                                   static_cast<uint32_t>(value >> 32),
                                   kNumSMs * kNumRanks);
                            #pragma unroll
                            for (uint32_t rank_idx = 0;
                                 rank_idx < kNumRanks; ++rank_idx) {
                                const auto rank_value = ptx::ld_volatile(
                                    workspace.get_expert_recv_count_ptr(
                                        rank_idx, expert_idx));
                                printf(" rank%u=(tokens=%u,arrivals=%u)",
                                       rank_idx,
                                       static_cast<uint32_t>(rank_value),
                                       static_cast<uint32_t>(rank_value >> 32));
                            }
                            printf("\n");
                        }
                        timeout_reported = true;
                    }
                    if (elapsed_clocks >=
                        (DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS + 1ll) *
                            2000000000ll) {
                        DG_DEVICE_ASSERT(false and
                                         "Scheduler recv-count timeout");
                    }
#endif
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_pool_block_offset(kNumExpertsPerRank);
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1BlockNs;
        const uint32_t num_total_l1_waves =
            math::ceil_div(num_total_l1_tasks, kNumSMs);
        const uint32_t min_l1_warmup_waves = static_cast<uint32_t>(get_num_l1_warmup_waves(
            static_cast<int>(num_total_m_blocks), static_cast<int>(kNumSMs),
            static_cast<int>(kNumL1BlockNs), static_cast<int>(kNumL2BlockNs)));
        num_l1_warmup_waves =
            cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

    CUTLASS_DEVICE task_info_t create_task(
            const BlockPhase& block_phase,
            const uint32_t& task_idx,
            const uint32_t& num_n_blocks,
            const uint32_t& shape_n,
            const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;

        task_info_t result(
            block_phase, 0, 0, n_block_idx, pool_block_idx, 0,
            shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks =
                math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset =
                block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank &&
                pool_block_idx >= lane_pool_block_offset &&
                pool_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            if (owner_mask) {
                const uint32_t owner_lane_idx =
                    static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx =
                    pool_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m =
                    cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_task_idx(uint32_t* task_count_ptr) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr, 1u);
        return ptx::exchange(result, 0);
    }

    // Producer-side dynamic claim.  Task counters describe issued work, while
    // the activation loader's acquire waits below the scheduler protect actual
    // data completion.
    CUTLASS_DEVICE task_info_t claim_next_task() {
        while (true) {
            if (num_l1_warmup_waves != kNumL1WavesDone &&
                num_l1_warmup_waves > 0) {
                -- num_l1_warmup_waves;
                const uint32_t task_idx =
                    get_next_task_idx(workspace.get_l1_task_count_ptr());
                if (task_idx >= num_total_m_blocks * kNumL1BlockNs) {
                    num_l1_warmup_waves = kNumL1WavesDone;
                    continue;
                }
                return create_task(
                    BlockPhase::Linear1, task_idx, kNumL1BlockNs,
                    L1_SHAPE_N, L1_SHAPE_K);
            }

            const uint32_t task_idx =
                get_next_task_idx(workspace.get_l2_task_count_ptr());
            if (task_idx >= num_total_m_blocks * kNumL2BlockNs)
                break;

            if (num_l1_warmup_waves != kNumL1WavesDone)
                num_l1_warmup_waves = 1;

            auto task_info = create_task(
                BlockPhase::Linear2, task_idx, kNumL2BlockNs,
                L2_SHAPE_N, L2_SHAPE_K);
            const uint32_t num_required_l1_tasks =
                (task_info.pool_block_idx + 1) * kNumL1BlockNs;
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
            const auto start_clock = clock64();
#endif
            while (true) {
                const auto current_l1_tasks =
                    ptx::ld_volatile(workspace.get_l1_task_count_ptr());
                if (current_l1_tasks >= num_required_l1_tasks)
                    break;
#if defined(DG_SM90_NVFP4_MOE_COUNTER_DEBUG)
                if (clock64() - start_clock >=
                    (DG_SM90_NVFP4_MOE_TIMEOUT_SECONDS + 2ll) *
                        2000000000ll) {
                    printf("DeepGEMM scheduler L1-task timeout: "
                           "current=%u, target=%u, pool_block=%u\n",
                           current_l1_tasks, num_required_l1_tasks,
                           task_info.pool_block_idx);
                    DG_DEVICE_ASSERT(false and
                                     "Scheduler L1-task timeout");
                }
#endif
            }
            return task_info;
        }
        return task_info_t();
    }

    CUTLASS_DEVICE void wait_task_slot_empty() const {
        wait_schedule_barrier(
            task_info_empty_barriers[sched_stage_idx],
            sched_phase ^ 1u, 1u);
    }

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info) {
        if (cute::elect_one_sync()) {
            task_infos[sched_stage_idx] = task_info;
            __threadfence_block();
            task_info_full_barriers[sched_stage_idx].arrive();
        }
        __syncwarp();
        advance_schedule_pipeline();
    }

    // Consumer-side mailbox read.  Every role has an independent mailbox
    // cursor but observes the same two-stage sequence.
    CUTLASS_DEVICE bool get_published_task(task_info_t& task_info) {
        wait_schedule_barrier(
            task_info_full_barriers[sched_stage_idx], sched_phase, 2u);
        asm volatile("" ::: "memory");
        task_info = task_infos[sched_stage_idx];
        advance_schedule_pipeline();
        return task_info.is_valid();
    }

    // Called once by each math warp after the first GEMM stage is ready.  At
    // that point the activation loader has necessarily consumed the payload,
    // so the producer may safely recycle the mailbox slot.
    CUTLASS_DEVICE void release_task_info(const uint32_t& lane_idx) const {
        if (lane_idx == 0)
            task_info_empty_barriers[sched_stage_idx ^ 1u].arrive();
    }
};

#endif

} // namespace deep_gemm::sched
