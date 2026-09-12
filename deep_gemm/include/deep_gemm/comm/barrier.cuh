#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier_with_timeout_policy.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::comm {

// 60s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

// `WorkspaceT` is any workspace exposing `get_grid_sync_count_ptr<>`, `get_nvl_barrier_counter_ptr()` and
// `get_nvl_barrier_signal_ptr()`. Under `TrapOnly` both waits keep the same timeout and call the same handler, but
// leave the loop first: a `trap;` inside the wait loop makes ptxas allocate the registers of the region containing
// the loop against the kernel's launch bound and ignore the region's `setmaxnreg.inc` (spills in the SM90 fused
// kernel). `Diagnostic` keeps the loop it had, since its handler pulls in a `printf`.
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0,
          BarrierTimeoutPolicy kTimeoutPolicy = BarrierTimeoutPolicy::Diagnostic,
          typename WorkspaceT, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const WorkspaceT& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.template get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        if constexpr (kTimeoutPolicy == BarrierTimeoutPolicy::Diagnostic) {
            const auto start_clock = clock64();
            do {
                new_value = ptx::ld_acq(count_ptr);
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    handle_grid_sync_timeout<kTimeoutPolicy>(
                        sm_idx, thread_idx, kGridSyncIndex, old_value, new_value,
                        old_value ^ kFinishSumTag);
                }
            } while (((new_value ^ old_value) & kFinishSumTag) == 0);
        } else {
            const auto start_clock = clock64();
            bool timed_out = false;
            do {
                new_value = ptx::ld_acq(count_ptr);
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    timed_out = true;
                    break;
                }
            } while (((new_value ^ old_value) & kFinishSumTag) == 0);
            if (timed_out)
                handle_grid_sync_timeout<kTimeoutPolicy>(
                    sm_idx, thread_idx, kGridSyncIndex, old_value, new_value,
                    old_value ^ kFinishSumTag);
        }
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag,
          BarrierTimeoutPolicy kTimeoutPolicy = BarrierTimeoutPolicy::Diagnostic,
          typename WorkspaceT, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const WorkspaceT& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex, kTimeoutPolicy>(
            workspace, sm_idx, thread_idx, sync_scope);

    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            const auto start_clock = clock64();
            if constexpr (kTimeoutPolicy == BarrierTimeoutPolicy::Diagnostic) {
                while (ptx::ld_acq_sys(signal_ptr) != target) {
                    if (clock64() - start_clock >= kNumTimeoutCycles) {
                        handle_nvlink_barrier_timeout<kTimeoutPolicy>(
                            sym_buffer.rank_idx, *counter_ptr,
                            ptx::ld_acq_sys(signal_ptr), target,
                            signal_phase, signal_sign, kTag);
                    }
                }
            } else {
                bool timed_out = false;
                while (ptx::ld_acq_sys(signal_ptr) != target) {
                    if (clock64() - start_clock >= kNumTimeoutCycles) {
                        timed_out = true;
                        break;
                    }
                }
                if (timed_out)
                    handle_nvlink_barrier_timeout<kTimeoutPolicy>(
                        sym_buffer.rank_idx, *counter_ptr,
                        ptx::ld_acq_sys(signal_ptr), target,
                        signal_phase, signal_sign, kTag);
            }
        }
    }

    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex, kTimeoutPolicy>(
            workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
