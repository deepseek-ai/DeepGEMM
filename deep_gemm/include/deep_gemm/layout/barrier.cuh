#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

struct BarrierSignals {
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    uint32_t grid_sync_count[kNumMaxGridSyncCounters];
    uint32_t nvl_barrier_counter;
    int nvl_barrier_signals[2];  // The NVLink protocol alternates between +1 and -1.

    template <uint32_t kIndex = 0>
    CUTLASS_HOST_DEVICE uint32_t* get_grid_sync_count_ptr() {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return grid_sync_count + kIndex;
    }

    CUTLASS_HOST_DEVICE uint32_t* get_nvl_barrier_counter_ptr() {
        return &nvl_barrier_counter;
    }

    CUTLASS_HOST_DEVICE int* get_nvl_barrier_signal_ptr(const uint32_t& phase) {
        return nvl_barrier_signals + phase;
    }
};

DG_STATIC_ASSERT(sizeof(BarrierSignals) == 28, "Invalid barrier signal size");
DG_STATIC_ASSERT(offsetof(BarrierSignals, nvl_barrier_counter) == 16, "Invalid NVLink counter offset");
DG_STATIC_ASSERT(offsetof(BarrierSignals, nvl_barrier_signals) == 20, "Invalid NVLink signal offset");

} // namespace deep_gemm::layout
