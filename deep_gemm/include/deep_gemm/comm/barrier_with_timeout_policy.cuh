#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::comm {

enum class BarrierTimeoutPolicy {
    Diagnostic,
    TrapOnly,
};

template <BarrierTimeoutPolicy kTimeoutPolicy>
CUTLASS_DEVICE void handle_grid_sync_timeout(
    const uint32_t sm_idx, const uint32_t thread_idx,
    const uint32_t grid_sync_idx, const uint32_t old_value,
    const uint32_t new_value, const uint32_t expected_tag) {
    if constexpr (kTimeoutPolicy == BarrierTimeoutPolicy::Diagnostic) {
        printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
               sm_idx, thread_idx, grid_sync_idx, old_value, new_value,
               expected_tag);
        DG_DEVICE_ASSERT(false and "Grid sync timeout");
    } else {
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "Grid sync timeout");
    }
}

template <BarrierTimeoutPolicy kTimeoutPolicy>
CUTLASS_DEVICE void handle_nvlink_barrier_timeout(
    const int rank_idx, const int counter, const int signal,
    const int target, const int phase, const int sign, const int tag) {
    if constexpr (kTimeoutPolicy == BarrierTimeoutPolicy::Diagnostic) {
        printf("DeepGEMM NVLink barrier timeout: rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
               rank_idx, counter, signal, target, phase, sign, tag);
        DG_DEVICE_ASSERT(false and "NVLink barrier timeout");
    } else {
        DG_TRAP_ONLY_DEVICE_ASSERT(false and "NVLink barrier timeout");
    }
}

} // namespace deep_gemm::comm
