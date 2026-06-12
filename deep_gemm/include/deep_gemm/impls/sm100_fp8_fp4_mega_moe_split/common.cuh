#pragma once

#include <stdint.h>

#include <deep_gemm/common/compile.cuh>

namespace deep_gemm::mega_moe_split {

// Shared `state` tensor layout for the split-kernel pipeline: small device counters used to
// couple the kernels through the CUDA-graph dependency edges and to report progress to the
// host (consumed L2 blocks, launched CTAs, reduced tokens).
enum class SplitStateOffset : uint32_t {
    K1ReadyTasks = 0,
    K1DoneBlocks = 1,
    K2ClaimCounter = 2,
    K2DoneTasks = 3,
    K2DoneBlocks = 4,
    K3DoneElements = 5,
    K2Checksum = 6,
    NumOffsets = 7,
};

constexpr CUTLASS_HOST_DEVICE uint32_t get_state_offset(const SplitStateOffset offset) {
    return static_cast<uint32_t>(offset);
}

} // namespace deep_gemm::mega_moe_split
