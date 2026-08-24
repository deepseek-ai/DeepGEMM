#pragma once

#include <cuda/std/cstdint>

namespace deep_gemm::layout {

// Single production schedule for the Hopper Humming FP8 x MXFP4 kernel.
// Host shared-memory sizing, TMA descriptors, JIT instantiation, and device
// launch bounds all consume this contract.
struct Sm90HummingMoeSchedule {
    static constexpr uint32_t block_m = 64;
    static constexpr uint32_t block_n = 128;
    static constexpr uint32_t block_k = 128;
    static constexpr uint32_t num_stages = 3;
    static constexpr uint32_t num_dispatch_threads = 64;
    static constexpr uint32_t num_non_epilogue_threads = 64;
    static constexpr uint32_t num_epilogue_threads = 128;
};

} // namespace deep_gemm::layout
