#pragma once

#include <cute/arch/mma_sm100_desc.hpp>

namespace deep_gemm {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
    MXF4        = 2,
};

enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
    KGroupedContiguousWithPsumLayout    = 6,
};

constexpr CUTLASS_HOST_DEVICE bool is_m_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::MGroupedContiguous:                  return true;
        case GemmType::MGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

constexpr CUTLASS_HOST_DEVICE bool is_k_grouped_contiguous(const GemmType& gemm_type) {
    switch (gemm_type) {
        case GemmType::KGroupedContiguous:                  return true;
        case GemmType::KGroupedContiguousWithPsumLayout:    return true;
        default: return false;
    }
}

// Number of warps assigned along N for the warp-level `mma.sync` SM89 GEMM kernel.
// Shared by the host-side heuristics and the JIT-compiled device code so that both agree on
// the warp tiling without having to plumb it through `Layout`/`LaunchConfig`.
// Returns 0 when no valid split exists.
constexpr CUTLASS_HOST_DEVICE uint32_t get_mma_sync_warps_n(const uint32_t& num_warps,
                                                             const uint32_t& block_m,
                                                             const uint32_t& block_n) {
    // One FP32 accumulator register per (16x8) MMA tile per lane; keeping this at 64 leaves
    // enough registers for the double-buffered accumulators, the A/B fragments and the addresses.
    constexpr uint32_t kMaxAccumRegs = 64;

    uint32_t best_warps_n = 0;
    uint32_t best_score = 0xffffffffu;
    for (uint32_t warps_n = 1; warps_n <= num_warps; warps_n *= 2) {
        if (num_warps % warps_n != 0 or block_n % warps_n != 0)
            continue;
        const uint32_t warps_m = num_warps / warps_n;
        if (block_m % warps_m != 0)
            continue;

        const uint32_t warp_m = block_m / warps_m, warp_n = block_n / warps_n;
        if (warp_m % 16 != 0 or warp_n % 8 != 0)
            continue;
        if ((warp_m / 16) * (warp_n / 8) * 4 > kMaxAccumRegs)
            continue;

        // Prefer the most square warp tile: it balances the `ldmatrix` traffic of both operands
        const uint32_t score = warp_m > warp_n ? warp_m - warp_n : warp_n - warp_m;
        if (score < best_score)
            best_score = score, best_warps_n = warps_n;
    }
    return best_warps_n;
}

enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};

// Host-provided epilogue parameters. Operators keep their per-thread state separately;
// only the fields consumed by the selected operator are meaningful.
struct EpilogueOperatorArgs {
    uint32_t* sfd = nullptr;
    uint32_t sfd_stride = 0;
    uint32_t shape_m = 0, shape_n = 0;
    float alpha = 1.0f;
    float clamp_value = 0.0f;  // Zero disables clamping
    uint32_t* clamped_count = nullptr;
    uint32_t* clamp_count_ready = nullptr;  // Per-stream GPU launch tag for in-kernel initialization
};

static constexpr uint32_t kNumDeviceLocalityDomains = 2;

} // namespace deep_gemm
