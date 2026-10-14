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
