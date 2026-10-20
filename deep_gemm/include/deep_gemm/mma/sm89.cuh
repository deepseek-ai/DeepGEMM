#pragma once

#include <cuda_fp8.h>

#include <deep_gemm/common/exception.cuh>

#if defined(DG_IN_CUDA_COMPILATION)
namespace deep_gemm::mma::sm89 {

// `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` is the Ada Lovelace (SM89) FP8 Tensor Core
// instruction. It lowers to `QMMA.16832.F32.E4M3.E4M3`; Ada is the first architecture to have FP8
// Tensor Cores, so this is *not* available on Ampere (SM80).
//
// Fragment layouts (all `groupID`/`threadID_in_group` refer to the lane index within the warp):
//   A (16x32, row-major, K-contiguous): a0/a1 -> rows `lane / 4` / `lane / 4 + 8`, k = `(lane % 4) * 4 + [0, 4)`
//                                    a2/a3 -> same rows, k offset by 16
//   B (32x8, col-major i.e. K-contiguous): b0 -> col `lane / 4`, k = `(lane % 4) * 4 + [0, 4)`
//                                    b1 -> same col, k offset by 16
//   C/D (16x8, row-major): d0/d1 -> row `lane / 4`, col `(lane % 4) * 2 + [0, 2]`
//                         d2/d3 -> row `lane / 4 + 8`, same columns
struct FP8MMAM16N8K32 {
    static constexpr uint32_t M = 16;
    static constexpr uint32_t N = 8;
    static constexpr uint32_t K = 32;
    static constexpr uint32_t kNumAccum = 4;
    static constexpr uint32_t kNumRegsA = 4;
    static constexpr uint32_t kNumRegsB = 2;

    CUTLASS_DEVICE static void mma(float& d_0, float& d_1, float& d_2, float& d_3,
                                   const uint32_t (&a)[kNumRegsA],
                                   const uint32_t (&b)[kNumRegsB]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d_0), "+f"(d_1), "+f"(d_2), "+f"(d_3)
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};

} // namespace deep_gemm::mma::sm89
#endif