#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::sm120 {

// B128 XOR swizzle: col_byte ^ ((row & 7) << 4)
__device__ __forceinline__ int swizzle_b128(int row, int col_byte) {
    return col_byte ^ ((row & 7) << 4);
}

// ldmatrix A operand address (16x32 FP8 → ldmatrix.x4.b16)
// Maps 32 threads to 4 matrix fragments covering 16 rows × 32 cols
__device__ __forceinline__ void* ldmatrix_a_addr(
    char* smem_tile, int lane, int m_tile, int k_step,
    int row_stride, int mma_k
) {
    int row = (lane & 7) + ((lane >> 3) & 1) * 8 + m_tile * 16;
    int col = (lane >> 4) * 16 + k_step * mma_k;
    int col_swizzled = swizzle_b128(row, col);
    return smem_tile + row * row_stride + col_swizzled;
}

// ldmatrix B operand address (8x32 FP8 → ldmatrix.x2.b16)
// Maps 16 active threads to 2 matrix fragments covering 8 rows × 32 cols
__device__ __forceinline__ void* ldmatrix_b_addr(
    char* smem_tile, int lane, int n_tile, int k_step,
    int row_stride, int mma_k
) {
    int row = (lane & 7) + n_tile * 8;
    int col = ((lane >> 3) & 1) * 16 + k_step * mma_k;
    int col_swizzled = swizzle_b128(row, col);
    return smem_tile + row * row_stride + col_swizzled;
}

// Load A fragment via ldmatrix.x4
__device__ __forceinline__ void load_a_fragment(
    uint32_t (&frag)[4], char* smem_a, int lane, int m_tile, int k_step,
    int row_stride, int mma_k
) {
    void* addr = ldmatrix_a_addr(smem_a, lane, m_tile, k_step, row_stride, mma_k);
    ptx::SM90_U32x4_LDSM_N::copy(frag[0], frag[1], frag[2], frag[3], addr);
}

// Load B fragment via ldmatrix.x2
__device__ __forceinline__ void load_b_fragment(
    uint32_t (&frag)[2], char* smem_b, int lane, int n_tile, int k_step,
    int row_stride, int mma_k
) {
    void* addr = ldmatrix_b_addr(smem_b, lane, n_tile, k_step, row_stride, mma_k);
    ptx::SM90_U32x2_LDSM_N::copy(frag[0], frag[1], addr);
}

// Load UE8M0 scale factor from SMEM (packed 4 per int32)
// Returns the packed int32 containing 4 UE8M0 bytes for the thread's group
__device__ __forceinline__ uint32_t load_sf(const char* smem_sf, int idx) {
    return *reinterpret_cast<const uint32_t*>(smem_sf + idx * sizeof(int32_t));
}

} // namespace deep_gemm::sm120

#endif
