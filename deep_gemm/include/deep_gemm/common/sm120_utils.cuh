#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

#include <cute/swizzle.hpp>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::sm120 {

// Swizzle: uses CuTe Swizzle<B,M,S> for position-independent swizzle math.
// TMA writes with hardware swizzle (B128, B64, B32). The XOR pattern uses
// physical SMEM address bits. With 128B-aligned base (guaranteed by __align__(1024)),
// the swizzle becomes position-independent.
//
// CuTe mapping: Swizzle<BBits, 4, 3> where BBits = log2(swizzle_bytes / 16).
//   B128: Swizzle<3,4,3>  B64: Swizzle<2,4,3>  B32: Swizzle<1,4,3>
//
// SwizzleContext pre-computes the row-dependent XOR bits once per row,
// then only XOR + ADD per ldmatrix call (saves 2 instrs per K-step).

template <int swizzle_bytes>
using CuTeSwizzle = cute::Swizzle<__builtin_ctz(swizzle_bytes) - 4, 4, 3>;

template <int swizzle_bytes>
struct SwizzleContext {
    int row_base_addr;
    int row_xor_bits;

    __device__ __forceinline__ void init(int row, int row_stride) {
        row_base_addr = row * row_stride;
        row_xor_bits = CuTeSwizzle<swizzle_bytes>::apply(row_base_addr) ^ row_base_addr;
    }

    __device__ __forceinline__ void* addr(char* smem_tile, int col_byte) const {
        return smem_tile + row_base_addr + (col_byte ^ row_xor_bits);
    }
};

template <int swizzle_bytes>
__device__ __forceinline__ int swizzle(int row, int col_byte, int row_stride) {
    int flat = row * row_stride + col_byte;
    return CuTeSwizzle<swizzle_bytes>::apply(flat) - row * row_stride;
}

// ldmatrix wrappers

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, void* smem) {
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
         : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
         : "l"(__cvta_generic_to_shared(smem))
         : "memory");
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t& d0, uint32_t& d1, void* smem) {
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
         : "=r"(d0), "=r"(d1)
         : "l"(__cvta_generic_to_shared(smem))
         : "memory");
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& d0, uint32_t& d1, void* smem) {
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0, %1}, [%2];\n"
         : "=r"(d0), "=r"(d1)
         : "l"(__cvta_generic_to_shared(smem))
         : "memory");
}

// ldmatrix for FP4 .b4x16_p64: reads packed+padded FP4 from SMEM, unpacks to uint8 in regs
// .m8n16.x2: loads 2 matrices of 8×16 4-bit elements. Threads 0-15 provide addresses.
__device__ __forceinline__ void ldmatrix_m8n16_x2_b4x16_p64(
    uint32_t& d0, uint32_t& d1, void* smem) {
    asm volatile ("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
         : "=r"(d0), "=r"(d1)
         : "l"(__cvta_generic_to_shared(smem))
         : "memory");
}

// ldmatrix for FP4 .b4x16_p64, 4 matrices (A operand, m16k32 needs 4 regs).
// .m8n16.x4: loads 4 matrices of 8×16 4-bit elements; all 32 lanes provide addresses.
__device__ __forceinline__ void ldmatrix_m8n16_x4_b4x16_p64(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, void* smem) {
    asm volatile ("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0, %1, %2, %3}, [%4];\n"
         : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
         : "l"(__cvta_generic_to_shared(smem))
         : "memory");
}

// Fragment loaders: pre-computed swizzle variants (fast path)

// Load B fragment from MN-major SMEM B[BLOCK_K, BLOCK_N] via ldmatrix.trans.x2
// MN-major: rows=K, cols=N(bf16 contiguous). .trans reads columns (N-contiguous).
// Each thread provides the address of a K-row at the N-tile start.
// Threads 0-7: K-rows [k_base..k_base+7], threads 8-15: K-rows [k_base+8..k_base+15].
template <int swizzle_bytes>
__device__ __forceinline__ void load_b_fragment_trans_x2(
    uint32_t (&frag)[2], char* smem_b,
    int lane, int k_step, int mma_k, int n_tile, int row_stride_bytes
) {
    const int k_row = (lane & 7) + ((lane >> 3) & 1) * 8 + k_step * mma_k;
    const int n_col_byte = n_tile * 8 * 2;
    int flat = k_row * row_stride_bytes + n_col_byte;
    if constexpr (swizzle_bytes > 0)
        flat = CuTeSwizzle<swizzle_bytes>::apply(flat);
    void* addr = smem_b + flat;
    ldmatrix_x2_trans(frag[0], frag[1], addr);
}


// Load A fragment using pre-computed SwizzleContext
// ctx must be initialized with the A row for this lane: (lane&7) + ((lane>>3)&1)*8 + m_tile*16
template <int swizzle_bytes>
__device__ __forceinline__ void load_a_fragment(
    uint32_t (&frag)[4], char* smem_a,
    const SwizzleContext<swizzle_bytes>& ctx, int lane, int k_step, int mma_k
) {
    int col = (lane >> 4) * 16 + k_step * mma_k;
    void* addr = ctx.addr(smem_a, col);
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], addr);
}

// Load A fragment from .b4x16_p64 padded SMEM via ldmatrix.m8n16.x4 (fp4 A operand).
// Same address computation as load_a_fragment (fp8); only the ldmatrix opcode differs.
// Caller must shift each returned reg << 2 (same as the fp4 B path).
template <int swizzle_bytes>
__device__ __forceinline__ void load_a_fragment_b4x16(
    uint32_t (&frag)[4], char* smem_a,
    const SwizzleContext<swizzle_bytes>& ctx, int lane, int k_step, int mma_k
) {
    int col = (lane >> 4) * 16 + k_step * mma_k;
    void* addr = ctx.addr(smem_a, col);
    ldmatrix_m8n16_x4_b4x16_p64(frag[0], frag[1], frag[2], frag[3], addr);
}

// Load B fragment pair (2 N-tiles) using pre-computed SwizzleContext
// ctx must be initialized with the B row for this lane: (lane&7) + ((lane>>3)&1)*8 + np*16
template <int swizzle_bytes>
__device__ __forceinline__ void load_b_fragment_x4(
    uint32_t (&frag)[4], char* smem_b,
    const SwizzleContext<swizzle_bytes>& ctx, int lane, int k_step, int mma_k
) {
    int col = (lane >> 4) * 16 + k_step * mma_k;
    void* addr = ctx.addr(smem_b, col);
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], addr);
}

// Load B fragment for one N-tile covering 2 K-steps via ldmatrix.x4.
// 4 thread groups (8 lanes each) load 4 consecutive 16-byte K-groups:
//   group 0 (lanes 0-7):   K bytes [base, base+16)
//   group 1 (lanes 8-15):  K bytes [base+16, base+32)
//   group 2 (lanes 16-23): K bytes [base+32, base+48)
//   group 3 (lanes 24-31): K bytes [base+48, base+64)
// Output: {r0,r1} = K-step 0 B operand (consecutive regs), {r2,r3} = K-step 1.
// ctx must be initialized with: row = (lane&7) + nt*8, stride = kSMEMKBytes
template <int swizzle_bytes>
__device__ __forceinline__ void load_b_per_ntile_x4(
    uint32_t (&frag)[4], char* smem_b,
    const SwizzleContext<swizzle_bytes>& ctx, int lane, int ks_pair_base, int ldm_k
) {
    int col = (lane >> 3) * 16 + ks_pair_base * ldm_k;
    void* addr = ctx.addr(smem_b, col);
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], addr);
}

// Load single B N-tile via ldmatrix.x2 — no MOV overhead for fragment rearrangement
// ctx must be initialized with: row = (lane&7) + n_tile*8, stride = BLOCK_K
template <int swizzle_bytes>
__device__ __forceinline__ void load_b_fragment_x2(
    uint32_t (&frag)[2], char* smem_b,
    const SwizzleContext<swizzle_bytes>& ctx, int lane, int k_step, int mma_k
) {
    int col = ((lane >> 3) & 1) * 16 + k_step * mma_k;
    void* addr = ctx.addr(smem_b, col);
    ldmatrix_x2(frag[0], frag[1], addr);
}

// Load B fragment from .b4x16_p64 padded SMEM via ldmatrix.m8n16.x2
// SMEM layout: [8 bytes data + 8 bytes pad] per 16 FP4 elements → 16 bytes per group
// For MMA K=32: 2 groups per step → col offset = (ks*2 + mat)*16 where mat=(lane/8)%2
// Address computation identical to load_b_fragment_x2 for FP8 (same byte offsets)
// Threads 16-31 provide dummy addr (hardware ignores for .m8n16.x2)
template <int swizzle_bytes>
__device__ __forceinline__ void load_b_fragment_b4x16_p64(
    uint32_t (&frag)[2], char* smem_b,
    const SwizzleContext<swizzle_bytes>& ctx, int lane, int k_step, int mma_k
) {
    int col = ((lane >> 3) & 1) * 16 + k_step * mma_k;
    void* addr = ctx.addr(smem_b, col);
    ldmatrix_m8n16_x2_b4x16_p64(frag[0], frag[1], addr);
}

// Fragment loaders: legacy (full address computation each call)

template <int swizzle_bytes>
__device__ __forceinline__ void* ldmatrix_a_addr(
    char* smem_tile, int lane, int m_tile, int k_step, int row_stride, int mma_k
) {
    int row = (lane & 7) + ((lane >> 3) & 1) * 8 + m_tile * 16;
    int col = (lane >> 4) * 16 + k_step * mma_k;
    int col_swizzled = swizzle<swizzle_bytes>(row, col, row_stride);
    return smem_tile + row * row_stride + col_swizzled;
}

template <int swizzle_bytes>
__device__ __forceinline__ void* ldmatrix_b_addr(
    char* smem_tile, int lane, int n_tile, int k_step, int row_stride, int mma_k
) {
    int row = (lane & 7) + n_tile * 8;
    int col = ((lane >> 3) & 1) * 16 + k_step * mma_k;
    int col_swizzled = swizzle<swizzle_bytes>(row, col, row_stride);
    return smem_tile + row * row_stride + col_swizzled;
}

template <int swizzle_bytes = 128>
__device__ __forceinline__ void load_a_fragment(
    uint32_t (&frag)[4], char* smem_a, int lane, int m_tile, int k_step,
    int row_stride, int mma_k
) {
    void* addr = ldmatrix_a_addr<swizzle_bytes>(smem_a, lane, m_tile, k_step, row_stride, mma_k);
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], addr);
}

template <int swizzle_bytes = 128>
__device__ __forceinline__ void load_b_fragment(
    uint32_t (&frag)[2], char* smem_b, int lane, int n_tile, int k_step,
    int row_stride, int mma_k
) {
    void* addr = ldmatrix_b_addr<swizzle_bytes>(smem_b, lane, n_tile, k_step, row_stride, mma_k);
    ldmatrix_x2(frag[0], frag[1], addr);
}

template <int swizzle_bytes = 128>
__device__ __forceinline__ void load_b_fragment_x4(
    uint32_t (&frag)[4], char* smem_b, int lane, int n_tile_pair, int k_step,
    int row_stride, int mma_k
) {
    int row = (lane & 7) + ((lane >> 3) & 1) * 8 + n_tile_pair * 16;
    int col = (lane >> 4) * 16 + k_step * mma_k;
    int col_swizzled = swizzle<swizzle_bytes>(row, col, row_stride);
    void* addr = smem_b + row * row_stride + col_swizzled;
    ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], addr);
}

// Scale factor loading

__device__ __forceinline__ uint32_t load_sf(const char* smem_sf, int idx) {
    return *reinterpret_cast<const uint32_t*>(smem_sf + idx * sizeof(int32_t));
}

} // namespace deep_gemm::sm120

#endif
