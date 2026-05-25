#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include <cute/swizzle.hpp>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)) || defined(__CLION_IDE__)

namespace deep_gemm::sm120 {

// cp.async.cg based B-tile loader for small N (swap-AB path).
// Replaces TMA for the B tile when N_tensor < TMA minimum dimension (8).
// All threads in the producer warp cooperate to load the tile.
//
// Key difference from TMA: cp.async.cg does NOT auto-apply swizzle.
// We compute swizzled SMEM destination addresses per 16B chunk.
//
// Usage in producer loop:
//   cpasync_load_b_tile<BK, kSwizzleBMode>(smem_b, gmem_b, n_rows, bk, lane_idx);
//   asm volatile("cp.async.commit_group;\n" ::: "memory");
//   asm volatile("cp.async.wait_group 0;\n" ::: "memory");
//   // B is now in SMEM with correct swizzle layout

template <int BK, int kSwizzleBMode>
CUTLASS_DEVICE void cpasync_load_b_tile(
    char* smem_b,                     // SMEM destination (swizzled layout)
    const void* gmem_b,               // GMEM source for this K-block
    int n_rows,                       // actual N rows to load
    int bk,                           // block K bytes to load per row
    int gmem_row_stride,              // GMEM stride between rows (= shape_k for K-major FP8)
    int lane_idx                      // lane within warp (0..31)
) {
    const auto* src = reinterpret_cast<const uint8_t*>(gmem_b);
    const int smem_row_stride = bk;   // SMEM row stride = BK (contiguous within tile)
    const int num_col_slots = bk / 16;  // number of 16B slots per row

    // Total work items: n_rows × num_col_slots
    const int total_items = n_rows * num_col_slots;

    // Each thread handles items in stride-32 pattern
    for (int item = lane_idx; item < total_items; item += 32) {
        int row = item / num_col_slots;
        int col_slot = item % num_col_slots;
        int col_byte = col_slot * 16;

        // GMEM source: row stride may differ from BK (full K dimension vs K-block)
        const uint8_t* gmem_addr = src + row * gmem_row_stride + col_byte;

        // SMEM destination: contiguous BK bytes per row, with swizzle
        int smem_row_base = row * smem_row_stride;

        int swizzled_col;
        if constexpr (kSwizzleBMode > 0) {
            // B128 swizzle = Swizzle<3, 4, 3>, B64 = Swizzle<2, 4, 3>
            using SwizzleOp = cute::Swizzle<__builtin_ctz(kSwizzleBMode) - 4, 4, 3>;
            int flat = smem_row_base + col_byte;
            int swizzled_flat = SwizzleOp::apply(flat);
            swizzled_col = swizzled_flat - smem_row_base;
        } else {
            swizzled_col = col_byte;
        }

        uint32_t smem_dest = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_b + smem_row_base + swizzled_col));

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(smem_dest), "l"(gmem_addr)
            : "memory"
        );
    }
}

// Convenience: commit and wait for all outstanding cp.async groups
CUTLASS_DEVICE void cpasync_commit_and_wait() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// Commit group (non-blocking)
CUTLASS_DEVICE void cpasync_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait for N outstanding groups (0 = wait all)
template <int N>
CUTLASS_DEVICE void cpasync_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Zero-fill a region of SMEM (for padding rows when N < BN)
CUTLASS_DEVICE void smem_zero_fill(char* smem, int bytes, int lane_idx) {
    auto* ptr = reinterpret_cast<uint32_t*>(smem);
    int words = bytes / 4;
    for (int i = lane_idx; i < words; i += 32)
        ptr[i] = 0;
}

} // namespace deep_gemm::sm120

#endif // __CUDA_ARCH__ >= 1200
