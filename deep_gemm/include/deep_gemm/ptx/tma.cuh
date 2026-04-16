#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_desc.hpp>

namespace deep_gemm::ptx {

// Tensor-map instructions
CUTLASS_DEVICE void tensor_map_release_gpu() {
    asm volatile ("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}

CUTLASS_DEVICE void tensor_map_acquire_gpu(const cute::TmaDescriptor* gmem_desc_ptr) {
    auto gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
    asm volatile ("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(gmem_int_desc) : "memory");
}

CUTLASS_DEVICE void tensor_map_replace_global_addr_in_smem(cute::TmaDescriptor* smem_desc, const void* new_addr) {
    auto smem_int_desc = static_cast<uint32_t>(__cvta_generic_to_shared(smem_desc));
    const auto new_int64_addr = reinterpret_cast<uint64_t>(new_addr);
    asm volatile ("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "r"(smem_int_desc), "l"(new_int64_addr));
}

CUTLASS_DEVICE void tensor_map_replace_global_inner_dim_stride_in_smem(cute::TmaDescriptor* smem_desc, const uint32_t& new_dim, const uint64_t& new_stride) {
    auto smem_int_desc = __cvta_generic_to_shared(smem_desc);
    asm volatile ("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" :: "l"(smem_int_desc), "r"(new_dim));
#if ((__CUDACC_VER_MAJOR__ > 12) or ((__CUDACC_VER_MAJOR__ == 12) and (__CUDACC_VER_MINOR__ >= 3)))
    asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "l"(smem_int_desc), "l"(new_stride));
#else
    DG_STATIC_ASSERT(false, "Invalid CUDA version");
#endif
}

/// TMA instructions
CUTLASS_DEVICE void mbarrier_arrive(
    cutlass::arch::ClusterTransactionBarrier* ptr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

CUTLASS_DEVICE void mbarrier_arrive_and_set_tx(
    cutlass::arch::ClusterTransactionBarrier* ptr, const uint32_t& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

CUTLASS_DEVICE void mbarrier_wait_and_flip_phase(
    cutlass::arch::ClusterTransactionBarrier* ptr, uint32_t& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase), "r"(0x989680));
    phase ^= 1;
}

CUTLASS_DEVICE void tma_load_1d(
    const void* dst_ptr, const void* src_ptr,
    cutlass::arch::ClusterTransactionBarrier* mbarrier_ptr,
    const uint32_t& num_bytes,
    const cute::TMA::CacheHintSm90& hint = cute::TMA::CacheHintSm90::EVICT_FIRST) {
    // NOTES: normally, the loaded part will be evicted soon
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),
        "l"(src_ptr),
        "r"(num_bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr))),
        "l"(hint)
        : "memory");
}

CUTLASS_DEVICE void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const uint32_t& num_bytes,
    const cute::TMA::CacheHintSm90& hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    // NOTES: normally, the stored part will be used soon
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes),
                 "l"(hint)
                 : "memory");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    // NOTES: this function does not have `.read`
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");
}

CUTLASS_DEVICE
void tma_gather4(const void* desc_ptr, cutlass::arch::ClusterTransactionBarrier& mbarrier,
                 void* smem_ptr, const uint32_t& col_idx, const int4& row_idxs, const uint64_t& cache_hint) {
    const auto smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    const auto mbarrier_addr = cute::cast_smem_ptr_to_uint(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx),
          "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w),
          "r"(mbarrier_addr), "l"(cache_hint)
        : "memory"
    );
}

} // namespace deep_gemm::ptx
