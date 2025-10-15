#pragma once
#include <cutlass/arch/barrier.h>
#include <cute/arch/cluster_sm90.hpp>

namespace deep_gemm {

template <uint32_t kSplitKSlices>
static constexpr uint32_t get_reduce_iterations() {
    uint32_t num_iterations = 0;
    for (int i = 1; i < kSplitKSlices; i <<= 1) {
        num_iterations++;
    }
    return num_iterations;
}

// Reduce `accums` on registers in place across `kSplitKSlices` CTAs within a cluster,
// using distributed shared memory and memory barriers for communication.
// The reduction is done by a shuffle-down manner.
// Before reduction, each thread should hold `kNumAccum` float values in `accums`.
// After reduction, only the leader CTA holds the final results and will write them back to shared memory and then global memory.
template <uint32_t kSplitKSlices, uint32_t kNumMathThreads, uint32_t kNumAccum, typename Barrier>
__device__  __forceinline__ void 
split_k_reduce(float* smem_d, float* accums, Barrier* split_k_reduce_empty_barrier, Barrier* split_k_reduce_full_barrier, uint32_t current_iter){
    constexpr uint32_t reduce_iterations = get_reduce_iterations<kSplitKSlices>();
    // kSplitKSlices CTAs with same cute::block_id_in_cluster().x and different cute::block_id_in_cluster().y within a cluster make a reduction group.
    DG_TRAP_ONLY_DEVICE_ASSERT(cute::cluster_shape().y == kSplitKSlices);
    auto k_partition_id = cute::block_id_in_cluster().y;
    auto smem_d_u32 =  cute::cast_smem_ptr_to_uint(smem_d);
    uint32_t reduce_full_barrier_u32 = cute::cast_smem_ptr_to_uint(split_k_reduce_full_barrier);
    #pragma unroll
    for (uint32_t i = 1, mask = 1; i <= reduce_iterations; i ++, mask <<= 1) {
        int peer_rank = (k_partition_id ^ mask) * cute::cluster_shape().x + cute::block_id_in_cluster().x;

        // notify all CTAs whthin the reduction group that I'm ready for current reduce iteration
        if (threadIdx.x < kSplitKSlices) {
            split_k_reduce_empty_barrier->arrive(threadIdx.x * cute::cluster_shape().x + cute::block_id_in_cluster().x);
        }
        // wait all CTAs whthin the reduction group to be ready
        split_k_reduce_empty_barrier->wait((current_iter * reduce_iterations + i + 1) & 1);
        if ((k_partition_id & mask) != 0) { // send
            #pragma unroll
            for (uint32_t j = 0, smem_offset = threadIdx.x; j < kNumAccum; ++j, smem_offset += kNumMathThreads) {
                uint32_t smem_d_shifted_u32 = smem_d_u32 + smem_offset * sizeof(float);
                // store to peer smem
                cute::store_shared_remote(*reinterpret_cast<uint32_t*>(&accums[j]), smem_d_shifted_u32, reduce_full_barrier_u32, peer_rank);
            }
            if (threadIdx.x == 0) {
                split_k_reduce_full_barrier->arrive_and_expect_tx(kNumAccum * kNumMathThreads * sizeof(float), peer_rank);
                // arrive on local full_barrier, to make barrier phases aligned
                split_k_reduce_full_barrier->arrive();
            }
            // to make barrier phases aligned
            split_k_reduce_full_barrier->wait((current_iter * reduce_iterations + i + 1) & 1);
        } else { // receive
            split_k_reduce_full_barrier->wait((current_iter * reduce_iterations + i + 1) & 1);
            #pragma unroll
            for (uint32_t j = 0, smem_offset = threadIdx.x; j < kNumAccum; ++j, smem_offset += kNumMathThreads) {
                accums[j] += smem_d[smem_offset];
            }
        }
    }
}


}; // namespace deep_gemm