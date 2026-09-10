#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/packing.cuh>

namespace deep_gemm::layout {

template <uint32_t kNumStages, uint32_t kNumEpilogueStages, uint32_t kNumTMAStoreStages,
          uint32_t LOAD_BLOCK_M, uint32_t LOAD_BLOCK_N, uint32_t BLOCK_K,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          typename cd_dtype_t>
struct SM100BF16GemmSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using cd_dtype = cd_dtype_t;
    using cd_stage_t = cd_dtype_t[STORE_BLOCK_M * STORE_BLOCK_N];
    using a_stage_t = cutlass::bfloat16_t[LOAD_BLOCK_M * BLOCK_K];
    using b_stage_t = cutlass::bfloat16_t[LOAD_BLOCK_N * BLOCK_K];
    DG_STATIC_ASSERT(kNumTMAStoreStages * sizeof(cd_stage_t) % 1024 == 0 and
                     sizeof(a_stage_t) % 1024 == 0 and sizeof(b_stage_t) % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");
    alignas(1024) cd_stage_t cd[kNumTMAStoreStages];
    a_stage_t a[kNumStages];
    b_stage_t b[kNumStages];
    Barrier full_barriers[kNumStages];
    Barrier empty_barriers[kNumStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    // Keep the BF16 barrier layout aligned with the block-scaled GEMM layout.
    Barrier reserved_barriers[kNumStages];
    Barrier tensor_core_full_barrier;
    uint32_t tmem_ptr;
};

template <uint32_t kNumStages, uint32_t kNumEpilogueStages, uint32_t kNumTMAStoreStages,
          uint32_t LOAD_BLOCK_M, uint32_t LOAD_BLOCK_N, uint32_t BLOCK_K,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N, uint32_t SF_BLOCK_K,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t>
struct SM100FP8FP4GemmSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using cd_dtype = cd_dtype_t;
    using cd_stage_t = cd_dtype_t[STORE_BLOCK_M * STORE_BLOCK_N];
    using a_stage_t = a_dtype_t[LOAD_BLOCK_M * BLOCK_K / get_smem_pack_factor<a_dtype_t>()];
    using b_stage_t = b_dtype_t[LOAD_BLOCK_N * BLOCK_K / get_smem_pack_factor<b_dtype_t>()];
    using sfa_stage_t = uint32_t[SF_BLOCK_M * SF_BLOCK_K];
    using sfb_stage_t = uint32_t[SF_BLOCK_N * SF_BLOCK_K];
    DG_STATIC_ASSERT(kNumTMAStoreStages * sizeof(cd_stage_t) % 1024 == 0 and
                     sizeof(a_stage_t) % 1024 == 0 and sizeof(b_stage_t) % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");
    alignas(1024) cd_stage_t cd[kNumTMAStoreStages];
    a_stage_t a[kNumStages];
    b_stage_t b[kNumStages];
    sfa_stage_t sfa[kNumStages];
    sfb_stage_t sfb[kNumStages];
    Barrier full_barriers[kNumStages];
    Barrier sf_full_barriers[kNumStages];
    Barrier empty_barriers[kNumStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    Barrier tmem_overlap_barriers[kNumEpilogueStages];
    uint32_t tmem_ptr;
};

} // namespace deep_gemm::layout
