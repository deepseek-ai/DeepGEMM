#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::layout::sparse_mqa_logits {

inline constexpr uint32_t kNumSparseSlotBits = 16;
inline constexpr uint32_t kInvalidSparseSlot = (1u << kNumSparseSlotBits) - 1;
inline constexpr uint32_t kNumHeads = 32;
inline constexpr uint32_t kHeadDim = 128;
inline constexpr uint32_t kNumUTCCPAlignedElems = 128;
inline constexpr uint32_t kNumKVTokensPerTMA = 128;
inline constexpr uint32_t kBlockQ = 2;

CUTLASS_HOST_DEVICE constexpr uint32_t get_num_threads(const uint32_t num_math_warpgroups) {
    return (num_math_warpgroups + 1 + num_math_warpgroups / 4) * 128;
}

struct alignas(16) MetadataHeader {
    uint32_t num_kv_splits;
    uint32_t num_waves;
    uint32_t use_unaligned_ks;
};

struct alignas(16) KVSplitHeader {
    static constexpr uint32_t kContiguousFlag = 0x80000000u;

    uint32_t q_token_base;
    uint32_t packed_num_kv_blocks;
    uint32_t q0_slot_base;
    uint32_t q1_slot_base;

    KVSplitHeader() = default;
    CUTLASS_HOST_DEVICE KVSplitHeader(const uint32_t q_token_base, const uint32_t num_kv_blocks,
                                     const bool is_contiguous, const uint32_t q0_slot_base,
                                     const uint32_t q1_slot_base):
            q_token_base(q_token_base), packed_num_kv_blocks(num_kv_blocks | (is_contiguous ? kContiguousFlag : 0)),
            q0_slot_base(q0_slot_base), q1_slot_base(q1_slot_base) {}

    CUTLASS_HOST_DEVICE static constexpr uint32_t get_num_kv_blocks(const uint32_t packed_num_kv_blocks) {
        return packed_num_kv_blocks & ~kContiguousFlag;
    }

    CUTLASS_HOST_DEVICE static constexpr bool is_contiguous(const uint32_t packed_num_kv_blocks) {
        return (packed_num_kv_blocks & kContiguousFlag) != 0;
    }
};

struct KVBlockInfo {
    uint32_t physical_kv_block_idx;
    uint32_t packed_slot_offsets;

    KVBlockInfo() = default;
    CUTLASS_HOST_DEVICE KVBlockInfo(const uint32_t physical_kv_block_idx,
                                   const uint32_t q0_slot_offset, const uint32_t q1_slot_offset):
            physical_kv_block_idx(physical_kv_block_idx),
            packed_slot_offsets(q0_slot_offset | (q1_slot_offset << kNumSparseSlotBits)) {}
};

struct alignas(16) ScheduleEntry {
    uint32_t kv_split_begin;
    uint32_t kv_split_end;
    uint32_t q_token_base;
    uint32_t num_q_tokens;

    ScheduleEntry() = default;
    CUTLASS_HOST_DEVICE ScheduleEntry(const uint32_t kv_split_begin, const uint32_t kv_split_end,
                                      const uint32_t q_token_base, const uint32_t num_q_tokens):
            kv_split_begin(kv_split_begin), kv_split_end(kv_split_end), q_token_base(q_token_base), num_q_tokens(num_q_tokens) {}
};

template <uint32_t kNumKVBlocksPerSplit>
struct KVSplit {
    KVSplitHeader header;
    KVBlockInfo kv_block_infos[kNumKVBlocksPerSplit];
};

struct QBlock {
    uint32_t q_token_base;
    uint32_t num_q_tokens;
    uint32_t num_kv_splits;

    QBlock() = default;
    CUTLASS_HOST_DEVICE QBlock(const uint32_t q_token_base, const uint32_t num_q_tokens,
                              const uint32_t num_kv_splits)
        : q_token_base(q_token_base), num_q_tokens(num_q_tokens), num_kv_splits(num_kv_splits) {}
};

template <uint32_t BLOCK_Q, uint32_t SPARSE_BLOCK_KV, uint32_t SPLIT_KV, uint32_t kNumQStages,
          uint32_t kNumKVStages, uint32_t kNumTmemStages, typename qk_dtype_t>
struct SharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t kPackFactor = get_smem_pack_factor<qk_dtype_t>();
    static constexpr uint32_t kNumKVBlocksPerSplit = SPLIT_KV / SPARSE_BLOCK_KV;
    static constexpr uint32_t kNumSFQ = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kSwizzleAlignment = 8 * kHeadDim / kPackFactor;

    DG_STATIC_ASSERT(BLOCK_Q == 2, "Sparse metadata packs exactly two Q slots");
    DG_STATIC_ASSERT(SPARSE_BLOCK_KV == 8 or SPARSE_BLOCK_KV == 16, "Invalid sparse KV block size");
    DG_STATIC_ASSERT(SPLIT_KV % SPARSE_BLOCK_KV == 0 and SPLIT_KV % 128 == 0, "Invalid sparse KV split size");
    DG_STATIC_ASSERT(kNumKVBlocksPerSplit <= kInvalidSparseSlot,
                     "Sparse split-local slots must not use the invalid-slot value");

    alignas(kSwizzleAlignment) qk_dtype_t q[kNumQStages][BLOCK_Q * kNumHeads][kHeadDim / kPackFactor];
    alignas(kSwizzleAlignment) qk_dtype_t kv[kNumKVStages][SPLIT_KV][kHeadDim / kPackFactor];
    alignas(128) uint32_t sf_q[kNumQStages][kNumSFQ];
    alignas(128) uint32_t sf_kv[kNumKVStages][SPLIT_KV];
    alignas(128) nv_bfloat16 weights[kNumQStages][BLOCK_Q * kNumHeads];
    alignas(16) KVBlockInfo kv_block_infos[kNumKVStages][kNumKVBlocksPerSplit];
    alignas(16) QBlock q_blocks[kNumQStages];
    alignas(16) KVSplitHeader kv_split_headers[kNumKVStages];

    Barrier full_q_barriers[kNumQStages];
    Barrier full_sf_q_barriers[kNumQStages];
    Barrier empty_q_barriers[kNumQStages];
    Barrier full_metadata_barriers[kNumKVStages];
    Barrier full_sf_copy_barriers[kNumKVStages];
    Barrier full_kv_barriers[kNumKVStages];
    Barrier empty_kv_barriers[kNumKVStages];
    Barrier full_tmem_barriers[kNumTmemStages];
    Barrier empty_tmem_barriers[kNumTmemStages];
    uint32_t tmem_ptr_in_smem;
};

// Metadata kernel workspace
struct alignas(128) WorkspaceState {
    // Keep concurrently updated counters on separate L2 lines
    uint32_t num_kv_splits;
    alignas(128) uint32_t next_q_offset;
    alignas(128) uint32_t num_finished_ctas;
};

struct alignas(8) QBlockInfo {
    uint32_t kv_split_base;
    uint32_t num_kv_splits;

    CUTLASS_HOST_DEVICE QBlockInfo(const uint32_t kv_split_base, const uint32_t num_kv_splits):
            kv_split_base(kv_split_base), num_kv_splits(num_kv_splits) {}
};

} // namespace deep_gemm::layout::sparse_mqa_logits
