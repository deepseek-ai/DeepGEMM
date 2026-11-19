#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/sm120_utils.cuh>
#include <deep_gemm/layout/sparse_mqa_logits.cuh>
#include <deep_gemm/mma/sm120.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sm120_sparse_mqa_detail {

using namespace layout::sparse_mqa_logits;

template <bool kIsFP4, uint32_t SPARSE_BLOCK_KV, bool kPipeline>
struct SharedStorage {
    static constexpr uint32_t kRowBytes = kIsFP4 ? 64 : 128;
    static constexpr uint32_t kNumKVStages = kPipeline ? 2 : 1;

    alignas(1024) char q[2 * kNumHeads * kRowBytes];
    alignas(1024) char kv[kNumKVStages][64 * kRowBytes];
    uint32_t sf_q[2 * kNumHeads];
    uint32_t sf_kv[kNumKVStages][64];
    nv_bfloat16 weights[2 * kNumHeads];
    KVBlockInfo infos[kNumKVStages][64 / SPARSE_BLOCK_KV];
};

template <bool kIsFP4, uint32_t SPARSE_BLOCK_KV, bool kPipeline>
struct WarpSpecializedSharedStorage {
    using Barrier = cutlass::arch::ClusterBarrier;

    SharedStorage<kIsFP4, SPARSE_BLOCK_KV, kPipeline> tiles;
    Barrier full_q;
    Barrier empty_q;
    Barrier full_kv[2];
    Barrier empty_kv[2];
};

#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200) and (__CUDA_ARCH__ < 1300)) or defined(__CLION_IDE__)

// CUTLASS's barrier assembly lacks compiler memory clobbers for ordinary shared accesses.
CUTLASS_DEVICE void pipeline_wait(const cutlass::arch::ClusterBarrier& barrier, const uint32_t phase) {
    asm volatile("" ::: "memory");
    barrier.wait(phase);
    asm volatile("" ::: "memory");
}

CUTLASS_DEVICE void pipeline_arrive(const cutlass::arch::ClusterBarrier& barrier) {
    asm volatile("" ::: "memory");
    barrier.arrive();
    asm volatile("" ::: "memory");
}

CUTLASS_DEVICE uint4 load_vector(const uint8_t* ptr) {
    if ((reinterpret_cast<uint64_t>(ptr) & 15u) == 0)
        return *reinterpret_cast<const uint4*>(ptr);
    uint4 value;
    auto bytes = reinterpret_cast<uint8_t*>(&value);
    #pragma unroll
    for (uint32_t i = 0; i < 16; ++ i)
        bytes[i] = ptr[i];
    return value;
}

CUTLASS_DEVICE uint32_t load_scale(const uint8_t* ptr) {
    uint32_t value = 0;
    #pragma unroll
    for (uint32_t i = 0; i < sizeof(uint32_t); ++ i)
        value |= static_cast<uint32_t>(ptr[i]) << (8 * i);
    return value;
}

struct KVTokenRef {
    const uint8_t* values;
    const uint8_t* scale;
};

template <uint32_t kRowBytes>
struct ContiguousKVAccessor {
    const uint8_t* kv;
    const uint8_t* sf_kv;
    uint64_t num_tokens;

    CUTLASS_DEVICE KVTokenRef resolve(const uint32_t physical_kv_block_idx, const uint32_t within) const {
        const uint64_t token = static_cast<uint64_t>(physical_kv_block_idx) + within;
        if (token >= num_tokens)
            return {nullptr, nullptr};
        return {kv + token * kRowBytes, sf_kv + token * sizeof(uint32_t)};
    }
};

template <uint32_t kRowBytes, uint32_t SPARSE_BLOCK_KV, uint32_t PAGE_KV>
struct PagedKVAccessor {
    const uint8_t* kv;
    uint64_t num_pages;
    uint64_t page_stride;

    CUTLASS_DEVICE KVTokenRef resolve(const uint32_t physical_kv_block_idx, const uint32_t within) const {
        constexpr uint32_t kBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
        const uint64_t page_idx = physical_kv_block_idx / kBlocksPerPage;
        if (page_idx >= num_pages)
            return {nullptr, nullptr};
        const uint32_t token = (physical_kv_block_idx % kBlocksPerPage) * SPARSE_BLOCK_KV + within;
        const auto page = kv + page_idx * page_stride;
        return {page + token * kRowBytes,
                page + static_cast<uint64_t>(PAGE_KV) * kRowBytes + token * sizeof(uint32_t)};
    }
};

template <bool kIsFP4, uint32_t SPARSE_BLOCK_KV, bool kPipeline, uint32_t kNumLoadThreads>
CUTLASS_DEVICE void load_q(SharedStorage<kIsFP4, SPARSE_BLOCK_KV, kPipeline>& smem,
                          const uint8_t* q, const uint8_t* sf_q, const nv_bfloat16* weights,
                          const uint64_t weights_stride, const ScheduleEntry& entry) {
    constexpr uint32_t kRowBytes = kIsFP4 ? 64 : 128;
    for (uint32_t item = threadIdx.x; item < 2 * kNumHeads * kRowBytes / 16; item += kNumLoadThreads) {
        const uint32_t qi = item / (kNumHeads * kRowBytes / 16);
        uint4 value = {0, 0, 0, 0};
        if (qi < entry.num_q_tokens)
            value = load_vector(q + static_cast<uint64_t>(entry.q_token_base) * kNumHeads * kRowBytes + item * 16);
        *reinterpret_cast<uint4*>(smem.q + sm120::CuTeSwizzle<kRowBytes>::apply(item * 16)) = value;
    }
    for (uint32_t item = threadIdx.x; item < 2 * kNumHeads; item += kNumLoadThreads) {
        const uint32_t qi = item / kNumHeads, h = item % kNumHeads;
        const uint64_t row = static_cast<uint64_t>(entry.q_token_base) + qi;
        smem.sf_q[item] = qi < entry.num_q_tokens ? load_scale(sf_q + (row * kNumHeads + h) * sizeof(uint32_t)) : 0;
        smem.weights[item] = qi < entry.num_q_tokens ? weights[row * weights_stride + h] : __float2bfloat16_rn(0.0f);
    }
}

template <bool kCacheQ>
struct QOperands {};

template <>
struct QOperands<true> {
    uint32_t scales[2][4];
    nv_bfloat162 weights[2][4];
    nv_bfloat162 partner_weights[2][4];
};

template <bool kIsFP4, uint32_t SPARSE_BLOCK_KV, bool kPipeline>
CUTLASS_DEVICE void load_q_operands(QOperands<true>& operands,
                                   const SharedStorage<kIsFP4, SPARSE_BLOCK_KV, kPipeline>& smem,
                                   const uint32_t math_tid) {
    const uint32_t lane = math_tid % 32, g = lane / 4, t = lane % 4;
    #pragma unroll
    for (uint32_t qi = 0; qi < 2; ++ qi) {
        #pragma unroll
        for (uint32_t nt = 0; nt < 4; ++ nt) {
            operands.scales[qi][nt] = smem.sf_q[qi * kNumHeads + nt * 8 + g];
            const uint32_t h = qi * kNumHeads + nt * 8 + t * 2;
            const auto weight = __halves2bfloat162(smem.weights[h], smem.weights[h + 1]);
            operands.weights[qi][nt] = weight;
            operands.partner_weights[qi][nt] = ptx::exchange(weight, g * 4 + (t ^ 2));
        }
    }
}

template <bool kCacheQ = false, bool kIsFP4, uint32_t SPARSE_BLOCK_KV, bool kPipeline>
CUTLASS_DEVICE void compute_chunk(SharedStorage<kIsFP4, SPARSE_BLOCK_KV, kPipeline>& smem,
                                 const uint32_t stage, const uint32_t math_tid,
                                 const ScheduleEntry& entry, const KVSplitHeader& split_header,
                                 nv_bfloat16* logits, const uint64_t logits_stride, const uint32_t output_cols,
                                 const QOperands<kCacheQ>& operands = {}) {
    constexpr uint32_t kRowBytes = kIsFP4 ? 64 : 128;
    constexpr uint32_t kKSteps = kIsFP4 ? 2 : 4;
    const uint32_t lane = math_tid % 32, warp = math_tid / 32;
    const uint32_t g = lane / 4, t = lane % 4;
    sm120::SwizzleContext<kRowBytes> a_ctx;
    a_ctx.init(warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, kRowBytes);
    const uint32_t sf_a = smem.sf_kv[stage][warp * 16 + g + (t & 1) * 8];
    uint32_t a[kKSteps][4];
    #pragma unroll
    for (uint32_t ks = 0; ks < kKSteps; ++ ks)
        sm120::load_a_fragment(a[ks], smem.kv[stage], a_ctx, lane, ks, 32);
    #pragma unroll
    for (uint32_t qi = 0; qi < 2; ++ qi) {
        if (qi >= entry.num_q_tokens)
            continue;
        nv_bfloat162 partial_sum[2] = {__floats2bfloat162_rn(0.0f, 0.0f), __floats2bfloat162_rn(0.0f, 0.0f)};
        #pragma unroll
        for (uint32_t nt = 0; nt < 4; ++ nt) {
            sm120::SwizzleContext<kRowBytes> b_ctx;
            b_ctx.init(qi * kNumHeads + nt * 8 + (lane & 7), kRowBytes);
            const uint32_t sf_b = [&]() {
                if constexpr (kCacheQ)
                    return operands.scales[qi][nt];
                else
                    return smem.sf_q[qi * kNumHeads + nt * 8 + g];
            }();
            float d[4] = {0, 0, 0, 0};
            #pragma unroll
            for (uint32_t ks = 0; ks < kKSteps; ++ ks) {
                uint32_t b[2];
                sm120::load_b_fragment_x2(b, smem.q, b_ctx, lane, ks, 32);
                if constexpr (kIsFP4)
                    mma::sm120::fp4_mma_block_scaled(d, a[ks], b,
                        mma::sm120::extract_sf_pair(sf_a, 2 * ks), mma::sm120::extract_sf_pair(sf_b, 2 * ks));
                else
                    mma::sm120::fp8_mma_block_scaled(d, a[ks], b,
                        mma::sm120::extract_sf_byte(sf_a, ks), mma::sm120::extract_sf_byte(sf_b, ks));
            }
            nv_bfloat162 weight, partner_weight;
            if constexpr (kCacheQ) {
                weight = operands.weights[qi][nt];
                partner_weight = operands.partner_weights[qi][nt];
            } else {
                const uint32_t h = qi * kNumHeads + nt * 8 + t * 2;
                weight = __halves2bfloat162(smem.weights[h], smem.weights[h + 1]);
                partner_weight = ptx::exchange(weight, g * 4 + (t ^ 2));
            }
            #pragma unroll
            for (uint32_t mi = 0; mi < 2; ++ mi) {
                const auto score = ptx::cvt_relu_bf16x2_f32(make_float2(d[2 * mi], d[2 * mi + 1]));
                const auto partner_score = ptx::exchange(score, g * 4 + (t ^ 2));
                if (t < 2) {
                    partial_sum[mi] = __hfma2(score, weight, partial_sum[mi]);
                    partial_sum[mi] = __hfma2(partner_score, partner_weight, partial_sum[mi]);
                }
            }
        }
        #pragma unroll
        for (uint32_t mi = 0; mi < 2; ++ mi) {
            const auto sum_1 = ptx::exchange(partial_sum[mi], g * 4 + 1);
            if (t == 0) {
                const uint32_t row = warp * 16 + g + mi * 8;
                const auto info = smem.infos[stage][row / SPARSE_BLOCK_KV];
                const uint32_t slot = (info.packed_slot_offsets >> (qi * kNumSparseSlotBits)) & kInvalidSparseSlot;
                if (slot != kInvalidSparseSlot) {
                    const uint32_t slot_base = qi == 0 ? split_header.q0_slot_base : split_header.q1_slot_base;
                    const uint64_t col = (static_cast<uint64_t>(slot_base) + slot) * SPARSE_BLOCK_KV + row % SPARSE_BLOCK_KV;
                    if (col < output_cols) {
                        const auto sum = __hadd2_rn(partial_sum[mi], sum_1);
                        logits[(static_cast<uint64_t>(entry.q_token_base) + qi) * logits_stride + col] = __hadd_rn(sum.x, sum.y);
                    }
                }
            }
        }
    }
}

template <uint32_t kBytes>
CUTLASS_DEVICE void copy_async(void* dst, const uint8_t* src, const bool valid) {
    const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const uint32_t src_bytes = valid ? kBytes : 0;
    if constexpr (kBytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
            "r"(addr), "l"(src), "r"(src_bytes) : "memory");
    else {
        DG_STATIC_ASSERT(kBytes == 4, "Invalid sparse async copy size");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::
            "r"(addr), "l"(src), "r"(src_bytes) : "memory");
    }
}

template <uint32_t kNumLoadThreads = 128, bool kIsFP4, uint32_t SPARSE_BLOCK_KV, typename KVAccessor>
CUTLASS_DEVICE void gather_async(SharedStorage<kIsFP4, SPARSE_BLOCK_KV, true>& smem,
                                 const uint32_t stage, const KVBlockInfo* infos,
                                 const uint32_t block_base, const uint32_t num_blocks,
                                 const uint32_t num_q_tokens, const KVAccessor& accessor) {
    constexpr uint32_t kRowBytes = kIsFP4 ? 64 : 128;
    for (uint32_t item = threadIdx.x; item < 64 * kRowBytes / 16; item += kNumLoadThreads) {
        const uint32_t row = item / (kRowBytes / 16), chunk = item % (kRowBytes / 16);
        const uint32_t block = row / SPARSE_BLOCK_KV;
        const auto info = block_base + block < num_blocks ? infos[block_base + block] :
            KVBlockInfo(0, kInvalidSparseSlot, kInvalidSparseSlot);
        if (row % SPARSE_BLOCK_KV == 0 and chunk == 0)
            smem.infos[stage][block] = info;
        KVTokenRef ref = {nullptr, nullptr};
        const bool q0_valid = (info.packed_slot_offsets & kInvalidSparseSlot) != kInvalidSparseSlot;
        const bool q1_valid = num_q_tokens > 1 and (info.packed_slot_offsets >> kNumSparseSlotBits) != kInvalidSparseSlot;
        if (block_base + block < num_blocks and (q0_valid or q1_valid))
            ref = accessor.resolve(info.physical_kv_block_idx, row % SPARSE_BLOCK_KV);
        const bool valid = ref.values != nullptr;
        copy_async<16>(smem.kv[stage] + sm120::CuTeSwizzle<kRowBytes>::apply(item * 16),
            valid ? ref.values + chunk * 16 : accessor.kv, valid);
        if (chunk == 0)
            copy_async<4>(&smem.sf_kv[stage][row], valid ? ref.scale : accessor.sf_kv, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <bool kIsFP4, uint32_t SPARSE_BLOCK_KV, uint32_t kNumSMs,
          bool kUseUnalignedKs, uint32_t kWorkPartitions, bool kPipeline, typename KVAccessor>
CUTLASS_DEVICE void compute(const uint8_t* q, const uint8_t* sf_q,
                            const nv_bfloat16* weights, const uint8_t* metadata,
                            nv_bfloat16* logits, const uint64_t num_q_tokens,
                            const uint64_t metadata_bytes, const uint64_t weights_stride,
                            const uint64_t logits_stride, const uint32_t output_cols,
                            const KVAccessor& accessor) {
    constexpr uint32_t kRowBytes = kIsFP4 ? 64 : 128;
    constexpr uint32_t kSplitKV = kIsFP4 ? 640 : 512;
    constexpr uint32_t kBlocksPerSplit = kSplitKV / SPARSE_BLOCK_KV;
    constexpr uint32_t kBlocksPerChunk = 64 / SPARSE_BLOCK_KV;
    using Split = KVSplit<kBlocksPerSplit>;
    DG_STATIC_ASSERT(SPARSE_BLOCK_KV == 8 or SPARSE_BLOCK_KV == 16, "Invalid sparse block size");
    DG_STATIC_ASSERT(kNumHeads == 32 and kHeadDim == 128 and kBlockQ == 2, "Invalid sparse MQA layout");
    DG_STATIC_ASSERT(kWorkPartitions > 0, "Invalid work partition count");
    DG_DEVICE_ASSERT(blockDim.x == 128 and gridDim.x == kNumSMs * kWorkPartitions);

    extern __shared__ __align__(1024) uint8_t storage[];
    auto& smem = *reinterpret_cast<SharedStorage<kIsFP4, SPARSE_BLOCK_KV, kPipeline>*>(storage);
    const uint32_t tid = threadIdx.x;
    const uint32_t entry_lane = blockIdx.x / kWorkPartitions, partition = blockIdx.x % kWorkPartitions;

    if (metadata_bytes < sizeof(MetadataHeader))
        return;
    const auto header = *reinterpret_cast<const MetadataHeader*>(metadata);
    DG_DEVICE_ASSERT(header.use_unaligned_ks == kUseUnalignedKs);
    if (header.use_unaligned_ks != kUseUnalignedKs)
        return;
    const uint64_t schedule_offset = sizeof(MetadataHeader) + static_cast<uint64_t>(header.num_kv_splits) * sizeof(Split);
    if (schedule_offset > metadata_bytes)
        return;
    if (static_cast<uint64_t>(header.num_waves) * kNumSMs > (metadata_bytes - schedule_offset) / sizeof(ScheduleEntry))
        return;
    const auto splits = reinterpret_cast<const Split*>(metadata + sizeof(MetadataHeader));
    const auto entries = reinterpret_cast<const ScheduleEntry*>(metadata + schedule_offset);

    for (uint32_t wave = 0; wave < header.num_waves; ++ wave) {
        const auto entry = entries[static_cast<uint64_t>(wave) * kNumSMs + entry_lane];
        if (entry.kv_split_begin == entry.kv_split_end)
            continue;
        if (entry.kv_split_begin > entry.kv_split_end or entry.kv_split_end > header.num_kv_splits or
            entry.num_q_tokens == 0 or entry.num_q_tokens > 2 or
            static_cast<uint64_t>(entry.q_token_base) + entry.num_q_tokens > num_q_tokens)
            continue;

        bool q_loaded = false;
        uint64_t chunk_ordinal = 0;
        for (uint32_t split_idx = entry.kv_split_begin; split_idx < entry.kv_split_end; ++ split_idx) {
            const auto split_header = splits[split_idx].header;
            const uint32_t num_blocks = KVSplitHeader::get_num_kv_blocks(split_header.packed_num_kv_blocks);
            if (num_blocks > kBlocksPerSplit or split_header.q_token_base != entry.q_token_base)
                continue;
            const uint32_t chunk_count = (num_blocks + kBlocksPerChunk - 1) / kBlocksPerChunk;
            const uint32_t chunk_begin = kPipeline ? chunk_count * partition / kWorkPartitions : 0;
            const uint32_t chunk_end = kPipeline ? chunk_count * (partition + 1) / kWorkPartitions : chunk_count;
            if (chunk_begin == chunk_end)
                continue;
            for (uint32_t block_base = chunk_begin * kBlocksPerChunk; block_base < chunk_end * kBlocksPerChunk;
                 block_base += kBlocksPerChunk, ++ chunk_ordinal) {
                if constexpr (not kPipeline and kWorkPartitions > 1) {
                    if (chunk_ordinal % kWorkPartitions != partition)
                        continue;
                }
                if (not q_loaded) {
                    load_q<kIsFP4, SPARSE_BLOCK_KV, kPipeline, 128>(smem, q, sf_q, weights, weights_stride, entry);
                    q_loaded = true;
                }
                const uint32_t chunk = block_base / kBlocksPerChunk;
                const uint32_t stage = kPipeline ? chunk & 1u : 0;
                if constexpr (kPipeline) {
                    if (chunk == chunk_begin) {
                        gather_async(smem, stage, splits[split_idx].kv_block_infos,
                            block_base, num_blocks, entry.num_q_tokens, accessor);
                        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                        __syncthreads();
                    }
                    if (chunk + 1 < chunk_end)
                        gather_async(smem, stage ^ 1u, splits[split_idx].kv_block_infos,
                            block_base + kBlocksPerChunk, num_blocks, entry.num_q_tokens, accessor);
                } else {
                    if (tid < kBlocksPerChunk) {
                        smem.infos[0][tid] = block_base + tid < num_blocks ? splits[split_idx].kv_block_infos[block_base + tid] :
                            KVBlockInfo(0, kInvalidSparseSlot, kInvalidSparseSlot);
                    }
                    __syncthreads();
                    for (uint32_t item = tid; item < 64 * kRowBytes / 16; item += 128) {
                        const uint32_t row = item / (kRowBytes / 16), chunk = item % (kRowBytes / 16);
                        const uint32_t block = row / SPARSE_BLOCK_KV;
                        KVTokenRef ref = {nullptr, nullptr};
                        if (block_base + block < num_blocks)
                            ref = accessor.resolve(smem.infos[0][block].physical_kv_block_idx, row % SPARSE_BLOCK_KV);
                        const uint4 value = ref.values != nullptr ? load_vector(ref.values + chunk * 16) : uint4{0, 0, 0, 0};
                        *reinterpret_cast<uint4*>(smem.kv[0] + sm120::CuTeSwizzle<kRowBytes>::apply(item * 16)) = value;
                        if (chunk == 0)
                            smem.sf_kv[0][row] = ref.scale != nullptr ? load_scale(ref.scale) : 0;
                    }
                    __syncthreads();
                }
                compute_chunk(smem, stage, tid, entry, split_header, logits, logits_stride, output_cols);
                if constexpr (kPipeline)
                    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

template <bool kIsFP4, uint32_t SPARSE_BLOCK_KV, uint32_t kNumSMs,
          bool kUseUnalignedKs, uint32_t kWorkPartitions, bool kCacheQ, bool kEntryBalance, typename KVAccessor>
CUTLASS_DEVICE void compute_warp_specialized(const uint8_t* q, const uint8_t* sf_q,
                                             const nv_bfloat16* weights, const uint8_t* metadata,
                                             nv_bfloat16* logits, const uint64_t num_q_tokens,
                                             const uint64_t metadata_bytes, const uint64_t weights_stride,
                                             const uint64_t logits_stride, const uint32_t output_cols,
                                             const KVAccessor& accessor) {
    constexpr uint32_t kNumProducerThreads = 128, kNumMathThreads = 128;
    constexpr uint32_t kBlocksPerSplit = 640 / SPARSE_BLOCK_KV;
    constexpr uint32_t kBlocksPerChunk = 64 / SPARSE_BLOCK_KV;
    using Split = KVSplit<kBlocksPerSplit>;
    DG_STATIC_ASSERT(kIsFP4 and SPARSE_BLOCK_KV == 8 and kWorkPartitions == 4, "Invalid warp-specialized sparse shape");
    DG_DEVICE_ASSERT(blockDim.x == kNumProducerThreads + kNumMathThreads and gridDim.x == kNumSMs * kWorkPartitions);
    DG_DEVICE_ASSERT(output_cols == 2048 * SPARSE_BLOCK_KV);

    const uint32_t tid = threadIdx.x;
    const bool is_producer = tid < kNumProducerThreads;
    const uint32_t entry_lane = blockIdx.x / kWorkPartitions, partition = blockIdx.x % kWorkPartitions;
    if (metadata_bytes < sizeof(MetadataHeader))
        return;
    const auto header = *reinterpret_cast<const MetadataHeader*>(metadata);
    DG_DEVICE_ASSERT(header.use_unaligned_ks == kUseUnalignedKs);
    if (header.use_unaligned_ks != kUseUnalignedKs)
        return;
    const uint64_t schedule_offset = sizeof(MetadataHeader) + static_cast<uint64_t>(header.num_kv_splits) * sizeof(Split);
    if (schedule_offset > metadata_bytes)
        return;
    if (static_cast<uint64_t>(header.num_waves) * kNumSMs > (metadata_bytes - schedule_offset) / sizeof(ScheduleEntry))
        return;
    const auto splits = reinterpret_cast<const Split*>(metadata + sizeof(MetadataHeader));
    const auto entries = reinterpret_cast<const ScheduleEntry*>(metadata + schedule_offset);

    extern __shared__ __align__(1024) uint8_t storage[];
    auto& pipeline = *reinterpret_cast<WarpSpecializedSharedStorage<kIsFP4, SPARSE_BLOCK_KV, true>*>(storage);
    auto& smem = pipeline.tiles;
    if (tid == 0) {
        pipeline.full_q.init(kNumProducerThreads);
        pipeline.empty_q.init(kNumMathThreads);
        #pragma unroll
        for (uint32_t stage = 0; stage < 2; ++ stage) {
            pipeline.full_kv[stage].init(kNumProducerThreads);
            pipeline.empty_kv[stage].init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    const auto run_pipeline = [&]<bool kIsProducer>() {
        if constexpr (kIsProducer)
            cutlass::arch::warpgroup_reg_dealloc<40>();
        else
            cutlass::arch::warpgroup_reg_alloc<88>();

        uint64_t q_ordinal = 0, chunk_ordinal = 0;
        for (uint32_t wave = 0; wave < header.num_waves; ++ wave) {
            const auto entry = entries[static_cast<uint64_t>(wave) * kNumSMs + entry_lane];
            if (entry.kv_split_begin == entry.kv_split_end)
                continue;
            if (entry.kv_split_begin > entry.kv_split_end or entry.kv_split_end > header.num_kv_splits or
                entry.num_q_tokens == 0 or entry.num_q_tokens > 2 or
                static_cast<uint64_t>(entry.q_token_base) + entry.num_q_tokens > num_q_tokens)
                continue;

            const auto count_chunks = [&](const KVSplitHeader& split_header) -> uint32_t {
                const uint32_t num_blocks = KVSplitHeader::get_num_kv_blocks(split_header.packed_num_kv_blocks);
                if (num_blocks > kBlocksPerSplit or split_header.q_token_base != entry.q_token_base)
                    return 0;
                return (num_blocks + kBlocksPerChunk - 1) / kBlocksPerChunk;
            };
            uint64_t entry_begin = 0, entry_end = 0, chunk_prefix = 0;
            if constexpr (kEntryBalance) {
                uint64_t total_chunks = 0;
                for (uint32_t split_idx = entry.kv_split_begin; split_idx < entry.kv_split_end; ++ split_idx)
                    total_chunks += count_chunks(splits[split_idx].header);
                entry_begin = total_chunks * partition / kWorkPartitions;
                entry_end = total_chunks * (partition + 1) / kWorkPartitions;
            }
            const uint32_t q_phase = q_ordinal & 1u;
            QOperands<kCacheQ and not kIsProducer> operands;
            if constexpr (kIsProducer) {
                pipeline_wait(pipeline.empty_q, q_phase ^ 1u);
                load_q<kIsFP4, SPARSE_BLOCK_KV, true, kNumProducerThreads>(smem, q, sf_q, weights, weights_stride, entry);
                pipeline_arrive(pipeline.full_q);
            } else {
                pipeline_wait(pipeline.full_q, q_phase);
                if constexpr (kCacheQ)
                    load_q_operands(operands, smem, tid - kNumProducerThreads);
            }
            for (uint32_t split_idx = entry.kv_split_begin; split_idx < entry.kv_split_end; ++ split_idx) {
                const auto split_header = splits[split_idx].header;
                const uint32_t num_blocks = KVSplitHeader::get_num_kv_blocks(split_header.packed_num_kv_blocks);
                const uint32_t chunk_count = count_chunks(split_header);
                uint32_t chunk_begin = chunk_count * partition / kWorkPartitions;
                uint32_t chunk_end = chunk_count * (partition + 1) / kWorkPartitions;
                if constexpr (kEntryBalance) {
                    chunk_begin = entry_begin <= chunk_prefix ? 0 :
                        static_cast<uint32_t>(min(static_cast<uint64_t>(chunk_count), entry_begin - chunk_prefix));
                    chunk_end = entry_end <= chunk_prefix ? 0 :
                        static_cast<uint32_t>(min(static_cast<uint64_t>(chunk_count), entry_end - chunk_prefix));
                    chunk_prefix += chunk_count;
                }
                for (uint32_t chunk = chunk_begin; chunk < chunk_end; ++ chunk, ++ chunk_ordinal) {
                    const uint32_t stage = chunk_ordinal & 1u;
                    const uint32_t phase = (chunk_ordinal / 2) & 1u;
                    if constexpr (kIsProducer) {
                        pipeline_wait(pipeline.empty_kv[stage], phase ^ 1u);
                        gather_async<kNumProducerThreads>(smem, stage, splits[split_idx].kv_block_infos,
                            chunk * kBlocksPerChunk, num_blocks, entry.num_q_tokens, accessor);
                        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                        pipeline_arrive(pipeline.full_kv[stage]);
                    } else {
                        pipeline_wait(pipeline.full_kv[stage], phase);
                        compute_chunk<kCacheQ>(smem, stage, tid - kNumProducerThreads, entry, split_header,
                            logits, logits_stride, output_cols, operands);
                        pipeline_arrive(pipeline.empty_kv[stage]);
                    }
                }
            }
            if constexpr (not kIsProducer)
                pipeline_arrive(pipeline.empty_q);
            ++ q_ordinal;
        }
    };
    if (is_producer)
        run_pipeline.template operator()<true>();
    else
        run_pipeline.template operator()<false>();
    __syncthreads();
}

#endif

} // namespace deep_gemm::sm120_sparse_mqa_detail

namespace deep_gemm {

template <bool kIsFP4, bool kIsPaged, uint32_t SPARSE_BLOCK_KV, uint32_t PAGE_KV,
          uint32_t kNumSMs, bool kUseUnalignedKs, uint32_t kWorkPartitions = 1, bool kPipeline = false,
          bool kWarpSpecialized = false, bool kCacheQ = false, bool kEntryBalance = false>
CUTLASS_GLOBAL __launch_bounds__(kWarpSpecialized ? 256 : 128, kWarpSpecialized ? 4 : 1)
void sm120_fp8_fp4_sparse_mqa_logits_kernel(
    const uint8_t* q, const uint8_t* sf_q, const uint8_t* kv, const uint8_t* sf_kv,
    const nv_bfloat16* weights, const uint8_t* metadata, nv_bfloat16* logits,
    const uint64_t num_q_tokens, const uint64_t num_kv_tokens_or_pages,
    const uint64_t kv_page_stride, const uint64_t metadata_bytes,
    const uint64_t weights_stride, const uint64_t logits_stride, const uint32_t output_cols) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200) and (__CUDA_ARCH__ < 1300)) or defined(__CLION_IDE__)
    using namespace sm120_sparse_mqa_detail;
    constexpr uint32_t kRowBytes = kIsFP4 ? 64 : 128;
    DG_STATIC_ASSERT(not kEntryBalance or kWarpSpecialized, "Entry balance requires the warp-specialized path");
    DG_STATIC_ASSERT(not kIsPaged or not kPipeline, "Paged sparse gather uses the synchronous path");
    DG_STATIC_ASSERT(not kWarpSpecialized or (not kIsPaged and kPipeline and kIsFP4 and SPARSE_BLOCK_KV == 8 and kWorkPartitions == 4),
                     "Warp-specialized sparse gather only supports nonpaged aligned MXFP4 S8 P4");
    cudaGridDependencySynchronize();
    if constexpr (kIsPaged) {
        DG_STATIC_ASSERT(PAGE_KV > 0 and PAGE_KV % SPARSE_BLOCK_KV == 0 and not kUseUnalignedKs, "Invalid paged sparse shape");
        const PagedKVAccessor<kRowBytes, SPARSE_BLOCK_KV, PAGE_KV> accessor = {kv, num_kv_tokens_or_pages, kv_page_stride};
        compute<kIsFP4, SPARSE_BLOCK_KV, kNumSMs, kUseUnalignedKs, kWorkPartitions, false>(q, sf_q, weights, metadata, logits,
            num_q_tokens, metadata_bytes, weights_stride, logits_stride, output_cols, accessor);
    } else {
        const ContiguousKVAccessor<kRowBytes> accessor = {kv, sf_kv, num_kv_tokens_or_pages};
        if constexpr (kWarpSpecialized) {
            DG_STATIC_ASSERT(kPipeline, "Warp-specialized gather requires aligned asynchronous copies");
            compute_warp_specialized<kIsFP4, SPARSE_BLOCK_KV, kNumSMs, kUseUnalignedKs, kWorkPartitions, kCacheQ, kEntryBalance>(
                q, sf_q, weights, metadata, logits, num_q_tokens, metadata_bytes, weights_stride, logits_stride, output_cols, accessor);
        } else {
            compute<kIsFP4, SPARSE_BLOCK_KV, kNumSMs, kUseUnalignedKs, kWorkPartitions, kPipeline>(q, sf_q, weights, metadata, logits,
                num_q_tokens, metadata_bytes, weights_stride, logits_stride, output_cols, accessor);
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports SM120 family");
#endif
}

} // namespace deep_gemm
