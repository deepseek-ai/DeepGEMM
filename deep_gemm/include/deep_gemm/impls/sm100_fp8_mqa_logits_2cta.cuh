#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/utils.cuh>
using namespace deep_gemm::math;
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {
using namespace deep_gemm::ptx;
using namespace deep_gemm::sm90;
using namespace deep_gemm::sm100;
using namespace deep_gemm::utils;

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

constexpr uint64_t CACHE_HINT = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
template <uint32_t CGA_SIZE, bool IS_2CTA, typename dtype_t>
struct TmaCopy {
    __device__ __forceinline__ void operator()(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
         dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
         const uint32_t& num_tma_multicast = 1) {
        static_assert(CGA_SIZE > 0, "Invalid CGA_SIZE");
        using TmaLdg = std::conditional<IS_2CTA, cute::SM100_TMA_2SM_LOAD_MULTICAST_2D, cute::SM90_TMA_LOAD_MULTICAST_2D>::type;
        TmaLdg::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr), (1 << CGA_SIZE) -1, CACHE_HINT, smem_ptr, inner_idx, outer_idx);
    }
};
template <bool IS_2CTA, typename dtype_t>
struct TmaCopy<1, IS_2CTA, dtype_t> {
    __device__ __forceinline__ void operator()(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
         dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx) {
        using TmaLdg = std::conditional<IS_2CTA, cute::SM100_TMA_2SM_LOAD_2D, cute::SM90_TMA_LOAD_2D>::type;
        TmaLdg::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr), CACHE_HINT, smem_ptr, inner_idx, outer_idx);
    }
};

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode, uint32_t CGA_SIZE, bool IS_2CTA,
          typename dtype_t>
__device__ __forceinline__ void
tma_copy_impl(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
         dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx) {
    constexpr uint32_t BLOCK_INNER_ATOM = get_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();
    
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
        TmaCopy<CGA_SIZE, IS_2CTA, dtype_t>()(desc_ptr, barrier_ptr, smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
            inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
    }     
}

__device__ __forceinline__ void arrive_cta(void *smem_ptr, uint32_t cta_id) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);

    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id));
}

__device__ __forceinline__ uint32_t cluster_idx() {
     uint32_t ret;
     asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(ret));
     return ret;
 }

 __device__ __forceinline__ uint32_t cluster_dim() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %nclusterid.x;\n" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ void bulk_copy_g2s(void const* gmem_ptr, void* mbar_ptr, void* smem_ptr, int32_t load_bytes) {
    uint32_t smem_int_mbar = cute::cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cute::cast_smem_ptr_to_uint(smem_ptr);

    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n"
                     :
                     : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar), "n"(3)
                     : "memory");
}


template <uint32_t kNumHeads,
          uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q,
          uint32_t BLOCK_KV,
          uint32_t kNumQStages,
          uint32_t kNumKVStages,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp8_mqa_logits_2cta(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t max_seqlen_k, const uint64_t stride_logits,
                          uint32_t* cu_seq_len_k_start_and_end,
                          float* logits,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // Normally, `h (kNumHeads) == 32` and `d (kHeadDim) == 64`
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    constexpr uint32_t BLOCK_Q_2CTA = BLOCK_Q * 2;
    const auto& num_q_blocks = ceil_div(seq_len, BLOCK_Q_2CTA);

    // Types
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warp_in_group_idx = warp_idx % 4;
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q_2CTA * kNumHeads * sizeof(__half);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = (BLOCK_KV / 2) * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = (BLOCK_KV / 2) * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    // Align to 512 bytes for swizzle-64B
    extern __shared__ __align__(512) uint8_t smem_buffer[];

    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__half*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_kv_scales =  PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    static constexpr uint32_t SMEM_KV_OFFSET_SIZE_PER_STAGE = BLOCK_Q_2CTA * sizeof(uint2);
    auto smem_kv_offsets =  PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint2*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages + SMEM_KV_OFFSET_SIZE_PER_STAGE * i);
    });

    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_offsets[kNumQStages]);
    auto full_q_barriers     = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    
    constexpr uint32_t kNumTmemCols = 512;
    constexpr uint32_t UMMA_M = 256;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = BLOCK_Q_2CTA * kNumHeads;
    constexpr uint32_t kNumUMMAStages = kNumTmemCols / UMMA_N;
    auto full_umma_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumUMMAStages + i); });
    
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumUMMAStages * 2);
    cute::TMEM::Allocator2Sm allocator;
    const bool& is_load_q_warp = (warp_idx == (kNumMathThreads / 32));
    const bool& is_load_kv_warp = (warp_idx == (kNumMathThreads / 32 + 1));
    const bool& is_umma_warp = (warp_idx == (kNumMathThreads / 32 + 2));
    
    if (is_load_q_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumUMMAStages; ++ i) {
            full_umma_barriers[i]->init(1);
            empty_umma_barriers[i]->init(kNumMathThreads * 2);
        }
        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (is_umma_warp) {
        // Allocate tensor memory
        allocator.allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 24;
    constexpr uint32_t kNumMathRegisters = 224;

    // Block scheduler (reverse order: num_q_blocks-1 -> 0)
    int32_t block_q_idx = (int32_t)num_q_blocks - 1 - (int32_t)cluster_idx();
    uint32_t q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<int32_t, uint32_t> {
        return {block_q_idx - (int32_t)cluster_dim(), q_iter_idx + 1};
    };

    const auto& get_q_pipeline = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t> {
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                };
    };

    uint32_t num_total_kv_blocks = 0;
    const auto& get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks  + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    const auto& get_umma_pipeline = [&](const uint32_t& umma_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (umma_block_idx) % kNumUMMAStages,         // UMMA pipeline stage
            ((umma_block_idx) / kNumUMMAStages) & 1    // UMMA pipeline phase
        };
    };

    uint32_t seq_k_start[BLOCK_Q_2CTA];
    const auto& get_kv_offsets = [&](const uint32_t& q_stage_idx) -> cute::tuple<uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        uint2* kv_offsets_ptr = smem_kv_offsets[q_stage_idx];

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q_2CTA; ++ i) {
            uint2 kv_offset = kv_offsets_ptr[i];
            seq_k_start[i] = kv_offset.x;
            start = min(start, kv_offset.x);
            end = max(end, kv_offset.y);
        }
        start = min(start, seq_len_kv);
        start = start / 4 * 4;
        end = min(end, seq_len_kv);
        return {start, ceil_div(end - start, BLOCK_KV)};
    };

    if (is_load_q_warp) {
        cutlass::arch::warpgroup_reg_dealloc<48>();

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            const uint32_t q_cta_offset = (block_idx * BLOCK_Q_2CTA + cta_rank_in_cluster * BLOCK_Q) * kNumHeads;
            tma_copy_impl<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim, 1, true>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_cta_offset);

            const uint32_t w_cta_shared_offset = cta_rank_in_cluster * BLOCK_Q * kNumHeads;
            const uint32_t w_cta_offset = block_idx * BLOCK_Q_2CTA + cta_rank_in_cluster * BLOCK_Q;
            tma_copy_impl<kNumHeads, BLOCK_Q, 0, 2, false>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx] + w_cta_shared_offset, 0, w_cta_offset);
            if (cta_rank_in_cluster == 0) {
                // bulk_copy_g2s(void const* gmem_ptr, uint64_t* mbar_ptr, void* smem_ptr, int32_t load_bytes)
                const uint32_t* start_and_end_ptr = cu_seq_len_k_start_and_end + block_idx * BLOCK_Q_2CTA * 2;
                bulk_copy_g2s(start_and_end_ptr, full_q_barriers[stage_idx], smem_kv_offsets[stage_idx], SMEM_KV_OFFSET_SIZE_PER_STAGE);

                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE * 2 + SMEM_WEIGHT_SIZE_PER_STAGE + SMEM_KV_OFFSET_SIZE_PER_STAGE);
            } else {
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_WEIGHT_SIZE_PER_STAGE + SMEM_KV_OFFSET_SIZE_PER_STAGE);
            }
        };

        if (cute::elect_one_sync()) {
            if (block_q_idx >= 0) {
                issue_tma_q(0, block_q_idx);
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }

            while (block_q_idx >= 0) {
                CUTE_TIE_DECL(get_q_pipeline(), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, block_q_idx);
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (is_load_kv_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            while (block_q_idx >= 0) {
                CUTE_TIE_DECL(get_q_pipeline(), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);
                CUTE_TIE_DECL(get_kv_offsets(q_stage_idx), kv_start, num_kv_blocks);
                const uint32_t kv_cta_offset = cta_rank_in_cluster * (BLOCK_KV / 2);
                #pragma unroll 1
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    tma_copy_impl<kHeadDim, BLOCK_KV / 2, kHeadDim, 1, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                           smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV + kv_cta_offset);

                    tma_copy_impl<BLOCK_KV / 2, 1, 0, 1, false>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                        smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV + kv_cta_offset, 0);
                    if (cta_rank_in_cluster == 0) {
                        full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE * 2 + SMEM_KV_SCALE_SIZE_PER_STAGE);
                    } else {
                        full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SCALE_SIZE_PER_STAGE);
                    }

                }
                num_total_kv_blocks += num_kv_blocks;
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (is_umma_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cta_rank_in_cluster != 0)
            goto EXIT;

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        uint32_t umma_block_idx = 0;
        while (block_q_idx >= 0) {
            CUTE_TIE_DECL(get_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            CUTE_TIE_DECL(get_kv_offsets(q_stage_idx), kv_start, num_kv_blocks);

            #pragma unroll 1
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                CUTE_TIE_DECL(get_umma_pipeline(umma_block_idx), umma_stage_idx, umma_phase);
                empty_umma_barriers[umma_stage_idx]->wait(umma_phase ^ 1);
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                    auto a_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(smem_kv[kv_stage_idx], 0, k * UMMA_K);
                    auto b_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(smem_q[q_stage_idx], 0, k * UMMA_K);
                    cute::SM100_MMA_F8F6F4_2x1SM_SS::fma(a_desc, b_desc, umma_stage_idx * UMMA_N, k, runtime_instr_desc);
                }
                cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(full_umma_barriers[umma_stage_idx]), 0x03);
                umma_block_idx += 1;
            }
            num_total_kv_blocks += num_kv_blocks;
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    } else if (warp_idx >= kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        constexpr uint32_t kNumHeadsInHalf2 = kNumHeads / 2;
        __half2 weights[BLOCK_Q][kNumHeadsInHalf2];

        uint32_t umma_block_idx = 0;
        while (block_q_idx >= 0) {
            CUTE_TIE_DECL(get_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            __half2 *weights_offset = reinterpret_cast<__half2*>(smem_weights[q_stage_idx] + warpgroup_idx * BLOCK_Q * kNumHeads);
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                for (uint32_t j = 0; j < kNumHeadsInHalf2; ++ j) {
                    weights[i][j] = *(weights_offset + i * kNumHeadsInHalf2 + j);
                }
            }
            CUTE_TIE_DECL(get_kv_offsets(q_stage_idx), kv_start, num_kv_blocks);

            // Compute over KV blocks
            #pragma unroll 1
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);

                const auto& warp_offset = warp_in_group_idx * 32;
                const auto& v_offset = lane_idx;
                
                CUTE_TIE_DECL(get_umma_pipeline(umma_block_idx), umma_stage_idx, umma_phase);
                full_umma_barriers[umma_stage_idx]->wait(umma_phase);
                float scale_kv = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_offset);
                empty_kv_barriers[kv_stage_idx]->arrive();

                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + cta_rank_in_cluster * BLOCK_KV / 2 + warp_offset;

                constexpr uint32_t kNumLDTMElems = kNumHeads * BLOCK_Q;
                DG_STATIC_ASSERT(kNumLDTMElems == 32 or kNumLDTMElems == 64 or kNumLDTMElems == 128, "Invalid kNumLDTMElems");
                uint32_t shifted_accum[kNumLDTMElems/2];
                auto tmem_load = [&](auto offset, auto... Is) {
                    if constexpr (kNumLDTMElems == 32) {
                        cute::SM100_TMEM_LOAD_32dp32b16x_16b::copy(offset, shifted_accum[Is]...);
                    } else if constexpr (kNumLDTMElems == 64) {
                        cute::SM100_TMEM_LOAD_32dp32b32x_16b::copy(offset, shifted_accum[Is]...);
                    } else if constexpr (kNumLDTMElems == 128) {
                        cute::SM100_TMEM_LOAD_32dp32b64x_16b::copy(offset, shifted_accum[Is]...);
                    }
                };
                uint32_t tmem_start = umma_stage_idx * UMMA_N + warpgroup_idx * UMMA_N / 2;
                [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load(tmem_start, Is...); }(cute::make_index_sequence<kNumLDTMElems/2>{});
                cutlass::arch::fence_view_async_tmem_load();
                if (kv_block_idx == num_kv_blocks - 1) {
                    empty_q_barriers[q_stage_idx]->arrive();
                }
                arrive_cta(empty_umma_barriers[umma_stage_idx], 0);
                umma_block_idx += 1;
                
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    __half2* accum = reinterpret_cast<__half2*>(shifted_accum + i * kNumHeads / 2);

                    

                    auto sum_0 = __float2half2_rn(0.0f);
                    auto sum_1 = __float2half2_rn(0.0f);

                    const auto& transform_reg = [&](const uint32_t& j, const __half2& sum) {
                        __half2 acc = __hmax2(accum[j], __float2half2_rn(0.0f));
                        return __hfma2(acc, weights[i][j], sum);                    
                    };

                    #pragma unroll
                    for (int j = 0; j < kNumHeadsInHalf2; j += 2) {
                        sum_0 = transform_reg(j, sum_0);
                        sum_1 = transform_reg(j + 1, sum_1);
                    }

                    auto result0 = __hadd2(sum_0, sum_1);
                    float2 result1 = __half22float2(result0);
                    float result = (result1.x + result1.y) * scale_kv;
                    
                    const uint32_t& q_idx = block_q_idx * BLOCK_Q_2CTA + warpgroup_idx * BLOCK_Q + i;
                    if constexpr (kIsCompressedLogits) {
                        if (kv_offset + v_offset >= seq_k_start[i])
                            logits[q_idx * stride_logits + kv_offset + v_offset - seq_k_start[i + warpgroup_idx * BLOCK_Q]] = result;
                    } else {
                        logits[q_idx * stride_logits + kv_offset + v_offset] = result;
                    }
                }
            }
            num_total_kv_blocks += num_kv_blocks;
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }

EXIT:
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    if (is_load_q_warp)
        allocator.free(0, kNumTmemCols);
}

} // namespace deep_gemm
