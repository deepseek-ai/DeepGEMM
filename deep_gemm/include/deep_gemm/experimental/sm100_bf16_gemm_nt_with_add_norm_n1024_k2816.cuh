#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/math.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::experimental {

// Global, relaxed-GPU, and shared-memory PTX helpers
CUTLASS_DEVICE void st_global_cg_v4(void* ptr, const uint4& v) { asm volatile("st.global.cg.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
CUTLASS_DEVICE void st_global_cs_v4(void* ptr, const uint4& v) { asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
CUTLASS_DEVICE void st_global_cs_256b(void* ptr, const uint4& v0, const uint4& v1, const bool aligned_32b) {
    if (aligned_32b) {
        asm volatile("st.global.cs.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                     :: "l"(ptr), "r"(v0.x), "r"(v0.y), "r"(v0.z), "r"(v0.w),
                        "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w) : "memory");
    } else {
        const auto u4_ptr = static_cast<uint4*>(ptr);
        st_global_cs_v4(u4_ptr, v0);
        st_global_cs_v4(u4_ptr + 1, v1);
    }
}

CUTLASS_DEVICE uint32_t ld_relaxed_gpu_u32(const uint32_t* ptr) { uint32_t ret; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr)); return ret; }
CUTLASS_DEVICE float ld_relaxed_gpu_f32(const float* ptr) { float ret; asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr)); return ret; }
CUTLASS_DEVICE void st_relaxed_gpu_f32(float* ptr, const float val) { asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" :: "l"(ptr), "f"(val) : "memory"); }
CUTLASS_DEVICE float4 ld_relaxed_gpu_f32x4(const float4* ptr) {
    float4 ret;
    asm volatile("ld.relaxed.gpu.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
    return ret;
}
CUTLASS_DEVICE uint4 ld_global_ca_u4(const uint4* ptr) {
    uint4 ret;
    asm("ld.global.ca.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint4 ld_global_u4_16b_or_2b(const void* base_ptr, const uint32_t u4_idx, const uint32_t const_u4_offset = 0) {
    if ((reinterpret_cast<uintptr_t>(base_ptr) & 15u) == 0u)
        return ld_global_ca_u4(static_cast<const uint4*>(base_ptr) + u4_idx + const_u4_offset);
    const auto u16_ptr = static_cast<const uint16_t*>(base_ptr) + u4_idx * 8 + const_u4_offset * 8;
    const auto pack_pair = [&](const uint32_t elem_idx) {
        return static_cast<uint32_t>(u16_ptr[elem_idx]) | (static_cast<uint32_t>(u16_ptr[elem_idx + 1]) << 16);
    };
    return make_uint4(pack_pair(0), pack_pair(2), pack_pair(4), pack_pair(6));
}

CUTLASS_DEVICE void st_global_u4_16b_or_2b(void* base_ptr, const uint64_t elem_offset_idx, const uint4& val) {
    const auto dst_u16 = static_cast<uint16_t*>(base_ptr) + elem_offset_idx;
    if ((reinterpret_cast<uintptr_t>(dst_u16) & 15u) == 0u) {
        st_global_cs_v4(dst_u16, val);
    } else {
        #pragma unroll
        for (uint32_t word_idx = 0; word_idx < 4; ++ word_idx) {
            const uint32_t word = (&val.x)[word_idx];
            dst_u16[word_idx * 2] = static_cast<uint16_t>(word);
            dst_u16[word_idx * 2 + 1] = static_cast<uint16_t>(word >> 16);
        }
    }
}

CUTLASS_DEVICE uint4 ld_smem_u4(const uint32_t smem_addr, const uint32_t byte_offset = 0) {
    uint4 ret;
    asm volatile("{\n\t.reg .u32 addr;\n\tadd.u32 addr, %4, %5;\n\tld.shared.v4.u32 {%0, %1, %2, %3}, [addr];\n\t}"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "r"(smem_addr), "r"(byte_offset) : "memory");
    return ret;
}
CUTLASS_DEVICE void st_smem_u4(const uint32_t smem_addr, const uint4& v) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(smem_addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
CUTLASS_DEVICE constexpr uint32_t swizzle_128b_offset_bytes(const uint32_t local_row_idx, const uint32_t vec_u4_idx) {
    return local_row_idx * 128u + (vec_u4_idx ^ (local_row_idx & 7u)) * 16u;
}
CUTLASS_DEVICE uint4 ld_swizzled_u4(const void* base_ptr, const uint32_t local_row_idx, const uint32_t vec_u4_idx) {
    const uint32_t row_addr = cute::cast_smem_ptr_to_uint(base_ptr) + swizzle_128b_offset_bytes(local_row_idx, 0);
    uint4 ret;
    asm volatile("{\n\t.reg .u32 addr;\n\txor.b32 addr, %4, %5;\n\tld.shared.v4.u32 {%0, %1, %2, %3}, [addr];\n\t}"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "r"(row_addr), "r"(vec_u4_idx * 16u) : "memory");
    return ret;
}
CUTLASS_DEVICE void st_swizzled_u4(void* base_ptr, const uint32_t local_row_idx, const uint32_t vec_u4_idx, const uint4& val) {
    const uint32_t row_addr = cute::cast_smem_ptr_to_uint(base_ptr) + swizzle_128b_offset_bytes(local_row_idx, 0);
    asm volatile("{\n\t.reg .u32 addr;\n\txor.b32 addr, %0, %1;\n\tst.shared.v4.u32 [addr], {%2, %3, %4, %5};\n\t}"
                 :: "r"(row_addr), "r"(vec_u4_idx * 16u), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
}
CUTLASS_DEVICE void sync_epi_group_barrier(const uint32_t group_idx) {
    asm volatile("{\n\t.reg .u32 bar_idx;\n\tadd.u32 bar_idx, %0, 9;\n\tbar.sync bar_idx, 128;\n\t}" :: "r"(group_idx) : "memory");
}
CUTLASS_DEVICE void wait_smem_mbarrier(const void* bar_ptr, const uint32_t phase_idx) {
    const uint32_t bar_addr = cute::cast_smem_ptr_to_uint(bar_ptr);
    asm volatile("{\n\t.reg .pred p;\n\tMBAR_WAIT_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 10000000;\n\t@!p bra MBAR_WAIT_%=;\n\t}"
                 :: "r"(bar_addr), "r"(phase_idx) : "memory");
}
CUTLASS_DEVICE uint4 shfl_u4_sync(const uint4& val, const int src_lane_idx) {
    return make_uint4(__shfl_sync(0xffffffff, val.x, src_lane_idx), __shfl_sync(0xffffffff, val.y, src_lane_idx),
                      __shfl_sync(0xffffffff, val.z, src_lane_idx), __shfl_sync(0xffffffff, val.w, src_lane_idx));
}

template <typename... Descs>
CUTLASS_DEVICE void prefetch_tma_descs_warp0(const uint32_t warp_idx, const Descs&... descs) {
    if (warp_idx == 0 and cute::elect_one_sync())
        (cute::prefetch_tma_descriptor(&descs), ...);
    __syncwarp();
}

template <uint32_t kNumPages, typename PageArray>
CUTLASS_DEVICE void issue_fenced_tma_store_2d_pages(
    const cute::TmaDescriptor& desc, const PageArray& pages, const uint32_t base_n_idx,
    const uint32_t m_idx, const uint32_t page_col_stride) {
    cute::tma_store_fence();
    #pragma unroll
    for (uint32_t page_idx = 0; page_idx < kNumPages; ++ page_idx) {
        ptx::tma_store_2d(&desc, pages[page_idx], base_n_idx + page_idx * page_col_stride,
                          m_idx, cute::TMA::CacheHintSm100::EVICT_FIRST);
    }
    cute::tma_store_arrive();
}

// Fused BF16 residual-add, sum-of-squares, and RMSNorm scaling helpers
template <uint32_t kShapeN>
CUTLASS_DEVICE float compute_rstd_from_sumsq(const float sumsq) {
    const float eps_mean = sumsq * (1.0f / static_cast<float>(kShapeN)) + 1e-6f;
    float ret;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(eps_mean));
    return ret;
}

CUTLASS_DEVICE uint32_t scale_bf16x2_by_rstd_gamma(const nv_bfloat162& val, const nv_bfloat162& gamma_val, const float rstd) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    float prod_x, prod_y;
    asm("fma.rn.f32.bf16 %0, %1, %2, 0f00000000;" : "=f"(prod_x)
        : "h"(*reinterpret_cast<const uint16_t*>(&val.x)), "h"(*reinterpret_cast<const uint16_t*>(&gamma_val.x)));
    asm("fma.rn.f32.bf16 %0, %1, %2, 0f00000000;" : "=f"(prod_y)
        : "h"(*reinterpret_cast<const uint16_t*>(&val.y)), "h"(*reinterpret_cast<const uint16_t*>(&gamma_val.y)));
    const auto rounded = math::cast_into_bf16x2(prod_x * rstd, prod_y * rstd);
#else
    const auto res_f32 = ptx::cvt_bf16x2_to_fp32x2(val), gamma_f32 = ptx::cvt_bf16x2_to_fp32x2(gamma_val);
    const auto rounded = math::cast_into_bf16x2((res_f32.x * gamma_f32.x) * rstd, (res_f32.y * gamma_f32.y) * rstd);
#endif
    return *reinterpret_cast<const uint32_t*>(&rounded);
}

CUTLASS_DEVICE uint4 scale_uint4_bf16_with_gamma(const uint4& res_u4, const uint4& gamma_u4, const float rstd) {
    const auto res_pairs = reinterpret_cast<const nv_bfloat162*>(&res_u4);
    const auto gamma_pairs = reinterpret_cast<const nv_bfloat162*>(&gamma_u4);
    return make_uint4(scale_bf16x2_by_rstd_gamma(res_pairs[0], gamma_pairs[0], rstd),
                      scale_bf16x2_by_rstd_gamma(res_pairs[1], gamma_pairs[1], rstd),
                      scale_bf16x2_by_rstd_gamma(res_pairs[2], gamma_pairs[2], rstd),
                      scale_bf16x2_by_rstd_gamma(res_pairs[3], gamma_pairs[3], rstd));
}

template <bool kHasBias>
CUTLASS_DEVICE uint4 accumulate_bias_residual_u4(
    const float2* acc_pairs, const uint4& bias_u4, const uint4& res_u4, float2& row_sumsq) {
    const auto bias_pairs = reinterpret_cast<const nv_bfloat162*>(&bias_u4);
    const auto res_pairs = reinterpret_cast<const nv_bfloat162*>(&res_u4);
    uint32_t packed[4];
    #pragma unroll
    for (uint32_t pair_idx = 0; pair_idx < 4; ++ pair_idx) {
        auto acc = acc_pairs[pair_idx];
        if constexpr (kHasBias)
            ptx::accumulate(acc, bias_pairs[pair_idx]);
        ptx::accumulate(acc, res_pairs[pair_idx]);
        const auto rounded = math::cast_into_bf16x2(acc.x, acc.y);
        packed[pair_idx] = *reinterpret_cast<const uint32_t*>(&rounded);
        ptx::accumulate_square(row_sumsq, rounded);
    }
    return make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

// Pipeline, counter, and UMMA helpers
template <uint32_t kCount, typename Bar>
CUTLASS_DEVICE void init_barriers(Bar (&bars)[kCount], const uint32_t lane_idx, const uint32_t arrivals) {
    DG_STATIC_ASSERT(kCount <= 32, "Warp-parallel barrier init requires kCount <= 32");
    if (lane_idx < kCount)
        bars[lane_idx].init(arrivals);
}

template <uint32_t kCount, typename FullBar, typename EmptyBar>
CUTLASS_DEVICE void init_barrier_pairs(FullBar (&full)[kCount], EmptyBar (&empty)[kCount], const uint32_t lane_idx,
                                       const uint32_t full_arrivals, const uint32_t empty_arrivals) {
    DG_STATIC_ASSERT(kCount <= 32, "Warp-parallel barrier init requires kCount <= 32");
    if (lane_idx < kCount) {
        full[lane_idx].init(full_arrivals);
        empty[lane_idx].init(empty_arrivals);
    }
}

CUTLASS_DEVICE void fence_warp_barrier_init() {
    __syncwarp();
    if (cute::elect_one_sync())
        cutlass::arch::fence_barrier_init();
    __syncwarp();
}

template <typename Allocator, uint32_t kNumTmemCols>
CUTLASS_DEVICE void allocate_tmem_checked(uint32_t* tmem_ptr) {
    Allocator().allocate(kNumTmemCols, tmem_ptr);
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr) == 0);
}

template <uint32_t kNumStages>
CUTLASS_DEVICE void advance_stage(uint32_t& stage_idx, uint32_t& phase_idx) {
    stage_idx = (kNumStages == 1 or stage_idx + 1u == kNumStages) ? 0u : (stage_idx + 1u);
    phase_idx ^= (stage_idx == 0u);
}

CUTLASS_DEVICE void wait_counter_at_least(const uint32_t* counter_ptr, const uint32_t target) {
    const auto start_clock = clock64();
    while (ld_relaxed_gpu_u32(counter_ptr) < target)
        DG_TRAP_ONLY_DEVICE_ASSERT(clock64() - start_clock < comm::kNumTimeoutCycles);
}

// Counters live in uninitialized memory: zero them in-kernel, then wait for every CTA before any arrival
CUTLASS_DEVICE void init_counters_and_grid_sync(uint32_t* counters, const uint32_t num_rows, const uint32_t num_row_slots) {
    cudaGridDependencySynchronize();
    for (uint32_t row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x)
        counters[row_idx * num_row_slots] = 0u;
    cooperative_groups::this_grid().sync();
}

template <uint32_t UMMA_M, uint32_t UMMA_N>
CUTLASS_DEVICE auto make_bf16_umma_desc() {
    return cute::UMMA::make_runtime_instr_desc(
        cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                    UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>());
}

template <typename MMAOp, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K = 64>
CUTLASS_DEVICE void issue_bf16_umma_k_block(
    cute::UMMA::SmemDescriptor& a_desc, cute::UMMA::SmemDescriptor& b_desc,
    const uint32_t a_base_lo, const uint32_t b_base_lo,
    const uint32_t tmem_addr, const bool accum_prev_k, const uint64_t runtime_instr_desc) {
    constexpr uint32_t BLOCK_ATOM_K = 64, UMMA_K = 16, kNumSwizzleBytes = 128;
    #pragma unroll
    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
        const uint32_t atom_k_idx = umma_k_idx * UMMA_K / BLOCK_ATOM_K, inner_k_idx = umma_k_idx * UMMA_K % BLOCK_ATOM_K;
        a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_M, kNumSwizzleBytes, cutlass::bfloat16_t>(
            a_base_lo, atom_k_idx * BLOCK_M * BLOCK_ATOM_K, inner_k_idx);
        b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_N, kNumSwizzleBytes, cutlass::bfloat16_t>(
            b_base_lo, atom_k_idx * BLOCK_N * BLOCK_ATOM_K, inner_k_idx);
        MMAOp::fma(a_desc, b_desc, tmem_addr, umma_k_idx > 0 or accum_prev_k, runtime_instr_desc);
    }
}

template <typename MMAOp, uint32_t UMMA_M, uint32_t UMMA_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t BLOCK_K, uint32_t kNumAStages, uint32_t kNumBStages, bool kIs2CTA,
          typename SmemT, typename NextTileFn>
CUTLASS_DEVICE void run_umma_warp1_pipeline(
    SmemT& smem, const uint32_t lane_idx, uint32_t tile_idx, const uint32_t num_tiles,
    const uint32_t k_start_idx, const uint32_t k_end_idx, NextTileFn&& next_tile_fn) {
    if (tile_idx >= num_tiles)
        return;
    constexpr bool kAsymmetricAB = kNumAStages != kNumBStages;
    const auto runtime_instr_desc = make_bf16_umma_desc<UMMA_M, UMMA_N>();
    auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_M, 64, 128>(smem.a[0], 0, 0);
    auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_N, 64, 128>(smem.b[0], 0, 0);
    const uint32_t a_desc_lo = lane_idx < kNumAStages ? a_desc.lo + lane_idx * sizeof(smem.a[0]) / 16 : 0u;
    const uint32_t b_desc_lo = lane_idx < kNumBStages ? b_desc.lo + lane_idx * sizeof(smem.b[0]) / 16 : 0u;
    uint32_t a_stage_idx = 0, a_phase_idx = 0, b_stage_idx = 0, b_phase_idx = 0, iter_idx = 0;
    do {
        const uint32_t accum_stage_idx = iter_idx & 1u, tmem_col_idx = accum_stage_idx * 256u;
        smem.tmem_empty_barriers[accum_stage_idx].wait(((iter_idx >> 1u) & 1u) ^ 1u);
        ptx::tcgen05_after_thread_sync();
        for (uint32_t k_block_idx = k_start_idx; k_block_idx < k_end_idx; ++ k_block_idx) {
            smem.a_full_barriers[a_stage_idx].wait(a_phase_idx);
            if constexpr (kAsymmetricAB)
                smem.b_full_barriers[b_stage_idx].wait(b_phase_idx);
            ptx::tcgen05_after_thread_sync();
            const auto a_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(a_stage_idx));
            const auto b_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(kAsymmetricAB ? b_stage_idx : a_stage_idx));
            if (cute::elect_one_sync()) {
                issue_bf16_umma_k_block<MMAOp, BLOCK_M, BLOCK_N, BLOCK_K>(
                    a_desc, b_desc, a_base_lo, b_base_lo, tmem_col_idx, k_block_idx > k_start_idx, runtime_instr_desc);
                if constexpr (not kIs2CTA) {
                    ptx::umma_arrive_no_elect(smem.a_empty_barriers[a_stage_idx]);
                    if (k_block_idx + 1 == k_end_idx)
                        ptx::umma_arrive_no_elect(smem.tmem_full_barriers[accum_stage_idx]);
                }
            }
            __syncwarp();
            if constexpr (kIs2CTA) {
                cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&smem.a_empty_barriers[a_stage_idx]), 0x3);
                if constexpr (kAsymmetricAB)
                    cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&smem.b_empty_barriers[b_stage_idx]), 0x3);
                if (k_block_idx + 1 == k_end_idx)
                    cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&smem.tmem_full_barriers[accum_stage_idx]), 0x3);
                __syncwarp();
            }
            advance_stage<kNumAStages>(a_stage_idx, a_phase_idx);
            if constexpr (kAsymmetricAB)
                advance_stage<kNumBStages>(b_stage_idx, b_phase_idx);
        }
        ++ iter_idx;
    } while (next_tile_fn(tile_idx));
    smem.tmem_empty_barriers[(iter_idx - 1) & 1u].wait(((iter_idx - 1) >> 1u) & 1u);
}

// 1. Small-M 1-CTA Weight-Stationary + Fine-Slice SS (BLOCK_N = 16/32) + Split-K
template <uint32_t kShapeK, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kSplitK = 1>
struct alignas(1024) SM100BF16GemmAddNormSmallSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static constexpr bool kUseFineSS = BLOCK_N < 64;
    static constexpr uint32_t kUmmaM = kUseFineSS ? 64 : BLOCK_M;
    static constexpr uint32_t kNumStageBytes = (kUmmaM + BLOCK_N) * BLOCK_K * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t kNumKBlocks = kShapeK / BLOCK_K;
    static constexpr uint32_t kNumKBlocksPerSplit = (kNumKBlocks + kSplitK - 1) / kSplitK;
    static constexpr uint32_t kNumMaxStages = 221184 / kNumStageBytes;
    static constexpr uint32_t kNumStageCap = kShapeK > 1024 ? (BLOCK_K == 128 ? 8 : 12) : 16;
    static constexpr uint32_t kNumCappedStages = kNumMaxStages < kNumStageCap ? kNumMaxStages : kNumStageCap;
    static constexpr uint32_t kNumStages = kNumKBlocksPerSplit == 0 ? 1 : (kNumKBlocksPerSplit < kNumCappedStages ? kNumKBlocksPerSplit : kNumCappedStages);

    alignas(1024) cutlass::bfloat16_t a[kNumStages][kUmmaM * BLOCK_K];
    cutlass::bfloat16_t b[kNumStages][BLOCK_N * BLOCK_K];
    Barrier a_full_barriers[kNumStages];
    Barrier a_empty_barriers[kNumStages];
    Barrier tmem_full_barriers[2];
    Barrier tmem_empty_barriers[2];
    uint32_t tmem_ptr;
    alignas(16) float fold_sumsq[256];
};

template <uint32_t kShapeK, bool kHasBias, int kNumSMs,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kSplitK = 1>
CUTLASS_DEVICE void sm100_bf16_gemm_add_norm_small_device(
    uint32_t shape_m, const cute::TmaDescriptor& tensor_map_a, const cute::TmaDescriptor& tensor_map_b,
    void* __restrict__ residual, void* __restrict__ d,
    const void* __restrict__ bias, const void* __restrict__ gamma,
    float* __restrict__ partials, uint32_t* __restrict__ counters) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Allocator = cute::TMEM::Allocator1Sm;
    using SharedStorage = SM100BF16GemmAddNormSmallSharedStorage<kShapeK, BLOCK_M, BLOCK_N, BLOCK_K, kSplitK>;

    constexpr bool kUseFineSS = SharedStorage::kUseFineSS;
    constexpr uint32_t kUmmaM = SharedStorage::kUmmaM;
    constexpr uint32_t kUmmaN = BLOCK_N;
    constexpr uint32_t kShapeN = 1024, kNumSwizzleBytes = 128;
    constexpr uint32_t kNumStages = SharedStorage::kNumStages, kNumKBlocks = SharedStorage::kNumKBlocks;
    constexpr uint32_t kNumControlThreads = 128, kNumEpilogueThreads = 128, kNumControlWarps = 4;
    constexpr int kNumControlRegisters = kNumStages > 6 ? 96 : 64, kNumEpilogueRegisters = 232;
    constexpr uint32_t kNumNSlices = kShapeN / BLOCK_N, kNumCounterSlots = 32;
    constexpr uint32_t kNumTotalFolds = kNumEpilogueThreads / BLOCK_M, kNumUmmaColsPerFold = kUmmaN / kNumTotalFolds;
    constexpr uint32_t kNumFolds = kUseFineSS ? 1 : (BLOCK_N < kNumUmmaColsPerFold ? 1 : (BLOCK_N / kNumUmmaColsPerFold)), kNumElemsPerLoad = 8;
    constexpr uint32_t kNumLoadsPerFold = (BLOCK_N / kNumFolds) / kNumElemsPerLoad;
    constexpr bool kNeedRegRealloc = kSplitK > 1;
    constexpr uint32_t kNumSplitU4PerTile = (BLOCK_M * BLOCK_N) / 4, kNumTmemCols = 512, kNumTmemStageCols = 256;
    constexpr bool kStoreResAtEnd = kSplitK > 1;

    DG_STATIC_ASSERT((BLOCK_M == 32 or BLOCK_M == 64) and (BLOCK_N == 16 or BLOCK_N == 32 or BLOCK_N == 64) and kNumLoadsPerFold <= 4, "Invalid block size");
    DG_STATIC_ASSERT((BLOCK_K == 64 or BLOCK_K == 128) and kShapeK % BLOCK_K == 0, "Invalid block K");
    DG_STATIC_ASSERT(kSplitK >= 1 and kNumKBlocks >= kSplitK and kNumStages <= 16, "Invalid stage or Split-K config");
    DG_STATIC_ASSERT(kNumControlRegisters * kNumControlThreads + kNumEpilogueRegisters * kNumEpilogueThreads <= 64512, "Register budget exceeded");
    DG_STATIC_ASSERT(sizeof(SharedStorage) == kNumStages * SharedStorage::kNumStageBytes + 2048 and
                     sizeof(SharedStorage) <= 232448, "Unexpected shared storage size");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const auto warp_idx = static_cast<uint32_t>(cutlass::canonical_warp_idx_sync());
    const auto lane_idx = ptx::get_lane_idx();
    prefetch_tma_descs_warp0(warp_idx, tensor_map_a, tensor_map_b);

    if (warp_idx == 1) {
        init_barrier_pairs(smem.a_full_barriers, smem.a_empty_barriers, lane_idx, 1, 1);
        if constexpr (not kNeedRegRealloc)
            init_barrier_pairs(smem.tmem_full_barriers, smem.tmem_empty_barriers, lane_idx, 1, kNumEpilogueThreads);
        fence_warp_barrier_init();
    } else if (warp_idx == 2) {
        allocate_tmem_checked<Allocator, kNumTmemCols>(&smem.tmem_ptr);
    } else if (kNeedRegRealloc and warp_idx == 3) {
        init_barrier_pairs(smem.tmem_full_barriers, smem.tmem_empty_barriers, lane_idx, 1, kNumEpilogueThreads);
        fence_warp_barrier_init();
    }
    __syncthreads();

    const uint32_t num_m_blocks = math::ceil_div(shape_m, BLOCK_M), num_tiles = num_m_blocks * kNumNSlices;
    init_counters_and_grid_sync(counters, num_m_blocks + (kSplitK > 1 ? num_tiles : 0u), kNumCounterSlots);

    const uint32_t box_m = shape_m < BLOCK_M ? shape_m : BLOCK_M;
    constexpr uint32_t kNumAlignedSMs = (static_cast<uint32_t>(kNumSMs) / kNumNSlices) * kNumNSlices;
    DG_STATIC_ASSERT(kNumAlignedSMs >= kNumNSlices, "Invalid SM alignment");
    const uint32_t num_active_sms = kSplitK > 1
        ? (num_tiles * kSplitK)
        : ((math::ceil_div(num_tiles, kNumAlignedSMs) == math::ceil_div(num_tiles, static_cast<uint32_t>(kNumSMs)))
               ? kNumAlignedSMs : static_cast<uint32_t>(kNumSMs));
    const uint32_t split_idx = kSplitK > 1 ? (blockIdx.x / num_tiles) : 0u;
    const uint32_t start_tile_idx = kSplitK > 1
        ? (split_idx < kSplitK ? (blockIdx.x % num_tiles) : num_tiles)
        : (blockIdx.x < num_active_sms ? blockIdx.x : num_tiles);
    const uint32_t k_start_idx = kSplitK > 1 ? ((split_idx * kNumKBlocks) / kSplitK) : 0u;
    const uint32_t k_end_idx = kSplitK > 1 ? (((split_idx + 1u) * kNumKBlocks) / kSplitK) : kNumKBlocks;

    if (warp_idx < kNumControlWarps) {
        if constexpr (kNeedRegRealloc) {
            cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
            __syncwarp();
        }
        if (warp_idx == 0 and cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            constexpr uint64_t kEvictLast = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST);
            const uint32_t num_arrival_bytes = (box_m + BLOCK_N) * BLOCK_K * sizeof(cutlass::bfloat16_t);
            uint32_t stage_idx = 0, phase_idx = 0;
            for (uint32_t tile_idx = start_tile_idx; tile_idx < num_tiles; tile_idx += num_active_sms) {
                const uint32_t m_idx = (tile_idx / kNumNSlices) * BLOCK_M, n_idx = (tile_idx % kNumNSlices) * BLOCK_N;
                for (uint32_t k_block_idx = k_start_idx; k_block_idx < k_end_idx; ++ k_block_idx) {
                    smem.a_empty_barriers[stage_idx].wait(phase_idx ^ 1u);
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, kUmmaM, kNumSwizzleBytes, cutlass::bfloat16_t, false, kEvictLast>(
                        &tensor_map_a, &smem.a_full_barriers[stage_idx], smem.a[stage_idx], k_idx, m_idx, 1);
                    tma::copy<BLOCK_K, BLOCK_N, kNumSwizzleBytes, cutlass::bfloat16_t, false, kEvictLast>(
                        &tensor_map_b, &smem.a_full_barriers[stage_idx], smem.b[stage_idx], k_idx, n_idx, 1);
                    smem.a_full_barriers[stage_idx].arrive_and_expect_tx(num_arrival_bytes);
                    advance_stage<kNumStages>(stage_idx, phase_idx);
                }
            }
        } else if (warp_idx == 1) {
            using SmallMMAOp = cute::conditional_t<kUseFineSS, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_WS_SS>;
            run_umma_warp1_pipeline<SmallMMAOp, kUmmaM, kUmmaN, kUmmaM, BLOCK_N,
                                    BLOCK_K, kNumStages, kNumStages, false>(
                smem, lane_idx, start_tile_idx, num_tiles, k_start_idx, k_end_idx,
                [&](uint32_t& tile_idx) { tile_idx += num_active_sms; return tile_idx < num_tiles; });
        }
        __syncwarp();
    } else {
        if constexpr (kNeedRegRealloc) {
            cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
            __syncwarp();
        }
        cudaGridDependencySynchronize();

        const uint32_t epilogue_warp_idx = warp_idx - kNumControlWarps, epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t fold_idx = kUseFineSS ? 0u : (epilogue_thread_idx / BLOCK_M);
        const uint32_t local_row_idx = kUseFineSS ? (epilogue_warp_idx * 16 + (lane_idx & 15u)) : (epilogue_thread_idx % BLOCK_M);
        constexpr uint32_t kNumSplitThreads = kUseFineSS ? BLOCK_M : kNumEpilogueThreads;
        const uint32_t split_thread_idx = kUseFineSS ? (local_row_idx & (BLOCK_M - 1)) : epilogue_thread_idx;
        const bool is_epi_leader_lane = epilogue_warp_idx == 0 and cute::elect_one_sync();
        __syncwarp();

        const auto split_partials_u4 = reinterpret_cast<uint4*>(partials + static_cast<uint64_t>(num_m_blocks) * (kNumNSlices * BLOCK_M));
        if constexpr (kSplitK > 1) {
            if (split_idx > 0 and start_tile_idx < num_tiles) {
                smem.tmem_full_barriers[0].wait(0);
                ptx::tcgen05_after_thread_sync();
                const auto split_dst_base = split_partials_u4 +
                    static_cast<uint64_t>(start_tile_idx * (kSplitK - 1) + (split_idx - 1)) * kNumSplitU4PerTile + split_thread_idx;
                const bool has_split_row = kUseFineSS ? (lane_idx < 16u and local_row_idx < BLOCK_M) : true;
                uint32_t values[kNumLoadsPerFold][kNumElemsPerLoad];
                ptx::tmem_load_32dp32b<kNumLoadsPerFold * kNumElemsPerLoad>(0, &values[0][0]);
                cutlass::arch::fence_view_async_tmem_load();
                ptx::tcgen05_before_thread_sync();
                smem.tmem_empty_barriers[0].arrive();
                if (has_split_row) {
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < kNumLoadsPerFold; ++ load_idx) {
                        const auto v_u4 = reinterpret_cast<const uint4*>(values[load_idx]);
                        st_global_cg_v4(split_dst_base + (load_idx * 2) * kNumSplitThreads, v_u4[0]);
                        st_global_cg_v4(split_dst_base + (load_idx * 2 + 1) * kNumSplitThreads, v_u4[1]);
                    }
                }
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                if (is_epi_leader_lane)
                    ptx::red_add_rel(counters + static_cast<uint64_t>(num_m_blocks + start_tile_idx) * kNumCounterSlots, 1u);
                __syncwarp();
            }
        }

        if ((kSplitK == 1 or split_idx == 0) and start_tile_idx < num_tiles) {
            uint4 held[kNumLoadsPerFold];
            uint32_t iter_idx = 0;

            for (uint32_t tile_idx = start_tile_idx; tile_idx < num_tiles; tile_idx += num_active_sms, ++ iter_idx) {
                const uint32_t accum_stage_idx = iter_idx & 1u, tmem_col_idx = accum_stage_idx * kNumTmemStageCols;
                const uint32_t m_block_idx = tile_idx / kNumNSlices, slice_idx = tile_idx % kNumNSlices;
                const uint32_t m_idx = m_block_idx * BLOCK_M, n_base_idx = slice_idx * BLOCK_N, row_idx = m_idx + local_row_idx;
                const uint32_t col_u4_base_idx = (n_base_idx / kNumElemsPerLoad) + fold_idx * kNumLoadsPerFold;
                const bool has_fine_row = kUseFineSS ? (lane_idx < 16u and local_row_idx < BLOCK_M) : (fold_idx < kNumFolds);
                const bool valid_row = has_fine_row and (row_idx < shape_m);

                uint4 bias_packs[kNumLoadsPerFold], gamma_packs[kNumLoadsPerFold], preloaded_res[kNumLoadsPerFold];
                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < kNumLoadsPerFold; ++ load_idx) {
                    if constexpr (kHasBias)
                        bias_packs[load_idx] = has_fine_row ? ld_global_u4_16b_or_2b(bias, col_u4_base_idx, load_idx) : make_uint4(0, 0, 0, 0);
                    else
                        bias_packs[load_idx] = make_uint4(0, 0, 0, 0);
                    gamma_packs[load_idx] = has_fine_row ? ld_global_u4_16b_or_2b(gamma, col_u4_base_idx, load_idx) : make_uint4(0, 0, 0, 0);
                }

                const auto res_row_ptr = static_cast<const cutlass::bfloat16_t*>(residual) +
                                         static_cast<uint64_t>(row_idx) * kShapeN;
                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < kNumLoadsPerFold; ++ load_idx)
                    preloaded_res[load_idx] = valid_row ? ld_global_u4_16b_or_2b(res_row_ptr, col_u4_base_idx, load_idx) : make_uint4(0, 0, 0, 0);

                smem.tmem_full_barriers[accum_stage_idx].wait((iter_idx >> 1u) & 1u);
                ptx::tcgen05_after_thread_sync();
                uint32_t values[kNumLoadsPerFold][kNumElemsPerLoad];
                ptx::tmem_load_32dp32b<kNumLoadsPerFold * kNumElemsPerLoad>(tmem_col_idx, &values[0][0]);
                cutlass::arch::fence_view_async_tmem_load();
                ptx::tcgen05_before_thread_sync();
                smem.tmem_empty_barriers[accum_stage_idx].arrive();

                auto split_read_base = static_cast<const float4*>(nullptr);
                if constexpr (kSplitK > 1) {
                    if (is_epi_leader_lane) {
                        const auto split_counter_ptr = counters + static_cast<uint64_t>(num_m_blocks + tile_idx) * kNumCounterSlots;
                        wait_counter_at_least(split_counter_ptr, kSplitK - 1);
                    }
                    __syncwarp();
                    cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                    split_read_base = reinterpret_cast<const float4*>(
                        split_partials_u4 + static_cast<uint64_t>(tile_idx * (kSplitK - 1)) * kNumSplitU4PerTile + split_thread_idx);
                    asm volatile("" : "+l"(split_read_base) :: "memory");
                }

                float2 row_sumsq = make_float2(0.0f, 0.0f);
                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < kNumLoadsPerFold; ++ load_idx) {
                    const auto values_f32x2 = reinterpret_cast<const float2*>(values[load_idx]);
                    float2 acc_f32x2[4] = {values_f32x2[0], values_f32x2[1], values_f32x2[2], values_f32x2[3]};
                    if constexpr (kSplitK > 1) {
                        const auto load_base = split_read_base + (load_idx * 2) * kNumSplitThreads;
                        float4 split_p0[kSplitK - 1], split_p1[kSplitK - 1];
                        #pragma unroll
                        for (uint32_t s_idx = 0; s_idx < kSplitK - 1; ++ s_idx) {
                            split_p0[s_idx] = ld_relaxed_gpu_f32x4(load_base + s_idx * kNumSplitU4PerTile);
                            split_p1[s_idx] = ld_relaxed_gpu_f32x4(load_base + s_idx * kNumSplitU4PerTile + kNumSplitThreads);
                        }
                        #pragma unroll
                        for (uint32_t s_idx = 0; s_idx < kSplitK - 1; ++ s_idx) {
                            acc_f32x2[0] = make_float2(acc_f32x2[0].x + split_p0[s_idx].x, acc_f32x2[0].y + split_p0[s_idx].y);
                            acc_f32x2[1] = make_float2(acc_f32x2[1].x + split_p0[s_idx].z, acc_f32x2[1].y + split_p0[s_idx].w);
                            acc_f32x2[2] = make_float2(acc_f32x2[2].x + split_p1[s_idx].x, acc_f32x2[2].y + split_p1[s_idx].y);
                            acc_f32x2[3] = make_float2(acc_f32x2[3].x + split_p1[s_idx].z, acc_f32x2[3].y + split_p1[s_idx].w);
                        }
                    }
                    held[load_idx] = accumulate_bias_residual_u4<kHasBias>(acc_f32x2, bias_packs[load_idx], preloaded_res[load_idx], row_sumsq);
                    if constexpr (kSplitK > 1 and kNumLoadsPerFold * (kSplitK - 1) > 12)
                        asm volatile("" ::: "memory");
                }

                const auto block_partials = partials + static_cast<uint64_t>(m_block_idx) * (kNumNSlices * BLOCK_M);
                const auto slice_partials = block_partials + slice_idx * BLOCK_M;
                const auto counter_ptr = counters + m_block_idx * kNumCounterSlots;
                if constexpr (kNumFolds == 1) {
                    if (has_fine_row)
                        st_relaxed_gpu_f32(slice_partials + local_row_idx, valid_row ? (row_sumsq.x + row_sumsq.y) : 0.0f);
                    if constexpr (BLOCK_M == 32 and not kUseFineSS)
                        __syncwarp();
                    else
                        cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                } else if constexpr (kNumFolds == 2 and BLOCK_M == 32) {
                    if (epilogue_warp_idx == 1)
                        ptx::st_shared(&smem.fold_sumsq[lane_idx], row_sumsq.x + row_sumsq.y);
                    cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                    if (epilogue_warp_idx == 0) {
                        const float slice_sumsq = (row_sumsq.x + row_sumsq.y) + ptx::ld_shared(&smem.fold_sumsq[lane_idx]);
                        st_relaxed_gpu_f32(slice_partials + lane_idx, valid_row ? slice_sumsq : 0.0f);
                        __syncwarp();
                    }
                } else if constexpr (kNumFolds == 2 and BLOCK_M == 64) {
                    if (epilogue_thread_idx >= 64)
                        ptx::st_shared(&smem.fold_sumsq[local_row_idx], row_sumsq.x + row_sumsq.y);
                    cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                    if (epilogue_thread_idx < 64) {
                        const float slice_sumsq = (row_sumsq.x + row_sumsq.y) + ptx::ld_shared(&smem.fold_sumsq[local_row_idx]);
                        st_relaxed_gpu_f32(slice_partials + local_row_idx, valid_row ? slice_sumsq : 0.0f);
                    }
                    cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                } else {
                    ptx::st_shared(&smem.fold_sumsq[epilogue_thread_idx], row_sumsq.x + row_sumsq.y);
                    cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
                    if (epilogue_warp_idx == 0) {
                        const float slice_sumsq = (ptx::ld_shared(&smem.fold_sumsq[lane_idx]) + ptx::ld_shared(&smem.fold_sumsq[32 + lane_idx])) +
                                                  (ptx::ld_shared(&smem.fold_sumsq[64 + lane_idx]) + ptx::ld_shared(&smem.fold_sumsq[96 + lane_idx]));
                        st_relaxed_gpu_f32(slice_partials + lane_idx, valid_row ? slice_sumsq : 0.0f);
                        __syncwarp();
                    }
                }
                if (is_epi_leader_lane)
                    ptx::red_add_rel(counter_ptr, 1u);
                __syncwarp();

                const auto res_out_ptr = static_cast<cutlass::bfloat16_t*>(residual) +
                                         static_cast<uint64_t>(row_idx) * kShapeN + col_u4_base_idx * kNumElemsPerLoad;
                if constexpr (not kStoreResAtEnd) {
                    if (valid_row) {
                        #pragma unroll
                        for (uint32_t load_idx = 0; load_idx < kNumLoadsPerFold; ++ load_idx)
                            st_global_u4_16b_or_2b(res_out_ptr, load_idx * kNumElemsPerLoad, held[load_idx]);
                    }
                }
                if (is_epi_leader_lane)
                    wait_counter_at_least(counter_ptr, kNumNSlices);
                __syncwarp();
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);

                constexpr uint32_t kNumRowVecs = BLOCK_M / 4, kNumTotalVecs = kNumNSlices * kNumRowVecs;
                constexpr uint32_t kNumActiveWarps = (kNumTotalVecs < kNumEpilogueThreads ? kNumTotalVecs : kNumEpilogueThreads) / 32;
                constexpr uint32_t kNumVecLoads = (kNumTotalVecs + kNumEpilogueThreads - 1) / kNumEpilogueThreads;
                const auto block_partials_f4 = reinterpret_cast<const float4*>(block_partials);
                float4 thread_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (kNumTotalVecs >= kNumEpilogueThreads or epilogue_thread_idx < kNumTotalVecs) {
                    #pragma unroll
                    for (uint32_t v_iter_idx = 0; v_iter_idx < kNumVecLoads; ++ v_iter_idx) {
                        const float4 v = ld_relaxed_gpu_f32x4((block_partials_f4 + epilogue_thread_idx) + v_iter_idx * kNumEpilogueThreads);
                        thread_sum = make_float4(thread_sum.x + v.x, thread_sum.y + v.y, thread_sum.z + v.z, thread_sum.w + v.w);
                    }
                }
                const auto shfl_xor_add_f4 = [](float4& v, const int lane_mask) CUTLASS_LAMBDA_FUNC_INLINE {
                    v.x += __shfl_xor_sync(0xffffffff, v.x, lane_mask);
                    v.y += __shfl_xor_sync(0xffffffff, v.y, lane_mask);
                    v.z += __shfl_xor_sync(0xffffffff, v.z, lane_mask);
                    v.w += __shfl_xor_sync(0xffffffff, v.w, lane_mask);
                };
                if constexpr (kNumRowVecs <= 16)
                    shfl_xor_add_f4(thread_sum, 16);
                if constexpr (kNumRowVecs == 8)
                    shfl_xor_add_f4(thread_sum, 8);
                if (epilogue_warp_idx < kNumActiveWarps and lane_idx < kNumRowVecs) {
                    ptx::st_shared(&smem.fold_sumsq[epilogue_warp_idx * BLOCK_M + lane_idx * 4],
                                   __float_as_uint(thread_sum.x), __float_as_uint(thread_sum.y),
                                   __float_as_uint(thread_sum.z), __float_as_uint(thread_sum.w));
                }
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);

                if (valid_row) {
                    float total_sumsq = ptx::ld_shared(&smem.fold_sumsq[local_row_idx]) + ptx::ld_shared(&smem.fold_sumsq[BLOCK_M + local_row_idx]);
                    if constexpr (kNumActiveWarps == 4)
                        total_sumsq += (ptx::ld_shared(&smem.fold_sumsq[2 * BLOCK_M + local_row_idx]) + ptx::ld_shared(&smem.fold_sumsq[3 * BLOCK_M + local_row_idx]));
                    const float rstd = compute_rstd_from_sumsq<kShapeN>(total_sumsq);
                    const auto d_out_ptr = static_cast<cutlass::bfloat16_t*>(d) +
                                           static_cast<uint64_t>(row_idx) * kShapeN + col_u4_base_idx * kNumElemsPerLoad;
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < kNumLoadsPerFold; ++ load_idx) {
                        if constexpr (kStoreResAtEnd)
                            st_global_u4_16b_or_2b(res_out_ptr, load_idx * kNumElemsPerLoad, held[load_idx]);
                        st_global_u4_16b_or_2b(d_out_ptr, load_idx * kNumElemsPerLoad, scale_uint4_bf16_with_gamma(held[load_idx], gamma_packs[load_idx], rstd));
                    }
                }
                if (tile_idx + num_active_sms < num_tiles)
                    cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
            }
        }
    }

    __syncthreads();
    if (warp_idx == 2)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100");
#endif
}

// 2. N-Sliced 2-CTA Pipeline (`512t Control + RawAdd + Norm` or `384t Direct-TMEM Ping-Pong`)
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumAStages, uint32_t kNumCDPages,
          uint32_t kNumResPages = 0, uint32_t kNumBStages = kNumAStages,
          bool kUseDirectPingPong = false>
struct alignas(1024) SM100BF16GemmAddNormSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static constexpr bool kAsymmetricAB = kNumAStages != kNumBStages;
    static constexpr uint32_t kNumActualResPages = kNumResPages > 0 ? kNumResPages : kNumCDPages;
    static constexpr uint32_t kNumPayloadBytes =
        ((kNumCDPages + kNumResPages) * BLOCK_M * 64 +
         (kNumAStages * BLOCK_M + kNumBStages * BLOCK_N) * 64) * sizeof(cutlass::bfloat16_t);

    alignas(1024) cutlass::bfloat16_t cd[kNumCDPages + kNumResPages][BLOCK_M * 64];
    cutlass::bfloat16_t a[kNumAStages][BLOCK_M * 64];
    cutlass::bfloat16_t b[kNumBStages][BLOCK_N * 64];
    alignas(16) uint4 gamma_u4[128];
    Barrier a_full_barriers[kNumAStages];
    Barrier a_empty_barriers[kNumAStages];
    Barrier b_full_barriers[kAsymmetricAB ? kNumBStages : 1];
    Barrier b_empty_barriers[kAsymmetricAB ? kNumBStages : 1];
    Barrier residual_full_barriers[2][kNumActualResPages];
    cutlass::arch::ClusterBarrier residual_empty_barriers[kNumActualResPages];
    Barrier tmem_full_barriers[2];
    Barrier tmem_empty_barriers[2];
    uint32_t tmem_ptr;
    cutlass::arch::ClusterBarrier page_full_barriers[kUseDirectPingPong ? 1 : 2][kNumCDPages];
    cutlass::arch::ClusterBarrier page_empty_barriers[kNumCDPages];
    float fold_sumsq[BLOCK_M < 128 ? 64 : 1];
};

template <uint32_t kShapeK, bool kHasBias, int kNumSMs,
          uint32_t BLOCK_M = 128, uint32_t BLOCK_N = 128, uint32_t kNumStages = 6, uint32_t kNumCDPages = 2,
          bool kUseResTmaStore = false, uint32_t kNumResPages = 0,
          uint32_t kNumBStages = kNumStages, bool kUseDirectPingPong = false>
CUTLASS_DEVICE void sm100_bf16_gemm_add_norm_device(
    uint32_t shape_m, const cute::TmaDescriptor& tensor_map_a, const cute::TmaDescriptor& tensor_map_b,
    const cute::TmaDescriptor& tensor_map_residual, void* __restrict__ residual,
    const void* __restrict__ bias, const void* __restrict__ gamma, void* __restrict__ d,
    float* __restrict__ partials, uint32_t* __restrict__ counters,
    const cute::TmaDescriptor* tensor_map_d_ptr = nullptr) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Allocator = cute::TMEM::Allocator2Sm;
    constexpr uint32_t kNumAStages = kNumStages;
    constexpr bool kAsymmetricAB = kNumAStages != kNumBStages;
    using SharedStorage = SM100BF16GemmAddNormSharedStorage<
        BLOCK_M, BLOCK_N, kNumAStages, kNumCDPages, kNumResPages, kNumBStages, kUseDirectPingPong>;
    constexpr bool kHasDedicatedResIn = kNumResPages > 0;
    constexpr uint32_t kNumActiveResPages = kHasDedicatedResIn ? kNumResPages : kNumCDPages;

    constexpr uint32_t kShapeN = 1024, BLOCK_K = 64, kNumSwizzleBytes = 128;
    constexpr uint32_t kNumControlThreads = 128, kNumRawThreads = kUseDirectPingPong ? 0 : 128;
    constexpr uint32_t kNumNormThreads = 256, kNumNormGroupThreads = 128;
    constexpr int kNumControlRegisters = kUseDirectPingPong ? 56 : (BLOCK_M == 64 ? 48 : 40);
    constexpr int kNumRawRegisters = kUseDirectPingPong ? 0 : (BLOCK_M == 64 ? 120 : 112);
    constexpr int kNumNormRegisters = kUseDirectPingPong ? 224 : (BLOCK_M == 64 ? 168 : 176);
    constexpr uint32_t kRawWarpStartIdx = 4, kNormWarpStartIdx = kUseDirectPingPong ? 4 : 8, kNumKBlocks = kShapeK / BLOCK_K;
    constexpr uint32_t kNumNSlices = kShapeN / (2 * BLOCK_N), kNumMulticast = 2, kNumCounterSlots = 32;
    constexpr uint32_t UMMA_M = 2 * BLOCK_M, UMMA_N = 2 * BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = BLOCK_M, STORE_BLOCK_N = 64, kNumStores = UMMA_N / STORE_BLOCK_N;
    constexpr bool kDualFoldSingleStore = BLOCK_M == 64 and BLOCK_N == 32;
    constexpr uint32_t kNumFolds = kUseDirectPingPong ? 1u : (kNumRawThreads / BLOCK_M);
    constexpr uint32_t kNumStoresPerFold = kDualFoldSingleStore ? 1u : (kNumStores / kNumFolds);
    constexpr uint32_t kNumElemsPerLoad = 8, kNumLoadsPerStore = STORE_BLOCK_N / kNumElemsPerLoad;
    constexpr uint32_t kNumRawLoadsPerStep = kDualFoldSingleStore ? 4u : kNumLoadsPerStore;
    constexpr uint32_t kNumRawPageArrivals = kDualFoldSingleStore ? kNumRawThreads : BLOCK_M;
    constexpr uint32_t kNumAArrivalBytes = BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kNumBArrivalBytes = BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kNumArrivalBytes = kNumAArrivalBytes + kNumBArrivalBytes;
    constexpr uint32_t kNumTmemCols = 512, kNumTmemStageCols = 256;

    constexpr auto step_to_store_idx = [](const uint32_t step_idx) constexpr -> uint32_t {
        return (kNumFolds == 1 or kDualFoldSingleStore) ? step_idx : ((step_idx % kNumFolds) * kNumStoresPerFold + (step_idx / kNumFolds));
    };

    DG_STATIC_ASSERT((BLOCK_M == 64 or BLOCK_M == 128) and (BLOCK_N == 32 or BLOCK_N == 64 or BLOCK_N == 128), "Invalid block size");
    DG_STATIC_ASSERT(not kUseDirectPingPong or (not kHasBias and BLOCK_M == 128 and BLOCK_N == 128 and kNumCDPages == 2 and kUseResTmaStore and kNumResPages >= 1), "Invalid PingPong configuration");
    DG_STATIC_ASSERT(((kNumCDPages == 1 and (BLOCK_M == 128 or kDualFoldSingleStore)) or kNumCDPages == 2 or kNumCDPages == 4) and
                     kNumCDPages <= kNumStores and (kDualFoldSingleStore or kNumCDPages % kNumFolds == 0), "Invalid CD page count");
    DG_STATIC_ASSERT((kNumResPages == 0 or (kNumResPages == 1 and (BLOCK_M == 128 or kDualFoldSingleStore)) or kNumResPages == 2 or kNumResPages == 4) and
                     kNumActiveResPages <= kNumStores and (kDualFoldSingleStore or kNumActiveResPages % kNumFolds == 0), "Invalid residual page count");
    DG_STATIC_ASSERT(kNumAStages <= 32 and kNumBStages <= 32 and
                     sizeof(SharedStorage) == SharedStorage::kNumPayloadBytes + 3072 and
                     sizeof(SharedStorage) <= 232448, "Shared storage exceeded");
    DG_STATIC_ASSERT(kNumControlRegisters * kNumControlThreads + kNumRawRegisters * kNumRawThreads +
                     kNumNormRegisters * kNumNormThreads <= 64512, "Register budget exceeded");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const auto warp_idx = static_cast<uint32_t>(cutlass::canonical_warp_idx_sync());
    const auto lane_idx = ptx::get_lane_idx();
    const auto cta_rank_idx = static_cast<uint32_t>(cute::block_rank_in_cluster());
    const bool is_leader_cta = cta_rank_idx == 0;
    if constexpr (kUseDirectPingPong)
        prefetch_tma_descs_warp0(warp_idx, tensor_map_a, tensor_map_b, tensor_map_residual, *tensor_map_d_ptr);
    else
        prefetch_tma_descs_warp0(warp_idx, tensor_map_a, tensor_map_b, tensor_map_residual);

    if (warp_idx == 1) {
        init_barrier_pairs(smem.a_full_barriers, smem.a_empty_barriers, lane_idx, kNumMulticast, 1);
        init_barrier_pairs(smem.tmem_full_barriers, smem.tmem_empty_barriers, lane_idx, 1, kNumMulticast * kNumNormGroupThreads);
        fence_warp_barrier_init();
    } else if (warp_idx == 2) {
        init_barriers(smem.residual_full_barriers[0], lane_idx, 1);
        if constexpr (kUseDirectPingPong) {
            init_barriers(smem.residual_full_barriers[1], lane_idx, 1);
            init_barriers(smem.residual_empty_barriers, lane_idx, kNumNormGroupThreads);
        } else {
            if constexpr (kHasDedicatedResIn)
                init_barriers(smem.residual_empty_barriers, lane_idx, kNumRawPageArrivals);
            init_barriers(smem.page_empty_barriers, lane_idx, kNumNormGroupThreads);
            init_barrier_pairs(smem.page_full_barriers[0], smem.page_full_barriers[1], lane_idx, kNumRawPageArrivals, kNumRawPageArrivals);
        }
        fence_warp_barrier_init();
    } else if (kAsymmetricAB and warp_idx == 3) {
        init_barrier_pairs(smem.b_full_barriers, smem.b_empty_barriers, lane_idx, kNumMulticast, 1);
        fence_warp_barrier_init();
    }
    comm::cluster_sync_with_relaxed_arrive();

    const uint32_t num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
    init_counters_and_grid_sync(counters, math::ceil_div(num_m_blocks, 2u) * 2u, kNumCounterSlots);
    const uint32_t num_tiles = math::ceil_div(num_m_blocks, 2u) * kNumNSlices;
    DG_STATIC_ASSERT(kNumSMs > 0 and kNumSMs % kNumMulticast == 0, "Invalid SM count");
    const uint32_t start_tile_idx = blockIdx.x / kNumMulticast, num_active_clusters = gridDim.x / kNumMulticast;

    const auto advance_tile = [&](uint32_t& tile_idx) CUTLASS_LAMBDA_FUNC_INLINE {
        tile_idx += num_active_clusters;
        return tile_idx < num_tiles;
    };

    if (warp_idx < kRawWarpStartIdx) {
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
        __syncwarp();
        if (warp_idx == 0 and cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            constexpr uint64_t kAHint = static_cast<uint64_t>(
                (not kUseDirectPingPong) ? cute::TMA::CacheHintSm100::EVICT_LAST : cute::TMA::CacheHintSm100::EVICT_NORMAL);
            constexpr uint64_t kBHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST);
            const uint32_t rank_n_idx = cta_rank_idx * BLOCK_N;
            uint32_t a_stage_idx = 0, a_phase_idx = 0, tile_idx = start_tile_idx;
            if (tile_idx < num_tiles) {
                do {
                    const uint32_t m_idx = ((tile_idx / kNumNSlices) * 2 + cta_rank_idx) * BLOCK_M;
                    const uint32_t n_idx = (tile_idx % kNumNSlices) * UMMA_N + rank_n_idx;
                    for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                        smem.a_empty_barriers[a_stage_idx].wait(a_phase_idx ^ 1u);
                        const uint32_t k_idx = k_block_idx * BLOCK_K;
                        tma::copy<BLOCK_K, BLOCK_M, kNumSwizzleBytes, cutlass::bfloat16_t, false, kAHint>(
                            &tensor_map_a, &smem.a_full_barriers[a_stage_idx], smem.a[a_stage_idx], k_idx, m_idx, kNumMulticast);
                        if constexpr (not kAsymmetricAB) {
                            tma::copy<BLOCK_K, BLOCK_N, kNumSwizzleBytes, cutlass::bfloat16_t, false, kBHint>(
                                &tensor_map_b, &smem.a_full_barriers[a_stage_idx], smem.b[a_stage_idx], k_idx, n_idx, kNumMulticast);
                        }
                        if (is_leader_cta)
                            smem.a_full_barriers[a_stage_idx].arrive_and_expect_tx((kAsymmetricAB ? kNumAArrivalBytes : kNumArrivalBytes) * kNumMulticast);
                        else
                            smem.a_full_barriers[a_stage_idx].arrive(0u);
                        advance_stage<kNumAStages>(a_stage_idx, a_phase_idx);
                    }
                } while (advance_tile(tile_idx));
            }
        } else if (warp_idx == 1) {
            allocate_tmem_checked<Allocator, kNumTmemCols>(&smem.tmem_ptr);
            if (is_leader_cta) {
                run_umma_warp1_pipeline<ptx::SM100_MMA_F16BF16_2x1SM_SS, UMMA_M, UMMA_N, BLOCK_M, BLOCK_N,
                                        BLOCK_K, kNumAStages, kNumBStages, true>(
                    smem, lane_idx, start_tile_idx, num_tiles, 0u, kNumKBlocks, advance_tile);
            }
        } else if (warp_idx == 2 and cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            constexpr uint32_t kResPageBaseIdx = kHasDedicatedResIn ? kNumCDPages : 0;
            uint32_t page_phase_idx = 0, iter_idx = 0, tile_idx = start_tile_idx;
            if (tile_idx < num_tiles) {
                if constexpr (BLOCK_N != 64 or kShapeK > 1024)
                    smem.a_empty_barriers[1].wait(0);
                do {
                    const uint32_t group_idx = kUseDirectPingPong ? (iter_idx & 1u) : 0u;
                    const uint32_t m_idx = ((tile_idx / kNumNSlices) * 2 + cta_rank_idx) * BLOCK_M;
                    const uint32_t base_n_idx = (tile_idx % kNumNSlices) * UMMA_N;
                    #pragma unroll
                    for (uint32_t step_idx = 0; step_idx < kNumStores; ++ step_idx) {
                        const uint32_t store_idx = step_to_store_idx(step_idx);
                        const uint32_t res_page_idx = step_idx % kNumActiveResPages;
                        if constexpr (kUseDirectPingPong)
                            smem.residual_empty_barriers[res_page_idx].wait(((step_idx / kNumActiveResPages) & 1u) ^ 1u);
                        else if constexpr (kHasDedicatedResIn)
                            smem.residual_empty_barriers[res_page_idx].wait(page_phase_idx ^ 1u);
                        else
                            smem.page_empty_barriers[res_page_idx].wait(page_phase_idx ^ 1u);
                        if (res_page_idx + 1 == kNumActiveResPages)
                            page_phase_idx ^= 1u;
                        tma::copy<STORE_BLOCK_N, STORE_BLOCK_M, kNumSwizzleBytes, cutlass::bfloat16_t>(
                            &tensor_map_residual, &smem.residual_full_barriers[group_idx][res_page_idx],
                            smem.cd[kResPageBaseIdx + res_page_idx], base_n_idx + store_idx * STORE_BLOCK_N, m_idx, 1);
                        smem.residual_full_barriers[group_idx][res_page_idx].arrive_and_expect_tx(sizeof(smem.cd[0]));
                    }
                    ++ iter_idx;
                } while (advance_tile(tile_idx));
            }
        } else if (kAsymmetricAB and warp_idx == 3 and cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            constexpr uint64_t kBHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST);
            const uint32_t rank_n_idx = cta_rank_idx * BLOCK_N;
            uint32_t b_stage_idx = 0, b_phase_idx = 0, tile_idx = start_tile_idx;
            if (tile_idx < num_tiles) {
                do {
                    const uint32_t n_idx = (tile_idx % kNumNSlices) * UMMA_N + rank_n_idx;
                    for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                        smem.b_empty_barriers[b_stage_idx].wait(b_phase_idx ^ 1u);
                        tma::copy<BLOCK_K, BLOCK_N, kNumSwizzleBytes, cutlass::bfloat16_t, false, kBHint>(
                            &tensor_map_b, &smem.b_full_barriers[b_stage_idx], smem.b[b_stage_idx],
                            k_block_idx * BLOCK_K, n_idx, kNumMulticast);
                        if (is_leader_cta)
                            smem.b_full_barriers[b_stage_idx].arrive_and_expect_tx(kNumBArrivalBytes * kNumMulticast);
                        else
                            smem.b_full_barriers[b_stage_idx].arrive(0u);
                        advance_stage<kNumBStages>(b_stage_idx, b_phase_idx);
                    }
                } while (advance_tile(tile_idx));
            }
        }
        __syncwarp();
    } else if constexpr (not kUseDirectPingPong) {
        if (threadIdx.x < kNormWarpStartIdx * 32) {
            cutlass::arch::warpgroup_reg_dealloc<kNumRawRegisters>();
            __syncwarp();
            cudaGridDependencySynchronize();

            const uint32_t raw_warp_idx = static_cast<uint32_t>(cutlass::canonical_warp_idx_sync()) - kRawWarpStartIdx, raw_thread_idx = raw_warp_idx * 32 + lane_idx;
            const uint32_t fold_idx = raw_thread_idx / BLOCK_M, local_row_idx = raw_thread_idx % BLOCK_M;
            constexpr uint32_t kNumBiasQuads = kDualFoldSingleStore ? 1u : 4u;
            uint4 bias_regs[kNumBiasQuads];
            if constexpr (kHasBias) {
                #pragma unroll
                for (uint32_t quad_idx = 0; quad_idx < kNumBiasQuads; ++ quad_idx)
                    bias_regs[quad_idx] = ld_global_u4_16b_or_2b(bias, kDualFoldSingleStore ? ((start_tile_idx * 8u + lane_idx) & 127u) : lane_idx, quad_idx * 32);
            }

            uint32_t iter_idx = 0, tile_idx = start_tile_idx;
            if (tile_idx < num_tiles) {
                do {
                    const uint32_t cur_tile_idx = tile_idx;
                    const uint32_t accum_stage_idx = iter_idx & 1u, norm_group_idx = iter_idx & 1u;
                    smem.tmem_full_barriers[accum_stage_idx].wait((iter_idx >> 1u) & 1u);
                    ptx::tcgen05_after_thread_sync();

                    const uint32_t base_n_idx = (cur_tile_idx % kNumNSlices) * UMMA_N;
                    const uint32_t bias_quad_idx = base_n_idx / 256, bias_base_lane_idx = (kDualFoldSingleStore or BLOCK_N == 128) ? 0u : ((base_n_idx / 8) & 31u);
                    const uint4 slice_bias_reg = kDualFoldSingleStore ? bias_regs[0] : (bias_quad_idx == 0 ? bias_regs[0] : (bias_quad_idx == 1 ? bias_regs[1] : (bias_quad_idx == 2 ? bias_regs[2] : bias_regs[3])));
                    float2 row_sumsq = make_float2(0.0f, 0.0f);
                    uint32_t values[2][kNumElemsPerLoad];

                    #pragma unroll
                    for (uint32_t local_store_idx = 0; local_store_idx < kNumStoresPerFold; ++ local_store_idx) {
                        const uint32_t step_idx = (kNumFolds == 1 or kDualFoldSingleStore) ? local_store_idx : (local_store_idx * kNumFolds + fold_idx);
                        const uint32_t store_idx = (kNumFolds == 1 or kDualFoldSingleStore) ? local_store_idx : (fold_idx * kNumStoresPerFold + local_store_idx);
                        const uint32_t page_idx = step_idx % kNumCDPages, res_page_idx = step_idx % kNumActiveResPages;
                        const uint32_t cd_wait_phase_idx = ((iter_idx * (kNumStores / kNumCDPages)) + (step_idx / kNumCDPages)) & 1u;
                        const uint32_t res_wait_phase_idx = ((iter_idx * (kNumStores / kNumActiveResPages)) + (step_idx / kNumActiveResPages)) & 1u;
                        const uint32_t raw_vec_base_idx = kDualFoldSingleStore ? (fold_idx * 4u) : 0u;
                        constexpr uint32_t kResPageBaseIdx = kHasDedicatedResIn ? kNumCDPages : 0;

                        uint4 residual_page[kNumRawLoadsPerStep];
                        const uint32_t store_tmem_col_idx = accum_stage_idx * kNumTmemStageCols + local_store_idx * STORE_BLOCK_N;
                        if (local_store_idx == 0)
                            ptx::tmem_load_32dp32b<kNumElemsPerLoad>(store_tmem_col_idx, values[0]);
                        smem.residual_full_barriers[0][res_page_idx].wait(res_wait_phase_idx);
                        #pragma unroll
                        for (uint32_t preload_idx = 0; preload_idx < kNumRawLoadsPerStep; ++ preload_idx)
                            residual_page[preload_idx] = ld_swizzled_u4(smem.cd[kResPageBaseIdx + res_page_idx], local_row_idx, raw_vec_base_idx + preload_idx);
                        if constexpr (kHasDedicatedResIn) {
                            cutlass::arch::fence_view_async_shared();
                            smem.residual_empty_barriers[res_page_idx].arrive();
                            smem.page_empty_barriers[page_idx].wait(cd_wait_phase_idx ^ 1u);
                        }

                        #pragma unroll
                        for (uint32_t load_idx = 0; load_idx < kNumRawLoadsPerStep; ++ load_idx) {
                            cutlass::arch::fence_view_async_tmem_load();
                            if (local_store_idx + 1 == kNumStoresPerFold and load_idx + 1 == kNumRawLoadsPerStep) {
                                ptx::tcgen05_before_thread_sync();
                                smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                            }
                            if (load_idx + 1 < kNumRawLoadsPerStep)
                                ptx::tmem_load_32dp32b<kNumElemsPerLoad>(store_tmem_col_idx + (load_idx + 1) * kNumElemsPerLoad, values[(load_idx + 1) & 1]);
                            else if (local_store_idx + 1 < kNumStoresPerFold)
                                ptx::tmem_load_32dp32b<kNumElemsPerLoad>(store_tmem_col_idx + STORE_BLOCK_N, values[0]);
                            uint4 bias_pack = make_uint4(0, 0, 0, 0);
                            if constexpr (kHasBias)
                                bias_pack = shfl_u4_sync(slice_bias_reg, static_cast<int>((bias_base_lane_idx + store_idx * kNumLoadsPerStore + raw_vec_base_idx + load_idx) & 31u));
                            st_swizzled_u4(smem.cd[page_idx], local_row_idx, raw_vec_base_idx + load_idx,
                                           accumulate_bias_residual_u4<kHasBias>(
                                               reinterpret_cast<const float2*>(values[load_idx & 1]), bias_pack, residual_page[load_idx], row_sumsq));
                        }
                        smem.page_full_barriers[norm_group_idx][page_idx].arrive();
                    }

                    uint32_t post_tile_idx = cur_tile_idx;
                    asm volatile("" : "+r"(post_tile_idx));
                    const uint32_t m_block_idx = (post_tile_idx / kNumNSlices) * 2 + cta_rank_idx, slice_idx = post_tile_idx % kNumNSlices;
                    const uint32_t row_idx = m_block_idx * BLOCK_M + local_row_idx;
                    const auto slice_partials = partials + (static_cast<uint64_t>(m_block_idx) * kNumNSlices + slice_idx) * BLOCK_M + local_row_idx;
                    const float my_sumsq = row_sumsq.x + row_sumsq.y;
                    if constexpr (kNumFolds > 1) {
                        if (fold_idx == 1)
                            smem.fold_sumsq[local_row_idx] = my_sumsq;
                        cutlass::arch::NamedBarrier::sync(kNumRawThreads, 0);
                        if (fold_idx == 0 and row_idx < shape_m)
                            st_relaxed_gpu_f32(slice_partials, my_sumsq + smem.fold_sumsq[local_row_idx]);
                    } else if (row_idx < shape_m) {
                        st_relaxed_gpu_f32(slice_partials, my_sumsq);
                    }
                    cutlass::arch::NamedBarrier::sync(kNumRawThreads, 0);
                    if (raw_warp_idx == 0 and cute::elect_one_sync())
                        ptx::red_add_rel(counters + m_block_idx * kNumCounterSlots, 1u);
                    __syncwarp();
                    ++ iter_idx;
                } while (advance_tile(tile_idx));
            }
        } else {
            cutlass::arch::warpgroup_reg_alloc<kNumNormRegisters>();
            __syncwarp();
            cudaGridDependencySynchronize();
            const uint32_t norm_warp_idx = static_cast<uint32_t>(cutlass::canonical_warp_idx_sync()) - kNormWarpStartIdx, norm_thread_idx = norm_warp_idx * 32 + lane_idx;
            const uint32_t norm_group_idx = norm_warp_idx / 4, group_warp_idx = norm_warp_idx % 4;
            constexpr uint32_t kNormLeaderWarpIdx = kAsymmetricAB ? 0u : 3u;
            const bool is_norm_leader_lane = group_warp_idx == kNormLeaderWarpIdx and cute::elect_one_sync();
            __syncwarp();

            if (norm_thread_idx < 128)
                smem.gamma_u4[norm_thread_idx] = ld_global_u4_16b_or_2b(gamma, norm_thread_idx);
            cutlass::arch::NamedBarrier::sync(kNumNormThreads, 3);

            constexpr uint32_t kNumRowsPerWarp = BLOCK_M / 4, kNumVecsPerRow = 4, kNumRowsPerBatch = 32 / kNumVecsPerRow;
            constexpr uint32_t kNumBatchesPerPage = kNumRowsPerWarp / kNumRowsPerBatch, kNumBatchesPerStage = 2;
            const uint32_t lane_vec_idx = lane_idx % kNumVecsPerRow, lane_row_in_batch_idx = lane_idx / kNumVecsPerRow;

            uint4 held[kNumStores][kNumBatchesPerPage][2];
            uint32_t group_page_phase_idx = 0;
            uint32_t tile_idx = start_tile_idx + norm_group_idx * num_active_clusters;

            if (tile_idx < num_tiles) {
                uint32_t iter_idx = 0;
                do {
                    const uint32_t cur_tile_idx = tile_idx;
                    const uint32_t m_block_idx = (cur_tile_idx / kNumNSlices) * 2 + cta_rank_idx;
                    const uint32_t m_idx = m_block_idx * BLOCK_M, base_n_idx = (cur_tile_idx % kNumNSlices) * UMMA_N;
                    const auto store_held_batches = [&](cutlass::bfloat16_t* dst_ptr, const bool aligned_32b,
                                                        const uint32_t store_idx, const uint32_t begin_batch_idx,
                                                        const uint32_t end_batch_idx) CUTLASS_LAMBDA_FUNC_INLINE {
                        const uint32_t col_idx = base_n_idx + store_idx * STORE_BLOCK_N + lane_vec_idx * 16;
                        #pragma unroll
                        for (uint32_t batch_idx = begin_batch_idx; batch_idx < end_batch_idx; ++ batch_idx) {
                            const uint32_t row_idx = m_idx + group_warp_idx * kNumRowsPerWarp + batch_idx * kNumRowsPerBatch + lane_row_in_batch_idx;
                            if (row_idx < shape_m)
                                st_global_cs_256b(dst_ptr + static_cast<uint64_t>(row_idx) * kShapeN + col_idx, held[store_idx][batch_idx][0], held[store_idx][batch_idx][1], aligned_32b);
                        }
                    };

                    #pragma unroll
                    for (uint32_t step_idx = 0; step_idx < kNumStores; ++ step_idx) {
                        const uint32_t store_idx = step_to_store_idx(step_idx);
                        const uint32_t page_idx = step_idx % kNumCDPages;
                        smem.page_full_barriers[norm_group_idx][page_idx].wait(group_page_phase_idx);
                        if (page_idx + 1 == kNumCDPages)
                            group_page_phase_idx ^= 1u;
                        if constexpr (kUseResTmaStore) {
                            if (is_norm_leader_lane)
                                issue_fenced_tma_store_2d_pages<1>(tensor_map_residual, &smem.cd[page_idx], base_n_idx + store_idx * STORE_BLOCK_N, m_idx, STORE_BLOCK_N);
                        }
                        #pragma unroll
                        for (uint32_t batch_idx = 0; batch_idx < kNumBatchesPerPage; ++ batch_idx) {
                            const uint32_t local_row_idx = group_warp_idx * kNumRowsPerWarp + batch_idx * kNumRowsPerBatch + lane_row_in_batch_idx;
                            held[store_idx][batch_idx][0] = ld_swizzled_u4(smem.cd[page_idx], local_row_idx, lane_vec_idx * 2);
                            held[store_idx][batch_idx][1] = ld_swizzled_u4(smem.cd[page_idx], local_row_idx, lane_vec_idx * 2 + 1);
                        }
                        cutlass::arch::fence_view_async_shared();
                        if (kUseResTmaStore and is_norm_leader_lane)
                            cute::tma_store_wait<0>();
                        smem.page_empty_barriers[page_idx].arrive();
                    }

                    constexpr uint32_t kPrefetchSplitStoreIdx = (kUseResTmaStore or BLOCK_N < 128)
                        ? kNumStores : (BLOCK_M == 128 ? 3u : 2u);
                    const auto residual_ptr = static_cast<cutlass::bfloat16_t*>(residual);
                    const bool res_aligned_32b = (reinterpret_cast<uintptr_t>(residual_ptr) & 31u) == 0u;
                    if constexpr (not kUseResTmaStore) {
                        #pragma unroll
                        for (uint32_t store_idx = 0; store_idx < kPrefetchSplitStoreIdx; ++ store_idx)
                            store_held_batches(residual_ptr, res_aligned_32b, store_idx, 0, kNumBatchesPerPage);
                    }

                    constexpr bool kPreloadGamma0 = BLOCK_M == 128 or kUseResTmaStore or BLOCK_N <= 64;
                    const auto load_gamma_pair = [&](const uint32_t store_idx, uint4& g0, uint4& g1) CUTLASS_LAMBDA_FUNC_INLINE {
                        const uint32_t g_idx = (base_n_idx / kNumElemsPerLoad) + store_idx * 8 + lane_vec_idx * 2;
                        g0 = ptx::ld_shared(&smem.gamma_u4[g_idx]);
                        g1 = ptx::ld_shared(&smem.gamma_u4[g_idx + 1]);
                    };
                    uint4 g_pack_0, g_pack_1;
                    if constexpr (kPreloadGamma0)
                        load_gamma_pair(0, g_pack_0, g_pack_1);

                    const auto counter_ptr = counters + m_block_idx * kNumCounterSlots;
                    if (is_norm_leader_lane)
                        wait_counter_at_least(counter_ptr, kNumNSlices);
                    __syncwarp();
                    cutlass::arch::NamedBarrier::sync(kNumNormGroupThreads, norm_group_idx + 1);

                    const uint32_t local_owner_row_idx = group_warp_idx * kNumRowsPerWarp + lane_idx;
                    const bool has_owner_row = lane_idx < kNumRowsPerWarp and (m_idx + local_owner_row_idx) < shape_m;
                    DG_STATIC_ASSERT(kNumNSlices % 4 == 0, "Invalid slice count");
                    float partial_vals[kNumNSlices];
                    if (has_owner_row) {
                        const auto block_partials = partials + static_cast<uint64_t>(m_block_idx) * (kNumNSlices * BLOCK_M) + local_owner_row_idx;
                        #pragma unroll
                        for (uint32_t s_idx = 0; s_idx < kNumNSlices; ++ s_idx)
                            partial_vals[s_idx] = ld_relaxed_gpu_f32(block_partials + s_idx * BLOCK_M);
                    }

                    if constexpr (not kUseResTmaStore and kPrefetchSplitStoreIdx < kNumStores) {
                        #pragma unroll
                        for (uint32_t store_idx = kPrefetchSplitStoreIdx; store_idx < kNumStores; ++ store_idx)
                            store_held_batches(residual_ptr, res_aligned_32b, store_idx, 0, kNumBatchesPerPage);
                    }

                    float rstd = 0.0f;
                    if (has_owner_row) {
                        float sumsq = 0.0f;
                        #pragma unroll
                        for (uint32_t s_idx = 0; s_idx < kNumNSlices; s_idx += 4)
                            sumsq = ((sumsq + partial_vals[s_idx]) + partial_vals[s_idx + 1]) + (partial_vals[s_idx + 2] + partial_vals[s_idx + 3]);
                        rstd = compute_rstd_from_sumsq<kShapeN>(sumsq);
                    }
                    if constexpr (not kPreloadGamma0)
                        load_gamma_pair(0, g_pack_0, g_pack_1);
                    __syncwarp();

                    float batch_rstd[kNumBatchesPerPage];
                    #pragma unroll
                    for (uint32_t batch_idx = 0; batch_idx < kNumBatchesPerPage; ++ batch_idx)
                        batch_rstd[batch_idx] = __shfl_sync(0xffffffff, rstd, static_cast<int>(batch_idx * kNumRowsPerBatch + lane_row_in_batch_idx));

                    const auto d_ptr = static_cast<cutlass::bfloat16_t*>(d);
                    const bool d_aligned_32b = (reinterpret_cast<uintptr_t>(d_ptr) & 31u) == 0u;
                    #pragma unroll
                    for (uint32_t store_idx = 0; store_idx < kNumStores; ++ store_idx) {
                        #pragma unroll
                        for (uint32_t batch_base_idx = 0; batch_base_idx < kNumBatchesPerPage; batch_base_idx += kNumBatchesPerStage) {
                            #pragma unroll
                            for (uint32_t stage_batch_idx = 0; stage_batch_idx < kNumBatchesPerStage; ++ stage_batch_idx) {
                                const uint32_t batch_idx = batch_base_idx + stage_batch_idx;
                                held[store_idx][batch_idx][0] = scale_uint4_bf16_with_gamma(held[store_idx][batch_idx][0], g_pack_0, batch_rstd[batch_idx]);
                                held[store_idx][batch_idx][1] = scale_uint4_bf16_with_gamma(held[store_idx][batch_idx][1], g_pack_1, batch_rstd[batch_idx]);
                            }
                            if (batch_base_idx + kNumBatchesPerStage == kNumBatchesPerPage and store_idx + 1 < kNumStores)
                                load_gamma_pair(store_idx + 1, g_pack_0, g_pack_1);
                            store_held_batches(d_ptr, d_aligned_32b, store_idx, batch_base_idx, batch_base_idx + kNumBatchesPerStage);
                        }
                    }
                    ++ iter_idx;
                    tile_idx += 2 * num_active_clusters;
                } while (tile_idx < num_tiles);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumNormRegisters>();
        __syncwarp();
        cudaGridDependencySynchronize();

        const auto& tensor_map_d = *tensor_map_d_ptr;
        const uint32_t epi_warp_idx = warp_idx - kNormWarpStartIdx, epi_thread_idx = epi_warp_idx * 32 + lane_idx;
        const uint32_t group_idx = epi_warp_idx / 4, group_warp_idx = epi_warp_idx % 4;
        const uint32_t local_row_idx = group_warp_idx * 32 + lane_idx;
        constexpr uint32_t kLeaderWarpIdx = kAsymmetricAB ? 0u : 3u;
        const bool is_group_leader = group_warp_idx == kLeaderWarpIdx and cute::elect_one_sync();
        __syncwarp();

        if (epi_thread_idx < 128)
            smem.gamma_u4[epi_thread_idx] = ld_global_u4_16b_or_2b(gamma, epi_thread_idx);
        cutlass::arch::NamedBarrier::sync(kNumNormThreads, 3);

        uint4 held[kNumStores][kNumLoadsPerStore];
        uint32_t group_iter_idx = 0;
        uint32_t tile_idx = start_tile_idx + group_idx * num_active_clusters;

        if (tile_idx < num_tiles) {
            uint32_t iter_idx = 0;
            do {
                uint32_t is_my_turn;
                asm("and.b32 %0, %1, 2;" : "=r"(is_my_turn) : "r"(group_idx));
                const uint32_t cur_tile_idx = tile_idx;
                if (is_my_turn == 0u) {
                    #pragma unroll
                    for (uint32_t pre_idx = 0; pre_idx < kNumActiveResPages; ++ pre_idx) {
                        wait_smem_mbarrier(&smem.residual_full_barriers[group_idx][pre_idx], 0u);
                        #pragma unroll
                        for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                            held[pre_idx][load_idx] = ld_swizzled_u4(smem.cd[kNumCDPages + pre_idx], local_row_idx, load_idx);
                        cutlass::arch::fence_view_async_shared();
                        smem.residual_empty_barriers[pre_idx].arrive();
                    }
                }
                if (is_my_turn == 0u) {
                    wait_smem_mbarrier(&smem.tmem_full_barriers[group_idx], group_iter_idx & 1u);
                    ptx::tcgen05_after_thread_sync();

                    const uint32_t group_tmem_col_idx = group_idx * kNumTmemStageCols;
                    uint32_t values[2][kNumElemsPerLoad];
                    ptx::tmem_load_32dp32b<kNumElemsPerLoad>(group_tmem_col_idx, values[0]);
                    const uint32_t m_block_idx = (cur_tile_idx / kNumNSlices) * 2 + cta_rank_idx, slice_idx = cur_tile_idx % kNumNSlices;
                    const uint32_t m_idx = m_block_idx * BLOCK_M, row_idx = m_idx + local_row_idx, base_n_idx = slice_idx * UMMA_N;
                    const auto counter_ptr = counters + m_block_idx * kNumCounterSlots;
                    float2 row_sumsq = make_float2(0.0f, 0.0f);

                    #pragma unroll
                    for (uint32_t store_idx = 0; store_idx < kNumStores; ++ store_idx) {
                        const uint32_t store_tmem_col_idx = group_tmem_col_idx + store_idx * STORE_BLOCK_N;
                        const uint32_t res_page_idx = store_idx % kNumActiveResPages;
                        if (store_idx >= kNumActiveResPages) {
                            wait_smem_mbarrier(&smem.residual_full_barriers[group_idx][res_page_idx], (store_idx / kNumActiveResPages) & 1u);
                            if (store_idx + 1 < kNumStores) {
                                #pragma unroll
                                for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                                    held[store_idx][load_idx] = ld_swizzled_u4(smem.cd[kNumCDPages + res_page_idx], local_row_idx, load_idx);
                                cutlass::arch::fence_view_async_shared();
                                smem.residual_empty_barriers[res_page_idx].arrive();
                            }
                        }
                        #pragma unroll
                        for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx) {
                            const uint4 res_u4 = (store_idx + 1 == kNumStores)
                                ? ld_swizzled_u4(smem.cd[kNumCDPages + res_page_idx], local_row_idx, load_idx)
                                : held[store_idx][load_idx];
                            if (store_idx + 1 == kNumStores and load_idx + 1 == kNumLoadsPerStore) {
                                cutlass::arch::fence_view_async_shared();
                                smem.residual_empty_barriers[res_page_idx].arrive();
                            }
                            cutlass::arch::fence_view_async_tmem_load();
                            if (store_idx + 1 == kNumStores and load_idx + 1 == kNumLoadsPerStore) {
                                ptx::tcgen05_before_thread_sync();
                                smem.tmem_empty_barriers[group_idx].arrive(0u);
                            }
                            if (load_idx + 1 < kNumLoadsPerStore)
                                ptx::tmem_load_32dp32b<kNumElemsPerLoad>(store_tmem_col_idx + (load_idx + 1) * kNumElemsPerLoad, values[(load_idx + 1) & 1]);
                            else if (store_idx + 1 < kNumStores)
                                ptx::tmem_load_32dp32b<kNumElemsPerLoad>(store_tmem_col_idx + STORE_BLOCK_N, values[0]);
                            held[store_idx][load_idx] = accumulate_bias_residual_u4<false>(
                                reinterpret_cast<const float2*>(values[load_idx & 1]), make_uint4(0, 0, 0, 0), res_u4, row_sumsq);
                            asm volatile("" : "+r"(held[store_idx][load_idx].x), "+r"(held[store_idx][load_idx].y),
                                              "+r"(held[store_idx][load_idx].z), "+r"(held[store_idx][load_idx].w),
                                              "+f"(row_sumsq.x), "+f"(row_sumsq.y));
                        }
                        if (store_idx + 1 == kNumStores and row_idx < shape_m)
                            st_relaxed_gpu_f32(partials + (static_cast<uint64_t>(m_block_idx) * kNumNSlices + slice_idx) * BLOCK_M + local_row_idx, row_sumsq.x + row_sumsq.y);
                        if (store_idx > 0 or group_iter_idx > 0) {
                            if (is_group_leader)
                                cute::tma_store_wait<0>();
                            sync_epi_group_barrier(group_idx);
                        }
                        if (store_idx + 1 == kNumStores and is_group_leader)
                            ptx::red_add_rel(counter_ptr, 1u);
                        #pragma unroll
                        for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                            st_swizzled_u4(smem.cd[group_idx], local_row_idx, load_idx, held[store_idx][load_idx]);
                        sync_epi_group_barrier(group_idx);
                        if (is_group_leader) {
                            issue_fenced_tma_store_2d_pages<1>(tensor_map_residual, &smem.cd[group_idx], base_n_idx + store_idx * STORE_BLOCK_N, m_idx, STORE_BLOCK_N);
                            if (store_idx + 1 == kNumStores)
                                wait_counter_at_least(counter_ptr, kNumNSlices);
                        }
                        __syncwarp();
                    }
                    sync_epi_group_barrier(group_idx);

                    DG_STATIC_ASSERT(kNumNSlices % 4 == 0, "Invalid slice count");
                    float rstd = 0.0f;
                    if (row_idx < shape_m) {
                        const auto block_partials = partials + static_cast<uint64_t>(m_block_idx) * (kNumNSlices * BLOCK_M) + local_row_idx;
                        float partial_vals[kNumNSlices];
                        #pragma unroll
                        for (uint32_t s_idx = 0; s_idx < kNumNSlices; ++ s_idx)
                            partial_vals[s_idx] = ld_relaxed_gpu_f32(block_partials + s_idx * BLOCK_M);
                        float sumsq = 0.0f;
                        #pragma unroll
                        for (uint32_t s_idx = 0; s_idx < kNumNSlices; s_idx += 4)
                            sumsq = ((sumsq + partial_vals[s_idx]) + partial_vals[s_idx + 1]) + (partial_vals[s_idx + 2] + partial_vals[s_idx + 3]);
                        rstd = compute_rstd_from_sumsq<kShapeN>(sumsq);
                    }

                    #pragma unroll
                    for (uint32_t store_idx = 0; store_idx < kNumStores; ++ store_idx) {
                        #pragma unroll
                        for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx) {
                            const uint4 gamma_pack = ld_smem_u4(cute::cast_smem_ptr_to_uint(smem.gamma_u4) + ((base_n_idx / kNumElemsPerLoad) + store_idx * kNumLoadsPerStore + load_idx) * 16u);
                            held[store_idx][load_idx] = scale_uint4_bf16_with_gamma(held[store_idx][load_idx], gamma_pack, rstd);
                        }
                        if (is_group_leader)
                            cute::tma_store_wait<0>();
                        sync_epi_group_barrier(group_idx);
                        #pragma unroll
                        for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                            st_swizzled_u4(smem.cd[group_idx], local_row_idx, load_idx, held[store_idx][load_idx]);
                        sync_epi_group_barrier(group_idx);
                        if (is_group_leader)
                            issue_fenced_tma_store_2d_pages<1>(tensor_map_d, &smem.cd[group_idx], base_n_idx + store_idx * STORE_BLOCK_N, m_idx, STORE_BLOCK_N);
                        __syncwarp();
                    }
                    ++ group_iter_idx;
                }
                ++ iter_idx;
                tile_idx += 2 * num_active_clusters;
            } while (tile_idx < num_tiles);
        }
    }

    comm::cluster_sync_with_relaxed_arrive();
    if ((threadIdx.x / 32) == 1)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100");
#endif
}

// 3. Whole-Row Single-K-Pass 2-CTA Cluster (`BLOCK_M x 1024`, Zero Scratch)
template <uint32_t BLOCK_M, uint32_t kNumCDPagesPerGroup = 2, uint32_t kNumAStages = 4,
          uint32_t kNumBStages = 3, uint32_t kNumSharedResPages = 0>
struct alignas(1024) SM100BF16GemmAddNormWholeRowSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static constexpr uint32_t UMMA_N = 256, BLOCK_K = 64, kNumSlicesPerWarp = 128 / BLOCK_M;
    static constexpr uint32_t kNumFlatPages = 2 * kNumCDPagesPerGroup + kNumSharedResPages;
    static constexpr uint32_t kNumResBarriersPerGroup = kNumSharedResPages > 0 ? 1 : kNumCDPagesPerGroup;

    alignas(1024) cutlass::bfloat16_t cd_flat[kNumFlatPages][kNumSlicesPerWarp][BLOCK_M * 64];
    alignas(1024) cutlass::bfloat16_t a[kNumAStages][BLOCK_M * BLOCK_K];
    alignas(1024) cutlass::bfloat16_t b[kNumBStages][UMMA_N * BLOCK_K];
    alignas(16) float group_sumsq[2][BLOCK_M];
    alignas(16) float peer_cta_sumsq[2][BLOCK_M];
    Barrier a_full_barriers[kNumAStages];
    Barrier a_empty_barriers[kNumAStages];
    Barrier b_full_barriers[kNumBStages];
    Barrier b_empty_barriers[kNumBStages];
    Barrier res_full_barriers[2][kNumResBarriersPerGroup];
    cutlass::arch::ClusterBarrier cd_empty_barriers[2][kNumResBarriersPerGroup];
    cutlass::arch::ClusterBarrier shared_res_empty_barrier;
    Barrier tmem_full_barriers[2];
    cutlass::arch::ClusterBarrier tmem_empty_barriers[2];
    Barrier peer_sumsq_full_barriers[2];
    cutlass::arch::ClusterBarrier peer_sumsq_empty_barriers[2];
    uint32_t tmem_ptr;
};

template <uint32_t kShapeK, bool kHasBias, int kNumSMs, uint32_t BLOCK_M,
          uint32_t kNumCDPagesPerGroup = 2, uint32_t kNumAStages = 4, uint32_t kNumBStages = 3,
          bool kUnalignedIO = false, uint32_t kNumSharedResPages = 0>
CUTLASS_DEVICE void sm100_bf16_gemm_add_norm_whole_row_device(
    uint32_t shape_m, const cute::TmaDescriptor& tensor_map_a, const cute::TmaDescriptor& tensor_map_b,
    const cute::TmaDescriptor& tensor_map_residual, const cute::TmaDescriptor& tensor_map_d,
    void* __restrict__ residual, const void* __restrict__ bias, const void* __restrict__ gamma, void* __restrict__ d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Allocator = cute::TMEM::Allocator1Sm;
    using SharedStorage = SM100BF16GemmAddNormWholeRowSharedStorage<
        BLOCK_M, kNumCDPagesPerGroup, kNumAStages, kNumBStages, kNumSharedResPages>;
    constexpr bool kHasSharedResIn = kNumSharedResPages > 0;
    constexpr uint32_t kSharedResPageIdx = 2 * kNumCDPagesPerGroup;

    constexpr uint32_t kShapeN = 1024, kNumSwizzleBytes = 128;
    constexpr uint32_t UMMA_M = BLOCK_M, UMMA_N = 256, BLOCK_K = 64, BLOCK_ATOM_K = 64;
    constexpr uint32_t kNumKBlocks = kShapeK / BLOCK_K, kNumLocalSlices = 2, kNumMulticast = 2;
    constexpr uint32_t kNumCtaCols = kNumLocalSlices * UMMA_N;
    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
    constexpr uint32_t STORE_BLOCK_M = BLOCK_M, STORE_BLOCK_N = 64;
    constexpr uint32_t kNumSlicesPerWarp = 128 / BLOCK_M, kNumLanesPerSubpage = 32 / kNumSlicesPerWarp;
    constexpr uint32_t kNumStoresPerThread = (UMMA_N / STORE_BLOCK_N) / kNumSlicesPerWarp;
    constexpr uint32_t kNumGroupCols = kNumStoresPerThread * STORE_BLOCK_N;
    constexpr uint32_t kNumElemsPerLoad = 8, kNumLoadsPerStore = STORE_BLOCK_N / kNumElemsPerLoad;
    constexpr uint32_t kNumControlThreads = 128, kNumGroupThreads = 128, kNumEpilogueThreads = 256;
    constexpr uint32_t kEpiWarpStartIdx = 4, kNumTmemCols = 512, kNumTmemSliceCols = 256;
    constexpr bool kNeedExtraEpiRegs = kHasBias and kNumCDPagesPerGroup == 1;
    constexpr int kNumControlRegisters = kNeedExtraEpiRegs ? 24 : 40;
    constexpr int kNumEpilogueRegisters = kNeedExtraEpiRegs ? 240 : 232;

    DG_STATIC_ASSERT((BLOCK_M == 64 or BLOCK_M == 128) and (kNumCDPagesPerGroup == 1 or kNumCDPagesPerGroup == 2) and
                     kNumCDPagesPerGroup <= kNumStoresPerThread and kNumSharedResPages <= 1 and
                     (not kHasSharedResIn or kNumStoresPerThread % 2 == 0), "Invalid WholeRow config");
    DG_STATIC_ASSERT(kNumAStages <= 32 and kNumBStages <= 32 and sizeof(SharedStorage) <= 232448, "Shared storage exceeded");
    DG_STATIC_ASSERT(kNumControlRegisters * kNumControlThreads + kNumEpilogueRegisters * kNumEpilogueThreads <= 64512, "Register budget exceeded");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const auto warp_idx = static_cast<uint32_t>(cutlass::canonical_warp_idx_sync());
    const auto lane_idx = ptx::get_lane_idx();
    const auto cta_rank_idx = static_cast<uint32_t>(cute::block_rank_in_cluster());
    const bool is_leader_cta = cta_rank_idx == 0;
    if constexpr (kUnalignedIO)
        prefetch_tma_descs_warp0(warp_idx, tensor_map_a, tensor_map_b);
    else
        prefetch_tma_descs_warp0(warp_idx, tensor_map_a, tensor_map_b, tensor_map_residual, tensor_map_d);

    if (warp_idx == 1) {
        init_barrier_pairs(smem.a_full_barriers, smem.a_empty_barriers, lane_idx, 1, kNumMulticast);
        init_barrier_pairs(smem.tmem_full_barriers, smem.tmem_empty_barriers, lane_idx, 1, BLOCK_M == 64 ? kNumEpilogueThreads : kNumGroupThreads);
        fence_warp_barrier_init();
    } else if (warp_idx == 2) {
        init_barrier_pairs(smem.res_full_barriers[0], smem.cd_empty_barriers[0], lane_idx, 1, kNumGroupThreads);
        init_barrier_pairs(smem.res_full_barriers[1], smem.cd_empty_barriers[1], lane_idx, 1, kNumGroupThreads);
        if constexpr (kHasSharedResIn) {
            if (lane_idx == 0)
                smem.shared_res_empty_barrier.init(kNumGroupThreads);
        }
        fence_warp_barrier_init();
    } else if (warp_idx == 3) {
        init_barrier_pairs(smem.b_full_barriers, smem.b_empty_barriers, lane_idx, 1, 1);
        init_barrier_pairs(smem.peer_sumsq_full_barriers, smem.peer_sumsq_empty_barriers, lane_idx, 1, kNumLocalSlices);
        fence_warp_barrier_init();
    }
    comm::cluster_sync_with_relaxed_arrive();

    DG_STATIC_ASSERT(kNumSMs > 0 and kNumSMs % kNumMulticast == 0, "Invalid SM count");
    constexpr uint32_t kWaveStrideM = (static_cast<uint32_t>(kNumSMs) / kNumMulticast) * BLOCK_M;

    if (warp_idx < kEpiWarpStartIdx) {
        const auto alloc_and_sync_tmem = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
            if (warp_idx == 2)
                allocate_tmem_checked<Allocator, kNumTmemCols>(&smem.tmem_ptr);
            if (warp_idx == 1 or warp_idx == 2)
                cutlass::arch::NamedBarrier::sync(64, 4);
        };
        if constexpr (kNumControlRegisters < 32)
            alloc_and_sync_tmem();
        cutlass::arch::warpgroup_reg_dealloc<kNumControlRegisters>();
        __syncwarp();
        if constexpr (kNumControlRegisters >= 32)
            alloc_and_sync_tmem();
        const uint32_t m_start_idx = (blockIdx.x / kNumMulticast) * BLOCK_M;
        if (warp_idx == 0 and cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            uint32_t a_stage_idx = 0, a_phase_idx = 0;
            #pragma unroll 1
            for (uint32_t m_idx = m_start_idx; m_idx < shape_m; m_idx += kWaveStrideM) {
                #pragma unroll 1
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                    smem.a_empty_barriers[a_stage_idx].wait(a_phase_idx ^ 1u);
                    if (is_leader_cta) {
                        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
                            &tensor_map_a, reinterpret_cast<uint64_t*>(&smem.a_full_barriers[a_stage_idx]),
                            kCTAMask, static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST),
                            smem.a[a_stage_idx], static_cast<int>(k_block_idx * BLOCK_K), static_cast<int>(m_idx));
                    }
                    smem.a_full_barriers[a_stage_idx].arrive_and_expect_tx(sizeof(smem.a[0]));
                    advance_stage<kNumAStages>(a_stage_idx, a_phase_idx);
                }
            }
        } else if (warp_idx == 3 and cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            constexpr uint64_t kBHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST);
            const uint32_t cta_n_idx = cta_rank_idx * kNumCtaCols;
            uint32_t b_stage_idx = 0, b_phase_idx = 0;
            #pragma unroll 1
            for (uint32_t m_idx = m_start_idx; m_idx < shape_m; m_idx += kWaveStrideM) {
                #pragma unroll 1
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                    #pragma unroll
                    for (uint32_t local_slice_idx = 0; local_slice_idx < kNumLocalSlices; ++ local_slice_idx) {
                        smem.b_empty_barriers[b_stage_idx].wait(b_phase_idx ^ 1u);
                        tma::copy<BLOCK_K, UMMA_N, kNumSwizzleBytes, cutlass::bfloat16_t, false, kBHint>(
                            &tensor_map_b, &smem.b_full_barriers[b_stage_idx],
                            smem.b[b_stage_idx], k_block_idx * BLOCK_K, cta_n_idx + local_slice_idx * UMMA_N, 1);
                        smem.b_full_barriers[b_stage_idx].arrive_and_expect_tx(sizeof(smem.b[0]));
                        advance_stage<kNumBStages>(b_stage_idx, b_phase_idx);
                    }
                }
            }
        } else if (warp_idx == 1) {
            const auto runtime_instr_desc = make_bf16_umma_desc<UMMA_M, UMMA_N>();
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_ATOM_K, kNumSwizzleBytes>(smem.a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, UMMA_N, BLOCK_ATOM_K, kNumSwizzleBytes>(smem.b[0], 0, 0);
            const uint32_t a_desc_base_lo = a_desc.lo, b_desc_base_lo = b_desc.lo;
            uint32_t a_stage_idx = 0, a_phase_idx = 0, b_stage_idx = 0, b_phase_idx = 0, tile_iter_idx = 0;
            #pragma unroll 1
            for (uint32_t m_idx = m_start_idx; m_idx < shape_m; m_idx += kWaveStrideM, ++ tile_iter_idx) {
                const uint32_t tmem_stage_idx = tile_iter_idx & 1u;
                const uint32_t tmem_wait_phase_idx = ((BLOCK_M == 64 ? (tile_iter_idx >> 1u) : tile_iter_idx) & 1u) ^ 1u;
                #pragma unroll 1
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                    smem.a_full_barriers[a_stage_idx].wait(a_phase_idx);
                    const uint32_t a_base_lo = a_desc_base_lo + a_stage_idx * static_cast<uint32_t>(sizeof(smem.a[0]) / 16);
                    #pragma unroll
                    for (uint32_t local_slice_idx = 0; local_slice_idx < kNumLocalSlices; ++ local_slice_idx) {
                        if (k_block_idx == 0 and (BLOCK_M != 64 or local_slice_idx == 0))
                            smem.tmem_empty_barriers[BLOCK_M == 64 ? tmem_stage_idx : local_slice_idx].wait(tmem_wait_phase_idx);
                        smem.b_full_barriers[b_stage_idx].wait(b_phase_idx);
                        ptx::tcgen05_after_thread_sync();

                        const uint32_t b_base_lo = b_desc_base_lo + b_stage_idx * static_cast<uint32_t>(sizeof(smem.b[0]) / 16);
                        const uint32_t tmem_addr = BLOCK_M == 64
                            ? ((local_slice_idx * (16u << 16)) | (tmem_stage_idx * kNumTmemSliceCols))
                            : (local_slice_idx * kNumTmemSliceCols);
                        if (cute::elect_one_sync()) {
                            issue_bf16_umma_k_block<ptx::SM100_MMA_F16BF16_SS, BLOCK_M, UMMA_N, BLOCK_K>(
                                a_desc, b_desc, a_base_lo, b_base_lo, tmem_addr, k_block_idx > 0, runtime_instr_desc);
                            ptx::umma_arrive_no_elect(smem.b_empty_barriers[b_stage_idx]);
                            if (local_slice_idx + 1 == kNumLocalSlices)
                                cutlass::arch::umma_arrive_multicast_no_elect(reinterpret_cast<uint64_t*>(&smem.a_empty_barriers[a_stage_idx]), kCTAMask);
                            if (k_block_idx + 1 == kNumKBlocks and (BLOCK_M != 64 or local_slice_idx + 1 == kNumLocalSlices))
                                ptx::umma_arrive_no_elect(smem.tmem_full_barriers[BLOCK_M == 64 ? tmem_stage_idx : local_slice_idx]);
                        }
                        __syncwarp();
                        advance_stage<kNumBStages>(b_stage_idx, b_phase_idx);
                    }
                    advance_stage<kNumAStages>(a_stage_idx, a_phase_idx);
                }
            }
        } else if (not kUnalignedIO and warp_idx == 2 and cute::elect_one_sync()) {
            cudaGridDependencySynchronize();
            constexpr uint64_t kEvictFirst = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_FIRST);
            const uint32_t cta_n_idx = cta_rank_idx * kNumCtaCols;
            uint32_t res_phase_idx = 0;
            #pragma unroll 1
            for (uint32_t m_idx = m_start_idx; m_idx < shape_m; m_idx += kWaveStrideM) {
                if (m_idx == m_start_idx)
                    smem.a_full_barriers[1].wait(0);
                if constexpr (kHasSharedResIn) {
                    #pragma unroll
                    for (uint32_t store_idx = 0; store_idx < kNumStoresPerThread; ++ store_idx) {
                        #pragma unroll
                        for (uint32_t group_idx = 0; group_idx < kNumLocalSlices; ++ group_idx) {
                            smem.shared_res_empty_barrier.wait(res_phase_idx ^ 1u);
                            res_phase_idx ^= 1u;
                            #pragma unroll
                            for (uint32_t sub_idx = 0; sub_idx < kNumSlicesPerWarp; ++ sub_idx) {
                                tma::copy<STORE_BLOCK_N, STORE_BLOCK_M, kNumSwizzleBytes, cutlass::bfloat16_t, false, kEvictFirst>(
                                    &tensor_map_residual, &smem.res_full_barriers[group_idx][0], smem.cd_flat[kSharedResPageIdx][sub_idx],
                                    cta_n_idx + sub_idx * UMMA_N + group_idx * kNumGroupCols + store_idx * STORE_BLOCK_N, m_idx, 1);
                            }
                            smem.res_full_barriers[group_idx][0].arrive_and_expect_tx(sizeof(smem.cd_flat[0]));
                        }
                    }
                } else {
                    constexpr uint32_t kNumBatches = kNumStoresPerThread / kNumCDPagesPerGroup;
                    #pragma unroll
                    for (uint32_t batch_idx = 0; batch_idx < kNumBatches; ++ batch_idx) {
                        #pragma unroll
                        for (uint32_t page_idx = 0; page_idx < kNumCDPagesPerGroup; ++ page_idx) {
                            const uint32_t store_idx = batch_idx * kNumCDPagesPerGroup + page_idx;
                            #pragma unroll
                            for (uint32_t group_idx = 0; group_idx < kNumLocalSlices; ++ group_idx) {
                                smem.cd_empty_barriers[group_idx][page_idx].wait(res_phase_idx ^ 1u);
                                #pragma unroll
                                for (uint32_t sub_idx = 0; sub_idx < kNumSlicesPerWarp; ++ sub_idx) {
                                    tma::copy<STORE_BLOCK_N, STORE_BLOCK_M, kNumSwizzleBytes, cutlass::bfloat16_t, false, kEvictFirst>(
                                        &tensor_map_residual, &smem.res_full_barriers[group_idx][page_idx],
                                        smem.cd_flat[group_idx * kNumCDPagesPerGroup + page_idx][sub_idx],
                                        cta_n_idx + sub_idx * UMMA_N + group_idx * kNumGroupCols + store_idx * STORE_BLOCK_N, m_idx, 1);
                                }
                                smem.res_full_barriers[group_idx][page_idx].arrive_and_expect_tx(sizeof(smem.cd_flat[0]));
                            }
                        }
                        res_phase_idx ^= 1u;
                    }
                }
            }
        }
        __syncwarp();
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        __syncwarp();
        cudaGridDependencySynchronize();

        const uint32_t epi_warp_idx = warp_idx - kEpiWarpStartIdx, group_idx = epi_warp_idx / 4, group_warp_idx = epi_warp_idx % 4;
        const uint32_t sub_idx = lane_idx / kNumLanesPerSubpage, sub_lane_idx = lane_idx % kNumLanesPerSubpage;
        const uint32_t local_row_idx = group_warp_idx * kNumLanesPerSubpage + sub_lane_idx;
        const uint32_t base_group_n_idx = cta_rank_idx * kNumCtaCols + group_idx * kNumGroupCols;
        const uint32_t my_col_base_idx = base_group_n_idx + sub_idx * UMMA_N;
        const bool is_group_leader = group_warp_idx == 0 and cute::elect_one_sync();
        __syncwarp();

        uint4 my_bias_u4 = make_uint4(0, 0, 0, 0);
        if constexpr (kHasBias)
            my_bias_u4 = ld_global_u4_16b_or_2b(static_cast<const cutlass::bfloat16_t*>(bias) + my_col_base_idx, sub_lane_idx);
        const uint4 my_gamma_u4 = ld_global_u4_16b_or_2b(static_cast<const cutlass::bfloat16_t*>(gamma) + my_col_base_idx, sub_lane_idx);

        const uint32_t smem_group_base = cute::cast_smem_ptr_to_uint(smem.cd_flat[group_idx * kNumCDPagesPerGroup][sub_idx]) + local_row_idx * kNumSwizzleBytes;
        constexpr uint32_t kNumSlotBytes = static_cast<uint32_t>(sizeof(smem.cd_flat[0]));
        const auto get_cell_addr = [smem_group_base, local_row_idx](const uint32_t page_idx, const uint32_t load_idx) CUTLASS_LAMBDA_FUNC_INLINE {
            return smem_group_base + page_idx * kNumSlotBytes + (load_idx ^ (local_row_idx & 7u)) * 16u;
        };
        const auto issue_group_tma_store = [&smem, group_idx, is_group_leader, base_group_n_idx](
                                               const cute::TmaDescriptor& desc, const uint32_t page_idx,
                                               const uint32_t store_idx, const uint32_t cur_m_idx) CUTLASS_LAMBDA_FUNC_INLINE {
            cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
            if (is_group_leader) {
                issue_fenced_tma_store_2d_pages<kNumSlicesPerWarp>(
                    desc, smem.cd_flat[group_idx * kNumCDPagesPerGroup + page_idx],
                    base_group_n_idx + store_idx * STORE_BLOCK_N, cur_m_idx, UMMA_N);
            }
            __syncwarp();
        };

        uint4 held[kNumStoresPerThread][kNumLoadsPerStore];
        uint32_t tile_iter_idx = 0;
        #pragma unroll 1
        for (uint32_t m_idx = (blockIdx.x / kNumMulticast) * BLOCK_M; m_idx < shape_m; m_idx += kWaveStrideM, ++ tile_iter_idx) {
            const uint32_t tile_phase_idx = tile_iter_idx & 1u, sumsq_buf_idx = tile_iter_idx & 1u;
            const uint32_t sumsq_phase_idx = (tile_iter_idx >> 1u) & 1u;

            if constexpr (kUnalignedIO) {
                const uint32_t row_idx = m_idx + local_row_idx;
                const auto residual_ptr = static_cast<const cutlass::bfloat16_t*>(residual);
                #pragma unroll
                for (uint32_t store_idx = 0; store_idx < kNumStoresPerThread; ++ store_idx) {
                    const auto res_row_ptr = residual_ptr + static_cast<uint64_t>(row_idx) * kShapeN + my_col_base_idx + store_idx * STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                        held[store_idx][load_idx] = row_idx < shape_m ? ld_global_u4_16b_or_2b(res_row_ptr, load_idx) : make_uint4(0, 0, 0, 0);
                }
                cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
            } else if constexpr (kHasSharedResIn) {
                const uint32_t smem_shared_res_base = cute::cast_smem_ptr_to_uint(smem.cd_flat[kSharedResPageIdx][sub_idx]) + local_row_idx * kNumSwizzleBytes;
                #pragma unroll
                for (uint32_t store_idx = 0; store_idx < kNumStoresPerThread; ++ store_idx) {
                    smem.res_full_barriers[group_idx][0].wait(store_idx & 1u);
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                        held[store_idx][load_idx] = ld_smem_u4(smem_shared_res_base + (load_idx ^ (local_row_idx & 7u)) * 16u);
                    cutlass::arch::fence_view_async_shared();
                    smem.shared_res_empty_barrier.arrive();
                }
                if (tile_iter_idx > 0) {
                    if (is_group_leader)
                        cute::tma_store_wait<0>();
                    cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
                }
            } else {
                constexpr uint32_t kNumResBatches = kNumStoresPerThread / kNumCDPagesPerGroup;
                const uint32_t res_base_phase_idx = (kNumResBatches & 1u) ? tile_phase_idx : 0u;
                #pragma unroll
                for (uint32_t store_idx = 0; store_idx < kNumStoresPerThread; ++ store_idx) {
                    const uint32_t page_idx = store_idx % kNumCDPagesPerGroup;
                    smem.res_full_barriers[group_idx][page_idx].wait(res_base_phase_idx ^ ((store_idx / kNumCDPagesPerGroup) & 1u));
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                        held[store_idx][load_idx] = ld_smem_u4(get_cell_addr(page_idx, load_idx));
                    if (store_idx + kNumCDPagesPerGroup < kNumStoresPerThread) {
                        cutlass::arch::fence_view_async_shared();
                        smem.cd_empty_barriers[group_idx][page_idx].arrive();
                    }
                }
                cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
            }

            const uint32_t tmem_bar_idx = BLOCK_M == 64 ? tile_phase_idx : group_idx;
            smem.tmem_full_barriers[tmem_bar_idx].wait(BLOCK_M == 64 ? sumsq_phase_idx : tile_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const uint32_t group_tmem_col_idx = (BLOCK_M == 64 ? tile_phase_idx * kNumTmemSliceCols : 0u) + group_idx * kNumGroupCols;
            float2 row_sumsq = make_float2(0.0f, 0.0f);
            uint32_t values[2][kNumElemsPerLoad];
            ptx::tmem_load_32dp32b<kNumElemsPerLoad>(group_tmem_col_idx, values[0]);
            #pragma unroll
            for (uint32_t store_idx = 0; store_idx < kNumStoresPerThread; ++ store_idx) {
                const uint32_t store_tmem_col_idx = group_tmem_col_idx + store_idx * STORE_BLOCK_N;
                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx) {
                    cutlass::arch::fence_view_async_tmem_load();
                    if (store_idx + 1 == kNumStoresPerThread and load_idx + 1 == kNumLoadsPerStore) {
                        ptx::tcgen05_before_thread_sync();
                        smem.tmem_empty_barriers[tmem_bar_idx].arrive();
                    }
                    if (load_idx + 1 < kNumLoadsPerStore)
                        ptx::tmem_load_32dp32b<kNumElemsPerLoad>(store_tmem_col_idx + (load_idx + 1) * kNumElemsPerLoad, values[(load_idx + 1) & 1]);
                    else if (store_idx + 1 < kNumStoresPerThread)
                        ptx::tmem_load_32dp32b<kNumElemsPerLoad>(store_tmem_col_idx + STORE_BLOCK_N, values[0]);
                    const uint4 res_pack = held[store_idx][load_idx];
                    uint4 bias_pack = make_uint4(0, 0, 0, 0);
                    if constexpr (kHasBias)
                        bias_pack = shfl_u4_sync(my_bias_u4, static_cast<int>(sub_idx * kNumLanesPerSubpage + store_idx * kNumLoadsPerStore + load_idx));
                    const uint4 new_res_u4 = accumulate_bias_residual_u4<kHasBias>(
                        reinterpret_cast<const float2*>(values[load_idx & 1]), bias_pack, res_pack, row_sumsq);
                    held[store_idx][load_idx] = new_res_u4;
                    if constexpr (kUnalignedIO) {
                        const uint32_t row_idx = m_idx + local_row_idx;
                        if (row_idx < shape_m)
                            st_global_u4_16b_or_2b(residual, static_cast<uint64_t>(row_idx) * kShapeN + my_col_base_idx + store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerLoad, new_res_u4);
                    } else if (store_idx < kNumCDPagesPerGroup) {
                        st_smem_u4(get_cell_addr(store_idx, load_idx), new_res_u4);
                    }
                }
                if constexpr (not kUnalignedIO) {
                    if (store_idx < kNumCDPagesPerGroup)
                        issue_group_tma_store(tensor_map_residual, store_idx, store_idx, m_idx);
                }
            }

            float thread_sumsq = row_sumsq.x + row_sumsq.y;
            if constexpr (kNumSlicesPerWarp == 2)
                thread_sumsq += __shfl_xor_sync(0xffffffff, thread_sumsq, 16);
            if (sub_idx == 0)
                ptx::st_shared(&smem.group_sumsq[group_idx][local_row_idx], thread_sumsq);
            cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);

            if (epi_warp_idx == 0) {
                if (tile_iter_idx >= 2)
                    smem.peer_sumsq_empty_barriers[sumsq_buf_idx].wait(((tile_iter_idx - 2) >> 1u) & 1u);
                if (lane_idx < BLOCK_M / 4) {
                    const float4 s0 = ptx::ld_shared(reinterpret_cast<const float4*>(smem.group_sumsq[0]) + lane_idx);
                    const float4 s1 = ptx::ld_shared(reinterpret_cast<const float4*>(smem.group_sumsq[1]) + lane_idx);
                    ptx::st_async_cluster(reinterpret_cast<float4*>(smem.peer_cta_sumsq[sumsq_buf_idx]) + lane_idx,
                                          make_float4(s0.x + s1.x, s0.y + s1.y, s0.z + s1.z, s0.w + s1.w),
                                          cta_rank_idx ^ 1u, smem.peer_sumsq_full_barriers[sumsq_buf_idx]);
                }
                if (cute::elect_one_sync())
                    smem.peer_sumsq_full_barriers[sumsq_buf_idx].arrive_and_expect_tx(BLOCK_M * sizeof(float));
                __syncwarp();
            }

            const float my_cta_sumsq_val = ptx::ld_shared(&smem.group_sumsq[0][local_row_idx]) +
                                           ptx::ld_shared(&smem.group_sumsq[1][local_row_idx]);
            if constexpr (kUnalignedIO) {
                cutlass::arch::NamedBarrier::sync(kNumEpilogueThreads, 0);
            } else {
                #pragma unroll
                for (uint32_t store_idx = kNumCDPagesPerGroup; store_idx < kNumStoresPerThread; ++ store_idx) {
                    const uint32_t page_idx = store_idx % kNumCDPagesPerGroup;
                    if (is_group_leader)
                        cute::tma_store_wait<kNumCDPagesPerGroup - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx)
                        st_smem_u4(get_cell_addr(page_idx, load_idx), held[store_idx][load_idx]);
                    issue_group_tma_store(tensor_map_residual, page_idx, store_idx, m_idx);
                }
            }

            smem.peer_sumsq_full_barriers[sumsq_buf_idx].wait(sumsq_phase_idx);
            const float rstd = compute_rstd_from_sumsq<kShapeN>(
                my_cta_sumsq_val + ptx::ld_shared(&smem.peer_cta_sumsq[sumsq_buf_idx][local_row_idx]));

            #pragma unroll
            for (uint32_t store_idx = 0; store_idx < kNumStoresPerThread; ++ store_idx) {
                const uint32_t page_idx = store_idx % kNumCDPagesPerGroup;
                if constexpr (not kUnalignedIO) {
                    #pragma unroll
                    for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx) {
                        const int src_lane_idx = static_cast<int>(sub_idx * kNumLanesPerSubpage + store_idx * kNumLoadsPerStore + load_idx);
                        held[store_idx][load_idx] = scale_uint4_bf16_with_gamma(held[store_idx][load_idx], shfl_u4_sync(my_gamma_u4, src_lane_idx), rstd);
                    }
                    if (is_group_leader)
                        cute::tma_store_wait<kNumCDPagesPerGroup - 1>();
                }
                const bool need_peer_empty = store_idx == 0 and (m_idx + 2 * kWaveStrideM < shape_m);
                if (not kUnalignedIO or need_peer_empty)
                    cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
                if (need_peer_empty) {
                    cutlass::arch::fence_view_async_shared();
                    if (is_group_leader)
                        smem.peer_sumsq_empty_barriers[sumsq_buf_idx].arrive(cta_rank_idx ^ 1u);
                    __syncwarp();
                }
                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < kNumLoadsPerStore; ++ load_idx) {
                    if constexpr (kUnalignedIO) {
                        const int src_lane_idx = static_cast<int>(sub_idx * kNumLanesPerSubpage + store_idx * kNumLoadsPerStore + load_idx);
                        const uint4 out_u4 = scale_uint4_bf16_with_gamma(held[store_idx][load_idx], shfl_u4_sync(my_gamma_u4, src_lane_idx), rstd);
                        const uint32_t row_idx = m_idx + local_row_idx;
                        if (row_idx < shape_m)
                            st_global_u4_16b_or_2b(d, static_cast<uint64_t>(row_idx) * kShapeN + my_col_base_idx + store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerLoad, out_u4);
                    } else {
                        st_smem_u4(get_cell_addr(page_idx, load_idx), held[store_idx][load_idx]);
                    }
                }
                if constexpr (not kUnalignedIO)
                    issue_group_tma_store(tensor_map_d, page_idx, store_idx, m_idx);
            }

            if constexpr (not kUnalignedIO and not kHasSharedResIn) {
                if (m_idx + kWaveStrideM < shape_m) {
                    if constexpr (kNumCDPagesPerGroup == 2) {
                        if (is_group_leader)
                            cute::tma_store_wait<1>();
                        cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
                        cutlass::arch::fence_view_async_shared();
                        smem.cd_empty_barriers[group_idx][0].arrive();
                    }
                    if (is_group_leader)
                        cute::tma_store_wait<0>();
                    cutlass::arch::NamedBarrier::sync(kNumGroupThreads, group_idx + 1);
                    cutlass::arch::fence_view_async_shared();
                    smem.cd_empty_barriers[group_idx][kNumCDPagesPerGroup - 1].arrive();
                }
            }
        }
    }

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 2)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100");
#endif
}

template <int kNumSMs, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kSplitK = 1>
CUTLASS_GLOBAL void __launch_bounds__(256, 1)
sm100_bf16_gemm_nt_with_add_norm_n1024_k2816_small_impl(
    uint32_t shape_m,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    void* __restrict__ residual,
    void* __restrict__ d,
    const void* __restrict__ gamma,
    float* __restrict__ partials,
    uint32_t* __restrict__ counters) {
    sm100_bf16_gemm_add_norm_small_device<2816, false, kNumSMs, BLOCK_M, BLOCK_N, BLOCK_K, kSplitK>(
        shape_m, tensor_map_a, tensor_map_b,
        residual, d, nullptr, gamma, partials, counters);
}

template <int kNumSMs, uint32_t BLOCK_M = 128, uint32_t BLOCK_N = 128,
          uint32_t kNumStages = 6, uint32_t kNumCDPages = 2,
          bool kUseResTmaStore = false,
          uint32_t kNumResPages = 0, uint32_t kNumBStages = kNumStages>
CUTLASS_GLOBAL void __launch_bounds__(512, 1)
sm100_bf16_gemm_nt_with_add_norm_n1024_k2816_impl(
    uint32_t shape_m,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
    void* __restrict__ residual,
    const void* __restrict__ gamma,
    void* __restrict__ d,
    float* __restrict__ partials,
    uint32_t* __restrict__ counters) {
    sm100_bf16_gemm_add_norm_device<2816, false, kNumSMs, BLOCK_M, BLOCK_N,
                                    kNumStages, kNumCDPages, kUseResTmaStore,
                                    kNumResPages, kNumBStages>(
        shape_m, tensor_map_a, tensor_map_b, tensor_map_residual,
        residual, nullptr, gamma, d, partials, counters);
}

template <int kNumSMs, uint32_t kNumStages, uint32_t kNumResPages, uint32_t kNumBStages>
CUTLASS_GLOBAL void __launch_bounds__(384, 1)
sm100_bf16_gemm_nt_with_add_norm_n1024_k2816_pingpong_impl(
    uint32_t shape_m,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
    const __grid_constant__ cute::TmaDescriptor tensor_map_d,
    const void* __restrict__ gamma,
    float* __restrict__ partials,
    uint32_t* __restrict__ counters) {
    sm100_bf16_gemm_add_norm_device<
        2816, false, kNumSMs, 128, 128,
        kNumStages, 2, true, kNumResPages, kNumBStages, true>(
        shape_m, tensor_map_a, tensor_map_b, tensor_map_residual,
        nullptr, nullptr, gamma, nullptr, partials, counters, &tensor_map_d);
}

template <int kNumSMs, uint32_t BLOCK_M = 128, uint32_t kNumCDPagesPerGroup = 2,
          uint32_t kNumAStages = 4, uint32_t kNumBStages = 3, bool kUnalignedIO = false, uint32_t kNumSharedResPages = 0>
CUTLASS_GLOBAL void __launch_bounds__(384, 1)
sm100_bf16_gemm_nt_with_add_norm_n1024_k2816_whole_row_impl(
    uint32_t shape_m,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
    const __grid_constant__ cute::TmaDescriptor tensor_map_d,
    void* __restrict__ residual,
    const void* __restrict__ gamma,
    void* __restrict__ d) {
    sm100_bf16_gemm_add_norm_whole_row_device<
        2816, false, kNumSMs, BLOCK_M, kNumCDPagesPerGroup, kNumAStages, kNumBStages, kUnalignedIO, kNumSharedResPages>(
        shape_m, tensor_map_a, tensor_map_b, tensor_map_residual, tensor_map_d,
        residual, nullptr, gamma, d);
}

} // namespace deep_gemm::experimental
#pragma clang diagnostic pop
