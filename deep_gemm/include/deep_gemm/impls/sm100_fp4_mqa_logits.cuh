#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp4_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t max_seqlen_k,
                          const uint32_t logits_stride,
                          const uint32_t* cu_seq_len_k_start,
                          const uint32_t* cu_seq_len_k_end,
                          logits_dtype_t* logits,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // Prefetch TMA descriptors
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    // UMMA configs
    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_K = 64;
    static constexpr uint32_t kNumSFQ  = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(BLOCK_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(BLOCK_KV == kNumMathWarpGroups * UMMA_M and BLOCK_KV % kNumUTCCPAlignedElems == 0, "Invalid `BLOCK_KV`");

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = 8 * (kHeadDim / 2);
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE      = BLOCK_Q * kNumHeads * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE   = kNumSFQ * sizeof(int);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE     = BLOCK_KV * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE  = kNumSFKV * sizeof(int);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE  % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q and KV data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i;
    });
    const auto smem_sf_ptr = smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
    auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * i);
    });
    auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages
                                                    + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });

    // Barriers and TMEM pointer on shared memory
    const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    const auto tmem_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
    auto full_tmem_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + i; });
    auto empty_tmem_barriers = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
    auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);

    // Tensor memory configs
    constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Initialize barriers
    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            full_tmem_barriers[i]->init(1);
            empty_tmem_barriers[i]->init(128);
        }
        cutlass::arch::fence_barrier_init();
    }

    // Allocate tensor memory
    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    __syncthreads();

    // Scheduler
    const uint32_t num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    auto load_schedule = [&](const uint32_t& q_idx) -> cute::tuple<uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto row_idx = cute::min(q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cute::min(cu_seq_len_k_start[row_idx], seq_len_kv);
            seq_k_end[i] = cute::min(cu_seq_len_k_end[row_idx], seq_len_kv);
            start = cute::min(start, seq_k_start[i]);
            end = cute::max(end, seq_k_end[i]);
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        return {start, math::ceil_div(end - start, BLOCK_KV)};
    };

    // Make Q, KV and TMEM pipeline
    auto make_pipeline = [](const uint32_t& num_stages) {
        // Return current stage and phase, and advance pipeline by steps
        return [iter_idx = 0u, num_stages](const uint32_t& step = 1) mutable -> cute::tuple<uint32_t, uint32_t> {
            uint32_t current_idx = iter_idx;
            iter_idx += step;
            return {current_idx % num_stages, (current_idx / num_stages) & 1};
        };
    };
    auto advance_q_pipeline    = make_pipeline(kNumQStages);
    auto advance_kv_pipeline   = make_pipeline(kNumKVStages);
    auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        // TMA warp for loading Q
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Enumerate Q blocks
        if (cute::elect_one_sync()) {
            for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
                // Wait Q consumer release
                CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                cute::SM90_TMA_LOAD_2D::copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[q_stage_idx]),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_q[q_stage_idx], 0, q_idx * BLOCK_Q * kNumHeads);
                tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, full_q_barriers[q_stage_idx], smem_sf_q[q_stage_idx], 0, q_idx * BLOCK_Q);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[q_stage_idx], smem_weights[q_stage_idx], 0, q_idx * BLOCK_Q);
                full_q_barriers[q_stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + kRealNumSFQ * sizeof(int) + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        // TMA warp for loading KV cache
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        if (cute::elect_one_sync()) {
            // Enumerate Q blocks
            for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
                // Load KV block ranges
                CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);

                // Enumerate KV blocks
                for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                    // Wait KV consumer release
                    CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    cute::SM90_TMA_LOAD_2D::copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                                                 static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                 smem_kv[kv_stage_idx], 0, kv_start + kv_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
                                              smem_sf_kv[kv_stage_idx],
                                              kv_start + kv_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        // UMMA warp
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        // Enumerate Q blocks
        for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
            // Load KV block ranges
            CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);

            // Wait TMA Q arrivals
            CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Transpose and copy SF Q
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                utccp_required_smem_warp_transpose(smem_ptr);
                cutlass::arch::fence_view_async_shared();
                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                if (cute::elect_one_sync())
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                __syncwarp();
            }

            // Enumerate KV blocks
            for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                // Wait TMA KV arrivals
                CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Transpose
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                    cutlass::arch::fence_view_async_shared();
                }

                // UMMA with SF
                if (cute::elect_one_sync()) {
                    // Copy SF KV
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        // Wait TMEM release
                        CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        // Issue UMMA with SF
                        #pragma unroll
                        for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                            auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * 2, k * 2);
                            // TODO: generalize umma desc
                            DG_STATIC_ASSERT(kHeadDim == 128, "Invalid head dim");
                            auto a_desc = mma::sm100::make_smem_desc(
                                cute::UMMA::LayoutType::SWIZZLE_64B,
                                smem_kv[kv_stage_idx] + i * UMMA_M * (kHeadDim / 2) + k * UMMA_K / 2,
                                8 * (kHeadDim / 2), 0);
                            auto b_desc = mma::sm100::make_smem_desc(
                                cute::UMMA::LayoutType::SWIZZLE_64B,
                                smem_q[q_stage_idx] + k * UMMA_K / 2,
                                8 * (kHeadDim / 2), 0);
                            ptx::SM100_MMA_MXF4_SS::fma(
                                a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                        }
                        // TODO: move this into `deep_gemm/ptx/tcgen05.cuh`
                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
            }

            // UMMA warp must also arrive on empty_q to prevent running ahead
            // of math warps in the Q pipeline. Without this, UMMA can consume
            // kNumQStages Q blocks before math warps release any, causing a
            // circular dependency: UMMA waits full_q -> TMA_Q waits empty_q
            // -> Math waits full_tmem -> UMMA (already moved on).
            empty_q_barriers[q_stage_idx]->arrive();
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        // Math warpgroups for reduce
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = threadIdx.x;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr uint32_t N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Math warpgroups process TMEM stages alternately
        // Advance pipeline to align with the assigned stage
        advance_tmem_pipeline(math_warpgroup_idx);

        // Local register buffers
        float accum[kNumHeads];
        float weights[BLOCK_Q][kNumHeads];

        // Enumerate Q blocks
        for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
            // Load KV block ranges
            CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);

            // Wait TMA Q arrivals
            CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            // TODO: optimize bank conflicts
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
            }

            // Enumerate KV blocks
            for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                // Calculate KV offset in advance
                auto kv_offset = kv_start + kv_idx * BLOCK_KV + math_thread_idx;

                // Advance pipeline by `kNumMathWarpGroups` steps
                // Wait UMMA arrival
                CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
                ptx::tcgen05_after_thread_sync();

                // Reduce over the head dim and store
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Load accumulator from TMEM
                    uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                    tmem_load(cute::Int<kNumHeads>{}, tmem_addr, accum);

                    // Release TMEM empty
                    if (i == BLOCK_Q - 1) {
                        ptx::tcgen05_before_thread_sync();
                        empty_tmem_barriers[tmem_stage_idx]->arrive();
                    }

                    // Accumulate weighted ReLU in parallel
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);

                    const auto transform = [&](const uint32_t& j, const float2& sum) {
                        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                        auto b = make_float2(weights[i][j], weights[i][j + 1]);
                        return __ffma2_rn(a, b, sum);
                    };

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        sum_0 = transform(j, sum_0);
                        sum_1 = transform(j + 2, sum_1);
                    }

                    auto sum = __fadd2_rn(sum_0, sum_1);
                    auto result = static_cast<logits_dtype_t>(sum.x + sum.y);

                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    // TODO: optimize performance
                    const auto q_offset = (q_idx * BLOCK_Q + i) * static_cast<uint64_t>(logits_stride);
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i])
                            logits[q_offset + kv_offset - seq_k_start[i]] = result;
                    } else {
                        logits[q_offset + kv_offset] = result;
                    }
                    __syncwarp();
                }
            }

            // Release last Q empty
            empty_q_barriers[q_stage_idx]->arrive();
        }

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm
