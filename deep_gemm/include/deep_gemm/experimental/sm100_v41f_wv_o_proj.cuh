#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/operators.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::experimental {

template <uint32_t BLOCK_O, uint32_t kNumStages = BLOCK_O <= 128 ? 9 : 8,
          uint32_t kExtent = cute::max(128u, BLOCK_O),
          uint32_t SF_BLOCK_N = math::constexpr_align(kExtent, 128u)>
struct alignas(1024) SM100V41FWVOProjSharedStorage {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using FP8 = cutlass::float_e4m3_t;

    template <uint32_t kNumTMAStoreStages>
    struct StoreStorage {
        using cd_dtype = cutlass::bfloat16_t;
        alignas(1024) cd_dtype cd[kNumTMAStoreStages][128 * 16];
    };

    alignas(1024) FP8 a[kNumStages][128 * 128];
    alignas(1024) FP8 b[kNumStages][(kExtent / 2) * 128];
    uint32_t sfa[kNumStages][128];
    uint32_t sfb[kNumStages][SF_BLOCK_N];
    alignas(1024) cutlass::bfloat16_t cd_dedicated[BLOCK_O <= 128 ? 1 : 2 * 128 * 16];
    Barrier full_barriers[kNumStages];
    Barrier sf_full_barriers[kNumStages];
    Barrier empty_barriers[kNumStages];
    Barrier tmem_full_barriers[2];
    Barrier tmem_empty_barriers[2];
    uint32_t tmem_ptr;

    template <bool kAliasStore>
    CUTLASS_DEVICE auto& get_store_storage() {
        return *reinterpret_cast<StoreStorage<kAliasStore ? 8 : 2>*>(
            kAliasStore ? reinterpret_cast<void*>(a) : reinterpret_cast<void*>(cd_dedicated));
    }
};

template <uint32_t kNumMBlocks, uint32_t BLOCK_O, uint32_t kNumSMs>
CUTLASS_GLOBAL void __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
v41f_wv_o_proj_kernel(uint32_t shape_m, uint64_t* ready, cutlass::float_e4m3_t* mid, uint32_t* mid_sf,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_x,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_wv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_x_sf,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_wv_sf,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_mid,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_o,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_mid_sf,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_o_sf,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_y) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Allocator = cute::TMEM::Allocator2Sm;
    using FP8 = cutlass::float_e4m3_t;
    using SharedStorage = SM100V41FWVOProjSharedStorage<BLOCK_O>;

    constexpr uint32_t kNumGroups = 8, kInputRank = 4096, kOutputRank = 1024, kHidden = 5120;
    constexpr uint32_t BLOCK_M = 128, BLOCK_WV = 128, BLOCK_K = 128;
    constexpr uint32_t kNumStages = BLOCK_O <= 128 ? 9 : 8;
    constexpr auto kMajor = cute::UMMA::Major::K;
    constexpr uint32_t kExtent = cute::max(BLOCK_WV, BLOCK_O);
    constexpr uint32_t SF_BLOCK_M = 128, SF_BLOCK_N = math::constexpr_align(kExtent, 128u);
    constexpr uint32_t kSwizzleCDMode = 32, STORE_BLOCK_N = kSwizzleCDMode / sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kTmemStartColOfSFA = 2 * kExtent;
    constexpr uint32_t kTmemStartColOfSFB = kTmemStartColOfSFA + SF_BLOCK_M / 32;

    constexpr uint32_t kNumWVBlocksN = kNumGroups * kOutputRank / BLOCK_WV;
    constexpr uint32_t kNumOBlocksN = math::constexpr_ceil_div(kHidden, BLOCK_O);
    constexpr uint32_t kNumWVTasks = kNumMBlocks * kNumWVBlocksN;
    constexpr uint32_t kNumTasks = kNumWVTasks + kNumMBlocks * kNumOBlocksN;
    constexpr uint64_t kActivationCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_LAST);
    constexpr uint64_t kWeightCacheHint = static_cast<uint64_t>(
        kNumMBlocks == 2 ? cute::TMA::CacheHintSm100::EVICT_FIRST : cute::TMA::CacheHintSm100::EVICT_NORMAL);

    DG_STATIC_ASSERT(sizeof(SharedStorage) <= 231424, "Shared storage exceeds the launch allocation");
    DG_STATIC_ASSERT(kNumMBlocks >= 2 and kNumMBlocks % 2 == 0, "Invalid M block count");
    DG_STATIC_ASSERT(kNumWVBlocksN == 64, "Each row block uses one 64-bit readiness word");
    DG_STATIC_ASSERT(BLOCK_O % 16 == 0 and kExtent <= 256, "Invalid MMA tile");
    DG_STATIC_ASSERT(kTmemStartColOfSFB + SF_BLOCK_N / 32 <= 512 and kNumSMs % 2 == 0, "Invalid TMEM/cluster config");

    const uint32_t aligned_m = math::align(shape_m, 4u);

    comm::cluster_sync_with_relaxed_arrive();
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    const auto generations = reinterpret_cast<uint32_t*>(ready + kNumMBlocks);
    const auto cta_idx = cute::block_rank_in_cluster();
    const bool is_leader_cta = cta_idx == 0;

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            smem.sf_full_barriers[i].init(1);
            smem.empty_barriers[i].init(1);
            smem.full_barriers[i].init((1 + 32) * 2);
        }
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++ i) {
            smem.tmem_full_barriers[i].init(1);
            smem.tmem_empty_barriers[i].init(256);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(512, &smem.tmem_ptr);
    }
    comm::cluster_sync_with_relaxed_arrive();
    DG_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_ptr == 0);
    cudaGridDependencySynchronize();

    uint32_t stage_idx = 0, phase = 0;
    const auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    const auto for_each_tile = [&](auto body) {
        uint32_t task_idx = blockIdx.x, iter_idx = 0;
        for (; task_idx < kNumWVTasks; task_idx += kNumSMs, ++ iter_idx)
            body(cute::true_type(), task_idx % kNumMBlocks, task_idx / kNumMBlocks, iter_idx);
        for (; task_idx < kNumTasks; task_idx += kNumSMs, ++ iter_idx) {
            const uint32_t o_task_idx = task_idx - kNumWVTasks;
            body(cute::false_type(), o_task_idx % kNumMBlocks, o_task_idx / kNumMBlocks, iter_idx);
        }
    };

    if (warp_idx == 0 and cute::elect_one_sync()) {
        const auto generation_idx = generations[blockIdx.x] + 1;
        generations[blockIdx.x] = generation_idx;
        const uint64_t expected_mask = generation_idx & 1 ? ~uint64_t(0) : 0;

        for_each_tile([&]<bool kIsWV>(cute::bool_constant<kIsWV>, uint32_t m_block_idx, uint32_t n_block_idx, uint32_t) {
            constexpr uint32_t BLOCK_N = kIsWV ? BLOCK_WV : BLOCK_O;
            constexpr uint32_t PHASE_SF_BLOCK_N = kIsWV ? 128 : SF_BLOCK_N;
            constexpr uint32_t kNumKBlocks = (kIsWV ? kInputRank : kNumGroups * kOutputRank) / BLOCK_K;
            constexpr bool kSplitWait = not kIsWV and kNumWVTasks > kNumSMs;

            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t group_idx = kIsWV ? n_block_idx / (kOutputRank / BLOCK_WV) : 0;

            if constexpr (not kIsWV) {
                comm::wait_until([&]() {
                    const auto diff = ptx::ld_acq_gpu(ready + m_block_idx) ^ expected_mask;
                    return (kSplitWait ? (diff & 0xffffffffull) : diff) == 0;
                }, [&]() {
                    printf("o_proj row dependency timeout: row=%u, col=%u\n", m_block_idx, n_block_idx);
                });
                ptx::fence_async_global();
            }

            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                smem.empty_barriers[stage_idx].wait(phase ^ 1);
                if constexpr (kSplitWait) {
                    if (k_block_idx == kNumKBlocks / 2) {
                        comm::wait_until([&]() { return ptx::ld_acq_gpu(ready + m_block_idx) == expected_mask; }, [&]() {
                            printf("o_proj row dependency timeout: row=%u, col=%u\n", m_block_idx, n_block_idx);
                        });
                        ptx::fence_async_global();
                    }
                }

                const auto sf_k_idx = group_idx * (kInputRank / BLOCK_K) + k_block_idx;
                tma::copy<SF_BLOCK_M, 1, 0, uint32_t, false, kActivationCacheHint>(
                    kIsWV ? &tensor_map_x_sf : &tensor_map_mid_sf, &smem.sf_full_barriers[stage_idx],
                    smem.sfa[stage_idx], m_idx, sf_k_idx);
                smem.sf_full_barriers[stage_idx].arrive_and_expect_tx((SF_BLOCK_M + PHASE_SF_BLOCK_N) * sizeof(uint32_t));

                tma::copy<BLOCK_K, BLOCK_M, 128, FP8, true, kActivationCacheHint>(
                    kIsWV ? &tensor_map_x : &tensor_map_mid, &smem.full_barriers[stage_idx],
                    smem.a[stage_idx], k_block_idx * BLOCK_K, m_idx, 2, group_idx);
                if (is_leader_cta)
                    smem.full_barriers[stage_idx].arrive_and_expect_tx((BLOCK_M + BLOCK_N / 2) * BLOCK_K * 2);
                else
                    smem.full_barriers[stage_idx].arrive(0u);
            }
        });
        cudaTriggerProgrammaticLaunchCompletion();
    } else if (warp_idx == 1 and is_leader_cta) {
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            FP8, FP8, float, cutlass::float_ue8m0_t, 256, kExtent, kMajor, kMajor>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);
        auto a_desc = mma::sm100::make_umma_desc<kMajor, BLOCK_M, BLOCK_K, 128>(smem.a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajor, kExtent / 2, BLOCK_K, 128>(smem.b[0], 0, 0);
        const uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(smem.a[0]) / 16 : 0;
        const uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(smem.b[0]) / 16 : 0;

        for_each_tile([&]<bool kIsWV>(cute::bool_constant<kIsWV>, uint32_t, uint32_t, uint32_t iter_idx) {
            constexpr uint32_t BLOCK_N = kIsWV ? BLOCK_WV : BLOCK_O;
            constexpr uint32_t PHASE_SF_BLOCK_N = kIsWV ? 128 : SF_BLOCK_N;
            constexpr uint32_t kNumKBlocks = (kIsWV ? kInputRank : kNumGroups * kOutputRank) / BLOCK_K;

            const uint32_t accum_stage_idx = iter_idx % 2;
            smem.tmem_empty_barriers[accum_stage_idx].wait(((iter_idx / 2) & 1) ^ 1);
            ptx::tcgen05_after_thread_sync();
            mma::sm100::update_instr_desc_with_umma_n(instr_desc, BLOCK_N);

            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                smem.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();

                if (cute::elect_one_sync()) {
                    mma::sm100::replace_smem_desc_addr(sf_desc, smem.sfa[stage_idx]);
                    cute::SM100_UTCCP_4x32dp128bit_2cta::copy(sf_desc, kTmemStartColOfSFA);
                    #pragma unroll
                    for (uint32_t sf_idx = 0; sf_idx < PHASE_SF_BLOCK_N / 128; ++ sf_idx) {
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem.sfb[stage_idx] + sf_idx * 128);
                        cute::SM100_UTCCP_4x32dp128bit_2cta::copy(sf_desc, kTmemStartColOfSFB + sf_idx * 4);
                    }
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < 4; ++ umma_k_idx) {
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajor, BLOCK_M, 128, FP8>(a_desc_base_lo, 0, umma_k_idx * 32);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajor, kExtent / 2, 128, FP8>(b_desc_base_lo, 0, umma_k_idx * 32);
                        const auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, umma_k_idx, umma_k_idx);
                        ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                            a_desc, b_desc, accum_stage_idx * kExtent, k_block_idx > 0 or umma_k_idx > 0,
                            runtime_instr_desc, kTmemStartColOfSFA, kTmemStartColOfSFB);
                    }
                }
                cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&smem.empty_barriers[stage_idx]), 3);
                if (k_block_idx + 1 == kNumKBlocks)
                    cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&smem.tmem_full_barriers[accum_stage_idx]), 3);
                __syncwarp();
            }
        });
    } else if (warp_idx == 2) {
        const auto utccp_required_smem_warp_transpose = [&](uint32_t* smem_ptr) {
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };
        for_each_tile([&]<bool kIsWV>(cute::bool_constant<kIsWV>, uint32_t, uint32_t, uint32_t) {
            constexpr uint32_t PHASE_SF_BLOCK_N = kIsWV ? 128 : SF_BLOCK_N;
            constexpr uint32_t kNumKBlocks = (kIsWV ? kInputRank : kNumGroups * kOutputRank) / BLOCK_K;
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                smem.sf_full_barriers[stage_idx].wait(phase);
                utccp_required_smem_warp_transpose(smem.sfa[stage_idx]);
                #pragma unroll
                for (uint32_t sf_idx = 0; sf_idx < PHASE_SF_BLOCK_N / 128; ++ sf_idx)
                    utccp_required_smem_warp_transpose(smem.sfb[stage_idx] + sf_idx * 128);
                cutlass::arch::fence_view_async_shared();
                smem.full_barriers[stage_idx].arrive(0u);
            }
        });
    } else if (warp_idx == 3 and cute::elect_one_sync()) {
        for_each_tile([&]<bool kIsWV>(cute::bool_constant<kIsWV>, uint32_t, uint32_t n_block_idx, uint32_t) {
            constexpr uint32_t BLOCK_N = kIsWV ? BLOCK_WV : BLOCK_O;
            constexpr uint32_t PHASE_SF_BLOCK_N = kIsWV ? 128 : SF_BLOCK_N;
            constexpr uint32_t kNumKBlocks = (kIsWV ? kInputRank : kNumGroups * kOutputRank) / BLOCK_K;

            const uint32_t group_idx = kIsWV ? n_block_idx / (kOutputRank / BLOCK_WV) : 0;
            const uint32_t n_idx = kIsWV ? n_block_idx % (kOutputRank / BLOCK_WV) * BLOCK_WV : n_block_idx * BLOCK_O;

            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                smem.empty_barriers[stage_idx].wait(phase ^ 1);
                const auto sf_k_idx = group_idx * (kInputRank / BLOCK_K) + k_block_idx;
                tma::copy<PHASE_SF_BLOCK_N, 1, 0, uint32_t, false, kWeightCacheHint>(
                    kIsWV ? &tensor_map_wv_sf : &tensor_map_o_sf, &smem.sf_full_barriers[stage_idx],
                    smem.sfb[stage_idx], n_idx, sf_k_idx);
                tma::copy<BLOCK_K, BLOCK_N / 2, 128, FP8, true, kWeightCacheHint>(
                    kIsWV ? &tensor_map_wv : &tensor_map_o, &smem.full_barriers[stage_idx],
                    smem.b[stage_idx], k_block_idx * BLOCK_K, n_idx + cta_idx * (BLOCK_N / 2), 2, group_idx);
            }
        });
    } else if (warp_idx >= 4) {
        const auto epilogue_warp_idx = warp_idx - 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        uint32_t tma_stage_idx = 0;
        const auto bf16_args = EpilogueOperatorArgs();
        auto bf16_op = epilogue::operators::Identity(bf16_args);

        for_each_tile([&]<bool kIsWV>(cute::bool_constant<kIsWV>, uint32_t m_block_idx, uint32_t n_block_idx, uint32_t iter_idx) {
            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t accum_stage_idx = iter_idx % 2;
            const uint32_t tmem_base_addr = accum_stage_idx * kExtent;

            smem.tmem_full_barriers[accum_stage_idx].wait((iter_idx / 2) & 1);
            ptx::tcgen05_after_thread_sync();

            if constexpr (kIsWV) {
                const auto row_idx = m_idx + epilogue_thread_idx;
                const auto dst = reinterpret_cast<ptx::longlong4_t*>(
                    mid + (static_cast<uint64_t>(row_idx) * kNumWVBlocksN + n_block_idx) * BLOCK_WV);
                uint32_t sf_word = 0;

                #pragma unroll
                for (uint32_t load_idx = 0; load_idx < BLOCK_WV / 32; ++ load_idx) {
                    uint32_t values[32];
                    ptx::tmem_load_32dp32b<32>(tmem_base_addr + load_idx * 32, values);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (load_idx + 1 == BLOCK_WV / 32) {
                        ptx::tcgen05_before_thread_sync();
                        smem.tmem_empty_barriers[accum_stage_idx].arrive(0u);
                    }

                    nv_bfloat162 rounded[16];
                    #pragma unroll
                    for (uint32_t pair_idx = 0; pair_idx < 16; ++ pair_idx)
                        rounded[pair_idx] = math::cast_into_bf16x2(values[2 * pair_idx], values[2 * pair_idx + 1]);
                    const auto sf_exp = math::get_ue8m0_sf_exp<FP8>(math::get_packed_bf16_amax(rounded));
                    const auto sf_inv = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp));
                    ptx::longlong4_t payload;
                    const auto packed = reinterpret_cast<uint32_t*>(&payload);
                    #pragma unroll
                    for (uint32_t word_idx = 0; word_idx < 8; ++ word_idx)
                        packed[word_idx] = math::scale_bf16x2_into_fp8x4(rounded[2 * word_idx], rounded[2 * word_idx + 1], sf_inv, sf_inv);
                    if (row_idx < shape_m)
                        dst[load_idx] = payload;
                    sf_word |= sf_exp << (load_idx * 8);
                }
                if (row_idx < shape_m)
                    mid_sf[static_cast<uint64_t>(n_block_idx) * aligned_m + row_idx] = sf_word;
                cutlass::arch::NamedBarrier::sync(128, 0);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync())
                    ptx::red_xor_rel(ready + m_block_idx, 1ull << n_block_idx);
            } else {
                constexpr bool kAliasStore = kNumTasks - kNumWVTasks <= kNumSMs;
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_O, BLOCK_M, STORE_BLOCK_N, kSwizzleCDMode, kAliasStore ? 8 : 2, 128, 0, GemmType::Normal, false>(
                    smem.template get_store_storage<kAliasStore>(), tma_stage_idx, tmem_base_addr, m_idx, n_block_idx * BLOCK_O, 0,
                    false, epilogue_warp_idx, lane_idx, bf16_op, false,
                    nullptr, &smem.tmem_empty_barriers[accum_stage_idx], tensor_map_y);
            }
        });
    }
    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 2)
        Allocator().free(0, 512);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100");
#endif
}

} // namespace deep_gemm::experimental

#pragma clang diagnostic pop
