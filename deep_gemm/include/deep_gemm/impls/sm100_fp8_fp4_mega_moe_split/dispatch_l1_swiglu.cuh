#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_split/common.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_split.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

namespace deep_gemm::mega_moe_split {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kLocalOnly,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_split_dispatch_l1_swiglu_impl(
    int* cumulative_local_expert_recv_stats,
    const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output
) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
    }

    const auto workspace = layout::SplitWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout = layout::Data(kHidden);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "Invalid SF_BLOCK_N");

    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr());
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr());
    (void) combine_token_buffer;

    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;

    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;
    constexpr uint32_t UMMA_BLOCK_K = 128;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");

    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t) * (BLOCK_K / 128);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t) * (BLOCK_K / 128);
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE = SMEM_CD_L1_SIZE / kNumTMAStoreStages;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemoryAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemoryAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemoryAlignment == 0,
                     "Shared memory of CD/A/B must be aligned to 1024 bytes");

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);
    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(smem_gemm_base, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(
        smem_gemm_base, SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });
    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumStages]);
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + STORE_BLOCK_M * kNumEpilogueWarps / 2);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i;
    });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages + i;
    });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + kNumEpilogueWarps * 2);

    comm::cluster_sync_with_relaxed_arrive();

    if (warp_idx == 0) {
        if (cute::elect_one_sync())
            ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
    } else if (warp_idx == 1) {
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                full_barriers[i]->init(2 * 2);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                tmem_full_barriers[i]->init(1);
                tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    comm::cluster_sync_with_relaxed_arrive();

    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumRanks>(workspace);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kDispatchGridSyncIndex = 0;

    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    DG_STATIC_ASSERT(kNumTopk <= 64 and kNumExpertsPerRank <= 1024 and kNumMaxTokensPerRank <= 65536,
                     "Route entry bit-packing constraints");
    const auto pull_meta_base = workspace.get_src_token_topk_idx_ptr(0, 0, 0);
    constexpr uint32_t kRouteCountOffset = kNumExpertsPerRank * kNumRanks * kNumMaxTokensPerRank;
    constexpr uint32_t kRouteEntriesOffset = kRouteCountOffset + kNumRanks * kNumMaxTokensPerRank;

    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int32_t expert_idx = -1;
                if (i + lane_idx / kNumTopk < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int32_t>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        read_topk_idx([&](const uint32_t& token_topk_idx, const int32_t& expert_idx) {
            (void) token_topk_idx;
            atomicAdd_block(smem_expert_count + static_cast<uint32_t>(expert_idx), 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        #pragma unroll
        for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
             i < num_tokens;
             i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
            const uint32_t src_token_idx = i + lane_idx / kNumTopk;
            const bool in_range = src_token_idx < num_tokens and lane_idx < kNumActivateLanes;
            int32_t expert_idx = -1;
            if (in_range)
                expert_idx = static_cast<int32_t>(
                    __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
            const bool routed = in_range and expert_idx >= 0;
            const int32_t dst_rank_idx = routed ? expert_idx / static_cast<int32_t>(kNumExpertsPerRank) : 0;
            const uint32_t local_expert_idx = routed ? static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank : 0u;

            uint32_t dst_slot_idx = 0;
            if (routed) {
                dst_slot_idx = atomicAdd_block(smem_expert_count + static_cast<uint32_t>(expert_idx), 1);
                *sym_buffer.map(pull_meta_base + (local_expert_idx * kNumRanks + sym_buffer.rank_idx)
                                * kNumMaxTokensPerRank + dst_slot_idx, static_cast<uint32_t>(dst_rank_idx)) =
                    i * kNumTopk + lane_idx;
            }

            #pragma unroll
            for (uint32_t dst_idx = 0; dst_idx < kNumRanks; ++dst_idx) {
                const uint32_t targets_dst = __ballot_sync(0xffffffffu, routed and dst_rank_idx == int32_t(dst_idx));
                const uint32_t group_base = (lane_idx / kNumTopk) * kNumTopk;
                const uint32_t our_routes = targets_dst & (((1u << kNumTopk) - 1u) << group_base);
                const uint32_t route_count = __popc(our_routes);
                if (in_range and lane_idx == group_base)
                    *sym_buffer.map(pull_meta_base + kRouteCountOffset
                                    + sym_buffer.rank_idx * kNumMaxTokensPerRank + src_token_idx, dst_idx) =
                        route_count;
                if (routed and dst_rank_idx == int32_t(dst_idx) and route_count > 1) {
                    const uint32_t packed = (dst_slot_idx << 16) | (local_expert_idx << 6) | (lane_idx - group_base);
                    const uint32_t slot_idx = __popc(our_routes & ((1u << lane_idx) - 1u));
                    *sym_buffer.map(pull_meta_base + kRouteEntriesOffset
                                    + (sym_buffer.rank_idx * kNumMaxTokensPerRank + src_token_idx) * kNumTopk + slot_idx,
                                    dst_idx) = packed;
                }
            }
            __syncwarp();
        }

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); });

        if (sm_idx == 0) {
            if constexpr (kLocalOnly) {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExpertsPerRank; i += kNumDispatchThreads) {
                    const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                    const auto num_recv_tokens = static_cast<uint32_t>(expert_status);
                    #pragma unroll
                    for (uint32_t rank_idx = 0; rank_idx < kNumRanks; ++rank_idx) {
                        *workspace.get_expert_recv_count_ptr(rank_idx, i) =
                            rank_idx == 0 ? num_recv_tokens : 0u;
                    }
                    *workspace.get_expert_recv_count_sum_ptr(i) =
                        (static_cast<uint64_t>(kNumSMs) * kNumRanks << 32) | num_recv_tokens;
                }
            } else {
                #pragma unroll
                for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                    const auto dst_rank_idx = i / kNumExpertsPerRank;
                    const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                    const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                    if constexpr (kNumRanks == 1) {
                        *workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx) =
                            expert_status & 0xffffffff;
                        ptx::atomic_add(
                            workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx),
                            expert_status);
                    } else {
                        *sym_buffer.map(
                            workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                            dst_rank_idx) = expert_status & 0xffffffff;
                        ptx::atomic_add_sys(
                            sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                            expert_status);
                    }
                }
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        if constexpr (kLocalOnly or kNumRanks == 1) {
            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); });
        } else {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                false,
                true);
        }

        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        scheduler.fetch_expert_recv_count();

        const auto smem_per_rank_count = smem_expert_count;
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExpertsPerRank * kNumRanks; i += kNumDispatchThreads)
            smem_per_rank_count[i] = static_cast<uint32_t>(
                *workspace.get_expert_recv_count_ptr(i % kNumRanks, i / kNumRanks));
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int32_t current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            int32_t old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= int32_t(kNumExpertsPerRank))
                    break;

                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(static_cast<uint32_t>(current_expert_idx));
            }

            if (current_expert_idx >= int32_t(kNumExpertsPerRank))
                break;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks
                        ? smem_per_rank_count[static_cast<uint32_t>(current_expert_idx) * kNumRanks + j] : 0u;
                }
            }

            uint32_t current_rank_in_expert_idx, token_idx_in_rank;
            utils::peel_forward<kNumRanks>(stored_rank_count,
                                           token_idx - expert_start_idx,
                                           current_rank_in_expert_idx,
                                           token_idx_in_rank);

            const uint32_t src_token_topk_idx = *(pull_meta_base
                + (static_cast<uint32_t>(current_expert_idx) * kNumRanks + current_rank_in_expert_idx)
                    * kNumMaxTokensPerRank
                + token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;

            uint32_t route_count = 0;
            if (lane_idx == 0) {
                const auto route_count_ptr = pull_meta_base + kRouteCountOffset
                    + current_rank_in_expert_idx * kNumMaxTokensPerRank + src_token_idx;
                route_count = *route_count_ptr;
                if (route_count > 1)
                    route_count = atomicExch(route_count_ptr, 0u);
            }
            route_count = __shfl_sync(0xffffffffu, route_count, 0);
            if (route_count == 0)
                continue;

            if (cute::elect_one_sync()) {
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                   current_rank_in_expert_idx),
                    pull_mbarrier, kHidden);
            }
            __syncwarp();

            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            constexpr uint32_t kNumSFUint32PerLane = math::constexpr_ceil_div(kNumSFUint32, 32u);
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            uint32_t stored_sf[kNumSFUint32PerLane] = {};
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFUint32PerLane; ++i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    stored_sf[i] = remote_sf_ptr[j];
            }
            float stored_weight = 0.0f;
            if (lane_idx < kNumTopk)
                stored_weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_idx * kNumTopk + lane_idx,
                    current_rank_in_expert_idx);

            if (cute::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
            }
            __syncwarp();

            const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            if (route_count == 1) {
                const uint32_t route_topk_slot_idx = src_token_topk_idx - src_token_idx * kNumTopk;
                const uint32_t route_token_idx_in_expert = token_idx - expert_start_idx;
                const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + route_token_idx_in_expert;
                const uint32_t sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M
                                                  + transform_sf_token_idx(route_token_idx_in_expert);
                const uint32_t stored_pool_block_idx = expert_pool_block_offset + route_token_idx_in_expert / BLOCK_M;
                const float weight = __shfl_sync(0xffffffffu, stored_weight, route_topk_slot_idx);
                __syncwarp();
                if (cute::elect_one_sync()) {
                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, route_topk_slot_idx};
                    cute::tma_store_arrive();
                }

                #pragma unroll
                for (uint32_t i = 0; i < kNumSFUint32PerLane; ++i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFUint32)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = stored_sf[i];
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(workspace.get_l1_arrival_count_ptr(stored_pool_block_idx), 1);
                }
                __syncwarp();
                continue;
            }

            uint32_t stored_pool_blocks[kNumTopk] = {};
            #pragma unroll
            for (uint32_t route_idx = 0; route_idx < route_count; ++route_idx) {
                const uint32_t packed = *(pull_meta_base + kRouteEntriesOffset
                    + (current_rank_in_expert_idx * kNumMaxTokensPerRank + src_token_idx) * kNumTopk + route_idx);
                const uint32_t route_topk_slot_idx = packed & 0x3Fu;
                const uint32_t route_local_expert_idx = (packed >> 6) & 0x3FFu;
                const uint32_t route_token_idx_in_rank = packed >> 16;

                uint32_t route_token_idx_in_expert = token_idx - expert_start_idx;
                if (route_local_expert_idx != static_cast<uint32_t>(current_expert_idx)
                        or route_token_idx_in_rank != token_idx_in_rank) {
                    uint32_t route_rank_counts[kNumRanksPerLane];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                        const uint32_t j = i * 32 + lane_idx;
                        route_rank_counts[i] = j < kNumRanks
                            ? smem_per_rank_count[route_local_expert_idx * kNumRanks + j] : 0u;
                    }
                    route_token_idx_in_expert = utils::peel_inverse<kNumRanks>(
                        route_rank_counts, current_rank_in_expert_idx, route_token_idx_in_rank);
                }

                const uint32_t route_pool_block_offset = scheduler.get_pool_block_offset(route_local_expert_idx);
                const uint32_t pool_token_idx = route_pool_block_offset * BLOCK_M + route_token_idx_in_expert;
                const uint32_t sf_pool_token_idx = route_pool_block_offset * SF_BLOCK_M
                                                  + transform_sf_token_idx(route_token_idx_in_expert);
                stored_pool_blocks[route_idx] = route_pool_block_offset + route_token_idx_in_expert / BLOCK_M;
                const float weight = __shfl_sync(0xffffffffu, stored_weight, route_topk_slot_idx);
                __syncwarp();
                if (cute::elect_one_sync()) {
                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, route_topk_slot_idx};
                    cute::tma_store_arrive();
                }

                #pragma unroll
                for (uint32_t i = 0; i < kNumSFUint32PerLane; ++i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFUint32)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = stored_sf[i];
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(workspace.get_l1_arrival_count_ptr(stored_pool_blocks[route_idx]), 1);
                }
            }

            __syncwarp();
        }

        if (sm_idx == 0 and cumulative_local_expert_recv_stats != nullptr) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExpertsPerRank; i += kNumDispatchThreads) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
            }
        }
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void) local_expert_idx;
            (void) n_block_idx;
            if (block_phase != sched::BlockPhase::Linear1)
                return;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
            const auto expected = scheduler.template get_valid_m<false>();
            while (ptx::ld_acq(ptr) != expected);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx * (BLOCK_K / 128);
                if (not is_leader_cta)
                    m_idx += scheduler.template get_valid_m<true>() / 2;

                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        &tensor_map_l1_acts, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        &tensor_map_l1_acts_sf, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE * 2 + SMEM_SFA_SIZE_PER_STAGE * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void) m_block_idx;
            if (block_phase != sched::BlockPhase::Linear1)
                return;

            const auto shape_k = L1_SHAPE_K;
            const auto shape_n = L1_SHAPE_N;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx * (BLOCK_K / 128);

                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        &tensor_map_l1_weights, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 2);
                    tma::copy<BLOCK_N, 1, 0>(
                        &tensor_map_l1_weights_sf, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        if (is_leader_cta) {
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K>();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, UMMA_BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, UMMA_BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            uint32_t current_iter_idx = 0;
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                (void) local_expert_idx;
                (void) m_block_idx;
                (void) n_block_idx;
                if (block_phase != sched::BlockPhase::Linear1)
                    return;

                mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());

                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx++ / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++umma_k_block_idx) {
                            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i) {
                                auto smem_ptr = smem_sfa[stage_idx] + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i) {
                                auto smem_ptr = smem_sfb[stage_idx] + umma_k_block_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }

                            #pragma unroll
                            for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++k) {
                                const auto runtime_instr_desc =
                                    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                                a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                        a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M * sizeof(a_dtype_t), k * UMMA_K);
                                b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                                        b_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_N * sizeof(b_dtype_t), k * UMMA_K);
                                ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                    b_desc, a_desc, accum_stage_idx * UMMA_N,
                                    k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
                                    kTmemStartColOfSFB, kTmemStartColOfSFA);
                            }
                        }
                    }
                    __syncwarp();

                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            });

            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        uint32_t current_iter_idx = 0;
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void) local_expert_idx;
            (void) num_k_blocks;
            if (block_phase != sched::BlockPhase::Linear1)
                return;

            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx++ / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            uint32_t m_idx = pool_block_idx * BLOCK_M;

            float stored_cached_weight = 0;
            #pragma unroll
            for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++s) {
                if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    break;
                }

                float2 swiglu_values[kNumAtomsPerStore * 2];
                float2 amax_values[kNumAtomsPerStore];
                #pragma unroll
                for (uint32_t i = 0; i < kNumAtomsPerStore; ++i) {
                    const uint32_t j = s * kNumAtomsPerStore + i;

                    DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                    if ((j * ATOM_M) % 32 == 0 and (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                        stored_cached_weight = *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                            .get_base_ptr<float>();
                    }

                    const float2 weights = {
                        ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                        ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                    };

                    uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                    uint32_t values[ATOM_M];
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                           values[0], values[1], values[2], values[3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                           values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();

                    if (j == WG_BLOCK_M / ATOM_M - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    auto fp32_values = reinterpret_cast<float*>(values);
                    #pragma unroll
                    for (uint32_t k = 0; k < 2; ++k) {
                        auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                        auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                            bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                            bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                            bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                        }

                        auto gate = __bfloat1622float2(bf16_gate);
                        auto neg_gate_exp = make_float2(
                            kFastMath ? __expf(-gate.x) : expf(-gate.x),
                            kFastMath ? __expf(-gate.y) : expf(-gate.y));
                        const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                        if constexpr (kFastMath) {
                            gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                        } else {
                            gate = {gate.x / denom.x, gate.y / denom.y};
                        }
                        const auto up = __bfloat1622float2(bf16_up);
                        swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                    }

                    amax_values[i].x = math::warp_reduce<4, true>(
                        cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                        math::ReduceMax<float>());
                    amax_values[i].y = math::warp_reduce<4, true>(
                        cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                        math::ReduceMax<float>());
                    if (lane_idx < 4)
                        smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                    __syncwarp();
                }

                const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                #pragma unroll
                for (uint32_t i = 0; i < kNumAtomsPerStore; ++i) {
                    const float2 wp_amax =
                        smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                    amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                    amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                    float2 sf, sf_inv;
                    math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                    const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                    const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                    const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                    uint32_t row = lane_idx;
                    uint32_t col = warp_idx_in_wg;
                    const auto smem_ptr = smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                                                 + i * ATOM_M * L1_OUT_BLOCK_N
                                                                 + row * L1_OUT_BLOCK_N
                                                                 + (col ^ (row / 2)) * kNumBankGroupBytes;
                    ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                    if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                        const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                        const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                        const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                        const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                        const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        __builtin_assume(token_base_idx < BLOCK_M);
                        const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                            + m_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                        const auto sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                        sf_base_ptr[sf_addr] =
                            (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                        sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                            (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                    }
                    __syncwarp();
                }
                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                    uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                    cute::tma_store_fence();
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_l1_output,
                        smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                        out_n_idx,
                        m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }

            ptx::tma_store_wait<0>();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                ptx::red_or_rel_gpu(
                    workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                    1ull << n_block_idx);
            }
            __syncwarp();
        });

        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);
    }
#endif
}

} // namespace deep_gemm::mega_moe_split
