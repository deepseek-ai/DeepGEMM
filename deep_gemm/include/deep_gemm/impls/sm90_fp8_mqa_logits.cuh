#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          typename logits_dtype_t, bool kInterleaveQHeads = false>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                         const uint32_t stride_logits,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         logits_dtype_t* logits,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(not kInterleaveQHeads or
                     (kNumHeads == 32 and kHeadDim == 128 and BLOCK_Q == 4 and kNumMathThreads == 512),
                     "Invalid interleaved Q configuration");

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });

    // Initialize barriers
    const bool is_tma_load_warp = kNumMathThreads <= threadIdx.x and threadIdx.x < kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
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

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // Block scheduler
    const auto sm_idx = blockIdx.x;
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, math::ceil_div(end - start, BLOCK_KV)};          // Task info
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Only the first warp remains
        if (not is_tma_load_warp)
            return;

        // Prefetch
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = ptx::get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q * kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            const auto own_token = lane_idx % 4;
            const auto own_q_idx = block_q_idx * BLOCK_Q + own_token;
            const auto own_q_offset = own_q_idx * static_cast<uint64_t>(stride_logits);
            uint32_t own_seq_k_start = 0, own_seq_k_end = 0;
            if constexpr (kInterleaveQHeads) {
                // Select with compile-time array indices once per Q block. A
                // dynamic seq_k_start[own_token] in the KV loop would place the
                // small arrays in local memory instead of keeping them in registers.
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    own_seq_k_start = own_token == i ? seq_k_start[i] : own_seq_k_start;
                    own_seq_k_end = own_token == i ? seq_k_end[i] : own_seq_k_end;
                }
            }

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            if constexpr (kInterleaveQHeads) {
                // Interleave pairs of heads from the four query tokens in shared memory:
                // [token, head_pair, head_in_pair, dim] ->
                // [head_pair, token, head_in_pair, dim]. Each lane of a quad then owns
                // all 32 heads of one token in the WGMMA accumulator. TMA and the
                // public Q layout stay unchanged; no extra kernel or buffer is needed.
                static constexpr uint32_t kVectorBytes = sizeof(uint4);
                static constexpr uint32_t kVectorsPerRow = kHeadDim / kVectorBytes;
                static constexpr uint32_t kVectorsPerThread = SMEM_Q_SIZE_PER_STAGE / (kNumMathThreads * kVectorBytes);
                uint4 fragments[kVectorsPerThread];

                // Both accesses account for TMA's 128-byte shared-memory swizzle.
                const auto q_vector_offset = [&](const uint32_t row, const uint32_t col) {
                    return row * kHeadDim + (col ^ (row & 7)) * kVectorBytes;
                };
                #pragma unroll
                for (uint32_t i = 0; i < kVectorsPerThread; ++ i) {
                    const auto vector_idx = thread_idx + i * kNumMathThreads;
                    const auto row = vector_idx / kVectorsPerRow;
                    const auto col = vector_idx % kVectorsPerRow;
                    fragments[i] = ptx::ld_shared(reinterpret_cast<const uint4*>(smem_q[q_stage_idx] + q_vector_offset(row, col)));
                }
                // Finish every read before overwriting the same shared-memory tile.
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
                #pragma unroll
                for (uint32_t i = 0; i < kVectorsPerThread; ++ i) {
                    const auto vector_idx = thread_idx + i * kNumMathThreads;
                    const auto row = vector_idx / kVectorsPerRow;
                    const auto col = vector_idx % kVectorsPerRow;
                    const auto token = row / kNumHeads;
                    const auto head = row % kNumHeads;
                    const auto interleaved_row = (head / 2) * (BLOCK_Q * 2) + token * 2 + (head & 1);
                    const auto& fragment = fragments[i];
                    ptx::st_shared(smem_q[q_stage_idx] + q_vector_offset(interleaved_row, col),
                                   fragment.x, fragment.y, fragment.z, fragment.w);
                }
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                const auto token = lane_idx % 4;
                #pragma unroll
                for (uint32_t h = 0; h < kNumHeads; ++ h)
                    weights[h] = ptx::ld_shared(smem_weights[q_stage_idx] + token * kNumHeads + h);
            } else {
                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i * (kNumHeads / 4) + j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv_0, scale_kv_1;
                if constexpr (kInterleaveQHeads) {
                    const auto scale_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                        smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset));
                    // Expose the implicit shared-memory reads to the compiler, so
                    // they stay before the generic-to-async fence below.
                    asm volatile("ld.shared.f32 %0, [%2];\n\t"
                                 "ld.shared.f32 %1, [%2 + 32];"
                                 : "=f"(scale_kv_0), "=f"(scale_kv_1)
                                 : "r"(scale_addr) : "memory");
                } else {
                    scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                    scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);
                }

                // Issue WGMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[q_stage_idx] + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                if constexpr (not kInterleaveQHeads) {
                    // Release KV empty
                    empty_kv_barriers[kv_stage_idx]->arrive();
                }

                // Reduce over the head dim and store
                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                if constexpr (kInterleaveQHeads) {
                    DG_STATIC_ASSERT(WGMMA::kNumAccum == kNumHeads * 2, "Invalid accumulation");
                    // Eight independent accumulation chains avoid a long FFMA dependency.
                    float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    #pragma unroll
                    for (uint32_t h = 0; h < kNumHeads / 2; ++ h) {
                        const auto p = (h & 1) * 4;
                        sum[p + 0] += fmaxf(accum[h * 4 + 0], 0) * weights[h * 2 + 0];
                        sum[p + 1] += fmaxf(accum[h * 4 + 1], 0) * weights[h * 2 + 1];
                        sum[p + 2] += fmaxf(accum[h * 4 + 2], 0) * weights[h * 2 + 0];
                        sum[p + 3] += fmaxf(accum[h * 4 + 3], 0) * weights[h * 2 + 1];
                    }
                    const auto v_0 = ((sum[0] + sum[1]) + (sum[4] + sum[5])) * scale_kv_0;
                    const auto v_1 = ((sum[2] + sum[3]) + (sum[6] + sum[7])) * scale_kv_1;
                    if (own_q_idx < seq_len and own_seq_k_start <= kv_offset + v_0_offset and kv_offset + v_0_offset < own_seq_k_end)
                        logits[own_q_offset + kv_offset + v_0_offset - own_seq_k_start] = static_cast<logits_dtype_t>(v_0);
                    if (own_q_idx < seq_len and own_seq_k_start <= kv_offset + v_1_offset and kv_offset + v_1_offset < own_seq_k_end)
                        logits[own_q_offset + kv_offset + v_1_offset - own_seq_k_start] = static_cast<logits_dtype_t>(v_1);

                    // Order generic-proxy scale reads before TMA's async-proxy
                    // overwrite. The existing release-arrive/acquire-wait handshake
                    // counts every reader before the producer reuses this stage.
                    cutlass::arch::fence_view_async_shared();
                    empty_kv_barriers[kv_stage_idx]->arrive();
                } else {
                    static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                    DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                    DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                    DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        auto shifted_accum = accum + i * kNumAccumPerReduce;
                        const auto transform = [&](const uint32_t& j) {
                            return fmaxf(shifted_accum[j], 0) * weights[i * (kNumHeads / 4) + (j / 4) * 2 + (j & 1)];
                        };

                        // Intra-thread reduction
                        float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                        #pragma unroll
                        for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                            #pragma unroll
                            for (uint32_t k = 0; k < 4; k ++)
                                sum[k] += transform(j * 4 + k);
                        }
                        float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                        float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                        // Inter-thread reduction
                        #pragma unroll
                        for (uint32_t j = 0; j < 2; ++ j) {
                            const auto& offset = static_cast<int>(1u << j);
                            v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                            v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                        }

                        // Store into the global memory
                        const auto q_offset = (block_q_idx * BLOCK_Q + i) * static_cast<uint64_t>(stride_logits);
                        if (seq_k_start[i] <= kv_offset + v_0_offset and kv_offset + v_0_offset < seq_k_end[i])
                            logits[q_offset + kv_offset + v_0_offset - seq_k_start[i]] = static_cast<logits_dtype_t>(v_0);
                        if (seq_k_start[i] <= kv_offset + v_1_offset and kv_offset + v_1_offset < seq_k_end[i])
                            logits[q_offset + kv_offset + v_1_offset - seq_k_start[i]] = static_cast<logits_dtype_t>(v_1);
                    }
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
}

} // namespace deep_gemm
