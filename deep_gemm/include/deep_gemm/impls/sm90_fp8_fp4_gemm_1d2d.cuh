#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t,
          bool kDecodeStub = false>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_fp4_gemm_1d2d_impl(int8_t* gmem_b_ptr, float* sfb, int* grouped_layout,
                            nv_bfloat16* gmem_d_ptr,
                            uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(
        math::constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or
        (math::constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    // Packed FP4 B is loaded by TMA into a separate buffer; each row is BLOCK_K / 2 bytes.
    static constexpr uint32_t BLOCK_K_PACKED = BLOCK_K / 2;
    static constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE = BLOCK_N * BLOCK_K_PACKED;
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t aligned_shape_n_sfb = math::align<uint32_t>(shape_n, 16u / sizeof(float));
    // SFB cache aliases smem_d when it fits. Small-M tiles may not have enough
    // smem_d capacity, so they fall back to a separate SFB region.
    const uint32_t smem_sfb_bytes = math::align<uint32_t>(shape_k_scales * BLOCK_N * sizeof(float), 16u);
    constexpr uint32_t COMPILED_SHAPE_K_SCALES = SHAPE_K == 0 ? 0 : math::constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t COMPILED_SMEM_SFB_BYTES =
        math::constexpr_align(COMPILED_SHAPE_K_SCALES * BLOCK_N * static_cast<uint32_t>(sizeof(float)), 16u);
    constexpr bool kUseSeparateSFB = SHAPE_K != 0 and COMPILED_SMEM_SFB_BYTES > SMEM_D_SIZE;
    constexpr uint32_t SMEM_SFB_SIZE = kUseSeparateSFB ? COMPILED_SMEM_SFB_BYTES : 0;

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    auto fp4_to_e4m3_bits = [](uint32_t code) -> uint32_t {
        // E2M1 mag {0..7} -> E4M3 magnitude bytes {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c}.
        // Use hardware __byte_perm as a single-cycle 8-entry byte LUT:
        //   LUT_LO[mag] = {0x00, 0x30, 0x38, 0x3c}, LUT_HI[mag-4] = {0x40, 0x44, 0x48, 0x4c}
        //   __byte_perm(LUT_LO, LUT_HI, mag) returns 4 copies of the byte at index `mag`
        //   (mag is in 0..7, fits in the lower nibble, used as the byte selector).
        // Sign: bit 3 of code shifted to MSB. We deliberately allow the FP4 "negative zero"
        // (mag=0, sign=1) to map to E4M3 0x80 (which is -0, not NaN). WGMMA treats -0 as 0,
        // so dropping the explicit mag!=0 mask saves ~4 ops per nibble (~128 per uint4 of
        // packed FP4) without affecting numerical results.
        constexpr uint32_t LUT_LO = 0x3c383000u;
        constexpr uint32_t LUT_HI = 0x4c484440u;
        const uint32_t mag      = code & 0x07u;
        const uint32_t mag_byte = __byte_perm(LUT_LO, LUT_HI, mag) & 0xffu;
        const uint32_t sign     = (code & 0x08u) << 4;          // 0 or 0x80
        return mag_byte | sign;
    };
    auto fp4_pair_to_e4m3_pair = [&](uint32_t packed) {
        const uint32_t lo = fp4_to_e4m3_bits(packed & 0x0fu);
        const uint32_t hi = fp4_to_e4m3_bits((packed >> 4) & 0x0fu);
        return lo | (hi << 8);
    };

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_B_PACKED_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_b_packed = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + SMEM_B_PACKED_OFFSET + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_B_PACKED_OFFSET + kNumStages * SMEM_B_PACKED_SIZE_PER_STAGE;
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SFB_OFFSET = SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE;
    // Prefer aliasing SFB onto smem_d. For small-M tiles smem_d is too small,
    // so allocate a separate SFB cache to enable BLOCK_M=64 masked kernels.
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + (kUseSeparateSFB ? SMEM_SFB_OFFSET : 0));
    if constexpr (not kUseSeparateSFB) {
        DG_TRAP_ONLY_DEVICE_ASSERT(smem_sfb_bytes <= SMEM_D_SIZE);
    }

    // Fill barriers.
    // After the A/packed-B barrier merge there is only one set of full/empty barriers.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_SFB_OFFSET + SMEM_SFB_SIZE);
    auto full_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // `gmem_b_ptr` is no longer used: B is now loaded by TMA into `smem_b_packed`.
    (void)gmem_b_ptr;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);
    auto get_current_group_idx = [&]() {
        if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return static_cast<uint32_t>(cute::max(0, grouped_layout[m_block_idx * BLOCK_M]));
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return scheduler.current_group_idx;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return scheduler.current_group_idx;
        } else {
            return 0u;
        }
    };

    // Pipeline and TMA phases (single shared pipeline for A/SFA/packed-B after the barrier merge)
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used.
        // We use the third warp, as warp 0/1 may be doing WGMMA with `BLOCK_M == 32`.
        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Assign TMA multicast number into A and B
                // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait consumer release for the (now merged) A/SFA/packed-B slot
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA A
                    constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                    const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                    constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_a, &full_barrier,
                             smem_a[stage_idx], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
                             num_tma_multicast_a, batch_idx);
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier,
                             smem_sfa[stage_idx], m_block_idx * BLOCK_M, scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales, 1, k_block_idx),
                             num_tma_multicast_a);

                    // Issue TMA B (packed FP4 bytes loaded as raw uint8 via FP8 alias) on the same barrier
                    const uint32_t k_idx_packed = k_block_idx * BLOCK_K_PACKED;
                    tma::copy<BLOCK_K_PACKED, BLOCK_N, 0, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full_barrier,
                             reinterpret_cast<__nv_fp8_e4m3*>(smem_b_packed[stage_idx]),
                             k_idx_packed,
                             scheduler.template get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b, batch_idx);

                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE +
                                                     SMEM_B_PACKED_SIZE_PER_STAGE);
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; ++ i) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                    phase ^= stage_idx == 0;
                }
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t current_group_idx = get_current_group_idx();

            // Cooperatively prefetch the SFB tile for this block from gmem to smem.
            // Layout in smem: [shape_k_scales, BLOCK_N] (k outer, n inner).
            // Out-of-bound n is filled with 1.0f to keep `n_idx >= shape_n` neutral.
            //
            // Optimization: use cp.async to copy gmem->smem directly (no register
            // round-trip). For MN-major (n innermost in gmem) we can issue 16-byte
            // (float4) cp.async per thread, cutting #instructions by 4x. K-major
            // and the OOB tail use scalar 4-byte cp.async / st.shared.
            {
                const uint32_t n_block_base = n_block_idx * BLOCK_N;
                if constexpr (kMajorSFB == cute::UMMA::Major::MN) {
                    constexpr uint32_t kVec = 4;
                    DG_STATIC_ASSERT(BLOCK_N % kVec == 0,
                                     "BLOCK_N must be a multiple of 4 for vectorized SFB load");
                    constexpr uint32_t kVecsPerRow = BLOCK_N / kVec;
                    const uint32_t total_vecs = shape_k_scales * kVecsPerRow;
                    const float* sfb_base = sfb +
                        current_group_idx * aligned_shape_n_sfb * shape_k_scales;
                    // Issue cp.async (16B per thread) for fully in-bounds vectors;
                    // fall back to scalar st.shared with 1.0f-fill for the tail.
                    for (uint32_t i = threadIdx.x; i < total_vecs; i += kNumMathThreads) {
                        const uint32_t k_idx = i / kVecsPerRow;
                        const uint32_t vec_n = i % kVecsPerRow;
                        const uint32_t n_off = vec_n * kVec;
                        const uint32_t n_idx = n_block_base + n_off;
                        float* smem_dst = smem_sfb + k_idx * BLOCK_N + n_off;
                        if (n_idx + kVec <= shape_n) {
                            const float* gmem_src = sfb_base +
                                k_idx * aligned_shape_n_sfb + n_idx;
                            ptx::cp_async_16(smem_dst, gmem_src);
                        } else {
                            // Tail: at least one element is OOB; use scalar st with 1.0f fill.
                            float4 vals;
                            vals.x = (n_idx + 0 < shape_n) ?
                                sfb_base[k_idx * aligned_shape_n_sfb + n_idx + 0] : 1.0f;
                            vals.y = (n_idx + 1 < shape_n) ?
                                sfb_base[k_idx * aligned_shape_n_sfb + n_idx + 1] : 1.0f;
                            vals.z = (n_idx + 2 < shape_n) ?
                                sfb_base[k_idx * aligned_shape_n_sfb + n_idx + 2] : 1.0f;
                            vals.w = (n_idx + 3 < shape_n) ?
                                sfb_base[k_idx * aligned_shape_n_sfb + n_idx + 3] : 1.0f;
                            ptx::st_shared(reinterpret_cast<float4*>(smem_dst), vals);
                        }
                    }
                } else {
                    // K-major: sfb is strided along n; cannot easily vectorize across n.
                    // Use scalar 4B cp.async for in-bounds, st.shared with 1.0f for OOB.
                    const uint32_t total = shape_k_scales * BLOCK_N;
                    for (uint32_t i = threadIdx.x; i < total; i += kNumMathThreads) {
                        const uint32_t k_idx = i / BLOCK_N;
                        const uint32_t n_off = i % BLOCK_N;
                        const uint32_t n_idx = n_block_base + n_off;
                        float* smem_dst = smem_sfb + k_idx * BLOCK_N + n_off;
                        if (n_idx >= shape_n) {
                            ptx::st_shared(smem_dst, 1.0f);
                        } else {
                            const float* gmem_src = sfb +
                                current_group_idx * shape_n * shape_k_scales +
                                n_idx * shape_k_scales + k_idx;
                            ptx::cp_async_4(smem_dst, gmem_src);
                        }
                    }
                }
                // Commit and wait for all cp.async issued above; pair with the
                // existing NamedBarrier so the smem cache is visible to all
                // math warps before they enter the K loop.
                ptx::cp_async_commit_group();
                ptx::cp_async_wait_group<0>();
                cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);
            }

            auto load_sfb = [&](uint32_t n_idx, uint32_t k_block_idx) {
                // SFB has been staged into smem above; out-of-bound `n_idx` already
                // resolves to 1.0f because we wrote 1.0f for those slots.
                const uint32_t n_off = n_idx - n_block_idx * BLOCK_N;
                return ptx::ld_shared(smem_sfb + k_block_idx * BLOCK_N + n_off);
            };

            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
            }

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
            
            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // Skip useless computations
            const bool is_cta_computation_valid = scheduler.is_computation_valid(m_block_idx, 0);
            if (is_cta_computation_valid) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / math::constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = math::constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // Decode helper: wait for the (merged) full[s] barrier, then
                // decode packed FP4 B from `smem_b_packed[s]` into the swizzled
                // `smem_b[s]`. Caller must subsequently issue
                // NamedBarrier::sync(...,7) to publish the decoded data to the
                // wgmma async-proxy. The packed slot is released together with
                // A/SFA via the single empty_barrier_arrive() after wgmma_wait.
                auto wait_and_decode = [&](uint32_t s, uint32_t p) {
                    full_barriers[s]->wait(p);
                    auto smem_b_packed_bytes = smem_b_packed[s];
                    auto smem_b_bytes = reinterpret_cast<uint8_t*>(smem_b[s]);
                    constexpr uint32_t kVecPackedBytes = 16;  // 16 packed bytes -> 32 e4m3 bytes
                    constexpr uint32_t kVecsPerRow = BLOCK_K_PACKED / kVecPackedBytes;
                    constexpr uint32_t kNumVecs = BLOCK_N * kVecsPerRow;
                    DG_STATIC_ASSERT(BLOCK_K_PACKED % kVecPackedBytes == 0,
                                     "Packed K must be multiple of 16-byte vector width");
                    DG_STATIC_ASSERT(BLOCK_K == 128,
                                     "Swizzle assumes BLOCK_K == 128 so 32B store stays in-range");
                    if constexpr (not kDecodeStub) {
                        for (uint32_t idx = threadIdx.x; idx < kNumVecs; idx += kNumMathThreads) {
                            const uint32_t tile_n = idx / kVecsPerRow;
                            const uint32_t vec_k  = idx % kVecsPerRow;
                            const uint32_t tile_k = vec_k * (kVecPackedBytes * 2);  // 32-byte step in K
                            const uint4 packed16 = *reinterpret_cast<const uint4*>(
                                smem_b_packed_bytes + tile_n * BLOCK_K_PACKED + vec_k * kVecPackedBytes);
                            uint64_t decoded[4] = {0, 0, 0, 0};
                            auto decode_u32 = [&](uint32_t packed_word, uint64_t& out) {
                                #pragma unroll
                                for (uint32_t b = 0; b < 4; ++ b) {
                                    const uint32_t pair = fp4_pair_to_e4m3_pair((packed_word >> (b * 8)) & 0xffu);
                                    out |= static_cast<uint64_t>(pair) << (b * 16);
                                }
                            };
                            decode_u32(packed16.x, decoded[0]);
                            decode_u32(packed16.y, decoded[1]);
                            decode_u32(packed16.z, decoded[2]);
                            decode_u32(packed16.w, decoded[3]);
                            const uint32_t n_group = tile_n / 8;
                            const uint32_t n_in_group = tile_n % 8;
                            const uint32_t row_base = n_group * 8 * BLOCK_K + n_in_group * BLOCK_K;
                            // Each segment writes 16 bytes (= 2 u64). Use a
                            // single st.shared.v2.u64 instead of two
                            // st.shared.u64, halving the store instruction
                            // count for the decoded B tile.
                            {
                                const uint32_t swizzled_k = tile_k ^ (n_in_group * 16);
                                ptx::st_shared_v2_u64(smem_b_bytes + row_base + swizzled_k,
                                                      decoded[0], decoded[1]);
                            }
                            {
                                const uint32_t swizzled_k = (tile_k + 16) ^ (n_in_group * 16);
                                ptx::st_shared_v2_u64(smem_b_bytes + row_base + swizzled_k,
                                                      decoded[2], decoded[3]);
                            }
                        }
                    }
                };


                // Dispatch `num_former_iters` and launch MMAs with decode/wgmma
                // pipeline overlap: decode runs one stage ahead of wgmma so that
                // each iter's wait+decode for stage k+1 hides under iter k's
                // wgmma async work.
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    // Lead pointers: track the stage that decode is currently
                    // working on (one ahead of wgmma's `stage_idx`).
                    uint32_t lead_stage = stage_idx, lead_phase = phase;
                    auto advance_lead = [&]() {
                        lead_stage = lead_stage == kNumStages - 1 ? 0 : lead_stage + 1;
                        lead_phase ^= lead_stage == 0;
                    };

                    // Prologue: decode the first stage so iter 0's wgmma can
                    // consume it without waiting.
                    wait_and_decode(stage_idx, phase);
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 7);
                    advance_lead();  // lead now points one stage ahead of wgmma

                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        // smem_b[stage_idx] is already decoded (prologue or the
                        // previous iter's hide-decode) and made visible by the
                        // matching NamedBarrier::sync below.
                        // smem_a[stage_idx] / smem_sfa[stage_idx] are guaranteed
                        // ready by the wait_and_decode that targeted stage_idx.

                        float scale_b_0_regs[WGMMA::kNumAccum / 4];
                        float scale_b_1_regs[WGMMA::kNumAccum / 4];
                        if (do_wgmma_store) {
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                const uint32_t n_idx = n_block_idx * BLOCK_N + i * 8 + col_idx * 2;
                                scale_b_0_regs[i] = load_sfb(n_idx, k_block_idx);
                                scale_b_1_regs[i] = load_sfb(n_idx + 1, k_block_idx);
                            }
                        }

                        const bool has_next = (k_block_idx + 1) < num_total_k_blocks;

                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // Read A scales
                            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                            auto scale_a_0 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
                            auto scale_a_1 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_1 + m_offset) : 0;

                            // Commit WGMMA instructions
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                WGMMA::wgmma(a_desc, b_desc, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);

                            // Hide decode of next stage under this wgmma's async
                            // work. Run only on the last wave so all warps follow
                            // the same wait/decode/sync sequence per K iter.
                            const bool is_last_wave = (local_idx == BLOCK_M / WAVE_BLOCK_M - 1);
                            if (is_last_wave and has_next) {
                                wait_and_decode(lead_stage, lead_phase);
                            }

                            ptx::warpgroup_wait<0>();

                            // Notify barrier arrival at the last warpgroup wave
                            if (is_last_wave)
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store)
                                continue;

                            // Promote with scales
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                const float scale_b_0 = scale_b_0_regs[i];
                                const float scale_b_1 = scale_b_1_regs[i];
                                shifted_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
                            }
                        }

                        // Publish next iter's decoded smem_b to the wgmma
                        // async-proxy. The packed-B slot is released together
                        // with A/SFA via the unified empty_barrier_arrive() above.
                        if (has_next) {
                            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 7);
                            advance_lead();
                        }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            // Psum layout can have a final partial M tile. TMA store writes a
            // full BLOCK_M tile and may go out of the tensor-map bounds, so use
            // a guarded scalar store for this layout.
            if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
                const uint32_t psum_store_end =
                    scheduler.current_group_idx + 1 < kNumGroups ?
                    math::align(scheduler.current_psum_m, BLOCK_M) : scheduler.current_psum_m;
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                    const uint32_t row_0 = m_block_idx * BLOCK_M + m_offset + r_0;
                    const uint32_t row_1 = m_block_idx * BLOCK_M + m_offset + r_1;
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        const uint32_t base_col = epilogue_type_t::template apply_index_n<8>(
                            n_block_idx * BLOCK_N + i * 8);
                        const uint32_t col = base_col + col_idx * 2;
                        const bool row_0_valid = (row_0 >= scheduler.last_psum_m and row_0 < psum_store_end);
                        const bool row_1_valid = (row_1 >= scheduler.last_psum_m and row_1 < psum_store_end);
                        if (row_0_valid and col < shape_n)
                            gmem_d_ptr[row_0 * shape_n + col] = __float2bfloat16_rn(shifted_accum[i * 4 + 0]);
                        if (row_0_valid and col + 1 < shape_n)
                            gmem_d_ptr[row_0 * shape_n + col + 1] = __float2bfloat16_rn(shifted_accum[i * 4 + 1]);
                        if (row_1_valid and col < shape_n)
                            gmem_d_ptr[row_1 * shape_n + col] = __float2bfloat16_rn(shifted_accum[i * 4 + 2]);
                        if (row_1_valid and col + 1 < shape_n)
                            gmem_d_ptr[row_1 * shape_n + col + 1] = __float2bfloat16_rn(shifted_accum[i * 4 + 3]);
                    }
                }
                __syncwarp();
                continue;
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // No swizzling, just padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                  n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
