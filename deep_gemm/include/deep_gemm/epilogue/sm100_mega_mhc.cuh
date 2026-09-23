#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/scheduler/mega_mhc.cuh>

namespace deep_gemm::epilogue::mega_mhc {

using namespace layout::mega_mhc;

// Reduce one token's split partials and produce its Mix coefficients.
template <uint32_t kHidden, bool kIsShifted, uint32_t kNumSplits>
CUTLASS_DEVICE float run_mix_task(const Workspace<kNumSplits>& workspace, uint64_t* gmem_split_barriers,
                                  const MixArgs& mix_args,
                                  const uint32_t& token_idx, const uint32_t& lane_idx) {
    const uint32_t m_block_idx = token_idx / BLOCK_M;
    const uint32_t row_idx = token_idx % BLOCK_M;
    sched::mega_mhc::Mix::wait<kNumSplits>(gmem_split_barriers, m_block_idx);

    // Reduce HC norm statistics and GEMM outputs across the split tasks.
    const bool is_hc_output_lane = lane_idx < kNumHCOutputs;
    const uint32_t first_task_idx = m_block_idx * kNumSplits;
    const auto hc_norm_sqr_sum_partials = workspace.get_hc_norm_sqr_sum_partial_ptr(first_task_idx);
    float hc_norm_sqr_sum = 0.0f;
    #pragma unroll
    for (uint32_t k_split_idx = lane_idx; k_split_idx < kNumSplits; k_split_idx += 32)
        hc_norm_sqr_sum += hc_norm_sqr_sum_partials[
            k_split_idx * kNumSqrSumPartialElementsPerTask + row_idx];
    hc_norm_sqr_sum = math::warp_reduce_sum(hc_norm_sqr_sum);

    const uint32_t safe_hc_output_idx = cute::min(lane_idx, kNumHCOutputs - 1);
    const auto gemm_partials = workspace.get_gemm_partial_ptr(first_task_idx);
    float hc_output = 0.0f;
    #pragma unroll
    for (uint32_t k_split_idx = 0; k_split_idx < kNumSplits; ++ k_split_idx)
        hc_output += gemm_partials[k_split_idx * kNumGemmPartialElementsPerTask +
                                   row_idx * kNumHCOutputs + safe_hc_output_idx];
    hc_output = is_hc_output_lane ? hc_output : 0.0f;
    const float hc_rms_scale = rsqrtf(hc_norm_sqr_sum * (1.0f / static_cast<float>(kNumRoutes * kHidden)) + mix_args.hc_norm_eps);

    // Warp lanes map to [Pre routes | Post routes | Comb routes x routes | inactive].
    constexpr uint32_t kFirstCombLane = 2 * kNumRoutes;
    const bool is_pre_lane = lane_idx < kNumRoutes;
    const bool is_pre_or_post_lane = lane_idx < kFirstCombLane;
    const bool is_comb_lane = is_hc_output_lane and not is_pre_or_post_lane;
    float mix_value = 0.0f;
    if (is_hc_output_lane) {
        const uint32_t mix_scale_idx = is_pre_lane ? 0u : is_pre_or_post_lane ? 1u : 2u;
        const float affine_value = hc_output * hc_rms_scale * mix_args.scales[mix_scale_idx] + mix_args.bases[lane_idx];
        if (is_pre_or_post_lane) {
            const float sigmoid = math::fast_rcp(1.0f + __expf(-affine_value));
            mix_value = is_pre_lane ? sigmoid + mix_args.hc_pre_eps : sigmoid * mix_args.hc_post_scale;
        } else {
            mix_value = affine_value;
        }
    }
    __syncwarp();

    const auto sinkhorn_normalize = [&](float value) {
        // Comb logits occupy a row-major route x route region after Pre and Post.
        value = is_comb_lane ? value : 0.0f;
        const float row_max = math::warp_reduce_max<kNumRoutes>(value);
        value = __expf(value - row_max);
        float row_sum = math::warp_reduce_sum<kNumRoutes>(value);
        value = value * math::fast_rcp(row_sum) + mix_args.sinkhorn_eps;

        const auto get_col_sum = [](const float& element) {
            // For four routes, these XORs combine adjacent rows, then the two row pairs.
            float sum = element + __shfl_xor_sync(0xffffffff, element, kNumRoutes);
            sum += __shfl_xor_sync(0xffffffff, sum, 6 * kNumRoutes);
            return sum;
        };

        value *= math::fast_rcp(get_col_sum(value) + mix_args.sinkhorn_eps);
        #pragma unroll 1
        for (uint32_t iteration = 1; iteration < mix_args.num_sinkhorn_iters; ++ iteration) {
            row_sum = math::warp_reduce_sum<kNumRoutes>(value);
            value *= math::fast_rcp(row_sum + mix_args.sinkhorn_eps);
            value *= math::fast_rcp(get_col_sum(value) + mix_args.sinkhorn_eps);
        }
        return value;
    };

    const float normalized_comb = sinkhorn_normalize(mix_value);

    // Only Shifted stores the new Pre coefficients.
    if (is_hc_output_lane and (kIsShifted or not is_pre_lane)) {
        float* mix_output = is_pre_lane ? mix_args.new_prev_mix :
                            is_comb_lane ? mix_args.new_comb_res_mix : mix_args.new_post_mix;
        const uint32_t mix_output_stride = is_comb_lane ? kNumRoutes * kNumRoutes : kNumRoutes;
        const uint32_t mix_output_idx = is_comb_lane ? lane_idx - kFirstCombLane : lane_idx & (kNumRoutes - 1);
        mix_output[token_idx * mix_output_stride + mix_output_idx] = is_comb_lane ? normalized_comb : mix_value;
    }
    __syncwarp();

    return mix_value;
}

// Every warp-wide pack spans 256 hidden elements.
constexpr uint32_t kHiddenPerNormPack = 256;
constexpr uint32_t kHiddenPerNormLane = kHiddenPerNormPack / 32;
constexpr uint32_t kNumBF16PairsPerLane = sizeof(uint4) / sizeof(nv_bfloat162);
constexpr uint32_t kNumLanesPerSF = kHiddenPerSF / kHiddenPerNormLane;
constexpr uint32_t kNumSFWordsPerNormPack = kHiddenPerNormPack / kHiddenPerSFWord;
constexpr uint32_t kNumLanesPerSFWord = kHiddenPerSFWord / kHiddenPerNormLane;

// Shared normalization and FP8 output helpers.
CUTLASS_DEVICE nv_bfloat162 normalize_bf16_pair(const nv_bfloat162& input, const nv_bfloat162& weight, const float& scale) {
    return __float22bfloat162_rn(__fmul2_rn(__fmul2_rn(__bfloat1622float2(input), {scale, scale}), __bfloat1622float2(weight)));
}

template <uint32_t SF_BLOCK_M>
CUTLASS_DEVICE uint32_t get_shared_sf_row_idx(const uint32_t& token_idx) {
    // Mega MoE SF pages interleave four 32-row groups within each 128-row page.
    const uint32_t idx = token_idx % SF_BLOCK_M;
    return token_idx / SF_BLOCK_M * ((SF_BLOCK_M + 127u) & ~127u) +
           (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
}

CUTLASS_DEVICE uint32_t store_fp8_and_pack_sf(const NormArgs& norm_args, const uint64_t& output_offset,
                                              const uint4& values, const uint32_t& sf_exp) {
    const auto bf16x2 = reinterpret_cast<const nv_bfloat162*>(&values);
    // Splat the BF16 sf_inv into both packed halves
    const auto sf_inv = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp));
    const uint2 fp8_values = {math::scale_bf16x2_into_fp8x4(bf16x2[0], bf16x2[1], sf_inv, sf_inv),
                              math::scale_bf16x2_into_fp8x4(bf16x2[2], bf16x2[3], sf_inv, sf_inv)};
    *reinterpret_cast<uint2*>(norm_args.y_fp8 + output_offset) = fp8_values;

    // Pack four 32-element SF groups from each 16-lane group into one int32.
    const uint32_t paired_sf = __byte_perm(sf_exp, __shfl_xor_sync(0xffffffff, sf_exp, kNumLanesPerSF), 0x1140);
    return __byte_perm(paired_sf, __shfl_xor_sync(0xffffffff, paired_sf, 2 * kNumLanesPerSF), 0x5410);
}

// A Normal task fuses Mix, Pre, and RMSNorm for one token.
template <uint32_t kHidden, uint32_t kNumSMs, bool kStoreBF16, bool kStoreFP8,
          uint32_t SF_BLOCK_M, uint32_t kNumSplits>
CUTLASS_DEVICE void run_normal_worker(SharedStorage& smem, const Workspace<kNumSplits>& workspace,
                                      const uint32_t& wg_idx_in_cta,
                                      const uint32_t& warp_idx_in_wg, const uint32_t& lane_idx,
                                      uint64_t* gmem_split_barriers,
                                      const MixArgs& mix_args,
                                      const NormArgs& norm_args) {
    constexpr uint32_t kNumWGsPerCTA = kNumThreads / (kNumWarpsPerWG * 32);
    constexpr uint32_t kNumHiddenPacksPerWarp = kHidden / (kHiddenPerNormPack * kNumWarpsPerWG);
    DG_STATIC_ASSERT(kHidden % (kHiddenPerNormPack * kNumWarpsPerWG) == 0, "Hidden size must divide evenly across Normal warps");
    constexpr uint32_t kNumWarpSqrSums = kNumWGsPerCTA * kNumWarpsPerWG;
    DG_STATIC_ASSERT(kNumWarpSqrSums * sizeof(float) +
                     kNumWGsPerCTA * (sizeof(uint32_t) + kNumRoutes * sizeof(float)) <= sizeof(smem.normal_scratch),
                     "Normal worker scratch is too small");

    // Scratch is laid out as [warp sums | token indices | Pre coefficients].
    const uint32_t hidden_pack_begin = warp_idx_in_wg * kNumHiddenPacksPerWarp;
    const uint32_t worker_barrier_idx = 1 + wg_idx_in_cta;
    const auto warp_sqr_sums = smem.normal_scratch;
    const auto token_indices = reinterpret_cast<uint32_t*>(warp_sqr_sums + kNumWarpSqrSums);
    const auto pre_coeffs = reinterpret_cast<float*>(token_indices + kNumWGsPerCTA) + wg_idx_in_cta * kNumRoutes;

    while (true) {
        if (warp_idx_in_wg == 0) {
            uint32_t task_ticket;
            if (cute::elect_one_sync())
                task_ticket = atomicAdd(&smem.next_norm_task_ticket, 1u);
            task_ticket = ptx::exchange(task_ticket, 0);
            const uint32_t token_idx = blockIdx.x + task_ticket * kNumSMs;
            token_indices[wg_idx_in_cta] = token_idx;

            // Mix is warp-sized; only its Pre coefficients must be broadcast.
            if (token_idx < norm_args.num_tokens) {
                const float pre_coeff = run_mix_task<kHidden, false>(
                    workspace, gmem_split_barriers, mix_args, token_idx, lane_idx);
                if (lane_idx < kNumRoutes)
                    pre_coeffs[lane_idx] = pre_coeff;
            }
        }
        cutlass::arch::NamedBarrier::sync(kNumWarpsPerWG * 32, worker_barrier_idx);
        const uint32_t token_idx = token_indices[wg_idx_in_cta];
        if (token_idx >= norm_args.num_tokens)
            return;
        const float pre_coeff = lane_idx < kNumRoutes ? pre_coeffs[lane_idx] : 0.0f;

        // Accumulate routes in FP32, then derive RMS from the BF16-rounded values.
        const auto token_residual = norm_args.new_residual + static_cast<uint64_t>(token_idx) * kNumRoutes * kHidden;
        uint4 retained[kNumHiddenPacksPerWarp];
        float2 sqr_sum = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (uint32_t hidden_pack_offset = 0; hidden_pack_offset < kNumHiddenPacksPerWarp; ++ hidden_pack_offset) {
            const uint32_t hidden_pack_idx = hidden_pack_begin + hidden_pack_offset;
            const uint32_t hidden_idx = hidden_pack_idx * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
            const auto output = reinterpret_cast<nv_bfloat162*>(&retained[hidden_pack_offset]);
            float2 values[kNumBF16PairsPerLane] = {};
            #pragma unroll
            for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
                const float coeff = ptx::exchange(pre_coeff, route_idx);
                const uint4 input = ptx::ld_evict_first(
                    reinterpret_cast<const uint4*>(token_residual + route_idx * kHidden + hidden_idx));
                const auto pairs = reinterpret_cast<const nv_bfloat162*>(&input);
                #pragma unroll
                for (uint32_t pair_idx = 0; pair_idx < kNumBF16PairsPerLane; ++ pair_idx)
                    values[pair_idx] = __ffma2_rn(__bfloat1622float2(pairs[pair_idx]), {coeff, coeff}, values[pair_idx]);
            }
            #pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < kNumBF16PairsPerLane; ++ pair_idx) {
                output[pair_idx] = __float22bfloat162_rn(values[pair_idx]);
                values[pair_idx] = __bfloat1622float2(output[pair_idx]);
                sqr_sum = __ffma2_rn(values[pair_idx], values[pair_idx], sqr_sum);
            }
        }

        const float warp_sqr_sum = math::warp_reduce_sum(sqr_sum.x + sqr_sum.y);
        warp_sqr_sums[wg_idx_in_cta * kNumWarpsPerWG + warp_idx_in_wg] = warp_sqr_sum;
        cutlass::arch::NamedBarrier::sync(kNumWarpsPerWG * 32, worker_barrier_idx);
        float rms_scale = lane_idx < kNumWarpsPerWG ? warp_sqr_sums[wg_idx_in_cta * kNumWarpsPerWG + lane_idx] : 0.0f;
        rms_scale = math::warp_reduce_sum<kNumWarpsPerWG>(rms_scale);
        rms_scale = ptx::exchange(rms_scale, 0);
        constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
        rms_scale = rsqrtf(rms_scale * kInvHidden + norm_args.eps) * norm_args.scale;
        const uint64_t output_offset = static_cast<uint64_t>(token_idx) * kHidden;
        uint32_t shared_sf_row_idx = 0;
        if constexpr (SF_BLOCK_M > 0)
            shared_sf_row_idx = get_shared_sf_row_idx<SF_BLOCK_M>(token_idx);

        const auto weight_packs = reinterpret_cast<const uint4*>(
            norm_args.weight + hidden_pack_begin * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane);
        uint4 weight = weight_packs[0];
        #pragma unroll
        for (uint32_t hidden_pack_offset = 0; hidden_pack_offset < kNumHiddenPacksPerWarp; ++ hidden_pack_offset) {
            const uint32_t hidden_pack_idx = hidden_pack_begin + hidden_pack_offset;
            const bool has_next_weight = hidden_pack_offset + 1 < kNumHiddenPacksPerWarp;
            uint4 next_weight;
            if (has_next_weight)
                next_weight = weight_packs[(hidden_pack_offset + 1) * (kHiddenPerNormPack / kHiddenPerNormLane)];

            const uint32_t hidden_idx = hidden_pack_idx * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
            const auto values = reinterpret_cast<nv_bfloat162*>(&retained[hidden_pack_offset]);
            const auto weight_pairs = reinterpret_cast<const nv_bfloat162*>(&weight);
            #pragma unroll
            for (uint32_t pair_idx = 0; pair_idx < kNumBF16PairsPerLane; ++ pair_idx)
                values[pair_idx] = normalize_bf16_pair(values[pair_idx], weight_pairs[pair_idx], rms_scale);
            if constexpr (kStoreBF16)
                *reinterpret_cast<uint4*>(norm_args.y_bf16 + output_offset + hidden_idx) = retained[hidden_pack_offset];
            if constexpr (kStoreFP8) {
                const auto amax = math::warp_reduce_max<kNumLanesPerSF>(
                    math::get_packed_bf16_amax(reinterpret_cast<const nv_bfloat162(&)[4]>(retained[hidden_pack_offset])));
                const uint32_t sf_word_idx = hidden_pack_idx * kNumSFWordsPerNormPack + lane_idx / kNumLanesPerSFWord;
                const uint32_t packed_sf = store_fp8_and_pack_sf(
                    norm_args, output_offset + hidden_idx, retained[hidden_pack_offset],
                    math::get_ue8m0_sf_exp(amax));
                if constexpr (SF_BLOCK_M > 0) {
                    if (lane_idx % kNumLanesPerSFWord == 0) {
                        norm_args.y_primary_sf[token_idx * norm_args.y_primary_sf_stride_token + sf_word_idx * norm_args.y_primary_sf_stride_word] = packed_sf;
                        norm_args.y_shared_sf[shared_sf_row_idx + sf_word_idx * norm_args.y_shared_sf_stride_word] = packed_sf;
                    }
                } else {
                    retained[hidden_pack_offset].x = packed_sf;
                }
            }

            if (has_next_weight)
                weight = next_weight;
        }

        // Keep column-major SF in dead value registers until contiguous output stores finish.
        if constexpr (kStoreFP8 and SF_BLOCK_M == 0) {
            if (lane_idx % kNumLanesPerSFWord == 0) {
                const int64_t sf_word_stride = norm_args.y_primary_sf_stride_word;
                auto sf_ptr = norm_args.y_primary_sf + token_idx * norm_args.y_primary_sf_stride_token +
                              (hidden_pack_begin * kNumSFWordsPerNormPack + lane_idx / kNumLanesPerSFWord) * sf_word_stride;

                #pragma unroll
                for (uint32_t hidden_pack_offset = 0; hidden_pack_offset < kNumHiddenPacksPerWarp; ++ hidden_pack_offset) {
                    *sf_ptr = retained[hidden_pack_offset].x;
                    sf_ptr += kNumSFWordsPerNormPack * sf_word_stride;
                }
            }
        }
    }
}

// Shifted Norm consumes materialized X1 and handles two tokens per task.
constexpr uint32_t kNumTokensPerShiftedNormTask = 2;

struct ShiftedNormPack {
    uint4 x1[kNumTokensPerShiftedNormTask];
    uint4 weight;
};

template <uint32_t kHidden, bool kStoreBF16, bool kStoreFP8, uint32_t SF_BLOCK_M, uint32_t kNumSplits>
CUTLASS_DEVICE void run_shifted_norm_task(const Workspace<kNumSplits>& workspace, uint64_t* gmem_split_barriers,
                                          const NormArgs& norm_args,
                                          const uint32_t& task_idx,
                                          const uint32_t& num_partitions, const uint32_t& lane_idx) {
    constexpr uint32_t kNumHiddenPacks = kHidden / kHiddenPerNormPack;
    // task_idx flattens [token pair, hidden-pack partition].
    const uint32_t partition_idx = task_idx % num_partitions;
    const uint32_t hidden_pack_begin = partition_idx * kNumHiddenPacks / num_partitions;
    const uint32_t hidden_pack_end = (partition_idx + 1) * kNumHiddenPacks / num_partitions;
    const uint32_t m_idx = task_idx / num_partitions * kNumTokensPerShiftedNormTask;
    const uint32_t m_block_idx = m_idx / BLOCK_M;
    const bool has_second_token = m_idx + 1 < norm_args.num_tokens;
    uint2 shared_sf_row_indices = {};

    sched::mega_mhc::Norm::wait<kNumSplits>(gmem_split_barriers, m_block_idx);
    const auto load_pack = [&](const uint64_t output_offset, const uint32_t hidden_idx) {
        return ShiftedNormPack{
            {ptx::ld_evict_first(reinterpret_cast<const uint4*>(norm_args.y_bf16 + output_offset)),
             ptx::ld_evict_first(reinterpret_cast<const uint4*>(
                 norm_args.y_bf16 + output_offset + (has_second_token ? kHidden : 0)))},
            *reinterpret_cast<const uint4*>(norm_args.weight + hidden_idx)};
    };

    // Keep current/next packs in registers and prefetch two ahead to hide the BF16 reread.
    ShiftedNormPack pack, next_pack;
    const uint32_t first_hidden_idx = hidden_pack_begin * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
    const uint64_t first_output_offset = static_cast<uint64_t>(m_idx) * kHidden + first_hidden_idx;
    pack = load_pack(first_output_offset, first_hidden_idx);
    if (hidden_pack_begin + 1 < hidden_pack_end)
        next_pack = load_pack(first_output_offset + kHiddenPerNormPack, first_hidden_idx + kHiddenPerNormPack);

    float x1_sqr_sum = 0.0f;
    if (lane_idx < kNumTokensPerShiftedNormTask and m_idx + lane_idx < norm_args.num_tokens) {
        const auto x1_sqr_sum_partials = workspace.get_x1_sqr_sum_partial_ptr(m_block_idx * kNumSplits);
        #pragma unroll
        for (uint32_t k_split_idx = 0; k_split_idx < kNumSplits; ++ k_split_idx)
            x1_sqr_sum += x1_sqr_sum_partials[k_split_idx * kNumSqrSumPartialElementsPerTask +
                                               m_idx % BLOCK_M + lane_idx];
    }
    constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
    const float rms_scale = rsqrtf(x1_sqr_sum * kInvHidden + norm_args.eps) * norm_args.scale;
    const float2 rms_scales = {ptx::exchange(rms_scale, 0), ptx::exchange(rms_scale, 1)};
    if constexpr (SF_BLOCK_M > 0) {
        shared_sf_row_indices.x = get_shared_sf_row_idx<SF_BLOCK_M>(m_idx);
        shared_sf_row_indices.y = get_shared_sf_row_idx<SF_BLOCK_M>(m_idx + 1);
    }
    for (uint32_t hidden_pack_idx = hidden_pack_begin; hidden_pack_idx < hidden_pack_end; ++ hidden_pack_idx) {
        const uint32_t hidden_idx = hidden_pack_idx * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
        const uint64_t output_offset = static_cast<uint64_t>(m_idx) * kHidden + hidden_idx;
        const bool has_prefetch = hidden_pack_idx + 2 < hidden_pack_end;
        ShiftedNormPack prefetch_pack;
        if (has_prefetch)
            prefetch_pack = load_pack(output_offset + 2 * kHiddenPerNormPack, hidden_idx + 2 * kHiddenPerNormPack);
        uint4 values[kNumTokensPerShiftedNormTask] = {};
        const auto input = reinterpret_cast<const nv_bfloat162*>(pack.x1);
        const auto weight = reinterpret_cast<const nv_bfloat162*>(&pack.weight);
        auto output = reinterpret_cast<nv_bfloat162*>(values);
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumBF16PairsPerLane; ++ pair_idx) {
            output[pair_idx] = normalize_bf16_pair(input[pair_idx], weight[pair_idx], rms_scales.x);
            output[kNumBF16PairsPerLane + pair_idx] =
                normalize_bf16_pair(input[kNumBF16PairsPerLane + pair_idx], weight[pair_idx], rms_scales.y);
        }
        if constexpr (kStoreBF16) {
            *reinterpret_cast<uint4*>(norm_args.y_bf16 + output_offset) = values[0];
            if (has_second_token)
                *reinterpret_cast<uint4*>(norm_args.y_bf16 + output_offset + kHidden) = values[1];
        }
        if constexpr (kStoreFP8) {
            const auto values_pairs = reinterpret_cast<const nv_bfloat162(*)[4]>(values);
            // Pack both tokens' amaxes into one BF16 pair to share a single warp reduction
            const auto packed_amax = math::warp_reduce_max<kNumLanesPerSF>(nv_bfloat162(
                math::get_packed_bf16_amax(values_pairs[0]),
                has_second_token ? math::get_packed_bf16_amax(values_pairs[1]) : __ushort_as_bfloat16(0)));
            const uint2 sf_exponents = {math::get_ue8m0_sf_exp(packed_amax.x),
                                        math::get_ue8m0_sf_exp(packed_amax.y)};
            const uint32_t sf_word_idx = hidden_pack_idx * kNumSFWordsPerNormPack + lane_idx / kNumLanesPerSFWord;
            const uint32_t first_packed_sf = store_fp8_and_pack_sf(norm_args, output_offset, values[0], sf_exponents.x);
            if (lane_idx % kNumLanesPerSFWord == 0) {
                norm_args.y_primary_sf[m_idx * norm_args.y_primary_sf_stride_token + sf_word_idx * norm_args.y_primary_sf_stride_word] = first_packed_sf;
                if constexpr (SF_BLOCK_M > 0)
                    norm_args.y_shared_sf[static_cast<int64_t>(sf_word_idx) * norm_args.y_shared_sf_stride_word + shared_sf_row_indices.x] = first_packed_sf;
            }
            if (has_second_token) {
                const uint32_t second_packed_sf = store_fp8_and_pack_sf(
                    norm_args, output_offset + kHidden, values[1], sf_exponents.y);
                if (lane_idx % kNumLanesPerSFWord == 0) {
                    norm_args.y_primary_sf[(m_idx + 1) * norm_args.y_primary_sf_stride_token + sf_word_idx * norm_args.y_primary_sf_stride_word] = second_packed_sf;
                    if constexpr (SF_BLOCK_M > 0)
                        norm_args.y_shared_sf[static_cast<int64_t>(sf_word_idx) * norm_args.y_shared_sf_stride_word + shared_sf_row_indices.y] = second_packed_sf;
                }
            }
        }
        if (hidden_pack_idx + 1 < hidden_pack_end)
            pack = next_pack;
        if (has_prefetch)
            next_pack = prefetch_pack;
    }
}

template <uint32_t kHidden, uint32_t kNumSMs, uint32_t kNumRegisters, bool kStoreBF16, bool kStoreFP8,
          uint32_t SF_BLOCK_M, uint32_t kNumSplits>
CUTLASS_DEVICE void run_shifted_norm_worker(SharedStorage& smem, const Workspace<kNumSplits>& workspace,
                                            uint64_t* gmem_split_barriers,
                                            const NormArgs& norm_args,
                                            const uint32_t& lane_idx) {
    constexpr uint32_t kNumTasksPerGridWave = kNumSMs * kNumWarpsPerWG;
    constexpr uint32_t kNumHiddenPacks = kHidden / kHiddenPerNormPack;
    const uint32_t num_token_pairs = math::ceil_div(norm_args.num_tokens, kNumTokensPerShiftedNormTask);
    // Only subdivide underfilled FP8 grids when a token pair can span multiple CTAs.
    constexpr uint32_t kNumFP8Partitions =
        kNumSplits > kDefaultNumSplits and kNumHiddenPacks > 2 * kNumWarpsPerWG ? kNumHiddenPacks / 2 : 2;
    const uint32_t num_partitions = kStoreFP8 ? kNumFP8Partitions : cute::max(
        math::ceil_div(kNumHiddenPacks, cute::max(
            1u, num_token_pairs * kNumHiddenPacks / kNumTasksPerGridWave)), 2u);
    const uint32_t num_tasks = num_token_pairs * num_partitions;
    const uint32_t cta_task_base = blockIdx.x * kNumWarpsPerWG;
    if (cta_task_base >= num_tasks)
        return;

    cutlass::arch::warpgroup_reg_alloc<kNumRegisters>();
    while (true) {
        uint32_t task_ticket;
        if (cute::elect_one_sync())
            task_ticket = atomicAdd(&smem.next_norm_task_ticket, 1u);
        task_ticket = ptx::exchange(task_ticket, 0);
        // Preserve the four CTA-local warp residue classes, then advance by one grid-wide worker wave.
        const uint32_t task_idx = cta_task_base + task_ticket % kNumWarpsPerWG +
                                  task_ticket / kNumWarpsPerWG * kNumTasksPerGridWave;
        if (task_idx >= num_tasks)
            return;
        run_shifted_norm_task<kHidden, kStoreBF16, kStoreFP8, SF_BLOCK_M>(
            workspace, gmem_split_barriers, norm_args, task_idx, num_partitions, lane_idx);
    }
}

} // namespace deep_gemm::epilogue::mega_mhc
