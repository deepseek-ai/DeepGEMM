#pragma once

#include <type_traits>

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/ptx/math.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/scheduler/mega_mhc_inference.cuh>

namespace deep_gemm::epilogue::mega_mhc {

using namespace layout::mega_mhc;

// Reduce one token's split partials and produce its Mix coefficients.
template <uint32_t kHidden, uint32_t kNumSplits, bool kSaveStats = false>
CUTLASS_DEVICE void run_mix_task(const Workspace<kNumSplits>& workspace, uint32_t* gmem_split_barriers,
                                 const MixArgs& mix_args, const MixSaveArgs& save_args,
                                 const uint32_t& token_idx, const uint32_t& lane_idx) {
    const uint32_t m_block_idx = token_idx / BLOCK_M;
    const uint32_t row_idx = token_idx % BLOCK_M;
    sched::mega_mhc::Mix::wait(gmem_split_barriers, m_block_idx);

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
    // NOTES: the backward rebuilds the mix value as `hc_output * hc_rms_scale`, so this
    // expression must stay identical to the one in the backward's `prepare_mix`.
    const float hc_rms_scale = rsqrtf(hc_norm_sqr_sum / static_cast<float>(kNumRoutes * kHidden) +
                                      mix_args.hc_norm_eps);

    if constexpr (kSaveStats) {
        if (is_hc_output_lane)
            save_args.hc_output[token_idx * kNumHCOutputs + lane_idx] = hc_output;
        if (lane_idx == 0)
            save_args.hc_norm_sqr_sum[token_idx] = hc_norm_sqr_sum;
    }

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

    if (is_hc_output_lane) {
        const auto mix_output = is_pre_lane  ? mix_args.new_prev_mix :
                                is_comb_lane ? mix_args.new_comb_res_mix : mix_args.new_post_mix;
        const uint32_t mix_output_stride = is_comb_lane ? kNumRoutes * kNumRoutes : kNumRoutes;
        const uint32_t mix_output_idx = is_comb_lane ? lane_idx - kFirstCombLane : lane_idx & (kNumRoutes - 1);
        mix_output[token_idx * mix_output_stride + mix_output_idx] = is_comb_lane ? normalized_comb : mix_value;
    }
    __syncwarp();
}

// Hidden elements handled by one lane of a norm pack.
constexpr uint32_t kHiddenPerNormLane = kHiddenPerNormPack / 32;
constexpr uint32_t kNumBF16PairsPerLane = sizeof(uint4) / sizeof(nv_bfloat162);
constexpr uint32_t kNumLanesPerSF = kHiddenPerSF / kHiddenPerNormLane;
constexpr uint32_t kNumSFWordsPerNormPack = kHiddenPerNormPack / kHiddenPerSFWord;
constexpr uint32_t kNumLanesPerSFWord = kHiddenPerSFWord / kHiddenPerNormLane;

// Shared normalization and FP8 output helpers.
CUTLASS_DEVICE nv_bfloat162 normalize_pair(const float2& input, const float2& weight, const float& scale) {
    return __float22bfloat162_rn(__fmul2_rn(__fmul2_rn(input, float2(scale, scale)), weight));
}

CUTLASS_DEVICE nv_bfloat162 normalize_pair(const nv_bfloat162& input, const nv_bfloat162& weight, const float& scale) {
    return normalize_pair(ptx::cvt_bf16x2_to_fp32x2(input), ptx::cvt_bf16x2_to_fp32x2(weight), scale);
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

// Norm consumes materialized X1 and handles two tokens per task.
template <typename dtype_t>
struct NormPack {
    // Eight values per lane in the residual stream's dtype (FP32 or BF16).
    uint4 x1[kNumTokensPerNormTask][kHiddenPerNormLane * sizeof(dtype_t) / sizeof(uint4)];
};

// Per-lane norm weights, shared by every token pair (FP32 or BF16).
template <typename dtype_t>
struct NormWeight {
    uint4 word[kHiddenPerNormLane * sizeof(dtype_t) / sizeof(uint4)];
};

template <uint32_t kHidden, bool kStoreBF16, bool kStoreFP8, uint32_t SF_BLOCK_M, uint32_t kNumSplits,
          uint32_t kNormPartitions, bool kStoreRho = false, typename dtype_t = nv_bfloat16>
CUTLASS_DEVICE void run_norm_task(const Workspace<kNumSplits>& workspace, uint32_t* gmem_split_barriers,
                                  const NormArgs& norm_args, const MixSaveArgs& save_args,
                                  const uint32_t& task_idx, const uint32_t& lane_idx) {
    constexpr uint32_t kNumHiddenPacks = kHidden / kHiddenPerNormPack;
    // task_idx flattens [token pair, hidden-pack partition].
    const uint32_t partition_idx = task_idx % kNormPartitions;
    const uint32_t hidden_pack_begin = partition_idx * kNumHiddenPacks / kNormPartitions;
    const uint32_t hidden_pack_end = (partition_idx + 1) * kNumHiddenPacks / kNormPartitions;
    const uint32_t m_idx = task_idx / kNormPartitions * kNumTokensPerNormTask;
    const uint32_t m_block_idx = m_idx / BLOCK_M;
    const bool has_second_token = m_idx + 1 < norm_args.num_tokens;
    uint2 shared_sf_row_indices = {};

    constexpr bool kInputFP32 = cute::is_same_v<dtype_t, float>;
    DG_STATIC_ASSERT(kInputFP32 or cute::is_same_v<dtype_t, nv_bfloat16>, "Invalid input data type");
    using InputPair = std::conditional_t<kInputFP32, float2, nv_bfloat162>;
    constexpr uint32_t kNumWords = kHiddenPerNormLane * sizeof(dtype_t) / sizeof(uint4);
    const auto input = static_cast<const dtype_t*>(norm_args.x1);
    const auto weights = static_cast<const dtype_t*>(norm_args.weight);

    sched::mega_mhc::Norm::wait(gmem_split_barriers, m_block_idx);
    const auto load_pack = [&](const uint64_t output_offset) {
        NormPack<dtype_t> pack;
        #pragma unroll
        for (uint32_t token_idx = 0; token_idx < kNumTokensPerNormTask; ++ token_idx) {
            const bool has_token = token_idx == 0 or has_second_token;
            const uint64_t token_offset = output_offset + (has_token ? token_idx * kHidden : 0);
            const auto input_ptr = reinterpret_cast<const uint4*>(input + token_offset);
            #pragma unroll
            for (uint32_t word_idx = 0; word_idx < kNumWords; ++ word_idx)
                pack.x1[token_idx][word_idx] = ptx::ld_evict_first(input_ptr + word_idx);
        }
        return pack;
    };
    const auto load_weight = [&](const uint32_t hidden_idx) {
        NormWeight<dtype_t> weight;
        const auto weight_ptr = reinterpret_cast<const uint4*>(weights + hidden_idx);
        #pragma unroll
        for (uint32_t word_idx = 0; word_idx < kNumWords; ++ word_idx)
            weight.word[word_idx] = weight_ptr[word_idx];
        return weight;
    };

    // Keep current/next packs in registers and prefetch two ahead to hide the activation reread.
    // Both tokens share a weight pack. Prefetch weights two packs ahead for inference,
    // but only one ahead for FP32 training to limit register pressure.
    NormPack<dtype_t> pack, next_pack;
    NormWeight<dtype_t> weight, next_weight;
    const uint32_t first_hidden_idx = hidden_pack_begin * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
    const uint64_t first_output_offset = static_cast<uint64_t>(m_idx) * kHidden + first_hidden_idx;
    pack = load_pack(first_output_offset);
    weight = load_weight(first_hidden_idx);
    if (hidden_pack_begin + 1 < hidden_pack_end) {
        next_pack = load_pack(first_output_offset + kHiddenPerNormPack);
        next_weight = load_weight(first_hidden_idx + kHiddenPerNormPack);
    }

    float x1_sqr_sum = 0.0f;
    if (lane_idx < kNumTokensPerNormTask and m_idx + lane_idx < norm_args.num_tokens) {
        const auto x1_sqr_sum_partials = workspace.get_x1_sqr_sum_partial_ptr(m_block_idx * kNumSplits);
        #pragma unroll
        for (uint32_t k_split_idx = 0; k_split_idx < kNumSplits; ++ k_split_idx)
            x1_sqr_sum += x1_sqr_sum_partials[k_split_idx * kNumSqrSumPartialElementsPerTask +
                                              m_idx % BLOCK_M + lane_idx];
    }
    constexpr float kInvHidden = 1.0f / static_cast<float>(kHidden);
    const float inv_rms = rsqrtf(x1_sqr_sum * kInvHidden + norm_args.eps);
    const float rms_scale = inv_rms * norm_args.scale;
    if constexpr (kStoreRho) {
        // One partition owns the row statistic; the other partitions recompute it unused.
        if (partition_idx == 0 and lane_idx < kNumTokensPerNormTask and m_idx + lane_idx < norm_args.num_tokens)
            save_args.rho[m_idx + lane_idx] = inv_rms;
    }
    const float2 rms_scales = {ptx::exchange(rms_scale, 0), ptx::exchange(rms_scale, 1)};
    if constexpr (SF_BLOCK_M > 0) {
        shared_sf_row_indices.x = get_shared_sf_row_idx<SF_BLOCK_M>(m_idx);
        shared_sf_row_indices.y = get_shared_sf_row_idx<SF_BLOCK_M>(m_idx + 1);
    }
    for (uint32_t hidden_pack_idx = hidden_pack_begin; hidden_pack_idx < hidden_pack_end; ++ hidden_pack_idx) {
        const uint32_t hidden_idx = hidden_pack_idx * kHiddenPerNormPack + lane_idx * kHiddenPerNormLane;
        const uint64_t output_offset = static_cast<uint64_t>(m_idx) * kHidden + hidden_idx;
        const bool has_prefetch = hidden_pack_idx + 2 < hidden_pack_end;
        NormPack<dtype_t> prefetch_pack;
        NormWeight<dtype_t> prefetch_weight;
        if (has_prefetch) {
            prefetch_pack = load_pack(output_offset + 2 * kHiddenPerNormPack);
            if constexpr (not kInputFP32)
                prefetch_weight = load_weight(hidden_idx + 2 * kHiddenPerNormPack);
        }
        uint4 values[kNumTokensPerNormTask] = {};
        const auto output = reinterpret_cast<nv_bfloat162*>(values);
        const auto input_pairs = reinterpret_cast<const InputPair*>(pack.x1);
        const auto weight_pairs = reinterpret_cast<const InputPair*>(weight.word);
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumBF16PairsPerLane; ++ pair_idx) {
            output[pair_idx] = normalize_pair(input_pairs[pair_idx], weight_pairs[pair_idx], rms_scales.x);
            output[kNumBF16PairsPerLane + pair_idx] = normalize_pair(
                input_pairs[kNumBF16PairsPerLane + pair_idx], weight_pairs[pair_idx], rms_scales.y);
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
        if (hidden_pack_idx + 1 < hidden_pack_end) {
            pack = next_pack;
            weight = next_weight;
        }
        if (has_prefetch) {
            next_pack = prefetch_pack;
            if constexpr (kInputFP32)
                next_weight = load_weight(hidden_idx + 2 * kHiddenPerNormPack);
            else
                next_weight = prefetch_weight;
        }
    }
}

template <uint32_t kHidden, uint32_t kNumSMs, uint32_t kNumRegisters, bool kStoreBF16, bool kStoreFP8,
          uint32_t SF_BLOCK_M, uint32_t kNumSplits, uint32_t kNormPartitions,
          bool kStoreRho = false, typename smem_t>
CUTLASS_DEVICE void run_norm_worker(smem_t& smem, const Workspace<kNumSplits>& workspace,
                                    uint32_t* gmem_split_barriers,
                                    const NormArgs& norm_args, const MixSaveArgs& save_args,
                                    const uint32_t& lane_idx) {
    using raw_x_t = std::remove_all_extents_t<decltype(smem_t::x)>;
    DG_STATIC_ASSERT(cute::is_same_v<raw_x_t, float> or cute::is_same_v<raw_x_t, nv_bfloat16> or
                     cute::is_same_v<raw_x_t, nv_bfloat162>,
                     "Invalid shared X element type");
    using dtype_t = std::conditional_t<cute::is_same_v<raw_x_t, float>, float, nv_bfloat16>;
    constexpr uint32_t kNumTasksPerGridWave = kNumSMs * kNumWarpsPerWG;
    const uint32_t num_token_pairs = math::ceil_div(norm_args.num_tokens, kNumTokensPerNormTask);
    const uint32_t num_tasks = num_token_pairs * kNormPartitions;
    const uint32_t cta_task_base = blockIdx.x * kNumWarpsPerWG;
    if (cta_task_base >= num_tasks)
        return;

    cutlass::arch::warpgroup_reg_alloc<kNumRegisters>();
    while (true) {
        uint32_t task_ticket;
        // The Norm worker control flow is warp-uniform, so lane 0 always takes the ticket
        if (lane_idx == 0)
            task_ticket = atomicAdd(&smem.next_norm_task_ticket, 1u);
        task_ticket = ptx::exchange(task_ticket, 0);
        // Preserve the four CTA-local warp residue classes, then advance by one grid-wide worker wave
        const uint32_t task_idx = cta_task_base + task_ticket % kNumWarpsPerWG +
                                  task_ticket / kNumWarpsPerWG * kNumTasksPerGridWave;
        if (task_idx >= num_tasks)
            return;
        run_norm_task<kHidden, kStoreBF16, kStoreFP8, SF_BLOCK_M, kNumSplits, kNormPartitions, kStoreRho, dtype_t>(
            workspace, gmem_split_barriers, norm_args, save_args, task_idx, lane_idx);
    }
}

} // namespace deep_gemm::epilogue::mega_mhc
