#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm100_mqa_logits_f16_weights(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& kv_scales, const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, const int& seq_len,
    const int& seq_len_kv, const int& max_seqlen_k, const int& stride_logits,
    const int& num_heads, const int& head_dim, const int& block_q,
    const int& block_kv, const bool& clean_logits) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(128 % num_heads == 0);
    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0);

    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 5, num_kv_stages = 8;
    constexpr int num_math_threads = 256;
    const bool is_compressed_logits = (max_seqlen_k > 0);
    DG_HOST_ASSERT(not (clean_logits and is_compressed_logits));
    auto weights_f16 = weights.to(torch::kFloat16).contiguous();

    const auto tensor_map_q = make_tma_2d_desc(
        q, head_dim, seq_len * num_heads, head_dim,
        block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv, head_dim, seq_len_kv, head_dim, block_kv / 2, head_dim, head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales, get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
        1, block_kv / 2, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights_f16, num_heads, seq_len, num_heads, block_q, num_heads, 0);

    const int block_q_2cta = block_q * 2;
    const int smem_q_per_stage = block_q * num_heads * head_dim;
    const int smem_weight_per_stage = block_q_2cta * num_heads * 2;
    const int smem_kv_per_stage = (block_kv / 2) * head_dim;
    const int smem_kv_scale_raw = (block_kv / 2) * static_cast<int>(kv_scales.element_size());
    const int smem_kv_scale_per_stage = (smem_kv_scale_raw + 511) / 512 * 512;
    const int smem_kv_offset_per_stage = block_q_2cta * 8;
    const int num_umma_stages = 512 / (block_q_2cta * num_heads);
    const int num_barriers = num_q_stages * 2 + num_kv_stages * 2 + num_umma_stages * 2;

    int smem_size = 0;
    smem_size += num_q_stages * smem_q_per_stage;
    smem_size += num_q_stages * smem_weight_per_stage;
    smem_size += num_kv_stages * smem_kv_per_stage;
    smem_size += num_kv_stages * smem_kv_scale_per_stage;
    smem_size += num_q_stages * smem_kv_offset_per_stage;
    smem_size += num_barriers * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Bulk offset loads have no TMA bounds fill; padded rows must be neutral.
    const int aligned_offset_rows = align(seq_len, block_q_2cta);
    auto cu_seq_len_k_start_and_end = torch::empty(
        {aligned_offset_rows, 2}, cu_seq_len_k_start.options());
    cu_seq_len_k_start_and_end.select(1, 0).fill_(-1);
    cu_seq_len_k_start_and_end.select(1, 1).zero_();
    auto valid_offsets = cu_seq_len_k_start_and_end.narrow(0, 0, seq_len);
    valid_offsets.select(1, 0).copy_(cu_seq_len_k_start);
    valid_offsets.select(1, 1).copy_(cu_seq_len_k_end);
    cu_seq_len_k_start_and_end = cu_seq_len_k_start_and_end.reshape({-1}).contiguous();

    const auto kernel = jit->compile("sm100_mqa_logits_f16_weights", std::format(R"(
#include <deep_gemm/impls/sm100_fp8_mqa_logits_f16_weights.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_mqa_logits_f16_weights<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", num_heads, head_dim, is_compressed_logits, block_q, block_kv,
        num_q_stages, num_kv_stages, num_specialized_threads, num_math_threads,
        to_string(logits_dtype)));

    jit->launch(kernel, {
        .num_smem_bytes = smem_size,
        .grid_dim = dim3(num_sms, 1, 1),
        .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        .cluster_dim = dim3(2, 1, 1),
        .enable_pdl = false,
    }, seq_len, seq_len_kv, max_seqlen_k, static_cast<uint64_t>(stride_logits),
        cu_seq_len_k_start_and_end.data_ptr<int>(), logits.data_ptr(),
        tensor_map_q, tensor_map_kv, tensor_map_kv_scales, tensor_map_weights);

    if (clean_logits) {
        const auto clean_kernel = jit->compile("smxx_clean_logits", std::format(R"(
#include <deep_gemm/impls/smxx_clean_logits.cuh>

using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_clean_logits<1, 8192, 8, {}>);
}};
)", to_string(logits_dtype)));
        jit->launch(clean_kernel, {
            .num_smem_bytes = 8192 * static_cast<int>(c10::elementSize(logits_dtype)),
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(256, 1, 1),
            .enable_pdl = false,
        }, seq_len, seq_len_kv, static_cast<uint64_t>(stride_logits),
            cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(), logits.data_ptr());
    }
}

} // namespace deep_gemm
