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
    const int& block_kv) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(128 % num_heads == 0);

    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 5, num_kv_stages = 8;
    constexpr int num_math_threads = 256;
    const bool is_compressed_logits = (max_seqlen_k > 0);
    auto weights_f16 = weights.to(torch::kFloat16).contiguous();

    // The two CTAs split each KV tile in half and share the same Q/weights tile.
    const auto tensor_map_q =
        make_tma_2d_desc(q, head_dim, seq_len * num_heads, head_dim,
                         block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv, head_dim, seq_len_kv, head_dim, block_kv / 2, head_dim, head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales,
        get_tma_aligned_size(seq_len_kv,
                             static_cast<int>(kv_scales.element_size())),
        1, block_kv / 2, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights_f16, num_heads, seq_len, num_heads, block_q, num_heads, 0);

    const int block_q_2cta = block_q * 2;
    const int smem_q_per_stage = block_q * num_heads * head_dim;
    const int smem_weight_per_stage = block_q_2cta * num_heads * 2;
    const int smem_kv_per_stage = (block_kv / 2) * head_dim;
    const int smem_kv_scale_raw =
        (block_kv / 2) * static_cast<int>(kv_scales.element_size());
    const int smem_kv_scale_per_stage =
        (smem_kv_scale_raw + 511) / 512 * 512;
    const int smem_kv_offset_per_stage = block_q_2cta * 8;
    const int num_umma_stages = 512 / (block_q_2cta * num_heads);
    const int num_barriers =
        num_q_stages * 2 + num_kv_stages * 2 + num_umma_stages * 2;

    int smem_size = 0;
    smem_size += num_q_stages * smem_q_per_stage;
    smem_size += num_q_stages * smem_weight_per_stage;
    smem_size += num_kv_stages * smem_kv_per_stage;
    smem_size += num_kv_stages * smem_kv_scale_per_stage;
    smem_size += num_q_stages * smem_kv_offset_per_stage;
    smem_size += num_barriers * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // The device kernel bulk-copies a full two-CTA Q tile without TMA
    // out-of-bounds zero fill. Pad the offsets buffer to that tile size;
    // {UINT32_MAX, 0} is neutral for the device-side min(start)/max(end)
    // reduction and suppresses compressed stores for padded rows.
    const int aligned_offset_rows = align(seq_len, block_q_2cta);
    torch::Tensor cu_seq_len_k_start_and_end = torch::empty(
        {aligned_offset_rows, 2}, cu_seq_len_k_start.options());
    cu_seq_len_k_start_and_end.select(1, 0).fill_(-1);
    cu_seq_len_k_start_and_end.select(1, 1).zero_();
    auto valid_offsets = cu_seq_len_k_start_and_end.narrow(0, 0, seq_len);
    valid_offsets.select(1, 0).copy_(cu_seq_len_k_start);
    valid_offsets.select(1, 1).copy_(cu_seq_len_k_end);
    cu_seq_len_k_start_and_end = cu_seq_len_k_start_and_end.reshape({-1}).contiguous();

    // Compile
    const auto kernel = jit->compile("sm100_mqa_logits_f16_weights", std::format(R"(
#include <deep_gemm/impls/sm100_fp8_mqa_logits_f16_weights.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_mqa_logits_f16_weights<
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {}
    >);
}};
)", num_heads, head_dim,
    is_compressed_logits,
    block_q, block_kv,
    num_q_stages, num_kv_stages,
    num_specialized_threads, num_math_threads,
    to_string(logits_dtype)));

    // Launch (two-CTA cluster)
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(runtime->get_num_sms(), 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        seq_len, seq_len_kv, max_seqlen_k,
        static_cast<uint64_t>(stride_logits),
        reinterpret_cast<uint32_t*>(cu_seq_len_k_start_and_end.data_ptr<int>()),
        logits.data_ptr(),
        tensor_map_q, tensor_map_kv, tensor_map_kv_scales, tensor_map_weights
    );
}

} // namespace deep_gemm
