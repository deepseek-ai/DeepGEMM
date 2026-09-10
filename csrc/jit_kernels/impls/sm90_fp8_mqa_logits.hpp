#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm90_fp8_mqa_logits(const torch::Tensor& q,
                                const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                const torch::Tensor& weights,
                                const torch::Tensor& cu_seq_len_k_start,
                                const torch::Tensor& cu_seq_len_k_end,
                                const torch::Tensor& logits,
                                const at::ScalarType& logits_dtype,
                                const int& seq_len, const int& seq_len_kv,
                                const int& max_seqlen_k, const int& stride_logits,
                                const int& num_heads, const int& head_dim,
                                const int& block_q, const int& block_kv,
                                const bool& clean_logits) {
    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 3, num_kv_stages = 3;
    constexpr int num_math_threads = 512;

    const bool is_compressed_logits = (max_seqlen_k > 0);
    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(not (clean_logits and is_compressed_logits));

    DG_HOST_ASSERT(jit->device.get_arch_major() == 9);
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv, head_dim, head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                       get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                       1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);

    int smem_size = 0;
    const int smem_q_size_per_stage = block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int smem_weight_size_per_stage = block_q * num_heads * static_cast<int>(weights.element_size());
    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_scale_size_per_stage = block_kv * static_cast<int>(kv_scales.element_size());
    smem_size += num_q_stages * smem_q_size_per_stage;
    smem_size += num_kv_stages * smem_kv_size_per_stage;
    smem_size += num_q_stages * smem_weight_size_per_stage;
    smem_size += num_kv_stages * kv_scale_size_per_stage;
    smem_size += (num_q_stages * 2 + num_kv_stages * 2 + (num_math_threads / 128) * 2) * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);

    DG_HOST_ASSERT(128 % num_heads == 0);

    // Compile
    const auto kernel = jit->compile("sm90_fp8_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
    num_heads, head_dim,
    is_compressed_logits, clean_logits,
    block_q, block_kv,
    num_q_stages, num_kv_stages,
    num_sms,
    num_specialized_threads, num_math_threads,
    to_string(logits_dtype)));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        },
        seq_len, seq_len_kv,
        max_seqlen_k, stride_logits,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        logits.data_ptr(),
        tensor_map_q, tensor_map_kv,
        tensor_map_kv_scales, tensor_map_weights
    );
}

static void sm90_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                           const torch::Tensor& schedule_metadata,
                                           const int& batch_size, const int& next_n,
                                           const int& block_kv, const int& num_sms,
                                           const bool& is_context_lens_2d,
                                           const bool& is_varlen, const int* indices_ptr) {
    constexpr int split_kv = 256;
    constexpr int num_threads = 32;
    const int aligned_batch_size = align(batch_size, 32);
    DG_HOST_ASSERT(split_kv % block_kv == 0);

    const int num_smem_ints = is_varlen ? 3 * aligned_batch_size + 1 : aligned_batch_size;
    const int smem_size = num_smem_ints * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm90_paged_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/scheduler/sm90_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm90_paged_mqa_logits_metadata<
        {}, {}, {}, {}
    >);
}};
)", aligned_batch_size, split_kv, num_sms, is_varlen ? "true" : "false"));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
        },
        batch_size,
        next_n,
        is_context_lens_2d,
        context_lens.data_ptr<int>(),
        const_cast<int*>(indices_ptr),
        schedule_metadata.data_ptr<int>()
    );
}

static void sm90_fp8_paged_mqa_logits(const torch::Tensor& q,
                                      const torch::Tensor& kv_cache,
                                      const torch::Tensor& kv_cache_scales,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& context_lens,
                                      const torch::Tensor& logits,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& indices,
                                      const torch::Tensor& schedule_meta,
                                      const at::ScalarType& logits_dtype,
                                      const int& batch_size, const int& next_n,
                                      const int& num_heads, const int& head_dim,
                                      const int& num_kv_blocks, const int& block_kv,
                                      const bool& is_context_lens_2d,
                                      const bool& is_varlen,
                                      const int& logits_stride,
                                      const int& block_table_stride,
                                      const int& num_sms,
                                      const int& split_kv) {
    constexpr int num_specialized_threads = 128;
    constexpr int mma_m = 64;
    const int num_math_warp_groups = split_kv / mma_m;
    const int num_math_threads = num_math_warp_groups * 128;
    constexpr int num_q_stages = 3, num_kv_stages = 3;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 9);
    DG_HOST_ASSERT(split_kv % mma_m == 0 and logits_stride % split_kv == 0);

    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_atom * num_heads,
                                               static_cast<int>(q.stride(2)),
                                               head_dim);
    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_cache_scales, block_kv, num_kv_blocks,
                                                       block_kv, 1,
                                                       static_cast<int>(kv_cache_scales.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_atom,
                                                     static_cast<int>(weights.stride(0)), 0);

    const int swizzle_alignment = head_dim * 8;
    const int smem_q_size_per_stage = next_n * num_heads * head_dim * static_cast<int>(q.element_size());
    const int aligned_smem_weight_size_per_stage = align(next_n * num_heads * static_cast<int>(weights.element_size()), swizzle_alignment);
    const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) + align(num_q_stages * 8 * 2, swizzle_alignment);
    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size());
    const int aligned_smem_kv_scale_size_per_stage = align(block_kv * static_cast<int>(kv_cache_scales.element_size()), swizzle_alignment);
    const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) + align(num_kv_stages * 8 * 2, swizzle_alignment);
    const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
    const int smem_tmem_ptr = 4;
    const int smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + smem_umma_barriers + smem_tmem_ptr;
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    DG_HOST_ASSERT(next_n == 1 or next_n == 2);

    DG_HOST_ASSERT(128 % num_heads == 0);

    // Compile
    const auto kernel = jit->compile("sm90_fp8_paged_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
    next_n, num_heads,
    head_dim, block_kv,
    is_context_lens_2d, is_varlen ? "true" : "false",
    num_q_stages, num_kv_stages,
    split_kv,
    num_specialized_threads, num_math_threads,
    to_string(logits_dtype)));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        },
        batch_size,
        logits_stride, block_table_stride,
        context_lens.data_ptr<int>(), logits.data_ptr(),
        block_table.data_ptr<int>(), is_varlen ? indices.data_ptr<int>() : nullptr, schedule_meta.data_ptr<int>(),
        tensor_map_q, tensor_map_kv,
        tensor_map_kv_scales, tensor_map_weights
    );
}

} // namespace deep_gemm
