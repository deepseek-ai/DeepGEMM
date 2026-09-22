#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm120.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm120_fp8_mqa_logits(const torch::Tensor& q,
                                 const torch::Tensor& kv,
                                 const torch::Tensor& kv_scales,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& cu_seq_len_k_start,
                                 const torch::Tensor& cu_seq_len_k_end,
                                 const torch::Tensor& logits,
                                 const at::ScalarType& logits_dtype,
                                 const int& seq_len, const int& seq_len_kv,
                                 const int& max_seqlen_k, const int& stride_logits,
                                 const int& num_heads, const int& head_dim,
                                 const int& block_q, const int& block_kv) {
    constexpr int num_tma_threads = 128;
    constexpr int num_q_stages = 2, num_kv_stages = 3;
    constexpr int num_math_threads = 256;
    const int num_sms = runtime->get_num_sms();
    const bool is_compressed_logits = max_seqlen_k > 0;

    // Split the KV range only when Q blocks underfill SM120. The split blocks
    // write disjoint output columns, so no reduction or atomic is needed.
    int kv_splits = 1;
    const int num_q_blocks = ceil_div(seq_len, block_q);
    const int max_kv_blocks = ceil_div(seq_len_kv, block_kv);
    if (num_q_blocks < num_sms and max_kv_blocks > 1) {
        const int max_splits = std::max(
            1, std::min(num_sms / std::max(1, num_q_blocks) + 1, max_kv_blocks / 4));
        double best_cost = 1e30;
        for (int s = 1; s <= max_splits; ++s) {
            const int waves = ceil_div(num_q_blocks * s, num_sms);
            const double cost = static_cast<double>(waves) / s;
            if (cost < best_cost - 1e-9) {
                best_cost = cost;
                kv_splits = s;
            }
        }
    }

    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT(128 % num_heads == 0);
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(
        q, head_dim, seq_len * num_heads,
        head_dim, block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv, head_dim, seq_len_kv, head_dim, block_kv, head_dim, head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales,
        get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
        1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, seq_len, num_heads, block_q,
        static_cast<int>(weights.stride(0)), 0);

    int smem_size = 0;
    const int smem_q_size_per_stage =
        block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int smem_weight_size_per_stage =
        block_q * num_heads * static_cast<int>(weights.element_size());
    const int smem_kv_size_per_stage =
        block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_scale_size_per_stage =
        block_kv * static_cast<int>(kv_scales.element_size());
    smem_size += num_q_stages * smem_q_size_per_stage;
    smem_size += num_kv_stages * smem_kv_size_per_stage;
    smem_size += num_q_stages * smem_weight_size_per_stage;
    smem_size += num_kv_stages * kv_scale_size_per_stage;
    smem_size += (num_q_stages * 2 + num_kv_stages * 2) * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm120_fp8_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_mqa_logits<
        {}, {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {}
    >);
}};
)", num_heads, head_dim, is_compressed_logits,
    block_q, block_kv, num_q_stages, num_kv_stages,
    num_sms,
    num_tma_threads, num_math_threads, to_string(logits_dtype)));

    // Launch: grid.y carries the KV splits
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, kv_splits, 1),
            .block_dim = dim3(num_tma_threads + num_math_threads, 1, 1),
        },
        seq_len, seq_len_kv, max_seqlen_k, stride_logits,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        logits.data_ptr(),
        tensor_map_q, tensor_map_kv,
        tensor_map_kv_scales, tensor_map_weights
    );
}

static void sm120_fp4_mqa_logits(const torch::Tensor& q,
                                 const torch::Tensor& sf_q,
                                 const torch::Tensor& kv,
                                 const torch::Tensor& sf_kv,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& cu_seq_len_k_start,
                                 const torch::Tensor& cu_seq_len_k_end,
                                 const torch::Tensor& logits,
                                 const at::ScalarType& logits_dtype,
                                 const int& seq_len, const int& seq_len_kv,
                                 const int& max_seqlen_k, const int& stride_logits,
                                 const int& num_heads, const int& head_dim,
                                 const int& block_q, const int& block_kv) {
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    constexpr int num_q_stages = 2, num_kv_stages = 5;
    const bool is_compressed_logits = max_seqlen_k > 0;
    const int num_sms = runtime->get_num_sms();

    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT(128 % num_heads == 0);
    DG_HOST_ASSERT(head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(
        q, head_dim, seq_len * num_heads,
        head_dim, block_q * num_heads, static_cast<int>(q.stride(1)),
        head_dim / 2, 0, false, false);
    const auto tensor_map_sf_q = make_tma_2d_desc(
        sf_q, num_heads, seq_len, num_heads, block_q,
        static_cast<int>(sf_q.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, seq_len, num_heads, block_q,
        static_cast<int>(weights.stride(0)), 0);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv, head_dim, seq_len_kv, head_dim, block_kv,
        static_cast<int>(kv.stride(0)), head_dim / 2, 0, false, false);
    const auto tensor_map_sf_kv = make_tma_2d_desc(
        sf_kv,
        get_tma_aligned_size(seq_len_kv, static_cast<int>(sf_kv.element_size())),
        1, block_kv, 1, 0, 0);

    const int smem_q_size_per_stage = block_q * num_heads * head_dim / 2;
    const int smem_sf_q_size_per_stage = block_q * num_heads * sizeof(int);
    const int smem_kv_size_per_stage = block_kv * head_dim / 2;
    const int smem_sf_kv_size_per_stage = block_kv * sizeof(int);
    const int smem_weight_size_per_stage = block_q * num_heads * sizeof(float);
    const int smem_barriers = (num_q_stages + num_kv_stages) * 2 * 8;
    const int smem_size =
        num_q_stages * (smem_q_size_per_stage + smem_sf_q_size_per_stage +
                        smem_weight_size_per_stage) +
        num_kv_stages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage) +
        smem_barriers;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm120_fp4_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm120_fp4_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp4_mqa_logits<
        {}, {}, {},
        {}, {}, {}, {},
        {}, {}, {}, {}
    >);
}};
)", num_heads, head_dim, is_compressed_logits,
    block_q, block_kv, num_q_stages, num_kv_stages,
    num_sms,
    num_tma_threads, num_math_threads, to_string(logits_dtype)));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_tma_threads + num_math_threads, 1, 1),
        },
        seq_len, seq_len_kv, max_seqlen_k, stride_logits,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        logits.data_ptr(),
        tensor_map_q, tensor_map_sf_q,
        tensor_map_kv, tensor_map_sf_kv, tensor_map_weights
    );
}

static void sm120_mqa_logits(const torch::Tensor& q,
                             const std::optional<torch::Tensor>& sf_q,
                             const torch::Tensor& kv, const torch::Tensor& sf_kv,
                             const torch::Tensor& weights,
                             const torch::Tensor& cu_seq_len_k_start,
                             const torch::Tensor& cu_seq_len_k_end,
                             const torch::Tensor& logits,
                             const at::ScalarType& logits_dtype,
                             const int& num_q_tokens, const int& num_kv_tokens,
                             const int& max_seqlen_k, const int& stride_logits,
                             const int& num_heads, const int& head_dim,
                             const int& block_q, const int& split_kv,
                             const bool& is_mx_sf,
                             const at::ScalarType& qk_dtype) {
    if (qk_dtype == kPackedFP4) {
        DG_HOST_ASSERT(is_mx_sf and sf_q.has_value());
        sm120_fp4_mqa_logits(
            q, sf_q.value(), kv, sf_kv, weights,
            cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
            num_q_tokens, num_kv_tokens, max_seqlen_k, stride_logits,
            num_heads, head_dim, block_q, split_kv);
    } else {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn and not is_mx_sf);
        sm120_fp8_mqa_logits(
            q, kv, sf_kv, weights,
            cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
            num_q_tokens, num_kv_tokens, max_seqlen_k, stride_logits,
            num_heads, head_dim, block_q, split_kv);
    }
}

static void sm120_paged_mqa_logits_metadata(
    const torch::Tensor& context_lens, const torch::Tensor& schedule_metadata,
    const int& batch_size, const int& next_n, const int& block_kv,
    const int& num_sms, const bool& is_context_lens_2d,
    const int& num_next_n_atoms, const bool& is_varlen,
    const int* indices_ptr) {
    constexpr int split_kv = 128;
    constexpr int num_threads = 32;
    const int aligned_batch_size = align(batch_size, 32);
    DG_HOST_ASSERT(split_kv % block_kv == 0);
    const int num_smem_ints =
        is_varlen ? 3 * aligned_batch_size + 1 : aligned_batch_size;
    const int smem_size = num_smem_ints * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm120_paged_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/scheduler/sm120_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm120_paged_mqa_logits_metadata<
        {}, {}, {}, {}
    >);
}};
)", aligned_batch_size, split_kv, num_sms,
    is_varlen ? "true" : "false"));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
        },
        batch_size, next_n, is_context_lens_2d,
        static_cast<uint32_t>(num_next_n_atoms),
        context_lens.data_ptr<int>(),
        const_cast<int*>(indices_ptr),
        schedule_metadata.data_ptr<int>()
    );
}

static void sm120_fp8_paged_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& kv_cache,
    const torch::Tensor& kv_cache_scales, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& logits,
    const torch::Tensor& block_table, const torch::Tensor& indices,
    const torch::Tensor& schedule_meta, const at::ScalarType& logits_dtype,
    const int& batch_size, const int& next_n, const int& num_heads,
    const int& head_dim, const int& num_kv_blocks, const int& block_kv,
    const bool& is_context_lens_2d, const bool& is_varlen,
    const int& logits_stride, const int& block_table_stride,
    const int& num_sms, const int& split_kv) {
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    constexpr int num_q_stages = 2, num_kv_stages = 3;
    const int num_groups = split_kv / block_kv;
    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT(128 % num_heads == 0);
    DG_HOST_ASSERT(block_kv == 64);
    DG_HOST_ASSERT(split_kv == 128 and logits_stride % split_kv == 0);

    const auto tensor_map_q = make_tma_2d_desc(
        q, head_dim, batch_size * next_n * num_heads,
        head_dim, next_n_atom * num_heads, static_cast<int>(q.stride(2)),
        head_dim);
    const auto tensor_map_kv = make_tma_3d_desc(
        kv_cache, head_dim, block_kv, num_kv_blocks,
        head_dim, block_kv, 1, static_cast<int>(kv_cache.stride(1)),
        static_cast<int>(kv_cache.stride(0)), head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_cache_scales, block_kv, num_kv_blocks, block_kv, 1,
        static_cast<int>(kv_cache_scales.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, batch_size * next_n, num_heads, next_n_atom,
        static_cast<int>(weights.stride(0)), 0);

    const int swizzle_alignment = head_dim * 8;
    const int smem_q_size_per_stage =
        next_n_atom * num_heads * head_dim * static_cast<int>(q.element_size());
    const int aligned_smem_weight_size_per_stage = align(
        next_n_atom * num_heads * static_cast<int>(weights.element_size()),
        swizzle_alignment);
    const int smem_q_pipe_size =
        num_q_stages * (smem_q_size_per_stage +
                        aligned_smem_weight_size_per_stage) +
        align(num_q_stages * 8 * 2, swizzle_alignment);
    const int smem_kv_size_per_stage =
        block_kv * head_dim * static_cast<int>(kv_cache.element_size());
    const int aligned_smem_kv_scale_size_per_stage = align(
        block_kv * static_cast<int>(kv_cache_scales.element_size()),
        swizzle_alignment);
    const int smem_kv_pipe_size =
        num_kv_stages * (smem_kv_size_per_stage +
                         aligned_smem_kv_scale_size_per_stage) +
        align(num_kv_stages * 8 * 2, swizzle_alignment);
    const int smem_size =
        smem_q_pipe_size + num_groups * smem_kv_pipe_size + 4;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm120_fp8_paged_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_paged_mqa_logits<
        {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}
    >);
}};
)", next_n, num_heads, head_dim, block_kv,
    is_context_lens_2d, is_varlen ? "true" : "false",
    num_q_stages, num_kv_stages, split_kv,
    num_tma_threads, num_math_threads, to_string(logits_dtype)));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_tma_threads + num_math_threads, 1, 1),
        },
        batch_size, logits_stride, block_table_stride,
        context_lens.data_ptr<int>(), logits.data_ptr(),
        block_table.data_ptr<int>(),
        is_varlen ? indices.data_ptr<int>() : nullptr,
        schedule_meta.data_ptr<int>(),
        tensor_map_q, tensor_map_kv,
        tensor_map_kv_scales, tensor_map_weights
    );
}

static void sm120_fp4_paged_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& sf_q,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& logits, const torch::Tensor& block_table,
    const torch::Tensor& indices, const torch::Tensor& schedule_meta,
    const at::ScalarType& logits_dtype, const int& batch_size,
    const int& next_n, const int& num_heads, const int& head_dim,
    const int& num_kv_blocks, const int& block_kv,
    const bool& is_context_lens_2d, const bool& is_varlen,
    const int& logits_stride, const int& block_table_stride,
    const int& num_sms, const int& split_kv) {
    constexpr int num_tma_threads = 128;
    constexpr int num_math_threads = 256;
    constexpr int num_q_stages = 2, num_kv_stages = 3;
    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT(split_kv == 128 and logits_stride % split_kv == 0);
    DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
    DG_HOST_ASSERT(head_dim == 128);

    const auto tensor_map_q = make_tma_2d_desc(
        q, head_dim, batch_size * next_n * num_heads,
        head_dim, next_n_atom * num_heads, static_cast<int>(q.stride(2)),
        head_dim / 2, 0, false, false);
    const auto tensor_map_sf_q = make_tma_2d_desc(
        sf_q, num_heads, batch_size * next_n, num_heads, next_n_atom,
        static_cast<int>(sf_q.stride(1)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, batch_size * next_n, num_heads, next_n_atom,
        static_cast<int>(weights.stride(0)), 0);
    const auto tensor_map_kv = make_tma_3d_desc(
        kv_cache, head_dim, block_kv, num_kv_blocks,
        head_dim, block_kv, 1, static_cast<int>(kv_cache.stride(1)),
        static_cast<int>(kv_cache.stride(0)), head_dim / 2, 0, false, false);
    const auto tensor_map_sf_kv = make_tma_2d_desc(
        kv_cache_sf, block_kv, num_kv_blocks, block_kv, 1,
        static_cast<int>(kv_cache_sf.stride(0)), 0);

    const int swizzle_alignment = head_dim / 2 * 8;
    const int smem_q_size_per_stage = next_n_atom * num_heads * head_dim / 2;
    const int aligned_smem_sf_q_size_per_stage = align(
        next_n_atom * num_heads * static_cast<int>(sizeof(int)),
        swizzle_alignment);
    const int aligned_smem_weight_size_per_stage = align(
        next_n_atom * num_heads * static_cast<int>(sizeof(float)),
        swizzle_alignment);
    const int smem_q_pipe_size =
        num_q_stages * (smem_q_size_per_stage +
                        aligned_smem_sf_q_size_per_stage +
                        aligned_smem_weight_size_per_stage) +
        align(num_q_stages * 8 * 2, swizzle_alignment);
    const int smem_kv_size_per_stage = block_kv * head_dim / 2;
    const int aligned_smem_sf_kv_size_per_stage = align(
        block_kv * static_cast<int>(sizeof(int)), swizzle_alignment);
    const int smem_kv_pipe_size =
        num_kv_stages * (smem_kv_size_per_stage +
                         aligned_smem_sf_kv_size_per_stage) +
        align(num_kv_stages * 8 * 2, swizzle_alignment);
    const int num_groups = split_kv / block_kv;
    const int smem_size =
        smem_q_pipe_size + num_groups * smem_kv_pipe_size + 4;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm120_fp4_paged_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp4_paged_mqa_logits<
        {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}
    >);
}};
)", next_n, num_heads, head_dim, block_kv,
    is_context_lens_2d, is_varlen ? "true" : "false",
    num_q_stages, num_kv_stages, split_kv,
    num_tma_threads, num_math_threads, to_string(logits_dtype)));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_tma_threads + num_math_threads, 1, 1),
        },
        batch_size, logits_stride, block_table_stride,
        context_lens.data_ptr<int>(), logits.data_ptr(),
        block_table.data_ptr<int>(),
        is_varlen ? indices.data_ptr<int>() : nullptr,
        schedule_meta.data_ptr<int>(),
        tensor_map_q, tensor_map_sf_q,
        tensor_map_kv, tensor_map_sf_kv, tensor_map_weights
    );
}

static void sm120_paged_mqa_logits(
    const torch::Tensor& q, const std::optional<torch::Tensor>& sf_q,
    const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_sf,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& logits, const torch::Tensor& block_table,
    const torch::Tensor& indices, const torch::Tensor& schedule_meta,
    const at::ScalarType& logits_dtype, const int& num_requests,
    const int& num_q_tokens_total, const int& tokens_per_request,
    const int& num_heads, const int& head_dim, const int& num_kv_blocks,
    const int& page_kv, const bool& is_context_lens_2d,
    const bool& is_varlen, const int& logits_stride,
    const int& block_table_stride, const int& num_sms, const int& split_kv,
    const bool& is_mx_sf, const at::ScalarType& qk_dtype) {
    // The SM120 kernel receives q.shape(0) and q.shape(1) separately, just as
    // the pre-#377 launcher did. For varlen, q.shape(0) is already the flattened
    // token count and tokens_per_request is one.
    (void)num_q_tokens_total;
    if (qk_dtype == kPackedFP4) {
        DG_HOST_ASSERT(is_mx_sf and sf_q.has_value());
        sm120_fp4_paged_mqa_logits(
            q, sf_q.value(), kv_cache, kv_cache_sf, weights, context_lens,
            logits, block_table, indices, schedule_meta, logits_dtype,
            num_requests, tokens_per_request, num_heads, head_dim,
            num_kv_blocks, page_kv, is_context_lens_2d, is_varlen,
            logits_stride, block_table_stride, num_sms, split_kv);
    } else {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn and not is_mx_sf);
        sm120_fp8_paged_mqa_logits(
            q, kv_cache, kv_cache_sf, weights, context_lens, logits,
            block_table, indices, schedule_meta, logits_dtype,
            num_requests, tokens_per_request, num_heads, head_dim,
            num_kv_blocks, page_kv, is_context_lens_2d, is_varlen,
            logits_stride, block_table_stride, num_sms, split_kv);
    }
}

} // namespace deep_gemm
