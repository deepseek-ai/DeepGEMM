#pragma once

#include <format>

#include <deep_gemm/layout/mqa_logits.cuh>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// SM100 paged metadata emits per-SM starts as (q_token_idx, kv_split_idx)
static void sm100_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                            const torch::Tensor& schedule_meta,
                                            const int& num_requests, const int& num_q_tokens_total,
                                            const int& next_n, const int& num_sms,
                                            const bool& is_context_lens_2d, const bool& is_varlen,
                                            const int* indices_ptr) {
    constexpr int split_kv = 256;
    constexpr int num_threads = 1024;
    // Request starts (varlen only), work prefix, warp sums, and block total
    const int smem_size = ((is_varlen ? 2 : 1) * num_requests + num_threads / 32 + 1) * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm100_paged_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm100_paged_mqa_logits_metadata<
        {}, {}, {}, {}, {}
    >);
}};
)", next_n, is_context_lens_2d ? "true" : "false", is_varlen ? "true" : "false",
    split_kv, num_sms));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
        },
        num_requests, num_q_tokens_total,
        context_lens.data_ptr<int>(), const_cast<int*>(indices_ptr), schedule_meta.data_ptr<int>()
    );
}

// Costs in units of one K split of main-loop work
static constexpr int kMQALogitsSplitCost = 1;
static constexpr int kMQALogitsSegmentCost = 1;
// Unroll depth of the builder's boundary search
static constexpr int kMQALogitsMaxNumQBlocksLog2 = 15;

static int get_mqa_logits_metadata_num_words(const int& num_q_tokens, const int& block_q, const int& num_sms) {
    DG_HOST_ASSERT(num_q_tokens > 0 and block_q > 0 and num_sms > 0);
    const int num_q_blocks = math::ceil_div(num_q_tokens, block_q);
    DG_HOST_ASSERT(num_q_blocks <= (1 << kMQALogitsMaxNumQBlocksLog2));
    return (3 * num_sms + 1) / 2 * 2 + 2 * num_q_blocks;
}

static void sm100_mqa_logits_metadata(const torch::Tensor& cu_seq_len_k_start,
                                      const torch::Tensor& cu_seq_len_k_end,
                                      const torch::Tensor& schedule_meta,
                                      const int& num_q_tokens,
                                      const int& num_kv_tokens,
                                      const int& block_q,
                                      const int& split_kv,
                                      const int& num_sms) {
    DG_STATIC_ASSERT(kMQALogitsSegmentCost > 0 and kMQALogitsSplitCost > 0, "Invalid schedule cost constants");
    const int num_q_blocks = math::ceil_div(num_q_tokens, block_q);
    const int64_t total_work_bound = static_cast<int64_t>(num_q_blocks) * math::ceil_div(num_kv_tokens, split_kv);
    const int64_t total_cost_bound = kMQALogitsSplitCost * total_work_bound + kMQALogitsSegmentCost * num_q_blocks;
    DG_HOST_ASSERT(total_work_bound <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(total_cost_bound <= std::numeric_limits<uint32_t>::max());
    constexpr int num_threads = 256;
    // The builder keeps the per-block work and cost prefixes in shared memory
    const int smem_size = 2 * num_q_blocks * static_cast<int>(sizeof(uint32_t));
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm100_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/scheduler/sm100_mqa_logits_metadata.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm100_mqa_logits_metadata<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", block_q, split_kv, num_sms,
        kMQALogitsSegmentCost, kMQALogitsSplitCost, kMQALogitsMaxNumQBlocksLog2));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
        },
        num_q_tokens, num_kv_tokens,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        schedule_meta.data_ptr<int>()
    );
}

static int get_mqa_logits_smem_size(const int& num_heads, const int& head_dim,
                                    const bool& is_mx_sf, const at::ScalarType& qk_dtype,
                                    const at::ScalarType& weights_dtype,
                                    const int& block_q, const int& split_kv,
                                    const int& num_q_stages, const int& num_kv_stages,
                                    const int& num_tmem_stages) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kTmaAlignment = 128;
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const uint32_t num_qk_bytes_per_token = is_fp4 ? head_dim / 2 : head_dim;
    const uint32_t swizzle_alignment = 8 * num_qk_bytes_per_token;
    const uint32_t block_qh = block_q * num_heads;
    const uint32_t umma_n = math::align(block_qh, 8u);
    const uint32_t num_sf_q = is_mx_sf ? math::align(umma_n, kNumUTCCPAlignedElems) : 1;
    const uint32_t num_sf_kv = is_mx_sf ? math::align(static_cast<uint32_t>(split_kv), kNumUTCCPAlignedElems) : split_kv;
    const uint32_t num_reduce_bytes = weights_dtype == torch::kBFloat16 ? sizeof(nv_bfloat16) : sizeof(float);

    uint32_t num_smem_bytes = 0;
    const auto add_region = [&](const uint32_t& num_region_bytes, const uint32_t& alignment) {
        num_smem_bytes = math::align(num_smem_bytes, alignment) + num_region_bytes;
    };
    add_region(num_q_stages * umma_n * num_qk_bytes_per_token, swizzle_alignment);
    add_region(num_kv_stages * split_kv * num_qk_bytes_per_token, swizzle_alignment);
    add_region(num_q_stages * num_sf_q * sizeof(uint32_t), kTmaAlignment);
    add_region(num_kv_stages * num_sf_kv * sizeof(uint32_t), kTmaAlignment);
    const uint32_t num_weight_bytes_per_row = math::align(num_heads * num_reduce_bytes, 16u);
    const uint32_t num_weight_bytes_per_stage = math::align(block_q * num_weight_bytes_per_row, kTmaAlignment);
    add_region(num_q_stages * num_weight_bytes_per_stage, kTmaAlignment);
    const uint32_t num_barriers = 2 * (num_q_stages + num_kv_stages + num_tmem_stages);
    add_region(num_barriers * sizeof(Barrier), alignof(Barrier));
    add_region(sizeof(uint32_t), alignof(uint32_t));
    return static_cast<int>(math::align(num_smem_bytes, swizzle_alignment));
}

// Unified contiguous-KV runtime for FP8 / MXFP4 / MXFP8; FP8 reuses the unused `sf_q` descriptor slot
static void sm100_mqa_logits(const torch::Tensor& q, const std::optional<torch::Tensor>& sf_q,
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
                             const bool& clean_logits,
                             const bool& is_mx_sf,
                             const at::ScalarType& qk_dtype,
                             const std::optional<torch::Tensor>& schedule_meta) {
    const bool is_fp4 = qk_dtype == kPackedFP4;

    constexpr int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    const int num_q_stages = 3;
    // Use the deepest KV pipeline that fits with headroom.
    const int num_kv_stages = is_fp4 ? 10 : 5;

    const bool is_compressed_logits = (max_seqlen_k > 0);
    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(not (clean_logits and is_compressed_logits));

    if (schedule_meta.has_value()) {
        const auto& workspace = schedule_meta.value();
        DG_HOST_ASSERT(workspace.is_cuda());
        DG_HOST_ASSERT(workspace.device() == q.device());
        DG_HOST_ASSERT(workspace.scalar_type() == torch::kInt32);
        DG_HOST_ASSERT(workspace.is_contiguous());
        // Metadata is accessed as 8-byte words
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 8 == 0);
        const int required_words = get_mqa_logits_metadata_num_words(num_q_tokens, block_q, num_sms);
        DG_HOST_ASSERT(workspace.numel() >= required_words);
    }
    // MX SF formats consume `sf_q`; FP8 fills that descriptor slot with KV scales
    CUtensorMap tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv;
    if (is_fp4)
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
    else
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);

    const int swizzle_mode = is_fp4 ? head_dim / 2 : head_dim;
    tensor_map_q = make_tma_2d_desc(q, head_dim, num_q_tokens * num_heads,
                                    head_dim, block_q * num_heads,
                                    static_cast<int>(q.stride(1)),
                                    swizzle_mode, 0, false, not is_fp4);
    tensor_map_kv = make_tma_2d_desc(kv, head_dim, num_kv_tokens,
                                     head_dim, split_kv,
                                     static_cast<int>(kv.stride(0)),
                                     swizzle_mode, 0, false, not is_fp4);
    tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                        get_tma_aligned_size(num_kv_tokens, static_cast<int>(sf_kv.element_size())), 1,
                                        split_kv, 1, 0, 0);
    if (is_mx_sf) {
        tensor_map_sf_q = make_tma_2d_desc(sf_q.value(), num_heads, num_q_tokens,
                                           num_heads, block_q,
                                           static_cast<int>(sf_q.value().stride(0)), 0);
    } else {
        tensor_map_sf_q = tensor_map_sf_kv;  // unused by FP8
    }
    const uint32_t weights_stride = static_cast<uint32_t>(weights.stride(0));
    const uint32_t num_weight_row_bytes = num_heads * static_cast<uint32_t>(weights.element_size());
    const uint32_t num_weight_elements_per_row = math::align(num_weight_row_bytes, 16u) / static_cast<uint32_t>(weights.element_size());
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, num_q_tokens,
        static_cast<int>(num_weight_elements_per_row), block_q,
        static_cast<int>(weights_stride), 0);

    const int smem_size = get_mqa_logits_smem_size(
        num_heads, head_dim, is_mx_sf, qk_dtype, weights.scalar_type(),
        block_q, split_kv, num_q_stages, num_kv_stages, 3);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm100_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {}, {}, {}, {}, {}, {}, {}, 3, {}, {}
>) == {}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mqa_logits<
        {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
    num_heads, head_dim, is_mx_sf ? "true" : "false",
    block_q, split_kv, num_q_stages, num_kv_stages,
    qk_dtype == kPackedFP4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t",
    weights.scalar_type() == torch::kBFloat16 ? "__nv_bfloat16" : "float",
    smem_size,
    num_heads, head_dim,
    is_mx_sf ? "true" : "false",
    is_compressed_logits, clean_logits,
    schedule_meta.has_value(),
    block_q, split_kv,
    num_q_stages, num_kv_stages,
    num_sms,
    num_specialized_threads, num_math_threads,
    qk_dtype == kPackedFP4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t",
    to_string(logits_dtype),
    weights.scalar_type() == torch::kBFloat16 ? "__nv_bfloat16" : "float"));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        },
        num_q_tokens, num_kv_tokens,
        stride_logits,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        schedule_meta.has_value() ? schedule_meta.value().data_ptr<int>() : nullptr,
        logits.data_ptr(),
        tensor_map_q, tensor_map_sf_q,
        tensor_map_kv, tensor_map_sf_kv,
        tensor_map_weights
    );
}

// Paged variant: separate host/runtime path, shared device core

// Unified paged runtime for FP8 / MXFP4 / MXFP8; FP8 reuses the unused `sf_q` descriptor slot
static void sm100_paged_mqa_logits(const torch::Tensor& q,
                                   const std::optional<torch::Tensor>& sf_q,
                                   const torch::Tensor& kv_cache,
                                   const torch::Tensor& kv_cache_sf,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& context_lens,
                                   const torch::Tensor& logits,
                                   const torch::Tensor& block_table,
                                   const torch::Tensor& indices,
                                   const torch::Tensor& schedule_meta,
                                   const at::ScalarType& logits_dtype,
                                   const int& num_requests, const int& num_q_tokens_total,
                                   const int& tokens_per_request,
                                   const int& num_heads, const int& head_dim,
                                   const int& num_kv_blocks, const int& page_kv,
                                   const bool& is_context_lens_2d,
                                   const bool& is_varlen,
                                   const int& logits_stride,
                                   const int& block_table_stride,
                                   const int& num_sms,
                                   const int& split_kv,
                                   const int& splits_per_chunk,
                                   const bool& is_mx_sf,
                                   const at::ScalarType& qk_dtype) {
    const bool is_fp4 = qk_dtype == kPackedFP4;

    const int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    DG_HOST_ASSERT(split_kv == 256 and logits_stride % split_kv == 0);

    const int num_q_stages = 3;
    // Match contiguous-KV pipeline depth.
    const int num_kv_stages = is_fp4 ? 10 : 5;
    DG_HOST_ASSERT(num_heads > 0 and num_heads <= 128 and num_heads % 4 == 0);
    const int block_q = 128 / num_heads;

    // MX SF formats consume `sf_q`; FP8 fills that descriptor slot with KV scales
    CUtensorMap tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv;
    if (is_fp4)
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
    else
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);

    const int swizzle_mode = is_fp4 ? head_dim / 2 : head_dim;
    tensor_map_q = make_tma_2d_desc(q, head_dim, num_requests * tokens_per_request * num_heads,
                                    head_dim, block_q * num_heads,
                                    static_cast<int>(q.stride(2)),
                                    swizzle_mode, 0, false, not is_fp4);
    tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, page_kv, num_kv_blocks,
                                     head_dim, page_kv, 1,
                                     static_cast<int>(kv_cache.stride(1)),
                                     static_cast<int>(kv_cache.stride(0)),
                                     swizzle_mode, 0, false, not is_fp4);
    tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, page_kv, num_kv_blocks,
                                        page_kv, 1,
                                        static_cast<int>(kv_cache_sf.stride(0)), 0);
    if (is_mx_sf) {
        tensor_map_sf_q = make_tma_2d_desc(sf_q.value(), num_heads, num_requests * tokens_per_request,
                                           num_heads, block_q,
                                           static_cast<int>(sf_q.value().stride(1)), 0);
    } else {
        tensor_map_sf_q = tensor_map_sf_kv;  // unused by FP8
    }
    const uint32_t weights_stride = static_cast<uint32_t>(weights.stride(0));
    const uint32_t num_weight_row_bytes = num_heads * static_cast<uint32_t>(weights.element_size());
    const uint32_t num_weight_elements_per_row = math::align(num_weight_row_bytes, 16u) / static_cast<uint32_t>(weights.element_size());
    const auto tensor_map_weights = make_tma_2d_desc(
        weights, num_heads, num_requests * tokens_per_request,
        static_cast<int>(num_weight_elements_per_row), block_q,
        static_cast<int>(weights_stride), 0);

    const int smem_size = get_mqa_logits_smem_size(
        num_heads, head_dim, is_mx_sf, qk_dtype, weights.scalar_type(),
        block_q, split_kv, num_q_stages, num_kv_stages, 3);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm100_paged_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {}, {}, {}, {}, {}, {}, {}, 3, {}, {}
>) == {}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {}, {}
    >);
}};
)",
    num_heads, head_dim, is_mx_sf ? "true" : "false",
    block_q, split_kv, num_q_stages, num_kv_stages,
    qk_dtype == kPackedFP4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t",
    weights.scalar_type() == torch::kBFloat16 ? "__nv_bfloat16" : "float",
    smem_size,
    tokens_per_request, num_heads,
    head_dim, page_kv,
    is_mx_sf ? "true" : "false",
    is_context_lens_2d, is_varlen ? "true" : "false",
    num_q_stages, num_kv_stages,
    split_kv, splits_per_chunk,
    num_specialized_threads, num_math_threads,
    qk_dtype == kPackedFP4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t",
    to_string(logits_dtype),
    weights.scalar_type() == torch::kBFloat16 ? "__nv_bfloat16" : "float"));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        },
        num_q_tokens_total,
        logits_stride, block_table_stride,
        context_lens.data_ptr<int>(), logits.data_ptr(),
        block_table.data_ptr<int>(), is_varlen ? indices.data_ptr<int>() : nullptr, schedule_meta.data_ptr<int>(),
        tensor_map_q, tensor_map_sf_q,
        tensor_map_kv, tensor_map_sf_kv,
        tensor_map_weights
    );
}

} // namespace deep_gemm
