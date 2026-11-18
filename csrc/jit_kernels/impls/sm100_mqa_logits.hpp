#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// The contiguous scheduler balances work across SMs using a cost per KV split and a startup cost per non-empty Q block;
// Q-block startup is modeled as 1.5 KV splits, giving a segment-to-split cost ratio of 3:2
static constexpr int kMQALogitsSplitCost = 2;
static constexpr int kMQALogitsSegmentCost = 3;
static constexpr int kMQALogitsSplitKV = 384;
static constexpr int kMQALogitsNumTmemStages = 3;
static constexpr int kMQALogitsSplitsPerChunk = 8;
static constexpr int kMQALogitsNumSpecializedThreads = 128;

// The Q tile always covers 128 (token, head) rows
static int get_mqa_logits_block_q(const int& num_heads) {
    DG_HOST_ASSERT(num_heads > 0 and num_heads <= 128 and num_heads % 4 == 0);
    return 128 / num_heads;
}

// Paged metadata: per-SM starts as (q_token_idx, kv_split_idx)
static void sm100_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                            const torch::Tensor& schedule_meta,
                                            const int& num_q_tokens_total, const int& num_sms,
                                            const int* indices_ptr) {
    // Tuned for H32, other head counts may have slightly less balanced schedules
    constexpr int block_q = 4;
    const uint32_t num_threads = num_q_tokens_total <= 512 ? 256 : (num_q_tokens_total <= 2048 ? 512 : 1024);
    // Request starts, work prefix, warp sums, and block total
    const int smem_size = (2 * num_q_tokens_total + num_threads / 32 + 1) * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm100_paged_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm100_paged_mqa_logits_metadata<
        {}, {}, {}, {}
    >);
}};
)", kMQALogitsSplitKV, num_sms, block_q, num_threads));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
        },
        num_q_tokens_total,
        context_lens.data_ptr<int>(), const_cast<int*>(indices_ptr), schedule_meta.data_ptr<int>()
    );
}

// Metadata words of the contiguous scheduler: per-SM starts and counts followed by one span per Q block
static int get_mqa_logits_metadata_num_words(const int& num_q_tokens, const int& block_q, const int& num_sms) {
    DG_HOST_ASSERT(num_q_tokens > 0 and block_q > 0 and num_sms > 0);
    const int num_q_blocks = ceil_div(num_q_tokens, block_q);
    // Int32 words: 3 per SM (start pair + split count), aligned to 2, then 2 per Q block (KV span)
    return align(3 * num_sms, 2) + 2 * num_q_blocks;
}

static void sm100_mqa_logits_metadata(const torch::Tensor& cu_seq_len_k_start,
                                      const torch::Tensor& cu_seq_len_k_end,
                                      const torch::Tensor& schedule_meta,
                                      const int& num_q_tokens,
                                      const int& num_kv_tokens,
                                      const int& block_q,
                                      const int& num_sms) {
    DG_STATIC_ASSERT(kMQALogitsSegmentCost > 0 and kMQALogitsSplitCost > 0, "Invalid schedule cost constants");
    constexpr int split_kv = kMQALogitsSplitKV;
    const int num_q_blocks = ceil_div(num_q_tokens, block_q);
    const int64_t total_work_bound = static_cast<int64_t>(num_q_blocks) * ceil_div(num_kv_tokens, split_kv);
    const int64_t total_cost_bound = kMQALogitsSplitCost * total_work_bound + kMQALogitsSegmentCost * num_q_blocks;
    DG_HOST_ASSERT(total_work_bound <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(total_cost_bound <= std::numeric_limits<uint32_t>::max());
    constexpr int num_threads = 256;
    // Per-block work and cost prefixes, warp sums
    const int smem_size = (2 * num_q_blocks + num_threads / 32) * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Compile
    const auto kernel = jit->compile("sm100_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm100_mqa_logits_metadata<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", block_q, split_kv, num_sms,
        kMQALogitsSegmentCost, kMQALogitsSplitCost, num_threads));

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

// Kernel geometry shared by the contiguous and paged runtimes: every math warpgroup consumes one UMMA_M = 128 KV
// tile, the Q tile always covers 128 (token, head) rows, and the FP4 pipelines are deeper (half-size stages)
struct MQALogitsConfig {
    bool is_fp4, is_paged;
    int num_heads, head_dim, block_q, umma_n;
    int num_q_stages, num_kv_stages;
    static constexpr int split_kv = kMQALogitsSplitKV;
    static constexpr int num_tmem_stages = kMQALogitsNumTmemStages;
    static constexpr int num_specialized_threads = kMQALogitsNumSpecializedThreads;
    static constexpr int num_math_threads = (kMQALogitsSplitKV / 128) * 128;

    MQALogitsConfig(const int& num_heads, const int& head_dim, const at::ScalarType& qk_dtype, const bool& is_paged):
            is_fp4(qk_dtype == kPackedFP4), is_paged(is_paged), num_heads(num_heads), head_dim(head_dim),
            block_q(get_mqa_logits_block_q(num_heads)) {
        DG_HOST_ASSERT(qk_dtype == kPackedFP4 or qk_dtype == torch::kFloat8_e4m3fn);
        if (is_fp4)
            DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
        else
            DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        umma_n = align(block_q * num_heads, 8);
        num_q_stages = is_fp4 ? 2 : 3;
        num_kv_stages = is_fp4 ? 8 : 3;
    }

    const char* qk_dtype_name() const {
        return is_fp4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t";
    }

    static auto compile(const std::string& name, const std::string& source) {
        deep_jit::cuda::CompilerOptions options;
        return jit->compile(name, source, options);
    }

    // TMA swizzle mode: one token row of Q / KV
    int swizzle_mode() const {
        return is_fp4 ? head_dim / 2 : head_dim;
    }

    // Weights rows are padded to 16 bytes for TMA
    int num_weight_elements_per_row() const {
        return static_cast<int>(align(num_heads * static_cast<uint32_t>(sizeof(nv_bfloat16)), 16u) / sizeof(nv_bfloat16));
    }

    // Template arguments of `layout::MQALogitsSharedStorage`; the generated source asserts `sizeof` == `smem_size()`
    std::string shared_storage_args() const {
        return std::format("{}, {}, {}, {}, {}, {}, {}, {}, {}",
                           num_heads, head_dim, block_q, split_kv, umma_n, num_q_stages, num_kv_stages, num_tmem_stages,
                           qk_dtype_name());
    }

    // Mirrors `layout::MQALogitsSharedStorage`
    int smem_size() const {
        constexpr uint32_t kNumBarrierBytes = 8;  // mbarrier size and alignment
        constexpr uint32_t kNumUTCCPAlignedElems = 128;
        constexpr uint32_t kTmaAlignment = 128;
        const uint32_t num_qk_bytes_per_token = is_fp4 ? head_dim / 2 : head_dim;
        const uint32_t swizzle_alignment = 8 * num_qk_bytes_per_token;
        const uint32_t num_sf_q = align(static_cast<uint32_t>(umma_n), kNumUTCCPAlignedElems);
        const uint32_t num_sf_kv = align(static_cast<uint32_t>(split_kv), kNumUTCCPAlignedElems);
        const uint32_t num_weight_bytes_per_stage = align(
            static_cast<uint32_t>(block_q * num_weight_elements_per_row() * sizeof(nv_bfloat16)), kTmaAlignment);

        uint32_t num_smem_bytes = 0;
        const auto add_region = [&](const uint32_t& num_region_bytes, const uint32_t& alignment) {
            num_smem_bytes = align(num_smem_bytes, alignment) + num_region_bytes;
        };
        add_region(num_q_stages * umma_n * num_qk_bytes_per_token, swizzle_alignment);
        add_region(num_kv_stages * split_kv * num_qk_bytes_per_token, swizzle_alignment);
        add_region(num_q_stages * num_sf_q * sizeof(uint32_t), kTmaAlignment);
        add_region(num_kv_stages * num_sf_kv * sizeof(uint32_t), kTmaAlignment);
        add_region(num_q_stages * num_weight_bytes_per_stage, kTmaAlignment);
        const uint32_t num_barriers = 3 * (num_q_stages + num_kv_stages) + 2 * num_tmem_stages;
        add_region(num_barriers * kNumBarrierBytes, kNumBarrierBytes);
        add_region(sizeof(uint32_t), alignof(uint32_t));
        const int smem_size = static_cast<int>(align(num_smem_bytes, swizzle_alignment));
        DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);
        return smem_size;
    }

    // `weights` is `[num_q_tokens, num_heads]` BF16 with an arbitrary row stride
    CUtensorMap make_tensor_map_weights(const torch::Tensor& weights, const int& num_q_tokens) const {
        return make_tma_2d_desc(weights, num_heads, num_q_tokens,
                                num_weight_elements_per_row(), block_q,
                                static_cast<int>(weights.stride(0)), 0);
    }
};

// Unified contiguous-KV runtime for MXFP4 / MXFP8
static void sm100_mqa_logits(const MQALogitsConfig& config,
                             const torch::Tensor& q, const torch::Tensor& sf_q,
                             const torch::Tensor& kv, const torch::Tensor& sf_kv,
                             const torch::Tensor& weights,
                             const torch::Tensor& cu_seq_len_k_start,
                             const torch::Tensor& cu_seq_len_k_end,
                             const torch::Tensor& logits,
                             const int& num_q_tokens, const int& num_kv_tokens,
                             const int& stride_logits,
                             const std::optional<torch::Tensor>& schedule_meta) {
    DG_HOST_ASSERT(not config.is_paged);
    const int num_sms = runtime->get_num_sms();

    if (schedule_meta.has_value()) {
        const auto& workspace = schedule_meta.value();
        DG_HOST_ASSERT(workspace.is_cuda());
        DG_HOST_ASSERT(workspace.device() == q.device());
        DG_HOST_ASSERT(workspace.scalar_type() == torch::kInt32);
        DG_HOST_ASSERT(workspace.is_contiguous());
        // Metadata is accessed as 8-byte words
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 8 == 0);
        const int required_words = get_mqa_logits_metadata_num_words(num_q_tokens, config.block_q, num_sms);
        DG_HOST_ASSERT(workspace.numel() >= required_words);
    }

    const auto tensor_map_q = make_tma_2d_desc(q, config.head_dim, num_q_tokens * config.num_heads,
                                               config.head_dim, config.block_q * config.num_heads,
                                               static_cast<int>(q.stride(1)),
                                               config.swizzle_mode(), 0, false, not config.is_fp4);
    const int kv_tma_tokens = config.split_kv % 256 == 0 ? 256 : 128;
    const auto tensor_map_kv = make_tma_2d_desc(kv, config.head_dim, num_kv_tokens,
                                                config.head_dim, kv_tma_tokens,
                                                static_cast<int>(kv.stride(0)),
                                                config.swizzle_mode(), 0, false, not config.is_fp4);
    const auto tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                                   get_tma_aligned_size(num_kv_tokens, static_cast<int>(sf_kv.element_size())), 1,
                                                   kv_tma_tokens, 1, 0, 0);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, config.num_heads, num_q_tokens,
                                                  config.num_heads, config.block_q,
                                                  static_cast<int>(sf_q.stride(0)), 0);
    const auto tensor_map_weights = config.make_tensor_map_weights(weights, num_q_tokens);
    const int smem_size = config.smem_size();

    // Compile
    const auto kernel = MQALogitsConfig::compile("sm100_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {}
>) == {}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mqa_logits<
        {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
    config.shared_storage_args(), smem_size,
    config.num_heads, config.head_dim,
    schedule_meta.has_value(),
    config.block_q, config.split_kv, config.umma_n,
    config.num_q_stages, config.num_kv_stages, config.num_tmem_stages,
    num_sms,
    config.num_specialized_threads, config.num_math_threads,
    config.qk_dtype_name()));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_specialized_threads + config.num_math_threads, 1, 1),
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

// Unified paged runtime for MXFP4 / MXFP8
static void sm100_paged_mqa_logits(const MQALogitsConfig& config,
                                   const torch::Tensor& q,
                                   const torch::Tensor& sf_q,
                                   const torch::Tensor& kv_cache,
                                   const torch::Tensor& kv_cache_sf,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& context_lens,
                                   const torch::Tensor& logits,
                                   const torch::Tensor& block_table,
                                   const torch::Tensor& indices,
                                   const torch::Tensor& schedule_meta,
                                   const int& num_q_tokens_total,
                                   const int& num_kv_blocks, const int& page_kv,
                                   const int& logits_stride,
                                   const int& block_table_stride) {
    DG_HOST_ASSERT(config.is_paged);
    DG_HOST_ASSERT(logits_stride % config.split_kv == 0);
    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(schedule_meta.size(0) == num_sms + 1);

    const auto tensor_map_q = make_tma_2d_desc(q, config.head_dim, num_q_tokens_total * config.num_heads,
                                               config.head_dim, config.block_q * config.num_heads,
                                               static_cast<int>(q.stride(2)),
                                               config.swizzle_mode(), 0, false, not config.is_fp4);
    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, config.head_dim, page_kv, num_kv_blocks,
                                                config.head_dim, page_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                config.swizzle_mode(), 0, false, not config.is_fp4);
    const auto tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, page_kv, num_kv_blocks,
                                                   page_kv, 1,
                                                   static_cast<int>(kv_cache_sf.stride(0)), 0);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, config.num_heads, num_q_tokens_total,
                                                  config.num_heads, config.block_q,
                                                  static_cast<int>(sf_q.stride(1)), 0);
    const auto tensor_map_weights = config.make_tensor_map_weights(weights, num_q_tokens_total);
    const int smem_size = config.smem_size();

    // Compile
    const auto kernel = MQALogitsConfig::compile("sm100_paged_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;

static_assert(sizeof(layout::MQALogitsSharedStorage<
    {}
>) == {}, "Incorrect MQA logits shared-memory size");

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        {}, {}, {},
        {}, {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
    config.shared_storage_args(), smem_size,
    config.num_heads, config.head_dim,
    page_kv,
    config.block_q, config.umma_n,
    config.num_q_stages, config.num_kv_stages, config.num_tmem_stages,
    config.split_kv, kMQALogitsSplitsPerChunk,
    config.num_specialized_threads, config.num_math_threads,
    config.qk_dtype_name()));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_specialized_threads + config.num_math_threads, 1, 1),
        },
        num_q_tokens_total,
        logits_stride, block_table_stride,
        context_lens.data_ptr<int>(), logits.data_ptr(),
        block_table.data_ptr<int>(), indices.data_ptr<int>(), schedule_meta.data_ptr<int>(),
        tensor_map_q, tensor_map_sf_q,
        tensor_map_kv, tensor_map_sf_kv,
        tensor_map_weights
    );
}

} // namespace deep_gemm
