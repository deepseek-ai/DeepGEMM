#pragma once

#include <format>

#include <deep_gemm/layout/sparse_mqa_logits.cuh>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

using namespace layout::sparse_mqa_logits;

static int get_sparse_mqa_split_kv(const at::ScalarType qk_dtype) {
    DG_HOST_ASSERT(qk_dtype == kPackedFP4 or qk_dtype == torch::kFloat8_e4m3fn);
    return qk_dtype == kPackedFP4 ? 640 : 512;
}

static int64_t get_num_metadata_bytes(const int num_q_tokens, const int num_max_sparse_blocks,
                                      const int sparse_block_kv, const int split_kv,
                                      const bool is_paged, const int num_sms) {
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const int64_t num_kv_blocks_per_split = split_kv / sparse_block_kv;
    const int64_t num_kv_split_bytes = sizeof(KVSplitHeader) + num_kv_blocks_per_split * sizeof(KVBlockInfo);
    const int64_t num_max_kv_splits = is_paged ?
        static_cast<int64_t>(num_q_tokens) * ceil_div<int64_t>(num_max_sparse_blocks, num_kv_blocks_per_split) :
        ceil_div<int64_t>(num_q_tokens, kBlockQ) *
            ceil_div<int64_t>(kBlockQ * static_cast<int64_t>(num_max_sparse_blocks), num_kv_blocks_per_split);
    DG_HOST_ASSERT(num_max_kv_splits <= std::numeric_limits<uint32_t>::max());
    const int64_t num_max_schedule_entries = align<int64_t>(num_max_kv_splits, num_sms);
    DG_HOST_ASSERT(num_max_schedule_entries <= std::numeric_limits<uint32_t>::max());
    return sizeof(MetadataHeader) + num_max_kv_splits * num_kv_split_bytes +
           num_max_schedule_entries * static_cast<int64_t>(sizeof(ScheduleEntry));
}

static void launch_sm100_sparse_mqa_logits(const bool is_paged, const bool use_unaligned_ks,
                                           const int sparse_block_kv,
                                           const torch::Tensor& q, const torch::Tensor& sf_q,
                                           const torch::Tensor& kv, const torch::Tensor& sf_kv,
                                           const torch::Tensor& weights, const torch::Tensor& metadata,
                                           const torch::Tensor& logits) {
    constexpr int kNumQStages = 2;
    constexpr int kNumTmemStages = 5;
    const bool is_fp4 = q.scalar_type() == kPackedFP4;
    const int split_kv = get_sparse_mqa_split_kv(q.scalar_type());
    const int num_math_warpgroups = split_kv / 128;
    const int num_kv_stages = is_fp4 ? 5 : 3;
    const auto qk_dtype_name = is_fp4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t";
    DG_HOST_ASSERT(not is_paged or not use_unaligned_ks);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    DG_HOST_ASSERT(metadata.dim() == 1 and metadata.scalar_type() == torch::kUInt8 and metadata.is_contiguous() and
                   metadata.numel() >= static_cast<int64_t>(sizeof(MetadataHeader)));
    const int num_q_tokens = static_cast<int>(q.size(0));
    const int swizzle_mode = is_fp4 ? kHeadDim / 2 : kHeadDim;
    const auto tensor_map_q = make_tma_2d_desc(q, kHeadDim, num_q_tokens * kNumHeads, kHeadDim, kBlockQ * kNumHeads,
                                               static_cast<int>(q.stride(is_paged ? 2 : 1)), swizzle_mode,
                                               0, false, not is_fp4);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, kNumHeads, num_q_tokens, kNumHeads, kBlockQ,
                                                  static_cast<int>(sf_q.stride(is_paged ? 1 : 0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, kNumHeads, num_q_tokens, kNumHeads, kBlockQ,
                                                     static_cast<int>(weights.stride(0)), 0);
    CUtensorMap tensor_map_kv{};
    CUtensorMap tensor_map_sf_kv{};
    if (not is_paged) {
        const int num_kv_tokens = static_cast<int>(kv.size(0));
        tensor_map_kv = make_tma_2d_desc(kv, kHeadDim, num_kv_tokens, kHeadDim, kNumKVTokensPerTMA,
                                         static_cast<int>(kv.stride(0)), swizzle_mode, 0, false, not is_fp4);
        tensor_map_sf_kv = make_tma_2d_desc(sf_kv, get_tma_aligned_size(num_kv_tokens, static_cast<int>(sf_kv.element_size())), 1,
                                            kNumKVTokensPerTMA, 1, 0, 0);
    }
    const int num_smem_bytes = is_fp4 ?
        (sparse_block_kv == 8 ?
            static_cast<int>(sizeof(SharedStorage<kBlockQ, 8, 640, kNumQStages, 5, kNumTmemStages, cutlass::float_e2m1_t>)) :
            static_cast<int>(sizeof(SharedStorage<kBlockQ, 16, 640, kNumQStages, 5, kNumTmemStages, cutlass::float_e2m1_t>))) :
        (sparse_block_kv == 8 ?
            static_cast<int>(sizeof(SharedStorage<kBlockQ, 8, 512, kNumQStages, 3, kNumTmemStages, cutlass::float_e4m3_t>)) :
            static_cast<int>(sizeof(SharedStorage<kBlockQ, 16, 512, kNumQStages, 3, kNumTmemStages, cutlass::float_e4m3_t>)));
    DG_HOST_ASSERT(num_smem_bytes <= SM100ArchSpec::smem_capacity);

    const int num_sms = runtime->get_num_sms();
    const auto instantiate = is_paged ? std::format(R"(
    auto ptr = reinterpret_cast<void*>(&sm100_paged_sparse_mqa_logits<
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
)", static_cast<int>(kv.size(1)), sparse_block_kv, kNumQStages, num_kv_stages, kNumTmemStages,
        num_math_warpgroups, num_sms, kBlockQ, is_fp4) : std::format(R"(
    auto ptr = reinterpret_cast<void*>(&sm100_sparse_mqa_logits<
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
)", sparse_block_kv, kNumQStages, num_kv_stages, kNumTmemStages, num_math_warpgroups, num_sms, kBlockQ,
        use_unaligned_ks, is_fp4);
    const auto kernel_name = is_paged ? "sm100_paged_sparse_mqa_logits" : "sm100_sparse_mqa_logits";
    const auto kernel = jit->compile(kernel_name, std::format(R"(
#include <deep_gemm/impls/sm100_sparse_mqa_logits.cuh>

using namespace deep_gemm;
using namespace deep_gemm::layout::sparse_mqa_logits;

static_assert(sizeof(SharedStorage<
    {}, {}, {}, {}, {}, {}, {}
>) == {}, "Incorrect sparse MQA logits shared-memory size");

static void __instantiate_kernel() {{
{}
}}
)", kBlockQ, sparse_block_kv, split_kv, kNumQStages, num_kv_stages, kNumTmemStages,
        qk_dtype_name, num_smem_bytes, instantiate));

    if (is_paged) {
        jit->launch(
            kernel, {
                .num_smem_bytes = num_smem_bytes,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(get_num_threads(num_math_warpgroups), 1, 1),
            },
            static_cast<int>(logits.stride(0)), static_cast<int>(kv.stride(0)), logits.data_ptr(),
            kv.data_ptr(), metadata.data_ptr<uint8_t>(), tensor_map_q, tensor_map_sf_q, tensor_map_weights
        );
    } else {
        jit->launch(
            kernel, {
                .num_smem_bytes = num_smem_bytes,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(get_num_threads(num_math_warpgroups), 1, 1),
            },
            static_cast<int>(logits.stride(0)), logits.data_ptr(), kv.data_ptr(), sf_kv.data_ptr<int>(),
            metadata.data_ptr<uint8_t>(), tensor_map_q, tensor_map_sf_q, tensor_map_weights,
            tensor_map_kv, tensor_map_sf_kv
        );
    }
}

static void launch_sm100_sparse_mqa_logits_metadata(const bool is_paged,
                                                    const bool use_unaligned_ks,
                                                    const int page_kv,
                                                    const int num_kv_tokens,
                                                    const int block_table_stride,
                                                    const torch::Tensor& sparse_kv_block_indices,
                                                    const torch::Tensor& metadata,
                                                    const torch::Tensor& workspace,
                                                    const int split_kv,
                                                    const int sparse_block_kv,
                                                    const int* cu_seq_len_k_start,
                                                    const int* cu_seq_len_k_end,
                                                    const int* context_lens,
                                                    const int* block_table,
                                                    const int* indices) {
    constexpr int kNumMetadataThreads = 256;
    constexpr int kNumKVSplitsPerEntry = 8;
    DG_HOST_ASSERT(not is_paged or not use_unaligned_ks);
    const int num_q_tokens = static_cast<int>(sparse_kv_block_indices.size(0));
    const int num_max_sparse_blocks = static_cast<int>(sparse_kv_block_indices.size(1));
    const int num_sms = runtime->get_num_sms();
    const int num_ctas = std::min(is_paged ? num_q_tokens : ceil_div<int>(num_q_tokens, kBlockQ), num_sms * 4);
    const int num_max_merged_kv_blocks = kBlockQ * num_max_sparse_blocks;
    const int num_max_kv_splits = ceil_div<int>(num_max_merged_kv_blocks, split_kv / sparse_block_kv);
    int num_smem_bytes = (kBlockQ * num_max_sparse_blocks + num_max_merged_kv_blocks + num_max_kv_splits) *
                         static_cast<int>(sizeof(uint32_t));
    num_smem_bytes += (kNumMetadataThreads / 32) * static_cast<int>(sizeof(uint32_t));
    num_smem_bytes += (1 + kNumMetadataThreads / 32 * (kNumKVSplitsPerEntry + 1)) * static_cast<int>(sizeof(uint32_t));
    num_smem_bytes = align(num_smem_bytes, 16);
    const int num_q_block_bytes = align<int>((3 + kBlockQ) * static_cast<int>(sizeof(uint32_t)), 16);
    num_smem_bytes = align(num_smem_bytes + num_q_block_bytes + static_cast<int>(sizeof(uint32_t)), 128);
    DG_HOST_ASSERT(num_ctas > 0 and num_smem_bytes <= SM100ArchSpec::smem_capacity);

    const auto kernel = jit->compile("sm100_sparse_mqa_logits_metadata", std::format(R"(
#include <deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh>

using namespace deep_gemm;
using namespace deep_gemm::sched::sparse_mqa_logits;

static void __instantiate_kernel() {{
    static_assert((sizeof(sched::sparse_mqa_logits::SharedStorage<
        {}, {}, {} / {}, {}, {}
    >) + 127) / 128 * 128 == {}, "Incorrect sparse MQA metadata shared-memory size");
    auto ptr = reinterpret_cast<void*>(&sm100_sparse_mqa_logits_metadata<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", kBlockQ, num_max_sparse_blocks, split_kv, sparse_block_kv,
        kNumKVSplitsPerEntry, kNumMetadataThreads, num_smem_bytes,
        is_paged ? "true" : "false", use_unaligned_ks ? "true" : "false",
        kBlockQ, split_kv, sparse_block_kv,
        num_max_sparse_blocks, page_kv, kNumKVSplitsPerEntry, num_sms, kNumMetadataThreads));

    jit->launch(
        kernel, {
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_ctas, 1, 1),
            .block_dim = dim3(kNumMetadataThreads, 1, 1),
        },
        num_q_tokens, num_kv_tokens,
        cu_seq_len_k_start, cu_seq_len_k_end,
        context_lens, block_table, block_table_stride,
        indices,
        sparse_kv_block_indices.data_ptr<int>(), metadata.data_ptr<uint8_t>(), workspace.data_ptr<uint8_t>()
    );
}

} // namespace deep_gemm
