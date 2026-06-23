#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"
#include <deep_gemm/layout/mqa_logits.cuh>

namespace deep_gemm {

// SM100 paged metadata emits per-SM starts as (q_token_idx, kv_split_idx)
class SM100PagedMQALogitsMetadataRuntime final: public LaunchRuntime<SM100PagedMQALogitsMetadataRuntime> {
public:
    struct Args {
        int next_n;
        bool is_context_lens_2d;
        bool is_varlen;
        int split_kv;
        int num_sms;

        int num_requests;
        int num_q_tokens_total;
        int* context_lens;
        int* indices;
        int* schedule_meta;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Metadata weights by q-token count, so BLOCK_Q is a placeholder here
        return fmt::format(R"(
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::sm100_paged_mqa_logits_metadata<
        {}, {}, {}, 1, {}, {}
    >);
}};
)", args.next_n, args.is_context_lens_2d ? "true" : "false", args.is_varlen ? "true" : "false",
    args.split_kv, args.num_sms);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.num_requests, args.num_q_tokens_total,
            args.context_lens, args.indices, args.schedule_meta
        ));
    }
};

static void sm100_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                            const torch::Tensor& schedule_meta,
                                            const int& num_requests, const int& num_q_tokens_total,
                                            const int& next_n, const int& num_sms,
                                            const bool& is_context_lens_2d, const bool& is_varlen,
                                            const int* indices_ptr) {
    constexpr int split_kv = 256;
    const int num_threads = 256;
    // smem: prefix_work[num_requests] + request_q_token_start[num_requests]
    const int smem_size = 2 * num_requests * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    const SM100PagedMQALogitsMetadataRuntime::Args args = {
        .next_n = next_n,
        .is_context_lens_2d = is_context_lens_2d,
        .is_varlen = is_varlen,
        .split_kv = split_kv,
        .num_sms = num_sms,
        .num_requests = num_requests,
        .num_q_tokens_total = num_q_tokens_total,
        .context_lens = context_lens.data_ptr<int>(),
        .indices = const_cast<int*>(indices_ptr),
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .launch_args = LaunchArgs(1, num_threads, smem_size)
    };
    const auto code = SM100PagedMQALogitsMetadataRuntime::generate(args);
    const auto runtime = compiler->build("sm100_paged_mqa_logits_metadata", code);
    SM100PagedMQALogitsMetadataRuntime::launch(runtime, args);
}

// Unified contiguous-KV runtime for FP4/FP8; FP8 reuses the unused `sf_q` descriptor slot
class SM100MQALogitsRuntime final: public LaunchRuntime<SM100MQALogitsRuntime> {
public:
    struct Args {
        bool is_fp4;
        int num_q_tokens;
        int num_kv_tokens;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int split_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_sf_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;
        at::ScalarType weights_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(128 % args.num_heads == 0);

        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mqa_logits<
        {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {}
    >);
}};
)", args.is_fp4 ? "true" : "false",
    args.num_heads, args.head_dim,
    args.is_compressed_logits,
    args.block_q, args.split_kv,
    args.num_q_stages, args.num_kv_stages,
    args.launch_args.grid_dim.first,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype),
    args.weights_dtype == torch::kBFloat16 ? "__nv_bfloat16" : "float");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.num_q_tokens, args.num_kv_tokens,
            args.stride_logits,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm100_mqa_logits(const bool& is_fp4,
                             const torch::Tensor& q, const std::optional<torch::Tensor>& sf_q,
                             const torch::Tensor& kv, const torch::Tensor& sf_kv,
                             const torch::Tensor& weights,
                             const torch::Tensor& cu_seq_len_k_start,
                             const torch::Tensor& cu_seq_len_k_end,
                             const torch::Tensor& logits,
                             const at::ScalarType& logits_dtype,
                             const int& num_q_tokens, const int& num_kv_tokens,
                             const int& max_seqlen_k, const int& stride_logits,
                             const int& num_heads, const int& head_dim,
                             const int& block_q, const int& split_kv) {
    constexpr int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    const int num_q_stages = 3;
    // Use the deepest KV pipeline that fits with headroom: FP4 10 stages, FP8 5
    const int num_kv_stages = is_fp4 ? 10 : 5;

    const bool is_compressed_logits = (max_seqlen_k > 0);

    // FP4 consumes `sf_q`; FP8 fills that descriptor slot with KV scales
    CUtensorMap tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv;
    if (is_fp4) {
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
        tensor_map_q = make_tma_2d_desc(q, head_dim, num_q_tokens * num_heads,
                                        head_dim, block_q * num_heads,
                                        static_cast<int>(q.stride(1)),
                                        head_dim / 2, 0, false, false);
        tensor_map_sf_q = make_tma_2d_desc(sf_q.value(), num_heads, num_q_tokens,
                                           num_heads, block_q,
                                           static_cast<int>(sf_q.value().stride(0)), 0);
        tensor_map_kv = make_tma_2d_desc(kv, head_dim, num_kv_tokens,
                                         head_dim, split_kv,
                                         static_cast<int>(kv.stride(0)),
                                         head_dim / 2, 0, false, false);
        tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                            get_tma_aligned_size(num_kv_tokens, static_cast<int>(sf_kv.element_size())), 1,
                                            split_kv, 1, 0, 0);
    } else {
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        tensor_map_q = make_tma_2d_desc(q, head_dim, num_q_tokens * num_heads,
                                        head_dim, block_q * num_heads, head_dim, head_dim);
        tensor_map_kv = make_tma_2d_desc(kv, head_dim, num_kv_tokens,
                                         head_dim, split_kv, head_dim, head_dim);
        tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                            get_tma_aligned_size(num_kv_tokens, static_cast<int>(sf_kv.element_size())),
                                            1, split_kv, 1, 0, 0);
        tensor_map_sf_q = tensor_map_sf_kv;  // unused by FP8
    }
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, num_q_tokens,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);

    // BLOCK_Q = 128 / num_heads keeps UMMA_N = 128 for all supported head counts
    const auto get_fp4_smem_size = [&](auto h, auto d) {
        constexpr int H = decltype(h)::value, D = decltype(d)::value;
        return static_cast<int>(sizeof(layout::MQALogitsSharedStorage<true, H, D, 128 / H, 256, 3, 10, 3>));
    };
    const auto get_fp8_smem_size = [&](auto h, auto d) {
        constexpr int H = decltype(h)::value, D = decltype(d)::value;
        return static_cast<int>(sizeof(layout::MQALogitsSharedStorage<false, H, D, 128 / H, 256, 3, 5, 3>));
    };
    const auto dispatch_heads = [&](auto&& fn) {
        switch (num_heads) {
            case 8:  return fn(cute::C<8>{});
            case 16: return fn(cute::C<16>{});
            case 32: return fn(cute::C<32>{});
            case 64: return fn(cute::C<64>{});
            default: DG_HOST_UNREACHABLE("Unsupported num_heads for MQA logits");
        }
    };

    int smem_size;
    if (is_fp4) {
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
        smem_size = dispatch_heads([&](auto h) {
            return head_dim == 64 ? get_fp4_smem_size(h, cute::C<64>{}) : get_fp4_smem_size(h, cute::C<128>{});
        });
    } else {
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        smem_size = dispatch_heads([&](auto h) {
            switch (head_dim) {
                case 32:  return get_fp8_smem_size(h, cute::C<32>{});
                case 64:  return get_fp8_smem_size(h, cute::C<64>{});
                case 128: return get_fp8_smem_size(h, cute::C<128>{});
                default:  DG_HOST_UNREACHABLE("Unsupported head_dim for FP8 MQA logits");
            }
        });
    }
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    const SM100MQALogitsRuntime::Args args = {
        .is_fp4 = is_fp4,
        .num_q_tokens = num_q_tokens,
        .num_kv_tokens = num_kv_tokens,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .split_kv = split_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .weights_dtype = weights.scalar_type(),
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100MQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_mqa_logits", code);
    SM100MQALogitsRuntime::launch(runtime, args);
}

// Paged variant: separate host/runtime path, shared device core

// Unified paged runtime for FP4/FP8; FP8 reuses the unused `sf_q` descriptor slot
class SM100PagedMQALogitsRuntime final: public LaunchRuntime<SM100PagedMQALogitsRuntime> {
public:
    struct Args {
        bool is_fp4;
        int num_q_tokens_total;
        int tokens_per_request;
        int num_heads;
        int head_dim;
        int page_kv;
        bool is_context_lens_2d;
        bool is_varlen;
        int block_table_stride;
        int logits_stride;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;
        int splits_per_chunk;

        int* context_lens;
        void* logits;
        int* block_table;
        int* indices;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_sf_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_sf_kv;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;
        at::ScalarType weights_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {}
    >);
}};
)", args.is_fp4 ? "true" : "false",
    args.tokens_per_request, args.num_heads,
    args.head_dim, args.page_kv,
    args.is_context_lens_2d, args.is_varlen ? "true" : "false",
    args.num_q_stages, args.num_kv_stages,
    args.split_kv, args.splits_per_chunk,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype),
    args.weights_dtype == torch::kBFloat16 ? "__nv_bfloat16" : "float");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.num_q_tokens_total,
            args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits,
            args.block_table, args.indices, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm100_paged_mqa_logits(const bool& is_fp4,
                                   const torch::Tensor& q,
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
                                   const int& splits_per_chunk) {
    const int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    DG_HOST_ASSERT(split_kv == 256 and logits_stride % split_kv == 0);

    const int num_q_stages = 3;
    // Match contiguous-KV pipeline depth: FP4 10 stages, FP8 5
    const int num_kv_stages = is_fp4 ? 10 : 5;
    // BLOCK_Q = 128 / num_heads; a Q-block holds up to BLOCK_Q request tokens
    DG_HOST_ASSERT(128 % num_heads == 0);
    const int block_q = 128 / num_heads;

    // FP4 consumes `sf_q`; FP8 fills that descriptor slot with KV scales
    CUtensorMap tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv;
    if (is_fp4) {
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
        tensor_map_q = make_tma_2d_desc(q, head_dim, num_requests * tokens_per_request * num_heads,
                                        head_dim, block_q * num_heads,
                                        static_cast<int>(q.stride(2)),
                                        head_dim / 2, 0, false, false);
        tensor_map_sf_q = make_tma_2d_desc(sf_q.value(), num_heads, num_requests * tokens_per_request,
                                           num_heads, block_q,
                                           static_cast<int>(sf_q.value().stride(1)), 0);
        tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, page_kv, num_kv_blocks,
                                         head_dim, page_kv, 1,
                                         static_cast<int>(kv_cache.stride(1)),
                                         static_cast<int>(kv_cache.stride(0)),
                                         head_dim / 2, 0, false, false);
        tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, page_kv, num_kv_blocks,
                                            page_kv, 1,
                                            static_cast<int>(kv_cache_sf.stride(0)), 0);
    } else {
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        tensor_map_q = make_tma_2d_desc(q, head_dim, num_requests * tokens_per_request * num_heads,
                                        head_dim, block_q * num_heads,
                                        static_cast<int>(q.stride(2)),
                                        head_dim);
        tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, page_kv, num_kv_blocks,
                                         head_dim, page_kv, 1,
                                         static_cast<int>(kv_cache.stride(1)),
                                         static_cast<int>(kv_cache.stride(0)),
                                         head_dim);
        tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, page_kv, num_kv_blocks,
                                            page_kv, 1,
                                            static_cast<int>(kv_cache_sf.stride(0)), 0);
        tensor_map_sf_q = tensor_map_sf_kv;  // unused by FP8
    }
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, num_requests * tokens_per_request,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);

    int smem_size;
    // Shared storage is sized per (H, D) via compile-time dispatch
    const auto dispatch_heads = [&](auto&& fn) {
        switch (num_heads) {
            case 8:  return fn(cute::C<8>{});
            case 16: return fn(cute::C<16>{});
            case 32: return fn(cute::C<32>{});
            case 64: return fn(cute::C<64>{});
            default: DG_HOST_UNREACHABLE("Unsupported num_heads for paged MQA logits");
        }
    };

    if (is_fp4) {
        DG_HOST_ASSERT(head_dim == 64 or head_dim == 128);
        smem_size = dispatch_heads([&](auto h) {
            constexpr int H = decltype(h)::value;
            const auto get_smem_size = [&](auto d) {
                constexpr int D = decltype(d)::value;
                return static_cast<int>(sizeof(layout::MQALogitsSharedStorage<true, H, D, 128 / H, 256, 3, 10, 3>));
            };
            return head_dim == 64 ? get_smem_size(cute::C<64>{}) : get_smem_size(cute::C<128>{});
        });
    } else {
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        smem_size = dispatch_heads([&](auto h) {
            constexpr int H = decltype(h)::value;
            switch (head_dim) {
                case 32:  return static_cast<int>(sizeof(layout::MQALogitsSharedStorage<false, H, 32, 128 / H, 256, 3, 5, 3>));
                case 64:  return static_cast<int>(sizeof(layout::MQALogitsSharedStorage<false, H, 64, 128 / H, 256, 3, 5, 3>));
                case 128: return static_cast<int>(sizeof(layout::MQALogitsSharedStorage<false, H, 128, 128 / H, 256, 3, 5, 3>));
                default:  DG_HOST_UNREACHABLE("Unsupported head_dim for FP8 paged MQA logits");
            }
        });
    }
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    const SM100PagedMQALogitsRuntime::Args args = {
        .is_fp4 = is_fp4,
        .num_q_tokens_total = num_q_tokens_total,
        .tokens_per_request = tokens_per_request,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .page_kv = page_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .is_varlen = is_varlen,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
        .splits_per_chunk = splits_per_chunk,
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .indices = is_varlen ? indices.data_ptr<int>() : nullptr,
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .weights_dtype = weights.scalar_type(),
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_paged_mqa_logits", code);
    SM100PagedMQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
