#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SMXXPagedMQALogitsMetadataRuntime final: public LaunchRuntime<SMXXPagedMQALogitsMetadataRuntime> {
public:
    struct Args {
        int aligned_batch_size;
        int split_kv;
        int num_sms;
        bool is_varlen;

        int batch_size;
        int next_n;
        bool is_context_lens_2d;
        int* context_lens;
        int* indices;
        int* schedule_metadata;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sched::smxx_paged_mqa_logits_metadata<
        {}, {}, {}, {}
    >);
}};
)", args.aligned_batch_size, args.split_kv, args.num_sms, args.is_varlen ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.next_n,
            args.is_context_lens_2d,
            args.context_lens,
            args.indices,
            args.schedule_metadata
        ));
    }
};

static void smxx_paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                           const torch::Tensor& schedule_metadata,
                                           const int& batch_size, const int& next_n,
                                           const int& block_kv, const int& num_sms,
                                           const bool& is_context_lens_2d,
                                           const bool& is_varlen, const int* indices_ptr) {
    constexpr int split_kv = 256;
    constexpr int num_threads = 32;
    const int aligned_batch_size = align(batch_size, 32);
    DG_HOST_ASSERT(split_kv % block_kv == 0);

    // Shared memory: prefix_sum[kAlignedBatchSize] + varlen_atom_token_start/context_len[kAlignedBatchSize] + varlen_num_atoms
    const int smem_size = (3 * aligned_batch_size + 1) * static_cast<int>(sizeof(int));
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SMXXPagedMQALogitsMetadataRuntime::Args& args = {
        .aligned_batch_size = aligned_batch_size,
        .split_kv = split_kv,
        .num_sms = num_sms,
        .is_varlen = is_varlen,
        .batch_size = batch_size,
        .next_n = next_n,
        .is_context_lens_2d = is_context_lens_2d,
        .context_lens = context_lens.data_ptr<int>(),
        .indices = const_cast<int*>(indices_ptr),
        .schedule_metadata = schedule_metadata.data_ptr<int>(),
        .launch_args = LaunchArgs(1, num_threads, smem_size)
    };
    const auto code = SMXXPagedMQALogitsMetadataRuntime::generate(args);
    const auto runtime = compiler->build("smxx_paged_mqa_logits_metadata", code);
    SMXXPagedMQALogitsMetadataRuntime::launch(runtime, args);
}

class SMXXFP8PagedMQALogitsRuntime final: public LaunchRuntime<SMXXFP8PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size;
        int next_n;
        int num_heads;
        int head_dim;
        int block_kv;
        bool is_context_lens_2d;
        bool is_varlen;
        int block_table_stride;
        int logits_stride;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;

        int* context_lens;
        void* logits;
        int* block_table;
        int* indices;
        int* schedule_meta;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", arch, arch,
    args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d, args.is_varlen ? "true" : "false",
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits,
            args.block_table, args.indices, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

class SM120FP8PagedMQALogitsReferenceRuntime final: public LaunchRuntime<SM120FP8PagedMQALogitsReferenceRuntime> {
public:
    struct Args {
        int batch_size;
        int next_n;
        int num_heads;
        int head_dim;
        int block_kv;
        bool is_context_lens_2d;
        int block_table_stride;
        int logits_stride;
        int tokens_per_block;
        int token_groups;
        bool cache_q;

        int64_t q_batch_stride;
        int64_t q_next_stride;
        int64_t q_head_stride;
        int64_t kv_block_stride;
        int64_t kv_token_stride;
        int64_t kv_scale_block_stride;
        int64_t weights_row_stride;

        void* q;
        void* kv_cache;
        float* kv_cache_scales;
        float* weights;
        int* context_lens;
        void* logits;
        int* block_table;
        at::ScalarType logits_dtype;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_paged_mqa_logits_reference<
        {}, {}, {},
        {}, {},
        {}
    >);
}};
)", args.num_heads, args.head_dim, args.block_kv,
    args.is_context_lens_2d ? "true" : "false", args.tokens_per_block,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.next_n,
            args.logits_stride,
            args.block_table_stride,
            args.q_batch_stride,
            args.q_next_stride,
            args.q_head_stride,
            args.kv_block_stride,
            args.kv_token_stride,
            args.kv_scale_block_stride,
            args.weights_row_stride,
            args.q,
            args.kv_cache,
            args.kv_cache_scales,
            args.weights,
            args.context_lens,
            args.logits,
            args.block_table
        ));
    }
};

class SM120FP8PagedMQALogitsTiledRuntime final: public LaunchRuntime<SM120FP8PagedMQALogitsTiledRuntime> {
public:
    using Args = SM120FP8PagedMQALogitsReferenceRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_paged_mqa_logits_mma_tiled<
        {},
        {},
        {},
        {},
        {}
    >);
}};
)", args.block_kv, args.is_context_lens_2d ? "true" : "false",
    args.token_groups,
    args.cache_q ? "true" : "false",
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.next_n,
            args.logits_stride,
            args.block_table_stride,
            args.q_batch_stride,
            args.q_next_stride,
            args.q_head_stride,
            args.kv_block_stride,
            args.kv_token_stride,
            args.kv_scale_block_stride,
            args.weights_row_stride,
            args.q,
            args.kv_cache,
            args.kv_cache_scales,
            args.weights,
            args.context_lens,
            args.logits,
            args.block_table
        ));
    }
};

static void sm120_fp8_paged_mqa_logits(const torch::Tensor& q,
                                       const torch::Tensor& kv_cache,
                                       const torch::Tensor& kv_cache_scales,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& context_lens,
                                       const torch::Tensor& logits,
                                       const torch::Tensor& block_table,
                                       const at::ScalarType& logits_dtype,
                                       const int& batch_size, const int& next_n,
                                       const int& num_heads, const int& head_dim,
                                       const int& block_kv,
                                       const bool& is_context_lens_2d,
                                       const int& logits_stride,
                                       const int& block_table_stride) {
    constexpr int tokens_per_block = 128;
    DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);

    const SM120FP8PagedMQALogitsReferenceRuntime::Args args = {
        .batch_size = batch_size,
        .next_n = next_n,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .tokens_per_block = tokens_per_block,
        .token_groups = 1,
        .cache_q = false,
        .q_batch_stride = q.stride(0),
        .q_next_stride = q.stride(1),
        .q_head_stride = q.stride(2),
        .kv_block_stride = kv_cache.stride(0),
        .kv_token_stride = kv_cache.stride(1),
        .kv_scale_block_stride = kv_cache_scales.stride(0),
        .weights_row_stride = weights.stride(0),
        .q = q.data_ptr(),
        .kv_cache = kv_cache.data_ptr(),
        .kv_cache_scales = kv_cache_scales.data_ptr<float>(),
        .weights = weights.data_ptr<float>(),
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .logits_dtype = logits_dtype,
        .launch_args = LaunchArgs(std::make_pair(batch_size * next_n, ceil_div(logits_stride, tokens_per_block)),
                                  tokens_per_block)
    };
    const bool use_tiled = get_env<int>("DG_SM120_PAGED_MQA_TILED", 0) != 0;
    if (use_tiled and num_heads == 64 and head_dim == 128 and block_kv == 64 and logits_dtype == torch::kFloat32) {
        const int token_groups = get_env<int>("DG_SM120_PAGED_MQA_TILED_GROUPS", 8);
        DG_HOST_ASSERT(token_groups == 1 or token_groups == 2 or token_groups == 4 or token_groups == 8);
        constexpr int tiled_token_tile = 8;
        constexpr int tiled_num_threads = 128;
        const int tokens_per_tiled_cta = tiled_token_tile * token_groups;
        const int tiled_smem_size = 64 * tokens_per_tiled_cta * sizeof(float);
        auto tiled_args = args;
        tiled_args.token_groups = token_groups;
        const bool should_cache_q_by_default = token_groups != 1;
        tiled_args.cache_q = get_env<int>(
            "DG_SM120_PAGED_MQA_TILED_CACHE_Q",
            should_cache_q_by_default ? 1 : 0) != 0;
        DG_HOST_ASSERT(not tiled_args.cache_q or token_groups == 2 or token_groups == 4 or token_groups == 8);
        tiled_args.launch_args = LaunchArgs(
            std::make_pair(batch_size * next_n, ceil_div(logits_stride, tokens_per_tiled_cta)),
            tiled_num_threads,
            tiled_smem_size);
        const auto code = SM120FP8PagedMQALogitsTiledRuntime::generate(tiled_args);
        const auto runtime = compiler->build("sm120_fp8_paged_mqa_logits_mma_tiled", code);
        SM120FP8PagedMQALogitsTiledRuntime::launch(runtime, tiled_args);
        return;
    }

    const auto code = SM120FP8PagedMQALogitsReferenceRuntime::generate(args);
    const auto runtime = compiler->build("sm120_fp8_paged_mqa_logits_reference", code);
    SM120FP8PagedMQALogitsReferenceRuntime::launch(runtime, args);
}

static void smxx_fp8_paged_mqa_logits(const torch::Tensor& q,
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
    const int num_specialized_threads = 128;
    const int mma_m = (device_runtime->get_arch_major() == 10 ? 128 : 64);
    const int num_math_warp_groups = split_kv / mma_m;
    const int num_math_threads = num_math_warp_groups * 128;
    const int num_q_stages = 3, num_kv_stages = (device_runtime->get_arch_major() == 10 ? 4 : 3);
    DG_HOST_ASSERT(split_kv % mma_m == 0 and logits_stride % split_kv == 0);

    // Construct TMAs
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

    // Calculate shared memory size
    int smem_size = 0;
    if (device_runtime->get_arch_major() == 9) {
        const int swizzle_alignment = head_dim * 8;

        const int smem_q_size_per_stage = next_n * num_heads * head_dim * static_cast<int>(q.element_size());
        const int aligned_smem_weight_size_per_stage = align(next_n * num_heads * static_cast<int>(weights.element_size()), swizzle_alignment);
        const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) + align(num_q_stages * 8 * 2, swizzle_alignment);

        const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int aligned_smem_kv_scale_size_per_stage = align(block_kv * static_cast<int>(kv_cache_scales.element_size()), swizzle_alignment);
        const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) + align(num_kv_stages * 8 * 2, swizzle_alignment);

        // Allocate some shared memory for UMMA barriers and tensor memory pointer, although it is not used in SM90
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
        DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    } else {
        const int smem_q_size_per_stage = next_n_atom * num_heads * head_dim * static_cast<int>(q.element_size());
        const int smem_kv_size_per_stage = split_kv * head_dim * static_cast<int>(kv_cache.element_size());
        const int smem_kv_scale_size_per_stage = split_kv * static_cast<int>(kv_cache_scales.element_size());
        const int smem_weight_size_per_stage = next_n_atom * num_heads * static_cast<int>(weights.element_size());

        const int smem_barriers = (num_q_stages + num_kv_stages) * 2 * 8;
        const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
        const int smem_tmem_ptr = 4;

        smem_size = num_q_stages * (smem_q_size_per_stage + smem_weight_size_per_stage) + 
                    num_kv_stages * (smem_kv_size_per_stage + smem_kv_scale_size_per_stage) + 
                    smem_barriers + smem_umma_barriers + smem_tmem_ptr;
        DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);
    }

    // Launch
    const SMXXFP8PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size,
        .next_n = next_n,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .is_varlen = is_varlen,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
        .context_lens = context_lens.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .block_table = block_table.data_ptr<int>(),
        .indices = is_varlen ? indices.data_ptr<int>() : nullptr,
        .schedule_meta = schedule_meta.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SMXXFP8PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("smxx_fp8_paged_mqa_logits", code);
    SMXXFP8PagedMQALogitsRuntime::launch(runtime, args);
}

class SM100FP4PagedMQALogitsRuntime final: public LaunchRuntime<SM100FP4PagedMQALogitsRuntime> {
public:
    struct Args {
        int batch_size;
        int next_n;
        int num_heads;
        int head_dim;
        int block_kv;
        bool is_context_lens_2d;
        bool is_varlen;
        int block_table_stride;
        int logits_stride;

        int num_q_stages;
        int num_kv_stages;
        int split_kv;

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

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", args.next_n, args.num_heads,
    args.head_dim, args.block_kv,
    args.is_context_lens_2d, args.is_varlen ? "true" : "false",
    args.num_q_stages, args.num_kv_stages,
    args.split_kv,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.batch_size,
            args.logits_stride, args.block_table_stride,
            args.context_lens, args.logits,
            args.block_table, args.indices, args.schedule_meta,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm100_fp4_paged_mqa_logits(const torch::Tensor& q,
                                       const torch::Tensor& sf_q,
                                       const torch::Tensor& kv_cache,
                                       const torch::Tensor& kv_cache_sf,
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
    const int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    DG_HOST_ASSERT(split_kv == 256 and logits_stride % split_kv == 0);

    // TODO: tuning num_stages
    const int num_q_stages = 3, num_kv_stages = 10, num_tmem_stages = 3;
    const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;

    // `head_dim` must be 128 for 64B swizzling
    DG_HOST_ASSERT(head_dim == 128);

    // Using 2D TMA as tensor q is asserted contiguous
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_atom * num_heads,
                                               static_cast<int>(q.stride(2)),
                                               head_dim / 2, 0, false, false);
    // NOTES: `sf_q` is a 3D tensor, while `weights` is a 2D tensor
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, num_heads, batch_size * next_n,
                                                  num_heads, next_n_atom,
                                                  static_cast<int>(sf_q.stride(1)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_atom,
                                                     static_cast<int>(weights.stride(0)), 0);

    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                head_dim / 2, 0, false, false);
    const auto tensor_map_sf_kv = make_tma_2d_desc(kv_cache_sf, block_kv, num_kv_blocks,
                                                   block_kv, 1,
                                                   static_cast<int>(kv_cache_sf.stride(0)), 0);

    // Calculate shared memory size
    const int smem_q_size_per_stage = next_n_atom * num_heads * head_dim / 2;
    const int smem_sf_q_size_per_stage = align(next_n_atom * num_heads, 128) * sizeof(int);
    const int smem_kv_size_per_stage = split_kv * head_dim / 2;
    const int smem_sf_kv_size_per_stage = align(split_kv, 128) * sizeof(int);
    const int smem_weight_size_per_stage = next_n_atom * num_heads * sizeof(float);

    const int smem_barriers = (num_q_stages + num_kv_stages + num_tmem_stages) * 2 * 8;
    const int smem_tmem_ptr = 4;
    const int smem_size = num_q_stages * (smem_q_size_per_stage + smem_sf_q_size_per_stage + smem_weight_size_per_stage) + 
                          num_kv_stages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage) + 
                          smem_barriers + smem_tmem_ptr;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SM100FP4PagedMQALogitsRuntime::Args args = {
        .batch_size = batch_size,
        .next_n = next_n,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .block_kv = block_kv,
        .is_context_lens_2d = is_context_lens_2d,
        .is_varlen = is_varlen,
        .block_table_stride = block_table_stride,
        .logits_stride = logits_stride,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .split_kv = split_kv,
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
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(num_sms,
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100FP4PagedMQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp4_paged_mqa_logits", code);
    SM100FP4PagedMQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
