#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SMXXFP8MQALogitsRuntime final: public LaunchRuntime<SMXXFP8MQALogitsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;

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
#include <deep_gemm/impls/sm{}_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_mqa_logits<
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", arch, arch,
    args.num_heads, args.head_dim,
    args.is_compressed_logits,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.launch_args.grid_dim.first,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv,
            args.max_seqlen_k, args.stride_logits,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

static void smxx_fp8_mqa_logits(const torch::Tensor& q,
                                const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                const torch::Tensor& weights,
                                const torch::Tensor& cu_seq_len_k_start,
                                const torch::Tensor& cu_seq_len_k_end,
                                const torch::Tensor& logits,
                                const at::ScalarType& logits_dtype,
                                const int& seq_len, const int& seq_len_kv,
                                const int& max_seqlen_k, const int& stride_logits,
                                const int& num_heads, const int& head_dim,
                                const int& block_q, const int& block_kv) {
    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 3, num_kv_stages = 3;
    const int num_math_threads = (device_runtime->get_arch_major() == 10 ? 256 : 512);

    // Use compressed logits format when max_seqlen_k is specified
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // Construct TMAs
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv, head_dim, head_dim);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                       get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                       1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q, num_heads, 0);

    // Calculate shared memory size
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
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SMXXFP8MQALogitsRuntime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SMXXFP8MQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("smxx_fp8_mqa_logits", code);
    SMXXFP8MQALogitsRuntime::launch(runtime, args);
}

class SM100FP8MQALogits2CTARuntime final: public LaunchRuntime<SM100FP8MQALogits2CTARuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        uint32_t* cu_seq_len_k_start_and_end;
        float* logits;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(128 % args.num_heads == 0);

        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_mqa_logits_2cta.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_mqa_logits_2cta<
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {}
    >);
}};
)",
        args.num_heads, args.head_dim,
        args.is_compressed_logits,
        args.block_q, args.block_kv,
        args.num_q_stages, args.num_kv_stages,
        args.num_specialized_threads, args.num_math_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv,
            args.max_seqlen_k, static_cast<uint64_t>(args.stride_logits),
            args.cu_seq_len_k_start_and_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

static void sm100_fp8_mqa_logits_2cta(const torch::Tensor& q,
                                      const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                      const torch::Tensor& weights,
                                      const torch::Tensor& cu_seq_len_k_start_and_end,
                                      const torch::Tensor& logits,
                                      const int& seq_len, const int& seq_len_kv,
                                      const int& max_seqlen_k, const int& stride_logits,
                                      const int& num_heads, const int& head_dim,
                                      const int& block_q, const int& block_kv) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);

    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 3, num_kv_stages = 3;
    constexpr int num_math_threads = 256;
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // Q: [seq_len * num_heads, head_dim], box [block_q * num_heads, head_dim]
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads, head_dim, head_dim);
    // KV: [seq_len_kv, head_dim], box [block_kv/2, head_dim] per CTA
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv / 2, head_dim, head_dim);
    // KV scales: [seq_len_kv], box [block_kv/2] per CTA
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                       get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                       1, block_kv / 2, 1, 0, 0);
    // Weights: [seq_len, num_heads], box [block_q, num_heads] per CTA
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q, num_heads, 0);

    // Shared memory layout (per CTA):
    //   smem_q[num_q_stages]      + smem_weights[num_q_stages]     (BLOCK_Q_2CTA weights)
    //   smem_kv[num_kv_stages]    + smem_kv_scales[num_kv_stages]  (BLOCK_KV/2 per CTA)
    //   smem_kv_offsets[num_q_stages]  + barriers + tmem_ptr
    const int block_q_2cta = block_q * 2;
    const int smem_q_per_stage     = block_q * num_heads * head_dim;         // FP8 = 1 byte
    const int smem_weight_per_stage = block_q_2cta * num_heads * 2;          // half = 2 bytes
    const int smem_kv_per_stage     = (block_kv / 2) * head_dim;             // FP8 = 1 byte
    const int smem_kv_scale_raw     = (block_kv / 2) * static_cast<int>(kv_scales.element_size());
    const int smem_kv_scale_per_stage = (smem_kv_scale_raw + 511) / 512 * 512; // align to 512
    const int smem_kv_offset_per_stage = block_q_2cta * 8;                   // uint2 = 8 bytes
    const int num_umma_stages       = 512 / (block_q_2cta * num_heads);
    const int num_barriers          = num_q_stages * 2 + num_kv_stages * 2 + num_umma_stages * 2;

    int smem_size = 0;
    smem_size += num_q_stages * smem_q_per_stage;
    smem_size += num_q_stages * smem_weight_per_stage;
    smem_size += num_kv_stages * smem_kv_per_stage;
    smem_size += num_kv_stages * smem_kv_scale_per_stage;
    smem_size += num_q_stages * smem_kv_offset_per_stage;
    smem_size += num_barriers * 8;  // ClusterTransactionBarrier = 8 bytes
    smem_size += 4;                 // tmem_ptr_in_smem
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    const SM100FP8MQALogits2CTARuntime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start_and_end = reinterpret_cast<uint32_t*>(cu_seq_len_k_start_and_end.data_ptr<int>()),
        .logits = logits.data_ptr<float>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size, 2 /* cluster_dim */)
    };
    const auto code = SM100FP8MQALogits2CTARuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_mqa_logits_2cta", code);
    SM100FP8MQALogits2CTARuntime::launch(runtime, args);
}

class SM100FP4MQALogitsRuntime final: public LaunchRuntime<SM100FP4MQALogitsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        void* logits;

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
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_mqa_logits<
        {}, {},
        {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", args.num_heads, args.head_dim,
    args.is_compressed_logits,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.launch_args.grid_dim.first,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv,
            args.max_seqlen_k, args.stride_logits,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_sf_q,
            args.tensor_map_kv, args.tensor_map_sf_kv,
            args.tensor_map_weights
        ));
    }
};

static void sm100_fp4_mqa_logits(const torch::Tensor& q, const torch::Tensor& sf_q,
                                 const torch::Tensor& kv, const torch::Tensor& sf_kv,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& cu_seq_len_k_start,
                                 const torch::Tensor& cu_seq_len_k_end,
                                 const torch::Tensor& logits,
                                 const at::ScalarType& logits_dtype,
                                 const int& seq_len, const int& seq_len_kv,
                                 const int& max_seqlen_k, const int& stride_logits,
                                 const int& num_heads, const int& head_dim,
                                 const int& block_q, const int& block_kv) {
    constexpr int num_specialized_threads = 128;
    const int num_math_threads = 2 * 128;
    constexpr int num_q_stages = 3, num_kv_stages = 6, num_tmem_stages = 3;

    // Use compressed logits format when max_seqlen_k is specified
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // Construct TMAs
    // `head_dim` must be 128 for 64B swizzling
    DG_HOST_ASSERT(head_dim == 128);
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads,
                                               static_cast<int>(q.stride(1)),
                                               head_dim / 2, 0, false, false);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, num_heads, seq_len,
                                                  num_heads, block_q,
                                                  static_cast<int>(sf_q.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv,
                                                static_cast<int>(kv.stride(0)),
                                                head_dim / 2, 0, false, false);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto tensor_map_sf_kv = make_tma_2d_desc(sf_kv,
                                                   get_tma_aligned_size(seq_len_kv, static_cast<int>(sf_kv.element_size())), 1,
                                                   block_kv, 1, 0, 0);

    // Calculate shared memory size
    const int smem_q_size_per_stage = block_q * num_heads * head_dim / 2;
    const int smem_sf_q_size_per_stage = align(block_q * num_heads, 128) * sizeof(int);
    const int smem_kv_size_per_stage = block_kv * head_dim / 2;
    const int smem_sf_kv_size_per_stage = align(block_kv, 128) * sizeof(int);
    const int smem_weight_size_per_stage = block_q * num_heads * sizeof(float);

    const int smem_barriers = (num_q_stages + num_kv_stages + num_tmem_stages) * 2 * 8;
    const int smem_tmem_ptr = 4;
    const int smem_size = num_q_stages * (smem_q_size_per_stage + smem_sf_q_size_per_stage + smem_weight_size_per_stage) + 
                          num_kv_stages * (smem_kv_size_per_stage + smem_sf_kv_size_per_stage) + 
                          smem_barriers + smem_tmem_ptr;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SM100FP4MQALogitsRuntime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_sf_q = tensor_map_sf_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_sf_kv = tensor_map_sf_kv,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto code = SM100FP4MQALogitsRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp4_mqa_logits", code);
    SM100FP4MQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
