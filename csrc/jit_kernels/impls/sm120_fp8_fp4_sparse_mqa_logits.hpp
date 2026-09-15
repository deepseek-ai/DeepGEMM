#pragma once

#include <format>
#include <limits>
#include <torch/python.h>

#include <deep_gemm/layout/sparse_mqa_logits.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

namespace sm120_sparse_mqa_runtime {
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuFuncGetAttribute);
}

static void launch_sm120_fp8_fp4_sparse_mqa_logits(
    const torch::Tensor& q, const torch::Tensor& sf_q,
    const torch::Tensor& kv, const torch::Tensor& sf_kv,
    const torch::Tensor& weights, const torch::Tensor& metadata,
    const torch::Tensor& logits, const bool is_paged,
    const bool use_unaligned_ks, const int sparse_block_kv) {
    using namespace layout::sparse_mqa_logits;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    DG_HOST_ASSERT(not is_paged or not use_unaligned_ks);
    const bool is_fp4 = q.scalar_type() == kPackedFP4;
    DG_HOST_ASSERT(is_fp4 or q.scalar_type() == torch::kFloat8_e4m3fn);
    const int row_bytes = is_fp4 ? 64 : 128;
    DG_HOST_ASSERT(q.is_cuda() and q.is_contiguous() and q.dim() == (is_paged ? 4 : 3));
    DG_HOST_ASSERT(q.size(-2) == kNumHeads and q.size(-1) == row_bytes);
    DG_HOST_ASSERT(not is_paged or q.size(1) == 1);
    const int64_t num_q_tokens = q.size(0);
    DG_HOST_ASSERT(num_q_tokens > 0 and num_q_tokens <= std::numeric_limits<uint32_t>::max());
    const auto check_device = [&](const torch::Tensor& tensor) {
        DG_HOST_ASSERT(tensor.is_cuda() and tensor.device() == q.device());
    };
    check_device(sf_q);
    check_device(kv);
    check_device(weights);
    check_device(metadata);
    check_device(logits);
    DG_HOST_ASSERT(sf_q.scalar_type() == torch::kInt32 and sf_q.is_contiguous());
    DG_HOST_ASSERT(sf_q.dim() == (is_paged ? 3 : 2) and sf_q.size(0) == num_q_tokens and sf_q.size(-1) == kNumHeads);
    DG_HOST_ASSERT(not is_paged or sf_q.size(1) == 1);
    DG_HOST_ASSERT(weights.dim() == 2 and weights.size(0) == num_q_tokens and weights.size(1) == kNumHeads);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16 and weights.stride(1) == 1);
    DG_HOST_ASSERT(metadata.dim() == 1 and metadata.scalar_type() == torch::kUInt8 and metadata.is_contiguous());
    DG_HOST_ASSERT(metadata.numel() >= static_cast<int64_t>(sizeof(MetadataHeader)));
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(metadata.data_ptr()) % alignof(MetadataHeader) == 0);
    DG_HOST_ASSERT(logits.dim() == 2 and logits.size(0) == num_q_tokens and logits.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(logits.stride(1) == 1 and logits.stride(0) >= logits.size(1) and logits.stride(0) % 512 == 0);
    DG_HOST_ASSERT(logits.size(1) % sparse_block_kv == 0);
    const int64_t num_max_sparse_blocks = logits.size(1) / sparse_block_kv;
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);

    if (is_paged) {
        DG_HOST_ASSERT(kv.dim() == 4 and kv.scalar_type() == torch::kUInt8);
        DG_HOST_ASSERT(kv.size(0) > 0 and kv.size(1) > 0 and kv.size(1) <= std::numeric_limits<uint32_t>::max());
        DG_HOST_ASSERT(kv.size(1) % sparse_block_kv == 0 and kv.size(2) == 1 and kv.size(3) == row_bytes + 4);
        DG_HOST_ASSERT(kv.stride(1) == row_bytes + 4 and kv.stride(3) == 1);
        DG_HOST_ASSERT(kv.stride(0) % 512 == 0);
    } else {
        check_device(sf_kv);
        DG_HOST_ASSERT(kv.dim() == 2 and kv.size(0) > 0 and kv.size(1) == row_bytes);
        DG_HOST_ASSERT(kv.is_contiguous() and kv.scalar_type() == q.scalar_type());
        DG_HOST_ASSERT(sf_kv.dim() == 1 and sf_kv.size(0) == kv.size(0));
        DG_HOST_ASSERT(sf_kv.scalar_type() == torch::kInt32 and sf_kv.is_contiguous());
    }

    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms > 0);
    int work_partitions = is_paged and num_q_tokens <= 16 ? 8 : 1;
    const bool pipeline = not is_paged and reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0 and
        reinterpret_cast<uintptr_t>(sf_kv.data_ptr()) % sizeof(uint32_t) == 0;
    if (pipeline and is_fp4 and sparse_block_kv == 8 and num_max_sparse_blocks == 2048)
        work_partitions = 4;
    const bool warp_specialized = pipeline and is_fp4 and sparse_block_kv == 8 and num_max_sparse_blocks == 2048;
    const bool cache_q = warp_specialized;
    const bool entry_balance = warp_specialized and num_q_tokens >= 64;
    const int num_kv_stages = pipeline ? 2 : 1;
    const int tile_bytes = align<int>(64 * (1 + num_kv_stages) * row_bytes +
        64 * (1 + num_kv_stages) * sizeof(uint32_t) + 64 * sizeof(nv_bfloat16) +
        num_kv_stages * (64 / sparse_block_kv) * sizeof(KVBlockInfo), 1024);
    const int smem_bytes = warp_specialized ?
        align<int>(tile_bytes + 6 * sizeof(cutlass::arch::ClusterBarrier), 1024) : tile_bytes;
    DG_HOST_ASSERT(smem_bytes <= jit->device.get_num_smem_bytes());

    const auto kernel = jit->compile("sm120_fp8_fp4_sparse_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_sparse_mqa_logits.cuh>

using namespace deep_gemm;

static_assert(sizeof(sm120_sparse_mqa_detail::{}<{}, {}, {}>) == {});

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_sparse_mqa_logits_kernel<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}};
)", warp_specialized ? "WarpSpecializedSharedStorage" : "SharedStorage",
        is_fp4, sparse_block_kv, pipeline, smem_bytes,
        is_fp4, is_paged, sparse_block_kv, is_paged ? kv.size(1) : 0, num_sms, use_unaligned_ks,
        work_partitions, pipeline, warp_specialized, cache_q, entry_balance));

    if (warp_specialized) {
        int registers = 0;
        DJ_CUDA_DRIVER_CHECK(sm120_sparse_mqa_runtime::lazy_cuFuncGetAttribute(
            &registers, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel->kernel_handle));
        DG_HOST_ASSERT(registers == 64 and "Sparse WG producer40/math88 requires compiled entry64");
    }
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_bytes,
            .grid_dim = dim3(num_sms * work_partitions, 1, 1),
            .block_dim = dim3(warp_specialized ? 256 : 128, 1, 1),
        },
        q.data_ptr(), sf_q.data_ptr(), kv.data_ptr(), is_paged ? nullptr : sf_kv.data_ptr(),
        weights.data_ptr(), metadata.data_ptr(), logits.data_ptr(),
        static_cast<uint64_t>(num_q_tokens), static_cast<uint64_t>(kv.size(0)),
        is_paged ? static_cast<uint64_t>(kv.stride(0)) : uint64_t{0},
        static_cast<uint64_t>(metadata.numel()), static_cast<uint64_t>(weights.stride(0)),
        static_cast<uint64_t>(logits.stride(0)), static_cast<uint32_t>(logits.size(1))
    );
}

} // namespace deep_gemm
