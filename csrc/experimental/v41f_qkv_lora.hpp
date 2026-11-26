#pragma once

#include <algorithm>
#include <format>
#include <limits>
#include <unordered_map>
#include <vector>

#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/python.h>

#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/math.hpp"
#include "../jit_kernels/impls/runtime_utils.hpp"

namespace deep_gemm::experimental {

static const torch::Tensor& get_v41f_qkv_barriers(const torch::TensorOptions& options, const int& num_m_blocks) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> barriers_by_stream;
    static std::vector<torch::Tensor> retired_barriers;
    auto& barriers = barriers_by_stream[stream];
    if (not barriers.defined() or barriers.numel() < 32 + 96ll * num_m_blocks) {
        // Initialize before capture; replay must not reset state shared with eager calls.
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        if (barriers.defined())
            retired_barriers.push_back(barriers);
        barriers = torch::zeros({32 + 96ll * std::max(64, num_m_blocks)}, options.dtype(torch::kInt));
    }
    return barriers;
}

// V41F Q-A/KV projection -> RMSNorm -> Q-B (with optional indexer Q) projection and KV RoPE/cache store.
// FP8 pairs use per-32 packed UE8M0 scales. Main Q-B weights are already permuted.
// Positions must index cos_sin; valid slots must be unique. Invalid slots are skipped.
// Warm up each M/row-warp bucket and Q-B output dim on its stream before CUDA Graph capture.
static void v41f_qkv_lora(
    const std::pair<torch::Tensor, torch::Tensor>& x,
    const std::pair<torch::Tensor, torch::Tensor>& qkv_weight,
    const std::pair<torch::Tensor, torch::Tensor>& q_b_weight,
    const torch::Tensor& q_norm_weight, const torch::Tensor& kv_norm_weight,
    const torch::Tensor& positions, const torch::Tensor& cos_sin, const torch::Tensor& slots,
    const torch::Tensor& q, const torch::Tensor& cache, const float& eps) {
    constexpr int kHidden = 5120, kQRank = 1280, kKVRank = 512, kNumHeads = 64;
    constexpr int kQDim = kNumHeads * kKVRank, kIndexQDim = 4096;
    constexpr int BLOCK_M = 128, BLOCK_K = 128, kNumMulticast = 2;
    constexpr int kRopeDim = 64, kNumCacheTokenBytes = kKVRank + kKVRank / 32;
    constexpr int kNumSmemBytes = 227840;

    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(x.first.dim() == 2 and x.first.size(0) >= 1 and x.first.size(0) <= std::numeric_limits<int>::max() - BLOCK_M);
    DG_HOST_ASSERT(q_b_weight.first.dim() == 2);
    const int m = static_cast<int>(x.first.size(0));
    const int q_out_dim = static_cast<int>(q_b_weight.first.size(0));
    DG_HOST_ASSERT(q_out_dim == kQDim or q_out_dim == kQDim + kIndexQDim);
    const int q_block_n = q_out_dim == kQDim ? 224 : 256;
    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms == jit->device.get_num_sms() and num_sms % 2 == 0 and "V41F requires all SMs");
    DG_HOST_ASSERT(kNumSmemBytes + 1024 <= jit->device.get_num_smem_bytes());
    const int num_row_warps = std::min(4, ceil_div(m, num_sms));
    const int num_warps = std::max(8, num_row_warps * 6);
    DG_HOST_ASSERT(x.first.is_cuda() and x.first.get_device() == at::cuda::current_device());
    for (const auto& t: {x.first, x.second, qkv_weight.first, qkv_weight.second, q_b_weight.first,
                         q_b_weight.second, q_norm_weight, kv_norm_weight, positions, cos_sin, slots, q, cache}) {
        DG_HOST_ASSERT(t.is_cuda() and t.device() == x.first.device());
    }
    for (const auto& t: {x.first, qkv_weight.first, q_b_weight.first}) {
        DG_HOST_ASSERT(t.scalar_type() == torch::kFloat8_e4m3fn and t.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
    }
    for (const auto& t: {q_norm_weight, kv_norm_weight, q}) {
        DG_HOST_ASSERT(t.scalar_type() == torch::kBFloat16 and t.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
    }
    for (const auto& sf: {x.second, qkv_weight.second, q_b_weight.second}) {
        DG_HOST_ASSERT(sf.dim() == 2 and sf.scalar_type() == torch::kInt);
        DG_HOST_ASSERT((sf.stride(0) == 1 or sf.size(0) == 1) and
                       sf.stride(1) >= align(static_cast<int>(sf.size(0)), 4) and
                       sf.stride(1) <= std::numeric_limits<int>::max() and sf.stride(1) % 4 == 0);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(sf.data_ptr()) % 16 == 0);
    }
    DG_HOST_ASSERT(x.first.sizes() == at::IntArrayRef({m, kHidden}));
    DG_HOST_ASSERT(x.second.sizes() == at::IntArrayRef({m, kHidden / 128}));
    DG_HOST_ASSERT(qkv_weight.first.sizes() == at::IntArrayRef({kQRank + kKVRank, kHidden}));
    DG_HOST_ASSERT(qkv_weight.second.sizes() == at::IntArrayRef({kQRank + kKVRank, kHidden / 128}));
    DG_HOST_ASSERT(q_b_weight.first.sizes() == at::IntArrayRef({q_out_dim, kQRank}));
    DG_HOST_ASSERT(q_b_weight.second.sizes() == at::IntArrayRef({q_out_dim, kQRank / 128}));
    DG_HOST_ASSERT(q_norm_weight.sizes() == at::IntArrayRef({kQRank}));
    DG_HOST_ASSERT(kv_norm_weight.sizes() == at::IntArrayRef({kKVRank}));
    DG_HOST_ASSERT(q.sizes() == at::IntArrayRef({m, q_out_dim}));
    for (const auto& t: {positions, slots})
        DG_HOST_ASSERT(t.scalar_type() == torch::kInt and t.is_contiguous() and t.sizes() == at::IntArrayRef({m}));
    DG_HOST_ASSERT(cos_sin.dim() == 2 and cos_sin.size(0) > 0 and cos_sin.size(1) == kRopeDim);
    DG_HOST_ASSERT(cos_sin.scalar_type() == torch::kFloat and cos_sin.is_contiguous());
    DG_HOST_ASSERT(cache.dim() == 4 and cache.size(0) > 0 and cache.size(1) > 0);
    DG_HOST_ASSERT(cache.size(2) == 1 and cache.size(3) == kNumCacheTokenBytes);
    DG_HOST_ASSERT(cache.scalar_type() == torch::kByte and cache.stride(1) == kNumCacheTokenBytes and cache.stride(3) == 1);
    DG_HOST_ASSERT(cache.stride(0) >= cache.size(1) * kNumCacheTokenBytes and cache.stride(0) % 4 == 0);
    DG_HOST_ASSERT(cache.stride(0) <= std::numeric_limits<uint32_t>::max() and
                   cache.size(0) * cache.size(1) <= std::numeric_limits<int>::max());
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(cache.data_ptr()) % 4 == 0);

    const int num_m_blocks = ceil_div(m, BLOCK_M);
    const int padded_m = num_m_blocks * BLOCK_M;
    int split_k = 1;
    if (not heuristics_runtime->get_deterministic_algorithms()) {
        const int num_a_tiles = num_m_blocks * ((kQRank + kKVRank) / 128);
        split_k = std::min(5, std::max(1, num_sms / num_a_tiles));
    }

    const auto& barriers = get_v41f_qkv_barriers(x.first.options(), num_m_blocks);
    const auto partials = torch::empty({static_cast<int64_t>(split_k) * padded_m, kQRank + kKVRank}, x.first.options().dtype(torch::kFloat));
    const auto q_a = torch::empty({m, kQRank}, x.first.options());
    const auto q_a_sf = torch::empty({kQRank / 128, align(m, 4)}, x.second.options()).transpose(0, 1).narrow(0, 0, m);

    const auto tensor_map_x = make_tma_a_desc(cute::UMMA::Major::K, x.first, m, kHidden,
                                              BLOCK_M / kNumMulticast, BLOCK_K, kHidden, 1, 128);
    const auto tensor_map_qkv_weight = make_tma_a_desc(cute::UMMA::Major::K, qkv_weight.first, kQRank + kKVRank, kHidden,
                                                       128, BLOCK_K, kHidden, 1, 128);
    const auto tensor_map_x_sf = make_tma_sf_desc(cute::UMMA::Major::MN, x.second, m, kHidden, 128, 32, 1, 0,
                                                  0, false, 1, static_cast<int>(x.second.stride(1)));
    const auto tensor_map_qkv_weight_sf = make_tma_sf_desc(cute::UMMA::Major::MN, qkv_weight.second, kQRank + kKVRank, kHidden, 128, 32, 1, 0,
                                                           0, false, 1, static_cast<int>(qkv_weight.second.stride(1)));
    const auto tensor_map_q_a = make_tma_a_desc(cute::UMMA::Major::K, q_a, m, kQRank,
                                                BLOCK_M, BLOCK_K, kQRank, 1, 128);
    const auto tensor_map_q_b_weight = make_tma_a_desc(cute::UMMA::Major::K, q_b_weight.first, q_out_dim, kQRank,
                                                       q_block_n / kNumMulticast, BLOCK_K, kQRank, 1, 128);
    const auto tensor_map_q_a_sf = make_tma_sf_desc(cute::UMMA::Major::MN, q_a_sf, m, kQRank, 128, 32, 1, 0);
    const auto tensor_map_q_b_weight_sf = make_tma_sf_desc(cute::UMMA::Major::MN, q_b_weight.second, q_out_dim, kQRank,
                                                           256, 32, 1, 0, 0, false, 1, static_cast<int>(q_b_weight.second.stride(1)));
    const auto tensor_map_q = make_tma_cd_desc(q, m, q_out_dim, 128, 32, q_out_dim, 1, 64);

    const auto kernel = jit->compile("sm100_v41f_qkv_lora", std::format(R"(
#include <deep_gemm/experimental/sm100_v41f_qkv_lora.cuh>

using namespace deep_gemm::experimental;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&v41f_qkv_lora_kernel<{}, {}, {}, {}, {}>);
}}
)", num_m_blocks, q_out_dim, num_sms, split_k, num_row_warps));

    jit->launch(kernel, {
        .num_smem_bytes = kNumSmemBytes,
        .grid_dim = dim3(num_sms, 1, 1),
        .block_dim = dim3(num_warps * 32, 1, 1),
        .cluster_dim = dim3(kNumMulticast, 1, 1),
        .cooperative = true,
    }, tensor_map_x, tensor_map_qkv_weight, tensor_map_x_sf, tensor_map_qkv_weight_sf,
       tensor_map_q_a, tensor_map_q_b_weight, tensor_map_q_a_sf, tensor_map_q_b_weight_sf, tensor_map_q,
       partials.data_ptr(), q_norm_weight.data_ptr(), kv_norm_weight.data_ptr(),
       positions.data_ptr(), cos_sin.data_ptr(), slots.data_ptr(), cache.data_ptr(),
       static_cast<uint32_t>(m),
       static_cast<uint32_t>(cache.size(0) * cache.size(1)), static_cast<uint32_t>(cache.size(1)),
       static_cast<uint32_t>(cache.stride(0)),
       q_a.data_ptr(), q_a_sf.data_ptr(), barriers.data_ptr(), eps);
}

} // namespace deep_gemm::experimental
