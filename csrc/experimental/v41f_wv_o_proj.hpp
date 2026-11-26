#pragma once

#include <algorithm>
#include <format>
#include <limits>
#include <unordered_map>

#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/python.h>

#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/math.hpp"
#include "../jit_kernels/impls/runtime_utils.hpp"

namespace deep_gemm::experimental {

static const torch::Tensor& get_v41f_wv_o_proj_ready(const torch::TensorOptions& options,
                                                     const int& num_m_blocks, const int& num_elements) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    static std::unordered_map<c10::cuda::CUDAStream, std::unordered_map<int, torch::Tensor>> ready_by_stream;
    auto& ready = ready_by_stream[stream][num_m_blocks];
    if (not ready.defined()) {
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        ready = torch::zeros({num_elements}, options.dtype(torch::kLong));
    }
    return ready;
}

// V41F grouped Wv projection -> per-32 FP8 quantization -> output projection.
// FP8 pairs use packed UE8M0 scales; Wv weights are already permuted.
// Warm up each 256-token M bucket on its stream before CUDA Graph capture.
static void v41f_wv_o_proj(const std::pair<torch::Tensor, torch::Tensor>& x,
                           const std::pair<torch::Tensor, torch::Tensor>& wv_weight,
                           const std::pair<torch::Tensor, torch::Tensor>& o_weight,
                           const torch::Tensor& y) {
    constexpr int kNumGroups = 8, kInputRank = 4096, kOutputRank = 1024, kHidden = 5120;
    constexpr int BLOCK_M = 128, BLOCK_WV = 128, BLOCK_K = 128, kSwizzleCDMode = 32;
    constexpr int kNumThreads = 256, kNumSmemBytes = 231424;
    constexpr int kNumInputSF = kInputRank / 128, kNumOutputSF = kNumGroups * kOutputRank / 128;

    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    DG_HOST_ASSERT(x.first.dim() == 3 and x.first.size(0) >= 1 and x.first.size(0) <= std::numeric_limits<int>::max() - 256);
    const int m = static_cast<int>(x.first.size(0)), aligned_m = align(m, 4);
    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms == jit->device.get_num_sms() and num_sms % 2 == 0 and "V41F requires all SMs");
    DG_HOST_ASSERT(kNumSmemBytes + 1024 <= jit->device.get_num_smem_bytes());
    DG_HOST_ASSERT(x.first.is_cuda() and x.first.get_device() == at::cuda::current_device());
    for (const auto& t: {x.first, x.second, wv_weight.first, wv_weight.second, o_weight.first, o_weight.second, y})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == x.first.device());
    for (const auto& t: {x.first, wv_weight.first, o_weight.first}) {
        DG_HOST_ASSERT(t.scalar_type() == torch::kFloat8_e4m3fn and t.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0);
    }
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0);
    for (const auto& sf: {x.second, wv_weight.second, o_weight.second}) {
        DG_HOST_ASSERT(sf.scalar_type() == torch::kInt);
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(sf.data_ptr()) % 16 == 0);
    }
    DG_HOST_ASSERT(x.first.sizes() == at::IntArrayRef({m, kNumGroups, kInputRank}));
    DG_HOST_ASSERT(wv_weight.first.sizes() == at::IntArrayRef({kNumGroups, kOutputRank, kInputRank}));
    DG_HOST_ASSERT(o_weight.first.sizes() == at::IntArrayRef({kHidden, kNumGroups * kOutputRank}));
    DG_HOST_ASSERT(y.sizes() == at::IntArrayRef({m, kHidden}));
    DG_HOST_ASSERT(x.second.sizes() == at::IntArrayRef({m, kNumGroups, kNumInputSF}));
    DG_HOST_ASSERT((x.second.stride(0) == 1 or m == 1) and x.second.stride(2) >= aligned_m and
                   x.second.stride(2) <= std::numeric_limits<int>::max() and
                   x.second.stride(2) % 4 == 0 and x.second.stride(1) == x.second.stride(2) * kNumInputSF);
    DG_HOST_ASSERT(wv_weight.second.sizes() == at::IntArrayRef({kNumGroups, kOutputRank, kNumInputSF}));
    DG_HOST_ASSERT(wv_weight.second.strides() == at::IntArrayRef({kOutputRank * kNumInputSF, 1, kOutputRank}));
    DG_HOST_ASSERT(o_weight.second.sizes() == at::IntArrayRef({kHidden, kNumOutputSF}));
    DG_HOST_ASSERT(o_weight.second.strides() == at::IntArrayRef({1, kHidden}));

    const int num_m_blocks = ceil_div(m, 256) * 2;
    const int block_o = std::min(144, align(ceil_div(kHidden, std::max(1, num_sms / num_m_blocks)), 16));
    const auto& ready = get_v41f_wv_o_proj_ready(x.first.options(), num_m_blocks, num_m_blocks + num_sms / 2);
    const auto mid = torch::empty({m, kNumGroups, kOutputRank}, x.first.options());
    const auto sf = torch::empty({kNumOutputSF, aligned_m}, x.second.options()).transpose(0, 1);

    const auto tensor_map_x = make_tma_3d_desc(x.first, kInputRank, m, kNumGroups, BLOCK_K, BLOCK_M, 1,
                                               kNumGroups * kInputRank, kInputRank, 128);
    const auto tensor_map_wv = make_tma_3d_desc(wv_weight.first, kInputRank, kOutputRank, kNumGroups, BLOCK_K, BLOCK_WV / 2, 1,
                                                kInputRank, kInputRank * kOutputRank, 128);
    const auto tensor_map_x_sf = make_tma_sf_desc(cute::UMMA::Major::MN, x.second, m, kInputRank, 128, 32, kNumGroups, 0,
                                                  0, false, 1, static_cast<int>(x.second.stride(2)));
    const auto tensor_map_wv_sf = make_tma_sf_desc(cute::UMMA::Major::MN, wv_weight.second, kOutputRank, kInputRank, 128, 32, kNumGroups, 0);
    const auto tensor_map_mid = make_tma_3d_desc(mid, kNumGroups * kOutputRank, m, 1, BLOCK_K, BLOCK_M, 1,
                                                 kNumGroups * kOutputRank, kNumGroups * kOutputRank, 128);
    const auto tensor_map_o = make_tma_3d_desc(o_weight.first, kNumGroups * kOutputRank, kHidden, 1, BLOCK_K, block_o / 2, 1,
                                               kNumGroups * kOutputRank, kHidden * kNumGroups * kOutputRank, 128);
    const auto tensor_map_mid_sf = make_tma_sf_desc(cute::UMMA::Major::MN, sf, m, kNumGroups * kOutputRank, 128, 32, 1, 0);
    const auto tensor_map_o_sf = make_tma_sf_desc(cute::UMMA::Major::MN, o_weight.second, kHidden, kNumGroups * kOutputRank,
                                                   align(block_o, 128), 32, 1, 0);
    const auto tensor_map_y = make_tma_cd_desc(y, m, kHidden, BLOCK_M, kSwizzleCDMode / 2, kHidden, 1, kSwizzleCDMode);

    const auto kernel = jit->compile("sm100_v41f_wv_o_proj", std::format(R"(
#include <deep_gemm/experimental/sm100_v41f_wv_o_proj.cuh>

using namespace deep_gemm::experimental;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&v41f_wv_o_proj_kernel<{}, {}, {}>);
}}
)", num_m_blocks, block_o, num_sms));

    jit->launch(kernel, {
        .num_smem_bytes = kNumSmemBytes,
        .grid_dim = dim3(num_sms, 1, 1),
        .block_dim = dim3(kNumThreads, 1, 1),
        .cluster_dim = dim3(2, 1, 1),
        .cooperative = true,
    }, static_cast<uint32_t>(m), ready.data_ptr(), mid.data_ptr(), sf.data_ptr(),
       tensor_map_x, tensor_map_wv, tensor_map_x_sf, tensor_map_wv_sf,
       tensor_map_mid, tensor_map_o, tensor_map_mid_sf, tensor_map_o_sf, tensor_map_y);
}

} // namespace deep_gemm::experimental
