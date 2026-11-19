#pragma once

#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include "../jit_kernels/impls/sm120_runtime.hpp"
#include "../jit_kernels/impls/runtime_utils.hpp"

namespace deep_gemm::sm120_moe_small {

static void launch(const torch::Tensor& input, const torch::Tensor& b,
                   const torch::Tensor& slots, const torch::Tensor& weights,
                   const torch::Tensor& output, bool l1, int bn, int stages, int bundle) {
    DG_HOST_ASSERT(input.is_cuda());
    const c10::cuda::CUDAGuard guard(input.device());
    const auto* device = at::cuda::getDeviceProperties(input.get_device());
    DG_HOST_ASSERT(device->major == 12 and device->minor == 0);
    DG_HOST_ASSERT(input.dim() == (l1 ? 2 : 3) and b.dim() == 3);
    DG_HOST_ASSERT(slots.dim() == 2 and weights.sizes() == slots.sizes());
    DG_HOST_ASSERT(output.dim() == (l1 ? 3 : 2));
    for (const auto& tensor: {input, b, output, slots, weights}) {
        DG_HOST_ASSERT(tensor.is_cuda() and tensor.device() == input.device() and tensor.is_contiguous());
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16u == 0u);
        for (const auto size: tensor.sizes()) DG_HOST_ASSERT(size > 0 and size <= INT_MAX);
    }
    DG_HOST_ASSERT(input.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(slots.scalar_type() == torch::kLong and weights.scalar_type() == torch::kFloat);
    for (const auto& tensor: {input, b, slots, weights}) DG_HOST_ASSERT(not output.is_alias_of(tensor));
    const int t = slots.size(0), topk = slots.size(1), e = b.size(0);
    const int d = l1 ? output.size(2) : output.size(1), k = input.size(-1);
    const int capacity = l1 ? output.size(1) : input.size(1);
    DG_HOST_ASSERT(t >= 1 and t <= 4 and topk >= 1 and topk <= 8 and topk <= e);
    DG_HOST_ASSERT(capacity >= t and capacity % 128 == 0);
    DG_HOST_ASSERT(b.size(1) == (l1 ? 2 : 1) * d and b.size(2) == k);
    DG_HOST_ASSERT((l1 ? input.size(0) : output.size(0)) == t);
    DG_HOST_ASSERT((l1 ? output.size(0) : input.size(0)) == e);
    DG_HOST_ASSERT(int64_t(e) * capacity <= INT_MAX and int64_t(e) * b.size(1) <= INT_MAX);
    DG_HOST_ASSERT((d == (l1 ? 768 : 2048) and k == (l1 ? 2048 : 768)) or
                   (d == (l1 ? 1536 : 4096) and k == (l1 ? 4096 : 1536)) or
                   (d == (l1 ? 1408 : 2048) and k == (l1 ? 2048 : 1408)) or
                   (d == (l1 ? 2560 : 3584) and k == (l1 ? 3584 : 2560)));
    DG_HOST_ASSERT(bn == 8 or bn == 16 or bn == 32);
    DG_HOST_ASSERT(stages >= 2 and stages <= 4 and bundle >= 1 and bundle <= topk and topk % bundle == 0);
    DG_HOST_ASSERT(l1 or bundle == topk);
    // Values of slots are a device-side caller contract: each route has a
    // unique slot in a valid expert row. No host readback or allocation here.
    const auto map_a = make_tma_2d_desc(input, k, l1 ? t : e * capacity, 64, 1, k, 0);
    const auto map_b = make_tma_2d_desc(b, k, e * b.size(1), 64, bn, k, 128);
    const auto code = std::format(R"(
#include <deep_gemm/impls/sm120_bf16_moe_small.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_bf16_moe_small<{}, {}, {}, {}, {}, {}, {}>);
}}
)", d, k, topk, bn, stages, bundle, l1 ? "true" : "false");
    const int a_bytes = stages * (l1 ? 1 : bundle) * 128;
    const int b_offset = (a_bytes + 1023) & ~1023;
    const int smem = b_offset + stages * bundle * (l1 ? 2 : 1) * bn * 128 +
                     stages * 16 + (l1 ? 0 : topk * bn * 2);
    DG_HOST_ASSERT(static_cast<size_t>(smem) <= device->sharedMemPerBlockOptin);
    const auto kernel = jit->compile(l1 ? "sm120_moe_small_l1" : "sm120_moe_small_l2", code);
    sm120_launch_kernel(kernel, SM120LaunchArgs(t * (d / bn) * (topk / bundle),
        (bundle + 4) * 32, smem, 1), slots.data_ptr(), weights.data_ptr(),
        output.data_ptr(), static_cast<uint32_t>(capacity), input.data_ptr(), b.data_ptr(), map_a, map_b);
}

static void register_apis(pybind11::module_& m) {
    using namespace pybind11::literals;
    // Private building block; the opt-in Python workspace owns fallback policy.
    m.def("_sm120_moe_small", &launch, "input"_a, "weight"_a, "slots"_a,
          "route_weights"_a, "output"_a, "l1"_a, "block_n"_a, "stages"_a, "bundle"_a);
}

} // namespace deep_gemm::sm120_moe_small
