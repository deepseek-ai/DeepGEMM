#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/layout.hpp"
#include "../heuristics/runtime.hpp"

namespace deep_gemm {

class PackFP32IntoUE8M0Runtime final {
public:
    struct Args {
        int num_groups, mn, sf_k, packed_sf_k, gran_k, k_alignment;
        bool use_psum_layout;
        int block_mn, block_packed_sf_k;
        void *sf, *out, *grouped_layout;

        deep_jit::cuda::LaunchOptions options;
    };

    static void compile_and_launch(const std::string& tag, const Args& args) {
        const auto kernel = jit->compile(tag, std::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pack_fp32_into_ue8m0<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", args.num_groups, args.options.block_dim->x, args.block_mn, args.block_packed_sf_k,
    "true", args.use_psum_layout ? "true" : "false"));

        // Launch
        jit->launch(
            kernel, args.options,
            args.sf, args.out, args.grouped_layout, args.mn, args.sf_k, args.packed_sf_k,
            args.gran_k, args.k_alignment
        );
    }
};

static std::tuple<int, int, int, int, int, torch::Tensor> preprocess_sf(const torch::Tensor& sf) {
    // NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    const auto dim = sf.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    DG_HOST_ASSERT(sf.scalar_type() == torch::kFloat);
    const auto batched_sf = dim == 2 ? sf.unsqueeze(0) : sf;

    const auto [num_sf_batches, mn, sf_k] = get_shape<3>(batched_sf);
    const auto tma_aligned_mn = get_tma_aligned_size(mn, static_cast<int>(sf.element_size()));
    return {dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf};
}

static torch::Tensor get_mn_major_tma_aligned_tensor(const torch::Tensor& sf) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);

    // The last kernel already gives a column-major TMA aligned layout
    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2) and batched_sf.stride(1) == 1 and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? batched_sf.squeeze(0) : batched_sf;

    const auto out = torch::empty_strided({num_sf_batches, mn, sf_k},
                                          {tma_aligned_mn * sf_k, 1, tma_aligned_mn},
                                          batched_sf.options());

    if (not batched_sf.is_contiguous()) {
        // Fallback to PyTorch's slow copy if not contiguous
        // ReSharper disable once CppExpressionWithoutSideEffects
        out.copy_(batched_sf);
    } else {
        constexpr int block_mn = 64;
        constexpr int block_sf_k = 128;
        constexpr int num_threads = 512;
        const auto smem_sf_k = sf_k < block_sf_k ? sf_k : block_sf_k;
        const auto smem_size = block_mn * (smem_sf_k + (1 - smem_sf_k % 2)) * static_cast<int>(sizeof(float));

        // Compile
        const auto kernel = jit->compile("transpose_fp32", std::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_fp32<
        {}, {}, {}, {}
    >);
}};
)", num_threads, block_mn, sf_k, block_sf_k));

        // Launch
        jit->launch(
            kernel, {
                .num_smem_bytes = smem_size,
                .grid_dim = dim3(ceil_div(mn, block_mn) * ceil_div(sf_k, block_sf_k), num_sf_batches, 1),
                .block_dim = dim3(num_threads, 1, 1),
            },
            batched_sf.data_ptr(), out.data_ptr(), static_cast<uint32_t>(mn)
        );
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(const torch::Tensor& sf) {
    const auto sf_reshaped = (sf.dim() == 2) ? sf.unsqueeze(0) : sf;

    // First, convert into UE8M0 `uint8_t`
    const auto ue8m0_tensor = sf_reshaped.view(torch::kInt32).bitwise_right_shift(23).to(torch::kUInt8);

    // Second, make padded packed tensors
    const auto [num_sf_batches, mn, k] = get_shape<3>(sf_reshaped);
    const auto aligned_mn = get_tma_aligned_size(mn, 4);
    const auto aligned_k  = align(k, 4);

    const auto options = torch::TensorOptions().device(sf.device()).dtype(torch::kUInt8);
    auto padded = torch::zeros({num_sf_batches, aligned_mn, aligned_k}, options);
    // ReSharper disable once CppExpressionWithoutSideEffects
    padded.slice(1, 0, mn).slice(2, 0, k).copy_(ue8m0_tensor);
    padded = padded.view(-1).view(torch::kInt32).view({num_sf_batches, aligned_mn, aligned_k / 4});

    // Finally, transpose
    auto out = torch::empty_strided({num_sf_batches, aligned_mn, aligned_k / 4},
                                    {aligned_mn * (aligned_k / 4), 1, aligned_mn},
                                    at::TensorOptions().device(sf.device()).dtype(torch::kInt32));
    out = out.copy_(padded).slice(1, 0, mn);
    return (sf.dim() == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::Tensor& sf,
                                                                  const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    const auto [dim, num_sf_batches, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);
    const auto packed_sf_k = ceil_div(sf_k, 4);
    const auto out = torch::empty_strided({num_sf_batches, mn, packed_sf_k},
                                          {packed_sf_k * tma_aligned_mn, 1, tma_aligned_mn},
                                          at::TensorOptions().device(batched_sf.device()).dtype(torch::kInt));

    // PSUM layout (always 2D contiguous) lets the pack kernel skip uninitialized MN gap rows
    const auto use_psum_layout = psum_layout.has_value();
    if (use_psum_layout) {
        DG_HOST_ASSERT(num_sf_batches == 1 and batched_sf.is_contiguous());
        DG_HOST_ASSERT(psum_layout->scalar_type() == torch::kInt and psum_layout->is_contiguous());
        DG_HOST_ASSERT(psum_layout->numel() > 0);
    }
    const auto m_alignment = use_psum_layout ? heuristics_runtime->get_mk_alignment_for_contiguous_layout() : 0;
    const auto num_psum_groups = use_psum_layout ? static_cast<int>(psum_layout->numel()) : 1;

    if (batched_sf.is_contiguous()) {
        if ((mn * sf_k) % 4 != 0 and num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);

        constexpr int block_mn = 48;
        constexpr int block_sf_k = 128;
        constexpr int num_threads = 512;
        const auto psum_smem_elems = use_psum_layout ? align(num_psum_groups * 2, 4) : 0;
        const auto smem_sf_k = sf_k < block_sf_k ? sf_k : block_sf_k + 1;
        const auto smem_size = block_mn * smem_sf_k * 4 + psum_smem_elems * 4;

        // Compile
        const auto kernel = jit->compile("transpose_and_pack_fp32_into_ue8m0", std::format(R"(
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&transpose_and_pack_fp32_into_ue8m0<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", num_threads, block_mn, sf_k, block_sf_k, num_psum_groups, use_psum_layout ? "true" : "false"));

        // Launch
        jit->launch(
            kernel, {
                .num_smem_bytes = smem_size,
                .grid_dim = dim3(ceil_div(mn, block_mn) * ceil_div(sf_k, block_sf_k), num_sf_batches, 1),
                .block_dim = dim3(num_threads, 1, 1),
            },
            batched_sf.data_ptr(), out.data_ptr(), static_cast<uint32_t>(mn),
            use_psum_layout ? psum_layout->data_ptr() : nullptr, static_cast<uint32_t>(m_alignment)
        );
    } else {
        DG_HOST_ASSERT(not use_psum_layout);
        if (mn % 4 != 0 or num_sf_batches > 1)
            return get_mn_major_tma_aligned_packed_ue8m0_tensor_torch(sf);
        DG_HOST_ASSERT(batched_sf.stride(1) == 1 and batched_sf.stride(2) == mn);

        constexpr int block_mn = 128;
        constexpr int block_packed_sf_k = 16;
        constexpr int num_threads = 512;

        // Compile and launch
        PackFP32IntoUE8M0Runtime::compile_and_launch("pack_fp32_into_ue8m0", {
            .num_groups = 1,
            .mn = mn,
            .sf_k = sf_k,
            .packed_sf_k = packed_sf_k,
            .gran_k = 128,
            .k_alignment = 128,
            .use_psum_layout = false,
            .block_mn = block_mn,
            .block_packed_sf_k = block_packed_sf_k,
            .sf = batched_sf.data_ptr(),
            .out = out.data_ptr(),
            .grouped_layout = nullptr,
            .options = {
                .grid_dim = dim3(ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k), 1),
                .block_dim = dim3(num_threads, 1, 1),
            }
        });
    }
    return (dim == 2) ? out.squeeze(0) : out;
}

static torch::Tensor get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(const torch::Tensor& sf,
                                                                            const torch::Tensor& grouped_layout,
                                                                            const std::optional<std::vector<int>>& ks_cpu,
                                                                            const int gran_k,
                                                                            const int k_alignment,
                                                                            const bool& use_psum_layout) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and (gran_k == 32 or gran_k == 128) and k_alignment % 128 == 0);
    const auto [sf_k, mn] = get_shape<2>(sf);
    const auto num_groups = static_cast<int>(grouped_layout.numel());

    DG_HOST_ASSERT(sf.is_contiguous());
    DG_HOST_ASSERT(num_groups <= 128 and mn % 4 == 0);
    DG_HOST_ASSERT(grouped_layout.is_contiguous() and grouped_layout.scalar_type() == torch::kInt);

    const auto has_synced_ks = ks_cpu.has_value() and not ks_cpu.value().empty();
    if (has_synced_ks) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
    } else {
        DG_HOST_ASSERT(use_psum_layout);
    }

    int packed_sf_k = 0;
    if (has_synced_ks) {
        int ref_sf_k = 0;
        for (const auto k: ks_cpu.value()) {
            DG_HOST_ASSERT(k % k_alignment == 0 and k % gran_k == 0);
            const auto group_sf_k = k / gran_k;
            ref_sf_k += group_sf_k;
            packed_sf_k += ceil_div(group_sf_k, 4);
        }
        DG_HOST_ASSERT(ref_sf_k == sf_k);
    } else {
        // Exact group sizes are read from the psum layout by the pack kernel.
        // This upper bound allows three tail-padding slots per group.
        packed_sf_k = (sf_k + num_groups * 3) / 4;
    }
    if (packed_sf_k == 0)
        return torch::empty({0, mn}, at::TensorOptions().device(sf.device()).dtype(torch::kInt));

    const auto out = torch::empty({packed_sf_k, mn}, at::TensorOptions().device(sf.device()).dtype(torch::kInt));

    constexpr int block_mn = 128;
    constexpr int block_packed_sf_k = 16;
    constexpr int num_threads = 512;

    // Compile and launch
    PackFP32IntoUE8M0Runtime::compile_and_launch("pack_fp32_into_ue8m0", {
        .num_groups = num_groups,
        .mn = mn,
        .sf_k = sf_k,
        .packed_sf_k = packed_sf_k,
        .gran_k = gran_k,
        .k_alignment = k_alignment,
        .use_psum_layout = use_psum_layout,
        .block_mn = block_mn,
        .block_packed_sf_k = block_packed_sf_k,
        .sf = sf.data_ptr(),
        .out = out.data_ptr(),
        .grouped_layout = grouped_layout.data_ptr(),
        .options = {
            .grid_dim = dim3(ceil_div(mn, block_mn), ceil_div(packed_sf_k, block_packed_sf_k), 1),
            .block_dim = dim3(num_threads, 1, 1),
        }
    });
    return out;
}

// Validate a user-provided, already packed UE8M0 (`int32`) SF tensor.
static torch::Tensor check_k_grouped_packed_ue8m0_tensor(const torch::Tensor& sf,
                                                         const torch::Tensor& grouped_layout,
                                                         const std::optional<std::vector<int>>& ks_cpu,
                                                         const int gran_k,
                                                         const int k_alignment,
                                                         const bool& use_psum_layout) {
    DG_HOST_ASSERT(sf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(gran_k == 32);
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(sf.is_contiguous());

    const auto [packed_sf_k, mn] = get_shape<2>(sf);
    const auto num_groups = static_cast<int>(grouped_layout.numel());
    DG_HOST_ASSERT(mn % 4 == 0);
    DG_HOST_ASSERT(sf.stride(0) == mn and sf.stride(1) == 1);
    DG_HOST_ASSERT(grouped_layout.is_contiguous() and grouped_layout.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(packed_sf_k > 0);

    const auto has_synced_ks = ks_cpu.has_value() and not ks_cpu.value().empty();
    if (has_synced_ks) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
        int required_packed_sf_k = 0;
        for (const auto k: ks_cpu.value()) {
            DG_HOST_ASSERT(k % k_alignment == 0);
            required_packed_sf_k += ceil_div(k, gran_k * 4);
        }
        DG_HOST_ASSERT(packed_sf_k >= required_packed_sf_k);
    } else {
        DG_HOST_ASSERT(use_psum_layout);
    }
    return sf;
}

} // namespace deep_gemm
