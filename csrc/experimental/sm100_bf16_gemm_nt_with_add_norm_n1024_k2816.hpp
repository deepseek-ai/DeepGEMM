#pragma once

#include <algorithm>
#include <format>
#include <optional>

#include <torch/python.h>

#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/math.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"
#include "../jit_kernels/impls/runtime_utils.hpp"

namespace deep_gemm::experimental {


struct AddNormLaunchPlan {
    bool is_small;
    bool is_whole_row = false;
    bool is_pingpong = false;
    bool unaligned_io = false;
    int block_m, block_n, block_k, split_k;
    int num_stages, num_b_stages, num_cd_pages, num_res_pages;
    bool use_res_tma, align_clusters;

    AddNormLaunchPlan(const bool is_small_in,
                      const int block_m_in, const int block_n_in, const int block_k_in,
                      const int split_k_in, const int num_stages_in,
                      const int num_cd_pages_in, const int num_res_pages_in,
                      const bool use_res_tma_in,
                      const bool align_clusters_in, const int num_b_stages_in = -1)
        : is_small(is_small_in),
          block_m(block_m_in), block_n(block_n_in), block_k(block_k_in), split_k(split_k_in),
          num_stages(num_stages_in), num_b_stages(num_b_stages_in > 0 ? num_b_stages_in : num_stages_in),
          num_cd_pages(num_cd_pages_in), num_res_pages(num_res_pages_in),
          use_res_tma(use_res_tma_in),
          align_clusters(align_clusters_in) {}

    static AddNormLaunchPlan Small(const int block_m_in, const int block_n_in,
                                   const int block_k_in = 128, const int split_k_in = 1) {
        return AddNormLaunchPlan(true, block_m_in, block_n_in, block_k_in, split_k_in,
                                 0, 0, 0, false, false);
    }

    static AddNormLaunchPlan Sliced(const int block_m_in, const int block_n_in,
                                    const int num_stages_in, const int num_cd_pages_in,
                                    const int num_res_pages_in = 0,
                                    const bool use_res_tma_in = false, const bool align_clusters_in = true,
                                    const int num_b_stages_in = -1) {
        return AddNormLaunchPlan(false, block_m_in, block_n_in, 64, 1,
                                 num_stages_in, num_cd_pages_in, num_res_pages_in,
                                 use_res_tma_in, align_clusters_in,
                                 num_b_stages_in);
    }

    static AddNormLaunchPlan WholeRow(const int block_m_in, const int num_cd_pages_in,
                                      const int num_a_stages_in, const int num_b_stages_in,
                                      const bool unaligned_io_in = false, const int num_shared_res_pages_in = 0) {
        AddNormLaunchPlan plan(false, block_m_in, 256, 64, 1, num_a_stages_in,
                               num_cd_pages_in, num_shared_res_pages_in, true, false, num_b_stages_in);
        plan.is_whole_row = true;
        plan.unaligned_io = unaligned_io_in;
        return plan;
    }

    static AddNormLaunchPlan PingPong(const int num_a_stages_in,
                                      const int num_b_stages_in, const int num_res_pages_in) {
        AddNormLaunchPlan plan(false, 128, 128, 64, 1, num_a_stages_in,
                               2, num_res_pages_in, true, false,
                               num_b_stages_in);
        plan.is_pingpong = true;
        return plan;
    }
};

template <int kShapeK>
static AddNormLaunchPlan select_add_norm_plan(const int m, const int num_sms, const bool tma_aligned_io,
                                              const bool deterministic) {
    const int full_clusters = std::max(1, num_sms / 2);
    if (not tma_aligned_io) {
        const bool use_m64 = not deterministic and m <= full_clusters * 64;
        return AddNormLaunchPlan::WholeRow(use_m64 ? 64 : 128, 1, 4, 4, true);
    }

    const int waves_32 = ceil_div(ceil_div(m, 128) * 8, full_clusters);
    const int waves_64 = ceil_div(ceil_div(m, 128) * 4, full_clusters);
    const int num_clusters_128 = ceil_div(m, 256) * 4;
    const int waves_128 = ceil_div(num_clusters_128, full_clusters);

    if (deterministic) {
        // Keep a fixed K-reduction (BLOCK_K=64, no split-K) and fixed 4x256-col sum-of-squares tree
        // across `WholeRow(128)`, `Sliced(128, 128)`, and `PingPong(128, 128)` for bitwise batch invariance.
        if constexpr (kShapeK == 1024) {
            if ((waves_128 > 1 and m <= full_clusters * 128) or
                (m >= (full_clusters * 2 - 1) * 128 and m <= full_clusters * 256))
                return AddNormLaunchPlan::WholeRow(128, 2, 3, 3, false, 1);
            const bool align_clusters = waves_128 <= 6 and
                                        ceil_div(num_clusters_128, (full_clusters / 4) * 4) == waves_128;
            return AddNormLaunchPlan::Sliced(128, 128, 6, 2, 0, false, align_clusters);
        } else {
            if (waves_128 > 1 and m <= full_clusters * 128)
                return AddNormLaunchPlan::WholeRow(128, 1, 4, 4);
            if (waves_128 <= 3)
                return AddNormLaunchPlan::PingPong(5, 5, 2);
            return AddNormLaunchPlan::PingPong(6, 5, 1);
        }
    }

    if constexpr (kShapeK == 1024) {
        if (m <= 32 and num_sms >= 128)
            return AddNormLaunchPlan::Small(32, 16);
        if (m <= 224 and num_sms >= 128)
            return AddNormLaunchPlan::Small(32, 64);
        if (ceil_div(m, 128) * 16 <= full_clusters)
            return AddNormLaunchPlan::Sliced(64, 32, 16, 1, 0, true);
        if (ceil_div(m, 64) * 16 <= num_sms and num_sms >= 128)
            return AddNormLaunchPlan::Small(64, 64);
        if (waves_32 <= 2)
            return AddNormLaunchPlan::Sliced(64, 64, 12, 2);
        if (waves_64 <= 3) {
            if (waves_32 < 2 * waves_64)
                return AddNormLaunchPlan::Sliced(64, 64, 12, 2, 2, true);
            return AddNormLaunchPlan::Sliced(64, 128, 8, 4);
        }
        if (m <= full_clusters * 128 or (m >= (full_clusters * 2 - 1) * 128 and m <= full_clusters * 256))
            return AddNormLaunchPlan::WholeRow(128, 2, 3, 3, false, 1);
        if (waves_64 <= 6 and waves_64 < 2 * waves_128)
            return AddNormLaunchPlan::Sliced(64, 128, 8, 4);

        const bool align_clusters = waves_128 <= 6 and
                                    ceil_div(num_clusters_128, (full_clusters / 4) * 4) == waves_128;
        return AddNormLaunchPlan::Sliced(128, 128, 6, 2, 0, false, align_clusters);
    } else {
        if (ceil_div(m, 32) * 48 <= num_sms and num_sms >= 128)
            return AddNormLaunchPlan::Small(32, 64, 128, 8);
        if (ceil_div(m, 64) * 48 <= num_sms and num_sms >= 128)
            return AddNormLaunchPlan::Small(64, 64, 128, 8);
        if (ceil_div(m, 128) * 16 <= full_clusters)
            return AddNormLaunchPlan::Sliced(64, 32, 14, 1, 0, true);
        if (waves_32 == 1)
            return AddNormLaunchPlan::Sliced(64, 64, 12, 2);
        if (waves_64 <= 3) {
            if (waves_32 < 2 * waves_64 and m <= 5504)
                return AddNormLaunchPlan::Sliced(64, 64, 13, 2);
            return AddNormLaunchPlan::Sliced(64, 128, 8, 4, 0, true);
        }
        if (m <= full_clusters * 128)
            return AddNormLaunchPlan::WholeRow(128, 1, 4, 4);
        if (waves_64 <= 6 and waves_64 < 2 * waves_128)
            return AddNormLaunchPlan::Sliced(64, 128, 10, 2, 2, true, true, 7);

        if (waves_128 <= 3)
            return AddNormLaunchPlan::PingPong(5, 5, 2);
        return AddNormLaunchPlan::PingPong(6, 5, 1);
    }
}

template <int kShapeK, bool kHasBias>
static void launch_bf16_gemm_nt_with_add_norm(
    const torch::Tensor& a, const torch::Tensor& b, const std::optional<torch::Tensor>& bias_opt,
    const torch::Tensor& gamma, const torch::Tensor& residual, const torch::Tensor& d) {
    constexpr int kShapeN = 1024, kNumBlockAtomK = 64, kNumStoreCols = 64, kNumSwizzleBytes = 128;
    constexpr int kNumSmallTrailerBytes = 2048, kNumTrailerBytes = 3072;
    constexpr int64_t kNumCounterSlots = 32;

    DG_HOST_ASSERT(a.dim() == 2 and b.dim() == 2);
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and gamma.is_contiguous() and
                   residual.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16 and
                   gamma.scalar_type() == torch::kBFloat16 and residual.scalar_type() == torch::kBFloat16 and
                   d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);

    const auto m = static_cast<int>(a.size(0));
    DG_HOST_ASSERT(a.size(1) == kShapeK and b.size(0) == kShapeN and b.size(1) == kShapeK and gamma.numel() == kShapeN);
    DG_HOST_ASSERT(residual.size(0) == m and residual.size(1) == kShapeN and d.size(0) == m and d.size(1) == kShapeN);
    DG_HOST_ASSERT(m == 0 or d.data_ptr() != residual.data_ptr());
    if constexpr (kHasBias) {
        DG_HOST_ASSERT(bias_opt.has_value() and bias_opt->is_contiguous() and
                       bias_opt->numel() == kShapeN and bias_opt->scalar_type() == torch::kBFloat16);
    }
    if (m == 0)
        return;

    const auto num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms == jit->device.get_num_sms() and num_sms % 2 == 0 and
                   num_sms >= 16 and "The kernel requires the full SM count");

    const auto res_ptr = static_cast<void*>(residual.data_ptr());
    const auto d_ptr = static_cast<void*>(d.data_ptr());
    const bool tma_aligned_io = ((reinterpret_cast<uintptr_t>(res_ptr) | reinterpret_cast<uintptr_t>(d_ptr)) & 0xfu) == 0;
    const bool deterministic = heuristics_runtime->get_deterministic_algorithms();
    auto plan = select_add_norm_plan<kShapeK>(m, num_sms, tma_aligned_io, deterministic);

    const int64_t num_m_blocks = ceil_div(m, plan.block_m);
    const int64_t num_m_pairs = (num_m_blocks + 1) >> 1;
    const int64_t num_slices = plan.is_whole_row ? 1 : (plan.is_small ? (kShapeN / plan.block_n) : (kShapeN / (2 * plan.block_n)));
    const int64_t num_tiles_small = num_m_blocks * num_slices;
    if (plan.is_small) {
        if (plan.split_k > 1)
            plan.split_k = std::min(plan.split_k, std::max(1, static_cast<int>(num_sms / num_tiles_small)));
    } else {
        plan.split_k = 1;
    }

    torch::Tensor partials, counters;
    if (not plan.is_whole_row) {
        const int64_t aligned_m_blocks = plan.is_small ? num_m_blocks : (num_m_pairs * 2);
        const bool is_split = plan.is_small and plan.split_k > 1;
        const int64_t split_extra_floats = is_split ? (num_tiles_small * (plan.split_k - 1) * plan.block_m * plan.block_n) : 0;
        partials = torch::empty({aligned_m_blocks * plan.block_m * num_slices + split_extra_floats}, a.options().dtype(torch::kFloat));
        counters = torch::empty({(aligned_m_blocks + (is_split ? num_tiles_small : 0)) * kNumCounterSlots}, a.options().dtype(torch::kInt32));
    }
    const auto partials_ptr = partials.defined() ? partials.data_ptr<float>() : nullptr;
    const auto counters_ptr = counters.defined() ? reinterpret_cast<uint32_t*>(counters.data_ptr<int>()) : nullptr;

    constexpr auto kBaseName = kHasBias
        ? "sm100_bf16_gemm_nt_with_bias_add_norm_n1024_k1024"
        : "sm100_bf16_gemm_nt_with_add_norm_n1024_k2816";
    std::string kernel_suffix, impl_args;
    int num_smem_bytes;
    if (plan.is_whole_row) {
        num_smem_bytes = ((2 * plan.num_cd_pages + plan.num_res_pages) * 128 * kNumStoreCols +
                          plan.num_stages * plan.block_m * kNumBlockAtomK +
                          plan.num_b_stages * 256 * kNumBlockAtomK) * 2 + kNumTrailerBytes;
        kernel_suffix = "_whole_row";
        impl_args = std::format("{}, {}, {}, {}, {}, {}, {}",
                                num_sms, plan.block_m, plan.num_cd_pages, plan.num_stages,
                                plan.num_b_stages, plan.unaligned_io ? "true" : "false", plan.num_res_pages);
    } else if (plan.is_small) {
        const int umma_m = plan.block_n < 64 ? 64 : plan.block_m;
        const int num_stage_bytes = (umma_m + plan.block_n) * plan.block_k * 2;
        const int num_stage_cap = kShapeK > 1024 ? (plan.block_k == 128 ? 8 : 12) : 16;
        const int num_stages = std::max(1, std::min(ceil_div(kShapeK / plan.block_k, plan.split_k),
                                                    std::min(num_stage_cap, 221184 / num_stage_bytes)));
        num_smem_bytes = num_stages * num_stage_bytes + kNumSmallTrailerBytes;
        kernel_suffix = "_small";
        impl_args = std::format("{}, {}, {}, {}, {}", num_sms, plan.block_m, plan.block_n, plan.block_k, plan.split_k);
    } else {
        num_smem_bytes = (plan.num_cd_pages + plan.num_res_pages) * plan.block_m * kNumStoreCols * 2 +
                         (plan.num_stages * plan.block_m + plan.num_b_stages * plan.block_n) * kNumBlockAtomK * 2 + kNumTrailerBytes;
        if (plan.is_pingpong) {
            kernel_suffix = "_pingpong";
            impl_args = std::format("{}, {}, {}, {}", num_sms, plan.num_stages, plan.num_res_pages, plan.num_b_stages);
        } else {
            impl_args = std::format("{}, {}, {}, {}, {}, {}, {}, {}", num_sms, plan.block_m, plan.block_n,
                                    plan.num_stages, plan.num_cd_pages,
                                    plan.use_res_tma ? "true" : "false", plan.num_res_pages, plan.num_b_stages);
        }
    }
    const auto kernel = jit->compile(std::format("{}{}", kBaseName, kernel_suffix), std::format(R"(
#include <deep_gemm/experimental/{0}.cuh>

using namespace deep_gemm::experimental;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{0}{1}_impl<{2}>);
}}
)", kBaseName, kernel_suffix, impl_args));

    const int box_m = plan.is_small ? std::min(plan.block_m, m) : plan.block_m;
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, kShapeK,
                                              box_m, kNumBlockAtomK, kShapeK, 1, kNumSwizzleBytes);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, kShapeN, kShapeK,
                                              plan.block_n, kNumBlockAtomK, kShapeK, 1, kNumSwizzleBytes);
    const auto tensor_map_residual = (plan.unaligned_io or plan.is_small)
        ? CUtensorMap() : make_tma_cd_desc(residual, m, kShapeN, box_m, kNumStoreCols, kShapeN, 1, kNumSwizzleBytes);
    const bool need_tma_d = plan.is_whole_row ? (not plan.unaligned_io) : plan.is_pingpong;
    const auto tensor_map_d = need_tma_d
        ? make_tma_cd_desc(d, m, kShapeN, box_m, kNumStoreCols, kShapeN, 1, kNumSwizzleBytes) : CUtensorMap();

    const auto invoke_launch = [&](const uint32_t grid_blocks, const uint32_t block_threads,
                                   const uint32_t cluster_blocks, auto&& launch_fn) {
        const auto do_launch = [&](const auto&... all_args) {
            jit->launch(kernel, {
                .num_smem_bytes = num_smem_bytes,
                .grid_dim = dim3(grid_blocks, 1, 1),
                .block_dim = dim3(block_threads, 1, 1),
                .cluster_dim = dim3(cluster_blocks, 1, 1),
                .cooperative = true,
                .enable_pdl = true,
            }, all_args...);
        };
        if constexpr (kHasBias)
            launch_fn(do_launch, static_cast<const void*>(bias_opt->data_ptr()));
        else
            launch_fn(do_launch);
    };

    const auto m_u32 = static_cast<uint32_t>(m);
    const auto gamma_ptr = static_cast<const void*>(gamma.data_ptr());
    if (plan.is_whole_row) {
        const auto launch_blocks = static_cast<uint32_t>(std::min<int64_t>(num_sms, num_m_blocks * 2));
        invoke_launch(launch_blocks, 384, 2, [&](const auto& do_launch, const auto&... bias_args) {
            do_launch(m_u32, tensor_map_a, tensor_map_b, tensor_map_residual, tensor_map_d,
                      res_ptr, bias_args..., gamma_ptr, d_ptr);
        });
    } else if (plan.is_small) {
        const int64_t total_ctas = num_tiles_small * plan.split_k;
        const int64_t aligned_sms = (num_sms / num_slices) * num_slices;
        const int64_t max_small_sms = (plan.split_k == 1 and aligned_sms > 0 and
                                       ceil_div(num_tiles_small, aligned_sms) == ceil_div(num_tiles_small, static_cast<int64_t>(num_sms)))
                                          ? aligned_sms : static_cast<int64_t>(num_sms);
        const auto launch_sms = static_cast<uint32_t>(plan.split_k > 1 ? total_ctas : std::min<int64_t>(max_small_sms, total_ctas));
        invoke_launch(launch_sms, 256, 1, [&](const auto& do_launch, const auto&... bias_args) {
            do_launch(m_u32, tensor_map_a, tensor_map_b,
                      res_ptr, d_ptr, bias_args..., gamma_ptr, partials_ptr, counters_ptr);
        });
    } else {
        const int64_t num_tiles = num_m_pairs * num_slices, raw_clusters = num_sms / 2;
        const int64_t aligned_clusters = (raw_clusters / num_slices) * num_slices;
        const int64_t max_clusters = (plan.align_clusters and aligned_clusters > 0 and
                                      ceil_div(num_tiles, aligned_clusters) == ceil_div(num_tiles, raw_clusters))
                                         ? aligned_clusters : raw_clusters;
        const auto launch_blocks = static_cast<uint32_t>(std::min<int64_t>(max_clusters * 2, num_tiles * 2));
        invoke_launch(launch_blocks, plan.is_pingpong ? 384 : 512, 2, [&](const auto& do_launch, const auto&... bias_args) {
            if (plan.is_pingpong) {
                do_launch(m_u32, tensor_map_a, tensor_map_b, tensor_map_residual, tensor_map_d,
                          bias_args..., gamma_ptr, partials_ptr, counters_ptr);
            } else {
                do_launch(m_u32, tensor_map_a, tensor_map_b, tensor_map_residual,
                          res_ptr, bias_args..., gamma_ptr, d_ptr, partials_ptr, counters_ptr);
            }
        });
    }
}

// torch: residual += a @ b.T (bf16); d = (residual.float() * rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
static void bf16_gemm_nt_with_add_norm_n1024_k2816(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& gamma,
    const torch::Tensor& residual, const torch::Tensor& d) {
    launch_bf16_gemm_nt_with_add_norm<2816, false>(a, b, std::nullopt, gamma, residual, d);
}

} // namespace deep_gemm::experimental
