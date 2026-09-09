#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm120_bf16_gemm.hpp"
#endif

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"
#include <torch/library.h>
#include "../torch_library_utils.hpp"

namespace deep_gemm::gemm {

static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == c.value().scalar_type());
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

#if DG_TENSORMAP_COMPATIBLE
// SM120 MMA consumes K-major operands: repack packed-FP4 (needs `logical_mn`) or copy
static torch::Tensor sm120_to_k_major(const torch::Tensor& t, const cute::UMMA::Major& major,
                                      const int& logical_mn) {
    if (major == cute::UMMA::Major::K)
        return t;
    return t.scalar_type() == kPackedFP4 ? fp4_repack_to_k_major(t, logical_mn) : t.contiguous();
}
#endif

static int check_k_grouped_args(const std::optional<std::vector<int>>& ks_cpu,
                                const torch::Tensor& grouped_layout,
                                const int& num_groups,
                                const bool& use_psum_layout,
                                const int& k_alignment,
                                const int& sum_k_if_ks_cpu_missing = 0) {
    DG_HOST_ASSERT(grouped_layout.is_contiguous());
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(static_cast<int>(grouped_layout.numel()) == num_groups);

    if (ks_cpu.has_value() and not ks_cpu.value().empty()) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
        int sum_k = 0;
        for (const auto k: ks_cpu.value()) {
            DG_HOST_ASSERT(k % k_alignment == 0);
            sum_k += k;
        }
        return sum_k;
    }
    DG_HOST_ASSERT(use_psum_layout);
    return sum_k_if_ks_cpu_missing;
}

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

// SM120: AB-swap decision must precede the single SF transform, so it owns its own flow
static void fp8_fp4_gemm_nt_sm120(const std::pair<torch::Tensor, torch::Tensor>& a,
                                  const std::pair<torch::Tensor, torch::Tensor>& b,
                                  const torch::Tensor& d,
                                  const std::optional<torch::Tensor>& c,
                                  const std::optional<std::tuple<int, int, int>>& recipe,
                                  const std::optional<std::tuple<int, int>>& recipe_a,
                                  const std::optional<std::tuple<int, int>>& recipe_b,
                                  const std::string& compiled_dims,
                                  const bool& disable_ue8m0_cast,
                                  const cute::UMMA::Major& major_a,
                                  const cute::UMMA::Major& major_b,
                                  const int& m, const int& n, const int& k) {
    // Force K-major operands
    const auto a_data = sm120_to_k_major(a.first, major_a, m);
    const auto b_data = sm120_to_k_major(b.first, major_b, n);
    constexpr auto k_major = cute::UMMA::Major::K;

    const bool is_mixed_fp4 = (a_data.scalar_type() != b_data.scalar_type()) and
                              (a_data.scalar_type() == kPackedFP4 or b_data.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(!is_mixed_fp4 or k % 128 == 0);

    // AB-swap for small-M decode: swap A↔B so small M becomes N (BN=16).
    // K-major B has N as TMA outer dim — no minimum size restriction.
    constexpr int kSwapAbMMax = 16;
    const bool swap_ab = (m >= 1 and m <= kSwapAbMMax
        and d.stride(-1) == 1 and !is_mixed_fp4 and !c.has_value());

    // Resolve actual granularities, swap if needed
    int ga, gb, gk;
    if (recipe.has_value()) {
        std::tie(ga, gb, gk) = recipe.value();
    } else if (recipe_a.has_value()) {
        ga = std::get<0>(recipe_a.value());
        gb = std::get<0>(recipe_b.value());
        gk = std::get<1>(recipe_a.value());
    } else {
        std::tie(ga, gb, gk) = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    }

    std::optional<std::tuple<int, int, int>> eff_recipe = std::nullopt;
    std::optional<std::tuple<int, int>> eff_recipe_a, eff_recipe_b;
    if (swap_ab) {
        eff_recipe_a = std::make_tuple(gb, gk);
        eff_recipe_b = std::make_tuple(ga, gk);
    } else if (recipe_a.has_value()) {
        eff_recipe_a = recipe_a;
        eff_recipe_b = recipe_b;
    } else {
        eff_recipe = recipe;
    }

    const auto& sf_a_raw = swap_ab ? b.second : a.second;
    const auto& sf_b_raw = swap_ab ? a.second : b.second;
    const int eff_m = swap_ab ? n : m;
    const int eff_n = swap_ab ? m : n;

    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        sf_a_raw, sf_b_raw, eff_m, eff_n, k, eff_recipe,
        eff_recipe_a, eff_recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);

    if (swap_ab) {
        sm120_fp8_fp4_gemm_1d1d(b_data, sfa, a_data, sfb, std::nullopt, d,
                                eff_m, eff_n, k, gran_k_a, gran_k_b,
                                k_major, k_major, compiled_dims,
                                std::nullopt, true);
    } else {
        sm120_fp8_fp4_gemm_1d1d(a_data, sfa, b_data, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                                k_major, k_major, compiled_dims);
    }
}

static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            std::optional<std::tuple<int, int, int>> recipe,
                            std::optional<std::tuple<int, int>> recipe_a,
                            std::optional<std::tuple<int, int>> recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [n , k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Dispatch into different arch implementations
    if (arch_major == 9 or arch_major == 10) {
        // SM90/SM100 share the "transform scaling factors, then dispatch" flow.
        const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
            a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);

        if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
            const int gran_n = recipe.has_value() ? std::get<1>(recipe.value()) : std::get<0>(recipe_b.value());
            if (gran_n == 1) {
                sm90_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
            } else {
                const auto major_sfb = get_major_type_ab(sfb);
                sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, major_sfb, compiled_dims);
            }
        } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
            sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                                    major_a, major_b, compiled_dims);
        } else {
            DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
        }
    } else if (arch_major == 12) {
        fp8_fp4_gemm_nt_sm120(a, b, d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast,
                              major_a, major_b, m, n, k);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void fp8_fp4_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt(a, {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)},
                    {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)}, b,
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void m_grouped_fp8_fp4_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 std::optional<std::tuple<int, int, int>> recipe,
                                                 std::optional<std::tuple<int, int>> recipe_a,
                                                 std::optional<std::tuple<int, int>> recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding,
                                                 const std::optional<int>& expected_m_for_psum_layout) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Pass PSUM layout so SFA packing skips gap rows
    const std::optional<torch::Tensor> psum_sfa_layout = use_psum_layout ? std::make_optional(grouped_layout) : std::nullopt;
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, num_groups, disable_ue8m0_cast,
        psum_sfa_layout);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                num_groups, m, n, k, major_a, major_b, major_sfb,
                                                compiled_dims, use_psum_layout, expected_m_for_psum_layout);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                     num_groups, m, n, k, gran_k_a, gran_k_b, major_a, major_b,
                                                     compiled_dims, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout);
    } else if (arch_major == 12 and sfa.scalar_type() == torch::kInt) {
        const auto b_data = sm120_to_k_major(b.first, major_b, n);
        const bool is_mixed_fp4 = (a.first.scalar_type() != b_data.scalar_type()) and
                                  (a.first.scalar_type() == kPackedFP4 or b_data.scalar_type() == kPackedFP4);
        DG_HOST_ASSERT(!is_mixed_fp4 or k % 128 == 0);
        sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b_data, sfb, d, grouped_layout,
                                                     num_groups, m, n, k, gran_k_a, gran_k_b, major_a, cute::UMMA::Major::K,
                                                     compiled_dims, use_psum_layout, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void m_grouped_fp8_fp4_gemm_nn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 const std::optional<std::tuple<int, int, int>>& recipe,
                                                 const std::optional<std::tuple<int, int>>& recipe_a,
                                                 const std::optional<std::tuple<int, int>>& recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding) {
    m_grouped_fp8_fp4_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)},
                                         d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast,
                                         use_psum_layout, ensure_zero_padding, std::nullopt);
}

static void m_grouped_fp8_fp4_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& expected_m,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             std::optional<std::tuple<int, int>> recipe_a,
                                             std::optional<std::tuple<int, int>> recipe_b,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_groups  , m , k ] = check_grouped_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups_ , n , k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Transform scaling factors
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, num_groups, num_groups, disable_ue8m0_cast);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_m_grouped_fp8_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, major_sfb, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                                 num_groups, m, n, k, expected_m, gran_k_a, gran_k_b,
                                                 major_a, major_b, compiled_dims);
    } else if (arch_major == 12 and sfa.scalar_type() == torch::kInt) {
        sm120_m_grouped_fp8_fp4_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                                 num_groups, m, n, k, expected_m, gran_k_a, gran_k_b,
                                                 major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void k_grouped_fp8_gemm_tn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::optional<std::vector<int>>& ks_cpu,
                                             const torch::Tensor& grouped_layout,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& use_psum_layout) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);

    const int gran_k = std::get<2>(recipe);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    const int k_alignment = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    DG_HOST_ASSERT(k_alignment % 32 == 0);

    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a.first);
    const auto [sum_k__, n_] = get_shape<2>(b.first);

    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, k_alignment, static_cast<int>(a.first.size(0)));
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);
    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, grouped_layout, gran_k, k_alignment,
                                      cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims, use_psum_layout);
    } else if (arch_major == 12) {
        DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());
        // SM120: single transpose [sum_k, M/N] → [M/N, sum_k] with constant stride=sum_k.
        // Kernel uses kKGroupedConstantStride: per-group only replaces addr+dim, not stride.
        const auto a_k = a.first.t().contiguous();
        const auto b_k = b.first.t().contiguous();
        const auto num_sms = device_runtime->get_num_sms();
        const auto tensor_map_buffer = torch::empty({num_sms * 4 * static_cast<int>(sizeof(CUtensorMap))},
                                                    a.first.options().dtype(torch::kByte));
        sm120_k_grouped_fp8_fp4_gemm_1d1d(a_k, sfa, b_k, sfb, c, d, m, n,
                                           ks_cpu.value(), grouped_layout, tensor_map_buffer,
                                           gran_k, gran_k,
                                           cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims,
                                           true, sum_k);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::optional<std::vector<int>>& ks_cpu,
                                             const torch::Tensor& grouped_layout,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& use_psum_layout) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);
    const int gran_k = std::get<2>(recipe);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);

    // Shape checks (K-grouped is FP8 only; FP4 K-grouped is not supported)
    DG_HOST_ASSERT(a.first.scalar_type() != kPackedFP4 and b.first.scalar_type() != kPackedFP4);
    // No psum on FP8 NT
    DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());

    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto sum_mk = a.first.numel();
    const auto sum_nk = b.first.numel();
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, 128);
    DG_HOST_ASSERT(sum_mk == static_cast<int64_t>(sum_k) * m);
    DG_HOST_ASSERT(sum_nk == static_cast<int64_t>(sum_k) * n);
    const auto arch_major = device_runtime->get_arch_major();

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, 128, false);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, 128, false);

    // Allocate tensormap buffer
    // `4` means the double buffering for both A and B operands (2 * 2)
    const auto num_sms = device_runtime->get_num_sms();
    const auto tensor_map_buffer = torch::empty({num_sms * 4 * static_cast<int>(sizeof(CUtensorMap))},
                                                a.first.options().dtype(torch::kByte));

    // Dispatch implementation
    if (arch_major == 9) {
        sm90_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks_cpu.value(), grouped_layout, tensor_map_buffer,
                                     cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims);
    } else if (arch_major == 12) {
        sm120_k_grouped_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n,
                                           ks_cpu.value(), grouped_layout, tensor_map_buffer,
                                           gran_k, gran_k,
                                           cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

#if DG_TENSORMAP_COMPATIBLE
static void bf16_gemm_nt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 12) {
        sm120_bf16_gemm(sm120_to_k_major(a, major_a, m), sm120_to_k_major(b, major_b, n),
                        c, d, m, n, k, cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bf16_gemm_nn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const bool& ensure_zero_padding,
                                              const std::optional<int>& expected_m_for_psum_layout) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto [m, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims,
                                            use_psum_layout, expected_m_for_psum_layout);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims,
                                             use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout);
    } else if (arch_major == 12) {
        sm120_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims,
                                             use_psum_layout, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void m_grouped_bf16_gemm_nn_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const bool& ensure_zero_padding) {
    m_grouped_bf16_gemm_nt_contiguous(a, b.transpose(1, 2),
                                      d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding, std::nullopt);
}

static void m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a, const torch::Tensor& b,
                                          const torch::Tensor& d, const torch::Tensor& masked_m,
                                          const int& expected_m, const std::string& compiled_dims) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto [num_groups, m, k] = get_shape<3>(a);
    const auto [num_groups_, n, k_] = get_shape<3>(b);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_m_grouped_gemm_masked(a, b, d, masked_m,
                                        num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_masked(a, b, d, masked_m,
                                         num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 12) {
        sm120_m_grouped_bf16_gemm_masked(a, b, d, masked_m,
                                         num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_bf16_gemm_tn_contiguous(const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const std::optional<std::vector<int>>& ks_cpu,
                                              const torch::Tensor& grouped_layout,
                                              const std::optional<torch::Tensor>& c,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout) {
    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a);
    const auto [sum_k__, n_] = get_shape<2>(b);

    const auto k_alignment = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    DG_HOST_ASSERT(k_alignment % 32 == 0);
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, k_alignment, static_cast<int>(a.size(0)));
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    // Contiguity checks
    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        // No psum on SM90
        DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());
        sm90_bf16_k_grouped_gemm(a, b, c, d, m, n, ks_cpu.value(), grouped_layout,
                                 cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_k_grouped_gemm(a, b, c, d, m, n, grouped_layout,
                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims, use_psum_layout);
    } else if (arch_major == 12) {
        DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());
        sm120_bf16_k_grouped_gemm(a, b, c, d, m, n, ks_cpu.value(), grouped_layout,
                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

static void cublaslt_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    cublaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value());
}

static void cublaslt_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a, b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b, d, c);
}

}  // namespace deep_gemm::gemm

namespace deep_gemm::torch_registration {

using namespace deep_gemm::torch_utils;

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void fp8_fp4_gemm_nt(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c,
    const c10::optional<std::vector<int64_t>>& recipe,
    const c10::optional<std::vector<int64_t>>& recipe_a,
    const c10::optional<std::vector<int64_t>>& recipe_b,
    const std::string& compiled_dims, const bool& disable_ue8m0_cast) {
    gemm::fp8_fp4_gemm_nt({a, sfa}, {b, sfb}, d, c,
                          list_to_recipe3(recipe), list_to_recipe2(recipe_a), list_to_recipe2(recipe_b),
                          compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_nn(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c,
    const c10::optional<std::vector<int64_t>>& recipe,
    const c10::optional<std::vector<int64_t>>& recipe_a,
    const c10::optional<std::vector<int64_t>>& recipe_b,
    const std::string& compiled_dims, const bool& disable_ue8m0_cast) {
    gemm::fp8_fp4_gemm_nn({a, sfa}, {b, sfb}, d, c,
                          list_to_recipe3(recipe), list_to_recipe2(recipe_a), list_to_recipe2(recipe_b),
                          compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tn(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c,
    const c10::optional<std::vector<int64_t>>& recipe,
    const c10::optional<std::vector<int64_t>>& recipe_a,
    const c10::optional<std::vector<int64_t>>& recipe_b,
    const std::string& compiled_dims, const bool& disable_ue8m0_cast) {
    gemm::fp8_fp4_gemm_tn({a, sfa}, {b, sfb}, d, c,
                          list_to_recipe3(recipe), list_to_recipe2(recipe_a), list_to_recipe2(recipe_b),
                          compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tt(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c,
    const c10::optional<std::vector<int64_t>>& recipe,
    const c10::optional<std::vector<int64_t>>& recipe_a,
    const c10::optional<std::vector<int64_t>>& recipe_b,
    const std::string& compiled_dims, const bool& disable_ue8m0_cast) {
    gemm::fp8_fp4_gemm_tt({a, sfa}, {b, sfb}, d, c,
                          list_to_recipe3(recipe), list_to_recipe2(recipe_a), list_to_recipe2(recipe_b),
                          compiled_dims, disable_ue8m0_cast);
}

static void m_grouped_fp8_fp4_gemm_nt_contiguous(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    const c10::optional<std::vector<int64_t>>& recipe,
    const c10::optional<std::vector<int64_t>>& recipe_a,
    const c10::optional<std::vector<int64_t>>& recipe_b,
    const std::string& compiled_dims, const bool& disable_ue8m0_cast,
    const bool& use_psum_layout, const bool& ensure_zero_padding,
    const c10::optional<int64_t>& expected_m_for_psum_layout) {
    gemm::m_grouped_fp8_fp4_gemm_nt_contiguous(
        {a, sfa}, {b, sfb}, d, grouped_layout,
        list_to_recipe3(recipe), list_to_recipe2(recipe_a), list_to_recipe2(recipe_b),
        compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding,
        expected_m_for_psum_layout.has_value()
            ? std::make_optional(static_cast<int>(expected_m_for_psum_layout.value()))
            : std::nullopt);
}

static void m_grouped_fp8_fp4_gemm_nn_contiguous(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    const c10::optional<std::vector<int64_t>>& recipe,
    const c10::optional<std::vector<int64_t>>& recipe_a,
    const c10::optional<std::vector<int64_t>>& recipe_b,
    const std::string& compiled_dims, const bool& disable_ue8m0_cast,
    const bool& use_psum_layout, const bool& ensure_zero_padding) {
    gemm::m_grouped_fp8_fp4_gemm_nn_contiguous(
        {a, sfa}, {b, sfb}, d, grouped_layout,
        list_to_recipe3(recipe), list_to_recipe2(recipe_a), list_to_recipe2(recipe_b),
        compiled_dims, disable_ue8m0_cast, use_psum_layout, ensure_zero_padding);
}

static void m_grouped_fp8_fp4_gemm_nt_masked(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& masked_m,
    const int64_t& expected_m,
    const c10::optional<std::vector<int64_t>>& recipe,
    const c10::optional<std::vector<int64_t>>& recipe_a,
    const c10::optional<std::vector<int64_t>>& recipe_b,
    const std::string& compiled_dims, const bool& disable_ue8m0_cast) {
    gemm::m_grouped_fp8_fp4_gemm_nt_masked(
        {a, sfa}, {b, sfb}, d, masked_m, static_cast<int>(expected_m),
        list_to_recipe3(recipe), list_to_recipe2(recipe_a), list_to_recipe2(recipe_b),
        compiled_dims, disable_ue8m0_cast);
}

static void k_grouped_fp8_gemm_tn_contiguous(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d,
    const c10::optional<std::vector<int64_t>>& ks_cpu,
    const torch::Tensor& grouped_layout,
    const c10::optional<torch::Tensor>& c,
    const std::vector<int64_t>& recipe,
    const std::string& compiled_dims, const bool& use_psum_layout) {
    gemm::k_grouped_fp8_gemm_tn_contiguous(
        {a, sfa}, {b, sfb}, d,
        list_to_optional_vector_int(ks_cpu), grouped_layout, c,
        list_to_tuple3(recipe), compiled_dims, use_psum_layout);
}

static void k_grouped_fp8_gemm_nt_contiguous(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d,
    const c10::optional<std::vector<int64_t>>& ks_cpu,
    const torch::Tensor& grouped_layout,
    const c10::optional<torch::Tensor>& c,
    const std::vector<int64_t>& recipe,
    const std::string& compiled_dims, const bool& use_psum_layout) {
    gemm::k_grouped_fp8_gemm_nt_contiguous(
        {a, sfa}, {b, sfb}, d,
        list_to_optional_vector_int(ks_cpu), grouped_layout, c,
        list_to_tuple3(recipe), compiled_dims, use_psum_layout);
}
#endif

#if DG_TENSORMAP_COMPATIBLE
static void bf16_gemm_nt(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    gemm::bf16_gemm_nt(a, b, d, c, compiled_dims);
}

static void bf16_gemm_nn(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    gemm::bf16_gemm_nn(a, b, d, c, compiled_dims);
}

static void bf16_gemm_tn(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    gemm::bf16_gemm_tn(a, b, d, c, compiled_dims);
}

static void bf16_gemm_tt(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const c10::optional<torch::Tensor>& c, const std::string& compiled_dims) {
    gemm::bf16_gemm_tt(a, b, d, c, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const torch::Tensor& grouped_layout, const std::string& compiled_dims,
    const bool& use_psum_layout, const bool& ensure_zero_padding,
    const c10::optional<int64_t>& expected_m_for_psum_layout) {
    gemm::m_grouped_bf16_gemm_nt_contiguous(
        a, b, d, grouped_layout, compiled_dims,
        use_psum_layout, ensure_zero_padding,
        expected_m_for_psum_layout.has_value()
            ? std::make_optional(static_cast<int>(expected_m_for_psum_layout.value()))
            : std::nullopt);
}

static void m_grouped_bf16_gemm_nn_contiguous(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const torch::Tensor& grouped_layout, const std::string& compiled_dims,
    const bool& use_psum_layout, const bool& ensure_zero_padding) {
    gemm::m_grouped_bf16_gemm_nn_contiguous(
        a, b, d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding);
}

static void m_grouped_bf16_gemm_nt_masked(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const torch::Tensor& masked_m, const int64_t& expected_m,
    const std::string& compiled_dims) {
    gemm::m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, static_cast<int>(expected_m), compiled_dims);
}

static void k_grouped_bf16_gemm_tn_contiguous(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const c10::optional<std::vector<int64_t>>& ks_cpu,
    const torch::Tensor& grouped_layout,
    const c10::optional<torch::Tensor>& c,
    const std::string& compiled_dims, const bool& use_psum_layout) {
    gemm::k_grouped_bf16_gemm_tn_contiguous(
        a, b, d, list_to_optional_vector_int(ks_cpu),
        grouped_layout, c, compiled_dims, use_psum_layout);
}
#endif

static void cublaslt_gemm_nt(
    const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c) {
    gemm::cublaslt_gemm_nt(a, b, d, c);
}

static void cublaslt_gemm_nn(
    const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c) {
    gemm::cublaslt_gemm_nn(a, b, d, c);
}

static void cublaslt_gemm_tn(
    const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c) {
    gemm::cublaslt_gemm_tn(a, b, d, c);
}

static void cublaslt_gemm_tt(
    const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& d, const c10::optional<torch::Tensor>& c) {
    gemm::cublaslt_gemm_tt(a, b, d, c);
}

}  // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    // GEMM — FP8/FP4
    m.def(
        "fp8_fp4_gemm_nt(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, Tensor? c=None, int[3]? recipe=None, int[2]? recipe_a=None, int[2]? recipe_b=None, str compiled_dims='nk', bool disable_ue8m0_cast=False) -> ()");
    m.def(
        "fp8_fp4_gemm_nn(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, Tensor? c=None, int[3]? recipe=None, int[2]? recipe_a=None, int[2]? recipe_b=None, str compiled_dims='nk', bool disable_ue8m0_cast=False) -> ()");
    m.def(
        "fp8_fp4_gemm_tn(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, Tensor? c=None, int[3]? recipe=None, int[2]? recipe_a=None, int[2]? recipe_b=None, str compiled_dims='mn', bool disable_ue8m0_cast=False) -> ()");
    m.def(
        "fp8_fp4_gemm_tt(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, Tensor? c=None, int[3]? recipe=None, int[2]? recipe_a=None, int[2]? recipe_b=None, str compiled_dims='mn', bool disable_ue8m0_cast=False) -> ()");
    m.def(
        "m_grouped_fp8_fp4_gemm_nt_contiguous(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, Tensor grouped_layout, int[3]? recipe=None, int[2]? recipe_a=None, int[2]? recipe_b=None, str compiled_dims='nk', bool disable_ue8m0_cast=False, bool use_psum_layout=False, bool ensure_zero_padding=True, int? expected_m_for_psum_layout=None) -> ()");
    m.def(
        "m_grouped_fp8_fp4_gemm_nn_contiguous(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, Tensor grouped_layout, int[3]? recipe=None, int[2]? recipe_a=None, int[2]? recipe_b=None, str compiled_dims='nk', bool disable_ue8m0_cast=False, bool use_psum_layout=False, bool ensure_zero_padding=True) -> ()");
    m.def(
        "m_grouped_fp8_fp4_gemm_nt_masked(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, Tensor masked_m, int expected_m, int[3]? recipe=None, int[2]? recipe_a=None, int[2]? recipe_b=None, str compiled_dims='nk', bool disable_ue8m0_cast=False) -> ()");
    m.def(
        "k_grouped_fp8_gemm_tn_contiguous(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, int[]? ks_cpu, Tensor grouped_layout, Tensor? c=None, int[3] recipe, str compiled_dims='mn', bool use_psum_layout=False) -> ()");
    m.def(
        "k_grouped_fp8_gemm_nt_contiguous(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, int[]? ks_cpu, Tensor grouped_layout, Tensor? c=None, int[3] recipe, str compiled_dims='mn', bool use_psum_layout=False) -> ()");
#endif

#if DG_TENSORMAP_COMPATIBLE
    // GEMM — BF16
    m.def(
        "bf16_gemm_nt(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None, str compiled_dims='nk') -> ()");
    m.def(
        "bf16_gemm_nn(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None, str compiled_dims='nk') -> ()");
    m.def(
        "bf16_gemm_tn(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None, str compiled_dims='mn') -> ()");
    m.def(
        "bf16_gemm_tt(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None, str compiled_dims='mn') -> ()");
    m.def(
        "m_grouped_bf16_gemm_nt_contiguous(Tensor a, Tensor b, Tensor(d!) d, Tensor grouped_layout, str compiled_dims='nk', bool use_psum_layout=False, bool ensure_zero_padding=True, int? expected_m_for_psum_layout=None) -> ()");
    m.def(
        "m_grouped_bf16_gemm_nn_contiguous(Tensor a, Tensor b, Tensor(d!) d, Tensor grouped_layout, str compiled_dims='nk', bool use_psum_layout=False, bool ensure_zero_padding=True) -> ()");
    m.def(
        "m_grouped_bf16_gemm_nt_masked(Tensor a, Tensor b, Tensor(d!) d, Tensor masked_m, int expected_m, str compiled_dims='nk') -> ()");
    m.def(
        "k_grouped_bf16_gemm_tn_contiguous(Tensor a, Tensor b, Tensor(d!) d, int[]? ks_cpu, Tensor grouped_layout, Tensor? c=None, str compiled_dims='mn', bool use_psum_layout=False) -> ()");
#endif

    // GEMM — cuBLASLt
    m.def("cublaslt_gemm_nt(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None) -> ()");
    m.def("cublaslt_gemm_nn(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None) -> ()");
    m.def("cublaslt_gemm_tn(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None) -> ()");
    m.def("cublaslt_gemm_tt(Tensor a, Tensor b, Tensor(d!) d, Tensor? c=None) -> ()");
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::torch_registration;

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.impl("fp8_fp4_gemm_nt", TORCH_FN(fp8_fp4_gemm_nt));
    m.impl("fp8_fp4_gemm_nn", TORCH_FN(fp8_fp4_gemm_nn));
    m.impl("fp8_fp4_gemm_tn", TORCH_FN(fp8_fp4_gemm_tn));
    m.impl("fp8_fp4_gemm_tt", TORCH_FN(fp8_fp4_gemm_tt));
    m.impl("m_grouped_fp8_fp4_gemm_nt_contiguous", TORCH_FN(m_grouped_fp8_fp4_gemm_nt_contiguous));
    m.impl("m_grouped_fp8_fp4_gemm_nn_contiguous", TORCH_FN(m_grouped_fp8_fp4_gemm_nn_contiguous));
    m.impl("m_grouped_fp8_fp4_gemm_nt_masked", TORCH_FN(m_grouped_fp8_fp4_gemm_nt_masked));
    m.impl("k_grouped_fp8_gemm_tn_contiguous", TORCH_FN(k_grouped_fp8_gemm_tn_contiguous));
    m.impl("k_grouped_fp8_gemm_nt_contiguous", TORCH_FN(k_grouped_fp8_gemm_nt_contiguous));
#endif

#if DG_TENSORMAP_COMPATIBLE
    m.impl("bf16_gemm_nt", TORCH_FN(bf16_gemm_nt));
    m.impl("bf16_gemm_nn", TORCH_FN(bf16_gemm_nn));
    m.impl("bf16_gemm_tn", TORCH_FN(bf16_gemm_tn));
    m.impl("bf16_gemm_tt", TORCH_FN(bf16_gemm_tt));
    m.impl("m_grouped_bf16_gemm_nt_contiguous", TORCH_FN(m_grouped_bf16_gemm_nt_contiguous));
    m.impl("m_grouped_bf16_gemm_nn_contiguous", TORCH_FN(m_grouped_bf16_gemm_nn_contiguous));
    m.impl("m_grouped_bf16_gemm_nt_masked", TORCH_FN(m_grouped_bf16_gemm_nt_masked));
    m.impl("k_grouped_bf16_gemm_tn_contiguous", TORCH_FN(k_grouped_bf16_gemm_tn_contiguous));
#endif

    m.impl("cublaslt_gemm_nt", TORCH_FN(cublaslt_gemm_nt));
    m.impl("cublaslt_gemm_nn", TORCH_FN(cublaslt_gemm_nn));
    m.impl("cublaslt_gemm_tn", TORCH_FN(cublaslt_gemm_tn));
    m.impl("cublaslt_gemm_tt", TORCH_FN(cublaslt_gemm_tt));
}
