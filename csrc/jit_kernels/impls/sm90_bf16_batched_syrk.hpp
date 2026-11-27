#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <format>
#include <limits>
#include <optional>
#include <utility>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm90.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

struct SM90BF16BatchedSyrkConfig {
    int block_m;
    int block_n;
    int block_k;
    int cluster_size;
    int swizzle_a_mode;
    int swizzle_d_mode;
    int num_stages;
    int num_d_stages;
    int smem_size;
    int num_tma_threads;
    int num_math_threads;
    int num_sms;
    int group_m;
};

// A singleton batch dimension takes the stride of a densely packed matrix
static int64_t get_tma_batch_stride(const torch::Tensor& t, const int& num_batches,
                                    const int64_t& stride_0, const int64_t& outer_dim) {
    return num_batches == 1 ? stride_0 * outer_dim : t.stride(0);
}

// Whether `cuTensorMapEncodeTiled` accepts the 3D view (see `get_tma_batch_stride`): 16-byte aligned base
// address, and strides that are multiples of 16 bytes and below 2^40 bytes
static bool can_encode_tma_3d(const torch::Tensor& t, const int& num_batches,
                              const int64_t& stride_0, const int64_t& outer_dim) {
    constexpr int64_t kMaxStrideBytes = 1ll << 40;
    const auto elem_size = static_cast<int64_t>(t.element_size());
    const auto is_valid_stride = [&](const int64_t& stride) {
        return stride >= 0 and stride < kMaxStrideBytes / elem_size and (stride * elem_size) % 16 == 0;
    };
    // Bound the singleton batch stride before multiplying to avoid overflow
    if (not is_valid_stride(stride_0))
        return false;
    if (num_batches == 1 and outer_dim > 0 and stride_0 >= kMaxStrideBytes / elem_size / outer_dim)
        return false;
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u and
           is_valid_stride(get_tma_batch_stride(t, num_batches, stride_0, outer_dim));
}

static int64_t get_syrk_num_tiles_per_batch(const int& n, const int& block_m, const int& block_n) {
    const int64_t num_m_blocks = ceil_div(n, block_m);
    if (block_n == block_m) {
        return num_m_blocks * (num_m_blocks + 1) / 2;
    } else {
        const int64_t q = num_m_blocks / 2;
        const int64_t tail = (num_m_blocks & 1) ? (q + 1) : 0;
        return q * (q + 1) + tail;
    }
}

static SM90BF16BatchedSyrkConfig get_sm90_bf16_batched_syrk_config(const int& num_batches,
                                                                   const int& n,
                                                                   const int& k,
                                                                   const at::ScalarType& d_dtype,
                                                                   const int& num_sms) {
    constexpr int block_k = 64;
    constexpr int kNumMaxStages = 12;
    constexpr int smem_barriers = 16 * 8 * 2;
    constexpr int group_m = 8;

    const int elem_size_d = static_cast<int>(c10::elementSize(d_dtype));
    const double l2_bw = std::min(64.0 * num_sms, 8e6 / 1.3e3);
    const double l1_bw = 128.0 * num_sms;

    // Internal test/benchmark controls (not a public API): force the tile shape or the cluster size
    const int force_block_m = deep_jit::get_env<int>("DG_SYRK_BLOCK_M", 0);
    const int force_block_n = deep_jit::get_env<int>("DG_SYRK_BLOCK_N", 0);
    const int force_cluster_size = deep_jit::get_env<int>("DG_SYRK_CLUSTER_SIZE", 0);

    double best_cycles = std::numeric_limits<double>::max();
    SM90BF16BatchedSyrkConfig best_config{};
    bool found = false;

    const std::pair<int, int> tile_candidates[] = {{64, 64}, {128, 128}, {128, 256}};
    for (const auto& [block_m, block_n] : tile_candidates) {
        if (force_block_m != 0 and block_m != force_block_m)
            continue;
        if (force_block_n != 0 and block_n != force_block_n)
            continue;
        // Restrict 128x256 super-tile to BF16 output to avoid register spills with FP32 epilogue
        if (block_n == 256 and d_dtype != torch::kBFloat16)
            continue;

        const int num_m_blocks = ceil_div(n, block_m);
        const int64_t num_tiles = static_cast<int64_t>(num_batches) * get_syrk_num_tiles_per_batch(n, block_m, block_n);
        const int64_t num_waves = ceil_div<int64_t>(num_tiles, num_sms);

        const int swizzle_a_mode = 128;
        const int swizzle_d_mode = d_dtype == torch::kBFloat16 ? 128 : 0;

        // D staging holds `block_m x block_m` sub-tiles
        const int smem_d_per_stage = align(block_m * block_m * elem_size_d, 1024);
        const int smem_ab_per_stage = (block_m + block_n) * block_k * 2;

        // Dual SMEM D staging avoids waiting between normal and mirror TMA stores
        const int min_stages = (block_m * block_n >= 128 * 256) ? 3 : 4;
        const int num_d_stages =
            (2 * smem_d_per_stage + min_stages * smem_ab_per_stage + smem_barriers <= SM90ArchSpec::smem_capacity) ? 2 : 1;
        const int smem_d_total = num_d_stages * smem_d_per_stage;
        const int num_stages = std::min(
            (SM90ArchSpec::smem_capacity - smem_d_total - smem_barriers) / smem_ab_per_stage,
            kNumMaxStages);
        if (num_stages < 3)
            continue;

        const int smem_size = smem_d_total + num_stages * smem_ab_per_stage + smem_barriers;
        const int num_tma_threads = 128;
        const int num_math_threads = block_m == 64 ? 128 : 256;

        for (const int cluster_size : {1, 2}) {
            if (force_cluster_size != 0 and cluster_size != force_cluster_size)
                continue;
            if (cluster_size > 1) {
                if (num_sms % cluster_size != 0 or num_waves <= 1 or num_m_blocks < 2)
                    continue;
            }

            const double wave_eff = static_cast<double>(num_tiles) / static_cast<double>(num_waves * num_sms);
            const double num_bytes_l2_ab = static_cast<double>(k) * (block_m + static_cast<double>(block_n) / cluster_size) * 2.0;
            const double num_bytes_l1_ab = static_cast<double>(k) * (block_m + block_n) * 2.0;
            const double num_bytes_cd = static_cast<double>(block_m) * block_n * elem_size_d * 2.0;

            const double l2_cycles = (num_bytes_l2_ab + num_bytes_cd) * num_tiles / l2_bw;
            const double l1_cycles = (2.0 * num_bytes_l1_ab + num_bytes_cd) * num_tiles / l1_bw;
            // Model L1/L2 cycles as in SM90ArchSpec; penalize single-WG 64x64 for half per-SM WGMMA throughput
            const double wg_penalty = (block_m == 64) ? 1.5 : 1.0;
            const double cycles = std::max(l1_cycles, l2_cycles) * wg_penalty / wave_eff;

            if (cycles < best_cycles) {
                best_cycles = cycles;
                best_config = {
                    .block_m = block_m,
                    .block_n = block_n,
                    .block_k = block_k,
                    .cluster_size = cluster_size,
                    .swizzle_a_mode = swizzle_a_mode,
                    .swizzle_d_mode = swizzle_d_mode,
                    .num_stages = num_stages,
                    .num_d_stages = num_d_stages,
                    .smem_size = smem_size,
                    .num_tma_threads = num_tma_threads,
                    .num_math_threads = num_math_threads,
                    .num_sms = num_sms,
                    .group_m = group_m,
                };
                found = true;
            }
        }
    }

    DG_HOST_ASSERT(found);
    return best_config;
}

static bool can_use_sm90_bf16_batched_syrk(const torch::Tensor& a_3d,
                                           const torch::Tensor& d_3d,
                                           const std::optional<torch::Tensor>& c_3d,
                                           const int& num_batches,
                                           const int& n,
                                           const int& k,
                                           const cute::UMMA::Major& major_a,
                                           const bool& skip_perf_heuristic) {
    if (jit->device.get_arch_major() != 9)
        return false;
    if (a_3d.scalar_type() != torch::kBFloat16)
        return false;
    if (d_3d.scalar_type() != torch::kBFloat16 and d_3d.scalar_type() != torch::kFloat)
        return false;
    if (c_3d.has_value() and
        c_3d->scalar_type() != torch::kBFloat16 and c_3d->scalar_type() != torch::kFloat)
        return false;

    const int64_t a_stride_0 = a_3d.stride(get_non_contiguous_dim(major_a));
    const int64_t d_stride_0 = d_3d.stride(-2);
    const int a_outer_dim = get_inner_outer_dims(major_a, k, n).second;
    if (not can_encode_tma_3d(a_3d, num_batches, a_stride_0, a_outer_dim) or
        not can_encode_tma_3d(d_3d, num_batches, d_stride_0, n))
        return false;
    if (skip_perf_heuristic)
        return true;

    // A conservative empirical cutoff from forced-native vs. forced-cuBLASLt sweeps on H100 (349 shapes): every
    // shape inside it was at least on par with cuBLASLt, at the cost of skipping some native wins (e.g. smaller
    // batches with `n` a multiple of 128). An active `C` or an FP32 `D` makes the epilogue memory-bound, so those
    // need larger problems before the halved WGMMA work pays off
    const int num_sms = runtime->get_num_sms();
    const int64_t num_tiles_128 = static_cast<int64_t>(num_batches) * get_syrk_num_tiles_per_batch(n, 128, 128);
    if (c_3d.has_value())
        return n >= 2048 and k >= 4096 and num_tiles_128 >= 6 * num_sms;
    const double num_dense_flops = 2.0 * num_batches * n * n * k;
    return n >= (d_3d.scalar_type() == torch::kFloat ? 1536 : 768) and k >= 1024 and
           num_tiles_128 >= 3 * num_sms and num_dense_flops >= 4e10;
}

static void sm90_bf16_batched_syrk(const torch::Tensor& a_3d,
                                   const torch::Tensor& d_3d,
                                   const std::optional<torch::Tensor>& c_3d,
                                   const int& num_batches,
                                   const int& n,
                                   const int& k,
                                   const cute::UMMA::Major& major_a,
                                   const float& alpha,
                                   const float& beta) {
    const int num_sms = runtime->get_num_sms();
    const auto config = get_sm90_bf16_batched_syrk_config(num_batches, n, k, d_3d.scalar_type(), num_sms);

    const bool has_c = c_3d.has_value() and beta != 0.0f;
    const at::ScalarType c_dtype = has_c ? c_3d->scalar_type() : d_3d.scalar_type();

    // Build 3D TMA descriptors for A, B (aliasing A with BLOCK_N box), and D
    const int64_t a_stride_0 = a_3d.stride(get_non_contiguous_dim(major_a));
    const auto [a_gmem_0, a_gmem_1] = get_inner_outer_dims(major_a, k, n);
    const int64_t a_stride_1 = get_tma_batch_stride(a_3d, num_batches, a_stride_0, a_gmem_1);
    const auto [a_smem_0, a_smem_1] = get_inner_outer_dims(major_a, config.block_k, config.block_m);
    const auto tensor_map_a = make_tma_3d_desc(a_3d,
                                               a_gmem_0, a_gmem_1, num_batches,
                                               a_smem_0, a_smem_1, 1,
                                               a_stride_0, a_stride_1,
                                               config.swizzle_a_mode);

    const auto [b_smem_0, b_smem_1] = get_inner_outer_dims(major_a, config.block_k, config.block_n);
    const auto tensor_map_b = (config.block_n == config.block_m)
        ? tensor_map_a
        : make_tma_3d_desc(a_3d,
                           a_gmem_0, a_gmem_1, num_batches,
                           b_smem_0, b_smem_1, 1,
                           a_stride_0, a_stride_1,
                           config.swizzle_a_mode);

    const int64_t d_stride_0 = d_3d.stride(-2);
    const int64_t d_stride_1 = get_tma_batch_stride(d_3d, num_batches, d_stride_0, n);
    const auto tensor_map_d = make_tma_3d_desc(d_3d,
                                               n, n, num_batches,
                                               config.block_m, config.block_m, 1,
                                               d_stride_0, d_stride_1,
                                               config.swizzle_d_mode);

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        printf("SM90 BF16 Batched SYRK: B=%d, N=%d, K=%d, major_a=%s, d_dtype=%s, has_c=%d -> "
               "BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d, cluster=%d, stages=%d, d_stages=%d, smem=%d, sms=%d\n",
               num_batches, n, k,
               major_a == cute::UMMA::Major::K ? "K" : "MN",
               c10::toString(d_3d.scalar_type()),
               static_cast<int>(has_c),
               config.block_m, config.block_n, config.block_k, config.cluster_size,
               config.num_stages, config.num_d_stages, config.smem_size, config.num_sms);
    }

    const auto kernel = jit->compile("sm90_bf16_batched_syrk_gemm", std::format(R"(
#include <deep_gemm/impls/sm90_bf16_batched_syrk.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_batched_syrk_gemm_impl<
        {},
        {}, {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
        to_string(major_a),
        0, 0,
        config.block_m, config.block_n, config.block_k,
        config.swizzle_a_mode, config.swizzle_d_mode,
        config.num_stages, config.num_d_stages,
        config.num_tma_threads, config.num_math_threads,
        config.cluster_size,
        config.num_sms,
        to_string(d_3d.scalar_type()),
        to_string(c_dtype),
        has_c ? "true" : "false",
        config.group_m));

    const void* c_data_ptr = has_c ? c_3d->data_ptr() : nullptr;
    const int64_t c_row_stride = has_c ? c_3d->stride(-2) : 0;
    const int64_t c_batch_stride = (has_c and num_batches > 1) ? c_3d->stride(0) : 0;

    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(config.num_sms, 1, 1),
            .block_dim = dim3(config.num_tma_threads + config.num_math_threads, 1, 1),
            .cluster_dim = dim3(config.cluster_size, 1, 1),
        },
        static_cast<uint32_t>(num_batches),
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(k),
        tensor_map_a,
        tensor_map_b,
        tensor_map_d,
        c_data_ptr,
        c_row_stride,
        c_batch_stride,
        alpha,
        has_c ? beta : 0.0f
    );
}

} // namespace deep_gemm
