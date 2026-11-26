#pragma once

#include <algorithm>
#include <format>
#include <torch/python.h>

#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/math.hpp"
#include "../jit_kernels/impls/runtime_utils.hpp"

namespace deep_gemm::experimental {

// torch: d = (F.silu(a @ b[:2816].T) * (a @ b[2816:].T)).bfloat16()
static void bf16_gemm_nt_with_swiglu_n5632_k1024(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d) {
    constexpr int kShapeN = 2816, kShapeBN = 5632, kShapeK = 1024;
    constexpr int BLOCK_K = 64, kNumSwizzleBytes = 128;

    DG_HOST_ASSERT(a.dim() == 2 and b.dim() == 2 and d.dim() == 2);
    DG_HOST_ASSERT(a.is_contiguous() and b.is_contiguous() and d.is_contiguous());
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16 and
                   d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);

    const auto m = static_cast<int>(a.size(0));
    DG_HOST_ASSERT(a.size(1) == kShapeK);
    DG_HOST_ASSERT(b.size(0) == kShapeBN and b.size(1) == kShapeK);
    DG_HOST_ASSERT(d.size(0) == m and d.size(1) == kShapeN);

    if (m == 0)
        return;

    const auto num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms == jit->device.get_num_sms() and num_sms % 2 == 0 and
                   "The kernel requires the full SM count");

    int mode_idx = 0, launch_sms = num_sms, grid_blocks = num_sms, num_threads = 256;
    uint32_t num_multicast = 2, block_m = 128, block_n = 128, tail_split_log2 = 0;
    uint32_t a_stages = 0, b_stages = 0;
    bool evict_last_a = true;
    const int num_m_groups = ceil_div(m, 256);

    if (m <= 64 and num_sms >= 118) {
        mode_idx = 0, num_multicast = 1, block_m = 64, block_n = 24;
    } else if (m <= 128 and num_sms >= 142) {
        mode_idx = 1, block_m = 64, block_n = 40;
        launch_sms = std::min(num_sms, 142);
        grid_blocks = launch_sms;
    } else if (m <= 256 and num_sms >= 142) {
        mode_idx = 2, block_n = 40, num_threads = 384;
        launch_sms = std::min(num_sms, 142);
        grid_blocks = launch_sms;
    } else if (m <= 384 and num_sms >= 144) {
        mode_idx = 3, block_m = 64, block_n = 120;
        launch_sms = std::min(num_sms, 144);
        grid_blocks = launch_sms;
    } else if (m > 512 and m <= 640 and num_sms >= 140) {
        mode_idx = 4, block_m = 64, block_n = 104;
        launch_sms = std::min(num_sms, 140);
        grid_blocks = launch_sms;
    } else if (num_m_groups == 3 and num_sms >= 132) {
        mode_idx = 5, block_n = 64;
        a_stages = 8, b_stages = 9;
        launch_sms = std::min(num_sms, 132);
        grid_blocks = launch_sms;
    } else if (num_m_groups == 5) {
        mode_idx = 7, block_n = 64;
        a_stages = 8, b_stages = 10;
    } else if (num_m_groups == 2 or num_m_groups == 4 or num_m_groups == 6 or
               num_m_groups == 8 or num_m_groups == 12 or num_m_groups == 14) {
        mode_idx = num_m_groups == 2 ? 18 : 6, block_n = 80, num_threads = 384;
        if (num_m_groups > 2)
            a_stages = 8, b_stages = 7;
        launch_sms = std::min(num_sms, 144);
        grid_blocks = launch_sms;
    } else if (num_m_groups == 7 and num_sms >= 140) {
        mode_idx = 8, block_n = 96, num_threads = 384;
        launch_sms = std::min(num_sms, 140);
        grid_blocks = launch_sms;
    } else if (num_m_groups == 9 and num_sms >= 148) {
        mode_idx = 9, block_n = 88, num_threads = 384;
    } else {
        evict_last_a = m <= 8192;
        const int num_clusters = num_sms / 2;
        const int num_tiles = num_m_groups * (kShapeN / 128);
        const int rem_tiles = num_tiles % num_clusters;
        const int max_last_wave_tasks = num_tiles >= 8 * num_clusters ? (num_clusters - 6) : num_clusters;
        if (rem_tiles > 0) {
            if (num_tiles < 58 * num_clusters and
                ((rem_tiles << 3) <= num_clusters or
                 (num_tiles >= 19 * num_clusters and (rem_tiles << 1) > max_last_wave_tasks and
                  (max_last_wave_tasks - rem_tiles) % 7 == 0))) {
                tail_split_log2 = 3;
            } else if (num_tiles >= 8 * num_clusters and num_tiles < 58 * num_clusters and
                       (rem_tiles << 2) <= num_clusters) {
                tail_split_log2 = 2;
            } else if (rem_tiles <= (num_tiles >= 8 * num_clusters ? max_last_wave_tasks - 4 : max_last_wave_tasks - 2)) {
                tail_split_log2 = 1;
            }
        }
        mode_idx = 10 + (evict_last_a ? 0 : 4) + static_cast<int>(tail_split_log2);
        const int split_gain = (1 << tail_split_log2) - 1;
        const int num_split_tiles = tail_split_log2 == 0
            ? 0
            : ((rem_tiles << tail_split_log2) > max_last_wave_tasks
                   ? ((max_last_wave_tasks - rem_tiles) / split_gain)
                   : rem_tiles);
        const int num_full_tiles = num_tiles - num_split_tiles;
        const int num_tail_subtiles = num_split_tiles << tail_split_log2;
        grid_blocks = (num_full_tiles + num_tail_subtiles) * 2;
    }

    const int box_m = std::min(m, static_cast<int>(block_m));
    const int b_main_rows = static_cast<int>(block_n);
    const int b_tail_rows = tail_split_log2 == 0 ? b_main_rows : static_cast<int>(block_n >> tail_split_log2);
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, kShapeK,
                                              box_m, BLOCK_K, kShapeK, 1, kNumSwizzleBytes);
    const auto tensor_map_b_main = make_tma_b_desc(cute::UMMA::Major::K, b, kShapeBN, kShapeK,
                                                   b_main_rows, BLOCK_K, kShapeK, 1, kNumSwizzleBytes);
    const auto tensor_map_b_tail = make_tma_b_desc(cute::UMMA::Major::K, b, kShapeBN, kShapeK,
                                                   b_tail_rows, BLOCK_K, kShapeK, 1, kNumSwizzleBytes);

    int cd_main_n = 64;
    if (block_n == 24) {
        cd_main_n = 16;
    } else if (block_n == 40) {
        cd_main_n = 32;
    }
    const int cd_main_swizzle = cd_main_n * 2;
    int cd_tail_n = cd_main_n, cd_tail_swizzle = cd_main_swizzle;
    if (block_n == 24 or block_n == 40) {
        cd_tail_n = 8, cd_tail_swizzle = 16;
    } else if (block_n == 80 or tail_split_log2 == 3) {
        cd_tail_n = 16, cd_tail_swizzle = 32;
    } else if (block_n == 96 or tail_split_log2 == 2) {
        cd_tail_n = 32, cd_tail_swizzle = 64;
    } else if (block_n == 88) {
        cd_tail_n = 24, cd_tail_swizzle = 0;
    } else if (block_n == 104) {
        cd_tail_n = 40, cd_tail_swizzle = 0;
    } else if (block_n == 120) {
        cd_tail_n = 56, cd_tail_swizzle = 0;
    }
    const auto tensor_map_cd = make_tma_cd_desc(d, m, kShapeN, box_m, cd_main_n, kShapeN, 1, cd_main_swizzle);
    const auto tensor_map_cd_tail = make_tma_cd_desc(d, m, kShapeN, box_m, cd_tail_n, kShapeN, 1, cd_tail_swizzle);

    const int b_stage_rows = num_multicast == 1 ? static_cast<int>(2 * block_n) : static_cast<int>(block_n);
    const int cd0_cols = (num_multicast == 2 and block_m == 64) ? (block_n == 40 ? 64 : 128) : cd_main_n;
    const int cd1_cols = (num_multicast == 2 and block_m == 64)
        ? (static_cast<int>(2 * block_n) - cd0_cols)
        : (block_n == 64 ? 0 : (block_n == 128 ? 64 : cd_tail_n));
    const int num_cd_bytes = static_cast<int>(block_m) * (cd0_cols + cd1_cols) * 2;
    const int num_a_stage_bytes = static_cast<int>(block_m) * BLOCK_K * 2;
    const int num_b_stage_bytes = b_stage_rows * BLOCK_K * 2;
    int num_smem_bytes = 0;
    if (a_stages > 0 and b_stages > 0) {
        num_smem_bytes = num_cd_bytes + static_cast<int>(a_stages) * num_a_stage_bytes +
                         static_cast<int>(b_stages) * num_b_stage_bytes + 1024;
    } else {
        const int num_stage_bytes = num_a_stage_bytes + num_b_stage_bytes;
        int num_stages = (232448 - num_cd_bytes - 1024) / num_stage_bytes;
        if (num_multicast == 1 and num_stages > 9)
            num_stages = 9;
        num_smem_bytes = num_cd_bytes + num_stages * num_stage_bytes + 1024;
    }

    const auto kernel = jit->compile("sm100_bf16_gemm_nt_with_swiglu_n5632_k1024", std::format(R"(
#include <deep_gemm/experimental/sm100_bf16_gemm_nt_with_swiglu_n5632_k1024.cuh>

using namespace deep_gemm::experimental;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_nt_with_swiglu_n5632_k1024_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", launch_sms, num_threads, num_multicast, block_m, block_n, tail_split_log2, evict_last_a, a_stages, b_stages));

    jit->launch(
        kernel, {
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(grid_blocks, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
            .cluster_dim = dim3(num_multicast, 1, 1),
            .enable_pdl = true,
        },
        static_cast<uint32_t>(m),
        tensor_map_a, tensor_map_b_main, tensor_map_b_tail,
        tensor_map_cd, tensor_map_cd_tail);
}

} // namespace deep_gemm::experimental
