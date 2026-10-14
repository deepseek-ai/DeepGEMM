#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

// Heuristics for the Ada Lovelace (SM89) FP8 GEMM, which runs on `mma.sync` + `cp.async` instead of
// TMA/WGMMA. Block K is pinned to 128 by the per-128-channel FP8 scaling granularity.
namespace deep_gemm {

struct SM89ArchSpec {
    // NOTES: SM89 allows up to 99 KiB of dynamic shared memory per block
    // (`sharedMemPerBlockOptin == 101376`), but keep a little slack for the launch driver
    static constexpr int smem_capacity = 99 * 1024;
    static constexpr int block_k = 128;

    static bool is_valid_warp_tiling(const int& block_m, const int& block_n, const int& num_threads) {
        if (num_threads % 32 != 0 or num_threads < 64)
            return false;
        // Every thread stages a whole number of 16-byte A/B chunks
        if (block_m * 8 % num_threads != 0 or block_n * 8 % num_threads != 0)
            return false;
        return get_mma_sync_warps_n(num_threads / 32, block_m, block_n) > 0;
    }

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        DG_HOST_ASSERT(desc.gemm_type == GemmType::Normal);
        DG_HOST_ASSERT(desc.a_dtype == torch::kFloat8_e4m3fn and desc.b_dtype == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(desc.major_a == cute::UMMA::Major::K and desc.major_b == cute::UMMA::Major::K);
        DG_HOST_ASSERT(desc.cd_dtype == torch::kFloat or desc.cd_dtype == torch::kBFloat16);

        // Block M candidates: smaller blocks help L2 locality and are needed for narrow problems
        std::vector<int> block_m_candidates;
        for (const int block_m: {128, 64, 32, 16}) {
            if (block_m == 16 and desc.m > 16) continue;
            if (block_m == 32 and desc.m > 32) continue;
            block_m_candidates.push_back(block_m);
        }
        DG_HOST_ASSERT(not block_m_candidates.empty());

        // Block N candidates: multiples of 32 keep the warp tiling (and therefore the `ldmatrix`
        // count and shared-memory bank pattern) sane; 8/16 only for very narrow problems.
        const int step = std::max<int>(8, heuristics_runtime->get_block_n_multiple_of());
        std::vector<int> block_n_candidates;
        const int min_block_n = desc.n <= 32 ? 8 : 32;
        for (int block_n = 128; block_n >= min_block_n; block_n -= 8)
            if (block_n % step == 0 and block_n % (min_block_n == 32 ? 32 : 8) == 0)
                block_n_candidates.push_back(block_n);
        DG_HOST_ASSERT(not block_n_candidates.empty());

        // The 1D2D recipe assigns one B scale per 128-wide N block. Any `BLOCK_N` that divides 128
        // keeps every column of a tile inside a single scale group, which the kernel maps with
        // `n_base / 128`.
        const bool is_1d2d = desc.kernel_type == KernelType::Kernel1D2D;

        std::vector<Layout> candidates;
        for (const int block_m: block_m_candidates) {
            for (const int block_n: block_n_candidates) {
                if (is_1d2d and (block_n > 128 or 128 % block_n != 0))
                    continue;

                for (const int num_threads: {256, 128, 64}) {
                    if (not is_valid_warp_tiling(block_m, block_n, num_threads))
                        continue;

                    const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};
                    const auto storage_config = get_storage_config(desc, layout);
                    if (get_pipeline_config(desc, layout, storage_config).num_stages < 2)
                        continue;

                    candidates.push_back(layout);
                    break;  // Prefer the largest CTA that fits this tile
                }
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        DG_HOST_ASSERT(layout.swap_ab == 0 and layout.cluster_m == 1 and layout.cluster_n == 1);
        // The kernel always uses a 128-byte XOR swizzle for A/B; there is no TMA store for C/D
        return {
            layout.block_m, layout.block_n,
            layout.block_m, layout.block_n,
            128, 128, 0
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout,
                                              const StorageConfig& storage_config) {
        static constexpr int kMaxStages = 8;

        const int elem_size_ab = c10::elementSize(desc.a_dtype);
        const int smem_a_per_stage = layout.block_m * block_k * elem_size_ab;
        const int smem_b_per_stage = layout.block_n * block_k * elem_size_ab;
        // SFB is only staged for the 1D1D recipe; 1D2D reads its single scale straight from GMEM
        const int smem_sf_per_stage = desc.kernel_type == KernelType::Kernel1D2D ?
            layout.block_m * static_cast<int>(sizeof(float)) :
            (layout.block_m + layout.block_n) * static_cast<int>(sizeof(float));

        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sf_per_stage;
        const int num_stages = std::min(smem_capacity / smem_per_stage, kMaxStages);
        return {num_stages * smem_per_stage, num_stages, 1};
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        const int num_threads = get_preferred_num_threads(layout);
        return {
            desc.num_sms, 1, num_threads,
            0, num_threads,
            0, 0
        };
    }

    // Mirrors the enumeration order of `get_layout_candidates`: the largest CTA that supports
    // the tile. The device code re-derives the warp split from `block_m`, `block_n` and this.
    static int get_preferred_num_threads(const Layout& layout) {
        for (const int num_threads: {256, 128, 64})
            if (is_valid_warp_tiling(layout.block_m, layout.block_n, num_threads))
                return num_threads;
        DG_HOST_UNREACHABLE("Unreachable");
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks = ceil_div(desc.get_expected_m(), layout.block_m) *
                                ceil_div(desc.get_expected_n(), layout.block_n) *
                                desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        // Utils
        const int l2_bandwidth_per_cycle = std::min(64. * desc.num_sms, 8e6 / (1.3e3));  // B/cycle
        const int l1_bandwidth_per_cycle = 128 * desc.num_sms;                            // B/cycle
        const int elem_size_ab = c10::elementSize(desc.a_dtype);
        const int elem_size_cd = c10::elementSize(desc.cd_dtype);

        // Data movement per block
        const int64_t expected_k = desc.get_expected_k();
        const int64_t num_bytes_l2_ab = expected_k * (layout.block_m + layout.block_n) * elem_size_ab;
        const int64_t num_bytes_l1_ab = expected_k * (layout.block_m + layout.block_n) * elem_size_ab;
        // Each MMA reads its A/B operands from shared memory through `ldmatrix`
        const int64_t num_bytes_l1_tc = expected_k *
                                        (layout.block_m + layout.block_n) * 2 * elem_size_ab +
                                        static_cast<int64_t>(layout.block_m) * layout.block_n * elem_size_cd;
        const int64_t num_bytes_l1_l2_cd = static_cast<int64_t>(layout.block_m) * layout.block_n *
                                           elem_size_cd * (desc.with_accumulation ? 2 : 1);

        // HBM bandwidth and total compute (Tensor/CUDA cores) are constant across configs
        int64_t num_l2_cycles = (num_bytes_l2_ab + num_bytes_l1_l2_cd) * num_blocks / l2_bandwidth_per_cycle;
        int64_t num_l1_cycles = (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) * num_blocks / l1_bandwidth_per_cycle;
        const float wave_efficiency = static_cast<float>(num_blocks) / (num_waves * desc.num_sms);
        const int64_t num_cycles = std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency;

        return {num_waves, last_wave_util, num_cycles, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        return a.num_cycles < b.num_cycles;
    }
};

}  // namespace deep_gemm