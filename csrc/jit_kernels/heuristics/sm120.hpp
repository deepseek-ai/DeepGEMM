#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "runtime.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM120ArchSpec {
    static constexpr int smem_capacity = 101376;  // 99KB

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        const int elem_size = get_element_size(desc.get_mma_kind());

        // SM120a always uses warp-specialized pipeline: BM=128, BK=128/elem_size
        const int block_m = 128;
        const int block_k = 128 / elem_size;

        // Block N candidates: must be multiples of 8 (mma.sync N=8)
        std::vector<int> block_n_candidates;
        int step = std::lcm(8, heuristics_runtime->get_block_n_multiple_of());
        for (int i = step; i <= 256; i += step) {
            if ((i * get_element_size(desc.get_mma_kind())) % 64 != 0)
                continue;
            block_n_candidates.push_back(i);
        }

        // MN-major B: ldmatrix.trans.x2 loads from SMEM using per-row swizzled addresses.
        // Constrain BLOCK_N to fit in a single swizzle atom so that each K-row's
        // N-contiguous data is within one atom (no multi-atom segmentation).
        const int mn_major_b_max_n = (desc.major_b == cute::UMMA::Major::MN)
            ? get_swizzle_mode(256, 1) / static_cast<int>(c10::elementSize(desc.b_dtype))
            : 256;

        std::vector<Layout> candidates;
        for (int block_n : block_n_candidates) {
            if (block_n > 128 or block_n > mn_major_b_max_n)
                continue;

            const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};
            const auto storage_config = get_storage_config(desc, layout);

            if (storage_config.swizzle_a_mode < 64 or storage_config.swizzle_b_mode < 64)
                continue;

            int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
            if (num_stages < 2)
                continue;

            candidates.push_back(layout);
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static int get_smem_bytes_per_k(const at::ScalarType& dtype, int block_k) {
        return (dtype == kPackedFP4) ? (block_k / 2) : (block_k * static_cast<int>(c10::elementSize(dtype)));
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        const auto load_block_m = layout.block_m;
        const auto load_block_n = layout.block_n;

        // FP4 packed: 0.5 bytes/elem → swizzle based on block_k/2 bytes per row
        const auto smem_k_bytes_a = get_smem_bytes_per_k(desc.a_dtype, layout.block_k);
        const auto swizzle_mode_a = get_swizzle_mode(smem_k_bytes_a, 1);
        // B swizzle: row stride depends on major — K-major rows span K, MN-major rows span N
        const auto smem_row_bytes_b = (desc.major_b == cute::UMMA::Major::K)
            ? get_smem_bytes_per_k(desc.b_dtype, layout.block_k)
            : layout.block_n * static_cast<int>(c10::elementSize(desc.b_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(smem_row_bytes_b, 1);

        const auto swizzle_mode_cd = (c10::elementSize(desc.cd_dtype) <= 2) ? 128 : 0;

        return {
            load_block_m, load_block_n,
            0, 0,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static int get_smem_d_size(const GemmDesc& desc, const Layout& layout) {
        const int cd_size = c10::elementSize(desc.cd_dtype);
        const int swizzle_cd = 128;
        if (cd_size <= 2 and layout.block_n * cd_size >= swizzle_cd
            and (layout.block_n * cd_size) % swizzle_cd == 0)
            return (layout.block_n * cd_size / swizzle_cd) * swizzle_cd * layout.block_m;
        return 0;
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 16;

        const int smem_barriers = kNumMaxStages * 8 * 2;
        const int smem_a_per_stage = storage_config.load_block_m * get_smem_bytes_per_k(desc.a_dtype, layout.block_k);
        const int smem_b_per_stage = storage_config.load_block_n * get_smem_bytes_per_k(desc.b_dtype, layout.block_k);

        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            smem_sfa_per_stage = align(layout.block_m * static_cast<int>(sizeof(int32_t)), 128);
            smem_sfb_per_stage = align(layout.block_n * static_cast<int>(sizeof(int32_t)), 128);
        }

        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 2 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        const int smem_d = get_smem_d_size(desc, layout);

        const int smem_extra = smem_barriers + smem_tensormap + smem_d;
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        const int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages
        };
    }

    static constexpr int mma_m = 16;

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        // Warp-specialized: 1 load warp group (128 threads) + 2 MMA warp groups (256 threads)
        return {
            desc.num_sms,
            1,
            384,              // total threads
            128, 256,         // kNumTMAThreads = 128, kNumMathThreads = 256
            0, 0
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        const int64_t expected_k = desc.get_expected_k();
        int64_t flops_per_block = 2LL * layout.block_m * layout.block_n * expected_k;

        // Empirical warp-spec MMA efficiency model.
        // A reuse = BN/8 (each A fragment reused across N-tiles).
        const double a_reuse = static_cast<double>(layout.block_n) / 8.0;
        double mma_efficiency = 0.69 + 0.07 * std::min(1.0, (a_reuse - 8.0) / 8.0);

        const double peak_flops_per_ns = (desc.a_dtype == at::kBFloat16) ? 380000.0 : 762000.0;
        double block_ns = flops_per_block / (peak_flops_per_ns * mma_efficiency);

        float wave_efficiency = static_cast<float>(num_blocks) / (num_waves * desc.num_sms);
        int64_t num_cycles = static_cast<int64_t>(block_ns * num_blocks / wave_efficiency);

        return {num_waves, last_wave_util, num_cycles, layout};
    }

    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        if (a.num_waves != b.num_waves and (a.num_waves == 1 or b.num_waves == 1))
            return a.num_waves < b.num_waves;

        if (a.num_cycles != b.num_cycles)
            return a.num_cycles < b.num_cycles;

        // Tie-break: prefer larger tile for better reuse
        if (a.layout.block_n != b.layout.block_n)
            return a.layout.block_n > b.layout.block_n;
        return false;
    }
};

} // namespace deep_gemm
