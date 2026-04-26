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
        const int block_k = 128 / get_element_size(desc.get_mma_kind());

        // Block M candidates: each math warp handles 16 rows via mma.sync m16n8k32
        std::vector<int> block_m_candidates;
        if (desc.gemm_type == GemmType::Normal or
            desc.gemm_type == GemmType::Batched or
            desc.gemm_type == GemmType::KGroupedContiguous) {
            block_m_candidates = {64, 128};
            if (desc.m <= 32) block_m_candidates.push_back(32);
            if (desc.m <= 16) block_m_candidates.push_back(16);
        } else if (desc.gemm_type == GemmType::MGroupedContiguous or
                   desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
            block_m_candidates = std::vector{heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
        } else if (desc.gemm_type == GemmType::MGroupedMasked) {
            block_m_candidates = {64, 128};
        }

        // Block N candidates: must be multiples of 8 (mma.sync N=8)
        std::vector<int> block_n_candidates;
        int step = std::lcm(8, heuristics_runtime->get_block_n_multiple_of());
        for (int i = step; i <= 256; i += step) {
            // Ensure large swizzle for K-major operands (avoid 32B which is slow)
            if ((i * get_element_size(desc.get_mma_kind())) % 64 != 0)
                continue;
            block_n_candidates.push_back(i);
        }

        // SM120a: cluster always 1x1 (TMA multicast unusable), no swap_ab
        std::vector<Layout> candidates;
        for (int block_m : block_m_candidates) {
            for (int block_n : block_n_candidates) {
                // At least one dim <= 128 to keep register pressure reasonable
                if (block_m > 128 and block_n > 128)
                    continue;

                const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};
                const auto storage_config = get_storage_config(desc, layout);

                // Require 128B swizzle for good ldmatrix throughput
                if (storage_config.swizzle_a_mode < 64 or storage_config.swizzle_b_mode < 64)
                    continue;

                // Need at least 3 pipeline stages to hide TMA latency
                int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
                if (num_stages < 3)
                    continue;

                candidates.push_back(layout);
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        // No multicast split, no TMA store (register epilogue)
        const auto load_block_m = layout.block_m;
        const auto load_block_n = layout.block_n;

        // Decide swizzle by inner dim (always K-major for SM120)
        const auto swizzle_mode_a = get_swizzle_mode(layout.block_k, c10::elementSize(desc.a_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(layout.block_k, c10::elementSize(desc.b_dtype));

        return {
            load_block_m, load_block_n,
            0, 0,  // no store blocks (register-based epilogue)
            swizzle_mode_a, swizzle_mode_b, 0  // no swizzle_cd (no TMA store)
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 16;

        const int smem_barriers = kNumMaxStages * 8 * 2;  // full + empty

        // A/B tiles per stage
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);

        // SF tiles per stage: packed UE8M0 (4 bytes per int32 element)
        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            smem_sfa_per_stage = align(layout.block_m * static_cast<int>(sizeof(int32_t)), 128);
            smem_sfb_per_stage = align(layout.block_n * static_cast<int>(sizeof(int32_t)), 128);
        }

        // Extra tensormap for K-grouped
        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 2 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        const int smem_extra = smem_barriers + smem_tensormap;
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
        const int num_tma_threads = 128;
        // Each math warp handles MMA_M=16 rows, so num_math_warps = block_m / 16
        const int num_math_warps = layout.block_m / mma_m;
        const int num_math_threads = num_math_warps * 32;
        return {
            desc.num_sms,
            1,  // cluster size always 1
            num_tma_threads + num_math_threads,
            num_tma_threads, num_math_threads,
            0, 0  // no separate epilogue threads
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

        // Perf model: L1/L2/HBM cycle estimation
        const int elem_size_ab = c10::elementSize(desc.a_dtype);
        const int64_t expected_k = desc.get_expected_k();

        // HBM data per block (no multicast)
        int64_t num_bytes_hbm = expected_k * (layout.block_m + layout.block_n) * elem_size_ab;
        // Compute cycles per block (FLOPs / peak)
        int64_t flops_per_block = 2LL * layout.block_m * layout.block_n * expected_k;

        // Simple roofline: max(compute_bound, memory_bound)
        // HBM BW ~1500 GB/s, FP8 peak ~814 TFLOPS
        constexpr double hbm_bw_bytes_per_ns = 1500.0;  // GB/s = bytes/ns
        constexpr double fp8_peak_flops_per_ns = 814000.0;  // TFLOPS = 1e12 flops/s = 1e3 flops/ns
        double compute_ns = flops_per_block / fp8_peak_flops_per_ns;
        double memory_ns = num_bytes_hbm / hbm_bw_bytes_per_ns;
        double block_ns = std::max(compute_ns, memory_ns);

        float wave_efficiency = static_cast<float>(num_blocks) / (num_waves * desc.num_sms);
        int64_t num_cycles = static_cast<int64_t>(block_ns * num_blocks / wave_efficiency);

        return {num_waves, last_wave_util, num_cycles, layout};
    }

    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        // Fewer waves is better
        if (a.num_waves != b.num_waves and (a.num_waves == 1 or b.num_waves == 1))
            return a.num_waves < b.num_waves;

        // Lower estimated cycles is better
        if (a.num_cycles != b.num_cycles)
            return a.num_cycles < b.num_cycles;

        // More pipeline stages (smaller tiles) is better for latency hiding
        return a.layout.block_m + a.layout.block_n < b.layout.block_m + b.layout.block_n;
    }
};

} // namespace deep_gemm
