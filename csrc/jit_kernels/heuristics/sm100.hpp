#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "runtime.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM100ArchSpec {
    static constexpr int smem_capacity = 232448;

    static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(
        const int& block_m, const int& block_n, const MmaKind& mma_kind) {
        constexpr int num_utccp_aligned_elems = 128;
        switch (mma_kind) {
            case MmaKind::BF16: return {0, 0};
            case MmaKind::MXFP8FP4: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            default: DG_HOST_UNREACHABLE("Unknown dtype");
        }
    }

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // Block K is always in a fixed manner
        const int block_k = 128 / get_element_size(desc.get_mma_kind());

        // Always enable swap A/B (and multicasting if possible) for m-grouped GEMMs
        if (desc.gemm_type == GemmType::MGroupedContiguous or
            desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout or
            desc.gemm_type == GemmType::MGroupedMasked) {
            const bool swap_ab = true;
            const auto block_n = 128;
            const auto block_m = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
            const auto cluster_m = 1;
            const auto cluster_n = ceil_div(desc.n, block_n) % 2 == 0 and desc.num_sms % 2 == 0 ? 2 : 1;
            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};
            std::vector<Layout> candidates = {layout};
            return candidates;
        }

        // Enumerate all candidates
        std::vector<Layout> candidates;
        for (int swap_ab = 0; swap_ab < 2; ++ swap_ab) {
            // Block M/N candidates
            std::vector<int> block_m_candidates;
            std::vector<int> block_n_candidates;            
            if (swap_ab) {
                int step = std::lcm(16, heuristics_runtime->get_block_m_multiple_of());
                int end = 256;
                for (int i = step; i <= end; i += step)
                    block_m_candidates.push_back(i);

                // TODO: consider other block N
                block_n_candidates = {128};
            } else {
                // NOTES: smaller block M can avoid TMA L2 OOB bound
                // TODO: consider block M = 256
                if (desc.m <= 32) block_m_candidates = {32};
                else if (desc.m <= 64) block_m_candidates = {64};
                else block_m_candidates = {128};

                // Small block size for small shape
                if (16 % heuristics_runtime->get_block_n_multiple_of() == 0)
                    block_n_candidates.push_back(16);
                int step = std::lcm(32, heuristics_runtime->get_block_n_multiple_of());
                // For small K, fewer store blocks improve store/compute overlap and reduce epilogue bottleneck
                int end = desc.k <= 256 ? 128 : 256;
                for (int i = step; i <= end; i += step)
                    block_n_candidates.push_back(i);
            }

            for (int cluster_m = 1; cluster_m <= 2; ++ cluster_m) {
                // After swapping, layout A/D can only do on cluster N
                if (swap_ab == 1 and cluster_m > 1)
                    continue;

                for (int cluster_n = 1; cluster_n <= 2; ++ cluster_n) {
                    // We only support cluster 2
                    if (cluster_m * cluster_n > 2)
                        continue;

                    // Only support layout A/D
                    if (swap_ab == 0 and cluster_n > 1)
                        continue;

                    // SM count must be divisible
                    if (desc.num_sms % (cluster_m * cluster_n) != 0)
                        continue;

                    for (int block_m: block_m_candidates) {
                        // Ensure large swizzle sizes (32B swizzle yields poor performance)
                        const auto swizzle_a_requirement = desc.a_dtype == kPackedFP4 ? 128 : 64;
                        // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                        const auto load_block_m_requirement = desc.major_a == cute::UMMA::Major::MN ? swizzle_a_requirement : 8;
                        if ((block_m / cluster_n) % load_block_m_requirement != 0)
                            continue;

                        // Shape must be divisible for multicast
                        if (ceil_div(desc.m, block_m) % cluster_m != 0)
                            continue;

                        for (int block_n: block_n_candidates) {
                            // Ensure large swizzle sizes (32B swizzle yields poor performance)
                            const auto swizzle_b_requirement = desc.b_dtype == kPackedFP4 ? 128 : 64;
                            // Enforce swizzle alignment for MN major; otherwise check base MMA shape
                            const auto load_block_n_requirement = desc.major_b == cute::UMMA::Major::MN ? swizzle_b_requirement : 8;
                            if ((block_n / cluster_m) % load_block_n_requirement != 0)
                                continue;

                            // Shape must be divisible for multicast
                            if (ceil_div(desc.n, block_n) % cluster_n != 0)
                                continue;

                            // SwapAB requires block N is layout A/D' UMMA M
                            constexpr int layout_ad_m = 128;
                            if (swap_ab and block_n != layout_ad_m)
                                continue;

                            // Check tensor memory capacity
                            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, desc.get_mma_kind());
                            const auto tmem_sf_cols = desc.get_mma_kind() == MmaKind::MXFP8FP4 ? sf_block_m / 32 + sf_block_n / 32 : 0;
                            const auto umma_n = swap_ab ? block_m : block_n;
                            if (2 * umma_n + tmem_sf_cols > 512)
                                continue;

                            const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};

                            // When neither A nor B is MN major, 128B swizzle is always feasible
                            if (desc.major_a == cute::UMMA::Major::K or desc.major_b == cute::UMMA::Major::K) {
                                const auto storage_config = get_storage_config(desc, layout);
                                if (storage_config.swizzle_a_mode != 128 or storage_config.swizzle_b_mode != 128)
                                    continue;
                            }

                            candidates.push_back(layout);
                        }
                    }
                }
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        constexpr int layout_ad_m = 128;
        constexpr int umma_step_n = 16;

        // Load/store block sizes (w/o consideration of swizzling atoms, w/ consideration of loop atoms)
        const auto load_block_m = layout.block_m / layout.cluster_n;
        const auto load_block_n = layout.block_n / layout.cluster_m;
        const auto store_block_m = layout.swap_ab ? umma_step_n : std::min(layout_ad_m, layout.block_m);
        const auto store_block_n = layout.block_n;

        // Decide swizzling by the inner dim
        // TODO: support FP4 sub-byte
        const auto swizzle_mode_a = get_swizzle_mode(
            desc.major_a == cute::UMMA::Major::K ? layout.block_k : load_block_m, c10::elementSize(desc.a_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(
            desc.major_b == cute::UMMA::Major::K ? layout.block_k : load_block_n, c10::elementSize(desc.b_dtype));
        const auto swizzle_mode_cd = get_swizzle_mode(
            store_block_n, c10::elementSize(desc.cd_dtype));

        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 32;

        // C/D for TMA stores
        const int smem_cd = layout.swap_ab ? storage_config.store_block_m * storage_config.store_block_n * c10::elementSize(desc.cd_dtype) * 2
                                           : storage_config.store_block_m * storage_config.swizzle_cd_mode * 2;

        // TODO: remove SF barriers for BF16 GEMMs
        // TMA full/empty barriers, with-SF full barriers, tensor memory full/empty barriers
        // NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
        // NOTES: the last barrier is for tensor core utilization control
        const int smem_barriers = kNumMaxStages * 8 * 3 + 2 * 8 * 2 + 8;

        // Tensor memory pointer
        const int smem_tmem_ptr = 4;

        // Calculate A/B per stages
        // TODO: consider FP4
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);

        // Calculate SF A/B per stages
        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(
                layout.block_m, layout.block_n, desc.get_mma_kind());
            smem_sfa_per_stage = sf_block_m * 4;
            smem_sfb_per_stage = sf_block_n * 4;
        }

        // Calculate stages
        int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
        int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        return {
            desc.num_sms,
            layout.get_cluster_size(),
            256,
            32, 128, 128, 128
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
        // TODO: calculate expected cycles
        return {num_waves, last_wave_util, 0, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        // Single wave is always better
        if ((a.num_waves == 1 or b.num_waves == 1) and a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Doing multicast is better
        if (a.layout.get_cluster_size() != b.layout.get_cluster_size())
            return a.layout.get_cluster_size() > b.layout.get_cluster_size();

        // Smaller number of waves is better
        if (a.num_waves != b.num_waves)
            return a.num_waves < b.num_waves;

        // Larger last wave utilization is better
        if (a.last_wave_util != b.last_wave_util)
            return a.last_wave_util > b.last_wave_util;

        // More stages is better
        // Same block M, smaller block N is better
        // Same block N, smaller block M is better
        if (a.layout.block_m + a.layout.block_n != b.layout.block_m + b.layout.block_n)
            return a.layout.block_m + a.layout.block_n < b.layout.block_m + b.layout.block_n;

        // Less shared memory C/D, more stages is better
        return a.layout.block_m * a.layout.block_n < b.layout.block_m * b.layout.block_n;
    }
};

} // namespace deep_gemm
