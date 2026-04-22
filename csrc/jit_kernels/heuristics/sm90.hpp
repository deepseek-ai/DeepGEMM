#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM90ArchSpec {
    static constexpr int smem_capacity = 232448;

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        // Block M candidates
        std::vector<int> block_m_candidates;
        if (desc.gemm_type == GemmType::Normal or
            desc.gemm_type == GemmType::Batched or
            desc.gemm_type == GemmType::KGroupedContiguous) {
            // TODO: check 256's performance
            block_m_candidates = {64, 128};
            // NOTES: smaller block M can avoid TMA L2 OOB bound
            if (desc.m <= 16) block_m_candidates.push_back(16);
            if (desc.m <= 32) block_m_candidates.push_back(32);

            // BF16 output GEMM supports 256
            if (desc.cd_dtype != torch::kFloat)
                block_m_candidates.push_back(256);
        } else if (desc.gemm_type == GemmType::MGroupedContiguous or
                   desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
            block_m_candidates = std::vector{heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
        } else if (desc.gemm_type == GemmType::MGroupedMasked) {
            block_m_candidates = {64, 128};
        }

        // Block N candidates
        std::vector<int> block_n_candidates;
        int step = std::lcm(16, heuristics_runtime->get_block_n_multiple_of());
        int start = step;
        // Avoid bank conflicts for 1D1D kernel FP32 output
        if (desc.kernel_type == KernelType::Kernel1D1D and desc.cd_dtype == torch::kFloat) {
            DG_HOST_ASSERT(desc.major_a == cute::UMMA::Major::K);
            DG_HOST_ASSERT(desc.major_b == cute::UMMA::Major::K);
            start = 24;
            block_n_candidates.push_back(16);
        }
        // Register spills
        int end = 256;
        if (desc.kernel_type == KernelType::Kernel1D2D)
            end = 192;
        if (desc.kernel_type == KernelType::Kernel1D1D)
            end = 160;
        // Enumerate
        for (int i = start; i <= end; i += step)
            block_n_candidates.push_back(i);

        // Block K is always in a fixed manner
        const int block_k = 128 / get_element_size(desc.get_mma_kind());

        // Disable multicast for performance
        const bool disable_multicast =
            // The number of k-groups is large (a heuristic)
            (desc.gemm_type == GemmType::KGroupedContiguous and desc.num_groups > 4) or
            // Not supported
            (desc.gemm_type == GemmType::Batched);

        // Enumerate all candidates
        std::vector<Layout> candidates;
        for (int cluster_m = 1; cluster_m <= (disable_multicast ? 1 : 2); ++ cluster_m) {
            for (int cluster_n = 1; cluster_n <= (disable_multicast ? 1 : 2); ++ cluster_n) {
                // We only support cluster 2
                if (cluster_m * cluster_n > 2)
                    continue;

                // SM count must be divisible
                if (desc.num_sms % (cluster_m * cluster_n) != 0)
                    continue;

                for (int block_m: block_m_candidates) {
                    for (int block_n: block_n_candidates) {
                        // 1D2D kernel unroll requirement
                        if (desc.kernel_type == KernelType::Kernel1D2D and block_n > block_k and (block_n % (block_n - block_k) != 0 and block_k % (block_n - block_k) != 0))
                            continue;

                        // Multicast legality for masked layout
                        // TODO: add some comments about it
                        if ((desc.gemm_type == GemmType::MGroupedMasked or desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) and
                            ceil_div(desc.n, block_n) % (cluster_m * cluster_n) != 0)
                            continue;

                        // The block sizes cannot be too large (for enough registers), so at least one dim less than 128
                        if (block_m > 128 and block_n > 128)
                            continue;

                        // Calculate swizzling
                        const auto layout = Layout{0, block_m, block_n, block_k, cluster_m, cluster_n};
                        const auto storage_config = get_storage_config(desc, layout);

                        // Make sure swizzling is large enough (32B's performance is low)
                        if (storage_config.swizzle_a_mode % 64 != 0 or storage_config.swizzle_b_mode % 64 != 0)
                            continue;
                        
                        // To hide TMA latency, the stage count should be at least 3; for small matrices, at least 4
                        int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
                        if (num_stages < 3 or (block_m * block_n < 128 * 192 and num_stages < 4))
                            continue;

                        candidates.push_back(layout);
                    }
                }
            }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        constexpr int wgmma_m = 64;

        // Load/store block sizes (w/o consideration of swizzling atoms, w/ consideration of loop atoms)
        // TODO: support swap AB
        DG_HOST_ASSERT(layout.swap_ab == 0);
        const auto load_block_m = layout.block_m;
        const auto load_block_n = layout.block_n;
        // 1D1D kernel will do single warp-group stores
        const auto store_block_m = desc.kernel_type == KernelType::Kernel1D1D ? wgmma_m : layout.block_m;
        const auto store_block_n = layout.block_n;

        // Decide swizzling by the inner dim
        const auto swizzle_mode_a = get_swizzle_mode(
            desc.major_a == cute::UMMA::Major::K ? layout.block_k : load_block_m, c10::elementSize(desc.a_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(
            desc.major_b == cute::UMMA::Major::K ? layout.block_k : load_block_n, c10::elementSize(desc.b_dtype));
        // We only enable swizzling for non-FP32 outputs
        const auto swizzle_mode_cd = desc.cd_dtype != torch::kFloat ?
            get_swizzle_mode(store_block_n, c10::elementSize(desc.cd_dtype)) : 0;

        return {
            load_block_m, load_block_n,
            store_block_m, store_block_n,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 16;

        // TODO: consider swap AB
        // C/D for TMA stores
        // NOTES: 1024 is for TMA swizzling alignment requirement
        const int smem_cd =
            align(layout.block_m * layout.block_n * static_cast<int>(c10::elementSize(desc.cd_dtype)), 1024);
        const int smem_barriers = kNumMaxStages * 8 * 2;

        // Calculate A/B per stages
        const int smem_a_per_stage = storage_config.load_block_m * layout.block_k * c10::elementSize(desc.a_dtype);
        const int smem_b_per_stage = storage_config.load_block_n * layout.block_k * c10::elementSize(desc.b_dtype);

        // Calculate SF A/B per stages
        const int smem_sfa_per_stage = desc.kernel_type == KernelType::KernelNoSF ?
            0 : align(layout.block_m * static_cast<int>(sizeof(float)), 128);
        const int smem_sfb_per_stage = desc.kernel_type != KernelType::Kernel1D1D ?
            0 : align(layout.block_n * static_cast<int>(sizeof(float)), 128);

        // Extra SFB sizes for 1D2D kernels
        const int use_uniform_sfb = layout.block_k % layout.block_n == 0 ? 1 : 2;
        const int smem_extra_sfb = desc.kernel_type != KernelType::Kernel1D2D ?
            0 : align<int>(ceil_div(desc.k, layout.block_k) * static_cast<int>(sizeof(float)) * use_uniform_sfb, 8);

        // Extra tensormap for 1D1D kernels
        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 4 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        // Calculate stages
        const int smem_extra = smem_cd + smem_barriers + smem_extra_sfb + smem_tensormap;
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        const int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages
        };
    }

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        const int num_tma_threads = 128;
        const int num_math_threads = layout.block_m <= 64 ? 128 : 256;
        return {
            desc.num_sms,
            layout.get_cluster_size(),
            num_tma_threads + num_math_threads,
            num_tma_threads, num_math_threads,
            0, 0 // Meaningless for SM90
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

        // Utils
        const int l2_bandwidth_per_cycle = std::min(64. * desc.num_sms, 8e6 / (1.3e3)); // B/cycle
        const int l1_bandwidth_per_cycle = 128 * desc.num_sms; // B/cycle
        const int wgmma_m = 64;
        const int elem_size_ab = c10::elementSize(desc.a_dtype);
        const int elem_size_cd = c10::elementSize(desc.cd_dtype);
        DG_HOST_ASSERT(desc.a_dtype == desc.b_dtype);

        // Data movement per block
        int64_t expected_k = desc.get_expected_k();
        int64_t num_bytes_l2_ab = expected_k * (layout.block_m / layout.cluster_n + layout.block_n / layout.cluster_m) * elem_size_ab;
        int64_t num_bytes_l1_ab = expected_k * (layout.block_m + layout.block_n) * elem_size_ab;
        int64_t num_bytes_l1_tc = expected_k * (std::max(wgmma_m, layout.block_m) + layout.block_n) * elem_size_ab
                                  + layout.block_m * layout.block_n * elem_size_cd;
        int64_t num_bytes_l1_l2_cd = layout.block_m * layout.block_n * elem_size_cd * (desc.with_accumulation ? 2 : 1);

        // HBM bandwidth and total compute (Tensor/CUDA cores) are constant across configs
        // We only model L1/L2 cycles as they are the primary variables between configs
        int64_t num_l2_cycles = (num_bytes_l2_ab + num_bytes_l1_l2_cd) * num_blocks / l2_bandwidth_per_cycle;
        int64_t num_l1_cycles = (num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) * num_blocks / l1_bandwidth_per_cycle;
        float wave_efficiency = static_cast<float>(num_blocks) / (num_waves * desc.num_sms);
        int64_t num_cycles = std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency;

        // Disable multicasting if only one wave exists
        if (layout.cluster_n * layout.cluster_m > 1 and num_waves <= 1)
            num_cycles = std::numeric_limits<int64_t>::max();

        return {num_waves, last_wave_util, num_cycles, layout};
    }

    // A regular comparator
    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        return a.num_cycles < b.num_cycles;
    }
};

} // namespace deep_gemm
