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

    static constexpr int kMinBlockM = 64;   // kMWarps(4) * MMA_M(16), both FP8 and BF16 with kNWarps=2

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        const int elem_size = get_element_size(desc.get_mma_kind());
        const int runtime_align = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
        const int expected_m = desc.get_expected_m();

        // BLOCK_M candidates: {64, 128} valid for both FP8 and BF16 (kNWarps=2, kMWarps=4).
        const int n_for_tile = desc.get_expected_n() > 0 ? desc.get_expected_n() : desc.n;
        const bool is_small_n = (n_for_tile > 0 and n_for_tile <= 32);

        std::vector<int> block_m_candidates;
        if (runtime_align <= kMinBlockM)
            block_m_candidates.push_back(64);
        else {
            if (is_small_n)
                block_m_candidates.push_back(64);
            block_m_candidates.push_back(128);
            if (expected_m > 0 and expected_m <= kMinBlockM)
                block_m_candidates.push_back(64);
            // For Normal GEMM with very few MN blocks: BM=64 creates more blocks
            // for split-K to fill SMs.
            const int eff_m = expected_m > 0 ? expected_m : desc.m;
            const int approx_mn_blocks_128 = ceil_div(eff_m, 128) * ceil_div(n_for_tile, 64);
            if (desc.gemm_type == GemmType::Normal
                and eff_m <= 128 and approx_mn_blocks_128 < desc.num_sms / 8)
                block_m_candidates.push_back(64);
        }
        if (block_m_candidates.empty())
            block_m_candidates.push_back(128);

        // Block K candidates: BK=64 enables 4 pipeline stages (better TMA hiding),
        // but only beneficial for large M (>= 2048) and non-mixed dtypes.
        const bool is_mixed = (desc.a_dtype != desc.b_dtype);
        std::vector<int> block_k_candidates;
        if (!is_mixed and expected_m >= 2048)
            block_k_candidates.push_back(64 / elem_size);
        block_k_candidates.push_back(128 / elem_size);

        // Block N candidates
        std::vector<int> block_n_candidates;
        if (is_small_n) {
            // BN=16 always valid: K-major B has N as TMA outer dim, no minimum.
            // Kernel epilogue bounds-checks shape_n for partial N tiles.
            block_n_candidates.push_back(16);
            if (n_for_tile > 16)
                block_n_candidates.push_back(32);
        } else {
            int step = std::lcm(8, heuristics_runtime->get_block_n_multiple_of());
            for (int i = step; i <= 256; i += step) {
                if ((i * get_element_size(desc.get_mma_kind())) % 64 != 0)
                    continue;
                block_n_candidates.push_back(i);
            }
        }

        const int mn_major_b_max_n = 128;

        std::vector<Layout> candidates;
        for (int block_m : block_m_candidates) {
        // For grouped contiguous GEMM, BLOCK_M must divide runtime_align
        // to prevent tiles from straddling expert boundaries.
        if (is_m_grouped_contiguous(desc.gemm_type) and block_m > runtime_align)
            continue;
        for (int block_k : block_k_candidates) {
        for (int block_n : block_n_candidates) {
            if (!is_small_n and (block_n > 128 or block_n > mn_major_b_max_n))
                continue;

            const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};
            const auto storage_config = get_storage_config(desc, layout);

            if (!is_small_n and (storage_config.swizzle_a_mode < 64 or storage_config.swizzle_b_mode < 64))
                continue;

            int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
            if (num_stages < 2)
                continue;

            candidates.push_back(layout);
        }
        }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static int get_smem_bytes_per_k(const at::ScalarType& dtype, int block_k) {
        return (dtype == kPackedFP4) ? (block_k / 2) : (block_k * static_cast<int>(c10::elementSize(dtype)));
    }

    static int get_smem_d_size_for_swizzle(const GemmDesc& desc, const Layout& layout, int swizzle_cd, int store_m) {
        const int cd_size = c10::elementSize(desc.cd_dtype);
        if (swizzle_cd > 0
            and layout.block_n * cd_size >= swizzle_cd
            and (layout.block_n * cd_size) % swizzle_cd == 0)
            return (layout.block_n * cd_size / swizzle_cd) * swizzle_cd * store_m;
        return 0;
    }

    static int get_smem_per_stage(const GemmDesc& desc, const Layout& layout) {
        const bool b_padded_fp4 = (desc.a_dtype != kPackedFP4 && desc.b_dtype == kPackedFP4);
        // Mixed FP4_A x FP8_B (swapAB): A uses .b4x16_p64 padded SMEM (row = block_k, like mixed B).
        const bool a_padded_fp4 = (desc.a_dtype == kPackedFP4 && desc.b_dtype != kPackedFP4);
        const int smem_a = layout.block_m *
            (a_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.a_dtype, layout.block_k));
        const int smem_b = layout.block_n *
            (b_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.b_dtype, layout.block_k));
        const int smem_sfa = (desc.kernel_type == KernelType::Kernel1D1D)
            ? align(layout.block_m * static_cast<int>(sizeof(int32_t)), 128) : 0;
        const int smem_sfb = (desc.kernel_type == KernelType::Kernel1D1D)
            ? align(layout.block_n * static_cast<int>(sizeof(int32_t)), 128) : 0;
        return smem_a + smem_b + smem_sfa + smem_sfb;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        const auto load_block_m = layout.block_m;
        const auto load_block_n = layout.block_n;

        const bool a_padded_fp4 = (desc.a_dtype == kPackedFP4 && desc.b_dtype != kPackedFP4);
        const auto smem_k_bytes_a = a_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.a_dtype, layout.block_k);
        const auto swizzle_mode_a = get_swizzle_mode(smem_k_bytes_a, 1);
        // Mixed FP8xFP4: B uses .b4x16_p64 padded SMEM (row stride = block_k, same as FP8)
        const bool b_padded_fp4 = (desc.a_dtype != kPackedFP4 && desc.b_dtype == kPackedFP4);
        const auto smem_row_bytes_b = (desc.major_b == cute::UMMA::Major::K)
            ? (b_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.b_dtype, layout.block_k))
            : layout.block_n * static_cast<int>(c10::elementSize(desc.b_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(smem_row_bytes_b, 1);

        const int cd_size = c10::elementSize(desc.cd_dtype);
        // cd_n_contiguous gates the TMA-store epilogue (off for AB-swap transposed output).
        const auto swizzle_mode_cd = (desc.cd_n_contiguous and layout.block_n * cd_size >= 128) ? 128 : 0;

        // Sub-tile epilogue: reduce SMEM_D by storing smaller M sub-tiles.
        // Try store_block_m = 64 (sub-tile) and see if it gains pipeline stages.
        constexpr int kNumMaxStages = 16;
        const int smem_barriers = kNumMaxStages * 8 * 2;
        const int per_stage = get_smem_per_stage(desc, layout);
        const int smem_d_full = get_smem_d_size_for_swizzle(desc, layout, swizzle_mode_cd, layout.block_m);
        const int stages_full = std::min((smem_capacity - smem_barriers - smem_d_full) / per_stage, kNumMaxStages);

        int store_m = layout.block_m;
        constexpr int kSubTileM = 64;
        const bool supports_subtile = (desc.kernel_type != KernelType::KernelNoSF);
        if (supports_subtile and swizzle_mode_cd > 0 and layout.block_m > kSubTileM) {
            const int smem_d_sub = get_smem_d_size_for_swizzle(desc, layout, swizzle_mode_cd, kSubTileM);
            const int stages_sub = std::min((smem_capacity - smem_barriers - smem_d_sub) / per_stage, kNumMaxStages);
            if (stages_sub > stages_full)
                store_m = kSubTileM;
        }

        return {
            load_block_m, load_block_n,
            store_m, 0,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static int get_smem_d_size(const GemmDesc& desc, const Layout& layout) {
        const auto storage = get_storage_config(desc, layout);
        return get_smem_d_size_for_swizzle(desc, layout, storage.swizzle_cd_mode, storage.store_block_m);
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 16;

        const int smem_barriers = kNumMaxStages * 8 * 2;
        const bool a_padded_fp4 = (desc.a_dtype == kPackedFP4 && desc.b_dtype != kPackedFP4);
        const int smem_a_per_stage = storage_config.load_block_m *
            (a_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.a_dtype, layout.block_k));
        const bool b_padded_fp4 = (desc.a_dtype != kPackedFP4 && desc.b_dtype == kPackedFP4);
        const int smem_b_per_stage = storage_config.load_block_n *
            (b_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.b_dtype, layout.block_k));

        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            smem_sfa_per_stage = align(layout.block_m * static_cast<int>(sizeof(int32_t)), 128);
            smem_sfb_per_stage = align(layout.block_n * static_cast<int>(sizeof(int32_t)), 128);
        }

        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 2 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        const int smem_d = get_smem_d_size_for_swizzle(desc, layout, storage_config.swizzle_cd_mode,
                                                       storage_config.store_block_m);

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
        const auto num_m_blocks = ceil_div(desc.get_expected_m(), layout.block_m);
        const auto num_n_blocks = ceil_div(desc.get_expected_n(), layout.block_n);
        const auto num_blocks = num_m_blocks * num_n_blocks * desc.get_expected_num_groups();
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        // TMA-bound latency model (empirically tuned on SM120a). Discrete num_waves
        // (ceil division) dominates the BM=64 vs BM=128 choice; more pipeline stages
        // reduce per-kblock barrier stall (modeled as kSyncBaseCy / sqrt(stages)).
        static constexpr double kCyPerTmaByte = 0.07;     // ~35 GB/s per SM
        static constexpr double kSyncBaseCy = 120.0;      // per-kblock barrier overhead
        static constexpr double kBlockOverheadCy = 2000;   // epilogue + scheduling

        const int64_t expected_k = desc.get_expected_k();
        const int k_blocks = ceil_div(static_cast<int>(expected_k), layout.block_k);

        const int elem_size = get_element_size(desc.get_mma_kind());
        const int sf_bytes_a = (desc.kernel_type == KernelType::Kernel1D1D)
            ? align(layout.block_m * 4, 128) : 0;
        const int sf_bytes_b = (desc.kernel_type == KernelType::Kernel1D1D)
            ? align(layout.block_n * 4, 128) : 0;
        const int64_t tma_bytes_per_kb = (int64_t)layout.block_m * layout.block_k * elem_size
                                       + (int64_t)layout.block_n * layout.block_k * elem_size
                                       + sf_bytes_a + sf_bytes_b;

        const auto storage_config = get_storage_config(desc, layout);
        const int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
        const double sync_per_kb = kSyncBaseCy / std::sqrt(static_cast<double>(num_stages));

        const double tma_per_kb = tma_bytes_per_kb * kCyPerTmaByte + sync_per_kb;

        // Account for split-K: too few blocks → it divides K across more partitions.
        const int split_k = (desc.gemm_type == GemmType::Normal) ? get_split_k_factor(desc, layout) : 1;
        const int k_blocks_eff = (split_k > 1) ? k_blocks / split_k : k_blocks;
        const int total_blocks = num_blocks * split_k;
        const auto num_waves_eff = ceil_div(total_blocks, desc.num_sms);

        const double block_time = k_blocks_eff * tma_per_kb + kBlockOverheadCy;
        // Split-K reduction overhead (~2-3 us on SM120a): fixed launch + per-element.
        const double reduce_fixed_cy = 5000.0;
        const double reduce_per_elem_cy = 0.01;
        const double reduce_overhead = (split_k > 1)
            ? reduce_fixed_cy + reduce_per_elem_cy * desc.get_expected_m() * desc.get_expected_n()
            : 0.0;
        const int64_t total_latency = static_cast<int64_t>(num_waves_eff * block_time + reduce_overhead);

        return {num_waves_eff, last_wave_util, total_latency, layout};
    }

    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        // Use 5% tolerance: within this band, prefer tile-shape tie-breaks
        const double ratio = (b.num_cycles > 0)
            ? static_cast<double>(a.num_cycles) / b.num_cycles : 1.0;
        if (ratio < 0.95) return true;   // a clearly better
        if (ratio > 1.05) return false;  // b clearly better

        // Within 5%: prefer larger N tile for better reuse
        if (a.layout.block_n != b.layout.block_n)
            return a.layout.block_n > b.layout.block_n;
        // Prefer smaller K tile (more pipeline stages for TMA hiding)
        if (a.layout.block_k != b.layout.block_k)
            return a.layout.block_k < b.layout.block_k;
        // Prefer larger M tile for better per-block efficiency
        if (a.layout.block_m != b.layout.block_m)
            return a.layout.block_m > b.layout.block_m;
        // Final: lower absolute latency
        return a.num_cycles < b.num_cycles;
    }

    static int get_split_k_factor(const GemmDesc& desc, const Layout& layout) {
        if (desc.gemm_type != GemmType::Normal or desc.kernel_type == KernelType::KernelNoSF)
            return 1;

        const int num_m_blocks = ceil_div(desc.get_expected_m(), layout.block_m);
        const int num_n_blocks = ceil_div(desc.get_expected_n(), layout.block_n);
        const int num_mn_blocks = num_m_blocks * num_n_blocks;
        const int num_k_blocks = ceil_div(static_cast<int>(desc.get_expected_k()), layout.block_k);

        if (num_mn_blocks >= desc.num_sms / 2)
            return 1;

        const int target_blocks = desc.num_sms * 3 / 4;
        int split_k = ceil_div(target_blocks, num_mn_blocks);

        // k_per_split must be divisible by the kernel's SF tile size so each
        // partition starts at an SF-aligned K-block boundary.  The kernel packs
        // 4 UE8M0 bytes per int32, spanning (4 * max_gran_k / block_k) k-blocks.
        const int kSFTileKBlocks = (4 * desc.max_gran_k) / layout.block_k;
        if (kSFTileKBlocks == 0)
            return 1;
        while (split_k > 1 and (num_k_blocks % split_k != 0 or (num_k_blocks / split_k) % kSFTileKBlocks != 0))
            --split_k;

        split_k = std::min(split_k, num_k_blocks / (2 * kSFTileKBlocks));

        constexpr int64_t kMaxWorkspaceBytes = 32 * 1024 * 1024;
        const int64_t mn_bytes = static_cast<int64_t>(desc.get_expected_m()) * desc.get_expected_n() * sizeof(float);
        if (mn_bytes > 0)
            split_k = std::min(split_k, std::max(static_cast<int>(kMaxWorkspaceBytes / mn_bytes), 1));

        return std::max(split_k, 1);
    }
};

} // namespace deep_gemm
