#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
#include <c10/core/ScalarType.h>
#include <deep_gemm/common/types.cuh>

#include "../../utils/math.hpp"

namespace deep_gemm {

/// GEMM descriptors
struct GemmDesc {
    GemmType gemm_type;
    KernelType kernel_type;
    int m, n, k, num_groups;
    at::ScalarType a_dtype, b_dtype, cd_dtype;
    cute::UMMA::Major major_a;
    cute::UMMA::Major major_b;
    bool with_accumulation;

    // Requirements from users
    int num_sms, tc_util;
    std::string compiled_dims;

    // Shape for heuristic generation
    int expected_m = 0, expected_n = 0, expected_k = 0, expected_num_groups = 0;
    int get_expected_m() const { return expected_m > 0 ? expected_m : m; }
    int get_expected_n() const { return expected_n > 0 ? expected_n : n; }
    int get_expected_k() const { return expected_k > 0 ? expected_k : k; }
    int get_expected_num_groups() const { return expected_num_groups > 0 ? expected_num_groups : num_groups; }

    MmaKind get_mma_kind() const {
        return a_dtype == torch::kBFloat16 ? MmaKind::BF16 : MmaKind::MXFP8FP4;
    }

    void check_validity() const {
        if (get_mma_kind() == MmaKind::BF16) {
            DG_HOST_ASSERT(a_dtype == torch::kBFloat16 and b_dtype == torch::kBFloat16);
        } else {
            DG_HOST_ASSERT(a_dtype == torch::kFloat8_e4m3fn or a_dtype == kPackedFP4);
            DG_HOST_ASSERT(b_dtype == torch::kFloat8_e4m3fn or b_dtype == kPackedFP4);
        }
        DG_HOST_ASSERT(cd_dtype == torch::kBFloat16 or cd_dtype == torch::kFloat);
        DG_HOST_ASSERT(num_sms % 2 == 0);
    }

    friend std::ostream& operator << (std::ostream& os, const GemmDesc& desc) {
        MmaKind mma_kind = desc.get_mma_kind();
        os << "GemmDesc(gemm_type=" << static_cast<int>(desc.gemm_type)
           << ", kernel_type=" << static_cast<int>(desc.kernel_type)
           << ", m=" << desc.m << ", n=" << desc.n << ", k=" << desc.k
           << ", num_groups=" << desc.num_groups
           << ", major_a=" << static_cast<int>(desc.major_a)
           << ", major_b=" << static_cast<int>(desc.major_b)
           << ", mma_kind=" << static_cast<int>(mma_kind)
           << ", a_dtype=" << c10::toString(desc.a_dtype)
           << ", b_dtype=" << c10::toString(desc.b_dtype)
           << ", cd_dtype=" << c10::toString(desc.cd_dtype)
           << ", with_accumulation=" << static_cast<int>(desc.with_accumulation)
           << ", num_sms=" << desc.num_sms
           << ", tc_util=" << desc.tc_util
           << ", compiled_dims=" << desc.compiled_dims
           << ", expected_m=" << desc.expected_m
           << ", expected_n=" << desc.expected_n
           << ", expected_k=" << desc.expected_k
           << ", expected_num_groups=" << desc.expected_num_groups << ")";
        return os;
    }
};

/// GEMM configs
struct Layout {
    int swap_ab;
    int block_m, block_n, block_k;
    int cluster_m, cluster_n;

    int get_cluster_size() const {
        return cluster_m * cluster_n;
    }

    friend std::ostream& operator << (std::ostream& os, const Layout& layout) {
        os << "Layout(swap_ab=" << layout.swap_ab
           << ", block_m=" << layout.block_m << ", block_n=" << layout.block_n << ", block_k=" << layout.block_k
           << ", cluster_m=" << layout.cluster_m << ", cluster_n=" << layout.cluster_n << ")";
        return os;
    }
};

struct StorageConfig {
    int load_block_m, load_block_n;
    int store_block_m, store_block_n;

    int swizzle_a_mode, swizzle_b_mode;
    int swizzle_cd_mode;

    friend std::ostream& operator << (std::ostream& os, const StorageConfig& config) {
        os << "StorageConfig("
           << "load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m << ", store_block_n=" << config.store_block_n
           << ", swizzle_a_mode=" << config.swizzle_a_mode << ", swizzle_b_mode=" << config.swizzle_b_mode
           << ", swizzle_cd_mode=" << config.swizzle_cd_mode << ")";
        return os;
    }
};

struct PipelineConfig {
    int smem_size;
    int num_stages;

    friend std::ostream& operator << (std::ostream& os, const PipelineConfig& config) {
        os << "PipelineConfig("
           << "smem_size=" << config.smem_size
           << ", num_stages=" << config.num_stages << ")";
        return os;
    }
};

struct LaunchConfig {
    int num_sms;
    int num_sms_per_cluster;
    int num_threads;

    int num_tma_threads;
    int num_math_threads;
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const LaunchConfig& config) {
        os << "LaunchConfig("
           << "num_sms=" << config.num_sms << ", num_sms_per_cluster=" << config.num_sms_per_cluster
           << ", num_threads=" << config.num_threads
           << ", num_tma_threads=" << config.num_tma_threads << ", num_math_threads=" << config.num_math_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

struct GemmConfig {
    Layout layout;
    StorageConfig storage_config;
    PipelineConfig pipeline_config;
    LaunchConfig launch_config;

    friend std::ostream& operator << (std::ostream& os, const GemmConfig& config) {
        os << "GemmConfig("
           << "layout=" << config.layout
           << ", storage_config=" << config.storage_config
           << ", pipeline_config=" << config.pipeline_config
           << ", launch_config=" << config.launch_config << ")";
        return os;
    }
};

/// Config comparators
struct LayoutInfo {
    int num_waves;
    int last_wave_util;
    int64_t num_cycles;
    Layout layout;

    friend std::ostream& operator << (std::ostream& os, const LayoutInfo& config) {
        os << "LayoutInfo("
           << "num_waves=" << config.num_waves
           << ", last_wave_util=" << config.last_wave_util 
           << ", num_cycles=" << config.num_cycles << ")";
        return os;
    }
};

}  // namespace deep_gemm
