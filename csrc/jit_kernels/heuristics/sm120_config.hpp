#pragma once

#include "common.hpp"

namespace deep_gemm {

struct SM120GemmDesc {
    GemmType gemm_type;
    KernelType kernel_type;
    int m, n, k, num_groups;
    at::ScalarType a_dtype, b_dtype, cd_dtype;
    cute::UMMA::Major major_a;
    cute::UMMA::Major major_b;
    bool with_accumulation;
    int num_sms, tc_util;
    std::string compiled_dims;
    int max_gran_k = 128;
    bool cd_n_contiguous = true;
    bool ensure_zero_padding = true;
    int expected_m = 0, expected_n = 0, expected_k = 0, expected_num_groups = 0;
    int get_expected_m() const { return expected_m > 0 ? expected_m : m; }
    int get_expected_n() const { return expected_n > 0 ? expected_n : n; }
    int get_expected_k() const { return expected_k > 0 ? expected_k : k; }
    int get_expected_num_groups() const { return expected_num_groups > 0 ? expected_num_groups : num_groups; }

    MmaKind get_mma_kind() const {
        if (a_dtype == torch::kBFloat16)
            return MmaKind::BF16;
        return a_dtype == kPackedFP4 and b_dtype == kPackedFP4 ? MmaKind::MXF4 : MmaKind::MXFP8FP4;
    }

    int get_heuristic_element_size() const {
        return get_mma_kind() == MmaKind::BF16 ? 2 : 1;
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

    friend std::ostream& operator << (std::ostream& os, const SM120GemmDesc& desc) {
        os << "SM120GemmDesc(gemm_type=" << static_cast<int>(desc.gemm_type)
           << ", kernel_type=" << static_cast<int>(desc.kernel_type)
           << ", m=" << desc.m << ", n=" << desc.n << ", k=" << desc.k
           << ", num_groups=" << desc.num_groups
           << ", major_a=" << static_cast<int>(desc.major_a)
           << ", major_b=" << static_cast<int>(desc.major_b)
           << ", mma_kind=" << static_cast<int>(desc.get_mma_kind())
           << ", a_dtype=" << c10::toString(desc.a_dtype)
           << ", b_dtype=" << c10::toString(desc.b_dtype)
           << ", cd_dtype=" << c10::toString(desc.cd_dtype)
           << ", with_accumulation=" << static_cast<int>(desc.with_accumulation)
           << ", num_sms=" << desc.num_sms << ", tc_util=" << desc.tc_util
           << ", compiled_dims=" << desc.compiled_dims
           << ", max_gran_k=" << desc.max_gran_k
           << ", cd_n_contiguous=" << desc.cd_n_contiguous
           << ", ensure_zero_padding=" << static_cast<int>(desc.ensure_zero_padding)
           << ", expected_m=" << desc.expected_m << ", expected_n=" << desc.expected_n
           << ", expected_k=" << desc.expected_k
           << ", expected_num_groups=" << desc.expected_num_groups << ")";
        return os;
    }
};

struct SM120GemmConfig {
    Layout layout;
    StorageConfig storage_config;
    PipelineConfig pipeline_config;
    LaunchConfig launch_config;
    int split_k_factor = 1;

    friend std::ostream& operator << (std::ostream& os, const SM120GemmConfig& config) {
        os << "SM120GemmConfig(layout=" << config.layout
           << ", storage_config=" << config.storage_config
           << ", pipeline_config=" << config.pipeline_config
           << ", launch_config=" << config.launch_config
           << ", split_k=" << config.split_k_factor << ")";
        return os;
    }
};

template <typename ArchSpec>
static SM120GemmConfig get_best_sm120_config(const SM120GemmDesc& desc) {
    desc.check_validity();
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);
    for (int i = 1; i < static_cast<int>(layout_candidates.size()); ++ i) {
        const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
        if (ArchSpec::compare(candidate_info, layout_info))
            layout = layout_candidates[i], layout_info = candidate_info;
    }
    const auto storage_config = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config = ArchSpec::get_launch_config(desc, layout);
    const auto gemm_config = SM120GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = pipeline_config,
        .launch_config = launch_config
    };
    if (deep_jit::get_env<int>("DG_JIT_DEBUG") or deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        std::stringstream ss;
        ss << desc;
        const auto key = ss.str();
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << desc << ": " << gemm_config << ", " << layout_info << std::endl;
            printed.insert(key);
        }
    }
    return gemm_config;
}

} // namespace deep_gemm
