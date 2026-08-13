#pragma once

#include <algorithm>
#include <cstdio>
#include <unordered_set>
#include <deep_gemm/common/types.cuh>

#include "config.hpp"
#include "runtime.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

template <typename ArchSpec>
static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();

    // Choose the best layout
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    DG_HOST_ASSERT(not layout_candidates.empty());
    auto layout = layout_candidates[0];
    auto layout_info = ArchSpec::get_layout_info(desc, layout);
    const auto forced_layout = get_env<std::string>("DG_JIT_FORCE_LAYOUT");
    if (forced_layout.empty()) {
        for (int i = 1; i < static_cast<int>(layout_candidates.size()); ++ i) {
            const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
            if (ArchSpec::compare(candidate_info, layout_info))
                layout = layout_candidates[i], layout_info = candidate_info;
        }
    } else {
        int block_m = 0, block_n = 0, block_k = 0;
        char trailing = 0;
        if (std::sscanf(forced_layout.c_str(), "%dx%dx%d%c", &block_m, &block_n, &block_k, &trailing) != 3)
            DG_HOST_UNREACHABLE("DG_JIT_FORCE_LAYOUT must use BMxBNxBK format");

        const auto it = std::find_if(layout_candidates.begin(), layout_candidates.end(), [&](const Layout& candidate) {
            return candidate.block_m == block_m and candidate.block_n == block_n and candidate.block_k == block_k;
        });
        if (it == layout_candidates.end())
            DG_HOST_UNREACHABLE("DG_JIT_FORCE_LAYOUT is not a valid candidate");
        layout = *it;
        layout_info = ArchSpec::get_layout_info(desc, layout);
    }

    // Infer other configs
    const auto storage_config = ArchSpec::get_storage_config(desc, layout);
    const auto pipeline_config = ArchSpec::get_pipeline_config(desc, layout, storage_config);
    const auto launch_config = ArchSpec::get_launch_config(desc, layout);
    const auto gemm_config = GemmConfig {
        .layout = layout,
        .storage_config = storage_config,
        .pipeline_config = pipeline_config,
        .launch_config = launch_config
    };

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
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
