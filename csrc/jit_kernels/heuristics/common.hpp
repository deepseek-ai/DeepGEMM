#pragma once

#include <cstdio>
#include <sstream>
#include <unordered_set>
#include <deep_gemm/common/types.cuh>

#include "config.hpp"
#include "runtime.hpp"
#include "../../utils/layout.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

inline int get_byte_addressable_element_size(const MmaKind& mma_kind) {
    const int element_size = get_element_size(mma_kind);
    DG_HOST_ASSERT(element_size != -1 and "Unknown MMA kind");
    DG_HOST_ASSERT(element_size > 0);
    return element_size;
}

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

        const Layout* matched_layout = nullptr;
        for (const auto& candidate: layout_candidates) {
            if (candidate.block_m != block_m or candidate.block_n != block_n or candidate.block_k != block_k)
                continue;
            if (matched_layout != nullptr and (
                candidate.swap_ab != matched_layout->swap_ab or
                candidate.cluster_m != matched_layout->cluster_m or
                candidate.cluster_n != matched_layout->cluster_n))
                DG_HOST_UNREACHABLE("DG_JIT_FORCE_LAYOUT is ambiguous; BMxBNxBK does not identify swap or cluster layout");
            matched_layout = &candidate;
        }
        if (matched_layout == nullptr) {
            std::stringstream details;
            details << "DG_JIT_FORCE_LAYOUT is not a valid candidate: " << forced_layout
                    << " for " << desc << "; valid candidates:";
            for (const auto& candidate: layout_candidates)
                details << " " << candidate;
            DG_HOST_UNREACHABLE(details.str());
        }
        layout = *matched_layout;
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
