#pragma once

#include "../../jit/device_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/lazy_init.hpp"

namespace deep_gemm {

class HeuristicsRuntime {
    static constexpr int kLegacyMKAlignmentForContiguousLayout = 128;

    bool ignore_compile_dims = false;
    int block_m_multiple_of = 1;
    int block_n_multiple_of = 1;
    int mk_alignment_for_contiguous_layout = kLegacyMKAlignmentForContiguousLayout;

public:
    void set_ignore_compile_dims(const bool& new_value) {
        ignore_compile_dims = new_value;
    }

    bool get_ignore_compile_dims() const {
        return ignore_compile_dims;
    }

    void set_block_size_multiple_of(const int& new_block_m_multiple_of, const int& new_block_n_multiple_of) {
        block_m_multiple_of = new_block_m_multiple_of;
        block_n_multiple_of = new_block_n_multiple_of;
    }

    int get_block_m_multiple_of() const {
        return block_m_multiple_of;
    }

    int get_block_n_multiple_of() const {
        return block_n_multiple_of;
    }

    void set_mk_alignment_for_contiguous_layout(const int& new_value) {
        mk_alignment_for_contiguous_layout = new_value;
    }

    int get_mk_alignment_for_contiguous_layout() const {
        return mk_alignment_for_contiguous_layout;
    }

    // Per-arch BLOCK_M search: start at `max_block_m`, step down by `step` (never below
    // `min_block_m`) until the tile no longer over-covers M.
    struct ContiguousMKAlignment { int max_block_m, min_block_m, step; };

    static ContiguousMKAlignment get_contiguous_mk_alignment(const int& arch_major) {
        // SM120: warp layout is kMWarps(4) * MMA_M(16), so BLOCK_M is a multiple of 64
        if (arch_major == 12)
            return {128, 64, 64};
        // SM100: 16-row MMA steps from 240 down to 32
        if (arch_major == 10)
            return {240, 32, 16};
        // SM90 and others: fixed legacy alignment, no shrinking
        return {kLegacyMKAlignmentForContiguousLayout, kLegacyMKAlignmentForContiguousLayout, 1};
    }

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m,
                                                                    const std::optional<int>& num_groups = std::nullopt) {
        const auto spec = get_contiguous_mk_alignment(device_runtime->get_arch_major());
        int block_m = spec.max_block_m;
        if (expected_m.has_value()) {
            // Grouped layouts must cover the per-group M, not the summed M
            int per_group_m = expected_m.value();
            if (num_groups.has_value() and num_groups.value() > 0)
                per_group_m = (per_group_m + num_groups.value() - 1) / num_groups.value();
            for (; block_m > spec.min_block_m and block_m - spec.step >= per_group_m; block_m -= spec.step);
        }
        return block_m;
    }
};

static auto heuristics_runtime = LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
