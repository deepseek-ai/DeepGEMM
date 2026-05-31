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

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m,
                                                                    const std::optional<int>& num_groups = std::nullopt) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major != 10 and arch_major != 12)
            return kLegacyMKAlignmentForContiguousLayout;

        // SM120: kMWarps=4, MMA_M=16, valid BLOCK_M must be multiple of 64
        int block_m = arch_major == 12 ? 128 : 240;
        int min_block_m = arch_major == 12 ? 64 : 32;
        int mma_step = arch_major == 12 ? 64 : 16;
        if (expected_m.has_value()) {
            int per_expert_m = expected_m.value();
            if (num_groups.has_value() and num_groups.value() > 0)
                per_expert_m = (per_expert_m + num_groups.value() - 1) / num_groups.value();
            for (; block_m > min_block_m and block_m - mma_step >= per_expert_m; block_m -= mma_step);
        }
        return block_m;
    }
};

static auto heuristics_runtime = LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
