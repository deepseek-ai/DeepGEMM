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

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major != 10 and arch_major != 12)
            return kLegacyMKAlignmentForContiguousLayout;

        // SM120's grouped FP8/FP4 kernel uses 4 M-warps × MMA_M=16, so
        // BLOCK_M ≥ 64. SM100 supports the full 32..max range.
        // Stepping in MMA_M=16 increments lets the kernel pick an exact
        // fit (e.g. 64, 80, 96, 112, 128) for the given expected_m.
        int block_m = arch_major == 12 ? 128 : 240;
        int min_block_m = arch_major == 12 ? 64 : 32;
        int mma_step = 16;
        if (expected_m.has_value()) {
            for (; block_m > min_block_m and block_m - mma_step >= expected_m.value(); block_m -= mma_step);
        }
        return block_m;
    }
};

static auto heuristics_runtime = LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
