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
        if (device_runtime->get_arch_major() != 10)
            return kLegacyMKAlignmentForContiguousLayout;

        int block_m = 224, mma_step = 32;
        if (expected_m.has_value()) {
            // Reduce `block_m` while ensuring it covers `m`
            for (; block_m > 32 and block_m - mma_step >= expected_m.value(); block_m -= mma_step);
        }
        return block_m;
    }

    static int get_recommended_mk_alignment_for_contiguous_layout(const bool& use_psum_layout,
                                                                  const std::optional<int>& expected_m,
                                                                  const std::optional<int>& expected_k,
                                                                  const std::optional<int>& expected_num_groups) {
        if (device_runtime->get_arch_major() == 9 and not use_psum_layout and
            expected_m.has_value() and expected_m.value() == 128 and
            expected_k.has_value() and expected_k.value() <= 256 and
            expected_num_groups.has_value() and expected_num_groups.value() == 4)
            return 64;
        return get_theoretical_mk_alignment_for_contiguous_layout(expected_m);
    }
};

static auto heuristics_runtime = LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
