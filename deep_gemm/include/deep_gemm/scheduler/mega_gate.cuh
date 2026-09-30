#pragma once

#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched::mega_gate {

using namespace layout::mega_gate;

// Every logical CTA of a token block arrives once; wait observes all their score slices through the
// terminal release sequence. wait_init prevents an arrival from racing the grid tag.
template <uint32_t kNumLogicalCtas>
struct ScoreBarrier {
    DG_STATIC_ASSERT(kNumLogicalCtas < kNumMaxLogicalCtas, "Too many logical CTAs for the score barrier");

    static CUTLASS_DEVICE uint64_t encode_state(const uint64_t grid_idx, const uint32_t num_arrived_ctas = 0) {
        return (grid_idx + 1) * kNumMaxLogicalCtas + num_arrived_ctas;
    }

    static CUTLASS_DEVICE void init(uint64_t* state_ptr) {
        ptx::st_rel(state_ptr, encode_state(ptx::get_grid_idx()));
    }

    static CUTLASS_DEVICE void wait_init(uint64_t* state_ptr) {
        const auto current_grid_barrier_base = encode_state(ptx::get_grid_idx());
        // NOTES: unsigned distance rejects stale grid tags but accepts any current-grid arrival count.
        while (ptx::ld_acq_gpu(state_ptr) - current_grid_barrier_base >= kNumLogicalCtas);
    }

    static CUTLASS_DEVICE void arrive(uint64_t* state_ptr) {
        ptx::red_async_inc_rel(state_ptr);
    }

    static CUTLASS_DEVICE void wait(uint64_t* state_ptr) {
        const auto all_ctas_arrived_state = encode_state(ptx::get_grid_idx(), kNumLogicalCtas);
        while (ptx::ld_acq_gpu(state_ptr) != all_ctas_arrived_state);
    }
};

} // namespace deep_gemm::sched::mega_gate
