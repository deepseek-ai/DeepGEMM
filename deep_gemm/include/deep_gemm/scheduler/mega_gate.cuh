#pragma once

#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched::mega_gate {

template <uint32_t kNumSMs, uint32_t kNumLogicalCtas, uint32_t kBlockTokens>
struct Scheduler {
    static constexpr uint32_t kNumWorkerGroups = kNumSMs / kNumLogicalCtas;
    uint32_t worker_group_idx;
    uint32_t num_token_blocks;

    CUTLASS_DEVICE Scheduler(const uint32_t& num_tokens):
        worker_group_idx(blockIdx.x / kNumLogicalCtas),
        num_token_blocks(math::ceil_div(num_tokens, kBlockTokens)) {}

    template <typename Task>
    CUTLASS_DEVICE void run(const Task& run_task) const {
        uint32_t iter_idx = 0;
        for (auto token_block_idx = worker_group_idx; token_block_idx < num_token_blocks;
             token_block_idx += kNumWorkerGroups, ++ iter_idx)
            run_task(token_block_idx, iter_idx);
    }
};

template <uint32_t kNumLogicalCtas>
struct ScoreBarrier {
    static CUTLASS_DEVICE uint64_t encode_state(const uint64_t grid_idx, const uint32_t num_arrived_ctas = 0) {
        DG_STATIC_ASSERT(kNumLogicalCtas < layout::mega_gate::kNumMaxLogicalCtas,
                         "Too many logical CTAs for the score barrier");
        return (grid_idx + 1) * layout::mega_gate::kNumMaxLogicalCtas + num_arrived_ctas;
    }

    static CUTLASS_DEVICE void init(uint64_t* state_ptr) {
        ptx::st_rel(state_ptr, encode_state(ptx::get_grid_idx()));
    }

    static CUTLASS_DEVICE void wait_init(uint64_t* state_ptr) {
        const auto state_base = encode_state(ptx::get_grid_idx());
        while (ptx::ld_acq_gpu(state_ptr) - state_base >= kNumLogicalCtas);
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
