#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

template <uint32_t kAlignedBatchSize, uint32_t SPLIT_KV, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(32, 1)
void smxx_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t next_n, const bool is_context_lens_2d,
                                    const uint32_t* context_lens, uint32_t* schedule_metadata) {
    DG_STATIC_ASSERT(kAlignedBatchSize % 32 == 0, "Invalid aligned batch size");
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    uint32_t num_segs[kAlignedBatchSize / 32];
    #pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++ k) {
        const uint32_t q_idx = k * 32 + lane_idx;
        const uint32_t lens_idx = (is_context_lens_2d ? q_idx * next_n + next_n - 1 : q_idx);
        const uint32_t context_len = (q_idx < batch_size ? context_lens[lens_idx] : 0);
        num_segs[k] = math::ceil_div(context_len, SPLIT_KV);
    }

    __shared__ uint32_t prefix_sum[kAlignedBatchSize];
    uint32_t sum = 0;
    #pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++ k) {
        uint32_t x = num_segs[k];
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffff, x, offset);
            x += (lane_idx >= offset ? y : 0);
        }
        x += sum;
        prefix_sum[k * 32 + lane_idx] = x;
        sum = __shfl_sync(0xffffffff, x, 31);
    }

    const uint32_t num_next_n_atoms = next_n / ((next_n % 2 == 0) ? 2 : 1);
    const uint32_t total = sum * num_next_n_atoms;
    const uint32_t q = total / kNumSMs, r = total % kNumSMs;
    for (uint32_t sm_idx = lane_idx; sm_idx <= kNumSMs; sm_idx += 32) {
        uint32_t seg_starts = sm_idx * q + min(sm_idx, r);
        uint32_t q_idx = 0;
        while (q_idx < batch_size and prefix_sum[q_idx] * num_next_n_atoms <= seg_starts)
            ++ q_idx;
        const uint32_t offset_in_q = (q_idx == 0 ? seg_starts : seg_starts - prefix_sum[q_idx - 1] * num_next_n_atoms);
        const uint32_t num_segs_q = (q_idx == 0 ? prefix_sum[0] : prefix_sum[q_idx] - prefix_sum[q_idx - 1]);
        const uint32_t atom_idx = num_segs_q > 0 ? offset_in_q / num_segs_q : 0;
        const uint32_t kv_split_idx = num_segs_q > 0 ? offset_in_q % num_segs_q : 0;
        const uint32_t q_atom_idx = q_idx * num_next_n_atoms + atom_idx;
        __syncwarp();

        schedule_metadata[sm_idx * 2] = q_atom_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}

template <uint32_t kNextN, bool kIsContextLens2D,
          uint32_t BLOCK_KV, uint32_t kNumBlocksPerSplit,
          uint32_t kNumNextNAtoms>
struct PagedMQALogitsScheduler {
    const uint32_t* context_lens;

    uint32_t current_q_atom_idx, current_kv_idx;
    uint32_t end_q_atom_idx, end_kv_idx;
    uint32_t current_num_kv;

    CUTLASS_DEVICE uint32_t get_num_kv(const uint32_t& q_atom_idx) const {
        const uint32_t q_idx = q_atom_idx / kNumNextNAtoms;
        const auto lens_idx = (kIsContextLens2D ? q_idx * kNextN + kNextN - 1 : q_idx);
        return math::ceil_div(context_lens[lens_idx], BLOCK_KV);
    }

    CUTLASS_DEVICE explicit PagedMQALogitsScheduler(const uint32_t& sm_idx, const uint32_t* context_lens, const uint32_t* schedule_meta) {
        this->context_lens = context_lens;

        const auto current_pack = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end_pack = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        current_q_atom_idx = current_pack.x, current_kv_idx = current_pack.y * kNumBlocksPerSplit;
        end_q_atom_idx = end_pack.x, end_kv_idx = end_pack.y * kNumBlocksPerSplit;

        current_num_kv = get_num_kv(current_q_atom_idx);
    }

    CUTLASS_DEVICE bool fetch_next_task(uint32_t &q_atom_idx, uint32_t &kv_idx, uint32_t &num_kv) {
        q_atom_idx = current_q_atom_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (current_q_atom_idx == end_q_atom_idx and current_kv_idx == end_kv_idx)
            return false;

        current_kv_idx += kNumBlocksPerSplit;
        if (current_kv_idx >= current_num_kv) {
            ++ current_q_atom_idx;
            current_kv_idx = 0;
            if (current_q_atom_idx % kNumNextNAtoms == 0 and exist_q_atom_idx(current_q_atom_idx)) {
                current_num_kv = get_num_kv(current_q_atom_idx);
            }
        }
        return true;
    }

    CUTLASS_DEVICE bool exist_q_atom_idx(const uint32_t& q_atom_idx) const {
        return q_atom_idx < end_q_atom_idx or (q_atom_idx == end_q_atom_idx and 0 < end_kv_idx);
    }
};

} // namespace deep_gemm::sched
