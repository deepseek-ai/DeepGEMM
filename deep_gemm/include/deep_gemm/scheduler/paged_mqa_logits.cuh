#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

template <uint32_t kAlignedBatchSize, uint32_t SPLIT_KV, uint32_t kNumSMs, bool kIsVarlen = false>
CUTLASS_GLOBAL __launch_bounds__(32, 1)
void smxx_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t next_n, const bool is_context_lens_2d,
                                    const uint32_t* context_lens, const uint32_t* indices, uint32_t* schedule_metadata) {
    DG_STATIC_ASSERT(kAlignedBatchSize % 32 == 0, "Invalid aligned batch size");
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    __shared__ uint32_t varlen_atom_token_start[kAlignedBatchSize];
    __shared__ uint32_t varlen_atom_context_len[kAlignedBatchSize];
    __shared__ uint32_t varlen_num_atoms_shared;
    uint32_t num_items;

    if constexpr (kIsVarlen) {
        if (lane_idx == 0) {
            uint32_t t = 0, atom_count = 0;
            while (t < batch_size) {
                varlen_atom_token_start[atom_count] = t;
                const bool is_paired = (t + 1 < batch_size and indices[t] == indices[t + 1]);
                varlen_atom_context_len[atom_count] = is_paired ? context_lens[t + 1] : context_lens[t];
                t += is_paired ? 2 : 1;
                ++ atom_count;
            }
            varlen_num_atoms_shared = atom_count;
        }
        __syncwarp();
        num_items = varlen_num_atoms_shared;
    } else {
        num_items = batch_size;
    }

    // Compute num_segs and prefix sum
    uint32_t num_segs[kAlignedBatchSize / 32];
    #pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++ k) {
        const uint32_t q_idx = k * 32 + lane_idx;
        uint32_t context_len;
        if constexpr (kIsVarlen) {
            context_len = (q_idx < num_items ? varlen_atom_context_len[q_idx] : 0);
        } else {
            const uint32_t lens_idx = (is_context_lens_2d ? q_idx * next_n + next_n - 1 : q_idx);
            context_len = (q_idx < batch_size ? context_lens[lens_idx] : 0);
        }
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

    // SM work distribution
    if constexpr (kIsVarlen) {
        const uint32_t total = sum;
        const uint32_t q = total / kNumSMs, r = total % kNumSMs;
        for (uint32_t sm_idx = lane_idx; sm_idx <= kNumSMs; sm_idx += 32) {
            uint32_t seg_starts = sm_idx * q + min(sm_idx, r);
            uint32_t lo = 0, hi = num_items;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                const bool pred = prefix_sum[mid] <= seg_starts;
                lo = pred ? mid + 1 : lo;
                hi = pred ? hi : mid;
            }
            const uint32_t atom_idx = lo;
            const uint32_t kv_split_idx = (atom_idx == 0 ? seg_starts : seg_starts - prefix_sum[atom_idx - 1]);
            const uint32_t q_atom_idx = (atom_idx < num_items ? varlen_atom_token_start[atom_idx] : batch_size);
            __syncwarp();

            schedule_metadata[sm_idx * 2] = q_atom_idx;
            schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
        }
    } else {
        const uint32_t next_n_atom = (next_n >= 2) ? 2 : 1;
        const uint32_t num_next_n_atoms = math::ceil_div(next_n, next_n_atom);
        const uint32_t total = sum * num_next_n_atoms;
        const uint32_t q = total / kNumSMs, r = total % kNumSMs;
        for (uint32_t sm_idx = lane_idx; sm_idx <= kNumSMs; sm_idx += 32) {
            uint32_t seg_starts = sm_idx * q + min(sm_idx, r);
            uint32_t lo = 0, hi = batch_size;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                const bool pred = prefix_sum[mid] * num_next_n_atoms <= seg_starts;
                lo = pred ? mid + 1 : lo;
                hi = pred ? hi : mid;
            }
            const uint32_t q_idx = lo;
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
}

// Conditional storage for varlen indices pointer (EBO: zero cost when unused)
template <bool kHasIndices>
struct IndicesStorage {
    const uint32_t* indices;
};

template <>
struct IndicesStorage<false> {};

template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t BLOCK_KV, uint32_t kNumBlocksPerSplit,
          uint32_t kNumNextNAtoms>
struct PagedMQALogitsScheduler : IndicesStorage<kIsVarlen> {
    const uint32_t* context_lens;
    uint32_t batch_size;

    uint32_t current_q_atom_idx, current_kv_idx;
    uint32_t end_q_atom_idx, end_kv_idx;
    uint32_t current_num_kv;

    CUTLASS_DEVICE static uint32_t atom_to_token_idx(const uint32_t& q_atom_idx) {
        if constexpr (kIsVarlen) {
            return q_atom_idx;
        } else {
            static constexpr bool kPadOddN = (not kIsVarlen) and (kNextN % 2 == 1) and (kNextN >= 3);
            static constexpr uint32_t kNextNAtom = (kIsVarlen or kNextN >= 2) ? 2 : 1;
            if constexpr (kPadOddN) {
                return q_atom_idx / kNumNextNAtoms * kNextN + q_atom_idx % kNumNextNAtoms * kNextNAtom;
            } else {
                return q_atom_idx * kNextNAtom;
            }
        }
    }

    CUTLASS_DEVICE static uint32_t atom_to_block_table_row(const uint32_t& q_atom_idx) {
        if constexpr (kIsVarlen) {
            return q_atom_idx;
        } else {
            return q_atom_idx / kNumNextNAtoms;
        }
    }

    CUTLASS_DEVICE uint32_t get_num_kv(const uint32_t& q_atom_idx) const {
        if constexpr (kIsVarlen) {
            const bool is_paired = (q_atom_idx + 1 < batch_size and
                                    this->indices[q_atom_idx] == this->indices[q_atom_idx + 1]);
            const uint32_t ctx_len = is_paired ? context_lens[q_atom_idx + 1] : context_lens[q_atom_idx];
            return math::ceil_div(ctx_len, BLOCK_KV);
        } else {
            const uint32_t q_idx = q_atom_idx / kNumNextNAtoms;
            const auto lens_idx = (kIsContextLens2D ? q_idx * kNextN + kNextN - 1 : q_idx);
            return math::ceil_div(context_lens[lens_idx], BLOCK_KV);
        }
    }

    CUTLASS_DEVICE explicit PagedMQALogitsScheduler(const uint32_t& sm_idx, const uint32_t& batch_size,
                                                    const uint32_t* context_lens,
                                                    const uint32_t* schedule_meta, const uint32_t* indices) {
        this->context_lens = context_lens;
        this->batch_size = batch_size;
        if constexpr (kIsVarlen) {
            this->indices = indices;
        }

        const auto current_pack = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end_pack = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        current_q_atom_idx = current_pack.x, current_kv_idx = current_pack.y * kNumBlocksPerSplit;
        end_q_atom_idx = end_pack.x, end_kv_idx = end_pack.y * kNumBlocksPerSplit;

        current_num_kv = get_num_kv(current_q_atom_idx);
    }

    // Advance step in q_atom_idx space when moving to the next atom.
    // Varlen: 1 or 2 depending on whether consecutive tokens share the same sequence.
    // Non-varlen: always 1 (one atom unit).
    CUTLASS_DEVICE uint32_t get_atom_advance(const uint32_t& q_atom_idx, const uint32_t& bound) const {
        if constexpr (kIsVarlen) {
            return (q_atom_idx + 1 < bound and this->indices[q_atom_idx] == this->indices[q_atom_idx + 1]) ? 2 : 1;
        } else {
            return 1;
        }
    }

    // Whether num_kv should be refreshed after advancing to q_atom_idx.
    // Varlen: always refresh (each atom may have a different context_len).
    // Non-varlen: only at atom-group boundaries (atoms within a group share context_len).
    CUTLASS_DEVICE bool should_refresh_num_kv(const uint32_t& q_atom_idx) const {
        if constexpr (kIsVarlen) {
            return true;
        } else {
            return q_atom_idx % kNumNextNAtoms == 0;
        }
    }

    CUTLASS_DEVICE bool fetch_next_task(uint32_t &q_atom_idx, uint32_t &kv_idx, uint32_t &num_kv) {
        q_atom_idx = current_q_atom_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (current_q_atom_idx == end_q_atom_idx and current_kv_idx == end_kv_idx)
            return false;

        current_kv_idx += kNumBlocksPerSplit;
        if (current_kv_idx >= current_num_kv) {
            current_kv_idx = 0;
            current_q_atom_idx += get_atom_advance(current_q_atom_idx, end_q_atom_idx);
            if (should_refresh_num_kv(current_q_atom_idx) and exist_q_atom_idx(current_q_atom_idx)) {
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
