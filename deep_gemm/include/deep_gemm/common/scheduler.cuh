#pragma once

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          // TODO: refactor this by other values
          uint32_t kNum1DBlocksPerGroup = 16>
struct Scheduler {
    int current_iter = -1;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // For grouped GEMM
    int* grouped_layout;
    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum;

    __device__ __forceinline__ explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                                  int* grouped_layout = nullptr) {
        num_m_blocks = ceil_div(shape_m, BLOCK_M);
        num_n_blocks = ceil_div(shape_n, BLOCK_N);
        if constexpr (kGemmType == GemmType::Normal) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = 0;
            this->grouped_layout = grouped_layout;
        }
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        // TODO: unify these 2 branches
        if constexpr (kIsMulticastOnA) {
            auto num_blocks_per_group = num_m_blocks * kNum1DBlocksPerGroup;
            auto group_idx = block_idx / num_blocks_per_group;
            auto first_n_block_idx = group_idx * kNum1DBlocksPerGroup;
            auto num_n_blocks_in_group = min(kNum1DBlocksPerGroup, num_n_blocks - first_n_block_idx);
            auto in_group_idx = block_idx % num_blocks_per_group;
            m_block_idx = in_group_idx / num_n_blocks_in_group;
            n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
        } else {
            auto num_blocks_per_group = num_n_blocks * kNum1DBlocksPerGroup;
            auto group_idx = block_idx / num_blocks_per_group;
            auto first_m_block_idx = group_idx * kNum1DBlocksPerGroup;
            auto num_m_blocks_in_group = min(kNum1DBlocksPerGroup, num_m_blocks - first_m_block_idx);
            auto in_group_idx = block_idx % num_blocks_per_group;
            m_block_idx = first_m_block_idx + in_group_idx % num_m_blocks_in_group;
            n_block_idx = in_group_idx / num_m_blocks_in_group;
        }
    }

    template <bool kWithGroupOffset>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kWithGroupOffset ? __ldg(grouped_layout + m_block_idx * BLOCK_M) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedMasked) {
            auto offset = kWithGroupOffset ? curr_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - curr_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm
