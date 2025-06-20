#pragma once

#include "utils.cuh"

namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked,
    GroupedWithOffset
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNum1DBlocksPerGroup = 16>
struct Scheduler {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks;

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;

    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum;

    __device__ __forceinline__ explicit Scheduler(const uint32_t& shape_m,
                                                  int* grouped_layout = nullptr) {
        num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
        if constexpr (kGemmType == GemmType::Normal) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
            this->grouped_layout = grouped_layout;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = 0;
            this->grouped_layout = grouped_layout;
        }
    }

    // ReSharper disable once CppNotAllPathsReturnValue
    __device__ __forceinline__ bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal) {
            return true;
        } else if constexpr (kGemmType == GemmType::GroupedContiguous) {
            return __ldg(grouped_layout + m_offset + m_block_idx * BLOCK_M) >= 0;
        } else if constexpr (kGemmType == GemmType::GroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < __ldg(grouped_layout + curr_group_idx);
        }
    }

    __device__ __forceinline__ bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::GroupedMasked) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::GroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsTMAMulticastOnA) {
                return true;
            } else {
                auto group_idx = __ldg(grouped_layout + m_block_idx * BLOCK_M);
                auto peer_group_idx = __ldg(grouped_layout + (m_block_idx ^ 1) * BLOCK_M);
                return group_idx == peer_group_idx;
            }
        }
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t& num_m_blocks, const uint32_t& block_idx,
                                                           uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        auto primary_num_blocks = kIsTMAMulticastOnA ? kNumNBlocks : num_m_blocks;
        auto secondary_num_blocks = kIsTMAMulticastOnA ? num_m_blocks : kNumNBlocks;
        auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        if (kNumTMAMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }

        // Convert to final M/N block indices
        if constexpr (kIsTMAMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t& shape_dim, const uint32_t& block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M));
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;

                // Within the current group
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            is_peer_cta_alive = kNumNBlocks % kNumTMAMulticast == 0 or          // Always aligned on N (constant bypass)
                                num_aligned_m_blocks % kNumTMAMulticast == 0 or // Always aligned on M (constant bypass)
                                (next_block_idx ^ 1) < num_blocks;              // Peer CTA in bound
            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }
};


template <uint32_t kNumTMAMulticast, uint32_t kNumNBlocks, uint32_t kNumNBlocksPerGroup>
__device__ __forceinline__ void offset_get_swizzled_block_idx(
    const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx)
{
    DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

    // Swizzle for better L2 usages
    auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup;
    auto group_idx = block_idx / num_blocks_per_group;
    auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
    auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
    auto in_group_idx = block_idx % num_blocks_per_group;
    m_block_idx = in_group_idx / num_n_blocks_in_group;
    n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
}



struct GroupedWithOffsetSchedulerInput
{
    uint32_t shape_m;
    int64_t* problem_m_offsets;
};

struct GroupedWithOffsetSchedulerInputSwapAB
{
    uint32_t shape_m;
    int64_t* problem_n_offsets;
};

struct StridedBatchedSchedulerInput
{
    uint32_t shape_m;
    uint64_t ld_a;
    uint64_t stride_a;
    uint64_t ld_b;
    uint64_t stride_b;
    uint64_t ld_d;
    uint64_t stride_d;
};

struct StridedBatchedSchedulerInputSwapAB
{
    uint32_t shape_n;
    uint64_t ld_a;
    uint64_t stride_a;
    uint64_t ld_b;
    uint64_t stride_b;
    uint64_t ld_d;
    uint64_t stride_d;
};


// Need to keep the same as the one in tests/unittest/_torch/thop/deep_gemm_tests.py
template <typename T_offset, typename T_index>
__host__ __device__ __forceinline__ T_offset compute_padded_offset(T_offset offset, T_index problem_idx)
{
    // This formulation ensures that padded_offset[i + 1] - padded_offset[i] >= offset[i + 1] - offset[i].
    constexpr T_offset alignment = 32;
    return (offset + problem_idx * (alignment - 1)) / alignment * alignment;
}

template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
    uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N), uint32_t kNumNBlocksPerGroup = 16>
struct GroupedWithOffsetScheduler
{
    static constexpr GemmType gemm_type = GemmType::GroupedWithOffset;

    int current_iter = -1;
    uint32_t curr_group_idx;
    uint32_t curr_cumsum;
    int64_t m_offset;
    int64_t m_padded_4_offset;
    int64_t m_boundary;
    int64_t* problem_m_offsets;

    using Input = GroupedWithOffsetSchedulerInput;
    Input input;

    GroupedWithOffsetScheduler() {}

    __device__ __forceinline__ GroupedWithOffsetScheduler(Input& input)
    {
        this->problem_m_offsets = input.problem_m_offsets;
        curr_group_idx = 0;
        curr_cumsum = 0;
    }

    __device__ __forceinline__ uint32_t get_global_m_idx(uint32_t const& block_idx)
    {
        return m_offset + block_idx * BLOCK_M;
    }

    __device__ __forceinline__ uint32_t get_global_n_idx(
        uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0)
    {
        return curr_group_idx * shape_dim + block_idx * block_size;
    }

    __device__ __forceinline__ uint32_t get_global_scales_a_idx(uint32_t const& block_idx)
    {
        return m_padded_4_offset + block_idx * BLOCK_M;
    }

    __device__ __forceinline__ uint32_t get_global_scales_b_idx(
        uint32_t const shape_dim, uint32_t const block_size, uint32_t const& block_idx, uint32_t const& m_block_idx = 0)
    {
        return curr_group_idx * shape_dim + block_idx * block_size;
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx)
    {
        ++current_iter;
        auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
        uint32_t num_m_blocks;
        while (true)
        {
            // End of the task
            if (curr_group_idx == kNumGroups)
                return false;
            m_offset = __ldg(problem_m_offsets + curr_group_idx);
            m_boundary = __ldg(problem_m_offsets + curr_group_idx + 1);
            m_padded_4_offset = compute_padded_offset(m_offset, curr_group_idx);
            auto m = m_boundary - m_offset;
            // Within current group
            num_m_blocks = ceil_div(m, static_cast<int64_t>(BLOCK_M));
            auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
            if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                break;

            // Move to check the next group
            curr_group_idx++;
            curr_cumsum = current_m_block_cumsum;
        }

        offset_get_swizzled_block_idx<kNumTMAMulticast, kNumNBlocks, kNumNBlocksPerGroup>(
            num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        return true;
    }
};

template <uint32_t SHAPE_M, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
    uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M), uint32_t kNumMBlocksPerGroup = 16>
struct GroupedWithOffsetSchedulerSwapAB
{
    static constexpr GemmType gemm_type = GemmType::GroupedWithOffset;

    int current_iter = -1;
    uint32_t curr_group_idx;
    uint32_t curr_cumsum;
    int64_t n_offset;
    int64_t n_padded_4_offset;
    int64_t n_boundary;
    int64_t* problem_n_offsets;

    using Input = GroupedWithOffsetSchedulerInputSwapAB;
    Input input;

    GroupedWithOffsetSchedulerSwapAB() {}

    __device__ __forceinline__ GroupedWithOffsetSchedulerSwapAB(Input& input)
    {
        this->problem_n_offsets = input.problem_n_offsets;
        curr_group_idx = 0;
        curr_cumsum = 0;
    }

    // weight
    __device__ __forceinline__ uint32_t get_global_m_idx(
        const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0)
    {
        return curr_group_idx * shape_dim + block_idx * block_size;
    }

    // act
    __device__ __forceinline__ uint32_t get_global_n_idx(uint32_t const& block_idx)
    {
        return n_offset + block_idx * BLOCK_N;
    }

    // act scales
    __device__ __forceinline__ uint32_t get_global_scales_b_idx(uint32_t const& block_idx)
    {
        return n_padded_4_offset + block_idx * BLOCK_N;
    }

    // weight scales
    __device__ __forceinline__ uint32_t get_global_scales_a_idx(
        const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0)
    {
        return curr_group_idx * shape_dim + block_idx * block_size;
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx)
    {
        ++current_iter;
        auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
        uint32_t num_n_blocks;
        while (true)
        {
            // End of the task
            if (curr_group_idx == kNumGroups)
                return false;
            n_offset = __ldg(problem_n_offsets + curr_group_idx);
            n_boundary = __ldg(problem_n_offsets + curr_group_idx + 1);
            n_padded_4_offset = compute_padded_offset(n_offset, curr_group_idx);
            auto n = n_boundary - n_offset;
            // Within current group
            num_n_blocks = ceil_div(n, static_cast<int64_t>(BLOCK_N));
            auto current_n_block_cumsum = curr_cumsum + num_n_blocks;
            if (next_block_idx < current_n_block_cumsum * kNumMBlocks)
                break;

            // Move to check the next group
            curr_group_idx++;
            curr_cumsum = current_n_block_cumsum;
        }

        offset_get_swizzled_block_idx<kNumTMAMulticast, kNumMBlocks, kNumMBlocksPerGroup>(
            num_n_blocks, next_block_idx - curr_cumsum * kNumMBlocks, n_block_idx, m_block_idx);
        return true;
    }
};

template <GemmType GT, uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumGroups, uint32_t kNumTMAMulticast, uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
    uint32_t kNumNBlocksPerGroup = 16>
struct SchedulerSelector
{
    static constexpr auto select_type()
    {
        if constexpr (GT == GemmType::GroupedWithOffset)
            return GroupedWithOffsetScheduler<SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kNumNBlocks,
                kNumNBlocksPerGroup>();
    }

    using type = decltype(select_type());
};

template <GemmType GT, uint32_t SHAPE_M, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumGroups, uint32_t kNumTMAMulticast, uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M),
    uint32_t kNumMBlocksPerGroup = 16>
struct SchedulerSelectorSwapAB
{
    static constexpr auto select_type()
    {
        static_assert(GT == GemmType::GroupedWithOffset || GT == GemmType::Normal,
            "Only GroupedWithOffset and Normal are supported for SwapAB");
        if constexpr (GT == GemmType::Normal)
            return NormalSchedulerSwapAB<SHAPE_M, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kNumMBlocks,
                kNumMBlocksPerGroup>();
        if constexpr (GT == GemmType::GroupedWithOffset)
            return GroupedWithOffsetSchedulerSwapAB<SHAPE_M, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast,
                kNumMBlocks, kNumMBlocksPerGroup>();
    }

    using type = decltype(select_type());
};

#pragma clang diagnostic pop

} // namespace deep_gemm
