#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::sched {

enum class IndexType {
    MN,
    K,
    SF_K,
};

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // Select the best from candidates
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto candidate: {8u, 16u}) {
        const auto usage = kIsMulticastOnA ?
            candidate * BLOCK_N + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // Grouping on N
            candidate * BLOCK_M + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // Grouping on M
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kEnsureZeroPadding = true,
          uint32_t kKAlignment = 128u,     // psum k-group start alignment
          uint32_t kSFKSpan = 128u,        // K covered by one k-grouped SF row
          uint32_t kNumLocalityDomains = 1u,
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {
    // With 2 locality domains, B is localized (N half `d` of every group in domain `d`), and the clusters of domain `d`
    // only take the blocks of N half `d`. `cluster_locality` (from the host): `[d]` is the number of clusters of domain `d`,
    // `[kNumLocalityDomains + cluster_idx]` is `(domain << 16) | rank` of cluster `blockIdx.x / 2` within its domain.
    // The table partitions the clusters, so every block is computed exactly once wherever the CTAs actually run;
    // the host builds it from the probed cluster placement, so that usually each cluster takes its own domain's blocks.
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or kNumLocalityDomains == 2, "Invalid locality domain count");
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or (kGemmType == GemmType::MGroupedContiguous and kNumMulticast == 2 and kIsMulticastOnA),
                     "Locality domains need m-grouped contiguous GEMMs with 2-CTA clusters on N");
    uint32_t num_local_clusters = 0, local_cluster_rank = 0, locality_domain_idx = 0;

    // A/B group starts must be aligned to whole K blocks. SF rows are packed
    // independently per group and tracked by `current_sf_k_cumsum`.
    DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or kKAlignment % 128 == 0,
                     "K alignment must be a multiple of BLOCK_K (128)");

    int current_iter = -1;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for contiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    // Only used for k-grouped layout
    // NOTES: `current_k_start` is the current group's physical K start offset
    // (always a multiple of `kKAlignment`), maintained by both psum and non-psum paths
    uint32_t current_shape_k, current_k_start = 0, current_sf_k_cumsum = 0;
    // Only used for `KGroupedContiguousWithPsumLayout`
    uint32_t current_k_end = 0;

    // Load the K-group selected by `current_group_idx`.
    CUTLASS_DEVICE void get_next_k_group() {
        if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
            // `grouped_layout[i]` is the psum end offset in K elements.
            const auto next_k_end = static_cast<uint32_t>(grouped_layout[current_group_idx]);
            current_k_start = math::align(current_k_end, kKAlignment);
            current_shape_k = next_k_end - current_k_start;
            current_k_end = next_k_end;
        } else {
            current_k_start += current_shape_k;
            current_shape_k = grouped_layout[current_group_idx];
        }
    }

    // ReSharper disable once CppPossiblyUninitializedMember
    CUTLASS_DEVICE explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                       const uint32_t& shape_k, int* grouped_layout = nullptr,
                                       const int* cluster_locality = nullptr) {
        if constexpr (kNumLocalityDomains > 1) {
            const auto entry = static_cast<uint32_t>(cluster_locality[kNumLocalityDomains + blockIdx.x / kNumMulticast]);
            locality_domain_idx = entry >> 16, local_cluster_rank = entry & 0xffff;
            num_local_clusters = static_cast<uint32_t>(cluster_locality[locality_domain_idx]);
        }
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        current_shape_k = is_k_grouped_contiguous(kGemmType) ? 0 : shape_k;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            this->grouped_layout = grouped_layout;
            current_psum_m = grouped_layout[0];
            num_m_blocks = math::ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
            get_next_k_group();
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        get_swizzled_block_idx(block_idx, num_m_blocks, num_n_blocks, m_block_idx, n_block_idx);
    }

    // The same swizzle over a `num_m_blocks_ x num_n_blocks_` sub-problem
    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, const uint32_t& num_m_blocks_, const uint32_t& num_n_blocks_,
                                               uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        const auto primary_num_blocks = kIsMulticastOnA ? num_n_blocks_ : num_m_blocks_;
        const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks_ : num_n_blocks_;
        const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        // NOTES: for SM90 only, as SM90 can dynamically disable TMA multicast
        // while SM100 uses 2-CTA, which can not be dynamically disabled
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }
#endif

        // Convert to final M/N block indices
        // `kIsMulticastOnA == true` leads to groups on N
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                             const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            const auto offset = kWithGroupOffset ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            auto offset = 0;
            if constexpr (kWithGroupOffset) {
                if constexpr (kIndexType == IndexType::MN) {
                    offset = current_group_idx * shape_dim;
                } else if constexpr (kIndexType == IndexType::K) {
                    offset = current_k_start;
                } else if constexpr (kIndexType == IndexType::SF_K) {
                    offset = current_sf_k_cumsum;
                }
            }
            return offset + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::Batched) {
            // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // For swap A/B and psum layout only
    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
        constexpr uint32_t UMMA_STEP_N = 16;
        DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding)
            return math::align(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1 ? current_psum_m - m_block_idx * BLOCK_M : BLOCK_M, UMMA_STEP_N);
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        if constexpr (kNumLocalityDomains > 1) {
            // Each domain walks the blocks of its N half with its own clusters; the two CTAs of a cluster keep adjacent N blocks
            const auto num_local_n_blocks = num_n_blocks / kNumLocalityDomains;
            const auto local_block_idx = (++ current_iter) * num_local_clusters * kNumMulticast +
                                         local_cluster_rank * kNumMulticast + cute::block_rank_in_cluster();
            if (local_block_idx >= num_m_blocks * num_local_n_blocks)
                return false;
            get_swizzled_block_idx(local_block_idx, num_m_blocks, num_local_n_blocks, m_block_idx, n_block_idx);
            n_block_idx += locality_domain_idx * num_local_n_blocks;
            return true;
        }

        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
            while (true) {
                // Within current group
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;

                // Move to check the next group
                if (++ current_group_idx == kNumGroups)
                    return false;

                // NOTES: `num_m_blocks` varies with the increase of the group index
                last_psum_m = math::align(current_psum_m, BLOCK_M);
                current_psum_m = grouped_layout[current_group_idx];
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M);
            }

            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);

            // NOTES: `last_psum_m` is aligned with block M
            m_block_idx += last_psum_m / BLOCK_M;
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                if (next_block_idx < (current_group_idx + 1) * num_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++;
                if (current_group_idx >= kNumGroups)
                    return false;

                const auto aligned_shape_k = math::align(current_shape_k, kKAlignment);
                current_sf_k_cumsum += math::ceil_div(aligned_shape_k, kSFKSpan);
                get_next_k_group();
            }

            get_swizzled_block_idx(next_block_idx - current_group_idx * num_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            current_group_idx = next_block_idx / num_blocks;
            const auto block_idx = next_block_idx - current_group_idx * num_blocks;
            if constexpr (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            // For SM90 only
            // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or                  // Always aligned on N (constant bypass)
                                num_m_blocks % kNumMulticast == 0 or                  // Always aligned on M (constant bypass)
                                (next_block_idx ^ 1) < num_blocks;                    // Peer CTA in bound
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // For SM90 only
    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                      kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;
            } else {
                const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
                const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
                return group_idx == peer_group_idx;
            }
        }
    }

    // For SM90 only
    // ReSharper disable once CppNotAllPathsReturnValue
    CUTLASS_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        } else {
            // Unreachable 
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm::sched
