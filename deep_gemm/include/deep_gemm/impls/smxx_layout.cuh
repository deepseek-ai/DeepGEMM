#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K,
          uint32_t PADDED_SF_K = SF_K + (1 - (SF_K % 2))>
CUTLASS_GLOBAL void transpose_fp32(const float* sf, float* out, const uint32_t mn) {
    typedef typename utils::Vectorized<sizeof(float) * SF_K>::vec_t in_vec_t;
    constexpr static uint32_t kNumElemsPerVec = sizeof(in_vec_t) / sizeof(float);
    constexpr static uint32_t SF_VEC_K = SF_K / kNumElemsPerVec;

    // Shapes and strides
    extern __shared__ float smem_buffer[];
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(float));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = math::align<uint32_t>(mn, kNumTMAAlignedElems);

    // Shift into the block
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * SF_K;
    const auto& local_sf = reinterpret_cast<const in_vec_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Load
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_VEC_K; i += kNumThreads) {
        auto in_vec = local_sf[i];
        const auto& in_values = reinterpret_cast<float*>(&in_vec);

        const auto& row = i / SF_VEC_K, col = (i % SF_VEC_K) * kNumElemsPerVec;
        #pragma unroll
        for (uint32_t j = 0; j < kNumElemsPerVec; ++ j)
            smem_buffer[row * PADDED_SF_K + col + j] = in_values[j];
    }
    __syncthreads();

    // Store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < in_block_mn * SF_K; i += kNumThreads) {
        const auto& sf_k_idx = i / in_block_mn, mn_idx = i % in_block_mn;
        const auto& global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx;
        out[sf_k_idx * tma_aligned_mn + global_mn_idx] = ptx::ld_shared(smem_buffer + mn_idx * PADDED_SF_K + sf_k_idx);
    }
}

// NOTES: the two kernels below always pack the K dimension

template <uint32_t kNumThreads, uint32_t BLOCK_MN, uint32_t SF_K,
          uint32_t kNumPsumGroups = 1, bool kUsePsumLayout = false>
CUTLASS_GLOBAL void transpose_and_pack_fp32_into_ue8m0(float* sf, uint32_t* out, const uint32_t mn,
                                                       const uint32_t* grouped_layout, const uint32_t m_alignment) {
    extern __shared__ uint32_t smem_buffer[];

    // Shapes and strides
    constexpr auto kNumPackedSFK = math::constexpr_ceil_div(SF_K, 4u);
    constexpr auto kNumTMAAlignedElems = static_cast<uint32_t>(16 / sizeof(int));
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto tma_aligned_mn = math::align<uint64_t>(mn, kNumTMAAlignedElems);

    // Wait for primary kernel completion before reading SFs or PSUM layout
    cudaGridDependencySynchronize();

    constexpr auto kNumPsumLayoutElems = kUsePsumLayout ? math::constexpr_align(kNumPsumGroups * 2, 4u) : 0;
    const auto group_mn_start = smem_buffer;
    const auto group_mn_end = smem_buffer + kNumPsumGroups;
    const auto sf_smem_buffer = smem_buffer + kNumPsumLayoutElems;

    // Precompute PSUM valid MN ranges into smem before the SF tile.
    if constexpr (kUsePsumLayout) {
        for (uint32_t g = threadIdx.x; g < kNumPsumGroups; g += kNumThreads) {
            const auto end_g = grouped_layout[g];
            group_mn_start[g] = g == 0 ? 0 : math::align(grouped_layout[g - 1], m_alignment);
            group_mn_end[g] = end_g;
        }
        __syncthreads();
    }

    // PSUM path skips uninitialized per-group MN gaps without early return.
    const auto is_valid_mn = [&](const uint32_t& global_mn_idx) -> bool {
        if constexpr (kUsePsumLayout) {
            bool valid = false;
            for (uint32_t g = 0; g < kNumPsumGroups; ++ g)
                valid |= (global_mn_idx >= group_mn_start[g] and global_mn_idx < group_mn_end[g]);
            return valid;
        } else {
            return global_mn_idx < mn;
        }
    };

    // Shift into the group
    sf = sf + static_cast<uint64_t>(blockIdx.y) * mn * SF_K;
    out = out + static_cast<uint64_t>(blockIdx.y) * tma_aligned_mn * kNumPackedSFK;

    // Load FP32 SFs
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block size");
    const auto local_sf = reinterpret_cast<uint32_t*>(sf + static_cast<uint64_t>(blockIdx.x) * (BLOCK_MN * SF_K));
    const auto num_values = in_block_mn * SF_K;
    const auto num_uint4 = num_values / 4;
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < num_uint4; i += kNumThreads) {
        const auto& [x, y, z, w] = reinterpret_cast<const uint4*>(local_sf)[i];
        ptx::st_shared(reinterpret_cast<uint4*>(sf_smem_buffer) + i, x, y, z, w);
    }

    // Fill unaligned values as well
    if (const auto unaligned_idx = num_uint4 * 4 + threadIdx.x; unaligned_idx < num_values)
        ptx::st_shared(sf_smem_buffer + unaligned_idx, local_sf[unaligned_idx]);
    __syncthreads();

    // Pack into UE8M0 and store
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < (kNumPackedSFK * BLOCK_MN); i += kNumThreads) {
        const auto sf_k_pack_idx = i / BLOCK_MN, mn_idx = i % BLOCK_MN;
        const auto global_mn_idx = blockIdx.x * BLOCK_MN + mn_idx;
        const auto in_bounds_mn = global_mn_idx < mn;
        const auto valid_mn = is_valid_mn(global_mn_idx);

        // Load shared memory
        uint32_t values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            const auto sf_k_idx = sf_k_pack_idx * 4 + j;
            values[j] = valid_mn and sf_k_idx < SF_K ? ptx::ld_shared(sf_smem_buffer + mn_idx * SF_K + sf_k_idx) : 0;
            // FP32 SFs must have a zero sign and mantissa (only the exponent is packed)
            DG_DEVICE_ASSERT((values[j] & 0x807fffffu) == 0);
        }

        // Pack and store
        uint32_t packed = 0;
        packed |= (values[0] >> 23u);
        packed |= (values[1] >> 15u);
        packed |= (values[2] >>  7u);
        packed |= (values[3] <<  1u);
        // Write safe finite scale codes for PSUM gap rows; UE8M0 0xff is NaN.
        if (in_bounds_mn)
            out[sf_k_pack_idx * tma_aligned_mn + global_mn_idx] = packed;
    }
}

template <uint32_t kNumGroups, uint32_t kNumThreads,
          uint32_t BLOCK_MN, uint32_t BLOCK_PACKED_SF_K, bool kTransposed = true, bool kUsePsumLayout = false>
CUTLASS_GLOBAL void pack_fp32_into_ue8m0(float* sf, uint32_t* out, uint32_t* grouped_layout,
                                         const uint32_t mn, const uint32_t sf_k, const uint32_t packed_sf_k,
                                         const uint32_t gran_k, const uint32_t k_alignment) {
    // Always packing the K dimension
    // NOTES: should also assert `mn % 4 == 0` at launch
    // psum layout may have input gaps, but packed output remains per-group compact.
    DG_STATIC_ASSERT(kTransposed, "Currently only support transposed SFs (MN-major)");
    DG_STATIC_ASSERT(BLOCK_MN % 4 == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_PACKED_SF_K == kNumThreads / 32, "Invalid block sizes");

    // Shapes and strides
    const auto in_block_mn = min(BLOCK_MN, mn - blockIdx.x * BLOCK_MN);
    const auto in_block_mn_uint4 = in_block_mn / 4;
    const auto in_block_packed_sf_k = min(BLOCK_PACKED_SF_K, packed_sf_k - blockIdx.y * BLOCK_PACKED_SF_K);

    // Shift into the right block along MN
    sf += blockIdx.x * BLOCK_MN;
    out += blockIdx.x * BLOCK_MN;

    // Each warp is responsible for a packed row
    const auto warp_idx = threadIdx.x / 32;
    const auto lane_idx = ptx::get_lane_idx();
    const auto packed_sf_k_idx = static_cast<uint64_t>(blockIdx.y) * BLOCK_PACKED_SF_K + warp_idx;
    if (warp_idx >= in_block_packed_sf_k)
        return;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Find the owner group of this warp's packed row and its input SF row range
    uint32_t num_padding_sf_rows = 0;
    uint32_t group_sf_row_start = 0, group_sf_row_end = sf_k;
    if constexpr (kNumGroups > 1 or kUsePsumLayout) {
        // Layout entries are group K sizes, or psum end offsets in psum mode
        DG_STATIC_ASSERT(kNumGroups <= 128, "Too many groups");
        uint32_t layout_cache[4];
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++ i) {
            const auto group_idx = lane_idx * 4 + i;
            layout_cache[i] = group_idx < kNumGroups ? grouped_layout[group_idx] : 0;
        }
        __syncwarp();

        uint32_t num_prefix_sf_rows = 0;
        uint32_t num_prefix_packed_rows = 0;
        uint32_t prev_group_end = 0;
        bool owner_group_found = false;
        #pragma unroll
        for (uint32_t group_idx = 0; group_idx < kNumGroups; ++ group_idx) {
            const auto layout_value = __shfl_sync(0xffffffff, layout_cache[group_idx % 4], group_idx / 4);
            uint32_t group_k;
            if constexpr (kUsePsumLayout) {
                group_k = layout_value - math::align(prev_group_end, k_alignment);
            } else {
                group_k = layout_value;
            }
            const auto num_group_sf_rows = math::ceil_div(group_k, gran_k);
            group_sf_row_start = num_prefix_sf_rows;
            group_sf_row_end = group_sf_row_start + num_group_sf_rows;
            num_prefix_sf_rows += num_group_sf_rows;
            num_prefix_packed_rows += math::ceil_div(num_group_sf_rows, 4u);
            if (packed_sf_k_idx < num_prefix_packed_rows) {
                owner_group_found = true;
                break;
            }
            if (const auto remainder = num_group_sf_rows % 4; remainder > 0)
                num_padding_sf_rows += 4 - remainder;
            if constexpr (kUsePsumLayout)
                prev_group_end = layout_value;
        }
        if (not owner_group_found)
            return;
    }

    for (uint32_t mn_idx = ptx::get_lane_idx(); mn_idx < in_block_mn_uint4; mn_idx += 32) {
        // Load
        uint4 values[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            values[j] = make_uint4(0, 0, 0, 0);
            const uint32_t sf_row_idx = packed_sf_k_idx * 4 + j - num_padding_sf_rows;
            if (sf_row_idx >= group_sf_row_start and sf_row_idx < group_sf_row_end)
                values[j] = reinterpret_cast<const uint4*>(sf + sf_row_idx * mn)[mn_idx];
            // FP32 SFs must have a zero sign and mantissa (only the exponent is packed)
            DG_DEVICE_ASSERT((values[j].x & 0x807fffffu) == 0);
            DG_DEVICE_ASSERT((values[j].y & 0x807fffffu) == 0);
            DG_DEVICE_ASSERT((values[j].z & 0x807fffffu) == 0);
            DG_DEVICE_ASSERT((values[j].w & 0x807fffffu) == 0);
        }

        // Pack and store
        uint4 packed;
        packed.x = (values[0].x >> 23u) | (values[1].x >> 15u) | (values[2].x >> 7u) | (values[3].x << 1u);
        packed.y = (values[0].y >> 23u) | (values[1].y >> 15u) | (values[2].y >> 7u) | (values[3].y << 1u);
        packed.z = (values[0].z >> 23u) | (values[1].z >> 15u) | (values[2].z >> 7u) | (values[3].z << 1u);
        packed.w = (values[0].w >> 23u) | (values[1].w >> 15u) | (values[2].w >> 7u) | (values[3].w << 1u);
        reinterpret_cast<uint4*>(out + packed_sf_k_idx * mn)[mn_idx] = packed;
    }
}

} // namespace deep_gemm
