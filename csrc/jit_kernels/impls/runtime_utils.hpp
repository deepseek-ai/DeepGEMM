#pragma once

#include <cuda.h>
#include <torch/python.h>

#include "../../utils/math.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

static std::pair<int, int> get_inner_outer_dims(const cute::UMMA::Major& major, const int& k, const int& mn) {
    return major == cute::UMMA::Major::K ? std::make_pair(k, mn) : std::make_pair(mn, k);
}

static int get_non_contiguous_dim(const cute::UMMA::Major& major) {
    return major == cute::UMMA::Major::K ? -2 : -1;
}

static int get_compiled_dim(const int& dim, const char& name, const std::string& compiled_dims) {
    for (const char& c: compiled_dims) {
        if (name == c)
            return dim;
    }
    return 0;
}

static std::string to_string(const cute::UMMA::Major& major) {
    switch (major) {
        case cute::UMMA::Major::K:  return "cute::UMMA::Major::K";
        case cute::UMMA::Major::MN: return "cute::UMMA::Major::MN";
    }
    DG_HOST_UNREACHABLE("Unknown major");
}

static std::string to_string(const GemmType& type) {
    switch (type) {
        case GemmType::Normal:              return "GemmType::Normal";
        case GemmType::MGroupedContiguous:  return "GemmType::MGroupedContiguous";
        case GemmType::MGroupedMasked:      return "GemmType::MGroupedMasked";
        case GemmType::KGroupedContiguous:  return "GemmType::KGroupedContiguous";
    }
    DG_HOST_UNREACHABLE("Unknown GEMM type");
}

static std::string to_string(const at::ScalarType& dtype) {
    switch (dtype) {
        case torch::kInt:           return "int";
        case torch::kFloat:         return "float";
        case torch::kBFloat16:      return "cutlass::bfloat16_t";
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static CUtensorMapDataType aten_dtype_to_tensor_map_dtype(const at::ScalarType& dtype) {
    switch (dtype) {
        case torch::kInt:           return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat:         return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kBFloat16:      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case torch::kFloat8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int& mode) {
    switch (mode) {
        case   0: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case  16: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case  32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case  64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: DG_HOST_UNREACHABLE("Unsupported swizzling mode");
    }
}

static CUtensorMap make_tma_2d_desc(const torch::Tensor& t,
                                    int gmem_inner_dim, int gmem_outer_dim,
                                    int smem_inner_dim, int smem_outer_dim,
                                    const int& gmem_outer_stride,
                                    const int& swizzle_mode) {
    const auto& elem_size = static_cast<int>(t.element_size());
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size), };
    const cuuint32_t elem_strides[2] = {1, 1};
    if (get_env<int>("DG_JIT_DEBUG")) {
        printf("Making TMA desc: global memory: %d %d, shared memory: %d %d, outer stride: %d, swizzle: %d, elem size: %d\n",
               gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
               gmem_outer_stride, swizzle_mode, elem_size);
    }
    DG_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type()),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

static CUtensorMap make_tma_a_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_m, const int& shape_k,
                                   const int& block_m, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode) {
    if (num_groups > 1)
        DG_HOST_ASSERT(major == cute::UMMA::Major::K);
    const auto& [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_m * num_groups);
    const auto& [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_m);
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode);
}

static CUtensorMap make_tma_b_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_n, const int& shape_k,
                                   const int& block_n, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode) {
    const auto& [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_n);
    const auto& [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_n);

    // `num_groups` is always applied into the outer dimensions
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim * num_groups,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode);
}

static CUtensorMap make_tma_cd_desc(const torch::Tensor& t,
                                    const int& shape_m, const int& shape_n,
                                    const int& block_m, const int& block_n,
                                    const int& outer_stride,
                                    const int& num_groups,
                                    const int& swizzle_mode) {

    // Swizzling requires the inner box dim to be less or equal than `kSwizzleCDMode`
    // bytes, so `BLOCK_N * sizeof(T) / kSwizzleCDMode` TMA stores are required
    return make_tma_2d_desc(t,
                            shape_n, shape_m * num_groups,
                            block_n, block_m,
                            outer_stride,
                            swizzle_mode);
}

static CUtensorMap make_tma_sf_desc(const cute::UMMA::Major& major,
                                    const torch::Tensor& t,
                                    int shape_mn, int shape_k,
                                    const int& block_mn, const int& block_k,
                                    const int& num_groups,
                                    const int& swizzle_mode) {
    DG_HOST_ASSERT(major == cute::UMMA::Major::MN);

    // TODO: maybe swizzle SF as well
    DG_HOST_ASSERT(swizzle_mode == 0);

    shape_mn = get_tma_aligned_size(shape_mn, static_cast<int>(t.element_size()));
    return make_tma_2d_desc(t,
                            shape_mn, ceil_div(shape_k, block_k * (t.scalar_type() == torch::kFloat ? 1 : 4)) * num_groups,
                            block_mn, 1,
                            shape_mn,
                            swizzle_mode);
}

} // namespace deep_gemm
