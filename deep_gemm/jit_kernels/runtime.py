import torch
import cuda.bindings.driver as cbd
from typing import Any, Dict, Tuple

from ..utils.math import ceil_div
from ..utils.layout import get_tma_aligned_size, GemmType, MajorTypeAB, MajorTypeCD


tmap_type_map: Dict[Any, str] = {
    torch.int8:            cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int16:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.int32:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.uint8:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint16:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.uint32:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    torch.uint64:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    torch.float32:         cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float16:         cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16:        cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn:   cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e4m3fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2:     cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_type_map = {
    0:   cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    16:  cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32:  cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64:  cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}

def make_tma_xd_desc(t: torch.Tensor,
                     gmem_dims: Tuple[cbd.cuuint64_t, ...], gmem_strides: Tuple[cbd.cuuint64_t, ...],
                     smem_dims: Tuple[cbd.cuuint32_t, ...],
                     swizzle_type: cbd.CUtensorMapSwizzle) -> cbd.CUtensorMap:
    num_dims = len(gmem_dims)
    assert len(gmem_strides) == num_dims - 1
    assert len(smem_dims) == num_dims

    tensor_dtype = tmap_type_map[t.dtype]
    res, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        num_dims,
        t.data_ptr(),
        gmem_dims,
        gmem_strides,
        smem_dims,
        (cbd.cuuint32_t(1), ) * num_dims,
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type,
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )

    if res != cbd.CUresult.CUDA_SUCCESS:
        raise Exception(f'Failed to encode tensor map: {res}')
    return tensor_map


def make_tma_2d_desc(t: torch.Tensor,
                     gmem_inner_dim: int, gmem_outer_dim: int,
                     smem_inner_dim: int, smem_outer_dim: int,
                     gmem_outer_stride: int,
                     swizzle_mode: int) -> cbd.CUtensorMap:
    # For swizzling pattern, multiple TMAs are required
    if swizzle_mode != 0:
        assert swizzle_mode % t.element_size() == 0
        smem_inner_dim = swizzle_mode // t.element_size()

    gmem_dims = (cbd.cuuint64_t(gmem_inner_dim), cbd.cuuint64_t(gmem_outer_dim))
    gmem_strides = (cbd.cuuint64_t(gmem_outer_stride * t.element_size()), )
    smem_dims = (cbd.cuuint32_t(smem_inner_dim), cbd.cuuint32_t(smem_outer_dim))
    return make_tma_xd_desc(t, gmem_dims, gmem_strides, smem_dims, swizzle_type_map[swizzle_mode])


def make_tma_a_desc(major_type: MajorTypeAB,
                    t: torch.Tensor,
                    shape_m: int, shape_k: int,
                    block_m: int, block_k: int,
                    outer_stride: int,
                    num_groups: int,
                    swizzle_mode: int) -> cbd.CUtensorMap:
    if num_groups > 1:
        assert major_type == MajorTypeAB.KMajor
    return make_tma_2d_desc(t,
                            *(shape_k, shape_m * num_groups)[::major_type.shape_direction()],
                            *(block_k, block_m)[::major_type.shape_direction()],
                            outer_stride,
                            swizzle_mode)


def make_tma_b_desc(major_type: MajorTypeAB,
                    t: torch.Tensor,
                    shape_n: int, shape_k: int,
                    block_n: int, block_k: int,
                    outer_stride: int,
                    num_groups: int,
                    swizzle_mode: int) -> cbd.CUtensorMap:
    # `num_groups` is always applied into the outer dimensions
    io_shapes = (shape_k, shape_n)[::major_type.shape_direction()]
    io_shapes = (io_shapes[0], io_shapes[1] * num_groups)

    return make_tma_2d_desc(t,
                            *io_shapes,
                            *(block_k, block_n)[::major_type.shape_direction()],
                            outer_stride,
                            swizzle_mode)


def make_tma_cd_desc(major_type: MajorTypeCD,
                     t: torch.Tensor,
                     shape_m: int, shape_n: int,
                     block_m: int, block_n: int,
                     outer_stride: int,
                     num_groups: int,
                     swizzle_mode: int) -> cbd.CUtensorMap:
    assert major_type == MajorTypeCD.NMajor

    # Swizzling requires the inner box dim to be less or equal than `kSwizzleCDMode`
    # bytes, so `BLOCK_N * sizeof(T) / kSwizzleCDMode` TMA stores are required
    layout_ad_m = 128
    return make_tma_2d_desc(t,
                            shape_n, shape_m * num_groups,
                            block_n, min(block_m, layout_ad_m),
                            outer_stride,
                            swizzle_mode)


def make_tma_sf_desc(major_type: MajorTypeAB,
                     t: torch.Tensor,
                     shape_mn: int, shape_k: int,
                     block_mn: int, block_k: int,
                     num_groups: int,
                     swizzle_mode: int) -> cbd.CUtensorMap:
    assert major_type == MajorTypeAB.MNMajor

    # TODO: maybe swizzle SF as well
    assert swizzle_mode == 0

    # Make TMA aligned to 16 bytes
    shape_mn = get_tma_aligned_size(shape_mn, t.element_size())
    return make_tma_2d_desc(t,
                            shape_mn, ceil_div(shape_k, block_k * 4) * num_groups,
                            block_mn, 1,
                            shape_mn,
                            swizzle_mode)
