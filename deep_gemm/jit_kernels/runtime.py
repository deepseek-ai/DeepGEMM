import ctypes
import os
import enum
import torch
import cuda.bindings.driver as cbd
from typing import Any, Dict, Tuple

from ..jit.runtime import Runtime


class Layout(enum.Enum):
    RowMajor = 0
    ColMajor = 1


class GemmType(enum.Enum):
    Normal = 0
    GroupedContiguous = 1
    GroupedMasked = 2

    def __str__(self) -> str:
        return {
            0: 'Normal',
            1: 'GroupedContiguous',
            2: 'GroupedMasked',
        }[self.value]


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
    32:  cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64:  cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


def get_num_math_warpgroups(block_m: int) -> int:
    return 1 if block_m == 64 else 2


def get_num_threads_per_sm(num_tma_threads: int, num_math_threads_per_group: int, block_m: int) -> int:
    assert num_math_threads_per_group == 128, 'Only support 128 threads per math group'
    return get_num_math_warpgroups(block_m) * num_math_threads_per_group + num_tma_threads


def make_2d_tma_copy_desc(global_address: torch.Tensor,
                          gmem_dim: Tuple[cbd.cuuint64_t, cbd.cuuint64_t],
                          stride_in_bytes: cbd.cuuint64_t,
                          smem_dim: Tuple[cbd.cuuint32_t, cbd.cuuint32_t],
                          swizzle_type: cbd.CUtensorMapSwizzle) -> cbd.CUtensorMap:
    tensor_dtype = tmap_type_map[global_address.dtype]
    res, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        2,
        global_address.data_ptr(),
        gmem_dim,
        (stride_in_bytes, ),
        smem_dim,
        (cbd.cuuint32_t(1), cbd.cuuint32_t(1)),
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type,
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )

    if res != cbd.CUresult.CUDA_SUCCESS:
        raise Exception(f'Failed to encode tensor map: {res}')
    return tensor_map


def make_2d_tma_desc(global_address: torch.Tensor, layout: Layout,
                     gmem_rows: int, gmem_cols: int,
                     smem_rows: int, smem_cols: int,
                     swizzle_type: cbd.CUtensorMapSwizzle = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B) -> cbd.CUtensorMap:
    if layout == Layout.RowMajor:
        gmem_dim = (cbd.cuuint64_t(gmem_cols), cbd.cuuint64_t(gmem_rows))
        smem_dim = (cbd.cuuint32_t(smem_cols), cbd.cuuint32_t(smem_rows))
        return make_2d_tma_copy_desc(global_address, gmem_dim, cbd.cuuint64_t(gmem_cols * global_address.element_size()), smem_dim, swizzle_type)
    else:
        gmem_dim = (cbd.cuuint64_t(gmem_rows), cbd.cuuint64_t(gmem_cols))
        smem_dim = (cbd.cuuint32_t(smem_rows), cbd.cuuint32_t(smem_cols))
        return make_2d_tma_copy_desc(global_address, gmem_dim, cbd.cuuint64_t(gmem_rows * global_address.element_size()), smem_dim, swizzle_type)


def make_2d_tma_a_desc(gemm_type: GemmType, global_address: torch.Tensor,
                       shape_m: int, shape_k: int,
                       block_m: int, block_k: int,
                       num_groups: int) -> cbd.CUtensorMap:
    return make_2d_tma_desc(global_address, Layout.RowMajor,
                            shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1), shape_k,
                            block_m, block_k)


def make_2d_tma_b_desc(gemm_type: GemmType, global_address: torch.Tensor,
                       shape_k: int, shape_n: int,
                       block_k: int, block_n: int,
                       num_groups: int) -> cbd.CUtensorMap:
    return make_2d_tma_desc(global_address, Layout.ColMajor,
                            shape_k, shape_n * (num_groups if gemm_type != GemmType.Normal else 1),
                            block_k, block_n)


def make_2d_tma_d_desc(gemm_type: GemmType, global_address: torch.Tensor,
                       shape_m: int, shape_n: int,
                       block_m: int, block_n: int,
                       num_groups: int, swizzle_mode: int) -> cbd.CUtensorMap:
    # Swizzling requires the inner box dim to be less or equal than `kSwizzleDMode`
    # bytes, so `BLOCK_N * sizeof(T) / kSwizzleDMode` TMA stores are required
    return make_2d_tma_desc(global_address, Layout.RowMajor,
                            shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1), shape_n,
                            block_m, block_n if swizzle_mode == 0 else swizzle_mode // global_address.element_size(),
                            swizzle_type_map[swizzle_mode])


def make_2d_tma_scales_a_desc(gemm_type: GemmType, global_address: torch.Tensor, shape_m: int, shape_k: int, block_m: int, block_k: int, num_groups: int = 1) -> cbd.CUtensorMap:
    # Make TMA aligned to 16 bytes
    tma_alignment = 16 / global_address.element_size()
    shape_m = (shape_m + tma_alignment - 1) // tma_alignment * tma_alignment

    return make_2d_tma_desc(global_address, Layout.ColMajor,
                            shape_m, (shape_k + block_k - 1) // block_k * (num_groups if gemm_type == GemmType.GroupedMasked else 1),
                            block_m, 1, cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE)


class FP8GemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path, [
            'NUM_TMA_MULTICAST',
            'M',
            'BLOCK_M',
            'GMEM_D',
            'SCALES_B',
            'GROUPED_LAYOUT',
            'NUM_SMS',
            'SMEM_SIZE',
            'TENSOR_MAP_A',
            'TENSOR_MAP_B',
            'TENSOR_MAP_SCALES_A',
            'TENSOR_MAP_D',
            'STREAM',
        ])

    @staticmethod
    def generate(**kwargs) -> str:
        code = f'''
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/fp8_gemm.cuh>

using namespace deep_gemm;

auto ptr = reinterpret_cast<void*>(&fp8_gemm_kernel<
    {kwargs['N']},
    {kwargs['K']},
    {kwargs['BLOCK_M']},
    {kwargs['BLOCK_N']},
    {kwargs['BLOCK_K']},
    {kwargs['BLOCK_N_PADDING']},
    {kwargs['SWIZZLE_D_MODE']},
    {kwargs['NUM_GROUPS']},
    {kwargs['NUM_STAGES']},
    {kwargs['NUM_TMA_THREADS']},
    {kwargs['NUM_MATH_THREADS_PER_GROUP']},
    {kwargs['NUM_TMA_MULTICAST']},
    {'true' if kwargs['IS_TMA_MULTICAST_ON_A'] else 'false'},
    GemmType::{kwargs['GEMM_TYPE']}
  >);
'''
        if int(os.getenv('DG_JIT_DEBUG', 0)):
            print(f'Generated FP8 GEMM code:\n{code}')
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, num_tma_multicast: int, shape_m: int,
               block_m: int, gmem_d: torch.Tensor, scales_b: torch.Tensor,
               grouped_layout: torch.Tensor, num_sms: int, smem_size: int,
               tensor_map_a: cbd.CUtensorMap, tensor_map_b: cbd.CUtensorMap,
               tensor_map_scales_a: cbd.CUtensorMap, tensor_map_d: cbd.CUtensorMap,
               stream: cbd.CUstream) -> cbd.CUresult:
        num_tma_threads = 128
        num_math_threads_per_group = 128

        res = cbd.cuKernelSetAttribute(cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size, kernel, cbd.CUdevice(gmem_d.device.index))[0]
        if res != cbd.CUresult.CUDA_SUCCESS:
            raise Exception(f'Failed to set max dynamic shared memory size: {res}')

        attr_val = cbd.CUlaunchAttributeValue()
        attr_val.clusterDim.x = num_tma_multicast
        attr_val.clusterDim.y = 1
        attr_val.clusterDim.z = 1
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value = attr_val

        config = cbd.CUlaunchConfig()
        config.numAttrs = 1
        config.attrs = [attr]
        config.gridDimX = num_sms
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = get_num_threads_per_sm(num_tma_threads, num_math_threads_per_group, block_m)
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = smem_size
        config.hStream = stream

        arg_values = (
            gmem_d.data_ptr(),
            scales_b.data_ptr(),
            grouped_layout.data_ptr(),
            shape_m,
            tensor_map_a,
            tensor_map_b,
            tensor_map_scales_a,
            tensor_map_d,
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            None,
            None,
            None,
            None,
        )
        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)
