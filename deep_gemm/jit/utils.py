import ctypes
from enum import Enum
from typing import Any, Dict, Tuple

import cuda.bindings.driver as cuda
import torch


class Layout(Enum):
    RowMajor = 0
    ColMajor = 1


class GemmType(Enum):
    Normal = 0
    GroupedContiguous = 1
    GroupedMasked = 2
    
    def __str__(self) -> str:
        return {
            0: 'Normal',
            1: 'GroupedContiguous',
            2: 'GroupedMasked',
        }[self.value]


typename_map: Dict[Any, str] = {
    torch.int8: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.int32: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.uint8: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.uint32: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    torch.uint64: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    torch.float32: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e4m3fnuz: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2fnuz: cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_map = {
    128: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
    64: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    32: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    0: cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
}

def get_num_math_warpgroups(block_m: int) -> int:
    return 1 if block_m == 64 else 2

def get_num_threads_per_sm(num_tma_threads: int, num_math_threads_per_group: int, block_m: int) -> int:
    assert num_math_threads_per_group == 128, "Only support 128 threads per math group"
    return get_num_math_warpgroups(block_m) * num_math_threads_per_group + num_tma_threads


def make_2d_tma_copy_desc(global_address: torch.Tensor, gmem_dim: Tuple[cuda.cuuint64_t, cuda.cuuint64_t], stride_in_bytes: cuda.cuuint64_t, smem_dim: Tuple[cuda.cuuint32_t, cuda.cuuint32_t], swizzle_type: cuda.CUtensorMapSwizzle) -> cuda.CUtensorMap:
    tensor_dtype = typename_map[global_address.dtype]
    res, tensor_map = cuda.cuTensorMapEncodeTiled(
        tensor_dtype,
        2,  # tensor rank
        global_address.data_ptr(),
        gmem_dim,
        (stride_in_bytes,),  # global strides
        smem_dim,
        (cuda.cuuint32_t(1), cuda.cuuint32_t(1)),  # element strides
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )

    if res != cuda.CUresult.CUDA_SUCCESS:
        raise Exception(f"Failed to encode tensor map: {res}")

    return tensor_map


def make_2d_tma_desc(global_address: torch.Tensor, layout: Layout, gmem_rows: int, gmem_cols: int, smem_rows: int, smem_cols: int, swizzle_type: cuda.CUtensorMapSwizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B) -> cuda.CUtensorMap:
    if layout == Layout.RowMajor:
        gmem_dim = (cuda.cuuint64_t(gmem_cols), cuda.cuuint64_t(gmem_rows))
        smem_dim = (cuda.cuuint32_t(smem_cols), cuda.cuuint32_t(smem_rows))
        return make_2d_tma_copy_desc(global_address, gmem_dim, cuda.cuuint64_t(gmem_cols * global_address.element_size()), smem_dim, swizzle_type)
    else:
        gmem_dim = (cuda.cuuint64_t(gmem_rows), cuda.cuuint64_t(gmem_cols))
        smem_dim = (cuda.cuuint32_t(smem_rows), cuda.cuuint32_t(smem_cols))
        return make_2d_tma_copy_desc(global_address, gmem_dim, cuda.cuuint64_t(gmem_rows * global_address.element_size()), smem_dim, swizzle_type)


def make_2d_tma_a_desc(gemm_type: GemmType, global_address: torch.Tensor, shape_m: int, shape_k: int, block_m: int, block_k: int, num_groups: int = 1) -> cuda.CUtensorMap:
    return make_2d_tma_desc(global_address, Layout.RowMajor, shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1), shape_k, block_m, block_k)


def make_2d_tma_b_desc(gemm_type: GemmType, global_address: torch.Tensor, shape_k: int, shape_n: int, block_k: int, block_n: int, num_groups: int = 1) -> cuda.CUtensorMap:
    return make_2d_tma_desc(global_address, Layout.ColMajor, shape_k, shape_n * (num_groups if gemm_type != GemmType.Normal else 1), block_k, block_n)


def make_2d_tma_d_desc(gemm_type: GemmType, swizzle_mode: int, global_address: torch.Tensor, shape_m: int, shape_n: int, block_m: int, block_n: int, num_groups: int = 1) -> cuda.CUtensorMap:
    # Swizzling requires the inner box dim less or equal than `kSwizzleDMode`
    # bytes So `BLOCK_N * sizeof(T) / kSwizzleDMode` TMA stores are required
    return make_2d_tma_desc(global_address, Layout.RowMajor, shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1), shape_n, block_m, block_n if swizzle_mode == 0 else swizzle_mode // global_address.element_size(), swizzle_map[swizzle_mode])


def make_2d_tma_scales_a_desc(gemm_type: GemmType, global_address: torch.Tensor, shape_m: int, shape_k: int, block_m: int, block_k: int, num_groups: int = 1) -> cuda.CUtensorMap:
    # Make TMA aligned to 16 bytes
    kAlignment = 16 / global_address.element_size()
    shape_m = (shape_m + kAlignment - 1) // kAlignment * kAlignment

    return make_2d_tma_desc(global_address, Layout.ColMajor, shape_m, (shape_k + block_k - 1) // block_k * (num_groups if gemm_type == GemmType.GroupedMasked else 1), block_m, 1, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE)


def run_gemm(kernel: cuda.CUkernel, num_tma_multicast: int, shape_m: int, block_m: int, gmem_d: torch.Tensor, scales_b: torch.Tensor, grouped_layout: torch.Tensor, num_sms: int, smem_size: int, tensor_map_a: cuda.CUtensorMap, tensor_map_b: cuda.CUtensorMap, tensor_map_scales_a: cuda.CUtensorMap, tensor_map_d: cuda.CUtensorMap, stream: cuda.CUstream) -> cuda.CUresult:
    num_tma_threads = 128
    num_math_threads_per_group = 128
    
    res = cuda.cuKernelSetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size, kernel, cuda.CUdevice(gmem_d.device.index))[0]
    if res != cuda.CUresult.CUDA_SUCCESS:
        raise Exception(f"Failed to set max dynamic shared memory size: {res}")

    attr_val = cuda.CUlaunchAttributeValue()
    attr_val.clusterDim.x = num_tma_multicast
    attr_val.clusterDim.y = 1
    attr_val.clusterDim.z = 1
    attr = cuda.CUlaunchAttribute()
    attr.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attr.value = attr_val
    
    config = cuda.CUlaunchConfig()
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
    
    kernelValues = (
        gmem_d.data_ptr(),
        scales_b.data_ptr(),
        grouped_layout.data_ptr(),
        shape_m,
        tensor_map_a,
        tensor_map_b,
        tensor_map_scales_a,
        tensor_map_d,
    )
    kernelTypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        None,
        None,
        None,
        None,
    )
    
    return cuda.cuLaunchKernelEx(config, kernel, (kernelValues, kernelTypes), 0)
