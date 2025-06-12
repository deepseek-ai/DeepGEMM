import ctypes
import os
import torch
import cuda.bindings.driver as cbd
from typing import Any, Dict, Optional

from ..runtime import (
    make_tma_a_desc, make_tma_b_desc,
    make_tma_cd_desc, make_tma_sf_desc
)
from ..heuristics.sm100_heuristics import get_best_configs
from ...config import get_num_sms
from ...jit import Runtime, build, pytypes_to_ctypes
from ...utils.math import align, ceil_div
from ...utils.layout import GemmType, MajorTypeAB, MajorTypeCD


class SM100FP8GemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(kwargs: Dict[str, Any]) -> str:
        assert kwargs['CD_DTYPE_T'] in (torch.bfloat16, torch.float)
        code = f'''
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_1d1d_impl<
        {kwargs['MAJOR_A']},
        {kwargs['MAJOR_B']},
        {kwargs['M'] if 'm' in kwargs['COMPILED_DIMS'] else 0},
        {kwargs['N'] if 'n' in kwargs['COMPILED_DIMS'] else 0},
        {kwargs['K'] if 'k' in kwargs['COMPILED_DIMS'] else 0},
        {kwargs['BLOCK_M']},
        {kwargs['BLOCK_N']},
        {kwargs['BLOCK_K']},
        {kwargs['NUM_GROUPS']},
        {kwargs['SWIZZLE_A_MODE']},
        {kwargs['SWIZZLE_B_MODE']},
        {kwargs['SWIZZLE_CD_MODE']},
        {kwargs['NUM_STAGES']},
        {kwargs['NUM_LAST_STAGES']},
        {kwargs['NUM_NON_EPILOGUE_THREADS']},
        {kwargs['NUM_EPILOGUE_THREADS']},
        {kwargs['NUM_MULTICAST']},
        {pytypes_to_ctypes[kwargs['IS_MULTICAST_ON_A']]},
        {kwargs['GEMM_TYPE']},
        {pytypes_to_ctypes[kwargs['WITH_ACCUMULATION']]},
        {pytypes_to_ctypes[kwargs['CD_DTYPE_T']]}
      >);
}};
'''
        if int(os.getenv('DG_JIT_DEBUG', 0)):
            print(f'Generated FP8 GEMM code:\n{code}')
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: Dict[str, Any]) -> cbd.CUresult:
        result = cbd.cuKernelSetAttribute(cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          kwargs['SMEM_SIZE'], kernel, cbd.CUdevice(kwargs['DEVICE_INDEX']))[0]
        assert result == cbd.CUresult.CUDA_SUCCESS, f'Failed to set max dynamic shared memory size: {result}'

        attr_val = cbd.CUlaunchAttributeValue()
        attr_val.clusterDim.x = kwargs['NUM_MULTICAST']
        attr_val.clusterDim.y = 1
        attr_val.clusterDim.z = 1
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value = attr_val

        config = cbd.CUlaunchConfig()
        config.numAttrs = 1
        config.attrs = [attr]
        config.gridDimX = kwargs['NUM_SMS']
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = kwargs['NUM_NON_EPILOGUE_THREADS'] + kwargs['NUM_EPILOGUE_THREADS']
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = kwargs['SMEM_SIZE']
        config.hStream = kwargs['STREAM']

        arg_values = (
            kwargs['GROUPED_LAYOUT'].data_ptr(),
            kwargs['M'],
            kwargs['N'],
            kwargs['K'],
            kwargs['TENSOR_MAP_A'],
            kwargs['TENSOR_MAP_B'],
            kwargs['TENSOR_MAP_SFA'],
            kwargs['TENSOR_MAP_SFB'],
            kwargs['TENSOR_MAP_C'],
            kwargs['TENSOR_MAP_D'],
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            None, None, None, None, None, None
        )
        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)


def fp8_gemm_nt(a: torch.Tensor, sfa: torch.Tensor,
                b: torch.Tensor, sfb: torch.Tensor,
                c: Optional[torch.Tensor], d: torch.Tensor,
                major_a: MajorTypeAB, major_b: MajorTypeAB,
                major_cd: MajorTypeCD,
                compiled_dims: str) -> None:
    m, k = a.shape
    n, _ = b.shape
    assert major_cd == MajorTypeCD.NMajor

    # K must be aligned to 128
    aligned_k = align(k, 128)

    num_sms = get_num_sms()
    num_sms, block_m, block_n, block_k, num_stages, multicast_config, smem_config = get_best_configs(
        GemmType.Normal, m, n, k, 1, major_a, major_b, major_cd, torch.float8_e4m3fn, d.dtype, num_sms)

    num_groups = 1
    tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                   multicast_config.get_ab_load_block_m(block_m), block_k,
                                   a.stride(major_a.non_contiguous_dim()), num_groups,
                                   smem_config.swizzle_a_mode)
    tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                   multicast_config.get_ab_load_block_n(block_n), block_k,
                                   b.stride(major_b.non_contiguous_dim()), num_groups,
                                   smem_config.swizzle_b_mode)
    tensor_map_d = make_tma_cd_desc(major_cd, d, m, n,
                                    block_m, block_n,
                                    d.stride(major_cd.non_contiguous_dim()), num_groups,
                                    smem_config.swizzle_cd_mode)
    tensor_map_c = make_tma_cd_desc(major_cd, c, m, n,
                                    block_m, block_n,
                                    c.stride(major_cd.non_contiguous_dim()), num_groups,
                                    smem_config.swizzle_cd_mode) if c is not None else tensor_map_d
    tensor_map_sfa = make_tma_sf_desc(MajorTypeAB.MNMajor, sfa, m, k, block_m, block_k, num_groups, smem_config.swizzle_sf_mode)
    tensor_map_sfb = make_tma_sf_desc(MajorTypeAB.MNMajor, sfb, n, k, block_n, block_k, num_groups, smem_config.swizzle_sf_mode)

    kwargs = {
        # Templated or runtime arguments according to the `COMPILED_DIMS`
        'COMPILED_DIMS': compiled_dims,
        'M': m, 'N': n, 'K': aligned_k,
        # Templated arguments
        'GEMM_TYPE': GemmType.Normal,
        'NUM_NON_EPILOGUE_THREADS': 128,
        'NUM_EPILOGUE_THREADS': 128,
        'MAJOR_A': major_a,
        'MAJOR_B': major_b,
        'NUM_GROUPS': 1,
        'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
        'NUM_STAGES': num_stages, 'NUM_LAST_STAGES': ceil_div(k, block_k) % num_stages,
        'SWIZZLE_A_MODE': smem_config.swizzle_a_mode,
        'SWIZZLE_B_MODE': smem_config.swizzle_b_mode,
        'SWIZZLE_CD_MODE': smem_config.swizzle_cd_mode,
        'NUM_MULTICAST': multicast_config.num_multicast,
        'IS_MULTICAST_ON_A': multicast_config.is_multicast_on_a,
        'WITH_ACCUMULATION': c is not None,
        'CD_DTYPE_T': d.dtype,
        # Runtime arguments
        'GROUPED_LAYOUT': torch.empty(0, dtype=torch.int32, device=d.device),
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config.smem_size,
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_SFA': tensor_map_sfa,
        'TENSOR_MAP_SFB': tensor_map_sfb,
        'TENSOR_MAP_C': tensor_map_c,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
        'DEVICE_INDEX': d.device.index
    }

    # Generate, build and run the kernel
    code = SM100FP8GemmRuntime.generate(kwargs)
    runtime = build('fp8_gemm', code, SM100FP8GemmRuntime, kwargs)
    runtime(**kwargs)


def m_grouped_fp8_gemm_nt_contiguous(a: torch.Tensor, sfa: torch.Tensor,
                                     b: torch.Tensor, sfb: torch.Tensor,
                                     d: torch.Tensor,
                                     m_indices: torch.Tensor,
                                     major_a: MajorTypeAB, major_b: MajorTypeAB,
                                     compiled_dims: str) -> None:
    m, k = a.shape
    num_groups, n, _ = b.shape
    major_d = MajorTypeCD.NMajor

    # K must be aligned to 128
    aligned_k = align(k, 128)

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    num_sms, block_m, block_n, block_k, num_stages, multicast_config, smem_config = get_best_configs(
        GemmType.GroupedContiguous, m, n, k, num_groups, major_a, major_b, major_d, torch.float8_e4m3fn, d.dtype, num_sms)

    # NOTES: you cannot distinguish groups for A, SFA, and D
    tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                   multicast_config.get_ab_load_block_m(block_m), block_k,
                                   a.stride(major_a.non_contiguous_dim()), num_groups=1,
                                   swizzle_mode=smem_config.swizzle_a_mode)
    tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                   multicast_config.get_ab_load_block_n(block_n), block_k,
                                   b.stride(major_b.non_contiguous_dim()), num_groups=num_groups,
                                   swizzle_mode=smem_config.swizzle_b_mode)
    tensor_map_d = make_tma_cd_desc(major_d, d, m, n,
                                    block_m, block_n,
                                    d.stride(major_d.non_contiguous_dim()), num_groups=1,
                                    swizzle_mode=smem_config.swizzle_cd_mode)
    tensor_map_sfa = make_tma_sf_desc(MajorTypeAB.MNMajor, sfa, m, k, block_m, block_k, num_groups=1, swizzle_mode=smem_config.swizzle_sf_mode)
    tensor_map_sfb = make_tma_sf_desc(MajorTypeAB.MNMajor, sfb, n, k, block_n, block_k, num_groups=num_groups, swizzle_mode=smem_config.swizzle_sf_mode)

    kwargs = {
        # Templated or runtime arguments according to the `COMPILED_DIMS`
        'COMPILED_DIMS': compiled_dims,
        'M': m, 'N': n, 'K': aligned_k,
        # Templated arguments
        'GEMM_TYPE': GemmType.GroupedContiguous,
        'NUM_NON_EPILOGUE_THREADS': 128,
        'NUM_EPILOGUE_THREADS': 128,
        'MAJOR_A': major_a,
        'MAJOR_B': major_b,
        'NUM_GROUPS': num_groups,
        'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
        'NUM_STAGES': num_stages, 'NUM_LAST_STAGES': ceil_div(k, block_k) % num_stages,
        'SWIZZLE_A_MODE': smem_config.swizzle_a_mode,
        'SWIZZLE_B_MODE': smem_config.swizzle_b_mode,
        'SWIZZLE_CD_MODE': smem_config.swizzle_cd_mode,
        'NUM_MULTICAST': multicast_config.num_multicast,
        'IS_MULTICAST_ON_A': multicast_config.is_multicast_on_a,
        'WITH_ACCUMULATION': False,
        'CD_DTYPE_T': d.dtype,
        # Runtime arguments
        'GROUPED_LAYOUT': m_indices,
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config.smem_size,
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_SFA': tensor_map_sfa,
        'TENSOR_MAP_SFB': tensor_map_sfb,
        'TENSOR_MAP_C': tensor_map_d,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
        'DEVICE_INDEX': d.device.index
    }

    # Generate, build and run the kernel
    code = SM100FP8GemmRuntime.generate(kwargs)
    runtime = build('fp8_m_grouped_gemm', code, SM100FP8GemmRuntime, kwargs)
    runtime(**kwargs)


def fp8_m_grouped_gemm_nt_masked(a: torch.Tensor, sfa: torch.Tensor,
                                 b: torch.Tensor, sfb: torch.Tensor,
                                 d: torch.Tensor,
                                 masked_m: torch.Tensor,
                                 expected_m: int,
                                 major_a: MajorTypeAB, major_b: MajorTypeAB,
                                 compiled_dims: str) -> None:
    num_groups, m, k = a.shape
    _, n, _ = b.shape
    major_d = MajorTypeCD.NMajor

    # K must be aligned to 128
    aligned_k = align(k, 128)

    num_sms = get_num_sms()
    num_sms, block_m, block_n, block_k, num_stages, multicast_config, smem_config = get_best_configs(
        GemmType.GroupedMasked, expected_m, n, k, num_groups, major_a, major_b, major_d, torch.float8_e4m3fn, d.dtype, num_sms)
    if num_groups > 1:
        assert m % block_m == 0

    tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                   multicast_config.get_ab_load_block_m(block_m), block_k,
                                   a.stride(major_a.non_contiguous_dim()), num_groups,
                                   smem_config.swizzle_a_mode)
    tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                   multicast_config.get_ab_load_block_n(block_n), block_k,
                                   b.stride(major_b.non_contiguous_dim()), num_groups,
                                   smem_config.swizzle_b_mode)
    tensor_map_d = make_tma_cd_desc(major_d, d, m, n,
                                    block_m, block_n,
                                    d.stride(major_d.non_contiguous_dim()), num_groups,
                                    smem_config.swizzle_cd_mode)
    tensor_map_sfa = make_tma_sf_desc(MajorTypeAB.MNMajor, sfa, m, k, block_m, block_k, num_groups, smem_config.swizzle_sf_mode)
    tensor_map_sfb = make_tma_sf_desc(MajorTypeAB.MNMajor, sfb, n, k, block_n, block_k, num_groups, smem_config.swizzle_sf_mode)

    kwargs = {
        # Templated or runtime arguments according to the `COMPILED_DIMS`
        'COMPILED_DIMS': compiled_dims,
        'M': m, 'N': n, 'K': aligned_k,
        # Templated arguments
        'GEMM_TYPE': GemmType.GroupedMasked,
        'NUM_NON_EPILOGUE_THREADS': 128,
        'NUM_EPILOGUE_THREADS': 128,
        'MAJOR_A': major_a,
        'MAJOR_B': major_b,
        'NUM_GROUPS': num_groups,
        'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
        'NUM_STAGES': num_stages, 'NUM_LAST_STAGES': ceil_div(k, block_k) % num_stages,
        'SWIZZLE_A_MODE': smem_config.swizzle_a_mode,
        'SWIZZLE_B_MODE': smem_config.swizzle_b_mode,
        'SWIZZLE_CD_MODE': smem_config.swizzle_cd_mode,
        'NUM_MULTICAST': multicast_config.num_multicast,
        'IS_MULTICAST_ON_A': multicast_config.is_multicast_on_a,
        'WITH_ACCUMULATION': False,
        'CD_DTYPE_T': d.dtype,
        # Runtime arguments
        'GROUPED_LAYOUT': masked_m,
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config.smem_size,
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_SFA': tensor_map_sfa,
        'TENSOR_MAP_SFB': tensor_map_sfb,
        'TENSOR_MAP_C': tensor_map_d,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
        'DEVICE_INDEX': d.device.index
    }

    # Generate, build and run the kernel
    code = SM100FP8GemmRuntime.generate(kwargs)
    runtime = build('fp8_m_grouped_gemm', code, SM100FP8GemmRuntime, kwargs)
    runtime(**kwargs)
