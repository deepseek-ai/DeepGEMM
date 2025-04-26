import os
from typing import Any, Dict


def generate(**kwargs: Dict[str, Any]) -> str:
    code = f'''
#ifdef __CUDACC_RTC__
#ifndef NVRTC_JIT_COMPILATION
#define NVRTC_JIT_COMPILATION
#endif

#include <deep_gemm/nvrtc_std.cuh>

#else

#include <string>
#include <cuda.h>

#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/fp8_gemm.cuh>

using namespace deep_gemm;

__global__ void dummy_kernel() {{
  void *ptr = (void *)&fp8_gemm_kernel<
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
  >;
}}
'''

    # Debug print
    if os.getenv('DG_JIT_DEBUG', None):
        print(f'Generated code:\n{code}')

    return code
