import ctypes
import os
import torch
from typing import Any, Dict

import cuda.bindings.driver as cuda

from deep_gemm import jit


def run_vector_add(kernel: cuda.CUkernel, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, stream: cuda.CUstream) -> cuda.CUresult:
    assert a.shape == b.shape == c.shape
    assert a.device == b.device == c.device
    assert a.dim() == 1

    n = a.numel()

    config = cuda.CUlaunchConfig()
    config.gridDimX = (n + 127) // 128
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 128
    config.blockDimY = 1
    config.blockDimZ = 1
    config.hStream = stream

    kernelValues = (
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        n,
    )
    kernelTypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )

    return cuda.cuLaunchKernelEx(config, kernel, (kernelValues, kernelTypes), 0)[0]


def generate_vector_add(**kwargs: Dict[str, Any]) -> str:
    return f"""
#ifdef __CUDACC_RTC__
#ifndef NVRTC_JIT_COMPILATION
#define NVRTC_JIT_COMPILATION
#endif
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#endif

#include <cuda_fp8.h>
#include <cuda_bf16.h>

template<typename T>
__global__ void vector_add(T* a, T* b, T* c, uint32_t N) {{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {{
        c[i] = a[i] + b[i];
    }}
}}

__global__ void dummy_kernel() {{
    void *ptr = (void *)&vector_add<{kwargs['T']}>;
}}
"""


class VectorAddRuntime(jit.Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path, 'vector_add', run_vector_add, [
            'A',
            'B',
            'C',
            'STREAM',
        ])


if __name__ == '__main__':
    # NVCC
    print(f'NVCC compiler version: {jit.NvccCompiler.__version__()}\n')
    print('Generated code:')
    code = generate_vector_add(T='float')
    print(code)
    print('Building ...')
    func = jit.NvccCompiler.build('test_func', code, VectorAddRuntime)

    a = torch.randn((1024, ), dtype=torch.float32, device='cuda')
    b = torch.randn((1024, ), dtype=torch.float32, device='cuda')
    c = torch.empty_like(a)
    ret = func(A=a, B=b, C=c, STREAM=torch.cuda.current_stream().cuda_stream)
    assert ret == cuda.CUresult.CUDA_SUCCESS, ret
    ref_output = a + b
    torch.testing.assert_close(c, ref_output)

    print('JIT test for NVCC passed\n')

    # NVRTC
    print(f'NVRTC compiler version: {jit.NvrtcCompiler.__version__()}\n')
    print('Generated code:')
    code = generate_vector_add(T='__nv_bfloat16')
    print(code)
    print('Building ...')
    func = jit.NvrtcCompiler.build('test_func', code, VectorAddRuntime)

    a = torch.randn((1024, ), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((1024, ), dtype=torch.bfloat16, device='cuda')
    c = torch.empty_like(a)
    ret = func(A=a, B=b, C=c, STREAM=torch.cuda.current_stream().cuda_stream)
    assert ret == cuda.CUresult.CUDA_SUCCESS, ret
    ref_output = a + b
    torch.testing.assert_close(c, ref_output)

    print('JIT test for NVRTC passed')