import ctypes
import os
import torch
import cuda.bindings.driver as cbd

from deep_gemm import jit

# Essential debugging staffs
os.environ['DG_JIT_DEBUG'] = os.getenv('DG_JIT_DEBUG', '1')
os.environ['DG_JIT_DISABLE_CACHE'] = os.getenv('DG_JIT_DISABLE_CACHE', '1')


class VectorAddRuntime(jit.Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path, [
            'A',
            'B',
            'C',
            'STREAM',
        ])

    @staticmethod
    def generate(**kwargs) -> str:
        return f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#endif

#include <cuda_fp8.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void vector_add(T* a, T* b, T* c, uint32_t n) {{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {{
        c[i] = a[i] + b[i];
    }}
}}

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&vector_add<{kwargs['T']}>);
}}
"""

    # noinspection PyShadowingNames,PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel,
               a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
               stream: cbd.CUstream) -> cbd.CUresult:
        assert a.shape == b.shape == c.shape
        assert a.device == b.device == c.device
        assert a.dim() == 1

        config = cbd.CUlaunchConfig()
        config.gridDimX = (a.numel() + 127) // 128
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = stream

        arg_values = (
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            a.numel(),
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )

        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)[0]


if __name__ == '__main__':
    print('Generated code:')
    code = VectorAddRuntime.generate(T='float')
    print(code)
    print()

    for compiler_name in ('NVCC', 'NVRTC'):
        # Get compiler
        compiler_cls = getattr(jit, f'{compiler_name}Compiler')
        print(f'Compiler: {compiler_name}, version: {compiler_cls.__version__()}')

        # Build
        print('Building ...')
        func = compiler_cls.build('test_func', code, VectorAddRuntime)

        # Run and check
        a = torch.randn((1024, ), dtype=torch.float32, device='cuda')
        b = torch.randn((1024, ), dtype=torch.float32, device='cuda')
        c = torch.empty_like(a)
        ret = func(A=a, B=b, C=c, STREAM=torch.cuda.current_stream().cuda_stream)
        assert ret == cbd.CUresult.CUDA_SUCCESS, ret
        torch.testing.assert_close(c, a + b)
        print(f'JIT test for {compiler_name} passed\n')
