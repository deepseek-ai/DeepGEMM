import os
import time
import subprocess
from typing import Any, Dict, Optional

import cuda.bindings.driver as cuda
from torch.utils.cpp_extension import CUDA_HOME

from .utils import run_gemm


def get_symbol(file_path: str, pattern: str) -> Optional[str]:
    command = [f'{CUDA_HOME}/bin/cuobjdump', '-symbols', file_path]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0
    for line in result.stdout.splitlines():
        if pattern in line:
            return line.split()[-1]
    return None


class Runtime:
    def __init__(self, path: str, kernel_name: str) -> None:
        self.path = path
        self.lib = None
        self.kernel = None
        self.kernel_name = kernel_name

        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Contains all necessary files
        files = ['kernel.cubin']
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, **kwargs: Dict[str, Any]) -> cuda.CUresult:
        # Load CUBIN
        if self.kernel is None:
            start_time = time.time_ns()
            res, lib = cuda.cuLibraryLoadFromFile(
                bytes(os.path.join(self.path, 'kernel.cubin'), 'utf-8'), [], [], 0, [], [], 0)
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to load library: {res}")

            print(f"Kernel name: {self.kernel_name}")
            res, kernel = cuda.cuLibraryGetKernel(
                lib, bytes(self.kernel_name, encoding='utf-8'))
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to get kernel: {res}")

            self.kernel = kernel
            if self.kernel is not None:
                self.lib = lib
            else:
                raise Exception("Failed to find fp8 gemm kernel")

            end_time = time.time_ns()
            elapsed_time = (end_time - start_time) / 1000
            if os.getenv('DG_JIT_DEBUG', None):
                print(
                    f'Loading JIT runtime {self.path} took {elapsed_time:.2f} us.')

        return run_gemm(
            self.kernel,
            kwargs['NUM_TMA_MULTICAST'],
            kwargs['M'],
            kwargs['BLOCK_M'],
            kwargs['GMEM_D'],
            kwargs['SCALES_B'],
            kwargs['GROUPED_LAYOUT'],
            kwargs['NUM_SMS'],
            kwargs['SMEM_SIZE'],
            kwargs['TENSOR_MAP_A'],
            kwargs['TENSOR_MAP_B'],
            kwargs['TENSOR_MAP_SCALES_A'],
            kwargs['TENSOR_MAP_D'],
            kwargs['STREAM'],
        )

    def __del__(self) -> None:
        if self.lib is not None:
            res = cuda.cuLibraryUnload(self.lib)[0]
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to unload library {self.path}: {res}")


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __getitem__(self, path: str) -> Optional[Runtime]:
        # In Python runtime
        if path in self.cache:
            return self.cache[path]

        # Already compiled
        if os.path.exists(path) and Runtime.is_path_valid(path):
            kernel_name = get_symbol(os.path.join(
                path, 'kernel.cubin'), 'fp8_gemm_kernel')
            runtime = Runtime(path, kernel_name)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime
