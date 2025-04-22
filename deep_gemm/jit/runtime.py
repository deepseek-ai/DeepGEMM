import os
import time
from typing import Any, Dict, Optional

import cuda.bindings.driver as cuda
import cuda.bindings.nvrtc as nvrtc
import torch

from .utils import run_gemm

class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.kernel = None

        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Contains all necessary files
        files = ['kernel.cu', 'kernel.cubin']
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, **kwargs: Dict[str, Any]) -> cuda.CUresult:
        # Load CUBIN
        if self.lib is None:
            start_time = time.time_ns()
            res, lib = cuda.cuLibraryLoadFromFile(
                bytes(os.path.join(self.path, 'kernel.cubin'), 'utf-8'), [], [], 0, [], [], 0)
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to load library: {res}")

            res, kernel_count = cuda.cuLibraryGetKernelCount(lib)
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to get kernel count: {res}")

            res, kernels = cuda.cuLibraryEnumerateKernels(kernel_count, lib)
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f"Failed to enumerate kernels: {res}")

            for kernel in kernels:
                res, kernel_name = cuda.cuKernelGetName(kernel)
                if res != cuda.CUresult.CUDA_SUCCESS:
                    raise Exception(f"Failed to get kernel name: {res}")
                if b"fp8" in kernel_name:
                    self.kernel = kernel
                    break

            if self.kernel is not None:
                self.lib = lib
            else:
                raise Exception("Failed to find fp8 gemm kernel")

            end_time = time.time_ns()
            elapsed_time = (end_time - start_time) / 1000
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
            runtime = Runtime(path)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime
