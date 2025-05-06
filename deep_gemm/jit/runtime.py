import os
import time
from typing import Any, Callable, Dict, List, Optional, Type

import cuda.bindings.driver as cuda


class Runtime:
    def __init__(self, path: str, kernel_name: str = None,
                 caller: Callable[..., cuda.CUresult] = None,
                 args: List[str] = None) -> None:
        self.path = path
        self.lib = None
        self.kernel = None
        self.kernel_name = kernel_name
        self.caller = caller
        self.args = args
        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Contains all necessary files
        files = ['kernel.cubin']
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, **kwargs) -> cuda.CUresult:
        # Load CUBIN
        if self.kernel is None:
            start_time = time.time_ns()
            res, lib = cuda.cuLibraryLoadFromFile(
                bytes(os.path.join(self.path, 'kernel.cubin'), 'utf-8'), [], [], 0, [], [], 0)
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f'Failed to load library: {res}')

            res, kernel_count = cuda.cuLibraryGetKernelCount(lib)
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f'Failed to get kernel count: {res}')
            
            res, kernels = cuda.cuLibraryEnumerateKernels(kernel_count, lib)
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f'Failed to enumerate kernels: {res}')

            for kernel in kernels:
                res, kernel_name = cuda.cuKernelGetName(kernel)
                if res != cuda.CUresult.CUDA_SUCCESS:
                    raise Exception(f'Failed to get kernel name: {res}')
                if bytes(self.kernel_name, encoding='utf-8') in kernel_name:
                    self.kernel = kernel
                    break

            if self.kernel is not None:
                self.lib = lib
            else:
                raise Exception('Failed to find required kernel')

            end_time = time.time_ns()
            elapsed_time = (end_time - start_time) / 1000
            if int(os.getenv('DG_JIT_DEBUG', 0)):
                print(f'Loading JIT runtime {self.path} took {elapsed_time:.2f} us.')

        return self.caller(
            self.kernel,
            *[kwargs[arg] for arg in self.args]
        )

    def __del__(self) -> None:
        if self.lib is not None:
            res = cuda.cuLibraryUnload(self.lib)[0]
            if res != cuda.CUresult.CUDA_SUCCESS:
                raise Exception(f'Failed to unload library {self.path}: {res}')


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime

    def get(self, path: str, runtime_cls: Type[Runtime]) -> Optional[Runtime]:
        # In Python runtime
        if path in self.cache:
            return self.cache[path]

        # Already compiled
        if not int(os.getenv('DG_DISABLE_CACHE', 0)) and os.path.exists(path) and Runtime.is_path_valid(path):
            runtime = runtime_cls(path)
            self.cache[path] = runtime
            return runtime
        return None
