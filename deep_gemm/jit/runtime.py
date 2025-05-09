import os
import subprocess
import time
import cuda.bindings.driver as cbd

from typing import List, Optional, Type
from torch.utils.cpp_extension import CUDA_HOME


class Runtime:
    def __init__(self, path: str, args: List[str] = None) -> None:
        self.path = path
        self.lib = None
        self.kernel = None
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

    @staticmethod
    def generate(**kwargs) -> str:
        raise NotImplemented

    @staticmethod
    def launch(kernel: cbd.CUkernel, **kwargs) -> cbd.CUresult:
        raise NotImplemented

    def __call__(self, **kwargs) -> cbd.CUresult:
        # Load CUBIN
        if self.kernel is None:
            start_time = time.time_ns()

            # Load CUBIN
            path = bytes(os.path.join(self.path, 'kernel.cubin'), 'utf-8')
            result, self.lib = cbd.cuLibraryLoadFromFile(path, [], [], 0, [], [], 0)
            assert result == cbd.CUresult.CUDA_SUCCESS, f'Failed to load library: {result}'

            # Extract the kernel name
            # TODO: use `cuda-bindings` API to do this (requires at least 12.8)
            command = [f'{CUDA_HOME}/bin/cuobjdump', '-symbols', path]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert result.returncode == 0
            kernel_names = [line.split()[-1] for line in result.stdout.splitlines()
                            if line.startswith('STT_FUNC') and '__instantiate_kernel' not in line]
            assert len(kernel_names) == 1, f'Too many kernels in the library: {kernel_names}'

            # Load kernel from the library
            result, self.kernel = cbd.cuLibraryGetKernel(self.lib, bytes(kernel_names[0], encoding='utf-8'))
            assert result == cbd.CUresult.CUDA_SUCCESS, f'Failed to load kernel: {result}'

            end_time = time.time_ns()
            elapsed_time = (end_time - start_time) / 1e6
            if int(os.getenv('DG_JIT_DEBUG', 0)):
                print(f'Loading JIT runtime {self.path} took {elapsed_time:.2f} ms.')

        # noinspection PyArgumentList
        return self.launch(self.kernel, *[kwargs[arg] for arg in self.args])

    def __del__(self) -> None:
        if self.lib is not None:
            res = cbd.cuLibraryUnload(self.lib)[0]
            if res != cbd.CUresult.CUDA_SUCCESS:
                raise Exception(f'Failed to unload library {self.path}: {res}')


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __setitem__(self, path: str, runtime: Runtime) -> None:
        self.cache[path] = runtime

    def get(self, path: str, runtime_cls: Type[Runtime]) -> Optional[Runtime]:
        # In Python runtime
        if path in self.cache:
            return self.cache[path]

        # Already compiled
        if not int(os.getenv('DG_JIT_DISABLE_CACHE', 0)) and os.path.exists(path) and Runtime.is_path_valid(path):
            runtime = runtime_cls(path)
            self.cache[path] = runtime
            return runtime
        return None
