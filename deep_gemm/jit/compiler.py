import abc
import functools
import hashlib
import os
import re
import subprocess
import time
import uuid
from typing import List, Tuple

import cuda.bindings
import cuda.bindings.nvrtc as nvrtc
from torch.utils.cpp_extension import CUDA_HOME

from . import interleave_ffma
from .runtime import Runtime, RuntimeCache, get_symbol

runtime_cache = RuntimeCache()


def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'include')


@functools.lru_cache(maxsize=None)
def get_deep_gemm_version() -> str:
    # Update include directories
    include_dir = os.path.join(get_jit_include_dir(), 'deep_gemm')
    assert os.path.exists(
        include_dir), f'Cannot find GEMM include directory {include_dir}'
    md5 = hashlib.md5()
    for filename in filter(lambda x: x.endswith('.cuh'), sorted(os.listdir(include_dir))):
        with open(os.path.join(include_dir, filename), 'rb') as f:
            md5.update(f.read())

    # Update `interleave_ffma.py`
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'interleave_ffma.py'), 'rb') as f:
        md5.update(f.read())
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def get_nvcc_compiler() -> Tuple[str, str]:
    paths = []
    if os.getenv('DG_NVCC_COMPILER'):
        paths.append(os.getenv('DG_NVCC_COMPILER'))
    paths.append(os.path.join(CUDA_HOME, 'bin', 'nvcc'))

    # Try to find the first available NVCC compiler
    least_version_required = '12.3'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            match = version_pattern.search(
                os.popen(f'{path} --version').read())
            version = match.group(1)
            assert match, f'Cannot get the version of NVCC compiler {path}'
            assert version >= least_version_required, f'NVCC {path} version {version} is lower than {least_version_required}'
            return path, version
    raise RuntimeError('Cannot find any available NVCC compiler')


@functools.lru_cache(maxsize=None)
def get_default_user_dir():
    if 'DG_CACHE_DIR' in os.environ:
        path = os.getenv('DG_CACHE_DIR')
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.join(os.path.expanduser('~'), '.deep_gemm')


@functools.lru_cache(maxsize=None)
def get_tmp_dir():
    return os.path.join(get_default_user_dir(), 'tmp')


@functools.lru_cache(maxsize=None)
def get_cache_dir():
    return os.path.join(get_default_user_dir(), 'cache')


def make_tmp_dir():
    tmp_dir = get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def put(path, data):
    is_binary = isinstance(data, bytes)
    
    # Write and do POSIX atomic replace
    tmp_file_path = os.path.join(make_tmp_dir(), f'file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}')
    with open(tmp_file_path, 'wb' if is_binary else 'w') as f:
        f.write(data)
    os.replace(tmp_file_path, path)


class Compiler(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def __version__() -> Tuple[int, int]:
        pass

    @classmethod
    @abc.abstractmethod
    def compile(cls, name: str, code: str, target_path: str) -> str:
        pass

    @staticmethod
    def flags() -> List[str]:
        cpp_standard = int(os.getenv('DG_NVCC_OVERRIDE_CPP_STANDARD', 20))
        return [f'-std=c++{cpp_standard}',
                '--ptxas-options=--register-usage-level=10' +
                (',--verbose' if 'DG_PTXAS_VERBOSE' in os.environ else ''),
                # Suppress some unnecessary warnings, such as unused variables for certain `constexpr` branch cases
                '--diag-suppress=39,161,174,177,940']

    @staticmethod
    def include_dirs() -> List[str]:
        return [get_jit_include_dir()]

    @classmethod
    def build(cls, name: str, code: str) -> Runtime:
        # Compiler flags
        flags = cls.flags()
        include_dirs = cls.include_dirs()

        # Build signature
        enable_sass_opt = get_nvcc_compiler()[1] <= '12.8' and int(
            os.getenv('DG_DISABLE_FFMA_INTERLEAVE', 0)) == 0
        signature = f'{name}$${get_deep_gemm_version()}$${code}$${get_nvcc_compiler()}$${flags}$${enable_sass_opt}'
        name = f'kernel.{name}.{hash_to_hex(signature)}'
        path = os.path.join(get_cache_dir(), name)

        # Check runtime cache or file system hit
        global runtime_cache
        if runtime_cache[path] is not None:
            if os.getenv('DG_JIT_DEBUG', None):
                print(f'Using cached JIT runtime {name} during build')
            return runtime_cache[path]

        # Compile into a temporary CU file
        os.makedirs(path, exist_ok=True)
        cubin_path = os.path.join(path, 'kernel.cubin')
        tmp_cubin_path = os.path.join(make_tmp_dir(), f'nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(cubin_path)}.cubin')

        start_time = time.time()
        kernel_name = cls.compile(name, code, tmp_cubin_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if os.getenv('DG_JIT_DEBUG', None):
            print(
                f'Compilation of JIT runtime {name} took {elapsed_time:.2f} seconds.')

        # Interleave FFMA reuse
        if enable_sass_opt:
            interleave_ffma.process(tmp_cubin_path)
            
        # Store kernel name
        put(f'{tmp_cubin_path}.name', kernel_name)

        # Atomic replace files
        os.replace(tmp_cubin_path, cubin_path)
        os.replace(f'{tmp_cubin_path}.name', f'{cubin_path}.name')

        # Put cache and return
        runtime_cache[path] = Runtime(path, kernel_name)
        return runtime_cache[path]


class NvccCompiler(Compiler):
    @staticmethod
    def __version__() -> Tuple[int, int]:
        _, version = get_nvcc_compiler()
        major, minor = map(int, version.split('.'))
        return (major, minor)

    @classmethod
    def flags(cls) -> List[str]:
        cxx_flags = ['-fPIC', '-O3',
                     '-Wno-deprecated-declarations', '-Wno-abi', '-fconcepts']
        return [*super().flags(), *[f'-I{d}' for d in cls.include_dirs()],
                '-gencode=arch=compute_90a,code=sm_90a',
                '-cubin', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                f'--compiler-options={",".join(cxx_flags)}']

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> str:
        # Write the code
        path = os.path.join(get_cache_dir(), name)
        src_path = os.path.join(path, 'kernel.cu')
        put(src_path, code)
        command = [get_nvcc_compiler()[0],
                   src_path, '-o', target_path,
                   *cls.flags()]
        if os.getenv('DG_JIT_DEBUG', None) or os.getenv('DG_JIT_PRINT_NVCC_COMMAND', False):
            print(f'Compiling JIT runtime {name} with command {command}')

        return_code = subprocess.check_call(command)
        assert return_code == 0, f'Failed to compile {src_path}'

        # NVCC needs to get the symbol name from the cubin file using `cuobjdump`
        return get_symbol(target_path, 'fp8_gemm_kernel')


class NvrtcCompiler(Compiler):
    @staticmethod
    def __version__() -> Tuple[int, int]:
        major, minor = map(int, cuda.bindings.__version__.split('.')[:2])
        return (major, minor)

    @staticmethod
    def include_dirs() -> List[str]:
        if CUDA_HOME is None:
            raise RuntimeError('CUDA_HOME is required for NVRTC compilation')
        return [get_jit_include_dir(), os.path.join(CUDA_HOME, 'include'), os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'include')]

    @classmethod
    def flags(cls) -> List[str]:
        base_flags = [*super().flags(), *[f'-I{d}' for d in cls.include_dirs()],
                      '--gpu-architecture=sm_90a', '-default-device']
        if cls.__version__() >= (12, 8):
            base_flags += ['--pch']
            if os.getenv('DG_JIT_DEBUG', None):
                base_flags += ['--pch-verbose=true']
        return base_flags

    @classmethod
    def compile(cls, name: str, code: str, target_path: str) -> str:
        code_bytes = bytes(code, 'utf-8')
        res, program = nvrtc.nvrtcCreateProgram(
            code_bytes, bytes(name, 'utf-8'), 0, [], [])
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise Exception(f"Failed to create program: {res}")

        kernel_regex = re.compile(r'fp8_gemm_kernel<[\S\s]*?>', re.MULTILINE)
        kernel_name = kernel_regex.search(code).group(
            0).replace('\n', '').replace(' ', '')
        res = nvrtc.nvrtcAddNameExpression(
            program, bytes(kernel_name, 'utf-8'))[0]
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise Exception(f"Failed to add name expression: {res}")

        options = [bytes(flag, 'utf-8') for flag in cls.flags()]
        compile_res = nvrtc.nvrtcCompileProgram(
            program, len(options), options)[0]

        if os.getenv('DG_JIT_DEBUG', None):
            res, log_size = nvrtc.nvrtcGetProgramLogSize(program)
            if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise Exception(f"Failed to get program log size: {res}")
            log_bytes = bytes(log_size)
            res = nvrtc.nvrtcGetProgramLog(program, log_bytes)[0]
            if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise Exception(f"Failed to get program log: {res}")
            log_str = log_bytes.decode('utf-8')
            print(log_str)

        if compile_res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise Exception(f"Failed to compile program: {compile_res}")

        # NVRTC can directly get the lowered name
        res, lowered_name = nvrtc.nvrtcGetLoweredName(
            program, bytes(kernel_name, 'utf-8'))
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise Exception(f"Failed to get lowered name: {res}")

        res, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise Exception(f"Failed to get CUBIN size: {res}")

        cubin_bytes = bytes(cubin_size)
        res = nvrtc.nvrtcGetCUBIN(program, cubin_bytes)[0]
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise Exception(f"Failed to get CUBIN: {res}")

        put(target_path, cubin_bytes)

        res = nvrtc.nvrtcDestroyProgram(program)[0]
        if res != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise Exception(f"Failed to destroy program: {res}")

        return lowered_name.decode('utf-8')


def build(name: str, code: str) -> Runtime:
    if os.getenv('DG_JIT_USE_NVRTC', '0') in ['1', 'true', 'True']:
        return NvrtcCompiler.build(name, code)
    else:
        return NvccCompiler.build(name, code)
