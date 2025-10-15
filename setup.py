import ast
import os
import re
import shutil
import setuptools
import subprocess
import sys
import torch
import platform
import urllib
import urllib.error
import urllib.request
from setuptools import find_packages
from setuptools.command.build_py import build_py
from packaging.version import parse
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


DG_SKIP_CUDA_BUILD = int(os.getenv('DG_SKIP_CUDA_BUILD', '0')) == 1
DG_FORCE_BUILD = int(os.getenv('DG_FORCE_BUILD', '0')) == 1
DG_USE_LOCAL_VERSION = int(os.getenv('DG_USE_LOCAL_VERSION', '1')) == 1
DG_JIT_USE_RUNTIME_API = int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')) == 1

# Compiler flags
cxx_flags = ['-std=c++17', '-O3', '-fPIC', '-Wno-psabi', '-Wno-deprecated-declarations',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']
if DG_JIT_USE_RUNTIME_API:
    cxx_flags.append('-DDG_JIT_USE_RUNTIME_API')

# Sources
current_dir = os.path.dirname(os.path.realpath(__file__))
sources = ['csrc/python_api.cpp']
build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    'deep_gemm/include',
    'third-party/cutlass/include',
    'third-party/fmt/include',
]
build_libraries = ['cuda', 'cudart', 'nvrtc']
build_library_dirs = [
    f'{CUDA_HOME}/lib64',
    f'{CUDA_HOME}/lib64/stubs'
]
third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]

# Release
base_wheel_url = 'https://github.com/DeepSeek-AI/DeepGEMM/releases/download/{tag_name}/{wheel_name}'


def get_package_version():
    with open(Path(current_dir) / 'deep_gemm' / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    revision = ''
    if DG_USE_LOCAL_VERSION:
        # noinspection PyBroadException
        try:
            status_cmd = ['git', 'status', '--porcelain']
            status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
            if status_output:
                print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
                assert False, 'Git working directory is not clean'

            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except:
            revision = '+local'
    return f'{public_version}{revision}'


def get_platform():
    if sys.platform.startswith('linux'):
        return f'linux_{platform.uname().machine}'
    else:
        raise ValueError('Unsupported platform: {}'.format(sys.platform))


def get_wheel_url():
    torch_version = parse(torch.__version__)
    if os.environ.get("NVIDIA_PRODUCT_NAME", "") == "PyTorch":
        torch_version = str(os.environ.get("NVIDIA_PYTORCH_VERSION"))
    else:
        torch_version = f'{torch_version.major}.{torch_version.minor}'

    python_version = f'cp{sys.version_info.major}{sys.version_info.minor}'
    platform_name = get_platform()
    deep_gemm_version = get_package_version()
    cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)

    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    cuda_version = parse(torch.version.cuda)
    cuda_version = f'{cuda_version.major}'

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f'deep_gemm-{deep_gemm_version}+cu{cuda_version}-torch{torch_version}-cxx11abi{cxx11_abi}-{python_version}-{platform_name}.whl'
    wheel_url = base_wheel_url.format(tag_name=f'v{deep_gemm_version}', wheel_name=wheel_filename)
    return wheel_url, wheel_filename


def get_ext_modules():
    if DG_SKIP_CUDA_BUILD:
        return []

    return [CUDAExtension(name='deep_gemm_cpp',
                          sources=sources,
                          include_dirs=build_include_dirs,
                          libraries=build_libraries,
                          library_dirs=build_library_dirs,
                          extra_compile_args=cxx_flags)]


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Second, make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Finally, run the regular build
        build_py.run(self)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_CPP_STANDARD'):
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        with open(os.path.join(self.build_lib, 'deep_gemm', 'envs.py'), 'w') as f:
            f.write(code)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, 'deep_gemm/include')
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        if DG_FORCE_BUILD or DG_USE_LOCAL_VERSION:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print(f'Try to download wheel from URL: {wheel_url}')
        try:
            with urllib.request.urlopen(wheel_url, timeout=1) as response:
                with open(wheel_filename, 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)

            # Make the archive
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)
            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f'{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}'
            wheel_path = os.path.join(self.dist_dir, archive_basename + '.whl')
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print('Precompiled wheel not found. Building from source...')
            # If the wheel could not be downloaded, build from source
            super().run()


if __name__ == '__main__':
    # noinspection PyTypeChecker
    setuptools.setup(
        name='deep_gemm',
        version=get_package_version(),
        packages=find_packages('.'),
        package_data={
            'deep_gemm': [
                'include/deep_gemm/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
            ]
        },
        ext_modules=get_ext_modules(),
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
            'bdist_wheel': CachedWheelsCommand,
        },
    )
