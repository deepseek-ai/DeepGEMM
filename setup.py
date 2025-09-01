import ast
import os
import re
import setuptools
import shutil
import subprocess
import sys
import urllib
import torch
import platform
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
from pathlib import Path
from packaging import version as parse
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

SKIP_CUDA_BUILD = os.getenv("DEEP_GEMM_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
NO_LOCAL_VERSION = os.getenv("DEEP_GEMM_NO_LOCAL_VERSION", "FALSE") == "TRUE"
FORCE_BUILD = os.getenv("DEEP_GEMM_FORCE_BUILD", "FALSE") == "TRUE"

BASE_WHEEL_URL = (
    "https://github.com/DeepSeek-AI/DeepGEMM/releases/download/{tag_name}/{wheel_name}"
)
PACKAGE_NAME = "deep_gemm"

current_dir = os.path.dirname(os.path.realpath(__file__))
cxx_flags = ['-std=c++17', '-O3', '-fPIC', '-Wno-psabi', '-Wno-deprecated-declarations',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']
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

# Use runtime API
if int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')):
    cxx_flags.append('-DDG_JIT_USE_RUNTIME_API')

def get_package_version():
    with open(Path(current_dir) / "deep_gemm" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    revision = ""

    if not NO_LOCAL_VERSION:
        try:
            cmd = ["git", "rev-parse", "--short", "HEAD"]
            revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
        except:
            revision = ""

    return f"{public_version}{revision}"

def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))

def get_wheel_url():
    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    grouped_gemm_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
    torch_cuda_version = parse(torch.version.cuda)
    # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.3
    # to save CI time. Minor versions should be compatible.
    torch_cuda_version = (
        parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
    )
    # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
    cuda_version = f"{torch_cuda_version.major}"

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{grouped_gemm_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"

    wheel_url = BASE_WHEEL_URL.format(
        tag_name=f"v{grouped_gemm_version}", wheel_name=wheel_filename
    )

    return wheel_url, wheel_filename

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

if not SKIP_CUDA_BUILD:
    ext_modules = [
        CUDAExtension(
            name="deep_gemm_cpp",
            sources=sources,
            include_dirs=build_include_dirs,
        )
    ]
else:
    ext_modules = []

class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all grouped gemm installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''

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
        ext_modules=[
            CUDAExtension(name='deep_gemm_cpp',
                          sources=sources,
                          include_dirs=build_include_dirs,
                          libraries=build_libraries,
                          library_dirs=build_library_dirs,
                          extra_compile_args=cxx_flags)
        ],
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
            'bdist_wheel': CachedWheelsCommand,
        },
    )
