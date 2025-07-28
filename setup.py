import os
import setuptools
import shutil
import subprocess
from setuptools import find_packages
from setuptools.command.build_py import build_py


current_dir = os.path.dirname(os.path.realpath(__file__))
cxx_flags = ['-std=c++20', '-O3', '-fPIC', '-Wno-psabi']
sources = ['csrc/python_api.cpp']


def get_cuda_home():
    """Get CUDA_HOME path, with fallback to environment variable"""
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        return CUDA_HOME
    except ImportError:
        import os
        return os.environ.get('CUDA_HOME', '/usr/local/cuda')


def get_build_include_dirs():
    """Get build include directories"""
    cuda_home = get_cuda_home()
    return [
        f'{cuda_home}/include',
        'deep_gemm/include',
        'third-party/cutlass/include',
        'third-party/fmt/include',
    ]


def get_build_library_dirs():
    """Get build library directories"""
    cuda_home = get_cuda_home()
    return [
        f'{cuda_home}/lib64',
        f'{cuda_home}/lib64/stub'
    ]


build_libraries = ['cuda', 'cudart']
third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]


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
        for name in ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_DISABLE_SHORTCUT_CACHE'):
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


if __name__ == '__main__':
    from torch.utils.cpp_extension import CppExtension
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''

    # noinspection PyTypeChecker
    setuptools.setup(
        name='deep_gemm',
        version='2.0.0' + revision,
        packages=find_packages('.'),
        install_requires=[
            'torch>=2.1.0',
        ],
        package_data={
            'deep_gemm': [
                'include/deep_gemm/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
            ]
        },
        ext_modules=[
            CppExtension(name='deep_gemm_cpp',
                         sources=sources,
                         include_dirs=get_build_include_dirs(),
                         libraries=build_libraries,
                         library_dirs=get_build_library_dirs(),
                         extra_compile_args=cxx_flags)
        ],
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
        },
    )
