import os
import setuptools
import shutil
import subprocess
import torch
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import re


def get_ppu_sdk_version():
    try:
        output = subprocess.check_output(['clang', '--version'], stderr=subprocess.STDOUT).decode('utf-8')
        match = re.search(r'\((\d+)\.(\d+)\.(\d+)(?:-[^)]*)?\)', output)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except Exception:
        pass
    try:
        output = subprocess.check_output(['hgcc', '--version'], stderr=subprocess.STDOUT).decode('utf-8')
        match = re.search(r'Release version (\d+)\.(\d+)\.(\d+)', output)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except Exception:
        pass
    return (0, 0, 0)


current_dir = os.path.dirname(os.path.realpath(__file__))

# PPU SDK path — provides hggc headers for actlize
ppu_sdk = os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')
ppu_include = os.path.join(ppu_sdk, 'targets', 'x86_64-linux', 'include')

sources = ['csrc/python_api.cpp']
build_include_dirs = [
    ppu_include,
    current_dir + '/deep_gemm/include',
    current_dir + '/third-party/actlize_v1.0.0/',
    current_dir + '/third-party/actlize_v1.0.0/include',
    current_dir + '/third-party/actlize_v1.0.0/tools',
    current_dir + '/third-party/fmt/include',
    current_dir + '/third-party/actlize_v1.0.0/include/cute',
]

build_libraries = ['hggc', 'hggcrt1', 'hgrtc', 'acblasLt']
build_library_dirs = [
    os.path.join(ppu_sdk, 'lib'),
]

third_party_include_dirs = [
    'third-party/actlize_v1.0.0/include/cute',
    'third-party/actlize_v1.0.0/include/cutlass',
    'third-party/actlize_v1.0.0/include/ppu_include.hpp',
    'third-party/actlize_v1.0.0/include/accutlass.hpp',
    'third-party/actlize_v1.0.0/tools'
]

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.make_jit_include_symlinks()

    @staticmethod
    def make_jit_include_symlinks():
        # Make symbolic links of third-party include directories, one subdirectory per actlize
        # library so that their colliding headers do not shadow each other
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            actlize_dirname = d.split('/')[1]
            src_dir = f'{current_dir}/{d}'
            dst_parent = f'{current_dir}/deep_gemm/include/{actlize_dirname}'
            dst_dir = f'{dst_parent}/{dirname}'
            assert os.path.exists(src_dir)
            os.makedirs(dst_parent, exist_ok=True)

            if os.path.islink(dst_dir) or os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)

            os.symlink(src_dir, dst_dir, target_is_directory=True)


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
            actlize_dirname = d.split('/')[1]
            src_dir = os.path.join(current_dir, d)
            include_dir = os.path.join(build_include_dir, actlize_dirname)
            os.makedirs(include_dir, exist_ok=True)
            dst_dir = os.path.join(include_dir, dirname)

            # Remove existing directory if it exists

            # Copy the directory
            if os.path.isfile(src_dir):
                if os.path.exists(dst_dir):
                    os.remove(dst_dir)
                shutil.copy(src_dir, dst_dir)
            else:
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)

def custom_local_scheme(version):
    return 'dev%03d.%s' % (version.distance, version.short_node)

def custom_version_scheme(version):
    return '2.0.0'

if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''
    extra_cxx_args = ["-O3", "-std=c++17", "-DUSE_HGGC"]
    extra_hgcc_args = ["-O3", "-std=c++17", "--use_fast_math", "-DUSE_HGGC"]

    ppu_sdk_version = get_ppu_sdk_version()
    if ppu_sdk_version >= (2, 0, 0):
        extra_cxx_args.append("-DDG_HGGC_SUPPORT_PCH=1")
        extra_hgcc_args.append("-DDG_HGGC_SUPPORT_PCH=1")

    ext_modules = []
    ext_modules.append(
        CppExtension(name='deep_gemm.deep_gemm_cpp',
                        sources=sources,
                        include_dirs=build_include_dirs,
                        libraries=build_libraries,
                        library_dirs=build_library_dirs,
                        extra_compile_args={
            "cxx": extra_cxx_args,
            "hgcc": extra_hgcc_args,
        }))
    setuptools.setup(
        name='deep_gemm',
        use_scm_version={
            "local_scheme": custom_local_scheme,
            "version_scheme": custom_version_scheme,
        },
        packages=find_packages('.'), # old version: packages=['deep_gemm', 'deep_gemm/jit', 'deep_gemm/jit_kernels', 'deep_gemm/deep_gemm_tuner'],,
        package_data={
            'deep_gemm': [
                'include/deep_gemm/**/*',
                'include/actlize_v1.0.0/**/*',
                'deep_gemm_tuner/configs/*',
            ]
        },
        ext_modules=ext_modules,
        zip_safe=False,
        cmdclass={
            'develop': PostDevelopCommand,
            'build_py': CustomBuildPy,
            'build_ext': BuildExtension,
        },
    )
