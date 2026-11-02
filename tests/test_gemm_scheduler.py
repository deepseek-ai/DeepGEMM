"""Host-only integer coverage tests for the actual GEMM scheduler header.

Run with `python tests/test_gemm_scheduler.py` (C++17 compiler required).
No PyTorch, CUDA toolkit, GPU, or initialized submodules are needed. Only the
CUDA/CUTLASS dependencies are shimmed; scheduler/gemm.cuh and common/types.cuh
are included from the source tree, not copied or translated. This checks tile
enumeration, not CUDA compilation, TMA, synchronization, or kernel correctness.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


HOST_MATH = r"""
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#define CUTLASS_DEVICE
#define CUTLASS_HOST_DEVICE
#define DG_STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#define DG_TRAP_ONLY_DEVICE_ASSERT(...) assert(__VA_ARGS__)
using std::min;
namespace cute {
using std::max;
using std::numeric_limits;
}
// A single CTA's block index, assigned before constructing its scheduler.
inline struct { uint32_t x; } blockIdx;
namespace deep_gemm::math {
template <typename T> constexpr T ceil_div(T a, T b) { return (a + b - 1) / b; }
template <typename T> constexpr T constexpr_ceil_div(T a, T b) { return ceil_div(a, b); }
template <typename T> constexpr T align(T a, T b) { return ceil_div(a, b) * b; }
}
"""


class TestGemmScheduler(unittest.TestCase):
    def test_grouped_coverage(self):
        root = Path(__file__).resolve().parents[1]
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            math_header = temporary / 'deep_gemm/common/math.cuh'
            math_header.parent.mkdir(parents=True)
            math_header.write_text(HOST_MATH)
            # common/types.cuh needs no MMA descriptors for its enums.
            cute_header = temporary / 'cute/arch/mma_sm100_desc.hpp'
            cute_header.parent.mkdir(parents=True)
            cute_header.write_text('#pragma once\n')
            for arch in (900, 1000):
                with self.subTest(arch=arch):
                    executable = temporary / f'scheduler_{arch}'
                    subprocess.run([
                        *compiler, '-std=c++17', '-O1', '-g',
                        *shlex.split(os.environ.get('CXXFLAGS', '')),
                        f'-D__CUDA_ARCH__={arch}', '-I', str(temporary),
                        '-I', str(root / 'deep_gemm/include'),
                        str(root / 'tests/gemm_scheduler_host.cpp'),
                        '-o', str(executable),
                    ], check=True, timeout=120)
                    subprocess.run([str(executable)], check=True, timeout=60)


if __name__ == '__main__':
    unittest.main()
