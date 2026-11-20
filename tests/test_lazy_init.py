"""Check lazy initialization, fork safety and runtime cleanup."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp
import deep_gemm


def main(local_rank: int):
    torch.cuda.set_device(local_rank)


_PROBE_SOURCE = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static uint32_t live_handles = 0, handle_token = 0;
static bool backend_dead = false;
static void verify_cleanup() {
    if (live_handles != 0) std::_Exit(92);
    std::fputs("DG_PROBE_CLEAN\n", stderr);
}
// Registered before the extension's static destructors, so this runs last.
__attribute__((constructor)) static void initialize_probe() {
    std::atexit(verify_cleanup);
}
static void stop_backend() { backend_dead = true; }
extern "C" uint32_t cublasLtCreate(void** handle) {
    std::atexit(stop_backend);
    *handle = &handle_token;
    ++live_handles;
    std::fputs("DG_PROBE_CREATE\n", stderr);
    return 0;
}
extern "C" uint32_t cublasLtDestroy(void* handle) {
    if (backend_dead) std::_Exit(91);
    if (handle != &handle_token or live_handles != 1) std::_Exit(93);
    --live_handles;
    std::fputs("DG_PROBE_DESTROY\n", stderr);
    return 0;
}
'''


@unittest.skipUnless(sys.platform.startswith('linux'), 'LD_PRELOAD requires Linux')
class TestNativeRuntimeShutdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        if not list((cls.repo / 'deep_gemm').glob('_C*.so')):
            raise unittest.SkipTest('requires the locally built DeepGEMM extension')
        compiler = shutil.which('c++')
        if compiler is None:
            raise unittest.SkipTest('requires a C++ compiler for the cuBLASLt probe')
        directory = tempfile.TemporaryDirectory(prefix='deep-gemm-shutdown-')
        cls.addClassCleanup(directory.cleanup)
        source = Path(directory.name) / 'probe.cpp'
        cls.probe = Path(directory.name) / 'probe.so'
        source.write_text(_PROBE_SOURCE)
        subprocess.run([compiler, '-shared', '-fPIC', '-std=c++17', str(source), '-o', str(cls.probe)],
                       check=True, capture_output=True, text=True, timeout=30)

    def test_native_shutdown(self):
        use_runtime = 'assert deep_gemm.get_tc_util() == deep_gemm.get_tc_util() == 100\n'
        close_runtime = textwrap.dedent('''\
            deep_gemm.shutdown()
            deep_gemm.shutdown()
            try:
                deep_gemm.get_tc_util()
            except RuntimeError:
                pass
            else:
                raise AssertionError('runtime was recreated after shutdown')
        ''')
        cases = (
            ('unused', '0', '', 0),
            ('owned', '0', use_runtime, 1),
            ('managed', '1', use_runtime, 0),
            ('explicit', '0', use_runtime + close_runtime, 1),
            ('managed_explicit', '1', use_runtime + close_runtime, 0),
            ('unused_explicit', '0', close_runtime, 0),
        )
        child_preamble = textwrap.dedent('''\
            from pathlib import Path
            import sys
            repo = Path(sys.argv[1]).resolve()
            sys.path.insert(0, str(repo))
            import deep_gemm
            assert Path(deep_gemm._C.__file__).absolute().parent == repo / 'deep_gemm'
        ''')
        for name, managed, body, expected_handles in cases:
            with self.subTest(case=name):
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = ''
                env['DG_USE_PYTORCH_CUBLASLT_HANDLE'] = managed
                env['LD_PRELOAD'] = str(self.probe) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
                result = subprocess.run([sys.executable, '-c', child_preamble + body, str(self.repo)],
                                        cwd=self.repo, env=env, capture_output=True, text=True, timeout=45)
                details = f'{name}: exit {result.returncode}\n{result.stdout}\n{result.stderr}'
                self.assertEqual(result.returncode, 0, details)
                events = result.stderr.splitlines()
                self.assertEqual(events.count('DG_PROBE_CREATE'), expected_handles, details)
                self.assertEqual(events.count('DG_PROBE_DESTROY'), expected_handles, details)
                self.assertEqual(events.count('DG_PROBE_CLEAN'), 1, details)

    def test_real_cublaslt_shutdown(self):
        child = textwrap.dedent('''\
            import torch
            if not torch.cuda.is_available():
                print('DG_NO_CUDA')
                raise SystemExit(0)
            import deep_gemm
            assert deep_gemm.get_tc_util() == 100
            print('DG_REAL_CUBLASLT_CREATED', flush=True)
        ''')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.repo) + os.pathsep + env.get('PYTHONPATH', '')
        env['DG_USE_PYTORCH_CUBLASLT_HANDLE'] = '0'
        # Expose the real cuBLASLt 13.1 use-after-free during native shutdown.
        # Freed storage must not retain values that hide the invalid second destroy.
        env['MALLOC_PERTURB_'] = '165'
        env['GLIBC_TUNABLES'] = env.get('GLIBC_TUNABLES', '') + ':glibc.malloc.tcache_count=0'
        result = subprocess.run([sys.executable, '-c', child], cwd=self.repo, env=env,
                                capture_output=True, text=True, timeout=90)
        details = f'exit {result.returncode}\n{result.stdout}\n{result.stderr}'
        self.assertEqual(result.returncode, 0, details)
        if 'DG_NO_CUDA' in result.stdout:
            self.skipTest('requires an available CUDA device')
        self.assertIn('DG_REAL_CUBLASLT_CREATED', result.stdout, details)

    def test_cuda_workspaces_are_released(self):
        child = textwrap.dedent('''\
            import atexit
            import os
            import torch
            if not torch.cuda.is_available():
                print('DG_NO_CUDA')
                raise SystemExit(0)
            allocated = []
            def check_cleanup():
                if os.environ['DG_USE_TEMP_CUBLASLT_WORKSPACE'] == '0':
                    assert allocated[0] - torch.cuda.memory_allocated() >= 2 * 16 * 1024 * 1024
                print('DG_CUDA_CLEAN', flush=True)
            # Runs after DeepGEMM's later-registered shutdown callback.
            atexit.register(check_cleanup)
            import deep_gemm
            a = torch.ones((32, 64), device='cuda', dtype=torch.bfloat16)
            b = torch.ones((48, 64), device='cuda', dtype=torch.bfloat16)
            outputs = [torch.empty((32, 48), device='cuda', dtype=torch.bfloat16) for _ in range(2)]
            streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
            streams[1].wait_stream(streams[0])
            for stream, output in zip(streams, outputs):
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        deep_gemm.cublaslt_gemm_nt(a, b, output)
            torch.cuda.synchronize()
            for output in outputs:
                assert torch.equal(output, torch.full_like(output, 64))
            allocated.append(torch.cuda.memory_allocated())
        ''')
        for managed in ('0', '1'):
            for temporary in ('0', '1'):
                with self.subTest(managed=managed, temporary=temporary):
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(self.repo) + os.pathsep + env.get('PYTHONPATH', '')
                    env['DG_USE_PYTORCH_CUBLASLT_HANDLE'] = managed
                    env['DG_USE_TEMP_CUBLASLT_WORKSPACE'] = temporary
                    result = subprocess.run([sys.executable, '-c', child], cwd=self.repo, env=env,
                                            capture_output=True, text=True, timeout=90)
                    details = f'exit {result.returncode}\n{result.stdout}\n{result.stderr}'
                    self.assertEqual(result.returncode, 0, details)
                    if 'DG_NO_CUDA' in result.stdout:
                        self.skipTest('requires an available CUDA device')
                    self.assertIn('DG_CUDA_CLEAN', result.stdout, details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lazy initialization')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    args, unittest_args = parser.parse_known_args()

    procs = [mp.Process(target=main, args=(i, ), ) for i in range(args.num_processes)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0, f'Lazy initialization worker exited with code {p.exitcode}'

    unittest.main(argv=[sys.argv[0], *unittest_args])
