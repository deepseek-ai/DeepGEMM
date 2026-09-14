import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch


def _make_test_module(directory):
    utils_path = Path(__file__).resolve().parents[1] / 'deep_gemm' / 'testing' / 'utils.py'
    path = Path(directory) / 'test_discovery_sample.py'
    path.write_text(f'''
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('testing_utils', {str(utils_path)!r})
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
test_filter = utils.test_filter
test_helper_alias = test_filter

@test_filter(lambda: True)
def test_enabled():
    Path(__file__).with_suffix('.ran').write_text('executed')

@test_filter(lambda: False)
def test_disabled():
    raise AssertionError('filtered test must not execute')

def test_filter_behavior():
    pass

def test_opted_out():
    raise AssertionError('opted-out test must not execute')

test_opted_out.__test__ = False
''')
    return path


def test_pytest_discovery():
    with tempfile.TemporaryDirectory() as directory:
        path = _make_test_module(directory)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='')
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', path.name]
        collected = subprocess.run(command + ['--collect-only'], cwd=directory, env=env,
                                   capture_output=True, text=True, timeout=60)
        assert collected.returncode == 0, collected.stdout + collected.stderr
        assert [line for line in collected.stdout.splitlines() if '::' in line] == [
            'test_discovery_sample.py::test_enabled',
            'test_discovery_sample.py::test_disabled',
            'test_discovery_sample.py::test_filter_behavior',
        ]
        result = subprocess.run(command + ['-s'], cwd=directory, env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert '3 passed' in result.stdout
        assert 'test_disabled:\n > Filtered by ' in result.stdout
        assert path.with_suffix('.ran').read_text() == 'executed'


def test_sanitizer_discovery():
    with tempfile.TemporaryDirectory() as directory:
        path = _make_test_module(directory)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        package = ModuleType('deep_gemm')
        package.__path__ = []
        runner = Path(__file__).with_name('test_sanitizer.py')
        with patch.dict(sys.modules, deep_gemm=package), \
             patch.object(sys, 'argv', [str(runner)]), \
             patch.object(os, 'listdir', return_value=[path.name, 'test_sanitizer.py', 'generators.py', 'test_mega_moe.py']), \
             patch.object(importlib, 'import_module', return_value=module) as import_module, \
             patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            namespace = runpy.run_path(str(runner), run_name='__main__')
        import_module.assert_called_once_with(path.stem)
        expected = ['test_disabled', 'test_enabled', 'test_filter_behavior']
        assert namespace['funcs'] == [(path.stem, name) for name in expected]
        assert run.call_count == 6
        for call, (name, tool) in zip(run.call_args_list,
                                     [(name, tool) for name in expected for tool in ['memcheck', 'synccheck']]):
            command = call.args[0]
            assert command[0] == '/usr/local/cuda/bin/compute-sanitizer'
            assert command[1] == f'--tool={tool}'
            assert f'from {path.stem} import {name}\n{name}()' in command[-1]
