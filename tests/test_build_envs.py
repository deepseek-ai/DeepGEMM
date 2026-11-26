"""Exercise the actual build method without running setup.py import-time code."""
import ast
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

source = Path(os.environ.get('DEEPGEMM_TEST_SOURCE', Path(__file__).resolve().parents[1])) / 'setup.py'
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CustomBuildPy')
node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_default_envs')
namespace = {'os': os}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
generate = namespace['generate_default_envs']
NAMES = ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_CPP_STANDARD')
VALUES = ('/tmp/dg-cache', "/tmp/Hardy's cache", '/tmp/"cache"', r'/tmp/run\tmp',
          '/tmp/cache\\', '/tmp/dg\ncache', '缓存/算子', '', 'a\rb\tc')

class TestBuildEnvs(unittest.TestCase):
    def check_roundtrip(self, expected):
        with patch.dict(os.environ, expected, clear=True), tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'deep_gemm'
            path.mkdir()
            generate(types.SimpleNamespace(build_lib=folder))
            text = (path / 'envs.py').read_text()
            result = {}
            exec(compile(text, 'generated-envs.py', 'exec'), result)
            self.assertEqual(result['persistent_envs'], expected)

    def test_unset(self):
        self.check_roundtrip({})

    def test_all_set(self):
        self.check_roundtrip(dict(zip(NAMES, (VALUES[1], VALUES[3], VALUES[5]))))

for i, name in enumerate(NAMES):
    for j, value in enumerate(VALUES):
        def test(self, name=name, value=value):
            self.check_roundtrip({name: value})
        setattr(TestBuildEnvs, f'test_variable_{i}_value_{j}', test)

if __name__ == '__main__':
    unittest.main()
