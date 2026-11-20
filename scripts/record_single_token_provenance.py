"""Record reused native build identity and the actual remote dependency trees."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final', type=Path, required=True)
    p.add_argument('--donor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.final/'source-manifest.json').read_text())
    native = [n for n in manifest if n.startswith(('csrc/', 'deep_gemm/include/deep_gemm/'))
              or n in ('setup.py', 'CMakeLists.txt')]
    same = {n: sha(a.final/n) == sha(a.donor/n) for n in native}
    assert all(same.values()), [n for n, equal in same.items() if not equal]
    extensions = {v.name: sha(v) for v in (a.final/'deep_gemm').glob('_C*.so')}
    assert extensions
    for name, digest in extensions.items():
        assert digest == sha(a.donor/'deep_gemm'/name)
    dependencies = {}
    for name in ('cutlass', 'deep_jit'):
        folder = a.final/'third-party'/name
        files = {str(v.relative_to(folder)): sha(v) for v in sorted(folder.rglob('*'))
                 if v.is_file() and '.git' not in v.relative_to(folder).parts}
        assert files
        dependencies[name] = dict(files=len(files),
            tree_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            file_sha256=files)
    report = dict(complete=True, native_build_reused=True, donor=str(a.donor),
        identical_native_files=len(native), native_hashes={n: sha(a.final/n) for n in native},
        extensions=extensions, dependencies=dependencies,
        note='Final native build is byte-identical to dev1; Python policy changed before final validation.')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Native sources identical:', len(native), 'extensions:', extensions)
