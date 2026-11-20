"""Seal the Git source plus this extension, excluding generated experiment data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
if out.exists() or out.with_name(out.name+'.sha256').exists():
    raise FileExistsError(f'Refusing to overwrite a source seal: {out}')
names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root).decode().split('\0')
names = sorted(n for n in names if n and not n.startswith('artifacts/')
               and n != 'source-manifest.json' and (root/n).is_file())
manifest = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
(root/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
out.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(out, 'w:gz') as tar:
    for n in names + ['source-manifest.json']:
        tar.add(root/n, arcname='source/'+n, recursive=False)
checksum = hashlib.sha256(out.read_bytes()).hexdigest()
out.with_name(out.name+'.sha256').write_text(checksum+'  '+out.name+'\n')
print(out, checksum, len(manifest))
