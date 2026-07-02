from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_cuda_home() -> str:
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        nvcc = shutil.which('nvcc')
        if nvcc is not None:
            cuda_home = str(Path(nvcc).parent.parent)
        else:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None

    assert cuda_home is not None
    return cuda_home


def get_cuda_arch(default: str = '9.0') -> str:
    try:
        status = subprocess.run(
            args=['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            check=True,
        )
        return status.stdout.decode('utf-8').strip().split('\n')[0]
    except Exception:
        return default
