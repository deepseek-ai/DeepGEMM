import argparse
import importlib
import inspect
import os
import subprocess
import sys

import deep_gemm


# Single test template
script_dir = os.path.dirname(os.path.abspath(__file__))
test_template = """
import random
import sys
import torch

# Necessary for `generators.py`
sys.path.append('{script_dir}')

torch.manual_seed(0)
random.seed(0)

from tests.{module_name} import {func_name}
{func_name}()
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--funcs', type=str, default='all')
    parser.add_argument('--tools', type=str, default='memcheck,synccheck')
    args = parser.parse_args()

    if args.funcs != 'all':
        funcs = []
        for name in [x.strip() for x in args.funcs.split(',')]:
            module_name, func_name = name.split('.')
            funcs.append((module_name, func_name))
    else:
        # Get all test functions except those related to cuBLAS
        files = [f for f in os.listdir(script_dir) if f.endswith('.py')]
        exclude_files = ['test_sanitizer.py', 'generators.py']
        funcs = [
            (module_name, name)
            for module_name in [os.path.splitext(f)[0] for f in files if f not in exclude_files]
            for name, obj in inspect.getmembers(importlib.import_module(module_name))
            if inspect.isfunction(obj) and name.startswith('test') and 'test_filter' not in name
        ]
    tools = [x.strip() for x in args.tools.split(',')]

    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['DG_JIT_PTXAS_CHECK'] = '1'
    env['DG_USE_NVIDIA_TOOLS'] = '1'
    env['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    env['TORCH_SHOW_CPP_STACKTRACES'] = '1'

    print(f'Library path: {deep_gemm.__path__}')
    for module_name, func_name in funcs:
        for tool in tools:
            cmd = [
                '/usr/local/cuda/bin/compute-sanitizer',
                f'--tool={tool}',
                '--target-processes=application-only',
                '--destroy-on-device-error=context',
                '--force-blocking-launches',
                '--check-api-memory-access=no',
                '--kernel-name-exclude', 'kns=nvjet',
                'python',
                '-c',
                test_template.format(module_name=module_name, func_name=func_name, script_dir=script_dir)
            ]
            print(f'\n{"=" * 60}')
            print(f'Running {module_name}.{func_name} with compute-sanitizer {tool}')
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                sys.exit(result.returncode)
