"""Run the final revision's correctness, inherited contracts and sanitizers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
from deep_gemm import _C

ROOT = Path(__file__).resolve().parents[1]


def sources():
    names = ('deep_gemm/small_moe.py', 'deep_gemm/small_moe_policy.py',
             'csrc/apis/sm120_moe_small.hpp',
             'deep_gemm/include/deep_gemm/impls/sm120_bf16_moe_small.cuh',
             'deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh',
             'deep_gemm/include/deep_gemm/mma/sm120.cuh',
             'tests/test_sm120_single_token_gpu.py',
             'tests/test_sm120_single_token_policy.py',
             'tests/run_sm120_single_token_validation.py')
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    unit = [sys.executable, '-u', '-m', 'unittest']
    compact = ['tests.test_sm120_small_moe_gpu.SmallMoEGpuTest.'+n for n in (
        'test_kernel_template_matrix', 'test_controlled_dyadic_oracle',
        'test_saved_midpoints_in_both_fused_layers', 'test_sparse_high_expert_slots',
        'test_odd_topk_and_token_fallback', 'test_other_shape_and_custom_capacity_on_nondefault_stream')]
    compact += ['tests.test_sm120_single_token_gpu.SingleTokenGpuTest.'+n for n in (
        'test_candidate_templates_graph_guards_and_high_slots',
        'test_candidates_preserve_midpoint_and_cancellation')]
    steps = [('public-correctness', unit+[
        'tests.test_sm120_small_moe_policy', 'tests.test_sm120_single_token_policy',
        'tests.test_sm120_small_moe_gpu', 'tests.test_sm120_single_token_gpu', '-v']),
        ('original-contract', [sys.executable, '-u', '-m', 'pytest',
                               'tests/test_sm120_grouped_swiglu.py', '-vv']),
        ('accuracy-regression', unit+['tests.test_sm120_bf16_accumulation', '-v'])]
    for tool in ('memcheck', 'synccheck', 'racecheck', 'initcheck'):
        steps.append((tool, ['compute-sanitizer', '--tool', tool, '--error-exitcode', '99',
                             '--print-limit', '32']+unit+compact+['-v']))
    report = dict(complete=False, all_steps_passed=False, source_hashes=sources(),
        total_tasks=len(steps), steps=[],
        extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        extension_path=_C.__file__, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), rtol=.01, atol=.002,
        universal_correct_rounding_asserted=False)
    started = time.monotonic()
    save = lambda: (a.output/'completion.json').write_text(json.dumps(report, indent=2)+'\n')
    try:
        for n, (name, command) in enumerate(steps, 1):
            print(f'[{n}/{len(steps)}] {name}', flush=True)
            path = a.output/(name+'.log'); before = time.monotonic()
            with path.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            text = path.read_text(errors='replace')
            passed = result.returncode == 0 and 'skipped' not in text
            if name == 'original-contract':
                passed &= bool(re.search(r'[1-9][0-9]* passed', text))
            else:
                passed &= bool(re.search(r'Ran [1-9][0-9]* tests', text))
            if name in ('memcheck', 'synccheck', 'initcheck'):
                passed &= 'ERROR SUMMARY: 0 errors' in text
            if name == 'racecheck':
                passed &= '0 hazards displayed (0 errors, 0 warnings)' in text
            report['steps'].append(dict(name=name, passed=bool(passed), command=command,
                exit_code=result.returncode, seconds=time.monotonic()-before,
                log=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            save()
            if not passed: raise RuntimeError(f'Validation failed: {path}')
        assert sources() == report['source_hashes']
        report['complete'] = report['all_steps_passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started; save()


if __name__ == '__main__': main()
