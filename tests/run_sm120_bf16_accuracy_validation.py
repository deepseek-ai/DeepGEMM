"""Collect revision-matched numerical and sanitizer receipts for PR #501."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'csrc/python_api.cpp', 'csrc/apis/gemm.hpp',
    'csrc/jit_kernels/impls/sm120_bf16_gemm.hpp',
    'deep_gemm/include/deep_gemm/mma/sm120.cuh',
    'deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh',
    'tests/test_sm120_grouped_swiglu.py', 'tests/test_sm120_bf16_accumulation.py',
    'tests/run_sm120_bf16_accuracy_validation.py',
    'tests/fixtures/sm120_moe/midpoint-inputs.pt',
    'tests/fixtures/sm120_moe/midpoint-seed2-inputs.pt',
    'experiments/sm120_grouped_swiglu_native.py',
)


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def main():
    import torch
    from deep_gemm import _C

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12,0):
        raise RuntimeError('SM120 GPU required; skipped tests are not validation')
    py = [sys.executable, '-u']
    compact = py + ['-m','unittest'] + [
        'tests.test_sm120_bf16_accumulation.Bf16AccumulationTest.'+name for name in (
            'test_saved_midpoint_projections_and_fp32_output',
            'test_saved_double_rounding_projection',
            'test_odd_shape_graph_on_nondefault_stream',
        )] + ['-v']
    steps = [
        ('midpoint-regression', py+['-m','unittest','tests.test_sm120_bf16_accumulation','-v']),
        ('original-contract', py+['-m','pytest','tests/test_sm120_grouped_swiglu.py','-vv']),
        ('native-regression', py+['-m','pytest','tests/test_sm120_bf16_gemm.py',
            'tests/test_sm120_bf16_odd_n.py','tests/test_sm120_native_rejections.py',
            '-k','not k_grouped_small_stride and not test_filter','-vv']),
    ]
    for tool in ('memcheck','synccheck','racecheck','initcheck'):
        steps.append((tool, ['compute-sanitizer','--tool',tool,'--error-exitcode','99',
                             '--print-limit','32']+compact))
    for rows in (16,128):
        steps.append((f'timing-m{rows}', py+['experiments/sm120_grouped_swiglu_native.py',
            '--rows', *([str(rows)]*4), '--max-m','128','--repeats','12',
            '--output',str(args.output/f'timing-m{rows}.json')]))
    started = time.monotonic()
    report = dict(complete=False,all_steps_passed=False,total_tasks=len(steps),steps=[],
        source_hashes=hashes(),extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version',
                                     '--format=csv,noheader'],text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,
        historical_failure_preserved=True,clocks_locked=False)
    def save():
        (args.output/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for index,(name,command) in enumerate(steps,1):
            print(f'[{index}/{len(steps)}] {name}',flush=True)
            log=args.output/(name+'.log')
            before=time.monotonic()
            with log.open('wb') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            text=log.read_text(errors='replace')
            passed=result.returncode==0
            if name in ('midpoint-regression','memcheck','synccheck','racecheck','initcheck'):
                passed=passed and bool(re.search(r'Ran [1-9][0-9]* tests',text)) and 'skipped' not in text
            if name in ('original-contract','native-regression'):
                passed=passed and bool(re.search(r'[1-9][0-9]* passed',text)) and 'skipped' not in text
            if name in ('memcheck','synccheck','initcheck'):
                passed=passed and 'ERROR SUMMARY: 0 errors' in text
            if name=='racecheck':
                passed=passed and '0 hazards displayed (0 errors, 0 warnings)' in text
            if name.startswith('timing-'):
                passed=passed and (args.output/(name+'.json')).is_file()
            report['steps'].append(dict(name=name,command=command,passed=passed,
                exit_code=result.returncode,log=log.name,
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                elapsed_seconds=time.monotonic()-before))
            save()
            if not passed:
                raise RuntimeError('Failed step '+name+'; inspect '+str(log))
        report['source_unchanged']=report['source_hashes']==hashes()
        if not report['source_unchanged']:
            raise RuntimeError('Source changed during validation')
        report['complete']=report['all_steps_passed']=True
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        save()


if __name__=='__main__':
    main()
