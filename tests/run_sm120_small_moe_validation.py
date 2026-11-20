"""Save correctness and four-sanitizer receipts for the final source revision."""
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


ROOT=Path(__file__).resolve().parents[1]
SOURCES=('csrc/python_api.cpp','csrc/apis/sm120_moe_small.hpp',
         'deep_gemm/include/deep_gemm/impls/sm120_bf16_moe_small.cuh',
         'deep_gemm/small_moe.py','deep_gemm/small_moe_policy.py',
         'tests/test_sm120_small_moe_gpu.py','tests/test_sm120_small_moe_policy.py',
         'tests/bench_sm120_small_moe.py','tests/run_sm120_small_moe_validation.py')
SOURCES += ('csrc/apis/gemm.hpp','csrc/jit_kernels/heuristics/sm120.hpp',
            'csrc/jit_kernels/heuristics/sm120_config.hpp','deep_gemm/include/deep_gemm/mma/sm120.cuh',
            'deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh',
            'csrc/jit_kernels/impls/sm120_bf16_gemm.hpp')


def hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12,0):
        raise RuntimeError('SM120 GPU required; skipped tests are not validation')
    report=dict(complete=False,all_steps_passed=False,steps=[],source_hashes=hashes(),
        extension_path=_C.__file__,extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version',
            '--format=csv,noheader'],text=True).strip(),
        inherited_accuracy_revision='1c31df4b4969ca668bee71c548bc4b023c29ca52',
        universal_correct_rounding_asserted=False,pr_acceptance_asserted=False)
    base=[sys.executable,'-u','-m','unittest']
    unit=base+['tests.test_sm120_small_moe_policy','tests.test_sm120_small_moe_gpu','-v']
    compact=base+['tests.test_sm120_small_moe_gpu.SmallMoEGpuTest.'+name for name in (
        'test_kernel_template_matrix','test_controlled_dyadic_oracle',
        'test_saved_midpoints_in_both_fused_layers',
        'test_sparse_high_expert_slots',
        'test_odd_topk_and_token_fallback',
        'test_other_shape_and_custom_capacity_on_nondefault_stream')]+['-v']
    steps=[('correctness',unit)]
    for tool in ('memcheck','synccheck','racecheck','initcheck'):
        steps.append((tool,['compute-sanitizer','--tool',tool,'--error-exitcode','99',
                           '--print-limit','32']+compact))
    report['total_tasks']=len(steps)
    started=time.monotonic()
    try:
        for index,(name,command) in enumerate(steps):
            print(f'[{index+1}/{len(steps)}] {name}',flush=True)
            before=time.monotonic()
            log=args.output/(name+'.log')
            with log.open('wb') as stream:
                completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            text=log.read_text(errors='replace')
            clean=bool(re.search(r'ERROR SUMMARY: 0 errors',text))
            if name=='racecheck':
                clean='0 hazards displayed (0 errors, 0 warnings)' in text
            passed=(completed.returncode==0 and bool(re.search(r'Ran [1-9][0-9]* tests',text))
                    and 'skipped' not in text)
            if name!='correctness': passed=passed and clean
            report['steps'].append(dict(name=name,command=command,exit_code=completed.returncode,
                log=log.name,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                elapsed_seconds=time.monotonic()-before,passed=passed,
                sanitizer_clean=clean if name!='correctness' else None))
            (args.output/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
            if not passed: raise RuntimeError('Failed validation: '+name+'; inspect '+str(log))
        report['source_unchanged']=hashes()==report['source_hashes']
        if not report['source_unchanged']: raise RuntimeError('Source changed during validation')
        report['complete']=report['all_steps_passed']=True
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'completion.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
