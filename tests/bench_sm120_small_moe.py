"""Current-accuracy full-chain controls, independent FP64 oracle, Graph timing.

All controls use the inherited repaired BF16 MMA/epilogue. Research donor
kernels with different arithmetic are deliberately excluded from this study.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch
import triton

from deep_gemm import _C
from deep_gemm.small_moe import SmallMoE, _dispatch_slots, _swiglu, _combine
from tests.test_sm120_small_moe_gpu import make_routes


ROOT=Path(__file__).resolve().parents[1]
SOURCES=('csrc/apis/gemm.hpp','csrc/apis/sm120_moe_small.hpp','csrc/python_api.cpp',
 'csrc/jit_kernels/heuristics/sm120.hpp','csrc/jit_kernels/heuristics/sm120_config.hpp',
 'csrc/jit_kernels/impls/sm120_bf16_gemm.hpp',
 'deep_gemm/include/deep_gemm/mma/sm120.cuh',
 'deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh',
 'deep_gemm/include/deep_gemm/impls/sm120_bf16_moe_small.cuh',
 'deep_gemm/small_moe.py','deep_gemm/small_moe_policy.py',
 'tests/test_sm120_small_moe_gpu.py','tests/bench_sm120_small_moe.py')


def hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def telemetry():
    return subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,'
        'temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu',
        '--format=csv,noheader'],text=True).strip()


@torch.no_grad()
def independent_oracle(rt,x,slots,weights):
    # FP64 dot -> FP32 -> BF16 at both materialized projection boundaries.
    # Reduction uses explicit FP32 multiply then add in the same route order.
    hs=[]; ys=[]
    for token in range(rt.t):
        total=torch.zeros(rt.d,device=x.device,dtype=torch.float32)
        for route in range(rt.k):
            slot=int(slots[token,route]); expert=slot//rt.capacity
            projection=(rt.gate_up[expert].double() @ x[token].double()).float().bfloat16().float()
            gate,up=projection.chunk(2)
            h=(torch.nn.functional.silu(gate)*up).bfloat16()
            hs.append(h)
            partial=(rt.down[expert].double() @ h.double()).float().bfloat16().float()
            total=total+partial*weights[token,route]
        ys.append(total.bfloat16())
    return torch.stack(hs),torch.stack(ys)


def compare(actual,expected,*,exact):
    if not bool(torch.isfinite(actual).all()): raise RuntimeError('Nonfinite output')
    equal=bool(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
    if exact and not equal:
        raise RuntimeError(f'Bitwise mismatch: count={int((actual!=expected).sum())}, '
            f'max_abs={float((actual.float()-expected.float()).abs().max())}')
    if not exact: torch.testing.assert_close(actual,expected,rtol=.01,atol=.002)
    return dict(bitwise=equal,max_abs=float((actual.float()-expected.float()).abs().max()),
                differing=int((actual!=expected).sum()))


def run(args,job,seed):
    t,e,k,d,i=(job[x] for x in ('t','e','k','d','i'))
    torch.manual_seed(seed)
    options=dict(device='cuda',dtype=torch.bfloat16)
    gu=torch.randn((e,2*i,d),**options).div_(d**.5)
    down=torch.randn((e,d,i),**options).div_(i**.5)
    rt=SmallMoE(t,gu,down,k)
    x=torch.randn((t,d),**options)
    counts,slots,weights=make_routes(t,e,k,rt.capacity,seed,job['distribution'])
    rt.validate_routes(counts,slots,weights)
    owners=[]
    for name in ('h','output'):
        value=getattr(rt,name)
        owner=torch.full((value.numel()+32,),17.,**options)
        setattr(rt,name,owner[16:-16].view(value.shape))
        owners.append(owner)
    paths={f'separate{a}_{b}':('separate',a,b) for a in (0,32,64,128) for b in (0,32,64,128)}
    paths.update({f'public_l1_{b}':('public',0,b) for b in (0,32,64,128)})
    if rt.config.l1 is not None:
        paths.update({f'small_l1_{b}':('small',0,b) for b in (0,32,64,128)})
    paths['selected']=('selected',0,0)
    expected_m=triton.cdiv(t*k,e)

    def launch(label):
        kind,a,b=paths[label]
        if kind=='selected':
            rt.forward(x,counts,slots,weights)
            return
        if kind=='small':
            _C._sm120_moe_small(x,gu,slots,weights,rt.h,True,*rt.config.l1)
        else:
            _dispatch_slots[(t*k,triton.cdiv(d,256))](x,rt.packed,slots,d,k,256,num_warps=4)
            if kind=='public':
                _C.m_grouped_bf16_swiglu_nt_masked(rt.packed,gu,rt.h,counts,expected_m)
            else:
                _C.m_grouped_bf16_gemm_nt_masked(rt.packed,gu,rt.projection,counts,
                    expected_m,sm120_m_tile=a)
                _swiglu[(t*k,triton.cdiv(i,128))](rt.projection,rt.h,slots,i,128,num_warps=4)
        _C.m_grouped_bf16_gemm_nt_masked(rt.h,down,rt.expert,counts,expected_m,sm120_m_tile=b)
        _combine[(t,triton.cdiv(d,256))](rt.expert,slots,weights,rt.output,d,k,256,
            num_warps=4,enable_fp_fusion=False)

    graphs={}; validations=[]
    for replay in range(2):
        if replay:
            new=make_routes(t,e,k,rt.capacity,seed+6,job['distribution'])
            for dst,src in zip((counts,slots,weights),new): dst.copy_(src)
            x.neg_().add_(.125); gu.neg_(); down.mul_(.75)
            rt.validate_routes(counts,slots,weights)
        flat=slots.flatten()
        oracle_h,oracle_y=independent_oracle(rt,x,slots,weights)
        launch('separate0_0'); torch.cuda.synchronize()
        expected=rt.output.clone()
        expected_h=rt.h.view(-1,i)[flat].clone()
        oracle_checks=dict(H=compare(expected_h,oracle_h,exact=False),
                           Y=compare(expected,oracle_y,exact=False))
        for label in paths:
            rt.output.fill_(float('nan'))
            launch(label); torch.cuda.synchronize()
            public=paths[label][0]=='public'
            compare(rt.h.view(-1,i)[flat],expected_h,exact=not public)
            compare(rt.output,expected,exact=not public)
            # Also compare each path directly with the independent oracle.
            compare(rt.h.view(-1,i)[flat],oracle_h,exact=False)
            compare(rt.output,oracle_y,exact=False)
            if replay==0:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.repeats): launch(label)
                graphs[label]=graph
            rt.output.fill_(float('nan'))
            graphs[label].replay(); torch.cuda.synchronize()
            compare(rt.output,expected,exact=not public)
            if not all(bool((o[:16]==17).all() and (o[-16:]==17).all()) for o in owners):
                raise RuntimeError('Output guard overwritten '+label)
        validations.append(dict(replay=replay,oracle=oracle_checks,graph=True,guards=True,
                                output_sha256=digest(expected),h_sha256=digest(expected_h)))
    raw=[]; rng=random.Random(seed)
    for batch in range(args.batches):
        labels=list(paths); rng.shuffle(labels)
        for label in labels:
            graphs[label].replay()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record(); graphs[label].replay(); end.record(); end.synchronize()
            raw.append(dict(batch=batch,label=label,us=begin.elapsed_time(end)*1000/args.repeats))
    medians={label:statistics.median(r['us'] for r in raw if r['label']==label) for label in paths}
    best=min((l for l in paths if l.startswith('separate')),key=medians.get)
    public=min((l for l in paths if l.startswith('public')),key=medians.get)
    small=min((l for l in paths if l.startswith('small')),key=medians.get,default=None)
    result=dict(job=job,seed=seed,config=repr(rt.config),validation=validations,
        fusion_enabled=rt.config.l1 is not None and rt.config.l2 is not None,
        raw=raw,medians_us=medians,best_separate=best,best_public_l1=public,best_small_l1=small,
        speedup=medians[best]/medians['selected'],
        speedup_vs_public_l1=medians[public]/medians['selected'],
        ordinary_fallback_control=f'separate{rt.fallback_m_tile}_{rt.fallback_m_tile}',
        telemetry=telemetry())
    del graphs,launch,rt,gu,down,owners
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[277,281])
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--batches',type=int,default=7)
    args=p.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    report=dict(complete=False,jobs=[],source_hashes=hashes(),
        extension_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        telemetry_before=telemetry(),torch=torch.__version__,triton=triton.__version__,
        inherited_accuracy_revision='1c31df4b4969ca668bee71c548bc4b023c29ca52',
        independent_oracle='FP64 dot -> FP32 -> BF16 gate/up -> FP32 SwiGLU -> BF16 H; '
            'FP64 down dot -> FP32 -> BF16 partial -> ordered FP32 multiply/add -> BF16 Y',
        rtol=.01,atol=.002,repeats=args.repeats,batches=args.batches,
        scope='X to Y, including dispatch, both layers, activation and combine',
        route_plan_precomputed=True,clocks_locked=False,performance_is_diagnostic=True)
    try:
        jobs=json.loads(args.jobs.read_text())
        report['total_tasks']=len(jobs)*len(args.seeds)
        for seed in args.seeds:
            for job in jobs:
                print(f'[{len(report["jobs"])+1}/{report["total_tasks"]}] seed={seed} {job}',flush=True)
                result=run(args,job,seed); report['jobs'].append(result)
                print(f'  speedup={result["speedup"]:.4f} selected={result["medians_us"]["selected"]:.3f}us',flush=True)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
        report['source_unchanged']=hashes()==report['source_hashes']
        if not report['source_unchanged']: raise RuntimeError('Source changed during benchmark')
        report['complete']=True
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['telemetry_after']=telemetry()
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
