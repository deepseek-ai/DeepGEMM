"""Graph timing and independent FP64 checks; never time oracle work."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import deep_gemm
import torch
import triton
import triton.testing
from experiments.sm120_grouped_swiglu_native import _valid_row_swiglu


def digest(value):
    return hashlib.sha256(value.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--sweep', type=int, required=True)
    args = p.parse_args()
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    scenes = []
    for k, n in ((896,4864),(4096,1024)):
        for m in (1,16,32,33,64,65,128,256):
            scenes.append((4,m,k,n,0,'uniform'))
    scenes += [(e,128,896,4864,0,'uniform') for e in (16,48)]
    scenes += [(4,m,896,4864,23,pattern) for m in (33,128)
               for pattern in ('empty','skew')]
    report = dict(variant=args.variant,sweep=args.sweep,complete=False,scenes=[],
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version',
                                     '--format=csv,noheader'],text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,clock_locked=False,
        rtol=0.01,atol=0.002,oracle='FP64 matmul -> FP32 -> BF16 projection -> FP32 SwiGLU -> BF16',
        scope='grouped gate/up GEMM plus SwiGLU; excludes dispatch/down/combine')
    def save():
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    for e,m,k,n,seed,pattern in scenes:
        capacity = max(128,m)
        rows = [m]*e if pattern=='uniform' else ([0,m//2,m,m//4] if pattern=='empty'
                                                     else [1,m//4,m//2,m])
        torch.manual_seed(seed)
        a = torch.randn((e,capacity,k),device='cuda',dtype=torch.bfloat16)
        b = torch.randn((e,2*n,k),device='cuda',dtype=torch.bfloat16)/k**0.5
        counts = torch.tensor(rows,device='cuda',dtype=torch.int32)
        projection = torch.full((e,capacity,2*n),9,device='cuda',dtype=torch.bfloat16)
        fused = torch.full((e,capacity,n),9,device='cuda',dtype=torch.bfloat16)
        separate = torch.full_like(fused,9)
        def ordinary():
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a,b,projection,counts,m)
        def activation():
            _valid_row_swiglu[(triton.cdiv(e*capacity,16),triton.cdiv(n,128))](
                projection,separate,counts,capacity,n,e,16,128,num_warps=4)
        def split():
            ordinary()
            activation()
        def combined():
            deep_gemm.m_grouped_bf16_swiglu_nt_masked(a,b,fused,counts,m)
        split(); combined(); torch.cuda.synchronize()
        entry = dict(id=f'e{e}-m{m}-k{k}-n{n}-seed{seed}-{pattern}',experts=e,
                     rows=rows,capacity=capacity,k=k,n=n,seed=seed,pattern=pattern,
                     input_sha256=dict(a=digest(a),b=digest(b)),oracle_checks=[])
        checked = range(e) if e<=4 else sorted({0,e//2,e-1})
        for group in checked:
            r=rows[group]
            if not r: continue
            reference=(a[group,:r].double() @ b[group].double().T).float().bfloat16().float()
            oracle=(torch.nn.functional.silu(reference[:,:n])*reference[:,n:]).bfloat16()
            for name,value in (('separate',separate[group,:r]),('fused',fused[group,:r])):
                delta=(value.float()-oracle.float()).abs()
                bad=int((delta > .002+.01*oracle.float().abs()).sum())
                entry['oracle_checks'].append(dict(group=group,path=name,bad=bad,
                    checked=delta.numel(),max_abs=float(delta.max())))
                torch.testing.assert_close(value,oracle,rtol=.01,atol=.002)
        for group,r in enumerate(rows):
            # Native ordinary masked GEMM may write padding rows. The fused
            # API explicitly preserves invalid rows; check that contract here.
            assert bool((fused[group,r:]==9).all())
            torch.testing.assert_close(fused[group,:r],separate[group,:r],rtol=.01,atol=.002)
        entry['output_sha256']={name:digest(torch.cat([value[g,:r].flatten()
            for g,r in enumerate(rows)])) for name,value in
            (('projection',projection),('separate',separate),('fused',fused))}
        functions=dict(ordinary=ordinary,activation=activation,separate=split,fused=combined)
        samples={name:[] for name in functions}
        for repeat in range(6):
            order=list(functions)
            if (repeat+args.sweep)%2: order.reverse()
            for name in order:
                value=1000*triton.testing.do_bench_cudagraph(functions[name],rep=20)
                assert math.isfinite(value) and value>0
                samples[name].append(value)
        entry.update(samples_us=samples,medians_us={n:statistics.median(v) for n,v in samples.items()})
        report['scenes'].append(entry); save()
        print(json.dumps(dict(scene=entry['id'],us=entry['medians_us'])),flush=True)
        del a,b,counts,projection,fused,separate,reference,oracle,delta,value
        torch.cuda.empty_cache()
    report['complete']=True; save()


if __name__=='__main__': main()
