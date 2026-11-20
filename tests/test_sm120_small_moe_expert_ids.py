"""Direct expert-ID routing must preserve the safe-slot arithmetic and bounds."""
import os
import unittest
import torch
import triton
import triton.language as tl

HAS_SM120=torch.cuda.is_available() and torch.cuda.get_device_capability()==(12,0)


@triton.jit
def prepare_safe_routes(IDS, WEIGHTS, SLOTS, SAFE_WEIGHTS, E: tl.constexpr,
                        K: tl.constexpr, B: tl.constexpr):
    """Independent safe-slot control; no optional framework adapter import."""
    r = tl.arange(0, B)
    expert = tl.load(IDS + r, r < K, 0).to(tl.int64)
    valid = (expert >= 0) & (expert < E)
    slot = tl.where(valid, expert * 128, r + 1)
    weight = tl.load(WEIGHTS + r, r < K, 0)
    tl.store(SLOTS + r, slot, r < K)
    tl.store(SAFE_WEIGHTS + r, tl.where(valid, weight, 0.), r < K)


@unittest.skipUnless(HAS_SM120,'SM120 GPU required')
class ExpertIDsTest(unittest.TestCase):
    @torch.no_grad()
    def test_safe_slots_and_direct_ids_match_with_guards_and_graphs(self):
        from deep_gemm import _C
        from deep_gemm.small_moe_policy import select_config, select_single_token_config
        cases=[(2048,768,128,1),(2048,768,128,8),(4096,1536,128,8),
               (2048,1408,60,4),(2048,1408,64,6)]
        instrumented=os.environ.get('SM120_DIRECT_SANITIZER')=='1'
        if instrumented:cases=[(2048,1408,60,4)]
        for d,i,policy_e,k in cases:
            torch.manual_seed(28197+k)
            e=8
            x=torch.randn((1,d),device='cuda',dtype=torch.bfloat16)
            gu=torch.randn((e,2*i,d),device='cuda',dtype=torch.bfloat16)*.02
            down=torch.randn((e,d,i),device='cuda',dtype=torch.bfloat16)*.02
            config=select_single_token_config(1,policy_e,k,d,i)
            if config.l1 is None:
                config=select_config(1,policy_e,k,d,i)
            self.assertIsNotNone(config.l1)
            gu_before,down_before=gu.clone(),down.clone()
            for swapped in ([True] if instrumented else [False,True]):
                weight=gu if not swapped else torch.cat((gu[:,i:],gu[:,:i]),dim=1)
                weight_before=weight.clone()
                for dtype in [torch.int32,torch.int64]:
                    with self.subTest(d=d,i=i,k=k,swapped=swapped,dtype=dtype):
                        ids=torch.empty((1,k),device='cuda',dtype=dtype)
                        rw=torch.empty((1,k),device='cuda',dtype=torch.float32)
                        slots=torch.empty((1,k),device='cuda',dtype=torch.int64)
                        safe=torch.empty_like(rw)
                        owners=[torch.full((e*128*i+256,),17,device='cuda',dtype=torch.bfloat16)
                                for _ in range(2)]
                        hs=[p[128:-128].view(e,128,i) for p in owners]
                        ys_owner=[torch.full((d+256,),19,device='cuda',dtype=torch.bfloat16)
                                  for _ in range(2)]
                        ys=[p[128:-128].view(1,d) for p in ys_owner]
                        def forward(direct):
                            if direct:
                                routes,weights,kw=ids,rw,{'expert_ids':True}
                            else:
                                prepare_safe_routes[(1,)](ids,rw,slots,safe,e,k,triton.next_power_of_2(k),num_warps=1)
                                routes,weights,kw=slots,safe,{}
                            _C._sm120_moe_small(x,weight,routes,weights,hs[direct],True,*config.l1,
                                               gate_up_swapped=swapped,**kw)
                            _C._sm120_moe_small(hs[direct],down,routes,weights,ys[direct],False,*config.l2,**kw)
                        ids.fill_(-1);rw.fill_(float('nan'))
                        for direct in [False,True]:forward(direct)
                        graphs=[]
                        for direct in [False,True]:
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):forward(direct)
                            graphs.append(graph)
                        valid=list(range(k))
                        mixed=[r if r%2==0 else -1 for r in range(k)]
                        extreme=[r if r%2==0 else torch.iinfo(dtype).max for r in range(k)]
                        scenes=[[-1]*k,valid,mixed,extreme,
                                [torch.iinfo(dtype).min]+list(range(1,k)),list(reversed(valid))]
                        for route in scenes:
                            ids.copy_(torch.tensor([route],device='cuda',dtype=dtype))
                            rw.copy_(torch.tensor([[.05*(r+1) if 0<=idx<e else float('nan')
                                                   for r,idx in enumerate(route)]],device='cuda'))
                            ids_before,rw_before=ids.clone(),rw.clone()
                            for replay in [False,True]:
                                for h in hs:h.fill_(float('nan'))
                                for y in ys:y.fill_(float('nan'))
                                if replay:
                                    for graph in graphs:graph.replay()
                                else:
                                    for direct in [False,True]:forward(direct)
                                torch.cuda.synchronize()
                                self.assertTrue(torch.equal(hs[0].view(torch.int16),hs[1].view(torch.int16)))
                                self.assertTrue(torch.equal(ys[0].view(torch.int16),ys[1].view(torch.int16)))
                                self.assertTrue(bool(torch.isfinite(ys[1]).all()))
                                self.assertTrue(torch.equal(ids,ids_before))
                                self.assertTrue(torch.equal(rw.view(torch.int32),rw_before.view(torch.int32)))
                                for p in owners:self.assertTrue(bool((p[:128]==17).all() and (p[-128:]==17).all()))
                                for p in ys_owner:self.assertTrue(bool((p[:128]==19).all() and (p[-128:]==19).all()))
                        self.assertTrue(torch.equal(weight,weight_before))
                        del graphs,graph,owners,hs,ys_owner,ys
                self.assertTrue(torch.equal(gu,gu_before))
                self.assertTrue(torch.equal(down,down_before))

    @torch.no_grad()
    def test_direct_ids_reject_multitoken_and_legacy_rejects_int32(self):
        from deep_gemm import _C
        x=torch.zeros((2,2048),device='cuda',dtype=torch.bfloat16)
        w=torch.zeros((8,2816,2048),device='cuda',dtype=torch.bfloat16)
        h=torch.empty((8,128,1408),device='cuda',dtype=torch.bfloat16)
        ids=torch.zeros((2,4),device='cuda',dtype=torch.int32)
        rw=torch.ones((2,4),device='cuda')
        with self.assertRaises(RuntimeError):
            _C._sm120_moe_small(x,w,ids,rw,h,True,16,3,4,expert_ids=True)
        with self.assertRaises(RuntimeError):
            _C._sm120_moe_small(x,w,ids,rw,h,True,16,3,4)


if __name__=='__main__':unittest.main()
