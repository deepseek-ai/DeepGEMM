"""Final entry-point tests; no research runtimes or generated-source patches."""
import gc
from pathlib import Path
import unittest

import torch

HAS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def make_routes(t, e, k, capacity, seed, distribution='uniform'):
    generator = torch.Generator().manual_seed(seed)
    counts = torch.zeros(e, dtype=torch.int32)
    slots = torch.empty((t,k), dtype=torch.int64)
    for token in range(t):
        if distribution == 'edge':
            ids = torch.arange(e-1,e-k-1,-1)
        elif distribution == 'concentrated':
            ids = torch.randperm(k, generator=generator)
        elif distribution == 'skew75' and k >= 4:
            ids = torch.cat((torch.randperm(e//2, generator=generator)[:3*k//4],
                             e//2 + torch.randperm(e-e//2, generator=generator)[:k-3*k//4]))
        else:
            ids = torch.randperm(e, generator=generator)[:k]
        for r, expert in enumerate(ids.tolist()):
            slots[token,r] = expert*capacity + counts[expert]
            counts[expert] += 1
    weights = torch.rand((t,k), generator=generator) + .05
    weights /= weights.sum(dim=1, keepdim=True)
    return tuple(x.cuda() for x in (counts,slots,weights))


@unittest.skipUnless(HAS_SM120, 'SM120 GPU required')
class SmallMoEGpuTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def make(self, t=1, k=2, d=2048, i=768, e=64, capacity=None):
        from deep_gemm.small_moe import SmallMoE
        torch.manual_seed(211)
        gate_up = torch.randn((e,2*i,d),device='cuda',dtype=torch.bfloat16) / d**.5
        down = torch.randn((e,d,i),device='cuda',dtype=torch.bfloat16) / i**.5
        return SmallMoE(t,gate_up,down,k,capacity=capacity)

    @staticmethod
    def poison_ordinary_outputs(rt):
        # On the tested CUDA 13.2 toolchain initcheck does not mark native
        # TMA-store output initialized. Poison, then require finite complete
        # results: this checks write coverage without trusting that tracking.
        rt.projection.fill_(float('nan'))
        rt.expert.fill_(float('nan'))

    def check_scene(self, t, k, d, i, *, distribution='uniform', zero_weights=False, e=64, kernel_matrix=False):
        from deep_gemm.small_moe import SmallMoE
        rt = self.make(t,k,d,i,e=e)
        if kernel_matrix:
            from deep_gemm.small_moe_policy import select_config
            rt.config=select_config(t,64,k,d,i)
        ref = SmallMoE(t,rt.gate_up,rt.down,k,enable_fusion=False)
        # Guard both formal kernel output allocations, not just the final Y.
        owners = []
        for name in ('h','output'):
            old = getattr(rt,name)
            owner = torch.full((old.numel()+32,),17.0,device='cuda',dtype=old.dtype)
            setattr(rt,name,owner[16:-16].view(old.shape))
            owners.append(owner)
        x = torch.randn((t,d),device='cuda',dtype=torch.bfloat16)
        graph = None
        for replay in range(2):
            counts,slots,weights = make_routes(t,rt.e,k,rt.capacity,223+replay,distribution)
            if zero_weights: weights.zero_()
            rt.validate_routes(counts,slots,weights)
            if replay == 0:
                stable = (counts.clone(),slots.clone(),weights.clone())
            else:
                for a,b in zip(stable,(counts,slots,weights)): a.copy_(b)
                x.neg_().add_(.125)
                rt.gate_up.neg_()
                rt.down.mul_(.75)
            counts,slots,weights = stable
            self.poison_ordinary_outputs(ref)
            expected = ref.forward(x,counts,slots,weights).clone()
            valid = torch.arange(rt.capacity,device='cuda')[None,:] < counts[:,None]
            expected_h = ref.h[valid].clone()
            self.assertTrue(bool(torch.isfinite(expected).all()))
            self.assertTrue(bool(torch.isfinite(expected_h).all()))
            self.poison_ordinary_outputs(rt)
            rt.h.fill_(float('nan'))
            actual = rt.forward(x,counts,slots,weights)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
            self.assertTrue(torch.equal(rt.h[valid].view(torch.int16),expected_h.view(torch.int16)))
            if rt.config.l1 is not None:
                self.assertTrue(bool(torch.isnan(rt.h[~valid]).all()))
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): rt.forward(x,counts,slots,weights)
            rt.output.fill_(float('nan'))
            self.poison_ordinary_outputs(rt)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(rt.output.view(torch.int16),expected.view(torch.int16)))
            for owner in owners:
                self.assertTrue(bool((owner[:16]==17).all() and (owner[-16:]==17).all()))

    def test_fused_direct_graph_and_changed_routing(self):
        for t,k,d,i,dist in ((1,1,2048,768,'uniform'),(1,8,2048,768,'skew75'),
                             (2,4,2048,768,'concentrated'),(4,2,2048,768,'uniform'),
                             (1,4,4096,1536,'uniform'),(2,2,4096,1536,'concentrated')):
            with self.subTest(t=t,k=k,d=d,distribution=dist):
                self.check_scene(t,k,d,i,distribution=dist)

    def test_odd_topk_and_token_fallback(self):
        self.check_scene(3,3,2048,768)

    def test_boundary_fallback(self):
        self.check_scene(4,2,4096,1536)

    def test_zero_route_weights(self):
        self.check_scene(1,2,2048,768,zero_weights=True)

    def test_sparse_high_expert_slots(self):
        self.check_scene(2,2,2048,768,distribution='edge')

    def test_kernel_template_matrix(self):
        from deep_gemm.small_moe_policy import select_config
        # E does not change template code; a compact E=8 fixture exercises
        # every supported tile/barrier configuration under all sanitizers.
        for d,i in ((2048,768),(4096,1536)):
            for t in (1,2,4):
                for k in (1,2,4,8):
                    if select_config(t,64,k,d,i).l1 is not None:
                        with self.subTest(t=t,k=k,d=d):
                            self.check_scene(t,k,d,i,e=8,kernel_matrix=True,
                                             distribution='concentrated')

    def test_controlled_dyadic_oracle(self):
        from deep_gemm.small_moe_policy import select_config
        rt=self.make(t=2,k=2,e=8)
        rt.config=select_config(2,64,2,rt.d,rt.i)
        rt.gate_up.zero_()
        rt.down.zero_()
        rt.gate_up[:,:rt.i,0]=.5
        rt.gate_up[:,rt.i:,1]=(torch.arange(rt.i,device='cuda')%8+1)/8
        rt.down[:,:,:4]=(torch.arange(rt.d,device='cuda')[:,None]%8+1)/16
        x=torch.zeros((2,rt.d),device='cuda',dtype=torch.bfloat16)
        x[:,0]=torch.tensor([.5,1.],device='cuda')
        x[:,1]=torch.tensor([.75,1.],device='cuda')
        counts,slots,weights=make_routes(2,8,2,rt.capacity,233,'concentrated')
        weights[:]=torch.tensor([.25,.75],device='cuda')
        rt.validate_routes(counts,slots,weights)
        actual=rt.forward(x,counts,slots,weights).cpu()
        expected=torch.zeros((2,rt.d),dtype=torch.float64)
        for token in range(2):
            for r in range(2):
                slot=int(slots[token,r]); expert=slot//rt.capacity
                projection=rt.gate_up[expert].cpu().double() @ x[token].cpu().double()
                gate,up=projection.bfloat16().float().chunk(2)
                h=(gate*torch.sigmoid(gate)*up).bfloat16()
                self.assertTrue(torch.equal(rt.h[expert,slot%rt.capacity].cpu().view(torch.int16),h.view(torch.int16)))
                partial=(rt.down[expert].cpu().double() @ h.double()).bfloat16().double()
                expected[token]+=partial*float(weights[token,r])
        self.assertTrue(torch.equal(actual.view(torch.int16),expected.bfloat16().view(torch.int16)))

    def test_saved_midpoints_in_both_fused_layers(self):
        from deep_gemm import _C
        from deep_gemm.small_moe_policy import select_config

        fixtures=Path(__file__).parent/'fixtures/sm120_moe'
        cases=torch.load(fixtures/'midpoint-inputs.pt',map_location='cpu',weights_only=True)
        seed2=torch.load(fixtures/'midpoint-seed2-inputs.pt',map_location='cpu',weights_only=True)
        cases=list(cases)+[dict(x=seed2['x'],weights=seed2['w'])]
        rt=self.make(t=1,k=2,d=4096,i=1536,e=8)
        cfg=select_config(1,64,2,rt.d,rt.i)
        counts,slots,weights=make_routes(1,8,2,rt.capacity,233,'edge')
        weights[:]=torch.tensor([.25,.75],device='cuda')
        for case in cases:
            x,w=case['x'],case['weights']
            dots=(w.double() @ x.double()).float().bfloat16()
            rt.gate_up.zero_(); rt.down.zero_()
            input=torch.zeros((1,rt.d),device='cuda',dtype=torch.bfloat16)
            input[0,:x.numel()].copy_(x)
            for slot in slots[0].tolist():
                expert=slot//rt.capacity
                rt.gate_up[expert,:2,:x.numel()].copy_(w)
                rt.gate_up[expert,rt.i:rt.i+2,:x.numel()].copy_(w)
            _C._sm120_moe_small(input,rt.gate_up,slots,weights,rt.h,True,*cfg.l1)
            expected=(torch.nn.functional.silu(dots.float())*dots.float()).bfloat16()
            for slot in slots[0].tolist():
                expert,row=divmod(slot,rt.capacity)
                torch.testing.assert_close(rt.h[expert,row,:2].cpu(),expected,rtol=0,atol=0)
        # L2 K=1536 cannot reuse saved K=4096 L1 operands. These exact
        # short dots straddle a BF16 midpoint, with cancellation inside K16.
        x=torch.ones(5,dtype=torch.bfloat16)
        w=torch.tensor([[256.,1.,1/256,2**-20,-256.],
                        [256.,1.,1/256,-2**-20,-256.]],dtype=torch.bfloat16)
        dots=(w.double() @ x.double()).float().bfloat16()
        rt.h.zero_(); rt.down.zero_()
        for slot in slots[0].tolist():
            expert,row=divmod(slot,rt.capacity)
            rt.h[expert,row,:x.numel()].copy_(x)
            rt.down[expert,:2,:x.numel()].copy_(w)
        _C._sm120_moe_small(rt.h,rt.down,slots,weights,rt.output,False,*cfg.l2)
        torch.testing.assert_close(rt.output[0,:2].cpu(),dots,rtol=0,atol=0)

    def test_other_shape_and_custom_capacity_on_nondefault_stream(self):
        from deep_gemm.small_moe import SmallMoE
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for d,i in ((2048,768),(3072,1024)):
                rt = self.make(t=2,k=2,d=d,i=i,e=16 if d==3072 else 64,capacity=256)
                ref = SmallMoE(2,rt.gate_up,rt.down,2,capacity=256,enable_fusion=False)
                counts,slots,weights = make_routes(2,rt.e,2,256,229,'concentrated')
                rt.validate_routes(counts,slots,weights)
                x = torch.randn((2,d),device='cuda',dtype=torch.bfloat16)
                self.poison_ordinary_outputs(rt)
                actual = rt.forward(x,counts,slots,weights).clone()
                self.poison_ordinary_outputs(ref)
                expected = ref.forward(x,counts,slots,weights)
                stream.synchronize()
                self.assertTrue(bool(torch.isfinite(expected).all()))
                self.assertTrue(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
        torch.cuda.current_stream().wait_stream(stream)

    def test_bad_routes_rejected_before_launch(self):
        from deep_gemm.small_moe import SmallMoE
        for e,d,i in ((0,2048,768),(8,0,768),(8,2048,0)):
            with self.assertRaises(ValueError):
                SmallMoE(1,torch.empty((e,2*i,d),device='cuda',dtype=torch.bfloat16),
                         torch.empty((e,d,i),device='cuda',dtype=torch.bfloat16),1)
        rt = self.make()
        counts,slots,weights = make_routes(rt.t,rt.e,rt.k,rt.capacity,227)
        rt.validate_routes(counts,slots,weights)
        for bad in (slots.clone().fill_(-1),slots.clone().fill_(0),slots.clone().fill_(rt.e*rt.capacity)):
            with self.assertRaises(ValueError): rt.validate_routes(counts,bad,weights)
        with self.assertRaises(ValueError): rt.validate_routes(counts+1,slots,weights)
        with self.assertRaises(ValueError): rt.validate_routes(counts,slots,weights*float('nan'))
        with self.assertRaises(ValueError): rt.forward(torch.empty((1,2048),device='cuda'),counts,slots,weights)
        with self.assertRaises(ValueError): rt.forward(torch.empty((1,2048),device='cuda',dtype=torch.bfloat16),counts,slots.int(),weights)
        with self.assertRaises(ValueError): rt.forward(rt.output,counts,slots,weights)
        from deep_gemm import _C
        x = torch.zeros((1,2048),device='cuda',dtype=torch.bfloat16)
        with self.assertRaises(Exception):
            _C._sm120_moe_small(x,rt.gate_up,slots,weights,rt.h,True,8,3,3)
        with self.assertRaises(Exception):
            _C._sm120_moe_small(x,rt.gate_up,slots,weights,rt.gate_up,True,8,3,2)


if __name__ == '__main__':
    unittest.main()
