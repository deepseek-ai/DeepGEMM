"""Regression for saved BF16 rounding edges, independent of activation error."""
from fractions import Fraction
from pathlib import Path
import unittest

import torch


HAS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@unittest.skipUnless(HAS_SM120, 'SM120 GPU required')
class Bf16AccumulationTest(unittest.TestCase):
    def test_additional_random_fp64_seeds(self):
        import deep_gemm
        from tests.test_sm120_grouped_swiglu import test_random_selected_experts_fp64_oracle

        previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        try:
            for seed in range(2, 32):
                with self.subTest(seed=seed):
                    test_random_selected_experts_fp64_oracle(seed)
        finally:
            deep_gemm.set_mk_alignment_for_contiguous_layout(previous)

    def test_saved_midpoint_projections_and_fp32_output(self):
        import deep_gemm

        fixture = Path(__file__).parent / 'fixtures/sm120_moe/midpoint-inputs.pt'
        cases = torch.load(fixture, map_location='cpu', weights_only=True)
        for case in cases:
            x, w = case['x'], case['weights']
            exact = [sum((Fraction(float(a)) * Fraction(float(b))
                          for a, b in zip(x, row)), Fraction()) for row in w]
            expected = torch.tensor([float(v) for v in exact], dtype=torch.float64).bfloat16()
            k = x.numel()
            a = torch.zeros((1, 128, k), device='cuda', dtype=torch.bfloat16)
            b = torch.zeros((1, 128, k), device='cuda', dtype=torch.bfloat16)
            a[0, 0].copy_(x)
            b[0, 0].copy_(w[0])
            b[0, 64].copy_(w[1])
            counts = torch.ones(1, device='cuda', dtype=torch.int32)
            for hint in (1, 33, 65):
                with self.subTest(identity=case['identity'], hint=hint):
                    out = torch.full((1, 128, 128), float('nan'), device='cuda', dtype=torch.bfloat16)
                    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, out, counts, hint)
                    self.assertTrue(torch.equal(out[0, 0, [0, 64]].cpu(), expected))
            out32 = torch.full((128, 128), float('nan'), device='cuda', dtype=torch.float32)
            deep_gemm.bf16_gemm_nt(a[0], b[0], out32)
            self.assertTrue(torch.equal(out32[0, [0, 64]].bfloat16().cpu(), expected))
            # Graph replay must consume current operands with the same rounding.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                deep_gemm.bf16_gemm_nt(a[0], b[0], out32)
            a.neg_()
            graph.replay()
            self.assertTrue(torch.equal(out32[0, [0, 64]].bfloat16().cpu(), -expected))

    def test_saved_double_rounding_projection(self):
        import deep_gemm

        fixture = Path(__file__).parent / 'fixtures/sm120_moe/midpoint-seed2-inputs.pt'
        case = torch.load(fixture, map_location='cpu', weights_only=True)
        x, w = case['x'], case['w']
        exact = [sum((Fraction(float(a)) * Fraction(float(b)) for a, b in zip(x, row)),
                     Fraction()) for row in w]
        expected = torch.tensor([float(v) for v in exact], dtype=torch.float64).bfloat16()
        a = torch.zeros((1,128,x.numel()), device='cuda', dtype=torch.bfloat16)
        b = torch.zeros((1,128,x.numel()), device='cuda', dtype=torch.bfloat16)
        a[0,0].copy_(x)
        b[0,0].copy_(w[0])
        b[0,64].copy_(w[1])
        counts = torch.ones(1,device='cuda',dtype=torch.int32)
        for hint in (1,33,65):
            out = torch.full((1,128,128),float('nan'),device='cuda',dtype=torch.bfloat16)
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a,b,out,counts,hint)
            self.assertTrue(torch.equal(out[0,0,[0,64]].cpu(),expected))
        fused = torch.full((1,128,64),float('nan'),device='cuda',dtype=torch.bfloat16)
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a,b,fused,counts,1)
        reference = (torch.nn.functional.silu(expected[0].float()) * expected[1].float()).bfloat16()
        torch.testing.assert_close(fused[0,0,0].cpu(),reference,rtol=0,atol=0)
        self.assertTrue(bool(torch.isnan(fused[:,1:]).all()))

    def test_odd_shape_graph_on_nondefault_stream(self):
        import deep_gemm

        previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        try:
            e, m, k, n = 4, 65, 136, 65
            a = (((torch.arange(e*m*k, device='cuda').reshape(e,m,k) % 17)-8).float()/16).bfloat16()
            b = (((torch.arange(e*2*n*k, device='cuda').reshape(e,2*n,k) % 13)-6).float()/64).bfloat16()
            counts = torch.tensor([0,1,31,65], device='cuda', dtype=torch.int32)
            storage = torch.full((17+e*m*n+16,), float('nan'), device='cuda', dtype=torch.bfloat16)
            out = storage[17:-16].view(e,m,n)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                invoke = lambda: deep_gemm.m_grouped_bf16_swiglu_nt_masked(a,b,out,counts,65)
                invoke()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    invoke()
                for rows in ((0,1,31,65),(65,0,7,1),(0,0,0,0)):
                    a.neg_()
                    counts.copy_(torch.tensor(rows, device='cuda', dtype=torch.int32))
                    out.fill_(float('nan'))
                    graph.replay()
                    projections = (a.double() @ b.double().transpose(-1,-2)).bfloat16().float()
                    reference = (torch.nn.functional.silu(projections[...,:n])*projections[...,n:]).bfloat16()
                    for group, count in enumerate(rows):
                        torch.testing.assert_close(out[group,:count],reference[group,:count],rtol=1e-2,atol=2e-3)
                        self.assertTrue(bool(torch.isnan(out[group,count:]).all()))
                    self.assertTrue(bool(torch.isnan(storage[:17]).all()))
                    self.assertTrue(bool(torch.isnan(storage[-16:]).all()))
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        finally:
            deep_gemm.set_mk_alignment_for_contiguous_layout(previous)


if __name__ == '__main__':
    unittest.main()
