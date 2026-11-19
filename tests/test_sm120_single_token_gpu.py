"""Public entry, candidate geometry, and midpoint tests on repaired arithmetic."""
import gc
import unittest

import torch

from tests.test_sm120_small_moe_gpu import HAS_SM120, SmallMoEGpuTest, make_routes


@unittest.skipUnless(HAS_SM120, 'SM120 GPU required')
class SingleTokenGpuTest(unittest.TestCase):
    check_scene = SmallMoEGpuTest.check_scene
    poison_ordinary_outputs = staticmethod(SmallMoEGpuTest.poison_ordinary_outputs)

    def tearDown(self):
        gc.collect(); torch.cuda.empty_cache()

    def make(self, t, k, d, i, e=8, capacity=None):
        from deep_gemm.small_moe import SmallMoE
        from deep_gemm.small_moe_policy import SmallMoEConfig
        torch.manual_seed(25043)
        gu = torch.randn((e, 2*i, d), device='cuda', dtype=torch.bfloat16).div_(d**.5)
        down = torch.randn((e, d, i), device='cuda', dtype=torch.bfloat16).div_(i**.5)
        rt = SmallMoE(t, gu, down, k, capacity=capacity, fusion_policy='single_token')
        if getattr(self, 'candidate', None):
            rt.config = SmallMoEConfig(*self.candidate, 'compact template correctness only')
        return rt

    def test_candidate_templates_graph_guards_and_high_slots(self):
        from deep_gemm.small_moe_policy import SINGLE_TOKEN_CANDIDATES
        try:
            for (d, i, _, k), cfg in SINGLE_TOKEN_CANDIDATES.items():
                self.candidate = cfg
                with self.subTest(d=d, i=i, k=k):
                    self.check_scene(1, k, d, i, e=8, distribution='edge')
        finally:
            self.candidate = None

    def test_candidates_preserve_midpoint_and_cancellation(self):
        from deep_gemm import _C
        from deep_gemm.small_moe_policy import SINGLE_TOKEN_CANDIDATES
        w = torch.tensor([[256., 1., 1/256, 2**-20, -256.],
                          [256., 1., 1/256, -2**-20, -256.]], dtype=torch.bfloat16)
        dots = (w.double() @ torch.ones(5, dtype=torch.float64)).float().bfloat16()
        for (d, i, _, k), cfg in SINGLE_TOKEN_CANDIDATES.items():
            with self.subTest(d=d, i=i, k=k):
                rt = self.make(1, k, d, i)
                rt.gate_up.zero_(); rt.down.zero_()
                x = torch.zeros((1, d), device='cuda', dtype=torch.bfloat16); x[0, :5] = 1
                counts, slots, weights = make_routes(1, 8, k, rt.capacity, 25043, 'edge')
                weights.fill_(1/k)
                for slot in slots[0].tolist():
                    expert = slot//rt.capacity
                    rt.gate_up[expert, :2, :5].copy_(w)
                    rt.gate_up[expert, i:i+2, :5].copy_(w)
                _C._sm120_moe_small(x, rt.gate_up, slots, weights, rt.h, True, *cfg[0])
                h = (torch.nn.functional.silu(dots.float())*dots.float()).bfloat16()
                for slot in slots[0].tolist():
                    expert, row = divmod(slot, rt.capacity)
                    torch.testing.assert_close(rt.h[expert, row, :2].cpu(), h, rtol=0, atol=0)
                rt.h.zero_()
                for slot in slots[0].tolist():
                    expert, row = divmod(slot, rt.capacity)
                    rt.h[expert, row, :5] = 1
                    rt.down[expert, :2, :5].copy_(w)
                _C._sm120_moe_small(rt.h, rt.down, slots, weights, rt.output, False, *cfg[1])
                total = torch.zeros(2, dtype=torch.float32)
                for _ in range(k): total = total + dots.float()*float(1/k)
                torch.testing.assert_close(rt.output[0, :2].cpu(), total.bfloat16(), rtol=0, atol=0)
                del rt

    def test_public_entry_supported_and_multitoken_fallback(self):
        from deep_gemm.small_moe_policy import SINGLE_TOKEN_ACCEPTED
        for d, i, e, k in sorted(SINGLE_TOKEN_ACCEPTED):
            with self.subTest(d=d, i=i, e=e, k=k):
                self.check_scene(1, k, d, i, e=e, distribution='edge')
        self.check_scene(2, 8, 2048, 768, e=64, distribution='concentrated')

    def test_configuration_and_force_split_contract(self):
        from deep_gemm.small_moe import SmallMoE
        from deep_gemm.small_moe_policy import SINGLE_TOKEN_ACCEPTED
        rt = self.make(1, 8, 2048, 768, e=128)
        self.assertEqual(rt.config.l1 is not None, (2048, 768, 128, 8) in SINGLE_TOKEN_ACCEPTED)
        forced = SmallMoE(1, rt.gate_up, rt.down, 8, fusion_policy='single_token', enable_fusion=False)
        self.assertIsNone(forced.config)
        with self.assertRaises(ValueError):
            SmallMoE(1, rt.gate_up, rt.down, 8, fusion_policy='unknown')


if __name__ == '__main__': unittest.main()
