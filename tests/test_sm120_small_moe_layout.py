"""Check native [gate;up] and framework [up;gate] share identical math."""
import unittest
from pathlib import Path

import torch


HAS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@unittest.skipUnless(HAS_SM120, "SM120 GPU required")
class SmallMoELayoutTest(unittest.TestCase):
    @torch.no_grad()
    def test_swapped_saved_rounding_edges(self):
        from deep_gemm import _C

        fixtures = Path(__file__).parent / 'fixtures/sm120_moe'
        cases = torch.load(fixtures / 'midpoint-inputs.pt', map_location='cpu', weights_only=True)
        extra = torch.load(fixtures / 'midpoint-seed2-inputs.pt', map_location='cpu', weights_only=True)
        cases.append({'x': extra['x'], 'weights': extra['w']})
        for case in cases:
            original_x, w = case['x'], case['weights']
            d, i = (4096, 1536) if original_x.numel() > 2048 else (2048, 768)
            x = torch.zeros((1, d), device='cuda', dtype=torch.bfloat16)
            x[0, :original_x.numel()].copy_(original_x)
            gu = torch.zeros((1, 2*i, d), device='cuda', dtype=torch.bfloat16)
            gu[0, 0, :original_x.numel()].copy_(w[0])
            gu[0, i, :original_x.numel()].copy_(w[1])
            swapped = torch.cat((gu[:, i:], gu[:, :i]), dim=1)
            slots = torch.zeros((1, 1), device='cuda', dtype=torch.int64)
            weights = torch.ones((1, 1), device='cuda', dtype=torch.float32)
            canonical_h = torch.empty((1, 128, i), device='cuda', dtype=torch.bfloat16)
            swapped_h = torch.empty_like(canonical_h)
            for _ in range(2):
                _C._sm120_moe_small(x, gu, slots, weights, canonical_h, True, 16, 3, 1)
                _C._sm120_moe_small(x, swapped, slots, weights, swapped_h, True, 16, 3, 1,
                                   gate_up_swapped=True)
                self.assertTrue(torch.equal(canonical_h[:, 0].view(torch.int16),
                                            swapped_h[:, 0].view(torch.int16)))
                projection = (w.double() @ x[0, :original_x.numel()].cpu().double()).bfloat16().float()
                expected = (torch.nn.functional.silu(projection[0]) * projection[1]).bfloat16()
                torch.testing.assert_close(swapped_h[0, 0, 0].cpu(), expected, rtol=.01, atol=.002)
                x.neg_()

    @torch.no_grad()
    def test_swapped_weights_changed_routes_and_graph(self):
        from deep_gemm import _C
        from deep_gemm.small_moe import SmallMoE
        from deep_gemm.small_moe_policy import select_single_token_config
        from tests.test_sm120_small_moe_gpu import make_routes

        for d, i, policy_e, k in ((2048, 768, 128, 8), (4096, 1536, 128, 8),
                                   (2048, 1408, 60, 4), (2048, 1408, 64, 6)):
            with self.subTest(d=d, i=i, k=k):
                torch.manual_seed(28119 + k)
                gu = torch.randn((8, 2*i, d), device="cuda", dtype=torch.bfloat16) / d**.5
                down = torch.randn((8, d, i), device="cuda", dtype=torch.bfloat16) / i**.5
                swapped = torch.cat((gu[:, i:], gu[:, :i]), dim=1)
                ref = SmallMoE(1, gu, down, k, enable_fusion=False, fallback_m_tile=32)
                x = torch.randn((1, d), device="cuda", dtype=torch.bfloat16)
                counts, slots, weights = make_routes(1, 8, k, 128, 28129)
                cfg = select_single_token_config(1, policy_e, k, d, i)
                # Sentinel guards also cover empty expert rows.
                owner = torch.full((8*128*i + 32,), 17, device="cuda", dtype=torch.bfloat16)
                h = owner[16:-16].view(8, 128, i)
                y = torch.empty_like(x)

                def forward():
                    _C._sm120_moe_small(x, swapped, slots, weights, h, True, *cfg.l1,
                                       gate_up_swapped=True)
                    _C._sm120_moe_small(h, down, slots, weights, y, False, *cfg.l2)

                forward()  # JIT outside capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward()
                for seed in (28131, 28151):
                    for a, b in zip((counts, slots, weights), make_routes(1, 8, k, 128, seed)):
                        a.copy_(b)
                    x.neg_().add_(.03125)
                    # initcheck does not track the reference's native TMA
                    # stores on the tested toolchain; poison and verify output.
                    ref.projection.fill_(float("nan"))
                    ref.expert.fill_(float("nan"))
                    expected = ref.forward(x, counts, slots, weights).clone()
                    valid = torch.arange(128, device="cuda")[None, :] < counts[:, None]
                    expected_h = ref.h[valid].clone()
                    for run in (forward, graph.replay):
                        h.fill_(float("nan"))
                        y.fill_(float("nan"))
                        run()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(h[valid].view(torch.int16), expected_h.view(torch.int16)))
                        self.assertTrue(torch.equal(y.view(torch.int16), expected.view(torch.int16)))
                        self.assertTrue(bool(torch.isnan(h[~valid]).all()))
                        self.assertTrue(bool((owner[:16] == 17).all() and (owner[-16:] == 17).all()))
                with self.assertRaises(RuntimeError):
                    _C._sm120_moe_small(h, down, slots, weights, y, False, *cfg.l2,
                                       gate_up_swapped=True)
                del graph, ref, gu, down, swapped, h, owner, y
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
