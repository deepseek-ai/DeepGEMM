"""The new opt-in policy cannot silently enable unvalidated request shapes."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('single_token_policy_cpu', ROOT/'deep_gemm/small_moe_policy.py')
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


class SingleTokenPolicyTest(unittest.TestCase):
    def test_frozen_support_set_and_rejected_pressure_boundary(self):
        self.assertEqual(policy.SINGLE_TOKEN_ACCEPTED, {
            (2048, 768, 128, 8), (4096, 1536, 128, 8),
            (2048, 1408, 60, 4), (2048, 1408, 64, 6),
        })
        self.assertIsNone(policy.select_single_token_config(1, 64, 8, 3584, 2560).l1)

    def test_only_evidence_keys_can_enable_fusion(self):
        self.assertLessEqual(policy.SINGLE_TOKEN_ACCEPTED, policy.SINGLE_TOKEN_CANDIDATES.keys())
        for key in policy.SINGLE_TOKEN_CANDIDATES:
            d, i, e, k = key
            cfg = policy.select_single_token_config(1, e, k, d, i)
            self.assertEqual(cfg.l1 is not None, key in policy.SINGLE_TOKEN_ACCEPTED)
            for t in (2, 3, 4, 8, 16, 32):
                with self.subTest(key=key, tokens=t):
                    fallback = policy.select_single_token_config(t, e, k, d, i)
                    self.assertIsNone(fallback.l1)
                    self.assertIsNone(fallback.l2)

    def test_nearby_unknown_dimensions_and_routes_fall_back(self):
        for args in ((1, 128, 8, 2048, 896), (1, 127, 8, 2048, 768),
                     (1, 128, 7, 2048, 768), (1, 8, 2, 4096, 14336),
                     (0, 128, 8, 2048, 768)):
            with self.subTest(args=args):
                self.assertIsNone(policy.select_single_token_config(*args).l1)


if __name__ == '__main__': unittest.main()
