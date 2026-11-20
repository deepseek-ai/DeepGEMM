"""CPU-only support-boundary tests, independent of Torch/CUDA availability."""
import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('small_moe_policy_cpu',
    Path(__file__).resolve().parents[1] / 'deep_gemm/small_moe_policy.py')
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


class SmallMoEPolicyTest(unittest.TestCase):
    def test_supported_tiles_have_valid_route_ownership(self):
        promoted = 0
        for d, i in ((2048, 768), (4096, 1536)):
            for t in (1, 2, 4):
                for k in (1, 2, 4, 8):
                    for e in (64, 128):
                        cfg = policy.select_config(t, e, k, d, i)
                        if cfg.l1 is None:
                            self.assertIsNone(cfg.l2)
                            continue
                        promoted += 1
                        self.assertEqual(k % cfg.l1[2], 0)
                        self.assertEqual(cfg.l2[2], k)
                        self.assertEqual(i % cfg.l1[0], 0)
                        self.assertEqual(d % cfg.l2[0], 0)
        self.assertEqual(promoted, 30)

    def test_unmeasured_shapes_fall_back(self):
        for args in ((3,128,3,4096,1536), (5,128,8,2048,768),
                     (1,32,8,2048,768), (1,128,8,3072,1024),
                     (4,128,8,2048,768), (4,128,2,4096,1536),
                     (2,64,8,2048,768), (4,64,4,2048,768),
                     (4,64,1,4096,1536),
                     (2,128,4,4096,1536), (4,128,4,4096,1536)):
            with self.subTest(args=args):
                self.assertIsNone(policy.select_config(*args).l1)
                self.assertIsNone(policy.select_config(*args).l2)

    def test_policy_is_immutable(self):
        cfg = policy.select_config(1,128,8,2048,768)
        with self.assertRaises(AttributeError):
            cfg.reason = 'changed'


if __name__ == '__main__':
    unittest.main()
