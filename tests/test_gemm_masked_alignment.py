"""SM100 masked group-stride regression; run with the built extension on SM100."""
import unittest

import torch
import deep_gemm
from generators import MajorTypeAB, grouped_cast_fp8_fp4_with_major


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TestMaskedAlignment(unittest.TestCase):
    def test_physical_group_stride(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest('requires SM100')
        previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
        self.addCleanup(deep_gemm.set_mk_alignment_for_contiguous_layout, previous)
        deep_gemm.set_mk_alignment_for_contiguous_layout(256)
        torch.manual_seed(492)
        for use_fp8 in (False, True):
            for m in (128, 256, 384, 512):
                with self.subTest(fp8=use_fp8, m=m):
                    # M=128 reproduces the original failure. M=384 checks a
                    # partial final tile after a full tile, not only M < block M.
                    rows = [1, 15 if m == 128 else m - 1, 16, 17, 0]
                    a = torch.randint(-2, 3, (5, m, 256), device='cuda').bfloat16()
                    b = torch.randint(-2, 3, (5, 128, 256), device='cuda').bfloat16()
                    reference = (a.float() @ b.float().mT).bfloat16()
                    counts = torch.tensor(rows, dtype=torch.int32, device='cuda')
                    valid = torch.arange(m, device='cuda')[None, :] < counts[:, None]
                    d = torch.full((5, m, 128), 42, dtype=torch.bfloat16, device='cuda')
                    if use_fp8:
                        a = grouped_cast_fp8_fp4_with_major(a, MajorTypeAB.KMajor, 128, False, True)
                        b = grouped_cast_fp8_fp4_with_major(b, MajorTypeAB.KMajor, 128, False, True,
                                                           use_block_cast_for_fp8=True)
                        launch = deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked
                    else:
                        launch = deep_gemm.m_grouped_bf16_gemm_nt_masked
                    if m % 256:
                        with self.assertRaisesRegex(RuntimeError, 'SM100 masked GEMM requires physical M aligned'):
                            launch(a, b, d, counts, max(1, sum(rows) // 5))
                        torch.cuda.synchronize()
                        self.assertTrue(torch.all(d == 42).item(), 'rejected input modified D')
                    else:
                        launch(a, b, d, counts, max(1, sum(rows) // 5))
                        torch.cuda.synchronize()
                        torch.testing.assert_close(d[valid], reference[valid], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
