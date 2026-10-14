#!/usr/bin/env python3
"""
Comprehensive hardware validation test suite for SM80 (Ampere) and SM89 (Ada Lovelace) GPU support in DeepGEMM.
This script can be executed directly on physical GPU hardware after cloning/building DeepGEMM.
"""

import unittest
import torch

try:
    import deep_gemm
    import deep_gemm._C as _C
    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False


class TestSM80SM89Support(unittest.TestCase):

    def setUp(self):
        if not HAS_DEEPGEMM:
            self.skipTest("DeepGEMM C++ extension not compiled/installed in environment.")
        if not torch.cuda.is_available():
            self.skipTest("CUDA device required for hardware testing.")

    def test_cuda_device_capability(self):
        major, minor = torch.cuda.get_device_capability()
        print(f"\n[GPU Hardware Info] Active GPU: {torch.cuda.get_device_name(0)}")
        print(f"[GPU Hardware Info] Compute Capability: SM{major}.{minor}")
        self.assertGreaterEqual(major, 8, "Expected GPU compute capability >= 8.0")

    def test_bf16_gemm_nt(self):
        """Test BF16 GEMM NT (Non-transposed A, Transposed B) layout across various shapes."""
        shapes = [(64, 128, 256), (128, 256, 512), (256, 512, 1024), (512, 1024, 2048)]
        for m, n, k in shapes:
            with self.subTest(shape=(m, n, k)):
                a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
                d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

                _C.bf16_gemm_nt(a, b, d, None, "nk", None, None)
                ref = torch.matmul(a, b.t())

                diff = torch.max(torch.abs(d - ref)).item()
                print(f"[BF16 GEMM NT] Shape ({m}, {n}, {k}) - Max Abs Diff: {diff:.6f}")
                self.assertLess(diff, 1e-1)

    def test_bf16_gemm_layouts(self):
        """Test BF16 GEMM with NN, TN, and TT layout combinations."""
        m, n, k = 128, 256, 512
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
        d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

        # Test NN layout (A @ B)
        _C.bf16_gemm_nn(a, b, d, None, "nk", None, None)
        ref_nn = torch.matmul(a, b)
        diff_nn = torch.max(torch.abs(d - ref_nn)).item()
        print(f"[BF16 GEMM NN] Max Abs Diff: {diff_nn:.6f}")
        self.assertLess(diff_nn, 1e-1)

        # Test TN layout (A.T @ B.T)
        a_t = a.t().contiguous()
        b_t = b.t().contiguous()
        _C.bf16_gemm_tn(a_t, b_t, d, None, "mn", None, None)
        ref_tn = torch.matmul(a, b)
        diff_tn = torch.max(torch.abs(d - ref_tn)).item()
        print(f"[BF16 GEMM TN] Max Abs Diff: {diff_tn:.6f}")
        self.assertLess(diff_tn, 1e-1)

        # Test TT layout (A.T @ B)
        _C.bf16_gemm_tt(a_t, b, d, None, "mn", None, None)
        ref_tt = torch.matmul(a, b)
        diff_tt = torch.max(torch.abs(d - ref_tt)).item()
        print(f"[BF16 GEMM TT] Max Abs Diff: {diff_tt:.6f}")
        self.assertLess(diff_tt, 1e-1)

    def test_bf16_gemm_accumulation(self):
        """Test BF16 GEMM with matrix C accumulation (D = A @ B.T + C)."""
        m, n, k = 128, 256, 512
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        c = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
        d = c.clone()

        _C.bf16_gemm_nt(a, b, d, c, "nk", None, None)
        ref = torch.matmul(a, b.t()) + c

        diff = torch.max(torch.abs(d - ref)).item()
        print(f"[BF16 GEMM Accumulation] Max Abs Diff: {diff:.6f}")
        self.assertLess(diff, 1e-1)

    def test_sm80_sm89_fp8_hardware_error(self):
        """Verify that SM80/SM89 GPUs raise explicit hardware error panics for unsupported FP8 operations."""
        major, minor = torch.cuda.get_device_capability()
        if major == 8:
            m, n, k = 128, 256, 512
            a_fp = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
            a_sf = torch.ones((m, k // 128), dtype=torch.float32, device="cuda")
            b_fp = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
            b_sf = torch.ones((n, k // 128), dtype=torch.float32, device="cuda")
            d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

            with self.assertRaises(RuntimeError) as ctx:
                _C.fp8_fp4_gemm_nt((a_fp, a_sf), (b_fp, b_sf), d, None, None, None, None, "nk", False, None, None)
            self.assertIn("not natively supported on NVIDIA Ampere/Ada (SM80/SM89)", str(ctx.exception))
            print(f"[Hardware Panic Check] Successfully verified SM80/SM89 FP8 error: {ctx.exception}")

    def test_sm80_sm89_fp4_hardware_error(self):
        """Verify that pre-SM100 GPUs raise explicit error panics for Blackwell-only FP4 operations."""
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            m, n, k = 128, 256, 512
            a_fp = torch.randint(0, 255, (m, k // 2), dtype=torch.uint8, device="cuda")
            a_sf = torch.ones((m, k // 128), dtype=torch.int32, device="cuda")
            b_fp = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device="cuda")
            b_sf = torch.ones((n, k // 128), dtype=torch.int32, device="cuda")
            d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

            with self.assertRaises(RuntimeError) as ctx:
                _C.fp8_fp4_gemm_nt((a_fp, a_sf), (b_fp, b_sf), d, None, None, None, None, "nk", False, None, None)
            self.assertIn("supported on NVIDIA Blackwell (SM100)", str(ctx.exception))
            print(f"[Hardware Panic Check] Successfully verified FP4 error: {ctx.exception}")


if __name__ == "__main__":
    unittest.main()
