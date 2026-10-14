#!/usr/bin/env python3
"""
SM89 (Ada Lovelace) hardware validation suite for DeepGEMM.

Ada is the first NVIDIA architecture with FP8 Tensor Cores, so SM89 is supported as a
first-class target: `fp8_gemm_*` runs a native kernel built on
`mma.sync.aligned.m16n8k32` + `ldmatrix` + `cp.async`. BF16 GEMMs are served by the
cuBLASLt fallback. Ampere (SM80) has no FP8 Tensor Cores and is not a supported target.

Run with::

    python3 -m unittest tests/test_sm89.py

The suite skips itself on non-SM89 devices, so it is safe to add to a cross-architecture
CI matrix. It covers:

* Native FP8 Tensor Core GEMMs (both SF recipes, both output dtypes, accumulation,
  ragged shapes, aliases, compiled-shape specialisation and bitwise determinism).
* BF16 GEMMs in every operand layout, plus the cuBLASLt entry points.
* FP32 scaling-factor layout kernels.
* Explicit, descriptive errors for operations SM89 cannot perform.
"""

import unittest
import torch

import deep_gemm
import deep_gemm._C as _C
from deep_gemm.testing import calc_diff, is_sm89
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8, ceil_div


# Small enough to fit in 8 GB of VRAM while still spanning >1 wave of the 24 SMs of
# the reference RTX 4060 Laptop (SM89) used for development.
SHAPES = [
    (1, 128, 128),        # decode-shaped M
    (16, 256, 256),       # single M block
    (128, 512, 512),
    (129, 320, 384),      # ragged M and N
    (333, 1000, 1024),    # ragged N, not a multiple of 128
    (129, 333, 384),      # odd N: the accumulator's column pair needs a scalar tail
    (1024, 1024, 1024),
    (2048, 512, 2048),    # multiple waves
    (4096, 128, 4096),
    # The persistent loop walks one tile after another, re-priming its software pipeline from
    # scratch each time. These shapes keep that path honest: a partial M block spread over many
    # N tiles, and a K so short that fewer stages are primed than the pipeline is deep (a stale
    # stage then leaks the previous tile's shared memory into the result).
    (1, 4096, 512),       # partial M block, 32 N tiles
    (64, 4096, 512),      # partial M block, many N tiles
    (128, 4096, 128),     # a single K block
]


def make_fp8(m, n, k, per_token_b: bool, use_ue8m0: bool = True):
    """Builds an `(a, sfa), (b, sfb)` NT pair with FP32 scales of granularity (1, 128)
    for A and either (1, 128) or (128, 128) for B."""
    a, sfa = per_token_cast_to_fp8(torch.randn((m, k), device='cuda', dtype=torch.bfloat16),
                                   use_ue8m0=use_ue8m0)
    cast = per_token_cast_to_fp8 if per_token_b else per_block_cast_to_fp8
    b, sfb = cast(torch.randn((n, k), device='cuda', dtype=torch.bfloat16), use_ue8m0=use_ue8m0)
    return (a, sfa), (b, sfb)


def reference_fp8_gemm(a, b, sfa, sfb, c, out_dtype, per_token_b):
    """FP32 reference for `D = C + sum_k (A * SFA) * (B * SFB)`.

    The 1D2D B scale has one row per 128 columns, so it is expanded back to per-token
    granularity first. Accumulation into C follows the library's contract: it happens in the
    dtype of D (an FP32 D accumulates in FP32, a BF16 D rounds first and then adds in BF16).
    """
    m, k = a.shape
    n = b.shape[0]
    if not per_token_b:
        sfb = sfb.repeat_interleave(128, dim=0)[:n]
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for i in range(ceil_div(k, 128)):
        ki, kk = i * 128, min(128, k - i * 128)
        out += (a[:, ki:ki + kk].float() * sfa[:, i:i + 1]) @ \
               (b[:, ki:ki + kk].float() * sfb[:, i:i + 1]).T
    out = out.to(out_dtype)
    if c is not None:
        out = (out.float() + c.float()).to(out_dtype) if out_dtype == torch.float32 \
            else (out + c)
    return out


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
class TestSM89Support(unittest.TestCase):

    def setUp(self):
        if not is_sm89():
            self.skipTest('SM89 (Ada Lovelace) GPU required')
        torch.manual_seed(0)

    # ------------------------------------------------------------------ FP8 GEMM

    def test_fp8_gemm_recipes(self):
        """Native FP8 GEMM for both supported SF recipes, both output dtypes and with/without C."""
        for per_token_b in (True, False):
            recipe = (1, 1, 128) if per_token_b else None
            for out_dtype in (torch.bfloat16, torch.float32):
                for accumulate in (False, True):
                    if out_dtype == torch.float32 and not accumulate:
                        continue    # an FP32 D always accumulates (DeepGEMM contract)
                    for m, n, k in SHAPES:
                        with self.subTest(recipe=recipe, out_dtype=out_dtype,
                                          accumulate=accumulate, m=m, n=n, k=k):
                            a, b = make_fp8(m, n, k, per_token_b)
                            d = torch.empty((m, n), device='cuda', dtype=out_dtype)
                            c = torch.randn((m, n), device='cuda', dtype=out_dtype) if accumulate else None
                            deep_gemm.fp8_gemm_nt(a, b, d, c=c, recipe=recipe)
                            ref = reference_fp8_gemm(a[0], b[0], a[1], b[1], c, out_dtype, per_token_b)
                            self.assertLess(calc_diff(d, ref), 1e-3)

    def test_fp8_gemm_explicit_1d2d_recipe(self):
        """The `(1, 128, 128)` recipe may also be requested explicitly."""
        m, n, k = 512, 1024, 2048
        a, b = make_fp8(m, n, k, per_token_b=False)
        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt(a, b, d, recipe=(1, 128, 128))
        ref = reference_fp8_gemm(a[0], b[0], a[1], b[1], None, torch.bfloat16, False)
        self.assertLess(calc_diff(d, ref), 1e-3)

    def test_fp8_gemm_recipe_pair(self):
        """`recipe_a` / `recipe_b` select the same kernels as the shared `recipe`."""
        m, n, k = 256, 512, 1024
        a, b = make_fp8(m, n, k, per_token_b=True)
        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt(a, b, d, recipe_a=(1, 128), recipe_b=(1, 128))
        ref = reference_fp8_gemm(a[0], b[0], a[1], b[1], None, torch.bfloat16, True)
        self.assertLess(calc_diff(d, ref), 1e-3)

    def test_fp8_gemm_aliases(self):
        """`fp4_gemm_*` aliases route to the same SM89 implementation."""
        m, n, k = 256, 512, 1024
        a, b = make_fp8(m, n, k, per_token_b=True)
        ref = reference_fp8_gemm(a[0], b[0], a[1], b[1], None, torch.bfloat16, True)
        for name in ('fp8_fp4_gemm_nt', 'fp8_gemm_nt', 'fp4_gemm_nt'):
            with self.subTest(name=name):
                d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
                getattr(deep_gemm, name)(a, b, d, recipe=(1, 1, 128))
                self.assertLess(calc_diff(d, ref), 1e-3)

    def test_fp8_gemm_compiled_dims(self):
        """Every `compiled_dims` choice produces identical results."""
        m, n, k = 256, 512, 1024
        a, b = make_fp8(m, n, k, per_token_b=True)
        ref = reference_fp8_gemm(a[0], b[0], a[1], b[1], None, torch.bfloat16, True)
        outputs = []
        for compiled_dims in ('nk', 'mnk', ''):
            d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            deep_gemm.fp8_gemm_nt(a, b, d, recipe=(1, 1, 128), compiled_dims=compiled_dims)
            self.assertLess(calc_diff(d, ref), 1e-3, compiled_dims)
            outputs.append(d)
        for other in outputs[1:]:
            self.assertTrue(torch.equal(outputs[0], other))

    def test_fp8_gemm_is_bitwise_deterministic(self):
        """Persistent scheduling must not make results depend on the launch."""
        m, n, k = 1024, 512, 2048
        a, b = make_fp8(m, n, k, per_token_b=True)
        d = torch.empty((m, n), device='cuda', dtype=torch.float32)
        c = torch.zeros((m, n), device='cuda', dtype=torch.float32)
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, recipe=(1, 1, 128))
        first = d.clone()
        for _ in range(20):
            deep_gemm.fp8_gemm_nt(a, b, d, c=c, recipe=(1, 1, 128))
            self.assertTrue(torch.equal(d, first))

    def test_fp8_gemm_accumulation_matches_zero_c(self):
        """`D = C + A @ B` must equal the same workload accumulated from a zero FP32 C."""
        m, n, k = 512, 512, 1024
        a, b = make_fp8(m, n, k, per_token_b=False)
        for out_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(out_dtype=out_dtype):
                direct = torch.empty((m, n), device='cuda', dtype=out_dtype)
                deep_gemm.fp8_gemm_nt(a, b, direct, recipe=(1, 128, 128))
                fp32 = torch.zeros((m, n), device='cuda', dtype=torch.float32)
                deep_gemm.fp8_gemm_nt(a, b, fp32, c=fp32, recipe=(1, 128, 128))
                expected = fp32 if out_dtype == torch.float32 else fp32.to(torch.bfloat16)
                self.assertTrue(torch.equal(direct, expected))

    # ------------------------------------------------------------------ BF16 GEMM

    def test_bf16_gemm(self):
        """BF16 GEMMs are served by cuBLASLt; they must not silently drop to TF32."""
        for m, n, k in [(1, 128, 256), (64, 512, 1024), (512, 1024, 2048), (129, 320, 512)]:
            for accumulate in (False, True):
                for out_dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(m=m, n=n, k=k, accumulate=accumulate, out_dtype=out_dtype):
                        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
                        c = torch.randn((m, n), device='cuda', dtype=out_dtype) if accumulate else None
                        d = c.clone() if accumulate else torch.empty((m, n), device='cuda', dtype=out_dtype)
                        deep_gemm.bf16_gemm_nt(a, b, d, c=c)
                        ref = a.float() @ b.float().t() + (c.float() if accumulate else 0)
                        self.assertLess(calc_diff(d, ref.to(out_dtype)), 5e-3)

    def test_bf16_gemm_layouts(self):
        """NN / TN / TT entry points must agree with `A @ B`."""
        m, n, k = 128, 256, 512
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
        ref = torch.matmul(a, b)

        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nn(a, b, d)
        self.assertLess(calc_diff(d, ref), 5e-3)

        # `bf16_gemm_tn(A, B)` computes `A.T @ B`, so A is `[K, M]` and B is `[K, N]`
        deep_gemm.bf16_gemm_tn(a.mT.contiguous(), b, d)
        self.assertLess(calc_diff(d, ref), 5e-3)

        # `bf16_gemm_tt(A, B)` computes `A.T @ B.T`
        deep_gemm.bf16_gemm_tt(a.mT.contiguous(), b.mT.contiguous(), d)
        self.assertLess(calc_diff(d, ref), 5e-3)

    def test_cublaslt_fp8_and_bf16(self):
        """The cuBLASLt entry points (the BF16 fallback on SM89) also handle FP8."""
        m, n, k = 256, 512, 1024
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.cublaslt_gemm_nt(a, b, d)
        self.assertLess(calc_diff(d, torch.matmul(a, b.mT)), 5e-3)

        a_fp8 = a.to(torch.float8_e4m3fn)
        b_fp8 = b.to(torch.float8_e4m3fn)
        deep_gemm.cublaslt_gemm_nt(a_fp8, b_fp8, d)
        self.assertLess(calc_diff(d, (a_fp8.float() @ b_fp8.float().mT).to(torch.bfloat16)), 1e-3)

    # ------------------------------------------------------------- SF layout kernels

    def test_sf_layout_kernels(self):
        """`get_mn_major_tma_aligned_tensor` must JIT-compile and run on SM89."""
        for mn, k in ((4096, 7168), (4097, 128), (333, 1024), (128, 8192)):
            with self.subTest(mn=mn, k=k):
                sf = torch.randn((mn, ceil_div(k, 128)), device='cuda', dtype=torch.float32)
                out = _C.get_mn_major_tma_aligned_tensor(sf)
                self.assertEqual(tuple(out.shape), tuple(sf.shape))
                self.assertEqual(out.stride(-2), 1)
                self.assertEqual(out.stride(-1), ceil_div(mn, 4) * 4)
                self.assertTrue(torch.equal(out[..., :mn], sf))

    # -------------------------------------------------------- unsupported operations

    def test_fp8_gemm_nn_tn_tt_are_rejected(self):
        """Only the NT layout is implemented for FP8 on SM89, exactly as on SM90."""
        m, n, k = 128, 256, 512
        a = torch.randn((m, k), device='cuda').to(torch.float8_e4m3fn)
        sfa = torch.ones((m, ceil_div(k, 128)), dtype=torch.float32, device='cuda')
        b = torch.randn((n, k), device='cuda').to(torch.float8_e4m3fn)
        sfb = torch.ones((n, ceil_div(k, 128)), dtype=torch.float32, device='cuda')
        d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        for name in ('fp8_fp4_gemm_nn', 'fp8_fp4_gemm_tn', 'fp8_fp4_gemm_tt'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                getattr(deep_gemm, name)((a, sfa), (b.mT, sfb), d, recipe=(1, 1, 128))

    def test_fp4_is_rejected(self):
        """FP4 Tensor Cores are Blackwell-only and must fail loudly."""
        m, n, k = 128, 256, 512
        # DeepGEMM uses `int8` as the storage dtype for packed FP4 operands
        a = torch.randint(0, 255, (m, k // 2), dtype=torch.uint8, device='cuda').to(torch.int8)
        b = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device='cuda').to(torch.int8)
        sfa = torch.ones((m, ceil_div(k, 128)), dtype=torch.int32, device='cuda')
        sfb = torch.ones((n, ceil_div(k, 128)), dtype=torch.int32, device='cuda')
        d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        with self.assertRaises(RuntimeError) as ctx:
            deep_gemm.fp8_fp4_gemm_nt((a, sfa), (b, sfb), d)
        self.assertIn('Blackwell (SM100)', str(ctx.exception))

    def test_mn_major_fp8_operands_are_rejected(self):
        """SM89 FP8 GEMMs only implement the K-major (NT) operand layout."""
        m, n, k = 128, 256, 512
        a = torch.randn((m, k), device='cuda').to(torch.float8_e4m3fn)
        sfa = torch.ones((m, ceil_div(k, 128)), dtype=torch.float32, device='cuda')
        b = torch.randn((k, n), device='cuda').to(torch.float8_e4m3fn)
        sfb = torch.ones((n, ceil_div(k, 128)), dtype=torch.float32, device='cuda')
        d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        with self.assertRaises(RuntimeError) as ctx:
            deep_gemm.fp8_fp4_gemm_nt((a, sfa), (b.mT, sfb), d)
        self.assertIn('K', str(ctx.exception))

    def test_grouped_fp8_is_rejected_with_a_clear_message(self):
        """M-grouped FP8 GEMMs are SM90+ only; the error must say so."""
        num_groups, m, n, k = 4, 256, 512, 1024
        a = torch.randn((m, k), device='cuda').to(torch.float8_e4m3fn)
        sfa = torch.ones((m, ceil_div(k, 128)), dtype=torch.float32, device='cuda')
        b = torch.randn((num_groups, n, k), device='cuda').to(torch.float8_e4m3fn)
        sfb = torch.ones((num_groups, n, ceil_div(k, 128)), dtype=torch.float32, device='cuda')
        grouped_layout = torch.full((m,), 0, dtype=torch.int32, device='cuda')
        d = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        with self.assertRaises(RuntimeError) as ctx:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a, sfa), (b, sfb), d, grouped_layout)
        self.assertIn('Hopper (SM90) or newer', str(ctx.exception))

    def test_pdl_is_rejected(self):
        """Programmatic dependent launch needs `griddepcontrol`, i.e. SM90+."""
        self.assertFalse(deep_gemm.get_pdl())
        with self.assertRaises(RuntimeError) as ctx:
            deep_gemm.set_pdl(True)
        self.assertIn('SM90', str(ctx.exception))
        self.assertFalse(deep_gemm.get_pdl())

    def test_mqa_logits_are_rejected(self):
        """MQA / indexer kernels require Hopper or newer."""
        with self.assertRaises(RuntimeError) as ctx:
            deep_gemm.get_mqa_logits_metadata(
                torch.zeros((4,), dtype=torch.int32, device='cuda'),
                torch.zeros((4,), dtype=torch.int32, device='cuda'), 8, 4)
        self.assertIn('Blackwell (SM100)', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()