"""Evaluate every expert of the original seed-0/1 strict FP64 scenarios."""
import unittest

import torch


HAS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12,0)


@unittest.skipUnless(HAS_SM120, 'SM120 GPU required')
class AllExpertsFp64Test(unittest.TestCase):
    def test_original_seeds_all_experts(self):
        import deep_gemm

        previous = deep_gemm.get_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        try:
            for seed in (0,1):
                # Keep the original generator, dimensions and tolerance. Only
                # extend the independently checked experts from three to 48.
                torch.manual_seed(seed)
                groups, m, k, n = 48, 128, 896, 4864
                a = torch.randn((groups,m,k), device='cuda', dtype=torch.bfloat16)
                b = torch.randn((groups,2*n,k), device='cuda', dtype=torch.bfloat16)/k**0.5
                counts = torch.full((groups,),m,device='cuda',dtype=torch.int32)
                projections = torch.empty((groups,m,2*n),device='cuda',dtype=torch.bfloat16)
                fused = torch.empty((groups,m,n),device='cuda',dtype=torch.bfloat16)
                deep_gemm.m_grouped_bf16_gemm_nt_masked(a,b,projections,counts,m)
                ordinary = (torch.nn.functional.silu(projections[...,:n].float())*
                            projections[...,n:].float()).bfloat16()
                deep_gemm.m_grouped_bf16_swiglu_nt_masked(a,b,fused,counts,m)
                for group in range(groups):
                    reference = (a[group].double() @ b[group].double().T).bfloat16().float()
                    oracle = (torch.nn.functional.silu(reference[:,:n])*reference[:,n:]).bfloat16()
                    for name, value in (('ordinary',ordinary[group]),('fused',fused[group])):
                        with self.subTest(seed=seed,expert=group,path=name):
                            torch.testing.assert_close(value,oracle,rtol=1e-2,atol=2e-3)
                del a,b,counts,projections,ordinary,fused,reference,oracle,value
        finally:
            deep_gemm.set_mk_alignment_for_contiguous_layout(previous)


if __name__=='__main__':
    unittest.main()
