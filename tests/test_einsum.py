import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench, bench_kineto,
    calc_diff, count_bytes
)


def test_bmk_bnk_mn() -> None:
    print('Testing "bmk, bnk -> mn":')
    for s in (129, 4096, 8192):
        for m, n, k in [(128, 384, 128), (256, 256, 256), (384, 128, 384)]:
            for dtype in (torch.float, torch.bfloat16):
                a = torch.randn((s, m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((s, n, k), dtype=torch.bfloat16, device='cuda')
                d = torch.randn((m, n), dtype=dtype, device='cuda')
                c = d if dtype == torch.float else None

                # Test correctness
                ref_d = (c if dtype == torch.float else 0) + torch.bmm(a.float(), b.float().mT).sum(0)
                deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)
                assert calc_diff(d, ref_d) < 1e-5

                t = bench_kineto(lambda: deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c), 'bmn_bnk_mn_gemm_impl', suppress_kineto_output=True)
                print(f' > Perf (b={s:4.0f}, {m=}, {n=}, {k=}, {"FP32" if dtype == torch.float else "BF16"}): ',
                    f'{t * 1e6:4.0f} us | '
                    f'{2 * s * m * n * k / t / 1e12:4.0f} TFLOPS | '
                    f'{(count_bytes(a, b) + (d.numel() * 4)) / 1e9 / t:4.0f} GB/s')
    print()


def test_bhr_hdr_bhd():
    print('Testing "bhr, hdr -> bhd":')
    for b in (128, 4096, 8192):
        for h, r, d in [(128, 512, 128)]:
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)
            z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
            deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:.0f} GB/s')
    print()


def test_bhd_hdr_bhr():
    print('Testing "bhd, hdr -> bhr":')
    for b in (128, 4096, 8192):
        for h, r, d in [(128, 512, 128)]:
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)
            z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)
            deep_gemm.einsum('bhd,hdr->bhr', x, y, z)
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_kineto(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_bmk_bnk_mn()
    test_bhr_hdr_bhd()
    test_bhd_hdr_bhr()
