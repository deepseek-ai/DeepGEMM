import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major, test_filter
)
from deep_gemm.utils.math import (
    ceil_div,
    per_block_cast_to_fp8, per_channel_cast_to_fp8, per_token_cast_to_fp8
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
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)
            z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
            deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), 'gemm', suppress_kineto_output=True)
            t_cublaslt = bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z, use_cublaslt=True), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


def test_bhd_hdr_bhr():
    print('Testing "bhd, hdr -> bhr":')
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)
            z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)
            deep_gemm.einsum('bhd,hdr->bhr', x, y, z)
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_kineto(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z), 'gemm', suppress_kineto_output=True)
            t_cublaslt = bench_kineto(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z, use_cublaslt=True), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


def test_fp8_bhr_hdr_bhd(use_ue8m0: bool = True):
    print('Testing FP8 "bhr, hdr -> bhd":')
    for h, r, d in [(8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)

            x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=use_ue8m0)
            x_fp8 = x_fp8[0].view(b, h, r), x_fp8[1].view(b, h, ceil_div(r, 128))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                     torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float))
            for i in range(h):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)
            z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)

            deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, z)
            assert calc_diff(z, ref_z) < 1e-3

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, z), 'fp8_gemm', suppress_kineto_output=True)
            t_cublaslt = bench_kineto(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z, use_cublaslt=True), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


@test_filter(lambda: get_arch_major() >= 10)
def test_fp8_bhd_hdr_bhr(use_ue8m0: bool = True):
    print('Testing FP8 "bhd, hdr -> bhr":')
    for h, r, d in [(8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)

            x_fp8 = per_token_cast_to_fp8(x.view(-1, d), use_ue8m0=use_ue8m0)
            x_fp8 = x_fp8[0].view(b, h, d), x_fp8[1].view(b, h, ceil_div(d, 128))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                     torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float))
            for i in range(h):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)
            z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)

            deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, z)
            assert calc_diff(z, ref_z) < 1e-3

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, z), 'fp8_gemm', suppress_kineto_output=True)
            t_cublaslt = bench_kineto(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z, use_cublaslt=True), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


@test_filter(lambda: get_arch_major() >= 10)
def test_fp8_bhd_bhr_hdr(use_ue8m0: bool = True):
    print('Testing FP8 "bhd, bhr -> hdr":')
    for h, r, d in [(8, 4096, 1024)]:
        for b in (4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            z_0 = torch.randn((h, d, r), device='cuda', dtype=torch.float32) * 10
            ref_z = z_0 + torch.einsum('bhd,bhr->hdr', x, y)

            x_fp8 = per_channel_cast_to_fp8(x.view(b, -1), use_ue8m0=use_ue8m0)
            y_fp8 = per_channel_cast_to_fp8(y.view(b, -1), use_ue8m0=use_ue8m0)
            x_fp8 = (x_fp8[0].view(b, h, d), x_fp8[1].view(ceil_div(b, 128), h, d))
            y_fp8 = (y_fp8[0].view(b, h, r), y_fp8[1].view(ceil_div(b, 128), h, r))
            z = z_0.clone()
            deep_gemm.fp8_einsum('bhd,bhr->hdr', x_fp8, y_fp8, z, z, recipe=(1, 1, 128))
            assert calc_diff(z, ref_z) < 1e-3

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhd,bhr->hdr', x_fp8, y_fp8, z, z, recipe=(1, 1, 128)), 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z, z)) / t / 1e9:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_bmk_bnk_mn()
    test_bhr_hdr_bhd()
    test_bhd_hdr_bhr()

    test_fp8_bhr_hdr_bhd()
    test_fp8_bhd_hdr_bhr()
    test_fp8_bhd_bhr_hdr()
