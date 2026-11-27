import torch

import deep_gemm
from deep_gemm.testing import assert_bitwise_equal, bench_kineto, get_arch_major
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8


def test() -> None:
    print('Testing segmented FP8 GEMM:')
    if get_arch_major() != 10:
        print(' > Skipped, the kernel requires SM100')
        return

    segment_k, segment_stride = 256, 384
    for n, k in ((5120, 3072), (7680, 3072), (11520, 3072),
                 (11520, 6144), (14080, 6144), (25600, 6144)):
        num_segments = k // segment_k
        b, b_sf = per_block_cast_to_fp8(
            torch.randn((n, k), device='cuda', dtype=torch.bfloat16), use_ue8m0=True, gran_k=32)
        b_sf = deep_gemm.transform_sf_into_required_layout(b_sf, n, k, (32, 32))
        for m in (1, 128, 4096):
            a, a_sf = per_token_cast_to_fp8(
                torch.randn((m, k), device='cuda', dtype=torch.bfloat16), use_ue8m0=True, gran_k=32)
            a_sf = deep_gemm.transform_sf_into_required_layout(a_sf, m, k, (1, 32))
            storage = torch.full((m, num_segments, segment_stride), float('nan'), device='cuda', dtype=torch.float8_e4m3fn)
            storage[:, :, :segment_k].copy_(a.view(m, num_segments, segment_k))
            a = storage[:, :, :segment_k]
            a_contiguous = a.contiguous().view(m, k)
            output = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            reference = torch.empty_like(output)

            def contiguous_func():
                deep_gemm.fp8_fp4_gemm_nt((a_contiguous, a_sf), (b, b_sf), reference, recipe=(1, 1, 32))

            def segmented_func():
                deep_gemm.experimental.fp8_gemm_nt_segmented_k256((a, a_sf), (b, b_sf), output)

            segmented_func()
            contiguous_func()
            assert_bitwise_equal(output, reference, f'{m=}, {n=}, {k=}')
            segmented_t = bench_kineto(segmented_func, 'gemm_', suppress_kineto_output=True)
            contiguous_t = bench_kineto(contiguous_func, 'gemm_', suppress_kineto_output=True)
            print(f' > Perf (m={m:4}, n={n}, k={k}): '
                  f'contiguous {contiguous_t * 1e6:.1f} us | {2 * m * n * k / contiguous_t / 1e12:.0f} TFLOPS, '
                  f'segmented {segmented_t * 1e6:.1f} us | {2 * m * n * k / segmented_t / 1e12:.0f} TFLOPS | '
                  f'{segmented_t / contiguous_t - 1:+.2%}')
    print()


if __name__ == '__main__':
    test()
