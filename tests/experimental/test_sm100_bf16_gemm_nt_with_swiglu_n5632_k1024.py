import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, assert_check, calc_diff, count_bytes, get_arch_major

SHAPE_N, SHAPE_K, SHAPE_B_N = 2816, 1024, 5632


def test() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    print('Testing bf16 GEMM with SwiGLU (n=5632, k=1024):')
    if get_arch_major() != 10:
        print(' > Skipped, the kernel requires SM100')
        return

    for m in (128, 256, 512, 1024, 4097):
        a = torch.randn((m, SHAPE_K), dtype=torch.bfloat16, device='cuda')
        b = torch.randn((SHAPE_B_N, SHAPE_K), dtype=torch.bfloat16, device='cuda')
        d = torch.empty((m, SHAPE_N), dtype=torch.bfloat16, device='cuda')

        def run() -> None:
            deep_gemm.experimental.bf16_gemm_nt_with_swiglu_n5632_k1024(a, b, d)

        run()
        z = a.float() @ b.float().T
        ref_d = (torch.nn.functional.silu(z[:, :SHAPE_N]) * z[:, SHAPE_N:]).bfloat16()
        diff = calc_diff(d, ref_d)
        assert_check(diff < 1e-5, f'{m=}, {diff=}')

        t = bench_kineto(run, 'swiglu', suppress_kineto_output=True)
        print(f' > Perf (m={m:5}, n={SHAPE_B_N}, k={SHAPE_K}): {t * 1e6:4.0f} us | '
              f'{2 * m * SHAPE_B_N * SHAPE_K / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')

    # Batch invariance under deterministic mode
    m_max = 16384
    a_full = torch.randn((m_max, SHAPE_K), dtype=torch.bfloat16, device='cuda')
    b_full = torch.randn((SHAPE_B_N, SHAPE_K), dtype=torch.bfloat16, device='cuda')
    deep_gemm.use_deterministic_algorithms(True)
    try:
        d_ref = torch.empty((m_max, SHAPE_N), dtype=torch.bfloat16, device='cuda')
        deep_gemm.experimental.bf16_gemm_nt_with_swiglu_n5632_k1024(a_full, b_full, d_ref)
        for m in (1, 32, 64, 128, 256, 384, 512, 640, 768, 1024, 1280, 1792, 2304, 4097, 8192):
            d_sub = torch.empty((m, SHAPE_N), dtype=torch.bfloat16, device='cuda')
            deep_gemm.experimental.bf16_gemm_nt_with_swiglu_n5632_k1024(a_full[:m].contiguous(), b_full, d_sub)
            assert_check(torch.equal(d_sub, d_ref[:m]),
                         f'Not batch-invariant in deterministic mode: {m=}')
        d_shift = torch.empty((137, SHAPE_N), dtype=torch.bfloat16, device='cuda')
        deep_gemm.experimental.bf16_gemm_nt_with_swiglu_n5632_k1024(a_full[37:37 + 137].contiguous(), b_full, d_shift)
        assert_check(torch.equal(d_shift, d_ref[37:37 + 137]),
                     'Shifted slice is not batch-invariant in deterministic mode')
    finally:
        deep_gemm.use_deterministic_algorithms(False)
    print()


if __name__ == '__main__':
    test()
