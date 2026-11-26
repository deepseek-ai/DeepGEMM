import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, assert_check, calc_diff, count_bytes, get_arch_major

SHAPE_N, SHAPE_K = 1024, 1024


def test() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    print('Testing bf16 GEMM + bias + add + norm (n=1024, k=1024):')
    if get_arch_major() != 10:
        print(' > Skipped, the kernel requires SM100')
        return

    for m in (128, 256, 512, 1024, 4097):
        a = torch.randn((m, SHAPE_K), dtype=torch.bfloat16, device='cuda')
        b = torch.randn((SHAPE_N, SHAPE_K), dtype=torch.bfloat16, device='cuda')
        bias = torch.randn((SHAPE_N,), dtype=torch.bfloat16, device='cuda')
        gamma = torch.randn((SHAPE_N,), dtype=torch.bfloat16, device='cuda')
        residual = torch.randn((m, SHAPE_N), dtype=torch.bfloat16, device='cuda')
        residual_ref = residual.clone()
        d = torch.empty((m, SHAPE_N), dtype=torch.bfloat16, device='cuda')

        def run() -> None:
            deep_gemm.experimental.bf16_gemm_nt_with_bias_add_norm_n1024_k1024(
                a, b, bias, gamma, residual, d)

        run()
        z = a.float() @ b.float().T + bias.float()
        ref_res = (z + residual_ref.float()).bfloat16()
        ref_rstd = torch.rsqrt(ref_res.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref_d = (ref_res.float() * ref_rstd * gamma.float()).bfloat16()
        res_diff, d_diff = calc_diff(residual, ref_res), calc_diff(d, ref_d)
        assert_check(res_diff < 1e-5 and d_diff < 1e-5, f'{m=}, {res_diff=}, {d_diff=}')

        t = bench_kineto(run, 'with_bias_add_norm_n1024_k1024', suppress_kineto_output=True)
        num_bytes = count_bytes(a, b, bias, gamma, residual, d)
        print(f' > Perf (m={m:5}, k={SHAPE_K}, bias): {t * 1e6:4.0f} us | '
              f'{2 * m * SHAPE_N * SHAPE_K / t / 1e12:4.0f} TFLOPS | '
              f'{num_bytes / 1e9 / t:4.0f} GB/s | {res_diff=:.2e}, {d_diff=:.2e}')

    # Batch invariance under deterministic mode
    m_max = 16384
    a_full = torch.randn((m_max, SHAPE_K), dtype=torch.bfloat16, device='cuda')
    b_full = torch.randn((SHAPE_N, SHAPE_K), dtype=torch.bfloat16, device='cuda')
    bias_full = torch.randn((SHAPE_N,), dtype=torch.bfloat16, device='cuda')
    gamma_full = torch.randn((SHAPE_N,), dtype=torch.bfloat16, device='cuda')
    res_full = torch.randn((m_max, SHAPE_N), dtype=torch.bfloat16, device='cuda')
    deep_gemm.use_deterministic_algorithms(True)
    try:
        res_ref = res_full.clone()
        d_ref = torch.empty((m_max, SHAPE_N), dtype=torch.bfloat16, device='cuda')
        deep_gemm.experimental.bf16_gemm_nt_with_bias_add_norm_n1024_k1024(
            a_full, b_full, bias_full, gamma_full, res_ref, d_ref)
        for m in (1, 7, 32, 128, 512, 1024, 4097, 8192, 10240, 12288):
            res_sub = res_full[:m].clone()
            d_sub = torch.empty((m, SHAPE_N), dtype=torch.bfloat16, device='cuda')
            deep_gemm.experimental.bf16_gemm_nt_with_bias_add_norm_n1024_k1024(
                a_full[:m].contiguous(), b_full, bias_full, gamma_full, res_sub, d_sub)
            assert_check(torch.equal(res_sub, res_ref[:m]) and torch.equal(d_sub, d_ref[:m]),
                         f'Not batch-invariant in deterministic mode: {m=}')
        # Unaligned residual/d must also match aligned output bitwise
        m_unaligned = 137
        res_unaligned = torch.empty(m_unaligned * SHAPE_N + 8, dtype=torch.bfloat16, device='cuda')[1:1 + m_unaligned * SHAPE_N].view(m_unaligned, SHAPE_N)
        res_unaligned.copy_(res_full[37:37 + m_unaligned])
        d_unaligned = torch.empty(m_unaligned * SHAPE_N + 8, dtype=torch.bfloat16, device='cuda')[1:1 + m_unaligned * SHAPE_N].view(m_unaligned, SHAPE_N)
        deep_gemm.experimental.bf16_gemm_nt_with_bias_add_norm_n1024_k1024(
            a_full[37:37 + m_unaligned].contiguous(), b_full, bias_full, gamma_full, res_unaligned, d_unaligned)
        assert_check(torch.equal(res_unaligned, res_ref[37:37 + m_unaligned]) and
                     torch.equal(d_unaligned, d_ref[37:37 + m_unaligned]),
                     'Unaligned deterministic mode is not batch-invariant')
    finally:
        deep_gemm.use_deterministic_algorithms(False)
    print()


if __name__ == '__main__':
    test()
