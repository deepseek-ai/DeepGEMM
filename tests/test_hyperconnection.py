import torch
import random

import deep_gemm
from deep_gemm.testing import (
    test_filter,
    bench_kineto,
    calc_diff, count_bytes
)
from deep_gemm.utils import align
from generators import get_arch_major


@test_filter(lambda: get_arch_major() >= 9)
def test_hc_prenorm_gemm() -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Testing hyperconnection prenorm GEMM:')
    for m in (13, 137, 4096, 8192):
        for n, k in [(24, 28672), (24, 7680), (24, 7168)]:
            for num_splits in [None, 16]:
                a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((n, k), dtype=torch.float, device='cuda')
                d = torch.empty((m, n), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
                s = torch.empty((m, ), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m), dtype=torch.float, device='cuda')
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
                final_d = d if num_splits is None else d.sum(0)
                final_s = s if num_splits is None else s.sum(0)

                ref_d = a.float() @ b.T
                ref_s = a.float().square().sum(-1)

                diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
                assert diff < 1e-8, f'{m=}, {n=}, {k=}, {diff:.10f}'

                t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits), 'tf32_hc_prenorm_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits={(num_splits or 0):2}): '
                      f'{t * 1e6:4.0f} us | '
                      f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                      f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
    print()




@test_filter(lambda: get_arch_major() == 9)
def test_hc_prenorm_split_k_reference_and_repeatability() -> None:
    # Regression test for the SM90 register-source WGMMA race: the in-flight
    # asynchronous WGMMA keeps reading the register A operands, so the next
    # stage must not overwrite them before `wgmma.wait_group` completes.
    #
    # The race only shows up with a multi-stage K loop (large K) and is
    # time-sensitive, so it is not deterministic. We enlarge the exposure by
    # sweeping split-k / PDL / side-stream paths and by asserting bit-exact
    # repeatability across repeated launches (a correct deterministic kernel
    # must produce identical results every time).
    previous_pdl = deep_gemm.get_pdl()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    # Use full-precision references (TF32 disabled) for the PyTorch GEMM.
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        print('Testing hyperconnection prenorm GEMM (split-k race regression):')
        for m, num_splits in [(256, 19), (512, 9), (1024, 4)]:
            for k in [4096, 16384]:
                for enable_pdl in [False, True]:
                    for use_side_stream in [False, True]:
                        deep_gemm.set_pdl(enable_pdl)
                        stream = torch.cuda.Stream() if use_side_stream else torch.cuda.current_stream()
                        with torch.cuda.stream(stream):
                            generator = torch.Generator(device='cuda').manual_seed(123)
                            n = 24
                            a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda', generator=generator)
                            b = torch.randn(n, k, dtype=torch.float32, device='cuda', generator=generator) * 0.02

                            blocks_per_split, remainder = divmod(k // 64, num_splits)
                            ref_d, ref_s = [], []
                            for split in range(num_splits):
                                start = (split * blocks_per_split + min(split, remainder)) * 64
                                end = start + (blocks_per_split + (split < remainder)) * 64
                                part = a[:, start:end].float()
                                ref_d.append(part @ b[:, start:end].T)
                                ref_s.append(part.square().sum(-1))
                            ref_d, ref_s = torch.stack(ref_d), torch.stack(ref_s)

                            previous_d, previous_s = None, None
                            for _ in range(3):
                                d = torch.empty(num_splits, m, n, device='cuda')
                                sqr_sum = torch.empty(num_splits, m, device='cuda')
                                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=num_splits)
                                stream.synchronize()
                                torch.testing.assert_close(d, ref_d, rtol=1e-3, atol=2e-3)
                                torch.testing.assert_close(sqr_sum, ref_s, rtol=1e-5, atol=2e-3)
                                if previous_d is not None:
                                    # A correct deterministic kernel is bit-exact across launches.
                                    torch.testing.assert_close(d, previous_d, rtol=0, atol=0)
                                    torch.testing.assert_close(sqr_sum, previous_s, rtol=0, atol=0)
                                previous_d, previous_s = d, sqr_sum
                        print(f' > OK (m={m:5}, k={k:5}, num_splits={num_splits:2}, '
                              f'pdl={int(enable_pdl)}, side_stream={int(use_side_stream)})')
    finally:
        deep_gemm.set_pdl(previous_pdl)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_hc_prenorm_gemm()
    test_hc_prenorm_split_k_reference_and_repeatability()
