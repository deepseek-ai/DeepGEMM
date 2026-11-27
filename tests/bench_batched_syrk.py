import ctypes
import math
import os
import tempfile
import torch

import deep_gemm
from deep_gemm.testing import bench, bench_kineto
from generators import enumerate_batched_syrk_symm

# Kernel names covering cuBLASLt (`nvjet`, `gemv`, `gemm`) and `sm90_bf16_batched_syrk_gemm_impl`
KERNEL_NAMES = ('nvjet', 'gemv', 'gemm')
MODES = ('default', 'native', 'fallback')


def run_with_mode(mode: str, fn):
    # `DG_SYRK_FORCE_*` are internal benchmark controls that bypass the dispatch cutoff
    key = {'native': 'DG_SYRK_FORCE_NATIVE', 'fallback': 'DG_SYRK_FORCE_FALLBACK'}.get(mode)
    if key is not None:
        os.environ[key] = '1'
    try:
        return fn()
    finally:
        if key is not None:
            os.environ.pop(key)


def get_native_config(fn):
    # Parse the native kernel config printed by `DG_PRINT_CONFIGS` (`None` means cuBLASLt ran)
    os.environ['DG_PRINT_CONFIGS'] = '1'
    with tempfile.TemporaryFile(mode='w+') as tmp:
        saved = os.dup(1)
        os.dup2(tmp.fileno(), 1)
        try:
            fn()
            torch.cuda.synchronize()
            ctypes.CDLL(None).fflush(None)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.environ.pop('DG_PRINT_CONFIGS')
        tmp.seek(0)
        for line in tmp.read().splitlines():
            if line.startswith('SM90 BF16 Batched SYRK'):
                return {k: int(v) for k, v in (kv.split('=') for kv in line.split('-> ')[1].replace(',', '').split())}
    return None


def get_issued_flops(b: int, n: int, k: int, config) -> int:
    # WGMMA work actually issued: full diagonal tiles, 128x256 super-tiles, and K padded to 64
    block_m, block_n = config['BLOCK_M'], config['BLOCK_N']
    t = math.ceil(n / block_m)
    tiles = t * (t + 1) // 2 if block_n == block_m else (t // 2) * (t // 2 + 1) + (t // 2 + 1) * (t % 2)
    return 2 * b * tiles * block_m * block_n * math.ceil(k / 64) * 64


def report(label: str, b: int, n: int, k: int, fn, kineto: bool = True):
    dense_flops = 2 * b * n * n * k
    for mode in MODES:
        config = run_with_mode(mode, lambda: get_native_config(fn))
        t = run_with_mode(mode, lambda: sum(bench_kineto(fn, KERNEL_NAMES, suppress_kineto_output=True)) if kineto else bench(fn))
        route = f"native {config['BLOCK_M']}x{config['BLOCK_N']} c{config['cluster']}" if config else 'cuBLASLt'
        issued = f'{get_issued_flops(b, n, k, config) / t / 1e12:5.0f}' if config else '    -'
        print(f' > {label} (b={b:2}, n={n:5}, k={k:5}) {mode:>8} [{route:>17}]: {t * 1e6:7.1f} us | '
              f'{dense_flops / t / 1e12:5.0f} effective TFLOPS (2BN^2K) | {issued} issued TFLOPS')


def bench_batched_syrk() -> None:
    print('Benchmarking batched SYRK (Kineto kernel time unless noted):')
    for b, m, k, dtype in enumerate_batched_syrk_symm():
        if dtype != torch.bfloat16:
            continue
        n, k = min(m, k), max(m, k)
        for k_major in (True, False):
            a = torch.randn((b, n, k) if k_major else (b, k, n), device='cuda', dtype=torch.bfloat16)
            a = a if k_major else a.mT
            d = torch.empty(b, n, n, device='cuda', dtype=torch.bfloat16)
            report(f'D = A @ A.mT, layout={"N" if k_major else "T"}', b, n, k, lambda: deep_gemm.batched_syrk(a, d))
    a = torch.randn(16, 1024, 1024, device='cuda', dtype=torch.bfloat16)
    d = torch.empty(16, 1024, 1024, device='cuda', dtype=torch.bfloat16)
    report('D = A @ A.mT, layout=N', 16, 1024, 1024, lambda: deep_gemm.batched_syrk(a, d))
    print()

    # Active `C` on either side of the dispatch cutoff (`n >= 2048` and `k >= 4096`), with BF16 and FP32 outputs;
    # mixed-dtype cases are timed end to end with CUDA events, as the fallback includes a dtype conversion kernel
    print('Benchmarking fused D = alpha * (A @ A.mT) + beta * C:')
    for b, n in ((16, 1536), (8, 2048), (8, 4096)):
        a = torch.randn(b, n, n, device='cuda', dtype=torch.bfloat16) / n ** 0.5
        for c_dtype, d_dtype in ((torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float), (torch.float, torch.bfloat16)):
            c = torch.randn(b, n, n, device='cuda', dtype=c_dtype)
            d = torch.empty(b, n, n, device='cuda', dtype=d_dtype)
            label = f'C={str(c_dtype)[6:]:>8} D={str(d_dtype)[6:]:>8}'
            is_cast = c_dtype != d_dtype
            report(label + (' (events)' if is_cast else ''), b, n, n,
                   lambda: deep_gemm.batched_syrk(a, d, c=c, alpha=2.0315, beta=-4.7750), kineto=not is_cast)
    print()

    # Muon Newton-Schulz step 2: P = b * G + c * (G @ G.mT). The numerical comparison is the fused cuBLASLt route
    # (forced fallback); the unfused sequence below uses this build's default SYRK dispatch plus two elementwise launches
    print('Benchmarking Muon step 2 (CUDA events):')
    for b, n in ((16, 4608), (16, 1536), (16, 576), (8, 4096), (8, 2048), (4, 5120)):
        g = torch.randn(b, n, n, device='cuda', dtype=torch.bfloat16)
        g = ((g.float() + g.float().mT) * 0.5 / n ** 0.5).to(torch.bfloat16)
        p, tmp = torch.empty_like(g), torch.empty_like(g)
        t_hist = bench(lambda: (deep_gemm.batched_syrk(g, tmp), torch.add(g, tmp, alpha=2.0315 / -4.7750).mul_(-4.7750)))
        print(f' > Muon (b={b:2}, n={n:5}) same-build unfused (SYRK + two elementwise launches): {t_hist * 1e6:7.1f} us')
        fused = lambda: deep_gemm.batched_syrk(g, p, c=g, alpha=2.0315, beta=-4.7750)
        times = {mode: run_with_mode(mode, lambda: bench(fused)) for mode in MODES}
        for mode in MODES:
            config = run_with_mode(mode, lambda: get_native_config(fused))
            t = times[mode]
            print(f' > Muon (b={b:2}, n={n:5}) fused {mode:>8} [{"native" if config else "cuBLASLt":>8}]: {t * 1e6:7.1f} us | '
                  f'{2 * b * n ** 3 / t / 1e12:5.0f} effective TFLOPS | {times["fallback"] / t:4.2f}x vs fused cuBLASLt | '
                  f'{t_hist / t:4.2f}x vs same-build unfused')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    bench_batched_syrk()
