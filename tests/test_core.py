# PyTorch has its own NVRTC, which may have a lower version than the system
# So try to disable PyTorch's NVRTC, or import NVRTC before PyTorch
import cuda.bindings.nvrtc as nvrtc
print(f'NVRTC version: {nvrtc.nvrtcVersion()[1:]}')

import random
import torch
from typing import List, Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from deep_gemm.jit_kernels.utils import get_m_alignment_for_contiguous_layout


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def construct_contiguous_grouped(num_groups: int, expected_m_per_group: int, k: int, n: int) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    alignment = get_m_alignment_for_contiguous_layout()
    group_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    m = sum([ceil_div(x, alignment) * alignment for x in group_ms])

    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)    
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        actual_end = start + group_m
        aligned_end = start + ceil_div(group_m, alignment) * alignment
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_out[start:aligned_end] = x[start:aligned_end] @ y[i].t()
        start = aligned_end
    ref_out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_out), ref_out)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    return m, x_fp8, y_fp8, m_indices, out, ref_out


def construct_masked_grouped(num_groups: int, max_m: int, expected_m_per_group: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert max_m % 4 == 0, f'TMA alignment error: {max_m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, max_m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))

    # Construct mask
    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m
    return x_fp8, y_fp8, masked_m, out, ref_out


def construct_wgrad(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    residual = torch.randn((m, n), device='cuda', dtype=torch.float) * 10
    out = residual.clone()
    ref_out = residual + (x.float() @ y.float().t())

    x_fp8 = per_token_cast_to_fp8(x)
    y_fp8 = per_token_cast_to_fp8(y)

    # NOTES: please do inplace add on the `out` later
    return x_fp8, y_fp8, residual, out, ref_out


def construct_k_grouped_wgrad(m: int, n: int, k_sizes: List[int]) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[int]]:
    num_groups, total_k = len(k_sizes), sum(k_sizes)

    x_flat = torch.empty((m * total_k,), device='cuda', dtype=torch.bfloat16)
    y_flat = torch.empty((n * total_k,), device='cuda', dtype=torch.bfloat16)
    out = torch.zeros((num_groups, m, n), device='cuda', dtype=torch.float)
    ref_out = torch.zeros((num_groups, m, n), device='cuda', dtype=torch.float)

    # Fill tensors with data and compute reference output
    x_offset, y_offset = 0, 0
    for idx, k in enumerate(k_sizes):
        x_chunk = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        y_chunk = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

        x_flat[x_offset:x_offset + m * k].copy_(x_chunk.flatten())
        y_flat[y_offset:y_offset + n * k].copy_(y_chunk.flatten())
        ref_out[idx] = x_chunk.float() @ y_chunk.float().t()
        
        x_offset += m * k
        y_offset += n * k

    x_fp8_flat = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    y_fp8_flat = torch.empty_like(y_flat, dtype=torch.float8_e4m3fn)

    total_scale_factors = sum(ceil_div(k, 128) for k in k_sizes)
    x_scales = torch.empty((total_scale_factors, m), device='cuda', dtype=torch.float)
    y_scales = torch.empty((total_scale_factors, n), device='cuda', dtype=torch.float)
    
    # Cast to FP8 and prepare scale factors
    x_offset, y_offset, scale_offset = 0, 0, 0
    for k in k_sizes:
        x_fp8_chunk, x_scale_chunk = per_token_cast_to_fp8(x_flat[x_offset:x_offset + m * k].view(m, k))
        y_fp8_chunk, y_scale_chunk = per_token_cast_to_fp8(y_flat[y_offset:y_offset + n * k].view(n, k))

        x_fp8_flat[x_offset:x_offset + m * k].copy_(x_fp8_chunk.flatten())
        y_fp8_flat[y_offset:y_offset + n * k].copy_(y_fp8_chunk.flatten())
        
        num_scales = ceil_div(k, 128)
        x_scales[scale_offset:scale_offset + num_scales].copy_(x_scale_chunk.T)
        y_scales[scale_offset:scale_offset + num_scales].copy_(y_scale_chunk.T)
        
        x_offset += m * k
        y_offset += n * k
        scale_offset += num_scales
    
    return (x_fp8_flat, x_scales), (y_fp8_flat, y_scales), out, ref_out, k_sizes


def test_gemm() -> None:
    print('Testing GEMM:')
    for m in (64, 128, 4096):
        for k, n in [(576, 7168), (7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Perf (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168),
                                                   (8, 4096, 7168, 4096), (8, 4096, 2048, 7168),
                                                   (32, 256, 7168, 4096), (32, 256, 2048, 7168)):
        # NOTES: we should mask the unfilled part before calculating difference
        m, x_fp8, y_fp8, m_indices, out, ref_out = construct_contiguous_grouped(num_groups, expected_m_per_group, k, n)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        out = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(out), out)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        valid_m = (m_indices != -1).sum().item()
        print(f' > Perf ({num_groups=:2}, {expected_m_per_group=:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, expected_m_per_group in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            # Test correctness
            for i in range(10):
                x_fp8, y_fp8, masked_m, out, ref_out = construct_masked_grouped(num_groups, 4096, expected_m_per_group, k, n)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{expected_m_per_group=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

            # noinspection PyShadowingNames
            def test_func():
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m_per_group)

            # Test performance with fixed shapes
            # noinspection PyUnboundLocalVariable
            valid_m = masked_m.sum().item()
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(valid_m * k + num_groups * k * n + valid_m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


def test_wgrad_gemm():
    print('Testing weight gradient GEMM:')

    for k in (4096, 8192):
        for m, n in ((7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)):
            # Test correctness
            x_fp8, y_fp8, residual, out, ref_out = construct_wgrad(m, k, n)
            deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
            x_fp8, y_fp8, residual, out, ref_out = construct_wgrad(m, k, n)

            # noinspection PyShadowingNames
            def test_func():
                deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_wgrad_gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_wgrad_gemm():
    print('Testing grouped weight gradient GEMM:')

    for num_groups, base_k in ((4, 4096), (4, 8192), (8, 4096)):
        for m, n in ((7168, 4096), (2048, 7168)):
            # Vary k sizes around base_k
            k_sizes = [base_k + random.randint(-1, 1) * 128 for _ in range(num_groups - 1)]
            k_sizes.append(base_k * num_groups - sum(k_sizes))
            
            # Test correctness
            x_fp8, y_fp8, out, ref_out, k_sizes = construct_k_grouped_wgrad(m, n, k_sizes)
            deep_gemm.k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(x_fp8, y_fp8, out, k_sizes)

            for idx in range(num_groups):
                diff = calc_diff(out[idx], ref_out[idx])
                assert diff < 0.001, f'{num_groups=}, {m=}, {n=}, k={k_sizes[idx]}, batch={idx}, {diff:.5f}'

            # Construct new tensors to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out, k_sizes = construct_k_grouped_wgrad(m, n, k_sizes)
            total_k = sum(k_sizes)
            
            def test_func():
                deep_gemm.k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(x_fp8, y_fp8, out, k_sizes)
            
            t = bench_kineto(test_func, 'fp8_wgrad_gemm', suppress_kineto_output=True, with_multiple_kernels=True) * num_groups
            print(f' > Performance ({num_groups=}, m={m:5}, n={n:5}, avg_k={total_k//num_groups:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * num_groups * m * n * (total_k/num_groups) / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * total_k + n * total_k + num_groups * m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()

    test_wgrad_gemm()
    test_k_grouped_wgrad_gemm()
