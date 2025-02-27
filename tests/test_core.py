import pytest
import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor

# Set seeds and TF32 flags up front
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(0)
random.seed(0)

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

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

def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float)
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float)
    )
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out

@pytest.mark.parametrize("m", [64, 128, 4096])
@pytest.mark.parametrize("k,n", [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)])
def test_gemm(m, k, n):
    """Test single GEMM with various dimensions."""
    x_fp8, y_fp8, out, ref_out = construct(m, k, n)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
    diff = calc_diff(out, ref_out)
    assert diff < 0.001, f'GEMM mismatch: m={m}, k={k}, n={n}, diff={diff:.5f}'

    # (Optional) performance timing
    def test_func():
        # Construct new tensors every time to avoid L2 cache effects
        x_fp8_local, y_fp8_local, out_local, _ = construct(m, k, n)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8_local, y_fp8_local, out_local)

    t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
    print(f'[test_gemm] (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:6.0f} us | '
          f'{2 * m * n * k / t / 1e12:5.2f} TFLOPS')

@pytest.mark.parametrize("num_groups,m,k,n", [
    (4, 8192, 7168, 4096),
    (4, 8192, 2048, 7168),
    (8, 4096, 7168, 4096),
    (8, 4096, 2048, 7168)
])
def test_m_grouped_gemm_contiguous(num_groups, m, k, n):
    """Test grouped GEMM with merged group+M dimension."""
    x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
    m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
    m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
    diff = calc_diff(out, ref_out)
    assert diff < 0.001, f'Grouped GEMM mismatch: m={m*num_groups}, k={k}, n={n}, diff={diff:.5f}'

    # (Optional) performance timing
    def test_func():
        x_fp8_local, y_fp8_local, out_local, _ = construct_grouped(num_groups, m, k, n, is_masked=False)
        m_indices_local = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
        m_indices_local = m_indices_local.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8_local, y_fp8_local, out_local, m_indices_local)

    t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
    print(f'[test_m_grouped_gemm_contiguous] (groups={num_groups}, m={m}, n={n}, k={k}): '
          f'{t * 1e6:6.0f} us | {2 * num_groups * m * n * k / t / 1e12:5.2f} TFLOPS')

@pytest.mark.parametrize("num_groups,m", [
    (1, 1024),
    (2, 512),
    (4, 256),
])
@pytest.mark.parametrize("k,n", [
    (7168, 4096),
    (2048, 7168),
])
def test_m_grouped_gemm_masked(num_groups, m, k, n):
    """Test grouped GEMM where each group can have a different 'masked' M dimension."""
    # Test correctness with random partial masks
    masked_m_candidates = list(filter(lambda candidate: candidate <= m, (64, 128, 192, 256, 320, 384)))
    for _ in range(10):
        x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
        masked_m = torch.empty((num_groups,), device='cuda', dtype=torch.int)
        for j in range(num_groups):
            masked_m[j] = random.choice(masked_m_candidates)
        # We pick an expected M that covers the average usage
        expected_m = min(int(masked_m.float().mean()) + 1, m)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m)

        for j in range(num_groups):
            diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
            assert diff < 0.001, (
                f'Grouped Masked GEMM mismatch: m={m}, k={k}, n={n}, group={j}, '
                f'masked_m={masked_m[j]}, groups={num_groups}, diff={diff:.5f}'
            )

    # Test performance with full dimension (no actual mask)
    def test_func():
        x_fp8_local, y_fp8_local, out_local, _ = construct_grouped(num_groups, m, k, n, is_masked=True)
        masked_m = torch.ones((num_groups,), device='cuda', dtype=torch.int) * m
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8_local, y_fp8_local, out_local, masked_m, m)

    t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
    print(f'[test_m_grouped_gemm_masked] (groups={num_groups}, m={m}, n={n}, k={k}): '
          f'{t * 1e6:6.0f} us | {2 * num_groups * m * n * k / t / 1e12:5.2f} TFLOPS')
