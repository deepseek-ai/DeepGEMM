import random
import torch
from typing import Tuple

from deep_gemm.utils.math import align, ceil_div, per_token_cast_to_fp8, per_block_cast_to_fp8
from deep_gemm.utils.layout import MajorTypeAB, get_m_alignment_for_contiguous_layout


def enumerate_normal():
    for m in (128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            for major_a, major_b in ((MajorTypeAB.KMajor,  MajorTypeAB.KMajor), (MajorTypeAB.KMajor,  MajorTypeAB.MNMajor),
                                     (MajorTypeAB.MNMajor, MajorTypeAB.KMajor), (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)):
                for out_dtype in (torch.bfloat16, torch.float):
                    for accumulate in (False, ) if out_dtype == torch.bfloat16 else (False, True):
                        yield m, k, n, major_a, major_b, accumulate, out_dtype


def enumerate_grouped_contiguous():
    for num_groups, expected_m_per_group, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
        for major_a, major_b in ((MajorTypeAB.KMajor,  MajorTypeAB.KMajor), (MajorTypeAB.KMajor,  MajorTypeAB.MNMajor)):
            yield num_groups, expected_m_per_group, k, n, major_a, major_b


def enumerate_grouped_masked():
    for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            yield num_groups, m, k, n


def generate_normal(m: int, k: int, n: int,
                    major_a: MajorTypeAB, major_b: MajorTypeAB,
                    accumulate: bool, out_dtype: torch.dtype):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    c = torch.randn((m, n), device='cuda', dtype=out_dtype) * 64 if accumulate else None
    d = torch.empty((m, n), device='cuda', dtype=out_dtype)
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    a_fp8, b_fp8 = per_token_cast_to_fp8(a), per_block_cast_to_fp8(b)
    a_fp8 = a_fp8 if major_a == MajorTypeAB.KMajor else (a_fp8[0].T.contiguous().T, a_fp8[1])
    b_fp8 = b_fp8 if major_b == MajorTypeAB.KMajor else (b_fp8[0].T.contiguous().T, b_fp8[1])
    return a_fp8, b_fp8, c, d, ref_d


def generate_grouped_contiguous(num_groups: int, expected_m_per_group: int, k: int, n: int, major_a: MajorTypeAB, major_b: MajorTypeAB) -> \
        Tuple[int, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    group_ms = [align(int(expected_m_per_group * random.uniform(0.7, 1.3)), get_m_alignment_for_contiguous_layout()) for _ in range(num_groups)]
    m = sum(group_ms)

    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    m_indices = torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, group_m in enumerate(group_ms):
        end = start + group_m
        m_indices[start:end] = i
        ref_d[start:end] = a[start:end] @ b[i].t()
        start = end

    assert major_a == MajorTypeAB.KMajor
    a_fp8 = per_token_cast_to_fp8(a)
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn),
             torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i])
    b_fp8 = b_fp8 if major_b == MajorTypeAB.KMajor else (b_fp8[0].mT.contiguous().mT, b_fp8[1])
    return m, a_fp8, b_fp8, m_indices, d, ref_d


def generate_grouped_masked(num_groups: int, m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    a = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    a_fp8 = (torch.empty_like(a, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, ceil_div(k, 128)), device='cuda', dtype=torch.float))
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = per_token_cast_to_fp8(a[i])
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i])

    return a_fp8, b_fp8, d, ref_d
