import enum
import itertools
import random
import torch
from math import prod
from typing import Generator, List, Optional, Tuple

from deep_gemm.testing import get_arch_major
from deep_gemm.utils import (
    align, ceil_div,
    per_token_cast_to_fp8, per_channel_cast_to_fp8, per_block_cast_to_fp8,
    per_token_cast_to_fp4, transpose_packed_fp4,
    get_mk_alignment_for_contiguous_layout,
    set_mk_alignment_for_contiguous_layout
)


class KernelType(enum.Enum):
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1
    

class QuantConfig:
    _legacy_quant_config = (128, 128, False, False)

    def __init__(self, value: Tuple[int, int, bool, bool] = _legacy_quant_config):
        self.gran_k_a, self.gran_k_b, self.is_fp4_a, self.is_fp4_b = value

    def print(self):
        print(f' > Testing with gran_k_a={self.gran_k_a}, gran_k_b={self.gran_k_b}, '
              f'is_fp4_a={self.is_fp4_a}, is_fp4_b={self.is_fp4_b}')

    def is_legacy(self) -> bool:
        return (self.gran_k_a, self.gran_k_b, self.is_fp4_a, self.is_fp4_b) == self._legacy_quant_config

    def is_fp4_fp4(self) -> bool:
        return self.is_fp4_a and self.is_fp4_b

    def get_recipes(self, is_wgrad: bool = False) -> Tuple[Tuple, Tuple, Tuple]:
        recipe, recipe_a, recipe_b = None, None, None
        if self.is_legacy():
            recipe = (1, 1, 128) if is_wgrad else None
        else:
            recipe_a = (1, self.gran_k_a)
            recipe_b = (1, self.gran_k_b) if self.is_fp4_b or is_wgrad else (self.gran_k_b, self.gran_k_b)
        return recipe, recipe_a, recipe_b

    def max_diff(self) -> float:
        if self.is_fp4_a and self.is_fp4_b:
            return 0.02
        if self.is_fp4_a or self.is_fp4_b:
            return 0.01
        return 0.001

    @staticmethod
    def get_list_from_dtype(dtype: torch.dtype) -> List:
        if dtype == torch.bfloat16:
            return [None]
        if dtype == torch.float4_e2m1fn_x2:
            return [QuantConfig((32, 32, True, True))]
        quant_config_list = [QuantConfig()]
        if get_arch_major() == 10:
            quant_config_list.append(QuantConfig((128, 32, False, True)))
            quant_config_list.append(QuantConfig((32, 32, True, True)))
        return quant_config_list


def reset_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_ue8m0_usage(kernel_type: KernelType) -> bool:
    if get_arch_major() == 9:
        return False
    return kernel_type.is_1d1d()


def get_kernel_types(dtype: torch.dtype) -> tuple:
    if dtype == torch.bfloat16:
        return (KernelType.KernelNoSF, )

    return (KernelType.Kernel1D2D, ) if get_arch_major() == 9 else (KernelType.Kernel1D1D, )


def get_major_ab(allow_a_mn_major: bool, allow_b_mn_major: bool) -> Generator:
    for major_a in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
        for major_b in (MajorTypeAB.KMajor, MajorTypeAB.MNMajor):
            if major_a.is_mn_major() and not allow_a_mn_major:
                continue
            if major_b.is_mn_major() and not allow_b_mn_major:
                continue
            yield major_a, major_b


def get_psum_layout_usage() -> tuple:
    return True, False


def enumerate_normal(dtype: torch.dtype, collect_cublas_scores: bool = False) -> Generator:
    assert dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.bfloat16)
    assert not collect_cublas_scores or dtype != torch.bfloat16

    quant_config_list = QuantConfig.get_list_from_dtype(dtype)
    fp32_output_nk = [(256, 7168), (129280, 7168)]
    bf16_output_nk = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
    m_fwd_list, m_bwd_list = [1, 128, 4096], [4096, ]
    nk_list = list(bf16_output_nk)

    # Only BF16 GEMM needs FP32 outputs
    if dtype == torch.bfloat16:
        nk_list += fp32_output_nk

    for kernel_type in get_kernel_types(dtype):
        for quant_config in quant_config_list:
            scores = []
            if len(quant_config_list) > 1:
                quant_config.print()
            reset_seed()

            def emit(*args):
                return (*args, scores) if collect_cublas_scores else args

            # Forward
            for m in m_fwd_list:
                for i, (n, k) in enumerate(nk_list):
                    out_dtype = torch.bfloat16 if i < len(bf16_output_nk) else torch.float
                    yield emit(kernel_type, quant_config, m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, out_dtype)
                    # BF16 accumulation: supported on all BF16 GEMMs, and SM100 FP8/FP4 GEMMs
                    if out_dtype == torch.bfloat16 and (dtype == torch.bfloat16 or get_arch_major() == 10):
                        yield emit(kernel_type, quant_config, m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, True, out_dtype)

            # Backward
            if quant_config is None or not quant_config.is_fp4_fp4():
                for m in m_bwd_list:
                    for n, k in nk_list:
                        override_major = MajorTypeAB.MNMajor
                        override_kernel_type = kernel_type
                        if get_arch_major() == 9 and dtype == torch.float8_e4m3fn:
                            override_major = MajorTypeAB.KMajor
                            override_kernel_type = KernelType.Kernel1D1D
                        yield emit(kernel_type,          quant_config, m, k, n, MajorTypeAB.KMajor, override_major, False, torch.bfloat16)  # Dgrad
                        yield emit(override_kernel_type, quant_config, n, m, k, override_major, override_major, True,  torch.float)         # Wgrad
                        yield emit(override_kernel_type, quant_config, n, m, k, override_major, override_major, False, torch.bfloat16)      # Wgrad
                        if dtype == torch.bfloat16 or get_arch_major() == 10:
                            yield emit(override_kernel_type, quant_config, n, m, k, override_major, override_major, False, torch.float)     # Wgrad

            if collect_cublas_scores and scores:
                quant_type = f'FP{4 if quant_config.is_fp4_a else 8}xFP{4 if quant_config.is_fp4_b else 8}'
                print(f'Average {quant_type} GEMM speedup over cuBLASLt: '
                      f'{prod(scores) ** (1.0 / len(scores)):.3f}x\n')


def enumerate_batched_syrk_symm() -> Generator:
    # Real Muon parameter shapes from 5120/7168 models. Batch merging is capped at 16 in hai-llm.
    b_m_k_list = [
        (16, 4608, 5120),
        (16, 1536, 5120),
        (16, 576, 5120),
        (8, 24576, 1536),
        (8, 32768, 512),
        (4, 5120, 16384),
        (8, 4096, 7168),
        (8, 7168, 2048),
    ]
    for dtype in (torch.bfloat16, torch.float):
        reset_seed()
        for b, m, k in b_m_k_list:
            yield b, m, k, dtype


def enumerate_m_grouped_contiguous(dtype: torch.dtype) -> Generator:
    quant_config_list = QuantConfig.get_list_from_dtype(dtype)
    m_group_list = [(4, 8192), (8, 4096)]
    n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]
    for kernel_type in get_kernel_types(dtype):
        for quant_config in quant_config_list:
            if len(quant_config_list) > 1:
                quant_config.print()
            for use_psum_layout in get_psum_layout_usage():
                for ensure_zero_padding in ((False, True) if use_psum_layout and get_arch_major() == 10 else (False, )):
                    reset_seed()
                    for num_groups, expected_m_per_group in m_group_list:
                        for n, k in n_k_list:
                            for major_a, major_b in get_major_ab(False, get_arch_major() != 9 or dtype != torch.float8_e4m3fn):
                                if quant_config is not None and quant_config.is_fp4_fp4() and major_b.is_mn_major():
                                    continue
                                yield kernel_type, quant_config, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout, ensure_zero_padding


def enumerate_m_grouped_masked(dtype: torch.dtype) -> Generator:
    quant_config_list = QuantConfig.get_list_from_dtype(dtype)
    max_m = 4096
    m_group_list = [(32, 192), (6, 1024), (32, 20), (6, 20)]
    n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]
    for kernel_type in get_kernel_types(dtype):
        for quant_config in quant_config_list:
            if len(quant_config_list) > 1:
                quant_config.print()
            for use_psum_layout in get_psum_layout_usage():
                reset_seed()
                for num_groups, m in m_group_list:
                    for n, k in n_k_list:
                        yield kernel_type, quant_config, num_groups, max_m, m, n, k, use_psum_layout


def enumerate_k_grouped_contiguous(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        sf_layout_list = [(128, 128)] if get_arch_major() == 9 else [(128, 128), (256, 256), (384, 384)]
    elif dtype == torch.float4_e2m1fn_x2:
        sf_layout_list = [(32, 256), (32, 512), (32, 768)]
    else:
        sf_layout_list = ([(128, 128)] if get_arch_major() == 9 else
                          [(32, 128), (128, 128), (32, 256), (128, 256), (32, 384), (128, 384)])
    # Only K-major is supported for SM90 FP8
    if get_arch_major() == 9 and dtype == torch.float8_e4m3fn:
        major_a, major_b = MajorTypeAB.KMajor, MajorTypeAB.KMajor
    elif dtype == torch.float4_e2m1fn_x2:
        major_a, major_b = MajorTypeAB.KMajor, MajorTypeAB.KMajor
    else:
        major_a, major_b = MajorTypeAB.MNMajor, MajorTypeAB.MNMajor
    psum_list = (False, True) if get_arch_major() == 10 else (False, )
    if get_arch_major() == 9:
        cd_options = [(True, torch.float)]
    else:
        cd_options = [(True, torch.float), (False, torch.float), (False, torch.bfloat16)]

    # NOTES: the first shape has many small groups, for stressing the SM90 in-place tensor map update
    for num_groups, m, n, expected_k_per_group in (( 8,  768, 2048,  128),
                                                   ( 4, 4096, 7168, 8192), ( 4, 7168, 2048, 8192),   # EP64
                                                   ( 8, 4096, 7168, 4096), ( 8, 7168, 2048, 4096),   # EP32
                                                   (16, 4096, 7168, 2048), (16, 7168, 2048, 2048)):  # EP16
        real_ks_cpu = [max(1, int(expected_k_per_group * random.uniform(0.7, 1.3))) for _ in range(num_groups)]
        for use_psum_layout in psum_list:
            for gran_k, k_alignment in sf_layout_list:
                set_mk_alignment_for_contiguous_layout(k_alignment)
                aligned_ks_cpu = [align(k, k_alignment) for k in real_ks_cpu]
                for accumulate, out_dtype in cd_options:
                    yield (num_groups, m, n, major_a, major_b, real_ks_cpu, aligned_ks_cpu,
                           expected_k_per_group, gran_k, k_alignment, use_psum_layout, accumulate, out_dtype)


def enumerate_k_grouped_contiguous_test_variants(real_ks_cpu: List[int]):
    yield list(real_ks_cpu)

    empty_ks_cpu = [0 if random.random() < 0.25 else k for k in real_ks_cpu]
    empty_ks_cpu[0], empty_ks_cpu[-1] = 0, real_ks_cpu[-1]
    yield empty_ks_cpu

    single_group_ks_cpu = [k if i == len(real_ks_cpu) // 2 else 0 for i, k in enumerate(real_ks_cpu)]
    yield single_group_ks_cpu


def enumerate_sf_layout():
    gran_k_list = (128, ) if get_arch_major() == 9 else (32, 128)
    for use_ue8m0 in (False, True):
        for with_transpose in (True, False):
            for mn in (4096, 4097, 8192):
                for k in (128, 7168, 7296):
                    for num_groups in (1, 2, 4):
                        for gran_k in gran_k_list:
                            set_mk_alignment_for_contiguous_layout(gran_k)
                            yield mn, k, with_transpose, use_ue8m0, num_groups, gran_k

    if get_arch_major() == 10:
        for sf_k, use_ue8m0 in ((908, False), (1211, True)):
            set_mk_alignment_for_contiguous_layout(32)
            yield 4096, sf_k * 32, True, use_ue8m0, 1, 32


def enumerate_k_grouped_sf_layout():
    for mn in (4096, 7168):
        for num_groups, avg_k in ((16, 2048), (8, 4096), (72, 384), (128, 256)):
            for gran_k, k_alignment in ((32, 128), (128, 128), (32, 256), (128, 256), (32, 384), (128, 384)):
                set_mk_alignment_for_contiguous_layout(k_alignment)
                ks_cpu = [align(int(random.uniform(0.7, 1.3) * avg_k), k_alignment)
                          for _ in range(num_groups)]
                yield mn, ks_cpu, num_groups, gran_k, k_alignment


def enumerate_k_grouped_psum_sf_layout():
    for mn, ks_cpu, num_groups, gran_k, k_alignment in enumerate_k_grouped_sf_layout():
        real_ks_cpu = [k - (gran_k // 2 if i % 2 else 0) for i, k in enumerate(ks_cpu)]
        aligned_ks_cpu = [align(k, k_alignment) for k in real_ks_cpu]
        psum_layout = build_psum_layout_from_ks(real_ks_cpu, k_alignment)
        yield mn, real_ks_cpu, aligned_ks_cpu, psum_layout, num_groups, gran_k, k_alignment


def enumerate_transpose():
    for mn in (64, 4096, 16384):
        for delta in (0, 101, 202, 303):
            for k in (128, 1024, 4096, 9984, 16384):
                yield mn + delta, k


def cast_fp8_fp4_with_major(x: torch.Tensor, major: MajorTypeAB, gran_k: int, is_fp4: bool,
                            use_ue8m0: bool, use_block_cast_for_fp8: bool = False):
    if is_fp4:
        x_fp4 = per_token_cast_to_fp4(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
        return x_fp4 if major.is_k_major() else (transpose_packed_fp4(x_fp4[0]).T, x_fp4[1])
    else:
        x_fp8 = per_block_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k) if use_block_cast_for_fp8 \
                else per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
        return x_fp8 if major.is_k_major() else (x_fp8[0].T.contiguous().T, x_fp8[1])


def grouped_cast_fp8_fp4_with_major(x: torch.Tensor, major: MajorTypeAB, gran_k: int, is_fp4: bool,
                                    use_ue8m0: bool, use_block_cast_for_fp8: bool = False):
    num_groups, mn, k = x.size()
    if is_fp4:
        x_fp4 = (torch.empty((num_groups, mn, k // 2), device='cuda', dtype=torch.int8) if major.is_k_major() else \
                 torch.empty((num_groups, k, mn // 2), device='cuda', dtype=torch.int8),
                 torch.empty((num_groups, mn, ceil_div(k, gran_k)), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            x_i_fp4 = per_token_cast_to_fp4(x[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
            x_fp4[0][i], x_fp4[1][i] = x_i_fp4 if major.is_k_major() else (transpose_packed_fp4(x_i_fp4[0]), x_i_fp4[1])
        return x_fp4 if major.is_k_major() else (x_fp4[0].mT, x_fp4[1])
    else:
        x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn),
                 torch.empty((num_groups, ceil_div(mn, gran_k), ceil_div(k, gran_k)), device='cuda', dtype=torch.float) if use_block_cast_for_fp8 \
                 else torch.empty((num_groups, mn, ceil_div(k, gran_k)), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            x_fp8[0][i], x_fp8[1][i] = per_block_cast_to_fp8(x[i], use_ue8m0=use_ue8m0, gran_k=gran_k) if use_block_cast_for_fp8 \
                                       else per_token_cast_to_fp8(x[i], use_ue8m0=use_ue8m0, gran_k=gran_k)
        return x_fp8 if major.is_k_major() else (x_fp8[0].mT.contiguous().mT, x_fp8[1])


def generate_normal(m: int, n: int, k: int,
                    major_a: MajorTypeAB, major_b: MajorTypeAB,
                    accumulate: bool, out_dtype: torch.dtype,
                    kernel_type: KernelType,
                    use_ue8m0: bool = False, use_bf16: bool = False,
                    quant_config: Optional[QuantConfig] = None,
                    alpha: Optional[float] = None):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.randn((m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((m, n), device='cuda', dtype=out_dtype)
    c = d if accumulate else None
    alpha_value = alpha if alpha is not None else 1.0
    ref_d = (alpha_value * (a.float() @ b.float().t()) + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d
    
    quant_config = QuantConfig() if quant_config is None else quant_config
    a = cast_fp8_fp4_with_major(a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    b = cast_fp8_fp4_with_major(b, major_b, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0,
                                use_block_cast_for_fp8=not (kernel_type.is_1d1d() and accumulate))

    return a, b, c, d, ref_d


def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int,
                                  major_a: MajorTypeAB, major_b: MajorTypeAB,
                                  use_ue8m0: bool = False, use_bf16: bool = False,
                                  use_psum_layout: bool = False,
                                  quant_config: Optional[QuantConfig] = None):
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout()) for actual_m in actual_ms]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    grouped_layout = torch.empty(num_groups, device='cuda', dtype=torch.int32) if use_psum_layout \
                     else torch.empty(m, device='cuda', dtype=torch.int32)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)
    valid_mask = torch.zeros(m, device='cuda', dtype=torch.bool)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        if use_psum_layout:
            grouped_layout[i] = actual_end
        else:
            grouped_layout[start: actual_end] = i
            grouped_layout[actual_end: aligned_end] = -1
        valid_mask[start:actual_end] = True
        # Zero BF16 padding so quantized SFA padding is regular, never uninitialized
        a[actual_end: aligned_end] = 0
        ref_d[start: aligned_end] = a[start: aligned_end] @ b[i].t()
        start = aligned_end

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, grouped_layout, d, ref_d, valid_mask

    assert major_a.is_k_major()
    quant_config = QuantConfig() if quant_config is None else quant_config
    a = cast_fp8_fp4_with_major(a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    b = grouped_cast_fp8_fp4_with_major(b, major_b, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)    

    return m, a, b, grouped_layout, d, ref_d, valid_mask


def layout_masked_to_psum(x: torch.Tensor, psum_m: torch.Tensor):
    num_groups, _, k = x.size()
    # PSUM gaps are intentionally left uninitialized to verify the pack kernel skips them
    x_psum = torch.empty((align(psum_m[-1].item(), get_mk_alignment_for_contiguous_layout()), k),
                         dtype=x.dtype, device=x.device)
    last_psum_m = 0
    for i in range(num_groups):
        x_psum[last_psum_m: psum_m[i]] = x[i, :psum_m[i] - last_psum_m]
        last_psum_m = align(psum_m[i], get_mk_alignment_for_contiguous_layout())
    return x_psum


def masked_valid_mask_to_psum(masked_m: torch.Tensor, psum_m: torch.Tensor) -> torch.Tensor:
    # Build the PSUM-layout valid mask directly: gaps must be `False` (never uninitialized),
    # otherwise selecting `d[valid_mask]` would read skipped rows and break determinism checks
    total_m = align(psum_m[-1].item(), get_mk_alignment_for_contiguous_layout())
    valid_mask = torch.zeros(total_m, device=masked_m.device, dtype=torch.bool)
    last_psum_m = 0
    for i in range(masked_m.numel()):
        valid_mask[last_psum_m: psum_m[i]] = True
        last_psum_m = align(psum_m[i], get_mk_alignment_for_contiguous_layout())
    return valid_mask


def generate_m_grouped_masked(num_groups: int, max_m: int, expected_m_per_group: int, n: int, k: int,
                              use_ue8m0: bool = False, use_bf16: bool = False,
                              use_psum_layout: bool = False,
                              quant_config: Optional[QuantConfig] = None):
    a = torch.randn((num_groups, max_m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    psum_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        psum_m[j] = (0 if j == 0 else align(psum_m[j - 1], get_mk_alignment_for_contiguous_layout())) + masked_m[j]
    assert masked_m.amax().item() <= max_m
    valid_mask = torch.arange(max_m, device='cuda')[None, :] < masked_m[:, None]

    if use_bf16:
        if use_psum_layout:
            a = layout_masked_to_psum(a, psum_m)
            d = layout_masked_to_psum(d, psum_m)
            ref_d = layout_masked_to_psum(ref_d, psum_m)
            valid_mask = masked_valid_mask_to_psum(masked_m, psum_m)
        return a, b, psum_m if use_psum_layout else masked_m, d, ref_d, valid_mask

    quant_config = QuantConfig() if quant_config is None else quant_config
    a = grouped_cast_fp8_fp4_with_major(a, MajorTypeAB.KMajor, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    b = grouped_cast_fp8_fp4_with_major(b, MajorTypeAB.KMajor, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)    

    if not use_psum_layout:
        # Zero SFA padding rows (beyond `masked_m`) so the pack kernel reads regular zeros
        for j in range(num_groups):
            a[1][j, masked_m[j].item():] = 0
    else:
        a = (layout_masked_to_psum(a[0], psum_m), layout_masked_to_psum(a[1], psum_m))
        d = layout_masked_to_psum(d, psum_m)
        ref_d = layout_masked_to_psum(ref_d, psum_m)
        valid_mask = masked_valid_mask_to_psum(masked_m, psum_m)

    return a, b, psum_m if use_psum_layout else masked_m, d, ref_d, valid_mask


def k_grouped_cast_fp8_fp4_with_major(x: torch.Tensor, ks_cpu: List[int], major: MajorTypeAB,
                                      use_ue8m0: bool, gran_k: int, is_fp4: bool,
                                      group_ends: Optional[List[int]] = None,
                                      sf_ks_cpu: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    if group_ends is None:
        group_ends = list(itertools.accumulate(ks_cpu))
        assert (group_ends[-1] if ks_cpu else 0) == x.size(0)
    if sf_ks_cpu is None:
        sf_ks_cpu = ks_cpu
    assert len(ks_cpu) == len(group_ends) == len(sf_ks_cpu)

    mn = x.size(1)
    if is_fp4:
        assert major.is_k_major() and gran_k == 32
        data = torch.zeros((mn, x.size(0) // 2), dtype=torch.int8, device=x.device)
        sf_groups = []
        for k, sf_k, end in zip(ks_cpu, sf_ks_cpu, group_ends):
            if sf_k == 0:
                continue
            start = end - k
            x_group = torch.zeros((sf_k, mn), dtype=x.dtype, device=x.device)
            x_group[:k] = x[start:end]
            x_group_fp4, x_group_sf = per_token_cast_to_fp4(
                x_group.T.contiguous(), use_ue8m0=use_ue8m0, gran_k=gran_k)
            data[:, start // 2:(start + sf_k) // 2] = x_group_fp4
            sf_groups.append(x_group_sf.T.contiguous())
        sf = torch.cat(sf_groups) if sf_groups else torch.empty((0, mn), dtype=torch.float, device=x.device)
        return data, sf

    # Cast each group independently. SF rows are compact by group and padded to
    # k_alignment, matching the K-grouped pack kernel's input contract.
    x_fp8 = torch.zeros(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    sf_groups = []
    for k, sf_k, end in zip(ks_cpu, sf_ks_cpu, group_ends):
        if sf_k == 0:
            continue
        start = end - k
        x_group = torch.zeros((sf_k, mn), dtype=x.dtype, device=x.device)
        x_group[:k] = x[start:end]
        x_group_fp8, x_group_sf = per_channel_cast_to_fp8(x_group, use_ue8m0=use_ue8m0, gran_k=gran_k)
        x_fp8[start:end] = x_group_fp8[:k]
        sf_groups.append(x_group_sf)
    sf = torch.cat(sf_groups) if sf_groups else torch.empty((0, mn), dtype=torch.float, device=x.device)
    if major.is_mn_major():
        return x_fp8, sf

    data = torch.zeros((x.size(0) * mn, ), dtype=x_fp8.dtype, device=x.device)
    for k, end in zip(ks_cpu, group_ends):
        start = end - k
        data[start * mn:end * mn] = x_fp8[start:end].T.flatten()
    return data, sf.T


def build_psum_layout_from_ks(real_ks: List[int], k_alignment: int) -> List[int]:
    psum, prev_end = [], 0
    for k in real_ks:
        end = align(prev_end, k_alignment) + k
        psum.append(end)
        prev_end = end
    return psum


def generate_k_grouped_contiguous(num_groups: int, m: int, n: int, major_a: MajorTypeAB, major_b: MajorTypeAB, logical_ks_cpu: List[int],
                                  use_ue8m0: bool = False, use_bf16: bool = False, gran_k = 128,
                                  quant_config: Optional[QuantConfig] = None,
                                  use_psum_layout: bool = False, k_alignment: Optional[int] = None,
                                  accumulate: bool = True, out_dtype: torch.dtype = torch.float):
    assert num_groups == len(logical_ks_cpu)
    k_alignment = get_mk_alignment_for_contiguous_layout() if k_alignment is None else k_alignment
    host_ks_cpu = [align(k, k_alignment) for k in logical_ks_cpu]
    if use_psum_layout:
        assert k_alignment % 32 == 0
        logical_group_ends = build_psum_layout_from_ks(logical_ks_cpu, k_alignment)
        total_k = sum(host_ks_cpu)
        grouped_layout_cpu = logical_group_ends
    else:
        total_k = sum(host_ks_cpu)
        logical_group_ends = []
        physical_start = 0
        for logical_k, host_k in zip(logical_ks_cpu, host_ks_cpu):
            logical_group_ends.append(physical_start + logical_k)
            physical_start += host_k
        grouped_layout_cpu = host_ks_cpu
    grouped_layout = torch.tensor(grouped_layout_cpu, device='cuda', dtype=torch.int32)

    # Physical psum gaps are guaranteed to be zero. A partial block must still
    # stop before the next group's nonzero data.
    a = torch.zeros((total_k, m), device='cuda', dtype=torch.bfloat16)
    b = torch.zeros((total_k, n), device='cuda', dtype=torch.bfloat16)
    d = torch.randn((num_groups, m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((num_groups, m, n), device='cuda', dtype=out_dtype)
    c = d if accumulate else None
    ref_d = torch.empty_like(d)

    for i, (group_k, end) in enumerate(zip(logical_ks_cpu, logical_group_ends)):
        start = end - group_k
        a[start:end] = torch.randn((group_k, m), device='cuda', dtype=torch.bfloat16)
        b[start:end] = torch.randn((group_k, n), device='cuda', dtype=torch.bfloat16)
        ref_d[i] = (a[start:end].float().T @ b[start:end].float() + (c[i] if accumulate else 0)).to(out_dtype)

    if use_bf16:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
        return total_k, a, b, c, d, ref_d, grouped_layout, host_ks_cpu

    is_fp4 = quant_config is not None and (quant_config.is_fp4_a or quant_config.is_fp4_b)
    if is_fp4:
        assert quant_config.is_fp4_a and quant_config.is_fp4_b
        assert (quant_config.gran_k_a, quant_config.gran_k_b) == (32, 32)
        assert (major_a, major_b) == (MajorTypeAB.KMajor, MajorTypeAB.KMajor)
        assert k_alignment % 256 == 0
    else:
        assert k_alignment % 128 == 0

    quantized_a = k_grouped_cast_fp8_fp4_with_major(
        a, logical_ks_cpu if use_psum_layout else host_ks_cpu, major_a, use_ue8m0, gran_k, is_fp4,
        logical_group_ends if use_psum_layout else None, host_ks_cpu)
    quantized_b = k_grouped_cast_fp8_fp4_with_major(
        b, logical_ks_cpu if use_psum_layout else host_ks_cpu, major_b, use_ue8m0, gran_k, is_fp4,
        logical_group_ends if use_psum_layout else None, host_ks_cpu)
    return total_k, quantized_a, quantized_b, c, d, ref_d, grouped_layout, host_ks_cpu
