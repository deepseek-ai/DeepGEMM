import enum
import itertools
import random
import torch
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
        quant_config_list = [QuantConfig()]
        if get_arch_major() == 10:
            quant_config_list.append(QuantConfig((128, 32, False, True)))
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


def enumerate_normal(dtype: torch.dtype) -> Generator:
    assert dtype in (torch.float8_e4m3fn, torch.bfloat16)

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
            if len(quant_config_list) > 1:
                quant_config.print()
            reset_seed()

            # Forward
            for m in m_fwd_list:
                for i in range(len(nk_list)):
                    n, k = nk_list[i]
                    out_dtype = torch.bfloat16 if i < len(bf16_output_nk) else torch.float
                    yield kernel_type, quant_config, m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, out_dtype
                    # BF16 accumulation: supported on all BF16 GEMMs, and SM100 FP8/FP4 GEMMs
                    if out_dtype == torch.bfloat16 and (dtype == torch.bfloat16 or get_arch_major() == 10):
                        yield kernel_type, quant_config, m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, True, out_dtype

            # Backward
            for m in m_bwd_list:
                for n, k in nk_list:
                    override_major = MajorTypeAB.MNMajor
                    override_kernel_type = kernel_type
                    if get_arch_major() == 9 and dtype == torch.float8_e4m3fn:
                        override_major = MajorTypeAB.KMajor
                        override_kernel_type = KernelType.Kernel1D1D
                    yield kernel_type,          quant_config, m, k, n, MajorTypeAB.KMajor, override_major, False, torch.bfloat16     # Dgrad
                    yield override_kernel_type, quant_config, n, m, k, override_major,     override_major, True,  torch.float        # Wgrad
                    yield override_kernel_type, quant_config, n, m, k, override_major,     override_major, False, torch.bfloat16     # Wgrad


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
        k_alignment_options = [(128, 128)] if get_arch_major() == 9 else [(32, 32), (128, 128), (192, 192)]
    else:
        k_alignment_options = [(128, 128)] if get_arch_major() == 9 else [(32, 32), (32, 128), (32, 160), (32, 224), (128, 128), (128, 160), (128, 224)]
    # Only K-major is supported for SM90 FP8
    major_a, major_b = (MajorTypeAB.KMajor, MajorTypeAB.KMajor) if get_arch_major() == 9 and dtype == torch.float8_e4m3fn \
                       else (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
    psum_list = (False, True) if get_arch_major() == 10 else (False, )
    # Must with FP32 accumulation and 1D1D kernels
    for num_groups, m, n, expected_k_per_group in (( 4, 4096, 7168, 8192), ( 4, 7168, 2048, 8192),   # EP64
                                                   ( 8, 768, 2048, 128), ( 8, 4096, 7168, 4096), ( 8, 7168, 2048, 4096),   # EP32
                                                   (16, 4096, 7168, 2048), (16, 7168, 2048, 2048)):  # EP16
        real_ks_cpu = [max(1, int(expected_k_per_group * random.uniform(0.7, 1.3))) for _ in range(num_groups)]
        for use_psum_layout in psum_list:
            for gran_k, k_alignment in k_alignment_options:
                set_mk_alignment_for_contiguous_layout(k_alignment)
                aligned_ks_cpu = [align(k, k_alignment) for k in real_ks_cpu]
                yield num_groups, m, n, major_a, major_b, real_ks_cpu, aligned_ks_cpu, expected_k_per_group, gran_k, k_alignment, use_psum_layout


def enumerate_k_grouped_contiguous_test_variants(real_ks_cpu: List[int], k_alignment: int,
                                                 use_psum_layout: bool, include_k_tail: bool = False):
    test_variants = [(False, False), (True, False)]
    if include_k_tail:
        test_variants.append((False, True))

    for test_empty_groups, test_k_tail in test_variants:
        test_real_ks_cpu = list(real_ks_cpu)
        if test_empty_groups and len(real_ks_cpu) > 1:
            test_real_ks_cpu[random.randint(0, len(real_ks_cpu) - 1)] = 0
        if test_k_tail and len(test_real_ks_cpu) > 1:
            test_real_ks_cpu[0] = k_alignment
        elif use_psum_layout and not test_empty_groups and len(test_real_ks_cpu) > 1 and test_real_ks_cpu[0] > 1:
            test_real_ks_cpu[0] -= random.randint(1, min(k_alignment - 1, test_real_ks_cpu[0] - 1))
        test_aligned_ks_cpu = [align(k, k_alignment) for k in test_real_ks_cpu]
        yield test_real_ks_cpu, test_aligned_ks_cpu, test_empty_groups, test_k_tail


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


def enumerate_k_grouped_sf_layout():
    gran_k_list = (128, ) if get_arch_major() == 9 else (32, 128)
    for mn in (4096, 7168):
        for num_groups, avg_k in ((16, 2048), (8, 4096), (72, 384), (128, 256)):
            for gran_k in gran_k_list:
                set_mk_alignment_for_contiguous_layout(gran_k)
                ks_cpu = [align(int(random.uniform(0.7, 1.3) * avg_k), gran_k) for _ in range(num_groups)]
                yield mn, ks_cpu, num_groups, gran_k


def enumerate_k_grouped_psum_sf_layout():
    for mn, ks_cpu, num_groups, gran_k in enumerate_k_grouped_sf_layout():
        k_alignment_list = (32, 128, 160, 224)
        for k_alignment in k_alignment_list:
            # Generate non-aligned K sizes to test ceil_div path in pack kernel.
            real_ks_cpu = [k + (gran_k // 2 if i % 2 else 0) for i, k in enumerate(ks_cpu)]
            psum_layout = build_psum_layout_from_ks(real_ks_cpu, k_alignment)
            aligned_ks_cpu = [align(k, k_alignment) for k in real_ks_cpu]
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
                    quant_config: Optional[QuantConfig] = None):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.randn((m, n), device='cuda', dtype=out_dtype) * 32 if accumulate else \
        torch.empty((m, n), device='cuda', dtype=out_dtype)
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

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

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        if use_psum_layout:
            grouped_layout[i] = actual_end
        else:
            grouped_layout[start: actual_end] = i
            grouped_layout[actual_end: aligned_end] = -1
        # Zero BF16 padding so quantized SFA padding is regular, never uninitialized
        a[actual_end: aligned_end] = 0
        ref_d[start: aligned_end] = a[start: aligned_end] @ b[i].t()
        start = aligned_end

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, grouped_layout, d, ref_d

    assert major_a.is_k_major()
    quant_config = QuantConfig() if quant_config is None else quant_config
    a = cast_fp8_fp4_with_major(a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    b = grouped_cast_fp8_fp4_with_major(b, major_b, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)    

    return m, a, b, grouped_layout, d, ref_d


def layout_masked_to_psum(x: torch.Tensor, psum_m: torch.Tensor):
    num_groups, max_m, _ = x.size()
    # PSUM gaps are intentionally left uninitialized to verify the pack kernel skips them
    x_psum = torch.empty_like(x).view(num_groups * max_m, -1)
    last_psum_m = 0
    for i in range(num_groups):
        x_psum[last_psum_m: psum_m[i]] = x[i, :psum_m[i] - last_psum_m]
        last_psum_m = align(psum_m[i], get_mk_alignment_for_contiguous_layout())
    return x_psum


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

    if use_bf16:
        return a, b, masked_m, psum_m, d, ref_d

    quant_config = QuantConfig() if quant_config is None else quant_config
    a = grouped_cast_fp8_fp4_with_major(a, MajorTypeAB.KMajor, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
    b = grouped_cast_fp8_fp4_with_major(b, MajorTypeAB.KMajor, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)    

    if not use_psum_layout:
        # Zero SFA padding rows (beyond `masked_m`) so the pack kernel reads regular zeros
        for j in range(num_groups):
            a[1][j, masked_m[j].item():] = 0

    return a, b, masked_m, psum_m, d, ref_d


def k_grouped_per_channel_cast_to_fp8(x: torch.Tensor, ks_cpu: List[int], use_ue8m0: bool, gran_k: int,
                                      group_ends: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # Cast each group independently so that the SF rows stay compact (`ceil_div(k, gran_k)` rows per group);
    # `group_ends` gives per-group end offsets (psum layouts), `None` means a contiguous prefix-sum layout
    assert x.dim() == 2
    if group_ends is None:
        group_ends = list(itertools.accumulate(ks_cpu))
        assert (group_ends[-1] if ks_cpu else 0) == x.size(0)

    n = x.size(1)
    x_fp8 = torch.zeros(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    sf_groups = []
    for k, end in zip(ks_cpu, group_ends):
        if k == 0:
            continue
        start = end - k
        x_group = torch.zeros((align(k, gran_k), n), dtype=x.dtype, device=x.device)
        x_group[:k] = x[start:end]
        x_group_fp8, x_group_sf = per_channel_cast_to_fp8(x_group, use_ue8m0=use_ue8m0, gran_k=gran_k)
        x_fp8[start:end] = x_group_fp8[:k]
        sf_groups.append(x_group_sf)
    sf = torch.cat(sf_groups) if sf_groups else torch.empty((0, n), dtype=torch.float, device=x.device)
    return x_fp8, sf


def generate_k_grouped_contiguous(num_groups: int, m: int, n: int, major_a: MajorTypeAB, major_b: MajorTypeAB, ks_cpu: List[int],
                                  use_ue8m0: bool = False, use_bf16: bool = False, gran_k = 128):
    k = sum(ks_cpu)
    grouped_layout = torch.tensor(ks_cpu, device='cuda', dtype=torch.int32)

    a = torch.randn((k, m), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
    c = torch.randn((num_groups, m, n), device='cuda', dtype=torch.float) * 32
    d = c
    ref_d = torch.empty_like(c)

    start = 0
    for i, group_k in enumerate(ks_cpu):
        end = start + group_k
        ref_d[i] = c[i] + (a[start:end].T @ b[start:end])
        start = end

    if use_bf16:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
        return k, a, b, c, d, ref_d, grouped_layout, ks_cpu

    assert get_mk_alignment_for_contiguous_layout() % 32 == 0

    a_fp8 = k_grouped_per_channel_cast_to_fp8(a, ks_cpu, use_ue8m0=use_ue8m0, gran_k=gran_k)
    b_fp8 = k_grouped_per_channel_cast_to_fp8(b, ks_cpu, use_ue8m0=use_ue8m0, gran_k=gran_k)

    # Transpose for K Major A/B
    if (major_a, major_b) == (MajorTypeAB.KMajor, MajorTypeAB.KMajor):
        a, sfa = a_fp8
        b, sfb = b_fp8
        new_a = torch.empty((sum(ks_cpu) * m, ), dtype=a.dtype, device=a.device)
        new_b = torch.empty((sum(ks_cpu) * n, ), dtype=b.dtype, device=b.device)
        prefix = 0
        for K in ks_cpu:
            new_a[prefix * m : (prefix + K) * m] = a[prefix : prefix + K, ].T.flatten()
            new_b[prefix * n : (prefix + K) * n] = b[prefix : prefix + K, ].T.flatten()
            prefix += K
        a_fp8, b_fp8 = (new_a, sfa.T), (new_b, sfb.T)
    else:
        assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)

    return k, a_fp8, b_fp8, c, d, ref_d, grouped_layout, ks_cpu


def build_psum_layout_from_ks(real_ks: List[int], k_alignment: int) -> List[int]:
    # Convert raw per-group K sizes to psum end offsets
    psum, prev_end = [], 0
    for k in real_ks:
        end = align(prev_end, k_alignment) + k
        psum.append(end)
        prev_end = end
    return psum


def generate_k_grouped_contiguous_psum(num_groups: int, m: int, n: int,
                                        major_a: MajorTypeAB, major_b: MajorTypeAB,
                                        real_ks: List[int], k_alignment: int,
                                        use_ue8m0: bool = False, use_bf16: bool = False,
                                        gran_k: int = 128):
    # psum k-grouped is SM100 MN-major only
    assert (major_a, major_b) == (MajorTypeAB.MNMajor, MajorTypeAB.MNMajor)
    assert k_alignment % 32 == 0

    # NOTES: `aligned_ks` round each group's K up to `k_alignment` (the group-start alignment),
    # so the host-side `k % k_alignment == 0` check passes and matches the psum layout below.
    aligned_ks = [align(k, k_alignment) for k in real_ks]

    grouped_layout = build_psum_layout_from_ks(real_ks, k_alignment)
    total_k = align(grouped_layout[-1] if num_groups > 0 else 0, k_alignment)

    # Keep padded K gaps zeroed
    a = torch.zeros((total_k, m), device='cuda', dtype=torch.bfloat16)
    b = torch.zeros((total_k, n), device='cuda', dtype=torch.bfloat16)
    c = torch.randn((num_groups, m, n), device='cuda', dtype=torch.float) * 32
    d = c
    ref_d = torch.empty_like(c)

    for i, k in enumerate(real_ks):
        if k == 0:
            ref_d[i] = c[i]
            continue
        start = grouped_layout[i] - k
        a_g = torch.randn((k, m), device='cuda', dtype=torch.bfloat16)
        b_g = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
        a[start:start + k] = a_g
        b[start:start + k] = b_g
        ref_d[i] = c[i] + (a_g.T @ b_g)

    grouped_layout_tensor = torch.tensor(grouped_layout, device='cuda', dtype=torch.int32)
    if use_bf16:
        return total_k, a, b, c, d, ref_d, grouped_layout_tensor, aligned_ks

    a_fp8 = k_grouped_per_channel_cast_to_fp8(a, real_ks, use_ue8m0=use_ue8m0, gran_k=gran_k, group_ends=grouped_layout)
    b_fp8 = k_grouped_per_channel_cast_to_fp8(b, real_ks, use_ue8m0=use_ue8m0, gran_k=gran_k, group_ends=grouped_layout)
    return total_k, a_fp8, b_fp8, c, d, ref_d, grouped_layout_tensor, aligned_ks
