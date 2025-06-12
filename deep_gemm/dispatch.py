import functools
import torch
from typing import Tuple, Optional

# TODO: add Ampere Triton/tile-lang kernels
from .jit.compiler import get_device_arch
from .jit_kernels.impls import (
    sm90_bf16_gemm,
    sm100_bf16_gemm,
    sm90_fp8_gemm_1d1d,
    sm90_fp8_gemm_1d2d,
    sm100_fp8_gemm_1d1d,
)
from .utils.layout import (
    MajorTypeAB, MajorTypeCD,
    get_major_type_ab, get_major_type_cd,
    transform_sf_into_required_layout
)


@functools.lru_cache(maxsize=None)
def must_be_k_major() -> bool:
    return {
        '90a': True,
        '100a': False,
    }[get_device_arch()]


@functools.lru_cache(maxsize=None)
def get_default_recipe(sfa_dtype: torch.dtype, sfb_dtype: torch.dtype) -> Tuple[int, int, int]:
    assert sfa_dtype in (torch.float, torch.int)
    return {
        ('90a',  torch.float): (1, 128, 128),
        ('100a', torch.float): (1, 128, 128),
        ('100a',   torch.int): (1,   1, 128),
    }[(get_device_arch(), sfb_dtype)]


def fp8_gemm_nt(a: Tuple[torch.Tensor, torch.Tensor],
                b: Tuple[torch.Tensor, torch.Tensor],
                d: torch.Tensor,
                c: Optional[torch.Tensor] = None,
                recipe: Optional[Tuple[int, int, int]] = None,
                compiled_dims: str = 'nk') -> None:
    """
    Perform `d = c + (a @ b)`.
    TODO: add more docs.
    """
    # Compiled dims can be upper cases
    compiled_dims = compiled_dims.lower()

    # NOTES: shape must be `[M, K] @ [N, K].T`
    major_a = get_major_type_ab(a[0])
    major_b = get_major_type_ab(b[0])
    if must_be_k_major():
        assert major_a == major_b == MajorTypeAB.KMajor

    a, sfa = a
    b, sfb = b
    m,  k  = a.shape
    n,  k_ = b.shape
    m_, n_ = d.shape

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    assert d.dtype in (torch.bfloat16, torch.float)

    # D must be N-major
    assert get_major_type_cd(d) == MajorTypeCD.NMajor

    # Check C as well
    if c is not None:
        assert c.dtype == d.dtype == torch.float
        assert get_major_type_cd(c) == MajorTypeCD.NMajor

    # Do nothing if the problem is empty
    if m == 0:
        return

    # Transform SFA and SFB into compute-required layout
    recipe = get_default_recipe(sfa.dtype, sfb.dtype) if recipe is None else recipe
    sfa = transform_sf_into_required_layout(sfa, mn=m, k=k, recipe=recipe, is_sfa=True)
    sfb = transform_sf_into_required_layout(sfb, mn=n, k=k, recipe=recipe, is_sfa=False)

    impl = {
        '100a': functools.partial(sm100_fp8_gemm_1d1d.fp8_gemm_nt,
                                  major_a=major_a, major_b=major_b, major_cd=MajorTypeCD.NMajor,
                                  compiled_dims=compiled_dims)
    }[get_device_arch()]
    impl(a, sfa, b, sfb, c, d)


def m_grouped_fp8_gemm_nt_contiguous(a: Tuple[torch.Tensor, torch.Tensor],
                                     b: Tuple[torch.Tensor, torch.Tensor],
                                     d: torch.Tensor,
                                     m_indices: torch.Tensor,
                                     recipe: Optional[Tuple[int, int, int]] = None,
                                     compiled_dims: str = 'nk') -> None:
    # Compiled dims can be upper cases
    compiled_dims = compiled_dims.lower()

    # NOTES: shape must be `[M, K] @ [G, N, K].mT`
    major_a = get_major_type_ab(a[0])
    major_b = get_major_type_ab(b[0])
    assert major_a == MajorTypeAB.KMajor
    if must_be_k_major():
        assert major_b == MajorTypeAB.KMajor
    assert m_indices.is_contiguous()

    a, sfa = a
    b, sfb = b
    m, k = a.shape
    num_groups, n, k_ = b.shape
    m_, n_ = d.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and n == n_ and k == k_
    assert n > 0 and k > 0 and num_groups > 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    assert d.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32

    # D must be N-major
    assert get_major_type_cd(d) == MajorTypeCD.NMajor

    # Do nothing if the problem is empty
    if m == 0:
        return

    # Transform SFA and SFB into compute-required layout
    recipe = get_default_recipe(sfa.dtype, sfb.dtype) if recipe is None else recipe
    sfa = transform_sf_into_required_layout(sfa, mn=m, k=k, recipe=recipe, is_sfa=True)
    sfb = transform_sf_into_required_layout(sfb, mn=n, k=k, recipe=recipe, num_groups=num_groups, is_sfa=False)

    impl = {
        '100a': functools.partial(sm100_fp8_gemm_1d1d.m_grouped_fp8_gemm_nt_contiguous, major_a=major_a, major_b=major_b, compiled_dims=compiled_dims)
    }[get_device_arch()]
    impl(a, sfa, b, sfb, d, m_indices)


def fp8_m_grouped_gemm_nt_masked(a: Tuple[torch.Tensor, torch.Tensor],
                                 b: Tuple[torch.Tensor, torch.Tensor],
                                 d: torch.Tensor,
                                 masked_m: torch.Tensor,
                                 expected_m: int,
                                 recipe: Optional[Tuple[int, int, int]] = None,
                                 compiled_dims: str = 'nk') -> None:
    # Compiled dims can be upper cases
    compiled_dims = compiled_dims.lower()

    # NOTES: shape must be `[G, M, K] @ [G, N, K].mT`
    major_a = get_major_type_ab(a[0])
    major_b = get_major_type_ab(b[0])
    assert major_a == major_b == MajorTypeAB.KMajor
    assert masked_m.is_contiguous()

    a, sfa = a
    b, sfb = b
    num_groups,   m,  k  = a.shape
    num_groups_,  n,  k_ = b.shape
    num_groups__, m_, n_ = d.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    assert d.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32

    # D must be N-major
    assert get_major_type_cd(d) == MajorTypeCD.NMajor

    # Transform SFA and SFB into compute-required layout
    recipe = get_default_recipe(sfa.dtype, sfb.dtype) if recipe is None else recipe
    sfa = transform_sf_into_required_layout(sfa, mn=m, k=k, recipe=recipe, num_groups=num_groups, is_sfa=True)
    sfb = transform_sf_into_required_layout(sfb, mn=n, k=k, recipe=recipe, num_groups=num_groups, is_sfa=False)

    impl = {
        '100a': functools.partial(sm100_fp8_gemm_1d1d.fp8_m_grouped_gemm_nt_masked, major_a=major_a, major_b=major_b, compiled_dims=compiled_dims)
    }[get_device_arch()]
    impl(a, sfa, b, sfb, d, masked_m, expected_m)
