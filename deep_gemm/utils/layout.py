import enum
import torch
from typing import Tuple, Optional

from .math import align, ceil_div
from ..jit.compiler import get_device_arch


class GemmType(enum.Enum):
    Normal = 0
    GroupedContiguous = 1
    GroupedMasked = 2

    def __str__(self) -> str:
        return {
            0: 'GemmType::Normal',
            1: 'GemmType::GroupedContiguous',
            2: 'GemmType::GroupedMasked',
        }[self.value]


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def shape_direction(self):
        return 1 if self.value == 0 else -1

    def non_contiguous_dim(self):
        return -2 if self.value == 0 else -1

    def __str__(self) -> str:
        return {
            0: 'cute::UMMA::Major::K',
            1: 'cute::UMMA::Major::MN'
        }[self.value]


class MajorTypeCD(enum.Enum):
    NMajor = 0
    MMajor = 1

    def non_contiguous_dim(self):
        return -2 if self.value == 0 else -1


def major_check(t: torch.Tensor):
    assert t.dim() in (2, 3)
    if t.dim() == 3:
        assert t.stride(0) == t.size(-2) * t.size(-1), 'Grouped dimension cannot have abnormal stride'
    assert t.stride(-2) == 1 or t.stride(-1) == 1


def get_major_type_ab(t: torch.Tensor):
    major_check(t)
    return MajorTypeAB.KMajor if t.stride(-1) == 1 else MajorTypeAB.MNMajor


def get_major_type_cd(t: torch.Tensor):
    major_check(t)
    return MajorTypeCD.NMajor if t.stride(-1) == 1 else MajorTypeCD.MMajor


def get_element_size(dtype: torch.dtype):
    return {
        torch.float8_e4m3fn: 1,
        torch.bfloat16:      2,
        torch.float:         4,
    }[dtype]


def get_m_alignment_for_contiguous_layout():
    return 128


def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return align(x, alignment)


def get_col_major_tma_aligned_packed_tensor(x: torch.Tensor) -> torch.Tensor:
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)

    # Finally, transpose
    transposed = torch.transpose(torch.empty((b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int), 1, 2)
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def check_sf_layout(sf: torch.Tensor,
                    mn: int, k: int, gran: Tuple[int, int],
                    num_groups: Optional[int],
                    tma_stride_check: bool = False,
                    type_check: Optional[torch.dtype] = None) -> torch.Tensor:
    # Type check
    if type_check is not None:
        assert sf.dtype == type_check

    # Always do shape checks
    assert sf.dtype in (torch.float, torch.int)
    assert sf.dim() == int(num_groups is not None) + 2
    if num_groups is not None:
        assert sf.size(-3) == num_groups
    assert sf.size(-2) == ceil_div(mn, gran[0])
    assert sf.size(-1) == ceil_div(k, gran[1] * (1 if sf.dtype == torch.float else 4))

    # TMA stride checks: TMA aligned and MN-major
    if tma_stride_check:
        if num_groups is not None:
            assert sf.stride(-3) == sf.stride(-1) * sf.size(-1)
        assert sf.stride(-2) == 1
        assert sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size())

    return sf


def transform_sf_into_required_layout(sf: torch.Tensor,
                                      mn: int, k: int,
                                      recipe: Tuple[int, int, int],
                                      num_groups: Optional[int] = None,
                                      is_sfa: bool = False):
    gran = (recipe[0 if is_sfa else 1], recipe[2])

    should_skip_transform = (
        (sf.dtype == torch.int and gran == (1, 128) and get_device_arch() == '100a')
        or (sf.dtype == torch.int and gran == (128, 128) and get_device_arch() == '100a')
    )

    if not should_skip_transform:
        # Pre-transform checks
        check_sf_layout(sf, mn=mn, k=k, gran=gran, num_groups=num_groups)

    # (FP32, 1, 128) on Hopper: transform to TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (1, 128) and get_device_arch() == '90a':
        raise NotImplemented

    # (FP32, 1, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (1, 128) and get_device_arch() == '100a':
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(sf, mn=mn, k=k, gran=(1, 128), num_groups=num_groups, tma_stride_check=True, type_check=torch.int)

    # (FP32, 128, 128) on Hopper: no need to transform, check shape and whatever-major
    if sf.dtype == torch.float and gran == (128, 128) and get_device_arch() == '90a':
        raise NotImplemented

    # (FP32, 128, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (128, 128) and get_device_arch() == '100a':
        sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(sf, mn=mn, k=k, gran=(1, 128), num_groups=num_groups, tma_stride_check=True, type_check=torch.int)

    if should_skip_transform:
        # TODO: add transpose kernel if SF layout is not satisfied
        return check_sf_layout(sf, mn=mn, k=k, gran=(1, 128), num_groups=num_groups, tma_stride_check=True, type_check=torch.int)

    assert False, f'Unknown cases: {sf.dtype=}, {gran=}, arch={get_device_arch()}'
