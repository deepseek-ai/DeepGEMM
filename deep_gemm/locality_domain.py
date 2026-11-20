import torch

from ._C.locality_domain import (
    get_num_locality_domains,
    get_granularity,
    is_localization_available,
    empty,
    empty_per_domain,
    is_localized,
    get_sm_locality_domains,
    get_balanced_sm_locality_domains,
    release_mlopart,
)


def localize(t: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """
    Distribute the weights across locality domains: `(*B, N, K)` -> `(num_domains, *B, N / num_domains, K)`.
    """
    num_domains, dim = get_num_locality_domains(), dim % t.dim()
    assert t.size(dim) % num_domains == 0, f'Weights `{tuple(t.shape)}` must slice into {num_domains} parts along dim {dim}'
    localized = empty_per_domain((*t.shape[:dim], t.size(dim) // num_domains, *t.shape[dim + 1:]), t.dtype)
    localized.copy_(t.unflatten(dim, (num_domains, -1)).movedim(dim, 0))
    return localized


def destroy_localizer() -> None:
    """
    Quit the allocator backed by MLOPart. The allocated tensors stay valid.
    """
    release_mlopart()
