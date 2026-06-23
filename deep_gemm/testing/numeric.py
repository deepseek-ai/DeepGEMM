import torch
from typing import Iterable


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def assert_bitwise_equal(x: torch.Tensor, y: torch.Tensor, label: str = ''):
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    x_bytes = x.contiguous().view(torch.uint8)
    y_bytes = y.contiguous().view(torch.uint8)
    if torch.equal(x_bytes, y_bytes):
        return

    mismatch = x_bytes != y_bytes
    mismatch_idx = mismatch.flatten().nonzero()[0].item()
    elem_idx = mismatch_idx // x.element_size()
    byte_in_elem = mismatch_idx % x.element_size()
    coord = tuple(torch.unravel_index(torch.tensor(elem_idx, device=x.device), x.shape))
    coord = tuple(v.item() for v in coord)
    raise AssertionError(
        f'bitwise mismatch{f" ({label})" if label else ""}: '
        f'num_bytes={mismatch.numel()}, num_mismatch={mismatch.sum().item()}, '
        f'first_byte={mismatch_idx}, elem={elem_idx}, coord={coord}, byte_in_elem={byte_in_elem}, '
        f'x_byte={x_bytes.flatten()[mismatch_idx].item()}, y_byte={y_bytes.flatten()[mismatch_idx].item()}, '
        f'x_val={x.flatten()[elem_idx].item()}, y_val={y.flatten()[elem_idx].item()}'
    )
