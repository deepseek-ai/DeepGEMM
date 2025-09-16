import torch
from typing import Iterable

def check_signal(num_local_expert, max_m, block_m, threshold, signal, masked_m):
    ceil_div = lambda a, b: (a + b - 1) // b

    expert_len = max_m // block_m
    for expert in range(num_local_expert):
        mask = masked_m[expert]
        start = expert * expert_len
        end = expert * expert_len + expert_len
        valid_len = ceil_div(mask, block_m)
        for i in range(start, end):
            if i < start + valid_len:
                assert signal[i] == threshold, f'{i=}, {signal[i]=}, {threshold=}'
            else:
                assert signal[i] == 0, f'{i=}, {signal[i]=}'

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
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
