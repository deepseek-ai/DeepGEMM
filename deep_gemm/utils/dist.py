import inspect
import os
import torch
import torch.distributed as dist
from typing import Tuple

_local_rank = None


def init_dist(local_rank: int, num_local_ranks: int) -> Tuple[int, int, dist.ProcessGroup]:
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    # Set local rank
    global _local_rank
    _local_rank = local_rank

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def uneven_all_gather(tensor: torch.Tensor, dim: int = 0, group: dist.ProcessGroup = None) -> torch.Tensor:
    world_size = dist.get_world_size(group)

    # Exchange sizes
    local_dim_size = torch.tensor([tensor.shape[dim]], device=tensor.device, dtype=torch.long)
    all_dim_sizes = [torch.zeros_like(local_dim_size) for _ in range(world_size)]
    dist.all_gather(all_dim_sizes, local_dim_size, group=group)
    all_dim_sizes = [s.item() for s in all_dim_sizes]
    max_dim_size = max(all_dim_sizes)

    # Pad
    if tensor.shape[dim] < max_dim_size:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_dim_size - tensor.shape[dim]
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat([tensor, padding], dim=dim)
    else:
        tensor_padded = tensor.contiguous()

    # All-gather
    gathered = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered, tensor_padded, group=group)

    # Remove padding
    trimmed = [
        torch.narrow(gathered[i], dim, 0, all_dim_sizes[i])
        for i in range(world_size)
    ]
    return torch.cat(trimmed, dim=dim)


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    global _local_rank
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    dist.barrier()
