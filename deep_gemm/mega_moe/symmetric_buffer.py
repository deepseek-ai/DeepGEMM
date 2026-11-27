import types
from typing import Optional

import torch
import torch.distributed as dist

try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
except ImportError:
    symm_mem = None


class SymmetricBuffer:
    """Shared allocation, peer mapping and lifetime for symmetric buffers."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_bytes: int,
        base: Optional['SymmetricBuffer'] = None
    ) -> None:
        assert num_bytes > 0, 'Symmetric buffer size must be positive'
        self.group = group
        if base is not None:
            # Reuse aliases the allocation; callers must finish earlier uses before reusing it.
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The requested symmetric buffer requires {num_bytes} bytes, '
                 f'but the base buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle
            return

        if group.size() == 1:
            self.buffer = torch.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
        else:
            assert symm_mem is not None, 'This PyTorch build does not support symmetric memory'
            self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        group.barrier()
        torch.cuda.synchronize()

    def destroy(self) -> None:
        """Release this owner's references; other buffers sharing the allocation remain valid."""
        self.handle = None
        self.buffer = None
        self.group = None
