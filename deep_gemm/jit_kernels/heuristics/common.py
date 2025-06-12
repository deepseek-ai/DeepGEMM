from ...jit.compiler import get_device_arch
from ...utils.math import ceil_div


class MulticastConfig:
    def __init__(self, num_multicast: int, is_multicast_on_a: bool):
        self.num_multicast = num_multicast
        self.is_multicast_on_a = is_multicast_on_a

    def get_ab_load_block_m(self, block_m: int):
        # NOTES: this for >= SM100 only
        assert get_device_arch() != '90a'
        return block_m // (self.num_multicast if self.is_multicast_on_a else 1)

    def get_ab_load_block_n(self, block_n: int):
        # NOTES: this for >= SM100 only
        assert get_device_arch() != '90a'
        return block_n // (1 if self.is_multicast_on_a else self.num_multicast)


class SharedMemoryConfig:
    def __init__(self, smem_size: int, swizzle_a_mode: int, swizzle_b_mode: int, swizzle_cd_mode: int):
        self.smem_size = smem_size
        self.swizzle_a_mode = swizzle_a_mode
        self.swizzle_b_mode = swizzle_b_mode
        # NOTES: sometimes the default swizzling pattern maybe not compatible (e.g., FP32 output)
        self.swizzle_cd_mode = swizzle_cd_mode
        # TODO: swizzle SF as well
        self.swizzle_sf_mode = 0

        assert self.swizzle_a_mode != 0
        assert self.swizzle_b_mode != 0
        assert self.swizzle_cd_mode > 16
        assert self.swizzle_sf_mode == 0


def is_multicast_legal(shape_dim: int, block_dim: int, num_multicast: int, num_sms: int,
                       require_divisible: bool = False) -> bool:
    divisible = ceil_div(shape_dim, block_dim) % num_multicast == 0 or not require_divisible
    return divisible and num_sms % num_multicast == 0


def get_swizzle_mode(block_size: int, elem_size: int) -> int:
    # `> 0` means interleaving
    # 16B actually means non-swizzling (but interleaving)
    for mode_bytes in (128, 64, 32, 16):
        if (block_size * elem_size) % mode_bytes == 0:
            return mode_bytes
    assert False, 'Invalid mode'
