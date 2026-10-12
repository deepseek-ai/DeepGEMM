import torch
import os
import functools
from enum import Enum
import copy

_num_sms = None

class GemmType(Enum):
    """Matches C++ enum GemmType in utils_rtc.cuh"""
    DenseGemm = 0
    GroupedContiguous = 1
    GroupedMasked = 2
    GroupedNoPad = 3
    GroupedFused = 4
    BatchGemm = 5

class CompuleMode(Enum):
    COMPILE_AND_RUN = 0
    #ONLY_COMPILE must be 1 to align with sglang deepgemm usage
    ONLY_COMPILE = 1

compile_mode = CompuleMode.COMPILE_AND_RUN

def set_compile_mode(mode):
    global compile_mode
    compile_mode = mode

def get_compile_mode():
    global compile_mode
    return compile_mode

def set_num_sms(num_sms: int) -> None:
    """
    Set the maximum SM count for all GEMM kernels to use.

    Arguments:
        num_sms: the desired maximum SM count for all GEMM kernels to use.
    """
    global _num_sms
    assert 0 < num_sms <= torch.cuda.get_device_properties(device='cuda').multi_processor_count
    _num_sms = num_sms


def get_num_sms() -> int:
    """
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    """
    global _num_sms
    if _num_sms is None:
        device_props = torch.cuda.get_device_properties(device='cuda')
        print(f'device_props.name:{device_props.name}')
        if "ZW810E" in device_props.name or "ZW610E" in device_props.name:
            _num_sms = 20
        else:
            _num_sms = device_props.multi_processor_count
    return _num_sms


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def get_m_alignment_for_contiguous_layout():
    """
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.

    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    """
    return 128

def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in a device kernel
    assert x.dim() in (2, 3)
    remove_dim = False
    m, n = x.shape[-2], x.shape[-1]
    # aligned_m = get_tma_aligned_size(m, x.element_size())
    aligned_m = m
    if x.dim() == 2:
        if x.stride(0) == 1 and x.stride(1) == aligned_m:
            return x
        x, remove_dim = x.unsqueeze(0), True

    b = x.shape[0]

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2)
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x

def get_col_major_tensor(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() in (2, 3), "Only 2-D or 3-D tensors supported"

    squeeze_dim = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_dim = True

    B, M, N = x.shape

    col_major = torch.empty((B, N, M), dtype=x.dtype, device=x.device)
    col_major = col_major.transpose(-2, -1)

    col_major[...] = x

    if squeeze_dim:
        col_major = col_major.squeeze(0)

    return col_major


def get_case_id() -> int:
    if not hasattr(get_case_id, 'case_id'):
        get_case_id.case_id = 0
    get_case_id.case_id += 1
    return get_case_id.case_id


@functools.lru_cache(maxsize=None)
def is_ppu1v5_device():
    device_prop = torch.cuda.get_device_properties(device='cuda')
    if device_prop.major == 8 and device_prop.minor == 9:
        return True
    else:
        return False

@functools.lru_cache(maxsize=None)
def get_sm_count():
    device_prop = torch.cuda.get_device_properties(device='cuda')
    return device_prop.multi_processor_count

@functools.lru_cache(maxsize=None)
def get_extra_info(m=0, n=0, k=0, dtype=torch.int8, api_type="dense") -> dict:
    extra_info = {}
    use_multistage_on_N = False
    use_moe_dynamic_tile = False

    if 'DG_USE_MULTISTAGE_ON_N' in os.environ:
        use_multistage_on_N = int(os.getenv('DG_USE_MULTISTAGE_ON_N'))
    extra_info['use_multistage_on_N'] = use_multistage_on_N

    if 'DG_USE_MOE_DYNAMIC_TILE' in os.environ:
        use_moe_dynamic_tile = int(os.getenv('DG_USE_MOE_DYNAMIC_TILE'))
    extra_info['use_moe_dynamic_tile'] = use_moe_dynamic_tile
    return extra_info

def get_search_space(d: torch.dtype, gemm_type : str, m:int=0, n:int=0, k:int=0) -> list:
    """
    Returns search space according input gemm type

    Arguments:
        gemm_type: nopad, masked, dense

    Returns:
        The tile list:{block_m, block_n, warp_m, warp_n, stage}
    """
    assert gemm_type in ('nopad', 'masked', 'dense','contiguous')

    block_k = 64 if d == torch.bfloat16 else 128
    tile_list = [
        # blockM = 16
        [16, 64, 16, 16, block_k, 2],
        [16, 64, 16, 16, block_k, 3],
        [16, 64, 16, 16, block_k, 4],
        [16, 64, 16, 16, block_k, 5],
        [16, 64, 16, 16, block_k * 2, 2],
        [16, 64, 16, 16, block_k * 2, 3],
        [16, 64, 16, 16, block_k * 2, 4],
        [16, 64, 16, 16, block_k * 2, 5],

        [16, 128, 16, 32, block_k, 2],
        [16, 128, 16, 32, block_k, 3],
        [16, 128, 16, 32, block_k, 4],
        [16, 128, 16, 32, block_k * 2, 2],
        [16, 128, 16, 32, block_k * 2, 3],
        [16, 128, 16, 32, block_k * 2, 4],
        [16, 128, 16, 16, block_k, 2],
        [16, 128, 16, 16, block_k, 3],
        [16, 128, 16, 16, block_k, 4],
        [16, 128, 16, 16, block_k * 2, 2],
        [16, 128, 16, 16, block_k * 2, 3],
        [16, 128, 16, 16, block_k * 2, 4],

        [16, 256, 16, 64, block_k, 2],
        [16, 256, 16, 64, block_k, 3],
        [16, 256, 16, 64, block_k * 2, 2],
        [16, 256, 16, 64, block_k * 2, 3],


        # blockM = 32
        [32, 64, 16, 32, block_k, 2],
        [32, 64, 16, 32, block_k, 3],
        [32, 64, 16, 32, block_k, 4],
        [32, 64, 16, 32, block_k, 5],
        [32, 64, 16, 32, block_k * 2, 2],
        [32, 64, 16, 32, block_k * 2, 3],
        [32, 64, 16, 32, block_k * 2, 4],
        [32, 64, 16, 16, block_k, 2],
        [32, 64, 16, 16, block_k, 3],
        [32, 64, 16, 16, block_k * 2, 2],
        [32, 64, 16, 16, block_k * 2, 3],

        [32, 128, 16, 64, block_k, 2],
        [32, 128, 16, 64, block_k, 3],
        [32, 128, 16, 64, block_k, 4],
        [32, 128, 16, 64, block_k * 2, 2],
        [32, 128, 16, 64, block_k * 2, 3],
        [32, 128, 16, 32, block_k, 2],
        [32, 128, 16, 32, block_k, 3],
        [32, 128, 16, 32, block_k * 2, 2],
        [32, 128, 16, 32, block_k * 2, 3],

        [32, 256, 16, 64, block_k, 2],
        [32, 256, 16, 64, block_k, 3],
        [32, 256, 16, 64, block_k * 2, 2],
        [32, 256, 16, 64, block_k * 2, 3],


        # blockM = 48
        [48, 64, 16, 16, block_k, 2],
        [48, 64, 16, 16, block_k * 2, 2],
        [48, 64, 16, 16, block_k, 3],
        [48, 64, 16, 16, block_k * 2, 3],
        [48, 64, 16, 32, block_k, 2],
        [48, 64, 16, 32, block_k * 2, 2],
        [48, 64, 16, 32, block_k, 3],
        [48, 64, 16, 32, block_k * 2, 3],

        [48, 128, 48, 32, block_k, 2],
        [48, 128, 48, 32, block_k * 2, 2],
        [48, 128, 48, 32, block_k, 3],
        [48, 128, 48, 32, block_k * 2, 3],


        # blockM = 64
        [64, 64, 32, 32, block_k, 2],
        [64, 64, 32, 32, block_k, 3],
        [64, 64, 32, 32, block_k, 4],
        [64, 64, 32, 32, block_k * 2, 2],
        [64, 64, 32, 32, block_k * 2, 3],
        [64, 64, 32, 32, block_k * 2, 4],

        [64, 128, 32, 64, block_k, 2],
        [64, 128, 32, 64, block_k, 3],
        [64, 128, 32, 32, block_k, 2],
        [64, 128, 32, 32, block_k, 3],
        [64, 128, 32, 64, block_k * 2, 2],
        [64, 128, 32, 64, block_k * 2, 3],
        [64, 128, 32, 32, block_k * 2, 2],
        [64, 128, 32, 32, block_k * 2, 3],

        [64, 256, 32, 64, block_k, 2],
        [64, 256, 32, 64, block_k, 3],
        [64, 256, 32, 64, block_k * 2, 2],
        [64, 256, 32, 64, block_k * 2, 3]
    ]

    if 'dense' in gemm_type:
        tile_list.extend([
            # blockM = 128
            [128, 128, 64, 64, block_k    , 2],
            [128, 128, 64, 64, block_k    , 3],
            [128, 128, 64, 64, block_k    , 4],
            [128, 128, 64, 64, block_k * 2, 3],
            [128, 256, 64, 64, block_k    , 2],
            [128, 256, 64, 64, block_k    , 3],

            # blockM = 160
            [160, 128, 80, 64, block_k    , 2],
            [160, 128, 80, 64, block_k    , 3],
            [160, 128, 80, 64, block_k    , 4],
            [160, 128, 80, 64, block_k * 2, 3],
            [160, 256, 80, 64, block_k    , 2],
            [160, 256, 80, 64, block_k    , 4],

            # blockM = 192
            [192, 128, 48, 64, block_k    , 2],
            [192, 128, 48, 64, block_k    , 3],
            [192, 128, 48, 64, block_k    , 4],
            [192, 128, 48, 64, block_k * 2, 3],
            [192, 256, 48, 64, block_k    , 2],
            [192, 256, 48, 64, block_k    , 4],

            # blockM = 256
            [256, 64, 32, 64, block_k,      2],
            [256, 64, 32, 64, block_k,      3],
            [256, 64, 32, 64, block_k * 2,  2],
            [256, 64, 32, 64, block_k * 2,  3],
            [256, 128, 64, 64, block_k    , 2],
            [256, 128, 64, 64, block_k    , 3],
            [256, 128, 64, 64, block_k    , 4],
            [256, 256, 64, 64, block_k,     4],

            # blockM = 320
            [320, 128, 80, 64, block_k    , 2],
            [320, 128, 80, 64, block_k    , 3],
            [320, 128, 80, 64, block_k    , 4],
            [320, 256, 80, 64, block_k    , 2],
            [320, 256, 80, 64, block_k    , 3],

            # blockK = 64B
            [128, 128, 64, 64, int(block_k / 2), 2],
            [128, 256, 64, 64, int(block_k / 2), 2],
            [256, 128, 64, 64, int(block_k / 2), 2],
        ])

    tile_list_rtn = []
    if d == torch.float8_e4m3fn:
        for tile in tile_list:
            if tile[0] in (16, 32, 64, 128, 192) and tile[4] == block_k:
                tile_list_rtn.append(tile)
                # if tile[4] == block_k and tile[0] != 192:
                #     tile_copy = copy.deepcopy(tile)
                #     tile_copy[4] = int(block_k / 2)
                #     tile_list_rtn.append(tile_copy)
        return tile_list_rtn

    # add block_k / 2 tile for k <256
    if k != 0 and k <= 512:
        for tile in tile_list:
            tile_list_rtn.append(tile)
            if tile[4] == block_k:
                tile_copy = copy.deepcopy(tile)
                tile_copy[4] = int(block_k / 2)
                tile_list_rtn.append(tile_copy)
    else:
        tile_list_rtn = tile_list

    return tile_list_rtn

@functools.lru_cache(maxsize=4096)
def get_paged_mqa_logits_tile(next_n, block_kv, num_heads, head_dim, datasize):
    assert block_kv == 64
    is_fp4 = (datasize == 1 and head_dim == 64)
    weights_size, logits_size = 4, 4
    def calc_smem(split_kv, stage_q, stage_k):
        block_m = split_kv
        block_n = next_n * num_heads
        block_k = head_dim
        smem_q = block_n * block_k * stage_q * datasize
        smem_k = block_m * block_k * stage_k * datasize
        smem_k_scale = block_m * stage_k * 4
        smem_q_scale = block_n * stage_q * 4 if is_fp4 else 0
        smem_weight = block_n * stage_q * weights_size
        smem = smem_q + smem_k + smem_k_scale + smem_q_scale + smem_weight
        return smem

    def calc_vreg(warp_kv):
        k_block_max = head_dim * datasize // 32
        mma_n = num_heads // 16
        mma_m = warp_kv // 16

        vreg_base = 18
        vreg_weights = 4 * mma_n * logits_size // 4
        vreg_acc = 8 * mma_m * mma_n
        vreg_A = 4 * mma_m * k_block_max
        vreg_B = 4 * mma_n * k_block_max
        vreg_fp4_scale = 2 if is_fp4 else 0

        orig_vreg = vreg_base + vreg_weights + vreg_acc + vreg_A + vreg_B + vreg_fp4_scale
        vreg = ceil_div(orig_vreg, 8) * 8
        # print(f"mma_m:{mma_m}, orig_vreg:{vreg}, vreg:{vreg}, vreg_base:{vreg_base}, vreg_weights:{vreg_weights}, vreg_acc:{vreg_acc}, vreg_A:{vreg_A}, vreg_B:{vreg_B}, vreg_fp4_scale:{vreg_fp4_scale}")
        return vreg

    def calc_num_threads(split_kv, warp_kv):
        warps_on_n = next_n
        warps_on_m = split_kv // warp_kv
        return warps_on_n * warps_on_m * 32

    def calc_tb_per_sm(num_threads, vreg, smem):
        warps_per_we = min(512 // vreg, 8)  # max 8 warp slots per warp engine (64 warps / 8 WEs)
        num_warps = 8 * warps_per_we
        vreg_tb_per_sm = num_warps // (num_threads // 32)
        smem_tb_per_sm = 262144 // smem
        return min(vreg_tb_per_sm, smem_tb_per_sm)

    def search_tile():
        tile_list = []
        for split_kv in [64, 128, 256]:
            for warp_kv in [16, 32, 64]:
                num_threads = calc_num_threads(split_kv, warp_kv)
                if num_threads < 128: continue
                if split_kv // warp_kv > 4 and num_threads != 512: continue
                stage_k = 3
                for split_mblock in [False, True]:
                    if split_mblock and (not (is_fp4 and warp_kv >= 32)): continue
                    vreg = calc_vreg(16 if split_mblock else warp_kv)
                    # Compare stage_q=1 and stage_q=2, pick one based on tb_per_sm
                    candidates = []
                    for stage_q in [1, 2]:
                        if num_threads == 512 and stage_q == 1: continue # warp-interleave not support stage_q = 1
                        smem = calc_smem(split_kv, stage_q, stage_k)
                        tb_per_sm = calc_tb_per_sm(num_threads, vreg, smem)
                        if tb_per_sm == 0: continue
                        candidates.append((stage_q, tb_per_sm))
                    if not candidates:
                        continue
                    elif len(candidates) == 1:
                        stage_q, tb_per_sm = candidates[0]
                    else:
                        (_, tb_per_sm_1), (_, tb_per_sm_2) = candidates
                        stage_q, tb_per_sm = candidates[0] if tb_per_sm_1 > tb_per_sm_2 else candidates[1]
                    tile_list.append((stage_q, stage_k, split_kv, warp_kv, split_mblock, tb_per_sm))
        tile = max(tile_list, key=lambda x: (x[5], x[0], 1 - int(x[4])))
        # print(f'tile_list: {tile_list}, selected_tile: {tile}')
        return tile

    tile_map = {
        # next_n = 1 tiles
        (1, 1, 32, 64): (2, 3, 256, 64, False, 4),
        (1, 1, 64, 64): (2, 3, 256, 64, False, 4),
        (1, 1, 32, 128): (1, 3, 64, 16, False, 8),
        (1, 1, 64, 128): (2, 3, 64, 16, False, 6),
        # int8 next_n > 1 tiles
        (1, 2, 32, 128): (2, 3, 256, 32, False, 2),
        (1, 4, 32, 128): (2, 3, 128, 32, False, 2),
        (1, 2, 64, 128): (2, 3, 256, 32, False, 1),
        (1, 4, 64, 128): (2, 3, 128, 32, False, 1),
        (1, 5, 64, 128): (2, 3, 64, 32, False, 1),
        (1, 6, 64, 128): (2, 3, 64, 32, False, 1),
        # fp4 next_n > 1 tiles, some using split_mblock
        (1, 2, 32, 64): (2, 3, 256, 64, True, 4),
        (1, 3, 32, 64): (1, 3, 256, 64, True, 4),
        (1, 4, 32, 64): (2, 3, 256, 64, True, 3),
        (1, 5, 32, 64): (1, 3, 64, 64, True, 9),
        (1, 6, 32, 64): (1, 3, 64, 64, True, 8),
        (1, 2, 64, 64): (1, 3, 256, 64, True, 4),
        (1, 3, 64, 64): (1, 3, 128, 64, True, 5),
    }
    tile_key = (datasize, next_n, num_heads, head_dim)
    if tile_key in tile_map:
        return tile_map[tile_key]
    # 4 heads are padded to 16 in the paged kernel: search with 16 heads and
    # ceil(next_n / 4) warp-tile "tokens" (fp4 keeps its own kernel layout)
    if num_heads == 4 and not is_fp4:
        num_heads = 16
        next_n = ceil_div(next_n, 4)
    return search_tile()
