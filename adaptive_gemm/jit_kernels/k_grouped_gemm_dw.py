import os
import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_col_major_tma_aligned_tensor, get_num_sms

os.environ["DG_DW_DEBUG"] = "1"

# C++ code templates
includes = ('"adaptive_gemm/fp8_gemm_dw.cuh"', )
template = """
using namespace adaptive_gemm;

// Templated args from Python JIT call
constexpr auto M = {M}, N = {N};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated grouped GEMM
using GemmType = GemmDW<M, N, BLOCK_M, BLOCK_N, 128, {NUM_GROUPS}, kNumStages, kNumTMAMulticast, GemmType::{GEMM_TYPE}>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, k);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs, k);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, k);
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out);
GemmType::run(out, rhs_scales, grouped_layout,
              k,
              tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
              stream, num_sms, smem_size);
"""


def ceil_div(a, b):
    return (a + b - 1) // b


def is_tma_multicast_legal(n: int, block_n: int, num_tma_multicast: int, num_sms: int) -> bool:
    if num_tma_multicast == 1:
        return True
    return (n % (block_n * num_tma_multicast) == 0) and num_sms % num_tma_multicast == 0


def get_smem_size(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128) -> int:
    smem_d = block_m * block_n * 2
    smem_a_per_stage = block_m * block_k
    smem_scales_a_per_stage = block_m * 4
    smem_b_per_stage = block_n * block_k
    # smem_scales_b = ceil_div(k, block_k) * 4
    smem_barrier = num_stages * 8 * 2

    smem_size = 0
    smem_size += smem_d
    smem_size += num_stages * smem_a_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += num_stages * smem_b_per_stage
    # smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    smem_size += smem_barrier
    return smem_size


def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     is_grouped_contiguous: bool = False) -> Tuple[int, int, int, int, int]:
    assert is_grouped_contiguous

    block_ms = (128, )
    block_ns = tuple(range(16, 129, 8))

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        for block_n in block_ns:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            if best_block_m is None or best_block_n is None:
                success = True
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util or (util == best_util and (block_m > best_block_m or (block_m == best_block_m and block_n < best_block_n)))
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)
            # if success and os.environ.get("DG_DW_DEBUG", None) is not None:
            #     print(f"BM, BN: [{block_m}, {block_n}]; num_waves: {num_waves}; last_wave_util: {get_last_wave_util(block_m, block_n)}%")
    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_size, sm90_capacity = None, None, 232448
    for num_stages in (6, 5, 4) if 128 % best_block_n != 0 else (8, 7, 6, 5, 4):
        best_smem_size = get_smem_size(num_stages, k, best_block_m, best_block_n)
        if best_smem_size <= sm90_capacity:
            best_num_stages = num_stages
            break
    assert best_num_stages is not None

    # Decide the number of TMA multicast
    best_num_tma_multicast = 2
    # if m >= 1024 and is_tma_multicast_legal(n, best_block_n, 2, num_sms) and num_groups == 1:
    #     best_num_tma_multicast = 2

    # print(best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size)
    return best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size


@torch.library.custom_op("moe::k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous", mutates_args=('out', ))
def k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    out: torch.Tensor, k_indices: torch.Tensor
) -> None:
    """
    Do a grouped GEMM (contiguous format) with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.
    
    **Note**: On the K axis, inputs are grouped into several batches, of which batch sizes aligned to
        `get_k_alignment_for_contiguous_layout()` (128).


    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k_sum]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k_sum / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k_sum]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k_sum / 128⌉]`.
        out: the BF16 output tensor of shape `[num_groups, m, n]`, representing the result.
        k_indices: a tensor of shape `[num_groups]` with type `torch.int`.
            `k_indices[i]` records the k-dim size of each group,
            which means that the k-dimension of i-th group of the problems is `k_indices[i]`.
    """
    # lhs, lhs_scales = lhs
    # rhs, rhs_scales = rhs
    m, k   = lhs.shape
    n, k_  = rhs.shape
    num_groups, m_, n_ = out.shape
    num_groups_ = k_indices.numel()

    # Type and shape checks
    assert m == m_ and k == k_ and n == n_ and num_groups == num_groups_
    assert k % 128 == 0 and k != 0
    assert lhs_scales.shape == (m, k // 128), f"{lhs_scales.shape}"
    assert rhs_scales.shape == ((n + 127) // 128, k // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert k_indices.dtype == torch.int32
    assert lhs.is_contiguous() and rhs.is_contiguous()
    assert out.is_contiguous() and k_indices.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `k` is zero
    if k == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    # When communication overlaps with computing, both operations compete for SM resources.
    # Disable persistent kernel can lead to better performance.
    num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count * 10
    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, num_groups, num_sms,
                                                                                  is_grouped_contiguous=True)
    args = (lhs, lhs_scales, rhs, rhs_scales, out,
            k_indices, k, num_groups,
            torch.cuda.current_stream(), num_sms, smem_size)
    # print(f"args:\n{args}")
    runtime = jit_tuner.compile_and_tune(
        name='k_grouped_gemm_dw_fp8_fp8_bf16_tn',
        keys={'M': m, 'N': n, 'BLOCK_M': block_m, 'BLOCK_N': block_n, 'NUM_GROUPS': num_groups,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast, 'GEMM_TYPE': 'GroupedContiguous'},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16),
                  ('grouped_layout', torch.int32), ('k', int), ('num_groups', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # for a in args:
    #     print(a)

    # Run the kernel
    runtime(*args)


@k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous.register_fake
def _(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    out: torch.Tensor, k_indices: torch.Tensor
) -> None:
    return

def per_tile_quant(A: torch.Tensor, gsize:int=128):
    import torch.nn.functional as F
    assert len(A.shape) == 2
    m, k = A.shape

    num_tiles_per_row = k // gsize

    pad_n = (gsize - (k % gsize)) % gsize
    if pad_n: A = F.pad(A, (0, pad_n), mode='constant', value=0)

    A_reshape = A.reshape(-1, gsize)
    scale = A_reshape.abs().amax(dim=1, keepdim=True).to(torch.float32).div(448.)

    q_tensor = A_reshape.div(scale).to(torch.float8_e4m3fn).reshape(*A.shape)[:,:k]
    fq_tensor = A_reshape.div(scale).to(torch.float8_e4m3fn).to(torch.float).mul(scale).reshape(*A.shape)[:,:k]

    return q_tensor, scale.reshape(m, num_tiles_per_row), fq_tensor

def per_block_quant(B: torch.Tensor, block_size: tuple[int, int] = (128, 128)):
    import torch.nn.functional as F
    assert len(B.shape) == 2
    bn, bk = block_size
    n, k = B.shape

    # Compute padding for rows and columns
    pad_m = (bn - (n % bn)) % bn
    pad_n = (bk - (k % bk)) % bk

    if pad_m or pad_n:
        B = F.pad(B, (0, pad_n, 0, pad_m), mode='constant', value=0)

    N, K = B.shape
    num_blocks_n = N // bn
    num_blocks_k = K // bk

    # Reshape to (num_block_rows, bm, num_block_cols, bn) and rearrange to (-1, bm, bn)
    B_blocks = (
        B
        .reshape(num_blocks_n, bn, num_blocks_k, bk)
        .permute(0, 2, 1, 3)
        .reshape(-1, bn, bk)
        .contiguous()
    )

    scale = B_blocks.abs().amax(dim=(1, 2), keepdim=True).to(torch.float32).div(448.)

    q_tensor = B_blocks.div(scale).to(torch.float8_e4m3fn)
    fq_tensor = q_tensor.to(torch.float).mul(scale)

    q_tensor = (
        q_tensor
        .reshape(num_blocks_n, num_blocks_k, bn, bk)
        .permute(0, 2, 1, 3)
        .reshape(N, K)[:n, :k]
    )
    fq_tensor = (
        fq_tensor
        .reshape(num_blocks_n, num_blocks_k, bn, bk)
        .permute(0, 2, 1, 3)
        .reshape(N, K)[:n, :k]
    )
    scale = scale.reshape(num_blocks_n, num_blocks_k)

    return q_tensor, scale, fq_tensor


def quant_input(a: torch.Tensor, b: torch.Tensor, k_indices: torch.Tensor, BLOCK_K: int = 128):
    """
    a: `[m, k_sum]` with type `torch.bfloat16`
    b: `[n, k_sum]` with type `torch.bfloat16`
    k_indices: `[num_groups]` with type `torch.int` which records the k-dim size of each group

    Returns:
        ((a_quant, a_scale), (b_quant, b_scale))
    """
    import torch.nn.functional as F

    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[1] == b.shape[1], "a and b must have the same k dimension size"
    assert k_indices.sum().item() == a.shape[1], "Sum of k_indices must match the k dimension"

    m, k_sum = a.shape
    n, _ = b.shape

    a_quant_list, a_scale_list, b_quant_list, b_scale_list = [], [], [], []
    new_k_indices = []

    k_start = 0
    for k_size in k_indices.tolist():
        # Align the current group's k dimension to BLOCK_K
        aligned_k_size = (k_size + BLOCK_K - 1) // BLOCK_K * BLOCK_K

        pad_size = aligned_k_size - k_size
        a_group = F.pad(a[:, k_start:k_start+k_size], (0, pad_size))
        b_group = F.pad(b[:, k_start:k_start+k_size], (0, pad_size))

        # Quantize the aligned groups
        a_quant, a_scale, _ = per_tile_quant(a_group, gsize=BLOCK_K)
        b_quant, b_scale, _ = per_block_quant(b_group, block_size=(BLOCK_K, BLOCK_K))

        a_quant_list.append(a_quant)
        a_scale_list.append(a_scale)
        b_quant_list.append(b_quant)
        b_scale_list.append(b_scale)
        new_k_indices.append(a_quant.shape[1])

        k_start += k_size

    # Concatenate the quantized tensors and scales
    a_quant = torch.cat(a_quant_list, dim=1)
    b_quant = torch.cat(b_quant_list, dim=1)

    a_scale = torch.cat(a_scale_list, dim=1)  # [m, total_tiles]
    b_scale = torch.cat(b_scale_list, dim=1)  # [num_blocks, total_blocks]

    return (a_quant, a_scale), (b_quant, b_scale), torch.tensor(new_k_indices, dtype=torch.int32, device='cuda')


def get_bfloat16_ref(lhs: tuple[torch.Tensor, torch.Tensor], rhs: tuple[torch.Tensor, torch.Tensor], k_indices: torch.Tensor):
    # since fp8 matmul is not supported, we use bfloat16 matmul as reference
    a, a_scale = lhs
    b, b_scale = rhs

    m, k = a.shape
    n, k_ = b.shape
    num_groups = k_indices.numel()

    BLOCK_K = 128
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == torch.float32 and b_scale.dtype == torch.float32
    assert k == k_ and k == k_indices.sum().item() and k % BLOCK_K == 0

    assert n % BLOCK_K == 0, f"Assume n is multiple of {BLOCK_K} for now"

    out = torch.zeros(num_groups, m, n, dtype=torch.bfloat16, device='cuda')
    a_dequant = (
        a
        .reshape(-1, BLOCK_K)
        .to(torch.float)
        .mul(a_scale.reshape(-1, 1))
        .reshape(m, k)
        .to(torch.bfloat16)
    )
    b_dequant = (
        b
        .reshape(-1, BLOCK_K, k // BLOCK_K, BLOCK_K)
        .transpose(1,2)
        .to(torch.float)
        .mul(b_scale.view(n//BLOCK_K, k//BLOCK_K, 1, 1))
        .transpose(1,2)
        .reshape(n, k)
        .to(torch.bfloat16)
    )
    k_start = 0
    for g, k_size in enumerate(k_indices.tolist()):
        a_group = a_dequant[:, k_start:k_start + k_size]
        b_group = b_dequant[:, k_start:k_start + k_size]
        out[g] = a_group@b_group.t().to(out.dtype)
        k_start += k_size

    return out


if __name__ == "__main__":

    VERIFICATION = True
    BENCHMARK = True

    # group, M, N, K
    TEST_SHAPES = [
        (4, 7168, 4096, 8192),
        (4, 2048, 7168, 8192),
        (8, 7168, 4096, 4096),
        (8, 4096, 7168, 4096),
        # (1, 4096, 4096, 4096),
    ]

    for (G, M, N, K) in TEST_SHAPES:
        A = torch.randn((M, K * G), device='cuda', dtype=torch.bfloat16)
        B = torch.randn((N, K * G), device='cuda', dtype=torch.bfloat16)
        D = torch.zeros((G, M, N), device='cuda', dtype=torch.bfloat16)
        k_indices = torch.tensor([K] * G, dtype=torch.int32, device='cuda')

        (A_quant, A_scale), (B_quant, B_scale), k_indices = quant_input(A, B, k_indices)

        print(f"{'='*50} [G, M, N, K]={[G, M, N,K]} {'='*50}")
        if VERIFICATION:
            # print(A_quant.shape)
            # print(A_scale.shape)
            # print(B_quant.shape)
            # print(B_scale.shape)
            # A_scale = torch.ones_like(A_scale)
            # B_scale = torch.ones_like(B_scale)
            ref = get_bfloat16_ref((A_quant, A_scale), (B_quant, B_scale), k_indices)
            k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous((A_quant, A_scale), (B_quant, B_scale), D, k_indices)
            if not torch.allclose(ref, D, atol=1, rtol=1e-1):
                print(f">>> Verification failed!")
                print(f"res:\n{D}")
                print(f"ref:\n{ref}")
            else:
                print(f">>> Verification passed!")

            amax = max(D.abs().max(), ref.abs().max())
            adiffmax = (D - ref).abs().max()
            rdiffmax = adiffmax / amax
            print(f"    max relative difference of the layer is {rdiffmax}")

        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        TEST_ITERS = 20
        cost = []
        for _ in range(TEST_ITERS):
            st.record()
            k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous((A_quant, A_scale), (B_quant, B_scale), D, k_indices)
            ed.record()
            torch.cuda.synchronize()
            cost.append(st.elapsed_time(ed))
        median = sorted(cost)[TEST_ITERS//2]
        print(f">>> time cost: {median:.3f} ms; TFLOPS: {G*M*N*K*2/median/1e9:.4f}")
