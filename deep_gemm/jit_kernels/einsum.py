import math
import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_m_alignment_for_contiguous_layout,get_col_major_tma_aligned_tensor, GemmType
from .gemm_fp8 import get_best_configs
from .gemm_int8 import get_best_configs as get_best_int8_configs
# C++ code templates
includes = ('<deep_gemm/impls/fp8_gemm.cuh>', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto BLOCK_N_PADDING = {BLOCK_N_PADDING};
constexpr auto kSwizzleDMode = {SWIZZLE_D_MODE};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated GEMM
using gemm_t = Fp8Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, GemmType::BatchGemm>;

// Launch kernel
gemm_t::run(d, a, b, sfa,
            sfb, nullptr, nullptr, m, 0,
            stream, num_sms, smem_size);
"""

includes_cutlass3 = ('<deep_gemm/impls/int8_gemm_cutlass3.cuh>', )
template_cutlass3 = """
using namespace deep_gemm;

// Templated args from Python JIT call
using ElementAB = {ElementAB};
using ElementAcc = {ElementAcc};
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto WARP_M = {WARP_M};
constexpr auto WARP_N = {WARP_N};
constexpr auto BLOCK_K = {BLOCK_K};
constexpr auto kNumGroups = {NUM_GROUPS};
constexpr auto kNumStages = {NUM_STAGES};

// Make a templated grouped GEMM
using gemm_t = Gemm<ElementAB, ElementAcc, N, K, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, kNumGroups, kNumStages, GemmType::BatchGemm>;

// Launch kernel
gemm_t::run(d, nullptr, nullptr,
            m, 0, (ElementAB*)a, sfa, (ElementAB*)b, sfb,
            stream, num_sms, smem_size);
"""

def int8_einsum(expr: str,
               a: Tuple[torch.Tensor, torch.Tensor],
               b: Tuple[torch.Tensor, torch.Tensor],
               d: torch.Tensor, c: torch.Tensor = None, recipe: list = [1, 1, 128]) -> None:
    if (expr == "bhr,hdr->bhd"):
        # Permute dims to satisfy the order of (batch_size, m, n, k)
        # (batch_size, m, n, k): (h, b, d, r)
        lhs, lhs_scales = a
        rhs, rhs_scales = b
        perm_a, perm_sfa = fused_permute(lhs, lhs_scales)
        int8_bmm(perm_a, perm_sfa, rhs, rhs_scales, d, c)
    elif (expr == "bhd,hdr->bhr"):
        raise NotImplementedError(
                "bhd,hdr->bhr is not yet supported in PPU int8_einsum."
            )
    elif (expr == "bhd,bhr->hdr"):
        raise NotImplementedError(
                "bhd,bhr->hdr is not yet supported in PPU int8_einsum."
            )
    else:
        raise ValueError(f"unsupported expr expression: {expr}!")
# --- Fused permute(1,0,2) kernel: includes & template (Python JIT pattern) ---
permute_includes = ('<deep_gemm/impls/fused_permute.cuh>', )
permute_template = """
using namespace deep_gemm;

// Compile-time constants from Python JIT
constexpr int VEC_SIZE = {VEC_SIZE};
constexpr int THREADS_PER_BLOCK = {THREADS_PER_BLOCK};
constexpr int ROWS_PER_ITER = {ROWS_PER_ITER};
constexpr int DIM1 = {DIM1};
constexpr int NUM_D2_VECS_A = {NUM_D2_VECS_A};
constexpr int TAIL_BYTES_A = {TAIL_BYTES_A};
constexpr int DIM2_SFA_BYTES = {DIM2_SFA_BYTES};
constexpr int D1_TILE = {D1_TILE};
constexpr bool USE_FAST_PATH = {USE_FAST_PATH};
constexpr bool SFA_D2_CONTIGUOUS = {SFA_D2_CONTIGUOUS};

// Grid: fast path uses (DIM1 * D2_BLOCKS, grid_y), SMEM path uses (D1_GROUPS, grid_y)
constexpr int D2_BLOCKS = (NUM_D2_VECS_A + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
constexpr int D1_GROUPS = DIM1 / D1_TILE;
int grid_x = USE_FAST_PATH ? (DIM1 * D2_BLOCKS) : D1_GROUPS;
int grid_y = (dim0 + ROWS_PER_ITER - 1) / ROWS_PER_ITER;
dim3 grid(grid_x, grid_y, 1);
dim3 block(THREADS_PER_BLOCK, 1, 1);
// SMEM: 0 for fast path, full tile for SMEM transpose path
constexpr int SMEM_SIZE = USE_FAST_PATH ? 0 : (ROWS_PER_ITER * D1_TILE * NUM_D2_VECS_A * VEC_SIZE);

fused_permute_kernel<VEC_SIZE, THREADS_PER_BLOCK, ROWS_PER_ITER, DIM1, NUM_D2_VECS_A, TAIL_BYTES_A, DIM2_SFA_BYTES, D1_TILE, USE_FAST_PATH, SFA_D2_CONTIGUOUS><<<grid, block, SMEM_SIZE, stream>>>(
    (const char*)src_a, (char*)dst_a,
    (const char*)src_sfa, (char*)dst_sfa,
    dim0, dim2_a,
    stride0_a_bytes, stride1_a_bytes,
    stride0_sfa_bytes, stride1_sfa_bytes, stride2_sfa_bytes);
__return_code = 0;
"""


def fused_permute(a: torch.Tensor, sfa: torch.Tensor):
    """
    Fused JIT kernel: permute(1,0,2) for two 3-D tensors in one launch.

    Both tensors must share the same (dim0, dim1) but may have different dim2 and dtypes.
    Supports non-contiguous inputs via byte strides for both a and sfa.
    Requires a.stride(-1) == 1; sfa d2 direction can be non-contiguous (scalar fallback).
    Returns (out_a, out_sfa) with shape (dim1, dim0, dim2_x) for each.
    """
    assert a.dim() == 3 and a.stride(-1) == 1, \
        'fused_permute: tensor a must have stride(-1) == 1 for vectorized loads'
    assert sfa.dim() == 3, \
        'fused_permute: tensor sfa must be a 3-D tensor'

    dim0, dim1, dim2_a = a.shape
    dim0_s, dim1_s, dim2_sfa = sfa.shape
    assert dim0 == dim0_s and dim1 == dim1_s, \
        'fused_permute: dim0 and dim1 must match between a and sfa'
    if any(d == 0 for d in (dim0, dim1, dim2_a, dim2_sfa)):
        return

    out_a = torch.empty((dim1, dim0, dim2_a), dtype=a.dtype, device=a.device)
    out_sfa = torch.empty((dim1, dim0, dim2_sfa), dtype=sfa.dtype, device=sfa.device)


    # Convert to byte-level dimensions
    elem_size_a = a.element_size()
    dim2_a_bytes = dim2_a * elem_size_a
    elem_size_sfa = sfa.element_size()
    dim2_sfa_bytes = dim2_sfa * elem_size_sfa

    num_d2_vecs_a = dim2_a_bytes // 16
    tail_bytes_a = dim2_a_bytes % 16
    max_smem = 65536  # 64KB target for good occupancy

    # Determine fast path early — it affects parameter selection.
    # Fast path: no SMEM, direct register load/store. Grid = (DIM1*D2_BLOCKS, grid_y).
    # For tiny data, we want MAXIMUM blocks to saturate 39 SMs.
    total_a_bytes = dim0 * dim1 * dim2_a_bytes
    use_fast_path = total_a_bytes <= 524288  # 512KB

    if use_fast_path:
        # Fast path: no SMEM constraint, maximize parallelism.
        # Dynamically choose THREADS_PER_BLOCK to ensure enough grid blocks for GPU saturation.
        # With k=2048, NUM_D2_VECS_A=128 → D2_BLOCKS=2 with THREADS_PER_BLOCK=64 (only 2 warps).
        # THREADS_PER_BLOCK=32 (1 warp) doubles D2_BLOCKS to 4, recovering block count.
        rows_per_iter = 1
        d1_tile = 1  # Not used for fast path, but needed for cache key
        d2_blocks_64 = (num_d2_vecs_a + 63) // 64
        grid_x_64 = dim1 * d2_blocks_64
        grid_y_fast = dim0  # rows_per_iter=1
        total_blocks_64 = grid_x_64 * grid_y_fast
        if total_blocks_64 <= 256 and grid_y_fast >= 8:
            threads_per_block = 32   # 1 warp: doubles D2_BLOCKS for more parallelism
        else:
            threads_per_block = 64   # 2 warps: sufficient parallelism
    else:
        # SMEM path: balance parallelism with SMEM constraints.
        # Choose THREADS_PER_BLOCK: larger blocks have lower dispatch overhead per thread
        if dim0 <= 64:
            threads_per_block = 128
        elif dim0 <= 256:
            threads_per_block = 256
        else:
            threads_per_block = 512

        # Choose rows_per_iter based on dim0 (m) size:
        # Large m → more blocks is fine (good parallelism), minimize per-block work
        # Small m → need high rows_per_iter to reduce block count (launch overhead)
        if dim0 >= 256:
            rows_per_iter = 4
        elif dim0 >= 64:
            rows_per_iter = 2
        else:
            max_rows_for_smem = max_smem // (dim1 * num_d2_vecs_a * 16)
            rows_per_iter = min(dim0, max(1, max_rows_for_smem))
            rows_per_iter = 1 << (rows_per_iter.bit_length() - 1)
            rows_per_iter = min(rows_per_iter, 16)

        # Compute D1_TILE: balance parallelism vs per-block overhead.
        # SMEM path uses grid_x = D1_GROUPS = DIM1 / D1_TILE.
        grid_y = (dim0 + rows_per_iter - 1) // rows_per_iter
        target_blocks = 128
        min_d1_groups = max(1, (target_blocks + grid_y - 1) // grid_y)
        max_d1_tile_by_parallelism = max(1, dim1 // min_d1_groups)
        max_d1_tile_by_smem = max(1, max_smem // (rows_per_iter * num_d2_vecs_a * 16))
        d1_tile = min(dim1, max_d1_tile_by_smem, max_d1_tile_by_parallelism)
        d1_tile = max(1, d1_tile)
        while dim1 % d1_tile != 0:
            d1_tile -= 1

    # Byte strides for non-contiguous support (d2 direction must be contiguous, stride(-1)==1)
    stride0_a_bytes = a.stride(0) * elem_size_a
    stride1_a_bytes = a.stride(1) * elem_size_a
    stride0_sfa_bytes = sfa.stride(0) * elem_size_sfa
    stride1_sfa_bytes = sfa.stride(1) * elem_size_sfa
    stride2_sfa_bytes = sfa.stride(2) * elem_size_sfa
    sfa_d2_contiguous = (sfa.stride(-1) == 1)

    global permute_includes, permute_template
    args = (a, out_a, sfa, out_sfa,
            dim0, dim2_a_bytes,
            stride0_a_bytes, stride1_a_bytes,
            stride0_sfa_bytes, stride1_sfa_bytes, stride2_sfa_bytes,
            torch.cuda.current_stream())
    ElementA = "cutlass::float_e4m3_t" if a.dtype == torch.float8_e4m3fn else "int8_t"
    runtime = jit_tuner.compile_and_tune(
        name='fused_permute_'+ ElementA,
        keys={'VEC_SIZE': 16, 'THREADS_PER_BLOCK': threads_per_block, 'ROWS_PER_ITER': rows_per_iter,
                'DIM1': dim1, 'NUM_D2_VECS_A': num_d2_vecs_a, 'TAIL_BYTES_A': tail_bytes_a,
                'DIM2_SFA_BYTES': dim2_sfa_bytes, 'D1_TILE': d1_tile,
                'USE_FAST_PATH': 'true' if use_fast_path else 'false',
                'SFA_D2_CONTIGUOUS': 'true' if sfa_d2_contiguous else 'false'},
        space=(),
        includes=permute_includes,
        arg_defs=(('src_a',    a.dtype), ('dst_a',    a.dtype),
                  ('src_sfa',  sfa.dtype), ('dst_sfa',  sfa.dtype),
                  ('dim0',     int), ('dim2_a',   int),
                  ('stride0_a_bytes',  int), ('stride1_a_bytes',  int),
                  ('stride0_sfa_bytes', int), ('stride1_sfa_bytes', int), ('stride2_sfa_bytes', int),
                  ('stream',   torch.cuda.Stream)),
        template=permute_template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )
    runtime(*args)

    return out_a, out_sfa


def fp8_einsum(expr: str,
               a: Tuple[torch.Tensor, torch.Tensor],
               b: Tuple[torch.Tensor, torch.Tensor],
               d: torch.Tensor, c: torch.Tensor = None, recipe: list = [1, 1, 128]) -> None:
    # recipe is not used for PPU, only support [1, 128, 128]
    if (expr == "bhr,hdr->bhd"):
        # Permute dims to satisfy the order of (batch_size, m, n, k)
        # (batch_size, m, n, k): (h, b, d, r)
        lhs, lhs_scales = a
        rhs, rhs_scales = b
        perm_a, perm_sfa = fused_permute(lhs, lhs_scales)
        fp8_bmm(perm_a, perm_sfa, rhs, rhs_scales, d, c)
    elif (expr == "bhd,hdr->bhr"):
        raise NotImplementedError(
                "bhd,hdr->bhr is not yet supported in PPU fp8_einsum."
            )
    elif (expr == "bhd,bhr->hdr"):
        raise NotImplementedError(
                "bhd,bhr->hdr is not yet supported in PPU fp8_einsum."
            )
    else:
        raise ValueError(f"unsupported expr expression: {expr}!")

def fp8_bmm(a: torch.Tensor, sfa: torch.Tensor,
            b: torch.Tensor, sfb: torch.Tensor,
            d: torch.Tensor, c: torch.Tensor = None, recipe: list = [1, 1, 128], compiled_dims: str = None,
            expr: str = None, configs = None) -> None:
    # recipe & compiled_dims are not used for PPU
    groups, m, k = a.shape
    groups_, n, k_ = b.shape
    m_, groups__, n_ = d.shape
    if sfa.shape == (groups, m, 1) and sfb.shape == (groups_, n, 1):
        return int8_bmm(a, sfa, b, sfb, d, c)

    assert k % 128 == 0, f"K={k} must be a multiple of 128"

    # Type and shape checks
    assert groups == groups_ and groups_ == groups__
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert a.dtype == torch.float8_e4m3fn and sfa.dtype == torch.float32
    assert b.dtype == torch.float8_e4m3fn and sfb.dtype == torch.float32
    assert d.dtype == torch.bfloat16
    assert a.is_contiguous() and b.is_contiguous() and d.is_contiguous()

    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    sfa = get_col_major_tma_aligned_tensor(sfa)
    assert sfb.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_configs(m, n, k, groups, num_sms, gemm_type=GemmType.BatchGemm)

    args = (a, sfa, b, sfb, d, m, torch.cuda.current_stream(), num_sms, smem_config[0])
    runtime = jit_tuner.compile_and_tune(
        name='batch_gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'BLOCK_K' : block_k, 'WARP_M' : warp_m, 'WARP_N' : warp_n,
              'NUM_GROUPS': groups,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_STAGES': num_stages},
        space=(),
        includes=includes,
        arg_defs=(('a', torch.float8_e4m3fn), ('sfa', torch.float),
                  ('b', torch.float8_e4m3fn), ('sfb', torch.float),
                  ('d', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args,
        jit_include_dir='actlize_v1.0.0'
    )
    # Run the kernel
    runtime(*args)

def int8_bmm(a: torch.Tensor, sfa: torch.Tensor,
             b: torch.Tensor, sfb: torch.Tensor,
             d: torch.Tensor, c: torch.Tensor = None, recipe: list = [1, 1, 128], compiled_dims: str = None,
             expr: str = None, configs = None) -> None:
    # recipe & compiled_dims are not used for PPU
    groups, m, k = a.shape
    groups_, n, k_ = b.shape
    m_, groups__, n_ = d.shape

    # Type and shape checks
    assert groups == groups_ and groups_ == groups__
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert a.dtype == torch.int8 or a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.int8 or b.dtype == torch.float8_e4m3fn
    assert a.dtype == b.dtype
    assert sfa.dtype == torch.float32 and sfb.dtype == torch.float32
    assert d.dtype == torch.bfloat16
    assert a.is_contiguous() and b.is_contiguous() and d.is_contiguous()
    assert sfb.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    if configs is not None:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = configs
    else:
        num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config = get_best_int8_configs(m, n, k, groups, num_sms, gemm_type=GemmType.BatchGemm)

    args = (a, sfa, b, sfb, d, m, torch.cuda.current_stream(), num_sms, smem_config[0])

    ElementAB = "cutlass::float_e4m3_t" if a.dtype == torch.float8_e4m3fn else "int8_t"
    ElementAcc = "float" if a.dtype == torch.float8_e4m3fn else "int32_t"
    runtime = jit_tuner.compile_and_tune(
        name='batch_gemm_' + ElementAB + '_bf16_nt',
        keys={'ElementAB' : ElementAB, "ElementAcc" : ElementAcc,
              'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'BLOCK_K' : block_k, 'WARP_M' : warp_m, 'WARP_N' : warp_n,
              'NUM_GROUPS': groups,
              'SWIZZLE_D_MODE': smem_config[1],
              'BLOCK_N_PADDING': smem_config[2],
              'NUM_STAGES': num_stages},
        space=(),
        includes=includes_cutlass3,
        arg_defs=(('a', a.dtype), ('sfa', torch.float),
                  ('b', b.dtype), ('sfb', torch.float),
                  ('d', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template_cutlass3,
        jit_include_dir='actlize_v1.0.0',
        args=args,
    )
    # Run the kernel
    runtime(*args)
