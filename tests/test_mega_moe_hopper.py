"""
H200 (SM90 / Hopper) mega-MoE: fused kernel + 同管线 baseline 性能对比。

结构对齐 tests/test_mega_moe.py（B 系列 SM100 FP4 路径），但所有路径都换成 H200 FP8：
  * fused：调用 `deep_gemm.fp8_mega_moe`（kernel symbol `sm90_fp8_mega_moe_impl`），
           使用 `transform_weights_for_mega_moe_sm90` 处理过的权重 + SymmBuffer。
  * baseline：DeepEP dispatch + 2 个 grouped FP8 GEMM + Triton SwiGLU + DeepEP combine，
              使用未变换的权重。由于当前 SM90 grouped GEMM 只支持 L2 activation
              per-128-K SFA，而 fused SM90 mega-MoE 的 L1 epilogue 为避免跨 CTA
              同步使用 per-64-K SFA，所以该 baseline 是同管线 legacy 参照，
              不是 bitwise apples-to-apples correctness oracle。
  * 性能输出涵盖：TFLOPS / overlap TFLOPS / HBM GB/s / NVL GB/s / fused us /
                  reduction us / `t_baseline / t_fused` legacy 比。
"""

import argparse
import math
import os
import random
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, get_arch_major

try:
    import deep_ep as _deep_ep
    _deep_ep_import_error = None
except Exception as ex:
    _deep_ep = None
    _deep_ep_import_error = ex


# 与 deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh 中模板入口同名，
# bench_kineto 用它从 trace 里挑出 fused mega-MoE 的 GPU 段
SM90_KERNEL_NAME = "sm90_fp8_mega_moe_impl"


# FP8 e4m3fn 的最大可表示值，量化时用 amax / 448 作为 scale 基准
FP8_E4M3_MAX = 448.0
# 新版 Triton（>= 3.x）强制：jit 内核读到的 Python 全局必须是 tl.constexpr 实例，
# 否则编译期 NameError。宿主 Python 侧仍用上面的普通 float 做 torch 运算。
_FP8_E4M3_MAX_TL = tl.constexpr(448.0)
L1_ACT_SF_GRAN = 128
FUSED_L2_ACT_SF_GRAN = 64
BASELINE_L2_ACT_SF_GRAN = 128
WEIGHT_SF_GRAN_MN = 128
WEIGHT_SF_GRAN_K = 128


# ============================================================================
# 模块 1：Triton SwiGLU + FP8 量化内核
# ----------------------------------------------------------------------------
# baseline 的 L2 仍走 DeepGEMM SM90 grouped FP8 GEMM，所以 activation SFA 只能按
# per-128-K 输入；但 scale 数值采用 fused epilogue 同款 UE8M0/power-of-two 规则，
# 避免再额外引入 exact-FP32-scale 差异。
# 输入  x        : (M, 2*H) bf16，内层是 [gate_part | up_part]
# 输入  topk_w   : (M,)     fp32，可选
# 输出  y        : (M, H)   fp8_e4m3fn
# 输出  y_sf     : (M, H/BLOCK_K) fp32 行主序
# ============================================================================


@triton.jit
def _swiglu_apply_weight_to_fp8_kernel(
    x_ptr,
    topk_w_ptr,
    y_ptr,
    y_sf_ptr,
    M,
    H,  # 运行时形状
    stride_xm,
    stride_xn,  # x: (M, 2H) 的 stride
    stride_ym,
    stride_yn,  # y: (M, H)  的 stride
    stride_sfm,
    stride_sfk,  # y_sf: (M, H/BLOCK_K) 的 stride
    clamp_value,  # 当 HAS_CLAMP=False 时这个参数无意义
    HAS_TOPK: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,  # = num_per_channels
):
    # 一个 program 处理 (BLOCK_M 个 token) × (第 pid_k 个 K-block 的 BLOCK_K 列)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # 行索引：本 program 负责 [pid_m*BLOCK_M, pid_m*BLOCK_M+BLOCK_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 当前 K-block 内的列索引（在 H 维度，不是 2H）
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # ---- 1) 载入 gate（x 的前半段 [0, H)）和 up（x 的后半段 [H, 2H)）----
    # 注意 stride_xn 是元素 stride（一般 == 1），但 H + offs_k 偏移是按"元素"算的
    gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
    up_ptrs = x_ptr + offs_m[:, None] * stride_xm + (H + offs_k[None, :]) * stride_xn
    gate = tl.load(gate_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # ---- 2) 可选 clamp（参考 tilelang 实现：gate 单边 max，up 双边）----
    if HAS_CLAMP:
        gate = tl.minimum(gate, clamp_value)
        up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)

    # ---- 3) SwiGLU：silu(gate) * up = gate * sigmoid(gate) * up（全程 FP32 累计）----
    y = gate * tl.sigmoid(gate) * up

    # ---- 4) 可选 MoE 权重缩放（per-token 标量）----
    if HAS_TOPK:
        w = tl.load(topk_w_ptr + offs_m, mask=mask_m, other=1.0)
        y = y * w[:, None]

    # ---- 5) 当前 K-block 内每行 absmax → scale ----
    amax = tl.max(tl.abs(y), axis=1)  # (BLOCK_M,)
    sf = tl.maximum(amax / _FP8_E4M3_MAX_TL, 1.0e-30)
    if USE_UE8M0_SCALE:
        # 对齐 deep_gemm/common/math.cuh::get_e4m3_sf_and_sf_inv:
        # scale = 2 ** ceil(log2(amax / 448)).
        sf = tl.exp2(tl.ceil(tl.log2(sf)))

    # ---- 6) 量化为 FP8 e4m3fn ----
    y_fp8 = (y / sf[:, None]).to(tl.float8e4nv)

    # ---- 7) 写回 y 和 sf ----
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yn
    tl.store(y_ptrs, y_fp8, mask=mask_m[:, None])

    sf_ptrs = y_sf_ptr + offs_m * stride_sfm + pid_k * stride_sfk
    tl.store(sf_ptrs, sf, mask=mask_m)


def swiglu_apply_weight_to_fp8_triton(
    x: torch.Tensor,
    topk_weights: torch.Tensor | None,
    clamp_value: float | None = None,
    num_per_channels: int = BASELINE_L2_ACT_SF_GRAN,
    use_ue8m0_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU + FP8 量化。语义等价于 PyTorch reference：
    gate, up = x[:, :H], x[:, H:]
    y = silu(gate.clamp(max=c)) * up.clamp(-c, c) * topk_w
    y_sf = y.view(M, H/np, np).abs().amax(-1) / 448
    if use_ue8m0_scale: y_sf = ceil_to_power_of_2(y_sf)
    y_fp8 = (y / y_sf.unsqueeze(-1)).to(fp8)
    """
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.is_contiguous(), "当前实现假设 x 是 contiguous 的，避免 stride 计算错位"
    M, two_H = x.shape
    H = two_H // 2
    assert H % num_per_channels == 0, f"H={H} 必须是 {num_per_channels} 的整数倍"

    y = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=x.device)
    y_sf = torch.empty((M, H // num_per_channels), dtype=torch.float32, device=x.device)

    # BLOCK_M 取 16：内核每个 program 处理 16 个 token × 128 列，寄存器压力小、容易调
    BLOCK_M = 16
    grid = (triton.cdiv(M, BLOCK_M), H // num_per_channels)

    # HAS_TOPK=False 时仍要传一个有效指针（Triton 不允许 nullptr），用 x 占位
    topk_ptr = topk_weights if topk_weights is not None else x

    _swiglu_apply_weight_to_fp8_kernel[grid](
        x,
        topk_ptr,
        y,
        y_sf,
        M,
        H,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_sf.stride(0),
        y_sf.stride(1),
        float(clamp_value) if clamp_value is not None else 0.0,
        HAS_TOPK=topk_weights is not None,
        HAS_CLAMP=clamp_value is not None,
        USE_UE8M0_SCALE=use_ue8m0_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_K=num_per_channels,
    )
    return y, y_sf


# ============================================================================
# 模块 2：grouped weight 的 (128, 128) FP8 块量化
# ----------------------------------------------------------------------------
# m_grouped_fp8_gemm_nt_contiguous 在 SM90 上对 weight 的输入约定：
#   每 (128, 128) 子块共享一个 FP32 SF，K 是 SF 的内层连续维（K-major）。
# 与 SM100 FP4 路径的差异：
#   * 不需要 deep_gemm.transform_sf_into_required_layout
#   * SF 是 FP32，不是 UE8M0 packed
# ============================================================================


def _quantize_grouped_fp8_block_128_128(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(G, N, K) bf16 → (G, N, K) fp8_e4m3fn + (G, N//128, K//128) fp32 SF。"""
    g, n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0, f"weight 的 N={n}, K={k} 都必须是 128 的倍数"

    chunk_g = 4
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    sf = torch.empty((g, n // 128, k // 128), dtype=torch.float, device=w.device)
    for start in range(0, g, chunk_g):
        end = min(start + chunk_g, g)
        w_view = w[start:end].view(end - start, n // 128, 128, k // 128, 128).float()
        sf_chunk = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4) / FP8_E4M3_MAX
        w_fp8[start:end].copy_(
            (w_view / sf_chunk.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn).view(end - start, n, k)
        )
        sf[start:end].copy_(sf_chunk)
    return w_fp8, sf.contiguous()


# ============================================================================
# 模块 3：尝试导入 deep_ep（用于 dispatch / combine）
# ============================================================================


def _import_deep_ep():
    if _deep_ep is None:
        dist_print(f"Failed to import deep_ep: {_deep_ep_import_error}", once_in_node=True)
        return None
    return _deep_ep


class _DeepEPHandle:
    def __init__(self, raw_handle, psum_num_recv_tokens_per_expert: torch.Tensor):
        self.raw_handle = raw_handle
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert


class _DeepEPBufferCompat:
    """Compatibility shim for newer DeepEP versions that expose Buffer, not ElasticBuffer."""

    def __init__(self, deep_ep, group, num_nvl_bytes: int):
        self.buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=0,
            explicitly_destroy=True,
        )

    def dispatch(
        self,
        x,
        *,
        topk_idx,
        topk_weights,
        num_experts: int,
        expert_alignment: int,
        **_,
    ):
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = (
            self.buffer.get_dispatch_layout(topk_idx, num_experts)
        )
        recv_x, _, recv_topk_weights, num_recv_tokens_per_expert, raw_handle, event = self.buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
        )
        psum = torch.tensor(
            num_recv_tokens_per_expert, dtype=torch.int, device=topk_idx.device
        ).cumsum(dim=0, dtype=torch.int)
        return recv_x, None, recv_topk_weights, _DeepEPHandle(raw_handle, psum), event

    def combine(self, x, *, handle):
        raw_handle = handle.raw_handle if isinstance(handle, _DeepEPHandle) else handle
        return self.buffer.combine(x, handle=raw_handle)

    def barrier(self, use_comm_stream: bool = False):
        torch.cuda.synchronize()
        dist.barrier()

    def destroy(self):
        self.buffer.destroy()


def _make_deep_ep_buffer(deep_ep, group, num_max_tokens_per_rank, hidden, num_topk, sym_buffer_bytes):
    if hasattr(deep_ep, "ElasticBuffer"):
        return deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=True,
            explicitly_destroy=True,
            allow_multiple_reduction=False,
        )
    nvl_alignment = 2 * 1024 * 1024
    num_nvl_bytes = ((int(sym_buffer_bytes) + nvl_alignment - 1) // nvl_alignment) * nvl_alignment
    return _DeepEPBufferCompat(deep_ep, group, num_nvl_bytes=num_nvl_bytes)


# ============================================================================
# 模块 4：CUDA event 中位数测时（避开对 tilelang.do_bench 的依赖）
# ============================================================================


def _bench_cuda_events(
    fn, num_warmup: int = 5, num_repeat: int = 20, l2_flush_gb: float = 8.0
) -> float:
    """返回 fn 的中位数耗时（秒）。"""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(num_repeat):
        # L2 flush，避免重复访问命中 cache 让测时偏低
        if l2_flush_gb > 0:
            free_bytes, _ = torch.cuda.mem_get_info()
            flush_bytes = min(int(l2_flush_gb * 1e9), int(free_bytes * 0.5))
            if flush_bytes >= 4:
                torch.empty(flush_bytes // 4, dtype=torch.int, device="cuda").zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times_ms.append(s.elapsed_time(e))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] / 1e3


# ============================================================================
# 模块 5：test() 主入口 — 在每个 rank 上跑一遍 baseline
# ============================================================================


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # 初始化分布式：rank_idx 是全局 rank，group 是默认 NCCL group
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(
            f"[SKIP] test_mega_moe_hopper requires SM90; got SM{get_arch_major()}0",
            once_in_node=True,
        )
        dist.destroy_process_group()
        return

    # 形状参数（与 test_mega_moe.py 同名同义）
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = (
        max(
            0,
            args.num_max_tokens_per_rank
            - random.randint(0, args.num_max_removed_tokens),
        )
        if args.num_tokens == 0
        else args.num_tokens
    )
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank
    assert num_experts % num_ranks == 0, (
        f"num_experts={num_experts} 必须能被 num_ranks={num_ranks} 整除"
    )

    # SM90 fused kernel 的形状约束（来自 csrc/apis/mega.hpp::fp8_mega_moe）：
    #   * H、IH 必须是 128 的倍数（L1 input per-128-K SF + block-(128,128) weight SF）
    #   * IH/64 ≤ 64 → IH ≤ 4096（l2_arrival_mask 是 uint64，每 bit 对应 64 列）
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64, (
        f"SM90 fused kernel 要求 intermediate_hidden <= 4096, 当前 {intermediate_hidden}"
    )

    # ---- 创建 BF16 输入：token 与两层 weight ----
    # x: 每 rank 本地 num_tokens 个 token，每个 token hidden 维
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    # L1 weight: 每个 expert 把 hidden → 2*intermediate_hidden（gate 和 up 拼一起）
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    # L2 weight: 每个 expert 把 intermediate_hidden → hidden
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )

    # 路由：scores → topk_idx (M, K) + topk_weights (M, K)
    eplb_replica_for = {}
    eplb_replica_slots: list[int] = []
    if args.num_redundant_experts > 0:
        assert args.num_redundant_experts % num_ranks == 0, "num_redundant_experts must divide num_ranks"
        num_replicas_per_rank = args.num_redundant_experts // num_ranks
        assert 0 < num_replicas_per_rank < num_experts_per_rank, "invalid redundant expert count"
        for r in range(num_ranks):
            base = r * num_experts_per_rank
            eplb_replica_slots += list(range(base + num_experts_per_rank - num_replicas_per_rank,
                                             base + num_experts_per_rank))
        logical_mask = torch.ones(num_experts, dtype=torch.bool, device="cuda")
        logical_mask[torch.tensor(eplb_replica_slots, dtype=torch.long, device="cuda")] = False
        if args.score_powerlaw_alpha > 0:
            expert_rank = torch.arange(1, num_experts + 1, dtype=torch.float, device="cuda")
            bias_for_hot = torch.pow(expert_rank, -args.score_powerlaw_alpha)
            bias_for_hot = (bias_for_hot - bias_for_hot.mean()) / (bias_for_hot.std() + 1e-6)
            hot_order = torch.argsort(bias_for_hot.masked_fill(~logical_mask, -float("inf")), descending=True).cpu().tolist()
        else:
            hot_order = torch.arange(num_experts, device="cuda")[logical_mask].cpu().tolist()
        hot_experts = hot_order[:args.num_redundant_experts]
        eplb_replica_for = {int(h): int(s) for h, s in zip(hot_experts, eplb_replica_slots)}
        if rank_idx == 0:
            print(
                f" > eplb_sim redundant={args.num_redundant_experts} "
                f"replicas_per_rank={num_replicas_per_rank} "
                f"dispatch={args.replica_dispatch}",
                flush=True,
            )

    def make_scores():
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
        if args.score_powerlaw_alpha > 0:
            expert_rank = torch.arange(1, num_experts + 1, dtype=torch.float, device="cuda")
            bias = torch.pow(expert_rank, -args.score_powerlaw_alpha)
            bias = (bias - bias.mean()) / (bias.std() + 1e-6)
            scores = scores + args.score_powerlaw_scale * bias[None, :]
        if eplb_replica_slots:
            scores[:, torch.tensor(eplb_replica_slots, dtype=torch.long, device="cuda")] = -float("inf")
        return scores

    def apply_eplb_replicas(idx: torch.Tensor) -> torch.Tensor:
        if not eplb_replica_for:
            return idx
        mapped = idx.clone()
        if args.replica_dispatch == "hash":
            token_ids = torch.arange(num_tokens, device="cuda")[:, None]
            slot_ids = torch.arange(num_topk, device="cuda")[None, :]
            choose_replica = ((token_ids * num_topk + slot_ids + rank_idx) & 1).bool()
            for logical_expert, replica_slot in eplb_replica_for.items():
                mapped = torch.where((idx == logical_expert) & choose_replica,
                                     torch.full_like(mapped, replica_slot), mapped)
        elif args.replica_dispatch == "static":
            for logical_expert, replica_slot in eplb_replica_for.items():
                logical_rank = logical_expert // num_experts_per_rank
                replica_rank = replica_slot // num_experts_per_rank
                if rank_idx == logical_rank:
                    chosen = logical_expert
                elif rank_idx == replica_rank:
                    chosen = replica_slot
                else:
                    chosen = replica_slot if ((rank_idx + logical_expert) & 1) else logical_expert
                if chosen != logical_expert:
                    mapped = torch.where(idx == logical_expert,
                                         torch.full_like(mapped, chosen), mapped)
        else:
            raise ValueError(f"unknown replica_dispatch={args.replica_dispatch}")
        return mapped

    if args.routing_mode == "balanced":
        assert args.masked_ratio == 0.0, "balanced routing does not support masked_ratio"
        assert (num_tokens * num_topk) % num_experts == 0, "balanced routing requires M*topk divisible by num_experts"
        token_ids = torch.arange(num_tokens, device="cuda", dtype=torch.long)[:, None]
        topk_offsets = torch.arange(num_topk, device="cuda", dtype=torch.long)[None, :]
        topk_idx = (token_ids * num_topk + topk_offsets) % num_experts
        topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float, device="cuda")
    elif args.routing_mode in ("balanced-shuffled", "balanced-shuffled-score"):
        assert args.masked_ratio == 0.0, f"{args.routing_mode} routing does not support masked_ratio"
        assert num_tokens % num_experts == 0, f"{args.routing_mode} requires M divisible by num_experts"
        assert num_experts % num_topk == 0, f"{args.routing_mode} requires experts divisible by topk"
        token_perm = torch.randperm(num_tokens, device="cuda")
        expert_perm = torch.randperm(num_experts, device="cuda")
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)
        slot_stride = num_experts // num_topk
        topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.long, device="cuda")
        for slot in range(num_topk):
            expert_ids = expert_perm[(positions + slot * slot_stride) % num_experts]
            topk_idx[token_perm, slot] = expert_ids
        if args.routing_mode == "balanced-shuffled-score":
            scores = make_scores()
            topk_weights = torch.gather(scores, 1, topk_idx)
        else:
            topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float, device="cuda")
    elif args.routing_mode == "topk-repair-budget-softmax":
        assert args.masked_ratio == 0.0, f"{args.routing_mode} routing does not support masked_ratio"
        assert (num_tokens * num_topk) % num_experts == 0, f"{args.routing_mode} requires exact local expert capacity"
        scores = make_scores()
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        expert_capacity = (num_tokens * num_topk) // num_experts
        scores_cpu = scores.cpu()
        probs_cpu = torch.softmax(scores_cpu, dim=1)
        selected = topk_idx.cpu().tolist()
        original_selected = [row[:] for row in selected]
        selected_sets = [set(row) for row in selected]
        counts = [0] * num_experts
        for row in selected:
            for expert in row:
                counts[expert] += 1

        selected_mass = 0.0
        for token, row in enumerate(selected):
            selected_mass += sum(float(probs_cpu[token, expert].item()) for expert in row)
        original_mass = selected_mass
        changed_slots = 0
        touched_tokens = set()
        budget = args.repair_mass_drop_budget

        while True:
            overflow = {e for e, c in enumerate(counts) if c > expert_capacity}
            underfull = [e for e, c in enumerate(counts) if c < expert_capacity]
            if not overflow or not underfull:
                break
            candidates = []
            for token, row in enumerate(selected):
                token_set = selected_sets[token]
                for slot, old_expert in enumerate(row):
                    if old_expert not in overflow:
                        continue
                    old_score = float(scores_cpu[token, old_expert].item())
                    old_mass = float(probs_cpu[token, old_expert].item())
                    for new_expert in underfull:
                        if new_expert in token_set:
                            continue
                        new_mass = float(probs_cpu[token, new_expert].item())
                        loss = old_score - float(scores_cpu[token, new_expert].item())
                        mass_loss = old_mass - new_mass
                        candidates.append((loss, mass_loss, token, slot, old_expert, new_expert))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            changed = False
            for _, mass_loss, token, slot, old_expert, new_expert in candidates:
                if counts[old_expert] <= expert_capacity or counts[new_expert] >= expert_capacity:
                    continue
                if selected[token][slot] != old_expert or new_expert in selected_sets[token]:
                    continue
                next_mass = selected_mass - mass_loss
                next_drop = (original_mass - next_mass) / max(original_mass, 1e-12)
                if next_drop > budget:
                    continue
                selected[token][slot] = new_expert
                selected_sets[token].remove(old_expert)
                selected_sets[token].add(new_expert)
                counts[old_expert] -= 1
                counts[new_expert] += 1
                selected_mass = next_mass
                changed_slots += 1
                touched_tokens.add(token)
                changed = True
            if not changed:
                break

        topk_idx = torch.tensor(selected, dtype=torch.long, device="cuda")
        topk_weights = torch.softmax(torch.gather(scores, 1, topk_idx), dim=-1)
        mass_drop = (original_mass - selected_mass) / max(original_mass, 1e-12)
        over_slots = sum(max(0, c - expert_capacity) for c in counts)
        max_count = max(counts)
        dist_print(
            f" > bounded_repair rank={rank_idx}: budget={budget:.3f} "
            f"changed={changed_slots / max(num_tokens * num_topk, 1) * 100:.1f}% "
            f"touched={len(touched_tokens) / max(num_tokens, 1) * 100:.1f}% "
            f"mass_drop={mass_drop * 100:.1f}% over_slots={over_slots} max_count={max_count}",
            once_in_node=False,
        )
    elif args.routing_mode == "global-repair-budget-softmax":
        assert args.masked_ratio == 0.0, f"{args.routing_mode} routing does not support masked_ratio"
        scores = make_scores()
        all_scores = uneven_all_gather(scores, group=group)
        local_num_tokens = torch.tensor([num_tokens], dtype=torch.long, device="cuda")
        all_num_tokens_t = [torch.zeros_like(local_num_tokens) for _ in range(num_ranks)]
        dist.all_gather(all_num_tokens_t, local_num_tokens, group=group)
        all_num_tokens = [int(x.item()) for x in all_num_tokens_t]
        local_offset = sum(all_num_tokens[:rank_idx])
        total_num_tokens = sum(all_num_tokens)
        assert (total_num_tokens * num_topk) % num_experts == 0, f"{args.routing_mode} requires exact global expert capacity"

        all_topk_weights, all_topk_idx = torch.topk(all_scores, num_topk, dim=-1, largest=True, sorted=False)
        expert_capacity = (total_num_tokens * num_topk) // num_experts
        scores_cpu = all_scores.cpu()
        probs_cpu = torch.softmax(scores_cpu, dim=1)
        score_order = torch.argsort(all_scores, dim=1, descending=True).cpu().tolist()
        selected = all_topk_idx.cpu().tolist()
        selected_sets = [set(row) for row in selected]
        counts = [0] * num_experts
        for row in selected:
            for expert in row:
                counts[expert] += 1

        selected_mass = 0.0
        for token, row in enumerate(selected):
            selected_mass += sum(float(probs_cpu[token, expert].item()) for expert in row)
        original_mass = selected_mass
        changed_slots = 0
        touched_tokens = set()
        budget = args.repair_mass_drop_budget
        max_rounds = max(1, args.repair_max_rounds)

        for _round in range(max_rounds):
            overflow = {e for e, c in enumerate(counts) if c > expert_capacity}
            if not overflow:
                break
            underfull = {e for e, c in enumerate(counts) if c < expert_capacity}
            if not underfull:
                break
            candidates = []
            for token, row in enumerate(selected):
                token_set = selected_sets[token]
                for slot, old_expert in enumerate(row):
                    if old_expert not in overflow:
                        continue
                    new_expert = -1
                    for cand in score_order[token]:
                        if cand in underfull and cand not in token_set:
                            new_expert = cand
                            break
                    if new_expert < 0:
                        continue
                    old_score = float(scores_cpu[token, old_expert].item())
                    old_mass = float(probs_cpu[token, old_expert].item())
                    new_mass = float(probs_cpu[token, new_expert].item())
                    loss = old_score - float(scores_cpu[token, new_expert].item())
                    mass_loss = old_mass - new_mass
                    candidates.append((loss, mass_loss, token, slot, old_expert, new_expert))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            changed = False
            for _, mass_loss, token, slot, old_expert, new_expert in candidates:
                if counts[old_expert] <= expert_capacity or counts[new_expert] >= expert_capacity:
                    continue
                if selected[token][slot] != old_expert or new_expert in selected_sets[token]:
                    continue
                next_mass = selected_mass - mass_loss
                next_drop = (original_mass - next_mass) / max(original_mass, 1e-12)
                if next_drop > budget:
                    continue
                selected[token][slot] = new_expert
                selected_sets[token].remove(old_expert)
                selected_sets[token].add(new_expert)
                counts[old_expert] -= 1
                counts[new_expert] += 1
                selected_mass = next_mass
                changed_slots += 1
                touched_tokens.add(token)
                changed = True
            if not changed:
                break

        all_topk_idx = torch.tensor(selected, dtype=torch.long, device="cuda")
        topk_idx = all_topk_idx[local_offset:local_offset + num_tokens].contiguous()
        topk_weights = torch.softmax(torch.gather(scores, 1, topk_idx), dim=-1)
        mass_drop = (original_mass - selected_mass) / max(original_mass, 1e-12)
        over_slots = sum(max(0, c - expert_capacity) for c in counts)
        max_count = max(counts)
        if rank_idx == 0:
            dist_print(
                f" > global_bounded_repair: budget={budget:.3f} "
                f"changed={changed_slots / max(total_num_tokens * num_topk, 1) * 100:.1f}% "
                f"touched={len(touched_tokens) / max(total_num_tokens, 1) * 100:.1f}% "
                f"mass_drop={mass_drop * 100:.1f}% over_slots={over_slots} max_count={max_count}",
                once_in_node=False,
            )
    elif args.routing_mode in ("topk-repair", "topk-repair-one", "topk-repair-softmax"):
        assert args.masked_ratio == 0.0, f"{args.routing_mode} routing does not support masked_ratio"
        assert (num_tokens * num_topk) % num_experts == 0, f"{args.routing_mode} requires exact local expert capacity"
        scores = make_scores()
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        expert_capacity = (num_tokens * num_topk) // num_experts
        scores_cpu = scores.cpu()
        selected = topk_idx.cpu().tolist()
        selected_sets = [set(row) for row in selected]
        counts = [0] * num_experts
        for row in selected:
            for expert in row:
                counts[expert] += 1

        while True:
            overflow = {e for e, c in enumerate(counts) if c > expert_capacity}
            underfull = [e for e, c in enumerate(counts) if c < expert_capacity]
            if not overflow:
                break
            candidates = []
            underfull_set = set(underfull)
            for token, row in enumerate(selected):
                token_set = selected_sets[token]
                for slot, old_expert in enumerate(row):
                    if old_expert not in overflow:
                        continue
                    old_score = float(scores_cpu[token, old_expert].item())
                    for new_expert in underfull:
                        if new_expert in token_set:
                            continue
                        loss = old_score - float(scores_cpu[token, new_expert].item())
                        candidates.append((loss, token, slot, old_expert, new_expert))
            assert candidates, "topk-repair could not find a repair candidate"
            candidates.sort(key=lambda x: x[0])
            changed = False
            for _, token, slot, old_expert, new_expert in candidates:
                if counts[old_expert] <= expert_capacity or counts[new_expert] >= expert_capacity:
                    continue
                if selected[token][slot] != old_expert or new_expert in selected_sets[token]:
                    continue
                selected[token][slot] = new_expert
                selected_sets[token].remove(old_expert)
                selected_sets[token].add(new_expert)
                counts[old_expert] -= 1
                counts[new_expert] += 1
                changed = True
            assert changed, "topk-repair made no progress"
        assert all(c == expert_capacity for c in counts), "topk-repair failed to reach exact capacity"
        topk_idx = torch.tensor(selected, dtype=torch.long, device="cuda")
        if args.routing_mode == "topk-repair-one":
            topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float, device="cuda")
        elif args.routing_mode == "topk-repair-softmax":
            topk_weights = torch.softmax(torch.gather(scores, 1, topk_idx), dim=-1)
        else:
            topk_weights = torch.gather(scores, 1, topk_idx)
    elif args.routing_mode in ("local-exact-score", "local-exact-score-one"):
        assert args.masked_ratio == 0.0, f"{args.routing_mode} routing does not support masked_ratio"
        assert (num_tokens * num_topk) % num_experts == 0, f"{args.routing_mode} requires exact local expert capacity"
        scores = make_scores()
        expert_capacity = (num_tokens * num_topk) // num_experts
        assert expert_capacity % num_topk == 0, f"{args.routing_mode} requires per-slot expert capacity"
        per_slot_capacity = expert_capacity // num_topk
        score_order = torch.argsort(scores, dim=1, descending=True).cpu().tolist()
        selected = [[-1] * num_topk for _ in range(num_tokens)]
        selected_sets = [set() for _ in range(num_tokens)]

        import sys
        sys.setrecursionlimit(max(10000, num_tokens * 4))
        for slot in range(num_topk):
            assignment = [-1] * num_tokens
            matched_tokens = [[] for _ in range(num_experts)]
            token_order = sorted(
                range(num_tokens),
                key=lambda t: scores[t, score_order[t][0]].item(),
                reverse=True,
            )

            def try_assign(token: int, seen_experts: set[int]) -> bool:
                for expert in score_order[token]:
                    if expert in selected_sets[token] or expert in seen_experts:
                        continue
                    seen_experts.add(expert)
                    if len(matched_tokens[expert]) < per_slot_capacity:
                        matched_tokens[expert].append(token)
                        assignment[token] = expert
                        return True
                    for idx, other_token in enumerate(list(matched_tokens[expert])):
                        if try_assign(other_token, seen_experts):
                            matched_tokens[expert][idx] = token
                            assignment[token] = expert
                            return True
                return False

            for token in token_order:
                assert try_assign(token, set()), "local-exact-score matching failed"
            assert all(expert >= 0 for expert in assignment), "local-exact-score left unassigned tokens"
            assert all(len(tokens) == per_slot_capacity for tokens in matched_tokens), "local-exact-score left capacity imbalance"
            for token, expert in enumerate(assignment):
                selected[token][slot] = expert
                selected_sets[token].add(expert)

        topk_idx = torch.tensor(selected, dtype=torch.long, device="cuda")
        if args.routing_mode == "local-exact-score-one":
            topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float, device="cuda")
        else:
            topk_weights = torch.gather(scores, 1, topk_idx)
    elif args.routing_mode == "local-capacity":
        assert args.masked_ratio == 0.0, "local-capacity routing does not support masked_ratio"
        scores = make_scores()
        expert_capacity = math.ceil(num_tokens * num_topk / num_experts)
        candidate_k = num_experts
        cand_vals, cand_idx = torch.topk(scores, candidate_k, dim=-1, largest=True, sorted=True)
        order = torch.argsort(cand_vals[:, 0], descending=True).cpu().tolist()
        cand_idx_cpu = cand_idx.cpu().tolist()
        remaining = [expert_capacity] * num_experts
        selected = [[-1] * num_topk for _ in range(num_tokens)]
        pending: list[tuple[int, int]] = []
        for token in order:
            used = 0
            for expert in cand_idx_cpu[token]:
                if remaining[expert] > 0:
                    selected[token][used] = expert
                    remaining[expert] -= 1
                    used += 1
                    if used == num_topk:
                        break
            if used < num_topk:
                pending.append((token, used))
        fallback_cursor = 0
        for token, used in pending:
            already = set(selected[token][:used])
            while used < num_topk:
                found = False
                for _ in range(num_experts):
                    expert = fallback_cursor % num_experts
                    fallback_cursor += 1
                    if remaining[expert] > 0 and expert not in already:
                        selected[token][used] = expert
                        remaining[expert] -= 1
                        already.add(expert)
                        used += 1
                        found = True
                        break
                if not found:
                    for expert in cand_idx_cpu[token]:
                        if expert not in already:
                            selected[token][used] = expert
                            already.add(expert)
                            used += 1
                            found = True
                            break
                assert found, "local-capacity routing could not fill all topk slots"
        topk_idx = torch.tensor(selected, dtype=torch.long, device="cuda")
        topk_weights = torch.gather(scores, 1, topk_idx)
    elif args.routing_mode == "capacity":
        assert args.masked_ratio == 0.0, "capacity routing does not support masked_ratio"
        scores = make_scores()
        all_scores = uneven_all_gather(scores, group=group)
        local_num_tokens = torch.tensor([num_tokens], dtype=torch.long, device="cuda")
        all_num_tokens_t = [torch.zeros_like(local_num_tokens) for _ in range(num_ranks)]
        dist.all_gather(all_num_tokens_t, local_num_tokens, group=group)
        all_num_tokens = [int(x.item()) for x in all_num_tokens_t]
        local_offset = sum(all_num_tokens[:rank_idx])
        total_num_tokens = sum(all_num_tokens)
        expert_capacity = math.ceil(total_num_tokens * num_topk / num_experts)
        candidate_k = num_experts
        cand_vals, cand_idx = torch.topk(all_scores, candidate_k, dim=-1, largest=True, sorted=True)
        order = torch.argsort(cand_vals[:, 0], descending=True).cpu().tolist()
        cand_idx_cpu = cand_idx.cpu().tolist()
        remaining = [expert_capacity] * num_experts
        selected = [[-1] * num_topk for _ in range(total_num_tokens)]
        pending: list[tuple[int, int]] = []
        for token in order:
            used = 0
            for expert in cand_idx_cpu[token]:
                if remaining[expert] > 0:
                    selected[token][used] = expert
                    remaining[expert] -= 1
                    used += 1
                    if used == num_topk:
                        break
            if used < num_topk:
                pending.append((token, used))
        fallback_cursor = 0
        for token, used in pending:
            already = set(selected[token][:used])
            while used < num_topk:
                found = False
                for _ in range(num_experts):
                    expert = fallback_cursor % num_experts
                    fallback_cursor += 1
                    if remaining[expert] > 0 and expert not in already:
                        selected[token][used] = expert
                        remaining[expert] -= 1
                        already.add(expert)
                        used += 1
                        found = True
                        break
                if not found:
                    for expert in cand_idx_cpu[token]:
                        if expert not in already:
                            selected[token][used] = expert
                            already.add(expert)
                            used += 1
                            found = True
                            break
                assert found, "capacity routing could not fill all topk slots"
        if rank_idx == 0:
            probs_cpu = torch.softmax(all_scores.cpu(), dim=1)
            natural_mass = 0.0
            selected_mass = 0.0
            changed_slots = 0
            touched_tokens = 0
            for token, row in enumerate(selected):
                natural = cand_idx_cpu[token][:num_topk]
                natural_set = set(natural)
                row_set = set(row)
                overlap = len(natural_set & row_set)
                changed_slots += num_topk - overlap
                touched_tokens += overlap != num_topk
                natural_mass += sum(float(probs_cpu[token, expert].item()) for expert in natural)
                selected_mass += sum(float(probs_cpu[token, expert].item()) for expert in row)
            mass_drop = (natural_mass - selected_mass) / max(natural_mass, 1e-12)
            used_counts = [expert_capacity - r for r in remaining]
            dist_print(
                f" > capacity_quality: changed={changed_slots / max(total_num_tokens * num_topk, 1) * 100:.1f}% "
                f"touched={touched_tokens / max(total_num_tokens, 1) * 100:.1f}% "
                f"mass_drop={mass_drop * 100:.1f}% max_count={max(used_counts)}",
                once_in_node=False,
            )
        all_topk_idx = torch.tensor(selected, dtype=torch.long, device="cuda")
        topk_idx = all_topk_idx[local_offset:local_offset + num_tokens].contiguous()
        topk_weights = torch.gather(scores, 1, topk_idx)
    else:
        scores = make_scores()
        topk_weights, topk_idx = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False
        )
    topk_idx = apply_eplb_replicas(topk_idx)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    # 累计接收统计：fused 与 baseline 各持一份避免相互覆盖
    phase_profile_enabled = os.environ.get("DG_SM90_MOE_PHASE_PROFILE", "0") not in ("", "0")
    phase_profile_extra = 64 if phase_profile_enabled else 0
    cum_stats_fused = torch.zeros(
        (num_experts_per_rank + phase_profile_extra,), dtype=torch.int, device="cuda"
    )
    cum_stats_baseline = cum_stats_fused.clone()

    # ---- BF16 → FP8 量化 ----
    # x_fp8 是元组：(token_fp8 (M, hidden), token_sf (M, hidden//128) fp32 行主序)
    # 注意 use_ue8m0=False, use_packed_ue8m0=False：SM90 不接受 UE8M0 packed SF
    x_fp8 = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )

    # weight 量化：(G, N, K) bf16 → ((G, N, K) fp8 e4m3fn, (G, N//128, K//128) fp32 SF)
    # baseline（DeepEP grouped GEMM）直接用这两个未变换的元组
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_weights_bf16)
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_weights_bf16)

    # fused 路径：FP8 weight 上做 gate/up gran-8 N-轴 interleave；SF 不变
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        l1_weights, l2_weights
    )

    # SwiGLU clamp：finite → 传给 fused/triton；inf → None（关闭 clamp，与 SM90 fused 一致）
    clamp_arg = args.activation_clamp if math.isfinite(args.activation_clamp) else None
    run_baseline_enabled = args.run_baseline or bool(args.check_output_diff)

    # ---- DeepGEMM grouped GEMM 的 M 维 alignment（baseline 走 DeepEP 时也用这个）----
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    # ---- 分配 fused 的 SymmBuffer 与输出 buffer ----
    sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    use_eplb_hint = bool(eplb_replica_for)
    use_skew_hint = args.score_powerlaw_alpha > 0.0
    use_masked_hint = args.masked_ratio > 0.0

    def run_fused():
        # NOTE: 跟 SM100 test_mega_moe.py 的处理一致 —— DG_COMM_KERNEL_DEBUG=1 时
        # kernel 出口会把 sym_buffer 整块清零，所以每次都要重新拷输入
        sym_buffer.x[:num_tokens].copy_(x_fp8[0])
        sym_buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

        old_eplb_hint = os.environ.get("DG_SM90_MOE_EPLB_HINT")
        old_skew_hint = os.environ.get("DG_SM90_MOE_SKEW_HINT")
        old_masked_hint = os.environ.get("DG_SM90_MOE_MASKED_HINT")
        if use_eplb_hint:
            os.environ["DG_SM90_MOE_EPLB_HINT"] = "1"
        if use_skew_hint:
            os.environ["DG_SM90_MOE_SKEW_HINT"] = "1"
        if use_masked_hint:
            os.environ["DG_SM90_MOE_MASKED_HINT"] = "1"
        try:
            deep_gemm.fp8_mega_moe(
                y_fused,
                transformed_l1,
                transformed_l2,
                sym_buffer,
                cumulative_local_expert_recv_stats=cum_stats_fused,
                recipe=(128, 128, 128),
                activation="swiglu",
                activation_clamp=clamp_arg,
                fast_math=bool(args.fast_math),
            )
        finally:
            if use_eplb_hint:
                if old_eplb_hint is None:
                    os.environ.pop("DG_SM90_MOE_EPLB_HINT", None)
                else:
                    os.environ["DG_SM90_MOE_EPLB_HINT"] = old_eplb_hint
            if use_skew_hint:
                if old_skew_hint is None:
                    os.environ.pop("DG_SM90_MOE_SKEW_HINT", None)
                else:
                    os.environ["DG_SM90_MOE_SKEW_HINT"] = old_skew_hint
            if use_masked_hint:
                if old_masked_hint is None:
                    os.environ.pop("DG_SM90_MOE_MASKED_HINT", None)
                else:
                    os.environ["DG_SM90_MOE_MASKED_HINT"] = old_masked_hint
        return y_fused

    # ---- 打印 config ----
    dist_print("Config (H200 fused mega-MoE):", once_in_node=True)
    dist_print(f" > Tokens: {num_tokens}/{num_max_tokens_per_rank}", once_in_node=True)
    dist_print(
        f" > Hidden: {hidden}, Intermediate: {intermediate_hidden}", once_in_node=True
    )
    dist_print(
        f" > Experts: {num_topk}/{num_experts} (per-rank: {num_experts_per_rank})",
        once_in_node=True,
    )
    dist_print(f" > Masked ratio: {args.masked_ratio}", once_in_node=True)
    dist_print(
        f" > Activation SF: fused L2 per-{FUSED_L2_ACT_SF_GRAN} UE8M0, "
        f"baseline L2 per-{BASELINE_L2_ACT_SF_GRAN} UE8M0 "
        f"(SM90 grouped GEMM constraint)",
        once_in_node=True,
    )
    dist_print(
        f" > Baseline: {'enabled' if run_baseline_enabled else 'disabled'}",
        once_in_node=True,
    )
    dist_print(
        f" > Buffer: {sym_buffer.buffer.nbytes / 2**30:.3f} GiB", once_in_node=True
    )
    dist_print(once_in_node=True)

    # 与社区版 test_mega_moe.py 对齐：NCU 模式只跑 fused kernel，避免 baseline 噪声。
    if args.ncu_profile_only:
        dist_print("Run fused SM90 mega-MoE kernel:", once_in_node=True)
        y = run_fused()
        torch.cuda.synchronize()
        assert y.shape == (num_tokens, hidden) and y.dtype == torch.bfloat16
        dist_print(" > Done, exiting", once_in_node=True)
        dist.barrier()
        sym_buffer.destroy()
        dist.destroy_process_group()
        return

    # ---- 分配 DeepEP buffer（baseline 用）----
    deep_ep = _import_deep_ep() if run_baseline_enabled else None
    ep_buffer = None
    if deep_ep is not None:
        ep_buffer = _make_deep_ep_buffer(
            deep_ep,
            group,
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            sym_buffer.buffer.nbytes,
        )

    # ----------------------------------------------------------------
    # baseline 主体：dispatch → L1 GEMM → SwiGLU+量化 → L2 GEMM → combine
    # 与 fused 用同一份 (FP8 weight, FP32 block-(128,128) SF) —— 但是 **未变换**
    # 的版本（baseline grouped GEMM 不需要 gate/up interleave）
    # ----------------------------------------------------------------
    def run_baseline():
        recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
            x_fp8,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cum_stats_baseline,
            num_experts=num_experts,
            expert_alignment=alignment,
            do_cpu_sync=False,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=False,  # SM90: row-major float SF
        )
        n = recv_x[0].size(0)

        # L1 GEMM：FP8 token @ FP8 W1 → BF16 中间激活 (gate||up 拼接)
        l1_y = torch.empty(
            (n, intermediate_hidden * 2), dtype=torch.bfloat16, device="cuda"
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            recv_x,
            l1_weights,
            l1_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )

        # Triton SwiGLU + FP8 量化（含 topk 权重乘法）
        # 注意：fused SM90 mega-MoE 的 L2 activation SFA 是 per-64-K；
        # 当前 DeepGEMM SM90 grouped GEMM 只支持 per-128-K SFA，所以性能 baseline
        # 只能用 per-128-K，但 scale 数值采用 fused 同款 UE8M0/power-of-two。
        l1_y = swiglu_apply_weight_to_fp8_triton(
            x=l1_y,
            topk_weights=recv_topk_weights,
            clamp_value=clamp_arg,
            num_per_channels=BASELINE_L2_ACT_SF_GRAN,
            use_ue8m0_scale=True,
        )

        # L2 GEMM：FP8 中间激活 @ FP8 W2 → BF16
        l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device="cuda")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            l1_y,
            l2_weights,
            l2_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            disable_ue8m0_cast=True,
        )

        # DeepEP combine：把每个 token 在 topk 个 expert 上的输出汇聚回源 rank
        return ep_buffer.combine(l2_y, handle=handle)[0]

    # ---- 跑一次确保不报错（fused + 可选 baseline）----
    y = run_fused()
    assert y.shape == (num_tokens, hidden) and y.dtype == torch.bfloat16, (
        f"fused 输出 shape/dtype 异常: shape={y.shape}, dtype={y.dtype}"
    )
    if args.check_split_vs_one_kernel:
        old_split_env = os.environ.get("DG_SM90_MOE_SPLIT_L1_L2")
        try:
            os.environ["DG_SM90_MOE_SPLIT_L1_L2"] = "1"
            y_split = run_fused().detach().clone()
            torch.cuda.synchronize()
            os.environ["DG_SM90_MOE_SPLIT_L1_L2"] = "0"
            y_one_kernel = run_fused().detach().clone()
            torch.cuda.synchronize()
        finally:
            if old_split_env is None:
                os.environ.pop("DG_SM90_MOE_SPLIT_L1_L2", None)
            else:
                os.environ["DG_SM90_MOE_SPLIT_L1_L2"] = old_split_env
        diff = (y_split.float() - y_one_kernel.float()).abs()
        denom = y_one_kernel.float().abs().mean().clamp_min(1e-12)
        dist_print(
            "Output diff (split two-kernel vs one-kernel):", once_in_node=True
        )
        dist_print(
            f" > max_abs={diff.max().item():.6e}, "
            f"mean_abs={diff.mean().item():.6e}, "
            f"mean_abs/mean_ref={diff.mean().div(denom).item():.6e}",
            once_in_node=True,
        )
        dist_print(once_in_node=True)
    if ep_buffer is not None:
        out_b = run_baseline()
        assert out_b.shape == (num_tokens, hidden) and out_b.dtype == torch.bfloat16, (
            f"baseline 输出 shape/dtype 异常: shape={out_b.shape}, dtype={out_b.dtype}"
        )
        if args.check_output_diff:
            diff = (y.float() - out_b.float()).abs()
            denom = out_b.float().abs().mean().clamp_min(1e-12)
            dist_print(
                "Output diff (fused vs legacy-per128 baseline):", once_in_node=True
            )
            dist_print(
                f" > max_abs={diff.max().item():.6e}, "
                f"mean_abs={diff.mean().item():.6e}, "
                f"mean_abs/mean_ref={diff.mean().div(denom).item():.6e}",
                once_in_node=True,
            )
            dist_print(once_in_node=True)

    # ---- 统计本 rank 实际接收的 token 数与触达的 expert 数 ----
    # 把所有 rank 的 topk_idx 收齐，再把不落在本 rank 持有 expert 范围内的条目
    # 标成 -1；剩下的非 -1 条目数即"被路由进本 rank 的 (token, slot) 总数"。
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    all_routed_topk_idx = gathered_topk_idx
    local_num_tokens_t = torch.tensor([num_tokens], dtype=torch.long, device="cuda")
    all_num_tokens_t = [torch.zeros_like(local_num_tokens_t) for _ in range(num_ranks)]
    dist.all_gather(all_num_tokens_t, local_num_tokens_t, group=group)
    all_num_tokens = [int(x.item()) for x in all_num_tokens_t]
    peer_recv_counts = []
    row_start = 0
    for src_tokens in all_num_tokens:
        src_topk = all_routed_topk_idx[row_start:row_start + src_tokens]
        peer_recv_counts.append(int(((src_topk >= rank_idx * num_experts_per_rank) &
                                     (src_topk < (rank_idx + 1) * num_experts_per_rank)).sum().item()))
        row_start += src_tokens
    max_peer_recv = max(peer_recv_counts) if peer_recv_counts else 0
    gathered_topk_idx = all_routed_topk_idx.clone()
    gathered_topk_idx[
        (gathered_topk_idx < rank_idx * num_experts_per_rank)
        | (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)
    ] = -1
    local_expert_ids = gathered_topk_idx[gathered_topk_idx != -1]
    num_recv_tokens = int(local_expert_ids.numel())
    num_touched_experts = int(torch.unique(local_expert_ids).numel())
    if num_recv_tokens > 0:
        local_counts = torch.bincount(
            local_expert_ids - rank_idx * num_experts_per_rank,
            minlength=num_experts_per_rank,
        )
        num_m_tiles = int(((local_counts + 63) // 64).sum().item())
        max_expert_tokens = int(local_counts.max().item())
    else:
        num_m_tiles = 0
        max_expert_tokens = 0

    # ---- benchmark ----
    # fused：bench_kineto 抓 sm90_fp8_mega_moe_impl 的 GPU 段（不含 host overhead）
    if phase_profile_enabled:
        cum_stats_fused.zero_()
    t_fused = bench_kineto(
        run_fused,
        SM90_KERNEL_NAME,
        num_tests=args.num_bench_tests,
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False)
        if ep_buffer is not None
        else dist.barrier(),
        trace_path=(
            f"{args.dump_profile_traces}/mega_moe_hopper_rank{rank_idx}.json"
            if args.dump_profile_traces
            else None
        ),
        with_multiple_kernels=os.environ.get(
            "DG_SM90_MOE_SPLIT_L1_L2",
            "1",
        ) != "0",
    )
    if phase_profile_enabled:
        cum_stats_fused.zero_()
        torch.cuda.synchronize()
        if ep_buffer is not None:
            ep_buffer.barrier(use_comm_stream=False)
        else:
            dist.barrier()
        phase_start = torch.cuda.Event(enable_timing=True)
        phase_end = torch.cuda.Event(enable_timing=True)
        phase_start.record()
        run_fused()
        phase_end.record()
        torch.cuda.synchronize()
        phase_event_us = phase_start.elapsed_time(phase_end) * 1000.0
        raw_i32 = cum_stats_fused[num_experts_per_rank:num_experts_per_rank + phase_profile_extra].detach().cpu().tolist()
        def _u64(slot: int) -> int:
            lo = raw_i32[slot * 2] & 0xffffffff
            hi = raw_i32[slot * 2 + 1] & 0xffffffff
            return lo | (hi << 32)
        names = (
            "dispatch_total",
            "dispatch_pull",
            "math_loop",
            "combine_barrier",
            "combine_reduce",
            "gemm_core",
            "l1_epilogue",
            "l2_epilogue",
        )
        num_profile_metrics = len(names)
        pieces = []
        for idx, name in enumerate(names):
            total = _u64(idx)
            max_cycles = _u64(num_profile_metrics + idx)
            count = _u64(2 * num_profile_metrics + idx)
            avg_us = (total / count / 1000.0) if count else 0.0
            max_us = max_cycles / 1000.0
            pieces.append(f"{name}:avg={avg_us:.1f}us,max={max_us:.1f}us,n={count},ns={total}/{max_cycles}")
        dist_print(f" > phase_profile rank={rank_idx}: event={phase_event_us:.1f}us; " + "; ".join(pieces))
    # baseline：cuda events 中位数（tilelang.do_bench 在 H200 不一定有，统一用 events）
    t_baseline = (
        _bench_cuda_events(
            run_baseline,
            num_warmup=args.num_warmup,
            num_repeat=args.num_repeat,
            l2_flush_gb=args.l2_flush_gb,
        )
        if ep_buffer is not None
        else 0.0
    )

    def safe_div(a, b):
        return float("nan") if b == 0 else a / b

    # 端到端 TFLOPS：3 个 matmul（L1 gate、L1 up、L2），每个 2*M*N*K，M=num_recv_tokens
    tflops = safe_div(
        2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused
    )

    # HBM 字节估算（SM90: weight 是 FP8 = 1B/elem，与 SM100 FP4=0.5B 不同）
    l1_weight_bytes = num_touched_experts * intermediate_hidden * 2 * hidden
    l2_weight_bytes = num_touched_experts * hidden * intermediate_hidden
    l1_weight_sf_bytes = (
        num_touched_experts
        * (intermediate_hidden * 2 // WEIGHT_SF_GRAN_MN)
        * (hidden // WEIGHT_SF_GRAN_K)
        * 4
    )
    l2_weight_sf_bytes = (
        num_touched_experts
        * (hidden // WEIGHT_SF_GRAN_MN)
        * (intermediate_hidden // WEIGHT_SF_GRAN_K)
        * 4
    )
    l1_input_sf_bytes = num_recv_tokens * (hidden // L1_ACT_SF_GRAN) * 4
    l2_act_sf_bytes = (
        num_recv_tokens * (intermediate_hidden // FUSED_L2_ACT_SF_GRAN) * 4
    )
    num_hbm_bytes = (
        l1_weight_bytes
        + l2_weight_bytes  # weights (FP8)
        + l1_weight_sf_bytes
        + l2_weight_sf_bytes  # weight SF (FP32)
        + num_recv_tokens * hidden
        + l1_input_sf_bytes  # L1 输入读 (FP8 + SF)
        + num_recv_tokens * intermediate_hidden
        + l2_act_sf_bytes  # L1 输出写 (FP8 + SF)
        + num_recv_tokens * intermediate_hidden
        + l2_act_sf_bytes  # L2 输入读 (FP8 + SF)
        + num_recv_tokens * hidden * 2  # L2 输出写 (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink 字节：dispatch 拉 token + input SF + topk weight，combine 写回 BF16
    num_nvlink_bytes = num_recv_tokens * (hidden + hidden // 32 + 4 + hidden * 2)
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # combine reduction 串行下界（解析估计；6.5e12 = HBM 串行 reduction 经验吞吐 B/s）
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # overlap 校正：扣掉 fused 中无法重叠的串行 reduction 段后估计稳态吞吐
    approx_factor = t_fused / max(t_fused - t_reduction, 1e-12)

    # baseline 用同一份 FLOPs / HBM 字节，时间换成 t_baseline
    tflops_baseline = safe_div(
        2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_baseline
    )
    hbm_gbs_baseline = safe_div(num_hbm_bytes / 1e9, t_baseline)
    nvlink_gbs_baseline = safe_div(num_nvlink_bytes / 1e9, t_baseline)

    def fmt_perf_line(
        name: str,
        t: float,
        compute_tflops: float,
        hbm_gbs_: float,
        nvlink_gbs_: float,
        reduction_us: float | None = None,
        speedup: float | None = None,
    ) -> str:
        reduction = f"{reduction_us:13.1f}" if reduction_us is not None else f"{'-':>13}"
        speedup_text = (
            f"{speedup:6.2f}x {'fused faster' if speedup > 1 else 'baseline faster'}"
            if speedup is not None else
            f"{'-':>21}"
        )
        return (
            f" > {name:<10} {rank_idx:2d}/{num_ranks:<2d} "
            f"{num_recv_tokens:12d} "
            f"{num_touched_experts:14d} {num_m_tiles:7d} {max_expert_tokens:8d} {max_peer_recv:8d} | "
            f"{compute_tflops:15.0f} "
            f"{hbm_gbs_:9.0f} "
            f"{nvlink_gbs_:9.0f} "
            f"{t * 1e6:9.0f} "
            f"{reduction} "
            f"{speedup_text}"
        )

    dist_print("Performance:", once_in_node=True)
    dist_print(
        " > kind       EP    recv_tokens active_experts m_tiles max_exp max_peer | "
        "compute(TFLOPS) HBM(GB/s) NVL(GB/s)  time(us) reduction(us) speedup",
        once_in_node=True,
    )
    dist_print(
        fmt_perf_line(
            "[fused]",
            t_fused,
            tflops * approx_factor,
            hbm_gbs * approx_factor,
            nvlink_gbs * approx_factor,
            reduction_us=t_reduction * 1e6,
        )
    )
    if ep_buffer is not None:
        speedup = safe_div(t_baseline, t_fused)
        dist_print(
            fmt_perf_line(
                "[baseline]",
                t_baseline,
                tflops_baseline,
                hbm_gbs_baseline,
                nvlink_gbs_baseline,
                speedup=speedup,
            )
        )
    else:
        reason = (
            "disabled; pass --run-baseline or --check-output-diff to compare"
            if not run_baseline_enabled
            else "deep_ep unavailable"
        )
        dist_print(f" > [baseline] ({reason})", once_in_node=True)

    # ---- 清理 ----
    dist.barrier()
    sym_buffer.destroy()
    if ep_buffer is not None:
        ep_buffer.destroy()
    dist.destroy_process_group()


# ============================================================================
# 模块 6：argparse + spawn
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H200 mega-MoE: fused (deep_gemm.fp8_mega_moe) vs DeepEP+grouped-FP8 baseline"
    )

    # 资源
    parser.add_argument(
        "--ncu-profile-only",
        action="store_true",
        help="只运行一次 fused SM90 kernel，便于 NCU/Nsight 采样",
    )
    parser.add_argument(
        "--num-processes", type=int, default=8, help="spawn 出来的进程数（一卡一进程）"
    )
    parser.add_argument(
        "--local-rank-idx",
        type=int,
        default=None,
        help="单进程模式的 local rank；用于外部 launcher/NCU 分别启动每个 rank",
    )

    # 模型形状
    # 注：SM90 fused kernel 要求 intermediate_hidden ≤ 4096
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=0,
        help="per-rank 实际 token 数；0 表示用 num-max-tokens-per-rank",
    )
    parser.add_argument(
        "--num-max-removed-tokens",
        type=int,
        default=0,
        help="num-tokens 为 0 时，每个 rank 随机移除的最大 token 数",
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument(
        "--intermediate-hidden",
        type=int,
        default=3072,
        help="中间层维度（≤ 4096，受 SM90 l2_arrival_mask 约束）",
    )
    parser.add_argument(
        "--activation-clamp",
        type=float,
        default=10.0,
        help="SwiGLU 前对 gate/up 的 clamp 阈值；传 inf 表示关闭",
    )
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="random",
        choices=("random", "balanced", "balanced-shuffled", "balanced-shuffled-score", "topk-repair", "topk-repair-one", "topk-repair-softmax", "topk-repair-budget-softmax", "global-repair-budget-softmax", "local-exact-score", "local-exact-score-one", "local-capacity", "capacity"),
        help="routing 构造方式；balanced/balanced-shuffled/topk-repair/local-exact-score/local-capacity/capacity 控制每 expert assignment 数",
    )
    parser.add_argument(
        "--masked-ratio",
        type=float,
        default=0.0,
        help="随机 mask 掉部分 topk expert selection，用于验证稀疏路由边界",
    )
    parser.add_argument(
        "--score-powerlaw-alpha",
        type=float,
        default=0.0,
        help="给 routing score 加 Zipf/power-law expert bias；0 表示关闭",
    )
    parser.add_argument(
        "--score-powerlaw-scale",
        type=float,
        default=1.0,
        help="power-law bias 的标准差尺度",
    )
    parser.add_argument(
        "--repair-mass-drop-budget",
        type=float,
        default=0.0,
        help="topk-repair-budget-softmax 的 aggregate selected softmax-mass drop 上限，例如 0.10",
    )
    parser.add_argument(
        "--repair-max-rounds",
        type=int,
        default=4,
        help="global-repair-budget-softmax 的 greedy repair 最大轮数，避免诊断代码超时",
    )
    parser.add_argument(
        "--num-redundant-experts",
        type=int,
        default=0,
        help="EPLB replica simulation: reserve physical expert slots as hot-expert replicas",
    )
    parser.add_argument(
        "--replica-dispatch",
        choices=("hash", "static"),
        default="hash",
        help="replica remap model: token-level hash or SGLang static source-rank approximation",
    )
    parser.add_argument(
        "--fast-math",
        type=int,
        default=1,
        help="fused 内 SwiGLU 是否启用 fast-math（0/1）",
    )

    # 测时
    parser.add_argument(
        "--num-bench-tests",
        type=int,
        default=30,
        help="bench_kineto 抓 fused 时的迭代数",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=5, help="baseline cuda events warmup"
    )
    parser.add_argument(
        "--num-repeat", type=int, default=20, help="baseline cuda events 测时迭代"
    )
    parser.add_argument(
        "--l2-flush-gb",
        type=float,
        default=8.0,
        help="baseline event 测时前用于 flush L2 的临时写入大小；0 表示关闭",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="启用 DeepEP+grouped-FP8 legacy baseline；默认关闭以避免 full-size 默认配置触发 baseline kernel 非法访问",
    )
    parser.add_argument(
        "--check-output-diff",
        type=int,
        default=0,
        help="非 0 时打印 fused 与 legacy-per128 baseline 的输出差异（预期非 bitwise）",
    )
    parser.add_argument(
        "--check-split-vs-one-kernel",
        type=int,
        default=0,
        help="非 0 时打印 split two-kernel 与 one-kernel fused 的输出差异",
    )
    parser.add_argument(
        "--dump-profile-traces",
        type=str,
        default="",
        help="非空时把 fused 的 Chrome trace 写到该目录（每 rank 一份）",
    )

    args = parser.parse_args()

    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # 单进程模式：由外部 launcher 分别设置 MASTER_ADDR/PORT/WORLD_SIZE/RANK。
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # 多进程启动：每个进程对应一个 GPU；test() 内部用 init_dist 建 NCCL group。
        torch.multiprocessing.spawn(
            test, args=(args.num_processes, args), nprocs=args.num_processes
        )
