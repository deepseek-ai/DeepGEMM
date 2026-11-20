"""Experimental vLLM 0.32.0 adapter; disabled unless SM120_VLLM_ENABLE=1.

Keep vLLM routing, shared experts, weight loading and native fallback. The
worker must serialize model forwards (no DBO, DP/EP or concurrent
independent streams). Each layer owns its scratch and output. CUDA Graphs are
supported after normal warmup; Python counters count dispatch/capture, not
Graph replays. This module is deliberately outside the default DeepGEMM API.
"""
import os
from functools import lru_cache

_registered = False


@lru_cache(maxsize=1)
def slot_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def prepare_routes(IDS, WEIGHTS, SLOTS, SAFE_WEIGHTS, E: tl.constexpr,
                       K: tl.constexpr, B: tl.constexpr):
        r = tl.arange(0, B)
        expert = tl.load(IDS + r, r < K, 0).to(tl.int64)
        valid = (expert >= 0) & (expert < E)
        # vLLM marks padding routes -1. Keep valid expert row zero intact;
        # padding gets distinct rows 1..K of expert zero (capacity is 128).
        slot = tl.where(valid, expert * 128, r + 1)
        weight = tl.load(WEIGHTS + r, r < K, 0)
        tl.store(SLOTS + r, slot, r < K)
        tl.store(SAFE_WEIGHTS + r, tl.where(valid, weight, 0.), r < K)

    return prepare_routes


def register():
    global _registered
    if _registered or os.environ.get("SM120_VLLM_ENABLE", "0") != "1":
        return
    from importlib.metadata import version

    if version("vllm") != "0.32.0":
        raise RuntimeError("The SM120 experiment requires exactly vLLM 0.32.0")

    import torch
    import triton
    from deep_gemm import _C
    from deep_gemm.small_moe_policy import select_single_token_config
    from sm120_tp_policy import select_adapter_config
    from vllm.config import get_current_vllm_config
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    logger = init_logger(__name__)

    make_slots = slot_kernel()

    @UnquantizedFusedMoEMethod.register_oot
    class SM120UnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
        def __init__(self, moe):
            super().__init__(moe)
            self.sm120_workspace = None
            self.sm120_dispatches = {"fused": 0, "native": 0, "capture_fused": 0}

        def process_weights_after_loading(self, layer):
            # Native conversion and backend selection remain authoritative.
            super().process_weights_after_loading(layer)
            self.sm120_workspace = None
            reason = self._support_reason(layer)
            if reason:
                logger.info_once("SM120 adapter retains native MoE: %s", reason)
                return
            e, twice_i, d = layer.w13_weight.shape
            i = twice_i // 2
            k = self.moe.experts_per_token
            config = select_adapter_config(1, e, k, d, i, self.moe.moe_parallel_config.tp_size)
            self.sm120_workspace = {
                "config": config,
                "tp_size": self.moe.moe_parallel_config.tp_size,
                "local_intermediate": i,
                "route_shape": (1, k),
                "direct_ids": os.environ.get("SM120_VLLM_DIRECT_IDS", "0") == "1",
                "h": torch.empty((e, 128, i), device=layer.w13_weight.device, dtype=torch.bfloat16),
                "y": torch.empty((1, d), device=layer.w13_weight.device, dtype=torch.bfloat16),
                "swapped": self.unquantized_backend == UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            }
            if not self.sm120_workspace["direct_ids"]:
                self.sm120_workspace["slots"] = torch.empty((1, k), device=layer.w13_weight.device, dtype=torch.int64)
                self.sm120_workspace["route_weights"] = torch.empty((1, k), device=layer.w13_weight.device, dtype=torch.float32)
            logger.info(
                "SM120 single-token adapter ready: E=%d D=%d I=%d k=%d native=%s L1=%s L2=%s",
                e, d, i, k, self.unquantized_backend, config.l1, config.l2,
            )

        def _support_reason(self, layer):
            if get_current_vllm_config().scheduler_config.async_scheduling:
                return "asynchronous scheduling is outside this experiment"
            if self.unquantized_backend not in (
                UnquantizedMoeBackend.TRITON,
                UnquantizedMoeBackend.FLASHINFER_CUTLASS,
            ):
                return "unhandled native weight layout/backend"
            pc = self.moe.moe_parallel_config
            if any(getattr(pc, key) != 1 for key in ("dp_size", "ep_size", "pcp_size", "sp_size")):
                return "parallel ranks are outside this experiment"
            if pc.tp_size != 1 and not (pc.tp_size == 2 and os.environ.get("SM120_VLLM_TP_ENABLE", "0") == "1"):
                return "TP research requires explicit opt-in and TP=2"
            if pc.enable_eplb or get_current_vllm_config().parallel_config.enable_dbo:
                return "EPLB/DBO is outside this experiment"
            if (self.moe.has_bias or self.moe.is_lora_enabled or self.moe.has_hash_routing
                    or self.moe._defer_moe_finalize
                    or self.moe.activation != MoEActivation.SILU
                    or any(getattr(self.moe, key) is not None
                           for key in ("swiglu_limit", "swiglu_alpha", "swiglu_beta"))
                    or layer.apply_router_weight_on_input or layer.expert_map is not None):
                return "unsupported activation, scaling, bias or expert map"
            gu, down = layer.w13_weight, layer.w2_weight
            if not gu.is_cuda or gu.ndim != 3 or down.ndim != 3:
                return "non-CUDA or unexpected weights"
            if gu.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
                return "BF16 required"
            e, twice_i, d = gu.shape
            i = twice_i // 2
            if (pc.tp_size == 2 and
                    (self.moe.intermediate_size_per_partition_unpadded != i
                     or self.moe.hidden_dim_unpadded != d)):
                return "padded TP weights are outside this experiment"
            if (twice_i != 2*i or down.shape != (e, d, i) or not gu.is_contiguous()
                    or not down.is_contiguous() or gu.data_ptr() % 16 or down.data_ptr() % 16
                    or self.moe.num_experts != e or layer.global_num_experts != e):
                return "padded, partitioned or noncontiguous weights"
            props = torch.cuda.get_device_properties(gu.device)
            if (props.major, props.minor) != (12, 0) or "RTX 5090" not in props.name or props.multi_processor_count != 170:
                return "single-token policy requires RTX 5090 / 170 SMs"
            config = select_adapter_config(1, e, self.moe.experts_per_token, d, i, pc.tp_size)
            if config.l1 is None or config.l2 is None:
                return config.reason
            return None

        def forward_native(self, layer, x, topk_weights, topk_ids,
                           shared_experts, shared_experts_input):
            ws = self.sm120_workspace
            if (ws is None or x.shape != ws["y"].shape or x.dtype != torch.bfloat16
                    or not x.is_contiguous() or x.device != ws["y"].device
                    or x.data_ptr() % 16 or topk_weights.dtype != torch.float32
                    or topk_weights.shape != ws["route_shape"]
                    or topk_ids.shape != ws["route_shape"]
                    or topk_ids.dtype not in (torch.int32, torch.int64)
                    or not topk_ids.is_contiguous() or not topk_weights.is_contiguous()
                    or topk_weights.device != x.device or topk_ids.device != x.device
                    or topk_weights.data_ptr() % 16):
                self.sm120_dispatches["native"] += 1
                return super().forward_native(
                    layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input,
                )
            capturing = torch.cuda.is_current_stream_capturing()
            self.sm120_dispatches["capture_fused" if capturing else "fused"] += 1
            k = topk_ids.shape[1]
            if ws["direct_ids"]:
                routes, route_weights, options = topk_ids, topk_weights, {"expert_ids": True}
            else:
                make_slots[(1,)](topk_ids, topk_weights, ws["slots"], ws["route_weights"],
                                 layer.w13_weight.shape[0], k, triton.next_power_of_2(k), num_warps=1)
                routes, route_weights, options = ws["slots"], ws["route_weights"], {}
            _C._sm120_moe_small(
                x, layer.w13_weight, routes, route_weights, ws["h"],
                True, *ws["config"].l1, gate_up_swapped=ws["swapped"], **options,
            )
            # The runner launches NO_OVERLAP / MULTI_STREAM shared experts.
            # A native backend can instead delegate this order to its kernel;
            # honor that delegation exactly once when replacing that kernel.
            if shared_experts is not None:
                shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
            _C._sm120_moe_small(
                ws["h"], layer.w2_weight, routes, route_weights, ws["y"],
                False, *ws["config"].l2, **options,
            )
            # This is the rank-local partial result. vLLM's MoERunner owns the
            # existing TP all-reduce, including capture and shared-expert order.
            return ws["y"]

    _registered = True
    logger.info("Registered opt-in SM120 adapter for vLLM 0.32.0")
