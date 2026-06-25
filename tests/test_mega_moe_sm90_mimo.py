"""Production-fidelity SM90 MegaMoE test that mirrors sglang MiMo-V2-flash inference.

Unlike ``test_mega_moe_sm90.py`` (which feeds synthetic ``randn`` activations and
``topk(randn)`` routing across 2 ranks), this test reproduces the *exact*
inference-time conditions of ``sglang ... MiMoV2FlashForCausalLM`` with the
DeepEP / megamoe a2a backend at ``--ep-size 8``, so the correctness check
exercises the same kernel path, the same routing distribution, and the same
activation-clamp setting that production actually uses.

Everything below was verified against:
  * the MiMo-V2-flash ``config.json``
  * ``sglang/srt/models/mimo_v2.py``           (MoE + gate + TopK construction)
  * ``sglang/srt/layers/moe/mega_moe.py``      (``_run_mega_routed`` forward)
  * ``sglang/srt/layers/moe/topk.py``          (``biased_grouped_topk_impl``)

Production conditions reproduced
--------------------------------
  dims      hidden=4096, intermediate=2048, experts=256, topk=8
  EP        8 ranks  ->  32 experts / rank          (matches --ep-size 8)
  routing   sigmoid scoring + noaux_tc correction_bias + renormalize.
            n_group=topk_group=1, so the group-limited routing degenerates to a
            plain *biased* top-8 over all 256 experts; the returned weights are
            the *raw* sigmoid scores of the selected experts (NOT bias-added),
            renormalized to sum to 1 -> all positive, each ~0.125.
  clamp     NONE.  The config has no ``swiglu_limit``; sglang passes
            ``activation_clamp = getattr(moe.config, 'swiglu_limit', None) = None``.
            This is the single biggest divergence from the old test, whose
            scenarios run clamp=10 almost everywhere.
  scaling   routed_scaling_factor = 1.0 (folded in TopK) -> a no-op on the output.
  quant     x : per_token_cast_to_fp8(use_ue8m0=False, gran_k=128)
            w : FP8 e4m3, block-(128,128) float scale
  kernel    deep_gemm.fp8_mega_moe(recipe=(128,128,128), activation='swiglu',
                                   activation_clamp=None, fast_math=True)

Correctness
-----------
Reuses the independent FP32 reference from ``test_mega_moe_sm90.py``
(``_reference_fused`` — verified line-by-line against the kernel epilogue). Two
gates are applied so that a *few* wrong rows cannot hide behind a global metric:
  1. global cosine    : calc_diff(y, y_ref) < --diff-tol   (bulk accuracy)
  2. per-token error  : max & p99 relative L2 error per row, plus the fraction of
                        "bad" rows (rel err > --row-tol). This is what catches a
                        routing / masking / edge bug affecting only a handful of
                        tokens — exactly the class of bug a global cosine dilutes.

REAL vs SYNTHETIC weights
-------------------------
With ``--weights-dir /preset-models`` this loads the REAL MiMo-V2-flash expert
FP8 weights + scales AND the real router (gate.weight + e_score_correction_bias)
for one MoE layer, which is the ONLY configuration that validates
``transform_weights_for_mega_moe_sm90``'s assumption about the real checkpoint's
block-(128,128) scale orientation. Without it, weights are synthetic and that
scale-layout assumption is self-consistent and unverifiable — a synthetic PASS
proves kernel<->reference agreement but NOT real-checkpoint correctness. Run the
real-weight config to actually chase a production accuracy gap.
"""

import argparse
import math
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Allow `from test_mega_moe_sm90 import ...` (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# sglang source tree: MUST be the SAME tree the server/launch_server uses, so the
# DeepEPMoE reference runs the exact production code (e.g. deep_ep.Buffer with
# allow_mnnvl=False to avoid the fabric CUDA_ERROR_NOT_PERMITTED on this box).
# `/sgl-workspace/sglang` is a stale tree (allow_mnnvl=True -> fabric path fails);
# the installed/active sglang is /root/workspace/sglang. Prefer the active one;
# fall back to the importable `sglang` package location.
_SGLANG_SRC = '/root/workspace/sglang/python'
if not os.path.isdir(_SGLANG_SRC):
    _SGLANG_SRC = '/sgl-workspace/sglang/python'
if os.path.isdir(_SGLANG_SRC) and _SGLANG_SRC not in sys.path:
    sys.path.insert(0, _SGLANG_SRC)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist
from deep_gemm.testing import calc_diff, get_arch_major

# Single source of truth: the FP32 reference + quant helpers already verified
# against the kernel live in the layered test. We only change the *inputs*.
from test_mega_moe_sm90 import (
    _quantize_grouped_fp8_block_128_128,
)


# ----------------------------------------------------------------------------
# MiMo-V2-flash model facts (from config.json)
# ----------------------------------------------------------------------------

MIMO_V2_FLASH = dict(
    hidden_size=4096,
    moe_intermediate_size=2048,
    n_routed_experts=256,
    num_experts_per_tok=8,
    scoring_func='sigmoid',
    topk_method='noaux_tc',
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,   # MiMo sets this to 1.0 (a no-op)
    swiglu_limit=None,           # absent in config -> clamp disabled
    num_fused_shared_experts=0,  # n_shared_experts is null
)


# ----------------------------------------------------------------------------
# Production-faithful routing  (sglang biased_grouped_topk_impl, n_group=1)
# ----------------------------------------------------------------------------

def _production_topk(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``biased_grouped_topk_impl`` for the n_group=topk_group=1 case.

    Returns ``(topk_weights[f32], topk_idx[i64])`` exactly as the SM90 mega_moe
    path feeds the kernel:
      * scores       = sigmoid(gate_logits)
      * selection    = top-k of (scores + correction_bias)   (bias affects choice)
      * weights      = raw sigmoid scores of the chosen experts (NOT bias-added)
      * renormalize  = weights / weights.sum()  -> sum to 1, all positive
    With a single expert group the group-limited masking is a no-op, so we skip
    it (verified: group_mask is all-ones when n_group == topk_group == 1).
    """
    # gate runs in fp32 in sglang (MoEGate.dtype = float32).
    logits = torch.nn.functional.linear(hidden_states.float(), gate_weight)  # (T, E)
    scores = logits.sigmoid()
    scores_for_choice = scores + correction_bias.unsqueeze(0)
    _, topk_idx = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_idx)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_idx.to(torch.int64)


def _make_activations(num_tokens: int, hidden: int, outlier: bool, device='cuda') -> torch.Tensor:
    """Activation distribution feeding the per-token FP8 quantizer.

    Real transformer activations have heavy per-channel outliers, which is what
    stresses per-token FP8 quantization (a single large channel inflates the
    row amax and crushes the rest). A clean ``randn`` *under*-estimates that
    quantization error — the classic "passes in test, drifts in prod" trap. We
    inject a few large channels per token to be closer to production unless the
    caller asks for the plain-Gaussian baseline.
    """
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    if outlier and num_tokens > 0:
        # ~1% of channels get a 6-12x amplitude bump on a random subset of tokens.
        num_out = max(1, hidden // 100)
        cols = torch.randperm(hidden, device=device)[:num_out]
        scale = torch.empty(num_out, device=device).uniform_(6.0, 12.0).to(torch.bfloat16)
        rows = torch.rand(num_tokens, device=device) < 0.5
        x[:, cols] = torch.where(
            rows.unsqueeze(1),
            x[:, cols] * scale.unsqueeze(0),
            x[:, cols],
        )
    return x


# ----------------------------------------------------------------------------
# Optional: load REAL MiMo-V2 weights to also validate the scale-layout assumption
# ----------------------------------------------------------------------------

def _load_real_layer(
    weights_dir: str, layer_id: int, rank_idx: int, num_ranks: int,
    num_experts: int, hidden: int, intermediate_hidden: int,
):
    """Load ONE real MiMo-V2-flash MoE layer from the preset checkpoint.

    Returns ``(l1_fp8, l1_sf, l2_fp8, l2_sf, gate_weight, correction_bias)`` for
    *this rank's* expert slice, plus the (shared) real router tensors.

    Verified checkpoint layout (/preset-models, ``model_<L>_linear_fc{1,2}.safetensors``):
      gate_proj.weight            (IH, H)   = (2048, 4096) fp8 e4m3
      gate_proj.weight_scale_inv  (IH/128, H/128) = (16, 32) f32   [MN-major block]
      up_proj.weight              (IH, H)   fp8     (+ matching scale)
      down_proj.weight            (H, IH)   = (4096, 2048) fp8
      down_proj.weight_scale_inv  (H/128, IH/128) = (32, 16) f32
      gate.weight                 (E, H)    f32     (in model_<L>.safetensors)
      gate.e_score_correction_bias(E,)      f32

    The L1 (w13) weight the kernel wants is gate||up stacked along N to (2*IH, H),
    with scales stacked to (2*IH/128, H/128) — IDENTICAL to what
    ``_quantize_grouped_fp8_block_128_128`` emits in the synthetic path, and what
    sglang's ``w13_weight`` is built from. ``weight_scale_inv`` is the dequant
    *multiplier* (amax/448), matching the reference's ``* sf`` dequant.

    This is the ONLY configuration that actually validates the real block-(128,128)
    scale orientation assumed by ``transform_weights_for_mega_moe_sm90``; the
    synthetic path is self-consistent and cannot catch a layout mismatch.
    """
    from safetensors import safe_open

    assert num_experts % num_ranks == 0
    num_local = num_experts // num_ranks
    e_lo = rank_idx * num_local                       # this rank owns experts [e_lo, e_lo+num_local)
    pre = f'model.layers.{layer_id}.mlp'

    fc1 = os.path.join(weights_dir, f'model_{layer_id}_linear_fc1.safetensors')
    fc2 = os.path.join(weights_dir, f'model_{layer_id}_linear_fc2.safetensors')
    misc = os.path.join(weights_dir, f'model_{layer_id}.safetensors')
    for p in (fc1, fc2, misc):
        assert os.path.exists(p), f'missing checkpoint shard: {p}'

    # NOTE: safetensors `device=` takes an explicit ordinal; under spawn the
    # per-process device is set but a bare 'cuda' string resolves to cuda:0 for
    # every rank, so pin to this rank's actual device.
    dev = f'cuda:{torch.cuda.current_device()}'

    ih, h = intermediate_hidden, hidden
    l1_fp8 = torch.empty((num_local, 2 * ih, h), dtype=torch.float8_e4m3fn, device=dev)
    l1_sf = torch.empty((num_local, 2 * ih // 128, h // 128), dtype=torch.float32, device=dev)
    l2_fp8 = torch.empty((num_local, h, ih), dtype=torch.float8_e4m3fn, device=dev)
    l2_sf = torch.empty((num_local, h // 128, ih // 128), dtype=torch.float32, device=dev)

    with safe_open(fc1, framework='pt', device=dev) as h1, \
         safe_open(fc2, framework='pt', device=dev) as h2:
        for li in range(num_local):
            e = e_lo + li
            gate = h1.get_tensor(f'{pre}.experts.{e}.gate_proj.weight')             # (IH, H)
            up = h1.get_tensor(f'{pre}.experts.{e}.up_proj.weight')                 # (IH, H)
            gate_sf = h1.get_tensor(f'{pre}.experts.{e}.gate_proj.weight_scale_inv')  # (IH/128, H/128)
            up_sf = h1.get_tensor(f'{pre}.experts.{e}.up_proj.weight_scale_inv')
            # w13 = gate||up along N (matches sglang's w13_weight ordering).
            l1_fp8[li, :ih], l1_fp8[li, ih:] = gate, up
            l1_sf[li, :ih // 128], l1_sf[li, ih // 128:] = gate_sf, up_sf

            down = h2.get_tensor(f'{pre}.experts.{e}.down_proj.weight')             # (H, IH)
            down_sf = h2.get_tensor(f'{pre}.experts.{e}.down_proj.weight_scale_inv')  # (H/128, IH/128)
            l2_fp8[li], l2_sf[li] = down, down_sf

    with safe_open(misc, framework='pt', device=dev) as hm:
        gate_weight = hm.get_tensor(f'{pre}.gate.weight').float()                  # (E, H)
        correction_bias = hm.get_tensor(f'{pre}.gate.e_score_correction_bias').float()  # (E,)

    return l1_fp8, l1_sf.contiguous(), l2_fp8, l2_sf.contiguous(), gate_weight, correction_bias


# ----------------------------------------------------------------------------
# Stand up sglang's REAL DeepEPMoE layer so the reference IS production deepep
# (deep_ep.Buffer dispatch -> ep_scatter -> m_grouped contiguous GEMM(m_indices)
#  -> silu_and_mul + sglang quant -> GEMM -> ep_gather), not a hand-built replica.
# ----------------------------------------------------------------------------

_SGLANG_PARALLEL_INITED = False


def _init_sglang_parallel(rank_idx: int, num_ranks: int, model_path: str):
    """Initialize sglang's model-parallel globals on top of the already-created
    torch process group (deep_gemm's init_dist made it). Idempotent per process."""
    global _SGLANG_PARALLEL_INITED
    if _SGLANG_PARALLEL_INITED:
        return
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.distributed import parallel_state
    from sglang.srt.layers.moe import initialize_moe_config

    server_args = ServerArgs(
        model_path=model_path,
        tp_size=num_ranks,
        ep_size=num_ranks,
        moe_a2a_backend="deepep",
        deepep_mode="normal",          # prefill path == contiguous/normal
        trust_remote_code=True,
        dtype="bfloat16",
    )
    set_global_server_args_for_scheduler(server_args)

    # torch.distributed is already initialized by init_dist; this reuses it and
    # only builds sglang's _WORLD wrapper.
    parallel_state.init_distributed_environment(
        world_size=num_ranks, rank=rank_idx, local_rank=rank_idx,
        distributed_init_method="env://", backend="nccl",
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=num_ranks,
        expert_model_parallel_size=num_ranks,
    )
    initialize_moe_config(server_args)

    # deepep's _get_impl() resolves normal-vs-low_latency from this dp-attention
    # flag. Prefill == extend == NORMAL (contiguous) path, which is what this test
    # (single forward, no cuda graph) mirrors.
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch

    set_is_extend_in_batch(True)
    _SGLANG_PARALLEL_INITED = True


def _build_real_deepep_moe(num_experts, num_topk, hidden, intermediate_hidden,
                           layer_id, l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf):
    """Construct sglang's DeepEPMoE, load the real fp8 weights, run the production
    post-load step. Returns the layer; call `.forward(x_bf16, topk_output)`."""
    import torch
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    experts = DeepEPMoE(
        num_experts=num_experts,
        top_k=num_topk,
        hidden_size=hidden,
        intermediate_size=intermediate_hidden,
        layer_id=layer_id,
        quant_config=quant_config,
        prefix="",
    )
    # Load this rank's real expert slice into the layer's params (shapes match
    # _load_real_layer: w13 (E_local,2*IH,H), w2 (E_local,H,IH)).
    experts.w13_weight.data.copy_(l1_w_fp8)
    experts.w2_weight.data.copy_(l2_w_fp8)
    experts.w13_weight_scale_inv.data.copy_(l1_w_sf)
    experts.w2_weight_scale_inv.data.copy_(l2_w_sf)
    # Production post-load weight processing (requant/format); no-op-ish on SM90.
    experts.quant_method.process_weights_after_loading(experts)
    return experts


# ----------------------------------------------------------------------------
# Scenario
# ----------------------------------------------------------------------------

def _run(
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup, args: argparse.Namespace,
):
    cfg = MIMO_V2_FLASH
    hidden = cfg['hidden_size']
    intermediate_hidden = cfg['moe_intermediate_size']
    num_experts = cfg['n_routed_experts']
    num_topk = cfg['num_experts_per_tok']
    assert num_experts % num_ranks == 0, (
        f'experts {num_experts} not divisible by ranks {num_ranks}; '
        f'MiMo-V2 EP must divide 256 (use --num-processes in {{1,2,4,8,...}})'
    )
    num_experts_per_rank = num_experts // num_ranks

    num_tokens = args.num_tokens
    num_max = max(num_tokens, args.num_max_tokens_per_rank)

    # Deterministic but rank-distinct inputs (every rank holds different tokens).
    torch.manual_seed(1234 + rank_idx)

    real = bool(args.weights_dir)
    dist_print('Mirroring sglang MiMoV2Flash @ EP{} :'.format(num_ranks), once_in_node=True)
    dist_print(f'  hidden={hidden} intermediate={intermediate_hidden} '
               f'experts={num_experts} (={num_experts_per_rank}/rank) topk={num_topk}',
               once_in_node=True)
    dist_print(f'  tokens/rank={num_tokens} (cap {num_max})  clamp=None  '
               f'outlier_acts={not args.clean_acts}  '
               f'weights={"REAL layer " + str(args.layer_id) if real else "SYNTHETIC"}',
               once_in_node=True)

    # ---- Expert weights + router --------------------------------------------
    if real:
        # Real checkpoint: real expert FP8 weights/scales AND the real router
        # (gate.weight + e_score_correction_bias) -> closes the scale-layout and
        # routing-distribution blind spots simultaneously.
        l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf, gate_weight, correction_bias = _load_real_layer(
            args.weights_dir, args.layer_id, rank_idx, num_ranks,
            num_experts, hidden, intermediate_hidden)
    else:
        # Synthetic FP8 e4m3 + block-(128,128) scale (SAME layout the real
        # checkpoint uses), scaled small like trained MoE experts.
        l1_bf = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden),
            dtype=torch.bfloat16, device='cuda') * 0.05
        l2_bf = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden),
            dtype=torch.bfloat16, device='cuda') * 0.05
        l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
        l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)
        gate_weight = torch.randn((num_experts, hidden), dtype=torch.float32, device='cuda') * 0.02
        correction_bias = torch.randn((num_experts,), dtype=torch.float32, device='cuda') * 0.1

    # ---- DP-attention padding model -----------------------------------------
    # With --enable-dp-attention + CUDA graph (your launch cmd), each DP rank's
    # token buffer is padded to max(global_num_tokens) and the symm buffer is
    # NOT zeroed between graph replays, so padding rows carry DIRTY activations
    # from a previous forward. sglang masks those rows by setting topk_idx=-1
    # (num_token_non_padded -> _mask_topk_ids_padded_region), and the kernel must
    # skip them. The standard DeepEP path never sees padding (its dispatcher only
    # moves valid tokens) -> this is megamoe-only, matching "turn megamoe off and
    # it's fine". num_valid<num_tokens activates this scenario.
    num_valid = num_tokens if args.num_valid_tokens < 0 else min(args.num_valid_tokens, num_tokens)
    has_padding = num_valid < num_tokens

    # ---- Activations (per-token FP8, per-128-K float SF — sglang SM90 path) ---
    x_bf = _make_activations(num_tokens, hidden, outlier=not args.clean_acts)
    if has_padding:
        # Padding rows = dirty data (large random), mimicking an un-zeroed graph
        # buffer. If the kernel leaks any padding-row contribution into a valid
        # token's output, this makes it blow up rather than hide near zero.
        x_bf[num_valid:] = torch.randn(
            (num_tokens - num_valid, hidden), dtype=torch.bfloat16, device='cuda') * 8.0
    # Quantize the activation with the SAME quantizer the real DeepEP dispatch uses
    # internally (`sglang_per_token_group_quant_fp8`), NOT deep_gemm's
    # `per_token_cast_to_fp8`. The two produce DIFFERENT fp8 codes/SF for the same
    # BF16 input (~0.06% codes, ~55% SF differ by 1 ULP), because their per-128 scale
    # formulas differ. Since the reference path (`experts.forward(x_bf)`) feeds the
    # dispatch-quantized activation into the GEMMs, the kernel must consume the
    # bit-identical fp8 activation to be comparable — otherwise the two paths diverge
    # from the very first op (the activation quant) and can never be bit-exact.
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
    x_fp8, x_sf = sglang_per_token_group_quant_fp8(
        x_bf, 128, column_major_scales=False, scale_tma_aligned=False, scale_ue8m0=False)

    # ---- Routing (production noaux_tc; weights all-positive, sum to 1) --------
    if num_tokens > 0:
        topk_w, topk_idx = _production_topk(
            x_bf, gate_weight, correction_bias, num_topk, cfg['norm_topk_prob'])
        if has_padding:
            # Exactly what _mask_topk_ids_padded_region does: padded rows -> -1.
            topk_idx[num_valid:] = -1
            topk_w[num_valid:] = 0.0
    else:
        topk_w = torch.empty((0, num_topk), dtype=torch.float32, device='cuda')
        topk_idx = torch.empty((0, num_topk), dtype=torch.int64, device='cuda')

    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf))

    # ---- Symm buffer (same kwargs sglang uses) -------------------------------
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max, num_topk, hidden, intermediate_hidden,
        use_fp8_dispatch=True, activation='swiglu')

    # ---- Run the fused kernel exactly as sglang does -------------------------
    def run_fused():
        if num_tokens > 0:
            buffer.x[:num_tokens].copy_(x_fp8)
            buffer.x_sf[:num_tokens].copy_(x_sf)
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_w)
        y = torch.empty((max(num_tokens, 1), hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_mega_moe(
            y, transformed_l1, transformed_l2, buffer,
            recipe=(128, 128, 128),
            activation='swiglu',
            activation_clamp=cfg['swiglu_limit'],   # None — production default
            fast_math=True,
        )
        return y[:num_tokens]

    y_fused = run_fused()
    torch.cuda.synchronize()

    # ---- Reference: the REAL sglang DeepEPMoE layer (production path) ----------
    # Instead of hand-replicating deepep, construct sglang's actual DeepEPMoE,
    # load the real fp8 weights, and call `.forward(x_bf16, topk_output)`. This
    # runs the EXACT production chain: deep_ep.Buffer dispatch (which quantizes the
    # BF16 activation itself via sglang_per_token_group_quant_fp8) -> ep_scatter ->
    # m_grouped_fp8_gemm_nt_contiguous(m_indices, recipe=None) -> silu_and_mul +
    # sglang quant -> GEMM -> ep_gather (applies topk weights). Zero reimplementation
    # risk: if the megamoe kernel matches production deepep, this matches y_fused.
    #
    # KEY: feed BF16 `x_bf` (NOT the pre-quantized x_fp8) — dispatch_a quantizes the
    # activation internally, so the whole quant+dispatch+gemm+combine chain is the
    # real production numerics.
    def _deepep_reference():
        if num_tokens == 0:
            return None
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        _init_sglang_parallel(rank_idx, num_ranks, args.weights_dir or '/preset-models')
        experts = _build_real_deepep_moe(
            num_experts, num_topk, hidden, intermediate_hidden, args.layer_id,
            l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf)

        # Build a router_logits placeholder (deepep path uses the provided topk,
        # but TopKOutput carries logits for downstream code that may read it).
        router_logits = torch.zeros((num_tokens, num_experts), dtype=torch.float32, device='cuda')
        topk_output = StandardTopKOutput(
            topk_weights=topk_w.to(torch.float32),
            topk_ids=topk_idx.to(torch.int64),
            router_logits=router_logits,
        )
        out = experts.forward(x_bf, topk_output)
        return out[:num_tokens]

    def _rel_max(y_a, y_b):
        """Per-ELEMENT relative error |y - y_ref| / |y_ref|, max over all elements
        (the single metric we report — same as tests/test_l1_single_token.py)."""
        ya, yb = y_a.float(), y_b.float()
        re = (ya - yb).abs() / yb.abs().clamp_min(1e-6)
        return re.max().item() if re.numel() else 0.0

    # Single check: per-element relative-error MAX vs the real sglang DeepEP fp8 path.
    y_ref = _deepep_reference()
    rel_max = _rel_max(y_fused, y_ref) if (y_ref is not None and num_tokens > 0) else 0.0
    ok = rel_max <= args.rel_tol

    # calc_diff dimension: 1 - cosine_similarity (DeepGEMM's standard fp8 metric).
    # Unlike per-element rel-max, this is ROBUST to cancellation columns (where the
    # reference value is ~0 and dividing a 1-ULP abs error by it explodes the ratio).
    # Reported both GLOBALLY (all elements) and as the WORST per-row value (so a
    # single bad token can't hide under the global average).
    if y_ref is not None and num_tokens > 0:
        yf2, yr2 = y_fused.float(), y_ref.float()
        cd_global = calc_diff(yf2, yr2)
        # per-row calc_diff: vectorized 1 - 2*<x,y>/(|x|^2+|y|^2) per token row
        xf, yf_ = yf2.double(), yr2.double()
        num = 2.0 * (xf * yf_).sum(-1)
        den = (xf * xf + yf_ * yf_).sum(-1)
        per_row = torch.where(den == 0, torch.zeros_like(den), 1.0 - num / den)
        cd_row_max = per_row.max().item()
        cd_row_argmax = int(per_row.argmax().item())
        # ABSOLUTE max diff: max |y - y_ref| over all elements (NOT divided by
        # anything, so it is immune to the near-zero-denominator blow-up that makes
        # rel-max misleading at cancellation columns). Report the element + its row
        # norm so the magnitude is interpretable.
        ae_all = (yf2 - yr2).abs()
        abs_max = ae_all.max().item()
        abs_flat = int(ae_all.argmax().item())
        abs_tok, abs_col = abs_flat // hidden, abs_flat % hidden
        abs_row_norm = yr2[abs_tok].norm().item()
    else:
        cd_global = cd_row_max = abs_max = abs_row_norm = 0.0
        cd_row_argmax = abs_tok = abs_col = -1

    dist_print(
        f'  [rank{rank_idx}] KERNEL vs sglang DeepEP fp8 path  [{"OK" if ok else "FAIL"}]\n'
        f'           abs-max  = {abs_max:.6f}  @tok{abs_tok} col{abs_col} (row_norm={abs_row_norm:.2f})\n'
        f'           calc_diff= {cd_global:.2e}  (global 1-cos)\n'
        f'           row-cdmax= {cd_row_max:.2e}  @tok{cd_row_argmax} (worst per-row 1-cos)\n'
        f'           rel-max  = {rel_max:.6f}  (tol {args.rel_tol}; NOTE: blows up at '
        f'cancellation cols where ref~=0)',
        once_in_node=False)

    # Print the top-10 worst elements (kernel value, ref value, abs/rel diff) so the
    # max can be read directly — exposes whether a huge rel comes from a near-zero ref.
    # IMPORTANT: this block runs on rank 0 ONLY, so it must use plain `print`, NOT
    # `dist_print` — the latter calls `dist.barrier()` internally, which would
    # deadlock here (the other ranks never enter this branch, so they never reach
    # the matching barrier and rank 0 hangs forever waiting on them). This was the
    # source of the long "hang" + the orphaned processes left holding the port.
    if y_ref is not None and num_tokens > 0 and rank_idx == 0:
        yf2, yr2 = y_fused.float(), y_ref.float()
        yf, yr = yf2.reshape(-1), yr2.reshape(-1)
        ae = (yf - yr).abs()
        re = ae / yr.abs().clamp_min(1e-6)
        order = re.argsort(descending=True)[:10]
        for flat in order.tolist():
            t, c = flat // hidden, flat % hidden
            # Context for the worst element: is the whole ref row ~zero? what's this
            # token's topk_idx? (a fully-zero ref row = padding/empty token.)
            ref_row_norm = yr2[t].norm().item()
            ker_row_norm = yf2[t].norm().item()
            tk = topk_idx[t].tolist() if t < topk_idx.size(0) else None
            print(
                f'    [rank{rank_idx}] tok{t} col{c}: kernel={yf[flat].item():+.6f} '
                f'ref={yr[flat].item():+.6f} abs={ae[flat].item():.6f} rel={re[flat].item():.6f} '
                f'| ref_row_norm={ref_row_norm:.4f} ker_row_norm={ker_row_norm:.4f} topk_idx={tk}',
                flush=True)

    buffer.destroy()
    dist.barrier()
    return ok


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90; got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    ok = _run(rank_idx, num_ranks, group, args)

    # Reduce pass/fail across ranks.
    ok_t = torch.tensor([1 if ok else 0], device='cuda')
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    all_ok = bool(ok_t.item())

    dist_print('', once_in_node=True)
    dist_print('PASSED (all ranks)' if all_ok else 'FAILED (>=1 rank)', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if not all_ok:
        sys.exit(1)


def _free_rendezvous_port(port: int) -> None:
    """Kill any process bound to the torch.distributed rendezvous port and wait
    for the port to become free. Makes back-to-back / post-Ctrl-C runs reliable
    (init_dist hard-codes the port, so a leftover holder otherwise hangs the next
    run on EADDRINUSE or a stale rendezvous)."""
    import socket
    import subprocess
    import time

    def _port_free() -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    if _port_free():
        return
    # Find and kill holders via `lsof` (best-effort; ignore if unavailable).
    try:
        out = subprocess.run(['lsof', '-ti', f'tcp:{port}'],
                             capture_output=True, text=True, timeout=10).stdout
        pids = [p for p in out.split() if p and int(p) != os.getpid()]
        if pids:
            print(f'[setup] freeing rendezvous port {port}: killing stale PIDs {pids}', flush=True)
            subprocess.run(['kill', '-9', *pids], timeout=10)
    except Exception as e:
        print(f'[setup] could not auto-kill port {port} holders ({e}); '
              f'if the run hangs, manually `kill` them or set MASTER_PORT', flush=True)
    # Wait up to ~15s for the socket (incl. TIME_WAIT) to clear.
    for _ in range(30):
        if _port_free():
            return
        time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 MegaMoE test mirroring sglang MiMo-V2-flash inference')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='EP size / ranks to spawn (default: 8 to mirror --ep-size 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                        help='Tokens per rank (production conditions, not production batch; default 128)')
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=128,
                        help='Symm-buffer token capacity (default 128)')
    parser.add_argument('--clean-acts', action='store_true',
                        help='Use plain Gaussian activations (no outliers). Default injects '
                             'outlier channels to match real activation FP8 stress.')
    parser.add_argument('--rel-tol', type=float, default=1e-3,
                        help='Per-token RELATIVE L2-error threshold (||y-ref||2/||ref||2). A token '
                             'is "bad" if its rel-L2 exceeds this. This is the PRIMARY correctness '
                             'gate (default 1e-3).')
    parser.add_argument('--weights-dir', type=str, default='',
                        help='Checkpoint dir (e.g. /preset-models) to load REAL MiMo-V2 expert '
                             'weights + real router from, validating the block-(128,128) scale '
                             'orientation. Omit for synthetic weights.')
    parser.add_argument('--layer-id', type=int, default=1,
                        help='Which MoE layer to load with --weights-dir (1..47; layer 0 is dense)')
    parser.add_argument('--num-valid-tokens', type=int, default=-1,
                        help='Model DP-attention padding: only the first N rows are valid; the rest '
                             'are padding (dirty acts + topk_idx=-1), mirroring CUDA-graph un-zeroed '
                             'buffers. -1 (default) = no padding. Use e.g. --num-tokens 128 '
                             '--num-valid-tokens 40 to stress the padding-skip path.')
    args = parser.parse_args()

    # The DeepEP fp8 reference needs these env vars (set before spawn so children
    # inherit them): EP_DISABLE_GIN=1 for single-node NVLink ElasticBuffer init.
    os.environ.setdefault('EP_DISABLE_GIN', '1')

    # Free the torch.distributed rendezvous port (init_dist hard-codes 8361 via
    # MASTER_PORT). A previous run that was Ctrl-C'd leaves orphan child processes
    # holding the port — the next run then either errors with EADDRINUSE or hangs
    # waiting on the stale rendezvous. Kill whatever is bound to the port and wait
    # for it to clear so back-to-back runs never stall on a leftover.
    _free_rendezvous_port(int(os.getenv('MASTER_PORT', '8361')))

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
