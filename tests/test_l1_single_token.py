"""Minimal L1-stage precision test: 1 rank, 1 token, topk=1, single expert.

With exactly one token routed to one expert, the kernel's L1 pool holds that
token's L1 result in row 0 of `buf.l2_acts` (the post-SwiGLU + per-128-fp8
down-projection input). We compare it, element by element, against the EXACT
sglang DeepEP fp8 op chain (same numerics sglang runs after dispatch):

    l1y  = DeepGEMM fp8 GEMM (x_fp8 @ W_l1)         -> bf16
    di   = flashinfer silu_and_mul(l1y)             -> bf16   (silu(gate)*up)
    q    = sglang per-128 fp8 quantize(di)          (amax/448, e4m3)

This isolates the L1 GEMM + SwiGLU + quantization with zero combine / L2 /
cross-rank / pool-layout ambiguity, and compares fp8-vs-fp8 (both sides use the
same DeepGEMM fp8 GEMM + flashinfer silu), so the result should be ~bit-exact.

    EP_DISABLE_GIN=1 FLASHINFER_DISABLE_VERSION_CHECK=1 python tests/test_l1_single_token.py
"""

import os
import sys
import torch
import torch.distributed as dist

os.environ.setdefault('FLASHINFER_DISABLE_VERSION_CHECK', '1')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/sgl-workspace/sglang/python')

import deep_gemm
import sgl_kernel  # registers torch.ops.sgl_kernel.silu_and_mul
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from test_mega_moe_sm90 import _quantize_grouped_fp8_block_128_128
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


def run(local_rank, num_local_ranks):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(1000 + rank_idx)

    hidden, ih, topk = 4096, 2048, 1
    E = num_ranks                       # 1 expert per rank (EP = num_ranks)
    Epr = E // num_ranks                # = 1
    nt = 1  # ONE token per rank

    # This rank owns exactly its 1 local expert (global expert = rank_idx).
    l1_bf = torch.randn((Epr, ih * 2, hidden), dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn((Epr, hidden, ih), dtype=torch.bfloat16, device='cuda') * 0.05
    l1_fp8, l1_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_fp8, l2_sf = _quantize_grouped_fp8_block_128_128(l2_bf)

    # one token, routed to THIS rank's own expert (global id = rank_idx), so its
    # down-input stays local in buf.l2_acts[0] for a clean per-rank comparison.
    x_bf = torch.randn((nt, hidden), dtype=torch.bfloat16, device='cuda')
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    idx = torch.full((nt, topk), rank_idx, dtype=torch.int64, device='cuda')  # local expert
    w = torch.ones((nt, topk), dtype=torch.float32, device='cuda')    # weight 1.0

    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe_sm90((l1_fp8, l1_sf), (l2_fp8, l2_sf))
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, E, nt, topk, hidden, ih,
                                                 use_fp8_dispatch=True, activation='swiglu')
    buf.x[:nt].copy_(x_fp8)
    buf.x_sf[:nt].copy_(x_sf)
    buf.topk_idx[:nt].copy_(idx)
    buf.topk_weights[:nt].copy_(w)

    y = torch.empty((nt, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.fp8_mega_moe(y, tl1, tl2, buf, recipe=(128, 128, 128),
                           activation='swiglu', activation_clamp=None, fast_math=True)
    torch.cuda.synchronize()

    # The kernel's L1 result for our one token = row 0 of the L2-input buffer.
    k_codes = buf.l2_acts[0].clone().float()             # (ih,) fp8 codes
    k_sf = buf.l2_acts_sf[0, :ih // 128].clone().float()  # (ih/128,) scales
    k_down = k_codes * k_sf.repeat_interleave(128)        # (ih,) dequantized down-input

    if True:  # every rank checks its own local token
        # ---- sglang fp8 reference: SAME grouped fp8 GEMM sglang uses + flashinfer silu + per-128 quant ----
        # m_grouped_fp8_gemm_nt_contiguous is the GEMM sglang's DeepEP path runs.
        # It needs: A=(m,k), B=(num_groups,n,k), out=(m,n), grouped_layout=(m,) int32
        # giving each row's group index, and m aligned to the contiguous alignment.
        align = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(align)
        m_pad = align  # pad our 1 token up to one aligned block
        a_pad = torch.zeros((m_pad, hidden), dtype=torch.float8_e4m3fn, device='cuda')
        a_sf_pad = torch.zeros((m_pad, hidden // 128), dtype=torch.float32, device='cuda')
        a_pad[:nt] = x_fp8
        a_sf_pad[:nt] = x_sf
        grouped_layout = torch.full((m_pad,), -1, dtype=torch.int32, device='cuda')
        grouped_layout[:nt] = 0  # our token belongs to expert/group 0
        l1y_pad = torch.empty((m_pad, ih * 2), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_pad, a_sf_pad), (l1_fp8, l1_sf), l1y_pad, grouped_layout,
            recipe=(1, 128, 128))
        l1y = l1y_pad[:nt].contiguous()
        # DEBUG: print the reference L1 GEMM gate output (token0, cols 0/1) in BOTH
        # bf16 (what m_grouped emits) and the fp32 value it represents, to bit-compare
        # against the kernel's raw fp32 accumulator printed under DG_MEGA_MOE_DEBUG_L1ACC.
        if rank_idx == 0:
            g0 = l1y[0, 0]; g1 = l1y[0, 1]
            g0b = g0.view(torch.int16).item() & 0xffff
            g1b = g1.view(torch.int16).item() & 0xffff
            print(f'[REFL1] gate row0 col0 bf16={g0.item():+.9e} bits=0x{g0b:04x} | '
                  f'col1 bf16={g1.item():+.9e} bits=0x{g1b:04x}', flush=True)
        # CONTIGUOUS down-proj path — sglang deepep NORMAL mode (CONFIRMED via runtime
        # probe: normal dispatches to `_run_contiguous_gemm`). The two real ops:
        #   silu_and_mul(l1y) -> BF16 down_input   (sgl_kernel; fast-math __expf, fp32
        #                                            silu*up, then store→bf16)
        #   sglang_per_token_group_quant_fp8(down_input, 128) -> fp8 codes + sf
        from sgl_kernel import silu_and_mul
        from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
        down_input = torch.empty((nt, ih), dtype=torch.bfloat16, device='cuda')
        silu_and_mul(l1y.view(-1, ih * 2), down_input)
        r_codes_t, r_sf_t = sglang_per_token_group_quant_fp8(
            down_input, 128, column_major_scales=False, scale_tma_aligned=False,
            scale_ue8m0=False)
        r_codes = r_codes_t[0].float()
        r_sf = r_sf_t[0].float()
        r_down = r_codes * r_sf.repeat_interleave(128)
        di = down_input[:1].float()  # for diagnostic prints

        # AUTO-FIND every per-128 group whose SF differs between kernel and reference,
        # and for each locate its amax column + show the dequantized down-input there.
        # This pinpoints WHICH element's L1-GEMM 1-ULP propagated into the SF.
        if rank_idx == 0:
            gdiff = (k_sf != r_sf).nonzero(as_tuple=False).flatten()
            print(f'[SFDIFF] groups with kernel!=sglang SF: {gdiff.tolist()}  '
                  f'(total {gdiff.numel()}/{ih//128})')
            for g in gdiff.tolist():
                lo = g * 128
                # reference amax element within the group (fp32 torch swiglu view)
                r_grp = di[0, lo:lo+128].abs()
                ai = int(r_grp.argmax().item())
                col = lo + ai
                kb = k_sf[g].view(torch.int32).item() & 0xffffffff
                rb = r_sf[g].view(torch.int32).item() & 0xffffffff
                # the bf16 gate/up that fed this amax element (from m_grouped l1y)
                gate_bf = l1y[0, col]; up_bf = l1y[0, ih + col]
                print(f'[SFDIFF] grp{g} amax@col{col}: '
                      f'kernel_sf=0x{kb:08x}({k_sf[g].item():.8e}) '
                      f'sglang_sf=0x{rb:08x}({r_sf[g].item():.8e}) '
                      f'fp32-ULP={abs(kb-rb)} | amax: k={k_sf[g].item()*448:.6f} '
                      f's={r_sf[g].item()*448:.6f} | refl1y gate(bf16)={gate_bf.item():+.6f} '
                      f'up(bf16)={up_bf.item():+.6f}')

        abs_max = (k_down - r_down).abs().max()
        code_match = (k_codes == r_codes).float().mean()
        sf_rel = ((k_sf - r_sf).abs() / r_sf.clamp_min(1e-12))

        print(f'[L1][rank{rank_idx}] down-input vs sglang fused ref: '
              f'fp8-code-match={code_match.item():.4f}  '
              f'#differ={int((k_codes != r_codes).sum())}/{ih}  SFrel-max={sf_rel.max().item():.2e}')

        # ---- show EXACTLY which codes differ and by how much (rank 0 only) ----
        diff = (k_codes != r_codes)
        bad = diff.nonzero(as_tuple=False).flatten()
        if rank_idx == 0:
            for col in bad[:20].tolist():
                grp = col // 128
                print(f'[L1]     col {col:4d} (grp{grp:2d}): code k={k_codes[col].item():+.1f} r={r_codes[col].item():+.1f} '
                      f'| dequant k={k_down[col].item():+.4f} r={r_down[col].item():+.4f}')

        # ---- L2 stage check: kernel final y vs reference L2 GEMM on the KERNEL's own down-input ----
        # (topk=1, single expert, weight=1 -> y should equal L2_GEMM(down_input)).
        # Use sglang's REAL L2 GEMM (m_grouped_fp8_gemm_nt_contiguous), not fp8_gemm_nt,
        # so the accumulation order matches what sglang actually runs.
        m_pad2 = align
        l2in_pad = torch.zeros((m_pad2, ih), dtype=torch.float8_e4m3fn, device='cuda')
        l2in_sf_pad = torch.zeros((m_pad2, ih // 128), dtype=torch.float32, device='cuda')
        l2in_pad[:nt] = k_codes.to(torch.float8_e4m3fn).view(nt, ih)
        l2in_sf_pad[:nt] = k_sf.view(nt, ih // 128)
        gl2 = torch.full((m_pad2,), -1, dtype=torch.int32, device='cuda'); gl2[:nt] = 0
        l2y_pad = torch.empty((m_pad2, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (l2in_pad, l2in_sf_pad), (l2_fp8, l2_sf), l2y_pad, gl2, recipe=(1, 128, 128))
        l2y_ref = l2y_pad[:nt].contiguous()
        yk = y[0].float()
        yr = l2y_ref[0].float()
        rel = (yk - yr).abs() / yr.abs().clamp_min(1e-6)
        rel_max = rel.max()
        ycode = (yk == yr).float().mean()
        print(f'[L2][rank{rank_idx}] kernel final-y '
              f'rel-max={rel_max.item():.6f}  bitmatch={ycode.item():.4f}  '
              f'y[:5]={[round(v_,4) for v_ in yk[:5].tolist()]} '
              f'r[:5]={[round(v_,4) for v_ in yr[:5].tolist()]}')

        # ---- top elements by relative error (only those that actually differ) ----
        # fp32 L2 reference: dequant(down_input) @ dequant(l2_weight)^T, exact fp32.
        l2w = (l2_fp8[0].float().view(hidden//128,128,ih//128,128)
               * l2_sf[0].view(hidden//128,1,ih//128,1)).view(hidden, ih)
        l2_fp32 = (k_down.view(1, ih) @ l2w.t())[0]   # (hidden,) true fp32 L2 output
        nz = (yk != yr).nonzero(as_tuple=False).flatten()
        if nz.numel() > 0:
            order = nz[rel[nz].argsort(descending=True)]
            for col in order[:10].tolist():
                print(f'[L2][rank{rank_idx}]   col {col:4d}: k={yk[col].item():+.6f} r={yr[col].item():+.6f} '
                      f'fp32={l2_fp32[col].item():+.6f} '
                      f'absdiff={abs(yk[col].item() - yr[col].item()):.6f} rel={rel[col].item():.6f}')

        # ---- the amax element of the bad group: is kernel's vs sglang's down-input value identical? ----
        if bad.numel() > 0:
            gbad = (bad[0].item() // 128)
            lo, hi = gbad * 128, gbad * 128 + 128
            r_grp = di[0, lo:hi].float()                 # sglang down-input (bf16) for the group
            k_grp = k_down[lo:hi]                        # kernel down-input (dequant) for the group
            r_amax_i = r_grp.abs().argmax().item()
            print(f'[L1]   grp{gbad} amax elem: local col {r_amax_i} (global {lo+r_amax_i}) '
                  f'sglang_di={r_grp[r_amax_i].item():+.6f} (abs={r_grp[r_amax_i].abs().item():.6f})')
            print(f'[L1]   grp{gbad} amax: sglang={r_grp.abs().max().item():.6f}  '
                  f'kernel(dequant)={k_grp.abs().max().item():.6f}')

    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    import sys as _sys
    nprocs = int(_sys.argv[1]) if len(_sys.argv) > 1 else 1
    torch.multiprocessing.spawn(run, args=(nprocs,), nprocs=nprocs)
