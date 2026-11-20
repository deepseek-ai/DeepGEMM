"""Untimed diagnostics on real model weights; never used by serving forward."""

def check_model_layer(model):
    import torch
    from deep_gemm.small_moe import SmallMoE
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

    for name, layer in model.named_modules():
        method = getattr(layer, "quant_method", None)
        ws = getattr(method, "sm120_workspace", None)
        if ws is None:
            continue
        e, twice_i, d = layer.w13_weight.shape
        i, k = twice_i // 2, method.moe.experts_per_token
        records = []
        with torch.inference_mode(), torch.random.fork_rng(devices=[layer.w13_weight.device]):
            torch.manual_seed(28191)
            for trial in range(3):
                ids = torch.randperm(e, device=layer.w13_weight.device)[:k].contiguous()
                gu = layer.w13_weight[ids].contiguous()
                if ws["swapped"]:
                    gu = torch.cat((gu[:, i:], gu[:, :i]), dim=1)
                down = layer.w2_weight[ids].contiguous()
                ref = SmallMoE(1, gu, down, k, enable_fusion=False, fallback_m_tile=32)
                x = torch.randn((1, d), device=gu.device, dtype=torch.bfloat16)
                weights = torch.softmax(torch.randn((1, k), device=gu.device), dim=-1)
                counts = torch.ones((k,), device=gu.device, dtype=torch.int32)
                slots = (torch.arange(k, device=gu.device, dtype=torch.int64)*128).view(1, k)
                ref.projection.fill_(float("nan"))
                ref.expert.fill_(float("nan"))
                expected = ref.forward(x, counts, slots, weights).clone()
                actual = method.forward_native(layer, x, weights, ids.view(1, k), None, None).clone()
                native = UnquantizedFusedMoEMethod.forward_native(
                    method, layer, x, weights, ids.view(1, k), None, None,
                ).clone()
                torch.cuda.synchronize()
                h = ws["h"][ids, 0]
                bitwise_h = torch.equal(h.view(torch.int16), ref.h[:, 0].view(torch.int16))
                bitwise_y = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                if not bitwise_h or not bitwise_y or not bool(torch.isfinite(native).all()):
                    raise RuntimeError(f"Real layer parity failed: H={bitwise_h}, Y={bitwise_y}")
                delta = actual.float() - native.float()
                records.append({
                    "trial": trial, "experts": ids.cpu().tolist(),
                    "ordinary_bm32_h_bitwise": bitwise_h, "ordinary_bm32_y_bitwise": bitwise_y,
                    "native_max_abs": delta.abs().max().item(),
                    "native_relative_l2": (delta.norm()/native.float().norm().clamp_min(1e-30)).item(),
                    "native_bitwise": torch.equal(actual.view(torch.int16), native.view(torch.int16)),
                    "native_comparison_is_diagnostic": True,
                })
                del ref, gu, down, actual, native, expected, h
            torch.cuda.empty_cache()
        return {"layer": name, "trials": records, "complete": True}
    raise RuntimeError("No eligible layer for real-weight parity check")
