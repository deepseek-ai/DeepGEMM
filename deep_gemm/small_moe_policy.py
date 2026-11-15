"""Bounded SM120 decode policy. No device readback or timing-based selection."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmallMoEConfig:
    l1: tuple[int, int, int] | None = None  # BN, stages, experts per CTA
    l2: tuple[int, int, int] | None = None
    reason: str = 'outside validated small-token range'


def select_config(tokens, experts, topk, hidden, intermediate):
    """Return separate GEMMs unless both shape and measured range match.

    The table does not assume common experts or uniform routing. New shapes
    require new evidence; the average T*k/E is not a per-expert row count.
    """
    if experts not in (64, 128) or tokens not in (1, 2, 4) or topk not in (1, 2, 4, 8):
        return SmallMoEConfig()
    if (hidden, intermediate) == (2048, 768):
        if (tokens == 2 and topk == 8) or (tokens == 4 and topk >= 4):
            return SmallMoEConfig(reason='full-chain benefit not robust across routing distributions')
        bundle = 2 if ((tokens == 1 and topk == 8) or
                       (tokens == 2 and topk == 4)) else topk
        return SmallMoEConfig((8, 3, bundle), (16, 3, topk) if tokens == 1 else (32, 2, topk),
                              'validated route-parallel decode')
    if (hidden, intermediate) == (4096, 1536):
        if tokens != 1 and not (tokens == 2 and topk <= 2):
            return SmallMoEConfig(reason='D4096 boundary benefit not established')
        bn, stages = (8, 4) if topk == 1 else (8, 3) if topk == 2 else (16, 2)
        bundle = 1 if tokens == 2 and topk == 2 else topk
        return SmallMoEConfig((bn, stages, bundle), (32, 2, topk), 'validated route-parallel decode')
    return SmallMoEConfig()
