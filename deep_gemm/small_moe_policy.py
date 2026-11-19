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


# Offline candidates revalidated with the PR's repaired arithmetic. Membership
# alone does not enable fusion: only the accepted subset below is public policy.
SINGLE_TOKEN_CANDIDATES = {
    (2048, 768, 128, 8): ((32, 3, 2), (16, 4, 8)),
    (4096, 1536, 128, 8): ((16, 2, 8), (32, 2, 8)),
    (2048, 1408, 60, 4): ((16, 3, 4), (16, 4, 4)),
    (3584, 2560, 64, 8): ((32, 3, 4), (32, 2, 8)),
    (2048, 1408, 64, 6): ((16, 3, 2), (16, 4, 6)),
}

# Passed calibration and independent-seed warm/pressure acceptance on RTX 5090.
# Qwen2 (3584, 2560, 64, 8) regressed under pressure and deliberately stays split.
SINGLE_TOKEN_ACCEPTED = frozenset({
    (2048, 768, 128, 8),
    (4096, 1536, 128, 8),
    (2048, 1408, 60, 4),
    (2048, 1408, 64, 6),
})


def select_single_token_config(tokens, experts, topk, hidden, intermediate):
    """Select an audited T1 plan using shape metadata only, never route values.

    Requires distinct top-k experts, as documented by SmallMoE. Consequently
    A=k and rho=1 without device readback. Multi-token requests are not inferred
    from this evidence and use ordinary GEMMs.
    """
    key = hidden, intermediate, experts, topk
    if tokens != 1 or key not in SINGLE_TOKEN_ACCEPTED:
        return SmallMoEConfig(reason='outside accepted single-token evidence')
    l1, l2 = SINGLE_TOKEN_CANDIDATES[key]
    return SmallMoEConfig(l1, l2, 'accepted single-token repaired-arithmetic plan')
