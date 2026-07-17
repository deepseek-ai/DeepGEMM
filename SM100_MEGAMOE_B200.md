# SM100 MegaMoE B200 Optimization

## Scope

This optimization targets the SM100 MXFP8 x MXFP4 MegaMoE path on B200. The
calibrated topology uses 8 ranks, 384 global experts, top-k 6, hidden size
7168, and intermediate hidden size 3072. Each specialization is gated by its
applicable dtype, shape, and token regime.

## Design

### Token-regime block selection

The block heuristic uses the expected routed tokens per expert,

`tokens_per_rank * num_ranks * topk / num_experts`,

to choose smaller store slices and fewer epilogue warpgroups for decode and
small-prefill workloads. Above 96.5 expected tokens per expert, the calibrated
topology selects a 16-token-aligned block that covers one BM192 wave plus an
imbalance allowance of `1.5 * sqrt(expected_tokens)`. This avoids an extra
wave at the measured BM128 and BM192 tail cliffs.

The token-pull size is reduced from 3584 bytes to 1792 bytes in the measured
small/mid regimes, improving communication and compute overlap without
changing the large-shape fallback.

### Large-token BM240 specialization

At 8192 or more tokens per rank, MXFP8 x MXFP4 uses BM240 with a 24-row store
slice. The larger candidate changes only internal large-token buffer sizing;
the public token-alignment contract remains 384 for compatibility and small
workloads keep the original candidate set.

### Native SM100 UE8M0 scale conversion

The L1 epilogue converts two positive FP32 scale candidates with
`cvt.rp.satfinite.ue8m0x2.f32`, then reconstructs the scale and reciprocal
scale from the encoded exponents. This replaces the generic exponent path in
the SM100 kernel while preserving its output.

### Kernel Factory direct combine

For the calibrated no-shared-expert, hidden-7168, top-k-6 case at 32768 or
more tokens per rank, the fused kernel stops after producing the combine
planes and launches a Kernel Factory-derived standalone reduction. The
specialization uses 768 threads, a two-CTA cluster, 128-bit non-coherent
loads, streaming stores, and ascending FP32 accumulation before BF16 output.
Smaller workloads and shared-expert configurations retain the fused combine.

## B200 performance

The baseline is commit `559d79f` (`Public release 26/07`). Measurements are
full-operator times from an 8-rank B200 job. Each candidate run is bracketed
by baseline runs before and after it; speedup is the averaged bracketed
baseline time divided by candidate time.

The dense small/mid sweep exceeds the baseline at every tested point and has
a 1.0506x geometric-mean speedup (5.06% throughput improvement, equivalent to
4.82% lower latency). The best point is 1792 tokens per rank at 1.1191x.

| Tokens/rank | Speedup | Improvement |
|---:|---:|---:|
| 832 | 1.1014x | +10.14% |
| 896 | 1.0608x | +6.08% |
| 960 | 1.0517x | +5.17% |
| 1024 | 1.0569x | +5.69% |
| 1088 | 1.0356x | +3.56% |
| 1152 | 1.0277x | +2.77% |
| 1216 | 1.0164x | +1.64% |
| 1280 | 1.0111x | +1.11% |
| 1536 | 1.0139x | +1.39% |
| 1664 | 1.0911x | +9.11% |
| 1792 | 1.1191x | +11.91% |
| 1920 | 1.0764x | +7.64% |
| 2048 | 1.0529x | +5.29% |
| 2176 | 1.0507x | +5.07% |
| 2304 | 1.0313x | +3.13% |
| 2432 | 1.0202x | +2.02% |

The wide sweep from 128 to 32768 tokens per rank has a 1.0206x geometric-mean
speedup. The only measured regression is 4096 tokens per rank at -0.03%,
which is within run-to-run noise.

| Tokens/rank | Speedup | Improvement |
|---:|---:|---:|
| 128 | 1.0065x | +0.65% |
| 256 | 1.0033x | +0.33% |
| 512 | 1.0172x | +1.72% |
| 768 | 1.0363x | +3.63% |
| 1024 | 1.0515x | +5.15% |
| 2048 | 1.0446x | +4.46% |
| 4096 | 0.9997x | -0.03% |
| 8192 | 1.0077x | +0.77% |
| 16384 | 1.0208x | +2.08% |
| 32768 | 1.0196x | +1.96% |

Across the union of 24 unique token counts, the geometric-mean speedup is
1.0382x (3.82%). Final-source digest checks at 832, 1792, 8192, and 32768
tokens per rank produced bitwise-identical outputs for all 32 rank pairs,
covering the small/mid, BM240, and direct-combine paths. The broader
masked/unmasked regression sweep across small, mid, and BM240 regimes was
also bitwise identical for all 96 tested rank pairs.
