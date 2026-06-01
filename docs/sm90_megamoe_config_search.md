# SM90 MegaMoE Config Search

This note records the H20 SM90 MegaMoE split-kernel defaults before moving the
selector to the DeepGEMM-style candidate-search path.

## Selector Model

Regular DeepGEMM GEMM selects parameters by:

1. generating legal config candidates,
2. estimating a cheap `LayoutInfo`-like score from the shape,
3. choosing the best candidate deterministically,
4. JIT-compiling only the selected kernel variant.

SM90 MegaMoE now follows the same shape.  The selector builds complete
`MegaMoESM90Config` candidates containing tile sizes, wave grouping, stage
count, thread layout, and scheduler/epilogue mode flags.  It then ranks them
with a block/wave score plus an empirical calibration layer for MoE-specific
dispatch/combine choices.

This is not online autotuning.  Runtime launch does not benchmark candidate
kernels.  The default search is deterministic and keeps the H20 empirical
choices as calibration targets.

## Search Space

Default candidate dimensions:

- `block_m`: default `64`; `128` can be included with
  `DG_SM90_MOE_SEARCH_BLOCK_SHAPES=1`.
- `block_n`: default `256` for split L1/L2, otherwise `128`; alternate block-N
  candidates can be included with `DG_SM90_MOE_SEARCH_BLOCK_SHAPES=1`.
- `num_epilogue_threads`: derived from block shape; `64x256` uses two epilogue
  warpgroups.
- `num_dispatch_threads`: compact split frontend uses `64`; otherwise `128`.
- `direct_l2_scatter`: candidate default from empirical policy, alternate
  included unless forced by `DG_SM90_MOE_DIRECT_L2_SCATTER`.
- `l2_nmajor_schedule`: candidate default from empirical policy, alternate
  included unless forced by `DG_SM90_MOE_L2_NMAJOR`.
- `one_warp_cleanup`: candidate default from empirical policy, alternate
  included unless forced by `DG_SM90_MOE_ONE_WARP_CLEANUP`.
- `num_stages`: empirical 4/5-stage target plus alternate 4/5 candidates unless
  forced by `DG_SM90_MOE_NUM_STAGES`; impossible forced values are clamped to
  the shared-memory-limited maximum stage count.
- `num_experts_per_wave`: empirical target plus `16` and full-rank candidates
  when legal unless forced by `DG_SM90_MOE_EXPERTS_PER_WAVE`.

Debugging:

```bash
DG_PRINT_CONFIGS=1 python3 tests/bench_mega_moe_sm90.py --num-processes 8 --batches 8 16 32 --num-tests 1
DG_SM90_MOE_PRINT_SEARCH=2 python3 tests/bench_mega_moe_sm90.py --num-processes 8 --batches 128 --num-tests 1
```

## H20 Empirical Reference

Configuration: `num_ranks=8`, `hidden=7168`, `intermediate_hidden=2048`,
`num_experts=256`, `num_experts_per_rank=32`, `topk=8`.

`expected_tokens_per_expert = M * topk / num_experts_per_rank = M / 4`.

| M | expected/expert | block_m | block_n | stages | dispatch/non-epi/epi threads | experts/wave | direct L2 scatter | L2 N-major | one-warp cleanup |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 8 | 2 | 64 | 256 | 5 | 64 / 64 / 256 | 16 | on | off | on |
| 16 | 4 | 64 | 256 | 5 | 64 / 64 / 256 | 16 | on | off | on |
| 32 | 8 | 64 | 256 | 4 | 64 / 64 / 256 | 32 | on | off | on |
| 64 | 16 | 64 | 256 | 5 | 64 / 64 / 256 | 32 | on | off | on |
| 128 | 32 | 64 | 256 | 5 | 64 / 64 / 256 | 32 | on | off | on |
| 256 | 64 | 64 | 256 | 4 | 64 / 64 / 256 | 32 | on | off | on |
| 260 | 65 | 64 | 256 | 4 | 64 / 64 / 256 | 32 | off | off | off |
| 512 | 128 | 64 | 256 | 5 | 64 / 64 / 256 | 16 | on | off | off |
| 1024 | 256 | 64 | 256 | 5 | 64 / 64 / 256 | 16 | on | on | off |

These are the pre-search empirical defaults.  Focused H20 sweeps after adding
candidate search found two better defaults for single-size runs:

- `M=32`: `num_experts_per_wave=16` was faster than the previous `32`.
- `M=260`: enabling both direct L2 scatter and one-warp cleanup was faster than
  the previous boundary behavior that turned both off after
  `expected_tokens_per_expert > 64`.

The selector calibration has been updated for those two cases.  Other tested
differences, such as 4-stage vs 5-stage at `M=512` and L2 N-major at `M=1024`,
were not stable enough to change the default.

## Validation Log

Commands:

```bash
DG_PRINT_CONFIGS=1 python3 tests/bench_mega_moe_sm90.py --num-processes 8 --batches 8 16 32 64 128 256 260 512 1024 --num-tests 1
python3 tests/test_mega_moe_sm90.py --num-processes 1 --layers 1 2 --fail-fast
python3 tests/test_mega_moe_sm90.py --num-processes 2 --layers 1 2 --fail-fast
python3 tests/bench_mega_moe_sm90.py --num-processes 8 --batches 8 16 32 64 128 256 260 512 --num-tests 20
```

Correctness after candidate selector:

- `--num-processes 1 --layers 1 2 --fail-fast`: passed all 5 scenarios,
  max printed diff `0.0006`.
- `--num-processes 2 --layers 1 2 --fail-fast`: passed all 5 scenarios,
  max printed diff `0.0006`.

Focused H20 benchmark after candidate selector:

```text
M       latency us
8          849.8
16         806.4
32         748.7
64         799.9
128        864.0
256       1160.3
260       1174.0
512       1894.0
1024      3226.0
```

The strongest improvements versus the pre-search defaults are at `M=32`
(`experts_per_wave=16`) and `M=260` (direct L2 scatter plus one-warp cleanup).
Other search candidates remain available through env overrides, but the default
calibration was not changed when repeated measurements were noisy or neutral.
