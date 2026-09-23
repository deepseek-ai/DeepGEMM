# DeepGEMM Scaling Factor Format

This document specifies the scaling factor (SF) tensor contract for DeepGEMM's FP8/FP4 GEMM APIs: required shapes, dtypes, strides, the transform pipeline, and how to call the APIs correctly. Source of truth:

- `csrc/apis/layout.hpp` — SF transform dispatch (`transform_sf_into_required_layout`, `transform_k_grouped_sf_into_required_layout`)
- `csrc/utils/layout.hpp` — SF layout validation (`check_sf_layout`)
- `csrc/apis/gemm.hpp` — GEMM argument checks
- `csrc/apis/attention.hpp`, `csrc/apis/mega_moe.hpp`, `csrc/apis/sm90_mega.hpp`, `csrc/apis/nvfp4_mega_moe.hpp`, `deep_gemm/mega/__init__.py` — attention / Mega MoE SF requirements (Section 6)
- `csrc/jit_kernels/impls/smxx_layout.hpp` — host-side transform launchers
- `deep_gemm/include/deep_gemm/impls/smxx_layout.cuh` — transform kernels
- `tests/test_layout.py`, `tests/test_fp8_fp4.py`, `tests/generators.py` — working usage examples

## 1. Definitions

### 1.1 Quantization Recipe

`recipe = (gran_m, gran_n, gran_k)` describes the **SF storage** granularity (not the compute granularity): one SF value covers a `gran_m x gran_k` block of A (or `gran_n x gran_k` of B).

Given A of shape `[·, M, K]` and B of shape `[·, N, K]` (the leading `·` batch/group dimension is optional), the untransformed SFs must have:

```
SFA shape: [·, ceil_div(M, gran_m), ceil_div(K, gran_k)]    dtype: float32
SFB shape: [·, ceil_div(N, gran_n), ceil_div(K, gran_k)]    dtype: float32
```

Rules:

- Pass exactly one of `recipe` or the `recipe_a` + `recipe_b` pair. Use `recipe_a = (gran_m, gran_k_a)` / `recipe_b = (gran_n, gran_k_b)` (2-tuples) when A and B use different K granularities.
- Supported SF transform `gran_k`: `32` or `128` on SM100 and SM120; `128` only on SM90. Individual kernels may impose tighter limits.

### 1.2 SF Value Constraint

When packing FP32 SFs into UE8M0, each value must have bit pattern `[0][8-bit exponent][23 mantissa bits = 0]` (positive powers of two, with zero also accepted). Sign bit and mantissa must be zero; this is asserted on device (`DG_TRAP_ONLY_DEVICE_ASSERT((value & 0x807fffffu) == 0)` in `smxx_layout.cuh`). This packing constraint does not apply to the SM90 FP32 compute format.

When producing SFs from a quantization cast kernel, pass `round_sf=True` to guarantee this.

### 1.3 Constants

- `packed_sf_dtype` = `int32` (4 UE8M0 exponents packed per element, see Section 7.1)
- `ALIGN_MN` = `16 bytes / sizeof(int32)` = `4` (TMA requires `stride(-1)` to be a multiple of 16 bytes)

## 2. Two Ways to Provide SFs

| Path | SF dtype | Extra kernel launch | When to use |
|---|---|---|---|
| A. Untransformed | `float32` | Yes — DeepGEMM launches a transform kernel per GEMM call | Prototyping, correctness testing |
| B. Pre-transformed | `int32` (packed UE8M0) | No — layout is only validated | Production; weights (transform once, cache) and activations (produce directly from the cast kernel) |

For path B, the recipe passed to the GEMM must have `gran_m = gran_n = 1` (the transform broadcasts SFs along MN), with `gran_k` unchanged.

## 3. Pre-transformed SF Format Contract (Path B)

A pre-transformed SF tensor for `mn` rows and `k` columns with granularity `(1, gran_k)` must satisfy (validated by `check_sf_layout` in `csrc/utils/layout.hpp`):

```
dtype:       int32 (packed UE8M0)
shape:       [·, mn, packed_sf_k]          where packed_sf_k = ceil_div(k, gran_k * 4)
stride(-3):  stride(-1) * size(-1)         # outer/group dimension packed tightly
stride(-2):  1                             # contiguous along MN ("MN-major")
stride(-1):  align(mn, 4)                  # TMA 16-byte alignment
```

Note the tensor is **MN-major**: the MN dimension is the contiguous one, so the underlying memory is `packed_sf_k` slices of `align(mn, 4)` elements each (see Section 7.2 for a diagram).

## 4. The Transform: `transform_sf_into_required_layout`

```python
deep_gemm.transform_sf_into_required_layout(
    sf,                        # torch.Tensor, float32 or int32
    mn,                        # int: M (if is_sfa) or N
    k,                         # int
    recipe,                    # (gran_m, gran_n, gran_k) or (gran_mn, gran_k)
    num_groups=None,           # int: set if sf has a leading group dimension
    is_sfa=None,               # bool: REQUIRED with a 3-tuple recipe; FORBIDDEN with a 2-tuple
    disable_ue8m0_cast=False,
    psum_layout=None,          # torch.Tensor: only for SFA under the PSUM layout (skips gap rows)
) -> torch.Tensor
```

Recipe form rules (asserted in `csrc/apis/layout.hpp`):

- 3-tuple `(gran_m, gran_n, gran_k)`: must also pass `is_sfa` (`True` selects `gran_m`, `False` selects `gran_n`).
- 2-tuple `(gran_mn, gran_k)`: must NOT pass `is_sfa`.

Dispatch table (`csrc/apis/layout.hpp`):

| Input dtype | `gran_mn` | `gran_k` | Arch | Action |
|---|---|---|---|---|
| `float32` | 1 | 128 | SM90 (or `disable_ue8m0_cast`) | Transpose to MN-major, TMA-aligned `float32` (no packing) |
| `float32` | 128 | 128 | SM90 (or `disable_ue8m0_cast`) | Validate only (no transform) |
| `float32` | positive | 32 or 128 | SM100 / SM120 | Broadcast along MN to `gran_mn=1`, then pack to UE8M0 `int32`, MN-major, TMA-aligned |
| `int32` | 1 | 32 or 128 | SM100 / SM120 | Validate only (already pre-transformed; must satisfy Section 3) |

For the SM100/SM120 packing row, the returned tensor is exactly the pre-transformed format defined in Section 3: shape `[·, mn, ceil_div(k, gran_k * 4)]`, dtype `int32`, strides `(align(mn, 4) * ceil_div(k, gran_k * 4), 1, align(mn, 4))`. It can be cached and passed back to later calls, which then hit the `int32` validate-only row.

On SM120, `psum_layout` and `num_groups` cannot both be supplied. The public transform retains the FP32 `disable_ue8m0_cast=True` cases above for compatibility; they do not enable an FP32-scale SM120 MMA path. Native SM120 FP8/FP4 GEMMs require UE8M0 scales.

### Example: transform weight SFs once and cache

```python
import deep_gemm

# Weights for fp8_einsum 'bhr,hdr->bhd': B operand = [h, d, r], so N=d, K=r, batch=h.
# Untransformed SFB: [h, ceil_div(d, 128), ceil_div(r, 128)], float32
sfw = deep_gemm.transform_sf_into_required_layout(
    sf=scale_factor,
    mn=d, k=r,
    recipe=(1, 128, 128),  # (gran_m, gran_n, gran_k) of the ORIGINAL quantization
    is_sfa=False,          # this is SFB, so gran_mn = gran_n = 128
    num_groups=h,
)
# sfw: [h, d, ceil_div(r, 128 * 4)], int32, MN-major, TMA-aligned. Cache and reuse.

deep_gemm.fp8_einsum(
    'bhr,hdr->bhd',
    (x_fp8, sfx),
    (w_fp8, sfw),          # pre-transformed: no transform kernel launched for SFB
    out,
    recipe=(1, 1, 128),    # gran_n is now 1 (broadcast during the transform); gran_k unchanged
)
```

This works for SFB because `'bhr,hdr->bhd'` does not permute the B operand. To also pre-transform the activation SFA, see Section 6.2 — `fp8_einsum` permutes SFs internally, which changes how the transform must be applied.

### Example: produce pre-transformed SFs directly from a cast kernel

Cast kernels can emit the packed `int32` SF directly, so no transform is ever needed (e.g., `per_token_cast` from `tile_kernels`):

```python
from tile_kernels.quant import per_token_cast

a_fp8, sfa = per_token_cast(
    x=activation,                       # [num_tokens, hidden], bf16
    fmt='e4m3',
    num_per_channels=128,               # = gran_k
    round_sf=True,                      # SF values are exact powers of 2 (Section 1.2)
    use_tma_aligned_col_major_sf=True,  # MN-major + TMA-aligned strides (Section 3)
    use_packed_ue8m0=True,              # packed int32 output
)
# a_fp8: [num_tokens, hidden], float8_e4m3fn
# sfa:   [num_tokens, ceil_div(hidden, 128 * 4)], int32, satisfies Section 3

deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a=(a_fp8, sfa),        # int32 SF: validated, not transformed
    b=(b_fp4, sfb),
    d=output,
    grouped_layout=grouped_layout,
    recipe_a=(1, 128),     # (gran_m, gran_k): gran_m must be 1 for pre-transformed SFs
    recipe_b=(1, 32),
    use_psum_layout=True,
)
```

### Example: simplest path (untransformed float32 SFs)

```python
# A: [m, k] FP8, quantized at 1x128; SFA: [m, ceil_div(k, 128)], float32
# B: [num_groups, n, k] FP4, quantized at 32x32; SFB: [num_groups, ceil_div(n, 32), ceil_div(k, 32)], float32
# grouped_layout: [num_groups], int32, PSUM row boundaries:
#   group i occupies A rows [align(layout[i-1], alignment), layout[i])
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a=(a_fp8, sfa),
    b=(b_fp4, sfb),
    d=output,
    grouped_layout=grouped_layout,
    recipe_a=(1, 128),
    recipe_b=(32, 32),     # float32 SFs may use any supported granularity
    use_psum_layout=True,
)
# DeepGEMM launches the transform kernel internally for both SFs on every call.
```

## 5. K-Grouped Contiguous Layout

K-grouped GEMM concatenates groups along K. The SM100/SM120 FP8 TN API uses logical operands `A = [sum_k, m]` and `B = [sum_k, n]`; the SM100 FP4 NT API uses `A = [m, sum_k]` and `B = [n, sum_k]`, backed by packed-byte storage `[m, sum_k / 2]` and `[n, sum_k / 2]`. Both produce `D = [num_groups, m, n]` and share the `grouped_layout` semantics below.

SM120 supports K-grouped BF16 and FP8, but **not K-grouped FP4 NT**. Its FP8 TN path accepts PSUM layouts and optionally omits `c`, with BF16 or FP32 output; the legacy FP8 NT path requires non-PSUM layouts and nonempty `ks_cpu`. SM120 TN accepts a global MK alignment divisible by 32, subject to the selected kernel's per-group alignment checks; NT requires 128-aligned group sizes. The SM100 accumulation example below is not an additional SM120 restriction. APIs:

```python
deep_gemm.k_grouped_fp8_gemm_tn_contiguous(   # SM100; SM90 uses k_grouped_fp8_gemm_nt_contiguous
    a,                       # (tensor, sf) pair
    b,                       # (tensor, sf) pair
    d,
    ks_cpu,                  # List[int] or None: per-group K sizes, on CPU
    grouped_layout,          # torch.Tensor: [num_groups], int32, on device (num_groups <= 128)
    c,                       # although the signature defaults to None, k-grouped asserts c is
                             # not None: pass an FP32 accumulator with the same shape as d
                             # (d = c + sum over groups); passing None fails a DG_HOST_ASSERT.
                             # Prefer passing the SAME tensor as d (c is d): the kernel then
                             # accumulates in place. If c is a different tensor, DeepGEMM first
                             # runs d.copy_(c) before the GEMM (an extra full copy)
    recipe=(1, 1, 128),      # gran_m and gran_n MUST be 1; gran_k in {32, 128} on SM100
    compiled_dims="mn",
    use_psum_layout=False,
)

deep_gemm.k_grouped_fp4_gemm_nt_contiguous(   # SM100 only; same arguments
    a, b, d, ks_cpu, grouped_layout, c,
    recipe=(1, 1, 32), compiled_dims="mn", use_psum_layout=False,
)
```

`k_alignment` below is the global MK alignment for contiguous layouts, set via `deep_gemm.set_mk_alignment_for_contiguous_layout(value)`. It must be a multiple of the kernel's `BLOCK_K`: 128 for FP8 on SM100 (SM90: exactly 128) and 256 for FP4. Every aligned K range must have zero A/B padding and valid corresponding SF padding.

### 5.1 `grouped_layout` and `ks_cpu` semantics

The meaning of `grouped_layout` depends on `use_psum_layout`:

| `use_psum_layout` | `grouped_layout[i]` contains | `ks_cpu` | `k_i` constraints |
|---|---|---|---|
| `False` | group `i`'s K size directly | required | each `k_i % k_alignment == 0` and `k_i % gran_k == 0` |
| `True` | cumulative (prefix-sum) end offset; group `i` occupies K range `[align(layout[i-1], k_alignment), layout[i])` | optional | `k_i` needs no alignment |

With `use_psum_layout=True`:

- If `ks_cpu` is provided, it must contain the **aligned** sizes `align(k_i, k_alignment)` (each entry must be a multiple of `k_alignment`); the exact SF shape is then computed on the host.
- If `ks_cpu` is `None` or `[]`, group sizes are read from `grouped_layout` on the device, and the host allocates an upper bound of `(sf_k + 3 * num_groups) / 4` packed rows.

### 5.2 K-grouped SF contract

The SF input (per operand) is 2D. Two accepted dtypes:

**`float32` (DeepGEMM packs it):** shape `[sum_sf_k, mn]` contiguous, where `sum_sf_k` = total SF rows over all groups (`k_i / gran_k` rows per group without PSUM; `align(k_i, k_alignment) / gran_k` with PSUM). Supported for `gran_k` 32 and 128. Requires `mn % 4 == 0`.

**`int32` pre-packed (validated only, no kernel):** accepted on SM100 with `gran_k = 32`, and on SM120 with `gran_k = 32` or `128`. SM100 requires `float32` for `gran_k = 128`. Contract:

```
shape:  [packed_sf_k, mn] where packed_sf_k >= sum(ceil_div(k_i, gran_k * 4))   # larger is OK, e.g.
        a buffer pre-allocated for the maximum K; unused trailing rows are ignored
stride: [mn, 1] (contiguous); mn % 4 == 0
layout: each group starts at a new packed row; when a group's SF row count is not a
        multiple of 4, the trailing UE8M0 slots of its last packed row are zero-filled
```

Note the k-grouped packed SF is plain contiguous `[packed_sf_k, mn]` — unlike Section 3, no padding is inserted along MN; instead `mn % 4 == 0` is a hard precondition.

### 5.3 Example: k-grouped with float32 SFs (non-PSUM)

```python
import torch, deep_gemm

gran_k, k_alignment = 32, 128
deep_gemm.set_mk_alignment_for_contiguous_layout(k_alignment)

ks = [2048, 4096, 1024]                    # each a multiple of k_alignment (and gran_k)
sum_k = sum(ks)
grouped_layout = torch.tensor(ks, device='cuda', dtype=torch.int32)  # per-group sizes (non-PSUM)

# a_fp8: [sum_k, m] float8_e4m3fn;  sfa: [sum_k // gran_k, m] float32, contiguous
# b_fp8: [sum_k, n] float8_e4m3fn;  sfb: [sum_k // gran_k, n] float32, contiguous
# m and n must be multiples of 4 (SF packing precondition, Section 5.2)
d = torch.zeros((num_groups, m, n), device='cuda', dtype=torch.float)
deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
    a=(a_fp8, sfa), b=(b_fp8, sfb), d=d,
    ks_cpu=ks, grouped_layout=grouped_layout,
    c=d,                     # same tensor as d: in-place accumulation, no extra copy
    recipe=(1, 1, gran_k),
)
```

### 5.4 Example: k-grouped with pre-packed int32 SFs

```python
# Pack once with the dedicated helper (or produce packed SFs from your cast kernel):
sfa_packed = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sfa,                     # [sum_sf_k, mn] float32, contiguous
    grouped_layout,          # semantics per Section 5.1
    ks_cpu=ks,               # or None/[] with use_psum_layout=True
    gran_k=32,               # SM100 example; SM120 also accepts gran_k=128 (Section 5.2)
    k_alignment=k_alignment,
    use_psum_layout=False,
)
# sfa_packed: [sum(ceil_div(k_i, 32*4)), mn], int32

deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
    a=(a_fp8, sfa_packed), b=(b_fp8, sfb_packed), d=d,
    ks_cpu=ks, grouped_layout=grouped_layout, c=d,
    recipe=(1, 1, 32),       # gran_k stays 32; layout is validated, no transform kernel runs
)
```

### 5.5 Example: PSUM layout (dynamic group sizes)

```python
# real_ks may be unaligned; groups are stored padded to k_alignment
def build_psum_layout(real_ks, k_alignment):
    psum, prev_end = [], 0
    for k in real_ks:
        end = (prev_end + k_alignment - 1) // k_alignment * k_alignment + k  # align(prev_end) + k
        psum.append(end)
        prev_end = end
    return psum

real_ks = [1000, 4096, 900]
grouped_layout = torch.tensor(build_psum_layout(real_ks, k_alignment), device='cuda', dtype=torch.int32)
aligned_ks = [(k + k_alignment - 1) // k_alignment * k_alignment for k in real_ks]

deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
    a=(a_fp8, sfa), b=(b_fp8, sfb), d=d,
    ks_cpu=aligned_ks,       # pass ALIGNED sizes; or None if only known on device
    grouped_layout=grouped_layout,
    c=d, recipe=(1, 1, gran_k),
    use_psum_layout=True,
)
```

## 6. SF Requirements Beyond Plain GEMM (Attention, Mega MoE)

Several non-GEMM APIs take SF tensors with **hardcoded** requirements that bypass the recipe/transform pipeline of Section 4. Passing the wrong SF dtype fails a `DG_HOST_ASSERT` immediately.

### 6.1 MQA Logits: `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`

Sources: `csrc/apis/attention.hpp` (host checks), `tests/test_attention.py` (SF construction).

```python
deep_gemm.fp8_fp4_mqa_logits(
    q,                       # (q_fp, q_sf or None): q_fp [seq_len, num_heads, head_dim]
    kv,                      # (kv_fp, kv_sf):       kv_fp [seq_len_kv, head_dim]
    weights,                 # [seq_len, num_heads]; FP32, or SM100 BF16/FP16 (restrictions below)
    cu_seq_len_k_start, cu_seq_len_k_end,
    clean_logits=True, max_seqlen_k=0, logits_dtype=torch.float32,
    schedule_meta=None,
)
```

SM100 dense **FP16 weights** select a non-MX FP8-only path: `q_sf=None`, contiguous FP32 `kv_sf[seq_len_kv]`, positive `seq_len % 4 == 0`, positive `seq_len_kv`, `num_heads` in 4/8/16/32/64/128, and `head_dim` in 32/64/128. Weights require `stride(1)==1`; the launcher materializes a contiguous FP16 copy as needed. The runtime SM count must be even. `schedule_meta` is unsupported, logits may be FP32 or BF16, and `max_seqlen_k >= 0`. Full-output cleaning uses a separate pass; compressed logits (`max_seqlen_k > 0`) require `clean_logits=False`. Paged MQA does not accept FP16 weights. SM100 BF16 weights still require BF16 logits.

**The dtype of `q_sf` and `kv_sf` is COUPLED.** Passing `q_sf` selects "MX mode", which flips the required `kv_sf` dtype (`kv_sf.scalar_type() == (is_mx_sf ? kInt32 : kFloat)` in `csrc/apis/attention.hpp`). There is no mixed mode:

| Mode | `q_sf` | `kv_sf` | Q/KV data dtype | Arch |
|---|---|---|---|---|
| MX (`q_sf` provided) | `int32` packed UE8M0, contiguous | `int32` packed UE8M0, contiguous | FP8 (MXFP8) or packed FP4 (MXFP4) | SM100; SM120 supports MXFP4 only |
| non-MX (`q_sf=None`) | — | `float32` (one plain scale per token), contiguous | FP8 only | SM90 / SM100 / SM120 |

SM120 dense/paged MQA supports 16, 32 or 64 heads, with FP8 head dimensions 32, 64 or 128 and MXFP4 head dimension 128. Weights must be FP32; logits may be BF16 or FP32. Dense `schedule_meta` and `get_mqa_logits_metadata` remain SM100-only. SM120 paged MQA uses `get_paged_mqa_logits_metadata`; supported page sizes are 64/128/256 for FP8 and 32/64/128/256 for MXFP4. Paged cleaning is unsupported; non-paged cleaning uses a separate pass and requires `max_seqlen_k=0`.

SM90 paged MQA accepts physical page sizes `block_kv=32/64`, `next_n=1/2/4`, 32 or 64 heads, and head dimension 32/64/128. It uses FP8 Q/KV with `q_sf=None`, FP32 weights and plain FP32 cache SFs; neither `indices` nor paged cleaning is supported. `context_lens` is contiguous int32 `[batch_size, next_n]`. For `next_n=4`, the runtime SM count must be even and the metadata helper's `num_sms` argument must be `deep_gemm.get_num_sms() // 2`; for `next_n=1/2`, pass the full runtime count. `get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)` takes scheduler slots and returns int32 `[num_sms + 1, 2]`. The `next_n=4` path schedules two CTAs per work item with independent KV copies; it does not promise literal KV multicast.

Additional rules:

- **FP4 Q/KV data requires MX mode** — `q_sf` must be provided.
- SF shapes: `q_sf` is `[seq_len, num_heads]` (non-paged) or `[batch_size, next_n, num_heads]` (paged); `kv_sf` is 1-D `[seq_len_kv]`. Both contiguous.
- MX SF granularity is per-32-element blocks along `head_dim` — with `head_dim <= 128` all (up to 4) UE8M0 exponents of one token/head fit in exactly **one `int32`**, hence the shapes above have no trailing K dimension.
- The legacy aliases `fp8_mqa_logits` / `fp8_paged_mqa_logits` hardwire `q_sf=None`, i.e., always the non-MX `float32` mode.
- Paged variant: the KV SF is **fused into the byte cache**, not a separate tensor. `kv_cache` is `uint8` of shape `[num_kv_blocks, block_kv, 1, kv_head_dim + 4]`, where `kv_head_dim` is `head_dim` for FP8 or `head_dim/2` for FP4. Within each page, all `block_kv * kv_head_dim` value bytes come first, followed by `block_kv` scales interpreted as `int32` (MX) or `float32` (non-MX). Despite the tensor shape, values and scales are not interleaved token by token; see the cache views in `csrc/apis/attention.hpp`.

```python
from deep_gemm.utils import per_token_cast_to_fp8, per_custom_dims_cast_to_fp8

# MX mode (SM100): per-token 1x32 quantization, packed UE8M0 for BOTH SFs
q_fp8_2d, q_sf = per_token_cast_to_fp8(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
kv_fp8, kv_sf = per_token_cast_to_fp8(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
logits = deep_gemm.fp8_fp4_mqa_logits(
    q=(q_fp8_2d.view(seq_len, num_heads, head_dim), q_sf.view(seq_len, num_heads)),  # int32
    kv=(kv_fp8, kv_sf.view(seq_len_kv)),                                             # int32
    weights=weights, cu_seq_len_k_start=ks, cu_seq_len_k_end=ke)

# non-MX mode: q_sf=None forces kv_sf to be plain float32 per-token
kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0,), False)   # kv_sf: [seq_len_kv], float32
logits = deep_gemm.fp8_fp4_mqa_logits(
    q=(q.to(torch.float8_e4m3fn), None),
    kv=(kv_fp8, kv_sf),
    weights=weights, cu_seq_len_k_start=ks, cu_seq_len_k_end=ke)
```

### 6.2 `fp8_einsum`: SFs Are Permuted Internally

`fp8_einsum` hardcodes its expressions and **permutes each operand AND its SF** into `(batch, m, n, k)` order before calling the internal batched GEMM (`csrc/apis/einsum.hpp`):

| Expression | `(batch, m, n, k)` | SFA permute | SFB permute |
|---|---|---|---|
| `'bhr,hdr->bhd'` | `(h, b, d, r)` | `(1, 0, 2)` | none |
| `'bhd,hdr->bhr'` (SM100 / SM120) | `(h, b, r, d)` | `(1, 0, 2)` | `(0, 2, 1)` |
| `'bhd,bhr->hdr'` (SM100 / SM120) | `(h, d, r, b)` | `(1, 2, 0)` | `(1, 2, 0)` |

SM120 supports BF16/FP32 outputs for all three expressions, including accumulation, but rejects quantized FP8 `(d, sfd)` output. For the latter two expressions it materializes K-major input data after permutation; SF permutation and validation still follow the table.

Consequences for pre-transformed (`int32`) SFs — the Section 3 stride contract is checked on the **post-permute** tensor:

- If the SF is not permuted (e.g., SFB of `'bhr,hdr->bhd'`, as in the Section 4 weight example), transform it directly with the operand's own `(num_groups, mn, k)`.
- If the SF is permuted, you must transform it **in post-permute coordinates** and permute it back before passing it to `fp8_einsum`. Do NOT `view`/`reshape` a 2-D transform output into 3-D — the transform output is MN-major (non-contiguous), so reshaping destroys the required strides.

```python
# WRONG: 2-D transform + reshape breaks the MN-major strides
sfx_2d = deep_gemm.transform_sf_into_required_layout(sfx_f32.view(b * h, -1), mn=b * h, k=r, recipe=(1, 128))
sfx_int32 = sfx_2d.reshape(b, h, -1)   # silently copies to contiguous strides;
                                       # stride(-2) != 1 after the internal permute -> DG_HOST_ASSERT fails

# CORRECT for 'bhr,hdr->bhd' SFA: fp8_einsum permutes SFA with (1, 0, 2), so the internal
# batched GEMM sees [h, b, packed_sf_k]. Transform with num_groups=h, mn=b, then permute back:
sfx_grouped = deep_gemm.transform_sf_into_required_layout(
    sfx_f32.permute(1, 0, 2).contiguous(),   # [h, b, ceil_div(r, 128)], float32
    mn=b, k=r, recipe=(1, 128), num_groups=h,
)                                            # [h, b, packed_sf_k], int32, Section 3 layout
sfx_int32 = sfx_grouped.permute(1, 0, 2)     # [b, h, packed_sf_k] view; einsum permutes it back
deep_gemm.fp8_einsum('bhr,hdr->bhd', (x_fp8, sfx_int32), (w_fp8, sfw), out, recipe=(1, 1, 128))
```

Untransformed `float32` SFs need no special care — the internal transform handles the permuted layout.

### 6.3 Mega MoE: `fp8_fp4_mega_moe`

Sources: `csrc/apis/mega_moe.hpp` (host checks), `deep_gemm/mega/__init__.py` (Python wrapper + weight transform), `tests/test_mega_moe.py` (SF construction). This is the SM100 FP8-dispatch path with FP4 or FP8 routed weights; the SM90 and NVFP4 contracts below are different. These `nv_dev` integrations use the current public APIs and DeepJIT.

- **Recipe is pinned to `(1, 1, 32)`**.
- **Weight SFs must already be packed UE8M0 `int32`** — unlike the GEMM APIs, there is no FP32 fallback or auto-transform. Before the Mega MoE shuffle, they use the Section 3 contract with `gran_k = 32`:
  - Routed L1 SF: `[num_experts_per_rank, 2*intermediate_hidden, hidden // 128]`; routed L2 SF: `[num_experts_per_rank, hidden, intermediate_hidden // 128]`. Both are MN-major, TMA-aligned `int32`. The divisor 128 is `gran_k * 4`.
  - Optional shared weights are FP8; their SFs use the same contract without the group dimension, with shared intermediate width `intermediate_hidden * num_shared_experts`.
- **An extra Mega-MoE-only layout step is mandatory**: pass weights + SFs through `deep_gemm.transform_weights_for_mega_moe` before the call. It interleaves gate/up rows of L1 weights **and SFs** at 8-row granularity and applies a UTCCP intra-128-row transpose to both L1 and L2 SFs (`reshape(-1, 4, 32, packed_sf_k).transpose(2, 3)`; requires `mn % 128 == 0`). Section 3 layout alone is not directly consumable by the kernel. Transform optional shared weights too.
- **Input activation SFs are caller-produced**, even though they are passed through buffer views rather than separate call arguments. Populate FP8 `buffer.x` and K-major `buffer.x_sf` (`[num_max_tokens_per_rank, hidden // 128]`, packed UE8M0 `int32`) before execution. For FP8 shared experts, `shared_l1_acts` aliases `x`, but the caller must also populate `shared_l1_acts_sf` in its padded MN-major shared-input layout; see `_to_shared_mega_moe_sf_layout` in `tests/test_mega_moe.py`. Routed pool scales and post-activation routed/shared L2 scales are generated internally. These different scale layouts are not interchangeable.
- **SiTU keeps this SF contract**: use `mma_type='fp8xfp4'` with FP4 routed weights, FP8 dispatch, and `activation='situ'` for buffer construction, weight transformation and execution. Both `situ_beta` and `situ_linear_beta` must be supplied, finite and strictly positive; `activation_clamp` must be `None`. The unified L1 epilogue applies SiTU to both routed and optional FP8 shared experts, not routed experts alone. FP8 routed weights, BF16 Mega MoE, SM90 and NVFP4 do not accept SiTU; SwiGLU does not accept SiTU beta arguments.

A world-size-one process group uses local CUDA storage; multiple ranks use symmetric-memory rendezvous. `DG_COMM_KERNEL_DEBUG=1` clears the entire symmetric buffer **after** the call, so caller-produced routed and shared inputs must be refilled before the next call.

```python
from deep_gemm.utils import per_token_cast_to_fp4

# Routed expert weights: [g, n, k] bf16 -> FP4 (1x32) + float32 SF [g, n, k/32]
w = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
w_sf = torch.empty((g, n, k // 32), device='cuda', dtype=torch.float)
for i in range(g):
    w[i], w_sf[i] = per_token_cast_to_fp4(w_bf16[i], use_ue8m0=True, gran_k=32)

# Step 1: pack to the Section 3 int32 layout (mandatory; float32 SFs are rejected)
w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups=g)

# Step 2: mega-MoE weight/SF shuffle (gate-up interleave + UTCCP SF transpose; mandatory)
(l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe((l1_w, l1_sf), (l2_w, l2_sf))
```

### 6.4 Sparse MQA (SM100 / SM120)

`fp8_fp4_sparse_mqa_logits` and `fp8_fp4_paged_sparse_mqa_logits` require MXFP8 or MXFP4 Q/KV with **both** scales packed as contiguous `int32` UE8M0. Unlike SM120 dense MQA, sparse MQA supports MXFP8. The fixed shape is 32 heads of dimension 128; weights and logits are BF16. Q SF shape is `[num_q_tokens, 32]`, or `[num_q_tokens, 1, 32]` for paged queries (`next_n=1`); non-paged KV SF shape is `[num_kv_tokens]`.

Use the corresponding `get_sparse_mqa_logits_metadata` or `get_paged_sparse_mqa_logits_metadata` API. Sparse blocks contain 8 or 16 KV tokens; `num_max_sparse_blocks` must be a positive multiple of 4, at most 4096. Valid sparse indices must be unique, strictly increasing and in range. Paged cache page size must be divisible by the sparse block size; each page stores values followed by scales, with page stride divisible by 512 bytes. The dense MQA page-size list does not describe this sparse cache contract. See `csrc/apis/attention.hpp` for device, pointer and stride checks.

### 6.5 SM90 FP8 Mega MoE

Sources: `csrc/apis/sm90_mega.hpp`, `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp`, `deep_gemm/mega/__init__.py`.

Use `SM90SymmBuffer` or `get_symm_buffer_for_sm90_mega_moe`, prepare weights with `transform_weights_for_mega_moe_sm90`, then call `fp8_mega_moe`. This path launches separate L1 and L2 kernels. It requires FP8 dispatch, `activation='swiglu'` and `recipe=(128, 128, 128)`; shared experts and SiTU are unsupported.

With `E` local experts, `H=hidden`, `I=intermediate_hidden`:

| Operand | Values | SFs |
|---|---|---|
| Routed L1 | contiguous E4M3 `[E, 2*I, H]` | contiguous FP32 `[E, 2*I//128, H//128]` |
| Routed L2 | contiguous E4M3 `[E, H, I]` | contiguous FP32 `[E, H//128, I//128]` |
| Caller input | `buffer.x`: E4M3 `[capacity, H]` | `buffer.x_sf`: K-major FP32 `[capacity, H//128]` |

Weight SFs represent natural 128-by-128 blocks, with ordinary contiguous strides, not the per-row MN-major packed layout in Section 3. **Only L1 weight rows are interleaved** at gate/up granularity 8. Neither weight SF tensor is interleaved, packed or UTCCP-transposed; L2 weights are unchanged. The helper validates matching weight/SF shapes, dtypes, devices and natural contiguity before transforming.

`H` must be a positive multiple of 256 and `I` a positive multiple of 128, with further combine-vectorization checks on the selected launch configuration. The factory aligns token capacity to 128; direct `SM90SymmBuffer` construction requires that alignment already. Populate `x`, `x_sf`, `topk_idx` (int64) and `topk_weights` (FP32) before calling with contiguous BF16 output `[num_tokens, H]`. The internal routed activation scales are FP32 and MN-major: L1 uses K granularity 128, while the post-SwiGLU L2 input uses granularity 64. These internal scales do not change the block-128 weight recipe. See `tests/test_mega_moe_sm90.py`.

### 6.6 SM100 NVFP4 Mega MoE

Sources: `csrc/apis/nvfp4_mega_moe.hpp`, `deep_gemm/mega/__init__.py`, `tests/test_nvfp4_mega_moe.py`.

Call `fp4_fp4_mega_moe` with `SymmBuffer(..., mma_type='fp4xfp4')`, `recipe=(1, 1, 16)` and `activation='swiglu'`. **NVFP4 is not MXFP4:** values are packed E2M1, but each 16-element K block has an E4M3 scale. Four E4M3 bytes are stored in an `int32`; they are not UE8M0 exponents. Do not run these scales through the UE8M0 transform from Section 4, which does not support this granularity or encoding.

With `E` local experts, `H=hidden`, `I=intermediate_hidden`, both `H` and `I` must be positive multiples of 512. Before the Mega MoE shuffle:

| Operand | Packed values | Packed E4M3 SFs |
|---|---|---|
| Routed L1 | contiguous int8 `[E, 2*I, H//2]` | int32 `[E, 2*I, H//64]` |
| Routed L2 | contiguous int8 `[E, H, I//2]` | int32 `[E, H, I//64]` |
| Caller input | `buffer.x`: uint8 `[capacity, H//2]` | `buffer.x_sf`: contiguous int32 `[capacity, H//64]` |

Weight SFs have `stride(-2)=1`, `stride(-1)=align(mn,4)` and tightly packed expert groups. This is the same stride pattern as Section 3, but with **E4M3 bytes and K granularity 16**. After packing those bytes and preparing MN-major storage, call `transform_weights_for_mega_moe`: L1 values and SFs receive the gate/up interleave; L1/L2 SFs receive the UTCCP transpose. The caller supplies NVFP4 input values/scales and routing tensors through the buffer; intermediate routed scales are generated internally.

Optional shared experts are **BF16**, not quantized tuples. Provide contiguous `shared_l1_weights[2*SI, H]`, `shared_l2_weights[H, SI]` and `x_bf16[at_least_num_tokens, H]` together, where `SI = I * num_shared_experts` agrees with the buffer. Run the BF16 shared weights through `transform_weights_for_mega_moe` (L1 gate/up interleave only). Shared input and intermediate activations have no quantization SFs. CUDA Graph replay must reuse the captured `x_bf16` allocation and token count, which are encoded in its tensor map; update the contents in place rather than replacing storage.

Optional model scales are contiguous FP32 tensors on the output CUDA device: `l1_alphas[E, 2]` (gate, up), `l2_alphas[E]`, and `a2_scales[E]`; omitted scales act as one. The caller folds the FC1 input global scale into `l1_alphas`. L1 alphas multiply GEMM results before BF16 gate/up rounding and SwiGLU. Each supplied `a2_scales` entry must be finite and strictly positive: intermediate E4M3 block scales are normalized by its reciprocal and the scale is restored through the L2 multiplier `l2_alpha * a2_scale`.

Routing weights apply **after L2**, not before NVFP4 intermediate quantization. The L2 result is rounded to BF16, multiplied by its routing weight and rounded to BF16 again. Combine sums routed partials in FP32, rounds to BF16, then applies finite `routed_scaling_factor` (default 1) with another BF16 rounding before adding the BF16 shared result and producing BF16 output. This ordering matters numerically.

The runtime SM count must be even and at least 2; `num_topk` must be positive and `num_topk + (num_shared_experts > 0) <= 32`. NVFP4 `base=` reuse requires a live buffer in the same process group with identical expert count, aligned token capacity, top-k, hidden dimensions, shared-expert count, MMA type, activation and runtime SM count. Keep that configuration and `num_sms` fixed from allocation through execution. Cross-protocol NVFP4/main FP8/BF16 reuse is rejected. The main FP8/BF16 `base=` capacity-reuse contract is otherwise unchanged.

## 7. Internals

Reference for kernel developers and for debugging layout mismatches.

### 7.1 UE8M0 Packing

The 8-bit exponents of 4 consecutive K positions are packed into one `int32`, little-endian by K index (`smxx_layout.cuh`):

```cpp
uint32_t packed = 0;
packed |= (values[0] >> 23u);   // exp of sf[4k+0] -> bits [7:0]
packed |= (values[1] >> 15u);   // exp of sf[4k+1] -> bits [15:8]
packed |= (values[2] >>  7u);   // exp of sf[4k+2] -> bits [23:16]
packed |= (values[3] <<  1u);   // exp of sf[4k+3] -> bits [31:24]
```

### 7.2 Memory Layout of the Transformed Tensor (non-k-grouped)

```
Shape:  [mn, packed_sf_k]    where packed_sf_k = ceil_div(k, gran_k * 4)
Stride: [1, align(mn, 4)]

Diagram (mn=6, packed_sf_k=3, align(6,4)=8), int32 elements:

Offset:   0  1  2  3  4  5  6  7 | 8  9  10 11 12 13 14 15 | 16 ...
Content: m0 m1 m2 m3 m4 m5 __ __ | m0 m1 m2 m3 m4 m5 __ __ | m0 ...
          <--- K-slice 0 ----->    <--- K-slice 1 ----->
```

Each K-slice occupies `align(mn, 4)` elements; the `__` padding exists only for 16-byte TMA alignment and is never read as data.

### 7.3 PSUM Gap Row Handling (M-grouped)

Under the M-grouped PSUM layout, gap rows exist between groups:

```
grouped_layout = [100, 250, 370], alignment = 128
Group 0: rows [0, 100)      valid
Gap:     rows [100, 128)    padding
Group 1: rows [128, 250)    valid
Gap:     rows [250, 256)    padding
Group 2: rows [256, 370)    valid
```

When `psum_layout` is passed to the SF transform, gap rows are not read from the input; the kernel writes `0` for them (a safe finite scale code — UE8M0 `0xff` is NaN). The GEMM kernel never consumes those values.

### 7.4 K-Grouped Packing Algorithm

Each group is packed independently, then concatenated along K (`pack_fp32_into_ue8m0` in `smxx_layout.cuh`):

1. Determine group `i`'s input SF row range:
   - Non-PSUM: `grouped_layout[i]` is the group K size `k_i` (a multiple of `k_alignment` and `gran_k`), covering `k_i / gran_k` rows.
   - PSUM: `k_i = grouped_layout[i] - align(grouped_layout[i-1], k_alignment)`; the group covers its aligned region, `align(k_i, k_alignment) / gran_k` rows (PSUM data is stored padded to `k_alignment`).
2. Emit `ceil_div(num_group_sf_rows, 4)` packed `int32` rows for the group; if `num_group_sf_rows % 4 != 0`, zero-fill the trailing UE8M0 slots of the last packed row.
3. Concatenate all groups: output shape `[sum(ceil_div(num_group_sf_rows_i, 4)), mn]`, contiguous.
