# DeepGEMM Scaling Factor Format

This document specifies the scaling factor (SF) tensor contract for DeepGEMM's FP8/FP4 GEMM APIs: required shapes, dtypes, strides, the transform pipeline, and how to call the APIs correctly. Source of truth:

- `csrc/apis/layout.hpp` — SF transform dispatch (`transform_sf_into_required_layout`, `transform_k_grouped_sf_into_required_layout`)
- `csrc/utils/layout.hpp` — SF layout validation (`check_sf_layout`)
- `csrc/apis/gemm.hpp` — GEMM argument checks
- `csrc/apis/attention.hpp`, `csrc/apis/mega_moe.hpp`, `deep_gemm/mega/__init__.py` — attention / mega MoE SF requirements (Section 6)
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
- Supported `gran_k`: `32` or `128` on SM100; `128` only on SM90.

### 1.2 SF Value Constraint

Every `float32` SF value must be an exact power of 2: bit pattern `[0][8-bit exponent][23 mantissa bits = 0]`. Sign bit and mantissa must be zero; this is asserted on device (`DG_TRAP_ONLY_DEVICE_ASSERT((value & 0x807fffffu) == 0)` in `smxx_layout.cuh`).

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
| `float32` | any | 32 or 128 | SM100 | Broadcast along MN to `gran_mn=1`, then pack to UE8M0 `int32`, MN-major, TMA-aligned |
| `int32` | 1 | 32 or 128 | SM100 | Validate only (already pre-transformed; must satisfy Section 3) |

For the SM100 `float32` row, the returned tensor is exactly the pre-transformed format defined in Section 3: shape `[·, mn, ceil_div(k, gran_k * 4)]`, dtype `int32`, strides `(align(mn, 4) * ceil_div(k, gran_k * 4), 1, align(mn, 4))`. It can be cached and passed back to later calls, which then hit the `int32` validate-only row.

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

K-grouped GEMM concatenates groups along K. The SM100 FP8 TN API uses logical operands `A = [sum_k, m]` and `B = [sum_k, n]`; the SM100 FP4 NT API uses `A = [m, sum_k]` and `B = [n, sum_k]`, backed by packed-byte storage `[m, sum_k / 2]` and `[n, sum_k / 2]`. Both produce `D = [num_groups, m, n]` and share the `grouped_layout` semantics below. APIs:

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

**`int32` pre-packed (validated only, no kernel):** ONLY accepted on SM100 with `gran_k = 32`. For `gran_k = 128` you must pass `float32`. Contract:

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
    gran_k=32,               # int32 pre-packing path requires gran_k == 32
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
    weights,                 # [seq_len, num_heads]; float32, or bf16 (SM100, forces bf16 logits)
    cu_seq_len_k_start, cu_seq_len_k_end,
    clean_logits=True, max_seqlen_k=0, logits_dtype=torch.float32,
)
```

**The dtype of `q_sf` and `kv_sf` is COUPLED.** Passing `q_sf` selects "MX mode", which flips the required `kv_sf` dtype (attention.hpp:123: `kv_sf.scalar_type() == (is_mx_sf ? kInt32 : kFloat)`). There is no mixed mode:

| Mode | `q_sf` | `kv_sf` | Q/KV data dtype | Arch |
|---|---|---|---|---|
| MX (`q_sf` provided) | `int32` packed UE8M0, contiguous | `int32` packed UE8M0, contiguous | FP8 (MXFP8) or packed FP4 (MXFP4) | SM100 only |
| non-MX (`q_sf=None`) | — | `float32` (one plain scale per token), contiguous | FP8 only | SM90 / SM100 |

Additional rules:

- **FP4 Q/KV data requires MX mode** — `q_sf` must be provided (attention.hpp:92).
- SF shapes: `q_sf` is `[seq_len, num_heads]` (non-paged) or `[batch_size, next_n, num_heads]` (paged); `kv_sf` is 1-D `[seq_len_kv]`. Both contiguous.
- MX SF granularity is per-32-element blocks along `head_dim` — with `head_dim <= 128` all (up to 4) UE8M0 exponents of one token/head fit in exactly **one `int32`**, hence the shapes above have no trailing K dimension.
- The legacy aliases `fp8_mqa_logits` / `fp8_paged_mqa_logits` hardwire `q_sf=None`, i.e., always the non-MX `float32` mode.
- Paged variant: the KV SF is **fused into the byte cache**, not a separate tensor. `kv_cache` is `uint8` of shape `[num_kv_blocks, block_kv, 1, kv_head_dim + 4]` — per token, the value bytes (`head_dim` for FP8, `head_dim/2` for FP4) are followed by 4 SF bytes interpreted as `int32` (MX) or `float32` (non-MX) (attention.hpp:266-285).

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

`fp8_einsum` hardcodes its expressions and **permutes each operand AND its SF** into `(batch, m, n, k)` order before calling the internal batched GEMM (`csrc/apis/einsum.hpp:209-232`):

| Expression | `(batch, m, n, k)` | SFA permute | SFB permute |
|---|---|---|---|
| `'bhr,hdr->bhd'` | `(h, b, d, r)` | `(1, 0, 2)` | none |
| `'bhd,hdr->bhr'` (SM100) | `(h, b, r, d)` | `(1, 0, 2)` | `(0, 2, 1)` |
| `'bhd,bhr->hdr'` (SM100) | `(h, d, r, b)` | `(1, 2, 0)` | `(1, 2, 0)` |

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

Sources: `csrc/apis/mega_moe.hpp` (host checks), `deep_gemm/mega/__init__.py` (Python wrapper + weight transform), `tests/test_mega_moe.py` (SF construction).

- **Recipe is pinned to `(1, 1, 32)`** (mega.hpp:179); `kGranK = 32` is also hardcoded in the JIT impl (sm100_fp8_fp4_mega_moe.hpp:165).
- **Weight SFs must already be packed `int32`** — unlike the GEMM APIs, there is NO float32 fallback and no auto-transform: `check_sf_layout(..., type_check=torch::kInt)` rejects `float32` outright (mega.hpp:206-209, 229-232). The required layout is exactly the Section 3 contract with `gran_k = 32`:
  - Routed L1 SF: `[num_experts_per_rank, 2*intermediate_hidden, ceil_div(hidden, 128)]`, `int32`, MN-major, TMA-aligned (`stride(-2)==1`, `stride(-1)==align(mn,4)`); routed L2 SF: `[num_experts_per_rank, hidden, ceil_div(intermediate_hidden, 128)]`. (The `/128` is `gran_k * 4 = 32 * 4`.)
  - Shared expert SFs (optional; weights are FP8 instead of FP4): same contract, 2-D without the group dimension.
- **An extra mega-MoE-only layout step is mandatory**: pass weights + SFs through `deep_gemm.transform_weights_for_mega_moe` before the call. It (a) interleaves gate/up rows of L1 (weights AND SF, 8-row granularity) and (b) applies a UTCCP intra-128-row transpose to both L1 and L2 SFs (`reshape(-1, 4, 32, packed_sf_k).transpose(2, 3)`; requires `mn % 128 == 0`) — `deep_gemm/mega/__init__.py:97-149`. The SFs in Section 3 layout alone are NOT directly consumable by the kernel.
- **Activation SFs are internal**: they live in slices of the symmetric buffer (`x_sf` is K-major `[num_max_tokens_per_rank, hidden // 128]` `int32`; the intermediate `l1/l2_acts_sf` are MN-major), produced by the kernel pipeline — not user-supplied arguments (mega.hpp:96-153).

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
