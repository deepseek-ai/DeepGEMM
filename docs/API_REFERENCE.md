# DeepGEMM API Reference

Comprehensive reference for DeepGEMM's JIT-compiled kernels, covering configuration, dense GEMMs (FP8/BF16), grouped MoE GEMMs, attention kernels, and utilities.

## Table of Contents

- [Conventions and Notes](#conventions-and-notes)
- [Configuration](#configuration)
- [Dense FP8 GEMM](#dense-fp8-gemm)
- [Dense BF16 GEMM](#dense-bf16-gemm)
- [M-Grouped GEMMs (MoE Forward)](#m-grouped-gemms-moe-forward)
- [K-Grouped GEMM (MoE Backward)](#k-grouped-gemm-moe-backward)
- [Attention Kernels](#attention-kernels)
- [Utilities](#utilities)
- [SM90 vs SM100 Differences](#sm90-vs-sm100-differences)
- [Common Pitfalls & Troubleshooting](#common-pitfalls--troubleshooting)
- [End-to-End Examples](#end-to-end-examples)

---

## Conventions and Notes

- **Layouts**: `nt` means A is non-transposed, B is transposed: `D = C + A @ B.T`. Other variants follow the same pattern (`nn`, `tn`, `tt`).
- **Tensor dtypes**: FP8 uses `torch.float8_e4m3fn`. BF16 uses `torch.bfloat16`.
- **Scaling factors**:
  - **SM90 (Hopper)**: FP32 scaling factors
  - **SM100 (Blackwell)**: Packed `UE8M0` scaling factors (4 values packed into `torch.int`)
- **Alignment**: Grouped GEMMs require group-boundary alignment from `get_mk_alignment_for_contiguous_layout()`. TMA-aligned tensors should respect `get_tma_aligned_size()`.
- **JIT Compilation**: Kernels are JIT-compiled; first call per shape/layout incurs compilation overhead.
- **Shape Convention**: M = rows of A/D, N = columns of D (rows of B for transposed), K = shared dimension.

---

## Configuration

### `set_num_sms(n: int) -> None`

Set the maximum number of SMs kernels may use.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Cap on SM count; 0 resets to device default |

### `get_num_sms() -> int`

Get the currently configured SM cap. Returns device SM count if not explicitly set.

### `set_tc_util(ratio: float) -> None`

Set desired tensor core utilization ratio.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ratio` | `float` | Fraction of TC usage (0.0–1.0); 1.0 for full utilization |

### `get_tc_util() -> float`

Get current tensor core utilization ratio.

**Usage Example**

```python
import deep_gemm as dg

dg.set_num_sms(0)        # Use all available SMs
dg.set_tc_util(1.0)      # Full tensor core utilization
print(f"Using {dg.get_num_sms()} SMs at {dg.get_tc_util():.0%} TC utilization")
```

---

## Dense FP8 GEMM

All FP8 GEMMs compute: `D = alpha * (A @ B{layout}) + beta * C`

### Function Signatures

```python
fp8_gemm_nt(a, b, d, c=None, alpha=1.0, beta=1.0, disable_ue8m0_cast=False, recipe=None, stream=None)
fp8_gemm_nn(a, b, d, c=None, ...)
fp8_gemm_tn(a, b, d, c=None, ...)
fp8_gemm_tt(a, b, d, c=None, ...)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Tuple[Tensor, Tensor]` | Input A: (data `[M, K]`, scale_factors) |
| `b` | `Tuple[Tensor, Tensor]` | Input B: (data, scale_factors), shape depends on layout |
| `d` | `Tensor` | Output buffer `[M, N]` |
| `c` | `Tensor \| None` | Optional accumulator/bias `[M, N]` |
| `alpha` | `float` | Scale for A @ B (default: 1.0) |
| `beta` | `float` | Scale for C (default: 1.0) |
| `disable_ue8m0_cast` | `bool` | Disable UE8M0 format conversion (default: False) |
| `recipe` | `Tuple \| None` | Optional kernel config override |
| `stream` | `cudaStream_t \| None` | Optional CUDA stream |

### Layout Shapes

| Layout | A Shape | B Shape | Operation |
|--------|---------|---------|-----------|
| `nt` | `[M, K]` | `[N, K]` | `D = A @ B.T` |
| `nn` | `[M, K]` | `[K, N]` | `D = A @ B` |
| `tn` | `[K, M]` | `[K, N]` | `D = A.T @ B` |
| `tt` | `[K, M]` | `[N, K]` | `D = A.T @ B.T` |

### Usage Example

```python
import torch
import deep_gemm as dg

M, N, K = 1024, 2048, 512

# Create FP8 inputs with scaling factors
a_data = torch.randn(M, K, device="cuda", dtype=torch.float8_e4m3fn)
a_scale = torch.ones(M, device="cuda", dtype=torch.float32)
a = (a_data, a_scale)

b_data = torch.randn(N, K, device="cuda", dtype=torch.float8_e4m3fn)
b_scale = torch.ones(N, device="cuda", dtype=torch.float32)
b = (b_data, b_scale)

# Output buffer
d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

# Compute D = A @ B.T
dg.fp8_gemm_nt(a, b, d)
```

---

## Dense BF16 GEMM

Same semantics as FP8, but with BF16 inputs/outputs. No scaling factors required.

### Function Signatures

```python
bf16_gemm_nt(a, b, d, c=None, alpha=1.0, beta=1.0, stream=None)
bf16_gemm_nn(a, b, d, c=None, ...)
bf16_gemm_tn(a, b, d, c=None, ...)
bf16_gemm_tt(a, b, d, c=None, ...)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Tensor` | Input A, dtype `torch.bfloat16` |
| `b` | `Tensor` | Input B, dtype `torch.bfloat16` |
| `d` | `Tensor` | Output buffer |
| `c` | `Tensor \| None` | Optional accumulator/bias |

### Usage Example

```python
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

dg.bf16_gemm_nn(a, b, d)  # D = A @ B
```

---

## M-Grouped GEMMs (MoE Forward)

Used when multiple experts share `N, K` dimensions but have different `M` (token counts).

### Contiguous Layout

Concatenates all expert tokens into a single tensor.

```python
m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices, disable_ue8m0_cast=False, stream=None)
m_grouped_fp8_gemm_nn_contiguous(a, b, d, m_indices, ...)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Tuple[Tensor, Tensor]` | Concatenated input `[sum(M_i), K]` with scales |
| `b` | `Tuple[Tensor, Tensor]` | Expert weights `[num_experts, N, K]` with scales |
| `d` | `Tensor` | Output `[sum(M_i), N]` |
| `m_indices` | `Tensor` | Expert assignment per row, or -1 for padding |

**Important**: Group boundaries must align to `get_mk_alignment_for_contiguous_layout()`.

### Masked Layout

For CUDA graph compatibility with static shapes.

```python
m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=False, stream=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Tuple[Tensor, Tensor]` | Input `[num_groups, max_m, K]` |
| `b` | `Tuple[Tensor, Tensor]` | Expert weights `[num_groups, N, K]` |
| `d` | `Tensor` | Output `[num_groups, max_m, N]` |
| `masked_m` | `Tensor` | Actual token count per group `[num_groups]` |
| `expected_m_per_group` | `int` | Static max M for graph capture |

### Usage Example

```python
align = dg.get_mk_alignment_for_contiguous_layout()

# Pad each expert's tokens to alignment boundary
# Build m_indices with expert assignment (-1 for padding)
dg.m_grouped_fp8_gemm_nt_contiguous(a_concat, b_experts, d_out, m_indices)
```

---

## K-Grouped GEMM (MoE Backward)

For weight gradient computation in MoE backward pass.

```python
k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, ks_tensor=None, c=None, stream=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Tuple[Tensor, Tensor]` | Concatenated `A.T` blocks `[K_total, M]` |
| `b` | `Tuple[Tensor, Tensor]` | Concatenated inputs `[K_total, N]` |
| `d` | `Tensor` | Gradient output `[M, N]` |
| `ks` | `List[int]` | Per-group K sizes |
| `ks_tensor` | `Tensor \| None` | Device tensor mirror of `ks` |
| `c` | `Tensor \| None` | Optional accumulator |

**Important**: K partitions in `ks` must match the concatenation order exactly.

---

## Attention Kernels

### `fp8_mqa_logits`

Compute MQA logits for prefilling.

```python
fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=True, stream=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `Tensor` | Query `[seq_len, num_heads, head_dim]`, FP8 |
| `kv` | `Tuple[Tensor, Tensor]` | KV cache `[seq_len_kv, head_dim]` with scales |
| `weights` | `Tensor` | Head weights `[seq_len, num_heads]` |
| `cu_seq_len_k_start` | `Tensor` | Start indices `[seq_len]` |
| `cu_seq_len_k_end` | `Tensor` | End indices `[seq_len]` |
| `clean_logits` | `bool` | Zero-initialize unfilled logits to -inf |

**Returns**: Output tensor `[seq_len, seq_len_kv]` with token-to-token logits.

### `fp8_paged_mqa_logits`

Paged version for decoding with KV cache paging.

```python
fp8_paged_mqa_logits(q, kv, weights, page_table, page_indices,
                     cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=True, stream=None)
```

Additional parameters:
- `page_table`: Mapping from logical to physical pages
- `page_indices`: Per-token page indices

---

## Utilities

### Layout and Alignment

```python
get_tma_aligned_size(size: int) -> int
```
Return size padded to TMA alignment requirements.

```python
get_mk_alignment_for_contiguous_layout() -> int
```
Return required alignment for M/K group boundaries in contiguous grouped GEMMs.

### Scaling Factor Transformation

```python
transform_sf_into_required_layout(sf: Tensor, ...) -> Tensor
```
Transform scaling factors into required layout for the target architecture.

### Tensor Utilities

```python
get_mn_major_tma_aligned_tensor(shape, dtype, device) -> Tensor
```
Create a TMA-aligned MN-major tensor.

```python
get_mn_major_tma_aligned_packed_ue8m0_tensor(shape, device) -> Tensor
```
Create packed UE8M0 tensor for SM100 scaling factors.

---

## SM90 vs SM100 Differences

| Aspect | SM90 (Hopper) | SM100 (Blackwell) |
|--------|---------------|-------------------|
| **Scaling Factors** | FP32 format | Packed UE8M0 (4 values per int32) |
| **Layout Support** | NT only | NT, NN, TN, TT |
| **Helper Functions** | `get_mn_major_tma_aligned_tensor` | `get_mn_major_tma_aligned_packed_ue8m0_tensor` |
| **CUDA Version** | 12.3+ (recommend 12.9+) | 12.9+ required |

### Converting Scaling Factors for SM100

```python
# SM90: Use FP32 scales directly
sf_fp32 = torch.ones(num_scales, device="cuda", dtype=torch.float32)

# SM100: Convert to packed UE8M0
sf_packed = dg.transform_sf_into_required_layout(sf_fp32, target="ue8m0")
```

---

## Common Pitfalls & Troubleshooting

### Layout Mismatches

**Problem**: Wrong results without errors.

**Solution**: Ensure function suffix (`nt`, `nn`, etc.) matches actual tensor memory layout. For `nt`, B should be `[N, K]` so `B.T` is `[K, N]`.

### Alignment Issues

**Problem**: Performance degradation or incorrect results.

**Solution**: Use `get_tma_aligned_size()` for tensor dimensions and `get_mk_alignment_for_contiguous_layout()` for group boundaries.

### Scaling Factor Format

**Problem**: Incorrect output values on SM100.

**Solution**: Convert FP32 scales to packed UE8M0 format using `transform_sf_into_required_layout()`.

### JIT Cold Start

**Problem**: High latency on first call.

**Solution**: Warm up critical paths before latency-sensitive runs. The compiled kernels are cached in `$HOME/.deep_gemm` by default.

### Grouped GEMM Order

**Problem**: Wrong gradients in K-grouped backward.

**Solution**: Ensure `ks` list matches the exact concatenation order of tensors.

---

## End-to-End Examples

### Dense FP8 GEMM

```python
import torch
import deep_gemm as dg

M, N, K = 4096, 4096, 4096

a = (torch.randn(M, K, device="cuda", dtype=torch.float8_e4m3fn),
     torch.ones(M, device="cuda", dtype=torch.float32))
b = (torch.randn(N, K, device="cuda", dtype=torch.float8_e4m3fn),
     torch.ones(N, device="cuda", dtype=torch.float32))
d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

dg.fp8_gemm_nt(a, b, d)
print(f"Output shape: {d.shape}, dtype: {d.dtype}")
```

### MoE Forward (M-Grouped Contiguous)

```python
num_experts = 8
tokens_per_expert = [128, 256, 64, 192, 128, 256, 64, 128]  # Variable
N, K = 4096, 1024

align = dg.get_mk_alignment_for_contiguous_layout()
# Pad and concatenate tokens, build m_indices
total_tokens = sum(tokens_per_expert)

a_concat = (torch.randn(total_tokens, K, device="cuda", dtype=torch.float8_e4m3fn),
            torch.ones(total_tokens, device="cuda", dtype=torch.float32))
b_experts = (torch.randn(num_experts, N, K, device="cuda", dtype=torch.float8_e4m3fn),
             torch.ones(num_experts, N, device="cuda", dtype=torch.float32))
d_out = torch.empty(total_tokens, N, device="cuda", dtype=torch.bfloat16)
m_indices = torch.zeros(total_tokens, device="cuda", dtype=torch.int32)  # Build appropriately

dg.m_grouped_fp8_gemm_nt_contiguous(a_concat, b_experts, d_out, m_indices)
```

### MQA Logits

```python
seq_len, num_heads, head_dim = 1024, 32, 128
seq_len_kv = 2048

q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float8_e4m3fn)
kv = (torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.float8_e4m3fn),
      torch.ones(seq_len_kv, device="cuda", dtype=torch.float32))
weights = torch.ones(seq_len, num_heads, device="cuda", dtype=torch.float32)
cu_start = torch.zeros(seq_len, device="cuda", dtype=torch.int32)
cu_end = torch.full((seq_len,), seq_len_kv, device="cuda", dtype=torch.int32)

logits = torch.empty(seq_len, seq_len_kv, device="cuda", dtype=torch.float32)
dg.fp8_mqa_logits(q, kv, logits, cu_start, cu_end, clean_logits=True)
```

---

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `DG_JIT_DEBUG` | `0`, `1` | Print JIT debugging info |
| `DG_JIT_CACHE_DIR` | path | Kernel cache directory (default: `$HOME/.deep_gemm`) |
| `DG_JIT_USE_NVRTC` | `0`, `1` | Use NVRTC instead of NVCC (faster compile, maybe slower runtime) |
| `DG_JIT_NVCC_COMPILER` | path | Custom NVCC compiler path |
| `DG_JIT_PTXAS_VERBOSE` | `0`, `1` | Show detailed PTXAS output |
| `DG_PRINT_CONFIGS` | `0`, `1` | Print selected configs per shape |

---

## Version

This documentation is for DeepGEMM v2.2.0.
