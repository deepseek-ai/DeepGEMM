# GPU Architecture Support Matrix and Porting Notes

## Overview

DeepGEMM has been extended to support NVIDIA **SM80 (Ampere)** and **SM89 (Ada Lovelace)** architectures while fully preserving the existing **SM90 (Hopper)** and **SM100 (Blackwell)** implementations.

---

## Architecture Support Matrix

| Operation / Feature | SM80 (Ampere) | SM89 (Ada Lovelace) | SM90 (Hopper) | SM100 (Blackwell) |
| :--- | :---: | :---: | :---: | :---: |
| **Dense BF16 GEMM (`bf16_gemm_*`)** | cuBLASLt Fallback | cuBLASLt Fallback | Native SM90 Kernel / cuBLASLt | Native SM100 Kernel / cuBLASLt |
| **Dense FP8 GEMM (`fp8_gemm_*`)** | Unsupported (Hardware Limit) | Unsupported (Hardware Limit) | Native SM90 Kernel (1D1D / 1D2D) | Native SM100 Kernel |
| **Microscaled FP4 GEMM (`fp4_gemm_*`)** | Unsupported (Blackwell Only) | Unsupported (Blackwell Only) | Unsupported (Blackwell Only) | Native SM100 FP4 Kernel |
| **Grouped BF16 GEMM (`m_grouped_bf16_*`)** | Unsupported | Unsupported | Native SM90 Kernel | Native SM100 Kernel |
| **Grouped FP8 GEMM (`m_grouped_fp8_*`)** | Unsupported (Hardware Limit) | Unsupported (Hardware Limit) | Native SM90 Kernel | Native SM100 Kernel |
| **V3.2 MQA Logits (`fp8_fp4_mqa_logits`)** | Unsupported (SM90/SM100 Only) | Unsupported (SM90/SM100 Only) | Native SM90 Kernel | Native SM100 Kernel |
| **Mega MoE (`fp8_fp4_mega_moe`)** | Unsupported (SM90/SM100 Only) | Unsupported (SM90/SM100 Only) | Native SM90 Kernel | Native SM100 Kernel |
| **HyperConnection & Mega Gate / mHC** | Unsupported (SM90/SM100 Only) | Unsupported (SM90/SM100 Only) | Native SM90 Kernel | Native SM100 Kernel |

---

## Hardware Limitations & Error Handling

1. **SM80 (Ampere) & SM89 (Ada Lovelace)**:
   - Standard dense BF16 GEMMs automatically dispatch through cuBLASLt fallbacks across SM80 and SM89.
   - Low-precision operations (FP8 / FP4) or Hopper/Blackwell-specific features (TMA, WGMMA, Mega MoE, MQA Logits) invoked on SM80 or SM89 raise explicit `RuntimeError` panics detailing the hardware capability limits.

2. **SM90 (Hopper) & SM100 (Blackwell)**:
   - Retain full native performance, TMA/WGMMA pipelines, UE8M0 scaling, and custom epilogue support without any changes or regressions.

---

## Running Validation Tests on GPU

A dedicated physical hardware test suite is provided in `tests/test_sm80_sm89.py`.

To run tests when cloned onto a system with an SM80 or SM89 GPU:

```bash
python3 -m unittest tests/test_sm80_sm89.py
```

This test checks:
1. GPU Compute Capability detection.
2. Correct execution of BF16 GEMM via cuBLASLt fallback against PyTorch references.
3. Proper handling and clear runtime architecture error messages for unsupported low-precision hardware features on SM80 and SM89.
