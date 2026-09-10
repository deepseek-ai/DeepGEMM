import os
import torch

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
from . import _C
from ._C import (
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
    set_ignore_compile_dims,
    set_block_size_multiple_of,
    set_pdl,
    get_pdl,
    use_deterministic_algorithms,
)

# cuBLASLt Kernels
from ._C import (
    cublaslt_gemm_nt, cublaslt_gemm_nn,
    cublaslt_gemm_tn, cublaslt_gemm_tt,
    cublaslt_nvfp4_gemm_nt,
    batched_syrk, batched_symm,
)

# DeepGEMM Kernels
from ._C import (
    # FP8-FP4 GEMMs (accept any FP8/FP4 operand combination)
    fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,
    fp8_fp4_gemm_tn, fp8_fp4_gemm_tt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nn_contiguous,
    m_grouped_fp8_fp4_gemm_nt_masked,
    # FP8/FP4 GEMMs (alias)
    fp4_gemm_nt,
    fp8_gemm_nt, fp8_gemm_nn,
    fp8_gemm_tn, fp8_gemm_tt,
    fp8_gemm_nt_skip_head_mid,
    m_grouped_fp4_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    m_grouped_fp4_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_masked,
    k_grouped_fp4_gemm_nt_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_tn_contiguous,
    # BF16 GEMMs
    bf16_gemm_nt, bf16_gemm_nn,
    bf16_gemm_tn, bf16_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    k_grouped_bf16_gemm_tn_contiguous,
    # MegaGate kernels
    bf16_mega_gate, get_bf16_mega_gate_config,
    # Einsum kernels
    einsum,
    fp8_einsum,
    # Attention kernels
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    get_paged_mqa_logits_metadata,
    get_sparse_mqa_logits_metadata,
    get_paged_sparse_mqa_logits_metadata,
    fp8_fp4_sparse_mqa_logits,
    fp8_fp4_paged_sparse_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    # Attention kernels (legacy)
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    # Hyperconnection kernels
    mega_mhc,
    tf32_hc_prenorm_gemm,
    # Layout kernels
    transform_sf_into_required_layout,
    # MegaMoE
    get_block_m_for_mega_moe,
)

# Mega kernels
from .mega import (
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
    bf16_mega_moe,
)

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {e}')

# Initialize CPP modules
_C.init(os.path.dirname(os.path.abspath(__file__)))

__version__ = '2.8.0'
