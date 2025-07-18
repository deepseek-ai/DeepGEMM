import os
import torch
import torch.utils.cpp_extension

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Import functions from the CPP module
import deep_gemm_cpp
deep_gemm_cpp.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    torch.utils.cpp_extension.CUDA_HOME         # CUDA home
)

# Configs
from deep_gemm_cpp import (
    set_num_sms,
    get_num_sms
)

# Kernels
from deep_gemm_cpp import (
    fp8_gemm_nt, fp8_gemm_nn,
    fp8_gemm_tn, fp8_gemm_tt,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    fp8_m_grouped_gemm_nt_masked,
    k_grouped_fp8_gemm_tn_contiguous
)

# Some utils
from . import testing
from . import utils
from .utils import *
