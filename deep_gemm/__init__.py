import os
import subprocess

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
import deep_gemm_cpp
from deep_gemm_cpp import (
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
)

# Kernels
from deep_gemm_cpp import (
    # FP8 GEMMs
    fp8_gemm_nt, fp8_gemm_nn,
    fp8_gemm_tn, fp8_gemm_tt,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    k_grouped_fp8_gemm_tn_contiguous,
    # BF16 GEMMs
    bf16_gemm_nt, bf16_gemm_nn,
    bf16_gemm_tn, bf16_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    # Layout kernels
    transform_sf_into_required_layout
)

# Some alias for legacy supports
# TODO: remove these later
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked

# Some utils
from . import testing
from . import utils
from .utils import *


# Initialize CPP modules
def _find_cuda_home() -> str:
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


deep_gemm_cpp.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    _find_cuda_home()                           # CUDA home
)
