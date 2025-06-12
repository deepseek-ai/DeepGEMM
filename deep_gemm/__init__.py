import os

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# All modules
from . import (
    dispatch,
    jit,
    jit_kernels,
    testing,
    utils
)

# All kernels
from .dispatch import *

# Some useful utils
from .utils.layout import (
    get_device_arch,
    get_m_alignment_for_contiguous_layout,
)
