"""
Pytest configuration and fixtures for DeepGEMM tests.
"""

import os
import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "sm90: marks tests for Hopper (SM90) GPU")
    config.addinivalue_line("markers", "sm100: marks tests for Blackwell (SM100) GPU")
    config.addinivalue_line("markers", "fp8: marks FP8 GEMM tests")
    config.addinivalue_line("markers", "bf16: marks BF16 GEMM tests")
    config.addinivalue_line("markers", "attention: marks attention kernel tests")
    config.addinivalue_line("markers", "einsum: marks einsum tests")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on file names and skip if GPU not available."""
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    skip_no_cuda = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        # Auto-mark based on test file name
        if "test_fp8" in item.nodeid:
            item.add_marker(pytest.mark.fp8)
            item.add_marker(pytest.mark.gpu)
        elif "test_bf16" in item.nodeid:
            item.add_marker(pytest.mark.bf16)
            item.add_marker(pytest.mark.gpu)
        elif "test_attention" in item.nodeid:
            item.add_marker(pytest.mark.attention)
            item.add_marker(pytest.mark.gpu)
        elif "test_einsum" in item.nodeid:
            item.add_marker(pytest.mark.einsum)
            item.add_marker(pytest.mark.gpu)

        # Skip GPU tests if CUDA not available
        if "gpu" in [m.name for m in item.iter_markers()]:
            if not cuda_available:
                item.add_marker(skip_no_cuda)


@pytest.fixture(scope="session")
def cuda_device():
    """Fixture providing CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def gpu_arch():
    """Fixture providing GPU architecture major version."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    major, _ = torch.cuda.get_device_capability()
    return major


@pytest.fixture(scope="session")
def is_hopper(gpu_arch):
    """Fixture checking if GPU is Hopper (SM90)."""
    return gpu_arch == 9


@pytest.fixture(scope="session")
def is_blackwell(gpu_arch):
    """Fixture checking if GPU is Blackwell (SM100)."""
    return gpu_arch == 10


@pytest.fixture(scope="session")
def deep_gemm_module():
    """Fixture providing the deep_gemm module."""
    import deep_gemm
    return deep_gemm


@pytest.fixture(autouse=True)
def reset_cuda_memory():
    """Automatically reset CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def random_seed():
    """Fixture for reproducible random state."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture(scope="session")
def env_vars():
    """Fixture providing commonly used environment variables."""
    return {
        "DG_CACHE_DIR": os.environ.get("DG_CACHE_DIR"),
        "DG_JIT_DEBUG": os.environ.get("DG_JIT_DEBUG"),
        "DG_JIT_USE_NVRTC": os.environ.get("DG_JIT_USE_NVRTC"),
        "DG_MINIMIZE_NUM_SMS": os.environ.get("DG_MINIMIZE_NUM_SMS"),
    }
