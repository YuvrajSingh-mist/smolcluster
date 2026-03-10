"""Pytest configuration and shared fixtures for smolcluster tests."""

import torch
import pytest


@pytest.fixture
def device():
    """Provide a CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_vocab_size():
    """Small vocabulary size for testing."""
    return 1000


@pytest.fixture
def small_seq_len():
    """Small sequence length for testing."""
    return 64


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 2


@pytest.fixture
def model_dim():
    """Small model dimension for testing."""
    return 128


@pytest.fixture
def num_heads():
    """Number of attention heads for testing."""
    return 4


@pytest.fixture
def num_layers():
    """Number of layers for testing."""
    return 2


@pytest.fixture
def sample_input(batch_size, small_seq_len, small_vocab_size):
    """Create a sample input tensor."""
    return torch.randint(0, small_vocab_size, (batch_size, small_seq_len))


@pytest.fixture
def sample_labels(batch_size, small_seq_len, small_vocab_size):
    """Create sample labels tensor."""
    return torch.randint(0, small_vocab_size, (batch_size, small_seq_len))


# MoE-specific fixtures
@pytest.fixture
def num_experts():
    """Number of experts for MoE testing."""
    return 4


@pytest.fixture
def top_k_experts():
    """Number of top-k experts for MoE routing."""
    return 2


# Configuration fixtures
@pytest.fixture
def dropout_rate():
    """Dropout rate for testing."""
    return 0.1


@pytest.fixture
def ff_dim(model_dim):
    """Feed-forward dimension for testing."""
    return model_dim * 4


# Skip tests requiring CUDA if not available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip CUDA tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
