# SmolCluster Tests

Pytest test suite for GPT-2 and MoE model implementations.

## Installation

```bash
uv pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=smolcluster --cov-report=html

# Specific test file
pytest src/smolcluster/tests/test_gpt.py
pytest src/smolcluster/tests/test_moe.py

# Specific test class or function
pytest src/smolcluster/tests/test_gpt.py::TestBaseTransformer
pytest src/smolcluster/tests/test_gpt.py::TestBaseTransformer::test_forward_pass

# Skip slow or CUDA tests
pytest -m "not slow"
pytest -m "not cuda"
```

## Test Coverage

**GPT-2 (BaseTransformer):** Initialization, forward pass, parameter counting, weight initialization, gradient flow, weight tying, sequence length variations, dropout, eval mode, batch independence, training steps, overfitting, CUDA compatibility.

**MoE (Mixtral):** Swish activation, rotary embeddings, text embeddings, layer normalization, SWiGLU experts, MoE routing, decoder blocks, full model, expert configurations, flash attention, gradient flow, training steps, overfitting, CUDA compatibility.

## Test Structure

```
tests/
├── conftest.py    # Shared fixtures
├── test_gpt.py    # GPT-2 tests
└── test_moe.py    # MoE tests
```

## CI Integration

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - run: uv pip install -e ".[dev]"
    - run: pytest --cov=smolcluster
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=smolcluster`
4. Format code: `ruff format .`
5. Run linter: `ruff check .`

