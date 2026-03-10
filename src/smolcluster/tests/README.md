# SmolCluster Tests

Professional pytest test suite for smolcluster models.

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and pytest configuration
├── test_gpt.py          # GPT-2 (BaseTransformer) model tests
├── test_moe.py          # MoE (Mixtral) model tests
└── README.md            # This file
```

## Running Tests

### Install Test Dependencies

```bash
# Install with dev dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# From project root
pytest

# With coverage
pytest --cov=smolcluster --cov-report=html

# Verbose output
pytest -v
```

### Run Specific Test Files

```bash
# Run only GPT tests
pytest src/smolcluster/tests/test_gpt.py

# Run only MoE tests
pytest src/smolcluster/tests/test_moe.py
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest src/smolcluster/tests/test_gpt.py::TestBaseTransformer

# Run specific test function
pytest src/smolcluster/tests/test_gpt.py::TestBaseTransformer::test_forward_pass
```

### Filter by Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only CUDA tests (if CUDA is available)
pytest -m cuda

# Skip CUDA tests
pytest -m "not cuda"
```

## Test Coverage

The test suite covers:

### GPT-2 (BaseTransformer) Tests
- ✅ Model initialization
- ✅ Forward pass correctness
- ✅ Shape validation
- ✅ Parameter counting
- ✅ Weight initialization
- ✅ Gradient flow
- ✅ Weight tying
- ✅ Different sequence lengths
- ✅ Dropout effects
- ✅ Deterministic eval mode
- ✅ Batch independence
- ✅ Training steps
- ✅ Overfitting capability
- ✅ CUDA compatibility (if available)
- ✅ Large model creation

### MoE (Mixtral) Tests
- ✅ Swish activation
- ✅ Rotary embeddings (RoPE)
- ✅ Text embeddings
- ✅ Layer normalization
- ✅ SWiGLU expert networks
- ✅ MoE layer with top-k routing
- ✅ Expert routing correctness
- ✅ Decoder block with MoE
- ✅ Full Mixtral model
- ✅ Different expert configurations
- ✅ Flash attention support
- ✅ Gradient flow through experts
- ✅ Training steps
- ✅ Overfitting capability
- ✅ CUDA compatibility (if available)
- ✅ Large model creation

## Test Fixtures

Shared fixtures in `conftest.py`:

- `device`: CPU device for testing
- `small_vocab_size`: Small vocabulary (1000 tokens)
- `small_seq_len`: Short sequence length (64 tokens)
- `batch_size`: Standard batch size (2)
- `model_dim`: Model dimension (128)
- `num_heads`: Number of attention heads (4)
- `num_layers`: Number of layers (2)
- `num_experts`: Number of MoE experts (4)
- `top_k_experts`: Top-k routing (2)
- `sample_input`: Random input tensor
- `sample_labels`: Random label tensor

## Test Markers

- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.cuda`: Tests requiring CUDA device

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest --cov=smolcluster --cov-report=xml -m "not cuda and not slow"
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Writing New Tests

### Test Structure

```python
import pytest
import torch
from smolcluster.models import YourModel

class TestYourModel:
    """Test suite for YourModel."""
    
    def test_initialization(self):
        """Test that model initializes correctly."""
        model = YourModel(...)
        assert model is not None
    
    def test_forward_pass(self, sample_input):
        """Test forward pass with fixtures."""
        model = YourModel(...)
        output = model(sample_input)
        assert output.shape == expected_shape
```

### Best Practices

1. **Use descriptive test names**: `test_forward_maintains_shape` not `test_1`
2. **Test one thing per test**: Keep tests focused and atomic
3. **Use fixtures**: Leverage conftest.py fixtures for common setup
4. **Add docstrings**: Explain what each test validates
5. **Test edge cases**: Don't just test the happy path
6. **Check shapes**: Always validate tensor shapes
7. **Test gradients**: Ensure gradients flow correctly
8. **Test both modes**: Check train and eval modes
9. **Use markers**: Tag slow or hardware-specific tests
10. **Keep tests fast**: Use small models/data for unit tests

## Debugging Failed Tests

```bash
# Show full output
pytest -v --tb=long

# Stop at first failure
pytest -x

# Show print statements
pytest -s

# Run specific failing test
pytest src/smolcluster/tests/test_gpt.py::TestBaseTransformer::test_forward_pass -v
```

## Performance Testing

```bash
# Profile tests
pytest --durations=10

# Show slowest tests
pytest --durations=0
```

## Code Coverage

```bash
# Generate HTML coverage report
pytest --cov=smolcluster --cov-report=html

# Open in browser
open htmlcov/index.html
```

## Integration with IDEs

### VS Code

Add to `.vscode/settings.json`:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "src/smolcluster/tests"
    ]
}
```

### PyCharm

PyCharm automatically detects pytest. Right-click on test files to run.

## Troubleshooting

### Import Errors

Make sure you've installed the package in development mode:

```bash
pip install -e .
```

### CUDA Tests Failing

CUDA tests will be skipped automatically if CUDA is not available. To force-skip:

```bash
pytest -m "not cuda"
```

### Dependencies Missing

Install all dev dependencies:

```bash
pip install -e ".[dev]"
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=smolcluster`
4. Format code: `ruff format .`
5. Run linter: `ruff check .`
