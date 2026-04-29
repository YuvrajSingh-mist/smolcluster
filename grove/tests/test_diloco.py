"""DiLoCo tests (single-process, world_size=1)."""

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

from grove.diloco import DiLoCo, _quantize_e3m0, _dequantize_e3m0


def test_diloco_basic():
    mx.random.seed(42)
    model = nn.Linear(16, 1)
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))
    diloco = DiLoCo(model, H=5, outer_lr=0.7, outer_momentum=0.9)

    X = mx.random.normal((32, 16))
    y = mx.random.normal((32, 1))

    losses = []
    for step in range(10):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.append(loss.item())
        diloco.step(model)

    assert diloco.outer_step == 2
    assert losses[-1] < losses[0]


def test_diloco_convergence():
    mx.random.seed(42)
    model = nn.Linear(8, 1)
    optimizer = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))
    diloco = DiLoCo(model, H=3, outer_lr=0.1, outer_momentum=0.9)

    X = mx.random.normal((16, 8))
    y = mx.random.normal((16, 1))

    initial_loss = None
    for step in range(30):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        diloco.step(model)
        if initial_loss is None:
            initial_loss = loss.item()

    assert loss.item() < initial_loss
    assert diloco.outer_step == 10


def test_e3m0_roundtrip():
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32) * 0.1
    code, scale = _quantize_e3m0(x)
    reconstructed = _dequantize_e3m0(code, scale)
    assert code.dtype == np.uint8
    assert np.all(code < 16)
    rel_error = np.mean(np.abs(x - reconstructed) / (np.abs(x) + 1e-8))
    assert rel_error < 0.5


def test_e3m0_zeros():
    x = np.zeros(100, dtype=np.float32)
    code, scale = _quantize_e3m0(x)
    reconstructed = _dequantize_e3m0(code, scale)
    assert np.allclose(reconstructed, 0.0)


def test_diloco_async():
    mx.random.seed(42)
    model = nn.Linear(16, 1)
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))
    diloco = DiLoCo(model, H=5, outer_lr=0.7, outer_momentum=0.9, overlap=True)

    X = mx.random.normal((32, 16))
    y = mx.random.normal((32, 1))

    losses = []
    for step in range(10):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.append(loss.item())
        diloco.step(model)

    assert diloco.outer_step == 2
    assert losses[-1] < losses[0]


def test_diloco_quantize():
    mx.random.seed(42)
    model = nn.Linear(16, 1)
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))
    diloco = DiLoCo(model, H=5, outer_lr=0.7, outer_momentum=0.9, quantize=True)

    X = mx.random.normal((32, 16))
    y = mx.random.normal((32, 1))

    losses = []
    for step in range(10):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.append(loss.item())
        diloco.step(model)

    assert diloco.outer_step == 2
    assert losses[-1] < losses[0]
