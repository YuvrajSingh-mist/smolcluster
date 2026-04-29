"""DeMo compressed optimizer tests."""

import pytest
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

from grove.demo import DeMo
from grove.compress import TopKCompressor, _make_dct_basis


def test_dct_roundtrip():
    for n in [8, 16, 32, 64]:
        basis = _make_dct_basis(n)
        x = np.random.randn(n).astype(np.float32)
        encoded = basis @ x
        decoded = basis.T @ encoded
        assert np.allclose(x, decoded, atol=1e-4), f"DCT roundtrip failed for n={n}"


def test_compressor_roundtrip_dct():
    comp = TopKCompressor(total_elems=256, chunk_target=64, topk=64, use_dct=True)
    x = np.random.randn(256).astype(np.float32)
    idx, val, transmitted = comp.compress(x)
    assert np.allclose(x, transmitted, atol=1e-4)


def test_compressor_roundtrip_direct():
    comp = TopKCompressor(total_elems=256, chunk_target=64, topk=64, use_dct=False)
    x = np.random.randn(256).astype(np.float32)
    idx, val, transmitted = comp.compress(x)
    assert np.allclose(x, transmitted, atol=1e-6)


def test_compressor_sparsity():
    comp = TopKCompressor(total_elems=256, chunk_target=64, topk=8, use_dct=True)
    x = np.random.randn(256).astype(np.float32)
    idx, val, transmitted = comp.compress(x)
    assert len(idx) == 4 * 8  # 4 chunks * 8 topk
    assert len(val) == 4 * 8


def test_demo_converges():
    mx.random.seed(42)
    model = nn.Linear(64, 64)
    demo = DeMo(model, lr=1e-2, topk=8, chunk=32)

    X = mx.random.normal((16, 64))
    y = mx.random.normal((16, 64))
    loss_fn = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))

    losses = []
    for step in range(50):
        loss, grads = loss_fn(model, X, y)
        mx.eval(loss)
        losses.append(loss.item())
        demo.step(model, grads)

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert demo.step_count == 50


def test_demo_mlp():
    mx.random.seed(42)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(32, 64), nn.Linear(64, 32)]
        def __call__(self, x):
            x = nn.relu(self.layers[0](x))
            return self.layers[1](x)

    model = MLP()
    demo = DeMo(model, lr=5e-3, topk=8, chunk=32)

    X = mx.random.normal((16, 32))
    y = mx.random.normal((16, 32))
    loss_fn = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))

    initial_loss = None
    for step in range(30):
        loss, grads = loss_fn(model, X, y)
        mx.eval(loss)
        if initial_loss is None:
            initial_loss = loss.item()
        demo.step(model, grads)

    assert loss.item() < initial_loss
