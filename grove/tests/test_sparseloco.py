"""SparseLoCo compressed DiLoCo tests."""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

from grove.sparseloco import SparseLoCo


def test_sparseloco_basic():
    mx.random.seed(42)
    model = nn.Linear(64, 64)
    optimizer = optim.Adam(learning_rate=1e-3)
    sloco = SparseLoCo(model, outer_lr=0.7, H=5, topk=16, chunk=64)

    X = mx.random.normal((16, 64))
    y = mx.random.normal((16, 64))
    loss_fn = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))

    losses = []
    synced = False
    for step in range(15):
        loss, grads = loss_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.append(loss.item())
        if sloco.step(model):
            synced = True

    assert synced
    assert sloco.outer_step == 3
    assert losses[-1] < losses[0]


def test_sparseloco_convergence():
    mx.random.seed(42)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(32, 64), nn.Linear(64, 32)]
        def __call__(self, x):
            x = nn.relu(self.layers[0](x))
            return self.layers[1](x)

    model = MLP()
    optimizer = optim.Adam(learning_rate=1e-3)
    sloco = SparseLoCo(model, outer_lr=0.5, H=5, topk=32, chunk=64)

    X = mx.random.normal((16, 32))
    y = mx.random.normal((16, 32))
    loss_fn = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))

    initial_loss = None
    for step in range(50):
        loss, grads = loss_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        if initial_loss is None:
            initial_loss = loss.item()
        sloco.step(model)

    assert loss.item() < initial_loss
    assert sloco.outer_step == 10


def test_sparseloco_async():
    mx.random.seed(42)
    model = nn.Linear(64, 64)
    optimizer = optim.Adam(learning_rate=1e-3)
    sloco = SparseLoCo(model, outer_lr=0.7, H=5, topk=16, chunk=64, overlap=True)

    X = mx.random.normal((16, 64))
    y = mx.random.normal((16, 64))
    loss_fn = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))

    losses = []
    for step in range(15):
        loss, grads = loss_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.append(loss.item())
        sloco.step(model)

    assert sloco.outer_step == 3
    assert losses[-1] < losses[0]


def test_sparseloco_compression_ratio():
    model = nn.Linear(4096, 4096)
    sloco = SparseLoCo(model, H=30, topk=64, chunk=4096)
    total_dense = sum(c.n_chunks * c.chunk_size for c in sloco._compressors)
    total_sparse = sum(c.n_chunks * c.topk for c in sloco._compressors)
    ratio = total_dense / total_sparse
    assert ratio >= 50, f"Expected >= 50x compression, got {ratio:.0f}x"
