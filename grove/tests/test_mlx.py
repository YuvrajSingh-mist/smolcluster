"""MLX gradient packing tests (single-threaded — Metal is not thread-safe)."""

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

from grove.mlx_comm import _GradientPacker


def test_flatten_unflatten_roundtrip():
    grads = {
        "layers": {
            "weight": mx.ones((10, 10)),
            "bias": mx.full((10,), 2.0),
        }
    }
    packer = _GradientPacker()
    buf = packer.flatten(grads)
    assert buf.dtype == np.float32
    assert buf.shape == (110,)  # 100 + 10

    result = packer.unflatten(buf, 1.0, grads)
    assert mx.allclose(result["layers"]["weight"], mx.ones((10, 10))).item()
    assert mx.allclose(result["layers"]["bias"], mx.full((10,), 2.0)).item()


def test_flatten_unflatten_with_scale():
    grads = {
        "w": mx.full((4, 4), 6.0),
    }
    packer = _GradientPacker()
    buf = packer.flatten(grads)
    result = packer.unflatten(buf, 0.5, grads)
    assert mx.allclose(result["w"], mx.full((4, 4), 3.0)).item()


def test_packer_preserves_nested_structure():
    grads = {
        "encoder": {
            "layer0": {"weight": mx.ones((8, 8)), "bias": mx.zeros((8,))},
            "layer1": {"weight": mx.full((8, 8), 2.0)},
        },
        "head": {"weight": mx.full((4, 8), 3.0)},
    }
    packer = _GradientPacker()
    buf = packer.flatten(grads)
    result = packer.unflatten(buf, 1.0, grads)

    assert mx.allclose(result["encoder"]["layer0"]["weight"], mx.ones((8, 8))).item()
    assert mx.allclose(result["encoder"]["layer0"]["bias"], mx.zeros((8,))).item()
    assert mx.allclose(result["encoder"]["layer1"]["weight"], mx.full((8, 8), 2.0)).item()
    assert mx.allclose(result["head"]["weight"], mx.full((4, 8), 3.0)).item()


def test_simulated_average():
    """Simulate what average_gradients does: flatten from N ranks, sum, scale by 1/N."""
    packer1 = _GradientPacker()
    packer2 = _GradientPacker()

    grads1 = {"w": mx.full((4,), 1.0)}
    grads2 = {"w": mx.full((4,), 3.0)}

    buf1 = packer1.flatten(grads1)
    buf2 = packer2.flatten(grads2)

    summed = buf1 + buf2
    result = packer1.unflatten(summed, 0.5, grads1)
    assert mx.allclose(result["w"], mx.full((4,), 2.0)).item()  # (1+3)/2
