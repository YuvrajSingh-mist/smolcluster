"""Distributed training across multiple Macs.

    grove start examples/distributed_train.py -n 2
"""

import grove


def main():
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    world = grove.init()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(8, 32), nn.Linear(32, 32), nn.Linear(32, 1)]

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = nn.relu(layer(x))
            return self.layers[-1](x)

    mx.random.seed(42)
    model = MLP()
    optimizer = optim.SGD(learning_rate=0.01)

    np.random.seed(world.rank())
    X = mx.array(np.random.randn(64, 8).astype(np.float32))
    y = mx.sum(X[:, :3], axis=1, keepdims=True)

    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))

    for step in range(1000):
        loss, grads = loss_and_grad(model, X, y)
        grads = grove.average_gradients(grads)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        grove.report(loss=loss.item())
        if world.rank() == 0 and step % 10 == 0:
            print(f"  Step {step:>3}: loss = {loss.item():.4f}")

    if world.rank() == 0:
        model.save_weights("model.safetensors")
        print(f"  Done across {world.size()} workers")


if __name__ == "__main__":
    main()
