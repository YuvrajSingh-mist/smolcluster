"""Distributed training across multiple machines using PyTorch.

    grove start examples/distributed_train_torch.py -n 2
"""

import grove


def main():
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

    world = grove.init()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.layers(x)

    torch.manual_seed(42)
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    np.random.seed(world.rank())
    X = torch.tensor(np.random.randn(64, 8).astype(np.float32))
    y = X[:, :3].sum(dim=1, keepdim=True)

    for step in range(1000):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()

        # average gradients across all workers
        grads = {name: param.grad for name, param in model.named_parameters()}
        grads = grove.average_gradients(grads)
        for name, param in model.named_parameters():
            param.grad = grads[name]

        optimizer.step()
        grove.report(loss=loss.item())

        if world.rank() == 0 and step % 10 == 0:
            print(f"  Step {step:>3}: loss = {loss.item():.4f}")

    if world.rank() == 0:
        torch.save(model.state_dict(), "model.pt")
        print(f"  Done across {world.size()} workers")


if __name__ == "__main__":
    main()
