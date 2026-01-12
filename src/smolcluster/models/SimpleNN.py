# model.py
import torch.nn as nn


class SimpleMNISTModel(nn.Module):
    def __init__(self, input_dim: int = 784, hidden: int = 128, out: int = 10):
        super().__init__()
        # Fully connected network: 784 → 256 → 128 → 10
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.norm = nn.LayerNorm(256)
        # self.fc2 = nn.Linear(256, 4 * hidden)
        self.relu2 = nn.ReLU()
        # self.norm2 = nn.LayerNorm(4 * hidden)
        self.fc3 = nn.Linear(4 * hidden, out)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
