"""Device selection helper — returns the best available compute device (CUDA, MPS, or CPU)."""
# device.py
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
