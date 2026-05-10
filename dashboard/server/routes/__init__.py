"""Route sub-package — groups API handlers by domain."""
from . import inference, misc, nodes, training

__all__ = ["inference", "nodes", "training", "misc"]
