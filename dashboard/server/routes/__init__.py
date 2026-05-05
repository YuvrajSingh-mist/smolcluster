"""Route sub-package — groups API handlers by domain."""
from . import inference, nodes, training, misc

__all__ = ["inference", "nodes", "training", "misc"]
