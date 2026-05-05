"""Node manager package — exports NodeManager and SSH target builder."""
from .manager import NodeManager
from .ssh import _build_ssh_target

__all__ = ["NodeManager", "_build_ssh_target"]
