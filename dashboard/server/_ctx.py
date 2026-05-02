"""Shared mutable state for the dashboard server.

All module-level globals that are initialized in `lifespan` live here so that
route handlers and background tasks can import this module and read the current
values without circular-import issues.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    from dashboard.node_manager import NodeManager

# Set by lifespan at startup — typed as Any here to avoid import cycles at runtime.
node_manager: "NodeManager | None" = None
server_hostname: str = ""
redis: "aioredis.Redis | None" = None

# Populated from ~/.ssh/config at startup; keyed by SSH alias.
static_nodes: dict = {}

# hostname → SSH alias (initially mirrors static_nodes keys)
ssh_aliases: dict = {}

# Lazily populated: hostname → probed SSH username
probed: dict = {}

# Lazily populated: hostname → {os, os_version, machine}
node_os: dict = {}
