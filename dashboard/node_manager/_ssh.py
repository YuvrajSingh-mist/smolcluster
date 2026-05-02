import asyncio
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def _build_ssh_target(ssh_user: str, hostname: str) -> str:
    """
    Build the SSH target from whatever the user provided:
      - bare alias  "mini2"               → use as-is  (SSH config handles user+key)
      - username    "yuvrajsingh2"        → "yuvrajsingh2@hostname.local"
      - user@host   "yuvrajsingh2@mini2"  → use as-is
      - empty                             → "hostname.local"
    Heuristic: a bare alias has no '@' and no '.' in it.
    """
    if not ssh_user:
        return f"{hostname}.local"
    if "@" in ssh_user:
        return ssh_user
    if "." not in ssh_user:
        return ssh_user
    return f"{ssh_user}@{hostname}.local"


class _ProbeMixin:
    @staticmethod
    async def probe_username(hostname: str, ssh_user: str = "") -> Optional[str]:
        info = await _ProbeMixin.probe_metadata(hostname, ssh_user)
        if info:
            return info.get("username") or None
        return None

    @staticmethod
    async def probe_metadata(hostname: str, ssh_user: str = "") -> Optional[dict]:
        target = _build_ssh_target(ssh_user, hostname)
        remote = (
            "sh -lc 'whoami; uname -s; uname -r; uname -m; "
            "if command -v sw_vers >/dev/null 2>&1; then sw_vers -productVersion; else echo; fi'"
        )
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            target, remote,
        ]
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True),
                timeout=8.0,
            )
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines()]
                if len(lines) >= 4:
                    os_version = lines[4] if len(lines) >= 5 and lines[4] else lines[2]
                    return {
                        "username":   lines[0],
                        "os":         lines[1],
                        "os_version": os_version,
                        "machine":    lines[3],
                    }
        except Exception as e:
            logger.debug(f"[node_manager] probe {hostname}: {e}")
        return None
