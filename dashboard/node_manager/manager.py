"""NodeManager — central coordinator that composes lifecycle, launch, cleanup, and SSH probe mixins."""
import asyncio
import logging
import time
from typing import Dict, Optional

from .constants import _sanitize_log_line
from .ssh import _ProbeMixin
from .lifecycle import _LifecycleMixin, _CleanupMixin
from .launch import _TrainLaunchMixin, _InferLaunchMixin

logger = logging.getLogger(__name__)


class NodeManager(
    _LifecycleMixin,
    _InferLaunchMixin,
    _TrainLaunchMixin,
    _CleanupMixin,
    _ProbeMixin,
):
    """
    selected:  hostname → {ssh_user, rank}
    processes: hostname → {proc, rank, algorithm, role}
    """

    def __init__(self):
        self.selected:  Dict[str, dict] = {}
        self.processes: Dict[str, dict] = {}
        self._lock  = asyncio.Lock()
        # Unbounded queue — drained by _log_broadcaster into the Redis Stream.
        self._queue: asyncio.Queue = asyncio.Queue()

    # ── Log helper ─────────────────────────────────────────────────────────────

    def _log(self, hostname: str, line: str) -> None:
        self._queue.put_nowait({
            "hostname": hostname, "line": line, "session": "", "ts": time.time(),
        })

    # ── Selection ──────────────────────────────────────────────────────────────

    async def select(self, hostname: str, ssh_user: str = "",
                     rank: Optional[int] = None) -> int:
        async with self._lock:
            if rank is None:
                taken = {v["rank"] for v in self.selected.values()}
                rank  = next(r for r in range(1, 100) if r not in taken)
            self.selected[hostname] = {"ssh_user": ssh_user, "rank": rank}
        logger.info(f"[node_manager] Selected {hostname} as rank {rank}")
        return rank

    async def deselect(self, hostname: str) -> None:
        async with self._lock:
            self.selected.pop(hostname, None)
        logger.info(f"[node_manager] Deselected {hostname}")

    # ── Snapshots ──────────────────────────────────────────────────────────────

    def snapshot_selected(self) -> Dict[str, dict]:
        return {
            h: {"rank": v["rank"], "ssh_user": v["ssh_user"]}
            for h, v in self.selected.items()
        }

    def snapshot_processes(self) -> Dict[str, dict]:
        def _status(v):
            rc = v["proc"].returncode
            if rc is None:
                return "running"
            if v["role"].endswith("_launcher") and rc == 0:
                return "launched"
            return f"exited:{rc}"

        return {
            h: {
                "rank": v["rank"], "algorithm": v["algorithm"],
                "role": v["role"], "status": _status(v),
            }
            for h, v in self.processes.items()
        }

    async def stop_all(self) -> None:
        await self.stop_training()
        async with self._lock:
            self.selected.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _stream(self, hostname: str, proc) -> None:
        """Read stdout/stderr, feed into log buffer, clean up on exit."""
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = _sanitize_log_line(raw.decode(errors="replace")).rstrip()
                if line:
                    self._log(hostname, line)
        except Exception as e:
            logger.debug(f"[node_manager] stream {hostname}: {e}")
        finally:
            await proc.wait()
            async with self._lock:
                info = self.processes.get(hostname, {})
                # Keep launcher entries alive so topology stays visible after the script
                # exits (inference continues in remote tmux sessions).
                if not (str(info.get("role", "")).endswith("_launcher") and proc.returncode == 0):
                    self.processes.pop(hostname, None)
            logger.info(f"[node_manager] {hostname} exited (rc={proc.returncode})")
