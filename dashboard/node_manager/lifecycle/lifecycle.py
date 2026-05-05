"""_LifecycleMixin — spawns training server + remote SSH workers, and starts inference processes."""
import asyncio
import logging
import os
import shlex

from ..constants import REMOTE_REPO
from ..ssh import _build_ssh_target

logger = logging.getLogger(__name__)


class _LifecycleMixin:
    async def start_training(self, algorithm: str, server_hostname: str) -> None:
        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        server_cmd = [
            "uv", "run", "python", "-m", "smolcluster.train",
            "server", server_hostname, "--algorithm", algorithm,
        ]
        self._log(server_hostname, f"$ {shlex.join(server_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *server_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": algorithm, "role": "server", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))
        await asyncio.sleep(2)  # let server bind socket

        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank     = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target   = _build_ssh_target(ssh_user, hostname)
            remote   = (
                f"cd {REMOTE_REPO} && PYTHONUNBUFFERED=1 "
                f"uv run python -m smolcluster.train "
                f"worker {rank} {hostname} --algorithm {algorithm}"
            )
            cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                target, remote,
            ]
            self._log(hostname, f"$ ssh {target} [rank {rank}]")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            async with self._lock:
                self.processes[hostname] = {
                    "rank": rank, "algorithm": algorithm, "role": "worker", "proc": proc,
                }
            asyncio.create_task(self._stream(hostname, proc))

    async def start_inference(self, server_hostname: str) -> None:
        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        num_workers = len(self.selected)

        server_cmd = [
            "uv", "run", "python", "-m", "smolcluster.applications.infer",
            "server", server_hostname, str(num_workers),
        ]
        self._log(server_hostname, f"$ {shlex.join(server_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *server_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": "infer", "role": "server", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))
        await asyncio.sleep(1)

        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank     = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target   = _build_ssh_target(ssh_user, hostname)
            remote   = (
                f"cd {REMOTE_REPO} && PYTHONUNBUFFERED=1 "
                f"uv run python -m smolcluster.applications.infer "
                f"worker {rank} {hostname} {server_hostname}"
            )
            cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                target, remote,
            ]
            self._log(hostname, f"$ ssh {target} → infer worker rank {rank}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            async with self._lock:
                self.processes[hostname] = {
                    "rank": rank, "algorithm": "infer", "role": "worker", "proc": proc,
                }
            asyncio.create_task(self._stream(hostname, proc))
