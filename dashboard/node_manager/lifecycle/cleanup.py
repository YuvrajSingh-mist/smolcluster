"""_CleanupMixin — terminates running processes and kills matching tmux sessions locally and on remote nodes."""
import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

from ..ssh import _build_ssh_target

logger = logging.getLogger(__name__)

_TMUX_PATTERN = (
    "^(server|worker[0-9]*"
    "|classicdp_worker[0-9]*|fsdp_worker[0-9]*|ep_worker[0-9]*|mp_pipeline_worker[0-9]*"
    "|syncps_inf_.*|classicdp_inf_.*|mp_inference_.*|mp_tablet_proxy[0-9]*"
    "|grpo_train|vllm_worker"
    "|syncps_api|syncps_frontend|classicdp_api|classicdp_frontend|mp_api|mp_frontend)$"
)
_KILL_CMD = (
    "tmux ls 2>/dev/null | cut -d: -f1 | "
    f"grep -E '{_TMUX_PATTERN}' | "
    "while IFS= read -r _s; do tmux kill-session -t \"$_s\" 2>/dev/null; echo \"killed:$_s\"; done"
)


class _CleanupMixin:
    async def stop_training(self) -> None:
        async with self._lock:
            procs    = dict(self.processes)
            selected = dict(self.selected)
            self.processes.clear()

        label = next(iter(procs), "local")
        self._log(label, "[stop] Terminating processes…")

        for hostname, info in procs.items():
            proc = info["proc"]
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
            logger.info(f"[node_manager] Stopped {hostname}")

        await self._cleanup_tmux_sessions(label, selected)

    async def run_cleanup_script(self, script_path: str, log_label: str) -> None:
        if not Path(script_path).exists():
            self._log(log_label, f"[stop] Cleanup script not found: {script_path}")
            return
        self._log(log_label, f"[stop] $ bash {script_path} --cleanup")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", script_path, "--cleanup",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=str(Path(script_path).parent.parent.parent),
            )
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self._log(log_label, line)
            await proc.wait()
            self._log(log_label, f"[stop] Cleanup script done (rc={proc.returncode})")
        except Exception as e:
            self._log(log_label, f"[stop] Cleanup script error: {e}")

    async def _cleanup_tmux_sessions(self, log_label: str, selected: Dict[str, dict]) -> None:
        self._log(log_label, "[stop] Killing local tmux sessions…")
        result = await asyncio.to_thread(
            subprocess.run, ["bash", "-lc", _KILL_CMD], capture_output=True, text=True,
        )
        killed = [
            ln.replace("killed:", "").strip()
            for ln in result.stdout.splitlines() if ln.startswith("killed:")
        ]
        if killed:
            for s in killed:
                self._log(log_label, f"[stop]   killed local: {s}")
        else:
            self._log(log_label, "[stop]   no matching local tmux sessions found")

        seen_targets: set = set()
        for hostname, info in selected.items():
            ssh_user = info.get("ssh_user", "")
            target   = _build_ssh_target(ssh_user, hostname)
            if target in seen_targets:
                continue
            seen_targets.add(target)
            self._log(log_label, f"[stop] Remote cleanup on {target}…")
            remote_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                target, _KILL_CMD,
            ]
            try:
                lr = await asyncio.to_thread(
                    subprocess.run, remote_cmd, capture_output=True, text=True,
                )
                r_killed = [
                    ln.replace("killed:", "").strip()
                    for ln in lr.stdout.splitlines() if ln.startswith("killed:")
                ]
                if r_killed:
                    for s in r_killed:
                        self._log(log_label, f"[stop]   killed {target}: {s}")
                else:
                    self._log(log_label, f"[stop]   no matching sessions on {target}")
            except Exception as e:
                self._log(log_label, f"[stop]   {target} unreachable: {e}")
