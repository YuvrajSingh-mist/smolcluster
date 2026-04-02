"""
NodeManager: manages cluster node selection and process lifecycle.

Flow:
    select(hostname, ssh_user)      — mark a node as "ready" (no SSH yet)
    deselect(hostname)              — unmark
    start_training(algorithm, ...)  — launch server locally + SSH workers
    start_inference(server_host)    — launch infer server + SSH workers
    launch_inference_script(...)    — write config YAML + run launch_inference.sh
    stop_training()                 — kill everything
    probe_username(hostname, user)  — SSH whoami
"""

import asyncio
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REMOTE_REPO = "~/smolcluster"
LOG_MAXLINES = 500  # global circular buffer

# algorithm → (config_filename, script_filename, topology_type)
# topology_type:
#   "flat_server"   — dedicated server (.server) + flat workers list (.workers[])
#   "nested_server" — dedicated server (.server) + .workers.{regular,tablets}
#   "allToAll"      — all nodes in .allToAllTopology.workers.regular (no dedicated server)
#   "pipeline"      — all nodes in .pipelineTopology.workers.regular (no dedicated server)
_TRAINING_ALGO_MAP: dict = {
    "syncps":      ("cluster_config_syncps.yaml",      "launch_syncps_train_gpt.sh",       "flat_server"),
    "mp":          ("cluster_config_mp.yaml",           "launch_mp_train_gpt.sh",           "nested_server"),
    "classicdp":   ("cluster_config_classicdp.yaml",   "launch_dp_train_gpt.sh",           "allToAll"),
    "fsdp":        ("cluster_config_fsdp.yaml",         "launch_fsdp_train_gpt.sh",         "allToAll"),
    "ep":          ("cluster_config_ep.yaml",           "launch_ep_train_moe.sh",           "allToAll"),
    "mp_pipeline": ("cluster_config_mp_pipeline.yaml", "launch_mp_pipeline_train_gpt.sh",  "pipeline"),
    "edp":         ("cluster_config_edp.yaml",          "launch_edp_train_gpt.sh",          "flat_server"),
}


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
    if "@" in ssh_user:       # already fully qualified
        return ssh_user
    if "." not in ssh_user:   # bare alias like "mini2"
        return ssh_user
    return f"{ssh_user}@{hostname}.local"


class NodeManager:
    """
    selected:  hostname → {ssh_user, rank}
    processes: hostname → {proc, rank, algorithm, role}
    """

    def __init__(self):
        self.selected:  Dict[str, dict] = {}
        self.processes: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        # Log streaming — global ordered list, trimmed to LOG_MAXLINES
        self._logs: List[dict] = []
        self._log_seq = 0

    # ── Log helpers ────────────────────────────────────────────────────────────

    def _log(self, hostname: str, line: str):
        self._log_seq += 1
        self._logs.append({"seq": self._log_seq, "hostname": hostname,
                            "line": line, "ts": time.time()})
        if len(self._logs) > LOG_MAXLINES:
            self._logs = self._logs[-LOG_MAXLINES:]

    def logs_since(self, seq: int = 0) -> List[dict]:
        return [l for l in self._logs if l["seq"] > seq]

    # ── Selection ──────────────────────────────────────────────────────────────

    async def select(self, hostname: str, ssh_user: str = "",
                     rank: Optional[int] = None) -> int:
        async with self._lock:
            if rank is None:
                taken = {v["rank"] for v in self.selected.values()}
                rank = next(r for r in range(1, 100) if r not in taken)
            self.selected[hostname] = {"ssh_user": ssh_user, "rank": rank}
        logger.info(f"[node_manager] Selected {hostname} as rank {rank}")
        return rank

    async def deselect(self, hostname: str) -> None:
        async with self._lock:
            self.selected.pop(hostname, None)
        logger.info(f"[node_manager] Deselected {hostname}")

    # ── Training lifecycle ─────────────────────────────────────────────────────

    async def start_training(self, algorithm: str, server_hostname: str) -> None:
        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        # ── Server (local) ────────────────────────────────────────────────────
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

        # ── Workers (SSH) ─────────────────────────────────────────────────────
        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target = _build_ssh_target(ssh_user, hostname)
            remote = (
                f"cd {REMOTE_REPO} && PYTHONUNBUFFERED=1 "
                f"uv run python -m smolcluster.train "
                f"worker {rank} {hostname} --algorithm {algorithm}"
            )
            cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                   target, remote]
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

    # ── Inference lifecycle ────────────────────────────────────────────────────

    async def start_inference(self, server_hostname: str) -> None:
        """
        Launch infer server locally + SSH each selected worker to run infer worker.
        """
        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        num_workers = len(self.selected)

        # ── Infer server (local) ──────────────────────────────────────────────
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

        # ── Workers (SSH) ─────────────────────────────────────────────────────
        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target = _build_ssh_target(ssh_user, hostname)
            remote = (
                f"cd {REMOTE_REPO} && PYTHONUNBUFFERED=1 "
                f"uv run python -m smolcluster.applications.infer "
                f"worker {rank} {hostname} {server_hostname}"
            )
            cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                   target, remote]
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

    # ── Inference script launcher ──────────────────────────────────────────────

    async def launch_inference_script(
        self,
        algorithm: str,
        server_hostname: str,
        nodes_info: Dict[str, dict],  # hostname → {ssh_alias, user, rank, ip}
        config_path: str,
        script_path: str,
    ) -> None:
        """
        1. Rewrite cluster_config_inference.yaml with current selected nodes.
        2. Run scripts/inference/launch_inference.sh --algorithm <algo>.
        3. Stream output to the log buffer.
        """
        from ruamel.yaml import YAML as _YAML
        _yaml = _YAML()
        _yaml.preserve_quotes = True
        _yaml.default_flow_style = False

        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        # ── Build the updated config ──────────────────────────────────────────
        p = Path(config_path)
        config = {}
        if p.exists():
            import io as _io
            config = _yaml.load(p.read_text()) or {}

        server_info = nodes_info.get(server_hostname, {})
        server_alias = server_info.get("ssh_alias") or server_hostname
        server_ip    = server_info.get("ip", "")

        # host_ip: start from existing, then upsert discovered IPs
        host_ip = dict(config.get("host_ip", {}))
        if server_ip:
            host_ip[server_alias] = server_ip

        workers_regular_raw = []
        for hostname, info in nodes_info.items():
            if hostname == server_hostname:
                continue
            alias = info.get("ssh_alias") or hostname
            ip    = info.get("ip", "")
            user  = info.get("user", "")
            rank  = info.get("rank", 1)
            workers_regular_raw.append({"hostname": alias, "user": user, "rank": rank})
            if ip:
                host_ip[alias] = ip

        workers_regular_raw.sort(key=lambda w: w["rank"])
        workers_regular = [
            {**worker, "rank": index}
            for index, worker in enumerate(workers_regular_raw, 1)
        ]

        if algorithm == "classicdp":
            # ClassicDP: server is also rank-0 worker; no separate server role
            server_user = server_info.get("user", "")
            all_workers = [{"hostname": server_alias, "user": server_user, "rank": 0}] + workers_regular
            config["server"]          = server_alias
            config["workers"]         = {"regular": all_workers, "tablets": []}
            config["num_workers"]     = len(all_workers)
            config["total_num_nodes"] = len(all_workers)
        else:
            config["server"]          = server_alias
            config["workers"]         = {"regular": workers_regular, "tablets": []}
            config["num_workers"]     = len(workers_regular)
            config["total_num_nodes"] = len(workers_regular) + 1  # +1 server

        config["host_ip"] = host_ip

        _buf = _io.StringIO()
        _yaml.dump(config, _buf)
        p.write_text(_buf.getvalue())
        self._log(server_hostname, f"[dashboard] Wrote {config_path}")
        self._log(server_hostname,
                  f"[dashboard] server={server_alias}, "
                  f"workers={[w['hostname'] for w in workers_regular]}, "
                  f"algorithm={algorithm}")

        # ── Run the launch script ─────────────────────────────────────────────
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        cmd = ["bash", script_path, "--algorithm", algorithm]
        self._log(server_hostname, f"$ {shlex.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(Path(script_path).parent.parent.parent),  # project root
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": algorithm, "role": "inference_launcher", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))

    # ── Training script launcher ───────────────────────────────────────────────

    async def launch_training_script(
        self,
        algorithm: str,
        server_hostname: str,
        nodes_info: Dict[str, dict],  # hostname → {ssh_alias, user, rank, ip}
        configs_dir: str,             # path to src/smolcluster/configs/
        scripts_dir: str,             # path to scripts/training/
    ) -> None:
        """
        1. Rewrite the algorithm-specific training config YAML with current nodes.
        2. Run the matching training shell script.
        3. Stream output to the log buffer.
        """
        import io as _io
        from ruamel.yaml import YAML as _YAML
        _yaml = _YAML()
        _yaml.preserve_quotes = True
        _yaml.default_flow_style = False

        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        if algorithm not in _TRAINING_ALGO_MAP:
            raise ValueError(f"Unknown training algorithm: {algorithm}")

        config_file, script_file, topology = _TRAINING_ALGO_MAP[algorithm]
        config_path = Path(configs_dir) / config_file
        script_path = Path(scripts_dir) / script_file

        config: dict = {}
        if config_path.exists():
            config = _yaml.load(config_path.read_text()) or {}

        server_info  = nodes_info.get(server_hostname, {})
        server_alias = server_info.get("ssh_alias") or server_hostname

        if topology in ("flat_server", "nested_server"):
            # Update top-level host_ip
            host_ip = dict(config.get("host_ip", {}))
            for hostname, info in nodes_info.items():
                alias = info.get("ssh_alias") or hostname
                ip    = info.get("ip", "")
                if ip:
                    host_ip[alias] = ip
            config["host_ip"] = host_ip

            config["server"] = server_alias

            workers_raw = [
                {
                    "hostname": (info.get("ssh_alias") or hostname),
                    "rank":     info["rank"],
                    "ip":       info.get("ip", ""),
                }
                for hostname, info in nodes_info.items()
                if hostname != server_hostname
            ]
            workers_raw.sort(key=lambda w: w["rank"])
            # Re-number workers from 1 (server is implicit rank 0)
            workers = [{**w, "rank": i} for i, w in enumerate(workers_raw, 1)]

            if topology == "flat_server":
                config["workers"]     = workers
            else:  # nested_server
                config["workers"]     = {"regular": workers, "tablets": []}
            config["num_workers"] = len(workers)

        elif topology in ("allToAll", "pipeline"):
            # All nodes are workers — sort by selection rank, assign 0-indexed ranks
            # server_hostname gets rank 0
            nodes_sorted = sorted(
                nodes_info.items(),
                key=lambda x: (0 if x[0] == server_hostname else x[1]["rank"]),
            )
            all_workers = []
            for i, (hostname, info) in enumerate(nodes_sorted):
                alias = info.get("ssh_alias") or hostname
                ip    = info.get("ip", "")
                entry: dict = {"hostname": alias, "rank": i, "ip": ip}
                if topology == "allToAll":
                    entry["port"] = 65432 + i
                all_workers.append(entry)

            topo_key = "allToAllTopology" if topology == "allToAll" else "pipelineTopology"
            config[topo_key]      = {"workers": {"regular": all_workers, "tablets": []}}
            config["num_workers"] = len(all_workers)
            config["num_nodes"]   = len(all_workers)

        _buf = _io.StringIO()
        _yaml.dump(config, _buf)
        config_path.write_text(_buf.getvalue())
        self._log(server_hostname, f"[dashboard] Wrote {config_path}")
        self._log(server_hostname,
                  f"[dashboard] algorithm={algorithm}, topology={topology}, "
                  f"nodes={list(nodes_info.keys())}")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        cmd = ["bash", str(script_path)]
        self._log(server_hostname, f"$ {shlex.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(Path(scripts_dir).parent.parent),  # project root
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": algorithm, "role": "training_launcher", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))

    async def stop_training(self) -> None:
        async with self._lock:
            procs = dict(self.processes)
            selected = dict(self.selected)
            self.processes.clear()

        for hostname, info in procs.items():
            proc = info["proc"]
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
            logger.info(f"[node_manager] Stopped {hostname}")

        # Launch scripts often spawn long-lived remote tmux sessions.
        # Stop should clean those up as well.
        await self._cleanup_tmux_sessions(selected)

    async def _cleanup_tmux_sessions(self, selected: Dict[str, dict]) -> None:
        tmux_pattern = (
            "^(server.*|worker.*|classicdp_worker.*|fsdp_worker.*|ep_worker.*|"
            "mp_pipeline_worker.*|syncps_inf_.*|classicdp_inf_.*|mp_inference_.*|mp_tablet_proxy.*|"
            "syncps_api|syncps_frontend|classicdp_api|classicdp_frontend|mp_api|mp_frontend)$"
        )

        local_cleanup = (
            "if command -v tmux >/dev/null 2>&1; then "
            "tmux ls 2>/dev/null | cut -d: -f1 | "
            f"grep -E '{tmux_pattern}' | "
            "xargs -r -I{} tmux kill-session -t {} 2>/dev/null || true; "
            "fi"
        )
        await asyncio.to_thread(
            subprocess.run,
            ["bash", "-lc", local_cleanup],
            capture_output=True,
            text=True,
        )

        seen_targets = set()
        for hostname, info in selected.items():
            target = _build_ssh_target(info.get("ssh_user", ""), hostname)
            if target in seen_targets:
                continue
            seen_targets.add(target)
            remote_cleanup = (
                "if command -v tmux >/dev/null 2>&1; then "
                "tmux ls 2>/dev/null | cut -d: -f1 | "
                f"grep -E '{tmux_pattern}' | "
                "xargs -r -I{} tmux kill-session -t {} 2>/dev/null || true; "
                "fi"
            )
            cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                target, remote_cleanup,
            ]
            try:
                await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
            except Exception as e:
                logger.debug(f"[node_manager] tmux cleanup on {target}: {e}")

    # ── Username probe ─────────────────────────────────────────────────────────

    @staticmethod
    async def probe_username(hostname: str, ssh_user: str = "") -> Optional[str]:
        target = _build_ssh_target(ssh_user, hostname)
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
               "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               target, "whoami"]
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True),
                timeout=8.0,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"[node_manager] probe {hostname}: {e}")
        return None

    # ── Snapshots ──────────────────────────────────────────────────────────────

    def snapshot_selected(self) -> Dict[str, dict]:
        return {h: {"rank": v["rank"], "ssh_user": v["ssh_user"]}
                for h, v in self.selected.items()}

    def snapshot_processes(self) -> Dict[str, dict]:
        def _status(v):
            rc = v["proc"].returncode
            if rc is None:
                return "running"
            if v["role"].endswith("_launcher") and rc == 0:
                return "launched"   # script done, remote tmux sessions are live
            return f"exited:{rc}"
        return {
            h: {
                "rank": v["rank"], "algorithm": v["algorithm"], "role": v["role"],
                "status": _status(v),
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
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self._log(hostname, line)
        except Exception as e:
            logger.debug(f"[node_manager] stream {hostname}: {e}")
        finally:
            await proc.wait()
            async with self._lock:
                info = self.processes.get(hostname, {})
                # Keep launcher entries alive so topology stays visible after
                # the script exits (inference continues in remote tmux sessions).
                # Only remove on failure or explicit stop_training().
                if not (str(info.get("role", "")).endswith("_launcher") and proc.returncode == 0):
                    self.processes.pop(hostname, None)
            logger.info(f"[node_manager] {hostname} exited (rc={proc.returncode})")
