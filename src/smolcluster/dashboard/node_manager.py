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
import io
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from ruamel.yaml import YAML as _YAML

logger = logging.getLogger(__name__)

REMOTE_REPO = "~/smolcluster"

# Strip C0 control chars except ESC so ANSI colors can be rendered in the UI.
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1A\x1C-\x1F\x7F]")


def _sanitize_log_line(line: str) -> str:
    return _CTRL_RE.sub("", line)

# algorithm → (config_filename, script_filename, topology_type)
# topology_type:
#   "flat_server"   — dedicated server (.server) + flat workers list (.workers[])
#   "nested_server" — dedicated server (.server) + .workers.{regular,tablets}
#   "allToAll"      — all nodes in .allToAllTopology.workers.regular (no dedicated server)
#   "pipeline"      — all nodes in .pipelineTopology.workers.regular (no dedicated server)
#   "grpo"          — dedicated server + inference-style workers.regular + total_num_nodes
_TRAINING_ALGO_MAP: dict = {
    "syncps":      ("cluster_config_syncps.yaml",      "launch_syncps_train_gpt.sh",       "flat_server"),
    "mp":          ("cluster_config_mp.yaml",           "launch_mp_train_gpt.sh",           "nested_server"),
    "classicdp":   ("cluster_config_classicdp.yaml",   "launch_dp_train_gpt.sh",           "allToAll"),
    "fsdp":        ("cluster_config_fsdp.yaml",         "launch_fsdp_train_gpt.sh",         "allToAll"),
    "ep":          ("cluster_config_ep.yaml",           "launch_ep_train_moe.sh",           "allToAll"),
    "mp_pipeline": ("cluster_config_mp_pipeline.yaml", "launch_mp_pipeline_train_gpt.sh",  "pipeline"),
    "edp":         ("cluster_config_edp.yaml",          "launch_edp_train_gpt.sh",          "flat_server"),
    "grpo":        ("inference/cluster_config_inference.yaml", "../../src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_train.sh", "grpo"),
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
        # Log streaming — unbounded queue, drained by _log_broadcaster into Redis Stream
        self._queue: asyncio.Queue = asyncio.Queue()

    # ── Log helpers ────────────────────────────────────────────────────────────

    def _log(self, hostname: str, line: str):
        self._queue.put_nowait({"hostname": hostname, "line": line,
                                "session": "", "ts": time.time()})

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
            config = _yaml.load(p.read_text()) or {}

        server_info = nodes_info.get(server_hostname, {})
        server_alias = server_info.get("ssh_alias") or server_hostname
        server_ip    = server_info.get("ip", "")

        # host_ip/port: start from existing, then upsert discovered values
        host_ip = dict(config.get("host_ip", {}))
        port_cfg = dict(config.get("port", {})) if isinstance(config.get("port", {}), dict) else {"default": config.get("port", 65432)}
        default_port = int(port_cfg.get("default", 65432))
        if not server_ip:
            # IP not discovered — try to reuse whatever the existing config already knows
            # about this server under either its alias or its raw hostname.
            server_ip = host_ip.get(server_alias) or host_ip.get(server_hostname, "")
        if server_ip:
            host_ip[server_alias] = server_ip
            # Also write under the raw discovery hostname so launch_api.sh can
            # resolve it even when the alias differs (e.g. "macmini1" vs "mini1").
            if server_alias != server_hostname:
                host_ip[server_hostname] = server_ip

        workers_regular_raw = []
        for hostname, info in nodes_info.items():
            if hostname == server_hostname:
                continue
            alias = info.get("ssh_alias") or hostname
            ip    = info.get("ip", "")
            user  = info.get("user", "")
            rank  = info.get("rank", 1)
            resolved_port = int(port_cfg.get(alias, port_cfg.get(hostname, default_port)))
            workers_regular_raw.append({
                "hostname": alias,
                "user": user,
                "rank": rank,
                "ip": ip or host_ip.get(alias, ""),
                "port": resolved_port,
            })
            if ip:
                host_ip[alias] = ip
            port_cfg[alias] = resolved_port

        workers_regular_raw.sort(key=lambda w: w["rank"])
        workers_regular = [
            {**worker, "rank": index}
            for index, worker in enumerate(workers_regular_raw, 1)
        ]

        if algorithm == "classicdp":
            # ClassicDP: server is also rank-0 worker; no separate server role
            server_user = server_info.get("user", "")
            server_port = int(port_cfg.get(server_alias, port_cfg.get(server_hostname, default_port)))
            all_workers = [{
                "hostname": server_alias,
                "user": server_user,
                "rank": 0,
                "ip": server_ip or host_ip.get(server_alias, ""),
                "port": server_port,
            }] + workers_regular
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
        config["port"] = port_cfg

        # ── IP validation: catch missing IPs before writing config ───────────
        _check_workers = all_workers if algorithm == "classicdp" else workers_regular
        _missing_ips = [w["hostname"] for w in _check_workers if not w.get("ip")]
        if not server_ip:
            _missing_ips.insert(0, server_alias)
        if _missing_ips:
            _cfg_name = Path(config_path).name
            _msg = (
                f"\n[ERROR] Cannot start inference — IP address unknown for: {', '.join(_missing_ips)}\n"
                f"\n  Fix: open  src/smolcluster/configs/inference/{_cfg_name}\n"
                f"  and add the missing entries under  host_ip:\n"
            )
            for _h in _missing_ips:
                _msg += f"    {_h}: \"<LAN IP of that machine>\"\n"
            _msg += "\n  Then click Infer again.\n"
            self._log(server_hostname, _msg)
            raise ValueError(_msg.strip())

        _buf = io.StringIO()
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

        if topology in ("flat_server", "nested_server", "grpo"):
            # Update top-level host_ip
            host_ip = dict(config.get("host_ip", {}))
            for hostname, info in nodes_info.items():
                alias = info.get("ssh_alias") or hostname
                ip    = info.get("ip", "")
                if ip:
                    host_ip[alias] = ip
            config["host_ip"] = host_ip

            config["server"] = server_alias

            workers_raw = []
            for hostname, info in nodes_info.items():
                if hostname == server_hostname:
                    continue
                alias = info.get("ssh_alias") or hostname
                # Fall back to existing host_ip in config when discovery returns no IP
                ip = info.get("ip", "") or host_ip.get(alias, "") or host_ip.get(hostname, "")
                workers_raw.append({"hostname": alias, "rank": info["rank"], "ip": ip})
            workers_raw.sort(key=lambda w: w["rank"])
            # Re-number workers from 1 (server is implicit rank 0)
            workers = [{**w, "rank": i} for i, w in enumerate(workers_raw, 1)]

            if topology == "flat_server":
                config["workers"]     = workers
            elif topology == "grpo":
                config["workers"]         = {"regular": workers, "tablets": []}
                config["total_num_nodes"] = len(workers) + 1  # +1 server
            else:  # nested_server
                config["workers"]     = {"regular": workers, "tablets": []}
            config["num_workers"] = len(workers)

        elif topology in ("allToAll", "pipeline"):
            # All nodes are workers — sort by selection rank, assign 0-indexed ranks
            # server_hostname gets rank 0
            host_ip_existing = dict(config.get("host_ip", {}))
            nodes_sorted = sorted(
                nodes_info.items(),
                key=lambda x: (0 if x[0] == server_hostname else x[1]["rank"]),
            )
            all_workers = []
            for i, (hostname, info) in enumerate(nodes_sorted):
                alias = info.get("ssh_alias") or hostname
                # Fall back to existing host_ip in config when discovery returns no IP
                ip = info.get("ip", "") or host_ip_existing.get(alias, "") or host_ip_existing.get(hostname, "")
                entry: dict = {"hostname": alias, "rank": i, "ip": ip}
                if topology == "allToAll":
                    entry["port"] = 65432 + i
                all_workers.append(entry)

            topo_key = "allToAllTopology" if topology == "allToAll" else "pipelineTopology"
            config[topo_key]      = {"workers": {"regular": all_workers, "tablets": []}}
            config["num_workers"] = len(all_workers)
            config["num_nodes"]   = len(all_workers)

        # ── IP validation: catch missing IPs before writing config ───────────
        _missing_ips: list[str] = []
        if topology in ("flat_server", "nested_server", "grpo"):
            for w in workers:
                if not w.get("ip"):
                    _missing_ips.append(w["hostname"])
        elif topology in ("allToAll", "pipeline"):
            for w in all_workers:
                if not w.get("ip"):
                    _missing_ips.append(w["hostname"])
        if _missing_ips:
            _cfg_rel = config_path.name
            _msg = (
                f"\n[ERROR] Cannot start training — IP address unknown for: {', '.join(_missing_ips)}\n"
                f"\n  Fix: open  src/smolcluster/configs/{_cfg_rel}\n"
                f"  and add the missing entries under  host_ip:\n"
            )
            for _h in _missing_ips:
                _msg += f"    {_h}: \"<LAN IP of that machine>\"\n"
            _msg += "\n  Then click Train again.\n"
            self._log(server_hostname, _msg)
            raise ValueError(_msg.strip())

        _buf = io.StringIO()
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

        # Kill any tmux sessions spawned by the launch scripts on all nodes.
        await self._cleanup_tmux_sessions(label, selected)

    async def run_cleanup_script(self, script_path: str, log_label: str) -> None:
        """
        Run an external cleanup script (e.g. launch_inference.sh --cleanup)
        and stream every line to the log buffer so the user can see progress.
        """
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
                cwd=str(Path(script_path).parent.parent.parent),  # project root
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
        # Matches every tmux session name that smolcluster scripts create.
        tmux_pattern = (
            "^(server|worker[0-9]*"
            "|classicdp_worker[0-9]*|fsdp_worker[0-9]*|ep_worker[0-9]*|mp_pipeline_worker[0-9]*"
            "|syncps_inf_.*|classicdp_inf_.*|mp_inference_.*|mp_tablet_proxy[0-9]*"
            "|grpo_train|vllm_worker"
            "|syncps_api|syncps_frontend|classicdp_api|classicdp_frontend|mp_api|mp_frontend)$"
        )
        # Only kill matching tmux sessions. Do not use a broad process-name kill,
        # because the dashboard itself runs under the smolcluster module path.
        _kill_cmd = (
            "tmux ls 2>/dev/null | cut -d: -f1 | "
            f"grep -E '{tmux_pattern}' | "
            "while IFS= read -r _s; do tmux kill-session -t \"$_s\" 2>/dev/null; echo \"killed:$_s\"; done"
        )

        # ── Local cleanup ─────────────────────────────────────────────────────
        self._log(log_label, "[stop] Killing local tmux sessions…")
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-lc", _kill_cmd],
            capture_output=True, text=True,
        )
        killed = [l.replace("killed:", "").strip()
                  for l in result.stdout.splitlines() if l.startswith("killed:")]
        if killed:
            for s in killed:
                self._log(log_label, f"[stop]   killed local: {s}")
        else:
            self._log(log_label, "[stop]   no matching local tmux sessions found")

        # ── Remote cleanup ────────────────────────────────────────────────────
        seen_targets: set = set()
        for hostname, info in selected.items():
            ssh_user = info.get("ssh_user", "")
            target = _build_ssh_target(ssh_user, hostname)
            if target in seen_targets:
                continue
            seen_targets.add(target)

            self._log(log_label, f"[stop] Remote cleanup on {target}…")
            remote_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                target, _kill_cmd,
            ]
            try:
                lr = await asyncio.to_thread(
                    subprocess.run, remote_cmd, capture_output=True, text=True
                )
                r_killed = [l.replace("killed:", "").strip()
                            for l in lr.stdout.splitlines() if l.startswith("killed:")]
                if r_killed:
                    for s in r_killed:
                        self._log(log_label, f"[stop]   killed {target}: {s}")
                else:
                    self._log(log_label, f"[stop]   no matching sessions on {target}")
            except Exception as e:
                self._log(log_label, f"[stop]   {target} unreachable: {e}")

    # ── Username probe ─────────────────────────────────────────────────────────

    @staticmethod
    async def probe_username(hostname: str, ssh_user: str = "") -> Optional[str]:
        info = await NodeManager.probe_metadata(hostname, ssh_user)
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
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
               "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               target, remote]
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
                        "username": lines[0],
                        "os": lines[1],
                        "os_version": os_version,
                        "machine": lines[3],
                    }
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
                line = _sanitize_log_line(raw.decode(errors="replace")).rstrip()
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
