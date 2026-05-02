import asyncio
import io
import logging
import os
import shlex
from pathlib import Path
from typing import Dict

from ruamel.yaml import YAML as _YAML

logger = logging.getLogger(__name__)


class _InferLaunchMixin:
    async def launch_inference_script(
        self,
        algorithm: str,
        server_hostname: str,
        nodes_info: Dict[str, dict],  # hostname → {ssh_alias, user, rank, ip}
        config_path: str,
        script_path: str,
    ) -> None:
        """Rewrite cluster_config_inference.yaml, then run launch_inference.sh."""
        _yaml = _YAML()
        _yaml.preserve_quotes = True
        _yaml.default_flow_style = False

        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        p = Path(config_path)
        config: dict = _yaml.load(p.read_text()) or {} if p.exists() else {}

        server_info  = nodes_info.get(server_hostname, {})
        server_alias = server_info.get("ssh_alias") or server_hostname
        server_ip    = server_info.get("ip", "")

        host_ip  = dict(config.get("host_ip", {}))
        port_raw = config.get("port", {})
        port_cfg = dict(port_raw) if isinstance(port_raw, dict) else {"default": port_raw or 65432}
        default_port = int(port_cfg.get("default", 65432))

        if not server_ip:
            server_ip = host_ip.get(server_alias) or host_ip.get(server_hostname, "")
        if server_ip:
            host_ip[server_alias] = server_ip
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
                "hostname": alias, "user": user, "rank": rank,
                "ip": ip or host_ip.get(alias, ""), "port": resolved_port,
            })
            if ip:
                host_ip[alias] = ip
            port_cfg[alias] = resolved_port

        workers_regular_raw.sort(key=lambda w: w["rank"])
        workers_regular = [
            {**w, "rank": idx}
            for idx, w in enumerate(workers_regular_raw, 1)
        ]

        if algorithm == "classicdp":
            server_user = server_info.get("user", "")
            server_port = int(port_cfg.get(server_alias, port_cfg.get(server_hostname, default_port)))
            all_workers = [{
                "hostname": server_alias, "user": server_user, "rank": 0,
                "ip": server_ip or host_ip.get(server_alias, ""), "port": server_port,
            }] + workers_regular
            config["server"]          = server_alias
            config["workers"]         = {"regular": all_workers, "tablets": []}
            config["num_workers"]     = len(all_workers)
            config["total_num_nodes"] = len(all_workers)
        else:
            config["server"]          = server_alias
            config["workers"]         = {"regular": workers_regular, "tablets": []}
            config["num_workers"]     = len(workers_regular)
            config["total_num_nodes"] = len(workers_regular) + 1

        config["host_ip"] = host_ip
        config["port"]    = port_cfg

        _check      = all_workers if algorithm == "classicdp" else workers_regular  # type: ignore[name-defined]
        _missing    = [w["hostname"] for w in _check if not w.get("ip")]
        if not server_ip:
            _missing.insert(0, server_alias)
        if _missing:
            _cfg = Path(config_path).name
            _msg = (
                f"\n[ERROR] Cannot start inference — IP unknown for: {', '.join(_missing)}\n"
                f"\n  Fix: edit  src/smolcluster/configs/inference/{_cfg}\n"
                f"  and add missing entries under  host_ip:\n"
            )
            for _h in _missing:
                _msg += f"    {_h}: \"<LAN IP>\"\n"
            _msg += "\n  Then click Infer again.\n"
            self._log(server_hostname, _msg)
            raise ValueError(_msg.strip())

        buf = io.StringIO()
        _yaml.dump(config, buf)
        p.write_text(buf.getvalue())
        self._log(server_hostname, f"[dashboard] Wrote {config_path}")
        self._log(server_hostname,
                  f"[dashboard] server={server_alias}, "
                  f"workers={[w['hostname'] for w in workers_regular]}, "
                  f"algorithm={algorithm}")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        cmd = ["bash", script_path, "--algorithm", algorithm]
        self._log(server_hostname, f"$ {shlex.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(Path(script_path).parent.parent.parent),
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": algorithm, "role": "inference_launcher", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))
