import asyncio
import io
import logging
import os
import shlex
from pathlib import Path
from typing import Dict

from ruamel.yaml import YAML as _YAML

from ._constants import _TRAINING_ALGO_MAP

logger = logging.getLogger(__name__)


class _TrainLaunchMixin:
    async def launch_training_script(
        self,
        algorithm: str,
        server_hostname: str,
        nodes_info: Dict[str, dict],  # hostname → {ssh_alias, user, rank, ip}
        configs_dir: str,
        scripts_dir: str,
    ) -> None:
        """Rewrite the algorithm's cluster config YAML and run the training launch script."""
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

        config: dict = _yaml.load(config_path.read_text()) or {} if config_path.exists() else {}
        server_info  = nodes_info.get(server_hostname, {})
        server_alias = server_info.get("ssh_alias") or server_hostname

        if topology in ("flat_server", "nested_server", "grpo"):
            host_ip = dict(config.get("host_ip", {}))
            for hostname, info in nodes_info.items():
                alias = info.get("ssh_alias") or hostname
                ip    = info.get("ip", "")
                if ip:
                    host_ip[alias] = ip
            config["host_ip"] = host_ip
            config["server"]  = server_alias

            workers_raw = []
            for hostname, info in nodes_info.items():
                if hostname == server_hostname:
                    continue
                alias = info.get("ssh_alias") or hostname
                ip    = info.get("ip", "") or host_ip.get(alias, "") or host_ip.get(hostname, "")
                workers_raw.append({"hostname": alias, "rank": info["rank"], "ip": ip})
            workers_raw.sort(key=lambda w: w["rank"])
            workers = [{**w, "rank": i} for i, w in enumerate(workers_raw, 1)]

            if topology == "flat_server":
                config["workers"]     = workers
            elif topology == "grpo":
                config["workers"]         = {"regular": workers, "tablets": []}
                config["total_num_nodes"] = len(workers) + 1
            else:  # nested_server
                config["workers"] = {"regular": workers, "tablets": []}
            config["num_workers"] = len(workers)

        elif topology in ("allToAll", "pipeline"):
            host_ip_existing = dict(config.get("host_ip", {}))
            nodes_sorted = sorted(
                nodes_info.items(),
                key=lambda x: (0 if x[0] == server_hostname else x[1]["rank"]),
            )
            all_workers = []
            for i, (hostname, info) in enumerate(nodes_sorted):
                alias = info.get("ssh_alias") or hostname
                ip    = info.get("ip", "") or host_ip_existing.get(alias, "") or host_ip_existing.get(hostname, "")
                entry: dict = {"hostname": alias, "rank": i, "ip": ip}
                if topology == "allToAll":
                    entry["port"] = 65432 + i
                all_workers.append(entry)
            topo_key = "allToAllTopology" if topology == "allToAll" else "pipelineTopology"
            config[topo_key]      = {"workers": {"regular": all_workers, "tablets": []}}
            config["num_workers"] = len(all_workers)
            config["num_nodes"]   = len(all_workers)

        _missing: list[str] = []
        if topology in ("flat_server", "nested_server", "grpo"):
            _missing = [w["hostname"] for w in workers if not w.get("ip")]  # type: ignore[name-defined]
        elif topology in ("allToAll", "pipeline"):
            _missing = [w["hostname"] for w in all_workers if not w.get("ip")]  # type: ignore[name-defined]
        if _missing:
            _cfg = config_path.name
            _msg = (
                f"\n[ERROR] Cannot start training — IP unknown for: {', '.join(_missing)}\n"
                f"\n  Fix: edit  src/smolcluster/configs/{_cfg}\n"
                f"  and add missing entries under  host_ip:\n"
            )
            for _h in _missing:
                _msg += f"    {_h}: \"<LAN IP>\"\n"
            _msg += "\n  Then click Train again.\n"
            self._log(server_hostname, _msg)
            raise ValueError(_msg.strip())

        buf = io.StringIO()
        _yaml.dump(config, buf)
        config_path.write_text(buf.getvalue())
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
            cwd=str(Path(scripts_dir).parent.parent),
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": algorithm, "role": "training_launcher", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))
