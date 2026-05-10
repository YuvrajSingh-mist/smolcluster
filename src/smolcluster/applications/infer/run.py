"""
smolcluster Inference Connectivity Test.

Server: listens for worker pings, tracks who connected, writes status to
        /tmp/smolcluster_inference.json so the dashboard can display it.
Worker: connects to server, reports hostname + env info, then exits.

Usage (via node_manager / SSH):
    python -m smolcluster.applications.infer server <server_hostname> <num_workers>
    python -m smolcluster.applications.infer worker <rank> <hostname> <server_hostname>
"""

import json
import platform
import socket
import sys
import time
from pathlib import Path

INFER_PORT = 65433
INFER_FILE = Path("/tmp/smolcluster_inference.json")


def _write(payload: dict):
    INFER_FILE.write_text(json.dumps(payload))


def run_server(server_hostname: str, num_workers: int):
    _write({
        "mode": "inference", "status": "waiting",
        "workers": [], "total": num_workers,
        "message": f"Waiting for {num_workers} worker(s)…",
    })

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", INFER_PORT))
    srv.listen(num_workers)
    srv.settimeout(60.0)

    workers = []
    while len(workers) < num_workers:
        try:
            conn, _addr = srv.accept()
            raw = conn.recv(4096).decode(errors="replace")
            info = json.loads(raw)
            workers.append(info)
            conn.send(json.dumps({"status": "ok"}).encode())
            conn.close()
            _write({
                "mode": "inference", "status": "connecting",
                "workers": workers, "total": num_workers,
                "message": f"{len(workers)} / {num_workers} workers connected",
            })
        except TimeoutError:
            break

    ok = len(workers) == num_workers
    _write({
        "mode": "inference",
        "status": "ready" if ok else "partial",
        "workers": workers, "total": num_workers,
        "message": (
            f"All {len(workers)} workers ready!"
            if ok else
            f"Only {len(workers)} / {num_workers} workers responded"
        ),
    })

    # Stay alive so the dashboard knows inference is still "running"
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass


def run_worker(rank: int, hostname: str, server_hostname: str):
    info = {
        "rank": rank,
        "hostname": hostname,
        "os": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "smolcluster": "unknown",
    }
    try:
        import smolcluster  # noqa: F401
        info["smolcluster"] = "ok"
    except ImportError as e:
        info["smolcluster"] = f"error: {e}"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30.0)
    sock.connect((f"{server_hostname}.local", INFER_PORT))
    sock.send(json.dumps(info).encode())
    sock.recv(4096)
    sock.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m smolcluster.applications.infer [server|worker] …")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "server":
        if len(sys.argv) < 4:
            print("Usage: … infer server <server_hostname> <num_workers>")
            sys.exit(1)
        run_server(sys.argv[2], int(sys.argv[3]))
    elif mode == "worker":
        if len(sys.argv) < 5:
            print("Usage: … infer worker <rank> <hostname> <server_hostname>")
            sys.exit(1)
        run_worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
