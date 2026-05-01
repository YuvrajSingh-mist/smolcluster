"""Grove CLI."""

import argparse
import atexit
import os
import platform
import signal
import subprocess
import sys


_ADJECTIVES = ["swift", "calm", "bold", "warm", "bright", "keen", "vast", "pure", "fair", "deep"]
_NOUNS = ["oak", "pine", "elm", "ash", "bay", "fern", "moss", "reed", "sage", "vale"]


def _generate_name() -> str:
    import random
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


def _get_system_info() -> dict:
    info = {"hostname": platform.node()}
    if platform.system() == "Darwin":
        info["os"] = f"macOS {platform.mac_ver()[0]}"
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, check=True)
            info["chip"] = r.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            info["chip"] = platform.processor() or "unknown"
        try:
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, check=True)
            info["memory"] = f"{int(r.stdout.strip()) / (1024**3):.0f} GB"
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
    else:
        info["os"] = platform.platform()
    info["python"] = platform.python_version()
    return info


def _load_main(script: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("worker_module", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "main", None) or getattr(module, "worker", None)
    if fn is None:
        print(f"Error: {script} must define a main() function")
        sys.exit(1)
    return fn


def _register_cleanup():
    import grove
    def _cleanup():
        if grove._comm and hasattr(grove._comm, '_group'):
            transport = grove._comm._group._transport
            if hasattr(transport, '_process'):
                try:
                    transport._process.terminate()
                    transport._process.wait(timeout=2)
                except Exception:
                    pass
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def _run_with_dashboard(worker_fn, stats_source, name, uid, rank=None):
    import logging
    from .dashboard import Dashboard, WorkerDashboard
    logging.disable(logging.NOTSET)
    if rank is not None:
        WorkerDashboard(stats_source, name, uid, rank).run_with_training(worker_fn)
    else:
        Dashboard(stats_source, name, uid).run_with_training(worker_fn)


def cmd_run(args):
    from .dashboard import _disable_tqdm_locks
    _disable_tqdm_locks()

    worker_fn = _load_main(args.script)

    from rich.console import Console
    Console(stderr=True).print(f"  [bold]grove[/]  {os.path.basename(args.script)}  [dim]single device[/]")

    import logging
    import threading
    logging.disable(logging.CRITICAL)

    class _Local:
        def __init__(self):
            self._lock = threading.Lock()
            self._step_counts = {}
            self._loss = {}
            self._sync_ms = {}
            self._hostnames = {0: platform.node()}
        def _build_stats(self):
            with self._lock:
                return {
                    "epoch": 0, "live_ranks": [0], "dead_ranks": [],
                    "steps": dict(self._step_counts), "loss": dict(self._loss),
                    "sync_ms": dict(self._sync_ms), "hostnames": dict(self._hostnames),
                }

    coord = _Local()
    import grove
    grove._coordinator = coord

    if args.logs:
        logging.disable(logging.NOTSET)
        worker_fn()
    else:
        _run_with_dashboard(worker_fn, coord, os.path.basename(args.script), "local", )


def cmd_start(args, passthrough=None):
    import logging
    if not args.logs:
        from .dashboard import _disable_tqdm_locks
        _disable_tqdm_locks()
    from .cluster import generate_uid

    name = args.name or _generate_name()
    uid = generate_uid()
    worker_fn = _load_main(args.script)

    from rich.console import Console
    console = Console(stderr=True)
    console.print(f"  [bold]grove[/]  {name}  [dim cyan]{uid}[/]  waiting for {args.n - 1} node{'s' if args.n > 2 else ''}...")

    logging.disable(logging.CRITICAL)
    _register_cleanup()

    os.environ["GROVE_SCRIPT"] = os.path.abspath(args.script)
    os.environ["GROVE_IS_COORDINATOR"] = "1"
    os.environ["GROVE_P2P_UID"] = uid

    transport = os.environ.get("GROVE_TRANSPORT", "p2p")
    zc = None
    if transport == "tcp":
        from .cluster import advertise_cluster
        zc, _ = advertise_cluster(
            name=name, uid=uid, expected=args.n, mode="open", script=args.script,
        )

    import grove
    grove.init(cluster=name, world_size=args.n, transport=transport)

    if grove._comm and hasattr(grove._comm, '_group'):
        t = grove._comm._group._transport
        if hasattr(t, '_upgraded'):
            if t._upgraded:
                console.print(f"  [green]wifi[/] direct TCP data plane")
            else:
                console.print(f"  [dim]awdl[/] peer-to-peer data plane")

    # Apply passthrough args so the script's main() sees them in sys.argv.
    # On workers, _distribute_script will transmit these and cmd_join restores them.
    if passthrough:
        sys.argv = [os.path.abspath(args.script)] + passthrough

    if args.logs:
        logging.disable(logging.NOTSET)
        worker_fn()
    elif grove._coordinator:
        _run_with_dashboard(worker_fn, grove._coordinator, name, uid, )
    else:
        logging.disable(logging.NOTSET)
        worker_fn()

    if zc:
        zc.unregister_all_services()
        zc.close()


def cmd_join(args, passthrough=None):
    import logging
    import time
    if not args.logs:
        from .dashboard import _disable_tqdm_locks
        _disable_tqdm_locks()

    from rich.console import Console
    console = Console()
    _register_cleanup()
    transport = os.environ.get("GROVE_TRANSPORT", "p2p")

    console.print("  [bold]grove[/]  [dim]scanning for clusters...[/]\n")

    if transport == "p2p":
        cluster = _discover_p2p(args, console)
    else:
        cluster = _discover_tcp(args, console)

    if cluster is None:
        sys.exit(0)

    cluster_name = cluster["name"]
    cluster_uid = cluster.get("uid", "")
    expected = int(cluster.get("expected", 2))

    console.print(f"  connecting to [bold]{cluster_name}[/] ({cluster_uid})...")
    logging.disable(logging.CRITICAL)

    import grove
    grove.init(cluster=cluster_name, world_size=expected, transport=transport)
    console.print(f"  [green]connected as rank {grove.rank}[/]")

    if grove._comm and hasattr(grove._comm, '_group'):
        t = grove._comm._group._transport
        if hasattr(t, '_upgraded'):
            if t._upgraded:
                console.print(f"  [green]wifi[/] direct TCP data plane")
            else:
                console.print(f"  [dim]awdl[/] peer-to-peer data plane")
    console.print()

    if grove._received_script is None:
        console.print("  [red]script not received from coordinator[/]")
        sys.exit(1)

    script_name, script_content = grove._received_script
    import tempfile
    script_path = os.path.join(tempfile.mkdtemp(prefix="grove_"), script_name)
    with open(script_path, "w") as f:
        f.write(script_content)
    console.print(f"  [dim]received {script_name} ({len(script_content)} bytes)[/]")

    worker_fn = _load_main(script_path)

    # Restore the argv the coordinator was called with so the script's main()
    # sees the same arguments on every node.
    if grove._received_argv:
        sys.argv = [script_path] + grove._received_argv

    if args.logs or not grove._worker_client:
        logging.disable(logging.NOTSET)
        worker_fn()
    else:
        _run_with_dashboard(
            worker_fn, grove._worker_client, cluster_name, cluster_uid,
            rank=grove.rank,
        )


def _discover_p2p(args, console):
    import time
    from .transport.p2p import P2PLiveBrowser, discover_p2p_clusters

    if args.cluster_name:
        clusters = discover_p2p_clusters(timeout=10.0)
        cluster = next((c for c in clusters if c.get("name") == args.cluster_name), None)
        if cluster is None:
            console.print(f"  [red]cluster '{args.cluster_name}' not found[/]")
            sys.exit(1)
        return cluster

    from .tui import JoinApp
    browser = P2PLiveBrowser()
    time.sleep(3.0)
    app = JoinApp(browser)
    app.run()
    browser.close()
    return app.selected_cluster


def _discover_tcp(args, console):
    import time
    from .cluster import browse_clusters_live, browse_clusters

    if args.cluster_name:
        clusters = browse_clusters(timeout=5.0)
        cluster = next((c for c in clusters if c.get("name") == args.cluster_name), None)
        if cluster is None:
            console.print(f"  [red]cluster '{args.cluster_name}' not found[/]")
            sys.exit(1)
        return cluster

    from .tui import JoinApp
    browser = browse_clusters_live()
    time.sleep(2.0)
    app = JoinApp(browser)
    app.run()
    browser.close()
    return app.selected_cluster



def cmd_status(_args):
    from rich.console import Console
    from rich.table import Table
    import grove

    console = Console()
    info = _get_system_info()

    console.print(f"\n  [bold]grove[/] v{grove.__version__}")
    console.print()
    for k, v in info.items():
        console.print(f"  [dim]{k:>10}[/]  {v}")
    try:
        import mlx.core as mx
        console.print(f"  [dim]{'mlx':>10}[/]  {mx.__version__}")
    except ImportError:
        console.print(f"  [dim]{'mlx':>10}[/]  [red]not installed[/]")
    console.print()

    try:
        from .cluster import browse_clusters
        console.print("  [dim]scanning for clusters...[/]", end="")
        clusters = browse_clusters(timeout=3.0)
        console.print("\r" + " " * 40 + "\r", end="")
        if clusters:
            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("Name", style="bold")
            table.add_column("ID", style="cyan dim")
            table.add_column("Nodes")
            table.add_column("Host", style="dim")
            for c in clusters:
                table.add_row(
                    c.get("name", "?"), c.get("uid", "?"),
                    f"{c.get('current', '?')}/{c.get('expected', '?')}",
                    c.get("hostname", "?"),
                )
            console.print("  [bold]clusters on network:[/]")
            console.print(table)
        else:
            console.print("  [dim]no clusters found on network[/]")
    except Exception:
        pass
    console.print()


def main():
    # Split off passthrough args (everything after --) before argparse sees them.
    # This lets `grove start train.py -n 2 -- discover -a classicdp` work cleanly.
    argv = sys.argv[1:]
    try:
        sep = argv.index("--")
        grove_argv = argv[:sep]
        passthrough = argv[sep + 1:]
    except ValueError:
        grove_argv = argv
        passthrough = []

    parser = argparse.ArgumentParser(prog="grove", description="grove — distributed ML for Apple Silicon")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run on a single device")
    run_p.add_argument("script")
    run_p.add_argument("--logs", action="store_true")

    start_p = sub.add_parser("start", help="Start a training cluster")
    start_p.add_argument("script")
    start_p.add_argument("-n", type=int, default=2)
    start_p.add_argument("--name", default=None)
    start_p.add_argument("--logs", action="store_true")

    join_p = sub.add_parser("join", help="Join a training cluster")
    join_p.add_argument("cluster_name", nargs="?", default=None)
    join_p.add_argument("--logs", action="store_true")

    sub.add_parser("status", help="System info and nearby clusters")

    # Parse grove args first, then treat unknown args as script passthrough for
    # run/start so users can pass script flags without requiring an explicit "--".
    args, unknown = parser.parse_known_args(grove_argv)
    if unknown:
        if args.command in {"run", "start"}:
            passthrough = unknown + passthrough
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    match args.command:
        case "run": cmd_run(args)
        case "start": cmd_start(args, passthrough)
        case "join": cmd_join(args, passthrough)
        case "status": cmd_status(args)
        case _: parser.print_help()


if __name__ == "__main__":
    main()
