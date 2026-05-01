"""CLI argument parsing utilities for train.py."""

import argparse
import os
import sys

import grove
from dashboard.__main__ import main as dashboard_main

ALGORITHMS = ["edp", "syncps", "mp", "mp_pipeline", "classicdp", "fsdp", "ep"]
MODES = ["server", "worker", "dashboard", "grove", "discover"]


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed GPT Training")
    parser.add_argument(
        "mode",
        choices=MODES,
        help="Run as server, worker, dashboard, grove, or discover (auto-discovery via grove)",
    )
    parser.add_argument("arg1", help="Hostname (server mode) or rank (worker mode)")
    parser.add_argument("arg2", nargs="?", help="Hostname (worker mode only)")
    parser.add_argument(
        "-a", "--algorithm",
        choices=ALGORITHMS,
        default="syncps",
        help="Training algorithm to use (default: syncps)",
    )
    parser.add_argument(
        "-r", "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser


def build_discover_parser(default_algorithm: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-a", "--algorithm", choices=ALGORITHMS, default=default_algorithm)
    parser.add_argument("-r", "--resume-checkpoint", default=None)
    return parser


def grove_world_size() -> int:
    return grove.world_size if grove.world_size > 1 else int(os.environ.get("SMOLCLUSTER_WORLD_SIZE", "2"))


def run_dashboard() -> None:
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    dashboard_main()


def should_autodiscover(argv: list[str]) -> bool:
    return grove.world_size > 1 and (len(argv) < 2 or argv[1] not in MODES)


def parse_server_worker_mode(parser: argparse.ArgumentParser) -> tuple[str, str, str, int | None, str | None]:
    if not any(flag in sys.argv for flag in ["--algorithm", "-a", "--resume-checkpoint", "-r"]):
        return _parse_legacy_mode(parser)

    args = parser.parse_args()
    if args.mode == "server":
        return args.mode, args.arg1, args.algorithm, None, args.resume_checkpoint

    if args.arg2 is None:
        print("Error: Worker mode requires both rank and hostname")
        parser.print_help()
        sys.exit(1)

    return args.mode, args.arg2, args.algorithm, int(args.arg1), args.resume_checkpoint


def _parse_legacy_mode(parser: argparse.ArgumentParser) -> tuple[str, str, str, int | None, str | None]:
    mode = sys.argv[1]
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    if mode == "server":
        return mode, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "syncps", None, None

    if len(sys.argv) < 4:
        parser.print_help()
        sys.exit(1)

    return mode, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "syncps", int(sys.argv[2]), None
