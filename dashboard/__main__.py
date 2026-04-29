"""
smolcluster Dashboard entry point.

Usage:
    python -m dashboard
    python -m dashboard --port 9090
"""

import argparse
import asyncio
import logging
import socket
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _make_sockets(port: int) -> list:
    """Return pre-bound, pre-listened sockets for IPv4 and IPv6 (dual-stack on macOS)."""
    socks = []
    for family, addr in [(socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")]:
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            s.bind((addr, port))
            s.listen(100)
            s.set_inheritable(True)
            socks.append(s)
        except OSError:
            pass
    return socks


def main():
    parser = argparse.ArgumentParser(description="smolcluster visual dashboard")
    parser.add_argument("--port", type=int, default=9090, help="Port (default: 9090)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    hostname = socket.gethostname().removesuffix(".local")
    print(f"\n  smolcluster dashboard  →  http://{hostname}.local:{args.port}\n")

    socks = _make_sockets(args.port)
    config = uvicorn.Config(
        "dashboard.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=socks))


if __name__ == "__main__":
    main()
