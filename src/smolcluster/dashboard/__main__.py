"""
smolcluster Dashboard entry point.

Usage:
    python -m smolcluster.dashboard                    
    python -m smolcluster.dashboard --port 9090
    python -m smolcluster.dashboard --host 0.0.0.0
"""

import argparse
import logging
import socket
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="smolcluster visual dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9090, help="Port (default: 9090)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    
    hostname = socket.gethostname().removesuffix(".local")
    print(f"\n  smolcluster dashboard  →  http://{hostname}.local:{args.port}\n")

    uvicorn.run(
        "smolcluster.dashboard.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
