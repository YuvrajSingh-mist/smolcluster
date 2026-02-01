import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from smolcluster.utils.common_utils import (
    receive_message,
    recv_tensor,
    send_message,
    send_tensor,
)

# Load configs
CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs"
with open(CONFIG_DIR / "model_parallelism" / "model_config_inference.yaml") as f:
    nn_config = yaml.safe_load(f)

with open(CONFIG_DIR / "model_parallelism" / "cluster_config_inference.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
NUM_WORKERS = cluster_config["num_workers"]
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1

# Get worker rank and hostname from command-line arguments
if len(sys.argv) > 1:
    WORKER_RANK = sys.argv[1]
else:
    WORKER_RANK = input(f"Enter worker rank (1 to {NUM_WORKERS}): ")

if len(sys.argv) > 2:
    HOSTNAME = sys.argv[2]
else:
    HOSTNAME = input("Enter tablet hostname: ")

# Set parameters - worker_rank is 1-indexed like regular MP
worker_rank = int(WORKER_RANK)
local_rank = worker_rank  # Keep 1-indexed for consistency

# Tablets connect to the server using the server's IP
SERVER_HOSTNAME = cluster_config["server"]
SERVER_IP = cluster_config["host_ip"][SERVER_HOSTNAME]
SERVER_PORT_CONFIG = cluster_config["port"]
if isinstance(SERVER_PORT_CONFIG, dict):
    SERVER_PORT = SERVER_PORT_CONFIG.get(
        SERVER_HOSTNAME, SERVER_PORT_CONFIG.get("default", 65432)
    )
else:
    SERVER_PORT = SERVER_PORT_CONFIG

# Tablet device connection (where actual computation happens)
TABLET_IP = cluster_config["host_ip"][HOSTNAME]
TABLET_PORT_CONFIG = cluster_config["port"]

if isinstance(TABLET_PORT_CONFIG, dict):
    TABLET_PORT = TABLET_PORT_CONFIG.get(
        HOSTNAME, TABLET_PORT_CONFIG.get("default", 8000)
    )
else:
    TABLET_PORT = TABLET_PORT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(f"[TABLET-PROXY-{local_rank}]")

logger.info(f"Tablet proxy {local_rank} for {HOSTNAME} starting")
logger.info(f"  -> Server connection: {SERVER_IP}:{SERVER_PORT}")
logger.info(f"  -> Tablet device connection: {TABLET_IP}:{TABLET_PORT}")

# Initialize model
model_name = "causal_gpt2"  # Set model name
model_config = nn_config[model_name]  # Get nested config

# Note: Tablet proxy doesn't load model - actual tablet device does the computation
logger.info(f"Tablet proxy configured for {model_name} model")


def connect_to_service(
    host: str,
    port: int,
    service_name: str,
    max_retries: int = 60,
    retry_delay: float = 3.0,
) -> socket.socket:
    """Connect to a service (server or tablet device) with retry logic."""
    logger.info(f"Connecting to {service_name} at {host}:{port}...")

    # Ping to warm up network
    try:
        subprocess.run(
            ["ping", "-c", "3", "-W", "1000", host], capture_output=True, timeout=10
        )
    except Exception as e:
        logger.warning(f"Network warmup ping to {service_name} failed: {e}")
    try:
        subprocess.run(
            ["ping", "-c", "3", "-W", "1000", host], capture_output=True, timeout=10
        )
    except Exception as e:
        logger.warning(f"Network warmup ping failed: {e}")

    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout for connection
        try:
            sock.connect((host, port))
            sock.settimeout(None)  # Remove timeout after connection
            logger.info(
                f"Connected to {service_name} at {host}:{port} on attempt {attempt + 1}"
            )
            return sock
        except (OSError, ConnectionRefusedError, socket.timeout) as e:
            sock.close()  # Close the failed socket
            # Re-ping every 5 attempts to keep network fresh
            if attempt > 0 and attempt % 5 == 0:
                logger.info(f"Re-pinging {service_name} to refresh network...")
                try:
                    subprocess.run(
                        ["ping", "-c", "2", "-W", "1000", host],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass
            if attempt < max_retries - 1:
                logger.warning(
                    f"Connection to {service_name} attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to {service_name} after {max_retries} attempts"
                )
                raise
    # This should never be reached, but just in case
    raise RuntimeError(f"Failed to connect to {service_name}")


def main():
    # Connect to server with retry logic
    logger.info(f"Tablet proxy {local_rank} connecting to server...")
    server_sock = connect_to_service(SERVER_IP, SERVER_PORT, "Server")

    # Connect to tablet device
    logger.info(f"Tablet proxy {local_rank} connecting to tablet device {HOSTNAME}...")
    tablet_sock = connect_to_service(TABLET_IP, TABLET_PORT, f"Tablet-{HOSTNAME}")

    # Register with the server
    logger.info(f"Registering as tablet proxy worker {local_rank} with server...")
    send_message(server_sock, ("register", local_rank))

    while True:
        recv_command = receive_message(server_sock)

        if recv_command == "start_inference":
            logger.info("Received start_inference command from server.")
            break

    logger.info("Tablet proxy waiting to forward generation requests...")

    while True:
        message = receive_message(server_sock)
        command, payload = message

        if command == "generate_activations":
            logger.info(
                f"Tablet proxy {local_rank} received activations from server, forwarding to tablet device {HOSTNAME}..."
            )

            # Forward activations to tablet device for computation
            activations = payload["activations"]
            send_tensor(tablet_sock, activations)

            logger.info(
                "Activations sent to tablet device, waiting for processed results..."
            )
            # Receive processed activations back from tablet device
            processed_activations = recv_tensor(tablet_sock)

            logger.info(
                f"Tablet proxy {local_rank} received processed activations from tablet device {HOSTNAME}"
            )

            # Forward results back to server
            logger.info(
                f"Tablet proxy {local_rank} forwarding activations to server (rank {local_rank} -> {local_rank + 1})"
            )

            send_message(
                server_sock,
                (
                    "forward_activations",
                    {
                        "from_rank": local_rank,
                        "to_rank": local_rank + 1,
                        "activations": processed_activations,
                        "tablet_device": HOSTNAME,  # Include tablet device info
                    },
                ),
            )

        elif command == "down":
            logger.info(
                "Received exit command from server. Shutting down tablet proxy."
            )
            # Forward shutdown to tablet device
            send_message(tablet_sock, ("down", None))
            break

    server_sock.close()
    tablet_sock.close()
    logger.info(
        f"Tablet proxy {local_rank} ({HOSTNAME}) completed and connections closed."
    )


if __name__ == "__main__":
    main()
