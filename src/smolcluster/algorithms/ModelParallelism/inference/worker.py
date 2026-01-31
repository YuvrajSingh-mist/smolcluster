import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
import sys
import yaml

from transformers import AutoConfig, GPT2LMHeadModel

from smolcluster.utils.common_utils import (
    receive_message,
    send_message
)
from smolcluster.utils.device import get_device
from smolcluster.utils.layers import (
    get_hfmodel_per_node,
    load_weights_per_node
)
from smolcluster.utils.model_downloader import ensure_model_weights


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
    HOSTNAME = input("Enter worker hostname: ")

# Set parameters - worker_rank is 1-indexed like regular MP
worker_rank = int(WORKER_RANK)
local_rank = worker_rank  # Keep 1-indexed for consistency

# Workers connect to the server using the IP specified for this worker's hostname
HOST_IP = cluster_config["host_ip"][HOSTNAME]

# Get port for this device (with fallback to default)
port_config = cluster_config["port"]
if isinstance(port_config, dict):
    PORT = port_config.get(HOSTNAME, port_config.get("default", 65432))
else:
    PORT = port_config  # Backward compatibility if port is still a single value

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(f"[WORKER-{local_rank}]")

logger.info(f"Worker {local_rank} starting. Connecting to server at {HOST_IP}:{PORT}")

# Initialize model
model_name = 'causal_gpt2'  # Set model name
model_config = nn_config[model_name]  # Get nested config
hf_model_name = model_config['hf_model_name']
num_nodes = model_config['num_nodes']
num_layers = model_config['num_layers']

config = AutoConfig.from_pretrained(hf_model_name)
if model_name == 'causal_gpt2':
    model = GPT2LMHeadModel(config)
else:
    raise ValueError(f"Unsupported model: {model_name}")

model = model.to(get_device())
logger.info(f"Model initialized on device: {get_device()}")

# Get weights model name from config
weights_model_name = model_config.get('weights_model_name', 'gpt2')
weights_filename = f"{weights_model_name}.safetensors"
# Go up 6 levels from worker.py to get project root: inference -> ModelParallelism -> algorithms -> smolcluster -> src -> project_root
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
weights_path = project_root / "src" / "data" / weights_filename

# Each worker downloads weights on their own machine before connecting to server
logger.info(f"Checking for model weights ({weights_model_name})...")
weights_path = ensure_model_weights(
    model_identifier=weights_model_name,
    weights_path=weights_path
)
logger.info(f"Model weights ready at: {weights_path}")


# Load model layers for this worker
layer_mapping, out_layers, results = get_hfmodel_per_node(
    model,
    num_nodes=num_nodes,
    local_rank=local_rank,
    model_name=model_name,
    total_layers=num_layers
)


model_layers = load_weights_per_node(
    model_name=model_name,
    weights_path=str(weights_path),
    out_layers=out_layers,
    layer_mapping=layer_mapping,
    local_rank=local_rank,
    num_nodes=num_nodes,
    results=results
)

model_layers = model_layers.to(get_device())
logger.info(f"Loaded {len(model_layers)} layers for worker {local_rank}")

def connect_to_server(
    host: str, port: int, max_retries: int = 60, retry_delay: float = 3.0
) -> socket.socket:
    """Connect to server with retry logic."""
    # Ping to warm up ARP cache (especially important for WiFi networks)
    logger.info(f"Warming up ARP cache by pinging {host}...")
    try:
        subprocess.run(
            ["ping", "-c", "3", "-W", "1000", host], capture_output=True, timeout=10
        )
    except Exception as e:
        logger.warning(f"ARP warmup ping failed: {e}")

    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout for connection
        try:
            sock.connect((host, port))
            sock.settimeout(None)  # Remove timeout after connection
            logger.info(
                f"Connected to server at {host}:{port} on attempt {attempt + 1}"
            )
            return sock
        except (OSError, ConnectionRefusedError, socket.timeout) as e:
            sock.close()  # Close the failed socket
            # Re-ping every 5 attempts to keep ARP fresh
            if attempt > 0 and attempt % 5 == 0:
                logger.info(f"Re-pinging {host} to refresh ARP cache...")
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
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to server after {max_retries} attempts"
                )
                raise
    # This should never be reached, but just in case
    raise RuntimeError("Failed to connect to server")


def main():
    
    # Connect to server with retry logic
    sock = connect_to_server(HOST_IP, PORT)

    # Register with the server
    # Register with the server
    logger.info(f"Registering as worker {local_rank} with server...")
    send_message(sock, ("register", local_rank))

    while True:
        recv_command = receive_message(sock)

        if recv_command == "start_inference":
            logger.info("Received start_inference command from server.")
            break

    logger.info("Waiting for generation requests...")
    
    while True:
        
        message = receive_message(sock)
        command, payload = message
        
        out = None
        
        if command == 'generate_activations':
        
            logger.info(f"Received command to generate text for rank {local_rank}.")
            
           
            out = payload['activations'].to(get_device())
            for layer in model_layers: 
                output = layer(out)
                out = output[0] if isinstance(output, tuple) else output
    
            logger.info(f"Finsihed generating activations for local_rank {local_rank} on device {out.device}")
        
            logger.info(f"Sending activations from rank {local_rank} to rank {local_rank + 1} on device {out.device}")
            
            send_message(sock, ('forward_activations', {"from_rank": local_rank, "to_rank": local_rank + 1, "activations": out.cpu()}))
            
            del out
           
            
        elif command == 'down':
            logger.info("Received exit command from server. Shutting down.")
            break
        
        
    sock.close()
    logger.info(f"Worker rank {local_rank} inferencing completed and connection closed.")

if __name__ == "__main__":
    main()
