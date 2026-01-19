import gc
import logging
import os
import socket
import sys
import threading
from collections import defaultdict
from pathlib import Path

import torch
import torchinfo
import torchvision
import wandb
import yaml
from torch.utils.data import DataLoader


from transformers import AutoConfig, GPT2LMHeadModel


from smolcluster.utils.common_utils import (
    get_gradients,
    get_weights,
    receive_message,
    send_message,
    set_gradients,
   
)
from smolcluster.utils.layers import (
    get_model_per_node
)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device

# Login to wandb using API key from environment variable
if "WANDB_API_TOKEN" in os.environ:
    wandb.login(key=os.environ["WANDB_API_TOKEN"], relogin=True)
    logger_temp = logging.getLogger("[SERVER-INIT]")
    logger_temp.info("✅ Logged into wandb using WANDB_API_TOKEN")
else:
    logger_temp = logging.getLogger("[SERVER-INIT]")
    logger_temp.warning("⚠️  WANDB_API_TOKEN not set - wandb may prompt for login")

# Get hostname from command-line argument
if len(sys.argv) > 1:
    HOSTNAME = sys.argv[1]
else:
    HOSTNAME = input("Enter server hostname: ")

# Load configs
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "model_parallelism"
with open(CONFIG_DIR / "gpt2_config.yaml") as f:
    nn_config = yaml.safe_load(f)

with open(CONFIG_DIR / "cluster_config_syncps.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Extract values with defaults
HOST_IP = "0.0.0.0"
PORT = cluster_config["port"]
NUM_WORKERS = cluster_config["num_workers"]
SEED = cluster_config.get("seed", 42)
WORLD_SIZE = NUM_WORKERS + 1
TIMEOUT = cluster_config["timeout"]

RANK = 0
num_nodes = nn_config['num_nodes']
model_name = nn_config['model_name']

criterion = torch.nn.CrossEntropyLoss()


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[LEADER]")
logger.info(f"Server will bind to IP: {HOST_IP}, Port: {PORT}")

step_event = threading.Event()
lock = threading.Lock()

workers = {}
grads_received = defaultdict(dict)




def handle_worker(conn: socket.SocketType, addr: tuple[str, int]) -> None:
    logger.info(f"Handling worker at {addr}")

    while True:
        try:
            message = receive_message(conn)

            # Handle connection closed or empty message
            if message is None:
                logger.info(f"Worker {addr} closed connection")
                break

            # Unpack the message tuple
            command, payload = message

            
            logger.info(
                f"Received message '{command}' from worker {addr} (rank {rank})"
            )

            if command == "get_model_layers":
                # logger.info(f"[Step {recv_step}] Storing gradients from worker {rank}")
                # with lock:
                #     curr_step = recv_step
                #     grads_received[curr_step][rank] = grads
                #     logger.info(
                #         f"[Step {recv_step}] Now have {len(grads_received[curr_step])} gradient sets"
                #     )
                # step_event.set()
                rank, n_layers, n_nodes = payload['rank'], payload['n_layers'], payload['n_nodes']
                
                model_layers = get_model_per_node(
                    model,
                    num_nodes=n_nodes,
                    local_rank=rank,
                    model_name=model_name,
                    total_layers=n_layers
                )
                
                send_message(conn, ("model_layers", model_layers))
            
            elif command == "disconnect":
                logger.info(f"Worker {addr} requested disconnection")
                break
            
            elif command == 'forward_activations':
                
                from_rank, to_rank, sock, activations = payload['from_rank'], payload['to_rank'], payload['sock'], payload['activations']
                    
                send_message(conn, {'forward_activations', activations, 'next_sock': sock, 'from': from_rank, 'to', to_rank})
                    
            # Add handling for other commands if needed, e.g., 'disconnect'
        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            break


    logger.info(f"Worker {addr} disconnected")
    conn.close()




config = AutoConfig.from_pretrained(nn_config["model_name"])

if nn_config['model_name'] == 'causal_gpt2':
    model = GPT2LMHeadModel(config)

model = model.to(get_device())
logger.info(f"Model initialized on device: {get_device()}")


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
sock.bind((HOST_IP, PORT))

# Listen for incoming connections
sock.listen(5)
logger.info(f"Server listening on {HOST_IP}:{PORT}")


def main():
   

    # Accept connections and wait for registration
    registered_workers = {}  # rank -> socket
    while len(registered_workers) < NUM_WORKERS:
        client_socket, client_address = sock.accept()
        logger.info(f"Accepted connection from {client_address}")

        # Wait for registration message
        try:
            message = receive_message(client_socket)
            if message is None:
                logger.warning(
                    f"Connection from {client_address} closed before registration"
                )
                client_socket.close()
                continue

            command, rank = message
            if command == "register":
                logger.info(f"Worker {rank} registered from {client_address}")
                registered_workers[rank] = client_socket
                workers[client_address] = client_socket
                threading.Thread(
                    target=handle_worker, args=(client_socket, client_address)
                ).start()
            else:
                logger.warning(f"Unexpected message from {client_address}: {command}")
                client_socket.close()
        except Exception as e:
            logger.error(f"Error during registration from {client_address}: {e}")
            client_socket.close()
            continue

    logger.info(f"All workers connected. Starting inference on {model_name}...")

    for worker_socket in registered_workers.values():
        send_message(worker_socket, "start_inference")

    logger.info(f"Starting inference for {model_name}.")
    
    while True:
        user_input = input("Enter text: ")
        if user_input.lower() in {"exit", "quit"}:
            logger.info("Exiting inference loop.")
            break
        prompt = user_input.strip()
        max_new_tokens = nn_config.get("max_new_tokens", 20)
        decoding_strategy = nn_config.get("decoding_strategy", "greedy")
        # Send generation request to all workers
        for rank, worker_socket in registered_workers.items():
            
            send_message(
                worker_socket,
                (
                    "generate_text",
                    {
                        "prompt": prompt,
                        "max_new_tokens": max_new_tokens,
                        "decoding_strategy": decoding_strategy,
                    },
                ),
            )

            ack = receive_message(worker_socket)
            
            if ack != "ack_generate":
                logger.error(f"Unexpected ack from worker {rank}: {ack}. Cannot continue.")
                break
        
    logger.info("Inference completed successfully!")
       
    sock.close()


if __name__ == "__main__":
    main()
