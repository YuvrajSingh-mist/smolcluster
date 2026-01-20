import gc
import heapq
import logging
import os
import socket
import sys
import threading
from collections import defaultdict
from pathlib import Path
import torch
from typing import List
import yaml

from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer


from smolcluster.utils.common_utils import (
    receive_message,
    send_message,
)
from smolcluster.utils.device import get_device


# Load configs
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" 
with open(CONFIG_DIR / "model_parallelism" / "model_config.yaml") as f:
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

model = None
tokenizer = None





config = AutoConfig.from_pretrained(nn_config["hf_model_name"])

if nn_config['model_name'] == 'causal_gpt2':
    model = GPT2LMHeadModel(config)
    tokenizer = AutoTokenizer.from_pretrained(nn_config["hf_model_name"])

model = model.to(get_device())
logger.info(f"Model initialized on device: {get_device()}")


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
sock.bind((HOST_IP, PORT))

# Listen for incoming connections
sock.listen(5)
logger.info(f"Server listening on {HOST_IP}:{PORT}")


def generate_text(activations: List[torch.Tensor], tokenizer: AutoTokenizer, max_new_tokens: int, decoding_strategy: str, temperature: float = 1.0) -> str:
    """
    Generate text from the final activations using the tokenizer.
    """
    generated_ids = activations.clone().detach().to(get_device())

    if decoding_strategy == "greedy":
        for _ in range(max_new_tokens):
            next_token_logits = generated_ids[:, -1, :] / temperature
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            print(f"Next token ID: {next_token_id.item()}")
            print(generated_ids)
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    else:
        raise NotImplementedError(f"Decoding strategy '{decoding_strategy}' not implemented.")

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
    return generated_text

def main():
   
    logger.info(f"Server is running at {HOST_IP}:{PORT}")
    
    # Accept connections and wait for registration
    # Use priority queue to maintain workers sorted by rank
    worker_queue = []  # Priority queue: [(rank, socket, address)]
    registered_workers = {}  # rank -> socket (for quick lookup)
    
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
                logger.info(f"Worker rank {rank} registered from {client_address}")
                registered_workers[rank] = client_socket
                workers[client_address] = client_socket
                # Add to priority queue sorted by rank
                heapq.heappush(worker_queue, (rank, client_socket, client_address))
                logger.info(f"Worker rank {rank} added to priority queue (queue size: {len(worker_queue)})")
                
            else:
                logger.warning(f"Unexpected message from {client_address}: {command}")
                client_socket.close()
        except Exception as e:
            logger.error(f"Error during registration from {client_address}: {e}")
            client_socket.close()
            continue

    logger.info(f"All workers connected. Starting inference on {model_name}...")
    logger.info(f"Worker priority queue (by rank): {[(rank, addr) for rank, _, addr in worker_queue]}")

    # Send start_inference to workers in rank order
    for rank, worker_socket, addr in sorted(worker_queue):
        logger.info(f"Sending start_inference to worker rank {rank} at {addr}")
        send_message(worker_socket, "start_inference")

    logger.info(f"Starting inference for {model_name}.")
    
    while True:
        user_input = input("Enter text: ")
        if user_input.lower() in {"exit", "quit"}:
            logger.info("Exiting inference loop.")
            break
        
        prompt = user_input.strip()
        
        tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(get_device())
        max_new_tokens = nn_config.get("max_new_tokens", 20)
        decoding_strategy = nn_config.get("decoding_strategy", "greedy")
        activations = None
        # Send generation request to all workers in rank order (0, 1, 2, ...)
        for rank, worker_socket, addr in sorted(worker_queue):
            logger.info(f"Sending generation request to worker rank {rank} at {addr}")
            
            send_message(
                worker_socket,
                (
                    "generate_activations",
                    {
                        "prompt": prompt,
                        "activations": activations,
                        "input_ids": tokenized_prompt,
                        "max_new_tokens": max_new_tokens,
                        "decoding_strategy": decoding_strategy,
                    },
                ),
            )

            message = receive_message(worker_socket)
            
            command, payload = message
            
            if command == 'forward_activations':
                activations = payload['activations']
                from_rank = payload['from_rank']
                to_rank = payload['to_rank']
                logger.info(f"Received activations forwarded from worker {from_rank} to worker {to_rank}")
                
            else:
                logger.error(f"Unexpected command from worker {rank}: {command}. Cannot continue.")
                break
        
        # After all workers have processed, generate the text from the latest activations
        
        text = generate_text(activations, tokenizer, max_new_tokens, decoding_strategy)
        logger.info(f"Generated text: {text}")
        
        print("Generated text is:\n", prompt + ' ' + text)
       
    logger.info("Inference completed successfully!")
       
    sock.close()


if __name__ == "__main__":
    main()
