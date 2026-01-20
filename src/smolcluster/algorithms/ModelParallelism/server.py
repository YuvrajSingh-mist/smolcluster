import gc
import heapq
import logging
import gc
import socket
import threading
from collections import defaultdict
from pathlib import Path
import yaml

from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer


from smolcluster.utils.common_utils import (
    receive_message,
    send_message,
)
from smolcluster.utils.device import get_device
from smolcluster.utils.decoding import sample_next_token


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
model_name = 'causal_gpt2'  # Set model name
model_config = nn_config[model_name]  # Get nested config
num_nodes = model_config['num_nodes']


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





config = AutoConfig.from_pretrained(model_config["hf_model_name"])

if model_name == 'causal_gpt2':
    model = GPT2LMHeadModel(config)
    tokenizer = AutoTokenizer.from_pretrained(model_config["hf_model_name"])

model = model.to(get_device())
logger.info(f"Model initialized on device: {get_device()}")


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
sock.bind((HOST_IP, PORT))

# Listen for incoming connections
sock.listen(5)
logger.info(f"Server listening on {HOST_IP}:{PORT}")


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
        max_new_tokens = model_config.get("max_new_tokens", 20)
        decoding_strategy = model_config.get("decoding_strategy", "greedy")
        temperature = model_config.get("temperature", 0.6)
        top_p = model_config.get("top_p", 0.9)
        
        # Generate tokens one at a time by looping through all workers for each token
        for token_idx in range(max_new_tokens):
            activations = None
            # Send generation request to all workers in rank order (0, 1, 2, ...)
            for rank, worker_socket, addr in sorted(worker_queue):
                
                send_message(
                    worker_socket,
                    (
                        "generate_activations",
                        {
                            "prompt": prompt,
                            "activations": activations,
                            "input_ids": tokenized_prompt,
                            "max_new_tokens": 1,  # Generate one token at a time
                            "decoding_strategy": decoding_strategy,
                        },
                    ),
                )

                message = receive_message(worker_socket)
                
                command, payload = message
                
                if command == 'forward_activations':
                    activations = payload['activations'].to(get_device())
                    from_rank = payload['from_rank']
                    to_rank = payload['to_rank']
                    logger.info(f"Received activations forwarded from worker {from_rank} to worker {to_rank}")
                    
                else:
                    logger.error(f"Unexpected command from worker {rank}: {command}. Cannot continue.")
                    break
            
            # After all workers process, sample next token from final activations
            tokenized_prompt, should_stop = sample_next_token(
                activations, 
                tokenized_prompt, 
                temperature, 
                tokenizer,
                decoding_strategy=decoding_strategy,
                top_p=top_p
            )
            
            if should_stop:
                break
        
        # Decode generated text
        text = tokenizer.decode(tokenized_prompt[0], skip_special_tokens=True)
       
        print("Generated text is: ", prompt + ' ' + text)
        
        del activations 
        
        gc.collect()
        activations = None
    
    for rank, worker_socket, addr in sorted(worker_queue):
        
        send_message(worker_socket, "down")
        
    logger.info("Inference completed successfully!")
       
    sock.close()


if __name__ == "__main__":
    main()
