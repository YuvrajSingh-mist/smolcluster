import gc
import heapq
import logging
import gc
import socket
import threading
from collections import defaultdict
from pathlib import Path
import yaml
import torch
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer


from smolcluster.utils.common_utils import (
    receive_message,
    send_message,
)
from smolcluster.utils.device import get_device
from smolcluster.utils.decoding import sample_next_token
from smolcluster.utils.model_downloader import ensure_model_weights
from smolcluster.utils.layers import (
    get_hfmodel_per_node,
    load_weights_per_node
)


# Load configs
CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs" 
with open(CONFIG_DIR / "model_parallelism" / "model_config_inference.yaml") as f:
    nn_config = yaml.safe_load(f)

with open(CONFIG_DIR / "model_parallelism" / "model_parallelism.yaml") as f:
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

# Ensure model weights are downloaded before workers connect
weights_model_name = model_config.get('weights_model_name', 'gpt2')
logger.info(f"Checking for model weights ({weights_model_name})...")
weights_path = ensure_model_weights(model_identifier=weights_model_name)
logger.info(f"Model weights ready at: {weights_path}")

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

# Load model layers for server (rank 0)
num_layers = model_config['num_layers']
logger.info(f"Loading server's share of model layers (rank {RANK})...")

layer_mapping, out_layers, results = get_hfmodel_per_node(
    model,
    num_nodes=num_nodes,
    local_rank=RANK,
    model_name=model_name,
    total_layers=num_layers
)

model_layers = load_weights_per_node(
    model_name=model_name,
    weights_path=str(weights_path),
    out_layers=out_layers,
    layer_mapping=layer_mapping,
    local_rank=RANK,
    num_nodes=num_nodes,
    results=results
)

model_layers = model_layers.to(get_device())
logger.info(f"Server loaded {len(model_layers)} layers")


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
    client_socket = None  # API client socket
    
    # Accept all connections (workers + API client)
    while len(registered_workers) < NUM_WORKERS or client_socket is None:
        conn, address = sock.accept()
        logger.info(f"Accepted connection from {address}")

        # Wait for registration message
        try:
            message = receive_message(conn)
            if message is None:
                logger.warning(
                    f"Connection from {address} closed before registration"
                )
                conn.close()
                continue

            command, rank = message
            if command == "register":
                logger.info(f"Worker rank {rank} registered from {address}")
                registered_workers[rank] = conn
                workers[address] = conn
                # Add to priority queue sorted by rank
                heapq.heappush(worker_queue, (rank, conn, address))
                logger.info(f"Worker rank {rank} added to priority queue (queue size: {len(worker_queue)})")
            elif command == "register_client":
                logger.info(f"API client registered from {address}")
                client_socket = conn
                send_message(client_socket, ("client_registered", None))
            else:
                logger.warning(f"Unexpected message from {address}: {command}")
                conn.close()
        except Exception as e:
            logger.error(f"Error during registration from {address}: {e}")
            conn.close()
            continue

    logger.info(f"All workers connected. Starting inference on {model_name}...")
    logger.info(f"Worker priority queue (by rank): {[(rank, addr) for rank, _, addr in worker_queue]}")

    # Send start_inference to workers in rank order
    for rank, worker_socket, addr in sorted(worker_queue):
        logger.info(f"Sending start_inference to worker rank {rank} at {addr}")
        send_message(worker_socket, "start_inference")

    logger.info(f"Starting inference for {model_name}.")
    logger.info("Waiting for inference requests from API client...")
    
    while True:
        # Wait for inference request from API client
        try:
            message = receive_message(client_socket)
            if message is None:
                logger.warning("Client disconnected")
                break
            
            command, payload = message
            
            if command == "disconnect":
                logger.info("Client requested disconnect")
                break
            elif command != "inference":
                logger.warning(f"Unexpected command: {command}")
                send_message(client_socket, ("error", {"message": f"Unknown command: {command}"}))
                continue
                
            # Extract inference parameters
            prompt = payload.get("prompt", "").strip()
            if not prompt:
                send_message(client_socket, ("error", {"message": "Empty prompt"}))
                continue
                
            logger.info(f"Received inference request: {prompt[:50]}...")
            
        except Exception as e:
            logger.error(f"Error receiving request: {e}")
            break
        
        tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(get_device())
        original_prompt_length = tokenized_prompt.shape[1]  # Track prompt length
        
        # Get active decoding strategy and its parameters
        active_strategy = model_config.get("active_decoding_strategy", "top_p")
        strategies = model_config.get("decoding_strategies", {})
        strategy_params = strategies.get(active_strategy, {})
        
        max_new_tokens = payload.get("max_tokens", model_config.get("max_new_tokens", 20))
        decoding_strategy = payload.get("decoding_strategy", active_strategy)
        temperature = payload.get("temperature", strategy_params.get("temperature", 1.0))
        top_p = payload.get("top_p", strategy_params.get("p", 0.9))
        top_k = payload.get("top_k", strategy_params.get("k", 50))
        
        # Generate tokens one at a time by looping through all workers for each token
        for token_idx in range(max_new_tokens):
            
            activations = None
        
            
            out = tokenized_prompt
        
            logger.info(f"Generating activations for input IDs for rank 0")

            with torch.no_grad():
    
                out = model_layers[0](out)
            
                pos_ids = torch.arange(out.shape[1], dtype=torch.long, device=get_device())
                out = out + model_layers[1](pos_ids)
                
                for layer in model_layers[2:]:
                    output = layer(out)
                    out = output[0] if isinstance(output, tuple) else output
            
            logger.info("Finsihed generating activations for local_rank 0")
            
            activations = out.cpu()
            # Send generation request to all workers in rank order (1, 2, ...)
            for rank, worker_socket, addr in sorted(worker_queue):
                
                send_message(
                    worker_socket,
                    (
                        "generate_activations",
                        {
                            
                            "activations": activations,
                            "input_ids": tokenized_prompt.cpu(),  # Move to CPU before sending
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
                top_p=top_p,
                top_k=top_k
            )
            
            if should_stop:
                break
        
        # Extract only the generated tokens (exclude original prompt)
        generated_tokens = tokenized_prompt[0, original_prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
       
        logger.info(f"Generated: {generated_text[:100]}...")
        
        # Send result back to API client
        try:
            send_message(client_socket, ("inference_result", {"text": generated_text}))
            logger.info("Sent result to client")
        except Exception as e:
            logger.error(f"Failed to send result to client: {e}")
        
        del activations 
        
        gc.collect()
        activations = None
    
    for rank, worker_socket, addr in sorted(worker_queue):
        
        send_message(worker_socket, "down")
        
    logger.info("Inference completed successfully!")
       
    sock.close()


if __name__ == "__main__":
    main()
