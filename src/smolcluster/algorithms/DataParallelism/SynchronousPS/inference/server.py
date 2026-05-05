"""SynchronousPS inference server — loads a model shard, coordinates token generation across workers, and streams results."""
import logging
import os
import socket
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from smolcluster.utils.common_utils import (
    get_effective_decoding_strategies,
    load_model_and_tokenizer,
    receive_message,
    resolve_generation_request_params,
    send_message,
)
from smolcluster.utils.decoding import sample_next_token
from smolcluster.utils.device import get_device
from smolcluster.utils.decoding import sample_next_token
                
CONFIG_DIR = Path(__file__).parent.parent.parent.parent.parent / "configs"

with open(CONFIG_DIR / "inference" / "model_config_inference.yaml") as f:
    inference_config = yaml.safe_load(f)

with open(CONFIG_DIR / "inference" / "cluster_config_inference.yaml") as f:
    cluster_config = yaml.safe_load(f)

model_configs = inference_config.get("dp", inference_config)
MODEL_NAME = model_configs.get("active_model", "hf_model")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or None


def resolve_model_config(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "hf_model_name" in cfg:
        return "hf_model", cfg

    model_name = cfg.get("active_model")
    if model_name and model_name in cfg:
        return model_name, cfg[model_name]

    raise ValueError("No valid model config found. Expected entry with 'hf_model_name'.")


MODEL_NAME, MODEL_CFG = resolve_model_config(model_configs)

HOST_IP = "0.0.0.0"
SERVER_HOSTNAME = cluster_config["server"]
port_config = cluster_config["port"]
if isinstance(port_config, dict):
    SERVER_PORT = port_config.get(SERVER_HOSTNAME, port_config.get("default", 65432))
else:
    SERVER_PORT = int(port_config)

NUM_WORKERS = int(cluster_config["num_workers"])

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[SYNCPS-INF-SERVER]")

EFFECTIVE_STRATEGIES = get_effective_decoding_strategies(
    MODEL_CFG,
    hf_token=HF_TOKEN,
    logger=logger,
)




def main() -> None:
    # Load model and tokenizer for server-side generation (rank 0)
    model, tokenizer = load_model_and_tokenizer(
        hf_model_name=MODEL_CFG["hf_model_name"],
        device=get_device(),
        hf_token=HF_TOKEN,
        tokenizer_cfg=MODEL_CFG.get("tokenizer", {}),
        load_model=True,
        load_tokenizer=True,
        logger=logger,
    )
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for SyncPS inference server")
    if model is None:
        logger.warning("Model not loaded - server rank 0 generation will not be available")
    
    device = get_device()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST_IP, SERVER_PORT))
    server_socket.listen(8)

    logger.info(f"SyncPS inference server listening on {HOST_IP}:{SERVER_PORT}")

    worker_sockets: dict[int, socket.socket] = {}
    client_socket = None

    while len(worker_sockets) < NUM_WORKERS or client_socket is None:
        conn, addr = server_socket.accept()
        logger.info(f"Accepted connection from {addr}")

        message = receive_message(conn)
        if message is None:
            conn.close()
            continue

        command, payload = message
        if command == "register":
            rank = int(payload)
            worker_sockets[rank] = conn
            logger.info(f"Registered worker rank {rank}")
        elif command == "register_client":
            client_socket = conn
            send_message(client_socket, ("client_registered", None))
            logger.info("Registered chat backend client")
        else:
            logger.warning(f"Unexpected registration command: {command}")
            conn.close()

    for rank, worker_socket in sorted(worker_sockets.items()):
        logger.info(f"Signaling worker {rank} to start inference")
        send_message(worker_socket, "start_inference")

    logger.info("Ready to process inference requests")

    while True:
        if client_socket is None:
            logger.info("Waiting for chat backend client connection...")
            while client_socket is None:
                conn, addr = server_socket.accept()
                logger.info(f"Accepted connection from {addr}")

                message = receive_message(conn)
                if message is None:
                    conn.close()
                    continue

                command, payload = message
                if command == "register_client":
                    client_socket = conn
                    send_message(client_socket, ("client_registered", None))
                    logger.info("Registered chat backend client")
                elif command == "register":
                    rank = int(payload)
                    worker_sockets[rank] = conn
                    logger.info(f"Registered worker rank {rank}")
                    send_message(conn, "start_inference")
                else:
                    logger.warning(f"Unexpected registration command: {command}")
                    conn.close()

        request = receive_message(client_socket)
        if request is None:
            logger.warning("Client disconnected; waiting for reconnection")
            try:
                client_socket.close()
            except Exception:
                pass
            client_socket = None
            continue

        command, payload = request
        if command == "disconnect":
            logger.info("Client requested disconnect; waiting for reconnection")
            try:
                client_socket.close()
            except Exception:
                pass
            client_socket = None
            continue
        if command != "inference":
            send_message(client_socket, ("error", {"message": f"Unknown command: {command}"}))
            continue

        prompt = (payload.get("prompt") or "").strip()
        messages = payload.get("messages")  # For instruction-based models
        worker_rank = payload.get("worker_rank")
        if worker_rank is None:
            worker_rank = min(worker_sockets.keys()) if worker_sockets else 0
        
        if not prompt and not messages:
            send_message(client_socket, ("error", {"message": "Empty prompt or messages"}))
            continue
        
        # Check if requesting rank 0 (server) or a worker - do this BEFORE worker socket validation
        if worker_rank == 0:
            # Server-side generation will be handled below after tokenization
            pass
        elif worker_rank not in worker_sockets:
            # Validate worker rank only if not rank 0
            send_message(client_socket, ("error", {"message": f"Worker {worker_rank} not available"}))
            continue

        # Check if using instruction-based model
        is_instruction_based = MODEL_CFG.get("is_instruction_based", False)

        # Handle instruction-based models with messages format
        if is_instruction_based and messages:
            # Format messages using chat template
            try:
                tokenizer_cfg = MODEL_CFG.get("tokenizer", {}) or {}
                tokenizer_overrides = tokenizer_cfg.get("overrides", {})
                add_generation_prompt = tokenizer_overrides.get(
                    "add_generation_prompt",
                    tokenizer_cfg.get("add_generation_prompt", True),
                )
                tokenized_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=add_generation_prompt,
                )
                # Older transformers versions return a BatchEncoding instead of a raw tensor
                if hasattr(tokenized_prompt, "input_ids"):
                    tokenized_prompt = tokenized_prompt.input_ids
                if logger:
                    logger.info(f"Applied chat template for instruction-based model")
            except Exception as e:
                logger.error(f"Failed to apply chat template: {e}")
                send_message(client_socket, ("error", {"message": f"Chat template error: {str(e)}"}))
                continue
        else:
            
            tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids

        original_prompt_length = tokenized_prompt.shape[1]
        try:
            max_tokens, decoding_strategy, temperature, top_p, top_k = resolve_generation_request_params(payload, MODEL_CFG, EFFECTIVE_STRATEGIES)

        except ValueError as exc:
            send_message(client_socket, ("error", {"message": str(exc)}))
            continue

        # SyncPS with logit averaging: aggregate logits from all workers
        logger.info(f"Generating with logit averaging across all {len(worker_sockets)} workers + server")
        
        # Include server rank 0 in ensemble if model is loaded
        current_ids = tokenized_prompt.clone()
        
        for token_idx in range(max_tokens):
            gathered_logits = []
            
            # Get server (rank 0) logits if model is loaded
            if model is not None:
                with torch.inference_mode():
                    server_logits = model(current_ids.to(device)).logits.cpu()
                gathered_logits.append(server_logits)
            
            # Request logits from all workers
            for rank, worker_socket in sorted(worker_sockets.items()):
                send_message(
                    worker_socket,
                    (
                        "generate_logits",
                        {
                            "input_ids": current_ids.cpu(),
                        },
                    ),
                )
            
            # Gather logits from all workers
            for rank, worker_socket in sorted(worker_sockets.items()):
                response = receive_message(worker_socket)
                if response is None:
                    logger.error(f"Worker {rank} disconnected during logit generation")
                    send_message(client_socket, ("error", {"message": f"Worker {rank} disconnected"}))
                    break
                
                resp_command, resp_payload = response
                if resp_command != "logits_ready":
                    logger.error(f"Unexpected response from worker {rank}: {resp_command}")
                    send_message(client_socket, ("error", {"message": f"Worker {rank} error: {resp_command}"}))
                    break
                
                gathered_logits.append(resp_payload["logits"])
            
            # Check if we had errors during gathering
            if len(gathered_logits) != len(worker_sockets) + (1 if model is not None else 0):
                logger.error("Failed to gather logits from all workers")
                break
            
            # Average logits across all workers + server
            averaged_logits = torch.stack(gathered_logits, dim=0).mean(dim=0)
            
            # Sample next token
            current_ids, should_stop = sample_next_token(
                averaged_logits,
                current_ids,
                temperature,
                tokenizer,
                decoding_strategy=decoding_strategy,
                top_p=top_p,
                top_k=top_k,
            )
            
            new_token_id = current_ids[0, -1].item()
            token_text = tokenizer.decode([new_token_id], skip_special_tokens=True)
            
            send_message(
                client_socket,
                (
                    "token",
                    {
                        "text": token_text,
                        "token_idx": token_idx,
                    },
                ),
            )
            
            if should_stop:
                break
        
        total_tokens = current_ids.shape[1] - original_prompt_length
        logger.info(f"Generated {total_tokens} tokens with ensemble averaging")
        send_message(client_socket, ("inference_complete", {"worker_rank": 0}))


if __name__ == "__main__":
    main()
