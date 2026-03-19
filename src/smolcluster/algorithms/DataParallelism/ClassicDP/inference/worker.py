import logging
import os
import socket
import subprocess
import sys
import time
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

CONFIG_DIR = Path(__file__).parent.parent.parent.parent.parent / "configs"

with open(CONFIG_DIR / "inference" / "model_config_inference.yaml") as f:
    inference_config = yaml.safe_load(f)

with open(CONFIG_DIR / "cluster_config_classicdp.yaml") as f:
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

    for key, value in cfg.items():
        if isinstance(value, dict) and "hf_model_name" in value:
            return key, value

    raise ValueError("No valid model config found. Expected entry with 'hf_model_name'.")


MODEL_NAME, MODEL_CFG = resolve_model_config(model_configs)
TOPOLOGY = cluster_config["allToAllTopology"]["workers"]["regular"]

if len(sys.argv) > 1:
    WORKER_RANK = int(sys.argv[1])
else:
    WORKER_RANK = int(input("Enter worker rank: "))

if len(sys.argv) > 2:
    HOSTNAME = sys.argv[2]
else:
    HOSTNAME = input("Enter worker hostname: ")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(f"[CLASSICDP-INF-WORKER-{WORKER_RANK}]")

EFFECTIVE_STRATEGIES = get_effective_decoding_strategies(
    MODEL_CFG,
    hf_token=HF_TOKEN,
    logger=logger,
)


def connect_with_retry(
    host: str, port: int, max_retries: int = 60, retry_delay: float = 2.0
) -> socket.socket:
    logger.info(f"Warming up network by pinging {host}...")
    try:
        subprocess.run(
            ["ping", "-c", "2", "-W", "1000", host], capture_output=True, timeout=8
        )
    except Exception:
        pass

    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        try:
            sock.connect((host, port))
            sock.settimeout(None)
            logger.info(f"Connected to {host}:{port} on attempt {attempt + 1}")
            return sock
        except (OSError, ConnectionRefusedError, socket.timeout) as exc:
            sock.close()
            if attempt < max_retries - 1:
                logger.warning(
                    f"Connection to {host}:{port} failed ({exc}), retrying in {retry_delay}s"
                )
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to connect to {host}:{port}") from exc

    raise RuntimeError(f"Failed to connect to {host}:{port}")




def get_worker_cfg(rank: int) -> dict:
    return next(w for w in TOPOLOGY if int(w["rank"]) == rank)


def run_rank_zero(
    model: Any,
    tokenizer: Any,
    server_sock: socket.socket,
    peer_sockets: dict[int, socket.socket],
) -> None:
    client_socket = None

    logger.info("Waiting for chat backend client registration...")
    while client_socket is None:
        conn, addr = server_sock.accept()
        logger.info(f"Accepted connection from {addr}")
        message = receive_message(conn)
        if message is None:
            conn.close()
            continue

        command, payload = message
        if command == "register_client":
            client_socket = conn
            send_message(client_socket, ("client_registered", None))
            logger.info("Client registered")
        else:
            logger.warning(f"Unexpected registration command: {command}")
            conn.close()

    for rank, peer_sock in sorted(peer_sockets.items()):
        logger.info(f"Signaling worker {rank} to start inference")
        send_message(peer_sock, "start_inference")

    device = get_device()
    while True:
        request = receive_message(client_socket)
        if request is None:
            logger.warning("Client disconnected")
            break

        command, payload = request
        if command == "disconnect":
            logger.info("Client requested disconnect")
            break
        if command != "inference":
            send_message(client_socket, ("error", {"message": f"Unknown command: {command}"}))
            continue

        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            send_message(client_socket, ("error", {"message": "Empty prompt"}))
            continue

        try:
            max_tokens, decoding_strategy, temperature, top_p, top_k = resolve_generation_request_params(payload, MODEL_CFG, EFFECTIVE_STRATEGIES)
        except ValueError as exc:
            send_message(client_socket, ("error", {"message": str(exc)}))
            continue
        tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids
        
        for token_idx in range(max_tokens):
            with torch.inference_mode():
                local_logits = model(tokenized_prompt.to(device)).logits.cpu()

            gathered_logits = [local_logits]
            for rank, peer_sock in sorted(peer_sockets.items()):
                send_message(
                    peer_sock,
                    (
                        "generate_activations",
                        {
                            "activations": None,
                            "input_ids": tokenized_prompt.cpu(),
                            "max_new_tokens": 1,
                            "decoding_strategy": decoding_strategy,
                        },
                    ),
                )
                response = receive_message(peer_sock)
                if response is None:
                    raise RuntimeError(f"Worker {rank} disconnected during inference")
                resp_command, resp_payload = response
                if resp_command != "forward_activations":
                    raise RuntimeError(
                        f"Unexpected response from worker {rank}: {resp_command}"
                    )
                gathered_logits.append(resp_payload["activations"])

            averaged_logits = torch.stack(gathered_logits, dim=0).mean(dim=0)
            tokenized_prompt, should_stop = sample_next_token(
                averaged_logits,
                tokenized_prompt,
                temperature,
                tokenizer,
                decoding_strategy=decoding_strategy,
                top_p=top_p,
                top_k=top_k,
            )

            new_token_id = tokenized_prompt[0, -1].item()
            new_token_text = tokenizer.decode([new_token_id], skip_special_tokens=True)
            send_message(
                client_socket,
                ("token", {"text": new_token_text, "token_idx": token_idx}),
            )

            if should_stop:
                break

        generated_tokens = tokenized_prompt[0, original_prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        send_message(client_socket, ("inference_complete", {"text": generated_text}))

    for rank, peer_sock in sorted(peer_sockets.items()):
        try:
            send_message(peer_sock, "down")
            peer_sock.close()
            logger.info(f"Closed connection to worker {rank}")
        except Exception:
            pass

    client_socket.close()
    server_sock.close()


def run_peer(model: Any, leader_socket: socket.socket) -> None:
    device = get_device()
    logger.info("Waiting for commands from rank 0")

    while True:
        message = receive_message(leader_socket)
        if message is None:
            logger.warning("Rank 0 disconnected")
            break

        if message == "start_inference":
            logger.info("Received start_inference")
            continue
        if message == "down":
            logger.info("Received shutdown")
            break

        command, payload = message
        if command != "generate_activations":
            logger.warning(f"Unknown command: {command}")
            continue

        input_ids = payload["input_ids"].to(device)
        with torch.inference_mode():
            logits = model(input_ids).logits.cpu()

        send_message(
            leader_socket,
            (
                "forward_activations",
                {
                    "from_rank": WORKER_RANK,
                    "to_rank": WORKER_RANK + 1,
                    "activations": logits,
                },
            ),
        )

    leader_socket.close()


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(
        hf_model_name=MODEL_CFG["hf_model_name"],
        device=get_device(),
        hf_token=HF_TOKEN,
        tokenizer_cfg=MODEL_CFG.get("tokenizer", {}),
        load_tokenizer=True,
        logger=logger,
    )
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for ClassicDP inference worker")

    my_cfg = get_worker_cfg(WORKER_RANK)
    my_port = int(my_cfg["port"])

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", my_port))
    listener.listen(len(TOPOLOGY) + 2)

    if WORKER_RANK == 0:
        logger.info(f"Rank 0 listening on port {my_port}")
        peer_sockets: dict[int, socket.socket] = {}

        for peer in sorted(TOPOLOGY, key=lambda x: x["rank"]):
            peer_rank = int(peer["rank"])
            if peer_rank == 0:
                continue
            peer_sock = connect_with_retry(peer["ip"], int(peer["port"]))
            send_message(peer_sock, ("register_peer", 0))
            peer_sockets[peer_rank] = peer_sock
            logger.info(f"Connected to worker {peer_rank} at {peer['ip']}:{peer['port']}")

        run_rank_zero(model, tokenizer, listener, peer_sockets)
    else:
        logger.info(f"Rank {WORKER_RANK} listening on port {my_port} for rank 0")
        leader_conn = None

        while leader_conn is None:
            conn, addr = listener.accept()
            message = receive_message(conn)
            if message is None:
                conn.close()
                continue

            command, rank = message
            if command == "register_peer" and int(rank) == 0:
                logger.info(f"Registered rank 0 connection from {addr}")
                leader_conn = conn
            else:
                logger.warning(f"Unexpected connection from {addr}: {message}")
                conn.close()

        listener.close()
        run_peer(model, leader_conn)


if __name__ == "__main__":
    main()
