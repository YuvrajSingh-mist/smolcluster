"""SynchronousPS inference worker — holds a model shard and processes forward-pass requests from the inference server."""
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
    load_model_and_tokenizer,
    receive_message,
    send_message,
)
from smolcluster.utils.decoding import sample_next_token
from smolcluster.utils.device import get_device

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

    for key, value in cfg.items():
        if isinstance(value, dict) and "hf_model_name" in value:
            return key, value

    raise ValueError("No valid model config found. Expected entry with 'hf_model_name'.")


MODEL_NAME, MODEL_CFG = resolve_model_config(model_configs)

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
logger = logging.getLogger(f"[SYNCPS-INF-WORKER-{WORKER_RANK}]")


def connect_to_server(
    host: str, port: int, max_retries: int = 60, retry_delay: float = 3.0
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
            logger.info(f"Connected to server at {host}:{port} on attempt {attempt + 1}")
            return sock
        except (OSError, ConnectionRefusedError, socket.timeout) as exc:
            sock.close()
            if attempt < max_retries - 1:
                logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {retry_delay}s"
                )
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Failed to connect to server at {host}:{port}"
                ) from exc

    raise RuntimeError(f"Failed to connect to server at {host}:{port}")


def main() -> None:
    server_hostname = cluster_config["server"]
    server_ip = cluster_config["host_ip"][server_hostname]

    port_config = cluster_config["port"]
    if isinstance(port_config, dict):
        server_port = int(port_config.get(server_hostname, port_config.get("default", 65432)))
    else:
        server_port = int(port_config)

    device = get_device()
    hf_model_name = MODEL_CFG["hf_model_name"]
    tokenizer_cfg = MODEL_CFG.get("tokenizer", {})

    logger.info(f"Loading model '{MODEL_NAME}' from {hf_model_name} on {device}")
    model, tokenizer = load_model_and_tokenizer(
        hf_model_name=hf_model_name,
        device=device,
        hf_token=HF_TOKEN,
        tokenizer_cfg=tokenizer_cfg,
        load_tokenizer=True,
        logger=logger,
    )
    
    if model is None:
        raise RuntimeError("Failed to load model")
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer")

    sock = connect_to_server(server_ip, server_port)
    send_message(sock, ("register", WORKER_RANK))

    logger.info("Waiting for start signal")
    while True:
        recv_command = receive_message(sock)
        if recv_command == "start_inference":
            logger.info("Received start_inference")
            break

    logger.info("Ready for generation requests")
    while True:
        message = receive_message(sock)
        if message is None:
            logger.warning("Server disconnected")
            break

        if message == "down":
            logger.info("Received shutdown")
            break

        command, payload = message
        
        if command == "generate_logits":
            # Compute logits for ensemble averaging (similar to ClassicDP)
            input_ids = payload["input_ids"].to(device)
            
            with torch.inference_mode():
                logits = model(input_ids).logits.cpu()
            
            send_message(
                sock,
                (
                    "logits_ready",
                    {
                        "worker_rank": WORKER_RANK,
                        "logits": logits,
                    },
                ),
            )
        
     
        else:
            logger.warning(f"Unknown command: {command}")
            continue

    sock.close()


if __name__ == "__main__":
    main()
