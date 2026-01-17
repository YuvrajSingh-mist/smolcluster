import logging
import math
import socket
import subprocess
import time
from pathlib import Path
import wandb
import torch

from smolcluster.utils.common_utils import (
    get_gradients,
    get_weights,
    receive_message,
    send_message,
    set_weights,
)
from smolcluster.utils.logging_utils import setup_cluster_logging
from torch.utils.data import DataLoader

# Setup logging (will be replaced by setup_cluster_logging in run_syncps_worker)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[WORKER]")



def evaluate(
    device: torch.device, model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module, decoder_type_ppl: bool = False
) -> tuple[float, float]:
    model.eval()
    total_val_loss = 0.0
    
    val_iter = iter(val_loader)

    with torch.no_grad():
        for _step in range(len(val_loader)):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)

            data, target = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            B,T,C = output.shape
            output = output.view(B*T, C)
            target = target.view(B*T)
            loss = criterion(output, target)
            total_val_loss += loss.item()
            # _, predicted = torch.max(output.data, 1)
            # total += target.size(0)
            # correct += (predicted == target).sum().item()
    avg_loss = total_val_loss / len(val_loader)
    ppl = math.exp(avg_loss) if decoder_type_ppl else None
    # accuracy = 100 * (correct / total)
    model.train()
    return avg_loss, ppl

def polyak_blend_weights(
    current_weights: dict[str, torch.Tensor],
    server_weights: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Blend current worker model with server's model using Polyak averaging.

    Args:
        current_weights: Current worker model weights
        server_weights: Server's averaged model weights
        alpha: Blending factor (0.5 = equal blend, higher = more server influence)

    Returns:
        Blended model weights
    """
    blended = {}
    for name in current_weights.keys():
        blended[name] = (
            alpha * server_weights[name] + (1.0 - alpha) * current_weights[name]
        )
    return blended



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


def run_syncps_worker(
    model,
    train_loader,
    val_loader,
    config,
    worker_rank,
    hostname,
    device,
    criterion,
    host_ip,
    port,
):
    """
    Run Synchronous Parameter Server worker training.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance (not used in SyncPS, gradients sent to server)
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        worker_rank: Worker rank (0-indexed)
        hostname: Worker hostname
        device: Device to run on
        criterion: Loss criterion
        host_ip: Server IP address
        port: Server port
    """
    global logger
    
    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="worker",
        rank=worker_rank,
        hostname=hostname,
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs")
    )
    logger.info(f"🚀 SyncPS Worker {worker_rank} starting up")
    
    # Extract configuration
   
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config.get("track_gradients", False)
    polyak_alpha = config.get("polyak_alpha", 0.5)
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    
    # Connect to server
    sock = connect_to_server(host_ip, port)
    
    # Register with server
    logger.info(f"Registering as worker {worker_rank} with server...")
    send_message(sock, ("register", worker_rank))
    
    logger.info(
        f"Data loaders ready. Train size: {len(train_loader)}, Test size: {len(val_loader)}"
    )
    
    # Wait for start signal
    logger.info("Waiting for start_training signal from server...")
    while True:
        recv_command = receive_message(sock)
        if recv_command == "start_training":
            logger.info("Received start_training command from server.")
            break
    
    logger.info("Starting training loop...")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            logger.info(f"[Step {step}] Starting forward and backward pass")
            
            data, target = data.to(device), target.to(device)
            output = model(data.view(data.size(0), -1))
            B,T,C = output.shape
            output = output.view(B*T, C)
            target = target.view(B*T)
            loss = criterion(output, target)
            total_loss += loss.item()
            train_ppl = math.exp(loss.item())
            
            wandb.log({
                "step": step,
                "epoch": epoch + 1,
                "losses/train_step": loss.item(),
                "losses/total_train": total_loss / (batch_idx + 1),
                "ppl/train": train_ppl,
            })
            
            model.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if config.get("gradient_clipping", {}).get("enabled", False):
                max_norm = config["gradient_clipping"].get("max_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                logger.info(
                    f"[Step {step}] Applied gradient clipping with max_norm={max_norm}"
                )
            
            grads = get_gradients(model)
            logger.info(f"[Step {step}] Computed local gradients")
            
            # Send gradients to server
            logger.info(f"[Step {step}] Sending gradients to server")
            send_message(sock, ("parameter_server_reduce", step, worker_rank, grads))
            
            # Receive updated weights from server
            logger.info(f"[Step {step}] Waiting for model weights from server")
            data_recv = receive_message(sock)
            command, recv_step, weights = data_recv
            logger.info(
                f"[Step {step}] Received '{command}' from server for step {recv_step}"
            )
            
            assert recv_step == step, "Step mismatch in communication with server."
            
            if command == "model_weights":
                # Apply Polyak averaging
                current_weights = get_weights(model)
                blended_weights = polyak_blend_weights(current_weights, weights, polyak_alpha)
                set_weights(blended_weights, model)
                logger.info(
                    f"[Step {step}] ✅ Applied Polyak-averaged weights (alpha={polyak_alpha})"
                )
            else:
                logger.warning(
                    f"[Step {step}] Expected 'model_weights' but got '{command}'"
                )
            
            # Log gradient norms if tracking enabled
            if track_gradients:
              
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log({
                            f"gradients/layer_{name}": grad_norm,
                            "step": step,
                            "epoch": epoch + 1,
                        })
        
            # Evaluation
            if step % eval_steps == 0:
                val_loss, val_ppl = evaluate(device, model, val_loader, criterion, decoder_type_ppl)
                
                if decoder_type_ppl:
                    wandb.log({
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                        "ppl/val": val_ppl,
                    })
                    logger.info(
                        f"Evaluation at step {step}: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}"
                    )
                else:
                    wandb.log({
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                    })
                    logger.info(
                        f"Evaluation at step {step}: Val Loss={val_loss:.4f}"
                    )
                model.train()
      
            wandb.log({
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/train_batch": loss.item(),
                })
           
            logger.info(f"Epoch {epoch + 1}, Step {step}: Loss={loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
 
        wandb.log({
                "epoch": epoch + 1,
                "losses/train_epoch": avg_loss,
            })
        
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed. Avg Loss: {avg_loss:.4f}"
        )
    
    sock.close()
    logger.info("Worker training completed and connection closed.")

