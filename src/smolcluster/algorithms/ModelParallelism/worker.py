import logging
import math
import socket
import subprocess
import time
from typing import Any, Optional

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from smolcluster.utils.checkpointing import CheckpointManager
from smolcluster.utils.common_utils import (
    get_network_metrics,
    receive_message,
    send_message,
)
from smolcluster.utils.layers import get_model_per_node
from smolcluster.utils.logging_utils import setup_cluster_logging


def evaluate(
    device: torch.device,
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    decoder_type_ppl: bool = False,
) -> tuple[float, Optional[float]]:
    """Evaluate model on validation set."""
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            B, T, C = output.shape
            output = output.view(B * T, C)
            target = target.view(B * T)
            loss = criterion(output, target)
            total_val_loss += loss.item()

    avg_loss = total_val_loss / len(val_loader)
    ppl = math.exp(avg_loss) if decoder_type_ppl else None

    return avg_loss, ppl


def should_save_checkpoint(
    step: int, epoch: int, checkpoint_steps: int, total_steps: int
) -> bool:
    """Determine if a checkpoint should be saved at the current step."""
    if checkpoint_steps <= 0:
        return False
    # Save at regular intervals and at the end of training
    return (step % checkpoint_steps == 0 and step != 0) or (step == total_steps - 1)


def compute_worker_activations(
    device: torch.device,
    model: torch.nn.Module,
    data: torch.Tensor,
) -> torch.Tensor:
    """Compute activations for worker node."""
    model.train()
    data = data.to(device)
    hidden = model(data)
    return hidden


def compute_loss(
    act: Any,
    target: Any,
    criterion: Any,
    device: torch.device,
) -> Any:
    """Compute loss for given data and target."""
    # # model.eval()
    # data, target = data.to(get_device()), target.to(get_device())
    # output = model(data)
    # B, T, C = output.shape
    act = act.to(device)
    target = target.to(device)
    B, T, C = act.shape
    output = act.view(B * T, C)
    target = target.view(B * T)
    loss = criterion(output, target)

    return loss


def clear_gpu_cache(device: torch.device) -> None:
    """Clear GPU cache for both MPS and CUDA devices."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# Setup logging (will be replaced by setup_cluster_logging in run_modelparallelism_worker)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[WORKER]")


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


def run_modelparallelism_worker(
    model,
    train_loader,
    val_loader,
    config,
    cluster_config,
    worker_rank,
    hostname,
    device,
    criterion,
    host_ip,
    port,
    resume_checkpoint_path=None,
):
    """
    Run Model Parallelism worker for distributed GPT training.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        worker_rank: Worker rank (1-indexed)
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
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs"),
    )
    logger.info(f"🚀 ModelParallelism Worker {worker_rank} starting up")

    # Extract configuration
    config["batch_size"]
    num_epochs = config["num_epochs"]
    config["eval_steps"]
    track_gradients = config.get("track_gradients", False)
    config.get("decoder_type", {}).get("ppl", False)

    # Set parameters
    local_rank = worker_rank  # Worker rank is already correct (1, 2, etc.)
    cluster_config["num_workers"]
    num_nodes = cluster_config["num_nodes"]
    cluster_config["model_name"]

    # Use provided host_ip and port (from train.py)
    HOST_IP = host_ip
    PORT = port

    # Network configuration
    buffer_size_mb = cluster_config.get("buffer_size", {}).get(hostname, 4)
    track_network_metrics = cluster_config.get("track_network_metrics", False)
    metrics_log_interval = cluster_config.get("metrics_log_interval", 50)
    logger.info(f"Network buffer size: {buffer_size_mb}MB")
    logger.info(f"Network metrics tracking: {track_network_metrics}")

    # Update logger with rank
    logger = logging.getLogger(f"[WORKER-{local_rank}]")

    logger.info(
        f"Worker {local_rank} starting. Connecting to server at {HOST_IP}:{PORT}"
    )

    # Initialize model
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")

    # Load model layers for this worker
    num_layers = config["num_layers"]
    logger.info(f"Loading worker's share of model layers (rank {local_rank})...")

    model_layers, out_layers = get_model_per_node(
        model, num_nodes=num_nodes, local_rank=local_rank, total_layers=num_layers
    )

    model_layers = model_layers.to(device)
    logger.info(f"Loaded {len(model_layers)} layers for worker {local_rank}")

    # Create optimizer for worker's layers only
    learning_rate = config["learning_rate"]
    optimizer = torch.optim.AdamW(model_layers.parameters(), lr=learning_rate)
    logger.info(f"Created optimizer for worker {local_rank} with lr={learning_rate}")

    # Initialize checkpointing
    save_checkpoints = config.get("save_checkpoints", True)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    checkpoint_steps = config.get("checkpoint_steps", 500)
    # Prioritize command-line resume path over config value
    resume_from_checkpoint = resume_checkpoint_path or config.get(
        "resume_from_checkpoint", None
    )
    max_checkpoints_to_keep = config.get("max_checkpoints_to_keep", 3)
    save_optimizer_state = config.get("save_optimizer_state", True)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints_to_keep,
        save_optimizer_state=save_optimizer_state,
        prefix=f"worker_{local_rank}",
    )
    logger.info(
        f"Checkpoint manager initialized: save_checkpoints={save_checkpoints}, checkpoint_steps={checkpoint_steps}, dir={checkpoint_dir}"
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if save_checkpoints and resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        # Create temporary model for loading
        temp_model = torch.nn.Sequential(*model_layers)
        checkpoint_data = checkpoint_manager.load_checkpoint(
            checkpoint_path=resume_from_checkpoint,
            model=temp_model,
            optimizer=optimizer if save_optimizer_state else None,
            scheduler=None,  # MP doesn't use scheduler
            device=device,
        )
        # Copy loaded state back to model_layers
        for i, layer in enumerate(model_layers):
            layer.load_state_dict(temp_model[i].state_dict())
        if checkpoint_data:
            start_epoch = checkpoint_data.get("epoch", 0)
            start_step = checkpoint_data.get("step", 0)
            logger.info(f"Resumed from epoch {start_epoch}, step {start_step}")
        else:
            logger.warning(
                f"Could not load checkpoint from {resume_from_checkpoint}, starting from scratch"
            )

    # Connect to server
    sock = connect_to_server(HOST_IP, PORT)

    # Register with the server
    logger.info(f"Registering as worker {local_rank} with server...")
    send_message(sock, ("register", local_rank))

    while True:
        recv_command = receive_message(sock)

        if recv_command == "start_training":
            logger.info("Received start_training command from server.")
            break

    # Initialize activation caches
    act_in_cache = {}
    act_out_cache = {}
    target_cache = {}

    logger.info("Starting training loop...")

    # Create epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc=f"Worker {local_rank} Epochs", ncols=100)
    
    for epoch in epoch_pbar:
        model_layers.train()
        total_loss = 0.0
        epoch_pbar.set_description(f"Worker {local_rank} - Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Create batch progress bar for this epoch
        batch_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Worker {local_rank} - Epoch {epoch + 1}",
            leave=False,
            ncols=100
        )
        
        for batch_idx, (data, target) in batch_pbar:
            step = epoch * len(train_loader) + batch_idx

            # Skip batches if resuming mid-epoch
            if step < start_step:
                continue

            logger.info(
                f"[Step {step} / {num_epochs * len(train_loader)}] Waiting for activations from server"
            )
            data = data.to(device)
            target = target.to(device)

            # Receive message from server
            message = receive_message(sock)
            command, recv_step, payload = message

            # Handle evaluation messages (doesn't affect step count)
            while command == "evaluate_forward":
                logger.info(
                    f"[Eval] Worker {local_rank} received evaluation activations"
                )

                # Get activations from previous node
                eval_activations = payload["activations"].to(device)

                # Forward through this worker's layers (no gradients for eval)
                with torch.no_grad():
                    model_layers.eval()
                    out = eval_activations
                    for layer in model_layers:
                        out = layer(out)
                        out = out[0] if isinstance(out, tuple) else out
                    model_layers.train()

                logger.info(
                    f"[Eval] Worker {local_rank} sending evaluation activations"
                )

                # Send activations to server
                send_message(
                    sock, ("eval_activations", 0, {"activations": out.detach().cpu()})
                )

                clear_gpu_cache(device)

                # Get next message (should be training message)
                message = receive_message(sock)
                command, recv_step, payload = message

            assert recv_step == step, f"Step mismatch: expected {step}, got {recv_step}"

            if command == "generate_activations_train":
                logger.info(
                    f"[Step {step}] Received command to generate activations for rank {local_rank}."
                )

                # Get activations from previous node
                act_in = payload["activations"].to(device)
                if payload.get("targets") is not None:
                    target_cache[step] = payload["targets"]
                act_in.requires_grad_(True)  # has activations from all prev layers

                out = act_in
                # Forward through this worker's layers
                for layer in model_layers:
                    out = layer(out)
                    out = out[0] if isinstance(out, tuple) else out

                act_out = out

                # Cache activations WITH computation graph (no detach for act_in!)
                act_in_cache[(step, local_rank)] = act_in
                act_out_cache[(step, local_rank)] = act_out

                # Clear unnecessary intermediate tensors
                clear_gpu_cache(device)

                logger.info(
                    f"[Step {step}] Finished generating activations for local_rank {local_rank}"
                )

                logger.info(
                    f"[Step {step}] Sending activations from rank {local_rank} to rank {local_rank + 1}"
                )

                # Send detached copy to next worker/server
                send_message(
                    sock,
                    (
                        "forward_activations",
                        step,
                        {
                            "from_rank": local_rank,
                            "to_rank": local_rank + 1,
                            "activations": act_out.detach().cpu(),
                        },
                    ),
                )

                # Don't delete - keep in cache for backward

            # Receive activations from server/previous worker
            message = receive_message(sock)
            command, recv_step, payload = message

            assert recv_step == step, f"Step mismatch: expected {step}, got {recv_step}"

            if command == "generate_gradients":
                logger.info(
                    f"[Step {step}] Received command to compute gradients for rank {local_rank}."
                )

                # Restore activations from cache (has computation graph)
                act_in = act_in_cache[(step, local_rank)]
                act_out = act_out_cache[(step, local_rank)]

                target_for_loss = target_cache.pop(step, None)
                if target_for_loss is None:
                    target_for_loss = target
                else:
                    target_for_loss = target_for_loss.to(device)

                loss = compute_loss(act_out, target_for_loss, criterion, device)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.info(
                    f"[Step {step}] Sending gradients from rank {local_rank} to rank {local_rank - 1}"
                )

                avg_loss = total_loss / (batch_idx + 1)
                wandb.log(
                    {
                        "step": step,
                        f"losses/train_{local_rank}": avg_loss,
                        "epoch": epoch + 1,
                    }
                )
                logger.info(
                    f"[Step {step}] Training loss: {avg_loss:.4f}"
                )
                
                # Update progress bars
                batch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'step': step
                })
                epoch_pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

                # Save checkpoint at regular intervals
                if save_checkpoints and should_save_checkpoint(
                    step, epoch, checkpoint_steps, num_epochs * len(train_loader)
                ):
                    # Wrap model_layers in Sequential for proper state_dict saving
                    temp_model = torch.nn.Sequential(*model_layers)
                    checkpoint_manager.save_checkpoint(
                        model=temp_model,
                        optimizer=optimizer,
                        scheduler=None,  # MP doesn't use scheduler
                        step=step,
                        epoch=epoch,
                        loss=loss.item(),
                        metadata={
                            "train_loss": total_loss / (batch_idx + 1),
                            "worker_rank": local_rank,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                    )
                    logger.info(f"[Step {step}] Worker {local_rank} checkpoint saved")

                # Send input gradients to previous worker
                send_message(
                    sock,
                    (
                        "forward_gradients",
                        step,
                        {
                            "from_rank": local_rank,
                            "to_rank": local_rank - 1,
                            "gradients": act_in.grad.detach().cpu(),
                        },
                    ),
                )

                # Clean up activations cache after backward pass
                del act_in_cache[(step, local_rank)]
                del act_out_cache[(step, local_rank)]
                clear_gpu_cache(device)

            elif command == "forward_gradients":
                rank, recv_grads = payload["to_rank"], payload["gradients"]
                logger.info(f"[Step {step}] Received gradients for rank {rank}.")

                if rank == local_rank:
                    logger.info(
                        f"[Step {step}] Computing backward pass for rank {local_rank}"
                    )

                    # Restore activations from cache (has computation graph)
                    act_in = act_in_cache[(step, local_rank)]
                    act_out = act_out_cache[(step, local_rank)]

                    # Apply received gradients to activations
                    torch.autograd.backward(act_out, recv_grads.to(device))

                send_message(
                    sock,
                    (
                        "forward_gradients",
                        step,
                        {
                            "from_rank": local_rank,
                            "to_rank": local_rank - 1,
                            "gradients": act_in.grad.detach().cpu(),
                        },
                    ),
                )

                # Clean up activations after backward
                del act_in_cache[(step, local_rank)]
                del act_out_cache[(step, local_rank)]
                clear_gpu_cache(device)

            elif command == "down":
                logger.info("Received exit command from server. Shutting down.")
                break

            # Clear GPU memory after optimizer step
            clear_gpu_cache(device)

            # Log gradient norms if tracking enabled
            if track_gradients:
                for name, param in model_layers.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/worker_{local_rank}_layer_{name}": grad_norm,
                                "step": step,
                                "epoch": epoch + 1,
                            }
                        )

            # Log network metrics if tracking enabled
            if track_network_metrics and step % metrics_log_interval == 0:
                network_stats = get_network_metrics(reset=True)
                if network_stats:
                    wandb.log(
                        {
                            f"network/worker_{local_rank}_send_bandwidth_mbps": network_stats.get(
                                "send_bandwidth_mbps", 0
                            ),
                            f"network/worker_{local_rank}_recv_bandwidth_mbps": network_stats.get(
                                "recv_bandwidth_mbps", 0
                            ),
                            f"network/worker_{local_rank}_avg_send_latency_ms": network_stats.get(
                                "avg_send_latency_ms", 0
                            ),
                            f"network/worker_{local_rank}_avg_recv_latency_ms": network_stats.get(
                                "avg_recv_latency_ms", 0
                            ),
                            f"network/worker_{local_rank}_avg_buffer_size_kb": network_stats.get(
                                "avg_buffer_size_kb", 0
                            ),
                            f"network/worker_{local_rank}_max_buffer_size_kb": network_stats.get(
                                "max_buffer_size_kb", 0
                            ),
                            "step": step,
                            "epoch": epoch + 1,
                        }
                    )
                    logger.info(
                        f"[Worker {local_rank} Step {step}] Network: Send={network_stats.get('send_bandwidth_mbps', 0):.2f}Mbps, "
                        f"Recv={network_stats.get('recv_bandwidth_mbps', 0):.2f}Mbps"
                    )

        # Close batch progress bar for this epoch
        batch_pbar.close()
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed.")

    # Close epoch progress bar
    epoch_pbar.close()
    
    sock.close()
    logger.info("Worker training completed and connection closed.")
