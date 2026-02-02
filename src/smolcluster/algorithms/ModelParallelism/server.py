import gc
import heapq
import logging
import math
import socket
from pathlib import Path
from typing import Optional

import torch
import torchinfo
import wandb
from torch.utils.data import DataLoader

from smolcluster.utils.checkpointing import CheckpointManager, should_save_checkpoint
from smolcluster.utils.common_utils import (
    get_network_metrics,
    receive_message,
    send_message,
)
from smolcluster.utils.layers import get_model_per_node
from smolcluster.utils.logging_utils import setup_cluster_logging


def compute_leader_activations(
    device: torch.device,
    model_layers: list[torch.nn.Module],
    data: torch.Tensor,
) -> torch.Tensor:
    """Compute gradients for leader/server node."""

    data = data.to(device)
    out = None
    # with torch.no_grad():

    out = model_layers[0](data)

    pos_ids = torch.arange(out.shape[1], dtype=torch.long, device=device)
    out = out + model_layers[1](pos_ids)

    for layer in model_layers[2:]:
        output = layer(out)
        out = output[0] if isinstance(output, tuple) else output

    return out


def evaluate(
    device: torch.device,
    model_layers: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    worker_queue: list,
    decoder_type_ppl: bool = False,
) -> tuple[float, Optional[float]]:
    """Evaluate model on validation set using distributed model layers.

    Activations flow through server layers -> worker 1 -> worker 2 -> back to server
    for loss computation, matching the training forward pass.
    """
    model_layers.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            # Server computes its layer activations
            activations = compute_leader_activations(device, model_layers, data)

            # Forward through workers in rank order
            for rank, worker_socket, _addr in sorted(worker_queue):
                send_message(
                    worker_socket,
                    (
                        "evaluate_forward",
                        0,  # eval_step placeholder
                        {
                            "activations": activations.detach().cpu(),
                        },
                    ),
                )

                # Receive activations from this worker
                message = receive_message(worker_socket)
                command, _, payload = message

                if command == "eval_activations":
                    activations = payload["activations"].to(device)
                else:
                    logger.error(
                        f"Unexpected eval command from worker {rank}: {command}"
                    )
                    break

            # Compute loss using final activations from last worker
            B, T, C = activations.shape
            output = activations.view(B * T, C)
            target_flat = target.view(B * T)
            loss = criterion(output, target_flat)
            total_val_loss += loss.item()

    avg_loss = total_val_loss / len(val_loader)
    ppl = math.exp(avg_loss) if decoder_type_ppl else None

    model_layers.train()
    return avg_loss, ppl


def clear_gpu_cache(device: torch.device) -> None:
    """Clear GPU cache for both MPS and CUDA devices."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def compute_train_loss(
    final_activations: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Compute training loss from final activations and targets.

    Args:
        final_activations: Output activations from the last worker [B, T, C]
        target: Target labels [B, T]
        criterion: Loss function
        device: Device to compute on

    Returns:
        Training loss as a float
    """
    final_activations.to(device)
    target_device = target.to(device)
    B, T, C = final_activations.shape
    output = final_activations.view(B * T, C)
    target_flat = target_device.view(B * T)
    train_loss = criterion(output, target_flat).item()
    return train_loss


def get_lr_schedule(warmup_iters, max_iters, learning_rate, min_lr):
    """Create learning rate schedule with linear warmup and cosine decay.

    Args:
        warmup_iters: Number of warmup iterations
        max_iters: Total training iterations
        learning_rate: Peak learning rate (after warmup)
        min_lr: Minimum learning rate (end of decay)

    Returns:
        Function that takes step and returns learning rate
    """

    def get_lr(step):
        # Linear warmup
        if step < warmup_iters:
            return learning_rate * (step + 1) / warmup_iters

        # Cosine decay after warmup
        if step > max_iters:
            return min_lr

        decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    return get_lr


# Setup logging (will be replaced by setup_cluster_logging in run_modelparallelism_server)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[LEADER]")


def run_modelparallelism_server(
    model,
    train_loader,
    val_loader,
    config,
    cluster_config,
    hostname,
    device,
    criterion,
    resume_checkpoint_path=None,
):
    """
    Run Synchronous Parameter Server training.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        hostname: Server hostname
        device: Device to run on
        criterion: Loss criterion
    """
    global logger

    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="server",
        rank=None,
        hostname=hostname,
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs"),
    )
    logger.info("🚀 ModelParallelism Server starting up")

    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config["track_gradients"]
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    learning_rate = config["learning_rate"]
    grad_clip_norm = config.get("grad_clip_norm", 0.0)
    cluster_config["num_workers"]
    model_name = cluster_config["model_name"]
    recv_grads = None
    RANK = 0
    NUM_WORKERS = cluster_config["num_workers"]

    num_nodes = cluster_config["num_nodes"]

    # Checkpoint configuration
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
    project_root = Path(__file__).parent.parent.parent.parent.parent
    full_checkpoint_dir = project_root / checkpoint_dir / "mp"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(full_checkpoint_dir),
        max_checkpoints=max_checkpoints_to_keep,
        save_optimizer=save_optimizer_state,
        rank=RANK,
        algorithm="mp",
    )

    # Network configuration
    buffer_size_mb = cluster_config.get("buffer_size", {}).get(hostname, 4)
    track_network_metrics = cluster_config.get("track_network_metrics", False)
    metrics_log_interval = cluster_config.get("metrics_log_interval", 50)
    logger.info(f"Network buffer size: {buffer_size_mb}MB")
    logger.info(f"Network metrics tracking: {track_network_metrics}")

    # Gradient clipping
    if grad_clip_norm > 0.0:
        logger.info(f"Gradient clipping enabled: max_norm={grad_clip_norm}")
    else:
        logger.info("Gradient clipping disabled")

    # Create socket
    HOST_IP = "0.0.0.0"
    port_config = cluster_config["port"]
    if isinstance(port_config, dict):
        server_hostname = cluster_config["server"]
        PORT = port_config.get(server_hostname, port_config.get("default", 65432))
    else:
        PORT = port_config

    logger.info(f"Server will bind to IP: {HOST_IP}, Port: {PORT}")
    workers = {}

    # Load tokenizer
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")

    # Log model summary
    model_summary = str(torchinfo.summary(model, verbose=0, device=device))
    logger.info("Model Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    # Load model layers for server (rank 0)
    num_layers = config["num_layers"]
    logger.info(f"Loading server's share of model layers (rank {RANK})...")

    model_layers, out_layers = get_model_per_node(
        model, num_nodes=num_nodes, local_rank=RANK, total_layers=num_layers
    )

    model_layers = model_layers.to(device)
    logger.info(f"Server loaded {len(model_layers)} layers")

    # Create optimizer for server's layers only
    optimizer = torch.optim.AdamW(model_layers.parameters(), lr=learning_rate)
    logger.info(f"Created optimizer for server with lr={learning_rate}")

    # Learning rate scheduler setup (after optimizer creation)
    use_lr_scheduler = config.get("use_lr_scheduler", False)
    total_steps = num_epochs * len(train_loader)
    scheduler = None
    if use_lr_scheduler:
        warmup_iters = config["warmup_iters"]
        min_lr = config["min_lr"]
        get_lr_fn = get_lr_schedule(warmup_iters, total_steps, learning_rate, min_lr)
        # Wrap custom LR function in LambdaLR scheduler for proper state saving
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: get_lr_fn(step) / learning_rate
        )
        logger.info(
            f"LR scheduler enabled: warmup={warmup_iters}, max_iters={total_steps}, peak_lr={learning_rate}, min_lr={min_lr}"
        )
    else:
        logger.info(f"LR scheduler disabled, using constant lr={learning_rate}")

    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if save_checkpoints and resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()
        else:
            checkpoint_path = resume_from_checkpoint

        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            # Create a temporary model with only server layers for loading
            temp_model = torch.nn.Sequential(*model_layers)
            metadata = checkpoint_manager.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=temp_model,
                optimizer=optimizer if save_optimizer_state else None,
                scheduler=scheduler,  # Load scheduler state if it exists
                device=device,
            )
            # Copy loaded state back to model_layers
            for i, layer in enumerate(model_layers):
                layer.load_state_dict(temp_model[i].state_dict())
            start_epoch = metadata.get("epoch", 0)
            start_step = metadata.get("step", 0)
            logger.info(f"Resumed from epoch={start_epoch}, step={start_step}")
        else:
            logger.warning("No checkpoint found to resume from, starting fresh")

    # Create and bind socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the host and port
    sock.bind((HOST_IP, PORT))
    sock.listen(5)
    logger.info(f"Server listening on {HOST_IP}:{PORT}")

    # Accept connections and wait for registration
    # Use priority queue to maintain workers sorted by rank
    worker_queue = []  # Priority queue: [(rank, socket, address)]
    registered_workers = {}  # rank -> socket (for quick lookup)

    # Accept all connections (workers + API client)
    while len(registered_workers) < NUM_WORKERS:
        conn, address = sock.accept()
        logger.info(f"Accepted connection from {address}")

        # Wait for registration message
        try:
            message = receive_message(conn)
            if message is None:
                logger.warning(f"Connection from {address} closed before registration")
                conn.close()
                continue

            command, rank = message
            if command == "register":
                logger.info(f"Worker rank {rank} registered from {address}")
                registered_workers[rank] = conn
                workers[address] = conn
                # Add to priority queue sorted by rank
                heapq.heappush(worker_queue, (rank, conn, address))
                logger.info(
                    f"Worker rank {rank} added to priority queue (queue size: {len(worker_queue)})"
                )

            else:
                logger.warning(f"Unexpected message from {address}: {command}")
                conn.close()
        except Exception as e:
            logger.error(f"Error during registration from {address}: {e}")
            conn.close()
            continue

    logger.info(f"All workers connected. Starting training on {model_name}...")
    logger.info(
        f"Worker priority queue (by rank): {[(rank, addr) for rank, _, addr in worker_queue]}"
    )

    # Send start_training to workers in rank order
    for rank, worker_socket, addr in sorted(worker_queue):
        logger.info(f"Sending start_training to worker rank {rank} at {addr}")
        send_message(worker_socket, "start_training", buffer_size_mb=buffer_size_mb)

    logger.info(f"Starting training for {model_name}.")

    # Initialize activation caches
    # act_in_cache = {}
    act_out_cache = {}

    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(start_epoch, num_epochs):
        model_layers.train()

        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx

            # Skip batches if resuming mid-epoch
            if step < start_step:
                continue

            data = data.to(device)
            target = target.to(device)
            # Update learning rate if scheduler enabled
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = learning_rate

            logger.info(
                f"[Step {step}  / {num_epochs * len(train_loader)}] Server computing leader activations"
            )
            leader_activations = compute_leader_activations(device, model_layers, data)
            leader_activations.requires_grad_(True)
            # act_in = None
            act_out = None
            activations = None

            # Clear GPU cache before caching activations
            clear_gpu_cache(device)

            # Cache server's activations WITH computation graph (no detach!)
            # act_in_cache[(step, RANK)] = data
            act_out_cache[(step, RANK)] = leader_activations
            activations = leader_activations

            # logger.info("Finsihed generating activations for local_rank 0")

            # activations = leader_activations
            # Send generation request to all workers in rank order (1, 2, ...)
            for rank, worker_socket, _addr in sorted(worker_queue):
                logger.info(f"[Step {step}] Sending activations to worker rank {rank}")
                send_message(
                    worker_socket,
                    (
                        "generate_activations_train",
                        step,
                        {
                            "activations": activations.detach().cpu(),
                            "targets": target.detach().cpu()
                            if rank == NUM_WORKERS
                            else None,
                        },
                    ),
                )

                message = receive_message(worker_socket)

                command, recv_step, payload = message

                assert recv_step == step, (
                    f"Step mismatch: expected {step}, got {recv_step}"
                )

                if command == "forward_activations":
                    activations = payload["activations"].to(device)
                    from_rank = payload["from_rank"]
                    to_rank = payload["to_rank"]
                    logger.info(
                        f"[Step {step}] Received activations forwarded from worker {from_rank} to worker {to_rank}"
                    )

                    # Clear GPU cache after moving activations to device
                    clear_gpu_cache(device)

                    # Cache worker's output activations
                    act_out_cache[(step, from_rank)] = activations

                else:
                    logger.error(
                        f"Unexpected command from worker {rank}: {command}. Cannot continue."
                    )
                    break

            # # Compute training loss using final activations from last worker

            # train_loss = compute_train_loss(activations, target, criterion, device)

            # # Log training loss
            # logger.info(f"[Step {step}] Train Loss: {train_loss:.4f}")
            # wandb.log({
            #     "losses/train": train_loss,
            #     "step": step,
            #     "epoch": epoch + 1,
            # })

            # Clear GPU cache before backward phase
            clear_gpu_cache(device)

            for rank, worker_socket, _addr in sorted(worker_queue, reverse=True):
                if rank == NUM_WORKERS:
                    logger.info(
                        f"[Step {step}] Sending generate_gradients command to last worker rank {rank}"
                    )
                    send_message(
                        worker_socket,
                        (
                            "generate_gradients",
                            step,
                            {
                                "gradients": None,
                            },
                        ),
                    )

                # Receiving the last worker nodes activations
                message = receive_message(worker_socket)

                command, recv_step, payload = message

                assert recv_step == step, (
                    f"Step mismatch: expected {step}, got {recv_step}"
                )

                if command == "forward_gradients":
                    recv_grads = payload["gradients"]
                    to_rank = payload["to_rank"]
                    from_rank = payload["from_rank"]
                    logger.info(
                        f"[Step {step}] Received gradients forwarded to server from worker {from_rank} for {to_rank}"
                    )

                    if to_rank == RANK:
                        logger.info(f"[Step {step}] Computing backward pass for server")
                        # Restore server's activations from cache (has computation graph)
                        act_out = act_out_cache[(step, RANK)]
                        act_out = act_out.to(device)

                        optimizer.zero_grad()
                        # Backward - this updates model parameters
                        torch.autograd.backward(act_out, recv_grads.to(device))
                        optimizer.step()

                        # Clean up server activation cache
                        if (step, RANK) in act_out_cache:
                            del act_out_cache[(step, RANK)]

                        # Clear GPU cache after backward pass
                        clear_gpu_cache(device)

                    else:
                        target_socket = next(
                            (s for r, s, _ in worker_queue if r == to_rank), None
                        )
                        if target_socket:
                            logger.info(
                                f"[Step {step}] Forwarding gradients to worker rank {to_rank} from {from_rank} via current rank {rank}"
                            )
                            send_message(
                                target_socket,
                                (
                                    "forward_gradients",
                                    step,
                                    {"gradients": recv_grads, "to_rank": to_rank},
                                ),
                            )

            # Clean up any remaining cached activations from this step
            keys_to_delete = [key for key in act_out_cache.keys() if key[0] == step]
            for key in keys_to_delete:
                del act_out_cache[key]

            # Apply gradient clipping
            if grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_layers.parameters(), grad_clip_norm
                )
                if step % 100 == 0:  # Log occasionally to avoid spam
                    logger.info(
                        f"[Step {step}] Gradient norm before clipping: {grad_norm:.4f}"
                    )

            # Apply gradients
            optimizer.step()

            # Step the scheduler after optimizer
            if scheduler is not None:
                scheduler.step()

            # Clear GPU memory after optimizer step
            clear_gpu_cache(device)

            # Log training metrics
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "batch_size": batch_size,
                }
            )

            # Log gradient norms if tracking enabled
            if track_gradients:
                for name, param in model_layers.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/layer_{name}": grad_norm,
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
                            "network/send_bandwidth_mbps": network_stats.get(
                                "send_bandwidth_mbps", 0
                            ),
                            "network/recv_bandwidth_mbps": network_stats.get(
                                "recv_bandwidth_mbps", 0
                            ),
                            "network/avg_send_latency_ms": network_stats.get(
                                "avg_send_latency_ms", 0
                            ),
                            "network/avg_recv_latency_ms": network_stats.get(
                                "avg_recv_latency_ms", 0
                            ),
                            "network/avg_buffer_size_kb": network_stats.get(
                                "avg_buffer_size_kb", 0
                            ),
                            "network/max_buffer_size_kb": network_stats.get(
                                "max_buffer_size_kb", 0
                            ),
                            "network/total_send_mb": network_stats.get(
                                "total_send_mb", 0
                            ),
                            "network/total_recv_mb": network_stats.get(
                                "total_recv_mb", 0
                            ),
                            "step": step,
                            "epoch": epoch + 1,
                        }
                    )
                    logger.info(
                        f"[Step {step}] Network: Send={network_stats.get('send_bandwidth_mbps', 0):.2f}Mbps, "
                        f"Recv={network_stats.get('recv_bandwidth_mbps', 0):.2f}Mbps, "
                        f"Buffer={network_stats.get('avg_buffer_size_kb', 0):.2f}KB"
                    )

            # Evaluation
            if step % eval_steps == 0 and step != 0:
                val_loss, val_ppl = evaluate(
                    device,
                    model_layers,
                    val_loader,
                    criterion,
                    worker_queue,
                    decoder_type_ppl,
                )

                if decoder_type_ppl:
                    wandb.log(
                        {
                            "step": step,
                            "epoch": epoch + 1,
                            "losses/val": val_loss,
                            "ppl/val": val_ppl,
                        }
                    )
                    eval_msg = f"[Step {step}] Evaluation: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}"
                    logger.info(eval_msg)
                    print(eval_msg)
                else:
                    wandb.log(
                        {
                            "step": step,
                            "epoch": epoch + 1,
                            "losses/val": val_loss,
                        }
                    )
                    eval_msg = f"[Step {step}] Evaluation: Val Loss={val_loss:.4f}"
                    logger.info(eval_msg)
                    print(eval_msg)

            # Save checkpoint at regular intervals
            if save_checkpoints and should_save_checkpoint(
                step, epoch, checkpoint_steps, num_epochs * len(train_loader)
            ):
                # Wrap model_layers in Sequential for proper state_dict saving
                temp_model = torch.nn.Sequential(*model_layers)
                checkpoint_manager.save_checkpoint(
                    model=temp_model,
                    optimizer=optimizer,
                    scheduler=scheduler,  # Save scheduler state
                    step=step,
                    epoch=epoch,
                    loss=None,
                    metadata={
                        "val_loss": val_loss
                        if step % eval_steps == 0 and step != 0
                        else None,
                        "learning_rate": current_lr,
                    },
                )
                logger.info(f"[Step {step}] Checkpoint saved")

                # Clean up activations tensor
                if activations is not None:
                    del activations

            gc.collect()
            activations = None

    for _rank, worker_socket, _addr in sorted(worker_queue):
        send_message(worker_socket, "down")

    logger.info("Training completed successfully!")

    sock.close()
