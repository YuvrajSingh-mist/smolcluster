import gc
import logging
import math
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchinfo
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from smolcluster.utils.checkpointing import CheckpointManager, should_save_checkpoint
from smolcluster.utils.common_utils import (
    get_gradients,
    get_network_metrics,
    receive_message,
    send_message,
    set_gradients,
    set_weights_by_layer,
)
from smolcluster.utils.layers import get_hfmodel_per_node, get_model_per_node
from smolcluster.utils.logging_utils import setup_cluster_logging

step = 0  # Global step counter to track training progress across threads


def reduce(
    grads_dict: dict[int, dict[str, torch.Tensor]], num_workers_connected: int
) -> dict[str, torch.Tensor]:
    """Average gradients from all workers."""
    grads_reduced = {}
    for worker_id in list(grads_dict):
        for name, worker_grads in grads_dict[worker_id].items():
            grads_reduced[name] = grads_reduced.get(name, 0.0) + (
                worker_grads / num_workers_connected
            )

    return grads_reduced


def compute_leader_activations(
    device: torch.device,
    model_layers: list[torch.nn.Module],
    data: torch.Tensor,
) -> torch.Tensor:
    """Compute gradients for worker rank 0 (leader node)."""

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
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    decoder_type_ppl: bool = False,
) -> tuple[float, float]:
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


def clear_gpu_cache(device: torch.device) -> None:
    """Clear GPU cache for both MPS and CUDA devices."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def compute_leader_gradients(
    device: torch.device,
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute gradients for leader/server node."""
    optimizer.zero_grad()
    model.train()
    data, target = data.to(device), target.to(device)
    output = model(data)
    B, T, C = output.shape
    output = output.view(B * T, C)
    target = target.view(B * T)
    loss = criterion(output, target)
    loss.backward()
    # Gradient clipping
    if config.get("gradient_clipping", {}).get("enabled", False):
        max_norm = config["gradient_clipping"].get("max_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    grads = get_gradients(model)
    return loss, grads


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


# Setup logging (will be replaced by setup_cluster_logging in run_modelparallelism_without_ps_worker)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = None  # Will be set in run_modelparallelism_pipeline_worker


def handle_worker(
    conn: socket.socket,
    addr: tuple[str, int],
    workers: dict,
    grads_received: dict,
    reduced_grads_received: dict,
    step_event: threading.Event,
    lock: threading.Lock,
    weights_received: dict,
) -> None:
    """Handle individual worker connections and gradient reception."""
    logger.info(f"Handling worker at {addr}")

    while True:
        try:
            message = receive_message(conn)

            # Handle connection closed or empty message
            if message is None:
                # logger.info(f"Worker {addr} closed connection")
                logger.warning(f"Received empty message from worker {addr}")
                break

            logger.debug(len(message))

            command, recv_step, rank, data = message

            if command == "all_gather":
                logger.info(
                    f"Received message '{command}' from worker {addr} (rank {rank}) for step {recv_step}"
                )
                logger.info(f"[Step {recv_step}] Storing gradients from worker {rank}")

                with lock:
                    grads_received[recv_step][rank] = data
                    logger.info(
                        f"[Step {recv_step}] Now have {len(grads_received[recv_step])} gradient sets"
                    )

                # reduced_grads = reduce(grads_received[recv_step], len(grads_received[recv_step]))
                step_event.set()

            elif command == "scatter_reduce":
                logger.info(
                    f"[Step {recv_step}] Received reduced gradients from worker {rank}"
                )
                
                # Buffer gradients by step - handle out-of-order delivery
                with lock:
                    reduced_grads_received[recv_step][rank] = data
                    logger.info(
                        f"[Step {recv_step}] Buffered reduced gradients from worker {rank}. "
                        f"Now have {len(reduced_grads_received[recv_step])} reduced gradient sets for this step"
                    )
                
                step_event.set()

            elif command == "broadcast_weights":
                logger.info(
                    f"Received broadcast weights message from worker {addr} (rank {rank})"
                )
                # Handle any broadcast messages if needed
                
                 # Buffer weights by step - handle out-of-order delivery
                with lock:
                    weights_received[recv_step][rank] = data
                    logger.info(
                        f"[Step {recv_step}] Buffered weights from worker {rank}. "
                        f"Now have {len(weights_received[recv_step])} weight sets for this step"
                    )
                
                step_event.set()
            elif command == "down":
                logger.info(f"Worker {addr} requested shutdown")
                break

        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            break
    logger.info(f"Worker {addr} disconnected")
    conn.close()


def accept_workers(
    sock: socket.socket,
    NUM_WORKERS: int,
    workers: dict,
    model_name: str,
    grads_received: dict,
    reduced_grads_received: dict,
    weights_received: dict,
    step_event: threading.Event,
    lock: threading.Lock,
) -> None:
    # Accept connections and wait for registration
    expected_peers = max(NUM_WORKERS - 1, 0)
    registered_workers = {}  # rank -> socket
    while len(registered_workers) < expected_peers:
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
                break

            command, worker_rank = message
            if command == "register":
                logger.info(f"Worker {worker_rank} registered from {client_address}")
                registered_workers[worker_rank] = client_socket
                workers[client_address] = client_socket
                threading.Thread(
                    target=handle_worker,
                    args=(
                        client_socket,
                        client_address,
                        workers,
                        grads_received,
                        reduced_grads_received,
                        step_event,
                        lock,
                        weights_received,
                    ),
                    daemon=True,
                ).start()
            else:
                logger.warning(f"Unexpected message from {client_address}: {command}")
                client_socket.close()
                break

        except Exception as e:
            logger.error(f"Error during registration from {client_address}: {e}")
            client_socket.close()
            break

    logger.info("All workers connected. Starting training...")


def run_fsdp_worker(
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
    Run FSDP (ZeRO Stage 1) training with optimizer state partitioning.
    Workers partition optimizer states while maintaining full model replica.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        hostname: Worker hostname
        device: Device to run on
        criterion: Loss criterion
    """
    global logger

    # Setup logger for this worker rank
    logger = logging.getLogger(f"[WORKER-{worker_rank}]")

    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="worker",
        rank=worker_rank,
        hostname=hostname,
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs"),
    )
    logger.info(f"🚀 FSDP Worker rank {worker_rank} starting up")

    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config["track_gradients"]
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    learning_rate = config["learning_rate"]
    grad_clip_norm = config.get("grad_clip_norm", 0.0)
    staleness_bound = cluster_config.get("staleness_bound", 0)  # 0 = strict sync, >0 = bounded async
    cluster_config["num_workers"]
    NUM_WORKERS = cluster_config["num_workers"]
    
    if staleness_bound > 0:
        logger.info(f"Bounded staleness enabled: staleness_bound={staleness_bound}")
    else:
        logger.info("Strict synchronous training enabled (staleness_bound=0)")

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
    full_checkpoint_dir = project_root / checkpoint_dir / "fsdp"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(full_checkpoint_dir),
        max_checkpoints=max_checkpoints_to_keep,
        save_optimizer=save_optimizer_state,
        rank=worker_rank,
        algorithm="fsdp",
    )

    # Network configuration
    buffer_size_mb = cluster_config.get("buffer_size", {}).get(hostname, 4)
    track_network_metrics = cluster_config.get("track_network_metrics", False)
    metrics_log_interval = cluster_config.get("metrics_log_interval", 50)
    logger.info(f"Network buffer size: {buffer_size_mb}MB")
    logger.info(f"Network metrics tracking: {track_network_metrics}")

    # Thread-safe data structures
    step_event = threading.Event()
    lock = threading.Lock()
    workers = {}
    outbound_worker_sockets = {}
    grads_received = defaultdict(dict)
    reduced_grads_received = defaultdict(dict)  # Buffer for scatter-reduce gradients
    weights_received = defaultdict(dict)  # Buffer for broadcast weights
    num_nodes = cluster_config["num_nodes"]
    num_layers = cluster_config["num_layers"]
    
    # Staleness tracking (only if staleness_bound > 0)
    staleness_stats = {
        "all_gather_step_diffs": [],  # Track step differences for all_gather gradients
        "scatter_reduce_step_diffs": [],  # Track step differences for scatter_reduce gradients
        "stale_gradient_count": 0,  # Count of gradients with step_diff > 0
        "max_step_diff": 0,  # Maximum step difference observed
        "broadcast_weights_step_diffs": [],  # Track step differences for broadcast weights
        "stale_weight_count": 0,  # Count of weights with step_diff > 0
    }

    # Gradient clipping
    if grad_clip_norm > 0.0:
        logger.info(f"Gradient clipping enabled: max_norm={grad_clip_norm}")
    else:
        logger.info("Gradient clipping disabled")

    # Load model
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")
    
    sharded_model, out_layers  = get_model_per_node(
        model=model,
        num_nodes=num_nodes,
        local_rank=worker_rank,
        total_layers=num_layers,
            
    )

    model_layers = out_layers
    
    # print(out_layers)
    logger.info(f"Worker rank {worker_rank} loaded {len(model_layers)} layers")
    # print(sharded_model)
    
    # ZeRO Stage 1: Create optimizer ONLY for this worker's owned layers
    # This partitions optimizer states (momentum, variance) across workers
    # Each worker only updates its owned parameters during optimizer.step()
    # Full model is synchronized via all-gather after each step
    # Collect all parameters from the dict of modules
    owned_params = [p for module in model_layers.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(owned_params, lr=learning_rate)
    logger.info(
        f"Created ZeRO Stage 1 optimizer for worker rank {worker_rank} with lr={learning_rate} "
        f"managing {len(owned_params)} parameter tensors"
    )

    
    # Log model summary
    model_summary = str(torchinfo.summary(model, verbose=0, device=device))
    logger.info("Model Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    # Load model layers for this worker rank
    config["num_layers"] = cluster_config["num_layers"]
    logger.info(f"Loading worker rank {worker_rank}'s share of model layers...")

    model = model.to(device)
    logger.info(
        f"Worker rank {worker_rank} loaded model layers and moved to device: {device}"
    )

    

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
            # Create a temporary model with only this worker's layers for loading

            metadata = checkpoint_manager.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer if save_optimizer_state else None,
                scheduler=scheduler,  # Load scheduler state if it exists
                device=device,
            )
            # Copy loaded state back to model_layers
            # for i, layer in enumerate(model_layers):
            model.load_state_dict(metadata["model_state_dict"])
            start_epoch = metadata.get("epoch", 0)
            start_step = metadata.get("step", 0)
            logger.info(f"Resumed from epoch={start_epoch}, step={start_step}")
        else:
            logger.warning("No checkpoint found to resume from, starting fresh")

    logger.info("Starting all-to-all topology setup for FSDP (ZeRO Stage 1).")

    # Get my worker configuration from allToAllTopology
    workers_list = cluster_config["allToAllTopology"]["workers"]["regular"]
    my_worker_config = next(w for w in workers_list if w["rank"] == worker_rank)
    my_port = my_worker_config["port"]

    # Step 1: Each worker binds to its configured port
    HOST_IP = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST_IP, my_port))
    sock.listen(NUM_WORKERS)  # Allow multiple connections for worker registration
    logger.info(f"Worker {worker_rank} listening on port {my_port}")

    # Step 2: Connect to next worker in linear topology (if not last worker)
    max_retries = 60
    retry_delay = 2

    for _ in range(NUM_WORKERS - 1):
        # Connect to next worker in the chain
        next_worker = next(w for w in workers_list if w["rank"] != worker_rank)
        next_ip = next_worker["ip"]
        next_port = next_worker["port"]
        del workers_list[
            workers_list.index(next_worker)
        ]  # Remove the next worker from the list to avoid duplicate connections

        logger.info(
            f"Worker {worker_rank} will connect to worker {worker_rank + 1} at {next_ip}:{next_port}"
        )
        time.sleep(worker_rank * 0.5)  # Stagger connections

        for attempt in range(max_retries):
            try:
                next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                next_sock.connect((next_ip, next_port))
                send_message(next_sock, ("register", worker_rank))

                logger.info(
                    f"Worker {worker_rank} connected to worker {worker_rank + 1} at {next_ip}:{next_port}"
                )
                outbound_worker_sockets[next_worker["rank"]] = (
                    next_sock  # This is important because this has the IP + PORT to which the nodes connected to it listen to which is what we have defined and not send stuff to the port we received through sock.accept()!
                )
                break
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection to worker {worker_rank + 1} refused (attempt {attempt + 1}/{max_retries} at IP: {next_ip}:{next_port}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to worker {worker_rank + 1} after {max_retries} attempts"
                    )
                    raise

    # Step 3: Accept connection from all workers
    accept_workers(
        sock,
        NUM_WORKERS,
        workers=workers,
        model_name="",
        grads_received=grads_received,
        reduced_grads_received=reduced_grads_received,
        weights_received=weights_received,
        step_event=step_event,
        lock=lock
    )

    logger.info(f"All workers connected. Starting training for {num_epochs} epochs.")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        val_loss = None #to make the ckpt manager happy at the end of the epoch when it tries to save the checkpoint and log the val_loss in the metadata
        
        model.train()

        global step  # Declare step as global to modify the global step counter

        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Create batch progress bar for this epoch (only for rank 0)
        batch_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            ncols=120,
            disable=(worker_rank != 0),
        )

        for batch_idx, (data, target) in batch_pbar:
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

            activations = None

            # Each worker computes its own gradients on its local data
            logger.info(
                f"[Step {step}/{num_epochs * len(train_loader)}] Worker rank {worker_rank} computing local gradients"
            )

            local_loss, local_grads = compute_leader_gradients(
                device, model, data, target, criterion, optimizer, config
            )
            with lock:
                grads_received[step][worker_rank] = local_grads

            total_loss += local_loss.item()

            # Clear GPU cache
            clear_gpu_cache(device)

            logger.info(
                f"[Step {step} / {num_epochs * len(train_loader)}] Worker {worker_rank} loss: {local_loss.item():.4f}"
            )
            train_ppl = math.exp(local_loss.item())

            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    f"losses/worker_{worker_rank}_step": local_loss.item(),
                    f"losses/worker_{worker_rank}_total_train": total_loss
                    / (batch_idx + 1),
                    f"ppl/worker_{worker_rank}_train": train_ppl,
                }
            )

            # All-gather: send local gradients to all peers via outbound connections
            logger.info(
                "Performing all-gather: broadcasting local gradients to all peers"
            )

            for peer_rank, peer_socket in outbound_worker_sockets.items():
                send_message(
                    peer_socket,
                    (
                        "all_gather",
                        step,
                        worker_rank,
                        local_grads,
                    ),
                )
                logger.info(
                    f"[Step {step}] Worker {worker_rank} sent gradients to worker {peer_rank}"
                )
            # Wait for all workers (all gather) to send their gradients for this step
            # Check for staleness violations and clean up stale gradients (only if staleness_bound > 0)
            while True:
                with lock:
                    # Only check staleness if bounded async is enabled
                    if staleness_bound > 0:
                        for recv_step in list(grads_received.keys()):
                            step_diff = abs(recv_step - step)
                            
                            # Track staleness statistics
                            staleness_stats["all_gather_step_diffs"].append(step_diff)
                            staleness_stats["max_step_diff"] = max(
                                staleness_stats["max_step_diff"], step_diff
                            )
                            if step_diff > 0:
                                staleness_stats["stale_gradient_count"] += 1
                            
                            if step_diff > staleness_bound:
                                logger.error(
                                    f"[Step {step}] STALENESS VIOLATION: Received gradient from step {recv_step} "
                                    f"(diff={step_diff} > bound={staleness_bound}). Training stopped."
                                )
                                raise RuntimeError(
                                    f"Staleness bound violated: step difference {step_diff} exceeds bound {staleness_bound}"
                                )
                    
                    curr_workers_len = len(grads_received[step])

                logger.info(
                    f"Worker {worker_rank} - Epoch {epoch + 1}/{num_epochs}, Step {step}/{num_epochs * len(train_loader)}: Received gradients from {curr_workers_len}/{NUM_WORKERS} workers."
                )
                if curr_workers_len < NUM_WORKERS:
                    logger.info(f"Waiting for more gradients for step {step}...")
                    step_event.wait(timeout=1.0)
                    step_event.clear()
                else:
                    break

            # Average gradients and update model
            if len(grads_received[step]) != 0:
                logger.info(
                    f"[Step {step}  / {num_epochs * len(train_loader)}] Averaging gradients from {len(grads_received[step])} participants"
                )

                # Reduce the grads
                grads_reduced = reduce(grads_received[step], len(grads_received[step]))

                logger.info(
                    f"[Step {step}] Worker {worker_rank} averaged gradients successfully"
                )

                # Scatter-reduce: broadcast averaged gradients to all peers via outbound connections
                for peer_rank, peer_socket in outbound_worker_sockets.items():
                    send_message(
                        peer_socket,
                        (
                            "scatter_reduce",
                            step,
                            worker_rank,
                            grads_reduced,
                        ),
                    )
                    logger.info(
                        f"[Step {step}] Worker {worker_rank} sent averaged gradients to worker {peer_rank}"
                    )

                logger.info(
                    f"[Step {step}] Worker {worker_rank} scatter-reduce complete"
                )

                # Wait for reduced gradients from all peers (for learning - shows full all-reduce flow)
                logger.info(f"[Step {step}] Worker {worker_rank} waiting for reduced gradients from peers...")
                while True:
                    with lock:
                        # Only check staleness if bounded async is enabled
                        if staleness_bound > 0:
                            for recv_step in list(reduced_grads_received.keys()):
                                step_diff = abs(recv_step - step)
                                
                                # Track staleness statistics
                                staleness_stats["scatter_reduce_step_diffs"].append(step_diff)
                                staleness_stats["max_step_diff"] = max(
                                    staleness_stats["max_step_diff"], step_diff
                                )
                                if step_diff > 0:
                                    staleness_stats["stale_gradient_count"] += 1
                                
                                if step_diff > staleness_bound:
                                    logger.error(
                                        f"[Step {step}] STALENESS VIOLATION in scatter-reduce: Received gradient from step {recv_step} "
                                        f"(diff={step_diff} > bound={staleness_bound}). Training stopped."
                                    )
                                    raise RuntimeError(
                                        f"Staleness bound violated in scatter-reduce: step difference {step_diff} exceeds bound {staleness_bound}"
                                    )
                        
                        curr_reduced_len = len(reduced_grads_received[step])
                    
                    logger.info(
                        f"[Step {step}] Worker {worker_rank} received {curr_reduced_len}/{NUM_WORKERS - 1} reduced gradient sets"
                    )
                    
                    if curr_reduced_len >= NUM_WORKERS - 1:
                        logger.info(f"[Step {step}] Worker {worker_rank} received all reduced gradients")
                        break
                    
                    step_event.wait()
                    step_event.clear()

                logger.info(
                    f"[Step {step}] Worker {worker_rank} applying averaged gradients to local model"
                )

                set_gradients(grads_reduced, model)
                optimizer.step()

                logger.info(
                    f"[Step {step}] Worker {worker_rank} model updated with averaged gradients"
                )
                
                
                logger.info(f"[Step {step}] Starting broadcasting weights to all peers after local update")
                
                # broadcast  weights to all peers via outbound connections
                for peer_rank, peer_socket in outbound_worker_sockets.items():
                    send_message(
                        peer_socket,
                        (
                            "broadcast_weights",
                            step,
                            worker_rank,
                            model.state_dict(),
                        ),
                    )
                    logger.info(
                        f"[Step {step}] Worker {worker_rank} sent averaged weights to worker {peer_rank}"
                    )

                logger.info(
                    f"[Step {step}] Worker {worker_rank} broadcast weights complete"
                )

              
                logger.info(f"[Step {step}] Worker {worker_rank} waiting for updated weights from all workers")
                
                # Wait for reduced gradients from all peers (for learning - shows full all-reduce flow)
                logger.info(f"[Step {step}] Worker {worker_rank} waiting for weights from peers...")
                while True:
                    with lock:
                        # Only check staleness if bounded async is enabled
                        if staleness_bound > 0:
                            for recv_step in list(weights_received.keys()):
                                step_diff = abs(recv_step - step)
                                
                                # Track staleness statistics
                                staleness_stats["broadcast_weights_step_diffs"].append(step_diff)
                                staleness_stats["max_step_diff"] = max(
                                    staleness_stats["max_step_diff"], step_diff
                                )
                                if step_diff > 0:
                                    staleness_stats["stale_weight_count"] += 1
                                
                                if step_diff > staleness_bound:
                                    logger.error(
                                        f"[Step {step}] STALENESS VIOLATION in broadcast weights: Received weight from step {recv_step} "
                                        f"(diff={step_diff} > bound={staleness_bound}). Training stopped."
                                    )
                                    raise RuntimeError(
                                        f"Staleness bound violated in broadcast weights: step difference {step_diff} exceeds bound {staleness_bound}"
                                    )
                        
                        curr_reduced_len = len(weights_received[step])
                    
                    logger.info(
                        f"[Step {step}] Worker {worker_rank} received {curr_reduced_len}/{NUM_WORKERS - 1} reduced weight sets"
                    )
                    
                    if curr_reduced_len >= NUM_WORKERS - 1:
                        logger.info(f"[Step {step}] Worker {worker_rank} received all reduced weights")
                        break
                    
                    step_event.wait()
                    step_event.clear()

                # Merge weights from all workers: take each worker's owned parameters
                set_weights_by_layer(
                    weights_received[step],
                    model,
                    worker_rank,
                    num_nodes,
                    num_layers,
                )
                logger.info(
                    f"[Step {step}] Worker {worker_rank} merged weights from all workers"
                )
                
                # Cleanup old weights beyond staleness window
                weights_received.pop(step, None)
                
                # Cleanup old gradients beyond staleness window to free memory (only if staleness_bound > 0)
                if staleness_bound > 0:
                    with lock:
                        for old_step in list(reduced_grads_received.keys()):
                            if old_step < step - staleness_bound:
                                logger.info(f"[Step {step}] Cleaning up stale reduced gradients from step {old_step}")
                                reduced_grads_received.pop(old_step, None)
                        
                        for old_step in list(grads_received.keys()):
                            if old_step < step - staleness_bound:
                                logger.info(f"[Step {step}] Cleaning up stale all-gather gradients from step {old_step}")
                                grads_received.pop(old_step, None)
                
                # Cleanup current step gradients
                reduced_grads_received.pop(step, None)
                grads_received.pop(step, None)
                del grads_reduced, local_grads
                gc.collect()
            else:
                logger.warning(
                    f"[Step {step}] Worker {worker_rank}: No gradients received. Skipping grad update."
                )
                del local_grads

            # Apply gradient clipping
            if grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm
                )
                if step % 100 == 0:  # Log occasionally to avoid spam
                    logger.info(
                        f"[Step {step}] Gradient norm before clipping: {grad_norm:.4f}"
                    )

            # Step the scheduler after optimizer
            if scheduler is not None:
                scheduler.step()

            # Clear GPU memory after optimizer step
            clear_gpu_cache(device)

            # Update batch progress bar with current metrics
            batch_pbar.set_postfix({"lr": f"{current_lr:.2e}", "step": step})

            # Log training metrics
            wandb_metrics = {
                "step": step,
                "epoch": epoch + 1,
                "lr": current_lr,
                "batch_size": batch_size,
            }
            
            # Log staleness metrics if bounded async is enabled and we have data
            if staleness_bound > 0 and step % metrics_log_interval == 0:
                with lock:
                    if staleness_stats["all_gather_step_diffs"]:
                        avg_all_gather_diff = sum(staleness_stats["all_gather_step_diffs"]) / len(
                            staleness_stats["all_gather_step_diffs"]
                        )
                        wandb_metrics[f"staleness/worker_{worker_rank}_all_gather_avg_step_diff"] = avg_all_gather_diff
                    
                    if staleness_stats["scatter_reduce_step_diffs"]:
                        avg_scatter_diff = sum(staleness_stats["scatter_reduce_step_diffs"]) / len(
                            staleness_stats["scatter_reduce_step_diffs"]
                        )
                        wandb_metrics[f"staleness/worker_{worker_rank}_scatter_reduce_avg_step_diff"] = avg_scatter_diff
                    
                    if staleness_stats["broadcast_weights_step_diffs"]:
                        avg_weights_diff = sum(staleness_stats["broadcast_weights_step_diffs"]) / len(
                            staleness_stats["broadcast_weights_step_diffs"]
                        )
                        wandb_metrics[f"staleness/worker_{worker_rank}_broadcast_weights_avg_step_diff"] = avg_weights_diff
                    
                    wandb_metrics[f"staleness/worker_{worker_rank}_max_step_diff"] = staleness_stats["max_step_diff"]
                    wandb_metrics[f"staleness/worker_{worker_rank}_stale_gradient_count"] = staleness_stats["stale_gradient_count"]
                    wandb_metrics[f"staleness/worker_{worker_rank}_stale_weight_count"] = staleness_stats["stale_weight_count"]
                    wandb_metrics[f"staleness/worker_{worker_rank}_staleness_bound"] = staleness_bound
                    
                    # Reset stats for next interval
                    staleness_stats["all_gather_step_diffs"] = []
                    staleness_stats["scatter_reduce_step_diffs"] = []
                    staleness_stats["broadcast_weights_step_diffs"] = []
                    staleness_stats["stale_gradient_count"] = 0
                    staleness_stats["stale_weight_count"] = 0
                    # Keep max_step_diff as cumulative max
                    
                    logger.info(
                        f"[Step {step}] Staleness stats - Max diff: {staleness_stats['max_step_diff']}, "
                        f"Stale grads in interval: {staleness_stats['stale_gradient_count']}, "
                        f"Stale weights in interval: {staleness_stats['stale_weight_count']}"
                    )
            
            wandb.log(wandb_metrics)

            # Log gradient norms if tracking enabled
            if track_gradients:
                for name, param in model.named_parameters():
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
                            f"network/worker_{worker_rank}_send_bandwidth_mbps": network_stats.get(
                                "send_bandwidth_mbps", 0
                            ),
                            f"network/worker_{worker_rank}_recv_bandwidth_mbps": network_stats.get(
                                "recv_bandwidth_mbps", 0
                            ),
                            f"network/worker_{worker_rank}_avg_send_latency_ms": network_stats.get(
                                "avg_send_latency_ms", 0
                            ),
                            f"network/worker_{worker_rank}_avg_recv_latency_ms": network_stats.get(
                                "avg_recv_latency_ms", 0
                            ),
                            f"network/worker_{worker_rank}_avg_buffer_size_kb": network_stats.get(
                                "avg_buffer_size_kb", 0
                            ),
                            f"network/worker_{worker_rank}_max_buffer_size_kb": network_stats.get(
                                "max_buffer_size_kb", 0
                            ),
                            "step": step,
                            "epoch": epoch + 1,
                        }
                    )
                    logger.info(
                        f"[Worker {worker_rank} Step {step}] Network: Send={network_stats.get('send_bandwidth_mbps', 0):.2f}Mbps, "
                        f"Recv={network_stats.get('recv_bandwidth_mbps', 0):.2f}Mbps"
                    )

            # Evaluation
            if step % eval_steps == 0:
                val_loss, val_ppl = evaluate(
                    device,
                    model,
                    val_loader,
                    criterion,
                    decoder_type_ppl=decoder_type_ppl,
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

                    # Update progress bar
                    batch_pbar.set_postfix(
                        {"val_loss": f"{val_loss:.4f}", "ppl": f"{val_ppl:.2f}"}
                    )
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

                    # Update progress bar
                    batch_pbar.set_postfix({"val_loss": f"{val_loss:.4f}"})

            # Save checkpoint at regular intervals
            if save_checkpoints and should_save_checkpoint(
                step, epoch, checkpoint_steps, num_epochs * len(train_loader)
            ):
                # Wrap model_layers in Sequential for proper state_dict saving

                checkpoint_manager.save_checkpoint(
                    model=model,
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

        # Close batch progress bar for this epoch
        batch_pbar.close()

    for worker_addr, worker_socket in workers.items():
        send_message(worker_socket, ("down", step, worker_rank, None))
        logger.info(f"Sent shutdown signal to worker at {worker_addr}")

    for peer_rank, peer_socket in outbound_worker_sockets.items():
        try:
            peer_socket.close()
            logger.info(f"Closed outbound socket to worker rank {peer_rank}")
        except Exception as e:
            logger.warning(
                f"Failed to close outbound socket to worker rank {peer_rank}: {e}"
            )

    logger.info("Training completed successfully!")
