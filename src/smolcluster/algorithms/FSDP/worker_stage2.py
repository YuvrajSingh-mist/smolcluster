import gc
import logging
import math
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from smolcluster.models.gpt import BaseTransformer
from smolcluster.utils.checkpointing import CheckpointManager, should_save_checkpoint
from smolcluster.utils.common_utils import (
    clear_skeleton_gradients,
    extract_owned_gradients,
    forward_through_shard,
    get_ordered_shard_layer_names,
    get_network_metrics,
    load_params_into_skeleton,
    receive_message,
    send_message,
    unload_params_from_skeleton,
)
from smolcluster.utils.layers import get_model_per_node
from smolcluster.utils.logging_utils import setup_cluster_logging

step = 0  # Global step counter to track training progress across threads

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


def compute_activations_sequential(
    device: torch.device,
    model_skeleton: torch.nn.Module,
    all_worker_params: dict,  # Dict of {rank: params_dict}
    data: torch.Tensor,
    num_workers: int,
    log_layers: bool = False,
) -> torch.Tensor:
    """
    FSDP Stage 2: Sequential forward pass through ALL worker shards.
    
    Like Model Parallelism, process shards sequentially:
    1. Load worker N's params into skeleton
    2. Forward through worker N's layers
    3. Cache activations (preserves computation graph)
    4. Unload worker N's params
    5. Repeat for worker N+1...
    
    Memory efficient: Only 1 shard loaded at a time!
    Returns final activations WITH computation graph for backward.
    """
    model_skeleton.train()
    activations = data.to(device)

    # Process each worker's shard sequentially
    for rank in sorted(all_worker_params.keys()):
        worker_params = all_worker_params[rank]
        
        # Log which layers this worker has (only on first call)
        if log_layers:
            sorted_layers = get_ordered_shard_layer_names(model_skeleton, rank, num_workers)
            logger.info(f"[LAYER DISTRIBUTION] Worker {rank} owns layers: {sorted_layers}")

        # The skeleton module has already been moved to `device` once in
        # `run_fsdp_worker()`. Here we only materialize the current worker's shard
        # tensors into that GPU-resident structure, execute its slice of the model,
        # and then free the storage again before loading the next shard.
        load_params_into_skeleton(model_skeleton, worker_params, device)
        
        # Run only the layers owned by this shard. The output tensor keeps the
        # autograd graph pointing back to the currently loaded parameter objects.
        activations = forward_through_shard(model_skeleton, activations, rank, num_workers, device)

        if log_layers:
            logger.info(
                f"[SHARD OUTPUT] Worker {rank} produced activation shape {tuple(activations.shape)}"
            )
        
        # Drop parameter storage for this shard to keep peak memory low.
        # The activation tensor still keeps the graph metadata needed for backward.
        unload_params_from_skeleton(model_skeleton)
        
        gc.collect()
        clear_gpu_cache(device)

    return activations


def compute_worker_gradients(
    device: torch.device,
    model_skeleton: torch.nn.Module,
    own_params: dict,
    all_worker_params: dict,
    final_output: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    worker_rank: int,
    num_workers: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute gradients for owned parameters using cached computation graph.
    
    CRITICAL: Must reload ALL worker parameters before backward.
    Unlike Model Parallelism (where each worker's forward uses only its params),
    our sequential forward builds a computation graph referencing ALL parameters.
    During backward, PyTorch needs all parameter tensors to be valid (not empty).
    
    Memory profile:
    - Forward: 1 shard at a time (~57MB peak)
    - Backward: All shards temporarily loaded (~114MB)
    - After gradient extraction: Back to empty skeleton
    
    Args:
        own_params: Dict of {layer_name: param_tensor} for our owned shard
        all_worker_params: Dict of {rank: params_dict} for all workers
        final_output: Model output with computation graph attached
        target: Target labels for loss computation
    
    Returns:
        (loss, grads): Loss value and gradients dict for owned parameters
    """
    optimizer.zero_grad()
    
    # CRITICAL: reload every shard into the already GPU-resident skeleton before
    # backward. The sequential forward built one graph spanning all shards, so when
    # autograd walks that graph it must find valid parameter storage for every layer
    # it traverses, not empty placeholders.
    for rank in sorted(all_worker_params.keys()):
        worker_params = all_worker_params[rank]
        load_params_into_skeleton(model_skeleton, worker_params, device)
    
    # Compute loss (final_output preserves computation graph from forward pass)
    target = target.to(device)
    
    B, T, C = final_output.shape
    logits = final_output.view(B * T, C)
    targets = target.view(B * T)
    loss = criterion(logits.view(-1, C), targets)
    
    # Backward through cached computation graph (all params now loaded)
    # PyTorch traces through all layers, computes gradients for all parameters
    loss.backward()
    
    # Apply gradient clipping to owned parameters only
    if config.get("gradient_clipping", {}).get("enabled", False):
        max_norm = config["gradient_clipping"].get("max_norm", 1.0)
        # Clip only parameters that have gradients (our owned shard)
        params_with_grads = [p for p in model_skeleton.parameters() if p.grad is not None]
        if params_with_grads:
            torch.nn.utils.clip_grad_norm_(params_with_grads, max_norm)
    
    # Extract only owned parameter gradients
    grads = extract_owned_gradients(model_skeleton, own_params)
    
    # Clean up skeleton to free memory
    unload_params_from_skeleton(model_skeleton)
    clear_skeleton_gradients(model_skeleton)
    gc.collect()
    clear_gpu_cache(device)
    
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
    grads_received: dict,
    parameters_received: dict,
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

            command, recv_step, rank, data, type = message

            if command == "all_reduce":
                
                logger.info(
                    f"Received message '{command}' from worker {addr} (rank {rank}) for step {recv_step}"
                )
                
                if type == "gradients":
                    
                    logger.info(f"[Step {recv_step}] Storing gradients from worker {rank}")

                    with lock:
                        grads_received[recv_step][rank] = data
                        logger.info(
                            f"[Step {recv_step}] Now have {len(grads_received[recv_step])} gradient sets"
                        )
                elif type == "parameters":
                    
                    logger.info(f"[Step {recv_step}] Storing parameters from worker {rank}")

                    with lock:
                        parameters_received[recv_step][rank] = data
                        logger.info(
                            f"[Step {recv_step}] Now have {len(parameters_received[recv_step])} parameter sets"
                        )
                    
                # reduced_grads = reduce(grads_received[recv_step], len(grads_received[recv_step]))
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
    grads_received: dict,
    weights_received: dict,
    parameters_received: dict,
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
                        grads_received,
                        parameters_received,
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
    model_skeleton,
    owned_params_dict,
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
    Run FSDP (ZeRO Stage 3) training with optimizer + gradient + parameter partitioning.
    Workers partition optimizer states, gradients, and model parameters for maximum memory efficiency.
    
    CRITICAL: This worker NEVER loads the full model. It receives an empty skeleton model
    and its parameter shard for sequential forward computation.

    Args:
        model_skeleton: Empty model skeleton (structure only, no weights)
        owned_params_dict: Dict of {layer_name: param_tensor} for this worker's shard
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
        algorithm="fsdp",
    )
    logger.info(f"🚀 FSDP Worker rank {worker_rank} starting up (ZeRO Stage 2)")

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
    parameters_received = defaultdict(dict)  # Buffer for all-gathered parameters
    num_nodes = cluster_config["num_nodes"]
    num_layers = cluster_config["num_layers"]
    
    # Staleness tracking (only if staleness_bound > 0)
    staleness_stats = {
        "all_gather_step_diffs": [],  # Track step differences for all_gather parameters
        "all_reduce_step_diffs": [],  # Track step differences for all_reduce gradients
        "stale_gradient_count": 0,  # Count of gradients with step_diff > 0
        "max_step_diff": 0,  # Maximum step difference observed
        "broadcast_weights_step_diffs": [],  # Track step differences for broadcast weights
        "stale_weight_count": 0,  # Count of weights with step_diff > 0
        "stale_parameter_count": 0,  # Count of parameters with step_diff > 0 (added for parameter staleness tracking)
    }

    # Gradient clipping
    if grad_clip_norm > 0.0:
        logger.info(f"Gradient clipping enabled: max_norm={grad_clip_norm}")
    else:
        logger.info("Gradient clipping disabled")

    # FSDP Stage 3: Use empty model skeleton received from train.py
    logger.info(f"Using empty model skeleton for worker rank {worker_rank} (received from train.py)...")
    
    # Move empty skeleton to device (takes minimal memory)
    model_skeleton = model_skeleton.to(device)
    logger.info(f"Model skeleton moved to device (no weights loaded on this worker)")
    
    # Track original owned param names for gradient accumulation
    original_owned_param_names = list(owned_params_dict.keys())
    logger.info(f"Worker rank {worker_rank} owns {len(owned_params_dict)} parameters (received from train.py)")
    
    # Create optimizer for owned parameters only
    # We'll create dummy parameters for optimizer
    owned_param_list = [torch.nn.Parameter(p.to(device)) for p in owned_params_dict.values()]
    optimizer = torch.optim.AdamW(owned_param_list, lr=learning_rate)
    logger.info(
        f"Created FSDP Stage 3 optimizer for worker rank {worker_rank} with lr={learning_rate} "
        f"managing {len(owned_params_dict)} parameter tensors"
    )
    
    # Clear GPU cache
    gc.collect()
    clear_gpu_cache(device)
    logger.info(f"FSDP Stage 3: Empty skeleton created, owns {len(owned_params_dict)} params (never loaded full model!)")

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
    checkpoint_path = None
    metadata = None
    
    if save_checkpoints and resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()
        else:
            checkpoint_path = resume_from_checkpoint

        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            
            # Create temporary full model to load checkpoint (only case where we create BaseTransformer)
            # Extract config from existing skeleton structure
            num_layers_from_skeleton = len(model_skeleton.blocks)
            
            # Get config from first block to match architecture
            first_block = model_skeleton.blocks[0]
            num_heads_from_skeleton = first_block.num_heads
            ff_dim_from_skeleton = first_block.ffn[0].out_features
            dropout_from_skeleton = first_block.dropout.p if hasattr(first_block.dropout, 'p') else 0.1
            
            temp_model = BaseTransformer(
                vocab_size=model_skeleton.vocab_size,
                max_seq_len=model_skeleton.max_seq_len,
                model_dim=model_skeleton.model_dim,
                num_layers=num_layers_from_skeleton,
                num_heads=num_heads_from_skeleton,
                ff_dim=ff_dim_from_skeleton,
                dropout=dropout_from_skeleton,
            )
            
            metadata = checkpoint_manager.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=temp_model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            
            # Update owned_params_dict from loaded model
            _, out_layers = get_model_per_node(
                model=temp_model,
                num_nodes=num_nodes,
                local_rank=worker_rank,
                total_layers=num_layers,
            )
            
            for layer_name, module in out_layers.items():
                for param_name, param in module.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    if full_param_name in owned_params_dict:
                        owned_params_dict[full_param_name] = param.data.cpu().clone()
            
            # Update owned_param_list from dict
            for i, name in enumerate(original_owned_param_names):
                owned_param_list[i].data = owned_params_dict[name].to(device)
            
            start_epoch = metadata.get("epoch", 0)
            start_step = metadata.get("step", 0)
            
            # Delete temporary model
            del temp_model, out_layers
            gc.collect()
            clear_gpu_cache(device)
            
            logger.info(f"Resumed owned shard from epoch={start_epoch}, step={start_step}")
        else:
            logger.warning("No checkpoint found to resume from, starting fresh")
    
    logger.info("Starting all-to-all topology setup for FSDP (ZeRO Stage 3).")

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
    max_retries = 120
    retry_delay = 2

    for _ in range(NUM_WORKERS - 1):
        # Connect to next worker in the chain
        next_worker = next(w for w in workers_list if w["rank"] != worker_rank)
        next_ip = next_worker["ip"]
        next_port = next_worker["port"]
        next_rank = next_worker["rank"]
        del workers_list[
            workers_list.index(next_worker)
        ]  # Remove the next worker from the list to avoid duplicate connections

        logger.info(
            f"Worker {worker_rank} will connect to worker {next_rank} at {next_ip}:{next_port}"
        )
        time.sleep(worker_rank * 0.5)  # Stagger connections

        for attempt in range(max_retries):
            try:
                next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                next_sock.connect((next_ip, next_port))
                send_message(next_sock, ("register", worker_rank))

                logger.info(
                    f"Worker {worker_rank} connected to worker {next_rank} at {next_ip}:{next_port}"
                )
                outbound_worker_sockets[next_worker["rank"]] = (
                    next_sock  # This is important because this has the IP + PORT to which the nodes connected to it listen to which is what we have defined and not send stuff to the port we received through sock.accept()!
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection to worker {next_rank} refused (attempt {attempt + 1}/{max_retries} at IP: {next_ip}:{next_port}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to worker {next_rank} after {max_retries} attempts"
                    )
                    raise

    # Step 3: Accept connection from all workers
    accept_workers(
        sock,
        NUM_WORKERS,
        workers=workers,
        grads_received=grads_received,
        weights_received=weights_received,
        parameters_received=parameters_received,
        step_event=step_event,
        lock=lock
    )

    logger.info(f"All workers connected. Starting training for {num_epochs} epochs.")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        val_loss = None #to make the ckpt manager happy at the end of the epoch when it tries to save the checkpoint and log the val_loss in the metadata
        
        model_skeleton.train()

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

            batch_start_time = time.time()
            data = data.to(device)
            target = target.to(device)
            # Update learning rate if scheduler enabled
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = learning_rate

            activations = None

            # FSDP Stage 3: All-gather parameters from all workers
            # Each worker sends their owned parameter shard
            
            # All-gather: send local parameters to all peers via outbound connections
            logger.info(
                "FSDP Stage 3: Broadcasting owned parameters to all peers for sequential activation computation"
            )

            owned_state_dict = {
                name: owned_params_dict[name] for name in original_owned_param_names
            }

            parameters_received[step][worker_rank] = owned_state_dict

            for peer_rank, peer_socket in outbound_worker_sockets.items():
                send_message(
                    peer_socket,
                    (
                        "all_reduce",
                        step,
                        worker_rank,
                        owned_state_dict,
                        'parameters',
                    ),
                )
                logger.info(
                    f"[Step {step}] Worker {worker_rank} sent {len(owned_state_dict)} owned parameters to worker {peer_rank}"
                )
                
            logger.info(
                f"[Step {step}] Worker {worker_rank} sent local parameters to all peers, now waiting to receive parameters from all peers for this step"
            )
            # Wait for all workers (all gather) to send their parameters for this step
            # Check for staleness violations and clean up stale parameters (only if staleness_bound > 0)

            while True:
                with lock:
                    # Clean up old parameters beyond staleness window FIRST
                    if staleness_bound > 0:
                        for old_step in list(parameters_received.keys()):
                            if old_step < step - staleness_bound:
                                logger.info(f"[Step {step}] Cleaning up stale parameters from step {old_step}")
                                parameters_received.pop(old_step, None)
                    
                    # Only check staleness if bounded async is enabled
                    if staleness_bound > 0:
                        for recv_step in list(parameters_received.keys()):
                            # Only check steps within the staleness window
                            if recv_step >= step - staleness_bound:
                                step_diff = abs(recv_step - step)
                                
                                # Track staleness statistics for all_gather (parameters)
                                staleness_stats["all_gather_step_diffs"].append(step_diff)
                                staleness_stats["max_step_diff"] = max(
                                    staleness_stats["max_step_diff"], step_diff
                                )
                                if step_diff > 0:
                                    staleness_stats["stale_parameter_count"] += 1
                                
                                if step_diff > staleness_bound:
                                    logger.error(
                                        f"[Step {step}] STALENESS VIOLATION: Received parameter from step {recv_step} "
                                        f"(diff={step_diff} > bound={staleness_bound}). Training stopped."
                                    )
                                    raise RuntimeError(
                                        f"Staleness bound violated: step difference {step_diff} exceeds bound {staleness_bound}"
                                    )
                    
                    curr_workers_len = len(parameters_received[step])

                logger.info(
                    f"Worker {worker_rank} - Epoch {epoch + 1}/{num_epochs}, Step {step}/{num_epochs * len(train_loader)}: Received parameters from {curr_workers_len}/{NUM_WORKERS} workers (including self)."
                )
                if curr_workers_len < NUM_WORKERS:
                    logger.info(f"Waiting for more parameters for step {step}...")
                    step_event.wait(timeout=1.0)
                    step_event.clear()
                else:
                    break

            
            logger.info(
                f"[Step {step}] Worker {worker_rank} received all parameters from peers, now updating local model parameters with all-gathered parameters"
            )
            
            # FSDP Stage 2: Compute activations sequentially through all worker shards
            if len(parameters_received[step]) == NUM_WORKERS:
                logger.info(
                    f"[Step {step}  / {num_epochs * len(train_loader)}] FSDP Stage 2: Sequential forward (one shard at a time)"
                )
                              
                # Step 1: Sequential forward through ALL shards (only one loaded at a time!)
                final_output = compute_activations_sequential(
                    device=device,
                    model_skeleton=model_skeleton,
                    all_worker_params=parameters_received[step],  # Dict of {rank: params}
                    data=data,
                    num_workers=NUM_WORKERS,
                    log_layers=(step == 0),  # Only log layer distribution on first step
                )
                
                logger.info(
                    f"[Step {step}] Sequential forward complete, now computing gradients for OUR shard only"
                )
            
                # Step 2: Compute gradients for OUR shard only (like Model Parallelism)
                local_loss, local_grads = compute_worker_gradients(
                    device=device,
                    model_skeleton=model_skeleton,
                    own_params=owned_params_dict,
                    all_worker_params=parameters_received[step],  # Need all params for backward
                    final_output=final_output,
                    target=target,
                    criterion=criterion,
                    optimizer=optimizer,
                    config=config,
                    worker_rank=worker_rank,
                    num_workers=NUM_WORKERS,
                )
                
                # Store our gradients
                with lock:
                    grads_received[step][worker_rank] = local_grads

                total_loss += local_loss.item()
                
                # Clean up activations
                del final_output
                gc.collect()
                clear_gpu_cache(device)
                
                logger.info(
                    f"[Step {step}] Worker {worker_rank} computed {len(local_grads)} gradients for owned layers (Stage 2: never loaded full model!)"
                )
            
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
                        "all_reduce",
                        step,
                        worker_rank,
                        local_grads,
                        'gradients',
                    ),
                )
                logger.info(
                    f"[Step {step}] Worker {worker_rank} sent gradients to worker {peer_rank}"
                )
            # Wait for all workers (all gather) to send their gradients for this step
            # Check for staleness violations and clean up stale gradients (only if staleness_bound > 0)
            
            while True:
                with lock:
                    # Clean up old gradients beyond staleness window FIRST
                    if staleness_bound > 0:
                        for old_step in list(grads_received.keys()):
                            if old_step < step - staleness_bound:
                                logger.info(f"[Step {step}] Cleaning up stale gradients from step {old_step}")
                                grads_received.pop(old_step, None)
                    
                    # Only check staleness if bounded async is enabled
                    if staleness_bound > 0:
                        for recv_step in list(grads_received.keys()):
                            # Only check steps within the staleness window
                            if recv_step >= step - staleness_bound:
                                step_diff = abs(recv_step - step)
                                
                                # Track staleness statistics
                                staleness_stats["all_reduce_step_diffs"].append(step_diff)
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
                    f"Worker {worker_rank} - Epoch {epoch + 1}/{num_epochs}, Step {step}/{num_epochs * len(train_loader)}: Received gradients from {curr_workers_len - 1}/{NUM_WORKERS - 1} peers (total: {curr_workers_len}/{NUM_WORKERS})."
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
                    f"[Step {step}  / {num_epochs * len(train_loader)}] Combining gradients from {len(grads_received[step])} participants"
                )

                logger.info(
                    f"[Step {step}] Worker {worker_rank} applying averaged gradients to owned parameters"
                )

                optimizer.zero_grad()

                # Average gradients across all workers for this worker's owned parameters only
                for peer_rank, peer_grads in grads_received[step].items():
                    logger.info(
                        f"[Step {step}] Worker {worker_rank} accumulating gradients from worker {peer_rank} (scaled by 1/{NUM_WORKERS})"
                    )

                    for i, layer_name in enumerate(original_owned_param_names):
                        if layer_name in peer_grads:
                            if owned_param_list[i].grad is None:
                                owned_param_list[i].grad = (
                                    peer_grads[layer_name].to(device) / NUM_WORKERS
                                )
                            else:
                                owned_param_list[i].grad += (
                                    peer_grads[layer_name].to(device) / NUM_WORKERS
                                )

                    logger.info(
                        f"[Step {step}] Worker {worker_rank} accumulated gradients from worker {peer_rank}"
                    )
                    
                optimizer.step()

                logger.info(
                    f"[Step {step}] Worker {worker_rank} model updated with averaged gradients"
                )
                
                
                logger.info(f"[Step {step}] Starting broadcasting weights to all peers after local update")
                
                # FSDP Stage 3: Sync owned_params_dict with optimizer state
                # The optimizer modified owned_param_list in-place, sync back to dict
                for i, name in enumerate(original_owned_param_names):
                    owned_params_dict[name] = owned_param_list[i].data.cpu().clone()
                
                # Only broadcast OUR owned parameters (not merged ones)
                owned_state_dict = {name: owned_params_dict[name] for name in original_owned_param_names}
                
                logger.info(f"[Step {step}] Broadcasting {len(owned_state_dict)} owned parameters (reduced from full state_dict)")
                
                # broadcast owned weights to all peers via outbound connections
                for peer_rank, peer_socket in outbound_worker_sockets.items():
                    send_message(
                        peer_socket,
                        (
                            "broadcast_weights",
                            step,
                            worker_rank,
                            owned_state_dict,
                            None
                        ),
                    )
                    logger.info(
                        f"[Step {step}] Worker {worker_rank} sent owned weights to worker {peer_rank}"
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

                # Keep only this worker's shard authoritative locally.
                # Peer weights are used only for synchronization visibility; the next
                # step's parameter all-gather provides the full set of remote shards.
                for i, name in enumerate(original_owned_param_names):
                    owned_param_list[i].data = owned_params_dict[name].to(device)
                
                logger.info(
                    f"[Step {step}] Worker {worker_rank} kept local shard state after weight broadcast"
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
                                
                                
                        for old_step in list(weights_received.keys()):
                            if old_step < step - staleness_bound:
                                logger.info(f"[Step {step}] Cleaning up stale weights from step {old_step}")
                                weights_received.pop(old_step, None)
                        
                        for old_step in list(grads_received.keys()):
                            if old_step < step - staleness_bound:
                                logger.info(f"[Step {step}] Cleaning up stale all-gather gradients from step {old_step}")
                                grads_received.pop(old_step, None)
                
                # Cleanup current step gradients
                reduced_grads_received.pop(step, None)
                grads_received.pop(step, None)
                del local_grads
                gc.collect()
            else:
                logger.warning(
                    f"[Step {step}] Worker {worker_rank}: No gradients received. Skipping grad update."
                )
                del local_grads

            # Apply gradient clipping
            if grad_clip_norm > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    owned_param_list, grad_clip_norm
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

            # Calculate tokens/sec
            batch_time = time.time() - batch_start_time
            tokens_processed = data.size(0) * data.size(1)
            tok_per_sec = tokens_processed / batch_time if batch_time > 0 else 0

            # Update batch progress bar with current metrics
            batch_pbar.set_postfix({"lr": f"{current_lr:.2e}", "step": step, "tok/s": f"{tok_per_sec:.0f}"})

            # Log training metrics
            wandb_metrics = {
                "step": step,
                "epoch": epoch + 1,
                "lr": current_lr,
                "batch_size": batch_size,
                f"throughput/worker_{worker_rank}_tok_per_sec": tok_per_sec,
            }
            
            # Log staleness metrics if bounded async is enabled and we have data
            if staleness_bound > 0 and step % metrics_log_interval == 0:
                with lock:
                    if staleness_stats["all_gather_step_diffs"]:
                        avg_all_gather_diff = sum(staleness_stats["all_gather_step_diffs"]) / len(
                            staleness_stats["all_gather_step_diffs"]
                        )
                        wandb_metrics[f"staleness/worker_{worker_rank}_all_gather_avg_step_diff"] = avg_all_gather_diff
                    
                    if staleness_stats["all_reduce_step_diffs"]:
                        avg_all_reduce_diff = sum(staleness_stats["all_reduce_step_diffs"]) / len(
                            staleness_stats["all_reduce_step_diffs"]
                        )
                        wandb_metrics[f"staleness/worker_{worker_rank}_all_reduce_avg_step_diff"] = avg_all_reduce_diff
                    
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
                    staleness_stats["all_reduce_step_diffs"] = []
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
                for i, name in enumerate(original_owned_param_names):
                    param = owned_param_list[i]
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

            # Evaluation (skip for FSDP Stage 3 - requires full model reconstruction)
            if step % eval_steps == 0:
                logger.info(f"[Step {step}] Skipping evaluation for FSDP Stage 3 (requires all-gather of full model)")
                val_loss = None
                val_ppl = None
                # Skip wandb logging for evaluation in FSDP Stage 3

            # Save checkpoint at regular intervals
            if save_checkpoints and should_save_checkpoint(
                step, epoch, checkpoint_steps, num_epochs * len(train_loader)
            ):
                # FSDP Stage 3: Save model skeleton + owned params
                # Create a state dict with owned params loaded into skeleton
                with torch.no_grad():
                    for layer_name, param_data in owned_params_dict.items():
                        clean_name = layer_name.replace('model.', '', 1) if layer_name.startswith('model.') else layer_name
                        module = model_skeleton
                        parts = clean_name.split('.')
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        param_name = parts[-1]
                        if hasattr(module, param_name):
                            param = getattr(module, param_name)
                            param.data = param_data.to(device)

                checkpoint_manager.save_checkpoint(
                    model=model_skeleton,
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
