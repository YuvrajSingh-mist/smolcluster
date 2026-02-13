import gc
import heapq
import logging
import math
import socket
import time
from pathlib import Path
from typing import Optional

import torch
import torchinfo
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import subprocess

from smolcluster.utils.checkpointing import CheckpointManager, should_save_checkpoint
from smolcluster.utils.common_utils import (
    calculate_bandwidth_metrics,
    get_network_metrics,
    receive_message,
    send_message,
)
from smolcluster.utils.layers import get_model_per_node
from smolcluster.utils.logging_utils import setup_cluster_logging


def get_tensor_size_mb(tensor: torch.Tensor) -> float:
    """Calculate tensor size in megabytes."""
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


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
    total_workers: int,
    step: int,
    epoch: int,
    device: torch.device,
    model_layers: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    next_sock: socket.socket,
    prev_sock: socket.socket,
    worker_rank: int = 0,
    decoder_type_ppl: bool = False,
    
) -> tuple[float, Optional[float]]:
    """Evaluate model on validation set using distributed model layers.

    Pipeline flow: Worker 0 → Worker 1 → Worker 2 (computes loss)
    All workers must call this function together to avoid blocking.
    """
    model_layers.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for data, target in tqdm(
            val_loader, 
            desc="Evaluating", 
            leave=False, 
            ncols=80,
            disable=(worker_rank != 0)
        ):
            data = data.to(device)
            target = target.to(device)

            # Worker 0: compute initial activations and send to worker 1
            if worker_rank == 0:
                activations = compute_leader_activations(device, model_layers, data)
                send_message(
                    next_sock,
                    (
                        "evaluate_forward",
                        0,
                        {
                            "activations": activations.detach().cpu(),
                            "target": target.detach().cpu(),
                        },
                    ),
                )
            
            # Workers 1 and 2: receive activations from previous worker
            if worker_rank > 0:
                message = receive_message(prev_sock)
                command, _, payload = message
             
                if command == "evaluate_forward":
                    activations = payload["activations"].to(device)
                    target = payload["target"].to(device)
                    
                    # Forward through local layers
                    out = activations
                    for layer in model_layers:
                        output = layer(out)
                        out = output[0] if isinstance(output, tuple) else output
                    activations = out
                    
                    if worker_rank == total_workers - 1:
                        # Last worker: compute loss
                        loss = compute_loss(activations, target, criterion)
                        total_val_loss += loss.item()
                    else:
                        # Middle workers: forward to next worker
                        send_message(
                            next_sock,
                            (
                                "evaluate_forward",
                                0,
                                {
                                    "activations": activations.detach().cpu(),
                                    "target": target.detach().cpu(),
                                },
                            ),
                        )

                clear_gpu_cache(device)
    
    logger.info(f"Worker {worker_rank} completed evaluation with total_val_loss={total_val_loss:.4f}")
    avg_loss = total_val_loss / len(val_loader)
    ppl = math.exp(avg_loss) if decoder_type_ppl else None
    
    if worker_rank == total_workers - 1:

        if decoder_type_ppl:
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/val": avg_loss,
                    "ppl/val": ppl,
                }
            )
            eval_msg = f"[Step {step}] Evaluation: Val Loss={avg_loss:.4f}, Val PPL={ppl:.2f}"
            logger.info(eval_msg)

        else:
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/val": avg_loss,
                }
            )
            eval_msg = f"[Step {step}] Evaluation: Val Loss={avg_loss:.4f}"
            logger.info(eval_msg)
        
    
        
    model_layers.train()
        
    return avg_loss, ppl
    


def clear_gpu_cache(device: torch.device) -> None:
    """Clear GPU cache for both MPS and CUDA devices."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def compute_loss(
    activations: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """Compute loss from activations and targets.
    
    Args:
        activations: Output activations [B, T, C]
        target: Target labels [B, T]
        criterion: Loss function
        
    Returns:
        Loss tensor
    """
    B, T, C = activations.shape
    output = activations.view(B * T, C)
    target_flat = target.view(B * T)
    loss = criterion(output, target_flat)
    return loss


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


# Setup logging (will be replaced by setup_cluster_logging in run_modelparallelism_without_ps_worker)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = None  # Will be set in run_modelparallelism_pipeline_worker



def run_modelparallelism_pipeline_worker(
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
    Run Model Parallelism Pipeline training (peer-to-peer workers in linear pipeline).

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
    logger.info(f"🚀 ModelParallelism Worker rank {worker_rank} starting up")

    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config["track_gradients"]
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    learning_rate = config["learning_rate"]
    grad_clip_norm = config.get("grad_clip_norm", 0.0)
    cluster_config["num_workers"]
    NUM_WORKERS = cluster_config["num_workers"]
    num_microbatches = config["num_microbatches"]
    recv_grads = None

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
        rank=worker_rank,
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
        # Get port for this hostname
        PORT = port_config.get(hostname, port_config.get("default", 65432))
    else:
        PORT = port_config

    # Load tokenizer
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")

    # Log model summary
    model_summary = str(torchinfo.summary(model, verbose=0, device=device))
    logger.info("Model Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    # Load model layers for this worker rank
    num_layers = config["num_layers"]
    logger.info(f"Loading worker rank {worker_rank}'s share of model layers...")

    model_layers, out_layers = get_model_per_node(
        model, num_nodes=num_nodes, local_rank=worker_rank, total_layers=num_layers
    )

    model_layers = model_layers.to(device)
    logger.info(f"Worker rank {worker_rank} loaded {len(model_layers)} layers")

    # Create optimizer for this worker's layers only
    optimizer = torch.optim.AdamW(model_layers.parameters(), lr=learning_rate)
    logger.info(f"Created optimizer for worker rank {worker_rank} with lr={learning_rate}")

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

   
    logger.info("Starting pipeline topology setup (linear forward, backward flow).")

    next_sock = None
    prev_sock = None
    
    # Step 1: Each worker binds to its own port
    my_port = PORT + worker_rank
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST_IP, my_port))
    sock.listen(1)
    logger.info(f"Worker {worker_rank} listening on add: {HOST_IP} at port {my_port}")
    
    # Step 2: Connect to next worker in pipeline (if not last worker)
    max_retries = 60
    retry_delay = 2
    
    if worker_rank < NUM_WORKERS - 1:
        # Connect to next worker in the pipeline
        workers_list = cluster_config["pipelineTopology"]["workers"]["regular"]
        next_worker = next(w for w in workers_list if w["rank"] == worker_rank + 1)
        next_ip = next_worker["ip"]
        next_port = my_port + 1
        
        logger.info(f"Worker {worker_rank} will connect to worker {worker_rank + 1} at {next_ip}:{next_port}")
        time.sleep(worker_rank * 0.5)  # Stagger connections
        
        for attempt in range(max_retries):
            try:
                next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                next_sock.connect((next_ip, next_port))
                logger.info(f"Worker {worker_rank} connected to worker {worker_rank + 1} at {next_ip}:{next_port}")
                break
            except (ConnectionRefusedError, TimeoutError, OSError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection to worker {worker_rank + 1} failed: {type(e).__name__} (attempt {attempt + 1}/{max_retries} at IP: {next_ip}:{next_port}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to worker {worker_rank + 1} after {max_retries} attempts")
                    raise
    else:
        # Last worker doesn't connect forward (end of pipeline)
        logger.info(f"Worker {worker_rank} is the last worker in pipeline (no forward connection)")
    
    # Step 3: Accept connection from previous worker (if not first worker)
    if worker_rank > 0:
        prev_sock, prev_addr = sock.accept()
        logger.info(f"Worker {worker_rank} accepted connection from worker {worker_rank - 1} at {prev_addr}")
    else:
        logger.info(f"Worker {worker_rank} is the first worker in pipeline (no backward connection for forward pass)")
    
 
    # Initialize activation caches
    act_in_cache = {}
    act_out_cache = {}

    # Initialize data transfer tracking
    activation_recv_times = []
    activation_recv_sizes = []
    activation_send_times = []
    activation_send_sizes = []
    gradient_send_times = []
    gradient_send_sizes = []

    logger.info(f"Starting training for {num_epochs} epochs.")
    
    for epoch in range(start_epoch, num_epochs):
        model_layers.train()
        total_loss = 0.0
        total_ppl = 0.0
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
            
            logger.info(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
            step = epoch * len(train_loader) + batch_idx
            
            data = data.to(device)
            target = target.to(device)
    
            # Check if microbatching is enabled
            num_microbatches = config.get("num_microbatches", None)
            use_microbatching = num_microbatches is not None and num_microbatches > 1
            
            optimizer.zero_grad()
            
            # Skip batches if resuming mid-epoch
            if step < start_step:
                continue
            
            losses = {}
            batch_loss = 0.0  # Track loss for THIS batch only
            batch_ppl = 0.0   # Track PPL for THIS batch only
            
            if use_microbatching:
                # ===== GPIPE MICROBATCHING PATH =====
                microbatch_size = data.size(0) // num_microbatches
                
                #fill phase: 
                for micro_idx in range(num_microbatches):
                    
                    micro_step = num_microbatches * step + micro_idx
                    micro_data = data[micro_idx * microbatch_size : (micro_idx + 1) * microbatch_size]
                    micro_target = target[micro_idx * microbatch_size : (micro_idx + 1) * microbatch_size]
                    
                    logger.info(f"Processing micro-batch {micro_idx + 1}/{num_microbatches} for step {step} (micro_step {micro_step})")
                    
                    # Update learning rate if scheduler enabled
                    if scheduler is not None:
                        current_lr = scheduler.get_last_lr()[0]
                    else:
                        current_lr = learning_rate

                    act_out = None
                    activations = None


                    if worker_rank == 0:
                        logger.info(
                            f"[Micro step {micro_step}/{num_microbatches}] Worker rank 0 computing leader activations"
                        )
                    
                        leader_activations = compute_leader_activations(device, model_layers, micro_data)
                        leader_activations.requires_grad_(True)
                    
                        
                        # Clear GPU cache before caching activations
                        clear_gpu_cache(device)
                        gc.collect()
                        
                        # Cache worker rank 0's activations WITH computation graph (no detach!)
                        # act_in_cache[(step, RANK)] = data
                        act_out_cache[(micro_step, worker_rank)] = leader_activations
                        activations = leader_activations
                        
                        next_rank = worker_rank + 1
                    
                        logger.info(f"[Step {step}] Sending activations to worker rank {next_rank}")
                        
                        # Track activation send size and time
                        act_send_size_mb = get_tensor_size_mb(activations.detach().cpu())
                        act_send_start = time.time()
                        
                        send_message(
                            next_sock,
                            (
                                "forward_activations",
                                micro_step,
                                {
                                    "activations": activations.detach().cpu(),
                                    "targets": micro_target.detach().cpu(),
                                    "from_rank": worker_rank,
                                    "to_rank": next_rank
                                },
                            ),
                        )
                        
                        act_send_time = time.time() - act_send_start
                        activation_send_times.append(act_send_time)
                        activation_send_sizes.append(act_send_size_mb)
                    
                    # Middle and last workers receive from previous worker and process forward pass
                    elif worker_rank > 0:
                        # Track activation receive time
                        act_recv_start = time.time()
                        message = receive_message(prev_sock)
                        act_recv_time = time.time() - act_recv_start
                        command, recv_step, payload = message
                        
                        logger.info("Received message from previous worker with addr: %s: command=%s, step=%d", prev_sock.getpeername(), command, recv_step)
                        
                        assert recv_step == micro_step, (
                            f"Micro step mismatch: expected {micro_step}, got {recv_step}"
                        )
                    
                    
                        if command == "forward_activations":
                            act_in = payload["activations"].to(device)
                            from_rank = payload["from_rank"]
                            to_rank = payload["to_rank"]
                            target = payload["targets"].to(device)
                            
                            # Track activation receive size
                            act_recv_size_mb = get_tensor_size_mb(payload["activations"])
                            activation_recv_times.append(act_recv_time)
                            activation_recv_sizes.append(act_recv_size_mb)
                            
                            logger.info(
                                f"[Micro step {micro_step}] Received activations forwarded from worker {from_rank} to worker {to_rank}"
                            )
                        
                            act_in.requires_grad_(True)
                            # Forward through local model layers
                            out = act_in
                            for layer in model_layers:
                                output = layer(out)
                                out = output[0] if isinstance(output, tuple) else output
                            
                            activations = out
                            act_out_cache[(micro_step, worker_rank)] = activations                    
                            act_in_cache[(micro_step, from_rank)] = act_in
                            
                            clear_gpu_cache(device)
                            gc.collect()
                            
                            logger.info(
                                f"[Micro step {micro_step}] Finished generating activations for local_rank {worker_rank}"
                            )

                     
                            
                            if worker_rank == NUM_WORKERS - 1:
                                # Last worker: compute loss and send gradients back
                                logger.info(f"[Micro step {micro_step}] Last worker, computing loss and starting backward pass")
                            
                                logger.info(f"[Micro step {micro_step}] Target shape: {target.shape}, dtype: {target.dtype}, act shape: {activations.shape}, dtype: {activations.dtype}")
                                
                                loss = compute_loss(activations, target, criterion)
                                
                                # Accumulate for batch average
                                batch_loss += loss.item()
                                batch_ppl += torch.exp(loss).item()
                                
                                # Accumulate for epoch average
                                total_loss += loss.item()
                                total_ppl += torch.exp(loss).item()
                                
                                losses[(micro_step, worker_rank)] = loss
                                
                                logger.info(f"[Micro step {micro_step}] Loss: {loss.item():.4f}")


                            else:
                                # Middle workers: forward activations to next worker
                                next_rank = worker_rank + 1
                                logger.info(f"[Micro step {micro_step}] Sending activations to worker rank {next_rank}")
                                
                                # Track activation send size and time
                                act_send_size_mb = get_tensor_size_mb(activations.detach().cpu())
                                act_send_start = time.time()
                                
                                send_message(
                                    next_sock,
                                    (
                                        "forward_activations",
                                        micro_step,
                                        {
                                            "activations": activations.detach().cpu(),
                                            "targets": target.detach().cpu(),
                                            "from_rank": worker_rank,
                                            "to_rank": next_rank
                                        },
                                    ),
                                )
                                
                                act_send_time = time.time() - act_send_start
                                activation_send_times.append(act_send_time)
                                activation_send_sizes.append(act_send_size_mb)
                                
                                logger.info(f"[Micro step {micro_step}] Sent activations to worker {next_rank}")
                            
                        
                            # Clear GPU cache
                            clear_gpu_cache(device)      
                    
                        
                # All workers (except the last) wait for gradients
                # Gradients flow backward: Worker 2 → Worker 1 → Worker 0
                # Each worker receives gradients from the next worker via next_sock
                
                
                #Drain Phase
                for micro_idx in range(num_microbatches):
                    
                    micro_step = num_microbatches * step + micro_idx
                    
                    logger.info(f"Processing backward pass for micro-batch {micro_idx + 1}/{num_microbatches} for step {step} (micro_step {micro_step})")
                    
                    if worker_rank == NUM_WORKERS - 1:
                        # Last worker already has the loss for this micro-batch, perform backward pass
                        worker_loss = losses.get((micro_step, worker_rank), None)
                        if worker_loss is None:
                            logger.warning(f"No loss found for micro_step {micro_step} at worker {worker_rank}, skipping backward pass for this micro-batch")
                            continue
                        else:
                            logger.info(f"Retrieved loss for micro_step {micro_step} at worker {worker_rank}")
                            worker_loss.backward()
                            logger.info(f"Completed backward pass for micro_step {micro_step} at worker {worker_rank}")

                            # Track gradient send size and time
                            grad_send_size_mb = get_tensor_size_mb(act_in_cache[(micro_step, worker_rank - 1)].grad.detach().cpu())
                            grad_send_start = time.time()
                            
                            send_message(
                                prev_sock,
                                (
                                    "forward_gradients",
                                    micro_step,
                                    {
                                        "gradients": act_in_cache[(micro_step, worker_rank - 1)].grad.detach().cpu(),
                                        "to_rank": worker_rank - 1,
                                        "from_rank": worker_rank
                                    },
                                ),
                            )
                            
                            grad_send_time = time.time() - grad_send_start
                            gradient_send_times.append(grad_send_time)
                            gradient_send_sizes.append(grad_send_size_mb)
                            
                            logger.info(f"[Micro step {micro_step}] Sent gradients back to worker {worker_rank - 1}")
                            
                            clear_gpu_cache(device)
                            
                    
                    elif worker_rank < NUM_WORKERS - 1:
                        # Non-last workers receive gradients from the next worker
                        message = receive_message(next_sock)
                        command, recv_step, payload = message
                        
                        logger.info("Received gradient message with command=%s, step=%d", command, recv_step)
                        
                        assert recv_step == micro_step, (
                            f"Micro step mismatch: expected {micro_step}, got {recv_step}"
                        )
                        
                        if command == 'forward_gradients':
                            logger.info(f"[Micro step {micro_step}] Received forward gradients")
                            
                            # Get gradients from payload
                            recv_grads = payload["gradients"]
                            to_rank = payload.get("to_rank")
                            from_rank = payload.get("from_rank")
                            
                            logger.info(f"[Micro step {micro_step}] Received gradients from worker {from_rank} for worker {to_rank}")
                            
                            # Retrieve cached activations for this step
                            act_out = act_out_cache[(micro_step, worker_rank)]
                                
                            # # Compute gradients locally using autograd
                            # optimizer.zero_grad()
                            torch.autograd.backward(act_out, recv_grads.to(device))
                            
                            # Clear GPU cache after backward pass
                            clear_gpu_cache(device)

                            if worker_rank > 0:
                                # Forward gradients to previous worker
                                prev_rank = worker_rank - 1
                                act_in_cached = act_in_cache.get((micro_step, prev_rank))
                                if act_in_cached is not None and act_in_cached.grad is not None:
                                    logger.info(f"[Micro step {micro_step}] Forwarding gradients to worker {prev_rank}")
                                    
                                    # Track gradient send size and time
                                    grad_send_size_mb = get_tensor_size_mb(act_in_cached.grad.detach().cpu())
                                    grad_send_start = time.time()
                                    
                                    send_message(
                                        prev_sock,
                                        (
                                            "forward_gradients",
                                            micro_step,
                                            {
                                                "gradients": act_in_cached.grad.detach().cpu(),
                                                "to_rank": prev_rank,
                                                "from_rank": worker_rank
                                            },
                                        ),
                                    )
                                    
                                    grad_send_time = time.time() - grad_send_start
                                    gradient_send_times.append(grad_send_time)
                                    gradient_send_sizes.append(grad_send_size_mb)
                            else:
                                logger.info(f"[Micro step {micro_step}] Worker 0 completed backward pass")
                
                # Clean up all cached activations from this training step's micro-batches
                step_start = num_microbatches * step
                step_end = num_microbatches * (step + 1)
                keys_to_delete = [key for key in act_out_cache.keys() if step_start <= key[0] < step_end]
                for key in keys_to_delete:
                    del act_out_cache[key]

                # Clean up all cached input activations from this training step's micro-batches
                keys_to_delete = [key for key in act_in_cache.keys() if step_start <= key[0] < step_end]
                for key in keys_to_delete:
                    del act_in_cache[key]
                    
            else:
                # ===== NAIVE PIPELINE (NO MICROBATCHING) =====
                logger.info(f"[Step {step}] Processing batch without microbatching")
                
                # Update learning rate if scheduler enabled
                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = learning_rate
                
                if worker_rank == 0:
                    # Worker 0: Forward through layers and send to next worker
                    logger.info(f"[Step {step}] Worker 0 computing activations")
                    activations = compute_leader_activations(device, model_layers, data)
                    activations.requires_grad_(True)
                    act_out_cache[(step, worker_rank)] = activations
                    
                    # Track and send
                    act_send_size_mb = get_tensor_size_mb(activations.detach().cpu())
                    act_send_start = time.time()
                    send_message(next_sock, ("forward_activations", step, {
                        "activations": activations.detach().cpu(),
                        "targets": target.detach().cpu(),
                        "from_rank": worker_rank,
                        "to_rank": worker_rank + 1
                    }))
                    act_send_time = time.time() - act_send_start
                    activation_send_times.append(act_send_time)
                    activation_send_sizes.append(act_send_size_mb)
                    
                    clear_gpu_cache(device)
                    
                elif worker_rank > 0:
                    # Receive from previous worker
                    act_recv_start = time.time()
                    message = receive_message(prev_sock)
                    act_recv_time = time.time() - act_recv_start
                    command, recv_step, payload = message
                    
                    assert recv_step == step, f"Step mismatch: expected {step}, got {recv_step}"
                    
                    if command == "forward_activations":
                        act_in = payload["activations"].to(device)
                        from_rank = payload["from_rank"]
                        target_recv = payload["targets"].to(device)
                        
                        act_recv_size_mb = get_tensor_size_mb(payload["activations"])
                        activation_recv_times.append(act_recv_time)
                        activation_recv_sizes.append(act_recv_size_mb)
                        
                        act_in.requires_grad_(True)
                        
                        # Forward through layers
                        out = act_in
                        for layer in model_layers:
                            output = layer(out)
                            out = output[0] if isinstance(output, tuple) else output
                        
                        activations = out
                        act_out_cache[(step, worker_rank)] = activations
                        act_in_cache[(step, from_rank)] = act_in
                        
                        clear_gpu_cache(device)
                        
                        if worker_rank == NUM_WORKERS - 1:
                            # Last worker: compute loss and backward
                            logger.info(f"[Step {step}] Last worker computing loss")
                            loss = compute_loss(activations, target_recv, criterion)
                            batch_loss = loss.item()
                            batch_ppl = torch.exp(loss).item()
                            total_loss += loss.item()
                            total_ppl += torch.exp(loss).item()
                            
                            logger.info(f"[Step {step}] Loss: {loss.item():.4f}")
                            
                            # Backward
                            loss.backward()
                            
                            # Send gradients back
                            grad_send_size_mb = get_tensor_size_mb(act_in.grad.detach().cpu())
                            grad_send_start = time.time()
                            send_message(prev_sock, ("forward_gradients", step, {
                                "gradients": act_in.grad.detach().cpu(),
                                "to_rank": worker_rank - 1,
                                "from_rank": worker_rank
                            }))
                            grad_send_time = time.time() - grad_send_start
                            gradient_send_times.append(grad_send_time)
                            gradient_send_sizes.append(grad_send_size_mb)
                            
                            clear_gpu_cache(device)
                            
                        else:
                            # Middle worker: forward to next
                            act_send_size_mb = get_tensor_size_mb(activations.detach().cpu())
                            act_send_start = time.time()
                            send_message(next_sock, ("forward_activations", step, {
                                "activations": activations.detach().cpu(),
                                "targets": target_recv.detach().cpu(),
                                "from_rank": worker_rank,
                                "to_rank": worker_rank + 1
                            }))
                            act_send_time = time.time() - act_send_start
                            activation_send_times.append(act_send_time)
                            activation_send_sizes.append(act_send_size_mb)
                            
                            clear_gpu_cache(device)
                
                # Wait for gradients (all non-last workers)
                if worker_rank < NUM_WORKERS - 1:
                    message = receive_message(next_sock)
                    command, recv_step, payload = message
                    
                    assert recv_step == step, f"Step mismatch: expected {step}, got {recv_step}"
                    
                    if command == "forward_gradients":
                        recv_grads = payload["gradients"]
                        act_out = act_out_cache[(step, worker_rank)]
                        
                        # Backward
                        torch.autograd.backward(act_out, recv_grads.to(device))
                        clear_gpu_cache(device)
                        
                        if worker_rank > 0:
                            # Forward gradients to previous worker
                            act_in_cached = act_in_cache.get((step, worker_rank - 1))
                            if act_in_cached is not None and act_in_cached.grad is not None:
                                grad_send_size_mb = get_tensor_size_mb(act_in_cached.grad.detach().cpu())
                                grad_send_start = time.time()
                                send_message(prev_sock, ("forward_gradients", step, {
                                    "gradients": act_in_cached.grad.detach().cpu(),
                                    "to_rank": worker_rank - 1,
                                    "from_rank": worker_rank
                                }))
                                grad_send_time = time.time() - grad_send_start
                                gradient_send_times.append(grad_send_time)
                                gradient_send_sizes.append(grad_send_size_mb)
                
                # Clean up caches for naive pipeline
                if (step, worker_rank) in act_out_cache:
                    del act_out_cache[(step, worker_rank)]
                if worker_rank > 0 and (step, worker_rank - 1) in act_in_cache:
                    del act_in_cache[(step, worker_rank - 1)]

            # Log batch-level metrics (after batch processing)
            if worker_rank == NUM_WORKERS - 1:
                if use_microbatching:
                    # Average loss/PPL for THIS batch (microbatching)
                    avg_batch_loss = batch_loss / num_microbatches
                    avg_batch_ppl = batch_ppl / num_microbatches
                    
                    # Cumulative epoch averages
                    num_microbatches_so_far = (batch_idx + 1) * num_microbatches
                    epoch_avg_loss = total_loss / num_microbatches_so_far
                    epoch_avg_ppl = total_ppl / num_microbatches_so_far
                else:
                    # Naive pipeline: batch_loss is already the single loss
                    avg_batch_loss = batch_loss
                    avg_batch_ppl = batch_ppl
                    
                    # Cumulative epoch averages
                    epoch_avg_loss = total_loss / (batch_idx + 1)
                    epoch_avg_ppl = total_ppl / (batch_idx + 1)
                
                logger.info(
                    f"[Step {step}] Batch {batch_idx + 1} completed - "
                    f"Batch Loss: {avg_batch_loss:.4f}, Batch PPL: {avg_batch_ppl:.4f} | "
                    f"Epoch Avg Loss: {epoch_avg_loss:.4f}, Epoch Avg PPL: {epoch_avg_ppl:.4f}"
                )
                
                wandb.log({
                    "losses/batch_loss": avg_batch_loss,
                    "losses/epoch_avg_loss": epoch_avg_loss,
                    "ppl/batch_ppl": avg_batch_ppl if decoder_type_ppl else None,
                    "ppl/epoch_avg_ppl": epoch_avg_ppl if decoder_type_ppl else None,
                    "training/step": step,
                    "training/epoch": epoch + 1,
                    "training/num_microbatches": num_microbatches if use_microbatching else 1,
                    "training/use_microbatching": use_microbatching,
                })
                
                batch_pbar.set_postfix({
                    'batch_loss': f'{avg_batch_loss:.4f}',
                    'epoch_loss': f'{epoch_avg_loss:.4f}',
                    'step': step
                })
          

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

            # Update batch progress bar with current metrics
            batch_pbar.set_postfix({
                'lr': f'{current_lr:.2e}',
                'step': step
            })
            
            # Log training metrics (per step)
            wandb.log(
                {
                    "training/step": step,
                    "training/epoch": epoch + 1,
                    "training/lr": current_lr,
                    "training/batch_size": batch_size,
                    "training/num_microbatches": num_microbatches,
                }
            )

            # Log gradient norms if tracking enabled
            if track_gradients:
                for name, param in model_layers.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad.detach(), 2).item()
                        wandb.log(
                            {
                                f"gradients/worker_{worker_rank}_layer_{name}": grad_norm,
                                "training/step": step,
                                "training/epoch": epoch + 1,
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
                            "training/step": step,
                            "training/epoch": epoch + 1,
                        }
                    )
                    
                    # Calculate bandwidth metrics using utility function
                    act_recv_metrics = calculate_bandwidth_metrics(
                        activation_recv_sizes, activation_recv_times, metrics_log_interval
                    )
                    act_send_metrics = calculate_bandwidth_metrics(
                        activation_send_sizes, activation_send_times, metrics_log_interval
                    )
                    grad_send_metrics = calculate_bandwidth_metrics(
                        gradient_send_sizes, gradient_send_times, metrics_log_interval
                    )
                    
                    wandb.log({
                        f"bandwidth/worker_{worker_rank}_activation_recv_mbps": act_recv_metrics["bandwidth_mbps"],
                        f"bandwidth/worker_{worker_rank}_activation_send_mbps": act_send_metrics["bandwidth_mbps"],
                        f"bandwidth/worker_{worker_rank}_gradient_send_mbps": grad_send_metrics["bandwidth_mbps"],
                        f"data_size/worker_{worker_rank}_activation_recv_mb": act_recv_metrics["avg_size_mb"],
                        f"data_size/worker_{worker_rank}_activation_send_mb": act_send_metrics["avg_size_mb"],
                        f"data_size/worker_{worker_rank}_gradient_send_mb": grad_send_metrics["avg_size_mb"],
                        "training/step": step,
                        "training/epoch": epoch + 1,
                    })
                    
                    logger.info(
                        f"[Worker {worker_rank} Step {step}] Network: Send={network_stats.get('send_bandwidth_mbps', 0):.2f}Mbps, "
                        f"Recv={network_stats.get('recv_bandwidth_mbps', 0):.2f}Mbps | "
                        f"Act Recv={act_recv_metrics['bandwidth_mbps']:.2f}Mbps, "
                        f"Act Send={act_send_metrics['bandwidth_mbps']:.2f}Mbps, "
                        f"Grad Send={grad_send_metrics['bandwidth_mbps']:.2f}Mbps"
                    )

            # Evaluation - ALL workers must participate
            if step % eval_steps == 0 and step != 0:
                logger.info(f"Worker {worker_rank} entering evaluation at step {step} (epoch {epoch + 1})")
                
                val_loss, val_ppl = evaluate(
                    NUM_WORKERS,
                    step,
                    epoch,
                    device,
                    model_layers,
                    val_loader,
                    criterion,
                    next_sock,
                    prev_sock,
                    worker_rank,
                    decoder_type_ppl,
                )

                
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
        
        # Close batch progress bar for this epoch
        batch_pbar.close()

    # Close ring sockets
    if next_sock:
        try:
            send_message(next_sock, "down")
        except:
            pass
        next_sock.close()
    if prev_sock:
        prev_sock.close()

    logger.info("Training completed successfully!")
