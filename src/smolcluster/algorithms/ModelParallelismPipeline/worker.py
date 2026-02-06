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
    model_layers: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    next_sock: socket.socket,
    prev_sock: socket.socket,
    worker_rank: int = 0,
    decoder_type_ppl: bool = False,
) -> tuple[float, Optional[float]]:
    """Evaluate model on validation set using distributed model layers.

    Activations flow through worker 0 layers -> worker 1 -> worker 2 -> back to worker 0
    for loss computation, matching the training forward pass.
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

            # Worker rank 0 computes its layer activations and sends to next worker
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
            
            # All workers receive activations, process, and forward (except last worker)
            if worker_rank > 0:
                message = receive_message(prev_sock)
                command, _, payload = message

                if command == "evaluate_forward":
                    activations = payload["activations"].to(device)
                    if worker_rank > 0:
                        target = payload.get("target", target).to(device)
                    
                    # Forward through local layers
                    out = activations
                    for layer in model_layers:
                        output = layer(out)
                        out = output[0] if isinstance(output, tuple) else output
                    activations = out
                    
                    # Send to next worker or back to worker 0
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
            
            # Worker 0 receives final activations and computes loss
            if worker_rank == 0:
                message = receive_message(prev_sock)
                command, _, payload = message
                if command == "evaluate_forward":
                    activations = payload["activations"].to(device)

            # Compute loss using final activations
            loss = compute_loss(activations, target, criterion)
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
    recv_grads = None
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
    logger.info(f"Worker {worker_rank} listening on port {my_port}")
    
    # Step 2: Connect to next worker in pipeline (if not last worker)
    max_retries = 10
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
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection to worker {worker_rank + 1} refused (attempt {attempt + 1}/{max_retries} at IP: {next_ip}:{next_port}). "
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
    
    # # Close listening socket (no longer needed)
    # sock.close()
    # logger.info(f"Ring topology ready for worker {worker_rank}")
    
    # Initialize activation caches
    act_in_cache = {}
    act_out_cache = {}

    logger.info(f"Starting training for {num_epochs} epochs.")
    
    for epoch in range(start_epoch, num_epochs):
        model_layers.train()
        total_loss = 0.0

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

            act_out = None
            activations = None


            if worker_rank == 0:
                logger.info(
                    f"[Step {step}/{num_epochs * len(train_loader)}] Worker rank 0 computing leader activations"
                )
            
                leader_activations = compute_leader_activations(device, model_layers, data)
                leader_activations.requires_grad_(True)
            
                
                # Clear GPU cache before caching activations
                clear_gpu_cache(device)

                # Cache worker rank 0's activations WITH computation graph (no detach!)
                # act_in_cache[(step, RANK)] = data
                act_out_cache[(step, worker_rank)] = leader_activations
                activations = leader_activations
                
                next_rank = worker_rank + 1
             
                logger.info(f"[Step {step}] Sending activations to worker rank {next_rank}")
                
                send_message(
                    next_sock,
                    (
                        "forward_activations",
                        step,
                        {
                            "activations": activations.detach().cpu(),
                            "targets": target.detach().cpu(),
                            "from_rank": worker_rank,
                            "to_rank": next_rank
                        },
                    ),
                )
             
            # Middle and last workers receive from previous worker and process forward pass
            if worker_rank > 0:
                message = receive_message(prev_sock)
                command, recv_step, payload = message
                
                logger.info("Received message from previous worker with addr: %s: command=%s, step=%d", prev_sock.getpeername(), command, recv_step)
                
                assert recv_step == step, (
                    f"Step mismatch: expected {step}, got {recv_step}"
                )
                
                
                if command == "forward_activations":
                    act_in = payload["activations"].to(device)
                    from_rank = payload["from_rank"]
                    to_rank = payload["to_rank"]
                    logger.info(
                        f"[Step {step}] Received activations forwarded from worker {from_rank} to worker {to_rank}"
                    )
                
                    act_in.requires_grad_(True)
                    # Forward through local model layers
                    out = act_in
                    for layer in model_layers:
                        output = layer(out)
                        out = output[0] if isinstance(output, tuple) else output
                    
                    activations = out
                    act_out_cache[(step, worker_rank)] = activations                    
                    act_in_cache[(step, from_rank)] = act_in
                    
                    
                    logger.info(
                        f"[Step {step}] Finished generating activations for local_rank {worker_rank}"
                    )

                    logger.info(
                        f"[Step {step}] Deciding next action for rank {worker_rank}"
                    )
                    
                    if worker_rank == NUM_WORKERS - 1:
                        # Last worker: compute loss and send gradients back
                        logger.info(f"[Step {step}] Last worker, computing loss and starting backward pass")
                        
                        target_payload = payload.get("targets")
                        if target_payload is not None:
                            target = target_payload.to(device)
                        
                        loss = compute_loss(activations, target, criterion)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Send gradients back to previous worker
                        send_message(
                            prev_sock,
                            (
                                "forward_gradients",
                                step,
                                {
                                    "gradients": act_in.grad.detach().cpu() if act_in.grad is not None else None,
                                    "to_rank": worker_rank - 1,
                                    "from_rank": worker_rank
                                },
                            ),
                        )
                        logger.info(f"[Step {step}] Sent gradients back to worker {worker_rank - 1}")
                    else:
                        # Middle workers: forward activations to next worker
                        next_rank = worker_rank + 1
                        logger.info(f"[Step {step}] Sending activations to worker rank {next_rank}")
                        send_message(
                            next_sock,
                            (
                                "forward_activations",
                                step,
                                {
                                    "activations": activations.detach().cpu(),
                                    "targets": target.detach().cpu() if 'targets' in payload else payload.get("targets"),
                                    "from_rank": worker_rank,
                                    "to_rank": next_rank
                                },
                            ),
                        )
                        logger.info(f"[Step {step}] Sent activations to worker {next_rank}")
                    
                    # Clear GPU cache
                    clear_gpu_cache(device)
                    
            # All workers (except the last) wait for gradients
            # Gradients flow backward: Worker 2 → Worker 1 → Worker 0
            # Each worker receives gradients from the next worker via next_sock
            if worker_rank < NUM_WORKERS - 1:
                # Non-last workers receive gradients from the next worker
                message = receive_message(next_sock)
                command, recv_step, payload = message
                
                logger.info("Received gradient message with command=%s, step=%d", command, recv_step)
                
                assert recv_step == step, (
                    f"Step mismatch: expected {step}, got {recv_step}"
                )
                
                if command == 'forward_gradients':
                    logger.info(f"[Step {step}] Received forward gradients")
                    
                    # Get gradients from payload
                    recv_grads = payload["gradients"]
                    to_rank = payload.get("to_rank")
                    from_rank = payload.get("from_rank")
                    
                    logger.info(f"[Step {step}] Received gradients from worker {from_rank} for worker {to_rank}")
                    
                    # Retrieve cached activations for this step
                    act_out = act_out_cache[(step, worker_rank)]
                        
                    # Compute gradients locally using autograd
                    optimizer.zero_grad()
                    torch.autograd.backward(act_out, recv_grads.to(device))
                    
                    # Clear GPU cache after backward pass
                    clear_gpu_cache(device)

                    if worker_rank > 0:
                        # Forward gradients to previous worker
                        prev_rank = worker_rank - 1
                        act_in_cached = act_in_cache.get((step, prev_rank))
                        if act_in_cached is not None and act_in_cached.grad is not None:
                            logger.info(f"[Step {step}] Forwarding gradients to worker {prev_rank}")
                            send_message(
                                prev_sock,
                                (
                                    "forward_gradients",
                                    step,
                                    {
                                        "gradients": act_in_cached.grad.detach().cpu(),
                                        "to_rank": prev_rank,
                                        "from_rank": worker_rank
                                    },
                                ),
                            )
                    else:
                        logger.info(f"[Step {step}] Worker 0 completed backward pass")
            

            # for rank, worker_socket, _addr in sorted(worker_queue, reverse=True):
            #     if rank == NUM_WORKERS:
            #         tqdm.write(
            #             f"[LEADER] [Step {step}] Sending generate_gradients command to last worker rank {rank}"
            #         )
            #         send_message(
            #             worker_socket,
            #             (
            #                 "generate_gradients",
            #                 step,
            #                 {
            #                     "gradients": None,
            #                 },
            #             ),
            #         )

            #     # Receiving the last worker nodes activations
            #     message = receive_message(worker_socket)

            #     command, recv_step, payload = message

            #     assert recv_step == step, (
            #         f"Step mismatch: expected {step}, got {recv_step}"
            #     )

            #     if command == "forward_gradients":
            #         recv_grads = payload["gradients"]
            #         to_rank = payload["to_rank"]
            #         from_rank = payload["from_rank"]
            #         tqdm.write(
            #             f"[LEADER] [Step {step}] Received gradients forwarded to server from worker {from_rank} for {to_rank}"
            #         )

            #         if to_rank == RANK:
            #             tqdm.write(f"[LEADER] [Step {step}] Computing backward pass for server")
            #             # Restore server's activations from cache (has computation graph)
            #             act_out = act_out_cache[(step, RANK)]
            #             act_out = act_out.to(device)

            #             optimizer.zero_grad()
            #             # Backward - this updates model parameters
            #             torch.autograd.backward(act_out, recv_grads.to(device))
            #             optimizer.step()

            #             # Clean up server activation cache
            #             if (step, RANK) in act_out_cache:
            #                 del act_out_cache[(step, RANK)]

            #             # Clear GPU cache after backward pass
            #             clear_gpu_cache(device)

            #         else:
            #             target_socket = next(
            #                 (s for r, s, _ in worker_queue if r == to_rank), None
            #             )
            #             if target_socket:
            #                 tqdm.write(
            #                     f"[LEADER] [Step {step}] Forwarding gradients to worker rank {to_rank} from {from_rank} via current rank {rank}"
            #                 )
            #                 send_message(
            #                     target_socket,
            #                     (
            #                         "forward_gradients",
            #                         step,
            #                         {"gradients": recv_grads, "to_rank": to_rank},
            #                     ),
            #                 )

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

            # Update batch progress bar with current metrics
            batch_pbar.set_postfix({
                'lr': f'{current_lr:.2e}',
                'step': step
            })
            
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
            if step % eval_steps == 0 and step != 0:
                val_loss, val_ppl = evaluate(
                    device,
                    model_layers,
                    val_loader,
                    criterion,
                    next_sock,
                    prev_sock,
                    worker_rank,
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
                    # Update progress bar
                    batch_pbar.set_postfix({'val_loss': f'{val_loss:.4f}', 'ppl': f'{val_ppl:.2f}'})
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
                    # Update progress bar
                    batch_pbar.set_postfix({'val_loss': f'{val_loss:.4f}'})

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
