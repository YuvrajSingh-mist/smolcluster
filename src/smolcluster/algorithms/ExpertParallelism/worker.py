import gc
import logging
import math
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    set_gradients
)
from smolcluster.utils.layers import get_expert_per_node, get_model_per_node
from smolcluster.utils.logging_utils import setup_cluster_logging

try:
    import grove as _grove
except ImportError:
    _grove = None

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


def get_expert_probs_and_indices(model: torch.nn.ModuleList, router: torch.nn.Module, data: torch.Tensor, local_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get expert probabilities and indices from the router."""
    
    if local_rank == 0:
        out = model[0](data)  # Pass through the first layer to get activations for routing
        expert_probs, expert_indices = router(out)
    else:
        expert_probs, expert_indices = router(data)
    
    return expert_probs, expert_indices
    


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
    config: dict
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
    tokens_received: dict,
    grads_received: dict,
    reduced_grads_received: dict,
    step_event: threading.Event,
    lock: threading.Lock,
    weights_received: dict,
    activations_received: dict,
    eval_losses_received: dict,
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

            if command == "route_tokens":
                logger.info(
                    f"Received message '{command}' from worker {addr} (for destination rank {rank}) for step {recv_step}"
                )
                logger.info(f"[Step {recv_step}] Storing routed tokens for worker {rank}")

                with lock:
                    tokens_received[recv_step][rank] = data
                    logger.info(
                        f"[Step {recv_step}] Now have {len(tokens_received[recv_step])} token sets"
                    )

                step_event.set()
            
            elif command == "parameter_server_reduce":
                logger.info(
                    f"[Step {recv_step}] Received '{command}' gradients from worker {rank}"
                )
                
                # Buffer gradients by step - handle out-of-order delivery
                with lock:
                    reduced_grads_received[recv_step][rank] = data
                    logger.info(
                        f"[Step {recv_step}] Buffered reduced gradients from worker {rank}. "
                        f"Now have {len(reduced_grads_received[recv_step])} reduced gradient sets for this step"
                    )
                
                step_event.set()


            elif command == "return_activations":
                logger.info(
                    f"[Step {recv_step}] Received expert activations from worker {rank}"
                )
                with lock:
                    activations_received[recv_step][rank] = data
                    logger.info(
                        f"[Step {recv_step}] Buffered activations from worker {rank}. "
                        f"Now have {len(activations_received[recv_step])} activation sets for this step"
                    )
                step_event.set()

            elif command == "eval_loss":
                with lock:
                    eval_losses_received[recv_step][rank] = data
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
    tokens_received: dict,
    grads_received: dict,
    reduced_grads_received: dict,
    weights_received: dict,
    activations_received: dict,
    eval_losses_received: dict,
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
                        tokens_received,
                        grads_received,
                        reduced_grads_received,
                        step_event,
                        lock,
                        weights_received,
                        activations_received,
                        eval_losses_received,
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

def route_tokens(
    expert_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    expert_shard: list,
    data: torch.Tensor,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Route tokens to per-expert buckets.

    Args:
        expert_probs:   [batch, seq, top_k] routing probabilities
        expert_indices: [batch, seq, top_k] selected expert indices
        num_experts:    total number of experts
        expert_shard:   expert indices assigned to this node
        data:           activations to route [batch, seq, embedding_dims]

    Returns:
        selected_tokens: {expert_idx: Tensor[num_selected, embedding_dims]}
        usage_counts:    {expert_idx: num_selected} for all experts
    """
    selected_tokens = {}
    usage_counts = {i: 0 for i in range(num_experts)}

    for expert_idx in range(num_experts):
        expert_mask = expert_indices == expert_idx
        expert_weights = (expert_probs * expert_mask).sum(dim=-1)  # [batch, seq]
        selected = expert_weights > 0
        num_selected = int(selected.sum().item())
        usage_counts[expert_idx] = num_selected
        if expert_idx in expert_shard and num_selected > 0:
            selected_tokens[expert_idx] = data[selected]

    return selected_tokens, usage_counts


def compute_router_usage_counts(
    expert_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> Dict[int, int]:
    """Count how many tokens were routed to each expert for the current step."""
    usage_counts = {i: 0 for i in range(num_experts)}
    for expert_idx in range(num_experts):
        expert_mask = expert_indices == expert_idx
        expert_weights = (expert_probs * expert_mask).sum(dim=-1)  # [batch, seq]
        usage_counts[expert_idx] = int((expert_weights > 0).sum().item())
    return usage_counts


def compute_expert_contributions(
    peer_tokens: Dict[int, torch.Tensor],
    expert_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_shard_indices: list,
    local_experts: torch.nn.ModuleDict,
    embedding_dims: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Run routed tokens through local expert blocks and scatter into a contribution tensor.

    Each selected token is passed through its assigned expert in full (all embedding_dims),
    weighted by the routing probability, and scattered back into the output tensor.

    Args:
        peer_tokens:          {expert_idx: Tensor[num_selected, embedding_dims]} from route_tokens
        expert_probs:         [batch, seq, top_k]
        expert_indices:       [batch, seq, top_k]
        expert_shard_indices: list of expert indices owned by this worker
        local_experts:        nn.ModuleDict {str(expert_idx): ExpertBlock}
        embedding_dims:       full hidden dimension
        batch_size, seq_len:  shape of the original activation batch
        device:               device for output tensor

    Returns:
        contribution: Tensor[batch, seq, embedding_dims]
    """
    contribution = torch.zeros(batch_size, seq_len, embedding_dims, device=device)

    for expert_idx in expert_shard_indices:
        if expert_idx not in peer_tokens:
            continue
        expert_mask = expert_indices == expert_idx
        expert_weights = (expert_probs * expert_mask).sum(dim=-1)  # [batch, seq]
        selected = expert_weights > 0  # [batch, seq] - tokens routed to this expert
        if not selected.any():
            continue
        expert_input = peer_tokens[expert_idx].to(device)          # [n_sel, embedding_dims]
        expert_out = local_experts[str(expert_idx)](expert_input)  # [n_sel, embedding_dims]
        contribution[selected] += expert_out * expert_weights[selected].unsqueeze(-1)

    return contribution


def run_distributed_eval_step(
    *,
    step: int,
    worker_rank: int,
    num_nodes: int,
    num_experts: int,
    num_workers: int,
    device: torch.device,
    val_loader,
    val_iter,
    text_embeddings,
    router,
    expert_shard_indices: list,
    local_experts: torch.nn.ModuleDict,
    embedding_dims: int,
    mixtral,
    criterion: torch.nn.Module,
    outbound_worker_sockets: dict,
    tokens_received: dict,
    activations_received: dict,
    eval_losses_received: dict,
    lock: threading.Lock,
    step_event: threading.Event,
) -> Tuple[Optional[float], Any]:
    """Run one distributed EP validation step and return (val_loss, val_iter)."""
   
    eval_step = ("eval", step)

    if worker_rank == 0:
        if val_iter is None:
            val_iter = iter(val_loader)
        try:
            eval_data, eval_target = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            eval_data, eval_target = next(val_iter)

        eval_data = eval_data.to(device)
        eval_target = eval_target.to(device)

        with torch.no_grad():
            eval_token_activations = text_embeddings(eval_data)
            eval_probs, eval_indices = router(eval_token_activations)

        for peer_rank in range(num_workers):
            peer_expert_indices = get_expert_per_node(peer_rank, num_nodes, num_experts)
            peer_eval_tokens, _ = route_tokens(
                eval_probs,
                eval_indices,
                num_experts,
                peer_expert_indices,
                eval_token_activations,
            )

            payload = {
                "tokens": peer_eval_tokens,
                "expert_probs": eval_probs,
                "expert_indices": eval_indices,
                "target": eval_target,
            }

            if peer_rank == 0:
                with lock:
                    tokens_received[eval_step][peer_rank] = payload
            else:
                peer_socket = outbound_worker_sockets[peer_rank]
                send_message(
                    peer_socket,
                    ("route_tokens", eval_step, peer_rank, payload),  # Use destination rank
                )
    else:
        while True:
            with lock:
                has_eval_tokens = worker_rank in tokens_received[eval_step]
            if has_eval_tokens:
                break
            step_event.wait()
            step_event.clear()

    eval_token_data = tokens_received[eval_step][worker_rank]
    eval_peer_tokens = eval_token_data["tokens"]
    eval_probs = eval_token_data["expert_probs"].to(device)
    eval_indices = eval_token_data["expert_indices"].to(device)
    eval_target = eval_token_data["target"].to(device)
    eval_batch_size, eval_seq_len, _ = eval_probs.shape

    with torch.no_grad():
        eval_contribution = compute_expert_contributions(
            peer_tokens=eval_peer_tokens,
            expert_probs=eval_probs,
            expert_indices=eval_indices,
            expert_shard_indices=expert_shard_indices,
            local_experts=local_experts,
            embedding_dims=embedding_dims,
            batch_size=eval_batch_size,
            seq_len=eval_seq_len,
            device=device,
        )

    if worker_rank != num_nodes - 1:
        last_rank_socket = outbound_worker_sockets[num_nodes - 1]
        send_message(
            last_rank_socket,
            ("return_activations", eval_step, worker_rank, eval_contribution.detach().cpu()),
        )
    else:
        with lock:
            activations_received[eval_step][worker_rank] = eval_contribution

    if worker_rank == num_nodes - 1:
        while True:
            with lock:
                curr_eval_act_len = len(activations_received[eval_step])
            if curr_eval_act_len >= num_nodes:  # Wait for all nodes including self
                break
            step_event.wait()
            step_event.clear()

        with torch.no_grad():
            eval_aggregated = torch.zeros(
                eval_batch_size, eval_seq_len, embedding_dims, device=device
            )
            with lock:
                for rank_contrib in activations_received[eval_step].values():
                    eval_aggregated += rank_contrib.to(device)

            eval_output = mixtral(eval_aggregated)
            B, T, C = eval_output.shape
            eval_output = eval_output.view(B * T, C)
            eval_target_flat = eval_target.view(B * T)
            eval_loss_tensor = criterion(eval_output, eval_target_flat)
            eval_loss_value = float(eval_loss_tensor.item())

        for peer_rank, peer_socket in outbound_worker_sockets.items():
            send_message(
                peer_socket,
                ("eval_loss", eval_step, worker_rank, eval_loss_value),
            )

        with lock:
            eval_losses_received[eval_step][worker_rank] = eval_loss_value

    while True:
        with lock:
            has_eval_loss = (num_nodes - 1) in eval_losses_received[eval_step]
        if has_eval_loss:
            break
        step_event.wait()
        step_event.clear()

    with lock:
        val_loss = eval_losses_received[eval_step][num_nodes - 1]

    tokens_received.pop(eval_step, None)
    activations_received.pop(eval_step, None)
    eval_losses_received.pop(eval_step, None)

    return val_loss, val_iter


def run_ep_worker(
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
    Run FSDP (ZeRO Stage 0) training with optimizer state partitioning.
    Workers partition optimizer states while maintaining full model and gradient replicas.

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
        algorithm="ep",
    )
    logger.info(f"🚀 FSDP Worker rank {worker_rank} starting up (ZeRO Stage 0)")

    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    num_experts = config["num_experts"]
    track_gradients = config["track_gradients"]
    learning_rate = config["learning_rate"]
    embedding_dims = config["embedding_dims"]
    grad_clip_norm = config.get("grad_clip_norm", 0.0)
    top_k = config["top_k"]
    num_layers = config["num_layers"]
    noisy_topk = config.get("noisy_topk", False)
    
    NUM_WORKERS = cluster_config["num_workers"]

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
    tokens_received = defaultdict(dict)  # Buffer for routed tokens from peers
    grads_received = defaultdict(dict)
    reduced_grads_received = defaultdict(dict)  # Buffer for scatter-reduce gradients
    weights_received = defaultdict(dict)  # Buffer for broadcast weights
    activations_received = defaultdict(dict)  # Buffer for expert activations from all workers (last rank)
    eval_losses_received = defaultdict(dict)  # Buffer for eval losses broadcast from last rank
    num_nodes = cluster_config["num_nodes"]
    
    # Expert usage tracking
    expert_usage_counts = {i: 0 for i in range(num_experts)}  # Track how many tokens each expert processes
    last_router_step_usage = {i: 0 for i in range(num_experts)}

    # Gradient clipping
    if grad_clip_norm > 0.0:
        logger.info(f"Gradient clipping enabled: max_norm={grad_clip_norm}")
    else:
        logger.info("Gradient clipping disabled")

    # Initialize EP model components for this rank via get_model_per_node
    _, ep_layers = get_model_per_node(
        model=model,
        num_nodes=num_nodes,
        local_rank=worker_rank,
        total_layers=num_layers,
        model_type="causal_mixtral",
        num_experts=num_experts,
        model_config={
            "top_k": top_k,
            "device": device,
            "noisy_topk": noisy_topk,
        },
    )
    expert_shard_indices = get_expert_per_node(worker_rank, num_nodes, num_experts)
    local_experts = torch.nn.ModuleDict({
        str(idx): ep_layers[f"model.expert_{idx}"] for idx in expert_shard_indices
    })
    text_embeddings = ep_layers.get("model.text_embeddings")
    router          = ep_layers.get("model.router")
    mixtral         = ep_layers.get("model.mixtral")

    logger.info(f"Worker rank {worker_rank} assigned experts: {expert_shard_indices}")
    logger.info(f"Expert partitioning across all {num_nodes} nodes:")
    for rank in range(num_nodes):
        logger.info(f"  Rank {rank}: experts {get_expert_per_node(rank, num_nodes, num_experts)}")
    logger.info(f"Worker rank {worker_rank} model components initialized on device: {device}")

    # Build optimizer over this worker's trainable parameters
    trainable_params = list(local_experts.parameters())
    if worker_rank == 0:
        trainable_params += (list(router.parameters()) + list(text_embeddings.parameters()))
    if worker_rank == num_nodes - 1 and mixtral is not None:
        trainable_params += list(mixtral.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    logger.info(f"Created optimizer for worker rank {worker_rank} with lr={learning_rate}")

    # Log model summary for local expert blocks
    model_summary = str(torchinfo.summary(local_experts, verbose=0, device=device))
    logger.info("Local Expert Blocks Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    config["num_layers"] = cluster_config["num_layers"]

    

    # Learning rate scheduler setup (after optimizer creation)
    use_lr_scheduler = config.get("use_lr_scheduler", False)
    total_steps = num_epochs * (len(train_loader) if train_loader is not None else 0)
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
                model=local_experts,
                optimizer=optimizer if save_optimizer_state else None,
                scheduler=scheduler,  # Load scheduler state if it exists
                device=device,
            )
            # Checkpoint manager already loaded state into model, just extract metadata
            start_epoch = metadata.get("epoch", 0)
            start_step = metadata.get("step", 0)
            logger.info(f"Resumed from epoch={start_epoch}, step={start_step}")
        else:
            logger.warning("No checkpoint found to resume from, starting fresh")

    logger.info("Starting all-to-all topology setup for FSDP (ZeRO Stage 0).")

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
                        f"Connection to worker {next_rank} failed (attempt {attempt + 1}/{max_retries} at IP: {next_ip}:{next_port}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to worker {next_rank} after {max_retries} attempts: {e}"
                    )
                    raise

    # Step 3: Accept connection from all workers
    accept_workers(
        sock,
        NUM_WORKERS,
        workers=workers,
        model_name="",
        tokens_received=tokens_received,
        grads_received=grads_received,
        reduced_grads_received=reduced_grads_received,
        weights_received=weights_received,
        activations_received=activations_received,
        eval_losses_received=eval_losses_received,
        step_event=step_event,
        lock=lock,
    )

    logger.info(f"All workers connected. Starting training for {num_epochs} epochs.")
    if _grove is not None:
        _grove.status("training")

    steps_per_epoch = len(train_loader) if train_loader is not None else 0
    if steps_per_epoch == 0:
        raise ValueError("run_ep_worker requires a non-empty train_loader on all ranks")

    val_iter = None
    train_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        grad_norm = 0.0
        val_loss = None #to make the ckpt manager happy at the end of the epoch when it tries to save the checkpoint and log the val_loss in the metadata
        
        local_experts.train()
        
        if worker_rank == 0 and text_embeddings is not None:
            text_embeddings.train()
        if worker_rank == num_nodes - 1 and mixtral is not None:
            mixtral.train()
            mixtral = mixtral.to(device)
        
        global step  # Declare step as global to modify the global step counter

        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Only rank 0 has the dataloader and processes batches
        if worker_rank == 0:
            # Create batch progress bar for this epoch (only for rank 0)
            batch_pbar = tqdm(
                enumerate(train_loader),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=True,
                ncols=120,
            )
            batch_iterator = batch_pbar
        else:
            # Other workers wait for tokens from rank 0
            batch_iterator = enumerate(range(steps_per_epoch))
            logger.info(f"Worker {worker_rank} waiting for tokens from rank 0")

        for batch_idx, batch_data in batch_iterator:
            step = epoch * steps_per_epoch + batch_idx

            # Skip batches if resuming mid-epoch
            if step < start_step:
                continue

            batch_start_time = time.time()
            
            # Update learning rate if scheduler enabled
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = learning_rate


            # Only rank 0 loads data and runs router
            if worker_rank == 0:
                data, target = batch_data
                data = data.to(device)
                target = target.to(device)

                logger.info(
                    f"[Step {step}/{num_epochs * steps_per_epoch}] Rank 0 computing embeddings and routing"
                )

                # Compute token embeddings (rank 0 only)
                token_activations = text_embeddings(data)  # [batch, seq, embedding_dims]

                # Route based on activations
                expert_probs, expert_indices = router(token_activations)

                # Phase 1: Route activations to workers based on expert assignment
                logger.info(
                    f"[Step {step}] Phase 1: Rank 0 routing activations to workers based on expert assignments"
                )

                # Track global expert usage (from rank 0's perspective)
                global_usage_counts = compute_router_usage_counts(
                    expert_probs=expert_probs,
                    expert_indices=expert_indices,
                    num_experts=num_experts,
                )
                for expert_idx, count in global_usage_counts.items():
                    expert_usage_counts[expert_idx] += count
                last_router_step_usage = global_usage_counts

                routing_summary = []

                # Determine which activations go to which worker's experts
                for peer_rank in range(NUM_WORKERS):
                    peer_expert_indices = get_expert_per_node(peer_rank, num_nodes, num_experts)
                    peer_routed_tokens, _ = route_tokens(expert_probs, expert_indices, num_experts, peer_expert_indices, token_activations)
                    
                    # Even with empty token dicts, send/store payloads so peers do not block waiting.
                    if len(peer_routed_tokens) == 0:
                        logger.warning(f"[Step {step}] No tokens routed to worker {peer_rank} (experts {peer_expert_indices})")
                    
                    # Store local tokens
                    if peer_rank == 0:
                        with lock:
                            tokens_received[step][peer_rank] = {
                                'tokens': peer_routed_tokens,
                                'expert_probs': expert_probs,
                                'expert_indices': expert_indices,
                                'target': target
                            }
                    else:
                        # Send to peer worker
                        peer_socket = outbound_worker_sockets[peer_rank]
                        send_message(
                            peer_socket,
                            (
                                "route_tokens",
                                step,
                                peer_rank,  # Use destination rank, not sender rank
                                {
                                    'tokens': peer_routed_tokens,
                                    'expert_probs': expert_probs,
                                    'expert_indices': expert_indices,
                                    'target': target
                                },
                            ),
                        )
                        logger.info(
                            f"[Step {step}] Rank 0 sent tokens for experts {peer_expert_indices} to worker {peer_rank}"
                        )

                    per_expert_counts = {
                        int(expert_idx): int(tokens.shape[0])
                        for expert_idx, tokens in peer_routed_tokens.items()
                    }
                    routing_summary.append(
                        f"rank={peer_rank} assigned={peer_expert_indices} "
                        f"active={sorted(list(peer_routed_tokens.keys()))} "
                        f"token_counts={per_expert_counts} total_tokens={sum(per_expert_counts.values())}"
                    )
                
                # Log expert usage for this step (only rank 0 has full view)
                logger.info(f"[Step {step}] Expert usage this step: {global_usage_counts}")
                logger.info(
                    f"[Step {step}] Routing summary | " + " ; ".join(routing_summary)
                )
                
                # Log to wandb periodically
                if step % metrics_log_interval == 0:
                    step_total_tokens = sum(last_router_step_usage.values())
                    cumulative_total_tokens = sum(expert_usage_counts.values())

                    # Create bar chart data for wandb
                    expert_usage_data = [[expert_idx, count] for expert_idx, count in sorted(expert_usage_counts.items())]
                    expert_usage_table = wandb.Table(data=expert_usage_data, columns=["Expert ID", "Token Count"])

                    router_step_freq = {
                        f"router/step_expert_{expert_idx}_freq": (
                            (count / step_total_tokens) if step_total_tokens > 0 else 0.0
                        )
                        for expert_idx, count in last_router_step_usage.items()
                    }
                    router_step_tokens = {
                        f"router/step_expert_{expert_idx}_tokens": count
                        for expert_idx, count in last_router_step_usage.items()
                    }

                    cumulative_router_freq = {
                        f"router/cumulative_expert_{expert_idx}_freq": (
                            (count / cumulative_total_tokens) if cumulative_total_tokens > 0 else 0.0
                        )
                        for expert_idx, count in expert_usage_counts.items()
                    }

                  
                    
                    wandb.log({
                        "expert_usage/cumulative_bar_chart": wandb.plot.bar(
                            expert_usage_table, 
                            "Expert ID", 
                            "Token Count",
                            title=f"Cumulative Expert Usage (Step {step})"
                        ),
                        "step": step,
                        "epoch": epoch + 1,
                        "router/step_total_tokens": step_total_tokens,
                        "router/cumulative_total_tokens": cumulative_total_tokens,
                        **router_step_tokens,
                        **router_step_freq,
                        **cumulative_router_freq,
                    })
                    
                    # Also log individual expert usage counts
                    expert_metrics = {
                        f"expert_usage/expert_{i}_tokens": count
                        for i, count in expert_usage_counts.items()
                    }
                    expert_metrics["step"] = step
                    expert_metrics["epoch"] = epoch + 1
                    wandb.log(expert_metrics)
            # Wait for tokens from rank 0
            elif worker_rank != 0:
                logger.info(f"[Step {step}] Worker {worker_rank} waiting for tokens from rank 0...")
                
                while True:
                    
                    with lock:
                        has_tokens = True if worker_rank in tokens_received[step] else False

                    # logger.info(f"[Step {step}] Tokens received: {tokens_received}. For {step} received: {tokens_received[step]}")
                    if not has_tokens:
                        logger.info(f"Waiting for more tokens for step {step}...")
                        step_event.wait()
                        step_event.clear()
                    else:
                        break

            # Process routed activations through local expert blocks
            logger.info(f"[Step {step}] Worker {worker_rank} processing activations through local expert blocks")

            # Get token data (activations) and routing info for this worker
            
            # try:
            token_data = tokens_received[step][worker_rank]  
            peer_tokens = token_data['tokens']   # Dict[expert_idx (int) -> activations]
            ep_probs = token_data['expert_probs'].to(device)    # [batch, seq, top_k]
            ep_indices = token_data['expert_indices'].to(device) # [batch, seq, top_k]
            target = token_data['target'].to(device)

            batch_size, seq_len, _ = ep_probs.shape

            local_token_counts = {
                int(expert_idx): int(tokens.shape[0])
                for expert_idx, tokens in peer_tokens.items()
            }
            logger.info(
                f"[Step {step}] Worker {worker_rank} batch summary: "
                f"assigned_experts={expert_shard_indices}, "
                f"active_experts={sorted(list(peer_tokens.keys()))}, "
                f"token_counts={local_token_counts}, "
                f"total_tokens={sum(local_token_counts.values())}"
            )

            logger.info(f"[Step {step}] Worker {worker_rank} processing experts: {expert_shard_indices}")

            # Run local experts and build contribution tensor
            contribution = compute_expert_contributions(
                peer_tokens=peer_tokens,
                expert_probs=ep_probs,
                expert_indices=ep_indices,
                expert_shard_indices=expert_shard_indices,
                local_experts=local_experts,
                embedding_dims=embedding_dims,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )

            logger.info(f"[Step {step}] Worker {worker_rank} built contribution tensor of shape {contribution.shape}")

            # Send contribution to last rank (or store locally if this is the last rank)
            if worker_rank != num_nodes - 1:
                last_rank_socket = outbound_worker_sockets[num_nodes - 1]
                send_message(
                    last_rank_socket,
                    ("return_activations", step, worker_rank, contribution.detach().cpu()),
                )
                logger.info(f"[Step {step}] Worker {worker_rank} sent activations to last rank {num_nodes - 1}")
                local_grads = {}
                local_loss = torch.tensor(0.0, device=device)
            else:
                with lock:
                    activations_received[step][worker_rank] = contribution

            # Last rank: gather expert activations from all workers and run Mixtral transformer
            if worker_rank == num_nodes - 1:
                logger.info(f"[Step {step}] Last rank waiting for activations from all workers...")
                while True:
                    with lock:
                        curr_act_len = len(activations_received[step])
                        
                    if curr_act_len >= NUM_WORKERS:
                        break
                    step_event.wait()
                    step_event.clear()

                # Aggregate: sum contributions from all workers
                aggregated = torch.zeros(batch_size, seq_len, embedding_dims, device=device)
                with lock:
                    for rank_contrib in activations_received[step].values():
                        aggregated += rank_contrib.to(device)

                logger.info(f"[Step {step}] Last rank aggregated activations of shape {aggregated.shape}")

                # Run Mixtral forward + loss + backward
                local_loss, local_grads = compute_leader_gradients(
                    device, mixtral, aggregated, target, criterion, optimizer, config
                )
                logger.info(f"[Step {step}] Last rank Mixtral loss: {local_loss.item():.4f}")
            
                with lock:
                    grads_received[step][worker_rank] = local_grads

                total_loss += local_loss.item()
            
          
                    
                # Clear GPU cache
                clear_gpu_cache(device)
                
                logger.info(
                    f"[Step {step} / {total_steps}] Worker {worker_rank} loss after expert processing: {local_loss.item():.4f}"
                )
                
                train_ppl = math.exp(local_loss.item()) if local_loss.item() > 0 else float('inf')

                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        f"losses/worker_{worker_rank}_step": local_loss.item(),
                        f"losses/worker_{worker_rank}_total_train": total_loss / (batch_idx + 1),
                        f"ppl/worker_{worker_rank}_train": train_ppl,
                    }
                )
            
            # Phase 2: last rank is the single gradient source and broadcasts to every worker
            if worker_rank == num_nodes - 1:
                logger.info(
                    f"[Step {step}] Last rank broadcasting gradients to all workers"
                )
                for peer_rank, peer_socket in outbound_worker_sockets.items():
                    send_message(
                        peer_socket,
                        (
                            "parameter_server_reduce",
                            step,
                            peer_rank,
                            local_grads,
                        ),
                    )
                    logger.info(
                        f"[Step {step}] Last rank sent gradients to worker {peer_rank}"
                    )

                # Last rank also keeps local copy so all workers follow same receive path
                with lock:
                    reduced_grads_received[step][worker_rank] = local_grads

            # All workers: Wait for gradients broadcast from last rank
            logger.info(
                f"[Step {step}] Worker {worker_rank} waiting for broadcast gradients from last rank..."
            )
            
            while True:
                    with lock:
                        has_grads = worker_rank in reduced_grads_received[step]

                    if has_grads:
                        break
                    
                    step_event.wait()
                    step_event.clear()
                    

            # Apply broadcast gradients to local model
            if len(reduced_grads_received[step]) > 0:
                grads_reduced = reduced_grads_received[step][worker_rank]
                
                logger.info(
                    f"[Step {step}] Worker {worker_rank} applying broadcast gradients to local model"
                )

                # Ensure optimizer step uses only this step's received gradients.
                optimizer.zero_grad(set_to_none=True)

                # Apply gradients to local model components by name matching
                set_gradients(grads_reduced, local_experts)
                if worker_rank == 0 and text_embeddings is not None:
                    set_gradients(grads_reduced, text_embeddings)
                if worker_rank == num_nodes - 1 and mixtral is not None:
                    set_gradients(grads_reduced, mixtral)
                optimizer.step()
                # Rank-0 emits a dashboard ping via stdout so the SSH-tailed log
                # stream carries it back to the controller machine.
                if worker_rank == 0:
                    print(f"[SMOL_PING] step={step}", flush=True)

                logger.info(
                    f"[Step {step}] Worker {worker_rank} model updated with broadcast gradients"
                )

                # Cleanup step data
                reduced_grads_received.pop(step, None)
                grads_received.pop(step, None)
                tokens_received.pop(step, None)
                activations_received.pop(step, None)
                del grads_reduced, local_grads
                gc.collect()
            else:
                logger.warning(
                    f"[Step {step}] Worker {worker_rank}: No broadcast gradients received. Skipping grad update."
                )
                del local_grads

            # Apply gradient clipping across local components
            if grad_clip_norm > 0.0:
                params_to_clip = list(local_experts.parameters())
                if worker_rank == 0 and text_embeddings is not None:
                    params_to_clip += list(text_embeddings.parameters())
                if worker_rank == num_nodes - 1 and mixtral is not None:
                    params_to_clip += list(mixtral.parameters())
                if params_to_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)
                    if worker_rank == num_nodes - 1 and step % 100 == 0:
                        logger.info(f"[Step {step}] Gradient norm before clipping: {grad_norm:.4f}")

            # Step the scheduler after optimizer
            if scheduler is not None:
                scheduler.step()

            # Clear GPU memory after optimizer step
            clear_gpu_cache(device)

            # Calculate tokens/sec
            batch_time = time.time() - batch_start_time
            if worker_rank == 0 and 'data' in locals():
                tokens_processed = data.size(0) * data.size(1)
            else:
                tokens_processed = batch_size * seq_len if 'batch_size' in locals() else 0
            tok_per_sec = tokens_processed / batch_time if batch_time > 0 else 0

            # Update batch progress bar with current metrics (only for rank 0)
            if worker_rank == 0:
                _elapsed = int(time.time() - train_start_time)
                _eh, _em, _es = _elapsed // 3600, (_elapsed % 3600) // 60, _elapsed % 60
                batch_pbar.set_postfix({"lr": f"{current_lr:.2e}", "grad_norm": f"{grad_norm:.3f}", "step": step, "tok/s": f"{tok_per_sec:.0f}", "elapsed": f"{_eh:02d}:{_em:02d}:{_es:02d}"})

            # Log training metrics
            wandb_metrics = {
                "step": step,
                "epoch": epoch + 1,
                "lr": current_lr,
                "batch_size": batch_size,
                f"throughput/worker_{worker_rank}_tok_per_sec": tok_per_sec,
            }
            
            wandb.log(wandb_metrics)

            if _grove is not None:
                _ns = get_network_metrics(reset=False)
                _grove.report(local_loss.item(), step=step, grad_norm=float(grad_norm),
                              tok_per_sec=tok_per_sec,
                              tx_mbps=_ns.get("send_bandwidth_mbps"), rx_mbps=_ns.get("recv_bandwidth_mbps"))
                _grove.status("training")

            # Log gradient norms if tracking enabled
            if track_gradients:
                local_modules = {"experts": local_experts}
                if worker_rank == num_nodes - 1 and mixtral is not None:
                    local_modules["mixtral"] = mixtral
                for module_name, module in local_modules.items():
                    for name, param in module.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log(
                                {
                                    f"gradients/{module_name}/{name}": grad_norm,
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

            # Minimal distributed validation: use one val batch routed through EP pipeline.
        
            if eval_steps > 0 and step % eval_steps == 0:
                logger.info(f"Step {step}: Starting distributed evaluation...")
                
                
                val_loss, val_iter = run_distributed_eval_step(
                    step=step,
                    worker_rank=worker_rank,
                    num_nodes=num_nodes,
                    num_experts=num_experts,
                    num_workers=NUM_WORKERS,
                    device=device,
                    val_loader=val_loader,
                    val_iter=val_iter,
                    text_embeddings=text_embeddings,
                    router=router,
                    expert_shard_indices=expert_shard_indices,
                    local_experts=local_experts,
                    embedding_dims=embedding_dims,
                    mixtral=mixtral,
                    criterion=criterion,
                    outbound_worker_sockets=outbound_worker_sockets,
                    tokens_received=tokens_received,
                    activations_received=activations_received,
                    eval_losses_received=eval_losses_received,
                    lock=lock,
                    step_event=step_event,
                )

                if worker_rank == 0 and val_loss is not None:
                    val_metrics = {
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                    }
                    if val_loss > 0:
                        val_metrics["ppl/val"] = math.exp(val_loss)
                    wandb.log(val_metrics)
                    logger.info(f"[Step {step}] Validation: Val Loss={val_loss:.4f}")
            else:
                val_loss = None

            # Save checkpoint at regular intervals
            if save_checkpoints and should_save_checkpoint(
                step, epoch, checkpoint_steps, total_steps
            ):
                # Wrap model_layers in Sequential for proper state_dict saving

                checkpoint_manager.save_checkpoint(
                    model=local_experts,
                    optimizer=optimizer,
                    scheduler=scheduler,  # Save scheduler state
                    step=step,
                    epoch=epoch,
                    loss=None,
                    metadata={
                        "val_loss": val_loss,
                        "learning_rate": current_lr,
                    },
                )
                logger.info(f"[Step {step}] Checkpoint saved")

            gc.collect()

        # Log epoch summary of expert usage (only rank 0 has complete view)
        if worker_rank == 0:
            logger.info(f"=== Epoch {epoch + 1} Expert Usage Summary ===")
            total_tokens = sum(expert_usage_counts.values())
            for expert_idx in sorted(expert_usage_counts.keys()):
                count = expert_usage_counts[expert_idx]
                percentage = (count / total_tokens * 100) if total_tokens > 0 else 0
                logger.info(f"  Expert {expert_idx}: {count:,} tokens ({percentage:.2f}%)")
            logger.info(f"  Total tokens processed: {total_tokens:,}")
            
            # Log final bar chart for this epoch
            expert_usage_data = [[expert_idx, count] for expert_idx, count in sorted(expert_usage_counts.items())]
            expert_usage_table = wandb.Table(data=expert_usage_data, columns=["Expert ID", "Token Count"])
            
            wandb.log({
                "expert_usage/epoch_bar_chart": wandb.plot.bar(
                    expert_usage_table, 
                    "Expert ID", 
                    "Token Count",
                    title=f"Expert Usage - Epoch {epoch + 1}"
                ),
                "epoch": epoch + 1,
            })

        # Close batch progress bar for this epoch (only for rank 0)
        if worker_rank == 0:
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
