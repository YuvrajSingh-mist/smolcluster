import gc
import logging
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path

import torch
import torchinfo
import torchvision
import wandb
import yaml
from torch.utils.data import DataLoader

from smolcluster.models.SimpleNN import SimpleMNISTModel
from smolcluster.utils.common_utils import (
    get_gradients,
    get_weights,
    receive_message,
    send_message,
    set_gradients,
)
from smolcluster.utils.data import get_data_indices
from smolcluster.utils.device import get_device
from smolcluster.utils.quantization import (
    dequantize_model_weights,
    quantize_model_weights,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[SERVER]")

gradients_event = threading.Event()
lock = threading.Lock()

model_version = 0  # Track global model version for elastic training

workers = {}
workers_grads_received = {}  # Single dict for all worker gradients: {(rank, recv_step, worker_version): grads}


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


def load_data(
    batch_size: int, WORLD_SIZE: int, SEED: int, RANK: int
) -> tuple[DataLoader, DataLoader]:
    # load MNIST
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    data = torchvision.datasets.MNIST(
        "../../../data", download=True, transform=transforms
    )
    lendata = len(data)
    torch.manual_seed(SEED)
    trainset, testset = torch.utils.data.random_split(
        data, [int(0.9 * lendata), lendata - int(0.9 * lendata)]
    )
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    batch_indices = get_data_indices(len(trainset), WORLD_SIZE, SEED)
    train_data = torch.utils.data.Subset(trainset, batch_indices[RANK])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    return train_loader, val_loader


def evaluate(
    device: torch.device, model: torch.nn.Module, val_loader: DataLoader, criterion: torch.nn.Module, decoder_type_ppl: bool = False
) -> tuple[float, float]:
    model.eval()
    total_val_loss = 0.0
    correct = 0
    total = 0
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


def compute_leader_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    
) -> tuple[torch.nn.Module, torch.Tensor]:
    model.train()
    
    # data, target = data.to(device), target.to(device)
    output = model(data)
    B,T,C = output.shape
    output = output.view(B*T, C)
    target = target.view(B*T)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()

    return model, loss


def polyak_average_weights(
    current_weights: dict[str, torch.Tensor],
    worker_weights: dict[str, torch.Tensor],
    staleness: int,
    # alpha_base: float = 1.0
) -> tuple[dict[str, torch.Tensor], float]:
    """
    Blend current model with worker's model using staleness-aware Polyak averaging.

    Args:
        current_weights: Current server model weights
        worker_weights: Worker's trained model weights
        staleness: |current_version - worker_version|
        alpha_base: Base weight for worker model (default: 1.0)

    Returns:
        Blended model weights
    """
    # Worker weight decreases with staleness
    staleness_factor = 1 / (1.0 + staleness)

    blended = {}
    for name in current_weights.keys():
        blended[name] = (
            staleness_factor * worker_weights[name]
            + (1.0 - staleness_factor) * current_weights[name]
        )

    return blended, staleness_factor


def run_edp_server(
    model,
    optimizer,
    train_loader,
    val_loader,
    config,
    cluster_config,
    hostname,
    device,
    criterion,
):
    """
    Run EDP parameter server training.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict (nn_config)
        cluster_config: Cluster configuration dict
        hostname: Server hostname
        device: Device to run on
        criterion: Loss criterion
    """
    global model_version
    shutdown_flag = threading.Event()

    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config["track_gradients"]
    learning_rate = config["learning_rate"]
    use_quantization = cluster_config["use_quantization"]
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    
    # Learning rate scheduler setup
    use_lr_scheduler = config.get("use_lr_scheduler", False)
    total_steps = num_epochs * len(train_loader)
    if use_lr_scheduler:
        warmup_iters = config["warmup_iters"]
        min_lr = config["min_lr"]
        get_lr_fn = get_lr_schedule(warmup_iters, total_steps, learning_rate, min_lr)
        logger.info(f"LR scheduler enabled: warmup={warmup_iters}, max_iters={total_steps}, peak_lr={learning_rate}, min_lr={min_lr}")
    else:
        get_lr_fn = None
        logger.info(f"LR scheduler disabled, using constant lr={learning_rate}")
    
    # Checkpoint settings
    save_checkpoints = config.get("save_checkpoints", False)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    checkpoint_steps = config.get("checkpoint_steps", 0)
   
    # Create checkpoint directory
    if save_checkpoints:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {checkpoint_path.absolute()}")
       
    # Create and bind socket
    HOST_IP = "0.0.0.0"
    PORT = cluster_config["port"]
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST_IP, PORT))
    sock.listen(5)
    logger.info(f"Server listening on {HOST_IP}:{PORT}")
    
    # Define handle_worker as nested function with access to model and use_quantization
    def handle_worker(conn: socket.SocketType, addr: tuple[str, int]) -> None:
        global model_version
        logger.info(f"Handling worker at {addr}")

        while True:
            message = receive_message(conn)

            # Handle connection closed or empty message
            if message is None:
                logger.info(f"Worker {addr} closed connection")
                break

            # Unpack the message tuple
            command, payload = message

            if command == "polyark_averaging":
                recv_step = payload["step"]
                rank = payload["rank"]
                worker_version = payload["model_version"]

                # Check if worker sent quantized weights, weights, or gradients
                if "quantized_weights" in payload:
                    # New approach: Dequantize and use Polyak averaging
                    quantized_weights = payload["quantized_weights"]
                    logger.info(
                        f"Received quantized weights from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                    )
                    # Dequantize weights back to float32 on the server's device
                    device_str = str(device)
                    weights = dequantize_model_weights(quantized_weights, device=device_str)
                    with lock:
                        workers_grads_received[(rank, recv_step, worker_version)] = {
                            "type": "weights",
                            "data": weights,
                        }
                elif "weights" in payload:
                    # Legacy: Full float32 weights (no compression)
                    weights = payload["weights"]
                    logger.info(
                        f"Received model weights from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                    )
                    with lock:
                        workers_grads_received[(rank, recv_step, worker_version)] = {
                            "type": "weights",
                            "data": weights,
                        }
                else:
                    # Old approach: Gradient scaling (kept for reference)
                    grads = payload["grads"]
                    logger.info(
                        f"Received gradients from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                    )
                    with lock:
                        workers_grads_received[(rank, recv_step, worker_version)] = {
                            "type": "grads",
                            "data": grads,
                        }

                gradients_event.set()
                logger.info(
                    f"Data stored successfully for worker {rank} at step {recv_step}"
                )

            elif command == "pull_weights":
                worker_version = payload  # payload is the worker's current model version
                logger.info(
                    f"Worker {addr} requested weights (worker version: {worker_version}, server version: {model_version})"
                )

                weights = get_weights(model)
                if use_quantization:
                    quantized_weights = quantize_model_weights(weights)
                    send_message(conn, (quantized_weights, model_version))
                    logger.info(f"Quantized weights sent to worker {addr}")
                else:
                    send_message(conn, (weights, model_version))
                    logger.info(f"Weights sent to worker {addr}")

            elif command == "disconnect":
                logger.info(f"Worker {addr} requested disconnection.")
                #  Remove disconnected worker
                with lock:
                    workers.pop(addr, None)

        conn.close()

    # Initialize W&B
    wandb.init(
        project="smolcluster",
        name=f"server-{hostname}_lr{learning_rate}_bs{batch_size}_workers{len(cluster_config['workers'])}",
        config={
            **config,
            "server_hostname": hostname,
            "worker_hostnames": cluster_config["workers"],
            "num_workers": len(cluster_config["workers"]),
        },
    )

    # Get input size from config (support both MNIST and GPT models)
    
    input_size = (batch_size, config['max_seq_len'])
    input_dtype = [torch.long]
    
    model_summary = str(
        torchinfo.summary(model, input_size=input_size, device=device, dtypes=input_dtype)
    )
    logger.info("Model Summary:")
    logger.info(model_summary)
    wandb.log({"model_structure": model_summary})

    # Start accepting worker connections in background
    def accept_workers():
        while not shutdown_flag.is_set():
            try:
                client_socket, client_address = sock.accept()
                logger.info(f"Accepted connection from {client_address}")

                # Wait for registration message
                message = receive_message(client_socket)
                if message is None:
                    logger.warning(
                        f"Connection from {client_address} closed before registration"
                    )
                    client_socket.close()
                    continue

                command, rank = message
                if command == "register":
                    logger.info(f"Worker {rank} registered from {client_address}")
                    with lock:
                        workers[rank] = client_socket
                    threading.Thread(
                        target=handle_worker,
                        args=(client_socket, client_address),
                        daemon=True,
                    ).start()
                else:
                    logger.warning(
                        f"Unexpected message from {client_address}: {command}"
                    )
                    client_socket.close()
            except OSError:
                # Socket closed, exit gracefully
                if shutdown_flag.is_set():
                    logger.info("Worker acceptance thread shutting down")
                    shutdown_flag.clear()
                    break
                else:
                    logger.error("Socket error occurred")
            except Exception as e:
                if not shutdown_flag.is_set():
                    logger.error(f"Error accepting worker: {e}")

    # Start worker acceptance thread
    threading.Thread(target=accept_workers, daemon=True).start()
    logger.info("Worker acceptance thread started")

    # Give workers a moment to connect
    time.sleep(2)


    logger.info(f"Starting training for {num_epochs} epochs.")

    # Send start signal to all connected workers
    with lock:
        for worker_socket in workers.values():
            try:
                send_message(worker_socket, "start_training")
            except Exception as e:
                logger.error(f"Error sending start signal to worker: {e}")

    train_iter = iter(train_loader)
    total_steps = num_epochs * len(train_loader)
    total_loss = 0.0
    step_start_time = time.time()

    for step in range(total_steps):
        model.train()
        
        # Update learning rate if scheduler enabled
        if get_lr_fn is not None:
            current_lr = get_lr_fn(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = learning_rate

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        data = batch[0].to(device)
        target = batch[1].to(device)

        epoch = step // len(train_loader)
        
       
       
        if track_gradients:
            logger.info("Tracking gradients in wandb...")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    
                    grad_norm = torch.norm(param.grad.detach(), 2).item()
                    wandb.log(
                        {
                            f"gradients/layer_{name}": grad_norm,
                            "step": step,
                        }
                    )
            logger.info("Gradient tracking complete.")

        # Collect any available worker data (non-blocking with small timeout)
        gradients_event.wait(timeout=0.01)  # Very short wait
        gradients_event.clear()

        with lock:
            workers_copy = dict(workers_grads_received)
            workers_grads_received.clear()

        if workers_copy:
            logger.info(f"Step {step}: Collected {len(workers_copy)} worker update(s)")

            # Polyak averaging with model weights
            current_weights = get_weights(model)

            for (rank, _recv_step, worker_version), worker_data in workers_copy.items():
                staleness = abs(model_version - worker_version)

                if worker_data["type"] == "weights":
                    # Polyak averaging: blend worker model with current model
                    worker_weights = worker_data["data"]
                    worker_weights = {
                        k: v.to(device) for k, v in worker_weights.items()
                    }
                    current_weights = {
                        k: v.to(device) for k, v in current_weights.items()
                    }

                    blended_weights, staleness_factor = polyak_average_weights(
                        current_weights, worker_weights, staleness
                    )

                    logger.info(
                        f"Applying worker {rank} model via Polyak averaging "
                        f"(staleness: {staleness}, alpha: {staleness_factor:.3f})"
                    )

                    # Update model with blended weights

                    model.load_state_dict(blended_weights, strict=False)

                    current_weights = blended_weights  # Update for next worker
                    
                      # Compute leader gradients
                    model, leader_loss = compute_leader_loss(
                        model, data, target, criterion, optimizer
                    )
                    logger.info(f"Epoch {epoch + 1}, Step: {step}: Computed leader loss.")

                    total_loss += leader_loss.item()
                    
                    # Gradient clipping
                    if config.get("grad_clip_norm", 0.0) != 0.0:
                        max_norm = config["grad_clip_norm"]
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    optimizer.step()
                    logger.info(
                        f"Applied leader gradients with workers. Step {step}: Updated to model version {model_version}"
                    )
                     # Calculate and log PPL for decoder models
                    if decoder_type_ppl:
                        train_ppl = math.exp(total_loss / (step + 1))
                        if step % 50 == 0:
                            wandb.log({"step": step, "epoch": epoch + 1, "train/ppl": train_ppl})

                    with lock:
                        model_version += 1

            del workers_copy
            gc.collect()

         # Compute leader gradients
        model, leader_loss = compute_leader_loss(
            model, data, target, criterion, optimizer
        )
        logger.info(f"Epoch {epoch + 1}, Step: {step}: Computed leader loss.")

        total_loss += leader_loss.item()
        
         # Calculate and log PPL for decoder models
        if decoder_type_ppl:
            train_ppl = math.exp(total_loss / (step + 1))
            if step % 50 == 0:
                wandb.log({"step": step, "epoch": epoch + 1, "train/ppl": train_ppl})

        # Gradient clipping
        if config.get("grad_clip_norm", 0.0) != 0.0:
            max_norm = config["grad_clip_norm"]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        with lock:
            model_version += 1


        logger.info(
            f"Applied leader gradients. Step {step}: Updated to model version {model_version}"
        )

        logger.info(
            f"Epoch {epoch + 1}, Step: {step}: Step loss = {leader_loss.item():.4f}"
        )
        
        # Calculate tokens/sec throughput
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        tokens_processed = batch_size * config['max_seq_len']
        tok_per_sec = tokens_processed / step_time if step_time > 0 else 0
        step_start_time = step_end_time  # Reset for next step
        
        # Save checkpoint based on steps
        if save_checkpoints and checkpoint_steps > 0 and step > 0 and step % checkpoint_steps == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_step_{step}.pt"
            torch.save({
                'step': step,
                'epoch': epoch + 1,
                'model_version': model_version,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / (step + 1),
                'config': config,
            }, checkpoint_file)
            logger.info(f"💾 Saved checkpoint: {checkpoint_file}")
            
            
           
        if step % 50 == 0:
            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/leader_step_loss": leader_loss.item(),
                    "losses/avg_loss": total_loss / (step + 1),
                }
            )

            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "batch_size": batch_size,
                    "throughput/tok_per_sec": tok_per_sec,
                }
            )

        if step % eval_steps == 0:
            logger.info(f"Evaluating model at step {step}...")

            val_loss, val_ppl = evaluate(device, model, val_loader, criterion, decoder_type_ppl)

            wandb.log(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/val": val_loss,
                }
            )
            if decoder_type_ppl and val_ppl is not None:
                wandb.log({"step": step, "epoch": epoch + 1, "val/ppl": val_ppl})

    logger.info(
        f"Training completed. Total steps: {step + 1}, Final model version: {model_version}"
    )
    logger.info("Waiting for any remaining worker updates...")

    gradients_event.wait(timeout=0.01)
    gradients_event.clear()

    while len(workers) > 0:
        gradients_event.wait(timeout=0.01)
        gradients_event.clear()

        with lock:
            workers_copy = dict(workers_grads_received)
            workers_grads_received.clear()

        if workers_copy:
            logger.info(f"Step {step}: Collected {len(workers_copy)} worker update(s)")

            # NEW APPROACH: Polyak averaging with model weights
            current_weights = get_weights(model)

            for (rank, _recv_step, worker_version), worker_data in workers_copy.items():
                staleness = abs(model_version - worker_version)

                if worker_data["type"] == "weights":
                    # Polyak averaging: blend worker model with current model
                    worker_weights = worker_data["data"]

                    worker_weights = {
                        k: v.to(device) for k, v in worker_weights.items()
                    }
                    current_weights = {
                        k: v.to(device) for k, v in current_weights.items()
                    }

                    blended_weights, staleness_factor = polyak_average_weights(
                        current_weights, worker_weights, staleness
                    )

                    logger.info(
                        f"Applying worker {rank} model via Polyak averaging "
                        f"(staleness: {staleness}, alpha: {staleness_factor:.3f})"
                    )

                    # Update model with blended weights
                    model.load_state_dict(blended_weights, strict=False)
                    current_weights = blended_weights  # Update for next worker

                    with lock:
                        model_version += 1


            del workers_copy

           

            with lock:
                model_version += 1

            step += 1

            epoch = step // len(train_loader)
            
            gc.collect()

            data = data.to(device)
            target = target.to(device)
            output = model(data)
            B,T,C = output.shape
            target = target.view(B*T)
            output = output.view(B*T, C)
            loss = criterion(output, target)
            total_loss += loss.item()

            logger.info(f"Step: {step}: Step loss = {loss.item():.4f}")

            if step % 50 == 0:
                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/step_loss": loss.item(),
                        "losses/avg_loss": total_loss / (step + 1),
                    }
                )

                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "lr": learning_rate,
                        "batch_size": batch_size,
                    }
                )

            if step % eval_steps == 0:
                logger.info(f"Evaluating model at step {step}...")

                val_loss, val_ppl = evaluate(device, model, val_loader, criterion, decoder_type_ppl)

                wandb.log(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                    }
                )
                if decoder_type_ppl and val_ppl is not None:
                    wandb.log({"step": step, "epoch": epoch + 1, "val/ppl": val_ppl})

    shutdown_flag.set()
    sock.close()
    logger.info("Server shutdown complete")
    
    # Cleanup DataLoaders to prevent resource leaks
    del train_loader, val_loader
    gc.collect()

    wandb.finish()


def main():
    """Legacy main function for backward compatibility."""
    global model_version
    
    # Login to wandb using API key from environment variable
    if "WANDB_API_TOKEN" in os.environ:
        wandb.login(key=os.environ["WANDB_API_TOKEN"], relogin=True)
        logger_temp = logging.getLogger("[SERVER-INIT]")
        logger_temp.info("✅ Logged into wandb using WANDB_API_TOKEN")
    else:
        logger_temp = logging.getLogger("[SERVER-INIT]")
        logger_temp.warning("⚠️  WANDB_API_TOKEN not set - wandb may prompt for login")

    # Get hostname from command-line argument
    if len(sys.argv) > 1:
        HOSTNAME = sys.argv[1]
    else:
        HOSTNAME = input("Enter server hostname: ")

    # Load configs
    CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
    with open(CONFIG_DIR / "nn_config.yaml") as f:
        nn_config = yaml.safe_load(f)

    with open(CONFIG_DIR / "cluster_config_edp.yaml") as f:
        cluster_config = yaml.safe_load(f)

    # Extract values with defaults
    NUM_WORKERS = cluster_config["num_workers"]
    SEED = cluster_config.get("seed", 42)
    WORLD_SIZE = NUM_WORKERS + 1
    RANK = 0

    batch_size = nn_config["batch_size"]
    criterion = torch.nn.CrossEntropyLoss()
    
    model = SimpleMNISTModel(
        input_dim=nn_config["model"]["input_dim"],
        hidden=nn_config["model"]["hidden"],
        out=nn_config["model"]["out"],
    )
    model = model.to(get_device())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])
    
    train_loader, val_loader = load_data(batch_size, WORLD_SIZE, SEED, RANK)
    
    run_edp_server(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=nn_config,
        cluster_config=cluster_config,
        hostname=HOSTNAME,
        device=get_device(),
        criterion=criterion,
    )


if __name__ == "__main__":
    main()
