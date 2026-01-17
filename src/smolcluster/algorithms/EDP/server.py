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
import wandb
from torch.utils.data import DataLoader

from smolcluster.utils.common_utils import (
    get_weights,
    receive_message,
    send_message,
)

from smolcluster.utils.quantization import (
    dequantize_model_weights,
    quantize_model_weights,
)
from smolcluster.utils.logging_utils import setup_cluster_logging
from queue import Queue


# Setup logging (will be replaced by setup_cluster_logging in run_edp_server)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[SERVER]")

# gradients_event = threading.Event()
lock = threading.Lock()

model_version = 0  # Track global model version for elastic training

workers = {}
workers_grads_received = {}  # Single dict for all worker gradients: {(rank, recv_step, worker_version): grads}


def sender_loop(sock, send_queue):
    """Sender thread that processes messages from a queue using send_message."""
    while True:
        # try:
        try:
            msg = send_queue.get(timeout=0.01)
            send_message(sock, msg)
            
        except Exception as e:
            # Queue timeout or socket error
            if isinstance(e, OSError):
                logger.error(f"Socket error in sender_loop: {e}")
                break
            continue


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


def compute_leader_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    use_fp16: bool = False,
    scaler = None,
) -> tuple[torch.nn.Module, torch.Tensor]:
    model.train()
    optimizer.zero_grad()
    
    # Use AMP if enabled
    if use_fp16 and scaler is not None:
        with torch.amp.autocast(device_type=device.type):
            output = model(data)
            B,T,C = output.shape
            output = output.view(B*T, C)
            target = target.view(B*T)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping with scaler
        if config.get("grad_clip_norm", 0.0) != 0.0:
            scaler.unscale_(optimizer)
            max_norm = config.get("grad_clip_norm")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    else:
        output = model(data)
        B,T,C = output.shape
        output = output.view(B*T, C)
        target = target.view(B*T)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        if config.get("grad_clip_norm", 0.0) != 0.0:
            max_norm = config.get("grad_clip_norm")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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

def enqeue_bounded_queue(bounded_queue: Queue, message):
    
    try:
        bounded_queue.put_nowait(message)
        logger.info("Message enqueued successfully.")   
    except Exception:
        
        try:
            discarded = bounded_queue.get_nowait()
            logger.warning("Bounded queue full, discarding oldest message.")
            bounded_queue.put_nowait(message)
    
        except Exception:
            logger.error("Failed to enqueue message after discarding oldest message.")
            raise


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
    global model_version, logger
    shutdown_flag = threading.Event()
    
    # Configure centralized logging (adds file handler to existing module-level logger)
    setup_cluster_logging(
        logger=logger,
        component="server",
        rank=None,
        hostname=hostname,
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs")
    )
    logger.info("🚀 EDP Server starting up")

    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config["track_gradients"]
    learning_rate = config["learning_rate"]
    use_quantization = cluster_config["use_quantization"]
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    use_fp16 = config.get("use_fp16", False)
    
    #Defining the bounded queue
    bounded_queue = Queue(maxsize=cluster_config['queue_size'])
    MAX_MSGS_PER_STEP = 2 #to how much messages from queue to process per step
    
    # Initialize AMP scaler if fp16 enabled (supports both CUDA and MPS)
    scaler = torch.amp.GradScaler(device.type) if use_fp16 and device.type in ['cuda', 'mps'] else None
    if use_fp16:
        logger.info(f"Mixed precision training (fp16) enabled on device: {device.type}")
    
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
    def handle_worker(conn: socket.SocketType, addr: tuple[str, int], bounded_queue: Queue) -> None:
        global model_version
        logger.info(f"Handling worker at {addr}")

        while True:
            message = receive_message(conn)

            # Handle connection closed or empty message
            if message is None:
                logger.info(f"Worker {addr} closed connection")
                break
            
            enqeue_bounded_queue(bounded_queue, (message, conn, addr))
            
            # gradients_event.set()
            
            # Unpack the message tuple

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
                    
                    # Create send queue and sender thread for this worker
                    send_queue = Queue(maxsize=32)
                    
                    threading.Thread(
                        target=sender_loop,
                        args=(client_socket, send_queue),
                        daemon=True
                    ).start()
                    
                    with lock:
                        workers[rank] = {
                            "conn": client_socket,
                            "send_queue": send_queue
                        }
                    
                    # Send start signal immediately to this worker via queue
                    try:
                        send_queue.put("start_training")
                        logger.info(f"Queued start_training signal for worker {rank}")
                    except Exception as e:
                        logger.error(f"Error queuing start signal for worker {rank}: {e}")
                    
                    threading.Thread(
                        target=handle_worker,
                        args=(client_socket, client_address, bounded_queue),
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
        
       
       
        # if track_gradients and step % 1000 == 0:
        #     logger.info("Tracking gradients in wandb...")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
                    
        #             grad_norm = torch.norm(param.grad.detach(), 2).item()
        #             wandb.log(
        #                 {
        #                     f"gradients/layer_{name}": grad_norm,
        #                     "step": step,
        #                 }
        #             )
        #     logger.info("Gradient tracking complete.")

            
        start_time = time.time()
        num_msgs_processed = 0
        while time.time() - start_time < 0.01 and num_msgs_processed < MAX_MSGS_PER_STEP:
            
        # for _ in range(MAX_MSGS_PER_STEP):
            
            try:
                message, conn, addr = bounded_queue.get_nowait()
            except Exception as e:
                logging.error(f"Error getting message from queue: {e}")
                break

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
                    
                    weights = payload["weights"]
                    logger.info(
                        f"Received model weights from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                    )
                    with lock:
                        workers_grads_received[(rank, recv_step, worker_version)] = {
                            "type": "weights",
                            "data": weights,
                        }
                
                
                logger.info(
                    f"Data stored successfully for worker {rank} at step {recv_step}"
                )

            elif command == "pull_weights":
                rank = payload["rank"]
                worker_version = payload["model_version"]
                logger.info(
                    f"Worker rank {rank} at {addr} requested weights (worker version: {worker_version}, server version: {model_version})"
                )

                weights = get_weights(model)
                
                if rank is not None and rank in workers:
                    if use_quantization:
                        quantized_weights = quantize_model_weights(weights)
                        workers[rank]["send_queue"].put((quantized_weights, model_version))
                        logger.info(f"Quantized weights queued for worker {rank}")
                    else:
                        workers[rank]["send_queue"].put((weights, model_version))
                        logger.info(f"Weights queued for worker {rank}")
                else:
                    logger.warning(f"Worker rank {rank} not found in workers dict, cannot send weights")

            elif command == "disconnect":
                logger.info(f"Worker {addr} requested disconnection.")
                #  Remove disconnected worker
                with lock:
                    workers.pop(addr, None)
            num_msgs_processed += 1
      
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
                        f"Applied worker {rank} model via Polyak averaging "
                        f"(staleness: {staleness}, alpha: {staleness_factor:.3f})"
                    )

                    # Update model with blended weights

                    model.load_state_dict(blended_weights, strict=False)

                    current_weights = blended_weights  # Update for next worker
                    
                      # Compute leader gradients
                    model, leader_loss = compute_leader_loss(
                        model, data, target, criterion, optimizer, config, device, use_fp16, scaler
                    )
                    logger.info(f"Epoch {epoch + 1}, Step: {step}: Computed leader loss.")

                    total_loss += leader_loss.item()
                    
                    # Gradient clipping (only if not using AMP, since AMP handles it)
                    if not use_fp16:
                        if config.get("grad_clip_norm", 0.0) != 0.0:
                            max_norm = config["grad_clip_norm"]
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    logger.info(
                        f"Applied leader gradients with worker {rank}. Step {step} / {total_steps}: Updated to model version {model_version}"
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
            
        else:
            
            # Compute leader gradients
            model, leader_loss = compute_leader_loss(
                model, data, target, criterion, optimizer, config, device, use_fp16, scaler
            )
            logger.info(f"Epoch {epoch + 1}, Step: {step} / {total_steps}: Computed leader loss.")

            total_loss += leader_loss.item()
            
         # Calculate and log PPL for decoder models
        if decoder_type_ppl:
            train_ppl = math.exp(total_loss / (step + 1))
            if step % 50 == 0:
                wandb.log({"step": step, "epoch": epoch + 1, "train/ppl": train_ppl})

        # Gradient clipping (only if not using AMP, since AMP handles it)
        if not use_fp16:
            if config.get("grad_clip_norm", 0.0) != 0.0:
                max_norm = config["grad_clip_norm"]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        with lock:
            model_version += 1


        logger.info(
            f"Applied leader gradients. Step {step} / {total_steps}: Updated to model version {model_version}"
        )

        logger.info(
            f"Epoch {epoch + 1}, Step: {step} / {total_steps}: Step loss = {leader_loss.item():.4f}"
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

    # gradients_event.wait(timeout=0.01)
    # gradients_event.clear()

    while len(workers) > 0:
        # gradients_event.wait(timeout=0.01)
        # gradients_event.clear()

        start_time = time.time()
        num_msgs_processed = 0
        while time.time() - start_time < 0.01 and num_msgs_processed < MAX_MSGS_PER_STEP:
            
        # for _ in range(MAX_MSGS_PER_STEP):
            
            try:
                message, conn, addr = bounded_queue.get_nowait()
            except Exception as e:
                logging.error(f"Error getting message from queue: {e}")
                break

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
                    
                    weights = payload["weights"]
                    logger.info(
                        f"Received model weights from worker {addr} rank {rank} for step {recv_step} (worker version: {worker_version}, server version: {model_version})"
                    )
                    with lock:
                        workers_grads_received[(rank, recv_step, worker_version)] = {
                            "type": "weights",
                            "data": weights,
                        }
                
                
                logger.info(
                    f"Data stored successfully for worker {rank} at step {recv_step}"
                )

            elif command == "pull_weights":
                rank = payload["rank"]
                worker_version = payload["model_version"]
                logger.info(
                    f"Worker rank {rank} at {addr} requested weights (worker version: {worker_version}, server version: {model_version})"
                )

                weights = get_weights(model)
                
                if rank is not None and rank in workers:
                    if use_quantization:
                        quantized_weights = quantize_model_weights(weights)
                        workers[rank]["send_queue"].put((quantized_weights, model_version))
                        logger.info(f"Quantized weights queued for worker {rank}")
                    else:
                        workers[rank]["send_queue"].put((weights, model_version))
                        logger.info(f"Weights queued for worker {rank}")
                else:
                    logger.warning(f"Worker rank {rank} not found in workers dict, cannot send weights")

            elif command == "disconnect":
                logger.info(f"Worker {addr} requested disconnection.")
                #  Remove disconnected worker
                with lock:
                    workers.pop(addr, None)
            num_msgs_processed += 1
            
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
            
            # Use AMP if enabled (for consistency)
            if use_fp16 and scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    output = model(data)
                    B,T,C = output.shape
                    target = target.view(B*T)
                    output = output.view(B*T, C)
                    loss = criterion(output, target)
            else:
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
   


if __name__ == "__main__":
    main()
