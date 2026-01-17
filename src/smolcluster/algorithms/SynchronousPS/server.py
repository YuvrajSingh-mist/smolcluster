import gc
import logging
import socket
import threading
from collections import defaultdict
import math
import torch
import torchinfo
from torch.utils.data import DataLoader

from smolcluster.utils.common_utils import (
    get_gradients,
    get_weights,
    receive_message,
    send_message,
    set_gradients,
)
from smolcluster.utils.logging_utils import setup_cluster_logging

# Setup logging (will be replaced by setup_cluster_logging in run_syncps_server)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[SERVER]")


def evaluate(
    device: torch.device, 
    model: torch.nn.Module, 
    val_loader: DataLoader, 
    criterion: torch.nn.Module,
    decoder_type_ppl: bool = False
) -> tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_val_loss = 0.0
 
    with torch.no_grad():
        for data, target in val_loader:
    
            data, target = data.to(device), target.to(device)
            output = model(data)
            B,T,C = output.shape
            output = output.view(B*T, C)
            target = target.view(B*T)
            loss = criterion(output, target)
            total_val_loss += loss.item()
    avg_loss = total_val_loss / len(val_loader)
    ppl = math.exp(avg_loss) if decoder_type_ppl else None
    
    return avg_loss, ppl

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
    model.train()
    data, target = data.to(device), target.to(device)
    output = model(data)
    B,T,C = output.shape
    output = output.view(B*T, C)
    target = target.view(B*T)
    loss = criterion(output, target)
    
    # Gradient clipping
    if config.get("gradient_clipping", {}).get("enabled", False):
        max_norm = config["gradient_clipping"].get("max_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    grads = get_gradients(model)
    return loss, grads



def handle_worker(conn: socket.socket, addr: tuple[str, int], workers: dict, grads_received: dict, step_event: threading.Event, lock: threading.Lock) -> None:
    """Handle individual worker connections and gradient reception."""
    logger.info(f"Handling worker at {addr}")

    while True:
        try:
            message = receive_message(conn)

            # Handle connection closed or empty message
            if message is None:
                logger.info(f"Worker {addr} closed connection")
                break

            # Unpack the message tuple
            command, recv_step, rank, grads = message

            logger.info(
                f"Received message '{command}' from worker {addr} (rank {rank}) for step {recv_step}"
            )

            if command == "parameter_server_reduce":
                logger.info(f"[Step {recv_step}] Storing gradients from worker {rank}")
                with lock:
                    grads_received[recv_step][rank] = grads
                    logger.info(
                        f"[Step {recv_step}] Now have {len(grads_received[recv_step])} gradient sets"
                    )
                step_event.set()

        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            break

    logger.info(f"Worker {addr} disconnected")
    conn.close()


def parameter_server_reduce(
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


def run_syncps_server(
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
    Run Synchronous Parameter Server training.
    
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
    global logger
    
    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="server",
        rank=None,
        hostname=hostname,
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs")
    )
    logger.info("🚀 SyncPS Server starting up")

    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    eval_steps = config["eval_steps"]
    track_gradients = config.get("track_gradients", False)
    decoder_type_ppl = config.get("decoder_type", {}).get("ppl", False)
    num_workers = cluster_config["num_workers"]
    world_size = num_workers + 1
    rank = 0  # Server is rank 0
    
    # Create socket
    HOST_IP = "0.0.0.0"
    PORT = cluster_config["port"]
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST_IP, PORT))
    sock.listen(5)
    logger.info(f"Server listening on {HOST_IP}:{PORT}")

    # Thread-safe data structures
    step_event = threading.Event()
    lock = threading.Lock()
    workers = {}
    grads_received = defaultdict(dict)
    
    # Log model summary
    try:
        import wandb
        model_summary = str(torchinfo.summary(model, verbose=0, device=device))
        logger.info("Model Summary:")
        logger.info(model_summary)
        wandb.log({"model_structure": model_summary})
    except Exception as e:
        logger.warning(f"Could not log model summary: {e}")

    # Accept connections and wait for registration
    registered_workers = {}  # rank -> socket
    while len(registered_workers) < num_workers:
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
                continue

            command, worker_rank = message
            if command == "register":
                logger.info(f"Worker {worker_rank} registered from {client_address}")
                registered_workers[worker_rank] = client_socket
                workers[client_address] = client_socket
                threading.Thread(
                    target=handle_worker, 
                    args=(client_socket, client_address, workers, grads_received, step_event, lock),
                    daemon=True
                ).start()
            else:
                logger.warning(f"Unexpected message from {client_address}: {command}")
                client_socket.close()
        except Exception as e:
            logger.error(f"Error during registration from {client_address}: {e}")
            client_socket.close()
            continue

    logger.info("All workers connected. Starting training...")

    # Send start signal to all workers
    for worker_socket in registered_workers.values():
        send_message(worker_socket, "start_training")

    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (data, target) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            logger.info(f"[Step {step}] Server computing leader gradients")
            leader_loss, leader_grads = compute_leader_gradients(
                device, model, data, target, criterion, optimizer, config
            )
            grads_received[step][rank] = leader_grads
            total_loss += leader_loss.item()
            logger.info(f"[Step {step}] Leader loss: {leader_loss.item():.4f}")

           
            wandb.log({
                    "step": step,
                    "epoch": epoch + 1,
                    "losses/leader_step": leader_loss.item(),
                    "losses/leader_total_train": total_loss / (step + 1),
                })
            

            # Wait for all workers
            while True:
                with lock:
                    curr_workers_len = len(grads_received[step])

                logger.info(
                    f"Epoch {epoch + 1}, Step: {step}, Batch {batch_idx}: Received gradients from {curr_workers_len}/{world_size} participants."
                )
                if curr_workers_len < num_workers:
                    logger.info(f"Waiting for more gradients for step {step}...")
                    step_event.wait()
                    step_event.clear()
                else:
                    break

            # Average gradients and update model
            if len(grads_received[step]) != 0:
                logger.info(
                    f"[Step {step}] Averaging gradients from {len(grads_received[step])} participants"
                )
                grads_reduced = parameter_server_reduce(
                    grads_received[step], len(grads_received[step])
                )

               
                logger.info(
                    f"[Step {step}] Applying averaged gradients to server model"
                )
                set_gradients(grads_reduced, model)
                optimizer.step()
                logger.info(f"[Step {step}] Server model updated")
                
                 # Send updated weights to workers
                for _worker_addr, worker_socket in workers.items():
                    weights = get_weights(model)
                    send_message(worker_socket, ("model_weights", step, weights))
                    logger.info(
                        f"[Step {step}] Sent updated model weights to worker at {_worker_addr}"
                    )
                    
                # Cleanup
                grads_received.pop(step, None)
                del grads_reduced, leader_grads
                gc.collect()
            else:
                logger.warning(
                    f"No gradients received for step {step}. Skipping grad update."
                )
                del leader_grads

            # Log gradient norms if tracking enabled
            if track_gradients:
                try:
                    import wandb
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = torch.norm(param.grad.detach(), 2).item()
                            wandb.log({
                                f"gradients/layer_{name}": grad_norm,
                                "step": step,
                                "epoch": epoch + 1,
                            })
                except:
                    pass

            # Log training metrics
            
            wandb.log({
                    "step": step,
                    "epoch": epoch + 1,
                    "lr": optimizer.param_groups[0]['lr'],
                    "batch_size": batch_size,
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
                        f"Step {step}: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}"
                    )
                else:
                    wandb.log({
                        "step": step,
                        "epoch": epoch + 1,
                        "losses/val": val_loss,
                    })
                    logger.info(
                        f"Step {step}: Val Loss={val_loss:.4f}"
                    )

        avg_loss = total_loss / len(train_loader)
       
        wandb.log({
                "epoch": epoch + 1,
                "losses/train_epoch": avg_loss,
            })
       

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed. Avg Loss: {avg_loss:.4f}"
        )

    logger.info("Training completed successfully!")
    sock.close()

