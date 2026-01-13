"""
Exact Paper Reproduction: Wikitext-2 Training

Reproduces arXiv 2512.19428 exactly:
- Dataset: Wikitext-2
- Model size: 13-18M parameters
- Compares Grassmann vs size-matched Transformer
"""
import json
import time
import math
import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchinfo
import wandb
from tqdm import tqdm
import yaml
from smolcluster.models.gpt import BaseTransformer
from smolcluster.data.wikitext import Wikitext2Dataset
from transformers import GPT2Tokenizer


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, get_lr_fn, device, epoch, log_interval=50, global_step=0, grad_clip_norm=1.0):
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for training data.
        optimizer: Optimizer.
        get_lr_fn: Learning rate schedule function (or None for no scheduling).
        device: Device to run on.
        epoch (int): Current epoch number.
        log_interval (int): Logging interval.
        global_step (int): Global training step counter.
        grad_clip_norm (float): Gradient clipping norm.

    Returns:
        tuple: (avg_loss, global_step)
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Update learning rate if scheduler provided
        if get_lr_fn is not None:
            lr = get_lr_fn(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()
        
        _, loss = model(x, labels=y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        global_step += 1

        total_loss += loss.item() * x.size(0)
        total_tokens += x.numel()

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{loss.exp().item():.2f}",
                "lr": f"{current_lr:.2e}",
                "tok/s": f"{tok_per_sec:.0f}",
                "grad_norm": f"{grad_norm.item():.4f}",
            })

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model on validation/test set.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for evaluation data.
        device: Device to run on.

    Returns:
        tuple: (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / total_count
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


def main():
    """Main training function for reproducing the paper's experiments."""
    parser = argparse.ArgumentParser(description="Paper Reproduction: Wikitext-2 & SNLI")
    parser.add_argument("--override", nargs='*', help="Override config values (key=value pairs)")
    args = parser.parse_args()

    # Load config
    config_path = "src/smolcluster/configs/gpt_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config['output']['output_dir'] is None:
        config['output']['output_dir'] = f"outputs/{config['training']['task']}_reproduction"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config['training']['task'] == "wikitext":
        return train_wikitext(config, device, output_dir)

def train_wikitext(config, device, output_dir):
    """Train on Wikitext-2 language modeling."""

    tokenizer = GPT2Tokenizer.from_pretrained(config['data']['tokenizer'])
    vocab_size = len(tokenizer)

    # Load datasets
    print(f"Loading Wikitext-2 (block_size={config['model']['max_seq_len']})...")
    train_dataset = Wikitext2Dataset("train", tokenizer, config['model']['max_seq_len'])
    val_dataset = Wikitext2Dataset("validation", tokenizer, config['model']['max_seq_len'])

    print(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    results = {}

    # Initialize W&B for this model
    wandb.init(
        project=config['logging']['project_name'],
        name=f"gpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_type": "gpt",
            "model_dim": config['model']['model_dim'],
            "num_layers": config['model']['num_layers'],
            "max_seq_len": config['model']['max_seq_len'],
            "batch_size": config['training']['batch_size'],
            "epochs": config['training']['epochs'],
            "lr": config['training']['learning_rate'],
        },
        dir=str(output_dir),
    )

    model = BaseTransformer(
        vocab_size=vocab_size,
        max_seq_len=config['model']['max_seq_len'],
        model_dim=config['model']['model_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ff_dim=config['model']['ff_dim'],
        dropout=config['model']['dropout'],
    )

    model = model.to(device)

    # Print model summary
    print("Model Summary:")
    summary = torchinfo.summary(
        model, 
        input_size=(config['training']['batch_size'], config['model']['max_seq_len']),
        device=device,
        dtypes=[torch.long]
    )
    print(summary)

    # Compile model with torch.compile for better performance
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    # Custom learning rate scheduler with warmup and cosine decay
    max_lr = config['training']['learning_rate']
    min_lr = config['lr_schedule']['min_lr']
    warmup_iters = config['lr_schedule']['warmup_iters']
    lr_decay_iters = len(train_loader) * config['training']['epochs']

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return max_lr * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    train_losses = []
    val_losses = []
    
    # Track best validation perplexity
    best_val_ppl = float('inf')
    best_epoch = 0
    
    # Create checkpoint directory
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    epoch_pbar = tqdm(range(1, config['training']['epochs'] + 1), desc="Training", unit="epoch")
    global_step = 0
    for epoch in epoch_pbar:
        train_loss, global_step = train_epoch(model, train_loader, optimizer, get_lr, device, epoch, log_interval=config['logging']['log_interval'], global_step=global_step, grad_clip_norm=config['training']['grad_clip_norm'])
        val_loss, val_ppl = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_pbar.set_postfix({
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_ppl": f"{val_ppl:.2f}",
            "best_ppl": f"{best_val_ppl:.2f}",
        })

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
        })

        # Save checkpoint every N epochs
        if config['output']['save_checkpoints'] and epoch % config['logging']['checkpoint_interval'] == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # Save best checkpoint
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, best_ckpt_path)
            print(f"✨ New best validation PPL: {val_ppl:.2f} at epoch {epoch} - saved to {best_ckpt_path}")

    print(f"\nFinal Results for GPT:")
    print(f"  Best Val PPL: {best_val_ppl:.2f} (epoch {best_epoch})")
    print(f"  Training complete. Checkpoints saved in {ckpt_dir}")
    print(f"  Note: Run eval_wikitext.py with best checkpoint for test perplexity")

    results["gpt"] = {
        "num_params": num_params,
        "best_val_ppl": float(best_val_ppl),
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    # Log final results to W&B
    wandb.log({
        "final/best_val_ppl": best_val_ppl,
        "final/best_epoch": best_epoch,
        "final/num_params": num_params,
    })

    # Finish W&B run for this model
    wandb.finish()

    # Save results
    with open(output_dir / "results.json", "w") as f:
        # Convert non-serializable items
        save_results = {}
        for k, v in results.items():
            save_results[k] = {
                "num_params": v["num_params"],
                "best_val_ppl": float(v["best_val_ppl"]),
                "best_epoch": v["best_epoch"],
            }
        json.dump(save_results, f, indent=2)

    
if __name__ == "__main__":
    main()