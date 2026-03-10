"""
Expert Parallelism worker for distributed MoE training.

This module handles the expert parallelism algorithm for training Mixture of Experts models.
Model definitions have been moved to smolcluster.models.moe for reusability across algorithms.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Import MoE model from the models module
from smolcluster.models.moe import Mixtral, topk_sampling

# Note: ModelArgs should be defined elsewhere or passed as configuration
# This is a placeholder to maintain backward compatibility
class ModelArgs:
    """Model configuration arguments. Should be loaded from config file."""

    # Model architecture
    embeddings_dims: int = 768
    vocab_size: int = 50257
    no_of_heads: int = 12
    no_of_decoder_layers: int = 12
    block_size: int = 1024
    batch_size: int = 32
    
    # MoE specific
    experts: int = 8
    top_experts: int = 2
    noisy_topk: bool = False
    
    # Training
    dropout: float = 0.1
    attn_dropout: float = 0.1
    max_lr: float = 3e-4
    
    # Optimization flags
    use_liger: bool = False
    use_flash_attention: bool = True
    use_checkpointing: bool = False
    
    # Device
    device: torch.device = torch.device("cpu")



# Placeholder tokenizer - should be imported from data module
tokenizer = None  # Will be set by the training script


def create_mixtral_model(config=None):
    """Create a Mixtral model instance from configuration.
    
    Args:
        config: Configuration object or dict with model parameters.
        
    Returns:
        Mixtral model instance.
    """
    if config is None:
        # Use ModelArgs defaults
        model = Mixtral(
            vocab_size=ModelArgs.vocab_size,
            embeddings_dims=ModelArgs.embeddings_dims,
            no_of_heads=ModelArgs.no_of_heads,
            no_of_decoder_layers=ModelArgs.no_of_decoder_layers,
            num_experts=ModelArgs.experts,
            top_k=ModelArgs.top_experts,
            max_seq_len=ModelArgs.block_size,
            device=ModelArgs.device,
            attn_dropout=ModelArgs.attn_dropout,
            dropout=ModelArgs.dropout,
            noisy_topk=ModelArgs.noisy_topk,
            use_checkpointing=ModelArgs.use_checkpointing,
            use_flash_attention=ModelArgs.use_flash_attention,
            use_liger=ModelArgs.use_liger,
            tokenizer=tokenizer,
        )
    else:
        # Use provided config
        model = Mixtral(
            vocab_size=config.get('vocab_size', ModelArgs.vocab_size),
            embeddings_dims=config.get('embeddings_dims', ModelArgs.embeddings_dims),
            no_of_heads=config.get('no_of_heads', ModelArgs.no_of_heads),
            no_of_decoder_layers=config.get('no_of_decoder_layers', ModelArgs.no_of_decoder_layers),
            num_experts=config.get('num_experts', ModelArgs.experts),
            top_k=config.get('top_k', ModelArgs.top_experts),
            max_seq_len=config.get('max_seq_len', ModelArgs.block_size),
            device=config.get('device', ModelArgs.device),
            attn_dropout=config.get('attn_dropout', ModelArgs.attn_dropout),
            dropout=config.get('dropout', ModelArgs.dropout),
            noisy_topk=config.get('noisy_topk', ModelArgs.noisy_topk),
            use_checkpointing=config.get('use_checkpointing', ModelArgs.use_checkpointing),
            use_flash_attention=config.get('use_flash_attention', ModelArgs.use_flash_attention),
            use_liger=config.get('use_liger', ModelArgs.use_liger),
            tokenizer=config.get('tokenizer', tokenizer),
        )
    return model


# Example model instantiation for backward compatibility
model = create_mixtral_model()
model = model.to(ModelArgs.device)

# Printing a summary of the architecture
idx = torch.randint(
    low=0,
    high=ModelArgs.vocab_size,
    size=(ModelArgs.batch_size, ModelArgs.block_size),
    dtype=torch.long
)
idx = idx.to(ModelArgs.device)

print(
    summary(
        model=model,
        input_data=idx,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
)

# Utility functions

def find_unused_parameters(model):
    """Find model parameters that didn't receive gradients.
    
    Args:
        model: PyTorch model to inspect.
        
    Returns:
        List of parameter names without gradients.
    """
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused


def save_to_file(step, text):
    """Save generated text to a file.
    
    Args:
        step: Training step number.
        text: Generated text to save.
    """
    with open(f'/generations_{step}.txt', 'w') as f:
        f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
        f.write(text + "\n\n")


# Training configuration and setup

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention
torch.set_float32_matmul_precision('high')

# Training hyperparameters
save_checkpoint_iter = 2000
total_iters = 20000
eval_iters = 200
eval_check = 200
warmup_iters = 1000
min_lr = 0.1 * ModelArgs.max_lr
lr_decay_iters = 20000
total_batch_size = 524288
micro_batch_size = ModelArgs.batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * (ModelArgs.block_size * 1))


# Learning rate scheduler (cosine with warmup)
# From https://github.com/karpathy/nanoGPT/blob/master/train.py

class CustomLRScheduler:
    """Custom learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr):
        """Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer.
            warmup_iters: Number of warmup iterations.
            lr_decay_iters: Number of iterations for LR decay.
            min_lr: Minimum learning rate.
            max_lr: Maximum learning rate.
        """
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.it = 0
        self._last_lr = [max_lr]  # Initialize with max_lr (matching PyTorch convention)
        
    def step(self):
        """Perform a scheduler step."""
        self._last_lr = [self._get_lr()]  # Store as list
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._last_lr[0]
        self.it += 1

    def get_last_lr(self):
        """Get the last computed learning rate.
        
        Returns:
            List containing the last learning rate.
        """
        return self._last_lr  # Returns list to match PyTorch convention
    
    def _get_lr(self):
        """Compute the current learning rate.
        
        Returns:
            Current learning rate value.
        """
        # 1) linear warmup for warmup_iters steps
        if self.it < self.warmup_iters:
            return self.max_lr * (self.it + 1) / (self.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if self.it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def state_dict(self):
        """Get the scheduler state dict.
        
        Returns:
            Dictionary containing scheduler state.
        """
        return {
            'warmup_iters': self.warmup_iters,
            'lr_decay_iters': self.lr_decay_iters,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'it': self.it
        }
    
    def load_state_dict(self, state_dict):
        """Load the scheduler state dict.
        
        Args:
            state_dict: Dictionary containing scheduler state.
        """
        self.warmup_iters = state_dict['warmup_iters']
        self.lr_decay_iters = state_dict['lr_decay_iters']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.it = state_dict['it']




