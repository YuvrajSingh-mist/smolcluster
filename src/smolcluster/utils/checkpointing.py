"""
Checkpointing utilities for distributed training.
Handles saving and loading model checkpoints, optimizer states, and training metadata.
"""

import logging
import os
import shutil
import glob
from pathlib import Path
from typing import Dict, Optional, Any, List
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving and loading for distributed training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_optimizer: bool = True,
        rank: int = 0,
        algorithm: str = "syncps",
        save_optimizer_state: Optional[bool] = None,
        prefix: Optional[str] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state (legacy parameter)
            rank: Rank of this process (0 for server, 1+ for workers)
            algorithm: Algorithm name (syncps, mp, edp)
            save_optimizer_state: Whether to save optimizer state (overrides save_optimizer if provided)
            prefix: Prefix for checkpoint filenames (e.g., "server", "worker_0")
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        # Use save_optimizer_state if provided, otherwise use save_optimizer
        self.save_optimizer = save_optimizer_state if save_optimizer_state is not None else save_optimizer
        self.rank = rank
        self.algorithm = algorithm
        self.prefix = prefix if prefix is not None else f"rank_{rank}"
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint manager initialized: dir={self.checkpoint_dir}, max_keep={max_checkpoints}")
    
    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save (optional)
            scheduler: Learning rate scheduler to save (optional)
            loss: Current loss value (optional)
            metadata: Additional metadata to save (optional)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step}_epoch_{epoch}_{self.prefix}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'rank': self.rank,
            'algorithm': self.algorithm,
        }
        
        if loss is not None:
            checkpoint_data['loss'] = loss
        
        if self.save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save scheduler state if it has state_dict method (PyTorch schedulers)
        if scheduler is not None and hasattr(scheduler, 'state_dict'):
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if metadata is not None:
            checkpoint_data['metadata'] = metadata
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path} (step={step}, epoch={epoch}, loss={loss})")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            device: Device to load checkpoint to (optional)
            
        Returns:
            Dictionary with checkpoint metadata (step, epoch, loss, etc.)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            if device is not None:
                checkpoint = torch.load(checkpoint_path, map_location=device)
            else:
                checkpoint = torch.load(checkpoint_path)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model state from {checkpoint_path}")
            
            # Load optimizer state if available
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Loaded optimizer state from {checkpoint_path}")
            
            # Load scheduler state if available
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                if hasattr(scheduler, 'load_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info(f"Loaded scheduler state from {checkpoint_path}")
            
            # Extract metadata
            metadata = {
                'step': checkpoint.get('step', 0),
                'epoch': checkpoint.get('epoch', 0),
                'loss': checkpoint.get('loss', None),
                'rank': checkpoint.get('rank', 0),
                'algorithm': checkpoint.get('algorithm', 'unknown'),
            }
            
            if 'metadata' in checkpoint:
                metadata.update(checkpoint['metadata'])
            
            logger.info(f"Checkpoint loaded: step={metadata['step']}, epoch={metadata['epoch']}, loss={metadata['loss']}")
            
            return metadata
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint for this rank.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        pattern = str(self.checkpoint_dir / f"checkpoint_*_rank_{self.rank}.pt")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            logger.info(f"No checkpoints found for rank {self.rank}")
            return None
        
        # Sort by step number (extract from filename)
        def get_step(path):
            try:
                # Extract step from filename like "checkpoint_step_500_epoch_1_rank_0.pt"
                basename = os.path.basename(path)
                step_str = basename.split('_')[2]  # Get the step number
                return int(step_str)
            except (IndexError, ValueError):
                return 0
        
        latest = max(checkpoints, key=get_step)
        logger.info(f"Found latest checkpoint: {latest}")
        return latest
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent max_checkpoints."""
        if self.max_checkpoints <= 0:
            return
        
        pattern = str(self.checkpoint_dir / f"checkpoint_*_rank_{self.rank}.pt")
        checkpoints = glob.glob(pattern)
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        
        # Delete oldest checkpoints
        num_to_delete = len(checkpoints) - self.max_checkpoints
        for checkpoint_path in checkpoints[:num_to_delete]:
            try:
                os.remove(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint_path}: {e}")
    
    def get_all_checkpoints(self) -> List[str]:
        """
        Get all checkpoints for this rank, sorted by step.
        
        Returns:
            List of checkpoint paths, sorted from oldest to newest
        """
        pattern = str(self.checkpoint_dir / f"checkpoint_*_rank_{self.rank}.pt")
        checkpoints = glob.glob(pattern)
        
        def get_step(path):
            try:
                basename = os.path.basename(path)
                step_str = basename.split('_')[2]
                return int(step_str)
            except (IndexError, ValueError):
                return 0
        
        return sorted(checkpoints, key=get_step)
    
    def delete_all_checkpoints(self):
        """Delete all checkpoints for this rank."""
        pattern = str(self.checkpoint_dir / f"checkpoint_*_rank_{self.rank}.pt")
        checkpoints = glob.glob(pattern)
        
        for checkpoint_path in checkpoints:
            try:
                os.remove(checkpoint_path)
                logger.info(f"Deleted checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")


def should_save_checkpoint(step: int, epoch: int, checkpoint_steps: int, total_steps: int) -> bool:
    """
    Determine if a checkpoint should be saved at this step.
    
    Args:
        step: Current training step
        epoch: Current epoch
        checkpoint_steps: Interval between checkpoints (0 to disable)
        total_steps: Total number of steps
        
    Returns:
        True if checkpoint should be saved
    """
    if checkpoint_steps <= 0:
        return False
    
    # Save at regular intervals
    if step > 0 and step % checkpoint_steps == 0:
        return True
    
    # Always save at the end of training
    if step == total_steps - 1:
        return True
    
    return False
