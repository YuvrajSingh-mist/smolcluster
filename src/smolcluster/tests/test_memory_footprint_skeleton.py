"""
Test suite for measuring memory footprint of FSDP Stage 3 model skeleton approach.

This module demonstrates the memory savings achieved by clearing model weights
and keeping only the model structure (skeleton) compared to loading the full model.

Uses the actual BaseTransformer (GPT-2) architecture with hyperparameters matching
the production configuration (gpt_config.yaml):
- vocab_size: 50257 (GPT-2 tokenizer)
- model_dim: 256
- num_layers: 6
- num_heads: 4
- ff_dim: 768
- dropout: 0.1
- max_seq_len: 1024
"""

import gc
import pytest
import torch
import torch.nn as nn
from typing import Tuple, Dict

from smolcluster.models.gpt import BaseTransformer


def get_gpu_memory_allocated(device: torch.device) -> float:
    """
    Get current GPU memory allocated in MB.
    
    Args:
        device: PyTorch device object
        
    Returns:
        Memory allocated in megabytes
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    elif device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    else:
        # CPU memory tracking is more complex, return 0 for now
        return 0.0


def get_gpu_memory_reserved(device: torch.device) -> float:
    """
    Get current GPU memory reserved/cached in MB.
    
    Args:
        device: PyTorch device object
        
    Returns:
        Memory reserved in megabytes
    """
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device) / (1024 ** 2)
    elif device.type == "mps":
        return torch.mps.driver_allocated_memory() / (1024 ** 2)
    else:
        return 0.0


def clear_gpu_cache(device: torch.device) -> None:
    """Clear GPU cache for CUDA and MPS devices."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.empty_cache()


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_parameter_memory_mb(model: nn.Module) -> float:
    """
    Calculate memory footprint of model parameters in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Memory footprint in megabytes
    """
    total_bytes = 0
    for param in model.parameters():
        if param.numel() > 0:
            # Calculate bytes: numel * bytes_per_element
            total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 ** 2)


def create_model_skeleton(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Create a model skeleton by clearing all parameter data to zeros.
    
    This simulates the FSDP Stage 3 approach where we keep the model structure
    with parameter shapes intact but set weights to zero to demonstrate memory usage.
    
    In actual FSDP Stage 3, parameters would be completely uninitialized/empty,
    but we use zeros here to maintain structure for testing.
    
    Args:
        model: Original model with weights
        device: Target device
        
    Returns:
        Model skeleton with zeroed parameters
    """
    skeleton = model
    with torch.no_grad():
        for param in skeleton.parameters():
            # Keep parameter shape but zero out data
            param.data = torch.zeros_like(param.data, device='cpu')
    skeleton = skeleton.to(device)
    return skeleton


def measure_memory_footprint(
    model: nn.Module, 
    device: torch.device,
    scenario: str
) -> Dict[str, float]:
    """
    Measure detailed memory footprint of a model.
    
    Args:
        model: PyTorch model to measure
        device: Device where model is loaded
        scenario: Description of the scenario
        
    Returns:
        Dictionary containing memory metrics
    """
    clear_gpu_cache(device)
    
    # Get baseline
    baseline_allocated = get_gpu_memory_allocated(device)
    baseline_reserved = get_gpu_memory_reserved(device)
    
    # Move model to device
    model = model.to(device)
    
    # Force synchronization
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure after loading
    after_allocated = get_gpu_memory_allocated(device)
    after_reserved = get_gpu_memory_reserved(device)
    
    # Calculate metrics
    param_count = count_parameters(model)
    param_memory_calculated = get_parameter_memory_mb(model)
    
    metrics = {
        "scenario": scenario,
        "parameter_count": param_count,
        "calculated_param_memory_mb": param_memory_calculated,
        "gpu_allocated_mb": after_allocated - baseline_allocated,
        "gpu_reserved_mb": after_reserved - baseline_reserved,
        "baseline_allocated_mb": baseline_allocated,
        "baseline_reserved_mb": baseline_reserved,
    }
    
    return metrics


@pytest.fixture
def device():
    """Fixture to provide appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def model_config():
    """Fixture to provide model configuration matching gpt_config.yaml."""
    return {
        "vocab_size": 50257,  # GPT-2 tokenizer vocab size
        "max_seq_len": 1024,
        "model_dim": 256,
        "num_layers": 6,
        "num_heads": 4,
        "ff_dim": 768,
        "dropout": 0.1,
    }


class TestModelMemoryFootprint:
    """Test suite for model memory footprint measurements."""
    
    def test_full_model_vs_skeleton_memory(self, device, model_config):
        """
        Test memory footprint comparison between full model and skeleton.
        
        This test demonstrates the memory savings achieved by the FSDP Stage 3
        skeleton approach where model weights are cleared after sharding.
        """
        print("\n" + "=" * 80)
        print("MEMORY FOOTPRINT TEST: Full Model vs Model Skeleton")
        print("=" * 80)
        
        # Skip if CPU (memory tracking not reliable)
        if device.type == "cpu":
            pytest.skip("Memory tracking not reliable on CPU device")
        
        # Create model instance
        model = BaseTransformer(
            vocab_size=model_config["vocab_size"],
            max_seq_len=model_config["max_seq_len"],
            model_dim=model_config["model_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ff_dim=model_config["ff_dim"],
            dropout=model_config["dropout"],
        )
        print(f"\nUsing model: BaseTransformer (GPT-2 architecture)")
        
        print(f"Device: {device}")
        print(f"Configuration: {model_config}")
        
        # Scenario 1: Full model with all weights
        print("\n" + "-" * 80)
        print("SCENARIO 1: Full Model (all weights loaded)")
        print("-" * 80)
        
        clear_gpu_cache(device)
        model1 = model
        metrics_full = measure_memory_footprint(model1, device, "Full Model")
        
        print(f"Parameter count: {metrics_full['parameter_count']:,}")
        print(f"Calculated param memory: {metrics_full['calculated_param_memory_mb']:.2f} MB")
        print(f"GPU allocated memory: {metrics_full['gpu_allocated_mb']:.2f} MB")
        print(f"GPU reserved memory: {metrics_full['gpu_reserved_mb']:.2f} MB")
        
        # Get model state dict for skeleton test
        state_dict = {name: param.data.clone() for name, param in model1.named_parameters()}
        
        # Clean up
        del model1
        clear_gpu_cache(device)
        
        # Scenario 2: Model skeleton with cleared weights
        print("\n" + "-" * 80)
        print("SCENARIO 2: Model Skeleton (weights cleared)")
        print("-" * 80)
        
        # Create fresh model instance
        model2 = BaseTransformer(
            vocab_size=model_config["vocab_size"],
            max_seq_len=model_config["max_seq_len"],
            model_dim=model_config["model_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ff_dim=model_config["ff_dim"],
            dropout=model_config["dropout"],
        )
        
        # Create skeleton
        skeleton = create_model_skeleton(model2, device)
        metrics_skeleton = measure_memory_footprint(skeleton, device, "Skeleton")
        
        print(f"Parameter count: {metrics_skeleton['parameter_count']:,}")
        print(f"Calculated param memory: {metrics_skeleton['calculated_param_memory_mb']:.2f} MB")
        print(f"GPU allocated memory: {metrics_skeleton['gpu_allocated_mb']:.2f} MB")
        print(f"GPU reserved memory: {metrics_skeleton['gpu_reserved_mb']:.2f} MB")
        
        # Scenario 3: Loading one parameter shard (simulate FSDP)
        print("\n" + "-" * 80)
        print("SCENARIO 3: Model Skeleton + 50% Parameter Shard (FSDP worker)")
        print("-" * 80)
        
        # Simulate loading 50% of parameters (one worker's shard)
        shard_percentage = 0.5
        param_names = list(state_dict.keys())
        shard_size = int(len(param_names) * shard_percentage)
        shard_param_names = param_names[:shard_size]
        
        # Load shard into skeleton
        with torch.no_grad():
            for name in shard_param_names:
                param = dict(skeleton.named_parameters())[name]
                param.data = state_dict[name].to(device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        after_shard_allocated = get_gpu_memory_allocated(device)
        after_shard_reserved = get_gpu_memory_reserved(device)
        
        shard_allocated = after_shard_allocated - metrics_skeleton['baseline_allocated_mb']
        shard_reserved = after_shard_reserved - metrics_skeleton['baseline_reserved_mb']
        
        # Calculate shard param memory
        shard_param_memory = sum(
            state_dict[name].numel() * state_dict[name].element_size()
            for name in shard_param_names
        ) / (1024 ** 2)
        
        print(f"Shard percentage: {shard_percentage * 100:.0f}%")
        print(f"Parameters in shard: {shard_size} / {len(param_names)}")
        print(f"Calculated shard memory: {shard_param_memory:.2f} MB")
        print(f"GPU allocated memory: {shard_allocated:.2f} MB")
        print(f"GPU reserved memory: {shard_reserved:.2f} MB")
        
        # Summary comparison
        print("\n" + "=" * 80)
        print("MEMORY SAVINGS SUMMARY")
        print("=" * 80)
        
        full_allocated = metrics_full['gpu_allocated_mb']
        skeleton_allocated = metrics_skeleton['gpu_allocated_mb']
        
        savings_skeleton = full_allocated - skeleton_allocated
        savings_percentage_skeleton = (savings_skeleton / full_allocated * 100) if full_allocated > 0 else 0
        
        savings_shard = full_allocated - shard_allocated
        savings_percentage_shard = (savings_shard / full_allocated * 100) if full_allocated > 0 else 0
        
        print(f"\n1. Full Model:")
        print(f"   GPU Memory: {full_allocated:.2f} MB")
        
        print(f"\n2. Skeleton (empty weights):")
        print(f"   GPU Memory: {skeleton_allocated:.2f} MB")
        print(f"   Savings: {savings_skeleton:.2f} MB ({savings_percentage_skeleton:.1f}%)")
        
        print(f"\n3. Skeleton + 50% Shard (FSDP worker):")
        print(f"   GPU Memory: {shard_allocated:.2f} MB")
        print(f"   Savings vs Full: {savings_shard:.2f} MB ({savings_percentage_shard:.1f}%)")
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS:")
        print("=" * 80)
        print("• Skeleton approach keeps structure but zeros weights (minimal memory)")
        print("• FSDP Stage 3 workers only keep their shard (~50% with 2 workers)")
        print("• Additional savings: optimizer states, gradients also sharded")
        print("• Peak memory during training: full model loaded temporarily during forward/backward")
        print("• Between steps: only skeleton + owned shard remains in memory")
        print("=" * 80 + "\n")
        
        # Assertions
        assert metrics_full['parameter_count'] > 0, \
            "Full model should have parameters"
        
        assert metrics_skeleton['parameter_count'] == metrics_full['parameter_count'], \
            "Skeleton should maintain same parameter structure (zeroed tensors)"
        
        assert skeleton_allocated < full_allocated, \
            "Skeleton should use less memory than full model"
        
        assert shard_allocated <= full_allocated / 2 + 10, \
            f"Skeleton + 50% shard should use approximately half memory of full model (expected ~{full_allocated/2:.2f} MB, got {shard_allocated:.2f} MB)"
        
        # Clean up
        del skeleton, state_dict
        clear_gpu_cache(device)
    
    def test_optimizer_memory_with_skeleton(self, device, model_config):
        """
        Test optimizer memory footprint with full model vs skeleton + shard.
        
        This demonstrates additional memory savings from sharded optimizer states.
        """
        print("\n" + "=" * 80)
        print("OPTIMIZER MEMORY TEST: Full vs Sharded")
        print("=" * 80)
        
        if device.type == "cpu":
            pytest.skip("Memory tracking not reliable on CPU device")
        
        # Create model
        model = BaseTransformer(
            vocab_size=model_config["vocab_size"],
            max_seq_len=model_config["max_seq_len"],
            model_dim=model_config["model_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ff_dim=model_config["ff_dim"],
            dropout=model_config["dropout"],
        )
        
        model = model.to(device)
        param_memory = get_parameter_memory_mb(model)
        
        # Test 1: Optimizer for full model
        print("\n1. FULL MODEL OPTIMIZER (AdamW):")
        clear_gpu_cache(device)
        baseline = get_gpu_memory_allocated(device)
        
        optimizer_full = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        after_optimizer = get_gpu_memory_allocated(device)
        optimizer_memory_full = after_optimizer - baseline
        
        print(f"   Parameter memory: {param_memory:.2f} MB")
        print(f"   Optimizer memory: {optimizer_memory_full:.2f} MB")
        print(f"   Total: {param_memory + optimizer_memory_full:.2f} MB")
        print(f"   Optimizer overhead: {(optimizer_memory_full / param_memory):.2f}x params")
        
        # Test 2: Optimizer for 50% shard
        print("\n2. SHARDED OPTIMIZER (50% parameters, FSDP-style):")
        
        # Get 50% of parameters
        all_params = list(model.parameters())
        shard_size = len(all_params) // 2
        shard_params = all_params[:shard_size]
        
        # Calculate shard memory
        shard_memory = sum(p.numel() * p.element_size() for p in shard_params) / (1024 ** 2)
        
        clear_gpu_cache(device)
        baseline_shard = get_gpu_memory_allocated(device)
        
        optimizer_shard = torch.optim.AdamW(shard_params, lr=0.001)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        after_optimizer_shard = get_gpu_memory_allocated(device)
        optimizer_memory_shard = after_optimizer_shard - baseline_shard
        
        print(f"   Parameter memory: {shard_memory:.2f} MB (50% of full)")
        print(f"   Optimizer memory: {optimizer_memory_shard:.2f} MB")
        print(f"   Total: {shard_memory + optimizer_memory_shard:.2f} MB")
        
        # Summary
        total_full = param_memory + optimizer_memory_full
        total_shard = shard_memory + optimizer_memory_shard
        savings = total_full - total_shard
        savings_pct = (savings / total_full * 100) if total_full > 0 else 0
        
        print("\n" + "-" * 80)
        print("MEMORY SAVINGS:")
        print(f"   Full model + optimizer: {total_full:.2f} MB")
        print(f"   Sharded (50%) + optimizer: {total_shard:.2f} MB")
        print(f"   Savings: {savings:.2f} MB ({savings_pct:.1f}%)")
        print("-" * 80 + "\n")
        
        # Assertions
        assert total_shard < total_full, "Sharded approach should use less total memory"
        
        # Clean up
        del model, optimizer_full, optimizer_shard
        clear_gpu_cache(device)

    def test_full_training_memory_breakdown(self, device, model_config):
        """
        Comprehensive memory analysis of full model during training.
        
        This test shows the complete memory footprint at each stage of training:
        1. Model parameters
        2. Optimizer states (AdamW: 2x parameters for momentum + variance)
        3. Gradients (1x parameters)
        4. Activations during forward pass
        5. Peak memory during backward pass
        
        This demonstrates what actually happens during training on GPU.
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TRAINING MEMORY ANALYSIS")
        print("=" * 80)
        
        # Skip if CPU
        if device.type == "cpu":
            pytest.skip("Memory tracking not reliable on CPU device")
        
        # Configuration
        batch_size = 4
        seq_len = 128
        print(f"\nDevice: {device}")
        print(f"Model config: {model_config}")
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        
        # Stage 0: Baseline (empty GPU)
        print("\n" + "-" * 80)
        print("STAGE 0: GPU Baseline (empty)")
        print("-" * 80)
        clear_gpu_cache(device)
        baseline = get_gpu_memory_allocated(device)
        print(f"Baseline GPU memory: {baseline:.2f} MB")
        
        # Stage 1: Load model
        print("\n" + "-" * 80)
        print("STAGE 1: Model Parameters")
        print("-" * 80)
        model = BaseTransformer(
            vocab_size=model_config["vocab_size"],
            max_seq_len=model_config["max_seq_len"],
            model_dim=model_config["model_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            ff_dim=model_config["ff_dim"],
            dropout=model_config["dropout"],
        ).to(device)
        
        param_count = count_parameters(model)
        after_model = get_gpu_memory_allocated(device)
        model_memory = after_model - baseline
        
        print(f"Parameter count: {param_count:,}")
        print(f"Model memory: {model_memory:.2f} MB")
        print(f"Total GPU allocated: {after_model:.2f} MB")
        
        # Breakdown by component
        print("\nModel component breakdown:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            module_mb = module_params * 4 / (1024 ** 2)  # float32
            print(f"  {name:20s}: {module_params:>10,} params ({module_mb:>6.2f} MB)")
        
        # Stage 2: Create optimizer
        print("\n" + "-" * 80)
        print("STAGE 2: Optimizer States (AdamW)")
        print("-" * 80)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        after_optimizer = get_gpu_memory_allocated(device)
        optimizer_memory = after_optimizer - after_model
        
        print(f"AdamW states: 2 buffers per parameter (momentum + variance)")
        print(f"Expected optimizer memory: ~{model_memory * 2:.2f} MB (2x model)")
        print(f"Actual optimizer memory: {optimizer_memory:.2f} MB")
        print(f"Total GPU allocated: {after_optimizer:.2f} MB")
        
        # Stage 3: Forward pass (creates activations)
        print("\n" + "-" * 80)
        print("STAGE 3: Forward Pass (Activations)")
        print("-" * 80)
        model.train()
        
        # Create dummy input
        input_ids = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len), device=device)
        targets = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len), device=device)
        
        before_forward = get_gpu_memory_allocated(device)
        logits = model(input_ids)
        after_forward = get_gpu_memory_allocated(device)
        
        activation_memory = after_forward - before_forward
        print(f"Activation memory: {activation_memory:.2f} MB")
        print(f"Total GPU allocated: {after_forward:.2f} MB")
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        after_loss = get_gpu_memory_allocated(device)
        loss_memory = after_loss - after_forward
        print(f"Loss computation memory: {loss_memory:.2f} MB")
        print(f"Total GPU allocated: {after_loss:.2f} MB")
        
        # Stage 4: Backward pass (creates gradients)
        print("\n" + "-" * 80)
        print("STAGE 4: Backward Pass (Gradients)")
        print("-" * 80)
        before_backward = get_gpu_memory_allocated(device)
        loss.backward()
        after_backward = get_gpu_memory_allocated(device)
        
        gradient_memory = after_backward - before_backward
        peak_memory = after_backward
        
        print(f"Gradient memory: {gradient_memory:.2f} MB")
        print(f"Expected gradient memory: ~{model_memory:.2f} MB (1x model)")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
        
        # Stage 5: Optimizer step
        print("\n" + "-" * 80)
        print("STAGE 5: Optimizer Step")
        print("-" * 80)
        before_step = get_gpu_memory_allocated(device)
        optimizer.step()
        optimizer.zero_grad()
        after_step = get_gpu_memory_allocated(device)
        
        step_memory = after_step - before_step
        print(f"Memory change during step: {step_memory:+.2f} MB")
        print(f"Memory after zero_grad: {after_step:.2f} MB")
        
        # Summary
        print("\n" + "=" * 80)
        print("MEMORY BREAKDOWN SUMMARY")
        print("=" * 80)
        print(f"\n{'Component':<30s} {'Memory (MB)':<15s} {'% of Total'}")
        print("-" * 80)
        
        total = peak_memory - baseline
        print(f"{'Model parameters':<30s} {model_memory:>10.2f} MB   {model_memory/total*100:>6.1f}%")
        print(f"{'Optimizer states (2x)':<30s} {optimizer_memory:>10.2f} MB   {optimizer_memory/total*100:>6.1f}%")
        print(f"{'Activations (forward)':<30s} {activation_memory:>10.2f} MB   {activation_memory/total*100:>6.1f}%")
        print(f"{'Gradients (backward)':<30s} {gradient_memory:>10.2f} MB   {gradient_memory/total*100:>6.1f}%")
        print("-" * 80)
        print(f"{'PEAK TRAINING MEMORY':<30s} {total:>10.2f} MB   100.0%")
        print("=" * 80)
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS FOR DISTRIBUTED TRAINING")
        print("=" * 80)
        print(f"• Model parameters: {model_memory:.2f} MB ({model_memory/total*100:.1f}% of peak)")
        print(f"• Optimizer states: {optimizer_memory:.2f} MB (AdamW keeps 2 states per param)")
        print(f"• Gradients: {gradient_memory:.2f} MB (same size as parameters)")
        print(f"• Activations: {activation_memory:.2f} MB (depends on batch size & seq length)")
        print(f"\n• FSDP Stage 1 (optimizer sharding): Saves {optimizer_memory/2:.2f} MB (~{optimizer_memory/total/2*100:.1f}% with 2 workers)")
        print(f"• FSDP Stage 2 (+ gradient sharding): Saves {(optimizer_memory + gradient_memory)/2:.2f} MB (~{(optimizer_memory + gradient_memory)/total/2*100:.1f}% with 2 workers)")
        print(f"• FSDP Stage 3 (+ param sharding): Saves {(model_memory + optimizer_memory + gradient_memory)/2:.2f} MB (~{(model_memory + optimizer_memory + gradient_memory)/total/2*100:.1f}% with 2 workers)")
        print(f"  Note: Parameters are loaded temporarily during forward/backward")
        print("=" * 80 + "\n")
        
        # Assertions
        assert model_memory > 0, "Model should have parameters"
        assert optimizer_memory > model_memory, "AdamW should use more memory than model (2 states)"
        assert gradient_memory > 0, "Gradients should be allocated after backward pass"
        assert after_step <= peak_memory, "Memory should not increase after optimizer step"
        
        # Clean up
        del model, optimizer, logits, loss
        clear_gpu_cache(device)


if __name__ == "__main__":
    """Run tests directly with detailed output."""
    pytest.main([__file__, "-v", "-s"])
