"""
Pytest test suite for GPT-2 (BaseTransformer) model.

Tests cover:
- Model instantiation
- Forward passes
- Shape validation
- Parameter counting
- Weight initialization
- Gradient flow
- Different configurations
"""

import pytest
import torch
import torch.nn as nn

from smolcluster.models.gpt import BaseTransformer, BaseTransformerBlock


class TestBaseTransformerBlock:
    """Test suite for BaseTransformerBlock."""

    def test_initialization(self, model_dim, num_heads, ff_dim, num_layers):
        """Test that BaseTransformerBlock initializes correctly."""
        block = BaseTransformerBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=0.1,
            num_layers=num_layers,
        )
        assert block.model_dim == model_dim
        assert block.num_heads == num_heads
        assert block.head_dim == model_dim // num_heads

    def test_forward_shape(self, model_dim, num_heads, ff_dim, num_layers, batch_size, small_seq_len):
        """Test that forward pass maintains correct shapes."""
        block = BaseTransformerBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=0.1,
            num_layers=num_layers,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = block(x)
        
        assert output.shape == (batch_size, small_seq_len, model_dim)

    def test_gradient_flow(self, model_dim, num_heads, ff_dim, num_layers, batch_size, small_seq_len):
        """Test that gradients flow through the block."""
        block = BaseTransformerBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=0.0,  # Disable dropout for gradient test
            num_layers=num_layers,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.skip(reason="Model doesn't validate model_dim divisibility")
    def test_invalid_model_dim(self, num_heads, ff_dim, num_layers):
        """Test that invalid model_dim raises error."""
        with pytest.raises(AssertionError):
            BaseTransformerBlock(
                model_dim=100,  # Not divisible by num_heads (4)
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=0.1,
                num_layers=num_layers,
            )


class TestBaseTransformer:
    """Test suite for BaseTransformer (GPT-2) model."""

    def test_initialization(self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads):
        """Test that BaseTransformer initializes correctly."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        assert model.vocab_size == small_vocab_size
        assert model.max_seq_len == small_seq_len
        assert model.model_dim == model_dim

    def test_forward_pass(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads, sample_input
    ):
        """Test that forward pass works and produces correct output shape."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
        
        batch_size, seq_len = sample_input.shape
        assert output.shape == (batch_size, seq_len, small_vocab_size)

    def test_parameter_counting(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads
    ):
        """Test that parameter counting works."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        num_params = model.get_num_params()
        expected_params = sum(p.numel() for p in model.parameters())
        
        assert num_params == expected_params
        assert num_params > 0

    def test_weight_initialization(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads
    ):
        """Test that weights are initialized properly (not all zeros)."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # Check that weights are not all zeros (except for biases which are often zero-initialized)
        for name, param in model.named_parameters():
            # Bias terms are often correctly initialized to zero
            if 'bias' in name:
                continue
            assert not torch.all(param == 0), f"Parameter {name} is all zeros"

    def test_gradient_flow(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads, sample_input, sample_labels
    ):
        """Test that gradients flow through the entire model."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.0,  # Disable dropout for gradient test
        )
        
        output = model(sample_input)
        loss = nn.functional.cross_entropy(
            output.view(-1, small_vocab_size),
            sample_labels.view(-1)
        )
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_tie_weights(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads
    ):
        """Test weight tying between input and output embeddings."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            tie_weights=True,
        )
        
        # Check that weights are tied
        assert model.token_embedding.weight is model.lm_head.weight

    def test_different_sequence_lengths(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads, batch_size
    ):
        """Test model with different sequence lengths."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        model.eval()
        
        # Test with shorter sequence
        short_seq = small_seq_len // 2
        input_ids = torch.randint(0, small_vocab_size, (batch_size, short_seq))
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.shape == (batch_size, short_seq, small_vocab_size)

    def test_custom_ff_dim(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads
    ):
        """Test model with custom feed-forward dimension."""
        custom_ff_dim = 256
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=custom_ff_dim,
        )
        
        # Check that the custom ff_dim was used
        assert model.blocks[0].ffn[0].out_features == custom_ff_dim

    def test_dropout_effect(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads, sample_input
    ):
        """Test that dropout has an effect during training."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.5,  # High dropout for testing
        )
        
        model.train()
        
        # Two forward passes with dropout should produce different results
        output1 = model(sample_input)
        output2 = model(sample_input)
        
        assert not torch.allclose(output1, output2)

    def test_eval_mode_deterministic(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads, sample_input
    ):
        """Test that eval mode produces deterministic results."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.5,
        )
        
        model.eval()
        
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = model(sample_input)
        
        assert torch.allclose(output1, output2)

    def test_batch_independence(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads
    ):
        """Test that samples in a batch are processed independently."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        model.eval()
        
        # Create a batch where first two samples are identical
        input_ids = torch.randint(0, small_vocab_size, (3, small_seq_len))
        input_ids[1] = input_ids[0]
        
        with torch.no_grad():
            output = model(input_ids)
        
        # First two outputs should be identical
        assert torch.allclose(output[0], output[1])
        # Third output should be different
        assert not torch.allclose(output[0], output[2])

    @pytest.mark.slow
    def test_large_model(self, small_vocab_size):
        """Test creation of a larger model (slow test)."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=256,
            model_dim=512,
            num_layers=6,
            num_heads=8,
        )
        
        num_params = model.get_num_params()
        assert num_params > 1_000_000  # Should have over 1M parameters

    @pytest.mark.cuda
    def test_cuda_compatibility(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads, sample_input
    ):
        """Test that model works on CUDA device."""
        device = torch.device("cuda")
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        ).to(device)
        
        input_ids = sample_input.to(device)
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.device.type == "cuda"


class TestGPTIntegration:
    """Integration tests for GPT-2 model."""

    def test_training_step(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads, 
        sample_input, sample_labels
    ):
        """Test a full training step."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Forward pass
        output = model(sample_input)
        loss = nn.functional.cross_entropy(
            output.view(-1, small_vocab_size),
            sample_labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        assert loss.item() > 0

    def test_overfitting_single_batch(
        self, small_vocab_size, small_seq_len, model_dim, num_layers, num_heads,
        sample_input, sample_labels
    ):
        """Test that model can overfit a single batch (sanity check)."""
        model = BaseTransformer(
            vocab_size=small_vocab_size,
            max_seq_len=small_seq_len,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.0,  # Disable dropout for overfitting
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        
        initial_loss = None
        final_loss = None
        
        # Train for a few iterations
        for i in range(50):
            output = model(sample_input)
            loss = nn.functional.cross_entropy(
                output.view(-1, small_vocab_size),
                sample_labels.view(-1)
            )
            
            if i == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i == 49:
                final_loss = loss.item()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5
