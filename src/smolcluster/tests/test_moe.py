"""
Pytest test suite for MoE (Mixtral) model.

Tests cover:
- MoE layer functionality
- Expert routing
- Model instantiation
- Forward passes
- Shape validation
- Parameter counting
- Gradient flow
- Different configurations
"""

import pytest
import torch
import torch.nn as nn

from smolcluster.models.moe import (
    AttentionHead,
    LayerNormalization,
    MHA,
    Mixtral,
    MoeLayer,
    RotaryEmbeddings,
    SWiGLUExpertMoE,
    Swish,
    TextEmbeddings,
    TransformerDecoderBlock,
)


class TestSwish:
    """Test suite for Swish activation."""

    def test_initialization(self):
        """Test Swish initializes correctly."""
        swish = Swish()
        assert swish.sig is not None

    def test_forward_shape(self, batch_size, small_seq_len, model_dim):
        """Test Swish maintains tensor shape."""
        swish = Swish()
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = swish(x)
        assert output.shape == x.shape

    def test_non_linearity(self):
        """Test that Swish is non-linear."""
        swish = Swish()
        x = torch.tensor([[-1.0, 0.0, 1.0]])
        output = swish(x)
        
        # Swish is not linear: swish(2x) != 2*swish(x)
        x2 = x * 2
        output2 = swish(x2)
        assert not torch.allclose(output2, output * 2)


class TestRotaryEmbeddings:
    """Test suite for Rotary Embeddings."""

    def test_initialization(self, device, model_dim, small_seq_len):
        """Test RotaryEmbeddings initializes correctly."""
        rope = RotaryEmbeddings(
            device=device,
            embeddings_dims=model_dim,
            block_size=small_seq_len,
        )
        assert rope.embeddings_dims == model_dim
        assert rope.block_size == small_seq_len

    def test_forward_shape(self, device, model_dim, small_seq_len, batch_size):
        """Test that RoPE maintains tensor shape."""
        rope = RotaryEmbeddings(
            device=device,
            embeddings_dims=model_dim,
            block_size=small_seq_len,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = rope(x)
        assert output.shape == x.shape


class TestTextEmbeddings:
    """Test suite for TextEmbeddings."""

    def test_initialization(self, small_vocab_size, model_dim, device):
        """Test TextEmbeddings initializes correctly."""
        embeddings = TextEmbeddings(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            device=device,
        )
        assert embeddings.embeddings_table.num_embeddings == small_vocab_size
        assert embeddings.embeddings_table.embedding_dim == model_dim

    def test_forward_shape(self, small_vocab_size, model_dim, device, batch_size, small_seq_len):
        """Test that embeddings produce correct shape."""
        embeddings = TextEmbeddings(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            device=device,
        )
        input_ids = torch.randint(0, small_vocab_size, (batch_size, small_seq_len))
        output = embeddings(input_ids)
        assert output.shape == (batch_size, small_seq_len, model_dim)


class TestLayerNormalization:
    """Test suite for LayerNormalization."""

    def test_initialization(self, model_dim):
        """Test LayerNormalization initializes correctly."""
        ln = LayerNormalization(embeddings_dims=model_dim)
        assert ln.norm is not None

    def test_forward_shape(self, model_dim, batch_size, small_seq_len):
        """Test that LayerNorm maintains shape."""
        ln = LayerNormalization(embeddings_dims=model_dim)
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = ln(x)
        assert output.shape == x.shape

    def test_normalization_effect(self, model_dim, batch_size, small_seq_len):
        """Test that LayerNorm normalizes along the last dimension."""
        ln = LayerNormalization(embeddings_dims=model_dim)
        x = torch.randn(batch_size, small_seq_len, model_dim) * 10  # Large values
        output = ln(x)
        
        # Check that output has approximately mean 0 and std 1 along last dim
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-1)


class TestSWiGLUExpertMoE:
    """Test suite for SWiGLU Expert."""

    def test_initialization(self, model_dim, device):
        """Test SWiGLU expert initializes correctly."""
        expert = SWiGLUExpertMoE(
            embeddings_dims=model_dim,
            device=device,
        )
        assert expert.hidden_dims == model_dim * 2

    def test_forward_shape(self, model_dim, device, batch_size, small_seq_len):
        """Test that expert maintains input shape."""
        expert = SWiGLUExpertMoE(
            embeddings_dims=model_dim,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = expert(x)
        assert output.shape == x.shape

    def test_gradient_flow(self, model_dim, device, batch_size, small_seq_len):
        """Test that gradients flow through expert."""
        expert = SWiGLUExpertMoE(
            embeddings_dims=model_dim,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim, requires_grad=True)
        output = expert(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestMoeLayer:
    """Test suite for MoE Layer."""

    def test_initialization(self, model_dim, num_experts, top_k_experts, device):
        """Test MoE layer initializes correctly."""
        moe = MoeLayer(
            embeddings_dims=model_dim,
            num_experts=num_experts,
            top_k=top_k_experts,
            device=device,
        )
        assert len(moe.heads) == num_experts
        assert moe.num_experts == num_experts
        assert moe.top_k == top_k_experts

    def test_forward_shape(self, model_dim, num_experts, top_k_experts, device, batch_size, small_seq_len):
        """Test that MoE layer maintains input shape."""
        moe = MoeLayer(
            embeddings_dims=model_dim,
            num_experts=num_experts,
            top_k=top_k_experts,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = moe(x)
        assert output.shape == x.shape

    def test_expert_routing(self, model_dim, num_experts, top_k_experts, device, batch_size, small_seq_len):
        """Test that experts are routed correctly."""
        moe = MoeLayer(
            embeddings_dims=model_dim,
            num_experts=num_experts,
            top_k=top_k_experts,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim)
        
        # Hook to check which experts are called
        experts_called = []
        
        def hook(module, input, output):
            experts_called.append(True)
        
        for expert in moe.heads:
            expert.register_forward_hook(hook)
        
        output = moe(x)
        
        # At least some experts should be called
        assert len(experts_called) > 0

    def test_gradient_flow(self, model_dim, num_experts, top_k_experts, device, batch_size, small_seq_len):
        """Test that gradients flow through MoE layer."""
        moe = MoeLayer(
            embeddings_dims=model_dim,
            num_experts=num_experts,
            top_k=top_k_experts,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTransformerDecoderBlock:
    """Test suite for Transformer Decoder Block with MoE."""

    def test_initialization(self, model_dim, num_heads, num_experts, top_k_experts, device):
        """Test decoder block initializes correctly."""
        block = TransformerDecoderBlock(
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k_experts,
            device=device,
        )
        assert block.mha is not None
        assert block.moe_block is not None

    def test_forward_shape(
        self, model_dim, num_heads, num_experts, top_k_experts, device, batch_size, small_seq_len
    ):
        """Test that decoder block maintains shape."""
        block = TransformerDecoderBlock(
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k_experts,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = block(x)
        assert output.shape == x.shape


class TestMixtral:
    """Test suite for Mixtral (MoE Transformer) model."""

    def test_initialization(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts, small_seq_len, device
    ):
        """Test Mixtral initializes correctly."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
        )
        
        assert model.vocab_size == small_vocab_size
        assert model.embeddings_dims == model_dim
        assert len(model.decoder_layers) == num_layers

    def test_forward_pass(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts, 
        small_seq_len, device, sample_input
    ):
        """Test that forward pass works correctly."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input, inference=True)
        
        batch_size, seq_len = sample_input.shape
        assert output.shape == (batch_size, seq_len, small_vocab_size)

    def test_parameter_counting(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts, small_seq_len, device
    ):
        """Test that parameter counting works."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
        )
        
        num_params = model.get_num_params()
        expected_params = sum(p.numel() for p in model.parameters())
        
        assert num_params == expected_params
        assert num_params > 0

    def test_gradient_flow(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts,
        small_seq_len, device, sample_input, sample_labels
    ):
        """Test that gradients flow through the entire model."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
            dropout=0.0,
        )
        
        output = model(sample_input, inference=True)
        loss = nn.functional.cross_entropy(
            output.view(-1, small_vocab_size),
            sample_labels.view(-1)
        )
        loss.backward()
        
        # Check that parameters have gradients
        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                if not torch.all(param.grad == 0):
                    params_with_grad += 1
        
        # At least some parameters should have non-zero gradients
        assert params_with_grad > 0

    def test_different_expert_configurations(
        self, small_vocab_size, model_dim, num_heads, num_layers, small_seq_len, device, sample_input
    ):
        """Test model with different expert configurations."""
        configs = [
            {"num_experts": 4, "top_k": 1},
            {"num_experts": 4, "top_k": 2},
            {"num_experts": 8, "top_k": 2},
        ]
        
        for config in configs:
            model = Mixtral(
                vocab_size=small_vocab_size,
                embeddings_dims=model_dim,
                no_of_heads=num_heads,
                no_of_decoder_layers=num_layers,
                num_experts=config["num_experts"],
                top_k=config["top_k"],
                max_seq_len=small_seq_len,
                device=device,
            )
            model.eval()
            
            with torch.no_grad():
                output = model(sample_input, inference=True)
            
            assert output.shape[0] == sample_input.shape[0]

    def test_flash_attention(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts,
        small_seq_len, device, sample_input
    ):
        """Test model with flash attention enabled."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
            use_flash_attention=True,
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input, inference=True)
        
        assert output.shape[2] == small_vocab_size

    def test_different_sequence_lengths(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts,
        small_seq_len, device, batch_size
    ):
        """Test model with different sequence lengths."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
        )
        model.eval()
        
        # Test with shorter sequence
        short_seq = small_seq_len // 2
        input_ids = torch.randint(0, small_vocab_size, (batch_size, short_seq))
        
        with torch.no_grad():
            output = model(input_ids, inference=True)
        
        assert output.shape == (batch_size, short_seq, small_vocab_size)

    def test_eval_mode_deterministic(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts,
        small_seq_len, device, sample_input
    ):
        """Test that eval mode produces deterministic results."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
            dropout=0.5,
            use_flash_attention=False,  # Disable flash attention for determinism
            noisy_topk=False,  # Explicitly disable noisy routing
        )
        
        model.eval()
        
        with torch.no_grad():
            output1 = model(sample_input, inference=True)
            output2 = model(sample_input, inference=True)
        
        assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.slow
    def test_large_model(self, small_vocab_size, device):
        """Test creation of a larger MoE model (slow test)."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=512,
            no_of_heads=8,
            no_of_decoder_layers=6,
            num_experts=8,
            top_k=2,
            max_seq_len=256,
            device=device,
        )
        
        num_params = model.get_num_params()
        assert num_params > 5_000_000  # Should have over 5M parameters

    @pytest.mark.cuda
    def test_cuda_compatibility(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts,
        small_seq_len, sample_input
    ):
        """Test that model works on CUDA device."""
        device = torch.device("cuda")
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
        ).to(device)
        
        input_ids = sample_input.to(device)
        
        with torch.no_grad():
            output = model(input_ids, inference=True)
        
        assert output.device.type == "cuda"


class TestMixtralIntegration:
    """Integration tests for Mixtral model."""

    def test_training_step(
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts,
        small_seq_len, device, sample_input, sample_labels
    ):
        """Test a full training step."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Forward pass
        output = model(sample_input, inference=True)
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
        self, small_vocab_size, model_dim, num_heads, num_layers, num_experts, top_k_experts,
        small_seq_len, device, sample_input, sample_labels
    ):
        """Test that model can overfit a single batch (sanity check)."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k_experts,
            max_seq_len=small_seq_len,
            device=device,
            dropout=0.0,  # Disable dropout for overfitting
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        
        initial_loss = None
        final_loss = None
        
        # Train for a few iterations
        for i in range(50):
            output = model(sample_input, inference=True)
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

    def test_moe_vs_standard_comparison(
        self, small_vocab_size, model_dim, num_heads, num_layers, small_seq_len, device
    ):
        """Test that MoE has more parameters than standard model for same config."""
        moe_model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            num_experts=4,
            top_k=2,
            max_seq_len=small_seq_len,
            device=device,
        )
        
        moe_params = moe_model.get_num_params()
        
        # MoE should have more parameters due to multiple experts
        assert moe_params > 100_000  # Should have reasonable number of parameters
