"""
Pytest test suite for MoE (Mixtral) model.

Tests cover:
- Expert routing
- Model instantiation
- Forward passes
- Shape validation
- Gradient flow
- Different configurations
"""

import pytest
import torch
import torch.nn as nn

from smolcluster.models.moe import (
    LayerNormalization,
    Mixtral,
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


class TestTransformerDecoderBlock:
    """Test suite for Transformer Decoder Block (attention-only; MoE is distributed externally)."""

    def test_initialization(self, model_dim, num_heads, device):
        """Test decoder block initializes correctly."""
        block = TransformerDecoderBlock(
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            device=device,
        )
        assert block.mha is not None
        assert block.layer_norm1 is not None

    def test_forward_shape(self, model_dim, num_heads, device, batch_size, small_seq_len):
        """Test that decoder block maintains shape."""
        block = TransformerDecoderBlock(
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim)
        output = block(x)
        assert output.shape == x.shape

    def test_gradient_flow(self, model_dim, num_heads, device, batch_size, small_seq_len):
        """Test that gradients flow through decoder block."""
        block = TransformerDecoderBlock(
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            device=device,
        )
        x = torch.randn(batch_size, small_seq_len, model_dim, requires_grad=True)
        output = block(x)
        output.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestMixtral:
    """Test suite for Mixtral (last-rank transformer).

    Mixtral now accepts expert-processed activations [batch, seq, embedding_dims],
    NOT raw token indices. Embeddings and routing live on rank 0.
    """

    def _make_activations(self, batch_size, seq_len, model_dim):
        return torch.randn(batch_size, seq_len, model_dim)

    def test_initialization(self, small_vocab_size, model_dim, num_heads, num_layers, device):
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
        )
        assert model.vocab_size == small_vocab_size
        assert model.embeddings_dims == model_dim
        assert len(model.decoder_layers) == num_layers

    def test_forward_shape(self, small_vocab_size, model_dim, num_heads, num_layers, device, batch_size, small_seq_len):
        """Forward pass accepts activations and returns logits."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
        )
        model.eval()
        activations = self._make_activations(batch_size, small_seq_len, model_dim)
        with torch.no_grad():
            output = model(activations)
        assert output.shape == (batch_size, small_seq_len, small_vocab_size)

    def test_parameter_counting(self, small_vocab_size, model_dim, num_heads, num_layers, device):
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
        )
        num_params = model.get_num_params()
        assert num_params == sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_gradient_flow(self, small_vocab_size, model_dim, num_heads, num_layers, device, batch_size, small_seq_len, sample_labels):
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
            dropout=0.0,
        )
        activations = self._make_activations(batch_size, small_seq_len, model_dim)
        output = model(activations)
        loss = nn.functional.cross_entropy(output.view(-1, small_vocab_size), sample_labels.view(-1))
        loss.backward()

        params_with_grad = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None and not torch.all(p.grad == 0)
        )
        assert params_with_grad > 0

    def test_different_sequence_lengths(self, small_vocab_size, model_dim, num_heads, num_layers, device, batch_size, small_seq_len):
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
        )
        model.eval()
        short_seq = small_seq_len // 2
        activations = self._make_activations(batch_size, short_seq, model_dim)
        with torch.no_grad():
            output = model(activations)
        assert output.shape == (batch_size, short_seq, small_vocab_size)

    def test_eval_mode_deterministic(self, small_vocab_size, model_dim, num_heads, num_layers, device, batch_size, small_seq_len):
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
            dropout=0.5,
        )
        model.eval()
        activations = self._make_activations(batch_size, small_seq_len, model_dim)
        with torch.no_grad():
            out1 = model(activations)
            out2 = model(activations)
        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.slow
    def test_large_model(self, small_vocab_size, device):
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=512,
            no_of_heads=8,
            no_of_decoder_layers=6,
            device=device,
        )
        assert model.get_num_params() > 5_000_000

    @pytest.mark.cuda
    def test_cuda_compatibility(self, small_vocab_size, model_dim, num_heads, num_layers, batch_size, small_seq_len):
        device = torch.device("cuda")
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
        ).to(device)
        activations = torch.randn(batch_size, small_seq_len, model_dim, device=device)
        with torch.no_grad():
            output = model(activations)
        assert output.device.type == "cuda"


class TestMixtralIntegration:
    """Integration tests for Mixtral model."""

    def test_training_step(self, small_vocab_size, model_dim, num_heads, num_layers, device, batch_size, small_seq_len, sample_labels):
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        activations = torch.randn(batch_size, small_seq_len, model_dim)
        output = model(activations)
        loss = nn.functional.cross_entropy(output.view(-1, small_vocab_size), sample_labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        assert loss.item() > 0

    def test_overfitting_single_batch(self, small_vocab_size, model_dim, num_heads, num_layers, device, batch_size, small_seq_len, sample_labels):
        """Model can overfit a fixed activation batch (sanity check)."""
        model = Mixtral(
            vocab_size=small_vocab_size,
            embeddings_dims=model_dim,
            no_of_heads=num_heads,
            no_of_decoder_layers=num_layers,
            device=device,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        activations = torch.randn(batch_size, small_seq_len, model_dim)
        initial_loss = None
        for i in range(50):
            output = model(activations)
            loss = nn.functional.cross_entropy(output.view(-1, small_vocab_size), sample_labels.view(-1))
            if i == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        assert loss.item() < initial_loss * 0.5
