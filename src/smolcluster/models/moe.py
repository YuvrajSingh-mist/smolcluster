"""
Mixture of Experts (MoE) Transformer implementation (Mixtral-style).

This module contains:
- Mixtral: Decoder-only transformer (attention + layernorm + lm_head) that accepts
  expert-processed activations as input. Does NOT compute embeddings or MoE routing.
- Router: Routing mechanism for expert selection with optional noisy top-k
- ExpertBlock: Wrapper for SwiGLU MoE experts (distributed across workers)
- TextEmbeddings: Token embedding layer (lives on rank 0)
- SWiGLUExpertMoE: Expert implementation using SwiGLU activation
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbeddings(nn.Module):
    """Rotary Position Embeddings (RoPE) for attention."""

    def __init__(
        self,
        device: torch.device,
        embeddings_dims: int,
        block_size: int,
    ):
        """Initialize rotary embeddings.

        Args:
            device: Device to place tensors on.
            embeddings_dims: Dimension of embeddings.
            block_size: Maximum sequence length.
        """
        super().__init__()
        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.device = device

    def apply_rope(
        self, seq: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embeddings to input sequence.

        Args:
            seq: Input sequence tensor.


        Returns:
            Tensor with rotary embeddings applied.
        """
        batch_size, seq_len, embeds_dims = seq.shape

        token_idx = torch.arange(0, seq_len, device=self.device).unsqueeze(1)
        positions = torch.arange(0, embeds_dims, 2, device=self.device).unsqueeze(0)
        theta = 10000 ** (-2 * positions / embeds_dims)
        angles = token_idx * theta
        angles = angles.expand(seq_len, -1)
        x_reshaped = seq.view(batch_size, seq_len, embeds_dims // 2, 2)

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        out = torch.stack(
            [
                x_reshaped[..., 0] * cos_angles - (x_reshaped[..., 1] * sin_angles),
                x_reshaped[..., 1] * cos_angles + x_reshaped[..., 0] * sin_angles,
            ],
            dim=-1,
        )
        out = out.view(batch_size, seq_len, embeds_dims)
        return out

    def forward(
        self, x: torch.Tensor, q: torch.Tensor | None = None, k: torch.Tensor | None = None
    ):
        """Forward pass for rotary embeddings.

        Args:
            x: Input tensor (used for shape reference when q and k are provided).
            q: Query tensor (optional).
            k: Key tensor (optional).

        Returns:
            Tensor with rotary embeddings applied, or tuple of (q, k) if both provided.
        """
        if q is not None and k is not None:
            # Flash attention case: apply to both q and k
            batch_size, num_heads, seq_len, head_dim = q.shape

            # Create rotary embeddings for this sequence length
            token_idx = torch.arange(0, seq_len, device=self.device).unsqueeze(1)
            positions = torch.arange(0, head_dim, 2, device=self.device).unsqueeze(0)
            theta = 10000 ** (-2 * positions / head_dim)
            angles = token_idx * theta

            cos_angles = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
            sin_angles = torch.sin(angles).unsqueeze(0).unsqueeze(0)

            # Apply to q
            q_reshaped = q.reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)
            q_rotated = torch.stack(
                [
                    q_reshaped[..., 0] * cos_angles - q_reshaped[..., 1] * sin_angles,
                    q_reshaped[..., 1] * cos_angles + q_reshaped[..., 0] * sin_angles,
                ],
                dim=-1,
            ).reshape(batch_size, num_heads, seq_len, head_dim)

            # Apply to k
            k_reshaped = k.reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)
            k_rotated = torch.stack(
                [
                    k_reshaped[..., 0] * cos_angles - k_reshaped[..., 1] * sin_angles,
                    k_reshaped[..., 1] * cos_angles + k_reshaped[..., 0] * sin_angles,
                ],
                dim=-1,
            ).reshape(batch_size, num_heads, seq_len, head_dim)

            return q_rotated, k_rotated
        else:
            # Non-flash attention case: apply to single tensor
            return self.apply_rope(x)


class TextEmbeddings(nn.Module):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, embeddings_dims: int, device: torch.device):
        """Initialize text embeddings.

        Args:
            vocab_size: Size of vocabulary.
            embeddings_dims: Dimension of embeddings.
            device: Device to place tensors on.
        """
        super().__init__()
        self.embeddings_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embeddings_dims, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for text embeddings.

        Args:
            x: Input token indices.

        Returns:
            Embedded tokens.
        """
        return self.embeddings_table(x)


class LayerNormalization(nn.Module):
    """Layer normalization."""

    def __init__(self, embeddings_dims: int):
        """Initialize layer normalization.

        Args:
            embeddings_dims: Dimension of embeddings.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embeddings_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return self.norm(x)


class Swish(nn.Module):
    """Swish activation function."""

    def __init__(self):
        """Initialize Swish activation."""
        super().__init__()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Swish.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return x * self.sig(x)


class SWiGLUExpertMoE(nn.Module):
    """Expert network using SwiGLU activation."""

    def __init__(
        self,
        embeddings_dims: int,
        device: torch.device,
    ):
        """Initialize SwiGLU expert.

        Args:
            embeddings_dims: Dimension of embeddings.
            device: Device to place tensors on.
        """
        super().__init__()
        self.hidden_dims = embeddings_dims * 2

        self.swish = Swish()
        self.linear_layer1 = nn.Linear(
            in_features=embeddings_dims, out_features=self.hidden_dims, bias=False, device=device
        )
        self.linear_layer2 = nn.Linear(
            in_features=embeddings_dims, out_features=self.hidden_dims, bias=False, device=device
        )
        self.linear_layer3 = nn.Linear(
            in_features=self.hidden_dims, out_features=embeddings_dims, bias=False, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwiGLU expert.

        Args:
            x: Input tensor.

        Returns:
            Expert output.
        """
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out


class ExpertBlock(nn.Module):
    """Expert block wrapper for SwiGLU MoE expert."""

    def __init__(
        self,
        embeddings_dims: int,
        device: torch.device,
    ):
        """Initialize expert block.

        Args:
            embeddings_dims: Dimension of embeddings.
            device: Device to place tensors on.
        """
        super().__init__()
        self.expert = SWiGLUExpertMoE(
            embeddings_dims=embeddings_dims,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for expert block.

        Args:
            x: Input tensor.

        Returns:
            Expert output.
        """
        return self.expert(x)


class Router(nn.Module):
    """Router for mixture of experts with top-k routing."""

    def __init__(
        self,
        embeddings_dims: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        noisy_topk: bool = False,
    ):
        """Initialize router.

        Args:
            embeddings_dims: Dimension of embeddings.
            num_experts: Number of experts.
            top_k: Number of experts to route to.
            device: Device to place tensors on.
            noisy_topk: Whether to use noisy top-k gating.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_topk = noisy_topk
        self.device = device

        self.gate = nn.Linear(
            in_features=embeddings_dims,
            out_features=num_experts,
            device=device,
            bias=False,
        )
        if noisy_topk:
            self.noise = nn.Linear(
                in_features=embeddings_dims,
                out_features=num_experts,
                device=device,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for router.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embeddings_dims).

        Returns:
            Tuple of (routing_probabilities, expert_indices):
                - routing_probabilities: Softmax probabilities for top-k experts [batch_size, seq_len, top_k]
                - expert_indices: Indices of top-k experts [batch_size, seq_len, top_k]
        """
        gate_out = self.gate(x)  # [batch_size, seq_len, num_experts]

        if self.noisy_topk:
            noise = self.noise(x)
            gaussian_noise = torch.normal(0, 1, size=gate_out.shape, device=self.device)
            noisy_router = F.softplus(noise) * gaussian_noise
            noisy_router += gate_out
        else:
            noisy_router = gate_out

        # Get top-k experts
        top_k_values, top_k_indices = torch.topk(
            noisy_router, k=self.top_k
        )  # [batch_size, seq_len, top_k]
        probs = F.softmax(top_k_values, dim=-1)  # [batch_size, seq_len, top_k]

        return probs, top_k_indices



class AttentionHead(nn.Module):
    """Single attention head with optional flash attention and RoPE."""

    def __init__(
        self,
        embeddings_dims: int,
        no_of_heads: int,
        device: torch.device,
        attn_dropout: float = 0.1,
    ):
        """Initialize attention head.

        Args:
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.
        """
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.no_of_heads = no_of_heads
        self.dropout_p = attn_dropout
        self.device = device


        self.rotary = RotaryEmbeddings(
            embeddings_dims=embeddings_dims, device=device, block_size=2048
        )
        self.keys = nn.Linear(embeddings_dims, self.head_size, bias=False, device=device)
        self.query = nn.Linear(embeddings_dims, self.head_size, bias=False, device=device)
        self.values = nn.Linear(embeddings_dims, self.head_size, bias=False, device=device)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for attention head.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embeddings_dims).

        Returns:
            Output tensor after attention.
        """
        _, block_size, _ = x.shape

        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        q = self.rotary(q)
        k = self.rotary(k)
        masked_table = torch.tril(torch.ones(block_size, block_size, device=self.device))
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        masked_values = weights.masked_fill(masked_table[:block_size, :block_size] == 0, float("-inf"))
        weights_normalized = F.softmax(masked_values, dim=-1)
        weights_normalized = self.dropout(weights_normalized)
        out = weights_normalized @ v
        return out



class MHA(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        embeddings_dims: int,
        no_of_heads: int,
        device: torch.device,
        attn_dropout: float = 0.1,

    ):
        """Initialize multi-head attention.

        Args:
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.

        """
        super().__init__()
        self.no_of_heads = no_of_heads


        # Non-flash attention uses separate heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    embeddings_dims=embeddings_dims,
                    no_of_heads=no_of_heads,
                    device=device,
                    attn_dropout=attn_dropout,
                )
                for _ in range(no_of_heads)
            ]
        )

        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(
            in_features=embeddings_dims,  # heads output head_size each (or full dim for flash), concatenated to embeddings_dims
            out_features=embeddings_dims,
            device=device,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after multi-head attention.
        """
        concat = torch.cat([head(x) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with attention and layer normalization.

    MoE experts are distributed across workers and processed externally.
    This block handles only attention and residual connection.
    """

    def __init__(
        self,
        embeddings_dims: int,
        no_of_heads: int,
        device: torch.device,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        """Initialize transformer decoder block.

        Args:
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.
            dropout: General dropout probability.
        """
        super().__init__()
        self.mha = MHA(
            embeddings_dims=embeddings_dims,
            no_of_heads=no_of_heads,
            device=device,
            attn_dropout=attn_dropout,
        )
        self.layer_norm1 = LayerNormalization(embeddings_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer decoder block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after attention.
        """
        x = x + self.mha(self.layer_norm1(x))
        return x


class Mixtral(nn.Module):
    """Mixtral: decoder-only transformer for the last rank in expert parallelism.

    Accepts expert-processed activations as input. Does NOT compute embeddings
    or perform MoE routing. Handles: attention, layernorm, decoder layers, lm_head.

    Input: activations from aggregated expert outputs [batch_size, seq_len, embeddings_dims]
    Output: logits [batch_size, seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        embeddings_dims: int = 768,
        no_of_heads: int = 12,
        no_of_decoder_layers: int = 12,
        device: torch.device = torch.device("cpu"),
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        """Initialize Mixtral model.

        Args:
            vocab_size: Size of vocabulary.
            embeddings_dims: Dimension of embeddings.
            no_of_heads: Number of attention heads.
            no_of_decoder_layers: Number of decoder layers.
            device: Device to place tensors on.
            attn_dropout: Dropout probability for attention.
            dropout: General dropout probability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings_dims = embeddings_dims

        self.linear_layer = nn.Linear(
            in_features=embeddings_dims, out_features=vocab_size, device=device, bias=False
        )
        self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embeddings_dims=embeddings_dims,
                    no_of_heads=no_of_heads,
                    device=device,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                )
                for _ in range(no_of_decoder_layers)
            ]
        )
        self.apply(self.kaiming_init_weights)

    def kaiming_init_weights(self, m: nn.Module) -> None:
        """Initialize weights using Kaiming initialization.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Forward pass for Mixtral.

        Args:
            activations: Expert-processed activations of shape
                         (batch_size, seq_len, embeddings_dims).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        x = activations
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.layer_norm(x)
        return self.linear_layer(x)

    def get_num_params(self) -> int:
        """Get the total number of parameters in the model.

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())
